#!/usr/bin/env python3
"""Two complementary CUDA patch encoders, trained from external observations.

No competition validation pixels or labels enter pretraining. A synthetic star
corpus supplements the downloaded observations with known correspondences.
Network A sees intensity; network B sees local high-pass structure. Both are
small, batched descriptors so expensive full-scene matching remains cached.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PatchEncoder(nn.Module):
    def __init__(self, mode="intensity"):
        super().__init__()
        self.mode = mode
        layers = []
        previous = 1
        for width in (32, 64, 128):
            layers.extend([nn.Conv2d(previous, width, 3, padding=1),
                           nn.GroupNorm(8, width), nn.SiLU(),
                           nn.Conv2d(width, width, 3, stride=2, padding=1),
                           nn.GroupNorm(8, width), nn.SiLU()])
            previous = width
        self.features = nn.Sequential(*layers)
        # Preserve coarse spatial structure instead of global average pooling.
        self.head = nn.Linear(128 * 2 * 2, 128)

    def forward(self, x):
        if self.mode == "highpass":
            x = x - F.avg_pool2d(F.pad(x, (3, 3, 3, 3), mode="reflect"), 7, 1)
        x = (x - x.mean((2, 3), keepdim=True)) / x.std((2, 3), keepdim=True).clamp_min(.025)
        z = F.adaptive_avg_pool2d(self.features(x.clamp(-5, 5)), 2).flatten(1)
        return F.normalize(self.head(z), dim=1)


def corpus(directory: Path, per_image=5000, synthetic=12000):
    rng = np.random.default_rng(6643)
    crops = []
    for path in sorted(directory.glob("*.jpg")):
        source = cv2.imread(str(path), 0)
        if source is None:
            raise ValueError(f"Unreadable external source: {path}")
        for factor in (1., .5):
            image = cv2.resize(source, None, fx=factor, fy=factor)
            response = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.)
            response -= cv2.GaussianBlur(image.astype(np.float32), (0, 0), 4.)
            maxima = (response == cv2.dilate(response, np.ones((7, 7), np.uint8)))
            maxima[:32] = maxima[-32:] = False
            maxima[:, :32] = maxima[:, -32:] = False
            ys, xs = np.where(maxima & (response > np.percentile(response, 85)))
            if len(xs):
                chosen = rng.choice(len(xs), min(per_image // 2, len(xs)), replace=False)
                crops.extend(image[y-16:y+16, x-16:x+16] for x, y in zip(xs[chosen], ys[chosen]))
    if not crops:
        raise ValueError("No external image crops found; run download_training_images.py")
    real_count = len(crops)
    yy, xx = np.mgrid[:32, :32].astype(np.float32)
    for _ in range(synthetic):
        im = np.full((32, 32), rng.uniform(0, 45), np.float32)
        for star in range(rng.integers(2, 12)):
            x, y = rng.uniform(0, 32, 2) if star else rng.uniform(13, 19, 2)
            sigma = rng.uniform(.5, 3.5)
            im += rng.uniform(12, 255) * np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma**2))
        im += rng.normal(0, rng.uniform(0, 5), (32, 32))
        crops.append(im.clip(0, 255).astype(np.uint8))
    print("CORPUS", {"external_crops": real_count, "synthetic": synthetic}, flush=True)
    return np.stack(crops)


def augment(x):
    b = len(x)
    device = x.device
    angle = torch.rand(b, device=device) * 2 * math.pi
    scale = torch.empty(b, device=device).uniform_(.72, 1.3)
    c, s = torch.cos(angle) / scale, torch.sin(angle) / scale
    translation = torch.empty(b, 2, device=device).uniform_(-.1, .1)
    theta = torch.stack((c, -s, translation[:, 0], s, c, translation[:, 1]), 1).reshape(b, 2, 3)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    y = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    mix = torch.rand(b, 1, 1, 1, device=device) * .8
    y = (1-mix)*y + mix*F.avg_pool2d(y, 3, 1, 1)
    y = y.clamp(0, 1).pow(torch.empty(b, 1, 1, 1, device=device).uniform_(.7, 1.5))
    gain = torch.empty(b, 1, 1, 1, device=device).uniform_(.6, 1.5)
    noise = torch.empty(b, 1, 1, 1, device=device).uniform_(.003, .065)
    return (y*gain + torch.randn_like(y)*noise).clamp(0, 1)


def train_models(image_dir: Path, output: Path, steps=4000, batch=192):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requested. Connect a Colab GPU; local CPU training is disabled.")
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256((image_dir / "manifest.json").read_bytes()).hexdigest()
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    signature = dict(source_sha256=source_hash, code_sha256=code_hash, steps=steps, batch=batch)
    images = None
    models = []
    for index, mode in enumerate(("intensity", "highpass")):
        seed = 6643 + index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = PatchEncoder(mode).cuda()
        path = output / f"encoder_{mode}.pt"
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state.get("signature") == signature:
                model.load_state_dict(state["state_dict"])
                models.append(model.eval())
                print("REUSE", path, flush=True)
                continue
        if images is None:
            images = torch.from_numpy(corpus(image_dir)[:, None]).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0008, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        model.train()
        for step in range(steps):
            # Distinct source indices avoid treating a crop as its own negative.
            indices = torch.randperm(len(images), device="cuda")[:min(batch, len(images))]
            x = images[indices].float()/255.
            a, b = model(augment(x)), model(augment(x))
            logits = a @ b.T / .1
            labels = torch.arange(len(indices), device="cuda")
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))/2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            scheduler.step()
            if step % 500 == 0 or step == steps-1:
                print("PRETRAIN", mode, step+1, "loss", round(loss.detach().item(), 4), flush=True)
        torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                        signature=signature, mode=mode, seed=seed), path)
        models.append(model.eval())
    return models


@torch.inference_mode()
def embed(model, images, batch=512):
    values = []
    device = next(model.parameters()).device
    for start in range(0, len(images), batch):
        x = torch.from_numpy(np.asarray(images[start:start+batch], np.float32)[:, None]/255.).to(device)
        values.append(model(x).float().cpu().numpy())
    return np.concatenate(values)


def radial_descriptor(images):
    """Rotation-invariant photometry, independent of the learned encoders."""
    yy, xx = np.mgrid[:32, :32]
    radius = np.sqrt((xx-15.5)**2+(yy-15.5)**2)
    features = []
    for im in images:
        row = []
        for left in range(0, 16, 2):
            ring = im[(radius >= left) & (radius < left+2)]
            row.extend((np.mean(ring), np.std(ring), np.percentile(ring, 90)))
        row = np.asarray(row, np.float32)
        features.append((row-row.mean())/(row.std()+1e-5))
    return np.asarray(features)
