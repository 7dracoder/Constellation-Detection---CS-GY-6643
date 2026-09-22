#!/usr/bin/env python3
"""Test contextual patch matching without changing constellation geometry.

Bright stellar cores can dominate whole-patch correlation. This experiment
adds rotation/scale-searched correlations outside the core, then calibrates
candidate ranking with entire scenes held out. It changes only coordinates
of already-present, non-member patches; presence, identity and m=1 are fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import constellation_pipeline as cp
from multimodel_pipeline import (digest, fit_candidate_model, load_scene,
                                 promotion, rerank, summary, write_csv)


def masks():
    y, x = np.mgrid[:32, :32]
    radius = np.hypot(x-15.5, y-15.5)
    return [((radius >= inner) & (radius < 15)).astype(np.float32) for inner in (0, 6, 10)]


def normalized_masked(values, mask):
    """Masked zero-mean vectors; pixels outside the mask contribute nothing."""
    values = np.asarray(values, np.float32).reshape(len(values), -1)
    weight = mask.reshape(1, -1)
    mean = (values * weight).sum(1, keepdims=True) / weight.sum()
    centered = (values - mean) * weight
    return centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-6)


def contextual_features(root, split, scene, out):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Context matching is restricted to the Colab GPU runtime")
    saved, image, queries, _, coords = load_scene(root, split, scene)
    neural = out / f"features_{split}_{scene}.npz"
    signature = digest(neural) + digest(Path(__file__))
    signature += digest(root / "outputs" / f"cache_{split}" / f"{scene}.json")
    signature += digest(root / split / scene / f"{scene}_image.png")
    for column in saved["columns"]:
        signature += digest(root / split / scene / "patches" / f"{column}.png")
    path = out / f"annular_{split}_{scene}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["signature"]) == signature:
                return saved, coords, cached["features"].copy(), image.shape
    with np.load(neural, allow_pickle=False) as cached:
        base = cached["features"].copy()
    config = cp.load_config(root / "matcher_config_gpu_wide.json")
    offsets = np.array([(x, y) for y in (-2, 0, 2) for x in (-2, 0, 2)], np.int32)
    extra = []
    with torch.inference_mode():
        for qi, query in enumerate(queries):
            variants = np.stack([cp.transform_query(query, float(angle), scale)
                                 for scale in config.scales
                                 for angle in np.linspace(0, 360, config.angles, endpoint=False)])
            locations = (coords[qi, :, None, :] + offsets[None, :, :]).reshape(-1, 2)
            crops = cp.padded_extract(image, locations)
            scores = []
            for mask in masks():
                q = torch.from_numpy(normalized_masked(variants, mask)).cuda()
                c = torch.from_numpy(normalized_masked(crops, mask)).cuda()
                corr = (q @ c.T).amax(0).reshape(16, len(offsets)).amax(1).cpu().numpy()
                scores.extend((corr, (corr-corr.mean()) / (corr.std()+1e-5)))
            extra.append(np.stack(scores, -1))
    features = np.concatenate((base, np.stack(extra)), -1)
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite annular features: {scene}")
    np.savez_compressed(path, features=features, signature=signature)
    print("ANNULAR_FEATURES", split, scene, len(queries), flush=True)
    return saved, coords, features, image.shape


def refine(source, data, model):
    saved, _, features, _ = data
    ranked = rerank(saved, features, model, alpha=.35)
    row = source.copy()
    for qi, column in enumerate(saved["columns"]):
        current = cp.parse_cell(source[column])
        if current is not None and current[2] == 0:
            point = ranked[qi][0]
            row[column] = cp.format_cell(cp.Prediction(point.x, point.y, m=0))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "outputs" / "multimodel"
    truth = {row["Id"]: row for row in cp.read_csv_rows(root / "train_ground_truth.csv")}
    data = {scene: contextual_features(root, "train", scene, out) for scene in truth}
    baseline = cp.read_csv_rows(out / "train_baseline.csv")
    rows = [refine(row, data[row["Id"]], fit_candidate_model(data, truth, exclude=row["Id"]))
            for row in baseline]
    b, c = summary(truth, baseline), summary(truth, rows)
    accepted = promotion(b, c)
    report = dict(baseline=b, challenger=c, promoted=accepted, alpha=.35,
                  caveat="Exploratory three-scene holdout diagnostic, not a Kaggle estimate.",
                  policy="Change only coordinates of already-present m=0 patches; keep v4 geometry.")
    (out / "annular_report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(out / "train_annular.csv", rows)
    print("ANNULAR_REPORT", json.dumps(dict(baseline=b["mean"], challenger=c["mean"], promoted=accepted)), flush=True)
    if not accepted:
        print("NO_NEW_RECOMMENDATION: annular experiment failed its gate", flush=True)
        return
    model = fit_candidate_model(data, truth)
    rows = []
    for source in cp.read_csv_rows(root / "outputs" / "submission_v4_presence.csv"):
        scene = source["Id"]
        rows.append(refine(source, contextual_features(root, "validation", scene, out), model))
    output = out / "submission_v7_annular.csv"
    write_csv(output, rows)
    cp.validate_submission(root, output)
    report["csv_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    (out / "annular_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("ANNULAR_CSV", output, report["csv_sha256"], flush=True)


if __name__ == "__main__":
    main()
