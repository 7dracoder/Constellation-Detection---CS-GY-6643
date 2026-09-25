#!/usr/bin/env python3
"""Pose-aware masked high-pass NCC reranking of an existing candidate cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from constellation_pipeline import (
    PATCH_RADIUS,
    PATCH_SIZE,
    image_files,
    padded_extract,
    patch_columns,
    read_csv_rows,
    read_grayscale,
)


DEFAULT_SCALES = (0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.40)


def pose_bank(
    query: np.ndarray,
    angles: int,
    scales: tuple[float, ...],
    highpass_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    centre = ((PATCH_SIZE - 1) / 2.0, (PATCH_SIZE - 1) / 2.0)
    yy, xx = np.mgrid[:PATCH_SIZE, :PATCH_SIZE]
    disk = ((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) <= (PATCH_RADIUS - 1) ** 2
    source_coverage = np.ones_like(query, dtype=np.float32)
    variants, masks = [], []
    for scale in scales:
        for angle_index in range(angles):
            angle = 360.0 * angle_index / angles
            matrix = cv2.getRotationMatrix2D(centre, angle, scale)
            warped = cv2.warpAffine(
                query,
                matrix,
                (PATCH_SIZE, PATCH_SIZE),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT101,
            )
            coverage = cv2.warpAffine(
                source_coverage,
                matrix,
                (PATCH_SIZE, PATCH_SIZE),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            highpass = warped - cv2.GaussianBlur(warped, (0, 0), highpass_sigma)
            variants.append(highpass.astype(np.float32))
            masks.append((disk & (coverage > 0.5)).astype(np.float32))
    return np.stack(variants), np.stack(masks)


def masked_ncc(variants: np.ndarray, masks: np.ndarray, patches: np.ndarray) -> np.ndarray:
    """Return pose-by-candidate NCC with a different validity mask per pose."""
    query = variants.reshape(len(variants), -1).astype(np.float32)
    mask = masks.reshape(len(masks), -1).astype(np.float32)
    scene = patches.reshape(len(patches), -1).astype(np.float32)
    count = np.maximum(mask.sum(axis=1), 1.0)
    query_mean = (query * mask).sum(axis=1) / count
    query_centered = (query - query_mean[:, None]) * mask
    query_norm = np.linalg.norm(query_centered, axis=1)
    scene_sum = mask @ scene.T
    scene_square_sum = mask @ (scene * scene).T
    scene_norm = np.sqrt(np.maximum(scene_square_sum - scene_sum * scene_sum / count[:, None], 1e-8))
    numerator = query_centered @ scene.T
    return numerator / np.maximum(query_norm[:, None] * scene_norm, 1e-8)


def rerank_scene(
    root: Path,
    split: str,
    scene: str,
    source_cache: Path,
    output_cache: Path,
    angles: int,
    scales: tuple[float, ...],
    highpass_sigma: float,
    masked_weight: float,
) -> None:
    saved = json.loads(source_cache.read_text())
    columns = saved["columns"]
    image = read_grayscale(image_files(root / split / scene, "*_image.png")[0])
    scene_highpass = image - cv2.GaussianBlur(image, (0, 0), highpass_sigma)
    reranked = []
    for query_index, (column, group) in enumerate(zip(columns, saved["candidates"])):
        query = read_grayscale(root / split / scene / "patches" / f"{column}.png")
        variants, masks = pose_bank(query, angles, scales, highpass_sigma)
        points = np.asarray([item[:2] for item in group], dtype=np.float32)
        patches = padded_extract(scene_highpass, np.rint(points).astype(np.int32))
        masked_scores = masked_ncc(variants, masks, patches).max(axis=0)
        raw_scores = np.asarray([float(item[2]) for item in group], dtype=np.float32)
        fused = (1.0 - masked_weight) * raw_scores + masked_weight * masked_scores
        order = np.argsort(fused)[::-1]
        ordered = []
        for rank, index in enumerate(order):
            next_score = float(fused[order[rank + 1]]) if rank + 1 < len(order) else float(fused[index])
            ordered.append([
                float(points[index, 0]),
                float(points[index, 1]),
                float(fused[index]),
                float(fused[index] - next_score),
                float(raw_scores[index]),
                float(masked_scores[index]),
            ])
        reranked.append(ordered)
        if (query_index + 1) % 10 == 0 or query_index + 1 == len(columns):
            print(f"  {scene}: {query_index + 1}/{len(columns)}", flush=True)
    payload = dict(saved)
    payload["candidates"] = reranked
    payload["reranker"] = "pose_masked_highpass_ncc"
    payload["reranker_angles"] = angles
    payload["reranker_scales"] = list(scales)
    payload["reranker_highpass_sigma"] = highpass_sigma
    payload["reranker_masked_weight"] = masked_weight
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    output_cache.write_text(json.dumps(payload))
    print(f"{scene}: wrote {output_cache}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--output-cache-dir", type=Path, required=True)
    parser.add_argument("--angles", type=int, default=72)
    parser.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    parser.add_argument("--highpass-sigma", type=float, default=3.0)
    parser.add_argument("--masked-weight", type=float, default=0.70)
    args = parser.parse_args()
    if args.angles < 1 or args.highpass_sigma <= 0.0:
        parser.error("--angles and --highpass-sigma must be positive")
    if not 0.0 <= args.masked_weight <= 1.0:
        parser.error("--masked-weight must be between zero and one")
    if not args.scales or any(scale <= 0.0 for scale in args.scales):
        parser.error("--scales must contain positive values")
    rows = read_csv_rows(
        args.root.resolve()
        / ("train_ground_truth.csv" if args.split == "train" else "sample_submission.csv")
    )
    for row in rows:
        scene = row["Id"]
        columns = patch_columns(int(row["n_patches"]))
        source = args.source_cache_dir.resolve() / f"{scene}.json"
        saved = json.loads(source.read_text())
        if saved.get("columns") != columns:
            raise ValueError(f"Column mismatch in {source}")
        rerank_scene(
            args.root.resolve(),
            args.split,
            scene,
            source,
            args.output_cache_dir.resolve() / f"{scene}.json",
            args.angles,
            tuple(args.scales),
            args.highpass_sigma,
            args.masked_weight,
        )


if __name__ == "__main__":
    main()
