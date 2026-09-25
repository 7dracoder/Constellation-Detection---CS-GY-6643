#!/usr/bin/env python3
"""Audit copy-paste neighborhood evidence across candidate locations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from constellation_pipeline import (
    image_files,
    padded_extract,
    parse_cell,
    patch_columns,
    read_csv_rows,
    read_grayscale,
)


def annulus_vectors(
    image: np.ndarray,
    points: np.ndarray,
    radius: int = 50,
    inner_radius: float = 20.0,
) -> np.ndarray:
    patches = padded_extract(image, np.rint(points).astype(np.int32), radius=radius)
    size = 2 * radius
    yy, xx = np.mgrid[:size, :size]
    centre = (size - 1) / 2.0
    distance = np.sqrt((xx - centre) ** 2 + (yy - centre) ** 2)
    mask = (distance >= inner_radius) & (distance <= radius - 2)
    vectors = patches[:, mask].astype(np.float32)
    vectors -= vectors.mean(axis=1, keepdims=True)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-6
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--radius", type=int, default=50)
    parser.add_argument("--inner-radius", type=float, default=20.0)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=(0.70, 0.80, 0.85, 0.90, 0.95)
    )
    args = parser.parse_args()
    root = args.root.resolve()
    values: dict[str, list[float]] = {"absent": [], "off_figure": [], "figure": []}
    records: list[tuple[float, str, str, str]] = []
    for row in read_csv_rows(root / "train_ground_truth.csv"):
        scene = row["Id"]
        image = read_grayscale(image_files(root / "train" / scene, "*_image.png")[0])
        saved = json.loads((args.cache_dir.resolve() / f"{scene}.json").read_text())
        columns = patch_columns(int(row["n_patches"]))
        for column, group in zip(columns, saved["candidates"]):
            points = np.asarray([item[:2] for item in group[: args.top_k]], dtype=np.float32)
            vectors = annulus_vectors(image, points, args.radius, args.inner_radius)
            best = float(np.max(vectors[1:] @ vectors[0])) if len(vectors) > 1 else -1.0
            truth = parse_cell(row[column])
            label = "absent" if truth is None else ("figure" if truth[2] == 1 else "off_figure")
            values[label].append(best)
            records.append((best, scene, column, label))
    for label, group in values.items():
        data = np.asarray(group, dtype=np.float64)
        quantiles = np.quantile(data, (0.0, 0.25, 0.5, 0.75, 1.0))
        print(
            f"{label:10} n={len(data):2d} min/q25/median/q75/max="
            + "/".join(f"{value:.3f}" for value in quantiles)
        )
    for threshold in args.thresholds:
        counts = {
            label: int(np.sum(np.asarray(group) >= threshold)) for label, group in values.items()
        }
        print(
            f"threshold={threshold:.2f} dup: absent={counts['absent']}/{len(values['absent'])} "
            f"off={counts['off_figure']}/{len(values['off_figure'])} "
            f"figure={counts['figure']}/{len(values['figure'])}"
        )
    strict = max(args.thresholds)
    for score, scene, column, label in sorted(records, reverse=True):
        if score >= strict:
            print(f"anchor={scene}/{column} class={label} annulus_ncc={score:.3f}")


if __name__ == "__main__":
    main()
