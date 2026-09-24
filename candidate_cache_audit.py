#!/usr/bin/env python3
"""Compare candidate-cache localisation recall on the labelled train scenes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from constellation_pipeline import parse_cell, patch_columns, read_csv_rows


def audit_cache(root: Path, cache_dir: Path, tolerance: float = 12.0) -> dict[str, float]:
    """Measure top-1 and retained-list recall for every labelled-present patch."""
    total = 0
    top1_hits = 0
    retained_hits = 0
    nearest_distances: list[float] = []
    missing_scenes: list[str] = []

    for truth_row in read_csv_rows(root / "train_ground_truth.csv"):
        scene = truth_row["Id"]
        cache_path = cache_dir / f"{scene}.json"
        if not cache_path.exists():
            missing_scenes.append(scene)
            continue
        saved = json.loads(cache_path.read_text())
        columns = list(saved["columns"])
        groups = saved["candidates"]
        if len(columns) != len(groups):
            raise ValueError(f"Cache column/candidate mismatch: {cache_path}")
        by_column = dict(zip(columns, groups))
        for column in patch_columns(int(truth_row["n_patches"])):
            truth = parse_cell(truth_row[column])
            if truth is None:
                continue
            total += 1
            candidates = by_column.get(column, [])
            if not candidates:
                nearest_distances.append(float("inf"))
                continue
            distances = [math.hypot(float(row[0]) - truth[0], float(row[1]) - truth[1]) for row in candidates]
            nearest_distances.append(min(distances))
            top1_hits += int(distances[0] <= tolerance)
            retained_hits += int(min(distances) <= tolerance)

    if missing_scenes:
        raise FileNotFoundError(
            f"{cache_dir} is missing labelled scenes: {', '.join(sorted(missing_scenes))}"
        )
    finite = [distance for distance in nearest_distances if np.isfinite(distance)]
    return {
        "labelled_present": float(total),
        "top1_hits": float(top1_hits),
        "retained_hits": float(retained_hits),
        "top1_recall": top1_hits / total if total else 0.0,
        "retained_recall": retained_hits / total if total else 0.0,
        "median_nearest_distance": float(np.median(finite)) if finite else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--cache-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--tolerance", type=float, default=12.0)
    args = parser.parse_args()
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")

    root = args.root.resolve()
    for cache_dir_arg in args.cache_dir:
        cache_dir = cache_dir_arg.resolve()
        result = audit_cache(root, cache_dir, args.tolerance)
        print(f"{cache_dir}")
        print(
            f"  top-1: {int(result['top1_hits'])}/{int(result['labelled_present'])} "
            f"({result['top1_recall']:.3f})"
        )
        print(
            f"  retained: {int(result['retained_hits'])}/{int(result['labelled_present'])} "
            f"({result['retained_recall']:.3f})"
        )
        print(f"  median nearest distance: {result['median_nearest_distance']:.2f}px")


if __name__ == "__main__":
    main()
