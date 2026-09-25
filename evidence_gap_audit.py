#!/usr/bin/env python3
"""Audit distinct-location relative match gaps by labelled patch class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from constellation_pipeline import parse_cell, patch_columns, read_csv_rows


def relative_gap(group: list[list[float]]) -> float:
    if len(group) < 2:
        return 0.0
    first, second = float(group[0][2]), float(group[1][2])
    return (first - second) / max(1.0 - first, 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=(0.05, 0.10, 0.16, 0.20, 0.30)
    )
    args = parser.parse_args()
    root = args.root.resolve()
    values: dict[str, list[float]] = {"absent": [], "off_figure": [], "figure": []}
    for row in read_csv_rows(root / "train_ground_truth.csv"):
        saved = json.loads((args.cache_dir.resolve() / f"{row['Id']}.json").read_text())
        columns = patch_columns(int(row["n_patches"]))
        if saved["columns"] != columns:
            raise ValueError(f"Column mismatch in {row['Id']}")
        for column, group in zip(columns, saved["candidates"]):
            truth = parse_cell(row[column])
            label = "absent" if truth is None else ("figure" if truth[2] == 1 else "off_figure")
            values[label].append(relative_gap(group))
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
            f"threshold={threshold:.3f} clear: absent={counts['absent']}/{len(values['absent'])} "
            f"off={counts['off_figure']}/{len(values['off_figure'])} "
            f"figure={counts['figure']}/{len(values['figure'])}"
        )


if __name__ == "__main__":
    main()
