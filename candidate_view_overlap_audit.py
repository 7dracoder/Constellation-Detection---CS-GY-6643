#!/usr/bin/env python3
"""Measure complementary top-1 localization hits between two candidate caches."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from clear_gap_refiner import candidate_relative_gap
from constellation_pipeline import parse_cell, patch_columns, read_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--left-cache", type=Path, required=True)
    parser.add_argument("--right-cache", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=12.0)
    args = parser.parse_args()
    counts: dict[str, dict[str, int]] = {}
    exclusive_gaps: dict[str, list[float]] = {"left_only": [], "right_only": []}
    for row in read_csv_rows(args.root.resolve() / "train_ground_truth.csv"):
        columns = patch_columns(int(row["n_patches"]))
        left = json.loads((args.left_cache.resolve() / f"{row['Id']}.json").read_text())
        right = json.loads((args.right_cache.resolve() / f"{row['Id']}.json").read_text())
        for column, left_group, right_group in zip(columns, left["candidates"], right["candidates"]):
            truth = parse_cell(row[column])
            if truth is None:
                continue
            label = "figure" if truth[2] == 1 else "off_figure"
            left_hit = math.dist(left_group[0][:2], truth[:2]) <= args.tolerance
            right_hit = math.dist(right_group[0][:2], truth[:2]) <= args.tolerance
            outcome = (
                "both" if left_hit and right_hit else
                "left_only" if left_hit else
                "right_only" if right_hit else "neither"
            )
            counts.setdefault(label, {key: 0 for key in ("both", "left_only", "right_only", "neither")})
            counts[label][outcome] += 1
            if outcome == "left_only":
                exclusive_gaps[outcome].append(candidate_relative_gap(left_group))
            elif outcome == "right_only":
                exclusive_gaps[outcome].append(candidate_relative_gap(right_group))
    for label, group in counts.items():
        print(label, group)
    for outcome, gaps in exclusive_gaps.items():
        if gaps:
            print(
                f"{outcome} winning-gap n={len(gaps)} min/median/max="
                f"{min(gaps):.3f}/{float(np.median(gaps)):.3f}/{max(gaps):.3f}"
            )


if __name__ == "__main__":
    main()
