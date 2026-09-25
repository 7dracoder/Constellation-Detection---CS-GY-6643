#!/usr/bin/env python3
"""Promote absent patches with a strong distinct-location relative gap."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from constellation_pipeline import (
    Prediction,
    format_cell,
    patch_columns,
    read_csv_rows,
    validate_submission,
)


def candidate_relative_gap(group: list[list[float]]) -> float:
    if len(group) < 2:
        return 0.0
    first, second = float(group[0][2]), float(group[1][2])
    return (first - second) / max(1.0 - first, 1e-6)


def refine(
    root: Path,
    split: str,
    input_path: Path,
    output_path: Path,
    cache_dir: Path,
    threshold: float,
) -> None:
    rows = read_csv_rows(input_path)
    promoted: list[str] = []
    for row in rows:
        columns = patch_columns(int(row["n_patches"]))
        saved = json.loads((cache_dir / f"{row['Id']}.json").read_text())
        if saved.get("columns") != columns:
            raise ValueError(f"Cache columns do not match {row['Id']}")
        for column, group in zip(columns, saved["candidates"]):
            if row[column].strip() != "-1" or candidate_relative_gap(group) < threshold:
                continue
            best = group[0]
            row[column] = format_cell(
                Prediction(int(round(float(best[0]))), int(round(float(best[1]))), m=0)
            )
            promoted.append(f"{row['Id']}/{column}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if split == "validation":
        validate_submission(root, output_path)
    print(f"promoted {len(promoted)} patches: {', '.join(promoted) if promoted else 'none'}")
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.16)
    args = parser.parse_args()
    if args.threshold < 0.0:
        parser.error("--threshold must be nonnegative")
    refine(
        args.root.resolve(),
        args.split,
        args.input.resolve(),
        args.output.resolve(),
        args.cache_dir.resolve(),
        args.threshold,
    )


if __name__ == "__main__":
    main()
