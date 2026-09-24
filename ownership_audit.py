#!/usr/bin/env python3
"""Compare graph-star ownership changes using independent candidate views.

This is a read-only diagnostic. Validation truth is never required or inferred.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from constellation_pipeline import parse_cell, patch_columns, read_csv_rows


def evidence(group: list[list[float]], cell: tuple[int, int, int] | None, tolerance: float) -> tuple[int, float]:
    """Return the best matching candidate's one-based rank and score."""
    if cell is None:
        return 0, float("nan")
    for rank, item in enumerate(group, 1):
        if math.dist((float(item[0]), float(item[1])), cell[:2]) <= tolerance:
            return rank, float(item[2])
    return 0, float("nan")


def load_groups(cache_dir: Path, scene: str, columns: list[str]) -> list[list[list[float]]]:
    path = cache_dir / f"{scene}.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    groups = saved.get("candidates", [])
    if saved.get("columns") != columns or len(groups) != len(columns):
        raise ValueError(f"Candidate cache does not match {scene}: {path}")
    return groups


def audit(
    baseline: Path,
    candidate: Path,
    raw_cache: Path,
    gaussian_cache: Path,
    tolerance: float = 8.0,
) -> list[dict[str, object]]:
    old_rows = read_csv_rows(baseline)
    new_rows = {row["Id"]: row for row in read_csv_rows(candidate)}
    if {row["Id"] for row in old_rows} != set(new_rows):
        raise ValueError("Baseline and candidate scene IDs differ")
    result: list[dict[str, object]] = []
    for old_row in old_rows:
        scene = old_row["Id"]
        new_row = new_rows[scene]
        columns = patch_columns(int(old_row["n_patches"]))
        if int(new_row["n_patches"]) != len(columns):
            raise ValueError(f"Patch count differs for {scene}")
        raw_groups = load_groups(raw_cache, scene, columns)
        gaussian_groups = load_groups(gaussian_cache, scene, columns)
        for index, column in enumerate(columns):
            if old_row[column] == new_row[column]:
                continue
            old = parse_cell(old_row[column])
            new = parse_cell(new_row[column])
            old_raw_rank, old_raw_score = evidence(raw_groups[index], old, tolerance)
            new_raw_rank, new_raw_score = evidence(raw_groups[index], new, tolerance)
            old_gauss_rank, old_gauss_score = evidence(gaussian_groups[index], old, tolerance)
            new_gauss_rank, new_gauss_score = evidence(gaussian_groups[index], new, tolerance)
            result.append(
                {
                    "scene": scene,
                    "patch": column,
                    "old": old_row[column],
                    "new": new_row[column],
                    "old_member": -1 if old is None else old[2],
                    "new_member": -1 if new is None else new[2],
                    "old_raw_rank": old_raw_rank,
                    "new_raw_rank": new_raw_rank,
                    "old_raw_score": old_raw_score,
                    "new_raw_score": new_raw_score,
                    "old_gaussian_rank": old_gauss_rank,
                    "new_gaussian_rank": new_gauss_rank,
                    "old_gaussian_score": old_gauss_score,
                    "new_gaussian_score": new_gauss_score,
                }
            )
    return result


def rank_direction(old_rank: int, new_rank: int) -> str:
    if old_rank == 0 or new_rank == 0:
        return "unmatched"
    return "better" if new_rank < old_rank else "worse" if new_rank > old_rank else "same"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--gaussian-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=8.0)
    args = parser.parse_args()
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    rows = audit(
        args.baseline.resolve(),
        args.candidate.resolve(),
        args.raw_cache.resolve(),
        args.gaussian_cache.resolve(),
        args.tolerance,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    counts = Counter(row["scene"] for row in rows)
    print(f"changed patches: {len(rows)} ({dict(sorted(counts.items()))})")
    both_members = [row for row in rows if row["old_member"] == row["new_member"] == 1]
    for view in ("raw", "gaussian"):
        directions = Counter(
            rank_direction(int(row[f"old_{view}_rank"]), int(row[f"new_{view}_rank"]))
            for row in both_members
        )
        print(f"member relocations, {view} rank: {dict(directions)}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
