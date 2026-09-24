#!/usr/bin/env python3
"""Reassign existing graph members to fixed star nodes using two image views.

No new member or graph node is introduced. This is an experimental ablation,
not part of the established submission pipeline.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from constellation_pipeline import parse_cell, patch_columns, read_csv_rows, validate_submission
from ownership_audit import evidence, load_groups


def rank_cost(rank: int, group_size: int) -> float:
    """Bounded per-query rank cost; missing Gaussian evidence is penalized."""
    return math.log1p(rank - 1) / math.log1p(max(group_size - 1, 1)) if rank else 1.25


def refine_row(
    row: dict[str, str],
    raw_groups: list[list[list[float]]],
    gaussian_groups: list[list[list[float]]],
    tolerance: float = 8.0,
    min_gain: float = 0.0,
) -> tuple[dict[str, str], int, float]:
    columns = patch_columns(int(row["n_patches"]))
    member_indices = [i for i, col in enumerate(columns) if (parse_cell(row[col]) or (0, 0, 0))[2] == 1]
    count = len(member_indices)
    if count < 2:
        return row.copy(), 0, 0.0
    nodes = [parse_cell(row[columns[i]]) for i in member_indices]
    assert all(node is not None for node in nodes)
    cost = np.full((count, count), 100.0, dtype=np.float64)
    for node_index, node in enumerate(nodes):
        for query_column, query_index in enumerate(member_indices):
            raw_rank, _ = evidence(raw_groups[query_index], node, tolerance)
            if not raw_rank:
                continue
            gaussian_rank, _ = evidence(gaussian_groups[query_index], node, tolerance)
            cost[node_index, query_column] = (
                rank_cost(raw_rank, len(raw_groups[query_index]))
                + rank_cost(gaussian_rank, len(gaussian_groups[query_index]))
            )
    old_cost = sum(cost[i, i] for i in range(count))
    row_ids, col_ids = linear_sum_assignment(cost)
    new_cost = sum(cost[i, j] for i, j in zip(row_ids, col_ids))
    if any(cost[i, j] >= 100.0 for i, j in zip(row_ids, col_ids)) or new_cost >= old_cost - min_gain - 1e-10:
        return row.copy(), 0, 0.0
    refined = row.copy()
    changed = 0
    for node_index, query_column in zip(row_ids, col_ids):
        query_index = member_indices[query_column]
        column = columns[query_index]
        node = nodes[node_index]
        assert node is not None
        value = f"({node[0]}, {node[1]}, 1)"
        changed += value != row[column]
        refined[column] = value
    return refined, changed, old_cost - new_cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--gaussian-cache", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=8.0)
    parser.add_argument("--min-gain", type=float, default=0.0)
    args = parser.parse_args()
    if args.tolerance <= 0 or args.min_gain < 0:
        parser.error("tolerance must be positive and min-gain nonnegative")
    rows = read_csv_rows(args.input.resolve())
    output_rows = []
    for row in rows:
        columns = patch_columns(int(row["n_patches"]))
        raw = load_groups(args.raw_cache.resolve(), row["Id"], columns)
        gaussian = load_groups(args.gaussian_cache.resolve(), row["Id"], columns)
        refined, changed, gain = refine_row(row, raw, gaussian, args.tolerance, args.min_gain)
        output_rows.append(refined)
        print(f"{row['Id']}: changed={changed} rank-gain={gain:.3f}")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    if args.split == "validation":
        validate_submission(args.root.resolve(), output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
