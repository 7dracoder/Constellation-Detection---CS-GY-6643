#!/usr/bin/env python3
"""Resolve only duplicate constellation labels using a distinct-label candidate.

Keeps the scored established row everywhere the established label is unique.
When two or more scenes share a label, those scenes take the candidate's rows
so the final CSV has no duplicate names — without rewriting strong unique fits.
Whole rows are swapped (never name-only) so ``m=1`` coordinates stay consistent
with the chosen pattern.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from constellation_pipeline import read_csv_rows, validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--established", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    established_rows = read_csv_rows(args.established.resolve())
    candidate = {row["Id"]: row for row in read_csv_rows(args.candidate.resolve())}
    if set(candidate) != {row["Id"] for row in established_rows}:
        raise ValueError("Scene ids differ between established and candidate CSVs")

    # Start from established rows.  Whenever any label repeats, replace every
    # scene that currently holds a duplicated label with the candidate's full
    # row.  Repeat until labels are unique or a round makes no progress (the
    # candidate itself may still collide).  This cascades: fixing canis-major
    # by promoting orion must then also refresh the prior unique orion scene.
    selected = [dict(row) for row in established_rows]
    swapped: list[str] = []
    initial_dups = sorted(
        name
        for name, count in Counter(row["constellation"] for row in selected).items()
        if name != "unknown" and count > 1
    )
    for _ in range(len(selected) + 1):
        counts = Counter(row["constellation"] for row in selected)
        duplicate_labels = {
            name for name, count in counts.items() if name != "unknown" and count > 1
        }
        if not duplicate_labels:
            break
        changed = False
        for index, row in enumerate(selected):
            if row["constellation"] not in duplicate_labels:
                continue
            replacement = dict(candidate[row["Id"]])
            if replacement == row:
                continue
            swapped.append(f"{row['Id']}:{row['constellation']}->{replacement['constellation']}")
            selected[index] = replacement
            changed = True
        if not changed:
            break

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)

    final_counts = Counter(row["constellation"] for row in selected)
    final_dups = sorted(name for name, count in final_counts.items() if name != "unknown" and count > 1)
    validate_submission(root, output)
    print(f"duplicate labels in established: {initial_dups or 'none'}")
    print(f"swapped rows: {', '.join(swapped) if swapped else 'none'}")
    print(f"final unique labels: {len(final_counts)}; remaining dups: {final_dups or 'none'}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
