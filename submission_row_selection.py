#!/usr/bin/env python3
"""Build an auditable submission using complete alternate rows for named scenes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from constellation_pipeline import read_csv_rows, validate_submission


def select_rows(
    established: list[dict[str, str]],
    alternative: list[dict[str, str]],
    scenes: set[str],
) -> list[dict[str, str]]:
    original = {row["Id"]: row for row in established}
    alternate = {row["Id"]: row for row in alternative}
    if len(original) != len(established) or len(alternate) != len(alternative) or set(original) != set(alternate):
        raise ValueError("Submissions must have the same unique scene IDs")
    if not scenes <= set(original):
        raise ValueError(f"Unknown scene IDs: {sorted(scenes - set(original))}")
    selected = []
    for row in established:
        other = alternate[row["Id"]]
        if row["n_patches"] != other["n_patches"] or set(row) != set(other):
            raise ValueError(f"Schema differs for {row['Id']}")
        selected.append(other if row["Id"] in scenes else row)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--established", type=Path, required=True)
    parser.add_argument("--alternative", type=Path, required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = read_csv_rows(args.established.resolve())
    alternative = read_csv_rows(args.alternative.resolve())
    selected = select_rows(original, alternative, set(args.scene))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(original[0]))
        writer.writeheader()
        writer.writerows(selected)
    validate_submission(args.root.resolve(), output)
    for scene in sorted(set(args.scene)):
        before = next(row for row in original if row["Id"] == scene)
        after = next(row for row in selected if row["Id"] == scene)
        changed = sum(before[column] != after[column] for column in before if column.startswith("patch_"))
        print(f"{scene}: {before['constellation']} -> {after['constellation']}, changed patches={changed}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
