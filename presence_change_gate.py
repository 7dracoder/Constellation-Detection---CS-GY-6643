#!/usr/bin/env python3
"""Gate presence-only candidate rows by their scene-level addition rate.

The gate is label-blind and scene-name-blind. It refuses candidates that alter
anything other than absent-to-nonmember cells, or add too many detections.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from constellation_pipeline import parse_cell, patch_columns, read_csv_rows, validate_submission


def select_rows(
    established: list[dict[str, str]],
    candidate: list[dict[str, str]],
    max_added_rate: float,
) -> tuple[list[dict[str, str]], list[tuple[str, int, int, bool]]]:
    proposed = {row["Id"]: row for row in candidate}
    if {row["Id"] for row in established} != set(proposed):
        raise ValueError("Scene IDs differ")
    selected = []
    audit = []
    for old in established:
        new = proposed[old["Id"]]
        if old["constellation"] != new["constellation"] or old["n_patches"] != new["n_patches"]:
            raise ValueError(f"Label or patch count changed for {old['Id']}")
        columns = patch_columns(int(old["n_patches"]))
        additions = 0
        for column in columns:
            if old[column] == new[column]:
                continue
            revised = parse_cell(new[column])
            if old[column] != "-1" or revised is None or revised[2] != 0:
                raise ValueError(f"Not an absent-to-nonmember change: {old['Id']}/{column}")
            additions += 1
        accepted = additions <= max_added_rate * len(columns)
        selected.append(new if accepted else old)
        audit.append((old["Id"], additions, len(columns), accepted))
    return selected, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--established", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-added-rate", type=float, default=0.15)
    args = parser.parse_args()
    if not 0.0 <= args.max_added_rate <= 1.0:
        parser.error("--max-added-rate must be in [0, 1]")
    old = read_csv_rows(args.established.resolve())
    new = read_csv_rows(args.candidate.resolve())
    selected, audit = select_rows(old, new, args.max_added_rate)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(old[0]))
        writer.writeheader()
        writer.writerows(selected)
    validate_submission(args.root.resolve(), output)
    for scene, added, total, accepted in audit:
        if added:
            print(f"{scene}: {'accepted' if accepted else 'rejected'} {added}/{total} additions")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
