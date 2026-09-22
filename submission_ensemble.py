#!/usr/bin/env python3
"""Build a conservative ensemble from a proven and an experimental CSV.

The affine solver is most useful when it confirms the identity selected by
the established similarity solver.  When the two solvers disagree, copying
only the old name while retaining affine ``m=1`` coordinates would create an
internally inconsistent row (the points would describe a different pattern).
This tool therefore keeps the complete established row on disagreements and
the complete candidate row on agreements.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from constellation_pipeline import read_csv_rows, validate_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--established", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    established = {row["Id"]: row for row in read_csv_rows(args.established.resolve())}
    candidate = read_csv_rows(args.candidate.resolve())
    if set(established) != {row["Id"] for row in candidate}:
        raise ValueError("The two submissions contain different scene ids")

    selected: list[dict[str, str]] = []
    accepted, rejected = [], []
    for row in candidate:
        prior = established[row["Id"]]
        if row["constellation"] == prior["constellation"]:
            selected.append(row)
            accepted.append(row["Id"])
        else:
            selected.append(prior)
            rejected.append(
                f"{row['Id']}:{prior['constellation']}!={row['constellation']}"
            )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    validate_submission(root, output)
    print(f"accepted affine rows: {len(accepted)} ({', '.join(accepted)})")
    print(f"kept established rows: {len(rejected)} ({', '.join(rejected)})")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
