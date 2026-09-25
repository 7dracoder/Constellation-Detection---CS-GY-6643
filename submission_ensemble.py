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

from constellation_pipeline import patch_columns, read_csv_rows, validate_submission


def changed_patch_count(prior: dict[str, str], candidate: dict[str, str]) -> int:
    """Count changed active patch cells without treating padding as evidence."""
    return sum(
        prior[column] != candidate[column]
        for column in patch_columns(int(prior["n_patches"]))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--established", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-patch-changes",
        type=int,
        help=(
            "Optional scene-level safety budget. Even when identities agree, "
            "keep the established row if the candidate rewrites more active "
            "patch cells than this. Omit to preserve the original behavior."
        ),
    )
    args = parser.parse_args()
    if args.max_patch_changes is not None and args.max_patch_changes < 0:
        parser.error("--max-patch-changes must be nonnegative")

    root = args.root.resolve()
    established = {row["Id"]: row for row in read_csv_rows(args.established.resolve())}
    candidate = read_csv_rows(args.candidate.resolve())
    if set(established) != {row["Id"] for row in candidate}:
        raise ValueError("The two submissions contain different scene ids")

    selected: list[dict[str, str]] = []
    accepted, rejected = [], []
    for row in candidate:
        prior = established[row["Id"]]
        changes = changed_patch_count(prior, row)
        same_identity = row["constellation"] == prior["constellation"]
        within_budget = (
            args.max_patch_changes is None or changes <= args.max_patch_changes
        )
        if same_identity and within_budget:
            selected.append(row)
            accepted.append(row["Id"])
        else:
            selected.append(prior)
            reason = (
                f"{prior['constellation']}!={row['constellation']}"
                if not same_identity
                else f"changes={changes}>{args.max_patch_changes}"
            )
            rejected.append(f"{row['Id']}:{reason}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    validate_submission(root, output)
    print(f"accepted candidate rows: {len(accepted)} ({', '.join(accepted)})")
    print(f"kept established rows: {len(rejected)} ({', '.join(rejected)})")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
