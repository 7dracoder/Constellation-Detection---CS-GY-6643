#!/usr/bin/env python3
"""Merge one deterministic geometric-RANSAC result per scene/pattern task."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from constellation_pipeline import read_csv_rows, validate_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = read_csv_rows(root / "sample_submission.csv")
    grouped: dict[str, list[dict[str, object]]] = {row["Id"]: [] for row in rows}
    for path in sorted(args.parts.glob("*.json")):
        data = json.loads(path.read_text())
        row = data["row"]
        details = data["diagnostics"]
        grouped[row["Id"]].append({"row": row, "diagnostics": details, "path": str(path)})
    selected: dict[str, dict[str, object]] = {}
    for row in rows:
        options = grouped[row["Id"]]
        if not options:
            raise RuntimeError(f"No completed geometric task for {row['Id']}")
        best = max(
            options,
            key=lambda item: (
                float(item["diagnostics"]["quality"]),
                int(item["diagnostics"]["support"]),
            ),
        )
        row.update(best["row"])
        selected[row["Id"]] = {"diagnostics": best["diagnostics"], "part": best["path"]}
    fieldnames = list(rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    validate_submission(root, args.output)
    args.output.with_suffix(".selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(f"merged {len(rows)} scenes into {args.output}")


if __name__ == "__main__":
    main()
