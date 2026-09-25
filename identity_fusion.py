#!/usr/bin/env python3
"""Fuse identity-only evidence into an established localization submission.

The published metric scores constellation name independently from presence,
localization, and geometric recovery.  This tool therefore changes only the
``constellation`` field, preserving every established patch decision and
coordinate byte-for-byte.  Identity sources may be complete/partial solver
CSVs or captured solver logs.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from constellation_pipeline import read_csv_rows, validate_submission


LOG_IDENTITY = re.compile(r"^(constellation_\d+):\s+([a-z][a-z-]*)\s+support=", re.MULTILINE)


def identities_from_log(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig")
    values = dict(LOG_IDENTITY.findall(text))
    if not values:
        raise ValueError(f"No solver identities found in {path}")
    return values


def identities_from_csv(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No identity rows found in {path}")
    return {row["Id"]: row["constellation"] for row in rows}


def fuse_identities(
    rows: list[dict[str, str]], sources: list[dict[str, str]]
) -> list[dict[str, str]]:
    known = {row["Id"] for row in rows}
    updates: dict[str, str] = {}
    for source in sources:
        unknown = set(source) - known
        if unknown:
            raise ValueError(f"Identity source contains unknown scenes: {sorted(unknown)}")
        updates.update(source)
    result = []
    for row in rows:
        revised = dict(row)
        revised["constellation"] = updates.get(row["Id"], row["constellation"])
        result.append(revised)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--identity-csv", type=Path, action="append", default=[])
    parser.add_argument("--identity-log", type=Path, action="append", default=[])
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Apply identity sources only to this scene; repeat as needed",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.identity_csv and not args.identity_log:
        parser.error("at least one --identity-csv or --identity-log is required")

    base = read_csv_rows(args.base.resolve())
    sources = [identities_from_log(path.resolve()) for path in args.identity_log]
    sources.extend(identities_from_csv(path.resolve()) for path in args.identity_csv)
    if args.scene:
        selected = set(args.scene)
        sources = [
            {scene: name for scene, name in source.items() if scene in selected}
            for source in sources
        ]
        missing = selected - {scene for source in sources for scene in source}
        if missing:
            raise ValueError(f"Selected scenes missing from identity sources: {sorted(missing)}")
    fused = fuse_identities(base, sources)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(base[0]))
        writer.writeheader()
        writer.writerows(fused)
    validate_submission(args.root.resolve(), output)
    for before, after in zip(base, fused):
        if before["constellation"] != after["constellation"]:
            print(
                f"{before['Id']}: {before['constellation']} -> {after['constellation']}"
            )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
