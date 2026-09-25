#!/usr/bin/env python3
"""Keep raw and filtered patch-location hypotheses for geometric fitting.

The first/raw candidate is retained for presence decisions. Alternate-view
candidates are interleaved so graph fitting can choose locations that differ
between image views. No labels or scene-specific rules are used.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def merge_group(
    baseline: list[list[float]], alternate: list[list[float]],
    limit: int, separation: float = 12.0,
) -> list[list[float]]:
    if not baseline:
        raise ValueError("A baseline query cannot have an empty candidate list")
    result: list[list[float]] = []
    for rank in range(max(len(baseline), len(alternate))):
        for view, group in (("baseline", baseline), ("alternate", alternate)):
            if rank >= len(group):
                continue
            source = group[rank]
            if any(math.dist(source[:2], prior[:2]) <= separation for prior in result):
                continue
            item = list(source)
            if view == "alternate":
                # NCC score scales shift substantially between views. Keep
                # the alternate rank but express its confidence on the raw
                # query's rank scale; the raw top-1 remains first.
                item[2] = float(baseline[min(rank, len(baseline) - 1)][2]) - 0.001
                item[3] = 0.0
            result.append(item)
            if len(result) >= limit:
                return result
    return result


def merge_caches(baseline_dir: Path, alternate_dir: Path, output_dir: Path) -> None:
    baseline_files = sorted(baseline_dir.glob("*.json"))
    if not baseline_files:
        raise ValueError(f"No baseline cache files in {baseline_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    for source in baseline_files:
        alternate_path = alternate_dir / source.name
        if not alternate_path.exists():
            raise FileNotFoundError(alternate_path)
        baseline = json.loads(source.read_text())
        alternate = json.loads(alternate_path.read_text())
        if baseline["columns"] != alternate["columns"]:
            raise ValueError(f"Patch columns differ: {source.name}")
        if len(baseline["candidates"]) != len(alternate["candidates"]):
            raise ValueError(f"Query counts differ: {source.name}")
        limit = int(baseline["top_k"]) + int(alternate["top_k"])
        merged = [
            merge_group(raw, filtered, limit)
            for raw, filtered in zip(baseline["candidates"], alternate["candidates"])
        ]
        output = {"columns": baseline["columns"], "top_k": limit, "candidates": merged}
        (output_dir / source.name).write_text(json.dumps(output))
        print(f"{source.stem}: {sum(len(group) for group in merged)} hypotheses")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--alternate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    merge_caches(args.baseline_dir.resolve(), args.alternate_dir.resolve(),
                 args.output_dir.resolve())


if __name__ == "__main__":
    main()
