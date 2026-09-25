#!/usr/bin/env python3
"""Reblend cached raw and masked-NCC scores without recomputing image poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def reblend_group(group: list[list[float]], weight: float) -> list[list[float]]:
    rescored = []
    for item in group:
        if len(item) < 6:
            raise ValueError("Expected reranked cache entries with raw and masked scores")
        raw, masked = float(item[4]), float(item[5])
        rescored.append((float((1.0 - weight) * raw + weight * masked), item))
    rescored.sort(key=lambda pair: pair[0], reverse=True)
    result = []
    for rank, (score, item) in enumerate(rescored):
        next_score = rescored[rank + 1][0] if rank + 1 < len(rescored) else score
        result.append([
            float(item[0]), float(item[1]), score, float(score - next_score),
            float(item[4]), float(item[5]),
        ])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    args = parser.parse_args()
    if any(not 0.0 <= weight <= 1.0 for weight in args.weights):
        parser.error("--weights must be between zero and one")
    source = args.source_cache_dir.resolve()
    files = sorted(source.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON caches in {source}")
    for weight in args.weights:
        suffix = f"{int(round(100 * weight)):03d}"
        output_dir = args.output_root.resolve() / f"cache_train_masked_ncc{suffix}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in files:
            payload = json.loads(path.read_text())
            payload["candidates"] = [reblend_group(group, weight) for group in payload["candidates"]]
            payload["reranker_masked_weight"] = weight
            (output_dir / path.name).write_text(json.dumps(payload))
        print(f"weight={weight:.2f}: wrote {output_dir}")


if __name__ == "__main__":
    main()
