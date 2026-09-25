#!/usr/bin/env python3
"""Compare two submissions in terms of what the published metric actually scores.

The four scored terms are Presence (the -1 vs point decision), Localization and
GeometricRecovery (both coordinate-only), and Identification (the name). The
reported `m` bit enters none of them, so this tool separates:

  * identity changes   -- 0.30 weight, one per scene
  * presence flips     -- 0.25 weight
  * coordinate moves   -- 0.20 + 0.25 weight, bucketed by distance
  * m-only flips       -- unscored, reported so they can be ignored

The 12 px / 36 px buckets matter because the reward ramp is flat below 12 px: a
move under 12 px that stays inside 12 px of truth changes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

from constellation_pipeline import parse_cell, patch_columns, read_csv_rows


def load(path: Path) -> dict[str, dict[str, str]]:
    return {row["Id"]: dict(row) for row in read_csv_rows(path)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="reference submission")
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()

    base, cand = load(args.base), load(args.candidate)
    if set(base) != set(cand):
        raise SystemExit("scene sets differ between the two files")

    identity_changes: list[tuple[str, str, str]] = []
    added = removed = m_only = 0
    moves: list[tuple[str, str, float]] = []
    per_scene: dict[str, int] = {}

    for scene in sorted(base):
        b, c = base[scene], cand[scene]
        if b["constellation"] != c["constellation"]:
            identity_changes.append((scene, b["constellation"], c["constellation"]))
        scene_scored = 0
        for column in patch_columns(int(b["n_patches"])):
            cell_b, cell_c = parse_cell(b[column]), parse_cell(c[column])
            if cell_b is None and cell_c is None:
                continue
            if cell_b is None:
                added += 1
                scene_scored += 1
                continue
            if cell_c is None:
                removed += 1
                scene_scored += 1
                continue
            distance = math.dist((cell_b[0], cell_b[1]), (cell_c[0], cell_c[1]))
            if distance == 0.0:
                if cell_b[2] != cell_c[2]:
                    m_only += 1
                continue
            moves.append((scene, column, distance))
            scene_scored += 1
        per_scene[scene] = scene_scored

    print(f"base      {args.base.name}\n          {sha256(args.base)}")
    print(f"candidate {args.candidate.name}\n          {sha256(args.candidate)}")

    print("\nSCORED CHANGES")
    print(f"  identity changes (0.30 per scene)   : {len(identity_changes)}")
    for scene, before, after in identity_changes:
        print(f"      {scene}: {before} -> {after}")
    print(f"  presence additions (absent -> point): {added}")
    print(f"  presence removals  (point -> absent): {removed}")
    print(f"  coordinate moves                    : {len(moves)}")
    if moves:
        under12 = sum(1 for *_, d in moves if d < 12.0)
        mid = sum(1 for *_, d in moves if 12.0 <= d < 36.0)
        far = sum(1 for *_, d in moves if d >= 36.0)
        print(f"      <12 px  (inside the flat reward zone): {under12}")
        print(f"      12-36 px (partial-credit zone)       : {mid}")
        print(f"      >=36 px  (a different star entirely) : {far}")

    print("\nUNSCORED CHANGES")
    print(f"  m-bit-only flips (no coordinate change): {m_only}")

    print("\nPER-SCENE scored cell changes")
    for scene, count in per_scene.items():
        if count:
            print(f"  {scene}: {count}")
    total = added + removed + len(moves)
    print(f"\ntotal scored cell changes: {total}  (plus {m_only} unscored m flips)")


if __name__ == "__main__":
    main()
