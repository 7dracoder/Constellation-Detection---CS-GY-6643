#!/usr/bin/env python3
"""Rank train-split predictions by the metric Kaggle actually scores.

`evaluate.py` reports two totals. `total_loose` mirrors the published Fall 2026
metric: Presence 0.25, Localization 0.20, GeometricRecovery 0.25 (true figure
stars matched one-to-one against *all* reported present points, nearest pairs
first), Identification 0.30. `total_strict` additionally restricts the geometry
term to `m=1` cells, which the published metric does not do.

The reported `m` bit does not appear in any of the four published terms. That is
verifiable: forcing every `m` to 1 or to 0 leaves `total_loose` unchanged. This
module therefore ranks candidates by `total_loose` only, and reports the
`m`-invariance check so the claim can be re-tested rather than trusted.

Usage:
    python kaggle_metric_audit.py                     # rank every outputs/train_*.csv
    python kaggle_metric_audit.py --predictions a.csv --predictions b.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from constellation_pipeline import parse_cell, patch_columns, read_csv_rows
from evaluate import score_scene

SCORED_KEYS = ("presence", "localisation", "geometry_loose", "identity", "total_loose")


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    return {row["Id"]: dict(row) for row in read_csv_rows(path)}


def score_file(truth: dict[str, dict[str, str]], rows: dict[str, dict[str, str]]) -> dict[str, float]:
    """Mean of the published per-scene terms over the labelled scenes."""
    missing = set(truth) - set(rows)
    if missing:
        raise ValueError(f"missing labelled scenes: {sorted(missing)}")
    acc = {key: 0.0 for key in SCORED_KEYS}
    acc["member_patch_f1"] = 0.0
    for scene, truth_row in truth.items():
        result = score_scene(truth_row, rows[scene])
        for key in acc:
            acc[key] += result[key] / len(truth)
    return acc


def force_membership(rows: dict[str, dict[str, str]], value: int) -> dict[str, dict[str, str]]:
    """Rewrite every present cell's m bit, leaving coordinates and presence alone."""
    out: dict[str, dict[str, str]] = {}
    for scene, row in rows.items():
        new = dict(row)
        for column in patch_columns(int(row["n_patches"])):
            cell = parse_cell(row[column])
            if cell is not None:
                new[column] = f"({cell[0]}, {cell[1]}, {value})"
        out[scene] = new
    return out


def membership_invariance(truth, rows) -> tuple[float, float, float]:
    """Return total_loose as produced, with all m=1, and with all m=0."""
    return (
        score_file(truth, rows)["total_loose"],
        score_file(truth, force_membership(rows, 1))["total_loose"],
        score_file(truth, force_membership(rows, 0))["total_loose"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--predictions",
        type=Path,
        action="append",
        default=None,
        help="Score specific files instead of every outputs/train_*.csv",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    truth = load_rows(root / "train_ground_truth.csv")

    paths = args.predictions or sorted((root / "outputs").glob("train_*.csv"))
    results: list[tuple[float, str, dict[str, float], Path]] = []
    skipped: list[tuple[str, str]] = []
    for path in paths:
        try:
            acc = score_file(truth, load_rows(path))
        except Exception as error:  # cache-build logs and ledgers share the prefix
            skipped.append((path.name, type(error).__name__))
            continue
        results.append((acc["total_loose"], path.name, acc, path))

    results.sort(reverse=True)
    header = (
        f"{'rank':>4}  {'presence':>8}{'local':>8}{'geomL':>8}{'ident':>7}"
        f"{'KAGGLE':>9}  {'memF1*':>7}  file"
    )
    print(header)
    print("-" * (len(header) + 18))
    for index, (total, name, acc, _path) in enumerate(results, start=1):
        print(
            f"{index:>4}  {acc['presence']:8.3f}{acc['localisation']:8.3f}"
            f"{acc['geometry_loose']:8.3f}{acc['identity']:7.2f}"
            f"{total:9.4f}  {acc['member_patch_f1']:7.3f}  {name}"
        )
    print("\n* memF1 is NOT scored by Kaggle; shown only to expose past mis-ranking.")

    if results:
        best_name, best_path = results[0][1], results[0][3]
        produced, ones, zeros = membership_invariance(truth, load_rows(best_path))
        print(f"\nm-bit invariance check on {best_name}:")
        print(f"  as produced {produced:.4f} | all m=1 {ones:.4f} | all m=0 {zeros:.4f}")
        print(
            "  -> m is unscored"
            if abs(produced - ones) < 1e-9 and abs(produced - zeros) < 1e-9
            else "  -> m DOES affect the total; re-read the metric"
        )
    if skipped:
        print(f"\nskipped {len(skipped)} non-prediction files under outputs/train_*.csv")


if __name__ == "__main__":
    main()
