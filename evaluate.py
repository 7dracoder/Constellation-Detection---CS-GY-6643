#!/usr/bin/env python3
"""Scene-held-out evaluator for the constellation competition.

The repository previously scored identity with
``float(row["constellation"] == "unknown")``, which awards credit for
declining to answer a scene that does have a label.  That made the old
"approximate train score" unusable for comparing configurations.

The official Kaggle formula is not published, so this mirrors the weighted
decomposition already documented in ``constellation_pipeline.py`` with the
identity term corrected, and additionally reports a strict reading of the
membership term (only ``m=1`` cells count) alongside the loose one (any
present cell counts).  Use the components, not just the total, when
comparing runs.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np

from constellation_pipeline import parse_cell, patch_columns, read_csv_rows


WEIGHTS = {"presence": 0.25, "localisation": 0.20, "geometry": 0.25, "identity": 0.30}


def macro_f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    values = []
    for positive in (False, True):
        tp = int(np.sum((actual == positive) & (predicted == positive)))
        fp = int(np.sum((actual != positive) & (predicted == positive)))
        fn = int(np.sum((actual == positive) & (predicted != positive)))
        values.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(values))


def point_reward(distance: float) -> float:
    """Full credit within 12 px, decaying linearly to zero at 36 px."""
    if distance <= 12.0:
        return 1.0
    return max(0.0, min(1.0, (36.0 - distance) / 24.0))


def greedy_reward(truth_points: list[tuple[int, int]], predicted: list[tuple[int, int]]) -> float:
    """Nearest-available matching of predicted points onto true figure stars."""
    if not truth_points:
        return 1.0
    remaining = list(predicted)
    rewards = []
    for x, y in truth_points:
        if not remaining:
            rewards.append(0.0)
            continue
        distances = [math.dist((x, y), point) for point in remaining]
        index = int(np.argmin(distances))
        rewards.append(point_reward(distances[index]))
        remaining.pop(index)
    return float(np.mean(rewards))


def score_scene(truth_row: dict[str, str], predicted_row: dict[str, str]) -> dict[str, float]:
    n_patches = int(truth_row["n_patches"])
    truth_present, predicted_present = [], []
    localisation = []
    truth_figure: list[tuple[int, int]] = []
    predicted_any: list[tuple[int, int]] = []
    predicted_member: list[tuple[int, int]] = []
    for column in patch_columns(n_patches):
        truth = parse_cell(truth_row[column])
        guess = parse_cell(predicted_row[column])
        truth_present.append(truth is not None)
        predicted_present.append(guess is not None)
        if truth is not None:
            distance = math.dist((truth[0], truth[1]), (guess[0], guess[1])) if guess else 1e9
            localisation.append(point_reward(distance) if guess else 0.0)
            if truth[2] == 1:
                truth_figure.append((truth[0], truth[1]))
        if guess is not None:
            predicted_any.append((guess[0], guess[1]))
            if guess[2] == 1:
                predicted_member.append((guess[0], guess[1]))

    presence = macro_f1(np.asarray(truth_present), np.asarray(predicted_present))
    localisation_score = float(np.mean(localisation)) if localisation else 0.0
    geometry_loose = greedy_reward(truth_figure, predicted_any)
    geometry_strict = greedy_reward(truth_figure, predicted_member)
    identity = float(predicted_row["constellation"] == truth_row["Id"])

    total_loose = (
        WEIGHTS["presence"] * presence
        + WEIGHTS["localisation"] * localisation_score
        + WEIGHTS["geometry"] * geometry_loose
        + WEIGHTS["identity"] * identity
    )
    total_strict = total_loose - WEIGHTS["geometry"] * (geometry_loose - geometry_strict)
    return {
        "presence": presence,
        "localisation": localisation_score,
        "geometry_loose": geometry_loose,
        "geometry_strict": geometry_strict,
        "identity": identity,
        "total_loose": total_loose,
        "total_strict": total_strict,
        "n_figure": len(truth_figure),
        "n_member": len(predicted_member),
    }


def evaluate(root: Path, predictions: Path) -> dict[str, float]:
    truth_rows = {row["Id"]: row for row in read_csv_rows(root / "train_ground_truth.csv")}
    predicted_rows = {row["Id"]: row for row in read_csv_rows(predictions)}
    missing = set(truth_rows) - set(predicted_rows)
    if missing:
        raise ValueError(f"Prediction file is missing labelled scenes: {sorted(missing)}")
    header = (
        f"{'scene':10}{'presence':>10}{'local':>8}{'geom(L)':>9}{'geom(S)':>9}"
        f"{'ident':>7}{'m=1':>6}{'true':>6}{'total(L)':>10}{'total(S)':>10}"
    )
    print(header)
    print("-" * len(header))
    totals: dict[str, list[float]] = {}
    for scene, truth_row in truth_rows.items():
        result = score_scene(truth_row, predicted_rows[scene])
        for key, value in result.items():
            totals.setdefault(key, []).append(value)
        print(
            f"{scene:10}{result['presence']:10.3f}{result['localisation']:8.3f}"
            f"{result['geometry_loose']:9.3f}{result['geometry_strict']:9.3f}"
            f"{result['identity']:7.0f}{result['n_member']:6d}{result['n_figure']:6d}"
            f"{result['total_loose']:10.4f}{result['total_strict']:10.4f}"
        )
    mean = {key: float(np.mean(values)) for key, values in totals.items()}
    print("-" * len(header))
    print(
        f"{'MEAN':10}{mean['presence']:10.3f}{mean['localisation']:8.3f}"
        f"{mean['geometry_loose']:9.3f}{mean['geometry_strict']:9.3f}"
        f"{mean['identity']:7.2f}{mean['n_member']:6.1f}{mean['n_figure']:6.1f}"
        f"{mean['total_loose']:10.4f}{mean['total_strict']:10.4f}"
    )
    print(
        f"\nidentity: {int(sum(totals['identity']))}/{len(totals['identity'])} scenes correct"
    )
    return mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.root.resolve(), args.predictions.resolve())


if __name__ == "__main__":
    main()
