#!/usr/bin/env python3
"""Scene-held-out evaluator for the constellation competition.

The repository previously scored identity with
``float(row["constellation"] == "unknown")``, which awards credit for
declining to answer a scene that does have a label.  That made the old
"approximate train score" unusable for comparing configurations.

The Fall 2026 Kaggle Evaluation text specifies a 25/20/25/30 decomposition,
12-to-36-pixel point reward, and nearest-pairs-first set matching for figure
stars. This local scorer mirrors those published rules and additionally
reports a strict diagnostic (only ``m=1`` cells count) alongside the scored
loose geometry term (any reported point counts). The set-based geometry term
does not require the right patch to own a figure star, so member_patch_f1
reports that correspondence separately. The train set has only three scenes;
its score is still not an unbiased estimate of the hidden split.
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
    """Globally nearest-pairs-first matching of figure stars to reported points."""
    if not truth_points:
        return 1.0
    pairs = sorted(
        (math.dist(truth, guess), truth_index, guess_index)
        for truth_index, truth in enumerate(truth_points)
        for guess_index, guess in enumerate(predicted)
    )
    used_truth: set[int] = set()
    used_guesses: set[int] = set()
    reward_sum = 0.0
    for distance, truth_index, guess_index in pairs:
        if truth_index in used_truth or guess_index in used_guesses:
            continue
        used_truth.add(truth_index)
        used_guesses.add(guess_index)
        reward_sum += point_reward(distance)
        if len(used_truth) == len(truth_points):
            break
    return reward_sum / len(truth_points)


def score_scene(truth_row: dict[str, str], predicted_row: dict[str, str]) -> dict[str, float]:
    n_patches = int(truth_row["n_patches"])
    truth_present, predicted_present = [], []
    localisation = []
    truth_figure: list[tuple[int, int]] = []
    predicted_any: list[tuple[int, int]] = []
    predicted_member: list[tuple[int, int]] = []
    correctly_assigned_members = 0
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
                if guess is not None and guess[2] == 1 and distance <= 12.0:
                    correctly_assigned_members += 1
        if guess is not None:
            predicted_any.append((guess[0], guess[1]))
            if guess[2] == 1:
                predicted_member.append((guess[0], guess[1]))

    presence = macro_f1(np.asarray(truth_present), np.asarray(predicted_present))
    localisation_score = float(np.mean(localisation)) if localisation else 0.0
    geometry_loose = greedy_reward(truth_figure, predicted_any)
    geometry_strict = greedy_reward(truth_figure, predicted_member)
    member_precision = correctly_assigned_members / len(predicted_member) if predicted_member else 0.0
    member_recall = correctly_assigned_members / len(truth_figure) if truth_figure else 0.0
    member_patch_f1 = (
        2.0 * member_precision * member_recall / (member_precision + member_recall)
        if member_precision + member_recall > 0.0
        else 0.0
    )
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
        "member_patch_f1": member_patch_f1,
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
        f"{'ident':>7}{'m=1':>6}{'true':>6}{'memberF1':>10}{'total(L)':>10}{'total(S)':>10}"
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
            f"{result['member_patch_f1']:10.3f}"
            f"{result['total_loose']:10.4f}{result['total_strict']:10.4f}"
        )
    mean = {key: float(np.mean(values)) for key, values in totals.items()}
    print("-" * len(header))
    print(
        f"{'MEAN':10}{mean['presence']:10.3f}{mean['localisation']:8.3f}"
        f"{mean['geometry_loose']:9.3f}{mean['geometry_strict']:9.3f}"
        f"{mean['identity']:7.2f}{mean['n_member']:6.1f}{mean['n_figure']:6.1f}"
        f"{mean['member_patch_f1']:10.3f}"
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
