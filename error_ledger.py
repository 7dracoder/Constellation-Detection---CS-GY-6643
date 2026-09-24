#!/usr/bin/env python3
"""Audit held-out train predictions against candidate caches and image signal.

The ledger is diagnostic only. Truth coordinates are used to score train
predictions and are never read by the inference pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from constellation_pipeline import parse_cell, patch_columns, read_csv_rows


def candidate_rank(group: list[list[float]], truth: tuple[int, int, int] | None, tolerance: float) -> int:
    """Return one-based rank of the first candidate near truth, or zero."""
    if truth is None:
        return 0
    for index, item in enumerate(group):
        if math.dist((float(item[0]), float(item[1])), truth[:2]) <= tolerance:
            return index + 1
    return 0


def classify_patch(
    truth: tuple[int, int, int] | None,
    prediction: tuple[int, int, int] | None,
    raw_rank: int,
    filtered_rank: int,
    tolerance: float,
) -> str:
    if truth is None:
        return "false_positive" if prediction is not None else "correct_absent"
    if prediction is None:
        return "retrieval_missing" if not (raw_rank or filtered_rank) else "presence_missed"
    if math.dist(truth[:2], prediction[:2]) > tolerance:
        return "retrieval_missing" if not (raw_rank or filtered_rank) else "location_selection_error"
    if truth[2] != prediction[2]:
        return "membership_missed" if truth[2] == 1 else "membership_false_positive"
    return "correct"


def patch_signal(image: np.ndarray) -> dict[str, float]:
    """Robust center-star contrast and width from the supplied query patch."""
    values = image.astype(np.float32)
    height, width = values.shape
    yy, xx = np.indices(values.shape)
    radius = np.hypot(xx - (width - 1) / 2.0, yy - (height - 1) / 2.0)
    annulus = values[(radius >= 9.0) & (radius <= 14.0)]
    center_mask = radius <= 6.0
    background = float(np.median(annulus))
    noise = float(1.4826 * np.median(np.abs(annulus - background)))
    peak = float(values[center_mask].max())
    weights = np.maximum(values - background, 0.0) * center_mask
    total = float(weights.sum())
    star_width = (
        math.sqrt(float((weights * radius**2).sum()) / (2.0 * total))
        if total > 0.0
        else float("nan")
    )
    return {
        "patch_peak": peak,
        "local_background": background,
        "local_noise": noise,
        "peak_over_noise": (peak - background) / max(noise, 1.0),
        "star_width_px": star_width,
    }


def load_cache(cache_dir: Path, scene: str, columns: list[str]) -> list[list[list[float]]]:
    path = cache_dir / f"{scene}.json"
    saved = json.loads(path.read_text())
    if saved.get("columns") != columns or len(saved.get("candidates", [])) != len(columns):
        raise ValueError(f"Candidate cache does not match the scene: {path}")
    return saved["candidates"]


def build_ledger(
    root: Path,
    predictions: Path,
    raw_cache: Path,
    filtered_cache: Path,
    tolerance: float = 12.0,
) -> list[dict[str, object]]:
    truth_rows = read_csv_rows(root / "train_ground_truth.csv")
    prediction_rows = {row["Id"]: row for row in read_csv_rows(predictions)}
    if set(prediction_rows) != {row["Id"] for row in truth_rows}:
        raise ValueError("Prediction and truth scene IDs differ")
    ledger: list[dict[str, object]] = []
    for truth_row in truth_rows:
        scene = truth_row["Id"]
        prediction_row = prediction_rows[scene]
        columns = patch_columns(int(truth_row["n_patches"]))
        if int(prediction_row["n_patches"]) != len(columns):
            raise ValueError(f"Patch count differs for {scene}")
        raw_groups = load_cache(raw_cache, scene, columns)
        filtered_groups = load_cache(filtered_cache, scene, columns)
        scene_image_paths = sorted((root / "train" / scene).glob("*_image.png"))
        if len(scene_image_paths) != 1:
            raise ValueError(f"Expected one sky image for {scene}")
        sky = cv2.imread(str(scene_image_paths[0]), cv2.IMREAD_GRAYSCALE)
        if sky is None:
            raise FileNotFoundError(scene_image_paths[0])
        height, width = sky.shape
        for index, column in enumerate(columns):
            truth = parse_cell(truth_row[column])
            prediction = parse_cell(prediction_row[column])
            raw_group = raw_groups[index]
            filtered_group = filtered_groups[index]
            if not raw_group or not filtered_group:
                raise ValueError(f"Empty candidate list for {scene}/{column}")
            raw_rank = candidate_rank(raw_group, truth, tolerance)
            filtered_rank = candidate_rank(filtered_group, truth, tolerance)
            patch_path = root / "train" / scene / "patches" / f"{column}.png"
            patch = cv2.imread(str(patch_path), cv2.IMREAD_GRAYSCALE)
            if patch is None:
                raise FileNotFoundError(patch_path)
            ledger.append(
                {
                    "scene": scene,
                    "patch": column,
                    "category": classify_patch(truth, prediction, raw_rank, filtered_rank, tolerance),
                    "truth_present": int(truth is not None),
                    "truth_member": truth[2] if truth is not None else "",
                    "prediction_present": int(prediction is not None),
                    "prediction_member": prediction[2] if prediction is not None else "",
                    "prediction_error_px": (
                        round(math.dist(truth[:2], prediction[:2]), 2)
                        if truth is not None and prediction is not None
                        else ""
                    ),
                    "raw_first_correct_rank": raw_rank,
                    "filtered_first_correct_rank": filtered_rank,
                    "raw_prediction_rank": candidate_rank(raw_group, prediction, tolerance),
                    "filtered_prediction_rank": candidate_rank(filtered_group, prediction, tolerance),
                    "raw_top_error_px": (
                        round(math.dist(truth[:2], raw_group[0][:2]), 2) if truth is not None else ""
                    ),
                    "filtered_top_error_px": (
                        round(math.dist(truth[:2], filtered_group[0][:2]), 2)
                        if truth is not None
                        else ""
                    ),
                    "raw_top_score": round(float(raw_group[0][2]), 4),
                    "filtered_top_score": round(float(filtered_group[0][2]), 4),
                    "border_distance_px": (
                        min(truth[0], truth[1], width - 1 - truth[0], height - 1 - truth[1])
                        if truth is not None
                        else ""
                    ),
                    **{key: round(value, 3) for key, value in patch_signal(patch).items()},
                }
            )
    return ledger


def print_summary(ledger: list[dict[str, object]]) -> None:
    for scene in sorted({str(row["scene"]) for row in ledger}):
        rows = [row for row in ledger if row["scene"] == scene]
        counts = Counter(str(row["category"]) for row in rows)
        detail = ", ".join(f"{category}={count}" for category, count in sorted(counts.items()))
        print(f"{scene}: {detail}")
    counts = Counter(str(row["category"]) for row in ledger)
    print("TOTAL: " + ", ".join(f"{category}={count}" for category, count in sorted(counts.items())))
    present = [row for row in ledger if row["truth_present"]]
    for name, key in (("raw", "raw_first_correct_rank"), ("filtered", "filtered_first_correct_rank")):
        top = sum(row[key] == 1 for row in present)
        retained = sum(int(row[key]) > 0 for row in present)
        print(f"{name} candidate recall: top-1 {top}/{len(present)}, retained {retained}/{len(present)}")
    snr = np.asarray([float(row["peak_over_noise"]) for row in present], dtype=np.float32)
    cut = float(np.median(snr))
    for label, subset in (
        ("lower-signal half", [row for row in present if float(row["peak_over_noise"]) <= cut]),
        ("higher-signal half", [row for row in present if float(row["peak_over_noise"]) > cut]),
    ):
        misses = sum(row["category"] != "correct" for row in subset)
        retrieval = sum(row["category"] == "retrieval_missing" for row in subset)
        print(f"{label}: {misses}/{len(subset)} errors, {retrieval} retrieval misses; median split={cut:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--filtered-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=12.0)
    args = parser.parse_args()
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    ledger = build_ledger(
        args.root.resolve(), args.predictions.resolve(), args.raw_cache.resolve(),
        args.filtered_cache.resolve(), args.tolerance,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    print_summary(ledger)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
