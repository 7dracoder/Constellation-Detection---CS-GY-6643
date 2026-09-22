#!/usr/bin/env python3
"""Refine non-member presence decisions with a course-data-only classifier.

The geometric solver deliberately uses a fixed per-scene present rate.  That
is robust, but it discards useful evidence in the shape of each query's
candidate-score distribution and in the query patch itself.  This module fits
a small regularised logistic model on the supplied training labels and then
post-processes an existing solver CSV:

* graph-supported ``m=1`` predictions are preserved exactly;
* other patches are marked present when the learned model clears its
  cross-validated threshold; and
* newly present non-members use the raw matcher's top-ranked location.

No scene name, external image, external label, or pretrained model is used as
a prediction feature.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from constellation_pipeline import (
    Prediction,
    format_cell,
    parse_cell,
    patch_columns,
    read_csv_rows,
    validate_submission,
)
from structural_refiner import patch_features


DEFAULT_REGULARIZATION = 0.15
DEFAULT_THRESHOLD = 0.58


@dataclass(frozen=True)
class PresenceClassifier:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    threshold: float

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        design = np.column_stack(
            (np.ones(len(features), dtype=np.float32), (features - self.mean) / self.scale)
        )
        logits = np.clip(design @ self.weights, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks / max(len(values) - 1, 1)


def load_candidate_cache(path: Path, columns: list[str]) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing candidate cache: {path}")
    saved = json.loads(path.read_text())
    if saved.get("columns") != columns:
        raise ValueError(f"Cache columns do not match scene template: {path}")
    if any(len(group) < 16 for group in saved["candidates"]):
        raise ValueError(f"Presence features require at least 16 candidates per patch: {path}")
    return saved


def scene_features(
    root: Path,
    split: str,
    scene: str,
    columns: list[str],
    cache_dir: Path,
) -> tuple[np.ndarray, dict]:
    saved = load_candidate_cache(cache_dir / f"{scene}.json", columns)
    score_matrix = np.asarray(
        [[float(item[2]) for item in group[:16]] for group in saved["candidates"]],
        dtype=np.float32,
    )
    top = score_matrix[:, 0]
    ranks = percentile_ranks(top)
    zscore = (top - top.mean()) / (top.std() + 1e-6)
    values: list[np.ndarray] = []
    for index, column in enumerate(columns):
        patch = cv2.imread(
            str(root / split / scene / "patches" / f"{column}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if patch is None:
            raise FileNotFoundError(f"Could not read {split}/{scene}/{column}.png")
        scores = score_matrix[index]
        values.append(
            np.concatenate(
                (
                    scores[[0, 1, 2, 3, 7, 15]],
                    np.asarray(
                        (
                            scores[0] - scores[1],
                            scores[0] - scores[3],
                            scores[0] - np.median(scores),
                            scores.std(),
                            ranks[index],
                            zscore[index],
                        ),
                        dtype=np.float32,
                    ),
                    patch_features(patch),
                )
            )
        )
    return np.stack(values).astype(np.float32), saved


def fit_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    regularization: float = DEFAULT_REGULARIZATION,
    threshold: float = DEFAULT_THRESHOLD,
) -> PresenceClassifier:
    mean = features.mean(axis=0)
    scale = features.std(axis=0) + 1e-5
    design = np.column_stack(
        (np.ones(len(features), dtype=np.float32), (features - mean) / scale)
    )
    weights = np.zeros(design.shape[1], dtype=np.float32)
    for _ in range(8_000):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = (design.T @ (probability - labels)) / len(labels)
        gradient[1:] += regularization * weights[1:]
        weights -= 0.05 * gradient
    return PresenceClassifier(mean, scale, weights, threshold)


def train_classifier(
    root: Path,
    cache_dir: Path,
    regularization: float = DEFAULT_REGULARIZATION,
    threshold: float = DEFAULT_THRESHOLD,
    excluded_scene: str | None = None,
) -> PresenceClassifier:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for row in read_csv_rows(root / "train_ground_truth.csv"):
        scene = row["Id"]
        if scene == excluded_scene:
            continue
        columns = patch_columns(int(row["n_patches"]))
        matrix, _ = scene_features(root, "train", scene, columns, cache_dir)
        features.append(matrix)
        labels.append(np.asarray([parse_cell(row[column]) is not None for column in columns]))
    return fit_classifier(
        np.vstack(features),
        np.concatenate(labels).astype(np.float32),
        regularization,
        threshold,
    )


def refine_rows(
    root: Path,
    split: str,
    rows: list[dict[str, str]],
    cache_dir: Path,
    train_cache_dir: Path,
    regularization: float,
    threshold: float,
    cross_validated: bool,
) -> list[dict[str, str]]:
    shared_model = None if cross_validated else train_classifier(
        root, train_cache_dir, regularization, threshold
    )
    result: list[dict[str, str]] = []
    for source_row in rows:
        row = source_row.copy()
        scene = row["Id"]
        columns = patch_columns(int(row["n_patches"]))
        features, saved = scene_features(root, split, scene, columns, cache_dir)
        model = shared_model or train_classifier(
            root, train_cache_dir, regularization, threshold, excluded_scene=scene
        )
        probabilities = model.probabilities(features)
        for index, column in enumerate(columns):
            current = parse_cell(row[column])
            if current is not None and current[2] == 1:
                continue
            if probabilities[index] >= model.threshold:
                best = saved["candidates"][index][0]
                row[column] = format_cell(
                    Prediction(x=int(best[0]), y=int(best[1]), m=0, score=float(best[2]))
                )
            else:
                row[column] = "-1"
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--train-cache-dir", type=Path, required=True)
    parser.add_argument("--regularization", type=float, default=DEFAULT_REGULARIZATION)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--cross-validated",
        action="store_true",
        help="For train-split diagnostics, refit while holding out each predicted scene",
    )
    args = parser.parse_args()
    if args.cross_validated and args.split != "train":
        parser.error("--cross-validated is only valid with --split train")
    root = args.root.resolve()
    rows = read_csv_rows(args.input.resolve())
    refined = refine_rows(
        root,
        args.split,
        rows,
        args.cache_dir.resolve(),
        args.train_cache_dir.resolve(),
        args.regularization,
        args.threshold,
        args.cross_validated,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(refined[0]))
        writer.writeheader()
        writer.writerows(refined)
    if args.split == "validation":
        validate_submission(root, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
