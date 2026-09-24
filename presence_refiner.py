#!/usr/bin/env python3
"""Refine non-member presence decisions with a course-data-only classifier.

The geometric solver deliberately uses a fixed per-scene present rate.  That
is robust, but it discards useful evidence in the shape of each query's
candidate-score distribution and in the query patch itself.  This module fits
a small regularised logistic model on the supplied training labels and then
post-processes an existing solver CSV:

* graph-supported ``m=1`` predictions are preserved exactly;
* other patches are marked present when the learned model clears its
  cross-validated threshold;
* one or more matched-filter candidate caches can be fused, including their
  cross-filter location agreement; and
* newly present non-members use the first cache's top-ranked location.

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


def _cache_dirs(value: Path | list[Path] | tuple[Path, ...]) -> list[Path]:
    """Normalize the legacy single-cache and new multi-cache APIs."""
    return [value] if isinstance(value, Path) else list(value)


def _score_features(score_matrix: np.ndarray) -> np.ndarray:
    """Per-query rank and separation features for one matcher view."""
    top = score_matrix[:, 0]
    ranks = percentile_ranks(top)
    zscore = (top - top.mean()) / (top.std() + 1e-6)
    values = []
    for index, scores in enumerate(score_matrix):
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
                )
            )
        )
    return np.stack(values).astype(np.float32)


def _agreement_features(saved_views: list[dict], query_index: int) -> np.ndarray:
    """Measure whether independently filtered views retrieve the same place.

    Features are pairwise and symmetric apart from the two nearest-list
    distances.  They use coordinates and ranks only; no scene identity or
    validation label enters the fusion model.
    """
    values: list[float] = []
    for left_index in range(len(saved_views)):
        left = np.asarray(saved_views[left_index]["candidates"][query_index][:16], dtype=np.float32)
        for right_index in range(left_index + 1, len(saved_views)):
            right = np.asarray(saved_views[right_index]["candidates"][query_index][:16], dtype=np.float32)
            left_xy, right_xy = left[:, :2], right[:, :2]
            top_distance = float(np.linalg.norm(left_xy[0] - right_xy[0]))
            left_to_right = float(np.linalg.norm(right_xy - left_xy[0], axis=1).min())
            right_to_left = float(np.linalg.norm(left_xy - right_xy[0], axis=1).min())
            score_delta = float(left[0, 2] - right[0, 2])
            values.extend(
                (
                    top_distance,
                    left_to_right,
                    right_to_left,
                    score_delta,
                    abs(score_delta),
                    float(top_distance <= 12.0),
                )
            )
    return np.asarray(values, dtype=np.float32)


def scene_features(
    root: Path,
    split: str,
    scene: str,
    columns: list[str],
    cache_dir: Path | list[Path] | tuple[Path, ...],
) -> tuple[np.ndarray, list[dict]]:
    cache_dirs = _cache_dirs(cache_dir)
    if not cache_dirs:
        raise ValueError("At least one candidate cache is required")
    saved_views = [
        load_candidate_cache(directory / f"{scene}.json", columns) for directory in cache_dirs
    ]
    score_features = []
    for saved in saved_views:
        score_matrix = np.asarray(
            [[float(item[2]) for item in group[:16]] for group in saved["candidates"]],
            dtype=np.float32,
        )
        score_features.append(_score_features(score_matrix))
    values: list[np.ndarray] = []
    for index, column in enumerate(columns):
        patch = cv2.imread(
            str(root / split / scene / "patches" / f"{column}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if patch is None:
            raise FileNotFoundError(f"Could not read {split}/{scene}/{column}.png")
        parts = [features[index] for features in score_features]
        if len(saved_views) > 1:
            parts.append(_agreement_features(saved_views, index))
        parts.append(patch_features(patch))
        values.append(np.concatenate(parts))
    return np.stack(values).astype(np.float32), saved_views


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
    cache_dir: Path | list[Path] | tuple[Path, ...],
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
    cache_dir: Path | list[Path] | tuple[Path, ...],
    train_cache_dir: Path | list[Path] | tuple[Path, ...],
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
        features, saved_views = scene_features(root, split, scene, columns, cache_dir)
        model = shared_model or train_classifier(
            root, train_cache_dir, regularization, threshold, excluded_scene=scene
        )
        probabilities = model.probabilities(features)
        for index, column in enumerate(columns):
            current = parse_cell(row[column])
            if current is not None and current[2] == 1:
                continue
            if probabilities[index] >= model.threshold:
                # Cache order is deliberate: the first view supplies final
                # coordinates while every view contributes confidence.
                best = saved_views[0]["candidates"][index][0]
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
    parser.add_argument(
        "--cache-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more prediction caches; the first supplies final coordinates",
    )
    parser.add_argument(
        "--train-cache-dir",
        type=Path,
        nargs="+",
        required=True,
        help="Training caches in the same view order as --cache-dir",
    )
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
        [path.resolve() for path in args.cache_dir],
        [path.resolve() for path in args.train_cache_dir],
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
