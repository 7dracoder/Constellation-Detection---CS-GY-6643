#!/usr/bin/env python3
"""Fit a constellation graph to localised query-patch centres.

This is the second stage of the competition pipeline.  It deliberately reads
an already validated localisation submission and only changes the membership
bit and the constellation label.  The fitter uses the supplied line drawings,
not scene names, patch counts, or memorised coordinates.

The expensive part - testing many similarity transforms against a small point
cloud - uses CUDA when PyTorch is available.  The NumPy fallback is intended
for small sanity checks, while the Cloud Bursting GPU job uses the CUDA path.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from constellation_pipeline import (
    Prediction,
    format_cell,
    parse_cell,
    parse_truth_rows,
    patch_columns,
    validate_submission,
)


PATTERN_NODE_MIN_AREA = 20
PATTERN_NODE_MAX_AREA = 200
CLUSTER_RADIUS = 18.0
MAX_TRANSFORMS_PER_PATTERN = 12_000


@dataclass(frozen=True)
class Pattern:
    name: str
    points: np.ndarray


@dataclass(frozen=True)
class Fit:
    pattern: Pattern
    support: int
    coverage: int
    mean_error: float
    tolerance: float
    mapped_points: np.ndarray

    @property
    def quality(self) -> float:
        """A size-normalised score used only to rank model hypotheses."""
        if self.support == 0:
            return float("-inf")
        fraction = self.support / min(len(self.mapped_points), len(self.pattern.points))
        return float(self.support + 0.25 * fraction + 0.03 * self.coverage - self.mean_error / self.tolerance)


@dataclass(frozen=True)
class MembershipClassifier:
    """Tiny regularised classifier for a query's target-figure likelihood.

    It is intentionally a shallow model: the training set contains only the
    supplied labelled query patches, so a large CNN would merely memorise the
    three training scenes.  The features describe local point-source contrast,
    not a scene name, image coordinate, or constellation identity.
    """

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    threshold: float

    def probability(self, patch: np.ndarray) -> float:
        values = (patch_features(patch)[None] - self.mean) / self.scale
        design = np.concatenate((np.ones((1, 1), dtype=np.float32), values), axis=1)
        score = float(np.clip(design @ self.weights, -30.0, 30.0)[0])
        return 1.0 / (1.0 + math.exp(-score))


def patch_features(image: np.ndarray) -> np.ndarray:
    """Brightness/contrast features robust to a small query-centre offset."""
    values = image.astype(np.float32)
    centre = values[13:19, 13:19]
    return np.asarray(
        (
            values.max(),
            np.percentile(values, 99),
            np.percentile(values, 98),
            centre.mean(),
            centre.max(),
            values.mean(),
            values.std(),
            np.mean(values >= 245),
            np.mean(centre >= 245),
        ),
        dtype=np.float32,
    )


def macro_f1(labels: np.ndarray, predicted: np.ndarray) -> float:
    values = []
    for target in (False, True):
        tp = int(np.sum((labels == target) & (predicted == target)))
        fp = int(np.sum((labels != target) & (predicted == target)))
        fn = int(np.sum((labels == target) & (predicted != target)))
        values.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(values))


def train_membership_classifier(root: Path) -> MembershipClassifier:
    """Fit only on provided training query patches that are known present."""
    truth = parse_truth_rows(root)
    features: list[np.ndarray] = []
    labels: list[int] = []
    for scene, values in truth.items():
        patch_dir = root / "train" / scene / "patches"
        for column, target in values.items():
            if target is None:
                continue
            patch = cv2.imread(str(patch_dir / f"{column}.png"), cv2.IMREAD_GRAYSCALE)
            if patch is None:
                raise FileNotFoundError(f"Could not read labelled patch {scene}/{column}")
            features.append(patch_features(patch))
            labels.append(target[2])
    matrix = np.stack(features)
    label = np.asarray(labels, dtype=np.float32)
    mean = matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, keepdims=True) + 1e-5
    design = np.concatenate((np.ones((len(matrix), 1), dtype=np.float32), (matrix - mean) / scale), axis=1)
    weights = np.zeros(design.shape[1], dtype=np.float32)
    # Deterministic full-batch logistic regression with an L2 penalty.  It is
    # fast, reproducible, and does not add an unapproved package dependency.
    for _ in range(4_000):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = (design.T @ (probability - label)) / len(label)
        gradient[1:] += 0.012 * weights[1:]
        weights -= 0.05 * gradient
    probability = 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -30.0, 30.0)))
    candidates = np.unique(np.quantile(probability, np.linspace(0.05, 0.95, 91)))
    threshold = max(candidates, key=lambda value: macro_f1(label.astype(bool), probability >= value))
    print(
        f"membership calibration: n={len(label)} macro-F1="
        f"{macro_f1(label.astype(bool), probability >= threshold):.3f} threshold={threshold:.3f}",
        flush=True,
    )
    return MembershipClassifier(mean[0], scale[0], weights, float(threshold))


def extract_pattern(path: Path) -> Pattern:
    """Extract the white star nodes from a supplied RGBA line drawing."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Could not read pattern image: {path}")
    rgb = image[:, :, :3]
    # Node outlines are near-white while figure edges are green.
    white = (rgb.min(axis=2) > 180) & ((rgb.max(axis=2) - rgb.min(axis=2)) < 50)
    count, _, stats, centres = cv2.connectedComponentsWithStats(white.astype(np.uint8), 8)
    points = np.asarray(
        [
            centres[index]
            for index in range(1, count)
            if PATTERN_NODE_MIN_AREA <= stats[index, cv2.CC_STAT_AREA] <= PATTERN_NODE_MAX_AREA
        ],
        dtype=np.float32,
    )
    if len(points) < 2:
        raise ValueError(f"Pattern has fewer than two recoverable nodes: {path}")
    return Pattern(path.name.removesuffix("_pattern.png"), points)


def load_patterns(root: Path) -> list[Pattern]:
    return [extract_pattern(path) for path in sorted((root / "patterns").glob("*_pattern.png"))]


def cluster_predictions(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse repeated localiser hits while retaining query-to-cluster links."""
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int32)
    centres: list[np.ndarray] = []
    counts: list[int] = []
    mapping: list[int] = []
    for point in points.astype(np.float32):
        if not centres:
            centres.append(point.copy())
            counts.append(1)
            mapping.append(0)
            continue
        existing = np.asarray(centres)
        distances = np.linalg.norm(existing - point, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= CLUSTER_RADIUS:
            counts[nearest] += 1
            centres[nearest] = centres[nearest] + (point - centres[nearest]) / counts[nearest]
            mapping.append(nearest)
        else:
            centres.append(point.copy())
            counts.append(1)
            mapping.append(len(centres) - 1)
    return np.asarray(centres, dtype=np.float32), np.asarray(mapping, dtype=np.int32)


def upper_pairs(point_count: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(point_count, 1)


def candidate_scales(pattern_points: np.ndarray, observed_points: np.ndarray) -> np.ndarray:
    """Find likely image-to-pattern scales with a distance-ratio Hough vote."""
    pi, pj = upper_pairs(len(pattern_points))
    oi, oj = upper_pairs(len(observed_points))
    pattern_lengths = np.linalg.norm(pattern_points[pj] - pattern_points[pi], axis=1)
    observed_lengths = np.linalg.norm(observed_points[oj] - observed_points[oi], axis=1)
    ratios = (observed_lengths[:, None] / pattern_lengths[None, :]).ravel()
    ratios = ratios[(ratios > 0.25) & (ratios < 25.0)]
    if len(ratios) == 0:
        return np.empty(0, dtype=np.float32)
    edges = np.arange(math.log(0.25), math.log(25.0) + 0.10, 0.10)
    histogram, edges = np.histogram(np.log(ratios), bins=edges)
    best = np.argsort(histogram)[-5:][::-1]
    return np.exp((edges[best] + edges[best + 1]) / 2).astype(np.float32)


def candidate_transforms(pattern_points: np.ndarray, observed_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return similarity matrices and translations from compatible point pairs."""
    pi, pj = upper_pairs(len(pattern_points))
    oi, oj = upper_pairs(len(observed_points))
    pattern_delta = pattern_points[pj] - pattern_points[pi]
    observed_delta = observed_points[oj] - observed_points[oi]
    pattern_length = np.linalg.norm(pattern_delta, axis=1)
    observed_length = np.linalg.norm(observed_delta, axis=1)
    matrices: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    for coarse_scale in candidate_scales(pattern_points, observed_points):
        ratio = observed_length[:, None] / pattern_length[None, :]
        compatible = np.argwhere(np.abs(np.log(ratio / coarse_scale)) <= 0.075)
        # The same transform can vote in adjacent Hough bins.  Evenly sampling
        # keeps a GPU batch bounded without privileging a filename or patch ID.
        if len(compatible) > MAX_TRANSFORMS_PER_PATTERN // 4:
            chosen = np.linspace(0, len(compatible) - 1, MAX_TRANSFORMS_PER_PATTERN // 4).round().astype(int)
            compatible = compatible[chosen]
        for reflected in (False, True):
            source = pattern_points * np.array([1.0, -1.0], dtype=np.float32) if reflected else pattern_points
            delta = source[pj] - source[pi]
            for reverse_observed in (False, True):
                source_pair = compatible[:, 1]
                observed_pair = compatible[:, 0]
                left = oi[observed_pair] if reverse_observed else oj[observed_pair]
                right = oj[observed_pair] if reverse_observed else oi[observed_pair]
                source_vector = delta[source_pair]
                target_vector = observed_points[right] - observed_points[left]
                source_norm = np.linalg.norm(source_vector, axis=1)
                target_norm = np.linalg.norm(target_vector, axis=1)
                cosine = np.sum(source_vector * target_vector, axis=1) / (source_norm * target_norm)
                sine = (
                    source_vector[:, 0] * target_vector[:, 1]
                    - source_vector[:, 1] * target_vector[:, 0]
                ) / (source_norm * target_norm)
                scale = target_norm / source_norm
                rotation = np.stack(
                    (
                        np.stack((cosine, -sine), axis=1),
                        np.stack((sine, cosine), axis=1),
                    ),
                    axis=1,
                )
                linear = rotation * scale[:, None, None]
                source_anchor = source[pi[source_pair]]
                offsets.append(
                    observed_points[left] - np.einsum("bij,bj->bi", linear, source_anchor)
                )
                # `linear` acts on the optionally reflected coordinates.  Fold
                # that reflection into the returned transform so the evaluator
                # always receives the original pattern coordinates.
                if reflected:
                    linear = linear @ np.diag(np.array([1.0, -1.0], dtype=np.float32))
                matrices.append(linear)
    if not matrices:
        return np.empty((0, 2, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    matrix = np.concatenate(matrices, axis=0).astype(np.float32)
    offset = np.concatenate(offsets, axis=0).astype(np.float32)
    if len(matrix) > MAX_TRANSFORMS_PER_PATTERN:
        chosen = np.linspace(0, len(matrix) - 1, MAX_TRANSFORMS_PER_PATTERN).round().astype(int)
        matrix, offset = matrix[chosen], offset[chosen]
    return matrix, offset


def evaluate_numpy(pattern: Pattern, observed: np.ndarray, matrices: np.ndarray, offsets: np.ndarray) -> Optional[Fit]:
    best: Optional[Fit] = None
    for start in range(0, len(matrices), 256):
        matrix = matrices[start : start + 256]
        offset = offsets[start : start + 256]
        mapped = np.einsum("bij,pj->bpi", matrix, pattern.points) + offset[:, None, :]
        distances = np.linalg.norm(mapped[:, :, None, :] - observed[None, None, :, :], axis=3)
        nearest_observed = distances.min(axis=1)
        nearest_pattern = distances.min(axis=2)
        scale = np.sqrt(np.abs(np.linalg.det(matrix)))
        tolerance = np.maximum(28.0, 7.5 * scale)
        inliers = nearest_observed <= tolerance[:, None]
        support = inliers.sum(axis=1)
        residual = np.where(inliers, nearest_observed, 0.0).sum(axis=1) / np.maximum(support, 1)
        coverage = (nearest_pattern <= tolerance[:, None]).sum(axis=1)
        for index in np.argsort(support)[-8:]:
            fit = Fit(pattern, int(support[index]), int(coverage[index]), float(residual[index]), float(tolerance[index]), mapped[index])
            if best is None or fit.quality > best.quality:
                best = fit
    return best


def evaluate_torch(pattern: Pattern, observed: np.ndarray, matrices: np.ndarray, offsets: np.ndarray, device: str) -> Optional[Fit]:
    import torch

    observed_t = torch.as_tensor(observed, device=device)
    pattern_t = torch.as_tensor(pattern.points, device=device)
    best: Optional[Fit] = None
    for start in range(0, len(matrices), 2048):
        matrix = torch.as_tensor(matrices[start : start + 2048], device=device)
        offset = torch.as_tensor(offsets[start : start + 2048], device=device)
        mapped = torch.einsum("bij,pj->bpi", matrix, pattern_t) + offset[:, None, :]
        distances = torch.cdist(mapped, observed_t)
        nearest_observed = distances.amin(dim=1)
        nearest_pattern = distances.amin(dim=2)
        scale = torch.sqrt(torch.abs(torch.linalg.det(matrix)))
        tolerance = torch.clamp(7.5 * scale, min=28.0)
        inliers = nearest_observed <= tolerance[:, None]
        support = inliers.sum(dim=1)
        residual = torch.where(inliers, nearest_observed, torch.zeros_like(nearest_observed)).sum(dim=1) / support.clamp_min(1)
        coverage = (nearest_pattern <= tolerance[:, None]).sum(dim=1)
        candidate = torch.argsort(support)[-8:].tolist()
        for index in candidate:
            fit = Fit(
                pattern,
                int(support[index].item()),
                int(coverage[index].item()),
                float(residual[index].item()),
                float(tolerance[index].item()),
                mapped[index].detach().cpu().numpy(),
            )
            if best is None or fit.quality > best.quality:
                best = fit
    return best


def fit_pattern(pattern: Pattern, observed: np.ndarray, device: str) -> Optional[Fit]:
    if len(observed) < 2:
        return None
    matrices, offsets = candidate_transforms(pattern.points, observed)
    if len(matrices) == 0:
        return None
    if device == "cuda":
        return evaluate_torch(pattern, observed, matrices, offsets, device)
    return evaluate_numpy(pattern, observed, matrices, offsets)


def choose_fit(patterns: list[Pattern], observed: np.ndarray, device: str) -> tuple[Optional[Fit], Optional[Fit]]:
    # Two points always define a similarity transform, so two-point fits carry
    # no identification evidence and must not outrank a real partial graph.
    fits = [
        fit
        for pattern in patterns
        if (fit := fit_pattern(pattern, observed, device)) is not None and fit.support >= 4
    ]
    fits.sort(key=lambda fit: fit.quality, reverse=True)
    return (fits[0], fits[1]) if len(fits) >= 2 else (fits[0] if fits else None, None)


def fit_membership(fit: Fit, clusters: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(clusters[:, None, :] - fit.mapped_points[None, :, :], axis=2)
    return distances.min(axis=1) <= fit.tolerance


def local_brightness(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Return a small-window peak, robust to a one-pixel centre offset."""
    values = []
    height, width = image.shape
    for x, y in points:
        cx, cy = int(round(float(x))), int(round(float(y)))
        left, right = max(0, cx - 1), min(width, cx + 2)
        top, bottom = max(0, cy - 1), min(height, cy + 2)
        values.append(float(image[top:bottom, left:right].max()))
    return np.asarray(values, dtype=np.float32)


def is_confident(best: Optional[Fit], runner_up: Optional[Fit], point_count: int) -> bool:
    if best is None or best.support < 4:
        return False
    coverage = best.support / min(len(best.pattern.points), point_count)
    if coverage < 0.38:
        return False
    if runner_up is None:
        return True
    # A name is withheld when the two structural explanations are effectively tied.
    return best.quality - runner_up.quality >= 0.16


def refine_submission(
    root: Path, input_path: Path, output_path: Path, device: str, brightness_floor: float
) -> None:
    validate_submission(root, input_path)
    patterns = load_patterns(root)
    membership_classifier = train_membership_classifier(root)
    with input_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        fieldnames = list(csv.DictReader((root / "sample_submission.csv").open(newline="", encoding="utf-8")).fieldnames or [])
    for row in rows:
        active = patch_columns(int(row["n_patches"]))
        locations: list[tuple[int, int]] = []
        present_columns: list[str] = []
        for column in active:
            value = parse_cell(row[column])
            if value is not None:
                locations.append((value[0], value[1]))
                present_columns.append(column)
        clusters, point_to_cluster = cluster_predictions(np.asarray(locations, dtype=np.float32))
        image_path = root / "validation" / row["Id"] / f"{row['Id']}_image.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read validation image: {image_path}")
        query_probabilities = []
        for column in present_columns:
            patch = cv2.imread(
                str(root / "validation" / row["Id"] / "patches" / f"{column}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            if patch is None:
                raise FileNotFoundError(f"Could not read validation patch {row['Id']}/{column}")
            query_probabilities.append(membership_classifier.probability(patch))
        cluster_probabilities = np.zeros(len(clusters), dtype=np.float32)
        if len(point_to_cluster):
            np.maximum.at(cluster_probabilities, point_to_cluster, np.asarray(query_probabilities))
        likely_figure = cluster_probabilities >= membership_classifier.threshold
        bright = local_brightness(image, clusters) >= brightness_floor
        # The learned, patch-only score excludes distractor constellations before
        # graph fitting.  Brightness remains a label-free fallback for a scene
        # whose target stars happen to sit below the learned global threshold.
        if int(likely_figure.sum()) >= 4:
            model_points = clusters[likely_figure]
            selection_source = "query"
        elif int(bright.sum()) >= 4:
            model_points = clusters[bright]
            selection_source = "brightness"
        else:
            model_points = clusters
            selection_source = "all"
        best, runner_up = choose_fit(patterns, model_points, device)
        if not is_confident(best, runner_up, len(model_points)):
            row["constellation"] = "unknown"
            continue
        assert best is not None
        membership = fit_membership(best, clusters)
        row["constellation"] = best.pattern.name
        for column, cluster_index in zip(present_columns, point_to_cluster):
            x, y, _ = parse_cell(row[column])  # type: ignore[misc]
            row[column] = format_cell(Prediction(x=x, y=y, m=int(membership[cluster_index])))
        print(
            f"{row['Id']}: {best.pattern.name} support={best.support} coverage={best.coverage} "
            f"error={best.mean_error:.1f} selector={selection_source} "
            f"runner={runner_up.pattern.name if runner_up else 'none'}",
            flush=True,
        )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    validate_submission(root, output_path)
    print(f"wrote {output_path}")


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--brightness-floor", type=float, default=252.0)
    args = parser.parse_args()
    root = args.root.resolve()
    device = resolve_device(args.device)
    if device == "cuda":
        print("using CUDA for structural hypothesis scoring")
    else:
        print("using CPU structural hypothesis scoring")
    refine_submission(root, args.input.resolve(), args.output.resolve(), device, args.brightness_floor)


if __name__ == "__main__":
    main()
