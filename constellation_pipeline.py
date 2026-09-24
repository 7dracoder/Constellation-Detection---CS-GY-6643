#!/usr/bin/env python3
"""Reproducible baseline for the CS-GY 6643 constellation competition.

The pipeline never uses scene identifiers, patch counts, or memorized locations
as prediction features.  It uses only pixels in each scene, the supplied
reference patterns, and labels in train_ground_truth.csv for calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


PATCH_SIZE = 32
PATCH_RADIUS = PATCH_SIZE // 2
CELL_RE = re.compile(r"-?\d+")


@dataclass(frozen=True)
class Prediction:
    x: int
    y: int
    m: int = 0
    score: float = 0.0
    margin: float = 0.0


@dataclass
class MatcherConfig:
    background_sigma: float = 10.0
    response_sigma_small: float = 0.8
    response_sigma_large: float = 2.2
    response_percentile: float = 99.20
    max_candidates: int = 15000
    min_candidate_distance: int = 3
    coarse_factor: int = 3
    coarse_peaks_per_transform: int = 5
    max_global_candidates: int = 120
    candidate_refinement_limit: int = 120
    variants_per_candidate: int = 2
    fine_top_pairs: int = 8
    intensity_weight: float = 0.45
    angles: int = 18
    scales: tuple[float, ...] = (0.76, 0.86, 0.96, 1.00, 1.12, 1.28)
    refine_radius: int = 8
    # The threshold is learned from train labels by the calibrate command.
    presence_threshold: float = 0.46
    margin_threshold: float = 0.00
    # Optional matched denoising applied to both the scene and query before
    # template matching.  The default preserves the established raw path.
    denoise_method: str = "none"
    denoise_sigma: float = 0.0
    denoise_kernel: int = 3


def read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image.astype(np.float32)


def denoise_for_matching(image: np.ndarray, config: MatcherConfig) -> np.ndarray:
    """Apply the same conservative denoiser to a scene and its query patches."""
    method = config.denoise_method.lower()
    if method == "none":
        return image
    if method == "gaussian":
        if config.denoise_sigma <= 0.0:
            raise ValueError("Gaussian denoising requires denoise_sigma > 0")
        return cv2.GaussianBlur(image, (0, 0), config.denoise_sigma).astype(np.float32)
    if method == "median":
        kernel = int(config.denoise_kernel)
        if kernel < 3 or kernel % 2 == 0:
            raise ValueError("Median denoising requires an odd denoise_kernel >= 3")
        return cv2.medianBlur(image, kernel).astype(np.float32)
    raise ValueError(f"Unknown denoise_method: {config.denoise_method!r}")


def parse_cell(value: str) -> Optional[tuple[int, int, int]]:
    if value.strip() == "-1":
        return None
    values = [int(v) for v in CELL_RE.findall(value)]
    if len(values) < 2:
        raise ValueError(f"Invalid prediction cell: {value!r}")
    return values[0], values[1], values[2] if len(values) >= 3 else 0


def format_cell(prediction: Optional[Prediction]) -> str:
    if prediction is None:
        return "-1"
    return f"({prediction.x}, {prediction.y}, {prediction.m})"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def patch_columns(n_patches: int) -> list[str]:
    return [f"patch_{index:02d}" for index in range(1, n_patches + 1)]


def normalize_image(image: np.ndarray, background_sigma: float) -> np.ndarray:
    """Suppress slow illumination while retaining point-source structure."""
    background = cv2.GaussianBlur(image, (0, 0), background_sigma)
    high_pass = image - background
    local_scale = cv2.GaussianBlur(np.abs(high_pass), (0, 0), background_sigma)
    normalized = high_pass / (local_scale + 4.0)
    return np.clip(normalized, -8.0, 8.0).astype(np.float32)


def blob_response(normalized: np.ndarray, config: MatcherConfig) -> np.ndarray:
    small = cv2.GaussianBlur(normalized, (0, 0), config.response_sigma_small)
    large = cv2.GaussianBlur(normalized, (0, 0), config.response_sigma_large)
    return small - large


def non_maximum_points(
    response: np.ndarray, config: MatcherConfig, border: int = PATCH_RADIUS + 4
) -> tuple[np.ndarray, np.ndarray]:
    """Return star-centre candidates as xy coordinates and response strengths."""
    size = 2 * config.min_candidate_distance + 1
    local_max = response == cv2.dilate(response, np.ones((size, size), np.uint8))
    threshold = float(np.percentile(response, config.response_percentile))
    valid = local_max & (response >= threshold)
    valid[:border, :] = False
    valid[-border:, :] = False
    valid[:, :border] = False
    valid[:, -border:] = False
    ys, xs = np.nonzero(valid)
    strengths = response[ys, xs]
    order = np.argsort(strengths)[::-1][: config.max_candidates]
    points = np.column_stack((xs[order], ys[order])).astype(np.int32)
    return points, strengths[order].astype(np.float32)


def padded_extract(image: np.ndarray, centres: np.ndarray, radius: int = PATCH_RADIUS) -> np.ndarray:
    """Extract fixed centred windows; reflection padding protects image borders."""
    padded = cv2.copyMakeBorder(image, radius, radius, radius, radius, cv2.BORDER_REFLECT101)
    return extract_from_padded(padded, centres, radius)


def extract_from_padded(
    padded: np.ndarray, centres: np.ndarray, radius: int = PATCH_RADIUS
) -> np.ndarray:
    patches = np.empty((len(centres), 2 * radius, 2 * radius), dtype=np.float32)
    for index, (x, y) in enumerate(centres):
        patches[index] = padded[y : y + 2 * radius, x : x + 2 * radius]
    return patches


def circular_mask(size: int = PATCH_SIZE) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    centre = (size - 1) / 2.0
    return ((xx - centre) ** 2 + (yy - centre) ** 2 <= (size / 2.1) ** 2).astype(np.float32)


MASK = circular_mask()


def normalized_vectors(patches: np.ndarray) -> np.ndarray:
    """Illumination-invariant vectors for normalized cross-correlation."""
    weighted = patches * MASK
    mean = weighted.sum(axis=(1, 2), keepdims=True) / MASK.sum()
    vectors = ((patches - mean) * MASK).reshape(len(patches), -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-6)


def gradient_vectors(patches: np.ndarray) -> np.ndarray:
    """Normalized Sobel vectors that discount smooth brightness drift.

    The query transform is explicitly undone before this comparison, so gradient
    direction remains useful rather than being discarded as a rotation-sensitive
    feature.
    """
    grad_x = np.stack([cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3) for patch in patches])
    grad_y = np.stack([cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3) for patch in patches])
    return np.concatenate((normalized_vectors(grad_x), normalized_vectors(grad_y)), axis=1) / math.sqrt(2)


def structural_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Small-window SSIM, computed after the candidate's fine alignment."""
    mean_first = cv2.GaussianBlur(first, (0, 0), 1.5)
    mean_second = cv2.GaussianBlur(second, (0, 0), 1.5)
    var_first = cv2.GaussianBlur(first * first, (0, 0), 1.5) - mean_first * mean_first
    var_second = cv2.GaussianBlur(second * second, (0, 0), 1.5) - mean_second * mean_second
    covariance = cv2.GaussianBlur(first * second, (0, 0), 1.5) - mean_first * mean_second
    c1, c2 = 6.5, 58.5
    ssim_map = ((2 * mean_first * mean_second + c1) * (2 * covariance + c2)) / (
        (mean_first * mean_first + mean_second * mean_second + c1)
        * (var_first + var_second + c2)
    )
    return float(np.mean(ssim_map * MASK) / np.mean(MASK))


def rotation_invariant_descriptor(patches: np.ndarray) -> np.ndarray:
    """Compact polar brightness descriptor used only to shortlist candidates."""
    size = patches.shape[-1]
    yy, xx = np.mgrid[:size, :size]
    centre = (size - 1) / 2.0
    radii = np.hypot(xx - centre, yy - centre)
    angles = np.mod(np.arctan2(yy - centre, xx - centre), 2 * np.pi)
    features: list[np.ndarray] = []
    positive = np.maximum(patches, 0.0)
    for low, high in ((0, 3), (3, 6), (6, 10), (10, 14), (14, 17)):
        ring = (radii >= low) & (radii < high)
        # Rotation changes phase, not the magnitude of angular Fourier terms.
        bins = []
        for bin_index in range(12):
            start = 2 * np.pi * bin_index / 12
            end = 2 * np.pi * (bin_index + 1) / 12
            sector = ring & (angles >= start) & (angles < end)
            bins.append(positive[:, sector].mean(axis=1))
        spectrum = np.abs(np.fft.rfft(np.stack(bins, axis=1), axis=1))[:, :4]
        features.append(spectrum)
    descriptor = np.concatenate(features, axis=1).astype(np.float32)
    descriptor /= np.maximum(np.linalg.norm(descriptor, axis=1, keepdims=True), 1e-6)
    return descriptor


def transform_query(query: np.ndarray, angle: float, scale: float) -> np.ndarray:
    centre = ((PATCH_SIZE - 1) / 2.0, (PATCH_SIZE - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, scale)
    return cv2.warpAffine(
        query,
        matrix,
        (PATCH_SIZE, PATCH_SIZE),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )


class SceneMatcher:
    """Coarse-to-fine matcher with no training-scene-specific information.

    Direct star detection loses too many dim patch centres in this data.  The
    coarse stage therefore searches the whole image at reduced resolution, then
    scores only a small set of full-resolution candidates.
    """

    def __init__(self, image_path: Path, config: MatcherConfig):
        self.config = config
        self.image = read_grayscale(image_path)
        self.matching_image = denoise_for_matching(self.image, config)
        self.padded_image = cv2.copyMakeBorder(
            self.matching_image,
            PATCH_RADIUS,
            PATCH_RADIUS,
            PATCH_RADIUS,
            PATCH_RADIUS,
            cv2.BORDER_REFLECT101,
        )
        height, width = self.image.shape
        self.coarse_shape = (width // config.coarse_factor, height // config.coarse_factor)
        self.coarse_image = cv2.resize(
            self.matching_image, self.coarse_shape, interpolation=cv2.INTER_AREA
        )
        self.last_candidate_count = 0

    def _query_variants(self, query: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
        angles = np.linspace(0, 360, self.config.angles, endpoint=False)
        variants = []
        transforms = []
        for scale in self.config.scales:
            for angle in angles:
                variants.append(transform_query(query, float(angle), scale))
                transforms.append((float(angle), float(scale)))
        return np.stack(variants), transforms

    def _coarse_candidates(self, variants: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        factor = self.config.coarse_factor
        small_size = max(6, round(PATCH_SIZE / factor))
        raw_candidates: list[tuple[float, int, int, int]] = []
        peak_kernel = np.ones((7, 7), np.uint8)
        for variant_index, variant in enumerate(variants):
            template = cv2.resize(
                variant, (small_size, small_size), interpolation=cv2.INTER_AREA
            )
            response = cv2.matchTemplate(self.coarse_image, template, cv2.TM_CCOEFF_NORMED)
            peaks = response == cv2.dilate(response, peak_kernel)
            ys, xs = np.nonzero(peaks)
            values = response[ys, xs]
            keep = min(self.config.coarse_peaks_per_transform, len(values))
            if keep == 0:
                continue
            selected = np.argpartition(values, -keep)[-keep:]
            for index in selected:
                # matchTemplate gives a source-window top-left; convert to its centre.
                x = int(xs[index] * factor + PATCH_RADIUS)
                y = int(ys[index] * factor + PATCH_RADIUS)
                raw_candidates.append((float(values[index]), x, y, variant_index))
        raw_candidates.sort(reverse=True)
        selected_points: list[tuple[int, int]] = []
        selected_variants: list[int] = []
        minimum_distance = (factor * 3) ** 2
        for _, x, y, variant_index in raw_candidates:
            if all((x - old_x) ** 2 + (y - old_y) ** 2 > minimum_distance for old_x, old_y in selected_points):
                selected_points.append((x, y))
                selected_variants.append(variant_index)
            if len(selected_points) >= self.config.max_global_candidates:
                break
        if not selected_points:
            raise RuntimeError("Coarse matching produced no candidates")
        return np.asarray(selected_points, dtype=np.int32), np.asarray(selected_variants, dtype=np.int32)

    def _coarse_neighbourhood(self, variant_index: int) -> list[int]:
        """Return fine transforms near the one that created a coarse peak."""
        scale_index, angle_index = divmod(int(variant_index), self.config.angles)
        nearby = []
        for scale_offset in (-1, 0, 1):
            candidate_scale = scale_index + scale_offset
            if not 0 <= candidate_scale < len(self.config.scales):
                continue
            for angle_offset in (-1, 0, 1):
                candidate_angle = (angle_index + angle_offset) % self.config.angles
                nearby.append(candidate_scale * self.config.angles + candidate_angle)
        return nearby

    def _refine(
        self, vector: np.ndarray, x: int, y: int
    ) -> tuple[int, int, float]:
        radius = self.config.refine_radius
        offsets = np.array(
            [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)],
            dtype=np.int32,
        )
        centres = offsets + np.array([x, y], dtype=np.int32)
        patches = extract_from_padded(self.padded_image, centres)
        scores = normalized_vectors(patches) @ vector
        index = int(np.argmax(scores))
        return int(centres[index, 0]), int(centres[index, 1]), float(scores[index])

    def match_candidates(
        self, query_path: Path, limit: int = 16, minimum_separation: float = 18.0
    ) -> list[Prediction]:
        """Return spatially distinct, fine-scored locations for one query.

        The one-best location is enough for the simple baseline, but it loses
        useful information in a dense star field: the correct location is
        often in the top few visual matches.  The joint geometric solver uses
        this bounded shortlist together with the supplied constellation graph
        to resolve those ambiguities.  Candidates are still computed solely
        from the query patch and its own scene.
        """
        query = denoise_for_matching(read_grayscale(query_path), self.config)
        if query.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Unexpected patch shape {query.shape} in {query_path}")
        if limit < 1:
            raise ValueError("limit must be positive")
        variants, _ = self._query_variants(query)
        candidates, coarse_variants = self._coarse_candidates(variants)
        self.last_candidate_count = len(candidates)
        candidate_patches = extract_from_padded(self.padded_image, candidates)
        variant_vectors = normalized_vectors(variants)
        candidate_vectors = normalized_vectors(candidate_patches)
        intensity_similarity = variant_vectors @ candidate_vectors.T
        gradient_similarity = gradient_vectors(variants) @ gradient_vectors(candidate_patches).T
        similarity = 0.70 * intensity_similarity + 0.30 * gradient_similarity
        candidate_order = np.argsort(similarity.max(axis=0))[::-1][
            : self.config.candidate_refinement_limit
        ]
        refined: list[tuple[float, int, int]] = []
        for candidate_index in candidate_order:
            candidate_x, candidate_y = candidates[candidate_index]
            # The coarse search already tells us which rotation/scale produced
            # this location.  Preserve it and immediate transform neighbours:
            # that cue is more reliable than an unaligned crop similarity in a
            # dense, repetitive star field.
            shortlist = np.argsort(similarity[:, candidate_index])[::-1][
                : self.config.variants_per_candidate
            ]
            variant_order = list(
                dict.fromkeys(
                    self._coarse_neighbourhood(int(coarse_variants[candidate_index]))
                    + [int(item) for item in shortlist]
                )
            )
            for variant_index in variant_order:
                refined_x, refined_y, intensity_score = self._refine(
                    variant_vectors[variant_index], int(candidate_x), int(candidate_y)
                )
                refined_patch = extract_from_padded(
                    self.padded_image,
                    np.array([[refined_x, refined_y]], dtype=np.int32),
                )
                gradient_score = (
                    gradient_vectors(variants[variant_index : variant_index + 1])
                    @ gradient_vectors(refined_patch).T
                ).item()
                ssim_score = structural_similarity(variants[variant_index], refined_patch[0])
                score = (
                    self.config.intensity_weight * intensity_score
                    + 0.20 * gradient_score
                    + (0.80 - self.config.intensity_weight) * ssim_score
                )
                refined.append((score, refined_x, refined_y))
        if not refined:
            raise RuntimeError(f"No fine candidates for {query_path}")
        refined.sort(reverse=True)
        selected: list[tuple[float, int, int]] = []
        separation_squared = minimum_separation**2
        for score, x, y in refined:
            if all((x - old_x) ** 2 + (y - old_y) ** 2 > separation_squared for _, old_x, old_y in selected):
                selected.append((score, x, y))
            if len(selected) >= limit:
                break
        result: list[Prediction] = []
        for index, (score, x, y) in enumerate(selected):
            next_score = selected[index + 1][0] if index + 1 < len(selected) else float(
                np.partition(similarity.ravel(), -2)[-2]
            )
            result.append(
                Prediction(x=int(x), y=int(y), score=float(score), margin=float(score - next_score))
            )
        return result

    def match(self, query_path: Path) -> Prediction:
        """Return the ordinary one-best prediction for a query patch."""
        return self.match_candidates(query_path, limit=1)[0]


def parse_truth_rows(root: Path) -> dict[str, dict[str, Optional[tuple[int, int, int]]]]:
    result: dict[str, dict[str, Optional[tuple[int, int, int]]]] = {}
    for row in read_csv_rows(root / "train_ground_truth.csv"):
        n_patches = int(row["n_patches"])
        result[row["Id"]] = {column: parse_cell(row[column]) for column in patch_columns(n_patches)}
    return result


def image_files(directory: Path, pattern: str) -> list[Path]:
    """Return real image files, excluding macOS AppleDouble metadata sidecars."""
    return sorted(path for path in directory.glob(pattern) if not path.name.startswith("._"))


def iter_scene_patches(root: Path, split: str) -> Iterable[tuple[str, Path, list[Path]]]:
    for scene_dir in sorted((root / split).iterdir()):
        if not scene_dir.is_dir():
            continue
        image_paths = image_files(scene_dir, "*_image.png")
        if len(image_paths) != 1:
            raise ValueError(f"Expected exactly one sky image in {scene_dir}")
        patches = sorted((scene_dir / "patches").glob("patch_*.png"))
        yield scene_dir.name, image_paths[0], patches


def fit_presence_threshold(
    records: list[tuple[Prediction, bool]], config: MatcherConfig
) -> MatcherConfig:
    scores = np.array([record[0].score for record in records])
    labels = np.array([record[1] for record in records], dtype=bool)
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101)))
    best = (-1.0, config.presence_threshold)
    for threshold in candidates:
        predicted = scores >= threshold
        values = []
        for positive_class in (False, True):
            tp = int(np.sum((predicted == positive_class) & (labels == positive_class)))
            fp = int(np.sum((predicted == positive_class) & (labels != positive_class)))
            fn = int(np.sum((predicted != positive_class) & (labels == positive_class)))
            denom = 2 * tp + fp + fn
            values.append(0.0 if denom == 0 else 2 * tp / denom)
        macro_f1 = float(np.mean(values))
        if macro_f1 > best[0]:
            best = (macro_f1, float(threshold))
    config.presence_threshold = best[1]
    print(f"presence calibration: macro-F1={best[0]:.4f}, threshold={best[1]:.4f}")
    return config


def score_train_predictions(
    root: Path, predictions: dict[str, dict[str, Optional[Prediction]]]
) -> dict[str, float]:
    """Score the localisation components only.

    This is not the competition metric: it has no access to a predicted
    constellation name and therefore omits the 0.30 identity term.  Use
    ``evaluate.py`` to compare configurations.
    """
    truth_rows = read_csv_rows(root / "train_ground_truth.csv")
    scene_scores = []
    for row in truth_rows:
        scene = row["Id"]
        n_patches = int(row["n_patches"])
        labels = []
        predicted_present = []
        localization_rewards = []
        truth_figure = []
        predicted_points = []
        for column in patch_columns(n_patches):
            truth = parse_cell(row[column])
            prediction = predictions[scene][column]
            labels.append(truth is not None)
            predicted_present.append(prediction is not None)
            if truth is not None:
                reward = 0.0
                if prediction is not None:
                    distance = math.dist((truth[0], truth[1]), (prediction.x, prediction.y))
                    reward = max(0.0, min(1.0, (36.0 - distance) / 24.0))
                    if distance <= 12:
                        reward = 1.0
                localization_rewards.append(reward)
                if truth[2] == 1:
                    truth_figure.append((truth[0], truth[1]))
            if prediction is not None:
                predicted_points.append((prediction.x, prediction.y))
        f1s = []
        for positive_class in (False, True):
            actual = np.array(labels) == positive_class
            guessed = np.array(predicted_present) == positive_class
            tp = int(np.sum(actual & guessed))
            fp = int(np.sum(~actual & guessed))
            fn = int(np.sum(actual & ~guessed))
            f1s.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
        geom_rewards = []
        remaining = list(predicted_points)
        for x, y in truth_figure:
            if not remaining:
                geom_rewards.append(0.0)
                continue
            distances = [math.dist((x, y), point) for point in remaining]
            index = int(np.argmin(distances))
            distance = distances[index]
            remaining.pop(index)
            geom_rewards.append(1.0 if distance <= 12 else max(0.0, (36.0 - distance) / 24.0))
        # This helper only sees localisation predictions, never a predicted
        # constellation, so it cannot evaluate the 0.30 identity term at all.
        # It previously substituted ``row["constellation"] == "unknown"``,
        # which awarded that term for declining to name a scene that does
        # have a label, and so reported a number that rose as identification
        # got worse.  The identity term is dropped here and the remaining
        # components are renormalised; use ``evaluate.py`` for the full
        # weighted score.
        score = (
            0.25 * float(np.mean(f1s))
            + 0.20 * float(np.mean(localization_rewards))
            + 0.25 * float(np.mean(geom_rewards))
        ) / 0.70
        scene_scores.append(score)
    return {"localisation_only_mean": float(np.mean(scene_scores))}


def calibrate(root: Path, output_config: Path, config: MatcherConfig) -> None:
    truth = parse_truth_rows(root)
    records: list[tuple[Prediction, bool]] = []
    raw_predictions: dict[str, dict[str, Prediction]] = {}
    for scene, image_path, patches in iter_scene_patches(root, "train"):
        print(f"indexing {scene} ({len(patches)} patches)")
        matcher = SceneMatcher(image_path, config)
        raw_predictions[scene] = {}
        for patch_path in patches:
            prediction = matcher.match(patch_path)
            raw_predictions[scene][patch_path.stem] = prediction
            label = truth[scene][patch_path.stem] is not None
            location_error = None
            if label:
                x, y, _ = truth[scene][patch_path.stem]  # type: ignore[misc]
                location_error = math.dist((x, y), (prediction.x, prediction.y))
            records.append((prediction, label))
            print(
                f"  {patch_path.name}: score={prediction.score:.3f} margin={prediction.margin:.3f} "
                f"truth={'present' if label else 'absent'}"
                + (f" error={location_error:.1f}px" if location_error is not None else "")
            )
    config = fit_presence_threshold(records, config)
    thresholded = {
        scene: {
            column: prediction if prediction.score >= config.presence_threshold else None
            for column, prediction in scene_predictions.items()
        }
        for scene, scene_predictions in raw_predictions.items()
    }
    print(
        "localisation-only train score (NOT a leaderboard estimate; run evaluate.py):",
        score_train_predictions(root, thresholded),
    )
    output_config.write_text(json.dumps(asdict(config), indent=2) + "\n")
    print(f"wrote {output_config}")


def load_config(path: Optional[Path]) -> MatcherConfig:
    if path is None:
        return MatcherConfig()
    values = json.loads(path.read_text())
    values["scales"] = tuple(values["scales"])
    return MatcherConfig(**values)


def match_validation_scene(
    root: Path, scene: str, n_patches: int, config: MatcherConfig
) -> tuple[str, dict[str, str]]:
    # Independent scenes make safe process-level parallelism possible.
    cv2.setNumThreads(1)
    scene_dir = root / "validation" / scene
    image_path = image_files(scene_dir, "*_image.png")[0]
    matcher = SceneMatcher(image_path, config)
    cells: dict[str, str] = {}
    for column in patch_columns(n_patches):
        patch_path = scene_dir / "patches" / f"{column}.png"
        prediction = matcher.match(patch_path)
        cells[column] = format_cell(
            prediction if prediction.score >= config.presence_threshold else None
        )
    return scene, cells


def write_submission(
    root: Path,
    output_path: Path,
    config: MatcherConfig,
    constellation: str = "unknown",
    workers: int = 1,
) -> None:
    template_path = root / "sample_submission.csv"
    rows = read_csv_rows(template_path)
    fieldnames = list(rows[0])
    work = [(row["Id"], int(row["n_patches"])) for row in rows]
    scene_cells: dict[str, dict[str, str]] = {}
    if workers == 1:
        for scene, n_patches in work:
            print(f"matching {scene}: {n_patches} patches")
            completed_scene, cells = match_validation_scene(root, scene, n_patches, config)
            scene_cells[completed_scene] = cells
    else:
        print(f"matching {len(work)} scenes with {workers} worker processes")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(match_validation_scene, root, scene, n_patches, config): scene
                for scene, n_patches in work
            }
            for future in as_completed(futures):
                scene, cells = future.result()
                scene_cells[scene] = cells
                print(f"completed {scene}")
    for row in rows:
        row.update(scene_cells[row["Id"]])
        row["constellation"] = constellation
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    validate_submission(root, output_path)
    print(f"wrote {output_path}")


def validate_submission(root: Path, path: Path) -> None:
    template_path = root / "sample_submission.csv"
    template = read_csv_rows(template_path)
    with template_path.open(newline="", encoding="utf-8") as stream:
        template_fields = list(csv.reader(stream))[0]
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != template_fields:
            raise ValueError("Submission header does not exactly match sample_submission.csv")
        output = list(reader)
    if any(None in row or None in row.values() for row in output):
        raise ValueError("Submission contains an extra field or a malformed CSV row")
    if len(template) != len(output):
        raise ValueError("Submission has a different number of rows from the template")
    allowed = {
        pattern.name.removesuffix("_pattern.png")
        for pattern in image_files(root / "patterns", "*_pattern.png")
    } | {"unknown"}
    for expected, actual in zip(template, output):
        missing = [column for column in template_fields if column not in actual]
        blank = [column for column in template_fields if not actual.get(column, "").strip()]
        if missing:
            raise ValueError(f"Missing fields in submission row: {missing}")
        if blank:
            raise ValueError(f"Submission contains blank/null-like fields in {actual.get('Id', '<unknown>')}: {blank}")
        if expected["Id"] != actual["Id"] or expected["n_patches"] != actual["n_patches"]:
            raise ValueError(f"Changed ID or patch count for {expected['Id']}")
        if actual["constellation"] not in allowed:
            raise ValueError(f"Invalid constellation: {actual['constellation']}")
        n_patches = int(actual["n_patches"])
        for column in patch_columns(n_patches):
            prediction = parse_cell(actual[column])
            if prediction is not None and not (0 <= prediction[0] < 3000 and 0 <= prediction[1] < 3000):
                raise ValueError(f"Out-of-range coordinate in {actual['Id']} {column}")
        # Padding cells are ignored by Kaggle, but forcing the sample's explicit
        # -1 sentinel removes a common source of accidental blank/null fields.
        for column in patch_columns(87)[n_patches:]:
            if actual[column] != "-1":
                raise ValueError(f"Padding cell must be -1 in {actual['Id']} {column}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("calibrate", "predict", "validate"))
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--constellation", default="unknown")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "calibrate":
        if args.output is None:
            parser.error("calibrate requires --output CONFIG.json")
        calibrate(root, args.output, load_config(args.config))
    elif args.command == "predict":
        if args.output is None:
            parser.error("predict requires --output submission.csv")
        write_submission(
            root,
            args.output,
            load_config(args.config),
            args.constellation,
            max(1, args.workers),
        )
    else:
        if args.output is None:
            parser.error("validate requires --output submission.csv")
        validate_submission(root, args.output)
        print(f"valid submission: {args.output}")


if __name__ == "__main__":
    main()
