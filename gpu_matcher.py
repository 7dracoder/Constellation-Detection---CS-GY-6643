#!/usr/bin/env python3
"""GPU-accelerated, course-data-only matcher for constellation queries.

This keeps the baseline's full-resolution verification, but replaces its
OpenCV-by-variant coarse search with batched normalized cross correlation on a
CUDA GPU.  The wider rotation/scale bank is deliberately generated only from
the supplied query patches; it does not download, train on, or assume any
external sky data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from constellation_pipeline import (
    PATCH_RADIUS,
    PATCH_SIZE,
    MatcherConfig,
    Prediction,
    SceneMatcher,
    bandpass_for_matching,
    denoise_for_matching,
    extract_from_padded,
    fit_presence_threshold,
    format_cell,
    gradient_vectors,
    image_files,
    iter_scene_patches,
    load_config,
    normalized_vectors,
    parse_truth_rows,
    patch_columns,
    read_grayscale,
    read_csv_rows,
    score_train_predictions,
    structural_similarity,
    validate_submission,
)


def resolve_device(requested: str) -> str:
    """Fail closed when CUDA was requested but is unavailable."""
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false")
    return requested


class TorchCoarseSceneMatcher(SceneMatcher):
    """Run the reduced-resolution NCC bank in batches on one Torch device."""

    def __init__(
        self,
        image_path: Path,
        config: MatcherConfig,
        device: str,
        batch_size: int,
    ):
        super().__init__(image_path, config)
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.coarse_tensor = torch.from_numpy(self.coarse_image).to(
            self.device, dtype=torch.float32
        )[None, None]
        self.coarse_bandpass_tensor = (
            torch.from_numpy(self.coarse_bandpass).to(self.device, dtype=torch.float32)[None, None]
            if self.coarse_bandpass is not None else None
        )

    def _coarse_candidates(self, variants: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Equivalent to TM_CCOEFF_NORMED, evaluated for many filters at once."""
        torch = self.torch
        functional = torch.nn.functional
        factor = self.config.coarse_factor
        small_size = max(6, round(PATCH_SIZE / factor))
        template_count = small_size * small_size
        ones = torch.ones((1, 1, small_size, small_size), device=self.device)
        with torch.inference_mode():
            local_sum = functional.conv2d(self.coarse_tensor, ones)
            local_sum_sq = functional.conv2d(self.coarse_tensor.square(), ones)
            local_energy = (local_sum_sq - local_sum.square() / template_count).clamp_min_(1e-6)
            if self.coarse_bandpass_tensor is not None:
                band_sum = functional.conv2d(self.coarse_bandpass_tensor, ones)
                band_sum_sq = functional.conv2d(self.coarse_bandpass_tensor.square(), ones)
                band_energy = (
                    band_sum_sq - band_sum.square() / template_count
                ).clamp_min_(1e-6)

            # Preserve the transformation that produced every coarse peak.
            # The CPU matcher uses that information to test a small local
            # rotation/scale neighbourhood during fine alignment; dropping it
            # here made the GPU path incompatible with the newer joint solver
            # and unnecessarily weakened fine candidate ranking.
            raw_candidates: list[tuple[float, int, int, int]] = []
            for start in range(0, len(variants), self.batch_size):
                chunk = variants[start : start + self.batch_size]
                templates = np.stack(
                    [
                        cv2.resize(item, (small_size, small_size), interpolation=cv2.INTER_AREA)
                        for item in chunk
                    ]
                ).astype(np.float32)
                filters = torch.from_numpy(templates).to(self.device)[:, None]
                filters -= filters.mean(dim=(2, 3), keepdim=True)
                filter_norm = filters.flatten(1).norm(dim=1).clamp_min_(1e-6)
                # conv2d's first axis is the (single) scene-image batch;
                # remove it so every remaining leading axis is a query variant.
                response = functional.conv2d(self.coarse_tensor, filters)[0]
                response /= local_energy.sqrt()[0]
                response /= filter_norm[:, None, None]
                if self.coarse_bandpass_tensor is not None:
                    band_templates = np.stack(
                        [
                            cv2.resize(
                                bandpass_for_matching(item, self.config),
                                (small_size, small_size), interpolation=cv2.INTER_AREA,
                            )
                            for item in chunk
                        ]
                    ).astype(np.float32)
                    band_filters = torch.from_numpy(band_templates).to(self.device)[:, None]
                    band_filters -= band_filters.mean(dim=(2, 3), keepdim=True)
                    band_norm = band_filters.flatten(1).norm(dim=1).clamp_min_(1e-6)
                    band_response = functional.conv2d(self.coarse_bandpass_tensor, band_filters)[0]
                    band_response /= band_energy.sqrt()[0]
                    band_response /= band_norm[:, None, None]
                    weight = self.config.bandpass_weight
                    response = (1.0 - weight) * response + weight * band_response

                # Keep separated maxima per transformation before the global
                # coordinate-level deduplication below.  This mirrors the CPU
                # coarse-stage semantics without moving full response maps back.
                pooled = functional.max_pool2d(response[:, None], 7, stride=1, padding=3)[:, 0]
                response.masked_fill_(response < pooled, float("-inf"))
                peak_count = min(self.config.coarse_peaks_per_transform, response.shape[-1] * response.shape[-2])
                values, flat_indices = torch.topk(response.flatten(1), peak_count, dim=1)
                map_width = response.shape[-1]
                for variant_index in range(response.shape[0]):
                    for score, flat_index in zip(values[variant_index].tolist(), flat_indices[variant_index].tolist()):
                        if not np.isfinite(score):
                            continue
                        top = int(flat_index) // map_width
                        left = int(flat_index) % map_width
                        raw_candidates.append(
                            (
                                float(score),
                                int(left * factor + PATCH_RADIUS),
                                int(top * factor + PATCH_RADIUS),
                                start + variant_index,
                            )
                        )

        raw_candidates.sort(reverse=True)
        selected_points: list[tuple[int, int]] = []
        selected_variants: list[int] = []
        minimum_distance = (factor * 3) ** 2
        for _, x, y, variant_index in raw_candidates:
            if all(
                (x - old_x) ** 2 + (y - old_y) ** 2 > minimum_distance
                for old_x, old_y in selected_points
            ):
                selected_points.append((x, y))
                selected_variants.append(variant_index)
            if len(selected_points) >= self.config.max_global_candidates:
                break
        if not selected_points:
            raise RuntimeError("GPU coarse matching produced no candidates")
        return (
            np.asarray(selected_points, dtype=np.int32),
            np.asarray(selected_variants, dtype=np.int32),
        )

    def match_candidates(
        self, query_path: Path, limit: int = 16, minimum_separation: float = 18.0
    ) -> list[Prediction]:
        """Return fine-ranked candidates without recomputing each local window.

        ``SceneMatcher`` refines each rotation/scale variant separately.  That
        recomputes the same 441 nearby 32x32 windows roughly ten times for a
        given coarse location.  The candidate and transform bank are unchanged
        here; we only score all of a location's candidate transforms against
        its one local-window bank at once.  This preserves the original
        coarse-to-fine matching rule while making the wider, course-data-only
        transform bank practical for the full validation split.
        """
        query = denoise_for_matching(read_grayscale(query_path), self.config)
        if query.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Unexpected patch shape {query.shape} in {query_path}")
        if limit < 1:
            raise ValueError("limit must be positive")

        variants, _ = self._query_variants(query)
        band_variants = (
            np.stack([bandpass_for_matching(item, self.config) for item in variants])
            if self.padded_bandpass is not None else None
        )
        candidates, coarse_variants = self._coarse_candidates(variants)
        self.last_candidate_count = len(candidates)
        candidate_patches = extract_from_padded(self.padded_image, candidates)
        variant_vectors = normalized_vectors(variants)
        candidate_vectors = normalized_vectors(candidate_patches)
        intensity_similarity = variant_vectors @ candidate_vectors.T
        if band_variants is None:
            gradient_similarity = gradient_vectors(variants) @ gradient_vectors(candidate_patches).T
            similarity = 0.70 * intensity_similarity + 0.30 * gradient_similarity
            band_vectors = None
        else:
            band_vectors = normalized_vectors(band_variants)
            band_patches = extract_from_padded(self.padded_bandpass, candidates)
            band_similarity = band_vectors @ normalized_vectors(band_patches).T
            weight = self.config.bandpass_weight
            similarity = (1.0 - weight) * intensity_similarity + weight * band_similarity
        candidate_order = np.argsort(similarity.max(axis=0))[::-1][
            : self.config.candidate_refinement_limit
        ]

        radius = self.config.refine_radius
        offsets = np.asarray(
            [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)],
            dtype=np.int32,
        )
        refined: list[tuple[float, int, int]] = []
        for candidate_index in candidate_order:
            candidate_x, candidate_y = candidates[candidate_index]
            shortlist = np.argsort(similarity[:, candidate_index])[::-1][
                : self.config.variants_per_candidate
            ]
            variant_order = list(
                dict.fromkeys(
                    self._coarse_neighbourhood(int(coarse_variants[candidate_index]))
                    + [int(item) for item in shortlist]
                )
            )
            centres = offsets + np.asarray((candidate_x, candidate_y), dtype=np.int32)
            local_patches = extract_from_padded(self.padded_image, centres)
            # The old per-variant _refine call normalised local_patches for
            # every transform.  One matrix product obtains exactly the same
            # maxima for all variants from a single normalised local bank.
            local_vectors = normalized_vectors(local_patches)
            local_scores = variant_vectors[np.asarray(variant_order)] @ local_vectors.T
            if band_vectors is not None:
                local_band_patches = extract_from_padded(self.padded_bandpass, centres)
                local_band_vectors = normalized_vectors(local_band_patches)
                band_scores = band_vectors[np.asarray(variant_order)] @ local_band_vectors.T
                weight = self.config.bandpass_weight
                local_scores = (1.0 - weight) * local_scores + weight * band_scores
            best_offsets = np.argmax(local_scores, axis=1)
            for order_index, (variant_index, offset_index) in enumerate(zip(variant_order, best_offsets)):
                refined_x, refined_y = centres[int(offset_index)]
                refined_patch = local_patches[int(offset_index)]
                if band_vectors is not None:
                    refined.append((
                        float(local_scores[order_index, offset_index]),
                        int(refined_x), int(refined_y),
                    ))
                    continue
                gradient_score = (
                    gradient_vectors(variants[variant_index : variant_index + 1])
                    @ gradient_vectors(refined_patch[None]).T
                ).item()
                ssim_score = structural_similarity(variants[variant_index], refined_patch)
                score = (
                    self.config.intensity_weight * float(local_scores[order_index, offset_index])
                    + 0.20 * gradient_score
                    + (0.80 - self.config.intensity_weight) * ssim_score
                )
                refined.append((score, int(refined_x), int(refined_y)))

        if not refined:
            raise RuntimeError(f"No fine candidates for {query_path}")
        refined.sort(reverse=True)
        selected: list[tuple[float, int, int]] = []
        separation_squared = minimum_separation**2
        for score, x, y in refined:
            if all(
                (x - old_x) ** 2 + (y - old_y) ** 2 > separation_squared
                for _, old_x, old_y in selected
            ):
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


def calibrate(root: Path, output_config: Path, config: MatcherConfig, device: str, batch_size: int) -> None:
    truth = parse_truth_rows(root)
    records: list[tuple[Prediction, bool]] = []
    raw_predictions: dict[str, dict[str, Prediction]] = {}
    for scene, image_path, patches in iter_scene_patches(root, "train"):
        print(f"indexing {scene} ({len(patches)} patches) on {device}", flush=True)
        matcher = TorchCoarseSceneMatcher(image_path, config, device, batch_size)
        raw_predictions[scene] = {}
        for patch_path in patches:
            prediction = matcher.match(patch_path)
            raw_predictions[scene][patch_path.stem] = prediction
            records.append((prediction, truth[scene][patch_path.stem] is not None))
    config = fit_presence_threshold(records, config)
    # The supplied training rows are used only to calibrate presence.  Do not
    # print the repository's old "approximate score" here: it treats every
    # training constellation as unknown and therefore cannot validate the
    # identity component of this competition's metric.
    print(f"calibrated {sum(len(values) for values in raw_predictions.values())} query patches", flush=True)
    output_config.write_text(json.dumps(asdict(config), indent=2) + "\n")
    print(f"wrote {output_config}", flush=True)


def write_submission(
    root: Path,
    output_path: Path,
    config: MatcherConfig,
    device: str,
    batch_size: int,
) -> None:
    template_rows = read_csv_rows(root / "sample_submission.csv")
    fieldnames = list(template_rows[0])
    for row in template_rows:
        scene = row["Id"]
        n_patches = int(row["n_patches"])
        print(f"matching {scene}: {n_patches} patches on {device}", flush=True)
        scene_dir = root / "validation" / scene
        image_path = image_files(scene_dir, "*_image.png")[0]
        matcher = TorchCoarseSceneMatcher(image_path, config, device, batch_size)
        for column in patch_columns(n_patches):
            prediction = matcher.match(scene_dir / "patches" / f"{column}.png")
            row[column] = format_cell(
                prediction if prediction.score >= config.presence_threshold else None
            )
        row["constellation"] = "unknown"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = __import__("csv").DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(template_rows)
    validate_submission(root, output_path)
    print(f"wrote {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("calibrate", "predict", "validate"))
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "validate":
        validate_submission(root, args.output)
        print(f"valid submission: {args.output}")
        return
    device = resolve_device(args.device)
    config = load_config(args.config)
    if args.command == "calibrate":
        calibrate(root, args.output, config, device, args.batch_size)
    else:
        write_submission(root, args.output, config, device, args.batch_size)


if __name__ == "__main__":
    main()
