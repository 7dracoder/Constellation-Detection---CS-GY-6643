#!/usr/bin/env python3
"""Multi-scale Gaussian star bank + rotation/scale patch matching.

Builds a high-confidence list of ~150 star centres from a 2–4 scale Difference-
of-Gaussians response, then matches each query patch against those anchors
with a compact rotation/scale bank and local refinement.  The result is merged
into an existing GPU/CPU candidate cache so RANSAC geometry still sees the
dense top-k cloud, while missed true locations (especially dim non-members)
get another chance to appear.

This does not download external data and does not change constellation labels
by itself; it only improves candidate recall for the existing solvers.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from constellation_pipeline import (
    PATCH_RADIUS,
    PATCH_SIZE,
    extract_from_padded,
    gradient_vectors,
    image_files,
    normalized_vectors,
    patch_columns,
    read_csv_rows,
    read_grayscale,
    structural_similarity,
    transform_query,
)

# Four DoG pairings built from adjacent Gaussians.  Small sigmas catch sharp
# point sources; larger ones catch slightly bloated / saturated stars.
DEFAULT_SIGMAS = (0.8, 1.4, 2.2, 3.4)
DEFAULT_BANK_SIZE = 150
DEFAULT_ANGLES = 24
DEFAULT_SCALES = (0.70, 0.85, 1.0, 1.18, 1.40)
DEFAULT_REFINE_RADIUS = 8
DEFAULT_MIN_SEPARATION = 6.0


def multi_scale_dog(normalized: np.ndarray, sigmas: tuple[float, ...] = DEFAULT_SIGMAS) -> np.ndarray:
    """Max absolute DoG across adjacent Gaussian pairs."""
    blurred = [cv2.GaussianBlur(normalized, (0, 0), sigma) for sigma in sigmas]
    response = np.zeros_like(normalized)
    for small, large in zip(blurred[:-1], blurred[1:]):
        dog = small - large
        response = np.maximum(response, np.abs(dog))
    return response.astype(np.float32)


def detect_star_bank(
    image: np.ndarray,
    bank_size: int = DEFAULT_BANK_SIZE,
    sigmas: tuple[float, ...] = DEFAULT_SIGMAS,
    min_separation: float = DEFAULT_MIN_SEPARATION,
    border: int = PATCH_RADIUS + 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (xy points, strengths) for the strongest ~bank_size stars."""
    background = cv2.GaussianBlur(image, (0, 0), 10.0)
    high_pass = image - background
    local_scale = cv2.GaussianBlur(np.abs(high_pass), (0, 0), 10.0)
    normalized = np.clip(high_pass / (local_scale + 4.0), -8.0, 8.0).astype(np.float32)
    response = multi_scale_dog(normalized, sigmas)

    # Soft local-max suppression, then keep the strongest peaks with spacing.
    kernel = max(3, int(2 * min_separation) | 1)
    local_max = response == cv2.dilate(response, np.ones((kernel, kernel), np.uint8))
    # 99th percentile keeps only the brightest cores; dim labelled non-members
    # on the train scenes sit near ranks 5k–20k, so the default enrichment
    # bank uses a softer cut and lets ``bank_size`` do the real truncation.
    threshold = float(np.percentile(response, 92.0))
    valid = local_max & (response >= threshold)
    valid[:border, :] = False
    valid[-border:, :] = False
    valid[:, :border] = False
    valid[:, -border:] = False
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        raise RuntimeError("star bank found no peaks")
    strengths = response[ys, xs]
    order = np.argsort(strengths)[::-1]
    selected: list[tuple[int, int]] = []
    selected_strengths: list[float] = []
    sep2 = min_separation ** 2
    for index in order:
        x, y = int(xs[index]), int(ys[index])
        if all((x - ox) ** 2 + (y - oy) ** 2 >= sep2 for ox, oy in selected):
            selected.append((x, y))
            selected_strengths.append(float(strengths[index]))
        if len(selected) >= bank_size:
            break
    return np.asarray(selected, dtype=np.int32), np.asarray(selected_strengths, dtype=np.float32)


def _query_variants(
    query: np.ndarray,
    angles: int = DEFAULT_ANGLES,
    scales: tuple[float, ...] = DEFAULT_SCALES,
) -> np.ndarray:
    variants = []
    for scale in scales:
        for angle_index in range(angles):
            angle = 360.0 * angle_index / angles
            variants.append(transform_query(query, angle, scale))
    return np.stack(variants).astype(np.float32)


def match_query_to_bank(
    query: np.ndarray,
    padded_image: np.ndarray,
    bank_xy: np.ndarray,
    angles: int = DEFAULT_ANGLES,
    scales: tuple[float, ...] = DEFAULT_SCALES,
    refine_radius: int = DEFAULT_REFINE_RADIUS,
    top_k: int = 16,
    minimum_separation: float = 18.0,
) -> list[tuple[float, int, int, float]]:
    """Score a query against each star-bank anchor; return (score, x, y, margin)."""
    variants = _query_variants(query, angles, scales)
    variant_vectors = normalized_vectors(variants)
    bank_patches = extract_from_padded(padded_image, bank_xy)
    bank_vectors = normalized_vectors(bank_patches)
    intensity = variant_vectors @ bank_vectors.T
    gradient = gradient_vectors(variants) @ gradient_vectors(bank_patches).T
    similarity = 0.70 * intensity + 0.30 * gradient

    # Shortlist anchors, then locally refine the best few transforms.
    anchor_order = np.argsort(similarity.max(axis=0))[::-1][: min(48, len(bank_xy))]
    offsets = np.asarray(
        [
            (dx, dy)
            for dy in range(-refine_radius, refine_radius + 1)
            for dx in range(-refine_radius, refine_radius + 1)
        ],
        dtype=np.int32,
    )
    refined: list[tuple[float, int, int]] = []
    for anchor_index in anchor_order:
        ax, ay = bank_xy[int(anchor_index)]
        shortlist = np.argsort(similarity[:, anchor_index])[::-1][:4]
        centres = offsets + np.asarray((ax, ay), dtype=np.int32)
        local_patches = extract_from_padded(padded_image, centres)
        local_vectors = normalized_vectors(local_patches)
        local_scores = variant_vectors[shortlist] @ local_vectors.T
        best_offsets = np.argmax(local_scores, axis=1)
        for order_index, (variant_index, offset_index) in enumerate(zip(shortlist, best_offsets)):
            rx, ry = centres[int(offset_index)]
            refined_patch = local_patches[int(offset_index)]
            gradient_score = (
                gradient_vectors(variants[int(variant_index) : int(variant_index) + 1])
                @ gradient_vectors(refined_patch[None]).T
            ).item()
            ssim_score = structural_similarity(variants[int(variant_index)], refined_patch)
            score = (
                0.50 * float(local_scores[order_index, int(offset_index)])
                + 0.20 * gradient_score
                + 0.30 * ssim_score
            )
            refined.append((score, int(rx), int(ry)))

    if not refined:
        return []
    refined.sort(reverse=True)
    selected: list[tuple[float, int, int]] = []
    sep2 = minimum_separation ** 2
    for score, x, y in refined:
        if all((x - ox) ** 2 + (y - oy) ** 2 >= sep2 for _, ox, oy in selected):
            selected.append((score, x, y))
        if len(selected) >= top_k:
            break
    result = []
    for index, (score, x, y) in enumerate(selected):
        next_score = selected[index + 1][0] if index + 1 < len(selected) else score - 0.05
        result.append((float(score), int(x), int(y), float(score - next_score)))
    return result


def merge_candidate_lists(
    primary: list[list[float]],
    secondary: list[tuple[float, int, int, float]],
    top_k: int,
    minimum_separation: float = 18.0,
) -> list[list[float]]:
    """Union two candidate lists, preferring higher scores, then re-separate."""
    combined = [(float(row[2]), int(row[0]), int(row[1]), float(row[3]) if len(row) > 3 else 0.0) for row in primary]
    combined.extend(secondary)
    combined.sort(reverse=True)
    selected: list[list[float]] = []
    sep2 = minimum_separation ** 2
    for score, x, y, margin in combined:
        if all((x - int(row[0])) ** 2 + (y - int(row[1])) ** 2 >= sep2 for row in selected):
            selected.append([float(x), float(y), float(score), float(margin)])
        if len(selected) >= top_k:
            break
    return selected


def enrich_scene_cache(
    root: Path,
    split: str,
    scene: str,
    cache_path: Path,
    output_path: Path,
    bank_size: int = DEFAULT_BANK_SIZE,
    top_k: int = 16,
    angles: int = DEFAULT_ANGLES,
    scales: tuple[float, ...] = DEFAULT_SCALES,
) -> dict:
    """Enrich one scene's candidate cache with star-bank matches."""
    image_paths = image_files(root / split / scene, "*_image.png")
    if len(image_paths) != 1:
        raise ValueError(f"Expected one sky image for {split}/{scene}")
    image = read_grayscale(image_paths[0])
    bank_xy, bank_strengths = detect_star_bank(image, bank_size=bank_size)
    padded = cv2.copyMakeBorder(
        image, PATCH_RADIUS, PATCH_RADIUS, PATCH_RADIUS, PATCH_RADIUS, cv2.BORDER_REFLECT101
    )

    saved = json.loads(cache_path.read_text()) if cache_path.exists() else None
    if saved is None:
        raise FileNotFoundError(f"Missing base cache {cache_path}; run the GPU matcher first")
    active = list(saved["columns"])
    base_top_k = int(saved.get("top_k", top_k))
    keep = max(base_top_k, top_k)

    enriched = []
    recovered = 0
    for query_index, column in enumerate(active):
        query = read_grayscale(root / split / scene / "patches" / f"{column}.png")
        if query.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Unexpected patch shape {query.shape} in {scene}/{column}")
        bank_hits = match_query_to_bank(
            query, padded, bank_xy, angles=angles, scales=scales, top_k=keep
        )
        primary = saved["candidates"][query_index]
        merged = merge_candidate_lists(primary, bank_hits, keep)
        # Count a recovery when the bank contributed a new top-ranked location
        # that was not within 12 px of any previous candidate.
        if bank_hits:
            bx, by = bank_hits[0][1], bank_hits[0][2]
            if all(math.hypot(bx - row[0], by - row[1]) > 12.0 for row in primary):
                if any(math.hypot(bx - row[0], by - row[1]) <= 12.0 for row in merged[:3]):
                    recovered += 1
        enriched.append(merged)
        if (query_index + 1) % 10 == 0 or query_index + 1 == len(active):
            print(f"  {scene}: {query_index + 1}/{len(active)} patches  bank={len(bank_xy)}", flush=True)

    payload = {
        "columns": active,
        "top_k": keep,
        "star_bank_size": int(len(bank_xy)),
        "star_bank_strength_max": float(bank_strengths[0]) if len(bank_strengths) else 0.0,
        "candidates": enriched,
        "source_cache": str(cache_path),
        "enrichment": "multi_scale_dog_star_bank",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload))
    print(f"{scene}: wrote {output_path} (new-top candidates≈{recovered})", flush=True)
    return payload


def enrich_split(
    root: Path,
    split: str,
    cache_dir: Path,
    output_dir: Path,
    bank_size: int,
    top_k: int,
) -> None:
    if split == "validation":
        scenes = [row["Id"] for row in read_csv_rows(root / "sample_submission.csv")]
    else:
        scenes = [row["Id"] for row in read_csv_rows(root / "train_ground_truth.csv")]
    output_dir.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        enrich_scene_cache(
            root,
            split,
            scene,
            cache_dir / f"{scene}.json",
            output_dir / f"{scene}.json",
            bank_size=bank_size,
            top_k=top_k,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bank-size", type=int, default=DEFAULT_BANK_SIZE)
    parser.add_argument("--top-k", type=int, default=16)
    args = parser.parse_args()
    enrich_split(
        args.root.resolve(),
        args.split,
        args.cache_dir.resolve(),
        args.output_dir.resolve(),
        args.bank_size,
        args.top_k,
    )


if __name__ == "__main__":
    main()
