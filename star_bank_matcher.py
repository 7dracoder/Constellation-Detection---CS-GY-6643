#!/usr/bin/env python3
"""Multi-scale Gaussian star bank + rotation/scale patch matching.

Builds a high-confidence list of star centres from a multi-scale Difference-
of-Gaussians response, then matches each query patch against those anchors
with a compact rotation/scale bank and local refinement.  ``--bank-size 0``
selects an adaptive number of peaks per scene, bounded by ``--max-bank-size``.
Detected and locally matched star centres are refined to sub-pixel coordinates
before being merged into an existing GPU/CPU candidate cache.  RANSAC geometry
therefore still sees the dense top-k cloud, while missed true locations
(especially dim non-members) get another chance to appear.

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
DEFAULT_MAX_BANK_SIZE = 5000
DEFAULT_RESPONSE_PERCENTILE = 92.0
DEFAULT_RELATIVE_STRENGTH_FLOOR = 0.65
DEFAULT_ANGLES = 24
DEFAULT_SCALES = (0.70, 0.85, 1.0, 1.18, 1.40)
DEFAULT_REFINE_RADIUS = 8
DEFAULT_MIN_SEPARATION = 6.0
DEFAULT_CENTROID_RADIUS = 2


def multi_scale_dog(normalized: np.ndarray, sigmas: tuple[float, ...] = DEFAULT_SIGMAS) -> np.ndarray:
    """Max absolute DoG across adjacent Gaussian pairs."""
    blurred = [cv2.GaussianBlur(normalized, (0, 0), sigma) for sigma in sigmas]
    response = np.zeros_like(normalized)
    for small, large in zip(blurred[:-1], blurred[1:]):
        dog = small - large
        response = np.maximum(response, np.abs(dog))
    return response.astype(np.float32)


def star_response(
    image: np.ndarray, sigmas: tuple[float, ...] = DEFAULT_SIGMAS
) -> np.ndarray:
    """Return the locally normalized multi-scale DoG response for one scene."""
    background = cv2.GaussianBlur(image, (0, 0), 10.0)
    high_pass = image - background
    local_scale = cv2.GaussianBlur(np.abs(high_pass), (0, 0), 10.0)
    normalized = np.clip(high_pass / (local_scale + 4.0), -8.0, 8.0).astype(np.float32)
    return multi_scale_dog(normalized, sigmas)


def subpixel_centroid(
    response: np.ndarray, x: float, y: float, radius: int = DEFAULT_CENTROID_RADIUS
) -> tuple[float, float]:
    """Refine a peak with a small response-weighted centroid.

    Subtracting the local floor prevents the surrounding background from
    pulling the result toward the geometric centre.  Squaring the remaining
    weights keeps the refinement attached to the dominant point source.
    """
    height, width = response.shape
    centre_x = int(round(x))
    centre_y = int(round(y))
    x0, x1 = max(0, centre_x - radius), min(width, centre_x + radius + 1)
    y0, y1 = max(0, centre_y - radius), min(height, centre_y + radius + 1)
    patch = response[y0:y1, x0:x1].astype(np.float64, copy=False)
    if patch.size == 0:
        return float(x), float(y)
    floor = float(np.percentile(patch, 25.0))
    weights = np.maximum(patch - floor, 0.0) ** 2
    total = float(weights.sum())
    if total <= 1e-12:
        return float(x), float(y)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    refined_x = float(np.sum(weights * xx) / total)
    refined_y = float(np.sum(weights * yy) / total)
    return refined_x, refined_y


def detect_star_bank(
    image: np.ndarray,
    bank_size: int | None = DEFAULT_BANK_SIZE,
    sigmas: tuple[float, ...] = DEFAULT_SIGMAS,
    min_separation: float = DEFAULT_MIN_SEPARATION,
    border: int = PATCH_RADIUS + 4,
    response_percentile: float = DEFAULT_RESPONSE_PERCENTILE,
    max_bank_size: int = DEFAULT_MAX_BANK_SIZE,
    relative_strength_floor: float = DEFAULT_RELATIVE_STRENGTH_FLOOR,
    response: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sub-pixel ``(xy points, strengths)`` for one scene.

    ``bank_size=None`` keeps separated peaks above both the response percentile
    and a scene-relative fraction of the strongest response, capped by
    ``max_bank_size``.  This makes the retained count adapt to the amount and
    contrast of usable star structure in each scene.
    """
    if response is None:
        response = star_response(image, sigmas)

    # Soft local-max suppression, then keep the strongest peaks with spacing.
    kernel = max(3, int(2 * min_separation) | 1)
    local_max = response == cv2.dilate(response, np.ones((kernel, kernel), np.uint8))
    # 99th percentile keeps only the brightest cores; dim labelled non-members
    # on the train scenes sit near ranks 5k–20k, so the default enrichment
    # bank uses a softer cut and lets ``bank_size`` do the real truncation.
    threshold = float(np.percentile(response, response_percentile))
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
    selected: list[tuple[float, float]] = []
    selected_strengths: list[float] = []
    sep2 = min_separation ** 2
    limit = max_bank_size if bank_size is None else min(bank_size, max_bank_size)
    for index in order:
        if bank_size is None and strengths[index] < strengths[order[0]] * relative_strength_floor:
            break
        x, y = int(xs[index]), int(ys[index])
        if all((x - ox) ** 2 + (y - oy) ** 2 >= sep2 for ox, oy in selected):
            selected.append(subpixel_centroid(response, x, y))
            selected_strengths.append(float(strengths[index]))
        if len(selected) >= limit:
            break
    return np.asarray(selected, dtype=np.float32), np.asarray(selected_strengths, dtype=np.float32)


def prepare_bank_features(
    padded_image: np.ndarray, bank_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute scene-side descriptors shared by every query patch."""
    integer_bank_xy = np.rint(bank_xy).astype(np.int32)
    bank_patches = extract_from_padded(padded_image, integer_bank_xy)
    return integer_bank_xy, normalized_vectors(bank_patches), gradient_vectors(bank_patches)


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
    response: np.ndarray | None = None,
    centroid_radius: int = DEFAULT_CENTROID_RADIUS,
    prepared_bank: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> list[tuple[float, float, float, float]]:
    """Score a query against each star-bank anchor; return (score, x, y, margin)."""
    variants = _query_variants(query, angles, scales)
    variant_vectors = normalized_vectors(variants)
    if prepared_bank is None:
        integer_bank_xy, bank_vectors, bank_gradients = prepare_bank_features(
            padded_image, bank_xy
        )
    else:
        integer_bank_xy, bank_vectors, bank_gradients = prepared_bank
    intensity = variant_vectors @ bank_vectors.T
    gradient = gradient_vectors(variants) @ bank_gradients.T
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
        ax, ay = integer_bank_xy[int(anchor_index)]
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
    selected: list[tuple[float, float, float]] = []
    sep2 = minimum_separation ** 2
    for score, x, y in refined:
        refined_x, refined_y = (
            subpixel_centroid(response, x, y, centroid_radius)
            if response is not None
            else (float(x), float(y))
        )
        if all(
            (refined_x - ox) ** 2 + (refined_y - oy) ** 2 >= sep2
            for _, ox, oy in selected
        ):
            selected.append((score, refined_x, refined_y))
        if len(selected) >= top_k:
            break
    result = []
    for index, (score, x, y) in enumerate(selected):
        next_score = selected[index + 1][0] if index + 1 < len(selected) else score - 0.05
        result.append((float(score), float(x), float(y), float(score - next_score)))
    return result


def merge_candidate_lists(
    primary: list[list[float]],
    secondary: list[tuple[float, float, float, float]],
    top_k: int,
    minimum_separation: float = 18.0,
) -> list[list[float]]:
    """Union two candidate lists, preferring higher scores, then re-separate."""
    combined = [
        (float(row[2]), float(row[0]), float(row[1]), float(row[3]) if len(row) > 3 else 0.0)
        for row in primary
    ]
    combined.extend(secondary)
    combined.sort(reverse=True)
    selected: list[list[float]] = []
    sep2 = minimum_separation ** 2
    for score, x, y, margin in combined:
        if all((x - float(row[0])) ** 2 + (y - float(row[1])) ** 2 >= sep2 for row in selected):
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
    bank_size: int | None = DEFAULT_BANK_SIZE,
    top_k: int = 16,
    angles: int = DEFAULT_ANGLES,
    scales: tuple[float, ...] = DEFAULT_SCALES,
    response_percentile: float = DEFAULT_RESPONSE_PERCENTILE,
    max_bank_size: int = DEFAULT_MAX_BANK_SIZE,
    relative_strength_floor: float = DEFAULT_RELATIVE_STRENGTH_FLOOR,
    centroid_radius: int = DEFAULT_CENTROID_RADIUS,
) -> dict:
    """Enrich one scene's candidate cache with star-bank matches."""
    image_paths = image_files(root / split / scene, "*_image.png")
    if len(image_paths) != 1:
        raise ValueError(f"Expected one sky image for {split}/{scene}")
    image = read_grayscale(image_paths[0])
    response = star_response(image)
    bank_xy, bank_strengths = detect_star_bank(
        image,
        bank_size=bank_size,
        response_percentile=response_percentile,
        max_bank_size=max_bank_size,
        relative_strength_floor=relative_strength_floor,
        response=response,
    )
    padded = cv2.copyMakeBorder(
        image, PATCH_RADIUS, PATCH_RADIUS, PATCH_RADIUS, PATCH_RADIUS, cv2.BORDER_REFLECT101
    )
    prepared_bank = prepare_bank_features(padded, bank_xy)

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
            query,
            padded,
            bank_xy,
            angles=angles,
            scales=scales,
            top_k=keep,
            response=response,
            centroid_radius=centroid_radius,
            prepared_bank=prepared_bank,
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
        "star_bank_mode": "adaptive" if bank_size is None else "fixed",
        "star_bank_requested_size": bank_size,
        "star_bank_max_size": max_bank_size,
        "response_percentile": response_percentile,
        "relative_strength_floor": relative_strength_floor,
        "subpixel_centroid_radius": centroid_radius,
        "star_bank_strength_max": float(bank_strengths[0]) if len(bank_strengths) else 0.0,
        "candidates": enriched,
        "source_cache": str(cache_path),
        "enrichment": "adaptive_multi_scale_dog_star_bank_subpixel",
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
    bank_size: int | None,
    top_k: int,
    response_percentile: float,
    max_bank_size: int,
    relative_strength_floor: float,
    centroid_radius: int,
    requested_scenes: list[str] | None,
    overwrite: bool,
) -> None:
    if split == "validation":
        scenes = [row["Id"] for row in read_csv_rows(root / "sample_submission.csv")]
    else:
        scenes = [row["Id"] for row in read_csv_rows(root / "train_ground_truth.csv")]
    if requested_scenes:
        unknown = sorted(set(requested_scenes) - set(scenes))
        if unknown:
            raise ValueError(f"Unknown {split} scenes: {', '.join(unknown)}")
        requested = set(requested_scenes)
        scenes = [scene for scene in scenes if scene in requested]
    output_dir.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        output_path = output_dir / f"{scene}.json"
        if output_path.exists() and not overwrite:
            print(f"{scene}: keeping existing {output_path} (use --overwrite to replace)", flush=True)
            continue
        enrich_scene_cache(
            root,
            split,
            scene,
            cache_dir / f"{scene}.json",
            output_path,
            bank_size=bank_size,
            top_k=top_k,
            response_percentile=response_percentile,
            max_bank_size=max_bank_size,
            relative_strength_floor=relative_strength_floor,
            centroid_radius=centroid_radius,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bank-size",
        type=int,
        default=0,
        help="Fixed number of stars; 0 selects an adaptive count per scene (default)",
    )
    parser.add_argument("--max-bank-size", type=int, default=DEFAULT_MAX_BANK_SIZE)
    parser.add_argument("--response-percentile", type=float, default=DEFAULT_RESPONSE_PERCENTILE)
    parser.add_argument(
        "--relative-strength-floor",
        type=float,
        default=DEFAULT_RELATIVE_STRENGTH_FLOOR,
        help="Adaptive mode keeps peaks at least this fraction of the strongest scene peak",
    )
    parser.add_argument("--centroid-radius", type=int, default=DEFAULT_CENTROID_RADIUS)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help="Process only this scene; repeat for multiple scenes",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.bank_size < 0:
        parser.error("--bank-size must be zero (adaptive) or a positive integer")
    if args.max_bank_size < 1:
        parser.error("--max-bank-size must be positive")
    if not 0.0 < args.response_percentile < 100.0:
        parser.error("--response-percentile must be between 0 and 100")
    if not 0.0 < args.relative_strength_floor <= 1.0:
        parser.error("--relative-strength-floor must be in (0, 1]")
    if args.centroid_radius < 0:
        parser.error("--centroid-radius must be non-negative")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    enrich_split(
        args.root.resolve(),
        args.split,
        args.cache_dir.resolve(),
        args.output_dir.resolve(),
        None if args.bank_size == 0 else args.bank_size,
        args.top_k,
        args.response_percentile,
        args.max_bank_size,
        args.relative_strength_floor,
        args.centroid_radius,
        args.scenes,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
