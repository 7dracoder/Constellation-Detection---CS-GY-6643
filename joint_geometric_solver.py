#!/usr/bin/env python3
"""Joint patch-localisation and constellation-geometry solver.

This is a course-data-only second-stage solver.  A query patch has many
visually plausible star-field locations, so selecting its visual top-1 before
looking at the constellation graph discards useful evidence.  Here each query
keeps a bounded shortlist, and a similarity transform of one supplied pattern
selects a mutually consistent, one-to-one set of figure-star locations.

No scene name, patch count, external image, external label, or pretrained
model is used as a prediction feature.  Training labels calibrate the existing
present/absent threshold and the small patch-only membership selector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from constellation_pipeline import (
    MatcherConfig,
    Prediction,
    SceneMatcher,
    format_cell,
    image_files,
    load_config,
    parse_cell,
    patch_columns,
    read_csv_rows,
    validate_submission,
)
from structural_refiner import MembershipClassifier, Pattern, load_patterns, train_membership_classifier


@dataclass(frozen=True)
class Candidate:
    query_index: int
    x: int
    y: int
    score: float


@dataclass(frozen=True)
class GraphFit:
    pattern: Pattern
    mapped_points: np.ndarray
    assignments: dict[int, Candidate]
    support: int
    mean_error: float
    tolerance: float
    quality: float


def candidate_pairs(candidates: list[Candidate]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directed pairs of locations belonging to distinct query patches."""
    points = np.asarray([(item.x, item.y) for item in candidates], dtype=np.float32)
    queries = np.asarray([item.query_index for item in candidates], dtype=np.int32)
    left, right = np.triu_indices(len(candidates), 1)
    keep = queries[left] != queries[right]
    left, right = left[keep], right[keep]
    lengths = np.linalg.norm(points[right] - points[left], axis=1)
    keep = lengths >= 24.0
    left, right, lengths = left[keep], right[keep], lengths[keep]
    # A candidate-pair direction is not informative; retain both to cover the
    # 180-degree alternative when proposing transforms from one pattern pair.
    return (
        np.concatenate((left, right)),
        np.concatenate((right, left)),
        np.concatenate((lengths, lengths)),
    )


def similarity_from_pairs(
    source_points: np.ndarray,
    candidate_points: np.ndarray,
    node_left: np.ndarray,
    node_right: np.ndarray,
    candidate_left: np.ndarray,
    candidate_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised similarity transforms mapping source-pair to candidate-pair."""
    source_a = source_points[node_left]
    source_delta = source_points[node_right] - source_a
    target_a = candidate_points[candidate_left]
    target_delta = candidate_points[candidate_right] - target_a
    source_norm = np.linalg.norm(source_delta, axis=1)
    target_norm = np.linalg.norm(target_delta, axis=1)
    scale = target_norm / source_norm
    cosine = np.sum(source_delta * target_delta, axis=1) / (source_norm * target_norm)
    sine = (
        source_delta[:, 0] * target_delta[:, 1] - source_delta[:, 1] * target_delta[:, 0]
    ) / (source_norm * target_norm)
    matrix = scale[:, None, None] * np.stack(
        (
            np.stack((cosine, -sine), axis=1),
            np.stack((sine, cosine), axis=1),
        ),
        axis=1,
    )
    offset = target_a - np.einsum("bij,bj->bi", matrix, source_a)
    return matrix.astype(np.float32), offset.astype(np.float32)


def propose_transforms(
    source_points: np.ndarray,
    candidates: list[Candidate],
    proposals: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw scale-compatible RANSAC proposals without a giant pair cross-product."""
    if len(candidates) < 2:
        return np.empty((0, 2, 2), np.float32), np.empty((0, 2), np.float32)
    points = np.asarray([(item.x, item.y) for item in candidates], dtype=np.float32)
    candidate_left, candidate_right, candidate_lengths = candidate_pairs(candidates)
    node_left, node_right = np.triu_indices(len(source_points), 1)
    node_lengths = np.linalg.norm(source_points[node_right] - source_points[node_left], axis=1)
    valid_nodes = node_lengths >= 8.0
    node_left, node_right, node_lengths = (
        node_left[valid_nodes],
        node_right[valid_nodes],
        node_lengths[valid_nodes],
    )
    if len(candidate_lengths) == 0 or len(node_lengths) == 0:
        return np.empty((0, 2, 2), np.float32), np.empty((0, 2), np.float32)
    rng = np.random.default_rng(seed)
    matrices: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    accepted = 0
    attempts = 0
    while accepted < proposals and attempts < 20:
        attempts += 1
        draw = max(2_048, 2 * (proposals - accepted))
        c_index = rng.integers(len(candidate_lengths), size=draw)
        n_index = rng.integers(len(node_lengths), size=draw)
        ratio = candidate_lengths[c_index] / node_lengths[n_index]
        # These broad bounds are calibrated only from the labelled examples;
        # they reject numerically implausible maps but retain all observed
        # training-scene scales with room for held-out variation.
        keep = (ratio >= 2.0) & (ratio <= 10.0)
        if not np.any(keep):
            continue
        c_index, n_index = c_index[keep], n_index[keep]
        take = min(proposals - accepted, len(c_index))
        c_index, n_index = c_index[:take], n_index[:take]
        matrix, offset = similarity_from_pairs(
            source_points,
            points,
            node_left[n_index],
            node_right[n_index],
            candidate_left[c_index],
            candidate_right[c_index],
        )
        matrices.append(matrix)
        offsets.append(offset)
        accepted += len(matrix)
    if not matrices:
        return np.empty((0, 2, 2), np.float32), np.empty((0, 2), np.float32)
    return np.concatenate(matrices), np.concatenate(offsets)


def top_hypotheses(
    source_points: np.ndarray,
    candidates: list[Candidate],
    matrices: np.ndarray,
    offsets: np.ndarray,
    keep: int = 24,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Use nearest candidate points to cheaply pre-rank transform proposals."""
    if len(matrices) == 0:
        return []
    points = np.asarray([(item.x, item.y) for item in candidates], dtype=np.float32)
    tree = cKDTree(points)
    scored: list[tuple[float, int, float]] = []
    for start in range(0, len(matrices), 512):
        matrix = matrices[start : start + 512]
        offset = offsets[start : start + 512]
        mapped = np.einsum("bij,pj->bpi", matrix, source_points) + offset[:, None, :]
        distance, _ = tree.query(mapped.reshape(-1, 2), k=1)
        distance = distance.reshape(len(mapped), len(source_points))
        scale = np.sqrt(np.abs(np.linalg.det(matrix)))
        tolerance = np.clip(4.5 * scale, 18.0, 45.0)
        inlier = distance <= tolerance[:, None]
        support = inlier.sum(axis=1)
        residual = np.where(inlier, distance, 0.0).sum(axis=1) / np.maximum(support, 1)
        quality = support - 0.20 * residual / tolerance
        for local_index in np.argsort(quality)[-min(12, len(quality)) :]:
            scored.append((float(quality[local_index]), start + int(local_index), float(tolerance[local_index])))
    scored.sort(reverse=True)
    result: list[tuple[np.ndarray, np.ndarray, float]] = []
    for _, index, tolerance in scored[:keep]:
        result.append((matrices[index], offsets[index], tolerance))
    return result


def estimate_similarity(source: np.ndarray, target: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Least-squares rotation, scale, and translation; source may already be reflected."""
    if len(source) < 2:
        return None
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered
    left, singular, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    scale = float(singular.sum() / max(float(np.sum(source_centered * source_centered)), 1e-6))
    matrix = (scale * rotation).astype(np.float32)
    offset = (target_mean - source_mean @ matrix.T).astype(np.float32)
    return matrix, offset


def assign_queries(
    pattern: Pattern,
    source_points: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
    candidates: list[Candidate],
) -> Optional[GraphFit]:
    """Assign at most one query patch to each mapped pattern node and vice versa."""
    mapped = source_points @ matrix.T + offset
    query_ids = sorted({item.query_index for item in candidates})
    if not query_ids:
        return None
    query_to_column = {query: column for column, query in enumerate(query_ids)}
    scale = float(np.sqrt(abs(np.linalg.det(matrix))))
    tolerance = float(np.clip(4.5 * scale, 18.0, 45.0))
    # Cost is normalized localisation error, with a small tie-breaker for the
    # visual score.  A dummy column lets a pattern node remain unmatched.
    costs = np.full((len(mapped), len(query_ids)), 1.25, dtype=np.float32)
    chosen = np.full((len(mapped), len(query_ids)), -1, dtype=np.int32)
    score_values = np.asarray([item.score for item in candidates], dtype=np.float32)
    score_floor, score_span = float(score_values.min()), float(np.ptp(score_values) + 1e-5)
    for candidate_index, candidate in enumerate(candidates):
        column = query_to_column[candidate.query_index]
        distance = np.linalg.norm(mapped - np.asarray((candidate.x, candidate.y), np.float32), axis=1)
        value = distance / tolerance - 0.10 * (candidate.score - score_floor) / score_span
        better = (distance <= tolerance) & (value < costs[:, column])
        costs[better, column] = value[better]
        chosen[better, column] = candidate_index
    dummy = np.full((len(mapped), len(mapped)), 0.92, dtype=np.float32)
    row_index, column_index = linear_sum_assignment(np.concatenate((costs, dummy), axis=1))
    assignments: dict[int, Candidate] = {}
    node_indices: list[int] = []
    target_points: list[tuple[int, int]] = []
    for node, column in zip(row_index, column_index):
        if column >= len(query_ids) or chosen[node, column] < 0 or costs[node, column] >= 0.92:
            continue
        candidate = candidates[int(chosen[node, column])]
        assignments[candidate.query_index] = candidate
        node_indices.append(int(node))
        target_points.append((candidate.x, candidate.y))
    # Refit a transform from the one-to-one assignment, then make one final
    # assignment pass.  This turns a two-point RANSAC proposal into a robust
    # partial-graph fit without allowing arbitrary affine distortion.
    refined = estimate_similarity(source_points[np.asarray(node_indices)], np.asarray(target_points, np.float32))
    if refined is not None and len(node_indices) >= 3:
        return assign_queries_once(pattern, source_points, *refined, candidates)
    return graph_fit_from_assignment(pattern, mapped, assignments, tolerance)


def assign_queries_once(
    pattern: Pattern,
    source_points: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
    candidates: list[Candidate],
) -> Optional[GraphFit]:
    """One non-recursive assignment pass used after least-squares refinement."""
    mapped = source_points @ matrix.T + offset
    query_ids = sorted({item.query_index for item in candidates})
    query_to_column = {query: column for column, query in enumerate(query_ids)}
    scale = float(np.sqrt(abs(np.linalg.det(matrix))))
    tolerance = float(np.clip(4.5 * scale, 18.0, 45.0))
    costs = np.full((len(mapped), len(query_ids)), 1.25, dtype=np.float32)
    chosen = np.full((len(mapped), len(query_ids)), -1, dtype=np.int32)
    score_values = np.asarray([item.score for item in candidates], dtype=np.float32)
    score_floor, score_span = float(score_values.min()), float(np.ptp(score_values) + 1e-5)
    for candidate_index, candidate in enumerate(candidates):
        column = query_to_column[candidate.query_index]
        distance = np.linalg.norm(mapped - np.asarray((candidate.x, candidate.y), np.float32), axis=1)
        value = distance / tolerance - 0.10 * (candidate.score - score_floor) / score_span
        better = (distance <= tolerance) & (value < costs[:, column])
        costs[better, column] = value[better]
        chosen[better, column] = candidate_index
    row_index, column_index = linear_sum_assignment(
        np.concatenate((costs, np.full((len(mapped), len(mapped)), 0.92, dtype=np.float32)), axis=1)
    )
    assignments: dict[int, Candidate] = {}
    for node, column in zip(row_index, column_index):
        if column < len(query_ids) and chosen[node, column] >= 0 and costs[node, column] < 0.92:
            candidate = candidates[int(chosen[node, column])]
            assignments[candidate.query_index] = candidate
    return graph_fit_from_assignment(pattern, mapped, assignments, tolerance)


def graph_fit_from_assignment(
    pattern: Pattern, mapped: np.ndarray, assignments: dict[int, Candidate], tolerance: float
) -> Optional[GraphFit]:
    if len(assignments) < 3:
        return None
    points = np.asarray([(item.x, item.y) for item in assignments.values()], dtype=np.float32)
    error = np.linalg.norm(points[:, None, :] - mapped[None, :, :], axis=2).min(axis=1)
    mean_error = float(error.mean())
    support = len(assignments)
    # The dominant signal is a set of distinct queries explained by one graph.
    # A small coverage term only resolves ties between patterns of different
    # node counts; it does not assume the full figure was issued as patches.
    coverage = support / min(len(pattern.points), support)
    quality = float(support + 0.20 * coverage - mean_error / tolerance)
    return GraphFit(pattern, mapped, assignments, support, mean_error, tolerance, quality)


def fit_pattern(pattern: Pattern, candidates: list[Candidate], proposals: int, seed: int) -> Optional[GraphFit]:
    """Fit both possible handednesses of a supplied reference pattern."""
    best: Optional[GraphFit] = None
    for reflected, source in enumerate((pattern.points, pattern.points * np.array((1.0, -1.0), np.float32))):
        matrices, offsets = propose_transforms(source, candidates, proposals, seed + 10_007 * reflected)
        for matrix, offset, _ in top_hypotheses(source, candidates, matrices, offsets):
            fit = assign_queries(pattern, source, matrix, offset, candidates)
            if fit is not None and (best is None or fit.quality > best.quality):
                best = fit
    return best


def _fit_pattern_task(args: tuple[Pattern, list[Candidate], int, int]) -> Optional[GraphFit]:
    pattern, candidates, proposals, seed = args
    return fit_pattern(pattern, candidates, proposals, seed)


def choose_fit(patterns: list[Pattern], candidates: list[Candidate], proposals: int, seed: int) -> tuple[Optional[GraphFit], Optional[GraphFit]]:
    # This runs after the CUDA scene matcher has initialized.  Threads avoid
    # forking a process with a live CUDA context, which can deadlock in Colab.
    # NumPy/SciPy do the expensive numeric work outside Python's GIL.
    tasks = [(pattern, candidates, proposals, seed + 101 * index) for index, pattern in enumerate(patterns)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        fits = [fit for fit in executor.map(_fit_pattern_task, tasks) if fit is not None]
    fits.sort(key=lambda item: item.quality, reverse=True)
    return (fits[0], fits[1]) if len(fits) >= 2 else (fits[0] if fits else None, None)


def cache_path(cache_dir: Optional[Path], scene: str) -> Optional[Path]:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{scene}.json"


def scene_candidates(
    root: Path,
    split: str,
    scene: str,
    active: list[str],
    matcher: SceneMatcher,
    top_k: int,
    cache_dir: Optional[Path],
) -> list[list[Candidate]]:
    path = cache_path(cache_dir, scene)
    if path is not None and path.exists():
        saved = json.loads(path.read_text())
        if saved.get("columns") == active and saved.get("top_k") == top_k:
            return [
                [Candidate(query_index=index, x=int(x), y=int(y), score=float(score)) for x, y, score in group]
                for index, group in enumerate(saved["candidates"])
            ]
    def match_one(item: tuple[int, str]) -> list[Candidate]:
        query_index, column = item
        predictions = matcher.match_candidates(
            root / split / scene / "patches" / f"{column}.png", limit=top_k
        )
        return [Candidate(query_index, item.x, item.y, item.score) for item in predictions]

    # Colab supplies two CPU cores.  Individual patch matches are independent
    # and the ordered executor map preserves deterministic query-to-column
    # alignment, so use both cores without altering the candidate set.
    # A CUDA matcher already batches its transform bank on the device.  Feeding
    # it from two Python threads competes for the same GPU and makes the
    # otherwise deterministic candidate ranking less reproducible.  CPU
    # matching remains safely parallel across Colab's two cores.
    workers = 1 if getattr(matcher, "uses_cuda", False) else min(2, len(active))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        result = list(executor.map(match_one, enumerate(active)))
    if path is not None:
        path.write_text(
            json.dumps(
                {
                    "columns": active,
                    "top_k": top_k,
                    "candidates": [[[item.x, item.y, item.score] for item in group] for group in result],
                }
            )
        )
    return result


def membership_probabilities(
    root: Path, split: str, scene: str, active: list[str], model: MembershipClassifier
) -> np.ndarray:
    values = []
    for column in active:
        patch = cv2.imread(str(root / split / scene / "patches" / f"{column}.png"), cv2.IMREAD_GRAYSCALE)
        if patch is None:
            raise FileNotFoundError(f"Could not read {scene}/{column}")
        values.append(model.probability(patch))
    return np.asarray(values, dtype=np.float32)


def predict_row(
    root: Path,
    split: str,
    row: dict[str, str],
    config: MatcherConfig,
    patterns: list[Pattern],
    membership_model: MembershipClassifier,
    top_k: int,
    proposals: int,
    cache_dir: Optional[Path],
    seed: int,
    device: str,
    batch_size: int,
) -> dict[str, str]:
    scene = row["Id"]
    active = patch_columns(int(row["n_patches"]))
    image_paths = image_files(root / split / scene, "*_image.png")
    if len(image_paths) != 1:
        raise ValueError(f"Expected one sky image for {split}/{scene}")
    if device == "cuda":
        # Imported lazily so the CPU-only course path never requires PyTorch.
        from gpu_matcher import TorchCoarseSceneMatcher

        matcher = TorchCoarseSceneMatcher(image_paths[0], config, device, batch_size)
        matcher.uses_cuda = True
    else:
        matcher = SceneMatcher(image_paths[0], config)
    candidates_by_query = scene_candidates(root, split, scene, active, matcher, top_k, cache_dir)
    probabilities = membership_probabilities(root, split, scene, active, membership_model)
    # The classifier is a useful *proposal* filter, not a final membership
    # decision: the graph must still be allowed to rescue a dim figure star.
    # The 0.60 factor is fixed from the three labelled scenes and is applied
    # identically to every unseen scene, rather than using a scene-size rule.
    selection_floor = 0.60 * membership_model.threshold
    selected_queries = np.nonzero(probabilities >= selection_floor)[0].tolist()
    # Do not infer a pattern from fewer than four patch identities; fall back
    # to all queries instead of using an arbitrary patch-count heuristic.
    if len(selected_queries) < 4:
        selected_queries = list(range(len(active)))
    graph_candidates = [item for query in selected_queries for item in candidates_by_query[query]]
    best, runner = choose_fit(patterns, graph_candidates, proposals, seed)
    graph_assignments: dict[int, Candidate] = {}
    if best is not None and best.support >= 4:
        graph_assignments = best.assignments
        row["constellation"] = best.pattern.name
    else:
        row["constellation"] = "unknown"
    for query_index, column in enumerate(active):
        direct = candidates_by_query[query_index][0]
        if query_index in graph_assignments:
            point = graph_assignments[query_index]
            row[column] = format_cell(Prediction(point.x, point.y, m=1, score=point.score))
        elif direct.score >= config.presence_threshold:
            row[column] = format_cell(Prediction(direct.x, direct.y, m=0, score=direct.score))
        else:
            row[column] = "-1"
    error_text = f"{best.mean_error:.1f}" if best is not None else "n/a"
    print(
        f"{scene}: {row['constellation']} support={best.support if best else 0} "
        f"error={error_text} runner={runner.pattern.name if runner else 'none'}", flush=True
    )
    return row


def write_predictions(
    root: Path,
    split: str,
    template: Path,
    output: Path,
    config: MatcherConfig,
    top_k: int,
    proposals: int,
    cache_dir: Optional[Path],
    device: str,
    batch_size: int,
) -> None:
    patterns = load_patterns(root)
    membership_model = train_membership_classifier(root)
    rows = read_csv_rows(template)
    fieldnames = list(rows[0])
    for scene_index, row in enumerate(rows):
        predict_row(
            root,
            split,
            row,
            config,
            patterns,
            membership_model,
            top_k,
            proposals,
            cache_dir,
            seed=51_179 + scene_index,
            device=device,
            batch_size=batch_size,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    validate_submission(root, output)
    print(f"wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--proposals", type=int, default=6_000)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()
    if args.top_k < 2:
        parser.error("--top-k must be at least 2")
    if args.proposals < 100:
        parser.error("--proposals must be at least 100")
    root = args.root.resolve()
    write_predictions(
        root,
        "validation",
        root / "sample_submission.csv",
        args.output.resolve(),
        load_config(args.config.resolve()),
        args.top_k,
        args.proposals,
        args.cache_dir.resolve() if args.cache_dir else None,
        args.device,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
