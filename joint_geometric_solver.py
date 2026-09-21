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
from scipy.stats import binom

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


# Relative weights of the coverage and figure-count terms against the
# significance term.  Selected on the three labelled training scenes.
SCORE_COVERAGE_WEIGHT = 1.0
SCORE_COUNT_WEIGHT = 1.0


@dataclass(frozen=True)
class Candidate:
    query_index: int
    x: int
    y: int
    score: float


@dataclass(frozen=True)
class FitContext:
    """Per-scene quantities the size-fair pattern score needs.

    ``cloud_size`` and ``image_area`` define the uniform null model used to
    ask how surprising a support count is.  ``expected_figure`` is the
    train-derived prior on how many query patches belong to the target
    figure.  None of these depend on the scene identity or its patch count
    being meaningful as a label.
    """

    cloud_size: int
    image_area: float
    expected_figure: float

    @property
    def minimum_support(self) -> int:
        """Smallest support that counts as identification evidence.

        A similarity transform has four degrees of freedom, so a four-node
        diagram matching four of many candidate points is close to
        unfalsifiable: RANSAC searches a large number of four-point subsets,
        and the per-transform null probability does not charge for that
        search.  Requiring a fit to explain at least half of the stars the
        figure is expected to contribute removes those trivial wins without
        hard-coding a node count.
        """
        return max(5, int(round(0.5 * self.expected_figure)))


@dataclass(frozen=True)
class GraphFit:
    pattern: Pattern
    mapped_points: np.ndarray
    assignments: dict[int, Candidate]
    support: int
    mean_error: float
    tolerance: float
    quality: float
    matrix: Optional[np.ndarray] = None
    offset: Optional[np.ndarray] = None
    coverage: float = 0.0
    significance: float = 0.0


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
    context: FitContext,
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
        return assign_queries_once(pattern, source_points, *refined, candidates, context)
    return graph_fit_from_assignment(pattern, mapped, assignments, tolerance, context, matrix, offset)


def assign_to_mapped(
    mapped: np.ndarray, candidates: list[Candidate], tolerance: float
) -> dict[int, Candidate]:
    """One-to-one Hungarian assignment of query patches onto mapped nodes.

    Split out of ``assign_queries_once`` so the same rule can be reapplied to a
    denser candidate cloud once a transform has already been chosen from a
    sparse, high-precision one.
    """
    query_ids = sorted({item.query_index for item in candidates})
    if not query_ids:
        return {}
    query_to_column = {query: column for column, query in enumerate(query_ids)}
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
    return assignments


def assign_queries_once(
    pattern: Pattern,
    source_points: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
    candidates: list[Candidate],
    context: FitContext,
) -> Optional[GraphFit]:
    """One non-recursive assignment pass used after least-squares refinement."""
    mapped = source_points @ matrix.T + offset
    scale = float(np.sqrt(abs(np.linalg.det(matrix))))
    tolerance = float(np.clip(4.5 * scale, 18.0, 45.0))
    assignments = assign_to_mapped(mapped, candidates, tolerance)
    return graph_fit_from_assignment(pattern, mapped, assignments, tolerance, context, matrix, offset)


def graph_fit_from_assignment(
    pattern: Pattern,
    mapped: np.ndarray,
    assignments: dict[int, Candidate],
    tolerance: float,
    context: FitContext,
    matrix: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
) -> Optional[GraphFit]:
    """Rank a partial-graph fit in a way that does not reward node count.

    The previous expression was ``support + 0.20 * coverage - error/tol`` with
    ``coverage = support / min(len(pattern.points), support)``.  Support never
    exceeds the node count, so that ``min`` always selected ``support`` and the
    coverage term was identically 1.0: the ranking reduced to raw support.
    Raw support favours large diagrams, because a free similarity transform
    over a dense candidate cloud finds more coincidences the more nodes it has
    to place.  Measured on this project's own output, every predicted label sat
    among the largest supplied diagrams.

    Three terms replace it:

    * ``significance`` - how improbable this support is under a uniform-null
      cloud of the same density.  A 27-node diagram must earn more support
      than an 11-node diagram to reach the same score.
    * ``coverage`` - the real fraction of the diagram's own nodes explained.
    * ``count_penalty`` - the labelled scenes place the figure at roughly a
      third of the present patches; a fit far from that is likelier to be a
      coincidence than a constellation.
    """
    if len(assignments) < 3:
        return None
    points = np.asarray([(item.x, item.y) for item in assignments.values()], dtype=np.float32)
    error = np.linalg.norm(points[:, None, :] - mapped[None, :, :], axis=2).min(axis=1)
    mean_error = float(error.mean())
    support = len(assignments)
    nodes = len(pattern.points)

    coverage = support / nodes
    hit_probability = float(
        np.clip(context.cloud_size * math.pi * tolerance * tolerance / context.image_area, 1e-9, 1.0 - 1e-9)
    )
    # -log10 P(Binomial(nodes, hit) >= support): a size-fair surprise measure.
    significance = float(-binom.logsf(support - 1, nodes, hit_probability) / math.log(10.0))
    if not math.isfinite(significance):
        significance = 0.0
    count_penalty = abs(support - context.expected_figure) / max(context.expected_figure, 1.0)

    quality = float(
        significance
        + SCORE_COVERAGE_WEIGHT * coverage
        - mean_error / tolerance
        - SCORE_COUNT_WEIGHT * count_penalty
    )
    return GraphFit(
        pattern,
        mapped,
        assignments,
        support,
        mean_error,
        tolerance,
        quality,
        matrix,
        offset,
        coverage,
        significance,
    )


def fit_pattern(
    pattern: Pattern, candidates: list[Candidate], proposals: int, seed: int, context: FitContext
) -> Optional[GraphFit]:
    """Fit both possible handednesses of a supplied reference pattern."""
    best: Optional[GraphFit] = None
    for reflected, source in enumerate((pattern.points, pattern.points * np.array((1.0, -1.0), np.float32))):
        matrices, offsets = propose_transforms(source, candidates, proposals, seed + 10_007 * reflected)
        for matrix, offset, _ in top_hypotheses(source, candidates, matrices, offsets):
            fit = assign_queries(pattern, source, matrix, offset, candidates, context)
            if fit is not None and (best is None or fit.quality > best.quality):
                best = fit
    return best


def _fit_pattern_task(args: tuple[Pattern, list[Candidate], int, int, FitContext]) -> Optional[GraphFit]:
    pattern, candidates, proposals, seed, context = args
    return fit_pattern(pattern, candidates, proposals, seed, context)


def choose_fit(
    patterns: list[Pattern],
    candidates: list[Candidate],
    proposals: int,
    seed: int,
    context: FitContext,
) -> tuple[Optional[GraphFit], Optional[GraphFit]]:
    # This runs after the CUDA scene matcher has initialized.  Threads avoid
    # forking a process with a live CUDA context, which can deadlock in Colab.
    # NumPy/SciPy do the expensive numeric work outside Python's GIL.
    tasks = [
        (pattern, candidates, proposals, seed + 101 * index, context)
        for index, pattern in enumerate(patterns)
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Two points always define a similarity transform, so a three-node
        # agreement is not identification evidence.  Such a fit previously
        # could still win on quality and, because the caller requires
        # support >= 4 to emit a name, silently force the scene to "unknown"
        # while a genuine larger fit existed.
        found = [fit for fit in executor.map(_fit_pattern_task, tasks) if fit is not None]
    # Prefer fits that clear the expected-figure floor.  If none does, still
    # return the best three-degrees-of-freedom-beating fit rather than nothing:
    # the identity term is scored as accuracy, so declining to name a scene
    # earns exactly what a wrong name earns, and a ranked guess can only help.
    eligible = [fit for fit in found if fit.support >= context.minimum_support]
    fits = eligible if eligible else [fit for fit in found if fit.support >= 4]
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


def presence_cutoff(scores: np.ndarray, mode: str, config: MatcherConfig, present_rate: float) -> float:
    """Decide the per-scene present/absent cut.

    A single global score threshold generalised badly here: the published
    0.66890 run marked 72.8% of patches present overall, with six scenes above
    90% and one at 100%, while the labelled scenes sit at 53-66%.  Matcher
    scores drift between scenes, so a rank-based cut calibrated to the training
    present-rate transfers better than one absolute number.  The threshold mode
    is retained so the two can be compared directly.
    """
    if mode == "threshold":
        return float(config.presence_threshold)
    keep = max(1, int(round(present_rate * len(scores))))
    ordered = np.sort(scores)[::-1]
    return float(ordered[min(keep, len(ordered)) - 1])


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
    graph_top_k: int,
    presence_mode: str,
    present_rate: float,
    figure_rate: float,
    graph_query_factor: float,
    min_graph_queries: int,
    max_graph_queries: int,
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
    height, width = matcher.image.shape
    image_area = float(height * width)
    candidates_by_query = scene_candidates(root, split, scene, active, matcher, top_k, cache_dir)
    probabilities = membership_probabilities(root, split, scene, active, membership_model)

    direct_scores = np.asarray([group[0].score for group in candidates_by_query], dtype=np.float32)
    cutoff = presence_cutoff(direct_scores, presence_mode, config, present_rate)
    present_count = int(np.sum(direct_scores >= cutoff))
    expected_figure = max(4.0, figure_rate * present_count)
    # The classifier is a useful *proposal* filter, not a final membership
    # decision: the graph must still be allowed to rescue a dim figure star.
    # The 0.60 factor is fixed from the three labelled scenes and is applied
    # identically to every unseen scene, rather than using a scene-size rule.
    # Select by RANK, not by an absolute probability.  The classifier's score
    # scale shifts between scenes: the 0.60*threshold floor picked 12-17
    # queries on each labelled scene but only 4-9 on several validation
    # scenes, and with four queries the support can never exceed four, so
    # every pattern looks like a coincidence.  Taking a scene-adaptive number
    # of the most figure-like queries keeps the cloud in the range the
    # labelled scenes exercised, whatever the probabilities happen to be.
    target = int(
        np.clip(round(graph_query_factor * expected_figure), min_graph_queries, max_graph_queries)
    )
    order = np.argsort(probabilities)[::-1]
    selected_queries = sorted(int(index) for index in order[: min(target, len(active))])

    # Fit the graph on a deliberately sparse, high-precision cloud.  Support
    # earned by coincidence grows with candidate density: on the labelled
    # scenes the true pattern stays rank 1 against ~50 points but falls to
    # rank 4 against ~110, and the full top-16 cloud is 336-1392 points.
    fit_candidates = [
        item
        for query in selected_queries
        for item in candidates_by_query[query][: max(1, graph_top_k)]
    ]
    context = FitContext(len(fit_candidates), image_area, expected_figure)
    best, runner = choose_fit(patterns, fit_candidates, proposals, seed, context)

    graph_assignments: dict[int, Candidate] = {}
    if best is not None and best.support >= 4:
        # Precision chose the pattern; recall now places the stars.  Re-run the
        # one-to-one assignment against every retained candidate so a figure
        # star whose best location sat outside the sparse cloud is recovered.
        dense = [item for query in selected_queries for item in candidates_by_query[query]]
        graph_assignments = assign_to_mapped(best.mapped_points, dense, best.tolerance)
        if len(graph_assignments) < best.support:
            graph_assignments = best.assignments
        row["constellation"] = best.pattern.name
    else:
        row["constellation"] = "unknown"

    for query_index, column in enumerate(active):
        direct = candidates_by_query[query_index][0]
        if query_index in graph_assignments:
            point = graph_assignments[query_index]
            row[column] = format_cell(Prediction(point.x, point.y, m=1, score=point.score))
        elif direct.score >= cutoff:
            row[column] = format_cell(Prediction(direct.x, direct.y, m=0, score=direct.score))
        else:
            row[column] = "-1"
    error_text = f"{best.mean_error:.1f}" if best is not None else "n/a"
    detail = (
        f"cov={best.coverage:.2f} sig={best.significance:.1f} q={best.quality:.2f}"
        if best is not None
        else ""
    )
    print(
        f"{scene}: {row['constellation']} support={best.support if best else 0} "
        f"assigned={len(graph_assignments)} error={error_text} {detail} "
        f"present={present_count}/{len(active)} cloud={len(fit_candidates)} "
        f"runner={runner.pattern.name if runner else 'none'}",
        flush=True,
    )
    return row


def blank_template(root: Path, split: str) -> list[dict[str, str]]:
    """Build a prediction template for a split with known scene ids.

    For ``validation`` this is the supplied sample submission.  For ``train``
    it reuses the labelled file's ids and patch counts with every prediction
    cell cleared, so the solver can be scored by ``evaluate.py`` without ever
    reading a training answer during prediction.
    """
    if split == "validation":
        return read_csv_rows(root / "sample_submission.csv")
    rows = read_csv_rows(root / "train_ground_truth.csv")
    for row in rows:
        for column in patch_columns(87):
            row[column] = "-1"
        row["constellation"] = "unknown"
    return rows


def write_predictions(
    root: Path,
    split: str,
    output: Path,
    config: MatcherConfig,
    top_k: int,
    proposals: int,
    cache_dir: Optional[Path],
    device: str,
    batch_size: int,
    graph_top_k: int,
    presence_mode: str,
    present_rate: float,
    figure_rate: float,
    graph_query_factor: float,
    min_graph_queries: int,
    max_graph_queries: int,
) -> None:
    patterns = load_patterns(root)
    membership_model = train_membership_classifier(root)
    rows = blank_template(root, split)
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
            graph_top_k=graph_top_k,
            presence_mode=presence_mode,
            present_rate=present_rate,
            figure_rate=figure_rate,
            graph_query_factor=graph_query_factor,
            min_graph_queries=min_graph_queries,
            max_graph_queries=max_graph_queries,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if split == "validation":
        validate_submission(root, output)
    print(f"wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument(
        "--graph-top-k",
        type=int,
        default=3,
        help="Candidates per query used to FIT the graph; --top-k is still used to place stars afterwards",
    )
    parser.add_argument(
        "--proposals",
        type=int,
        default=20_000,
        help=(
            "RANSAC transform proposals per pattern.  At 6000 (the previous "
            "default) the true pattern's support on a labelled scene varied "
            "between 6 and 9 across seeds; from 12000 it was stable at 9."
        ),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--presence-mode", choices=("threshold", "quantile"), default="quantile")
    parser.add_argument(
        "--present-rate",
        type=float,
        default=0.61,
        help="Fraction of patches marked present per scene in quantile mode (train prior: 0.61)",
    )
    parser.add_argument(
        "--figure-rate",
        type=float,
        default=0.355,
        help="Expected figure stars as a fraction of present patches (train prior: 0.33-0.38)",
    )
    parser.add_argument("--graph-query-factor", type=float, default=2.0)
    parser.add_argument("--min-graph-queries", type=int, default=12)
    parser.add_argument("--max-graph-queries", type=int, default=30)
    args = parser.parse_args()
    if args.top_k < 2:
        parser.error("--top-k must be at least 2")
    if args.graph_top_k < 1 or args.graph_top_k > args.top_k:
        parser.error("--graph-top-k must be between 1 and --top-k")
    if args.proposals < 100:
        parser.error("--proposals must be at least 100")
    root = args.root.resolve()
    write_predictions(
        root,
        args.split,
        args.output.resolve(),
        load_config(args.config.resolve()),
        args.top_k,
        args.proposals,
        args.cache_dir.resolve() if args.cache_dir else None,
        args.device,
        args.batch_size,
        args.graph_top_k,
        args.presence_mode,
        args.present_rate,
        args.figure_rate,
        args.graph_query_factor,
        args.min_graph_queries,
        args.max_graph_queries,
    )


if __name__ == "__main__":
    main()
