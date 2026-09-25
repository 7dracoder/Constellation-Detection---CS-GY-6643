#!/usr/bin/env python3
"""Joint patch-localisation and constellation-geometry solver.

This is a course-data-only second-stage solver.  A query patch has many
visually plausible star-field locations, so selecting its visual top-1 before
looking at the constellation graph discards useful evidence.  Here each query
keeps a bounded shortlist, and a transform of one supplied pattern selects a
mutually consistent, one-to-one set of figure-star locations.  The default
search uses a similarity transform, while ``--transform-model affine`` starts
from those stable hypotheses and refines them with a full affine transform.

No scene name, patch count, external image, external label, or pretrained
model is used as a prediction feature.  Training labels calibrate the existing
present/absent threshold and the small patch-only membership selector.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import betabinom, binom

from constellation_pipeline import (
    MatcherConfig,
    Prediction,
    SceneMatcher,
    blob_response,
    format_cell,
    image_files,
    load_config,
    non_maximum_points,
    normalize_image,
    padded_extract,
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

# Search-time-only RANSAC acceptance band for the candidate-pair / node-pair
# length ratio.  See the comment at its use site in propose_transforms.
RANSAC_SCALE_RATIO_MIN = 1.3
RANSAC_SCALE_RATIO_MAX = 18.0

# Weight of each candidate's own match confidence in the one-to-one
# assignment cost, relative to geometric distance (which is normalised to
# [0, 1] by construction).  See assignment_costs.
ASSIGNMENT_SCORE_WEIGHT = 0.10
# Margin (a candidate's score gap over the next-best at the same rank) is
# plumbed all the way through from the matcher to this cost function, but
# its weight defaults to 0 - disabled - because it was tested, not assumed.
# On the three labelled training scenes it made the assignment slightly
# WORSE (train total 0.8393/0.8143 -> 0.8331/0.8081 at weight 0.15), and
# tracing the specific collision that motivated it (two patches' candidates
# landing on the same star in taurus) showed the true candidate's own
# margin was smaller than the wrong one's - margin does not reliably
# separate correct from coincidental matches in this data, at least not on
# this little evidence.  Left at 0 rather than shipping an invented weight
# that measurably did not help; the plumbing remains available for a better
# use of the signal later, or for re-evaluation once more labelled scenes
# exist.
ASSIGNMENT_MARGIN_WEIGHT = 0.0
ASSIGNMENT_DUMMY_COST = 0.92
MAX_REFIT_ITERATIONS = 5
TOLERANCE_SCALE_FACTOR = 4.5
TOLERANCE_MIN = 18.0
TOLERANCE_MAX = 45.0


@dataclass(frozen=True)
class Candidate:
    query_index: int
    x: float
    y: float
    score: float
    # Gap between this candidate's own match score and the next-best score at
    # the SAME rank in its query's shortlist (0.0 when unknown, e.g. a cache
    # file written before this field existed).  A low margin means the
    # matcher itself could not clearly tell this location apart from a
    # near-tied alternative for that query, which is exactly the situation
    # that lets an unrelated, coincidentally-placed patch win a pattern node
    # away from its true occupant in the Hungarian assignment below.
    margin: float = 0.0


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
    image_width: float = 0.0
    image_height: float = 0.0
    total_patches: int = 0
    patch_budget_weight: float = 0.0
    patch_budget_center: float = 2.97
    patch_budget_width: float = 0.35
    patch_budget_clip: float = 20.0
    star_points: Optional[np.ndarray] = None
    star_evidence_weight: float = 0.0
    star_tolerance: float = 20.0
    star_evidence_clip: float = 20.0
    duplicate_count: int = 0
    duplicate_coverage_weight: float = 0.0
    duplicate_coverage_rate: float = 0.10
    duplicate_coverage_clip: float = 20.0

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
    null_hit_probability: float = 0.0
    null_overdispersion: float = 0.0
    in_frame_nodes: int = 0
    patch_budget_log_prior: float = 0.0
    unmatched_star_hits: int = 0
    unmatched_star_trials: int = 0
    unmatched_star_log_likelihood: float = 0.0
    duplicate_coverage_log_prior: float = 0.0
    consensus_votes: int = 0
    consensus_trials: int = 0


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
        # This is a search-time filter only: it thins out proposals before the
        # expensive assignment step, it does not decide correctness (tolerance
        # and significance still do that downstream).  Widening it costs a
        # little RANSAC search time, never accuracy, so the bounds are set
        # generously rather than tightly around what three labelled scenes
        # happened to show.  Those three fitted scale factors were 4.4-6.5;
        # a tight [2, 10] cutoff would silently exclude the true transform
        # for any held-out scene whose diagram-to-image scale falls outside
        # it, with no downstream signal that anything was missed.
        keep = (ratio >= RANSAC_SCALE_RATIO_MIN) & (ratio <= RANSAC_SCALE_RATIO_MAX)
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


def _triangle_index(points: np.ndarray, minimum_side: float) -> tuple[np.ndarray, np.ndarray]:
    """Return canonically ordered non-degenerate triples and scale-free signatures.

    Vertices are ordered by the length of the opposite side.  That ordering is
    invariant to translation, rotation, uniform scale and reflection, so the
    first two ordered vertices define a consistent similarity hypothesis after
    two triangles have been matched by their side-length ratios.
    """
    if len(points) < 3:
        return np.empty((0, 3), np.int32), np.empty((0, 2), np.float32)
    triples = np.asarray(list(itertools.combinations(range(len(points)), 3)), dtype=np.int32)
    p0, p1, p2 = points[triples[:, 0]], points[triples[:, 1]], points[triples[:, 2]]
    opposite = np.column_stack((
        np.linalg.norm(p1 - p2, axis=1),
        np.linalg.norm(p2 - p0, axis=1),
        np.linalg.norm(p0 - p1, axis=1),
    ))
    longest = opposite.max(axis=1)
    cross = np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    )
    keep = (opposite.min(axis=1) >= minimum_side) & (
        cross / np.maximum(longest * longest, 1e-6) >= 0.035
    )
    triples, opposite, longest = triples[keep], opposite[keep], longest[keep]
    if not len(triples):
        return np.empty((0, 3), np.int32), np.empty((0, 2), np.float32)
    vertex_order = np.argsort(opposite, axis=1)
    ordered = np.take_along_axis(triples, vertex_order, axis=1)
    signature = np.sort(opposite, axis=1)[:, :2] / longest[:, None]
    return ordered.astype(np.int32), signature.astype(np.float32)


def propose_triangle_transforms(
    source_points: np.ndarray,
    candidates: list[Candidate],
    proposals: int,
    seed: int,
    signature_tolerance: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Propose similarities from compatible invariant triangle signatures."""
    if proposals <= 0 or len(candidates) < 3 or len(source_points) < 3:
        return np.empty((0, 2, 2), np.float32), np.empty((0, 2), np.float32)
    points = np.asarray([(item.x, item.y) for item in candidates], dtype=np.float32)
    queries = np.asarray([item.query_index for item in candidates], dtype=np.int32)
    source_triples, source_signatures = _triangle_index(source_points, 8.0)
    if not len(source_triples):
        return np.empty((0, 2, 2), np.float32), np.empty((0, 2), np.float32)
    signature_tree = cKDTree(source_signatures)
    rng = np.random.default_rng(seed)
    matrices: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    accepted = 0
    attempts = 0
    while accepted < proposals and attempts < 24:
        attempts += 1
        draw = max(2_048, 3 * (proposals - accepted))
        sampled = rng.integers(len(candidates), size=(draw, 3))
        distinct = (
            (queries[sampled[:, 0]] != queries[sampled[:, 1]])
            & (queries[sampled[:, 0]] != queries[sampled[:, 2]])
            & (queries[sampled[:, 1]] != queries[sampled[:, 2]])
        )
        sampled = sampled[distinct]
        if not len(sampled):
            continue
        candidate_points = points[sampled.reshape(-1)].reshape(-1, 3, 2)
        # Reuse the same invariant construction without materialising every
        # possible candidate triple, which would be cubic in the cloud size.
        p0, p1, p2 = candidate_points[:, 0], candidate_points[:, 1], candidate_points[:, 2]
        opposite = np.column_stack((
            np.linalg.norm(p1 - p2, axis=1),
            np.linalg.norm(p2 - p0, axis=1),
            np.linalg.norm(p0 - p1, axis=1),
        ))
        longest = opposite.max(axis=1)
        cross = np.abs(
            (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
            - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
        )
        valid = (opposite.min(axis=1) >= 24.0) & (
            cross / np.maximum(longest * longest, 1e-6) >= 0.035
        )
        sampled, opposite, longest = sampled[valid], opposite[valid], longest[valid]
        if not len(sampled):
            continue
        candidate_order = np.argsort(opposite, axis=1)
        candidate_triples = np.take_along_axis(sampled, candidate_order, axis=1)
        signatures = np.sort(opposite, axis=1)[:, :2] / longest[:, None]
        signature_distance, source_index = signature_tree.query(signatures, k=1)
        compatible = signature_distance <= signature_tolerance
        candidate_triples = candidate_triples[compatible]
        source_match = source_triples[np.asarray(source_index[compatible], dtype=np.int32)]
        if not len(candidate_triples):
            continue
        take = min(proposals - accepted, len(candidate_triples))
        candidate_triples, source_match = candidate_triples[:take], source_match[:take]
        matrix, offset = similarity_from_pairs(
            source_points,
            points,
            source_match[:, 0],
            source_match[:, 1],
            candidate_triples[:, 0],
            candidate_triples[:, 1],
        )
        scale = np.sqrt(np.abs(np.linalg.det(matrix)))
        scale_ok = (scale >= RANSAC_SCALE_RATIO_MIN) & (scale <= RANSAC_SCALE_RATIO_MAX)
        if np.any(scale_ok):
            matrices.append(matrix[scale_ok])
            offsets.append(offset[scale_ok])
            accepted += int(np.sum(scale_ok))
    if not matrices:
        return np.empty((0, 2, 2), np.float32), np.empty((0, 2), np.float32)
    return np.concatenate(matrices)[:proposals], np.concatenate(offsets)[:proposals]


def propose_hybrid_transforms(
    source_points: np.ndarray,
    candidates: list[Candidate],
    proposals: int,
    seed: int,
    proposal_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch pair, triangle, or evenly budgeted hybrid hypotheses."""
    if proposal_mode == "pair":
        return propose_transforms(source_points, candidates, proposals, seed)
    if proposal_mode == "triangle":
        return propose_triangle_transforms(source_points, candidates, proposals, seed)
    if proposal_mode != "hybrid":
        raise ValueError(f"Unknown proposal mode: {proposal_mode}")
    triangle_budget = proposals // 2
    pair_matrix, pair_offset = propose_transforms(
        source_points, candidates, proposals - triangle_budget, seed
    )
    triangle_matrix, triangle_offset = propose_triangle_transforms(
        source_points, candidates, triangle_budget, seed + 65_537
    )
    return (
        np.concatenate((pair_matrix, triangle_matrix)),
        np.concatenate((pair_offset, triangle_offset)),
    )


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
        tolerance = transform_tolerance(scale)
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


def transform_tolerance(scale):
    """Localisation tolerance for a similarity transform's own scale factor.

    A larger sky-to-diagram scale spreads pattern nodes over more image
    pixels, so admissible geometric error should scale with it too.  The
    multiplier and the [18, 45]px clamp are fixed from the three labelled
    scenes and are not re-derived here; this only centralises them so every
    caller moves together if they are ever revisited, instead of the three
    separate copies that existed before.  Accepts either a scalar or an
    array of scale factors.
    """
    return np.clip(TOLERANCE_SCALE_FACTOR * scale, TOLERANCE_MIN, TOLERANCE_MAX)


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


def estimate_affine(source: np.ndarray, target: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Least-squares full affine transform with degeneracy safeguards.

    Three non-collinear correspondences determine an affine transform.  We
    normally call this with a larger one-to-one consensus produced by the
    similarity RANSAC search, which avoids the huge false-hypothesis space of
    blind three-point affine RANSAC.  The singular-value checks reject nearly
    collapsed or numerically explosive fits before they can attract unrelated
    candidates on the next ICP-style assignment pass.
    """
    if len(source) < 3:
        return None
    design = np.concatenate((source.astype(np.float64), np.ones((len(source), 1))), axis=1)
    if np.linalg.matrix_rank(design) < 3:
        return None
    coefficients, _, _, _ = np.linalg.lstsq(design, target.astype(np.float64), rcond=None)
    matrix = coefficients[:2].T
    offset = coefficients[2]
    singular = np.linalg.svd(matrix, compute_uv=False)
    if (
        not np.all(np.isfinite(singular))
        or singular[-1] < 0.35
        or singular[0] > 30.0
        or singular[0] / singular[-1] > 8.0
    ):
        return None
    return matrix.astype(np.float32), offset.astype(np.float32)


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    """Percentile rank of each value in [0, 1].

    Min-max scaling was tried first and measured to fail here: on a labelled
    training scene, per-candidate margin was almost always within 0.001-0.009
    of zero, with the scene-wide min/max (-0.15, 0.12) set by a couple of
    outliers.  Min-max normalising against that range squashed the entire
    ordinary distribution into a thin, barely-informative sliver.  Order
    statistics ignore the outliers' magnitude and keep the ranking they
    still correctly imply.
    """
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks / max(len(values) - 1, 1)


def assignment_costs(
    mapped: np.ndarray,
    candidates: list[Candidate],
    tolerance: float,
    rank_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build the one-to-one assignment cost matrix shared by every caller.

    Cost is normalised geometric distance minus a small confidence bonus
    from the candidate's OWN localisation.  The score term keeps its
    original min-max normalisation unchanged, so that setting
    ``ASSIGNMENT_MARGIN_WEIGHT = 0`` reproduces the pre-margin assignment
    exactly rather than also silently changing how score was weighted.
    Margin - the gap over the next-best candidate at the same rank in its
    query's shortlist - uses percentile-rank normalisation instead: on a
    labelled training scene, per-candidate margin was almost always within
    0.001-0.009 of zero, with the scene-wide min/max set by a couple of
    outliers, so min-max would have squashed the entire ordinary
    distribution into a barely-informative sliver.

    A candidate that lands near a node purely by coincidence often scores
    and separates poorly compared to the rest of the pool, which is the
    situation, traced directly on this project's training scenes, where an
    unrelated patch's spurious location can outcompete a different query's
    true candidate for the same node.  This signal is necessary but not
    sufficient: on that same traced case, the true candidate's own margin
    was smaller than the wrong one's, so this cannot be assumed to resolve
    every such collision, only to shift the odds.
    """
    query_ids = sorted({item.query_index for item in candidates})
    costs = np.full((len(mapped), len(query_ids)), 1.25, dtype=np.float32)
    chosen = np.full((len(mapped), len(query_ids)), -1, dtype=np.int32)
    if not query_ids:
        return costs, chosen, query_ids
    query_to_column = {query: column for column, query in enumerate(query_ids)}
    if rank_weight < 0.0:
        raise ValueError("rank_weight must be nonnegative")
    ranks: dict[int, int] = {}
    score_values = np.asarray([item.score for item in candidates], dtype=np.float32)
    score_floor, score_span = float(score_values.min()), float(np.ptp(score_values) + 1e-5)
    margin_rank = _rank_normalize(np.asarray([item.margin for item in candidates], dtype=np.float32))
    for candidate_index, candidate in enumerate(candidates):
        column = query_to_column[candidate.query_index]
        rank = ranks.get(candidate.query_index, 0)
        ranks[candidate.query_index] = rank + 1
        distance = np.linalg.norm(mapped - np.asarray((candidate.x, candidate.y), np.float32), axis=1)
        confidence = ASSIGNMENT_SCORE_WEIGHT * (candidate.score - score_floor) / score_span + (
            ASSIGNMENT_MARGIN_WEIGHT * margin_rank[candidate_index]
        )
        # The candidate list is ordered best-first within each query.  Apply
        # this optional prior only when placing stars after a pattern and
        # transform have already won; RANSAC fitting remains unchanged.
        rank_penalty = rank_weight * math.log1p(rank) / math.log1p(15)
        value = distance / tolerance - confidence + rank_penalty
        better = (distance <= tolerance) & (value < costs[:, column])
        costs[better, column] = value[better]
        chosen[better, column] = candidate_index
    return costs, chosen, query_ids


def assign_with_nodes(
    mapped: np.ndarray,
    candidates: list[Candidate],
    tolerance: float,
    rank_weight: float = 0.0,
) -> tuple[dict[int, int], dict[int, Candidate]]:
    """Hungarian assignment that also reports which node each query filled.

    The node index is needed to refit a transform from the current inliers
    (``source_points[node]`` pairs with ``assignments[query]``); callers that
    only need the placed points use :func:`assign_to_mapped` instead.
    """
    costs, chosen, query_ids = assignment_costs(mapped, candidates, tolerance, rank_weight)
    if not query_ids:
        return {}, {}
    dummy = np.full((len(mapped), len(mapped)), ASSIGNMENT_DUMMY_COST, dtype=np.float32)
    row_index, column_index = linear_sum_assignment(np.concatenate((costs, dummy), axis=1))
    node_for_query: dict[int, int] = {}
    assignments: dict[int, Candidate] = {}
    for node, column in zip(row_index, column_index):
        if column >= len(query_ids) or chosen[node, column] < 0 or costs[node, column] >= ASSIGNMENT_DUMMY_COST:
            continue
        candidate = candidates[int(chosen[node, column])]
        assignments[candidate.query_index] = candidate
        node_for_query[candidate.query_index] = int(node)
    return node_for_query, assignments


def assign_to_mapped(
    mapped: np.ndarray,
    candidates: list[Candidate],
    tolerance: float,
    rank_weight: float = 0.0,
) -> dict[int, Candidate]:
    """One-to-one Hungarian assignment of query patches onto mapped nodes.

    Used to re-place stars against a denser candidate cloud once a pattern
    and transform have already been chosen from a sparse one; see
    ``assign_with_nodes`` when the caller also needs to refit the transform.
    """
    _, assignments = assign_with_nodes(mapped, candidates, tolerance, rank_weight)
    return assignments


def assign_queries(
    pattern: Pattern,
    source_points: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
    candidates: list[Candidate],
    context: FitContext,
    transform_model: str = "similarity",
    max_iterations: int = MAX_REFIT_ITERATIONS,
) -> Optional[GraphFit]:
    """Assign query patches to mapped pattern nodes, refitting to convergence.

    A single assignment pass can leave a query's correct candidate just
    outside tolerance purely from the residual error of the initial
    two-point RANSAC proposal - measured directly on this project's own
    training scenes: one true figure star sat 37.6px from its nearest mapped
    node against a 29.5px tolerance, with nothing else contesting that node.
    Re-estimating the transform from the current inliers and reassigning,
    repeated until the assignment stops changing (or a small iteration cap
    is hit), is the same refit-then-reassign idea as ICP registration, and
    lets the fit tighten around its own consensus instead of staying frozen
    at the seed proposal's accuracy.

    Each iteration is scored the same way this module already ranks its many
    RANSAC proposals against each other (``GraphFit.quality``), and whichever
    iteration scores highest is kept - not simply the last one computed. A
    refit that trades a few borderline, possibly-spurious inliers for a
    tighter and more accurate transform is not guaranteed to raise (or
    lower) the raw assignment count, so ranking by the same quality measure
    used everywhere else in this file is more principled than either always
    trusting the newest refit or always preferring the largest count.
    """
    current_matrix, current_offset = matrix, offset
    previous: dict[int, Candidate] = {}
    best: Optional[GraphFit] = None
    for _ in range(max_iterations):
        mapped = source_points @ current_matrix.T + current_offset
        scale = float(np.sqrt(abs(np.linalg.det(current_matrix))))
        tolerance = float(transform_tolerance(scale))
        node_for_query, assignments = assign_with_nodes(mapped, candidates, tolerance)
        fit = graph_fit_from_assignment(
            pattern, mapped, assignments, tolerance, context, current_matrix, current_offset
        )
        if fit is not None and (best is None or fit.quality > best.quality):
            best = fit
        converged = set(assignments) == set(previous) and all(
            assignments[q].x == previous[q].x and assignments[q].y == previous[q].y for q in assignments
        )
        previous = assignments
        if converged or len(assignments) < 3:
            break
        target_points = np.asarray([(c.x, c.y) for c in assignments.values()], np.float32)
        node_indices = np.asarray([node_for_query[q] for q in assignments], dtype=np.int32)
        refined = (
            estimate_affine(source_points[node_indices], target_points)
            if transform_model == "affine"
            else estimate_similarity(source_points[node_indices], target_points)
        )
        if refined is None:
            break
        current_matrix, current_offset = refined
    return best


def patch_budget_log_prior(
    total_patches: int,
    in_frame_nodes: int,
    center: float = 2.97,
    width: float = 0.35,
    clip: float = 20.0,
) -> float:
    """Bounded plausibility of query count relative to visible figure size.

    The ratio is evaluated in log space so equally large multiplicative errors
    receive equal penalties.  All inputs are available for an unseen scene;
    no scene identifier or pattern name participates.
    """
    if total_patches <= 0 or in_frame_nodes <= 0:
        return 0.0
    if center <= 0.0 or width <= 0.0 or clip < 0.0:
        raise ValueError("Patch-budget center/width must be positive and clip nonnegative")
    z = (math.log(total_patches / in_frame_nodes) - math.log(center)) / width
    return float(max(-0.5 * z * z, -clip))


def duplicate_coverage_log_prior(
    duplicate_count: int,
    in_frame_nodes: int,
    rate: float = 0.10,
    clip: float = 20.0,
) -> float:
    """Bounded likelihood of the observed copy count for a visible figure size."""
    if duplicate_count < 0 or in_frame_nodes < 0:
        raise ValueError("Duplicate and node counts must be nonnegative")
    if not 0.0 < rate < 1.0 or clip < 0.0:
        raise ValueError("Duplicate rate must be in (0, 1) and clip nonnegative")
    value = float(binom.logpmf(duplicate_count, in_frame_nodes, rate))
    return float(max(value, -clip)) if math.isfinite(value) else float(-clip)


def duplicate_candidate_scores(
    image: np.ndarray,
    candidates_by_query: list[list[Candidate]],
    top_k: int = 10,
    radius: int = 50,
    inner_radius: float = 20.0,
) -> np.ndarray:
    """Measure copy-paste evidence between each query's top candidate regions.

    This evidence may promote a query or add a bounded prior, but it never
    removes a candidate from the geometric search.
    """
    if top_k < 2 or radius < 3 or not 0.0 <= inner_radius < radius - 1:
        raise ValueError("Invalid duplicate-evidence geometry")
    size = 2 * radius
    yy, xx = np.mgrid[:size, :size]
    centre = (size - 1) / 2.0
    distance = np.hypot(xx - centre, yy - centre)
    mask = (distance >= inner_radius) & (distance <= radius - 2)
    scores = np.full(len(candidates_by_query), -1.0, dtype=np.float32)
    for query_index, group in enumerate(candidates_by_query):
        retained = group[:top_k]
        if len(retained) < 2:
            continue
        points = np.asarray([(item.x, item.y) for item in retained], dtype=np.float32)
        patches = padded_extract(image, np.rint(points).astype(np.int32), radius=radius)
        vectors = patches[:, mask].astype(np.float32)
        vectors -= vectors.mean(axis=1, keepdims=True)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-6
        scores[query_index] = float(np.max(vectors[1:] @ vectors[0]))
    return scores


def unmatched_node_star_evidence(
    mapped: np.ndarray,
    assignments: dict[int, Candidate],
    context: FitContext,
) -> tuple[int, int, float]:
    """Likelihood evidence from in-frame nodes not explained by query patches.

    Assigned nodes already contributed patch-match evidence, so counting their
    nearby stars again would double-count the same observation.  For every
    remaining visible node we compare a nearby detected-star hit against the
    chance rate implied by the scene's star density.  Both rewards and missing-
    star penalties are bounded, preventing this auxiliary term from dominating
    a confident geometric fit.
    """
    stars = context.star_points
    if (
        stars is None
        or not len(stars)
        or context.image_width <= 0.0
        or context.image_height <= 0.0
        or context.star_tolerance <= 0.0
    ):
        return 0, 0, 0.0
    mapped = np.asarray(mapped, dtype=np.float32)
    in_frame = (
        (mapped[:, 0] >= 0.0)
        & (mapped[:, 0] < context.image_width)
        & (mapped[:, 1] >= 0.0)
        & (mapped[:, 1] < context.image_height)
    )
    assigned_nodes: set[int] = set()
    if assignments:
        points = np.asarray(
            [(item.x, item.y) for item in assignments.values()], dtype=np.float32
        )
        nearest = np.linalg.norm(points[:, None, :] - mapped[None, :, :], axis=2).argmin(axis=1)
        assigned_nodes.update(int(index) for index in nearest)
    tested = np.asarray(
        [index for index in np.flatnonzero(in_frame) if int(index) not in assigned_nodes],
        dtype=np.int32,
    )
    if not len(tested):
        return 0, 0, 0.0
    distances = cKDTree(np.asarray(stars, dtype=np.float32)).query(mapped[tested], k=1)[0]
    hits = int(np.sum(distances <= context.star_tolerance))
    trials = int(len(tested))
    chance = float(np.clip(
        len(stars) * math.pi * context.star_tolerance**2 / context.image_area,
        1e-4,
        0.75,
    ))
    expected = 0.90
    log_likelihood = (
        hits * math.log(expected / chance)
        + (trials - hits) * math.log((1.0 - expected) / (1.0 - chance))
    )
    bounded = float(np.clip(
        log_likelihood, -context.star_evidence_clip, context.star_evidence_clip
    ))
    return hits, trials, bounded


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

    if context.image_width > 0.0 and context.image_height > 0.0:
        in_frame = (
            (mapped[:, 0] >= 0.0)
            & (mapped[:, 0] < context.image_width)
            & (mapped[:, 1] >= 0.0)
            & (mapped[:, 1] < context.image_height)
        )
        in_frame_nodes = int(np.sum(in_frame))
    else:
        in_frame_nodes = nodes
    budget_prior = patch_budget_log_prior(
        context.total_patches,
        in_frame_nodes,
        context.patch_budget_center,
        context.patch_budget_width,
        context.patch_budget_clip,
    )
    duplicate_prior = duplicate_coverage_log_prior(
        context.duplicate_count,
        in_frame_nodes,
        context.duplicate_coverage_rate,
        context.duplicate_coverage_clip,
    )
    star_hits, star_trials, star_likelihood = unmatched_node_star_evidence(
        mapped, assignments, context
    )

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
        + context.patch_budget_weight * budget_prior
        + context.star_evidence_weight * star_likelihood
        + context.duplicate_coverage_weight * duplicate_prior
    )
    return GraphFit(
        pattern=pattern,
        mapped_points=mapped,
        assignments=assignments,
        support=support,
        mean_error=mean_error,
        tolerance=tolerance,
        quality=quality,
        matrix=matrix,
        offset=offset,
        coverage=coverage,
        significance=significance,
        in_frame_nodes=in_frame_nodes,
        patch_budget_log_prior=budget_prior,
        unmatched_star_hits=star_hits,
        unmatched_star_trials=star_trials,
        unmatched_star_log_likelihood=star_likelihood,
        duplicate_coverage_log_prior=duplicate_prior,
    )


def _support_tail_significance(
    support: int,
    nodes: int,
    null_supports: np.ndarray,
) -> tuple[float, float, float]:
    """Fit an over-dispersed binomial null to random-placement supports.

    A plain binomial assumes every pattern node independently sees a uniform
    point cloud. Real star candidates occur in clusters and along textured
    structures, so those node hits are correlated. The beta-binomial keeps
    the measured mean hit rate while using the variance across random rigid
    placements to estimate that correlation.
    """
    samples = np.asarray(null_supports, dtype=np.float64)
    if nodes <= 0 or not len(samples):
        return 0.0, 0.0, 0.0
    hit_probability = float(np.clip(samples.mean() / nodes, 1e-9, 1.0 - 1e-9))
    binomial_variance = nodes * hit_probability * (1.0 - hit_probability)
    if nodes <= 1 or binomial_variance <= 1e-12:
        overdispersion = 0.0
    else:
        overdispersion = float(np.clip(
            (samples.var(ddof=1) / binomial_variance - 1.0) / (nodes - 1),
            0.0,
            0.95,
        ))
    if overdispersion <= 1e-6:
        log_tail = float(binom.logsf(support - 1, nodes, hit_probability))
    else:
        concentration = (1.0 - overdispersion) / overdispersion
        alpha = hit_probability * concentration
        beta = (1.0 - hit_probability) * concentration
        # scipy's direct beta-binomial logsf can return NaN for otherwise
        # valid extreme alpha/beta values because it computes log(1-cdf).
        # Summing the short discrete upper tail in log space is stable.
        upper = np.arange(max(0, support), nodes + 1, dtype=np.int32)
        log_probability = betabinom.logpmf(upper, nodes, alpha, beta)
        log_tail = float(np.logaddexp.reduce(log_probability))
    significance = -log_tail / math.log(10.0)
    if not math.isfinite(significance):
        significance = 0.0
    return float(significance), hit_probability, overdispersion


def recalibrate_fit_for_clutter(
    fit: GraphFit,
    candidates: list[Candidate],
    context: FitContext,
    trials: int,
    seed: int,
) -> GraphFit:
    """Re-score a fit against this scene's measured spatial clutter.

    The fitted diagram is randomly rotated and translated over the same image.
    At each placement we count how many nodes land within the fitted tolerance
    of the actual candidate cloud. This preserves clustered false stars, seams,
    and other scene-specific structures that a uniform-density null misses.
    No labels participate in the calibration.
    """
    if (
        trials <= 0
        or context.image_width <= 0.0
        or context.image_height <= 0.0
        or not candidates
        or not len(fit.mapped_points)
    ):
        return fit
    candidate_points = np.asarray([(item.x, item.y) for item in candidates], dtype=np.float32)
    tree = cKDTree(candidate_points)
    mapped = np.asarray(fit.mapped_points, dtype=np.float32)
    centered = mapped - mapped.mean(axis=0)
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * math.pi, size=trials)
    cosine, sine = np.cos(angles), np.sin(angles)
    rotations = np.stack((cosine, -sine, sine, cosine), axis=1).reshape(
        -1, 2, 2
    ).astype(np.float32)
    rotated = np.einsum("tij,nj->tni", rotations, centered)
    centres = np.column_stack((
        rng.uniform(0.0, context.image_width, size=trials),
        rng.uniform(0.0, context.image_height, size=trials),
    )).astype(np.float32)
    placed = rotated + centres[:, None, :]
    inside = (
        (placed[:, :, 0] >= 0.0)
        & (placed[:, :, 0] < context.image_width)
        & (placed[:, :, 1] >= 0.0)
        & (placed[:, :, 1] < context.image_height)
    )
    distances = tree.query(placed.reshape(-1, 2), k=1)[0].reshape(trials, -1)
    null_supports = np.sum(inside & (distances <= fit.tolerance), axis=1)
    significance, hit_probability, overdispersion = _support_tail_significance(
        fit.support, len(fit.pattern.points), null_supports
    )
    count_penalty = abs(fit.support - context.expected_figure) / max(
        context.expected_figure, 1.0
    )
    quality = float(
        significance
        + SCORE_COVERAGE_WEIGHT * fit.coverage
        - fit.mean_error / fit.tolerance
        - SCORE_COUNT_WEIGHT * count_penalty
        + context.patch_budget_weight * fit.patch_budget_log_prior
        + context.star_evidence_weight * fit.unmatched_star_log_likelihood
        + context.duplicate_coverage_weight * fit.duplicate_coverage_log_prior
    )
    return replace(
        fit,
        quality=quality,
        significance=significance,
        null_hit_probability=hit_probability,
        null_overdispersion=overdispersion,
    )


def fit_pattern(
    pattern: Pattern,
    candidates: list[Candidate],
    proposals: int,
    seed: int,
    context: FitContext,
    transform_model: str,
    proposal_mode: str = "pair",
) -> Optional[GraphFit]:
    """Fit both possible handednesses of a supplied reference pattern."""
    best: Optional[GraphFit] = None
    for reflected, source in enumerate((pattern.points, pattern.points * np.array((1.0, -1.0), np.float32))):
        matrices, offsets = propose_hybrid_transforms(
            source, candidates, proposals, seed + 10_007 * reflected, proposal_mode
        )
        for matrix, offset, _ in top_hypotheses(source, candidates, matrices, offsets):
            fit = assign_queries(
                pattern, source, matrix, offset, candidates, context, transform_model=transform_model
            )
            if fit is not None and (best is None or fit.quality > best.quality):
                best = fit
    return best


def _fit_pattern_task(
    args: tuple[Pattern, list[Candidate], int, int, FitContext, str, str]
) -> Optional[GraphFit]:
    pattern, candidates, proposals, seed, context, transform_model, proposal_mode = args
    return fit_pattern(
        pattern, candidates, proposals, seed, context, transform_model, proposal_mode
    )


def choose_fit(
    patterns: list[Pattern],
    candidates: list[Candidate],
    proposals: int,
    seed: int,
    context: FitContext,
    transform_model: str = "similarity",
    proposal_mode: str = "pair",
    clutter_trials: int = 0,
    pattern_workers: int = 2,
) -> tuple[Optional[GraphFit], Optional[GraphFit]]:
    # This runs after the CUDA scene matcher has initialized.  Threads avoid
    # forking a process with a live CUDA context, which can deadlock in Colab.
    # NumPy/SciPy do the expensive numeric work outside Python's GIL.
    tasks = [
        (pattern, candidates, proposals, seed + 101 * index, context, transform_model, proposal_mode)
        for index, pattern in enumerate(patterns)
    ]
    with ThreadPoolExecutor(max_workers=pattern_workers) as executor:
        # Two points always define a similarity transform, so a three-node
        # agreement is not identification evidence.  Such a fit previously
        # could still win on quality and, because the caller requires
        # support >= 4 to emit a name, silently force the scene to "unknown"
        # while a genuine larger fit existed.
        found = [fit for fit in executor.map(_fit_pattern_task, tasks) if fit is not None]
    if clutter_trials > 0:
        found = [
            recalibrate_fit_for_clutter(
                fit,
                candidates,
                context,
                clutter_trials,
                seed + 1_000_003 + 7_919 * index,
            )
            for index, fit in enumerate(found)
        ]
    # Prefer fits that clear the expected-figure floor.  If none does, still
    # return the best three-degrees-of-freedom-beating fit rather than nothing:
    # the identity term is scored as accuracy, so declining to name a scene
    # earns exactly what a wrong name earns, and a ranked guess can only help.
    eligible = [fit for fit in found if fit.support >= context.minimum_support]
    fits = eligible if eligible else [fit for fit in found if fit.support >= 4]
    fits.sort(key=lambda item: item.quality, reverse=True)
    return (fits[0], fits[1]) if len(fits) >= 2 else (fits[0] if fits else None, None)


def choose_fit_consensus(
    patterns: list[Pattern],
    candidates: list[Candidate],
    proposals: int,
    seed: int,
    context: FitContext,
    trials: int,
    transform_model: str = "similarity",
    proposal_mode: str = "pair",
    clutter_trials: int = 0,
    pattern_workers: int = 2,
    selection_mode: str = "plurality",
    vote_weight: float = 0.75,
) -> tuple[Optional[GraphFit], Optional[GraphFit]]:
    """Run ``choose_fit`` at several seeds and report the plurality winner.

    RANSAC's own seed can decide which of several similarly-plausible
    patterns wins a single run.  Measured directly on this project's own
    validation scenes: one scene's best pattern won 5 of 6 independent
    seed draws, yet the one fixed, arbitrary seed the rest of this pipeline
    used happened to draw the other one.  Trusting whichever pattern wins
    most often across a handful of independent draws - and, among the
    trials that agreed with it, its own highest-quality fit - is more
    robust than depending on one arbitrary seed's outcome.  ``trials``
    independent seeds cost roughly ``trials`` times the RANSAC search time;
    the candidate cloud itself is unaffected and nothing here reads or
    depends on scene identity.
    """
    outcomes = [
        choose_fit(
            patterns,
            candidates,
            proposals,
            seed + 7_919 * trial,
            context,
            transform_model,
            proposal_mode,
            clutter_trials,
            pattern_workers,
        )
        for trial in range(max(1, trials))
    ]
    winners = [best for best, _ in outcomes if best is not None]
    if not winners:
        return None, None
    tally: dict[str, int] = {}
    for fit in winners:
        tally[fit.pattern.name] = tally.get(fit.pattern.name, 0) + 1
    pool = [fit for outcome in outcomes for fit in outcome if fit is not None]
    best_by_name: dict[str, GraphFit] = {}
    for fit in pool:
        prior = best_by_name.get(fit.pattern.name)
        if prior is None or fit.quality > prior.quality:
            best_by_name[fit.pattern.name] = fit
    if selection_mode == "plurality":
        top_name = max(tally, key=lambda name: (tally[name], name))
    elif selection_mode == "quality":
        top_name = max(best_by_name, key=lambda name: best_by_name[name].quality)
    elif selection_mode == "vote-quality":
        top_name = max(
            best_by_name,
            key=lambda name: (
                best_by_name[name].quality + vote_weight * tally.get(name, 0),
                tally.get(name, 0),
                name,
            ),
        )
    else:
        raise ValueError(f"Unknown consensus selection mode: {selection_mode}")
    consensus = replace(
        best_by_name[top_name],
        consensus_votes=tally.get(top_name, 0),
        consensus_trials=len(outcomes),
    )
    other = max(
        (fit for name, fit in best_by_name.items() if name != top_name),
        key=lambda fit: fit.quality,
        default=None,
    )
    if other is not None:
        other = replace(
            other,
            consensus_votes=tally.get(other.pattern.name, 0),
            consensus_trials=len(outcomes),
        )
    return consensus, other


def cache_path(cache_dir: Optional[Path], scene: str) -> Optional[Path]:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{scene}.json"


def merge_dense_candidate_views(
    primary: list[list[Candidate]],
    alternates: list[list[list[Candidate]]],
    minimum_separation: float = 6.0,
) -> list[list[Candidate]]:
    """Union candidate views for final assignment without changing fit order.

    NCC values from raw, Gaussian and band-pass images are not on a shared
    numeric scale.  Preserve each alternate candidate's *rank* by mapping its
    score onto the primary view's score at the same rank.  Otherwise the
    global score normalisation in :func:`assignment_costs` can prefer an
    entire filtered view merely because its NCC distribution is shifted.
    """
    merged = [list(group) for group in primary]
    separation2 = minimum_separation * minimum_separation
    for view in alternates:
        if len(view) != len(merged):
            raise ValueError("Dense candidate views contain different query counts")
        for query_index, additions in enumerate(view):
            baseline = primary[query_index]
            if not baseline:
                raise ValueError("Primary dense candidate group cannot be empty")
            for rank, candidate in enumerate(additions):
                if all(
                    (candidate.x - old.x) ** 2 + (candidate.y - old.y) ** 2 > separation2
                    for old in merged[query_index]
                ):
                    reference = baseline[min(rank, len(baseline) - 1)]
                    merged[query_index].append(
                        replace(candidate, score=reference.score - 0.001, margin=0.0)
                    )
    return merged


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
            # A cache written before ``margin`` existed has 3-tuple rows; a
            # missing margin defaults to 0.0, the same "no confidence signal"
            # value a fresh top-ranked candidate with no runner-up would get.
            # It never fabricates a value that would look like real evidence.
            return [
                [
                    Candidate(query_index=index, x=float(row[0]), y=float(row[1]), score=float(row[2]),
                              margin=float(row[3]) if len(row) > 3 else 0.0)
                    for row in group
                ]
                for index, group in enumerate(saved["candidates"])
            ]
    def match_one(item: tuple[int, str]) -> list[Candidate]:
        query_index, column = item
        predictions = matcher.match_candidates(
            root / split / scene / "patches" / f"{column}.png", limit=top_k
        )
        return [Candidate(query_index, item.x, item.y, item.score, item.margin) for item in predictions]

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
                    "candidates": [
                        [[item.x, item.y, item.score, item.margin] for item in group] for group in result
                    ],
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


def graph_query_targets(
    expected_figure: float,
    graph_query_factor: float,
    expansion_factor: float,
    minimum: int,
    maximum: int,
    available: int,
) -> tuple[int, ...]:
    """Return unique narrow/broad membership shortlist sizes."""
    base = int(np.clip(round(graph_query_factor * expected_figure), minimum, maximum))
    expanded = int(np.clip(round(base * expansion_factor), minimum, maximum))
    return tuple(dict.fromkeys((min(base, available), min(expanded, available))))


def final_assignment_queries(
    scope: str, selected_queries: list[int], query_count: int
) -> list[int]:
    """Choose who may fill nodes after the transform and identity are fixed.

    RANSAC still uses the precision-oriented membership shortlist.  ``all``
    only broadens the final one-to-one assignment so a dim query omitted by
    that shortlist can be rescued by landing on a mapped constellation node.
    """
    if scope == "selected":
        return selected_queries
    if scope == "all":
        return list(range(query_count))
    raise ValueError(f"Unknown final assignment query scope: {scope}")


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
    dense_cache_dirs: tuple[Path, ...],
    seed: int,
    device: str,
    batch_size: int,
    graph_top_k: int,
    presence_mode: str,
    present_rate: float,
    figure_rate: float,
    graph_query_factor: float,
    graph_query_expansion_factor: float,
    min_graph_queries: int,
    max_graph_queries: int,
    consensus_trials: int,
    transform_model: str,
    proposal_mode: str = "pair",
    clutter_trials: int = 0,
    final_rank_weight: float = 0.0,
    final_query_scope: str = "selected",
    pattern_workers: int = 2,
    consensus_selection: str = "plurality",
    consensus_vote_weight: float = 0.75,
    patch_budget_weight: float = 0.0,
    patch_budget_center: float = 2.97,
    patch_budget_width: float = 0.35,
    star_evidence_weight: float = 0.0,
    star_evidence_points: int = 300,
    star_evidence_tolerance: float = 20.0,
    duplicate_evidence_cache_dir: Optional[Path] = None,
    duplicate_evidence_threshold: float = 0.95,
    duplicate_evidence_top_k: int = 10,
    duplicate_coverage_weight: float = 0.0,
    duplicate_coverage_rate: float = 0.10,
    duplicate_anchor_priority: bool = False,
    duplicate_force_present: bool = False,
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
    star_points: Optional[np.ndarray] = None
    if star_evidence_weight > 0.0:
        normalized = normalize_image(matcher.image, config.background_sigma)
        response = blob_response(normalized, config)
        detected, _ = non_maximum_points(response, config, border=4)
        star_points = detected[:star_evidence_points].astype(np.float32)
    candidates_by_query = scene_candidates(root, split, scene, active, matcher, top_k, cache_dir)
    use_duplicate_evidence = (
        duplicate_coverage_weight > 0.0
        or duplicate_anchor_priority
        or duplicate_force_present
    )
    duplicate_source = candidates_by_query
    if use_duplicate_evidence and duplicate_evidence_cache_dir is not None:
        duplicate_source = scene_candidates(
            root, split, scene, active, matcher, top_k, duplicate_evidence_cache_dir
        )
    duplicate_scores = (
        duplicate_candidate_scores(
            matcher.image, duplicate_source, top_k=duplicate_evidence_top_k
        )
        if use_duplicate_evidence
        else np.full(len(active), -1.0, dtype=np.float32)
    )
    duplicate_flags = duplicate_scores >= duplicate_evidence_threshold
    duplicate_count = int(np.sum(duplicate_flags))
    alternate_dense_views = [
        scene_candidates(root, split, scene, active, matcher, top_k, directory)
        for directory in dense_cache_dirs
        if cache_dir is None or directory.resolve() != cache_dir.resolve()
    ]
    dense_candidates_by_query = merge_dense_candidate_views(
        candidates_by_query, alternate_dense_views
    )
    probabilities = membership_probabilities(root, split, scene, active, membership_model)

    direct_scores = np.asarray([group[0].score for group in candidates_by_query], dtype=np.float32)
    cutoff = presence_cutoff(direct_scores, presence_mode, config, present_rate)
    present_count = int(np.sum(direct_scores >= cutoff))
    expected_figure = max(4.0, figure_rate * present_count)
    # The classifier is a useful *proposal* filter, not a final membership
    # decision: the graph must still be allowed to rescue a dim figure star.
    # Select by RANK, not by an absolute probability.  The classifier's score
    # scale shifts between scenes: the 0.60*threshold floor picked 12-17
    # queries on each labelled scene but only 4-9 on several validation
    # scenes, and with four queries the support can never exceed four, so
    # every pattern looks like a coincidence.  Taking a scene-adaptive number
    # of the most figure-like queries keeps the cloud in the range the
    # labelled scenes exercised, whatever the probabilities happen to be.
    proposal_priority = probabilities.copy()
    if duplicate_anchor_priority and duplicate_count:
        proposal_priority[duplicate_flags] = (
            float(np.max(proposal_priority)) + 1.0 + duplicate_scores[duplicate_flags]
        )
    order = np.argsort(proposal_priority)[::-1]
    targets = graph_query_targets(
        expected_figure,
        graph_query_factor,
        graph_query_expansion_factor,
        min_graph_queries,
        max_graph_queries,
        len(active),
    )
    base_target = targets[0]

    # A single membership width creates a brittle precision/recall choice:
    # the narrow cloud suppresses coincidences but can omit a real dim figure
    # patch, while a wider cloud can recover it at the cost of more accidental
    # alignments.  Search both widths and compare them with GraphFit.quality,
    # whose binomial significance term explicitly accounts for cloud density.
    # This is scene-agnostic and label-free; a factor of 1.0 preserves the
    # original single-width behavior exactly.
    width_results: list[
        tuple[GraphFit, Optional[GraphFit], list[int], list[Candidate], FitContext]
    ] = []
    for width_index, target in enumerate(targets):
        selected = sorted(int(index) for index in order[: min(target, len(active))])

        # Fit the graph on a deliberately sparse, high-precision cloud.
        fit_cloud = [
            item
            for query in selected
            for item in candidates_by_query[query][: max(1, graph_top_k)]
        ]
        fit_context = FitContext(
            cloud_size=len(fit_cloud),
            image_area=image_area,
            expected_figure=expected_figure,
            image_width=float(width),
            image_height=float(height),
            total_patches=len(active),
            patch_budget_weight=patch_budget_weight,
            patch_budget_center=patch_budget_center,
            patch_budget_width=patch_budget_width,
            star_points=star_points,
            star_evidence_weight=star_evidence_weight,
            star_tolerance=star_evidence_tolerance,
            duplicate_count=duplicate_count,
            duplicate_coverage_weight=duplicate_coverage_weight,
            duplicate_coverage_rate=duplicate_coverage_rate,
        )
        width_best, width_runner = choose_fit_consensus(
            patterns,
            fit_cloud,
            proposals,
            seed + 104_729 * width_index,
            fit_context,
            consensus_trials,
            transform_model,
            proposal_mode,
            clutter_trials,
            pattern_workers,
            consensus_selection,
            consensus_vote_weight,
        )
        if width_best is not None:
            width_results.append(
                (width_best, width_runner, selected, fit_cloud, fit_context)
            )

    if width_results:
        best, runner, selected_queries, fit_candidates, context = max(
            width_results, key=lambda result: result[0].quality
        )
        other_fits = [
            fit
            for width_best, width_runner, _, _, _ in width_results
            for fit in (width_best, width_runner)
            if fit is not None and fit.pattern.name != best.pattern.name
        ]
        runner = max(other_fits, key=lambda fit: fit.quality, default=runner)
    else:
        best = runner = None
        selected_queries = sorted(
            int(index) for index in order[: min(base_target, len(active))]
        )
        fit_candidates = [
            item
            for query in selected_queries
            for item in candidates_by_query[query][: max(1, graph_top_k)]
        ]
        context = FitContext(
            cloud_size=len(fit_candidates),
            image_area=image_area,
            expected_figure=expected_figure,
            image_width=float(width),
            image_height=float(height),
            total_patches=len(active),
            patch_budget_weight=patch_budget_weight,
            patch_budget_center=patch_budget_center,
            patch_budget_width=patch_budget_width,
            star_points=star_points,
            star_evidence_weight=star_evidence_weight,
            star_tolerance=star_evidence_tolerance,
            duplicate_count=duplicate_count,
            duplicate_coverage_weight=duplicate_coverage_weight,
            duplicate_coverage_rate=duplicate_coverage_rate,
        )

    graph_assignments: dict[int, Candidate] = {}
    if best is not None and best.support >= 4:
        # Precision chose the pattern; recall now places the stars.  Re-run the
        # one-to-one assignment against every retained candidate so a figure
        # star whose best location sat outside the sparse cloud is recovered.
        assignment_queries = final_assignment_queries(
            final_query_scope, selected_queries, len(active)
        )
        dense = [
            item
            for query in assignment_queries
            for item in dense_candidates_by_query[query]
        ]
        graph_assignments = assign_to_mapped(
            best.mapped_points, dense, best.tolerance, rank_weight=final_rank_weight
        )
        if len(graph_assignments) < best.support:
            graph_assignments = best.assignments
        row["constellation"] = best.pattern.name
    else:
        row["constellation"] = "unknown"

    for query_index, column in enumerate(active):
        direct = candidates_by_query[query_index][0]
        if query_index in graph_assignments:
            point = graph_assignments[query_index]
            row[column] = format_cell(
                Prediction(int(round(point.x)), int(round(point.y)), m=1, score=point.score)
            )
        elif (duplicate_force_present and duplicate_flags[query_index]) or direct.score >= cutoff:
            row[column] = format_cell(
                Prediction(int(round(direct.x)), int(round(direct.y)), m=0, score=direct.score)
            )
        else:
            row[column] = "-1"
    error_text = f"{best.mean_error:.1f}" if best is not None else "n/a"
    clutter_detail = (
        f" null-p={best.null_hit_probability:.3f} rho={best.null_overdispersion:.3f}"
        if best is not None and clutter_trials > 0
        else ""
    )
    detail = (
        f"cov={best.coverage:.2f} sig={best.significance:.1f} q={best.quality:.2f} "
        f"nodes={best.in_frame_nodes} pb={best.patch_budget_log_prior:.2f} "
        f"dup={duplicate_count} dup-ll={best.duplicate_coverage_log_prior:.2f} "
        f"stars={best.unmatched_star_hits}/{best.unmatched_star_trials} "
        f"star-llr={best.unmatched_star_log_likelihood:.2f}{clutter_detail}"
        if best is not None
        else ""
    )
    runner_detail = (
        f"runner={runner.pattern.name} runner-q={runner.quality:.2f} "
        f"margin={best.quality - runner.quality:.2f}"
        if best is not None and runner is not None
        else "runner=none"
    )
    print(
        f"{scene}: {row['constellation']} support={best.support if best else 0} "
        f"assigned={len(graph_assignments)} error={error_text} {detail} "
        f"present={present_count}/{len(active)} queries={len(selected_queries)} "
        f"final-queries={final_query_scope} "
        f"votes={best.consensus_votes}/{best.consensus_trials} "
        f"cloud={len(fit_candidates)} {runner_detail}",
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
    dense_cache_dirs: tuple[Path, ...],
    device: str,
    batch_size: int,
    graph_top_k: int,
    presence_mode: str,
    present_rate: float,
    figure_rate: float,
    graph_query_factor: float,
    graph_query_expansion_factor: float,
    min_graph_queries: int,
    max_graph_queries: int,
    consensus_trials: int,
    transform_model: str,
    cross_validated_membership: bool = False,
    final_rank_weight: float = 0.0,
    final_query_scope: str = "selected",
    pattern_workers: int = 2,
    consensus_selection: str = "plurality",
    consensus_vote_weight: float = 0.75,
    proposal_mode: str = "pair",
    clutter_trials: int = 0,
    patch_budget_weight: float = 0.0,
    patch_budget_center: float = 2.97,
    patch_budget_width: float = 0.35,
    star_evidence_weight: float = 0.0,
    star_evidence_points: int = 300,
    star_evidence_tolerance: float = 20.0,
    duplicate_evidence_cache_dir: Optional[Path] = None,
    duplicate_evidence_threshold: float = 0.95,
    duplicate_evidence_top_k: int = 10,
    duplicate_coverage_weight: float = 0.0,
    duplicate_coverage_rate: float = 0.10,
    duplicate_anchor_priority: bool = False,
    duplicate_force_present: bool = False,
    requested_scenes: tuple[str, ...] = (),
    seed_offset: int = 0,
) -> None:
    if cross_validated_membership and split != "train":
        raise ValueError("Cross-validated membership is only valid for the train split")
    patterns = load_patterns(root)
    membership_model = None if cross_validated_membership else train_membership_classifier(root)
    all_rows = blank_template(root, split)
    fieldnames = list(all_rows[0])
    indexed_rows = list(enumerate(all_rows))
    if requested_scenes:
        requested = set(requested_scenes)
        known = {row["Id"] for row in all_rows}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown {split} scenes: {sorted(unknown)}")
        indexed_rows = [item for item in indexed_rows if item[1]["Id"] in requested]
    rows = [row for _, row in indexed_rows]
    for scene_index, row in indexed_rows:
        scene_membership_model = (
            train_membership_classifier(root, excluded_scene=row["Id"])
            if cross_validated_membership
            else membership_model
        )
        predict_row(
            root,
            split,
            row,
            config,
            patterns,
            scene_membership_model,
            top_k,
            proposals,
            cache_dir,
            dense_cache_dirs,
            seed=51_179 + scene_index + seed_offset,
            device=device,
            batch_size=batch_size,
            graph_top_k=graph_top_k,
            presence_mode=presence_mode,
            present_rate=present_rate,
            figure_rate=figure_rate,
            graph_query_factor=graph_query_factor,
            graph_query_expansion_factor=graph_query_expansion_factor,
            min_graph_queries=min_graph_queries,
            max_graph_queries=max_graph_queries,
            consensus_trials=consensus_trials,
            transform_model=transform_model,
            proposal_mode=proposal_mode,
            clutter_trials=clutter_trials,
            final_rank_weight=final_rank_weight,
            final_query_scope=final_query_scope,
            pattern_workers=pattern_workers,
            consensus_selection=consensus_selection,
            consensus_vote_weight=consensus_vote_weight,
            patch_budget_weight=patch_budget_weight,
            patch_budget_center=patch_budget_center,
            patch_budget_width=patch_budget_width,
            star_evidence_weight=star_evidence_weight,
            star_evidence_points=star_evidence_points,
            star_evidence_tolerance=star_evidence_tolerance,
            duplicate_evidence_cache_dir=duplicate_evidence_cache_dir,
            duplicate_evidence_threshold=duplicate_evidence_threshold,
            duplicate_evidence_top_k=duplicate_evidence_top_k,
            duplicate_coverage_weight=duplicate_coverage_weight,
            duplicate_coverage_rate=duplicate_coverage_rate,
            duplicate_anchor_priority=duplicate_anchor_priority,
            duplicate_force_present=duplicate_force_present,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if split == "validation" and not requested_scenes:
        validate_submission(root, output)
    elif requested_scenes:
        print(f"partial scene output: {', '.join(row['Id'] for row in rows)}", flush=True)
    print(f"wrote {output}", flush=True)


def main() -> None:
    # The RANSAC scale filter is consulted inside the proposal helpers, which
    # read these module globals rather than taking them as arguments. Pattern
    # fitting fans out over a ThreadPoolExecutor, which shares module state, so
    # rebinding them here reaches every worker.
    global RANSAC_SCALE_RATIO_MIN, RANSAC_SCALE_RATIO_MAX

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
    parser.add_argument(
        "--dense-cache-dir",
        type=Path,
        nargs="*",
        default=(),
        help=(
            "Additional candidate caches used only for final fixed-transform "
            "assignment; they never affect RANSAC fitting or identity ranking."
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--presence-mode", choices=("threshold", "quantile"), default="quantile")
    parser.add_argument(
        "--present-rate",
        type=float,
        default=0.625,
        help=(
            "Fraction of patches marked present per scene in quantile mode. "
            "0.625 is both the value that generated the 0.68196 Kaggle "
            "submission and the sweep optimum on the labelled scenes (train "
            "prior alone is 0.61; the two were previously different, so "
            "running with no flags reproduced neither)."
        ),
    )
    parser.add_argument(
        "--figure-rate",
        type=float,
        default=0.355,
        help="Expected figure stars as a fraction of present patches (train prior: 0.33-0.38)",
    )
    parser.add_argument("--graph-query-factor", type=float, default=2.0)
    parser.add_argument(
        "--graph-query-expansion-factor",
        type=float,
        default=1.0,
        help=(
            "Also search a wider membership-ranked query cloud and retain the "
            "fit with the better density-adjusted quality; 1.0 disables it"
        ),
    )
    parser.add_argument("--min-graph-queries", type=int, default=12)
    parser.add_argument("--max-graph-queries", type=int, default=30)
    parser.add_argument(
        "--consensus-trials",
        type=int,
        default=5,
        help=(
            "Independent RANSAC seeds voted per scene for the final pattern "
            "choice.  On this project's own validation scenes, a single fixed "
            "seed sometimes drew a pattern that won only 1 of 6 independent "
            "draws; voting trades roughly this factor more compute for not "
            "depending on one arbitrary seed."
        ),
    )
    parser.add_argument(
        "--transform-model",
        choices=("similarity", "affine"),
        default="similarity",
        help=(
            "Transform used after similarity-RANSAC initialization.  Affine "
            "adds independent axis scale and shear, matching the assignment's "
            "statement that pattern aspect ratio is unrelated to the scene."
        ),
    )
    parser.add_argument(
        "--proposal-mode",
        choices=("pair", "triangle", "hybrid"),
        default="pair",
        help=(
            "RANSAC seed generator. Hybrid spends half the fixed proposal "
            "budget on invariant triangle matches and preserves the pair baseline."
        ),
    )
    parser.add_argument(
        "--clutter-trials",
        type=int,
        default=0,
        help=(
            "Random label-free pattern placements used to calibrate graph "
            "significance against the scene's actual clustered candidate cloud. "
            "Zero preserves the established uniform-null baseline."
        ),
    )
    parser.add_argument(
        "--patch-budget-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the bounded log prior on total patches per transformed "
            "in-frame pattern node. Zero preserves established ranking."
        ),
    )
    parser.add_argument(
        "--patch-budget-center",
        type=float,
        default=2.97,
        help="Expected total-patches / in-frame-nodes ratio.",
    )
    parser.add_argument(
        "--patch-budget-width",
        type=float,
        default=0.35,
        help="Deliberately broad log-space standard deviation for the patch-budget prior.",
    )
    parser.add_argument(
        "--star-evidence-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of bounded likelihood evidence from detected stars at "
            "in-frame pattern nodes not already explained by patch assignments."
        ),
    )
    parser.add_argument(
        "--star-evidence-points",
        type=int,
        default=300,
        help="Number of strongest scene star peaks retained for node evidence.",
    )
    parser.add_argument(
        "--star-evidence-tolerance",
        type=float,
        default=20.0,
        help="Maximum pixel distance between an unmatched node and a detected star.",
    )
    parser.add_argument(
        "--duplicate-evidence-cache-dir",
        type=Path,
        help="Candidate cache used only to measure copy-region evidence.",
    )
    parser.add_argument(
        "--duplicate-evidence-threshold",
        type=float,
        default=0.95,
        help="Annulus NCC threshold for a duplicate/copy-evidence patch.",
    )
    parser.add_argument(
        "--duplicate-evidence-top-k",
        type=int,
        default=10,
        help="Candidate regions compared per query for copy evidence.",
    )
    parser.add_argument(
        "--duplicate-coverage-weight",
        type=float,
        default=0.0,
        help="Weight of the bounded duplicate-count likelihood in identity scoring.",
    )
    parser.add_argument(
        "--duplicate-coverage-rate",
        type=float,
        default=0.10,
        help="Expected duplicate-evidence rate per visible figure node.",
    )
    parser.add_argument(
        "--duplicate-anchor-priority",
        action="store_true",
        help="Place duplicate-evidence queries first in the RANSAC query shortlist.",
    )
    parser.add_argument(
        "--duplicate-force-present",
        action="store_true",
        help="Keep duplicate-evidence queries present without narrowing their candidates.",
    )
    parser.add_argument(
        "--cross-validated-membership",
        action="store_true",
        help="For train diagnostics, fit membership on the other scenes for each prediction",
    )
    parser.add_argument(
        "--final-rank-weight",
        type=float,
        default=0.0,
        help="Optional best-first rank penalty in final dense graph assignment only",
    )
    parser.add_argument(
        "--final-query-scope",
        choices=("selected", "all"),
        default="selected",
        help=(
            "Queries allowed to compete for nodes after identity and transform "
            "are fixed. 'all' can rescue dim figure patches excluded from the "
            "precision-oriented RANSAC shortlist."
        ),
    )
    parser.add_argument(
        "--pattern-workers",
        type=int,
        default=2,
        help="Pattern hypotheses evaluated concurrently for each RANSAC seed",
    )
    parser.add_argument(
        "--consensus-selection",
        choices=("plurality", "quality", "vote-quality"),
        default="plurality",
        help="How independent RANSAC seeds select the final identity",
    )
    parser.add_argument(
        "--consensus-vote-weight",
        type=float,
        default=0.75,
        help="Per-winning-seed bonus used by vote-quality consensus",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Process only this scene; repeat for checkpointable partial runs",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Shift every scene's RANSAC seed to test identity stability",
    )
    parser.add_argument(
        "--scale-ratio-min",
        type=float,
        default=RANSAC_SCALE_RATIO_MIN,
        help=(
            "Lower bound of the search-time diagram-to-sky scale band. All 48 "
            "supplied diagrams span 353-461 px, so this ratio is comparable "
            "across diagrams rather than an artefact of one canvas size; the "
            "three labelled scenes measure 4.42, 4.87 and 6.38. The default "
            "band is deliberately permissive and admits many spurious fits."
        ),
    )
    parser.add_argument(
        "--scale-ratio-max",
        type=float,
        default=RANSAC_SCALE_RATIO_MAX,
        help="Upper bound of the search-time diagram-to-sky scale band",
    )
    args = parser.parse_args()
    if args.top_k < 2:
        parser.error("--top-k must be at least 2")
    if args.graph_top_k < 1 or args.graph_top_k > args.top_k:
        parser.error("--graph-top-k must be between 1 and --top-k")
    if args.proposals < 100:
        parser.error("--proposals must be at least 100")
    if args.graph_query_expansion_factor < 1.0:
        parser.error("--graph-query-expansion-factor must be at least 1.0")
    if args.cross_validated_membership and args.split != "train":
        parser.error("--cross-validated-membership requires --split train")
    if args.final_rank_weight < 0.0:
        parser.error("--final-rank-weight must be nonnegative")
    if args.pattern_workers < 1:
        parser.error("--pattern-workers must be positive")
    if args.consensus_vote_weight < 0.0:
        parser.error("--consensus-vote-weight must be nonnegative")
    if args.clutter_trials < 0:
        parser.error("--clutter-trials must be nonnegative")
    if args.patch_budget_weight < 0.0:
        parser.error("--patch-budget-weight must be nonnegative")
    if args.patch_budget_center <= 0.0 or args.patch_budget_width <= 0.0:
        parser.error("--patch-budget-center and --patch-budget-width must be positive")
    if args.star_evidence_weight < 0.0:
        parser.error("--star-evidence-weight must be nonnegative")
    if args.star_evidence_points < 1 or args.star_evidence_tolerance <= 0.0:
        parser.error("--star-evidence-points and --star-evidence-tolerance must be positive")
    if not -1.0 <= args.duplicate_evidence_threshold <= 1.0:
        parser.error("--duplicate-evidence-threshold must be between -1 and 1")
    if args.duplicate_evidence_top_k < 2:
        parser.error("--duplicate-evidence-top-k must be at least 2")
    if args.duplicate_coverage_weight < 0.0:
        parser.error("--duplicate-coverage-weight must be nonnegative")
    if not 0.0 < args.duplicate_coverage_rate < 1.0:
        parser.error("--duplicate-coverage-rate must be between 0 and 1")
    if args.scale_ratio_min <= 0.0:
        parser.error("--scale-ratio-min must be positive")
    if args.scale_ratio_max <= args.scale_ratio_min:
        parser.error("--scale-ratio-max must exceed --scale-ratio-min")
    RANSAC_SCALE_RATIO_MIN = args.scale_ratio_min
    RANSAC_SCALE_RATIO_MAX = args.scale_ratio_max
    root = args.root.resolve()
    write_predictions(
        root,
        args.split,
        args.output.resolve(),
        load_config(args.config.resolve()),
        args.top_k,
        args.proposals,
        args.cache_dir.resolve() if args.cache_dir else None,
        tuple(path.resolve() for path in args.dense_cache_dir),
        args.device,
        args.batch_size,
        args.graph_top_k,
        args.presence_mode,
        args.present_rate,
        args.figure_rate,
        args.graph_query_factor,
        args.graph_query_expansion_factor,
        args.min_graph_queries,
        args.max_graph_queries,
        args.consensus_trials,
        args.transform_model,
        args.cross_validated_membership,
        args.final_rank_weight,
        args.final_query_scope,
        args.pattern_workers,
        args.consensus_selection,
        args.consensus_vote_weight,
        args.proposal_mode,
        args.clutter_trials,
        args.patch_budget_weight,
        args.patch_budget_center,
        args.patch_budget_width,
        args.star_evidence_weight,
        args.star_evidence_points,
        args.star_evidence_tolerance,
        args.duplicate_evidence_cache_dir.resolve()
        if args.duplicate_evidence_cache_dir else None,
        args.duplicate_evidence_threshold,
        args.duplicate_evidence_top_k,
        args.duplicate_coverage_weight,
        args.duplicate_coverage_rate,
        args.duplicate_anchor_priority,
        args.duplicate_force_present,
        tuple(args.scene),
        args.seed_offset,
    )


if __name__ == "__main__":
    main()
