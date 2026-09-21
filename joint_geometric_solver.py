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
    x: int
    y: int
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
    mapped: np.ndarray, candidates: list[Candidate], tolerance: float
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
    score_values = np.asarray([item.score for item in candidates], dtype=np.float32)
    score_floor, score_span = float(score_values.min()), float(np.ptp(score_values) + 1e-5)
    margin_rank = _rank_normalize(np.asarray([item.margin for item in candidates], dtype=np.float32))
    for candidate_index, candidate in enumerate(candidates):
        column = query_to_column[candidate.query_index]
        distance = np.linalg.norm(mapped - np.asarray((candidate.x, candidate.y), np.float32), axis=1)
        confidence = ASSIGNMENT_SCORE_WEIGHT * (candidate.score - score_floor) / score_span + (
            ASSIGNMENT_MARGIN_WEIGHT * margin_rank[candidate_index]
        )
        value = distance / tolerance - confidence
        better = (distance <= tolerance) & (value < costs[:, column])
        costs[better, column] = value[better]
        chosen[better, column] = candidate_index
    return costs, chosen, query_ids


def assign_with_nodes(
    mapped: np.ndarray, candidates: list[Candidate], tolerance: float
) -> tuple[dict[int, int], dict[int, Candidate]]:
    """Hungarian assignment that also reports which node each query filled.

    The node index is needed to refit a transform from the current inliers
    (``source_points[node]`` pairs with ``assignments[query]``); callers that
    only need the placed points use :func:`assign_to_mapped` instead.
    """
    costs, chosen, query_ids = assignment_costs(mapped, candidates, tolerance)
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
    mapped: np.ndarray, candidates: list[Candidate], tolerance: float
) -> dict[int, Candidate]:
    """One-to-one Hungarian assignment of query patches onto mapped nodes.

    Used to re-place stars against a denser candidate cloud once a pattern
    and transform have already been chosen from a sparse one; see
    ``assign_with_nodes`` when the caller also needs to refit the transform.
    """
    _, assignments = assign_with_nodes(mapped, candidates, tolerance)
    return assignments


def assign_queries(
    pattern: Pattern,
    source_points: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
    candidates: list[Candidate],
    context: FitContext,
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
        refined = estimate_similarity(source_points[node_indices], target_points)
        if refined is None:
            break
        current_matrix, current_offset = refined
    return best


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


def choose_fit_consensus(
    patterns: list[Pattern],
    candidates: list[Candidate],
    proposals: int,
    seed: int,
    context: FitContext,
    trials: int,
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
        choose_fit(patterns, candidates, proposals, seed + 7_919 * trial, context) for trial in range(max(1, trials))
    ]
    winners = [best for best, _ in outcomes if best is not None]
    if not winners:
        return None, None
    tally: dict[str, int] = {}
    for fit in winners:
        tally[fit.pattern.name] = tally.get(fit.pattern.name, 0) + 1
    top_name = max(tally, key=lambda name: (tally[name], name))
    agreeing = [fit for fit in winners if fit.pattern.name == top_name]
    consensus = max(agreeing, key=lambda fit: fit.quality)
    other = max(
        (fit for fit in winners if fit.pattern.name != top_name),
        key=lambda fit: fit.quality,
        default=None,
    )
    return consensus, other


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
            # A cache written before ``margin`` existed has 3-tuple rows; a
            # missing margin defaults to 0.0, the same "no confidence signal"
            # value a fresh top-ranked candidate with no runner-up would get.
            # It never fabricates a value that would look like real evidence.
            return [
                [
                    Candidate(query_index=index, x=int(row[0]), y=int(row[1]), score=float(row[2]),
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
    consensus_trials: int,
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
    best, runner = choose_fit_consensus(patterns, fit_candidates, proposals, seed, context, consensus_trials)

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
    consensus_trials: int,
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
            consensus_trials=consensus_trials,
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
        args.consensus_trials,
    )


if __name__ == "__main__":
    main()
