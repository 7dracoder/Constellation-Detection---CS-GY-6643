#!/usr/bin/env python3
"""Global distinct-label constellation solver.

The baseline :mod:`joint_geometric_solver` chooses each scene's constellation
independently, which lets two scenes claim the same figure (the scored v4/v5
outputs repeat ``canis-major`` and ``hydra``).  If the competition's validation
scenes are 16 *distinct* constellations - a property of how such datasets are
usually built, though not one the three labelled scenes can prove - then a
one-to-one assignment of scenes to constellations is strictly more information
than 16 independent choices, and can only help identity, which is 0.30 of the
metric.

This module:

1. reuses the existing per-scene candidate cloud, presence cut, membership
   selection, and RANSAC geometry from :mod:`joint_geometric_solver` (so
   nothing about localisation, presence, or the transform search changes);
2. collects, for each scene, a ranked table of *several* pattern fits with
   their quality scores rather than only the single best;
3. solves a global scene->constellation assignment that forbids duplicate
   labels while maximising total fit quality, optionally nudged by the public
   d3-celestial catalog cross-check as a tie-breaker; and
4. re-places the ``m=1`` figure stars using each scene's *assigned* pattern's
   own transform, so a swapped name never keeps another pattern's coordinates.

A scene only accepts a distinct label if doing so does not drop it far below
its own independent best fit.  When the constraint would force a clearly worse
pattern on a scene, that scene keeps its independent choice and the label may
still repeat - the assignment is used where it helps and abandoned where the
distinctness assumption fights the geometry.

No scene name, patch count, or memorised coordinate is used as a prediction
feature.  Training labels calibrate only the existing present/absent threshold
and the patch-only membership selector, exactly as before.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from constellation_pipeline import (
    MatcherConfig,
    Prediction,
    SceneMatcher,
    format_cell,
    image_files,
    load_config,
    patch_columns,
    read_csv_rows,
    validate_submission,
)
from joint_geometric_solver import (
    Candidate,
    FitContext,
    GraphFit,
    assign_to_mapped,
    choose_fit,
    membership_probabilities,
    presence_cutoff,
    scene_candidates,
)
from structural_refiner import Pattern, load_patterns, train_membership_classifier


# A scene keeps its own independent best pattern rather than accept a distinct
# label if the distinct label's fit quality is worse than the independent best
# by more than this margin.  Set from the ranking scale used in
# ``joint_geometric_solver`` (significance + coverage - error/tol - count),
# where a full correct constellation win over a coincidence is typically
# several units; a small tolerance here means "only swap when the alternative
# is genuinely competitive".  It is a guard against the distinctness
# assumption overriding clear geometry, not a tuned parameter.
DISTINCT_SWAP_TOLERANCE = 1.25

# How many ranked pattern fits to retain per scene for the global assignment.
# The assignment only ever needs enough alternatives to route around a
# collision; more than a handful of scenes rarely contend for the same figure.
FITS_PER_SCENE = 6


@dataclass
class SceneFits:
    """Everything the global assignment needs about one scene."""

    scene: str
    row: dict[str, str]
    active: list[str]
    candidates_by_query: list[list[Candidate]]
    selected_queries: list[int]
    cutoff: float
    present_count: int
    # Ranked, de-duplicated pattern fits (best first).  Each is a full GraphFit
    # so the assigned pattern's transform is available to re-place stars.
    ranked_fits: list[GraphFit]

    @property
    def independent_best(self) -> Optional[GraphFit]:
        return self.ranked_fits[0] if self.ranked_fits else None

    def fit_for(self, pattern_name: str) -> Optional[GraphFit]:
        for fit in self.ranked_fits:
            if fit.pattern.name == pattern_name:
                return fit
        return None


def collect_ranked_fits(
    patterns: list[Pattern],
    candidates: list[Candidate],
    proposals: int,
    seed: int,
    context: FitContext,
    consensus_trials: int,
    transform_model: str,
    keep: int,
) -> list[GraphFit]:
    """Return several distinct patterns' best fits for one scene, ranked.

    Runs ``choose_fit`` across independent RANSAC seeds (the same consensus
    idea the baseline uses for its single winner) and, for every pattern that
    ever wins or places second, keeps its highest-quality fit seen across the
    trials.  This gives the global assignment real alternatives to route a
    collision through, using only fits the ordinary solver would have trusted.
    """
    best_by_pattern: dict[str, GraphFit] = {}
    for trial in range(max(1, consensus_trials)):
        best, runner = choose_fit(
            patterns,
            candidates,
            proposals,
            seed + 7_919 * trial,
            context,
            transform_model,
        )
        for fit in (best, runner):
            if fit is None or fit.support < 4:
                continue
            current = best_by_pattern.get(fit.pattern.name)
            if current is None or fit.quality > current.quality:
                best_by_pattern[fit.pattern.name] = fit
    ranked = sorted(best_by_pattern.values(), key=lambda fit: fit.quality, reverse=True)
    return ranked[:keep]


def gather_scene_fits(
    root: Path,
    split: str,
    row: dict[str, str],
    config: MatcherConfig,
    patterns: list[Pattern],
    membership_model,
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
    transform_model: str,
) -> SceneFits:
    """Reproduce the per-scene stage of the baseline, but keep ranked fits."""
    scene = row["Id"]
    active = patch_columns(int(row["n_patches"]))
    image_paths = image_files(root / split / scene, "*_image.png")
    if len(image_paths) != 1:
        raise ValueError(f"Expected one sky image for {split}/{scene}")
    if device == "cuda":
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
    target = int(
        np.clip(round(graph_query_factor * expected_figure), min_graph_queries, max_graph_queries)
    )
    order = np.argsort(probabilities)[::-1]
    selected_queries = sorted(int(index) for index in order[: min(target, len(active))])

    fit_candidates = [
        item
        for query in selected_queries
        for item in candidates_by_query[query][: max(1, graph_top_k)]
    ]
    context = FitContext(len(fit_candidates), image_area, expected_figure)
    ranked_fits = collect_ranked_fits(
        patterns,
        fit_candidates,
        proposals,
        seed,
        context,
        consensus_trials,
        transform_model,
        FITS_PER_SCENE,
    )
    print(
        f"{scene}: ranked "
        + ", ".join(f"{fit.pattern.name}(q={fit.quality:.2f},s={fit.support})" for fit in ranked_fits[:4])
        + f" present={present_count}/{len(active)} cloud={len(fit_candidates)}",
        flush=True,
    )
    return SceneFits(
        scene=scene,
        row=row,
        active=active,
        candidates_by_query=candidates_by_query,
        selected_queries=selected_queries,
        cutoff=cutoff,
        present_count=present_count,
        ranked_fits=ranked_fits,
    )


def solve_distinct_assignment(
    scene_fits: list[SceneFits],
    catalog_scores: Optional[dict[str, dict[str, float]]] = None,
    catalog_weight: float = 0.0,
    strict: bool = False,
) -> dict[str, str]:
    """Assign each scene a distinct constellation, maximising total quality.

    Builds a scene x pattern reward matrix from the ranked fits, adds an
    optional small catalog-agreement bonus, and runs a one-to-one assignment
    (Hungarian) with the constraint that no constellation is used twice.
    Patterns a scene never fit at all get a strongly negative reward so the
    solver never routes a label through a scene it cannot geometrically
    support.  Returns ``scene -> chosen pattern name``.

    After the raw assignment, any scene whose assigned label is materially
    worse than its own independent best keeps the independent best instead:
    the distinctness assumption is used only where the geometry can afford it.
    """
    scenes = [sf.scene for sf in scene_fits]
    pattern_names = sorted({fit.pattern.name for sf in scene_fits for fit in sf.ranked_fits})
    if not pattern_names:
        return {sf.scene: (sf.independent_best.pattern.name if sf.independent_best else "unknown") for sf in scene_fits}

    name_to_col = {name: index for index, name in enumerate(pattern_names)}
    # Reward = fit quality (higher is better).  Absent fits get a large
    # negative so they are never chosen unless nothing else is available.
    absent = -1_000.0
    reward = np.full((len(scenes), len(pattern_names)), absent, dtype=np.float64)
    for row_index, sf in enumerate(scene_fits):
        for fit in sf.ranked_fits:
            value = fit.quality
            if catalog_scores and catalog_weight:
                value += catalog_weight * float(catalog_scores.get(sf.scene, {}).get(fit.pattern.name, 0.0))
            reward[row_index, name_to_col[fit.pattern.name]] = value

    # linear_sum_assignment minimises cost; negate the reward.  Pad columns so
    # every scene can be matched even when patterns are fewer than scenes
    # (they are not here, but keep it robust).
    cost = -reward
    if cost.shape[1] < cost.shape[0]:
        pad = np.full((cost.shape[0], cost.shape[0] - cost.shape[1]), -absent, dtype=np.float64)
        cost = np.concatenate((cost, pad), axis=1)
    row_index, col_index = linear_sum_assignment(cost)

    assigned: dict[str, str] = {}
    for r, c in zip(row_index, col_index):
        sf = scene_fits[r]
        if c < len(pattern_names) and reward[r, c] > absent / 2:
            assigned[sf.scene] = pattern_names[c]
        else:
            assigned[sf.scene] = sf.independent_best.pattern.name if sf.independent_best else "unknown"

    # Guard: unless strict distinctness is requested, never force a scene onto
    # a clearly worse pattern than its own best.  In strict mode the Hungarian
    # assignment is trusted as-is (every scene keeps a distinct label whenever
    # it has any valid fit), which is the right choice iff the 16 scenes really
    # are 16 distinct constellations.
    for sf in scene_fits:
        best = sf.independent_best
        if best is None:
            assigned[sf.scene] = "unknown"
            continue
        if strict:
            # Keep the assignment's choice if the scene can support it at all;
            # otherwise fall back to its own best rather than emit a label the
            # geometry cannot place.
            if sf.fit_for(assigned[sf.scene]) is None:
                assigned[sf.scene] = best.pattern.name
            continue
        chosen_fit = sf.fit_for(assigned[sf.scene])
        if chosen_fit is None or best.quality - chosen_fit.quality > DISTINCT_SWAP_TOLERANCE:
            assigned[sf.scene] = best.pattern.name
    return assigned


def realise_row(sf: SceneFits, chosen_pattern: str) -> dict[str, str]:
    """Fill a scene's CSV row given its final chosen pattern name."""
    row = sf.row
    fit = sf.fit_for(chosen_pattern)
    graph_assignments: dict[int, Candidate] = {}
    if fit is not None and fit.support >= 4:
        dense = [item for query in sf.selected_queries for item in sf.candidates_by_query[query]]
        graph_assignments = assign_to_mapped(fit.mapped_points, dense, fit.tolerance)
        if len(graph_assignments) < fit.support:
            graph_assignments = fit.assignments
        row["constellation"] = fit.pattern.name
    else:
        row["constellation"] = "unknown"

    for query_index, column in enumerate(sf.active):
        direct = sf.candidates_by_query[query_index][0]
        if query_index in graph_assignments:
            point = graph_assignments[query_index]
            row[column] = format_cell(Prediction(point.x, point.y, m=1, score=point.score))
        elif direct.score >= sf.cutoff:
            row[column] = format_cell(Prediction(direct.x, direct.y, m=0, score=direct.score))
        else:
            row[column] = "-1"
    assigned_count = len(graph_assignments)
    print(
        f"{sf.scene}: FINAL {row['constellation']} "
        f"support={fit.support if fit else 0} assigned={assigned_count}",
        flush=True,
    )
    return row


def load_catalog_scores(path: Optional[Path]) -> Optional[dict[str, dict[str, float]]]:
    """Optional catalog cross-check scores as {scene: {pattern: agreement}}.

    The file, if given, is a JSON object produced by an offline catalog run.
    Missing file or missing keys simply mean no bonus is applied; the catalog
    is a tie-breaker, never an override.
    """
    if path is None or not path.exists():
        return None
    import json

    data = json.loads(path.read_text())
    return {scene: {name: float(score) for name, score in scores.items()} for scene, scores in data.items()}


def blank_template(root: Path, split: str) -> list[dict[str, str]]:
    if split == "validation":
        return read_csv_rows(root / "sample_submission.csv")
    rows = read_csv_rows(root / "train_ground_truth.csv")
    for row in rows:
        for column in patch_columns(87):
            row[column] = "-1"
        row["constellation"] = "unknown"
    return rows


def write_predictions(root: Path, split: str, output: Path, args) -> None:
    patterns = load_patterns(root)
    membership_model = train_membership_classifier(root)
    rows = blank_template(root, split)
    fieldnames = list(rows[0])
    config = load_config(args.config.resolve())
    catalog_scores = load_catalog_scores(args.catalog_scores.resolve() if args.catalog_scores else None)

    scene_fits: list[SceneFits] = []
    for scene_index, row in enumerate(rows):
        scene_fits.append(
            gather_scene_fits(
                root,
                split,
                row,
                config,
                patterns,
                membership_model,
                args.top_k,
                args.proposals,
                args.cache_dir.resolve() if args.cache_dir else None,
                51_179 + scene_index,
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
                args.transform_model,
            )
        )

    assigned = solve_distinct_assignment(
        scene_fits, catalog_scores, args.catalog_weight, strict=args.strict_distinct
    )

    # Report which labels the distinct assignment changed versus independent best.
    changes = []
    for sf in scene_fits:
        independent = sf.independent_best.pattern.name if sf.independent_best else "unknown"
        if assigned[sf.scene] != independent:
            changes.append(f"{sf.scene}: {independent} -> {assigned[sf.scene]}")
    print("distinct-label changes: " + ("; ".join(changes) if changes else "none"), flush=True)

    result_rows = [realise_row(sf, assigned[sf.scene]) for sf in scene_fits]
    labels = [row["constellation"] for row in result_rows]
    duplicates = sorted({name for name in labels if name != "unknown" and labels.count(name) > 1})
    print(f"final distinct labels: {len(set(labels))} unique of {len(labels)}; duplicates: {duplicates or 'none'}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)
    if split == "validation":
        validate_submission(root, output)
    print(f"wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--graph-top-k", type=int, default=3)
    parser.add_argument("--proposals", type=int, default=20_000)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--presence-mode", choices=("threshold", "quantile"), default="quantile")
    parser.add_argument("--present-rate", type=float, default=0.625)
    parser.add_argument("--figure-rate", type=float, default=0.355)
    parser.add_argument("--graph-query-factor", type=float, default=2.0)
    parser.add_argument("--min-graph-queries", type=int, default=12)
    parser.add_argument("--max-graph-queries", type=int, default=30)
    parser.add_argument("--consensus-trials", type=int, default=5)
    parser.add_argument("--transform-model", choices=("similarity", "affine"), default="similarity")
    parser.add_argument(
        "--catalog-scores",
        type=Path,
        help="Optional JSON {scene:{pattern:agreement}} used only as an assignment tie-breaker",
    )
    parser.add_argument("--catalog-weight", type=float, default=0.0)
    parser.add_argument(
        "--strict-distinct",
        action="store_true",
        help="Trust the one-to-one assignment fully (no swap-guard fallback); use iff scenes are known distinct",
    )
    args = parser.parse_args()
    if args.top_k < 2:
        parser.error("--top-k must be at least 2")
    if args.graph_top_k < 1 or args.graph_top_k > args.top_k:
        parser.error("--graph-top-k must be between 1 and --top-k")
    write_predictions(args.root.resolve(), args.split, args.output.resolve(), args)


if __name__ == "__main__":
    main()
