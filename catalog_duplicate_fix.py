#!/usr/bin/env python3
"""Catalog-gated duplicate-only identity repair on top of a scored CSV.

Starts from an established submission (typically the scored v4 presence CSV).
Unique labels are frozen.  Only scenes that share a duplicated constellation
name are reconsidered.  For those scenes the solver:

1. rebuilds ranked pattern fits from the existing candidate cache (same RANSAC
   geometry the rest of the repo uses);
2. scores each fit against the public d3-celestial line catalog; and
3. picks a one-to-one assignment of *disputed* scenes to patterns that are not
   already claimed by frozen unique scenes, maximising
   ``geometry_quality + catalog_weight * catalog_agreement``.

A disputed scene is only moved off its established label when the chosen
alternative is competitive in geometry *and* not worse on the catalog than a
configurable margin.  Whole rows are rewritten from the assigned pattern's own
transform so ``m=1`` coordinates stay consistent with the name.

This deliberately does not force all 16 labels to be distinct when the catalog
and geometry both support the same name on two scenes — the uniqueness
assumption is used only where evidence allows it.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from constellation_pipeline import (
    MatcherConfig,
    Prediction,
    format_cell,
    image_files,
    load_config,
    parse_cell,
    patch_columns,
    read_csv_rows,
    validate_submission,
)
from distinct_label_solver import (
    DISTINCT_SWAP_TOLERANCE,
    FITS_PER_SCENE,
    SceneFits,
    collect_ranked_fits,
    realise_row,
)
from external_catalog_validator import load_d3_patterns
from joint_geometric_solver import (
    Candidate,
    FitContext,
    GraphFit,
    assign_to_mapped,
    choose_fit,
    fit_pattern,
    membership_probabilities,
    presence_cutoff,
    scene_candidates,
)
from structural_refiner import Pattern, load_patterns, train_membership_classifier


# How much better the catalog must like an alternative before we leave a
# duplicated established label.  Tuned as a soft preference, not a hard veto:
# catalog conventions differ from the course diagrams.
CATALOG_SWAP_MARGIN = 0.5


def catalog_quality_for_row(
    root: Path,
    split: str,
    row: dict[str, str],
    catalog_patterns: list[Pattern],
    proposals: int,
    seed: int,
) -> tuple[Optional[GraphFit], Optional[GraphFit]]:
    """Fit catalog geometry to a row's predicted m=1 points."""
    candidates: list[Candidate] = []
    for query_index, column in enumerate(patch_columns(int(row["n_patches"]))):
        value = parse_cell(row[column])
        if value is not None and value[2] == 1:
            candidates.append(Candidate(query_index, value[0], value[1], 1.0, 0.0))
    if len(candidates) < 4:
        return None, None
    paths = image_files(root / split / row["Id"], "*_image.png")
    image = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Missing sky image for {row['Id']}")
    context = FitContext(len(candidates), float(image.size), float(len(candidates)))
    return choose_fit(catalog_patterns, candidates, proposals, seed, context, transform_model="affine")


def catalog_score_map(
    root: Path,
    split: str,
    row: dict[str, str],
    catalog_patterns: list[Pattern],
    proposals: int,
    seed: int,
) -> dict[str, float]:
    """Return {pattern_name: quality} for the catalog best and runner-up."""
    best, runner = catalog_quality_for_row(root, split, row, catalog_patterns, proposals, seed)
    scores: dict[str, float] = {}
    for fit in (best, runner):
        if fit is not None:
            scores[fit.pattern.name] = max(scores.get(fit.pattern.name, -1e9), fit.quality)
    return scores


def gather_disputed(
    root: Path,
    split: str,
    row: dict[str, str],
    config: MatcherConfig,
    patterns: list[Pattern],
    membership_model,
    args,
) -> SceneFits:
    """Same per-scene gather as distinct_label_solver, for disputed scenes only."""
    scene = row["Id"]
    active = patch_columns(int(row["n_patches"]))
    image_paths = image_files(root / split / scene, "*_image.png")
    matcher_image = image_paths[0]
    from constellation_pipeline import SceneMatcher

    matcher = SceneMatcher(matcher_image, config)
    height, width = matcher.image.shape
    image_area = float(height * width)
    candidates_by_query = scene_candidates(
        root, split, scene, active, matcher, args.top_k, args.cache_dir
    )
    probabilities = membership_probabilities(root, split, scene, active, membership_model)
    direct_scores = np.asarray([group[0].score for group in candidates_by_query], dtype=np.float32)
    cutoff = presence_cutoff(direct_scores, args.presence_mode, config, args.present_rate)
    present_count = int(np.sum(direct_scores >= cutoff))
    expected_figure = max(4.0, args.figure_rate * present_count)
    target = int(
        np.clip(round(args.graph_query_factor * expected_figure), args.min_graph_queries, args.max_graph_queries)
    )
    order = np.argsort(probabilities)[::-1]
    selected_queries = sorted(int(index) for index in order[: min(target, len(active))])
    fit_candidates = [
        item
        for query in selected_queries
        for item in candidates_by_query[query][: max(1, args.graph_top_k)]
    ]
    context = FitContext(len(fit_candidates), image_area, expected_figure)
    ranked = collect_ranked_fits(
        patterns,
        fit_candidates,
        args.proposals,
        args.seed,
        context,
        args.consensus_trials,
        args.transform_model,
        FITS_PER_SCENE,
    )
    # Force-evaluate catalog-preferred competition diagrams so a catalog
    # favourite is available even when it never won a RANSAC plurality.
    by_name = {fit.pattern.name: fit for fit in ranked}
    prefer = list(catalog_prefer_names(args, row))
    # Under force-unique also probe every unused (not reserved) pattern so the
    # second-pass uniqueness breaker has somewhere to move a colliding scene.
    if getattr(args, "force_unique", False):
        reserved = set(getattr(args, "reserved_names", set()))
        for pattern in patterns:
            if pattern.name not in by_name and pattern.name not in reserved:
                prefer.append(pattern.name)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    prefer_unique: list[str] = []
    for name in prefer:
        if name not in seen:
            seen.add(name)
            prefer_unique.append(name)
    force_proposals = min(args.proposals, 4_000) if getattr(args, "force_unique", False) else args.proposals
    for name in prefer_unique:
        pattern = next((item for item in patterns if item.name == name), None)
        if pattern is None or name in by_name:
            continue
        forced = fit_pattern(
            pattern,
            fit_candidates,
            force_proposals,
            args.seed + 17,
            context,
            args.transform_model,
        )
        if forced is not None and forced.support >= 4:
            by_name[name] = forced
    keep = max(FITS_PER_SCENE, 8, len(by_name) if getattr(args, "force_unique", False) else 0)
    ranked = sorted(by_name.values(), key=lambda fit: fit.quality, reverse=True)[:keep]
    return SceneFits(
        scene=scene,
        row=row,
        active=active,
        candidates_by_query=candidates_by_query,
        selected_queries=selected_queries,
        cutoff=cutoff,
        present_count=present_count,
        ranked_fits=ranked,
    )


def catalog_prefer_names(args, row: dict[str, str]) -> list[str]:
    """Names suggested by a prior catalog pass stored on args (may be empty)."""
    return list(getattr(args, "catalog_hints", {}).get(row["Id"], []))


def resolve_duplicates(
    established_rows: list[dict[str, str]],
    disputed: dict[str, SceneFits],
    catalog_scores: dict[str, dict[str, float]],
    catalog_weight: float,
    force_unique: bool,
) -> dict[str, str]:
    """Assign patterns to disputed scenes; freeze everyone else."""
    labels = {row["Id"]: row["constellation"] for row in established_rows}
    counts = Counter(labels.values())
    duplicate_names = {name for name, count in counts.items() if name != "unknown" and count > 1}
    disputed_scenes = [row["Id"] for row in established_rows if labels[row["Id"]] in duplicate_names]
    frozen = {
        row["Id"]: row["constellation"]
        for row in established_rows
        if row["Id"] not in disputed_scenes
    }
    reserved = set(frozen.values()) - {"unknown"}

    # Candidate patterns for disputed scenes: their ranked fits, excluding names
    # already taken by frozen unique scenes.
    pattern_names = sorted(
        {
            fit.pattern.name
            for scene in disputed_scenes
            for fit in disputed[scene].ranked_fits
            if fit.pattern.name not in reserved
        }
    )
    if not pattern_names:
        return labels

    name_to_col = {name: index for index, name in enumerate(pattern_names)}
    absent = -1_000.0
    reward = np.full((len(disputed_scenes), len(pattern_names)), absent, dtype=np.float64)
    for row_index, scene in enumerate(disputed_scenes):
        sf = disputed[scene]
        for fit in sf.ranked_fits:
            if fit.pattern.name not in name_to_col:
                continue
            value = fit.quality
            value += catalog_weight * float(catalog_scores.get(scene, {}).get(fit.pattern.name, 0.0))
            # Prefer keeping the established label when qualities are close,
            # unless force_unique must break a collision.
            if fit.pattern.name == labels[scene] and not force_unique:
                value += 0.35
            reward[row_index, name_to_col[fit.pattern.name]] = value

    cost = -reward
    if cost.shape[1] < cost.shape[0]:
        pad = np.full((cost.shape[0], cost.shape[0] - cost.shape[1]), -absent, dtype=np.float64)
        cost = np.concatenate((cost, pad), axis=1)
    row_index, col_index = linear_sum_assignment(cost)

    assigned = dict(frozen)
    for r, c in zip(row_index, col_index):
        scene = disputed_scenes[r]
        sf = disputed[scene]
        established = labels[scene]
        if c >= len(pattern_names) or reward[r, c] <= absent / 2:
            assigned[scene] = established
            continue
        chosen = pattern_names[c]
        chosen_fit = sf.fit_for(chosen)
        established_fit = sf.fit_for(established)
        if chosen_fit is None:
            assigned[scene] = established
            continue
        if chosen == established:
            assigned[scene] = established
            continue
        # Geometry guard: do not accept a clearly worse pattern.
        if (
            not force_unique
            and established_fit is not None
            and established_fit.quality - chosen_fit.quality > DISTINCT_SWAP_TOLERANCE
        ):
            assigned[scene] = established
            continue
        # Catalog guard (skipped under force_unique — uniqueness is then the
        # primary goal and catalog only influenced the reward matrix).
        cat = catalog_scores.get(scene, {})
        if (
            not force_unique
            and cat.get(established, 0.0) > cat.get(chosen, 0.0) + CATALOG_SWAP_MARGIN
        ):
            assigned[scene] = established
            continue
        assigned[scene] = chosen

    if force_unique:
        # Second pass: if a duplicated name remains among disputed scenes,
        # keep it on the scene with the highest reward for that name and move
        # the others to their best unused alternative.
        used = Counter(assigned.values())
        for name, count in list(used.items()):
            if name == "unknown" or count < 2:
                continue
            holders = [scene for scene in disputed_scenes if assigned[scene] == name]
            holders.sort(
                key=lambda scene: (
                    catalog_scores.get(scene, {}).get(name, 0.0)
                    + (disputed[scene].fit_for(name).quality if disputed[scene].fit_for(name) else -1e9)
                ),
                reverse=True,
            )
            keep = holders[0]
            claimed = set(assigned.values()) - {name}
            claimed.add(name)  # reserved by keep
            for scene in holders[1:]:
                sf = disputed[scene]
                alternatives = [
                    fit
                    for fit in sf.ranked_fits
                    if fit.pattern.name not in claimed and fit.pattern.name not in reserved
                ]
                if not alternatives:
                    continue
                best_alt = max(
                    alternatives,
                    key=lambda fit: fit.quality
                    + catalog_weight * catalog_scores.get(scene, {}).get(fit.pattern.name, 0.0),
                )
                assigned[scene] = best_alt.pattern.name
                claimed.add(best_alt.pattern.name)
    return assigned


def write_submission(root: Path, args) -> None:
    established_rows = read_csv_rows(args.established.resolve())
    labels = [row["constellation"] for row in established_rows]
    dups = sorted({name for name in labels if name != "unknown" and labels.count(name) > 1})
    print(f"established duplicates: {dups or 'none'}", flush=True)
    if not dups:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(established_rows[0]))
            writer.writeheader()
            writer.writerows(established_rows)
        validate_submission(root, output)
        print(f"no duplicates; copied established to {output}")
        return

    config = load_config(args.config.resolve())
    patterns = load_patterns(root)
    membership_model = train_membership_classifier(root)
    catalog_patterns = load_d3_patterns(root, args.catalog_root.resolve())
    print(f"loaded {len(catalog_patterns)} catalog patterns", flush=True)

    disputed_ids = {
        row["Id"]
        for row in established_rows
        if row["constellation"] in dups
    }
    # Names already claimed by non-disputed scenes — force-unique may not reuse them.
    args.reserved_names = {
        row["constellation"]
        for row in established_rows
        if row["Id"] not in disputed_ids and row["constellation"] != "unknown"
    }
    # Catalog pass on established m=1 points first — supplies preferred names
    # to force-fit and the agreement scores used in the assignment.
    catalog_scores: dict[str, dict[str, float]] = {}
    catalog_hints: dict[str, list[str]] = {}
    for scene_index, row in enumerate(established_rows):
        if row["Id"] not in disputed_ids:
            continue
        scores = catalog_score_map(
            root,
            args.split,
            row,
            catalog_patterns,
            min(args.proposals, 12_000),
            91_003 + scene_index,
        )
        catalog_scores[row["Id"]] = scores
        catalog_hints[row["Id"]] = list(scores.keys())
        print(f"catalog {row['Id']}: {scores}", flush=True)
    args.catalog_hints = catalog_hints

    disputed: dict[str, SceneFits] = {}
    for scene_index, row in enumerate(established_rows):
        if row["Id"] not in disputed_ids:
            continue
        print(f"fitting disputed scene {row['Id']} ({row['constellation']})", flush=True)
        args.seed = 77_001 + scene_index
        disputed[row["Id"]] = gather_disputed(
            root, args.split, dict(row), config, patterns, membership_model, args
        )
        print(
            f"  ranked={[f'{fit.pattern.name}:{fit.quality:.2f}' for fit in disputed[row['Id']].ranked_fits[:5]]}",
            flush=True,
        )

    assigned = resolve_duplicates(
        established_rows,
        disputed,
        catalog_scores,
        args.catalog_weight,
        force_unique=args.force_unique,
    )
    changes = []
    result_rows = []
    for row in established_rows:
        scene = row["Id"]
        chosen = assigned[scene]
        if scene in disputed and chosen != row["constellation"]:
            changes.append(f"{scene}:{row['constellation']}->{chosen}")
            result_rows.append(realise_row(disputed[scene], chosen))
        else:
            result_rows.append(dict(row))

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    validate_submission(root, output)
    final_labels = [row["constellation"] for row in result_rows]
    final_dups = sorted({name for name in final_labels if name != "unknown" and final_labels.count(name) > 1})
    print(f"changes: {'; '.join(changes) if changes else 'none'}", flush=True)
    print(f"final unique={len(set(final_labels))} dups={final_dups or 'none'}", flush=True)
    print(f"wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--established", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("matcher_config_gpu_wide.json"))
    parser.add_argument("--catalog-root", type=Path, default=Path("external/d3-celestial"))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/cache_validation"))
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--graph-top-k", type=int, default=3)
    parser.add_argument("--proposals", type=int, default=12_000)
    parser.add_argument("--consensus-trials", type=int, default=5)
    parser.add_argument("--transform-model", choices=("similarity", "affine"), default="affine")
    parser.add_argument("--presence-mode", choices=("threshold", "quantile"), default="quantile")
    parser.add_argument("--present-rate", type=float, default=0.625)
    parser.add_argument("--figure-rate", type=float, default=0.355)
    parser.add_argument("--graph-query-factor", type=float, default=2.0)
    parser.add_argument("--min-graph-queries", type=int, default=12)
    parser.add_argument("--max-graph-queries", type=int, default=30)
    parser.add_argument("--catalog-weight", type=float, default=1.0)
    parser.add_argument(
        "--force-unique",
        action="store_true",
        help="After catalog-weighted assignment, break any remaining duplicate labels among disputed scenes",
    )
    parser.add_argument("--seed", type=int, default=77_001)
    args = parser.parse_args()
    write_submission(args.root.resolve(), args)


if __name__ == "__main__":
    main()
