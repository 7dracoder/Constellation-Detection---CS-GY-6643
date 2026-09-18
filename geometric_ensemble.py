#!/usr/bin/env python3
"""Course-data-only geometric ensemble for the constellation submission.

The query patches are noisy, degraded crops.  A single global template-match
peak is therefore often a distractor star.  This module keeps several
independent, deterministic match maps per supplied patch, then uses the
supplied constellation diagrams as a geometric consistency constraint.  No
network data, pretrained weights, or coordinates outside the coursework files
are used.

The search is checkpointed one validation scene at a time so it can run on the
CPU Cloud Bursting partition without depending on a permanently busy GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from scipy.spatial import cKDTree

from constellation_pipeline import (
    PATCH_RADIUS,
    format_cell,
    image_files,
    patch_columns,
    read_csv_rows,
    validate_submission,
)
from structural_refiner import Pattern, load_patterns


@dataclass(frozen=True)
class Candidate:
    query_index: int
    x: int
    y: int
    score: float


@dataclass(frozen=True)
class Hypothesis:
    pattern: Pattern
    matrix: np.ndarray
    offset: np.ndarray
    support: int
    error: float
    query_for_node: np.ndarray
    candidate_for_node: np.ndarray

    @property
    def quality(self) -> float:
        return float(self.support * 100.0 - self.error)


def preprocess(image: np.ndarray, mode: str) -> np.ndarray:
    """Feature maps selected by supplied training-scene validation only."""
    source = image.astype(np.float32)
    if mode == "raw":
        return source
    if mode == "dog":
        return cv2.GaussianBlur(source, (0, 0), 1.0) - cv2.GaussianBlur(source, (0, 0), 3.0)
    if mode == "grad":
        blurred = cv2.GaussianBlur(source, (0, 0), 1.0)
        return cv2.magnitude(
            cv2.Sobel(blurred, cv2.CV_32F, 1, 0),
            cv2.Sobel(blurred, cv2.CV_32F, 0, 1),
        )
    raise ValueError(f"Unknown preprocessing mode: {mode}")


def response_candidates(
    image: np.ndarray, patch: np.ndarray, query_index: int, per_mode: int
) -> list[Candidate]:
    """Return separated top locations from complementary match maps."""
    found: dict[tuple[int, int], Candidate] = {}
    for mode in ("raw", "dog", "grad"):
        response = cv2.matchTemplate(
            preprocess(image, mode), preprocess(patch, mode), cv2.TM_CCOEFF_NORMED
        )
        # Suppress nearby duplicate peaks.  Query locations are centres, hence
        # the PATCH_RADIUS offset after matching a top-left response position.
        peaks = response >= cv2.dilate(response, np.ones((13, 13), np.uint8))
        ys, xs = np.nonzero(peaks)
        if len(xs) == 0:
            continue
        scores = response[ys, xs]
        take = np.argsort(scores)[-min(per_mode, len(scores)) :]
        for item in take:
            key = (int(xs[item] + PATCH_RADIUS), int(ys[item] + PATCH_RADIUS))
            candidate = Candidate(query_index, key[0], key[1], float(scores[item]))
            prior = found.get(key)
            if prior is None or candidate.score > prior.score:
                found[key] = candidate
    return list(found.values())


def all_candidates(image: np.ndarray, patch_paths: Iterable[Path], per_mode: int) -> list[Candidate]:
    result: list[Candidate] = []
    for index, path in enumerate(patch_paths):
        patch = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if patch is None:
            raise FileNotFoundError(path)
        result.extend(response_candidates(image, patch, index, per_mode))
    return result


def candidate_pairs(candidates: list[Candidate]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique-query point-pairs suitable for similarity-transform proposals."""
    points = np.asarray([(item.x, item.y) for item in candidates], dtype=np.float32)
    query = np.asarray([item.query_index for item in candidates], dtype=np.int32)
    left, right = np.triu_indices(len(points), 1)
    keep = query[left] != query[right]
    left, right = left[keep], right[keep]
    delta = points[right] - points[left]
    length = np.linalg.norm(delta, axis=1)
    keep = length >= 20.0
    return left[keep], right[keep], length[keep]


def transform_from_pairs(
    pattern_points: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    node_a: int,
    node_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The two orientation choices are handled by ordered observed pairs."""
    source = pattern_points[node_b] - pattern_points[node_a]
    target = point_b - point_a
    scale = float(np.linalg.norm(target) / np.linalg.norm(source))
    cosine = float(np.dot(source, target) / (np.linalg.norm(source) * np.linalg.norm(target)))
    sine = float((source[0] * target[1] - source[1] * target[0]) / (np.linalg.norm(source) * np.linalg.norm(target)))
    matrix = scale * np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float32)
    offset = point_a - matrix @ pattern_points[node_a]
    return matrix, offset


def score_hypothesis(
    pattern: Pattern,
    matrix: np.ndarray,
    offset: np.ndarray,
    tree: cKDTree,
    candidates: list[Candidate],
    tolerance: float,
) -> Optional[Hypothesis]:
    mapped = pattern.points @ matrix.T + offset
    distance, index = tree.query(mapped, distance_upper_bound=tolerance)
    valid = index < len(candidates)
    if not np.any(valid):
        return None
    # A pattern should explain several distinct query crops, not repeated peaks
    # from one unusually bright or noisy query.
    query_for_node = np.full(len(mapped), -1, dtype=np.int32)
    candidate_for_node = np.full(len(mapped), -1, dtype=np.int32)
    best_by_query: dict[int, tuple[float, int, int]] = {}
    for node in np.nonzero(valid)[0]:
        candidate_index = int(index[node])
        candidate = candidates[candidate_index]
        old = best_by_query.get(candidate.query_index)
        if old is None or distance[node] < old[0]:
            best_by_query[candidate.query_index] = (float(distance[node]), int(node), candidate_index)
    for query_index, (_, node, candidate_index) in best_by_query.items():
        query_for_node[node] = query_index
        candidate_for_node[node] = candidate_index
    support = len(best_by_query)
    if support < 4:
        return None
    error = float(np.mean([item[0] for item in best_by_query.values()]))
    return Hypothesis(pattern, matrix, offset, support, error, query_for_node, candidate_for_node)


def fit_pattern(
    pattern: Pattern,
    candidates: list[Candidate],
    pair_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    proposals: int,
    seed: int,
) -> Optional[Hypothesis]:
    """Deterministic, bounded RANSAC over only supplied query candidates."""
    if len(candidates) < 2:
        return None
    points = np.asarray([(item.x, item.y) for item in candidates], dtype=np.float32)
    pair_left, pair_right, pair_length = pair_data
    if len(pair_length) == 0:
        return None
    pi, pj = np.triu_indices(len(pattern.points), 1)
    plen = np.linalg.norm(pattern.points[pj] - pattern.points[pi], axis=1)
    viable_nodes = np.nonzero(plen >= 8.0)[0]
    rng = np.random.default_rng(seed)
    tree = cKDTree(points)
    best: Optional[Hypothesis] = None
    # Scene-to-diagram scales measured on the supplied labelled scenes occupy
    # this interval.  The range remains deliberately broad for held-out scenes.
    for _ in range(proposals):
        node_pair = int(viable_nodes[rng.integers(len(viable_nodes))])
        # Rejection sampling avoids allocating a multi-million-entry
        # compatibility array for every RANSAC draw.  The deliberately broad
        # scale range accepts quickly on the supplied scenes.
        observed_pair = -1
        for _attempt in range(12):
            possible = int(rng.integers(len(pair_length)))
            ratio = float(pair_length[possible] / plen[node_pair])
            if 2.5 <= ratio <= 9.0:
                observed_pair = possible
                break
        if observed_pair < 0:
            continue
        for left, right in (
            (pair_left[observed_pair], pair_right[observed_pair]),
            (pair_right[observed_pair], pair_left[observed_pair]),
        ):
            matrix, offset = transform_from_pairs(
                pattern.points, points[left], points[right], int(pi[node_pair]), int(pj[node_pair])
            )
            hypothesis = score_hypothesis(pattern, matrix, offset, tree, candidates, tolerance=20.0)
            if hypothesis is not None and (best is None or hypothesis.quality > best.quality):
                best = hypothesis
    return best


def choose_pattern(patterns: list[Pattern], candidates: list[Candidate], proposals: int, seed: int) -> Optional[Hypothesis]:
    pair_data = candidate_pairs(candidates)
    hypotheses = [
        fit_pattern(pattern, candidates, pair_data, proposals, seed + index)
        for index, pattern in enumerate(patterns)
    ]
    valid = [item for item in hypotheses if item is not None]
    return max(valid, key=lambda item: item.quality) if valid else None


def default_cells(candidates: list[Candidate], count: int) -> list[Candidate]:
    """Use each query's best raw map candidate as a fallback coordinate."""
    result: list[Optional[Candidate]] = [None] * count
    for item in candidates:
        previous = result[item.query_index]
        if previous is None or item.score > previous.score:
            result[item.query_index] = item
    if any(item is None for item in result):
        raise RuntimeError("Every query needs at least one localisation candidate")
    return [item for item in result if item is not None]


def scene_prediction(
    root: Path, row: dict[str, str], patterns: list[Pattern], per_mode: int, proposals: int, seed: int
) -> tuple[dict[str, str], dict[str, object]]:
    """Score one scene against a selected group of reference patterns."""
    scene = row["Id"]
    active = patch_columns(int(row["n_patches"]))
    scene_dir = root / "validation" / scene
    image = cv2.imread(str(image_files(scene_dir, "*_image.png")[0]), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(scene)
    paths = [scene_dir / "patches" / f"{column}.png" for column in active]
    candidates = all_candidates(image, paths, per_mode)
    fallback = default_cells(candidates, len(active))
    fit = choose_pattern(patterns, candidates, proposals, seed=seed)
    for query_index, column in enumerate(active):
        item = fallback[query_index]
        membership = 0
        if fit is not None:
            node = np.nonzero(fit.query_for_node == query_index)[0]
            if len(node):
                item = candidates[int(fit.candidate_for_node[int(node[0])])]
                membership = 1
        row[column] = format_cell((item.x, item.y, membership))
    row["constellation"] = fit.pattern.name if fit is not None else "unknown"
    diagnostics: dict[str, object] = {
        "pattern": row["constellation"],
        "quality": round(fit.quality, 3) if fit else float("-inf"),
        "support": fit.support if fit else 0,
        "error": round(fit.error, 3) if fit else None,
        "candidate_count": len(candidates),
    }
    return row, diagnostics


def write_submission(root: Path, output: Path, per_mode: int, proposals: int) -> None:
    patterns = load_patterns(root)
    rows = read_csv_rows(root / "sample_submission.csv")
    fieldnames = list(rows[0])
    diagnostics: dict[str, dict[str, object]] = {}
    for scene_index, row in enumerate(rows):
        predicted, details = scene_prediction(root, row, patterns, per_mode, proposals, 13_579 + scene_index)
        row.update(predicted)
        diagnostics[row["Id"]] = details
        print(row["Id"], details, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    validate_submission(root, output)
    output.with_suffix(".diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(f"wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-mode", type=int, default=20)
    parser.add_argument("--proposals", type=int, default=100_000)
    parser.add_argument("--scene", help="One sample-submission scene for an array task")
    parser.add_argument("--pattern", help="One supplied reference pattern for an array task")
    parser.add_argument("--part-output", type=Path, help="JSON result for one scene/pattern task")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.part_output is None:
        if args.scene is not None or args.pattern is not None:
            parser.error("--scene/--pattern require --part-output")
        write_submission(root, args.output, args.per_mode, args.proposals)
        return
    if not args.scene or not args.pattern:
        parser.error("--part-output requires both --scene and --pattern")
    rows = {row["Id"]: row for row in read_csv_rows(root / "sample_submission.csv")}
    if args.scene not in rows:
        parser.error(f"Unknown sample scene: {args.scene}")
    patterns = {item.name: item for item in load_patterns(root)}
    if args.pattern not in patterns:
        parser.error(f"Unknown supplied pattern: {args.pattern}")
    predicted, details = scene_prediction(
        root, rows[args.scene], [patterns[args.pattern]], args.per_mode, args.proposals, 13_579
    )
    args.part_output.parent.mkdir(parents=True, exist_ok=True)
    args.part_output.write_text(json.dumps({"row": predicted, "diagnostics": details}, indent=2) + "\n")
    print(args.scene, details, flush=True)


if __name__ == "__main__":
    main()
