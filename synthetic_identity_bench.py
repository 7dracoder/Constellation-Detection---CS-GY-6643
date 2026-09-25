#!/usr/bin/env python3
"""Make Identification measurable by synthesising labelled candidate clouds.

Identification is 30% of the competition metric, but only three scenes are
labelled and the solver already scores 3/3 on them. That leaves no signal to
optimise against: any identity change is a coin flip on the largest single term.
This module removes that blind spot without using any external data.

Every generator parameter is measured from the supplied data rather than guessed:

  transform      pure similarity. Fitting each labelled scene's own diagram
                 against a clutter-free cloud of its true figure stars gives
                 anisotropy 1.000, 1.000 and 1.033 with an axis angle of exactly
                 90 degrees, so the curators used rotation + uniform scale +
                 translation, not a general affine.
  handedness     determinant positive on all three labelled scenes, so
                 reflection defaults to off and is exposed as a knob.
  scale          all 48 diagrams span 353-461 px, so the diagram-to-sky scale is
                 transferable across diagrams. Measured 4.42-6.38; sampled here
                 from 4.2-6.6.
  partial figure issued figure stars are 55-77% of diagram nodes (10 of 18,
                 10 of 13, 6 of 11).
  retention      the retained candidate list holds the true location for 67 of
                 71 labelled-present patches, so a figure star reaches the cloud
                 with probability 0.94.
  clutter        sampled from real cached candidate positions, so the false-star
                 cloud keeps its true clustering instead of being uniform.
  cloud shape    the identity cloud is graph_top_k candidates for each of up to
                 max_graph_queries selected patches, so it is built as 3 points
                 per selected query.

Usage:
    python synthetic_identity_bench.py --trials 32 --consensus 1 --workers 8
    python synthetic_identity_bench.py --trials 32 --config similarity_tight_clutter
"""

from __future__ import annotations

import os

# Each worker process would otherwise start a full BLAS thread pool, and a pool
# of workers then exhausts the Windows paging file. The search is already
# parallel across trials, so per-process BLAS threading buys nothing. These must
# be set before numpy is first imported, including in spawned child processes.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import joint_geometric_solver as J
from structural_refiner import Pattern, load_patterns

IMAGE_SIZE = 3000.0
# Mean issued figure size over the labelled scenes (10, 10, 6).
EXPECTED_FIGURE = 8.7
SCALE_LOW, SCALE_HIGH = 4.2, 6.6
FIGURE_FRACTION = (0.55, 0.77)
RETENTION = 0.94
QUERIES = (20, 30)
GRAPH_TOP_K = 3


@dataclass(frozen=True)
class Config:
    """A geometry configuration to benchmark."""

    name: str
    transform_model: str = "affine"
    proposal_mode: str = "hybrid"
    proposals: int = 3000
    trials: int = 1
    clutter_trials: int = 0
    scale_min: float = 1.3
    scale_max: float = 18.0
    consensus_selection: str = "plurality"
    consensus_vote_weight: float = 0.75


CONFIGS = {
    # What the scored 0.74973 submission used.
    "current": Config("current", "affine", "pair", 12000, 5, 0, 1.3, 18.0),
    # Cheaper search, same model, plus the empirical clutter null.
    "triangle_clutter": Config("triangle_clutter", "affine", "hybrid", 3000, 5, 256, 1.3, 18.0),
    # Transform model matched to the measured similarity transform.
    "similarity": Config("similarity", "similarity", "hybrid", 3000, 5, 0, 1.3, 18.0),
    # Similarity plus the measured diagram-to-sky scale band.
    "similarity_tightscale": Config(
        "similarity_tightscale", "similarity", "hybrid", 3000, 5, 0, 3.5, 8.0
    ),
    # Everything measured, together.
    "similarity_tight_clutter": Config(
        "similarity_tight_clutter", "similarity", "hybrid", 3000, 5, 256, 3.5, 8.0
    ),
    "affine_tight_clutter": Config(
        "affine_tight_clutter", "affine", "hybrid", 3000, 5, 256, 3.5, 8.0
    ),
    # Isolate the scale band alone, against the current model.
    "current_tightscale": Config("current_tightscale", "affine", "pair", 12000, 5, 0, 3.5, 8.0),
    # Scale-band width sensitivity, all on the similarity model. The band is a
    # prior derived from three labelled scenes, so a band that is too tight
    # fails catastrophically on any hidden scene outside it. These quantify how
    # much accuracy is traded for that safety margin.
    "sim_band_4_7": Config("sim_band_4_7", "similarity", "hybrid", 3000, 5, 0, 4.0, 7.0),
    "sim_band_3_9": Config("sim_band_3_9", "similarity", "hybrid", 3000, 5, 0, 3.0, 9.0),
    "sim_band_25_11": Config("sim_band_25_11", "similarity", "hybrid", 3000, 5, 0, 2.5, 11.0),
    "sim_band_2_14": Config("sim_band_2_14", "similarity", "hybrid", 3000, 5, 0, 2.0, 14.0),
}


def clutter_pool(root: Path) -> np.ndarray:
    """Pool real candidate positions so synthetic clutter keeps real clustering."""
    points: list[list[float]] = []
    for path in sorted((root / "outputs" / "cache_validation_adaptive").glob("*.json")):
        saved = json.loads(path.read_text())
        for group in saved["candidates"]:
            points.extend([float(c[0]), float(c[1])] for c in group[:GRAPH_TOP_K])
    if not points:
        raise SystemExit("no cached validation candidates found for the clutter pool")
    return np.asarray(points, dtype=np.float64)


def similarity_matrix(scale: float, angle: float, reflect: bool) -> np.ndarray:
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float64
    )
    flip = np.asarray([[-1.0, 0.0], [0.0, 1.0]]) if reflect else np.eye(2)
    return scale * rotation @ flip


@dataclass
class Trial:
    truth: str
    candidates: list[J.Candidate]
    context: J.FitContext
    n_issued: int


def make_trial(
    pattern: Pattern, pool: np.ndarray, rng: np.random.Generator, reflect_probability: float
) -> Trial | None:
    nodes = np.asarray(pattern.points, dtype=np.float64)
    if len(nodes) < 2:
        return None
    nodes = nodes - nodes.mean(axis=0)

    # Place the figure with a similarity transform, retrying until it fits the frame.
    for _ in range(200):
        matrix = similarity_matrix(
            rng.uniform(SCALE_LOW, SCALE_HIGH),
            rng.uniform(0.0, 2.0 * np.pi),
            rng.random() < reflect_probability,
        )
        mapped = nodes @ matrix.T
        span = mapped.max(axis=0) - mapped.min(axis=0)
        if span.max() >= IMAGE_SIZE - 80.0:
            continue
        low = -mapped.min(axis=0) + 40.0
        high = IMAGE_SIZE - 40.0 - mapped.max(axis=0)
        if np.any(high < low):
            continue
        mapped = mapped + rng.uniform(low, high)
        break
    else:
        return None

    # Only a subset of the figure's stars are issued as query patches.
    fraction = rng.uniform(*FIGURE_FRACTION)
    n_issued = max(2, min(len(mapped), int(round(fraction * len(mapped)))))
    issued = mapped[rng.choice(len(mapped), size=n_issued, replace=False)]

    n_queries = max(int(rng.integers(QUERIES[0], QUERIES[1] + 1)), n_issued)
    candidates: list[J.Candidate] = []
    query_index = 0

    for star in issued:
        group: list[np.ndarray] = []
        if rng.random() < RETENTION:
            jitter = rng.normal(0.0, 25.0 if rng.random() < 0.10 else 3.0, size=2)
            group.append(star + jitter)
        while len(group) < GRAPH_TOP_K:
            group.append(pool[rng.integers(len(pool))])
        rng.shuffle(group)
        for rank, point in enumerate(group):
            candidates.append(
                J.Candidate(query_index, float(point[0]), float(point[1]), 1.0 - 0.01 * rank, 0.0)
            )
        query_index += 1

    for _ in range(n_queries - n_issued):
        for rank in range(GRAPH_TOP_K):
            point = pool[rng.integers(len(pool))]
            candidates.append(
                J.Candidate(query_index, float(point[0]), float(point[1]), 0.9 - 0.01 * rank, 0.0)
            )
        query_index += 1

    context = J.FitContext(
        cloud_size=len(candidates),
        image_area=IMAGE_SIZE * IMAGE_SIZE,
        expected_figure=EXPECTED_FIGURE,
        image_width=IMAGE_SIZE,
        image_height=IMAGE_SIZE,
    )
    return Trial(pattern.name, candidates, context, n_issued)


_WORKER_PATTERNS: list[Pattern] = []


def _worker_init(root: str) -> None:
    global _WORKER_PATTERNS
    _WORKER_PATTERNS = load_patterns(Path(root))


def _worker_trial(task: tuple[Config, Trial, int]) -> tuple[str, str, int]:
    """Identify one synthetic scene. Returns (truth, predicted, issued figure stars)."""
    config, trial, seed = task
    J.RANSAC_SCALE_RATIO_MIN, J.RANSAC_SCALE_RATIO_MAX = config.scale_min, config.scale_max
    best, _runner = J.choose_fit_consensus(
        _WORKER_PATTERNS,
        trial.candidates,
        config.proposals,
        seed,
        trial.context,
        config.trials,
        config.transform_model,
        config.proposal_mode,
        config.clutter_trials,
        2,
        config.consensus_selection,
        config.consensus_vote_weight,
    )
    return trial.truth, (best.pattern.name if best else "none"), trial.n_issued


def run_config(config: Config, trials: list[Trial], root: Path, seed: int, workers: int) -> dict:
    """Evaluate one configuration, optionally across a process pool."""
    start = time.time()
    tasks = [(config, trial, seed) for trial in trials]
    if workers <= 1:
        _worker_init(str(root))
        outcomes = [_worker_trial(task) for task in tasks]
    else:
        import multiprocessing as mp

        with mp.Pool(workers, initializer=_worker_init, initargs=(str(root),)) as pool:
            outcomes = pool.map(_worker_trial, tasks, chunksize=1)

    correct = 0
    by_size: dict[str, list[int]] = {"small": [], "large": []}
    wrong: list[tuple[str, str]] = []
    for truth, predicted, issued in outcomes:
        hit = int(truth == predicted)
        correct += hit
        by_size["small" if issued < 8 else "large"].append(hit)
        if not hit:
            wrong.append((truth, predicted))
    return {
        "name": config.name,
        "accuracy": correct / len(trials),
        "correct": correct,
        "total": len(trials),
        "small": (sum(by_size["small"]), len(by_size["small"])),
        "large": (sum(by_size["large"]), len(by_size["large"])),
        "seconds": time.time() - start,
        "wrong": wrong,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--seed", type=int, default=6643)
    parser.add_argument("--reflect-probability", type=float, default=0.0)
    parser.add_argument(
        "--generator-scale",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help=(
            "Override the synthetic diagram-to-sky scale range. Widen it beyond "
            "a configuration's search band to measure what a hidden scene "
            "outside that band costs."
        ),
    )
    parser.add_argument("--config", action="append", default=None, choices=sorted(CONFIGS))
    parser.add_argument("--min-nodes", type=int, default=5,
                        help="skip diagrams too small to carry identity evidence")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--consensus", type=int, default=None,
                        help="override each config's consensus-seed count")
    parser.add_argument(
        "--consensus-selection",
        choices=("plurality", "quality", "vote-quality"),
        default=None,
    )
    parser.add_argument("--consensus-vote-weight", type=float, default=0.75)
    args = parser.parse_args()
    root = args.root.resolve()

    if args.generator_scale is not None:
        global SCALE_LOW, SCALE_HIGH
        SCALE_LOW, SCALE_HIGH = args.generator_scale
        print(f"generator scale range overridden to {SCALE_LOW}-{SCALE_HIGH}")

    patterns = load_patterns(root)
    usable = [p for p in patterns if len(p.points) >= args.min_nodes]
    pool = clutter_pool(root)
    print(f"{len(patterns)} diagrams, {len(usable)} with >= {args.min_nodes} nodes; "
          f"clutter pool {len(pool)} real candidate positions")

    rng = np.random.default_rng(args.seed)
    trials: list[Trial] = []
    while len(trials) < args.trials:
        pattern = usable[rng.integers(len(usable))]
        trial = make_trial(pattern, pool, rng, args.reflect_probability)
        if trial is not None:
            trials.append(trial)
    sizes = np.array([t.n_issued for t in trials])
    print(f"{len(trials)} synthetic scenes | issued figure stars "
          f"min {sizes.min()} median {int(np.median(sizes))} max {sizes.max()} | "
          f"cloud ~{trials[0].context.cloud_size} pts | workers {args.workers}\n")

    names = args.config or [
        "current", "similarity", "similarity_tightscale", "similarity_tight_clutter"
    ]
    print(f"{'config':28}{'identity':>14}{'small fig':>13}{'large fig':>13}{'secs':>7}")
    print("-" * 75)
    results = []
    for name in names:
        config = CONFIGS[name]
        if args.consensus is not None:
            config = replace(config, trials=args.consensus)
        if args.consensus_selection is not None:
            config = replace(
                config,
                consensus_selection=args.consensus_selection,
                consensus_vote_weight=args.consensus_vote_weight,
            )
        result = run_config(config, trials, root, args.seed, args.workers)
        results.append(result)
        print(f"{result['name']:28}{result['correct']:>5}/{result['total']:<3}"
              f"{result['accuracy']:>6.1%}"
              f"{result['small'][0]:>8}/{result['small'][1]:<4}"
              f"{result['large'][0]:>8}/{result['large'][1]:<4}"
              f"{result['seconds']:>7.0f}")

    best = max(results, key=lambda r: r["accuracy"])
    print(f"\nbest: {best['name']} at {best['accuracy']:.1%} "
          f"({best['correct']}/{best['total']})")
    tally: dict[tuple[str, str], int] = {}
    for pair in best["wrong"]:
        tally[pair] = tally.get(pair, 0) + 1
    if tally:
        print("most common confusions for the best config:")
        for (truth, got), count in sorted(tally.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {truth} -> {got}  x{count}")


if __name__ == "__main__":
    main()
