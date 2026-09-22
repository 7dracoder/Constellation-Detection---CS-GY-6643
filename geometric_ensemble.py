"""Share RANSAC proposals between similarity and affine ensemble branches.

The two branches use identical seeds and coarse hypotheses. Reusing those
arrays avoids repeating their sampling and nearest-neighbour pre-ranking;
refinement, eligibility, quality and consensus rules are unchanged.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import joint_geometric_solver as jg


def fit_both(task):
    pattern, candidates, proposals, seed, context = task
    best = {"similarity": None, "affine": None}
    for reflected, source in enumerate((pattern.points, pattern.points * np.array((1., -1.), np.float32))):
        matrices, offsets = jg.propose_transforms(source, candidates, proposals, seed + 10_007 * reflected)
        hypotheses = jg.top_hypotheses(source, candidates, matrices, offsets)
        for matrix, offset, _ in hypotheses:
            for mode in best:
                fit = jg.assign_queries(pattern, source, matrix, offset, candidates, context,
                                        transform_model=mode)
                if fit is not None and (best[mode] is None or fit.quality > best[mode].quality):
                    best[mode] = fit
    return best


def select(fits, context):
    found = [fit for fit in fits if fit is not None]
    eligible = [fit for fit in found if fit.support >= context.minimum_support]
    ranked = eligible if eligible else [fit for fit in found if fit.support >= 4]
    ranked.sort(key=lambda fit: fit.quality, reverse=True)
    return ranked[0] if ranked else None


def consensus(winners):
    winners = [fit for fit in winners if fit is not None]
    if not winners:
        return None
    tally = {}
    for fit in winners:
        tally[fit.pattern.name] = tally.get(fit.pattern.name, 0) + 1
    name = max(tally, key=lambda name: (tally[name], name))
    return max((fit for fit in winners if fit.pattern.name == name), key=lambda fit: fit.quality)


def choose_both(patterns, candidates, proposals, seed, context, trials):
    winners = {"similarity": [], "affine": []}
    with ThreadPoolExecutor(max_workers=2) as executor:
        for trial in range(max(1, trials)):
            tasks = [(pattern, candidates, proposals, seed + 7_919 * trial + 101 * index, context)
                     for index, pattern in enumerate(patterns)]
            found = list(executor.map(fit_both, tasks))
            for mode in winners:
                winners[mode].append(select([result[mode] for result in found], context))
    return {mode: consensus(values) for mode, values in winners.items()}
