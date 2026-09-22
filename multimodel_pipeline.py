#!/usr/bin/env python3
"""Externally pretrained, scene-cross-fitted ensemble with auditable promotion.

Models: cached CUDA correlation matcher; intensity CNN; high-pass CNN;
regularized candidate classifier; presence classifier; similarity and affine
geometry. All train-scene reports exclude that scene from both learned
classifiers. External CNN pretraining does not read competition images.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import constellation_pipeline as cp
import joint_geometric_solver as jg
import presence_refiner as pr
from evaluate import score_scene
from structural_refiner import train_membership_classifier
from download_training_images import download


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_scene(root, split, scene):
    path = root / "outputs" / f"cache_{split}" / f"{scene}.json"
    saved = json.loads(path.read_text())
    if len(saved["columns"]) != len(saved["candidates"]):
        raise ValueError(f"Cache column mismatch: {path}")
    if any(len(group) < 16 for group in saved["candidates"]):
        raise ValueError(f"Need 16 candidates per query: {path}")
    image = cp.read_grayscale(root / split / scene / f"{scene}_image.png")
    queries = np.stack([cp.read_grayscale(root / split / scene / "patches" / f"{c}.png")
                        for c in saved["columns"]])
    coords = np.asarray([[v[:2] for v in group[:16]] for group in saved["candidates"]], np.int32)
    crops = cp.padded_extract(image, coords.reshape(-1, 2))
    return saved, image, queries, crops, coords


def candidate_features(root, split, scene, models, output):
    from neural_patch_models import embed, radial_descriptor
    saved, image, queries, crops, coords = load_scene(root, split, scene)
    cache_path = root / "outputs" / f"cache_{split}" / f"{scene}.json"
    signature = digest(cache_path) + digest(Path(__file__))
    signature += digest(root / split / scene / f"{scene}_image.png")
    for column in saved["columns"]:
        signature += digest(root / split / scene / "patches" / f"{column}.png")
    for mode in ("intensity", "highpass"):
        signature += digest(output / f"encoder_{mode}.pt")
    feature_path = output / f"features_{split}_{scene}.npz"
    if feature_path.exists():
        with np.load(feature_path, allow_pickle=False) as data:
            if str(data["signature"]) == signature:
                return saved, coords, data["features"].copy(), image.shape
    raw = np.asarray([[v[2] for v in group[:16]] for group in saved["candidates"]], np.float32)
    ranks = np.broadcast_to(np.linspace(1, 0, 16), raw.shape)
    z = (raw-raw.mean(1, keepdims=True))/(raw.std(1, keepdims=True)+1e-5)
    features = [raw, z, ranks, raw-raw[:, :1]]
    for model in models:
        q = embed(model, queries)
        c = embed(model, crops).reshape(len(queries), 16, -1)
        sim = np.einsum("qd,qkd->qk", q, c)
        features.extend([sim, (sim-sim.mean(1, keepdims=True))/(sim.std(1, keepdims=True)+1e-5)])
    qr = radial_descriptor(queries)
    cr = radial_descriptor(crops).reshape(len(queries), 16, -1)
    features.append(np.einsum("qd,qkd->qk", qr, cr)/qr.shape[1])
    qstats = np.stack((queries.mean((1, 2)), queries.std((1, 2)), queries.max((1, 2))), 1)
    cstats = np.stack((crops.mean((1, 2)), crops.std((1, 2)), crops.max((1, 2))), 1)
    differences = np.abs(qstats[:, None, :]-cstats.reshape(len(queries), 16, 3))/255.
    features.extend([differences[:, :, i] for i in range(3)])
    result = np.stack(features, -1).astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite features: {scene}")
    np.savez_compressed(feature_path, signature=signature, features=result)
    return saved, coords, result, image.shape


class CandidateModel:
    def __init__(self, x, y):
        self.mean, self.scale = x.mean(0), x.std(0)+1e-5
        design = np.column_stack((np.ones(len(x)), (x-self.mean)/self.scale))
        # Each class receives half the loss; otherwise thousands of easy
        # negatives overwhelm the few true query-location correspondences.
        weights = np.where(y > .5, .5/max(y.sum(), 1), .5/max((1-y).sum(), 1))
        def objective(w):
            logits = design @ w
            loss = np.sum(weights*(np.logaddexp(0, logits)-y*logits)) + .05*np.sum(w[1:]**2)
            grad = design.T @ (weights*(expit(logits)-y))
            grad[1:] += .1*w[1:]
            return loss, grad
        solution = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B")
        if not solution.success:
            raise RuntimeError(f"Candidate calibration failed: {solution.message}")
        self.weights = solution.x

    def predict(self, x):
        return expit(self.weights[0] + ((x-self.mean)/self.scale)@self.weights[1:])


def fit_candidate_model(data, truth, exclude=None):
    x, y = [], []
    for scene, (_, coords, features, _) in data.items():
        if scene == exclude:
            continue
        for qi, col in enumerate(data[scene][0]["columns"]):
            target = cp.parse_cell(truth[scene][col])
            distances = np.full(16, 1e6) if target is None else np.linalg.norm(coords[qi]-target[:2], axis=1)
            keep = (distances <= 12) | (distances >= 36)
            x.extend(features[qi][keep])
            y.extend((distances[keep] <= 12).astype(float))
    return CandidateModel(np.asarray(x), np.asarray(y))


def rerank(saved, features, model, alpha=.35):
    probabilities = model.predict(features)
    groups = []
    for qi, group in enumerate(saved["candidates"]):
        raw = np.array([p[2] for p in group[:16]])
        # Combine comparable within-query standardized scores; retain original
        # correlation scores in Candidate so geometric costs keep their meaning.
        raw_z = (raw-raw.mean())/(raw.std()+1e-5)
        p = probabilities[qi]
        model_z = (p-p.mean())/(p.std()+1e-5)
        order = np.argsort((1-alpha)*raw_z + alpha*model_z)[::-1]
        groups.append([jg.Candidate(qi, int(group[j][0]), int(group[j][1]), float(group[j][2]),
                                    float(group[j][3]) if len(group[j]) > 3 else 0.) for j in order])
    return groups


def predict_scene(root, split, template, scene_data, rank_model, membership, presence,
                  patterns, proposals, trials, learned):
    scene = template["Id"]
    saved, coords, features, shape = scene_data
    columns = cp.patch_columns(int(template["n_patches"]))
    if saved["columns"] != columns:
        raise ValueError(f"Columns mismatch for {scene}")
    byq = rerank(saved, features, rank_model, alpha=.35 if learned else 0.)
    member_probs = jg.membership_probabilities(root, split, scene, columns, membership)
    px, _ = pr.scene_features(root, split, scene, columns, root/"outputs"/f"cache_{split}")
    present_probs = presence.probabilities(px)
    expected = max(4., .355*round(.625*len(columns)))
    count = int(np.clip(round(2*expected), 12, 30))
    selected = sorted(int(i) for i in np.argsort(member_probs)[::-1][:count])
    sparse = [p for q in selected for p in byq[q][:3]]
    dense = [p for q in selected for p in byq[q]]
    context = jg.FitContext(len(sparse), float(shape[0]*shape[1]), expected)
    # Seed derives only from input identity, with no label-specific parameters.
    seed = 51179 + int(hashlib.sha256(scene.encode()).hexdigest()[:6], 16)
    best_sim, _ = jg.choose_fit_consensus(patterns, sparse, proposals, seed, context, trials, "similarity")
    best_aff, _ = jg.choose_fit_consensus(patterns, sparse, proposals, seed, context, trials, "affine")
    chosen = best_sim
    if best_aff is not None and (best_sim is None or best_aff.pattern.name == best_sim.pattern.name):
        chosen = best_aff
    assigned = {} if chosen is None else jg.assign_to_mapped(chosen.mapped_points, dense, chosen.tolerance)
    row = template.copy()
    row["constellation"] = "unknown" if chosen is None else chosen.pattern.name
    for qi, column in enumerate(columns):
        point = assigned.get(qi)
        if point is not None:
            row[column] = cp.format_cell(cp.Prediction(point.x, point.y, m=1))
        elif present_probs[qi] >= presence.threshold:
            point = byq[qi][0]
            row[column] = cp.format_cell(cp.Prediction(point.x, point.y, m=0))
        else:
            row[column] = "-1"
    diagnostics = dict(scene=scene, learned=learned, label=row["constellation"],
                       similarity=None if best_sim is None else best_sim.pattern.name,
                       affine=None if best_aff is None else best_aff.pattern.name,
                       support=0 if chosen is None else chosen.support, assigned=len(assigned))
    print("GEOMETRY", json.dumps(diagnostics), flush=True)
    return row, diagnostics


def summary(truth, rows):
    per_scene = {row["Id"]: score_scene(truth[row["Id"]], row) for row in rows}
    mean = {k: float(np.mean([value[k] for value in per_scene.values()])) for k in next(iter(per_scene.values()))}
    return dict(mean=mean, per_scene=per_scene)


def promotion(baseline, challenger):
    """Require improvements in both metric interpretations and no large fold loss."""
    b, c = baseline["mean"], challenger["mean"]
    return bool(c["total_loose"] >= b["total_loose"]+.005
                and c["total_strict"] >= b["total_strict"]+.005
                and c["identity"] >= b["identity"]
                and all(challenger["per_scene"][s]["total_loose"] >= v["total_loose"]-.03
                        for s, v in baseline["per_scene"].items()))


def main():
    from neural_patch_models import train_models
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--proposals", type=int, default=12000)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.proposals < 100 or args.trials < 1:
        parser.error("Positive training steps, >=100 proposals and >=1 trial required")
    root = args.root.resolve()
    out = root/"outputs"/"multimodel"
    out.mkdir(parents=True, exist_ok=True)
    images = root/"external"/"training_images"
    if args.download:
        download(images)
    models = train_models(images, out, steps=args.steps)
    truth = {row["Id"]: row for row in cp.read_csv_rows(root/"train_ground_truth.csv")}
    train = {scene: candidate_features(root, "train", scene, models, out) for scene in truth}
    patterns = jg.load_patterns(root)
    baseline, challenger, audits = [], [], []
    for scene in truth:
        ranker = fit_candidate_model(train, truth, exclude=scene)
        membership = train_membership_classifier(root, excluded_scene=scene)
        presence = pr.train_classifier(root, root/"outputs"/"cache_train", excluded_scene=scene)
        template = truth[scene].copy()
        for key in template:
            if key.startswith("patch_"):
                template[key] = "-1"
        for learned, target in ((False, baseline), (True, challenger)):
            row, audit = predict_scene(root, "train", template, train[scene], ranker, membership,
                                       presence, patterns, args.proposals, args.trials, learned)
            target.append(row)
            audits.append(audit)
    b, c = summary(truth, baseline), summary(truth, challenger)
    accepted = promotion(b, c)
    report = dict(baseline=b, challenger=c, promoted=accepted, diagnostics=audits,
                  parameters=vars(args) | {"root": str(root)}, python=platform.python_version(),
                  caveat="Three held-out scenes; not a Kaggle estimate. No validation labels used.")
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    write_csv(out/"train_baseline.csv", baseline)
    write_csv(out/"train_challenger.csv", challenger)
    print("HELD_OUT_REPORT", json.dumps(dict(baseline=b["mean"], challenger=c["mean"], promoted=accepted)), flush=True)
    if args.train_only:
        return
    ranker = fit_candidate_model(train, truth)
    np.savez_compressed(out/"candidate_classifier.npz", mean=ranker.mean, scale=ranker.scale, weights=ranker.weights)
    membership = train_membership_classifier(root)
    presence = pr.train_classifier(root, root/"outputs"/"cache_train")
    rows = []
    for template in cp.read_csv_rows(root/"sample_submission.csv"):
        scene = template["Id"]
        data = candidate_features(root, "validation", scene, models, out)
        row, audit = predict_scene(root, "validation", template, data, ranker, membership, presence,
                                   patterns, args.proposals, args.trials, True)
        rows.append(row)
        audits.append(audit)
    experiment = out/"submission_multimodel_experimental.csv"
    write_csv(experiment, rows)
    cp.validate_submission(root, experiment)
    final = out/"submission_recommended.csv"
    if accepted:
        write_csv(final, rows)
    else:
        # Return the scored baseline when a new model fails its promotion gate.
        write_csv(final, cp.read_csv_rows(root/"outputs"/"submission_v4_presence.csv"))
    cp.validate_submission(root, final)
    report["recommended_source"] = experiment.name if accepted else "submission_v4_presence.csv"
    report["csv_sha256"] = digest(final)
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print("FINAL_CSV", final, "promoted", accepted, "sha256", digest(final), flush=True)


if __name__ == "__main__":
    main()
