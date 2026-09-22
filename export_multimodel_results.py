#!/usr/bin/env python3
"""Audit an existing completed run and bundle its outputs without retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np

import constellation_pipeline as cp
from download_training_images import SOURCES
from multimodel_pipeline import fit_candidate_model, load_scene, ranking_diagnostic, rerank, summary


def export(root: Path) -> Path:
    out = root / "outputs" / "multimodel"
    report = json.loads((out / "report.json").read_text())
    if "recommended_source" not in report:
        raise RuntimeError("Run is incomplete; wait for FINAL_CSV before exporting")
    truth = {row["Id"]: row for row in cp.read_csv_rows(root / "train_ground_truth.csv")}
    data = {}
    for scene in truth:
        saved, image, _, _, coords = load_scene(root, "train", scene)
        with np.load(out / f"features_train_{scene}.npz", allow_pickle=False) as cache:
            data[scene] = saved, coords, cache["features"].copy(), image.shape
    ranking = {scene: ranking_diagnostic(data[scene], truth[scene],
                                        fit_candidate_model(data, truth, exclude=scene))
               for scene in truth}
    # Isolate localisation from geometry: hold the classical presence decisions,
    # identities and member coordinates fixed while varying only non-member rank.
    # This is a diagnostic sweep, NOT another automatic model-selection gate.
    fixed = cp.read_csv_rows(out / "train_baseline.csv")
    sweep = {}
    for alpha in (0., .15, .35, .6, 1.):
        rows = []
        for source in fixed:
            scene = source["Id"]
            saved, _, features, _ = data[scene]
            ranker = fit_candidate_model(data, truth, exclude=scene)
            groups = rerank(saved, features, ranker, alpha=alpha)
            row = source.copy()
            for qi, column in enumerate(saved["columns"]):
                current = cp.parse_cell(row[column])
                if current is not None and current[2] == 0:
                    point = groups[qi][0]
                    row[column] = cp.format_cell(cp.Prediction(point.x, point.y, m=0))
            rows.append(row)
        sweep[str(alpha)] = summary(truth, rows)
    audit = {"scene_held_out_ranking": ranking,
             "fixed_geometry_localisation_sweep": sweep,
             "export_revision": subprocess.check_output(
                 ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
             "note": "Export revision may be newer than training; model signatures retain training code hashes."}
    for name in ("submission_recommended.csv", "submission_multimodel_experimental.csv"):
        cp.validate_submission(root, out / name)
    (out / "ranking_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    manifest = root / "external" / "training_images" / "manifest.json"
    sources = json.loads(manifest.read_text())
    credits = {identifier: credit for identifier, credit, _ in SOURCES}
    for source in sources:
        source["credit"] = credits[source["id"]]
    # Preserve the original manifest bytes because checkpoints fingerprint it.
    # Corrected credits are a separate attribution copy; no training data changed.
    bundle = root / "outputs" / "multimodel_results.zip"
    allowed = {".csv", ".json", ".pt", ".npz"}
    entries = sorted(p for p in out.iterdir() if p.suffix in allowed)
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in entries}
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in entries:
            archive.write(path, arcname=path.name)
        archive.write(manifest, arcname="external_sources_training_original.json")
        archive.writestr("external_sources.json", json.dumps(sources, indent=2) + "\n")
        archive.writestr("sha256.json", json.dumps(checksums, indent=2) + "\n")
    print("RANKING", json.dumps(ranking), flush=True)
    print("LOCALISATION_SWEEP", json.dumps({a: s["mean"] for a, s in sweep.items()}), flush=True)
    print("RECOMMENDED", report["recommended_source"], "PROMOTED", report["promoted"], flush=True)
    print("BUNDLE", bundle, bundle.stat().st_size, flush=True)
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    export(parser.parse_args().root.resolve())
