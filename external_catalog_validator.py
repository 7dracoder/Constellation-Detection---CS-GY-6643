#!/usr/bin/env python3
"""Read-only identity cross-check using d3-celestial constellation lines.

This deliberately does not modify a submission.  It loads the public catalog,
fits its independent star-line geometry to the rows' predicted ``m=1`` points,
and prints the best and runner-up identities.  The catalog drawing conventions
differ from the competition diagrams, so disagreement is diagnostic evidence,
not permission to overwrite a prediction blindly.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import cv2
import numpy as np

from constellation_pipeline import image_files, parse_cell, patch_columns, read_csv_rows
from joint_geometric_solver import Candidate, FitContext, choose_fit
from structural_refiner import Pattern, load_patterns


def slug(value: str) -> str:
    plain = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")


def load_d3_patterns(root: Path, catalog_root: Path) -> list[Pattern]:
    data = catalog_root / "data"
    metadata = {
        feature["id"]: feature["properties"]["name"]
        for feature in json.loads((data / "constellations.json").read_text())["features"]
    }
    available = {pattern.name for pattern in load_patterns(root)}
    aliases = {"corona-austrina": "corona-australis", "serpens": "serpens-caput"}
    grouped: dict[str, list[list[float]]] = {}
    features = json.loads((data / "constellations.lines.json").read_text())["features"]
    for feature in features:
        name = aliases.get(slug(metadata[feature["id"]]), slug(metadata[feature["id"]]))
        if name not in available:
            continue
        points = grouped.setdefault(name, [])
        for line in feature["geometry"]["coordinates"]:
            for coordinate in line:
                if coordinate not in points:
                    points.append(coordinate)

    patterns: list[Pattern] = []
    for name, values in sorted(grouped.items()):
        points = np.asarray(values, dtype=np.float32)
        # Two- and three-node figures cannot provide independent affine
        # evidence, so omit them from this validator.
        if len(points) < 4:
            continue
        if float(np.ptp(points[:, 0])) > 180.0:
            points[points[:, 0] < 0, 0] += 360.0
        points[:, 0] *= np.cos(np.deg2rad(float(points[:, 1].mean())))
        points -= points.mean(axis=0)
        rms = float(np.sqrt(np.mean(np.sum(points * points, axis=1))))
        points *= 140.0 / max(rms, 1e-6)
        patterns.append(Pattern(name, points))
    return patterns


def fit_rows(
    root: Path,
    split: str,
    rows: list[dict[str, str]],
    patterns: list[Pattern],
    proposals: int,
) -> None:
    recovered = 0
    scored = 0
    for scene_index, row in enumerate(rows):
        scene = row["Id"]
        candidates: list[Candidate] = []
        for query_index, column in enumerate(patch_columns(int(row["n_patches"]))):
            value = parse_cell(row[column])
            if value is not None and value[2] == 1:
                candidates.append(Candidate(query_index, value[0], value[1], 1.0, 0.0))
        if len(candidates) < 4:
            print(f"{scene}: insufficient m=1 points ({len(candidates)})")
            continue
        paths = image_files(root / split / scene, "*_image.png")
        image = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
        context = FitContext(len(candidates), float(image.size), float(len(candidates)))
        best, runner = choose_fit(
            patterns,
            candidates,
            proposals,
            91_003 + scene_index,
            context,
            transform_model="affine",
        )
        best_text = "none" if best is None else f"{best.pattern.name} support={best.support} q={best.quality:.2f}"
        runner_text = (
            "none" if runner is None else f"{runner.pattern.name} support={runner.support} q={runner.quality:.2f}"
        )
        submitted = row.get("constellation", row.get("Id", ""))
        agrees = best is not None and best.pattern.name == submitted
        print(f"{scene}: submitted={submitted} catalog={best_text} runner={runner_text} agree={agrees}")
        scored += 1
        recovered += int(agrees)
    print(f"catalog agreement: {recovered}/{scored}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--catalog-root", type=Path, default=Path("external/d3-celestial"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--proposals", type=int, default=12_000)
    args = parser.parse_args()
    root = args.root.resolve()
    patterns = load_d3_patterns(root, args.catalog_root.resolve())
    if not patterns:
        raise ValueError("No d3-celestial patterns matched the competition names")
    print(f"loaded {len(patterns)} independent catalog patterns")
    fit_rows(root, args.split, read_csv_rows(args.input.resolve()), patterns, args.proposals)


if __name__ == "__main__":
    main()
