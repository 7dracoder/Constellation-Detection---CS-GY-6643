#!/usr/bin/env python3
"""Measure whether scene peak detectors cover labelled figure-star centres."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from constellation_pipeline import (
    image_files,
    parse_cell,
    patch_columns,
    read_csv_rows,
    read_grayscale,
)
from star_bank_matcher import star_response


def peak_points(response: np.ndarray) -> np.ndarray:
    local = response == cv2.dilate(response, np.ones((7, 7), np.uint8))
    ys, xs = np.nonzero(local)
    strengths = response[ys, xs]
    order = np.argsort(strengths)[::-1]
    return np.column_stack((xs[order], ys[order])).astype(np.float32)


def response_views(image: np.ndarray) -> dict[str, np.ndarray]:
    local_background = cv2.GaussianBlur(image, (0, 0), 10.0)
    high_pass = image - local_background
    local_scale = cv2.GaussianBlur(np.abs(high_pass), (0, 0), 10.0)
    normalized = np.clip(high_pass / (local_scale + 4.0), -8.0, 8.0)
    return {
        "multiscale_abs_dog": star_response(image),
        "normalized_dog": cv2.GaussianBlur(normalized, (0, 0), 0.8)
        - cv2.GaussianBlur(normalized, (0, 0), 2.2),
        "raw_dog": cv2.GaussianBlur(image, (0, 0), 0.8)
        - cv2.GaussianBlur(image, (0, 0), 2.2),
        "local_contrast": image - cv2.GaussianBlur(image, (0, 0), 3.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--counts", type=int, nargs="+", default=(300, 1000, 3000))
    parser.add_argument("--tolerances", type=float, nargs="+", default=(12.0, 20.0, 30.0))
    args = parser.parse_args()
    root = args.root.resolve()
    rows = read_csv_rows(root / "train_ground_truth.csv")
    totals: dict[tuple[str, str, int, float], list[int]] = {}
    for row in rows:
        scene = row["Id"]
        path = image_files(root / "train" / scene, "*_image.png")[0]
        image = read_grayscale(path)
        truth = np.asarray(
            [
                cell[:2]
                for column in patch_columns(int(row["n_patches"]))
                if (cell := parse_cell(row[column])) is not None and cell[2] == 1
            ],
            dtype=np.float32,
        )
        for method, response in response_views(image).items():
            ranked = peak_points(response)
            for count in args.counts:
                points = ranked[: min(count, len(ranked))]
                for convention, coordinates in (("xy", truth), ("yx", truth[:, ::-1])):
                    distances = cKDTree(points).query(coordinates, k=1)[0]
                    for tolerance in args.tolerances:
                        key = (method, convention, count, tolerance)
                        totals.setdefault(key, [0, 0])
                        totals[key][0] += int(np.sum(distances <= tolerance))
                        totals[key][1] += len(truth)
    print("method,convention,count,tolerance,hits,total,recall")
    for (method, convention, count, tolerance), (hits, total) in sorted(totals.items()):
        print(
            f"{method},{convention},{count},{tolerance:g},{hits},{total},{hits / total:.3f}"
        )


if __name__ == "__main__":
    main()
