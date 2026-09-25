#!/usr/bin/env python3
"""Draw a diagnostic overlay of constellation predictions and optional truth."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2

from constellation_pipeline import parse_cell, patch_columns, read_csv_rows


def draw_overlay(
    root: Path, split: str, predictions: Path, scene: str, output: Path,
    display_clahe: bool = False,
) -> None:
    rows = {row["Id"]: row for row in read_csv_rows(predictions)}
    if scene not in rows:
        raise ValueError(f"Scene not found in predictions: {scene}")
    row = rows[scene]
    image_paths = sorted((root / split / scene).glob("*_image.png"))
    if len(image_paths) != 1:
        raise ValueError(f"Expected one sky image for {scene}")
    image = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_paths[0]}")
    truth = None
    if split == "train":
        truth_rows = {item["Id"]: item for item in read_csv_rows(root / "train_ground_truth.csv")}
        truth = truth_rows[scene]
    scale = min(1.0, 1500.0 / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if display_clahe:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12)).apply(gray)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        image = cv2.convertScaleAbs(image, alpha=1.35, beta=8)

    def point(cell: tuple[int, int, int]) -> tuple[int, int]:
        return round(cell[0] * scale), round(cell[1] * scale)

    for column in patch_columns(int(row["n_patches"])):
        predicted = parse_cell(row[column])
        actual = parse_cell(truth[column]) if truth is not None else None
        if actual is not None:
            cv2.circle(image, point(actual), 5, (70, 220, 80), 1, cv2.LINE_AA)
        if predicted is None:
            continue
        centre = point(predicted)
        color = (50, 230, 230) if predicted[2] else (230, 210, 60)
        cv2.circle(image, centre, 3 if not predicted[2] else 5, color, -1, cv2.LINE_AA)
        if predicted[2]:
            cv2.putText(image, column[-2:], (centre[0] + 6, centre[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
        if actual is not None and math.dist(predicted[:2], actual[:2]) > 12:
            cv2.line(image, centre, point(actual), (40, 60, 230), 1, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (image.shape[1], 31), (0, 0, 0), -1)
    legend = "member=yellow, other=cyan"
    if truth is not None:
        legend += ", truth=green, error=red"
    title = f"{scene}: {row['constellation']} | {legend}"
    cv2.putText(image, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise ValueError(f"Cannot write overlay: {output}")
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--display-clahe", action="store_true", help="Local contrast for visualization only")
    args = parser.parse_args()
    draw_overlay(args.root.resolve(), args.split, args.predictions.resolve(), args.scene,
                 args.output.resolve(), args.display_clahe)


if __name__ == "__main__":
    main()
