# Project status — Constellation Detection, CS-GY 6643

Last updated: September 18, 2026

## Scope and compliance

This project is intentionally restricted to the course-provided materials:

- supplied training and validation sky images;
- supplied 32×32 query patches;
- supplied constellation-pattern diagrams; and
- supplied training labels.

It does **not** use web images, external astronomical data, external labels,
pretrained weights, or external models. NYU's course guidance also requires
coursework to use Cloud Bursting/Open OnDemand instead of the researcher-facing
Torch cluster.

## Score record

| Submission stage | Kaggle score | Notes |
| --- | ---: | --- |
| Initial validated baseline | approximately 0.26710 | Correct schema, but weak localisation and all-`unknown` identity. |
| Earlier geometric submission | approximately 0.58 | Added candidate shortlists and geometric identity fitting. |
| A100 wide-search submission | **0.66890** | Current best: `outputs/submission_a100_wide_fixed.csv`. |

Kaggle provides a single hidden-label score, not component scores. A valid CSV
only proves that Kaggle can read it; it does not predict a leaderboard score.
No implementation or experiment in this repository establishes a 0.90 or 0.96
score guarantee.

## Current best submission

File: `outputs/submission_a100_wide_fixed.csv`

The file has been checked with `constellation_pipeline.py validate` and has:

- 16 data rows plus the header;
- 90 columns (`Id`, `n_patches`, 87 patch columns, `constellation`);
- no null or blank-like cells;
- `-1` padding for non-applicable patch columns; and
- cells formatted as `-1` or `(x, y, m)`.

The A100 output selected the following labels. These are predictions, not
ground truth.

| Scene | Predicted present stars | `m=1` stars | Predicted label |
| --- | ---: | ---: | --- |
| constellation_01 | 18 | 9 | eridanus |
| constellation_02 | 21 | 11 | eridanus |
| constellation_03 | 30 | 6 | carina |
| constellation_04 | 21 | 9 | canis-major |
| constellation_05 | 47 | 7 | pisces |
| constellation_06 | 37 | 4 | orion |
| constellation_07 | 13 | 7 | perseus |
| constellation_08 | 20 | 7 | orion |
| constellation_10 | 54 | 13 | centaurus |
| constellation_11 | 49 | 12 | canis-major |
| constellation_12 | 21 | 11 | lupus |
| constellation_13 | 30 | 7 | aquila |
| constellation_14 | 36 | 6 | pisces |
| constellation_15 | 44 | 6 | hydra |
| constellation_16 | 18 | 9 | eridanus |
| constellation_17 | 27 | 7 | ursa-minor |

## Method used by the current best run

### 1. Wide GPU patch localisation

`gpu_matcher.py` searches each query patch over the supplied sky image with a
coarse-to-fine CUDA matcher. It applies background normalization, a 36-angle
and 10-scale transform bank, coarse peak retrieval, and local refinement.

The configuration in `matcher_config_gpu_wide.json` uses scales from 0.58 to
1.46, a 10-pixel refinement radius, and weighted intensity, gradient, and SSIM
similarity. Rather than committing to the first visually plausible location,
it preserves the 16 best spatially distinct candidates for each patch.

On all 71 labelled-present training patches, the candidate-retrieval diagnostic
was:

| Diagnostic | Earlier CPU matcher | A100 wide matcher |
| --- | ---: | ---: |
| Exact top-1 location | 36 / 71 | 49 / 71 |
| Exact location within top 16 | 56 / 71 | 63 / 71 |

These figures measure training candidate retrieval only; they are not a
leaderboard score.

### 2. Constellation geometry and membership

`joint_geometric_solver.py` reads the supplied pattern diagrams, extracts their
star nodes, and tests each pattern against the query-candidate point cloud.

It evaluates scale, translation, rotation, and reflection using RANSAC-style
similarity-transform proposals. It then uses a one-to-one assignment between
patch queries and pattern nodes, preventing multiple patches from claiming the
same star. Graph-supported assignments become `m=1`; direct but non-graph
matches may remain `m=0`.

`structural_refiner.py` supplies a small regularized membership proposal model
based only on simple patch brightness and contrast features from the labelled
training patches. It is not a pretrained model.

### 3. A100 execution and recovery

The first CUDA solver run stalled because its geometry phase forked worker
processes after CUDA was initialized. Forking a live CUDA process can deadlock.
The solver was changed to use CUDA-safe threaded geometry workers. Candidate
results were cached per scene, and the corrected A100 run completed all 16
scenes before the final CSV was validated and downloaded.

## NYU compute history

The project was initially explored through NYU HPC access. The course
announcement stated that coursework must not run on the Torch research cluster,
so the supported NYU option is Cloud Bursting/Open OnDemand with NYU VPN. HPC
bootstrap and Slurm scripts remain under `hpc/`, but the current best score came
from the Colab A100 run.

## Known limitations and work still required

1. GPU presence calibration failed during the attempted calibration command, so
   the current best run used the configured default presence threshold of 0.46.
   This can cause false detections or missed stars.
2. The old local approximate-score helper does not correctly evaluate predicted
   constellation identity against the training labels. Its historical in-sample
   figure must not be used as a leaderboard forecast.
3. The geometry scorer's coverage expression currently degenerates to 1.0,
   preventing it from distinguishing between patterns that explain different
   fractions of their nodes.
4. Eight of the 71 labelled-present training patches did not include their true
   location in the A100 top-16 candidates. Geometry cannot correct a point that
   was never proposed.
5. Only three training scenes are labelled, so tuning directly on all of them
   can overfit. The next iteration needs scene-held-out evaluation before another
   Kaggle submission.

## Recommended next iteration

1. Implement a correct, scene-held-out evaluator for presence, localisation,
   membership/geometry, and identity.
2. Repair the calibration path and measure threshold sensitivity using only
   train-fold information.
3. Correct the graph-coverage score and introduce a graph-confidence margin so
   weak fits do not force an incorrect identity or `m=1` membership.
4. Compare only a small number of evidence-backed candidate settings, such as
   top-16 versus top-24/top-32, on the A100.
5. Submit only the strongest held-out configuration and keep the best Kaggle
   CSV by its actual leaderboard result.
