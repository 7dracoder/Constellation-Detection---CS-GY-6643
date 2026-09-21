# Constellation Detection — CS-GY 6643

This repository contains a reproducible, course-data-only computer-vision
pipeline for the CS-GY 6643 Constellation Detection competition. It uses only
the supplied sky images, query patches, pattern diagrams, and training labels:
no external images, labels, datasets, or pretrained models are used.

## Current status

The current submission is
[`outputs/submission_v2.csv`](outputs/submission_v2.csv),
which scored **0.68196** on Kaggle (September 21, 2026).

| Submission | Kaggle |
| --- | ---: |
| Initial validated baseline | 0.26710 |
| Earlier geometric submission | ~0.58 |
| A100 wide search | 0.66890 |
| Local RTX 5070 Ti rebuild | **0.68196** |

Candidate localisation is unchanged between the last two; the gain came from
the geometry, scoring, and calibration stages.  The largest single defect
fixed was RANSAC under-sampling: at the previous default of 6000 proposals the
true pattern's support on a labelled scene varied between 6 and 9 across
seeds, so constellation identity was partly decided by sampling luck.

None of this is evidence of a guaranteed future score.  See
[PROJECT_STATUS.md](PROJECT_STATUS.md) for the full experiment record, method,
scores, known limitations, and next validation plan.

## Competition output

The submission has 16 data rows and 90 columns: `Id`, `n_patches`, 87 patch
fields, and `constellation`. A patch field is either `-1` or `(x, y, m)`, where
`m=1` denotes a predicted constellation-member star. Always run the validator
before uploading a CSV.

## Core pipeline

The current best pipeline is composed of:

- `gpu_matcher.py`: CUDA/A100 coarse-to-fine patch localisation with a broad
  rotation/scale search and local intensity, gradient, and SSIM refinement.
- `joint_geometric_solver.py`: joint constellation-pattern fitting using the
  supplied diagrams, similarity transforms, reflection handling, and one-to-one
  star assignments.
- `structural_refiner.py`: supplied-pattern extraction and a small patch-only
  membership proposal classifier.
- `constellation_pipeline.py`: baseline matcher, CSV construction, and strict
  submission validation.
- `evaluate.py`: four-component scorer for a train-split prediction, used to
  compare configurations before spending a Kaggle submission.

The GPU-wide configuration keeps 16 spatially distinct candidates per query
patch. On the 71 labelled-present training patches, this improved exact top-1
localisation from 36 to 49 and exact top-16 candidate recall from 56 to 63.

## Setup

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## CPU baseline

Calibrate the present/absent threshold on the labelled scenes:

```sh
.venv/bin/python constellation_pipeline.py calibrate --output matcher_config.json
```

Generate and validate a Kaggle-format CSV:

```sh
.venv/bin/python constellation_pipeline.py predict --config matcher_config.json --output submission.csv
.venv/bin/python constellation_pipeline.py validate --output submission.csv
```

The initial matcher performs background normalization, star-candidate retrieval,
rotation/scale-tolerant local matching, and training-label threshold calibration.
It writes `unknown` as the constellation label and is a baseline rather than the
current best submission.

## Reproducing the current best

Requires a CUDA GPU.  Create the environment with a CUDA build of PyTorch
(the recorded run used `torch 2.11.0+cu128` on an RTX 5070 Ti, compute
capability 12.0), then:

```sh
.venv/Scripts/python.exe joint_geometric_solver.py --root . \
  --config matcher_config_gpu_wide.json \
  --output outputs/submission_v2.csv \
  --split validation --device cuda --top-k 16 --graph-top-k 3 \
  --proposals 20000 --cache-dir outputs/cache_validation \
  --presence-mode quantile --present-rate 0.625
```

Per-scene candidates are cached under `--cache-dir`, so changes confined to the
geometry or scoring stages re-run in minutes without repeating GPU matching.

## Evaluating a change before submitting

`evaluate.py` scores a train-split prediction on all four components with the
identity term corrected.  The older `score_train_predictions` awarded that term
for answering `unknown`, so its number rose as identification got worse; it now
reports localisation components only.

```sh
.venv/Scripts/python.exe joint_geometric_solver.py --root . \
  --config matcher_config_gpu_wide.json \
  --output outputs/train_pred.csv \
  --split train --device cuda --top-k 16 --graph-top-k 3 \
  --proposals 20000 --cache-dir outputs/cache_train \
  --presence-mode quantile --present-rate 0.625
.venv/Scripts/python.exe evaluate.py --predictions outputs/train_pred.csv
```

Only three scenes are labelled, and the membership classifier is fitted on
those same scenes, so treat the result as a regression guard rather than a
leaderboard estimate.  Kaggle's score is authoritative.

The superseded `outputs/submission_a100_wide_fixed.csv` (0.66890) is retained
for comparison.

## NYU Cloud Bursting notes

For CS-GY 6643 coursework, use NYU **Cloud Bursting Open OnDemand** rather than
the researcher-facing Torch cluster. The course announcement specifically asks
students not to use Torch for course workloads. Cloud Bursting may be accessed
through OOD while on NYU VPN and can provide L4 or A100 GPU partitions under the
course Slurm account.

Connect to NYU VPN, open `https://ood.torch.hpc.nyu.edu`, then use OOD's Files
page to upload the project (excluding `.venv` and `outputs`) to
`$HOME/constellation-detection`.  The supplied data are about 130 MB, so they
are small enough for OOD upload.  From an OOD terminal, create the environment
once, then submit the CPU job:

```sh
cd "$HOME/constellation-detection"
bash hpc/bootstrap_env.sh
sbatch hpc/submit_patchmatch_v1.sbatch
```

Check its state with `squeue --me`, and read the completion log shown by Slurm.
The resulting Kaggle-ready CSV is
`$HOME/constellation-detection/outputs/submission_patchmatch_hpc_v1.csv` and
can be downloaded through OOD Files for Kaggle upload.

The historical CPU Slurm CSV is a validated localisation baseline, not the
current best submission: it still reports `unknown` for constellation identity.
