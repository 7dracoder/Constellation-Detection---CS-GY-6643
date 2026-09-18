# Constellation Detection — CS-GY 6643

This repository contains a reproducible, course-data-only computer-vision
pipeline for the CS-GY 6643 Constellation Detection competition. It uses only
the supplied sky images, query patches, pattern diagrams, and training labels:
no external images, labels, datasets, or pretrained models are used.

## Current status

The current A100/Colab submission is
[`outputs/submission_a100_wide_fixed.csv`](outputs/submission_a100_wide_fixed.csv).
It passed the local Kaggle-format validator and scored **0.66890** on Kaggle.
This is an improvement over the earlier approximately `0.58` and `0.26710`
submissions, but it is not evidence of a guaranteed future score.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the full experiment record,
method, scores, known limitations, and next validation plan.

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

## Current A100/Colab output

The recommended current submission is the checked-in
`outputs/submission_a100_wide_fixed.csv`, produced from an A100 Colab runtime
with `matcher_config_gpu_wide.json`. The final Colab run completed all 16 scenes,
cached each scene's candidates, and validated the generated CSV before download.

Do not assume the older notebook's CPU path or any local approximate metric is a
leaderboard estimate; Kaggle's score is authoritative.

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
