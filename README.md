# Constellation Detection — CS-GY 6643

This repository contains a reproducible computer-vision pipeline for the
CS-GY 6643 Constellation Detection competition. The scored submissions through
v4 use only the supplied sky images, query patches, pattern diagrams, and
training labels. Experimental external-catalog work is kept separate and must
list its public source and licence.

## Current status

The current scored submission is
[`outputs/submission_v4_presence.csv`](outputs/submission_v4_presence.csv).
It scored **0.71544** on Kaggle (September 21, 2026), improving v3 by 0.03113.
The next validated candidate is
[`outputs/submission_v5_affine_gated.csv`](outputs/submission_v5_affine_gated.csv).
It has not been scored yet: it uses full-affine refinement on the 12 scenes
where affine and similarity geometry agree on identity, and retains the entire
v4 row on the four disagreements so the label and `m=1` coordinates cannot
describe different patterns.
A leave-one-scene-out presence pass
raised the labelled diagnostic from 0.8369 to **0.8709** (loose geometry) and
from 0.8119 to **0.8459** (strict geometry), while leaving graph-supported
`m=1` placements and constellation labels unchanged.

| Submission | Kaggle |
| --- | ---: |
| Initial validated baseline | 0.26710 |
| Earlier geometric submission | ~0.58 |
| A100 wide search | 0.66890 |
| Local RTX 5070 Ti rebuild | 0.68196 |
| Assignment-robustness pass | 0.68431 |
| Presence-refined pass | **0.71544** |
| Affine-gated candidate | not submitted |
| Teammate graphguard (v6) | 0.70102 (not promoted) |

Candidate localisation is unchanged across all three GPU runs; the gains came
from the geometry, scoring, and calibration stages.  The largest single
defect fixed in the rebuild was RANSAC under-sampling: at the previous
default of 6000 proposals the true pattern's support on a labelled scene
varied between 6 and 9 across seeds, so constellation identity was partly
decided by sampling luck.  The assignment-robustness pass then audited the
solver for other hard-coded assumptions tuned on the same three labelled
scenes and fixed the ones that could not have been exercised by them (a
RANSAC search-time filter, and single-seed dependence in the final pattern
choice); it also tested a margin-weighting idea, found it did not help, and
shipped it disabled rather than assumed beneficial.  See
[PROJECT_STATUS.md](PROJECT_STATUS.md) for the details.  The +0.00235 gain
is small and consistent with 11 of 16 validation labels being unchanged.

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
  supplied diagrams, similarity-RANSAC initialization, optional full-affine
  refinement, reflection handling, and one-to-one star assignments.
- `structural_refiner.py`: supplied-pattern extraction and a small patch-only
  membership proposal classifier.
- `constellation_pipeline.py`: baseline matcher, CSV construction, and strict
  submission validation.
- `evaluate.py`: four-component scorer for a train-split prediction, used to
  compare configurations before spending a Kaggle submission.
- `presence_refiner.py`: regularised, course-data-only presence classifier
  using matcher statistics and simple patch features. It preserves all
  graph-supported members and only revises non-member presence/localisation.
- `submission_ensemble.py`: keeps affine rows only when affine and established
  geometry agree on identity; disagreements retain the complete established
  row rather than mixing a name with another pattern's coordinates.
- `external_catalog_validator.py`: read-only identity cross-check against the
  public d3-celestial line catalog. It never rewrites a submission.

The GPU-wide configuration keeps 16 spatially distinct candidates per query
patch. On the 71 labelled-present training patches, this improved exact top-1
localisation from 36 to 49 and exact top-16 candidate recall from 56 to 63.

## Setup

For the external-image multi-model experiment, open
[`colab_multimodel.ipynb`](colab_multimodel.ipynb) in Colab with a GPU runtime.
It trains two complementary patch encoders, combines their evidence with the
cached correlation matcher and radial photometry, fits presence/membership
models, evaluates similarity and affine geometry, and downloads a CSV plus
the model/report archive. Sources and credits are in
[`EXTERNAL_TRAINING_DATA.md`](EXTERNAL_TRAINING_DATA.md).

The experiment holds entire scenes out of all three downstream classifiers.
Both neural encoders pretrain only on external observations and synthetic
patches. The report records whether the ensemble passes its promotion gate;
if it fails, `submission_recommended.csv` is explicitly the scored v4 fallback.
`submission_multimodel_experimental.csv` remains available for inspection.

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
  --output outputs/submission_v3.csv \
  --split validation --device cuda --top-k 16 --graph-top-k 3 \
  --proposals 20000 --cache-dir outputs/cache_validation \
  --presence-mode quantile --present-rate 0.625 --consensus-trials 5
```

Per-scene candidates are cached under `--cache-dir`, so changes confined to the
geometry or scoring stages re-run in minutes without repeating GPU matching.
`--consensus-trials 5` votes the final pattern choice across 5 independent
RANSAC seeds instead of trusting one fixed seed; it costs roughly 5x the
geometry-fitting time and needs no extra GPU matching.

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

The superseded `outputs/submission_v2.csv` (0.68196) and
`outputs/submission_a100_wide_fixed.csv` (0.66890) are retained for
comparison.

## Presence-refined scored submission

Starting from the scored `submission_v3.csv`, reproduce the new candidate:

```sh
.venv/bin/python presence_refiner.py --root . \
  --input outputs/submission_v3.csv \
  --output outputs/submission_v4_presence.csv \
  --split validation \
  --cache-dir outputs/cache_validation \
  --train-cache-dir outputs/cache_train
.venv/bin/python constellation_pipeline.py validate \
  --root . --output outputs/submission_v4_presence.csv
```

The default regularisation and decision threshold were chosen with
leave-one-scene-out predictions, not same-scene fitted predictions. The three
labelled scenes are still a very small validation set. Kaggle confirmed a
0.03113 public-leaderboard improvement over v3.

## Affine-gated v5 candidate

The assignment states that a diagram's aspect ratio is unrelated to its sky
appearance. The original solver nevertheless refit only similarity transforms,
which preserve aspect ratio. The affine mode keeps the stable two-point
similarity search, then refines each consensus with a safeguarded six-parameter
affine transform.

```sh
.venv/bin/python joint_geometric_solver.py --root . \
  --config matcher_config_gpu_wide.json \
  --output outputs/submission_v5_affine_base.csv \
  --top-k 16 --graph-top-k 3 --proposals 12000 \
  --cache-dir outputs/cache_validation \
  --presence-mode quantile --present-rate 0.625 \
  --consensus-trials 5 --transform-model affine
.venv/bin/python presence_refiner.py --root . \
  --input outputs/submission_v5_affine_base.csv \
  --output outputs/submission_v5_affine_presence.csv \
  --split validation --cache-dir outputs/cache_validation \
  --train-cache-dir outputs/cache_train
.venv/bin/python submission_ensemble.py --root . \
  --established outputs/submission_v4_presence.csv \
  --candidate outputs/submission_v5_affine_presence.csv \
  --output outputs/submission_v5_affine_gated.csv
```

On the three labeled scenes, five-seed affine + presence refinement raised the
loose diagnostic from 0.8625 to **0.8871** and the strict diagnostic from
0.8458 to **0.8621**, with identity remaining 3/3. These are regression checks,
not a leaderboard estimate.

Initialize the public catalog submodules and run the independent, read-only
identity check with:

```sh
git submodule update --init --recursive
.venv/bin/python external_catalog_validator.py --root . \
  --input outputs/submission_v5_affine_gated.csv
```

The validator recovers all three labeled identities from exact figure points
and agrees with 10/16 v5 identities. The remaining external fits are weak or
ambiguous, so catalog output is recorded as evidence rather than used as an
automatic label oracle.

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
