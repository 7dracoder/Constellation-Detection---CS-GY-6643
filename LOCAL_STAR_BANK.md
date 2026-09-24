# Local adaptive star-bank experiment

This experiment keeps the existing dense patch matcher and RANSAC solver intact.
It adds an adaptive multi-scale Difference-of-Gaussians star bank, sub-pixel
centroid refinement, and extra candidates in a separate cache directory. Never
overwrite the established cache or scored CSV while testing it.

## 1. Environment (WSL2 Ubuntu, verified on this machine)

The working CUDA environment is installed at `/opt/constellation-venv`. It uses
PyTorch `2.14.0+cu130` through WSL2 and the Windows NVIDIA driver. Do not install
a second Linux NVIDIA display driver inside WSL.

Open Windows PowerShell and enter Ubuntu:

```powershell
wsl -d Ubuntu -u root
```

Then run these commands in Ubuntu. The repository remains on the Windows drive,
while Python and its native libraries run inside Linux:

```bash
cd /mnt/c/Users/subhr/Documents/GitHub/Constellation-Detection---CS-GY-6643
PY=/opt/constellation-venv/bin/python

$PY -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
$PY -m unittest test_star_bank_matcher.py test_joint_geometric_solver.py
```

The expected device is `NVIDIA GeForce RTX 5070 Ti Laptop GPU`, and
`torch.cuda.is_available()` must print `True` before using `--device cuda`.

If the environment ever needs to be recreated, use disk-backed temporary
storage because the default WSL `/tmp` is too small for the CUDA wheel:

```bash
apt-get update
apt-get install -y python3-venv python3-pip git build-essential
python3 -m venv /opt/constellation-venv
mkdir -p /var/tmp/constellation-pip
TMPDIR=/var/tmp/constellation-pip /opt/constellation-venv/bin/python -m pip install --upgrade pip
TMPDIR=/var/tmp/constellation-pip /opt/constellation-venv/bin/python -m pip install -r requirements.txt
TMPDIR=/var/tmp/constellation-pip /opt/constellation-venv/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
```

### Windows-only fallback

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA matching, install a PyTorch build compatible with the local GPU and
driver, then verify it before running the matcher:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Use `--device cpu` in the commands below if CUDA is unavailable. CPU matching is
valid but substantially slower. Star-bank enrichment and RANSAC themselves run
on the CPU.

The remaining commands are Bash commands to run inside the Ubuntu session. If
using the Windows-only fallback instead, replace `$PY` with `python` and each
trailing `\` with PowerShell's backtick continuation character.

## 2. Run unit tests

```bash
$PY -m unittest test_star_bank_matcher.py
```

## 3. Build a fresh labelled-train base cache

```bash
$PY joint_geometric_solver.py --root . --config matcher_config_gpu_wide.json \
  --output outputs/train_base_local.csv --split train --device cuda \
  --top-k 16 --graph-top-k 3 --proposals 20000 \
  --cache-dir outputs/cache_train_local --presence-mode quantile \
  --present-rate 0.625 --consensus-trials 5
```

## 4. Add adaptive star-bank candidates

`--bank-size 0` means adaptive: every separated peak above the response
percentile is eligible, up to the stated maximum. `--top-k 24` preserves room
for new candidates rather than forcing them to displace all 16 base candidates.

```bash
$PY star_bank_matcher.py --root . --split train \
  --cache-dir outputs/cache_train_local \
  --output-dir outputs/cache_train_adaptive \
  --bank-size 0 --max-bank-size 5000 --response-percentile 92 \
  --relative-strength-floor 0.65 \
  --centroid-radius 2 --top-k 24
```

## 5. Build and audit the Gaussian-denoised cache

The scored noise-removal path applies the same mild Gaussian filter
(`sigma=0.65`) to the scene and query before template matching. Keep it in a
separate cache: raw candidates remain better for geometry, while denoised
candidates are better for direct presence/localisation.

```bash
$PY joint_geometric_solver.py --root . \
  --config matcher_config_gpu_wide_gaussian.json \
  --output outputs/train_gaussian_base.csv --split train --device cuda \
  --top-k 16 --graph-top-k 3 --proposals 12000 \
  --cache-dir outputs/cache_train_gaussian --presence-mode quantile \
  --present-rate 0.625 --consensus-trials 5 --transform-model affine

$PY candidate_cache_audit.py --root . --cache-dir \
  outputs/cache_train_local outputs/cache_train_adaptive \
  outputs/cache_train_gaussian
```

The verified Gaussian cache raised top-1 recall from `49/71` to `58/71` and
retained recall from `63/71` to `67/71`, with no retained-location losses.

### Gaussian-strength follow-up

The September 23 sweep compared raw matching with Gaussian `sigma=0.45`,
`0.65`, and `0.85`. Candidate recall was respectively 49/71, 54/71, 58/71,
and 58/71 at top-1; retained recall was 63/71, 65/71, 67/71, and 67/71.
`sigma=0.85` tied candidate recall but improved the complete labelled hybrid
from 0.8993 to **0.9014**. Kaggle subsequently scored the resulting gated
submission at **0.74973**, improving the `sigma=0.65` result by 0.01170.

Build and audit it with:

```bash
$PY joint_geometric_solver.py --root . \
  --config matcher_config_gpu_wide_gaussian085.json \
  --output outputs/train_gaussian085_cachebuild.csv \
  --split train --device cuda --top-k 16 --graph-top-k 1 \
  --proposals 100 --cache-dir outputs/cache_train_gaussian085 \
  --presence-mode quantile --present-rate 0.625 \
  --consensus-trials 1 --transform-model similarity

$PY candidate_cache_audit.py --root . --cache-dir \
  outputs/cache_train_local outputs/cache_train_gaussian045 \
  outputs/cache_train_gaussian outputs/cache_train_gaussian085

$PY presence_refiner.py --root . \
  --input outputs/train_adaptive_affine_multiwidth.csv \
  --output outputs/train_hybrid_gaussian085_presence.csv \
  --split train --cache-dir outputs/cache_train_gaussian085 \
  --train-cache-dir outputs/cache_train_gaussian085 --cross-validated

$PY evaluate.py --root . \
  --predictions outputs/train_hybrid_gaussian085_presence.csv
```

The refiner also accepts multiple caches, with the first cache supplying final
coordinates. The tested `sigma=0.85` + `sigma=0.65` fusion reached 0.9034 at
threshold 0.54 but was rejected: it added 44 validation detections after
changing only one labelled patch. Do not submit
`outputs/submission_hybrid_fusion085_065_t54_multiwidth_gated.csv`; the shift
indicates unstable probability calibration from only three labelled scenes.

The later corrected fold diagnostic also excludes each held-out scene from
membership training (`joint_geometric_solver.py
--cross-validated-membership`). It scores 0.9038, while patch-specific member
F1 is only 0.471. See `PROJECT_STATUS.md` for commands and
`outputs/train_strict_heldout_error_ledger.csv` for the per-patch failure
categories. This shows that figure-star assignment now merits closer work than
another global denoising sweep.

## 6. Run raw geometry with denoised presence/localisation

```bash
$PY candidate_cache_audit.py --root . --cache-dir \
  outputs/cache_train_local outputs/cache_train_adaptive
```

Candidate recall is a diagnostic, not the only gate: the multi-width geometric
search can also recover a stronger consensus from the same retained candidates.
Run both the narrow membership shortlist and a 1.5x expanded shortlist, then
let the density-adjusted fit quality select between them without scene labels:

```bash
$PY joint_geometric_solver.py --root . --config matcher_config_gpu_wide.json \
  --output outputs/train_adaptive_affine_multiwidth.csv --split train --device cpu \
  --top-k 24 --graph-top-k 3 --proposals 12000 \
  --cache-dir outputs/cache_train_adaptive --presence-mode quantile \
  --present-rate 0.625 --graph-query-factor 2.0 \
  --graph-query-expansion-factor 1.5 --max-graph-queries 30 \
  --consensus-trials 5 --transform-model affine

$PY presence_refiner.py --root . \
  --input outputs/train_adaptive_affine_multiwidth.csv \
  --output outputs/train_hybrid_gaussian_base_presence.csv \
  --split train --cache-dir outputs/cache_train_gaussian \
  --train-cache-dir outputs/cache_train_gaussian --cross-validated

$PY evaluate.py --root . \
  --predictions outputs/train_hybrid_gaussian_base_presence.csv
```

The second solver invocation reads the completed cache, so `--device cpu` avoids
requiring CUDA and does not redo patch matching.

## 7. Build validation candidates only after the train gate passes

```bash
$PY joint_geometric_solver.py --root . --config matcher_config_gpu_wide.json \
  --output outputs/validation_base_local.csv --split validation --device cuda \
  --top-k 16 --graph-top-k 3 --proposals 20000 \
  --cache-dir outputs/cache_validation_local --presence-mode quantile \
  --present-rate 0.625 --consensus-trials 5

$PY star_bank_matcher.py --root . --split validation \
  --cache-dir outputs/cache_validation_local \
  --output-dir outputs/cache_validation_adaptive \
  --bank-size 0 --max-bank-size 5000 --response-percentile 92 \
  --relative-strength-floor 0.65 \
  --centroid-radius 2 --top-k 24

$PY joint_geometric_solver.py --root . \
  --config matcher_config_gpu_wide_gaussian.json \
  --output outputs/validation_gaussian_base.csv \
  --split validation --device cuda --top-k 16 --graph-top-k 3 \
  --proposals 12000 --cache-dir outputs/cache_validation_gaussian \
  --presence-mode quantile --present-rate 0.625 \
  --consensus-trials 5 --transform-model affine

$PY joint_geometric_solver.py --root . --config matcher_config_gpu_wide.json \
  --output outputs/submission_adaptive_affine_multiwidth_base.csv \
  --split validation --device cpu --top-k 24 --graph-top-k 3 --proposals 12000 \
  --cache-dir outputs/cache_validation_adaptive --presence-mode quantile \
  --present-rate 0.625 --graph-query-factor 2.0 \
  --graph-query-expansion-factor 1.5 --max-graph-queries 30 \
  --consensus-trials 5 --transform-model affine

$PY presence_refiner.py --root . \
  --input outputs/submission_adaptive_affine_multiwidth_base.csv \
  --output outputs/submission_hybrid_gaussian_multiwidth.csv \
  --split validation --cache-dir outputs/cache_validation_gaussian \
  --train-cache-dir outputs/cache_train_gaussian

$PY constellation_pipeline.py validate --root . \
  --output outputs/submission_hybrid_gaussian_multiwidth.csv
```

The verified hybrid diagnostic is `0.8993` strict/loose with `3/3` identities,
`1.000` strict geometry, `0.858` presence, and `0.674` localisation. This is up
from `0.8625` for the single-width adaptive run and `0.8871` for raw
multi-width. The identity-gated hybrid subsequently scored **0.73803** on
Kaggle, improving the established v4 score of 0.71544 by 0.02259. Keep
`outputs/submission_v4_presence.csv` as the fallback; three labelled scenes
remain a small promotion gate rather than a leaderboard estimate.

To reproduce the scored `sigma=0.85` submission after the raw multi-width
geometry file already exists:

```bash
$PY joint_geometric_solver.py --root . \
  --config matcher_config_gpu_wide_gaussian085.json \
  --output outputs/validation_gaussian085_cachebuild.csv \
  --split validation --device cuda --top-k 16 --graph-top-k 1 \
  --proposals 100 --cache-dir outputs/cache_validation_gaussian085 \
  --presence-mode quantile --present-rate 0.625 \
  --consensus-trials 1 --transform-model similarity

$PY presence_refiner.py --root . \
  --input outputs/submission_adaptive_affine_multiwidth_base.csv \
  --output outputs/submission_hybrid_gaussian085_multiwidth.csv \
  --split validation --cache-dir outputs/cache_validation_gaussian085 \
  --train-cache-dir outputs/cache_train_gaussian085

$PY submission_ensemble.py --root . \
  --established outputs/submission_v4_presence.csv \
  --candidate outputs/submission_hybrid_gaussian085_multiwidth.csv \
  --output outputs/submission_hybrid_gaussian085_multiwidth_gated.csv

$PY constellation_pipeline.py validate --root . \
  --output outputs/submission_hybrid_gaussian085_multiwidth_gated.csv
```
