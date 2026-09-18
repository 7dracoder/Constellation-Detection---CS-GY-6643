#!/bin/bash
#SBATCH --job-name=constellation-wide
#SBATCH --account=cs_gy_6643-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:30:00
#SBATCH --output=logs/gpu_wide_%j.out
#SBATCH --error=logs/gpu_wide_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/constellation-detection}}"
cd "$PROJECT_ROOT"
mkdir -p logs outputs
# The Cloud Bursting image may provide this environment as a Conda-style
# prefix, which has a Python executable but no POSIX ``activate`` script.
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
command -v python

nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA PyTorch is required by this GPU job"
print("torch", torch.__version__, "device", torch.cuda.get_device_name(0))
PY

python gpu_matcher.py calibrate \
  --root "$PROJECT_ROOT" \
  --config matcher_config_gpu_wide.json \
  --output outputs/matcher_config_gpu_calibrated.json \
  --device cuda \
  --batch-size 32

python gpu_matcher.py predict \
  --root "$PROJECT_ROOT" \
  --config outputs/matcher_config_gpu_calibrated.json \
  --output outputs/submission_gpu_wide_raw.csv \
  --device cuda \
  --batch-size 32

python structural_refiner.py \
  --root "$PROJECT_ROOT" \
  --input outputs/submission_gpu_wide_raw.csv \
  --output outputs/submission_gpu_wide_structural.csv \
  --device cuda

python constellation_pipeline.py validate \
  --root "$PROJECT_ROOT" \
  --output outputs/submission_gpu_wide_raw.csv
python constellation_pipeline.py validate \
  --root "$PROJECT_ROOT" \
  --output outputs/submission_gpu_wide_structural.csv
