#!/usr/bin/env bash
# Run with: sbatch hpc/submit_structural_gpu.sbatch
# Requires hpc/bootstrap_gpu_env.sh to have completed once.
#SBATCH --job-name=constellation-graph
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/constellation-graph-%j.out
#SBATCH --error=logs/constellation-graph-%j.err

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs outputs

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python environment: ${PYTHON_BIN}" >&2
  exit 1
fi

nvidia-smi
"${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; do not run this structural GPU job on a CPU node")
print("using", torch.cuda.get_device_name(0))
PY

"${PYTHON_BIN}" structural_refiner.py \
  --input outputs/submission_patchmatch_hpc_v1.csv \
  --output outputs/submission_structural_gpu_v1.csv \
  --device cuda

"${PYTHON_BIN}" constellation_pipeline.py validate \
  --output outputs/submission_structural_gpu_v1.csv

echo "Submission file: ${PROJECT_ROOT}/outputs/submission_structural_gpu_v1.csv"
