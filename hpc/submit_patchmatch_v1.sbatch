#!/usr/bin/env bash
# Run with: sbatch hpc/submit_patchmatch_v1.sbatch
# Submit from an NYU Cloud Bursting OOD terminal; do not run matching in it.
#SBATCH --job-name=constellation-v1
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=8
# n2c48m24 has a 24 GB node-memory profile; 20 GB leaves scheduler headroom.
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --output=logs/constellation-v1-%j.out
#SBATCH --error=logs/constellation-v1-%j.err

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs outputs

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python environment: ${PYTHON_BIN}" >&2
  exit 1
fi

# Prevent each process/library from taking all cores at the same time.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

"${PYTHON_BIN}" constellation_pipeline.py predict \
  --config matcher_config_fast.json \
  --output outputs/submission_patchmatch_hpc_v1.csv \
  --workers 4

"${PYTHON_BIN}" constellation_pipeline.py validate \
  --output outputs/submission_patchmatch_hpc_v1.csv

echo "Submission file: ${PROJECT_ROOT}/outputs/submission_patchmatch_hpc_v1.csv"
