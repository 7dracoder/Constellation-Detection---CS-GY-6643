#!/bin/bash
# Submit this only after every element of the geometric array has succeeded.
#SBATCH --job-name=constellation-merge
#SBATCH --account=cs_gy_6643-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/geometric_merge_%j.out
#SBATCH --error=logs/geometric_merge_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/constellation-detection}}"
cd "$PROJECT_ROOT"
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"

python merge_geometric_parts.py \
  --root "$PROJECT_ROOT" \
  --parts outputs/geometric_parts \
  --output outputs/submission_geometric_cpu.csv
python constellation_pipeline.py validate \
  --root "$PROJECT_ROOT" \
  --output outputs/submission_geometric_cpu.csv
