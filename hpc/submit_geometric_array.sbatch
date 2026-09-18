#!/bin/bash
# Course-data-only CPU array search.  It deliberately does not request a GPU:
# the geometric RANSAC stage is CPU-bound and is checkpointed per
# scene/pattern pair.
#SBATCH --job-name=constellation-geometry
#SBATCH --account=cs_gy_6643-2026fa
#SBATCH --partition=n2c48m24
# There are 16 validation scenes and 48 supplied reference patterns.
#SBATCH --array=0-767%48
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/geometric_%A_%a.out
#SBATCH --error=logs/geometric_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/constellation-detection}}"
cd "$PROJECT_ROOT"
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
mkdir -p logs outputs/geometric_parts

mapfile -t SCENES < <(python - <<'PY'
import csv
with open('sample_submission.csv', newline='', encoding='utf-8') as stream:
    for row in csv.DictReader(stream):
        print(row['Id'])
PY
)
mapfile -t PATTERNS < <(find patterns -maxdepth 1 -name '*_pattern.png' -printf '%f\n' | sed 's/_pattern\.png$//' | sort)

SCENE_INDEX=$((SLURM_ARRAY_TASK_ID / ${#PATTERNS[@]}))
PATTERN_INDEX=$((SLURM_ARRAY_TASK_ID % ${#PATTERNS[@]}))
SCENE="${SCENES[$SCENE_INDEX]}"
PATTERN="${PATTERNS[$PATTERN_INDEX]}"
PART="outputs/geometric_parts/${SCENE}__${PATTERN}.json"

python geometric_ensemble.py \
  --root "$PROJECT_ROOT" \
  --output outputs/unused.csv \
  --scene "$SCENE" \
  --pattern "$PATTERN" \
  --part-output "$PART" \
  --per-mode 5 \
  --proposals 5000
