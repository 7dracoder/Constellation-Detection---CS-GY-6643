#!/usr/bin/env bash
# Usage: run_budget_duplicate_validation.sh TAG SEED_OFFSET PARALLEL SCENE...
# Writes outputs/budget_duplicate/<TAG>/<scene>.csv and .log; skips finished scenes.
set -u
cd /mnt/c/Users/subhr/Documents/GitHub/Constellation-Detection---CS-GY-6643
tag=$1; offset=$2; parallel=$3; shift 3
out=outputs/budget_duplicate/$tag
mkdir -p "$out"

run_scene() {
    scene=$1
    [ -s "$out/$scene.csv" ] && return 0
    /opt/constellation-venv/bin/python joint_geometric_solver.py --root . \
        --config matcher_config_gpu_wide.json --split validation --device cpu \
        --top-k 24 --graph-top-k 3 --proposals 3000 \
        --cache-dir outputs/cache_validation_adaptive --presence-mode quantile \
        --present-rate 0.625 --graph-query-factor 2.0 \
        --graph-query-expansion-factor 1.5 --max-graph-queries 30 \
        --consensus-trials 3 --transform-model similarity --proposal-mode hybrid \
        --scale-ratio-min 4.0 --scale-ratio-max 7.0 --patch-budget-weight 1.0 \
        --duplicate-evidence-cache-dir outputs/cache_validation_gaussian085 \
        --duplicate-evidence-threshold 0.95 --duplicate-evidence-top-k 10 \
        --duplicate-coverage-weight 0.5 --duplicate-coverage-rate 0.10 \
        --seed-offset "$offset" \
        --output "$out/$scene.csv.tmp" --scene "$scene" > "$out/$scene.log" 2>&1 \
        && mv "$out/$scene.csv.tmp" "$out/$scene.csv"
}
export -f run_scene
export out offset
printf '%s\n' "$@" | xargs -P "$parallel" -I{} bash -c 'run_scene {}'
echo "done $tag"
