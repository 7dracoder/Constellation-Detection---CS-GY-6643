#!/usr/bin/env bash
# Run once from a Cloud Bursting OOD terminal before submit_structural_gpu.sbatch.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing project environment: ${PYTHON_BIN}" >&2
  exit 1
fi

# Keep CUDA PyTorch separate from requirements.txt: macOS development and CPU
# runs do not need a multi-gigabyte CUDA wheel.  The wheel is selected at the
# NYU GPU node, where `torch.cuda.is_available()` is verified again by Slurm.
if ! "${PYTHON_BIN}" -c 'import torch' 2>/dev/null; then
  "${PYTHON_BIN}" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
fi

"${PYTHON_BIN}" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available from login node", torch.cuda.is_available())
PY
