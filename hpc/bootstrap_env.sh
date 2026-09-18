#!/usr/bin/env bash
# Create the Python environment once, from an NYU Cloud Bursting OOD terminal.
# This only installs packages; matching itself is submitted through Slurm.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"

if [[ -x "${PYTHON_BIN}" ]]; then
  # A Conda prefix is valid here too; it does not necessarily ship an
  # activate script, so use its interpreter directly throughout.
  BOOTSTRAP_PYTHON="${PYTHON_BIN}"
else
  BOOTSTRAP_PYTHON="python3"
fi

"${BOOTSTRAP_PYTHON}" - <<'PY'
import sys
if sys.version_info < (3, 9):
    raise SystemExit(f"Python 3.9+ is required; found {sys.version}")
print("using", sys.version)
PY

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${BOOTSTRAP_PYTHON}" -m venv "${VENV_DIR}"
fi

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install --requirement "${PROJECT_ROOT}/requirements.txt"
"${PYTHON_BIN}" - <<'PY'
import cv2, numpy, PIL, scipy
print("environment ready")
print("opencv", cv2.__version__)
print("numpy", numpy.__version__)
PY
