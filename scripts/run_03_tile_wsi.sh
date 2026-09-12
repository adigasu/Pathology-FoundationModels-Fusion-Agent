#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ -f "${REPO_ROOT}/.env_path_agent/bin/python" ]; then
    PYTHON_EXEC="${REPO_ROOT}/.env_path_agent/bin/python"
elif [ -f "${REPO_ROOT}/.env_agent/bin/python" ]; then
    PYTHON_EXEC="${REPO_ROOT}/.env_agent/bin/python"
elif [ -f "${REPO_ROOT}/.venv/bin/python" ]; then
    PYTHON_EXEC="${REPO_ROOT}/.venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

echo "========================================================================"
echo "Running Step 1.4: 20x Downsampling & Parallel Tissue Tiling"
echo "Repository Root   : ${REPO_ROOT}"
echo "Python Interpreter: ${PYTHON_EXEC}"
echo "========================================================================"

"${PYTHON_EXEC}" "${SCRIPT_DIR}/03_tile_wsi.py" "$@"
