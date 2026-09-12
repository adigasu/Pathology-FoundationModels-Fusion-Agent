#!/usr/bin/env bash
set -euo pipefail

# Avoid thread thrashing across NUMA nodes on small matrix operations
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# Determine repository root relative to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Resolve python interpreter (.env_path_agent, .env_agent, or system python)
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
echo "Phase 2: Autonomous Multi-Foundation-Model Fusion Agent"
echo "Repository Root   : ${REPO_ROOT}"
echo "Python Interpreter: ${PYTHON_EXEC}"
echo "Target Seed       : ${1:-42}"
echo "Threads Configured: OMP=4, OPENBLAS=4, MKL=4"
echo "========================================================================"

"${PYTHON_EXEC}" "${SCRIPT_DIR}/05_run_phase2_fusion_agent.py" --seed "${1:-42}" "${@:2}"
