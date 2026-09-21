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

# Parse positional arguments:
# Supports:
#   ./scripts/run_phase2.sh               -> Seed 42, default agent (v4)
#   ./scripts/run_phase2.sh 42            -> Seed 42, default agent (v4)
#   ./scripts/run_phase2.sh 42 v4         -> Seed 42, agent v4
#   ./scripts/run_phase2.sh v4            -> Seed 42 (default), agent v4
#   ./scripts/run_phase2.sh 42 --agent-version v4
SEED=42
AGENT_VERSION=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        v1|v2|v3|v4|v5|llm|V1|V2|V3|V4|V5|LLM)
            AGENT_VERSION="${1,,}"
            shift
            ;;
        --agent-version)
            AGENT_VERSION="${2,,}"
            shift 2
            ;;
        --agent-version=*)
            val="${1#*=}"
            AGENT_VERSION="${val,,}"
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --seed=*)
            SEED="${1#*=}"
            shift
            ;;
        [0-9]*)
            SEED="$1"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

CMD=("${PYTHON_EXEC}" "${SCRIPT_DIR}/05_run_phase2_fusion_agent.py" --seed "${SEED}")
if [ -n "${AGENT_VERSION}" ]; then
    CMD+=(--agent-version "${AGENT_VERSION}")
fi
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "========================================================================"
echo "Phase 2: Autonomous Multi-Foundation-Model Fusion Agent"
echo "Repository Root   : ${REPO_ROOT}"
echo "Python Interpreter: ${PYTHON_EXEC}"
echo "Target Seed       : ${SEED}"
echo "Agent Version     : ${AGENT_VERSION:-v4 (default)}"
echo "Threads Configured: OMP=4, OPENBLAS=4, MKL=4"
echo "========================================================================"

exec "${CMD[@]}"
