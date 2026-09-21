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

print_usage() {
    cat << 'HELP'
Usage: ./scripts/run_phase2.sh [OPTIONS] [SEED] [AGENT_VERSION]

Phase 2 Single-Seed Autonomous Fusion Agent Runner.

Options:
  -s, --seed SEED             Random seed integer (default: 42)
  -a, --agent,
      --agent-version VER     Agent version: v1, v2, v3, v4, v5, or llm (default: v4)
  -o, --output-dir DIR        Destination directory for results and metrics (default: artifacts/results)
  -h, --help                  Show this help message and exit

Positional arguments are also supported:
  ./scripts/run_phase2.sh 42
  ./scripts/run_phase2.sh 42 v4
  ./scripts/run_phase2.sh v4

Flagged examples:
  ./scripts/run_phase2.sh --seed 42 --agent v4 --output-dir artifacts/results/validation
  ./scripts/run_phase2.sh -s 1337 -a v2 -o artifacts/results/exp_v2
HELP
}

SEED=42
AGENT_VERSION=""
OUTPUT_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_usage
            exit 0
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        --seed=*)
            SEED="${1#*=}"
            shift
            ;;
        -a|--agent|--agent-version)
            AGENT_VERSION="${2,,}"
            shift 2
            ;;
        --agent=*|--agent-version=*)
            val="${1#*=}"
            AGENT_VERSION="${val,,}"
            shift
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-dir=*|-o=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        v1|v2|v3|v4|v5|llm|V1|V2|V3|V4|V5|LLM)
            AGENT_VERSION="${1,,}"
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
if [ -n "${OUTPUT_DIR}" ]; then
    CMD+=(--output-dir "${OUTPUT_DIR}")
fi
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "========================================================================"
echo "Phase 2: Autonomous Multi-Foundation-Model Fusion Agent (Single Seed)"
echo "Repository Root   : ${REPO_ROOT}"
echo "Python Interpreter: ${PYTHON_EXEC}"
echo "Target Seed       : ${SEED}"
echo "Agent Version     : ${AGENT_VERSION:-v4 (default)}"
echo "Output Directory  : ${OUTPUT_DIR:-artifacts/results (default)}"
echo "Threads Configured: OMP=4, OPENBLAS=4, MKL=4"
echo "========================================================================"

exec "${CMD[@]}"
