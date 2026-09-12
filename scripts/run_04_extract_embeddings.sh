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

# If first arg is a model name without --model flag, prepend --model
ARGS=("$@")
if [ $# -ge 1 ] && [[ "$1" =~ ^(uni2|virchow2|gigapath|prism2)$ ]]; then
    ARGS=("--model" "$1" "${@:2}")
elif [ $# -eq 0 ]; then
    ARGS=("--model" "uni2")
fi

echo "========================================================================"
echo "Running Step 1.5: Foundation Model Feature Extraction"
echo "Repository Root   : ${REPO_ROOT}"
echo "Python Interpreter: ${PYTHON_EXEC}"
echo "Arguments         : ${ARGS[*]}"
echo "========================================================================"

"${PYTHON_EXEC}" "${SCRIPT_DIR}/04_extract_embeddings.py" "${ARGS[@]}"
