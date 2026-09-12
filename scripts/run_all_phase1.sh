#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "========================================================================"
echo "PHASE 1: COMPLETE END-TO-END PIPELINE EXECUTION"
echo "========================================================================"

echo ""
echo ">>> STEP 1.2: Data Audit & Cohort Reduction..."
"${SCRIPT_DIR}/run_01_data_audit.sh"

echo ""
echo ">>> STEP 1.3: Multi-Seed Stratified Splitting..."
"${SCRIPT_DIR}/run_02_generate_splits.sh"

echo ""
echo ">>> STEP 1.4: 20x Preprocessing & Tiling..."
"${SCRIPT_DIR}/run_03_tile_wsi.sh"

echo ""
echo ">>> STEP 1.5: Feature Extraction (UNI2-h)..."
"${SCRIPT_DIR}/run_04_extract_embeddings.sh" uni2

echo ""
echo ">>> STEP 1.5: Feature Extraction (Virchow2)..."
"${SCRIPT_DIR}/run_04_extract_embeddings.sh" virchow2

echo ""
echo ">>> STEP 1.5: Feature Extraction (Prov-GigaPath)..."
"${SCRIPT_DIR}/run_04_extract_embeddings.sh" gigapath


echo ""
echo "========================================================================"
echo "PHASE 1 PIPELINE COMPLETED SUCCESSFULLY!"
echo "========================================================================"
