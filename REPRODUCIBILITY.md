# Reproducibility Guide: Pathology Foundation Models Fusion Agent

This guide provides end-to-end instructions for reproducing the entire data curation, tiling, feature extraction, autonomous agent fusion search, and evaluation benchmark across all experimental seeds (`42`, `1337`, `2026`).

---

> [!IMPORTANT]
> **CPU Thread Contention**: On multi-core servers, OpenMP / MKL thread pools can cause spin-lock overhead on small matrix operations. Always set thread limits when running evaluation scripts:
> ```bash
> export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
> ```

## 1. Environment Setup

### 1.1 Hardware Requirements
- **OS**: Linux (tested on Ubuntu 22.04 LTS / x86_64)
- **GPU**: NVIDIA GPU with >= 16 GB VRAM (e.g., RTX 3090, A5000, A100, V100, or Colab T4/A100)
- **Disk Space**: ~25 GB for WSI JPGs, ~5 GB for cached FP16 tile and slide embeddings

### 1.2 Installation Options

#### Option A: Quick Install via `requirements.txt` (Standard `pip`)
```bash
# 1. Create a virtual environment with Python 3.10+
python3 -m venv .env_path_agent
source .env_path_agent/bin/activate

# 2. Install PyTorch with your system's CUDA version (e.g., CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install core dependencies
pip install -r requirements.txt
```

#### Option B: Ultra-Fast Hermetic Install via `uv` (Recommended)
```bash
# 1. Create the dedicated virtual environment with Python 3.10
uv venv .env_path_agent --python /usr/bin/python3.10
export UV_LINK_MODE=copy

# 2. Install PyTorch with CUDA 12.4 wheels
uv pip install --python .env_path_agent/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install dependencies from requirements.txt
uv pip install --python .env_path_agent/bin/python -r requirements.txt

# 4. Verify GPU access
.env_path_agent/bin/python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0))"
```

### 1.3 Foundation Model Credentials & Access Links
The project leverages state-of-the-art pathology foundation models hosted on Hugging Face:

1. **UNI2-h** (`MahmoodLab/UNI2-h`): ViT-H/14 with SwiGLU, 1536-d ([Hugging Face](https://huggingface.co/MahmoodLab/UNI2-h)). Gated access.
2. **Virchow2** (`paige-ai/Virchow2`): ViT-H/14 with SwiGLU, 2560-d ([Hugging Face](https://huggingface.co/paige-ai/Virchow2)). Gated access.
3. **Prov-GigaPath** (`prov-gigapath/prov-gigapath`): ViT-Giant/14 (1.3B params), 1536-d ([Hugging Face](https://huggingface.co/prov-gigapath/prov-gigapath)). Instant gated access.
4. **Prism2** (`paige-ai/Prism2`): Multimodal Clinical VLM (Perceiver Resampler + Phi-3 3.8B LLM decoder, 3072-d diagnostic state) ([Hugging Face](https://huggingface.co/paige-ai/Prism2)).

Set your Hugging Face authentication token:
```bash
export HF_TOKEN="your_huggingface_access_token"
# Or run: huggingface-cli login
```

*(Optional) Flash-Attention for Prism2*:
```bash
uv pip install --python .env_path_agent/bin/python "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
```

---

## 2. Dataset Configuration & Canonical Paths

The pipeline resolves the image directory via the `LUNG_DATA_DIR` environment variable, falling back to `./data/lung_data` or `~/lung_data`:
```bash
# Export the path to your raw WSI JPG files:
export LUNG_DATA_DIR="/path/to/your/Lung_Pathology_Image_JPG"
```

Directory Structure:
- **Raw WSI JPGs**: `$LUNG_DATA_DIR`
- **Clinical Metadata**: `artifacts/CLWD.csv` (Source metadata)
- **Curated Metadata**: `artifacts/curated_patients.csv` (Audited 204-patient cohort)
- **Frozen Split Partitions**: `artifacts/splits/splits_seed_{42,1337,2026}.json`
- **Extracted Feature Caches**: `artifacts/features/` (Tile embeddings in FP16, slide vectors in FP32)
- **Tiling QC Overlays**: `artifacts/tiling_qc/`
- **Evaluation Figures**: `artifacts/figures/`

---

## 3. Phase 1: Data Audit, Tiling & Foundation Feature Extraction

Convenience bash scripts execute each stage deterministically:

### Step 1.1: Formal Data Audit & Cohort Curation (210 -> 204 Patients)
Audits the raw downloaded dataset against `artifacts/CLWD.csv`.

**Critical Data Audit Finding**:
Out of the 408 whole-slide image entries indexed in `CLWD.csv`, **22 image files are missing from the public Hugging Face repository download** (`kmmuleelab/Lung_Pathology_Image_JPG`). For **6 patients**, all associated slide images were completely omitted from the release:
- `8219606_75_Female` (slide WSI-343, Lepidic)
- `8221926_65_Male` (slide WSI-344, Lepidic)
- `8226738_42_Female` (slide WSI-351, Acinar)
- `8238589_51_Female` (slide WSI-365, Acinar)
- `8243660_65_Female` (slide WSI-376, Acinar)
- `8388318_57_Male` (slides WSI-78, WSI-79, Papillary)

Because these 6 patients have zero accessible images on disk, the curated cohort consists of the remaining **204 patients** (386 slides available on disk). Selecting the lowest available `WSI_ID` per patient under the 1-slide-per-patient protocol yields a frozen, verified cohort of exactly **204 whole-slide images (204 patients)** with 100% complete data on disk:
```bash
./scripts/run_01_data_audit.sh
```
*Output*: `artifacts/curated_patients.csv` (204 patients, 100% on disk)

### Step 1.2: Multi-Seed Stratified Splitting
Generates patient-level stratified 70/10/20 train/val/test partitions across seeds `42`, `1337`, and `2026`:
```bash
./scripts/run_02_generate_splits.sh
```
*Outputs*: `artifacts/splits/splits_seed_*.json` and `.csv`.

### Step 1.3: 20× Tiling & Background Rejection
Downsamples 80× slides by 4× to 20× equivalent, performs dual-criterion tissue segmentation (luminance < 220 & color-variance > 5), and extracts non-overlapping $224 \times 224$ tiles:
```bash
./scripts/run_03_tile_wsi.sh
```
*Outputs*: `artifacts/tiles_metadata.json` (398,649 valid tiles) and QC overlays in `artifacts/tiling_qc/`.

### Step 1.4: Foundation Model Feature Extraction
Extracts tile embeddings with model-specific normalizations and caches them in FP16:
```bash
# UNI2-h (1536-d)
./scripts/run_04_extract_embeddings.sh uni2

# Virchow2 (2560-d)
./scripts/run_04_extract_embeddings.sh virchow2

# Prov-GigaPath (1536-d)
./scripts/run_04_extract_embeddings.sh gigapath
```
*Master Script*: Run all Phase 1 steps end-to-end:
```bash
./scripts/run_all_phase1.sh
```

---

## 4. Phase 2: Autonomous Agent Fusion Search

### Step 2.1: 25-Trial Autonomous Search (Seed 42)
Executes the autonomous agent exploring Early, Late, and Intermediate fusion architectures with 1.0-SE overfitting guardrails:
```bash
./scripts/run_phase2.sh 42
```
*Outputs*:
- `artifacts/agent_decision_log_seed_42.json`: Auditable decision log conditioning proposals on fold performance.
- `artifacts/agent_search_summary_seed_42.md`: Step-by-step reasoning summary.
- `artifacts/phase2_comparison_table_seed_42.md`: 10-row baseline and fused model evaluation with 1,000-sample bootstrap CIs.

### Step 2.2: Multi-Seed Repeatability Verification (Seeds 42, 1337, 2026)
Executes independent end-to-end searches and evaluations across all 3 seeds with zero cross-seed contamination:
```bash
./scripts/run_all_phase2.sh
```
*Outputs*:
- `artifacts/phase2_multi_seed_comparison.md`: Multi-seed evaluation reporting Mean ± SD.
- `artifacts/phase2_multi_seed_comparison.json`: Machine-readable results.

---

## 5. Phase 3: Multimodal Vision-Language Benchmarks & Evaluation Figures

### Step 3.1: Prism2 Multimodal Feature Extraction
Extracts Perceiver Resampler and Phi-3 VLM diagnostic embeddings from cached Virchow2 tile representations:
```bash
# Extract Prism2 Base (2560-d), Latents Mean (2560-d), and VLM Diagnostic (3072-d):
.env_path_agent/bin/python scripts/extract_prism2_embeddings.py
```
*Outputs*: `artifacts/features/prism2_diag_slide.pt`, `artifacts/features/prism2_base_slide.pt`, `artifacts/features/prism2_latents_mean_slide.pt`

### Step 3.2: Multimodal Vision-Language Representation Benchmark
Evaluates Vision-Only Triad vs. Multimodal Quad-Model Late Fusion across all 3 seeds:
```bash
# Run Vision-Only and Multimodal benchmarks across all 3 seeds:
.env_path_agent/bin/python scripts/compare_prism2_representations.py --mode both
```
*Output*: `artifacts/prism2_benchmark_3seeds.json`

---

## 6. Main Benchmark Reproduction

Runs the master evaluation across all 3 seeds, reproducing the main results tables:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.env_path_agent/bin/python scripts/eval_3seeds_comparison.py --mode both
```

---


### 6.1: Ablation Study of Agent Search Architectures

To evaluate the comparative impact of different autonomous search paradigms, run the 5-paradigm agent architecture ablation study:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.env_path_agent/bin/python scripts/run_agent_search_ablation.py --seeds 42 1337 2026
```

### Step 6.2: Generate All Evaluation Figures & Failure Analysis Tables
Renders publication-quality figures and per-class statistical tables:
```bash
.env_path_agent/bin/python scripts/generate_evaluation_figures.py
```
*Generated Deliverables in `artifacts/figures/`*:
1. `confusion_matrix_quad_fusion.png`: Raw counts and row-normalized recall heatmap for Quad-Model Fusion on test cohort.
2. `confusion_matrix_tri_fusion.png`: Confusion matrix for Tri-Model Vision Fusion.
3. `per_class_auroc_comparison.png`: Grouped bar chart comparing all 7 subtypes across single models and fusion champions.
4. `roc_curves_multiclass.png`: Multi-class One-vs-Rest ROC curves with macro-average.
5. `calibration_impact.png`: Subtype recall before and after prior-shift adjustment ($\tau=0.4$).
6. `fusion_performance_summary.png`: Summary bar charts of AUROC and Balanced Accuracy with standard deviation error bars.
7. `artifacts/per_class_auroc_breakdown.md`: Complete numerical per-class AUROC table across all models and seeds.

---

## 7. Repository Layout

```
Pathology-FoundationModels-Fusion-Agent/
├── conf/
│   ├── agent.yaml           # Search agent hyperparameters & guardrails
│   ├── embeddings.yaml      # Foundation models and concat_mean_max pooling
│   ├── fusion.yaml          # Search spaces for Early, Late, and Intermediate fusion
│   └── data.yaml            # Dataset paths, cohort size (204), and magnification settings
├── src/
│   ├── agent/               # Autonomous search engines (v1–v5)
│   ├── fusion/              # Multimodal fusion models (Late, Early, StreamABMIL)
│   ├── embeddings/          # Model-specific feature extraction logic
│   └── eval/                # Bootstrap CIs, AUROC, and calibration metrics
├── scripts/
│   ├── eval_3seeds_comparison.py       # Master evaluation script (Main results table)
│   ├── run_agent_search_ablation.py    # 5-paradigm agent ablation runner
│   └── generate_evaluation_figures.py  # ROC and other figures generator
├── artifacts/
│   ├── splits/              # Frozen, zero-leakage patient-level splits (seeds 42, 1337, 2026)
│   ├── features/            # Precomputed FP16 slide-level embeddings
│   ├── figures/             # High-resolution PNG figures & confusion matrices
│   └── results/             # Machine-readable JSON logs for agent search trials
├── REPRODUCIBILITY.md       # This reproducibility guide
└── README.md                # Challenge task description and rubric
```

