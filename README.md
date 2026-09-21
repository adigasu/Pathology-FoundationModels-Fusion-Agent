# Pathology Foundation Models Fusion Agent

This is a guide for reproducing the entire data curation, tiling, feature extraction, autonomous agent fusion search, and evaluation benchmark across all experimental seeds (`42`, `1337`, `2026`).
(Note: the original `README.md` file is moved to `artifacts/README.md`.)

---

> [!IMPORTANT]
> **CPU Thread Contention**: On multi-core servers, OpenMP / MKL thread pools can cause spin-lock overhead on small matrix operations. Always set thread limits when running evaluation scripts:
> ```bash
> export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
> ```

---

## 1. Environment Setup

### 1.1 Hardware Requirements
- **OS**: Linux (tested on Ubuntu 22.04 LTS / x86_64)
- **GPU**: NVIDIA GPU with >= 12 GB VRAM (e.g., RTX 3090, A5000, A100, V100, or Colab T4/A100)
- **Disk Space**: 386 GB for original WSI JPGs, ~5 GB for cached FP16 tile and slide embeddings

### 1.2 Installation Options

#### Option A: Quick Install via `requirements.txt` (Standard `pip`)
```bash
# 1. Create a virtual environment with Python 3.10+
python3 -m venv .env_path_agent
source .env_path_agent/bin/activate

# 2. Install PyTorch with your system's CUDA version (e.g., CUDA 12.4 was used for this task)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Verify GPU access
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0))"
```

#### Option B: Fast Install via `uv` (Recommended)
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

### 2.1 Quickstart via Shared Precomputed Artifacts (`artifacts.tar.gz`)

For rapid reproduction without downloading raw WSIs and running feature extraction, the precomputed artifact bundle `artifacts.tar.gz` is shared. It contains precomputed slide embeddings (`artifacts/features/`), curated patient cohorts (`artifacts/curated_patients.csv`), frozen zero-leakage splits for all 3 seeds (`artifacts/splits/`), and other generated artifacts.

To unpack `artifacts.tar.gz`:

```bash
cd artifacts/
# Unpack the tarball
tar -xzvf /path/to/artifacts.tar.gz
```

*Verification*:
Ensure that `artifacts/features/` contains `uni2_slide_*.pt`, `virchow2_slide_*.pt`, `gigapath_slide_*.pt`, `prism2_diag_slide.pt` and `artifacts/curated_patients.csv`. With this, "3. Phase 1 (Data Audit, Tiling & Feature Extraction)" can be skipped and "4. Phase 2 (Autonomous Agent Fusion Search)" can be run directly.

### 2.2 Canonical Paths & Raw WSI Configuration

The pipeline resolves the raw image directory via the `LUNG_DATA_DIR` environment variable, falling back to `./data/Lung_Pathology_Image_JPG` or `~/Lung_Pathology_Image_JPG`:
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
- **Tiling QC Overlays**: `artifacts/tiling_qc/` (visualization of tiling)
- **Evaluation Figures**: `artifacts/figures/` (visualization of results)

---

## 3. Data Audit, Tiling & Foundation Feature Extraction (Phase 1)

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

### Step 1.5: Prism2 Multimodal Feature Extraction (for Quad-model fusion only)
Extracts Perceiver Resampler and Phi-3 VLM diagnostic embeddings from cached Virchow2 tile representations:
```bash
# Extract Prism2 Base (2560-d), Latents Mean (2560-d), and VLM Diagnostic (3072-d):
.env_path_agent/bin/python scripts/extract_prism2_embeddings.py
```
*Outputs*: `artifacts/features/prism2_diag_slide.pt`, `artifacts/features/prism2_base_slide.pt`, `artifacts/features/prism2_latents_mean_slide.pt`

---

## 4. Autonomous Agent Fusion Search (Phase 2)

The search engine supports 5 generations of autonomous agents (`v1` to `v5`):
- **`v1` (Sequential Agent)**: Fixed sequential exploration (T01–T12) across Early/Late/Intermediate followed by greedy parameter exploitation (T13–T25) under paired 1.0-SE guardrails.
- **`v2` (Exploitation Agent)**: Representation-capacity-first search prioritizing multi-resolution concat `[mean; max]` pooling before parameter exploitation.
- **`v3` (Hierarchical Agent)**: Decoupled 3-stage search (Stage 1: pooling isolation -> Stage 2: fusion topology exploration -> Stage 3: 15-fold cross-seed stability utility selection).
- **`v4` (Autonomous Agent - Default Champion)**: Hypothesis-driven search conditioned on complete trial history, domain pathology insights, and paired 1.0-SE guardrail early stopping.
- **`v5` (Unified Agent)**: Unified statistical pooling screen -> closed-loop autonomous reasoning -> 15-fold stability gating.

### LLM API Configuration (Agent v4 / LLM)
To enable live frontier LLM reasoning for Agent v4 hypothesis generation, export your Gemini API key:
```bash
export GEMINI_API_KEY="AIzaSy..."
# Optional model override (defaults to gemini-1.5-flash or gemini-2.0-flash):
export GEMINI_MODEL="gemini-2.0-flash"
```
> [!NOTE]
> **Zero-Dependency Fallback**: If no API key is exported, Agent v4 automatically routes to its deterministic offline heuristic reasoning engine (`provider = "fallback_heuristic"`), executing complete hypothesis generation and 1.0-SE guardrail pruning locally with zero external API calls or internet dependencies.

### Step 4.1: Running Agent Search for Tri-Model and Quad-Model Fusion

The following commands run the autonomous search for any agent type (`v1` to `v5`, e.g., champion `v4`) for either **Tri-Model Vision Fusion** (`UNI2` + `Virchow2` + `Prov-GigaPath`), **Multimodal Quad-Model Fusion** (+ `Prism2 VLM`), or **Both**:

#### A. Tri-Model Vision Fusion Search:
```bash
# Run 25-trial autonomous agent search (e.g. v4) on Tri-Model vision streams (Seed 42):
.env_path_agent/bin/python scripts/05_run_phase2_fusion_agent.py --agent-version v4 --modality tri --seed 42
```

#### B. Quad-Model (+ Prism2 VLM) Fusion Search:
```bash
# Run agent search evaluating both Tri-Model and Quad-Model (+ Prism2 VLM):
.env_path_agent/bin/python scripts/05_run_phase2_fusion_agent.py --agent-version v4 --modality quad --seed 42
```

#### C. Full Comparison (Both tri-model and quad-model):
```bash
# Evaluate Tri-Model and Quad-Model side-by-side with 1,000 bootstrap CIs (Seed 42):
.env_path_agent/bin/python scripts/05_run_phase2_fusion_agent.py --agent-version v4 --modality both --seed 42
```

*Outputs*:
- `artifacts/results/agent_decision_log_seed_42.json`: Auditable decision log conditioning proposals on fold performance.
- `artifacts/results/agent_search_summary_seed_42.md`: Step-by-step hypothesis, reasoning, and adoption summary.
- `artifacts/results/phase2_comparison_table_seed_42.md`: Baseline vs. Tri-Model & Quad-Model champion comparison table with 1,000-sample bootstrap 95% CIs.

### Step 4.2: Multi-Seed Repeatability Verification (Seeds 42, 1337, 2026)
Executes independent end-to-end searches and evaluations across all 3 seeds with zero cross-seed contamination:
```bash
# Run full 3-seed evaluation for any agent version (v1 to v5, say v4) across both modalities:
.env_path_agent/bin/python scripts/05_run_phase2_fusion_agent.py --agent-version v4 --modality both --all-seeds
```
*Outputs*:
- `artifacts/results/phase2_multi_seed_comparison.md`: Multi-seed evaluation reporting Mean ± SD.
- `artifacts/results/phase2_multi_seed_comparison.json`: Machine-readable results.

Convenience shell wrappers (supports positional or flagged arguments: `--seed`, `--agent`, `--output-dir`):
```bash
# Single-seed execution (default: seed 42, agent v4, output: artifacts/results):
./scripts/run_phase2.sh
# Single-seed execution with options (e.g. agent v4, seed 1337):
./scripts/run_phase2.sh --seed 1337 --agent v4 --output-dir artifacts/results/seed1337_v4
# Example with short options (e.g. agent v4):
./scripts/run_phase2.sh -s 1337 -a v4 -o artifacts/results/seed1337_v4

# Full multi-seed execution across seeds 42, 1337, 2026 (default: agent v4, output: artifacts/results):
./scripts/run_all_phase2.sh
# Full multi-seed execution with options (e.g. agent v4, output: artifacts/results/all_seeds_v4):
./scripts/run_all_phase2.sh --agent v4 --output-dir artifacts/results/all_seeds_v4
# Example with short options (e.g. agent v4):
./scripts/run_all_phase2.sh -a v4 -o artifacts/results/all_seeds_v4
```

### Step 4.3: Multimodal Vision-Language Representation Benchmark (Optional)
Evaluates Triad vs. Quad-Model Fusion across all 3 seeds:
```bash
.env_path_agent/bin/python scripts/compare_prism2_representations.py --mode both
```
*Output*: `artifacts/results/prism2_benchmark_3seeds.json`

---

## 5. Main Benchmark Reproduction

Runs the master evaluation across all 3 seeds, reproducing the main results tables:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.env_path_agent/bin/python scripts/eval_3seeds_comparison.py --mode both
```


### 5.1: Ablation Study of Agent Search Architectures

To evaluate the comparative impact of different autonomous search paradigms, run the 5-paradigm agent architecture ablation study:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.env_path_agent/bin/python scripts/run_agent_search_ablation.py --seeds 42 1337 2026
```

### Step 5.2: Generate All Evaluation Figures & Failure Analysis Tables
Renders publication-quality figures and per-class statistical tables (reads directly from `artifacts/results/` by default, or an explicit results folder):
```bash
.env_path_agent/bin/python scripts/generate_evaluation_figures.py
# Or specify custom results and output directories:
# .env_path_agent/bin/python scripts/generate_evaluation_figures.py --results-dir artifacts/results --output-dir artifacts/figures
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

## 6. Repository Layout

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
│   ├── 05_run_phase2_fusion_agent.py              # Autonomous agent search & multi-seed verification runner
│   ├── eval_3seeds_comparison.py                  # Master evaluation script (Main results table)
│   ├── run_agent_search_ablation.py               # 5-paradigm agent ablation runner
│   └── generate_evaluation_figures.py             # ROC, confusion matrix, and other analysis generator
├── artifacts/
│   ├── splits/              # Frozen, zero-leakage patient-level splits (seeds 42, 1337, 2026)
│   ├── features/            # Precomputed FP16 slide-level embeddings
│   ├── figures/             # High-resolution PNG figures & confusion matrices
│   └── results/             # Machine-readable JSON logs for agent search trials
├── REPRODUCIBILITY.md       # This reproducibility guide
└── README.md                # Challenge task description and rubric
```

