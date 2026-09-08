# Build an Agent for Foundation-Model Fusion in Lung Cancer Subtyping

* **Time box:** part-time (target ~15–30 hours of focused work)
* **Deliverable:** a reproducible repository + a short technical report

---

## Objective

Predict the **histologic growth pattern (tumor subtype)** of lung adenocarcinoma from whole-slide images (WSIs), by combining embeddings from **three different pathology foundation models** with patient metadata (age, sex).

> **Note: The fused model must outperform every single-foundation-model baseline** on the held-out test set, under an identical split.

A submission that reports a strong fused number without the single-model baselines to compare against cannot be scored.

---

## Dataset

**Source:** [`kmmuleelab/Lung_Pathology_Image_JPG`](https://huggingface.co/datasets/kmmuleelab/Lung_Pathology_Image_JPG) (public, Hugging Face)
**Reference publication:** [www.nature.com/articles/s41597-026-06906-z](https://www.nature.com/articles/s41597-026-06906-z)

| Property           | Value                                |
| ------------------ | ------------------------------------ |
| Whole-slide images | 408                                  |
| Patients           | 210                                  |
| Slides per patient | 1–5 (median 2)                      |
| Scan magnification | 80×                                 |
| Format             | JPG                                  |
| Disease            | Lung adenocarcinoma                  |
| Label              | Histologic growth pattern, 7 classes |

### Patient metadata — `CLWD.csv`

| Column                     | Type        | Notes                                                                 |
| -------------------------- | ----------- | --------------------------------------------------------------------- |
| `SampleNumber`           | str         | Patient / specimen identifier.**Use this as the grouping key.** |
| `WSI_ID`                 | str         | Slide identifier,`WSI-<n>`; joins to the image files                |
| `Sex`                    | categorical | `Female` (130 patients) / `Male` (79 patients)                    |
| `Age`                    | int         | Range 24–80, mean 55.2, median 55                                    |
| `Pathological_Diagnosis` | categorical | `AIS` (n=46 patients), `MIA` (n=6), `IA` (n=157)                |
| `Tumor Subtype`          | categorical | **Prediction target**, 7 classes                                |

### Class distribution

Patient-level counts (one label per patient), with metadata marginals:

| Subtype        | Patients | Slides | Mean age | % Female |
| -------------- | -------: | -----: | -------: | -------: |
| In situ        |       46 |     80 |     47.5 |      83% |
| Papillary      |       41 |     84 |     60.0 |      68% |
| Acinar         |       34 |     50 |     53.7 |      68% |
| Solid          |       29 |     69 |     56.0 |      45% |
| Micropapillary |       26 |     64 |     56.3 |      42% |
| Lepidic        |       19 |     30 |     57.8 |      63% |
| Cribriform     |       15 |     31 |     63.3 |      40% |

The problem is **7-class, imbalanced (3:1 head-to-tail), and small-n**. With 42 test patients, a single test patient moves balanced accuracy by roughly 2 points. Treat this as a core part of the problem, not a footnote — we care as much about how you quantify uncertainty as about the point estimate.

### Known data quirks

These are real properties of the released metadata. We expect you to find and handle them; how you handle them is part of the assessment.

1. **`SampleNumber` is not perfectly unique.** There are 209 distinct values for 210 patients — ID `8377886` appears twice with `Age` 68 and 69 (slides WSI-35/36 and WSI-103/104). Decide whether to treat these as one patient or two, and state the choice.
2. **Five patients carry conflicting subtype labels across their own slides** (e.g. `8248805`: WSI-401 → Lepidic, WSI-402 → Acinar). This is clinically plausible — adenocarcinoma is heterogeneous and multiple patterns coexist. Define and justify a patient-level label policy (dominant pattern, per-slide labels, or exclusion).

---

## Compute constraints and the reduced protocol

We assume no institutional GPU cluster. **Free Colab or Kaggle GPU is sufficient** for the intended scope.

To keep the task tractable, work on the reduced subset:

- **One slide per patient** → 210 images (state your selection rule; a deterministic rule such as lowest `WSI_ID` per patient is fine)
- **Downsample 80× → 20×** (4× linear downsample) before tiling
- Cache embeddings to disk once; never re-run a foundation model inside a hyperparameter loop

Using the full 408-slide set or higher magnification is welcome but **not** required and earns no extra credit by itself. If you do, keep the reduced-protocol run as the headline result so submissions stay comparable.

---

## Tasks

### Step 1 — Feature extraction with three foundation models

Extract slide-level embeddings using **three distinct pathology foundation models**. Suggested:

- [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) (Mahmood Lab)
- [Virchow2](https://huggingface.co/paige-ai/Virchow2) (Paige AI)
- [Prism2](https://huggingface.co/paige-ai/Prism2) (Paige AI)

Any three are acceptable — see the [THUNDER leaderboard](https://mics-lab.github.io/thunder/leaderboards/) for alternatives. Several of these are gated on Hugging Face; request access early, as approval can take a day or more.

Expected in your write-up:

- Tissue segmentation / background rejection, tiling strategy, tile size and stride at 20×
- Per-model preprocessing (each has its own normalization and input resolution — do not share one transform blindly)
- **Tile → slide aggregation**: mean pooling, attention-based (ABMIL-style), or the model's own slide encoder where one exists (Prism2 provides one; UNI2 and Virchow2 are tile encoders). Justify the choice.
- Embedding dimensionality per model, wall-clock cost, and where artifacts are cached

### Step 2 — Agent system for fusion-strategy search

Build an **agent system** that autonomously explores how to integrate the three embedding sets, rather than hand-tuning one pipeline.

Minimum viable scope:

- **A search space** spanning at least: early fusion (concatenation, with and without per-model L2 normalization or dimensionality reduction), late fusion (probability/logit averaging, weighted vote, stacking), and one intermediate/learned scheme (gated fusion, attention over model streams, or bilinear/cross-attention pooling)
- **An agent loop** that proposes a configuration, evaluates it, reads the result, and conditions its next proposal on what it has seen — logging its reasoning at each step
- **A stopping criterion and a trial budget**, both explicit
- **A decision log**: what was tried, the validation score, and why the next move was chosen

Implementation is open — an LLM-driven loop (tool-calling agent that writes/edits configs), or a structured search agent, or a hybrid. What we assess is whether the exploration is *principled and auditable*, not whether it is fashionable. If you use an LLM, keep API keys in the environment and out of the repo, and make the system degrade gracefully to a deterministic search when no key is present.

**Guardrail — the failure mode we look for:** with 21 validation patients, an agent running 200 unguided trials will overfit the validation set and the gain will not survive to test. Show that you thought about this (nested/repeated CV, a trial budget, selection-noise estimates, or a validation-variance-aware acceptance rule).

Note:`Age` and `Sex` must enter the pipeline. Both are required, and both are weak-but-real signals (see §2.2 — In situ patients are ~10 years younger and 83% female).

---

## Evaluation protocol

### Data Splits

| Split      | Fraction | Patients (of 210) |
| ---------- | -------: | ----------------: |
| Train      |      70% |              ~146 |
| Validation |      10% |               ~21 |
| Test       |      20% |               ~42 |

Mandatory:

- **Split at the patient level, by `SampleNumber`.** No patient may contribute slides to more than one split. A random slide-level split is the single most common disqualifying error on this task.
- **Stratify by subtype.** With Cribriform at 15 patients, an unstratified split can leave a class absent from validation.
- **Fix and commit the split.** Save the patient-ID → split assignment as a versioned artifact so every model and every agent trial is scored on exactly the same partition.
- **Touch the test set once.** All selection — fusion strategy, hyperparameters, aggregation, agent trials — uses train/validation only.
- **Seed everything** and report the seed.

### Metrics

Primary:

- **AUROC**, macro-averaged one-vs-rest (report per-class AUROC as well)
- **Balanced accuracy**

Also report:

- Confusion matrix on the test set
- **Uncertainty on the primary metrics** — bootstrap CIs over test patients, and/or mean ± sd across ≥3 seeds or repeated splits. A number without a spread is not interpretable at n=42.

### 5.3 Required comparison table

| Configuration                               | Macro AUROC | Balanced acc. |
| ------------------------------------------- | ----------- | ------------- |
| Metadata only (age + sex)                   |             |               |
| Foundation model A alone                    |             |               |
| Foundation model B alone                    |             |               |
| Foundation model C alone                    |             |               |
| Best single model + metadata                |             |               |
| **Fused (agent-selected) + metadata** |             |               |

Keep the downstream classifier family and tuning budget identical across rows, so the comparison isolates the fusion effect rather than the tuning effort.

---

## Deliverables

1. **Repository** — runnable end-to-end, with pinned dependencies (`uv`/`pip` lockfile or `requirements.txt`) and a documented entry point per stage. Suggested layout:

   ```
   conf/                 # configuration (Hydra or equivalent)
   src/
     data/               # subset selection, tiling, splits
     embeddings/         # one module per foundation model
     fusion/             # fusion strategies, registered by name
     agent/              # search loop, decision logging
     eval/               # metrics, bootstrap CIs, plots
   scripts/              # stage entry points
   notebooks/            # Colab/Kaggle runnable versions
   artifacts/            # splits, cached embeddings (gitignored), logs
   report.md
   ```
2. **Technical report** (≤ 6 pages, Markdown or PDF) covering: pipeline design, the three models and why; the agent's search space and decision policy; the results table from §5.3 with uncertainty; how the data quirks in §2.3 were handled; failure analysis of the confusion matrix; and what you would do with 10× the compute.

Note: AI tools (codex, claude code) are encouraged to be used for coding, but please don't directly use AI to write the technical report.

---

## Assessment rubric

| Weight | Dimension                           | What earns credit                                                                                                                    |
| -----: | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
|    15% | **Experimental rigor**        | Patient-level stratified splits, single test-set use, seeded runs, honest uncertainty, identical budgets across compared rows        |
|    25% | **Agent system design**       | A real search space, a genuine feedback loop, an auditable decision log, and explicit defenses against validation overfitting        |
|    20% | **Foundation-model handling** | Correct per-model preprocessing, sound tile→slide aggregation, sensible tiling at 20×, efficient caching                           |
|    20% | **Results and analysis**      | Does fusion beat every single model, and is the claim supported? Quality of failure analysis matters more than the size of the delta |
|    20% | **Engineering quality**       | Reproducibility, config-driven code, typed and documented modules, clean modular layout, no secrets in the repo                      |
