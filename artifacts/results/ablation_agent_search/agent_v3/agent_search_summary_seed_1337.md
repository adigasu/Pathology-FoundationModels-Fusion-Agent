# Phase 2 Agent v3 Decision Log (Hierarchical Stability Agent)

**Architecture**: 3-Stage Hierarchical Stability Search  
**Total Trials Run**: 16  
**Winning Pooling (Stage 1)**: `CONCAT`  
**Final Champion (Stage 3)**: `T15` with **15-Fold Grand Mean CV AUROC: 0.8305**  
**Stability Utility**: `0.8206` (Mean - 1.0*SE)  
**Champion Family**: `late`  
**Champion Params**: `{"strategy": "uniform", "c": 1.0, "use_metadata": true}`

## Trial Progression across Stages

| Trial | Stage | Strategy / Hypothesis | Family | Pooling | CV AUROC | Std / SE | Decision / Status |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **T01** | Stage 1 (Pooling) | Spatial mean pooling across all tiles | late | `mean` | 0.8098 | ±0.0558 | Evaluated |
| **T02** | Stage 1 (Pooling) | Element-wise max pooling across all tiles | late | `max` | 0.8227 | ±0.0277 | Evaluated |
| **T03** | Stage 1 (Pooling) | Top-5% norm tile mean pooling | late | `top5` | 0.7592 | ±0.0448 | Evaluated |
| **T04** | Stage 1 (Pooling) | Top-10% norm tile mean pooling | late | `top10` | 0.7739 | ±0.0370 | Evaluated |
| **T05** | Stage 1 (Pooling) | Top-20% norm tile mean pooling | late | `top20` | 0.7837 | ±0.0393 | Evaluated |
| **T06** | Stage 1 (Pooling) | Concatenated [mean; max] multi-resolution pooling | late | `concat` | 0.8234 | ±0.0360 | Winner |
| **T07** | Stage 2 (Fusion) | Early concatenation of raw stream vectors | early | `concat` | 0.8081 | ±0.0436 | Evaluated |
| **T08** | Stage 2 (Fusion) | Per-stream L2-normalization before concatenation | early | `concat` | 0.8081 | ±0.0436 | Evaluated |
| **T09** | Stage 2 (Fusion) | Train-set PCA (256-d, 95% variance) dimensionality reduction | early | `concat` | 0.8152 | ±0.0306 | Evaluated |
| **T10** | Stage 2 (Fusion) | Uniform probability soft voting ensemble | late | `concat` | 0.8234 | ±0.0360 | Evaluated |
| **T11** | Stage 2 (Fusion) | Temperature-scaled calibrated soft voting | late | `concat` | 0.8195 | ±0.0388 | Evaluated |
| **T12** | Stage 2 (Fusion) | Calibrated temperature soft voting (tau=1.2) | late | `concat` | 0.8196 | ±0.0356 | Evaluated |
| **T13** | Stage 2 (Fusion) | Validation simplex-constrained convex weighting | late | `concat` | 0.7952 | ±0.0348 | Evaluated |
| **T14** | Stage 2 (Fusion) | Stream-level Gated Fusion neural network | intermediate | `concat` | 0.7764 | ±0.0108 | Evaluated |
| **T15** | Stage 3 (Stability) | Multi-seed stability: Uniform probability soft voting ensemble | late | `concat` | **0.8305** | ±0.0381 (SE: 0.0098) | Champion (Utility: 0.8206) |
| **T16** | Stage 3 (Stability) | Multi-seed stability: Calibrated temperature soft voting (tau=1.2) | late | `concat` | **0.8300** | ±0.0373 (SE: 0.0096) | Runner-up |
