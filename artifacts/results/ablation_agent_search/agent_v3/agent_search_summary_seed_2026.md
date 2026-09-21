# Phase 2 Agent v3 Decision Log (Hierarchical Stability Agent)

**Architecture**: 3-Stage Hierarchical Stability Search  
**Total Trials Run**: 16  
**Winning Pooling (Stage 1)**: `MAX`  
**Final Champion (Stage 3)**: `T16` with **15-Fold Grand Mean CV AUROC: 0.8346**  
**Stability Utility**: `0.8263` (Mean - 1.0*SE)  
**Champion Family**: `late`  
**Champion Params**: `{"strategy": "temperature", "tau": 1.2, "c": 1.0, "use_metadata": true}`

## Trial Progression across Stages

| Trial | Stage | Strategy / Hypothesis | Family | Pooling | CV AUROC | Std / SE | Decision / Status |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **T01** | Stage 1 (Pooling) | Spatial mean pooling across all tiles | late | `mean` | 0.8256 | ±0.0343 | Evaluated |
| **T02** | Stage 1 (Pooling) | Element-wise max pooling across all tiles | late | `max` | 0.8450 | ±0.0302 | Winner |
| **T03** | Stage 1 (Pooling) | Top-5% norm tile mean pooling | late | `top5` | 0.7820 | ±0.0231 | Evaluated |
| **T04** | Stage 1 (Pooling) | Top-10% norm tile mean pooling | late | `top10` | 0.7958 | ±0.0150 | Evaluated |
| **T05** | Stage 1 (Pooling) | Top-20% norm tile mean pooling | late | `top20` | 0.8024 | ±0.0204 | Evaluated |
| **T06** | Stage 1 (Pooling) | Concatenated [mean; max] multi-resolution pooling | late | `concat` | 0.8409 | ±0.0243 | Evaluated |
| **T07** | Stage 2 (Fusion) | Early concatenation of raw stream vectors | early | `max` | 0.8232 | ±0.0263 | Evaluated |
| **T08** | Stage 2 (Fusion) | Per-stream L2-normalization before concatenation | early | `max` | 0.8232 | ±0.0263 | Evaluated |
| **T09** | Stage 2 (Fusion) | Train-set PCA (256-d, 95% variance) dimensionality reduction | early | `max` | 0.8426 | ±0.0309 | Evaluated |
| **T10** | Stage 2 (Fusion) | Uniform probability soft voting ensemble | late | `max` | 0.8450 | ±0.0302 | Evaluated |
| **T11** | Stage 2 (Fusion) | Temperature-scaled calibrated soft voting | late | `max` | 0.8439 | ±0.0253 | Evaluated |
| **T12** | Stage 2 (Fusion) | Calibrated temperature soft voting (tau=1.2) | late | `max` | 0.8445 | ±0.0284 | Evaluated |
| **T13** | Stage 2 (Fusion) | Validation simplex-constrained convex weighting | late | `max` | 0.8067 | ±0.0323 | Evaluated |
| **T14** | Stage 2 (Fusion) | Stream-level Gated Fusion neural network | intermediate | `max` | 0.7519 | ±0.0274 | Evaluated |
| **T15** | Stage 3 (Stability) | Multi-seed stability: Uniform probability soft voting ensemble | late | `max` | **0.8350** | ±0.0351 (SE: 0.0091) | Runner-up |
| **T16** | Stage 3 (Stability) | Multi-seed stability: Calibrated temperature soft voting (tau=1.2) | late | `max` | **0.8346** | ±0.0324 (SE: 0.0084) | Champion (Utility: 0.8263) |
