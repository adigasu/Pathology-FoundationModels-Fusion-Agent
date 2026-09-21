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
| **T01** | Stage 1 (Pooling) | Spatial mean pooling across all tiles | late | `mean` | 0.8028 | ±0.0495 | Evaluated |
| **T02** | Stage 1 (Pooling) | Element-wise max pooling across all tiles | late | `max` | 0.8373 | ±0.0480 | Winner |
| **T03** | Stage 1 (Pooling) | Top-5% norm tile mean pooling | late | `top5` | 0.7641 | ±0.0780 | Evaluated |
| **T04** | Stage 1 (Pooling) | Top-10% norm tile mean pooling | late | `top10` | 0.7701 | ±0.0751 | Evaluated |
| **T05** | Stage 1 (Pooling) | Top-20% norm tile mean pooling | late | `top20` | 0.7774 | ±0.0643 | Evaluated |
| **T06** | Stage 1 (Pooling) | Concatenated [mean; max] multi-resolution pooling | late | `concat` | 0.8270 | ±0.0546 | Evaluated |
| **T07** | Stage 2 (Fusion) | Early concatenation of raw stream vectors | early | `max` | 0.8225 | ±0.0365 | Evaluated |
| **T08** | Stage 2 (Fusion) | Per-stream L2-normalization before concatenation | early | `max` | 0.8225 | ±0.0365 | Evaluated |
| **T09** | Stage 2 (Fusion) | Train-set PCA (256-d, 95% variance) dimensionality reduction | early | `max` | 0.8146 | ±0.0302 | Evaluated |
| **T10** | Stage 2 (Fusion) | Uniform probability soft voting ensemble | late | `max` | 0.8373 | ±0.0480 | Evaluated |
| **T11** | Stage 2 (Fusion) | Temperature-scaled calibrated soft voting | late | `max` | 0.8336 | ±0.0437 | Evaluated |
| **T12** | Stage 2 (Fusion) | Calibrated temperature soft voting (tau=1.2) | late | `max` | 0.8338 | ±0.0404 | Evaluated |
| **T13** | Stage 2 (Fusion) | Validation simplex-constrained convex weighting | late | `max` | 0.8001 | ±0.0376 | Evaluated |
| **T14** | Stage 2 (Fusion) | Stream-level Gated Fusion neural network | intermediate | `max` | 0.7557 | ±0.0971 | Evaluated |
| **T15** | Stage 3 (Stability) | Multi-seed stability: Uniform probability soft voting ensemble | late | `max` | **0.8350** | ±0.0351 (SE: 0.0091) | Runner-up |
| **T16** | Stage 3 (Stability) | Multi-seed stability: Calibrated temperature soft voting (tau=1.2) | late | `max` | **0.8346** | ±0.0324 (SE: 0.0084) | Champion (Utility: 0.8263) |
