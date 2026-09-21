# Phase 2 Agent v5 Decision Log (Unified Agent - Seed 2026)

**Architecture**: 3-Stage Unified Search (Screen -> Reason -> 15-Fold Stability Gate)  
**Total Trials Run**: 13  
**Frozen Optimal Pooling (Stage 1)**: `MAX`  
**Final Champion (Stage 3)**: `T12` with **15-Fold Grand Mean CV AUROC: 0.8350**  
**Risk-Adjusted Stability Utility**: `0.8260` (Grand Mean - 1.0*SE)  
**Champion Family**: `late` | **gamma_meta**: `1.0`  
**Champion Params**: `{"strategy": "uniform", "c": 1.0, "use_metadata": true}`

## Complete Trial Progression across All 3 Stages

| Trial | Stage | Hypothesis | Family | Pooling | Gamma | CV AUROC | Delta vs Champ | SE(Delta) | Decision | Action / Reason |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **T01** | Stage 1 (Pooling Screen) | Spatial mean pooling across all tiles | late | `mean` | 1.0 | **0.8256** | - | - | `EVALUATED` | EVALUATED |
| **T02** | Stage 1 (Pooling Screen) | Element-wise max pooling across all tiles | late | `max` | 1.0 | **0.8450** | - | - | `STAGE1_WINNER` | STAGE1_WINNER |
| **T03** | Stage 1 (Pooling Screen) | Concatenated [mean; max] multi-resolution pooling | late | `concat_mean_max` | 1.0 | **0.8409** | - | - | `EVALUATED` | EVALUATED |
| **T04** | Stage 1 (Pooling Screen) | Top-20% norm tile mean pooling | late | `top20` | 1.0 | **0.8024** | - | - | `EVALUATED` | EVALUATED |
| **T05** | Stage 1 (Pooling Screen) | Tri-statistic [mean; max; std] (location + extrema + dispersion) | late | `mean_max_std` | 1.0 | **0.8426** | - | - | `EVALUATED` | EVALUATED |
| **T06** | Stage 2 (Reasoner) | Initial Baseline on R*=`max`: Uniform Late Voting + Metadata | late | `max` | 1.0 | **0.8450** | +0.0000 | 0.0000 | `INITIAL_CHAMPION` | Baseline configuration established on frozen representation R* |
| **T07** | Stage 2 (Reasoner) | Late Fusion with Calibrated Temperature Scaling (tau=1.5) on optimal representation R* | late | `max` | 1.0 | **0.8439** | -0.0011 | 0.0031 | `REJECT` | Underperformed champion by -0.0011 (SE: 0.0031). |
| **T08** | Stage 2 (Reasoner) | Late Fusion with Moderate Temperature Scaling (tau=1.2) on R* | late | `max` | 1.0 | **0.8445** | -0.0005 | 0.0026 | `REJECT` | Underperformed champion by -0.0005 (SE: 0.0026). |
| **T09** | Stage 2 (Reasoner) | Upweight clinical metadata prior (gamma_meta=1.5) on Calibrated Late Fusion | late | `max` | 1.5 | **0.8449** | -0.0001 | 0.0047 | `REJECT` | Underperformed champion by -0.0001 (SE: 0.0047). |
| **T10** | Stage 2 (Reasoner) | Strong clinical metadata scaling (gamma_meta=2.0) with tau=1.5 | late | `max` | 2.0 | **0.8418** | -0.0032 | 0.0061 | `REJECT` | Underperformed champion by -0.0032 (SE: 0.0061). |
| **T11** | Stage 2 (Reasoner) | Early Concatenation of foundation representations with L2 regularization C=1.0 | early | `max` | 1.0 | **0.8232** | -0.0217 | 0.0048 | `REJECT` | Underperformed champion by -0.0217 (SE: 0.0048). |
| **T12** | Stage 3 (Stability Gate) | 15-Fold Stability: Initial Baseline on R*=`max`: Uniform Late Voting + Metadata | late | `max` | 1.0 | **0.8350** | - | - | `EVAL` | - |
| **T13** | Stage 3 (Stability Gate) | 15-Fold Stability: Upweight clinical metadata prior (gamma_meta=1.5) on Calibrated Late Fusion | late | `max` | 1.5 | **0.8338** | - | - | `EVAL` | - |
