# Phase 2 Agent v5 Decision Log (Unified Agent - Seed 1337)

**Architecture**: 3-Stage Unified Search (Screen -> Reason -> 15-Fold Stability Gate)  
**Total Trials Run**: 13  
**Frozen Optimal Pooling (Stage 1)**: `MEAN_MAX_STD`  
**Final Champion (Stage 3)**: `T12` with **15-Fold Grand Mean CV AUROC: 0.8327**  
**Risk-Adjusted Stability Utility**: `0.8226` (Grand Mean - 1.0*SE)  
**Champion Family**: `late` | **gamma_meta**: `1.0`  
**Champion Params**: `{"strategy": "uniform", "c": 1.0, "use_metadata": true}`

## Complete Trial Progression across All 3 Stages

| Trial | Stage | Hypothesis | Family | Pooling | Gamma | CV AUROC | Delta vs Champ | SE(Delta) | Decision | Action / Reason |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **T01** | Stage 1 (Pooling Screen) | Spatial mean pooling across all tiles | late | `mean` | 1.0 | **0.8098** | - | - | `EVALUATED` | EVALUATED |
| **T02** | Stage 1 (Pooling Screen) | Element-wise max pooling across all tiles | late | `max` | 1.0 | **0.8227** | - | - | `EVALUATED` | EVALUATED |
| **T03** | Stage 1 (Pooling Screen) | Concatenated [mean; max] multi-resolution pooling | late | `concat_mean_max` | 1.0 | **0.8234** | - | - | `EVALUATED` | EVALUATED |
| **T04** | Stage 1 (Pooling Screen) | Top-20% norm tile mean pooling | late | `top20` | 1.0 | **0.7837** | - | - | `EVALUATED` | EVALUATED |
| **T05** | Stage 1 (Pooling Screen) | Tri-statistic [mean; max; std] (location + extrema + dispersion) | late | `mean_max_std` | 1.0 | **0.8276** | - | - | `STAGE1_WINNER` | STAGE1_WINNER |
| **T06** | Stage 2 (Reasoner) | Initial Baseline on R*=`mean_max_std`: Uniform Late Voting + Metadata | late | `mean_max_std` | 1.0 | **0.8276** | +0.0000 | 0.0000 | `INITIAL_CHAMPION` | Baseline configuration established on frozen representation R* |
| **T07** | Stage 2 (Reasoner) | Late Fusion with Calibrated Temperature Scaling (tau=1.5) on optimal representation R* | late | `mean_max_std` | 1.0 | **0.8226** | -0.0049 | 0.0069 | `REJECT` | Underperformed champion by -0.0049 (SE: 0.0069). |
| **T08** | Stage 2 (Reasoner) | Late Fusion with Moderate Temperature Scaling (tau=1.2) on R* | late | `mean_max_std` | 1.0 | **0.8208** | -0.0067 | 0.0072 | `REJECT` | Underperformed champion by -0.0067 (SE: 0.0072). |
| **T09** | Stage 2 (Reasoner) | Upweight clinical metadata prior (gamma_meta=1.5) on Calibrated Late Fusion | late | `mean_max_std` | 1.5 | **0.8211** | -0.0065 | 0.0088 | `REJECT` | Underperformed champion by -0.0065 (SE: 0.0088). |
| **T10** | Stage 2 (Reasoner) | Strong clinical metadata scaling (gamma_meta=2.0) with tau=1.5 | late | `mean_max_std` | 2.0 | **0.8202** | -0.0074 | 0.0096 | `REJECT` | Underperformed champion by -0.0074 (SE: 0.0096). |
| **T11** | Stage 2 (Reasoner) | Early Concatenation of foundation representations with L2 regularization C=1.0 | early | `mean_max_std` | 1.0 | **0.8085** | -0.0190 | 0.0115 | `REJECT` | Underperformed champion by -0.0190 (SE: 0.0115). |
| **T12** | Stage 3 (Stability Gate) | 15-Fold Stability: Initial Baseline on R*=`mean_max_std`: Uniform Late Voting + Metadata | late | `mean_max_std` | 1.0 | **0.8327** | - | - | `EVAL` | - |
| **T13** | Stage 3 (Stability Gate) | 15-Fold Stability: Late Fusion with Calibrated Temperature Scaling (tau=1.5) on optimal representation R* | late | `mean_max_std` | 1.0 | **0.8312** | - | - | `EVAL` | - |
