# Phase 2 Agent v2 Decision Log (Seed 2026)

**Agent Strategy**: Principled Exploitation with Paired SE Guardrail  
**Total Trials Run**: 18  
**Final Champion**: `T13` with **5-fold CV Macro AUROC: 0.8409**  
**Champion Family**: `late`  
**Champion Params**: `{"strategy": "uniform", "c": 1.0, "use_metadata": true}`  
**Champion Pooling**: `concat` | **gamma_meta**: `1.0`  
**Guardrail Policy**: Adopt if $\bar{\Delta} \ge 1.0 \times \text{SE}_{\Delta}$ (patience = 5)

## Trial Progression & SE Guardrail Decisions

| Trial | Stage | Hypothesis | Family | CV AUROC | Delta vs Champ | SE(Delta) | Threshold (k_se×SE) | Decision | Action/Reason |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **T01** | Exploration | Raw 5632-d concatenation + L2 LR | early | **0.8186** | +0.0000 | 0.0000 | 0.0000 | `INITIAL_CHAMPION` | First baseline trial established as champion benchmark |
| **T02** | Exploration | L2-normalized stream concatenation | early | **0.8186** | +0.0000 | 0.0000 | 0.0000 | `ADOPTED` | Exceeded champion by delta=+0.0000 >= 1.0*SE (0.0000, SE=0.0000) |
| **T03** | Exploration | Train-set PCA (256-d, 95% variance) | early | **0.8186** | +0.0000 | 0.0000 | 0.0000 | `ADOPTED` | Exceeded champion by delta=+0.0000 >= 1.0*SE (0.0000, SE=0.0000) |
| **T04** | Exploration | Ablation: Top Early without metadata | early | **0.8187** | +0.0001 | 0.0017 | 0.0017 | `REJECTED` | Failed guardrail: delta=+0.0001 < 1.0*SE (0.0017, SE=0.0017) |
| **T05** | Exploration | Uniform probability soft voting (1/3 each) | late | **0.8256** | +0.0069 | 0.0034 | 0.0034 | `ADOPTED` | Exceeded champion by delta=+0.0069 >= 1.0*SE (0.0034, SE=0.0034) |
| **T06** | Exploration | Temperature-scaled soft voting (tau=1.5) | late | **0.8258** | +0.0002 | 0.0044 | 0.0044 | `REJECTED` | Failed guardrail: delta=+0.0002 < 1.0*SE (0.0044, SE=0.0044) |
| **T07** | Exploration | Validation simplex-optimized weights | late | **0.8208** | -0.0047 | 0.0062 | 0.0062 | `REJECTED` | Failed guardrail: delta=-0.0047 < 1.0*SE (0.0062, SE=0.0062) |
| **T08** | Exploration | OOF stacking meta-classifier | late | **0.8156** | -0.0100 | 0.0143 | 0.0143 | `REJECTED` | Failed guardrail: delta=-0.0100 < 1.0*SE (0.0143, SE=0.0143) |
| **T09** | Exploration | Stream-level Gated Fusion (dropout=0.2) | intermediate | **0.8001** | -0.0254 | 0.0097 | 0.0097 | `REJECTED` | Failed guardrail: delta=-0.0254 < 1.0*SE (0.0097, SE=0.0097) |
| **T10** | Exploration | Stream-ABMIL multi-stream attention | intermediate | **0.8001** | -0.0254 | 0.0097 | 0.0097 | `REJECTED` | Failed guardrail: delta=-0.0254 < 1.0*SE (0.0097, SE=0.0097) |
| **T11** | Exploration | PCA (128-d per stream) -> Gated Fusion | intermediate | **0.8001** | -0.0254 | 0.0097 | 0.0097 | `REJECTED` | Failed guardrail: delta=-0.0254 < 1.0*SE (0.0097, SE=0.0097) |
| **T12** | Exploration | Ablation: Top Intermediate without metadata | intermediate | **0.8124** | -0.0132 | 0.0136 | 0.0136 | `REJECTED` | Failed guardrail: delta=-0.0132 < 1.0*SE (0.0136, SE=0.0136) |
| **T13** | Exploitation | Concatenated [mean; max] multi-resolution representation | late | **0.8409** | +0.0154 | 0.0049 | 0.0049 | `ADOPTED` | Exceeded champion by delta=+0.0154 >= 1.0*SE (0.0049, SE=0.0049) |
| **T14** | Exploitation | Temperature-scaled soft voting (tau=1.5) on concat | late | **0.8391** | -0.0018 | 0.0021 | 0.0021 | `REJECTED` | Failed guardrail: delta=-0.0018 < 1.0*SE (0.0021, SE=0.0021) |
| **T15** | Exploitation | Calibrated temperature soft voting (tau=1.2) on concat | late | **0.8411** | +0.0002 | 0.0024 | 0.0024 | `REJECTED` | Failed guardrail: delta=+0.0002 < 1.0*SE (0.0024, SE=0.0024) |
| **T16** | Exploitation | Conservative temperature soft voting (tau=2.0) on concat | late | **0.8413** | +0.0003 | 0.0031 | 0.0031 | `REJECTED` | Failed guardrail: delta=+0.0003 < 1.0*SE (0.0031, SE=0.0031) |
| **T17** | Exploitation | Regularization shrinkage C=0.5 | late | **0.8408** | -0.0001 | 0.0010 | 0.0010 | `REJECTED` | Failed guardrail: delta=-0.0001 < 1.0*SE (0.0010, SE=0.0010) |
| **T18** | Exploitation | Regularization expansion C=2.0 | late | **0.8413** | +0.0004 | 0.0014 | 0.0014 | `REJECTED` | Failed guardrail: delta=+0.0004 < 1.0*SE (0.0014, SE=0.0014) |
