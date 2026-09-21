# Phase 2 Agent v2 Decision Log (Seed 1337)

**Agent Strategy**: Principled Exploitation with Paired SE Guardrail  
**Total Trials Run**: 19  
**Final Champion**: `T14` with **5-fold CV Macro AUROC: 0.8195**  
**Champion Family**: `late`  
**Champion Params**: `{"strategy": "temperature", "tau": 1.5, "c": 1.0, "use_metadata": true}`  
**Champion Pooling**: `concat` | **gamma_meta**: `1.0`  
**Guardrail Policy**: Adopt if $\bar{\Delta} \ge 1.0 \times \text{SE}_{\Delta}$ (patience = 5)

## Trial Progression & SE Guardrail Decisions

| Trial | Stage | Hypothesis | Family | CV AUROC | Delta vs Champ | SE(Delta) | Threshold (k_se×SE) | Decision | Action/Reason |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **T01** | Exploration | Raw 5632-d concatenation + L2 LR | early | **0.7998** | +0.0000 | 0.0000 | 0.0000 | `INITIAL_CHAMPION` | First baseline trial established as champion benchmark |
| **T02** | Exploration | L2-normalized stream concatenation | early | **0.7998** | +0.0000 | 0.0000 | 0.0000 | `ADOPTED` | Exceeded champion by delta=+0.0000 >= 1.0*SE (0.0000, SE=0.0000) |
| **T03** | Exploration | Train-set PCA (256-d, 95% variance) | early | **0.7998** | +0.0000 | 0.0000 | 0.0000 | `ADOPTED` | Exceeded champion by delta=+0.0000 >= 1.0*SE (0.0000, SE=0.0000) |
| **T04** | Exploration | Ablation: Top Early without metadata | early | **0.7999** | +0.0001 | 0.0012 | 0.0012 | `REJECTED` | Failed guardrail: delta=+0.0001 < 1.0*SE (0.0012, SE=0.0012) |
| **T05** | Exploration | Uniform probability soft voting (1/3 each) | late | **0.8098** | +0.0100 | 0.0114 | 0.0114 | `REJECTED` | Failed guardrail: delta=+0.0100 < 1.0*SE (0.0114, SE=0.0114) |
| **T06** | Exploration | Temperature-scaled soft voting (tau=1.5) | late | **0.8062** | +0.0064 | 0.0054 | 0.0054 | `ADOPTED` | Exceeded champion by delta=+0.0064 >= 1.0*SE (0.0054, SE=0.0054) |
| **T07** | Exploration | Validation simplex-optimized weights | late | **0.7872** | -0.0190 | 0.0113 | 0.0113 | `REJECTED` | Failed guardrail: delta=-0.0190 < 1.0*SE (0.0113, SE=0.0113) |
| **T08** | Exploration | OOF stacking meta-classifier | late | **0.7686** | -0.0376 | 0.0203 | 0.0203 | `REJECTED` | Failed guardrail: delta=-0.0376 < 1.0*SE (0.0203, SE=0.0203) |
| **T09** | Exploration | Stream-level Gated Fusion (dropout=0.2) | intermediate | **0.7774** | -0.0288 | 0.0126 | 0.0126 | `REJECTED` | Failed guardrail: delta=-0.0288 < 1.0*SE (0.0126, SE=0.0126) |
| **T10** | Exploration | Stream-ABMIL multi-stream attention | intermediate | **0.7774** | -0.0288 | 0.0126 | 0.0126 | `REJECTED` | Failed guardrail: delta=-0.0288 < 1.0*SE (0.0126, SE=0.0126) |
| **T11** | Exploration | PCA (128-d per stream) -> Gated Fusion | intermediate | **0.7774** | -0.0288 | 0.0126 | 0.0126 | `REJECTED` | Failed guardrail: delta=-0.0288 < 1.0*SE (0.0126, SE=0.0126) |
| **T12** | Exploration | Ablation: Top Intermediate without metadata | intermediate | **0.7563** | -0.0499 | 0.0143 | 0.0143 | `REJECTED` | Failed guardrail: delta=-0.0499 < 1.0*SE (0.0143, SE=0.0143) |
| **T13** | Exploitation | Concatenated [mean; max] multi-resolution representation | late | **0.8195** | +0.0133 | 0.0058 | 0.0058 | `ADOPTED` | Exceeded champion by delta=+0.0133 >= 1.0*SE (0.0058, SE=0.0058) |
| **T14** | Exploitation | Temperature-scaled soft voting (tau=1.5) on concat | late | **0.8195** | +0.0000 | 0.0000 | 0.0000 | `ADOPTED` | Exceeded champion by delta=+0.0000 >= 1.0*SE (0.0000, SE=0.0000) |
| **T15** | Exploitation | Calibrated temperature soft voting (tau=1.2) on concat | late | **0.8196** | +0.0001 | 0.0020 | 0.0020 | `REJECTED` | Failed guardrail: delta=+0.0001 < 1.0*SE (0.0020, SE=0.0020) |
| **T16** | Exploitation | Conservative temperature soft voting (tau=2.0) on concat | late | **0.8169** | -0.0026 | 0.0008 | 0.0008 | `REJECTED` | Failed guardrail: delta=-0.0026 < 1.0*SE (0.0008, SE=0.0008) |
| **T17** | Exploitation | Regularization shrinkage C=0.5 | late | **0.8185** | -0.0010 | 0.0011 | 0.0011 | `REJECTED` | Failed guardrail: delta=-0.0010 < 1.0*SE (0.0011, SE=0.0011) |
| **T18** | Exploitation | Regularization expansion C=2.0 | late | **0.8213** | +0.0018 | 0.0034 | 0.0034 | `REJECTED` | Failed guardrail: delta=+0.0018 < 1.0*SE (0.0034, SE=0.0034) |
| **T19** | Exploitation | Heavy L2 penalty C=0.1 | late | **0.8166** | -0.0030 | 0.0034 | 0.0034 | `REJECTED` | Failed guardrail: delta=-0.0030 < 1.0*SE (0.0034, SE=0.0034) |
