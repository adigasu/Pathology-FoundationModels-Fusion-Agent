# Phase 2 Agent v2 Decision Log (Seed 42)

**Agent Strategy**: Principled Exploitation with Paired SE Guardrail  
**Total Trials Run**: 19  
**Final Champion**: `T14` with **5-fold CV Macro AUROC: 0.8287**  
**Champion Family**: `late`  
**Champion Params**: `{"strategy": "temperature", "pca_components": 256, "c": 1.0, "use_metadata": true, "tau": 1.5}`  
**Champion Pooling**: `concat` | **gamma_meta**: `1.0`  
**Guardrail Policy**: Adopt if $\bar{\Delta} \ge 1.0 \times \text{SE}_{\Delta}$ (patience = 5)

## Trial Progression & SE Guardrail Decisions

| Trial | Stage | Hypothesis | Family | CV AUROC | Delta vs Champ | SE(Delta) | Threshold (k_se×SE) | Decision | Action/Reason |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **T01** | Exploration | Raw 5632-d concatenation + L2 LR | early | **0.7980** | +0.0000 | 0.0000 | 0.0000 | `INITIAL_CHAMPION` | First baseline trial established as champion benchmark |
| **T02** | Exploration | L2-normalized stream concatenation | early | **0.7980** | +0.0000 | 0.0000 | 0.0000 | `ADOPTED` | Exceeded champion by delta=+0.0000 >= 1.0*SE (0.0000, SE=0.0000) |
| **T03** | Exploration | Train-set PCA (256-d, 95% variance) | early | **0.7980** | +0.0000 | 0.0000 | 0.0000 | `ADOPTED` | Exceeded champion by delta=+0.0000 >= 1.0*SE (0.0000, SE=0.0000) |
| **T04** | Exploration | Ablation: Top Early without metadata | early | **0.7974** | -0.0006 | 0.0022 | 0.0022 | `REJECTED` | Failed guardrail: delta=-0.0006 < 1.0*SE (0.0022, SE=0.0022) |
| **T05** | Exploration | Uniform probability soft voting (1/3 each) | late | **0.8028** | +0.0047 | 0.0080 | 0.0080 | `REJECTED` | Failed guardrail: delta=+0.0047 < 1.0*SE (0.0080, SE=0.0080) |
| **T06** | Exploration | Temperature-scaled soft voting (tau=1.5) | late | **0.8004** | +0.0024 | 0.0060 | 0.0060 | `REJECTED` | Failed guardrail: delta=+0.0024 < 1.0*SE (0.0060, SE=0.0060) |
| **T07** | Exploration | Validation simplex-optimized weights | late | **0.7903** | -0.0078 | 0.0085 | 0.0085 | `REJECTED` | Failed guardrail: delta=-0.0078 < 1.0*SE (0.0085, SE=0.0085) |
| **T08** | Exploration | OOF stacking meta-classifier | late | **0.7900** | -0.0080 | 0.0225 | 0.0225 | `REJECTED` | Failed guardrail: delta=-0.0080 < 1.0*SE (0.0225, SE=0.0225) |
| **T09** | Exploration | Stream-level Gated Fusion (dropout=0.2) | intermediate | **0.7804** | -0.0177 | 0.0180 | 0.0180 | `REJECTED` | Failed guardrail: delta=-0.0177 < 1.0*SE (0.0180, SE=0.0180) |
| **T10** | Exploration | Stream-ABMIL multi-stream attention | intermediate | **0.7804** | -0.0177 | 0.0180 | 0.0180 | `REJECTED` | Failed guardrail: delta=-0.0177 < 1.0*SE (0.0180, SE=0.0180) |
| **T11** | Exploration | PCA (128-d per stream) -> Gated Fusion | intermediate | **0.7804** | -0.0177 | 0.0180 | 0.0180 | `REJECTED` | Failed guardrail: delta=-0.0177 < 1.0*SE (0.0180, SE=0.0180) |
| **T12** | Exploration | Ablation: Top Intermediate without metadata | intermediate | **0.7738** | -0.0242 | 0.0045 | 0.0045 | `REJECTED` | Failed guardrail: delta=-0.0242 < 1.0*SE (0.0045, SE=0.0045) |
| **T13** | Exploitation | Concatenated [mean; max] multi-resolution representation | early | **0.8215** | +0.0235 | 0.0075 | 0.0075 | `ADOPTED` | Exceeded champion by delta=+0.0235 >= 1.0*SE (0.0075, SE=0.0075) |
| **T14** | Exploitation | Temperature-scaled soft voting (tau=1.5) on concat | late | **0.8287** | +0.0072 | 0.0037 | 0.0037 | `ADOPTED` | Exceeded champion by delta=+0.0072 >= 1.0*SE (0.0037, SE=0.0037) |
| **T15** | Exploitation | Calibrated temperature soft voting (tau=1.2) on concat | late | **0.8292** | +0.0005 | 0.0014 | 0.0014 | `REJECTED` | Failed guardrail: delta=+0.0005 < 1.0*SE (0.0014, SE=0.0014) |
| **T16** | Exploitation | Conservative temperature soft voting (tau=2.0) on concat | late | **0.8287** | +0.0000 | 0.0028 | 0.0028 | `REJECTED` | Failed guardrail: delta=+0.0000 < 1.0*SE (0.0028, SE=0.0028) |
| **T17** | Exploitation | Regularization shrinkage C=0.5 | early | **0.8213** | -0.0074 | 0.0019 | 0.0019 | `REJECTED` | Failed guardrail: delta=-0.0074 < 1.0*SE (0.0019, SE=0.0019) |
| **T18** | Exploitation | Regularization expansion C=2.0 | early | **0.8207** | -0.0080 | 0.0051 | 0.0051 | `REJECTED` | Failed guardrail: delta=-0.0080 < 1.0*SE (0.0051, SE=0.0051) |
| **T19** | Exploitation | Heavy L2 penalty C=0.1 | early | **0.8217** | -0.0070 | 0.0020 | 0.0020 | `REJECTED` | Failed guardrail: delta=-0.0070 < 1.0*SE (0.0020, SE=0.0020) |
