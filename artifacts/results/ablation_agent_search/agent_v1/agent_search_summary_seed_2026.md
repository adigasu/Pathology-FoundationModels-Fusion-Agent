# Phase 2 Autonomous Agent Decision Log (Seed 2026)

**Total Trials Run**: 18  
**Final Champion**: `T13` with **5-fold CV Macro AUROC: 0.8409**  
**Champion Family**: `late`  
**Champion Params**: `{"strategy": "uniform", "c": 1.0, "use_metadata": true, "class_weight": null}`  
**Champion Pooling**: `concat` | **gamma_meta**: `1.0`

## Trial Progression & SE Guardrail Decisions

| Trial | Stage | Hypothesis | Family | CV AUROC | Delta vs Champ | SE(Delta) | Threshold (k_se×SE) | Decision | Action/Reason |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **T01** | Exploration | Early Concatenation (Raw 5632-d) + L2 LogReg + Metadata | early | **0.8186** | +0.0000 | 0.0000 | 0.0000 | `INITIAL_CHAMPION` | First baseline trial established as champion benchmark |
| **T02** | Exploration | Per-stream L2-normalization before concatenation + Metadata | early | **0.7728** | -0.0458 | 0.0174 | 0.0174 | `REJECTED` | Failed guardrail: delta=-0.0458 < 1.0*SE (0.0174, SE=0.0174) |
| **T03** | Exploration | Train-set PCA (256-d, 95% variance) + L2 LogReg + Metadata | early | **0.8158** | -0.0028 | 0.0088 | 0.0088 | `REJECTED` | Failed guardrail: delta=-0.0028 < 1.0*SE (0.0088, SE=0.0088) |
| **T04** | Exploration | Ablation: Top Early configuration without metadata | early | **0.8247** | +0.0061 | 0.0082 | 0.0082 | `REJECTED` | Failed guardrail: delta=+0.0061 < 1.0*SE (0.0082, SE=0.0082) |
| **T05** | Exploration | Uniform Softmax Probability Averaging (1/3 weight each) + Metadata head | late | **0.8256** | +0.0069 | 0.0034 | 0.0034 | `ADOPTED` | Exceeded champion by delta=+0.0069 >= 1.0*SE (0.0034, SE=0.0034) |
| **T06** | Exploration | Temperature-scaled soft voting (tau=1.5) + Metadata head | late | **0.8258** | +0.0002 | 0.0044 | 0.0044 | `REJECTED` | Failed guardrail: delta=+0.0002 < 1.0*SE (0.0044, SE=0.0044) |
| **T07** | Exploration | Validation-optimized simplex weights (w1, w2, w3 >= 0, sum=1) | late | **0.8208** | -0.0047 | 0.0062 | 0.0062 | `REJECTED` | Failed guardrail: delta=-0.0047 < 1.0*SE (0.0062, SE=0.0062) |
| **T08** | Exploration | Meta-Stacking classifier (Logistic Regression on OOF probability vectors) | late | **0.8156** | -0.0100 | 0.0143 | 0.0143 | `REJECTED` | Failed guardrail: delta=-0.0100 < 1.0*SE (0.0143, SE=0.0143) |
| **T09** | Exploration | Stream-level Gated Fusion (learnable alpha_m(x)) + Metadata | intermediate | **0.8324** | +0.0069 | 0.0082 | 0.0082 | `REJECTED` | Failed guardrail: delta=+0.0069 < 1.0*SE (0.0082, SE=0.0082) |
| **T10** | Exploration | Stream-ABMIL (multi-stream attention pooling across backbones) | intermediate | **0.7967** | -0.0289 | 0.0139 | 0.0139 | `REJECTED` | Failed guardrail: delta=-0.0289 < 1.0*SE (0.0139, SE=0.0139) |
| **T11** | Exploration | Train-set PCA (128-d per stream) -> Gated Fusion | intermediate | **0.7204** | -0.1051 | 0.0115 | 0.0115 | `REJECTED` | Failed guardrail: delta=-0.1051 < 1.0*SE (0.0115, SE=0.0115) |
| **T12** | Exploration | Ablation: Top Intermediate configuration without metadata | intermediate | **0.8238** | -0.0017 | 0.0053 | 0.0053 | `REJECTED` | Failed guardrail: delta=-0.0017 < 1.0*SE (0.0053, SE=0.0053) |
| **T13** | Exploitation | Concatenated [mean; max] slide pooling representation | late | **0.8409** | +0.0154 | 0.0049 | 0.0049 | `ADOPTED` | Exceeded champion by delta=+0.0154 >= 1.0*SE (0.0049, SE=0.0049) |
| **T14** | Exploitation | Slide max pooling representation | late | **0.8450** | +0.0041 | 0.0079 | 0.0079 | `REJECTED` | Failed guardrail: delta=+0.0041 < 1.0*SE (0.0079, SE=0.0079) |
| **T15** | Exploitation | Temperature-scaled soft voting (tau=1.5) | late | **0.8391** | -0.0018 | 0.0021 | 0.0021 | `REJECTED` | Failed guardrail: delta=-0.0018 < 1.0*SE (0.0021, SE=0.0021) |
| **T16** | Exploitation | Calibrated temperature soft voting (tau=1.2) | late | **0.8411** | +0.0002 | 0.0024 | 0.0024 | `REJECTED` | Failed guardrail: delta=+0.0002 < 1.0*SE (0.0024, SE=0.0024) |
| **T17** | Exploitation | Regularization shrinkage C=0.5 | late | **0.8408** | -0.0001 | 0.0010 | 0.0010 | `REJECTED` | Failed guardrail: delta=-0.0001 < 1.0*SE (0.0010, SE=0.0010) |
| **T18** | Exploitation | Regularization C=2.0 | late | **0.8413** | +0.0004 | 0.0014 | 0.0014 | `REJECTED` | Failed guardrail: delta=+0.0004 < 1.0*SE (0.0014, SE=0.0014) |
