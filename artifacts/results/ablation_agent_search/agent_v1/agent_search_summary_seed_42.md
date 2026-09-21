# Phase 2 Autonomous Agent Decision Log (Seed 42)

**Total Trials Run**: 20  
**Final Champion**: `T15` with **5-fold CV Macro AUROC: 0.8287**  
**Champion Family**: `late`  
**Champion Params**: `{"c": 1.0, "l2_norm": false, "pca_dim": null, "use_metadata": true, "class_weight": null, "strategy": "temperature", "tau": 1.5}`  
**Champion Pooling**: `concat` | **gamma_meta**: `1.0`

## Trial Progression & SE Guardrail Decisions

| Trial | Stage | Hypothesis | Family | CV AUROC | Delta vs Champ | SE(Delta) | Threshold (k_se×SE) | Decision | Action/Reason |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **T01** | Exploration | Early Concatenation (Raw 5632-d) + L2 LogReg + Metadata | early | **0.7980** | +0.0000 | 0.0000 | 0.0000 | `INITIAL_CHAMPION` | First baseline trial established as champion benchmark |
| **T02** | Exploration | Per-stream L2-normalization before concatenation + Metadata | early | **0.7546** | -0.0434 | 0.0194 | 0.0194 | `REJECTED` | Failed guardrail: delta=-0.0434 < 1.0*SE (0.0194, SE=0.0194) |
| **T03** | Exploration | Train-set PCA (256-d, 95% variance) + L2 LogReg + Metadata | early | **0.8056** | +0.0076 | 0.0180 | 0.0180 | `REJECTED` | Failed guardrail: delta=+0.0076 < 1.0*SE (0.0180, SE=0.0180) |
| **T04** | Exploration | Ablation: Top Early configuration without metadata | early | **0.8016** | +0.0035 | 0.0173 | 0.0173 | `REJECTED` | Failed guardrail: delta=+0.0035 < 1.0*SE (0.0173, SE=0.0173) |
| **T05** | Exploration | Uniform Softmax Probability Averaging (1/3 weight each) + Metadata head | late | **0.8028** | +0.0047 | 0.0080 | 0.0080 | `REJECTED` | Failed guardrail: delta=+0.0047 < 1.0*SE (0.0080, SE=0.0080) |
| **T06** | Exploration | Temperature-scaled soft voting (tau=1.5) + Metadata head | late | **0.8004** | +0.0024 | 0.0060 | 0.0060 | `REJECTED` | Failed guardrail: delta=+0.0024 < 1.0*SE (0.0060, SE=0.0060) |
| **T07** | Exploration | Validation-optimized simplex weights (w1, w2, w3 >= 0, sum=1) | late | **0.7903** | -0.0078 | 0.0085 | 0.0085 | `REJECTED` | Failed guardrail: delta=-0.0078 < 1.0*SE (0.0085, SE=0.0085) |
| **T08** | Exploration | Meta-Stacking classifier (Logistic Regression on OOF probability vectors) | late | **0.7900** | -0.0080 | 0.0225 | 0.0225 | `REJECTED` | Failed guardrail: delta=-0.0080 < 1.0*SE (0.0225, SE=0.0225) |
| **T09** | Exploration | Stream-level Gated Fusion (learnable alpha_m(x)) + Metadata | intermediate | **0.7847** | -0.0134 | 0.0085 | 0.0085 | `REJECTED` | Failed guardrail: delta=-0.0134 < 1.0*SE (0.0085, SE=0.0085) |
| **T10** | Exploration | Stream-ABMIL (multi-stream attention pooling across backbones) | intermediate | **0.7771** | -0.0209 | 0.0129 | 0.0129 | `REJECTED` | Failed guardrail: delta=-0.0209 < 1.0*SE (0.0129, SE=0.0129) |
| **T11** | Exploration | Train-set PCA (128-d per stream) -> Gated Fusion | intermediate | **0.7136** | -0.0844 | 0.0245 | 0.0245 | `REJECTED` | Failed guardrail: delta=-0.0844 < 1.0*SE (0.0245, SE=0.0245) |
| **T12** | Exploration | Ablation: Top Intermediate configuration without metadata | intermediate | **0.7715** | -0.0265 | 0.0089 | 0.0089 | `REJECTED` | Failed guardrail: delta=-0.0265 < 1.0*SE (0.0089, SE=0.0089) |
| **T13** | Exploitation | Concatenated [mean; max] slide pooling representation | early | **0.8215** | +0.0235 | 0.0075 | 0.0075 | `ADOPTED` | Exceeded champion by delta=+0.0235 >= 1.0*SE (0.0075, SE=0.0075) |
| **T14** | Exploitation | Slide max pooling representation | early | **0.8225** | +0.0010 | 0.0068 | 0.0068 | `REJECTED` | Failed guardrail: delta=+0.0010 < 1.0*SE (0.0068, SE=0.0068) |
| **T15** | Exploitation | Temperature-scaled soft voting (tau=1.5) | late | **0.8287** | +0.0072 | 0.0037 | 0.0037 | `ADOPTED` | Exceeded champion by delta=+0.0072 >= 1.0*SE (0.0037, SE=0.0037) |
| **T16** | Exploitation | Calibrated temperature soft voting (tau=1.2) | late | **0.8292** | +0.0005 | 0.0014 | 0.0014 | `REJECTED` | Failed guardrail: delta=+0.0005 < 1.0*SE (0.0014, SE=0.0014) |
| **T17** | Exploitation | Regularization shrinkage C=0.5 | early | **0.8213** | -0.0074 | 0.0019 | 0.0019 | `REJECTED` | Failed guardrail: delta=-0.0074 < 1.0*SE (0.0019, SE=0.0019) |
| **T18** | Exploitation | Regularization C=2.0 | early | **0.8207** | -0.0080 | 0.0051 | 0.0051 | `REJECTED` | Failed guardrail: delta=-0.0080 < 1.0*SE (0.0051, SE=0.0051) |
| **T19** | Exploitation | Metadata scaling factor gamma_meta=0.5 | early | **0.8213** | -0.0074 | 0.0038 | 0.0038 | `REJECTED` | Failed guardrail: delta=-0.0074 < 1.0*SE (0.0038, SE=0.0038) |
| **T20** | Exploitation | Metadata scaling factor gamma_meta=0.2 | early | **0.8219** | -0.0067 | 0.0037 | 0.0037 | `REJECTED` | Failed guardrail: delta=-0.0067 < 1.0*SE (0.0037, SE=0.0037) |
