# Phase 2 Autonomous Agent Decision Log (Seed 1337)

**Total Trials Run**: 25  
**Final Champion**: `T23` with **5-fold CV Macro AUROC: 0.8213**  
**Champion Family**: `late`  
**Champion Params**: `{"strategy": "temperature", "c": 1.0, "tau": 1.5, "use_metadata": true, "class_weight": null}`  
**Champion Pooling**: `concat` | **gamma_meta**: `0.2`

## Trial Progression & SE Guardrail Decisions

| Trial | Stage | Hypothesis | Family | CV AUROC | Delta vs Champ | SE(Delta) | Threshold (k_se×SE) | Decision | Action/Reason |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **T01** | Exploration | Early Concatenation (Raw 5632-d) + L2 LogReg + Metadata | early | **0.7998** | +0.0000 | 0.0000 | 0.0000 | `INITIAL_CHAMPION` | First baseline trial established as champion benchmark |
| **T02** | Exploration | Per-stream L2-normalization before concatenation + Metadata | early | **0.7417** | -0.0581 | 0.0258 | 0.0258 | `REJECTED` | Failed guardrail: delta=-0.0581 < 1.0*SE (0.0258, SE=0.0258) |
| **T03** | Exploration | Train-set PCA (256-d, 95% variance) + L2 LogReg + Metadata | early | **0.7878** | -0.0120 | 0.0090 | 0.0090 | `REJECTED` | Failed guardrail: delta=-0.0120 < 1.0*SE (0.0090, SE=0.0090) |
| **T04** | Exploration | Ablation: Top Early configuration without metadata | early | **0.7790** | -0.0208 | 0.0094 | 0.0094 | `REJECTED` | Failed guardrail: delta=-0.0208 < 1.0*SE (0.0094, SE=0.0094) |
| **T05** | Exploration | Uniform Softmax Probability Averaging (1/3 weight each) + Metadata head | late | **0.8098** | +0.0100 | 0.0114 | 0.0114 | `REJECTED` | Failed guardrail: delta=+0.0100 < 1.0*SE (0.0114, SE=0.0114) |
| **T06** | Exploration | Temperature-scaled soft voting (tau=1.5) + Metadata head | late | **0.8062** | +0.0064 | 0.0054 | 0.0054 | `ADOPTED` | Exceeded champion by delta=+0.0064 >= 1.0*SE (0.0054, SE=0.0054) |
| **T07** | Exploration | Validation-optimized simplex weights (w1, w2, w3 >= 0, sum=1) | late | **0.7872** | -0.0190 | 0.0113 | 0.0113 | `REJECTED` | Failed guardrail: delta=-0.0190 < 1.0*SE (0.0113, SE=0.0113) |
| **T08** | Exploration | Meta-Stacking classifier (Logistic Regression on OOF probability vectors) | late | **0.7686** | -0.0376 | 0.0203 | 0.0203 | `REJECTED` | Failed guardrail: delta=-0.0376 < 1.0*SE (0.0203, SE=0.0203) |
| **T09** | Exploration | Stream-level Gated Fusion (learnable alpha_m(x)) + Metadata | intermediate | **0.8050** | -0.0012 | 0.0112 | 0.0112 | `REJECTED` | Failed guardrail: delta=-0.0012 < 1.0*SE (0.0112, SE=0.0112) |
| **T10** | Exploration | Stream-ABMIL (multi-stream attention pooling across backbones) | intermediate | **0.7812** | -0.0250 | 0.0183 | 0.0183 | `REJECTED` | Failed guardrail: delta=-0.0250 < 1.0*SE (0.0183, SE=0.0183) |
| **T11** | Exploration | Train-set PCA (128-d per stream) -> Gated Fusion | intermediate | **0.7120** | -0.0941 | 0.0189 | 0.0189 | `REJECTED` | Failed guardrail: delta=-0.0941 < 1.0*SE (0.0189, SE=0.0189) |
| **T12** | Exploration | Ablation: Top Intermediate configuration without metadata | intermediate | **0.7996** | -0.0066 | 0.0122 | 0.0122 | `REJECTED` | Failed guardrail: delta=-0.0066 < 1.0*SE (0.0122, SE=0.0122) |
| **T13** | Exploitation | Concatenated [mean; max] slide pooling representation | late | **0.8195** | +0.0133 | 0.0058 | 0.0058 | `ADOPTED` | Exceeded champion by delta=+0.0133 >= 1.0*SE (0.0058, SE=0.0058) |
| **T14** | Exploitation | Slide max pooling representation | late | **0.8242** | +0.0047 | 0.0076 | 0.0076 | `REJECTED` | Failed guardrail: delta=+0.0047 < 1.0*SE (0.0076, SE=0.0076) |
| **T15** | Exploitation | Temperature-scaled soft voting (tau=1.5) | late | **0.8195** | +0.0000 | 0.0000 | 0.0000 | `ADOPTED` | Exceeded champion by delta=+0.0000 >= 1.0*SE (0.0000, SE=0.0000) |
| **T16** | Exploitation | Calibrated temperature soft voting (tau=1.2) | late | **0.8196** | +0.0001 | 0.0020 | 0.0020 | `REJECTED` | Failed guardrail: delta=+0.0001 < 1.0*SE (0.0020, SE=0.0020) |
| **T17** | Exploitation | Regularization shrinkage C=0.5 | late | **0.8185** | -0.0010 | 0.0011 | 0.0011 | `REJECTED` | Failed guardrail: delta=-0.0010 < 1.0*SE (0.0011, SE=0.0011) |
| **T18** | Exploitation | Regularization C=2.0 | late | **0.8213** | +0.0018 | 0.0034 | 0.0034 | `REJECTED` | Failed guardrail: delta=+0.0018 < 1.0*SE (0.0034, SE=0.0034) |
| **T19** | Exploitation | Metadata scaling factor gamma_meta=0.5 | late | **0.8176** | -0.0020 | 0.0009 | 0.0009 | `REJECTED` | Failed guardrail: delta=-0.0020 < 1.0*SE (0.0009, SE=0.0009) |
| **T20** | Exploitation | Metadata scaling factor gamma_meta=0.2 | late | **0.8213** | +0.0018 | 0.0016 | 0.0016 | `ADOPTED` | Exceeded champion by delta=+0.0018 >= 1.0*SE (0.0016, SE=0.0016) |
| **T21** | Exploitation | Metadata scaling factor gamma_meta=2.0 | late | **0.8194** | -0.0019 | 0.0046 | 0.0046 | `REJECTED` | Failed guardrail: delta=-0.0019 < 1.0*SE (0.0046, SE=0.0046) |
| **T22** | Exploitation | Balanced class-prior loss weighting | late | **0.8190** | -0.0023 | 0.0027 | 0.0027 | `REJECTED` | Failed guardrail: delta=-0.0023 < 1.0*SE (0.0027, SE=0.0027) |
| **T23** | Exploitation | Uniform class weighting with optimal regularization | late | **0.8213** | +0.0000 | 0.0000 | 0.0000 | `ADOPTED` | Exceeded champion by delta=+0.0000 >= 1.0*SE (0.0000, SE=0.0000) |
| **T24** | Exploitation | Simplex-optimized weights on concatenated representation | late | **0.7970** | -0.0243 | 0.0146 | 0.0146 | `REJECTED` | Failed guardrail: delta=-0.0243 < 1.0*SE (0.0146, SE=0.0146) |
| **T25** | Exploitation | Optimal temperature-scaled fusion + calibrated metadata | late | **0.8195** | -0.0018 | 0.0016 | 0.0016 | `REJECTED` | Failed guardrail: delta=-0.0018 < 1.0*SE (0.0016, SE=0.0016) |
