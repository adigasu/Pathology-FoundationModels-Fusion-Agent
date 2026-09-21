# Phase 2 LLM Agent Decision Log (Seed 42)

**Agent**: LLM Autonomous Fusion Agent (`v4_llm`)  
**LLM Provider**: `GEMINI`  
**Total Trials Run**: 14 / 25  
**Final Champion**: `T09` with **5-fold CV Macro AUROC: 0.8309**  
**Champion Family**: `LATE`  
**Champion Pooling**: `concat` | **gamma_meta**: `1.5`  
**Champion Params**: `{"strategy": "temperature", "c": 1.0, "tau": 1.5, "use_metadata": true}`  
**Champion Hypothesis**: Upweight clinical metadata (gamma_meta=1.5) to inject stronger stage/age diagnostic priors.  
**Guardrail Policy**: Adopt if $\bar{\Delta} \ge 1.0 \times \text{SE}_{\Delta}$ (patience = 5)

## Trial Progression & Auditable Decision Records

| Trial | Source | Family | Pooling | CV AUROC | Delta vs Champ | SE(Delta) | Threshold (k_se×SE) | Decision | Action / Reason |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **T01** | `domain_reasoning_fallback` | `early` | `concat` | **0.8215** | +0.0000 | 0.0000 | 0.0000 | `ADOPT_INITIAL` | Initial baseline champion established (0.8215). |
| **T02** | `domain_reasoning_fallback` | `late` | `concat` | **0.8270** | +0.0055 | 0.0038 | 0.0038 | `ADOPT_CHAMPION` | Statistically meaningful improvement: Delta (+0.0055) >= 1.0*SE (0.0038). |
| **T03** | `domain_reasoning_fallback` | `late` | `concat` | **0.8287** | +0.0017 | 0.0029 | 0.0029 | `REJECT` | Marginal gain (+0.0017) did not exceed 1.0*SE guardrail (0.0029). Rejected to avoid Winner's Curse. |
| **T04** | `domain_reasoning_fallback` | `intermediate` | `concat` | **0.7768** | -0.0502 | 0.0085 | 0.0085 | `REJECT` | Underperformed champion by -0.0502 (SE: 0.0085). |
| **T05** | `domain_reasoning_fallback` | `late` | `concat` | **0.8292** | +0.0021 | 0.0019 | 0.0019 | `ADOPT_CHAMPION` | Statistically meaningful improvement: Delta (+0.0021) >= 1.0*SE (0.0019). |
| **T06** | `domain_reasoning_fallback` | `late` | `concat` | **0.8272** | -0.0020 | 0.0019 | 0.0019 | `REJECT` | Underperformed champion by -0.0020 (SE: 0.0019). |
| **T07** | `domain_reasoning_fallback` | `late` | `concat` | **0.8287** | -0.0005 | 0.0024 | 0.0024 | `REJECT` | Underperformed champion by -0.0005 (SE: 0.0024). |
| **T08** | `domain_reasoning_fallback` | `late` | `concat` | **0.8270** | -0.0021 | 0.0019 | 0.0019 | `REJECT` | Underperformed champion by -0.0021 (SE: 0.0019). |
| **T09** | `domain_reasoning_fallback` | `late` | `concat` | **0.8309** | +0.0017 | 0.0006 | 0.0006 | `ADOPT_CHAMPION` | Statistically meaningful improvement: Delta (+0.0017) >= 1.0*SE (0.0006). |
| **T10** | `domain_reasoning_fallback` | `late` | `concat` | **0.7897** | -0.0411 | 0.0064 | 0.0064 | `REJECT` | Underperformed champion by -0.0411 (SE: 0.0064). |
| **T11** | `domain_reasoning_fallback` | `late` | `concat` | **0.8276** | -0.0032 | 0.0028 | 0.0028 | `REJECT` | Underperformed champion by -0.0032 (SE: 0.0028). |
| **T12** | `domain_reasoning_fallback` | `early` | `concat` | **0.8215** | -0.0094 | 0.0034 | 0.0034 | `REJECT` | Underperformed champion by -0.0094 (SE: 0.0034). |
| **T13** | `domain_reasoning_fallback` | `late` | `concat` | **0.8270** | -0.0038 | 0.0024 | 0.0024 | `REJECT` | Underperformed champion by -0.0038 (SE: 0.0024). |
| **T14** | `domain_reasoning_fallback` | `late` | `concat` | **0.8287** | -0.0022 | 0.0012 | 0.0012 | `REJECT` | Underperformed champion by -0.0022 (SE: 0.0012). |
