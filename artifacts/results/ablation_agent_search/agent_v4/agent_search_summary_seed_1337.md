# Phase 2 LLM Agent Decision Log (Seed 1337)

**Agent**: LLM Autonomous Fusion Agent (`v4_llm`)  
**LLM Provider**: `GEMINI`  
**Total Trials Run**: 7 / 25  
**Final Champion**: `T02` with **5-fold CV Macro AUROC: 0.8234**  
**Champion Family**: `LATE`  
**Champion Pooling**: `concat` | **gamma_meta**: `1.0`  
**Champion Params**: `{"strategy": "uniform", "c": 1.0, "use_metadata": true}`  
**Champion Hypothesis**: Late uniform probability fusion allows foundation models to act as independent calibrated experts, reducing cross-stream feature interference.  
**Guardrail Policy**: Adopt if $\bar{\Delta} \ge 1.0 \times \text{SE}_{\Delta}$ (patience = 5)

## Trial Progression & Auditable Decision Records

| Trial | Source | Family | Pooling | CV AUROC | Delta vs Champ | SE(Delta) | Threshold (k_se×SE) | Decision | Action / Reason |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **T01** | `domain_reasoning_fallback` | `early` | `concat` | **0.8081** | +0.0000 | 0.0000 | 0.0000 | `ADOPT_INITIAL` | Initial baseline champion established (0.8081). |
| **T02** | `domain_reasoning_fallback` | `late` | `concat` | **0.8234** | +0.0153 | 0.0096 | 0.0096 | `ADOPT_CHAMPION` | Statistically meaningful improvement: Delta (+0.0153) >= 1.0*SE (0.0096). |
| **T03** | `domain_reasoning_fallback` | `late` | `concat` | **0.8195** | -0.0039 | 0.0048 | 0.0048 | `REJECT` | Underperformed champion by -0.0039 (SE: 0.0048). |
| **T04** | `domain_reasoning_fallback` | `intermediate` | `concat` | **0.7885** | -0.0349 | 0.0143 | 0.0143 | `REJECT` | Underperformed champion by -0.0349 (SE: 0.0143). |
| **T05** | `domain_reasoning_fallback` | `late` | `concat` | **0.8196** | -0.0038 | 0.0056 | 0.0056 | `REJECT` | Underperformed champion by -0.0038 (SE: 0.0056). |
| **T06** | `domain_reasoning_fallback` | `late` | `concat` | **0.8185** | -0.0049 | 0.0048 | 0.0048 | `REJECT` | Underperformed champion by -0.0049 (SE: 0.0048). |
| **T07** | `domain_reasoning_fallback` | `late` | `concat` | **0.8169** | -0.0065 | 0.0052 | 0.0052 | `REJECT` | Underperformed champion by -0.0065 (SE: 0.0052). |
