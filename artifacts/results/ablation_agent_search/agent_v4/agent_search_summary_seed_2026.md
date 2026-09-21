# Phase 2 LLM Agent Decision Log (Seed 2026)

**Agent**: LLM Autonomous Fusion Agent (`v4_llm`)  
**LLM Provider**: `GEMINI`  
**Total Trials Run**: 7 / 25  
**Final Champion**: `T02` with **5-fold CV Macro AUROC: 0.8409**  
**Champion Family**: `LATE`  
**Champion Pooling**: `concat` | **gamma_meta**: `1.0`  
**Champion Params**: `{"strategy": "uniform", "c": 1.0, "use_metadata": true}`  
**Champion Hypothesis**: Late uniform probability fusion allows foundation models to act as independent calibrated experts, reducing cross-stream feature interference.  
**Guardrail Policy**: Adopt if $\bar{\Delta} \ge 1.0 \times \text{SE}_{\Delta}$ (patience = 5)

## Trial Progression & Auditable Decision Records

| Trial | Source | Family | Pooling | CV AUROC | Delta vs Champ | SE(Delta) | Threshold (k_se×SE) | Decision | Action / Reason |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **T01** | `domain_reasoning_fallback` | `early` | `concat` | **0.8303** | +0.0000 | 0.0000 | 0.0000 | `ADOPT_INITIAL` | Initial baseline champion established (0.8303). |
| **T02** | `domain_reasoning_fallback` | `late` | `concat` | **0.8409** | +0.0107 | 0.0028 | 0.0028 | `ADOPT_CHAMPION` | Statistically meaningful improvement: Delta (+0.0107) >= 1.0*SE (0.0028). |
| **T03** | `domain_reasoning_fallback` | `late` | `concat` | **0.8391** | -0.0018 | 0.0021 | 0.0021 | `REJECT` | Underperformed champion by -0.0018 (SE: 0.0021). |
| **T04** | `domain_reasoning_fallback` | `intermediate` | `concat` | **0.8169** | -0.0240 | 0.0162 | 0.0162 | `REJECT` | Underperformed champion by -0.0240 (SE: 0.0162). |
| **T05** | `domain_reasoning_fallback` | `late` | `concat` | **0.8411** | +0.0002 | 0.0024 | 0.0024 | `REJECT` | Marginal gain (+0.0002) did not exceed 1.0*SE guardrail (0.0024). Rejected to avoid Winner's Curse. |
| **T06** | `domain_reasoning_fallback` | `late` | `concat` | **0.8403** | -0.0006 | 0.0027 | 0.0027 | `REJECT` | Underperformed champion by -0.0006 (SE: 0.0027). |
| **T07** | `domain_reasoning_fallback` | `late` | `concat` | **0.8413** | +0.0003 | 0.0031 | 0.0031 | `REJECT` | Marginal gain (+0.0003) did not exceed 1.0*SE guardrail (0.0031). Rejected to avoid Winner's Curse. |
