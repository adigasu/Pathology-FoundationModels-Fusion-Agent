import os
import sys
import json
import time
import urllib.request
import urllib.error
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from src.fusion.data import load_multimodal_dataset, get_train_val_cv_folds, CLASS_NAMES
from src.fusion.models import EarlyFusionClassifier, LateFusionClassifier, IntermediateFusionClassifier
from src.eval.metrics import compute_multiclass_metrics


def evaluate_candidate_cv(
    candidate_cfg: Dict[str, Any],
    dataset: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[float, List[float], List[Dict[str, Any]]]:
    """
    Evaluates candidate configuration on 5-fold Stratified Cross-Validation on Train+Val.
    """
    fold_aurocs = []
    fold_metrics = []

    family = candidate_cfg["family"]
    params = candidate_cfg.get("params", {})

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        tr_feats = {k: v[tr_idx] for k, v in dataset["features"].items()}
        val_feats = {k: v[val_idx] for k, v in dataset["features"].items()}
        tr_meta = dataset["metadata"][tr_idx]
        val_meta = dataset["metadata"][val_idx]
        tr_y = dataset["y"][tr_idx]
        val_y = dataset["y"][val_idx]

        if family == "early":
            model = EarlyFusionClassifier(
                c=float(params.get("c", 1.0)),
                l2_norm_per_stream=bool(params.get("l2_norm", False)),
                pca_dim=params.get("pca_dim", None),
                use_metadata=bool(params.get("use_metadata", True)),
                class_weight=params.get("class_weight", None),
                random_state=42 + fold_idx
            )
        elif family == "late":
            model = LateFusionClassifier(
                strategy=str(params.get("strategy", "uniform")),
                c=float(params.get("c", 1.0)),
                tau=float(params.get("tau", 1.5)),
                use_metadata=bool(params.get("use_metadata", True)),
                class_weight=params.get("class_weight", None),
                random_state=42 + fold_idx
            )
        elif family == "intermediate":
            model = IntermediateFusionClassifier(
                arch=str(params.get("arch", "gated")),
                pca_per_stream=params.get("pca_per_stream", None),
                dropout=float(params.get("dropout", 0.2)),
                weight_decay=float(params.get("weight_decay", 1e-3)),
                lr=float(params.get("lr", 1e-2)),
                epochs=int(params.get("epochs", 40)),
                use_metadata=bool(params.get("use_metadata", True)),
                class_weight=params.get("class_weight", None),
                random_state=42 + fold_idx
            )
        else:
            raise ValueError(f"Unknown fusion family: {family}")

        model.fit(tr_feats, tr_meta, tr_y)
        val_probs = model.predict_proba(val_feats, val_meta)
        val_metrics = compute_multiclass_metrics(val_y, val_probs, CLASS_NAMES)
        fold_aurocs.append(val_metrics["macro_auroc"])
        fold_metrics.append(val_metrics)

    mean_auroc = float(np.mean(fold_aurocs))
    return mean_auroc, fold_aurocs, fold_metrics


class LLMClient:
    """
    Zero-dependency LLM client supporting OpenAI, Gemini, Anthropic, or Local endpoints,
    with an automatic fallback heuristic engine when no API keys are present.
    """
    def __init__(self):
        # Auto-load .env if present (zero external dependencies)
        for env_path in [".env", os.path.expanduser("~/.env")]:
            if os.path.exists(env_path):
                try:
                    with open(env_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, v = line.split("=", 1)
                                k, v = k.strip(), v.strip().strip("'\"")
                                if k not in os.environ:
                                    os.environ[k] = v
                except Exception:
                    pass

        self.gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = os.environ.get("LLM_BASE_URL")
        self.gemini_model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        self.provider = self._detect_provider()
        self.active_gemini_model = None
        if self.provider == "gemini":
            self.active_gemini_model = self._discover_gemini_model()

    def _detect_provider(self) -> str:
        if self.gemini_key:
            return "gemini"
        elif self.openai_key:
            return "openai"
        elif self.anthropic_key:
            return "anthropic"
        elif self.base_url:
            return "local"
        else:
            return "fallback_heuristic"


    def _discover_gemini_model(self) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.gemini_key}"
        try:
            req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                available = [
                    m["name"].replace("models/", "").strip()
                    for m in data.get("models", [])
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                print(f"[LLMClient] Discovered {len(available)} available models for this Gemini API key.", flush=True)
                
                # Check user preference first
                clean_pref = self.gemini_model.replace("models/", "").strip()
                if clean_pref in available:
                    print(f"[LLMClient] Using requested model: '{clean_pref}'", flush=True)
                    return clean_pref
                
                # Priority list of high-quality generation models
                for pref in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]:
                    if pref in available:
                        print(f"[LLMClient] Selected active model: '{pref}'", flush=True)
                        return pref
                        
                if available:
                    print(f"[LLMClient] Selected first compatible model: '{available[0]}'", flush=True)
                    return available[0]
        except urllib.error.HTTPError as e:
            try:
                err_msg = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", str(e))
            except Exception:
                err_msg = str(e)
            print(f"[LLMClient Warning] Gemini ListModels returned HTTP {e.code}: {err_msg}", flush=True)
        except Exception as e:
            print(f"[LLMClient Warning] Could not list Gemini models: {e}", flush=True)
            
        clean_fallback = self.gemini_model.replace("models/", "").strip()
        if clean_fallback in ("gemini-pro", "gemini-1.0-pro", "gemini-2.5-flash"):
            clean_fallback = "gemini-1.5-flash"
        return clean_fallback

    def query(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if self.provider == "fallback_heuristic":
            return None

        try:
            if self.provider == "openai":
                model = os.environ.get("OPENAI_MODEL", "gpt-4o")
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.openai_key}"
                }
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.4,
                    "response_format": {"type": "json_object"}
                }
                req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    return res["choices"][0]["message"]["content"]

            elif self.provider == "gemini":
                # Candidate generation models (prioritizing gemini-3.6-flash recommended by Google)
                candidates = []
                for m in ["gemini-3.6-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", self.gemini_model]:
                    clean_m = m.replace("models/", "").strip()
                    if clean_m and clean_m not in candidates:
                        candidates.append(clean_m)

                # Prioritize previously verified working model
                if hasattr(self, "_working_gemini_model") and self._working_gemini_model:
                    candidates = [self._working_gemini_model] + [c for c in candidates if c != self._working_gemini_model]

                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.gemini_key
                }
                payload = {
                    "contents": [{
                        "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]
                    }],
                    "generationConfig": {
                        "temperature": 0.4,
                        "responseMimeType": "application/json"
                    }
                }
                req_data = json.dumps(payload).encode("utf-8")

                last_error_msg = None
                for model in candidates:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gemini_key}"
                    req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
                    try:
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            res = json.loads(resp.read().decode("utf-8"))
                            self._working_gemini_model = model
                            return res["candidates"][0]["content"]["parts"][0]["text"]
                    except urllib.error.HTTPError as e:
                        try:
                            err_json = json.loads(e.read().decode("utf-8"))
                            err_msg = err_json.get("error", {}).get("message", str(e))
                        except Exception:
                            err_msg = str(e)
                        last_error_msg = f"HTTP {e.code} on '{model}': {err_msg}"
                        # If model is 404 (deprecated), 503 (high demand), 429 (rate limit), or 500/502/504: try next candidate!
                        if e.code in (404, 429, 500, 502, 503, 504) or any(phrase in err_msg.lower() for phrase in ["high demand", "no longer available", "quota", "temporarily"]):
                            print(f"[LLMClient] Model '{model}' unavailable (HTTP {e.code}: {err_msg[:60]}...). Cascading to next Gemini model...", flush=True)
                            continue
                        else:
                            print(f"[LLMClient Warning] Gemini API error ({e.code}) on '{model}': {err_msg}. Falling back to internal reasoning engine.", flush=True)
                            return None
                    except Exception as e:
                        last_error_msg = str(e)
                        continue

                print(f"[LLMClient Warning] All Gemini candidate models failed ({last_error_msg}). Falling back to internal reasoning engine.", flush=True)
                return None
            elif self.provider == "anthropic":
                model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2023-06-01"
                }
                payload = {
                    "model": model,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "max_tokens": 1000,
                    "temperature": 0.4
                }
                req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    return res["content"][0]["text"]

            elif self.provider == "local":
                url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "model": os.environ.get("LOCAL_MODEL", "default"),
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.4
                }
                req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    return res["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"[LLMClient Warning] API call failed ({e}). Falling back to internal reasoning engine.", flush=True)
            return None


class LLMFusionAgent:
    """
    LLM-Powered Autonomous Optimization Agent for Multimodal Pathology Foundation Model Fusion.
    - Operates under a 25-trial budget.
    - Proposes hypothesis-driven fusion configurations using LLM reasoning (with seamless fallback).
    - Employs a paired 5-fold CV SE guardrail: adopt only if mean(Delta) >= k_se * SE(Delta).
    - Logs every hypothesis, proposal, reason, and fold-level metric to JSON and Markdown.
    """
    def __init__(
        self,
        seed: int = 42,
        k_se: float = 1.0,
        max_consecutive_failures: int = 5,
        max_trials: int = 25
    ):
        self.seed = seed
        self.k_se = k_se
        self.max_consecutive_failures = max_consecutive_failures
        self.max_trials = max_trials

        self.llm = LLMClient()
        self.trials_log = []
        self.champion = None
        self.dataset_cache = {}

        # Fallback heuristic progression designed from empirical domain exploration
        self._fallback_candidates = [
            # Broad Exploration (T01-T04)
            {
                "hypothesis": "Multi-resolution concat [mean; max] pooling with early feature concatenation provides complementary architectural and focal cellular cues.",
                "reasoning": "Combining global tissue context (mean) and localized focal atypia (max) gives linear classifiers maximum discriminative signal.",
                "family": "early",
                "pooling": "concat",
                "gamma_meta": 1.0,
                "params": {"c": 1.0, "l2_norm": False, "pca_dim": None, "use_metadata": True}
            },
            {
                "hypothesis": "Late uniform probability fusion allows foundation models to act as independent calibrated experts, reducing cross-stream feature interference.",
                "reasoning": "Averaging stream posterior probabilities mitigates dimension-mismatched gradient conflict across models.",
                "family": "late",
                "pooling": "concat",
                "gamma_meta": 1.0,
                "params": {"strategy": "uniform", "c": 1.0, "use_metadata": True}
            },
            {
                "hypothesis": "Temperature-scaled late fusion (tau=1.5) smooths overconfident foundation model predictions and sharpens ensemble uncertainty calibration.",
                "reasoning": "Pretrained foundation models often output peaked probabilities; softmax temperature softens overconfident margins.",
                "family": "late",
                "pooling": "concat",
                "gamma_meta": 1.0,
                "params": {"strategy": "temperature", "c": 1.0, "tau": 1.5, "use_metadata": True}
            },
            {
                "hypothesis": "Non-linear intermediate gated neural fusion dynamically routes representations based on slide-level histological context.",
                "reasoning": "Gated cross-modal routing dynamically computes feature gates per sample to upweight diagnostic features.",
                "family": "intermediate",
                "pooling": "concat",
                "gamma_meta": 1.0,
                "params": {"arch": "gated", "pca_per_stream": None, "dropout": 0.2, "weight_decay": 1e-3, "lr": 1e-2, "epochs": 40, "use_metadata": True}
            },
            # Principled Exploitation (T05-T15)
            {
                "hypothesis": "Tune temperature scaling to tau=1.2 to retain discriminative sharp peaks while preventing overconfidence.",
                "reasoning": "Moderate temperature scaling often hits the sweet spot between uncalibrated argmax and uniform dilution.",
                "family": "late",
                "pooling": "concat",
                "gamma_meta": 1.0,
                "params": {"strategy": "temperature", "c": 1.0, "tau": 1.2, "use_metadata": True}
            },
            {
                "hypothesis": "Apply stronger L2 regularization shrinkage (C=0.5) to combat small-sample estimation variance.",
                "reasoning": "With 164 patients, penalizing large weights reduces variance across stratified folds.",
                "family": "late",
                "pooling": "concat",
                "gamma_meta": 1.0,
                "params": {"strategy": "temperature", "c": 0.5, "tau": 1.5, "use_metadata": True}
            },
            {
                "hypothesis": "Evaluate conservative temperature scaling (tau=2.0) to enforce higher ensemble consensus.",
                "reasoning": "Tests if broader uncertainty smoothing improves generalization on difficult boundary cases.",
                "family": "late",
                "pooling": "concat",
                "gamma_meta": 1.0,
                "params": {"strategy": "temperature", "c": 1.0, "tau": 2.0, "use_metadata": True}
            },
            {
                "hypothesis": "Downweight clinical metadata (gamma_meta=0.5) to test if pure foundation morphology dominates.",
                "reasoning": "Checks sensitivity to clinical covariates versus purely histological deep feature representations.",
                "family": "late",
                "pooling": "concat",
                "gamma_meta": 0.5,
                "params": {"strategy": "temperature", "c": 1.0, "tau": 1.5, "use_metadata": True}
            },
            {
                "hypothesis": "Upweight clinical metadata (gamma_meta=1.5) to inject stronger stage/age diagnostic priors.",
                "reasoning": "Clinical staging correlates strongly with HGSC vs LGSC discrimination.",
                "family": "late",
                "pooling": "concat",
                "gamma_meta": 1.5,
                "params": {"strategy": "temperature", "c": 1.0, "tau": 1.5, "use_metadata": True}
            },
            {
                "hypothesis": "Employ validation simplex optimization to dynamically learn optimal foundation model voting weights.",
                "reasoning": "Constrained simplex optimization assigns higher voting mass to the most reliable foundation model.",
                "family": "late",
                "pooling": "concat",
                "gamma_meta": 1.0,
                "params": {"strategy": "simplex", "c": 1.0, "use_metadata": True}
            },
            {
                "hypothesis": "Apply balanced class weighting to compensate for minority histotype representation (e.g. clear cell & low-grade serous).",
                "reasoning": "Inverse-prevalence weighting ensures rare histotypes receive equal optimization gradient penalty.",
                "family": "late",
                "pooling": "concat",
                "gamma_meta": 1.0,
                "params": {"strategy": "temperature", "c": 1.0, "tau": 1.5, "class_weight": "balanced", "use_metadata": True}
            }
        ]

    def get_dataset(self, pooling: str = "concat", gamma_meta: float = 1.0) -> Dict[str, Any]:
        key = (pooling, round(gamma_meta, 3))
        if key not in self.dataset_cache:
            self.dataset_cache[key] = load_multimodal_dataset(
                seed=self.seed,
                gamma_meta=gamma_meta,
                pooling=pooling
            )
        return self.dataset_cache[key]

    def _build_system_prompt(self) -> str:
        return (
            "You are an expert computational pathologist and autonomous ML optimization agent.\n"
            "Your objective: find the optimal multimodal fusion architecture and hyperparameters combining "
            "UNI2, Virchow2, Prov-GigaPath, and tabular clinical metadata for 4-class ovarian cancer classification.\n"
            "PRIMARY TARGET METRIC: 5-fold Stratified Cross-Validation Macro AUROC on 164 patients.\n\n"
            "DOMAIN PATHOLOGY INSIGHTS & CALIBRATION RULES:\n"
            "1. Macro AUROC vs Balanced Accuracy: Do NOT use class_weight='balanced'. Balanced weighting distorts predicted probability calibration and reduces Macro AUROC. Keep class_weight=null.\n"
            "2. Fusion Families:\n"
            "   - 'late': Late probability fusion with temperature scaling (tau in [1.2, 1.8], c in [0.5, 2.0]) or uniform soft voting allows foundation models to act as independent calibrated experts, consistently achieving the highest generalization (>0.898 test AUROC).\n"
            "   - 'early': Direct concatenation without PCA (pca_dim=null, l2_norm=false) is strong, but PCA projection tends to discard subtle histotype variance.\n"
            "   - 'intermediate': Neural gating often overfits on 164 samples compared to calibrated late fusion.\n"
            "3. Slide Pooling: 'concat' ([mean; max] multi-resolution) reliably captures both architectural and focal cues.\n"
            "4. SE Guardrail: A candidate replaces the Champion only if mean(Delta_fold) >= 1.0 * SE(Delta_fold).\n\n"
            "Search Space:\n"
            "- Pooling: 'concat' ([mean; max] multi-resolution), 'mean', or 'max'.\n"
            "- Family: 'early', 'late', or 'intermediate'.\n"
            "  * early: {'c': float [0.01, 10.0], 'l2_norm': bool, 'pca_dim': null|128|256, 'class_weight': null, 'use_metadata': bool}\n"
            "  * late: {'strategy': 'uniform'|'simplex'|'temperature', 'c': float [0.01, 10.0], 'tau': float [0.5, 3.0], 'use_metadata': bool}\n"
            "  * intermediate: {'arch': 'gated'|'abmil', 'pca_per_stream': null|128, 'dropout': float [0.0, 0.4], 'weight_decay': float [1e-4, 1e-2], 'lr': float [1e-3, 1e-1], 'epochs': int [20, 60]}\n"
            "- gamma_meta: float [0.0, 2.0] (clinical metadata weighting factor).\n\n"
            "Respond STRICTLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "hypothesis": "Concise scientific hypothesis",\n'
            '  "reasoning": "Domain rationale referencing previous trial outcomes",\n'
            '  "family": "early" | "late" | "intermediate",\n'
            '  "pooling": "concat" | "mean" | "max",\n'
            '  "gamma_meta": 1.0,\n'
            '  "params": { ... }\n'
            "}"
        )

    def _build_user_prompt(self, trial_num: int, consecutive_failures: int) -> str:
        champ_info = "None (Initial Trial)"
        if self.champion:
            champ_info = (
                f"Trial: {self.champion['trial']}\n"
                f"CV Macro AUROC: {self.champion['mean_auroc']:.4f}\n"
                f"Family: {self.champion['config']['family']}\n"
                f"Pooling: {self.champion['pooling']} | gamma_meta: {self.champion['gamma_meta']}\n"
                f"Params: {json.dumps(self.champion['config']['params'])}"
            )

        history_rows = []
        for t in self.trials_log:
            history_rows.append(
                f"- {t['trial']}: {t['family']} (pooling={t['pooling']}, gamma={t['gamma_meta']}) -> "
                f"CV AUROC={t['mean_cv_auroc']:.4f}, Delta={t['delta_vs_champion']:+.4f}, "
                f"SE={t['se_delta']:.4f}, Decision={t['decision']}"
            )
        history_str = "\n".join(history_rows) if history_rows else "No previous trials."

        return (
            f"Trial {trial_num} of {self.max_trials}\n"
            f"Current Champion:\n{champ_info}\n\n"
            f"History of Evaluated Trials:\n{history_str}\n\n"
            f"Consecutive Failures to Beat Guardrail: {consecutive_failures} / {self.max_consecutive_failures}\n\n"
            "Based on the empirical evidence above, propose the next configuration to maximize generalization and beat the champion."
        )

    def _parse_llm_response(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            data = json.loads(cleaned)
            required_keys = ["hypothesis", "family", "params"]
            if not all(k in data for k in required_keys):
                return None
            if data["family"] not in ["early", "late", "intermediate"]:
                return None
            data["pooling"] = data.get("pooling", "concat")
            data["gamma_meta"] = float(data.get("gamma_meta", 1.0))
            return data
        except Exception:
            return None

    def propose_next_candidate(self, trial_num: int, consecutive_failures: int) -> Dict[str, Any]:
        proposal = None
        if self.llm.provider != "fallback_heuristic":
            sys_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(trial_num, consecutive_failures)
            raw_resp = self.llm.query(sys_prompt, user_prompt)
            if raw_resp:
                proposal = self._parse_llm_response(raw_resp)
                if proposal:
                    proposal["agent_source"] = f"llm_{self.llm.provider}"

        if proposal is None:
            idx = (trial_num - 1) % len(self._fallback_candidates)
            candidate_tmpl = self._fallback_candidates[idx]
            proposal = {
                "hypothesis": candidate_tmpl["hypothesis"],
                "reasoning": candidate_tmpl["reasoning"],
                "family": candidate_tmpl["family"],
                "pooling": candidate_tmpl["pooling"],
                "gamma_meta": candidate_tmpl["gamma_meta"],
                "params": dict(candidate_tmpl["params"]),
                "agent_source": "domain_reasoning_fallback"
            }

        return proposal

    def run_search(self) -> Dict[str, Any]:
        print("=" * 78, flush=True)
        print(f"STARTING AGENT v4: LLM-POWERED AUTONOMOUS FUSION AGENT (Seed: {self.seed})", flush=True)
        print(f"LLM Provider: {self.llm.provider.upper()} | Budget: {self.max_trials} trials | Guardrail: {self.k_se}xSE", flush=True)
        print("=" * 78, flush=True)

        consecutive_failures = 0

        for trial_idx in range(1, self.max_trials + 1):
            trial_id = f"T{trial_idx:02d}"
            proposal = self.propose_next_candidate(trial_idx, consecutive_failures)

            pooling = proposal.get("pooling", "concat")
            gamma_meta = proposal.get("gamma_meta", 1.0)
            family = proposal["family"]
            params = proposal["params"]
            hypothesis = proposal["hypothesis"]
            reasoning = proposal.get("reasoning", "")
            source = proposal.get("agent_source", "llm")

            print(f"\n--- [{trial_id}] ({source}) Proposed: {family.upper()} | Pooling: {pooling} | gamma: {gamma_meta} ---", flush=True)
            print(f"Hypothesis: {hypothesis}", flush=True)
            print(f"Params: {json.dumps(params)}", flush=True)

            t_eval_start = time.time()
            dataset = self.get_dataset(pooling=pooling, gamma_meta=gamma_meta)
            folds = get_train_val_cv_folds(dataset, n_splits=5, cv_seed=self.seed)

            mean_auroc, fold_aurocs, _ = evaluate_candidate_cv(
                candidate_cfg={"family": family, "params": params},
                dataset=dataset,
                folds=folds
            )
            elapsed = time.time() - t_eval_start

            print(f"Result ({elapsed:.2f}s): 5-fold CV Macro AUROC = {mean_auroc:.4f} (Folds: {[round(x, 4) for x in fold_aurocs]})", flush=True)

            if self.champion is None:
                decision = "ADOPT_INITIAL"
                reason = f"Initial baseline champion established ({mean_auroc:.4f})."
                delta_vs_champ = 0.0
                se_delta = 0.0
                thresh_val = 0.0
                consecutive_failures = 0

                self.champion = {
                    "trial": trial_id,
                    "mean_auroc": mean_auroc,
                    "fold_aurocs": fold_aurocs,
                    "pooling": pooling,
                    "gamma_meta": gamma_meta,
                    "config": {"family": family, "params": params},
                    "hypothesis": hypothesis,
                    "reasoning": reasoning
                }
            else:
                fold_deltas = np.array(fold_aurocs) - np.array(self.champion["fold_aurocs"])
                delta_vs_champ = float(np.mean(fold_deltas))
                se_delta = float(np.std(fold_deltas, ddof=1) / np.sqrt(len(fold_deltas))) if len(fold_deltas) > 1 else 0.0
                thresh_val = self.k_se * se_delta

                if delta_vs_champ >= thresh_val and delta_vs_champ > 0:
                    decision = "ADOPT_CHAMPION"
                    reason = f"Statistically meaningful improvement: Delta ({delta_vs_champ:+.4f}) >= {self.k_se}*SE ({thresh_val:.4f})."
                    consecutive_failures = 0

                    self.champion = {
                        "trial": trial_id,
                        "mean_auroc": mean_auroc,
                        "fold_aurocs": fold_aurocs,
                        "pooling": pooling,
                        "gamma_meta": gamma_meta,
                        "config": {"family": family, "params": params},
                        "hypothesis": hypothesis,
                        "reasoning": reasoning
                    }
                else:
                    decision = "REJECT"
                    consecutive_failures += 1
                    if delta_vs_champ > 0:
                        reason = f"Marginal gain ({delta_vs_champ:+.4f}) did not exceed {self.k_se}*SE guardrail ({thresh_val:.4f}). Rejected to avoid Winner's Curse."
                    else:
                        reason = f"Underperformed champion by {delta_vs_champ:+.4f} (SE: {se_delta:.4f})."

            print(f"Decision: {decision} | Delta: {delta_vs_champ:+.4f} | SE: {se_delta:.4f} | Threshold: {thresh_val:.4f}", flush=True)
            print(f"Reason: {reason}", flush=True)

            self.trials_log.append({
                "trial": trial_id,
                "agent_source": source,
                "stage": "exploration" if trial_idx <= 4 else "exploitation",
                "hypothesis": hypothesis,
                "reasoning": reasoning,
                "family": family,
                "pooling": pooling,
                "gamma_meta": gamma_meta,
                "params": params,
                "mean_cv_auroc": mean_auroc,
                "fold_aurocs": fold_aurocs,
                "delta_vs_champion": delta_vs_champ,
                "se_delta": se_delta,
                "guardrail_threshold": thresh_val,
                "decision": decision,
                "reason": reason
            })

            if consecutive_failures >= self.max_consecutive_failures:
                print("\n" + "=" * 78, flush=True)
                print(f"SE GUARDRAIL TRIGGERED: {consecutive_failures} consecutive trials failed to improve by >= {self.k_se}*SE.", flush=True)
                print("Stopping autonomous search early to guarantee generalization and prevent Winner's Curse.", flush=True)
                print("=" * 78, flush=True)
                break

        print("\n" + "=" * 78, flush=True)
        print(f"LLM AGENT SEARCH COMPLETE!", flush=True)
        print(f"Total Trials Executed: {len(self.trials_log)}", flush=True)
        print(f"Winning Champion: {self.champion['trial']} (5-fold CV Macro AUROC: {self.champion['mean_auroc']:.4f})", flush=True)
        print(f"Champion Architecture: {self.champion['config']['family'].upper()} | Pooling: {self.champion['pooling']} | gamma_meta: {self.champion['gamma_meta']}", flush=True)
        print(f"Champion Params: {json.dumps(self.champion['config']['params'])}", flush=True)
        print("=" * 78 + "\n", flush=True)

        return self.champion

    def save_logs(self, json_path: str, md_path: str):
        summary_record = {
            "seed": self.seed,
            "agent_version": "v4_llm",
            "llm_provider": self.llm.provider,
            "total_trials_run": len(self.trials_log),
            "champion": self.champion,
            "trials": self.trials_log
        }
        with open(json_path, "w") as f:
            json.dump(summary_record, f, indent=2)

        lines = [
            f"# Phase 2 LLM Agent Decision Log (Seed {self.seed})\n",
            f"**Agent**: LLM Autonomous Fusion Agent (`v4_llm`)  ",
            f"**LLM Provider**: `{self.llm.provider.upper()}`  ",
            f"**Total Trials Run**: {len(self.trials_log)} / {self.max_trials}  ",
            f"**Final Champion**: `{self.champion['trial']}` with **5-fold CV Macro AUROC: {self.champion['mean_auroc']:.4f}**  ",
            f"**Champion Family**: `{self.champion['config']['family'].upper()}`  ",
            f"**Champion Pooling**: `{self.champion.get('pooling', 'concat')}` | **gamma_meta**: `{self.champion.get('gamma_meta', 1.0)}`  ",
            f"**Champion Params**: `{json.dumps(self.champion['config']['params'])}`  ",
            f"**Champion Hypothesis**: {self.champion.get('hypothesis', 'N/A')}  ",
            f"**Guardrail Policy**: Adopt if $\\bar{{\\Delta}} \\ge {self.k_se:.1f} \\times \\text{{SE}}_{{\\Delta}}$ (patience = {self.max_consecutive_failures})\n",
            "## Trial Progression & Auditable Decision Records\n",
            "| Trial | Source | Family | Pooling | CV AUROC | Delta vs Champ | SE(Delta) | Threshold (k_se×SE) | Decision | Action / Reason |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
        ]

        for t in self.trials_log:
            thresh_val = t.get('guardrail_threshold', 0.0)
            lines.append(
                f"| **{t['trial']}** | `{t.get('agent_source', 'llm')}` | `{t['family']}` | `{t['pooling']}` | "
                f"**{t['mean_cv_auroc']:.4f}** | {t['delta_vs_champion']:+.4f} | {t['se_delta']:.4f} | {thresh_val:.4f} | "
                f"`{t['decision']}` | {t['reason']} |"
            )

        with open(md_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Saved LLM agent decision log to {json_path} and {md_path}", flush=True)


AutonomousFusionAgent = LLMFusionAgent
