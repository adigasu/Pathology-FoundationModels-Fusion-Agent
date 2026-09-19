"""
Agent v5: Unified Agent (search_v5_unified.py)
Synthesizes the empirically validated strengths of v1-v4:
- Stage 1: Decoupled statistical pooling screen (mean, max, concat, top20, mean_max_std) to isolate optimal representation R*.
- Stage 2: Closed-loop autonomous domain reasoning on R* with Paired 1.0-SE Guardrail and 5-trial early stopping.
- Stage 3: Multi-seed 15-fold cross-validation grid ranking finalists via Risk-Adjusted Stability Utility:
  U_stability = mu_CV - 1.0 * SE_CV
"""

from __future__ import annotations
import os
import sys
import json
import time
import math
import random
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.fusion.data import load_multimodal_dataset, get_train_val_cv_folds, CLASS_NAMES
from src.fusion.models import EarlyFusionClassifier, LateFusionClassifier, IntermediateFusionClassifier


def evaluate_candidate_cv(
    config: Dict[str, Any],
    dataset: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    gamma_meta: float = 1.0
) -> Tuple[float, List[float], np.ndarray]:
    """
    Evaluates candidate configuration across cross-validation folds.
    Returns: (mean_macro_auroc, fold_aurocs, oof_predictions)
    """
    family = config["family"]
    params = config.get("params", {})
    features = dataset["features"]
    metadata = dataset["metadata"] * gamma_meta
    y = dataset["y"]
    n_classes = len(CLASS_NAMES)
    n_samples = len(y)

    oof_preds = np.zeros((n_samples, n_classes), dtype=np.float32)
    fold_aurocs = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_feats = {k: v[train_idx] for k, v in features.items()}
        val_feats = {k: v[val_idx] for k, v in features.items()}
        train_meta = metadata[train_idx]
        val_meta = metadata[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        if family == "early":
            clf = EarlyFusionClassifier(
                c=params.get("c", 1.0),
                l2_norm_per_stream=params.get("l2_norm", False),
                pca_dim=params.get("pca_dim", None),
                use_metadata=params.get("use_metadata", True),
                class_weight=params.get("class_weight", None),
                random_state=42 + fold_idx
            )
        elif family == "late":
            clf = LateFusionClassifier(
                strategy=params.get("strategy", "uniform"),
                c=params.get("c", 1.0),
                tau=params.get("tau", 1.5),
                use_metadata=params.get("use_metadata", True),
                class_weight=params.get("class_weight", None),
                random_state=42 + fold_idx
            )
        elif family == "intermediate":
            clf = IntermediateFusionClassifier(
                arch=params.get("arch", "gated"),
                pca_per_stream=params.get("pca_per_stream", None),
                use_metadata=params.get("use_metadata", True),
                dropout=params.get("dropout", 0.2),
                weight_decay=params.get("weight_decay", 1e-3),
                lr=params.get("lr", 1e-2),
                epochs=params.get("epochs", 40),
                class_weight=params.get("class_weight", None),
                random_state=42 + fold_idx
            )
        else:
            raise ValueError(f"Unknown family: {family}")

        clf.fit(train_feats, train_meta, y_train)
        probs = clf.predict_proba(val_feats, val_meta)
        oof_preds[val_idx] = probs

        fold_auc_list = []
        for c_idx in range(n_classes):
            if len(np.unique(y_val == c_idx)) > 1:
                auc_c = roc_auc_score((y_val == c_idx).astype(int), probs[:, c_idx])
                fold_auc_list.append(auc_c)
        fold_macro = float(np.mean(fold_auc_list)) if fold_auc_list else 0.5
        fold_aurocs.append(fold_macro)

    mean_auroc = float(np.mean(fold_aurocs))
    return mean_auroc, fold_aurocs, oof_preds


class UnifiedAgent:
    """
    Agent v5: Unified Agent synthesizing:
    - Stage 1: Decoupled pooling representation screen (5 candidates) -> freezes R*
    - Stage 2: Autonomous closed-loop domain reasoner on R* with 1.0-SE guardrail & early stopping (patience=5)
    - Stage 3: Multi-seed 15-fold cross-validation grid ranking finalists via Risk-Adjusted Stability Utility
    """

    def __init__(
        self,
        seed: int = 42,
        k_se: float = 1.0,
        max_consecutive_failures: int = 5,
        max_stage2_trials: int = 20,
        stability_seeds: List[int] = (42, 1337, 2026)
    ):
        self.seed = seed
        self.k_se = k_se
        self.max_consecutive_failures = max_consecutive_failures
        self.max_stage2_trials = max_stage2_trials
        self.stability_seeds = list(stability_seeds)

        self.trials_log: List[Dict[str, Any]] = []
        self.champion: Optional[Dict[str, Any]] = None
        self.dataset_cache: Dict[str, Any] = {}
        self.winning_pooling: str = "concat_mean_max"
        self.finalists: List[Dict[str, Any]] = []

        self._domain_hypotheses = [
            {
                "hypothesis": "Late Fusion with Calibrated Temperature Scaling (tau=1.5) on optimal representation R*",
                "reasoning": "Calibrated temperature scaling dampens overconfident posteriors on ambiguous transitional subtypes.",
                "family": "late",
                "gamma_meta": 1.0,
                "params": {"strategy": "temperature", "c": 1.0, "tau": 1.5, "use_metadata": True}
            },
            {
                "hypothesis": "Late Fusion with Moderate Temperature Scaling (tau=1.2) on R*",
                "reasoning": "Slightly sharper temperature preserves high-confidence predictions on distinct solid sheets.",
                "family": "late",
                "gamma_meta": 1.0,
                "params": {"strategy": "temperature", "c": 1.0, "tau": 1.2, "use_metadata": True}
            },
            {
                "hypothesis": "Upweight clinical metadata prior (gamma_meta=1.5) on Calibrated Late Fusion",
                "reasoning": "Clinical staging and age offer diagnostic priors for In situ vs Invasive adenocarcinoma.",
                "family": "late",
                "gamma_meta": 1.5,
                "params": {"strategy": "temperature", "c": 1.0, "tau": 1.5, "use_metadata": True}
            },
            {
                "hypothesis": "Strong clinical metadata scaling (gamma_meta=2.0) with tau=1.5",
                "reasoning": "Test if stronger epidemiological weighting compensates for small cohort minority subtypes.",
                "family": "late",
                "gamma_meta": 2.0,
                "params": {"strategy": "temperature", "c": 1.0, "tau": 1.5, "use_metadata": True}
            },
            {
                "hypothesis": "Early Concatenation of foundation representations with L2 regularization C=1.0",
                "reasoning": "Direct joint feature interaction in early linear classifier to compare with late decision voting.",
                "family": "early",
                "gamma_meta": 1.0,
                "params": {"c": 1.0, "l2_norm": False, "pca_dim": None, "use_metadata": True}
            },
            {
                "hypothesis": "Early Concatenation with per-stream L2 normalization",
                "reasoning": "Balances gradient and magnitude discrepancies across 1536-d and 2560-d foundation backbones.",
                "family": "early",
                "gamma_meta": 1.0,
                "params": {"c": 1.0, "l2_norm": True, "pca_dim": None, "use_metadata": True}
            },
            {
                "hypothesis": "Balanced class weighting on Late Temperature Fusion",
                "reasoning": "Compensates for extreme class imbalance in Cribriform and In situ subtypes.",
                "family": "late",
                "gamma_meta": 1.0,
                "params": {"strategy": "temperature", "c": 1.0, "tau": 1.5, "class_weight": "balanced", "use_metadata": True}
            },
            {
                "hypothesis": "Conservative Temperature Scaling (tau=2.0) on Late Fusion",
                "reasoning": "Higher temperature enforces smoother probability consensus across foundation models.",
                "family": "late",
                "gamma_meta": 1.0,
                "params": {"strategy": "temperature", "c": 1.0, "tau": 2.0, "use_metadata": True}
            },
            {
                "hypothesis": "Stream-level Gated Neural Fusion (learnable non-linear routing)",
                "reasoning": "Test whether dynamic stream-gating outperforms linear late voting on R*.",
                "family": "intermediate",
                "gamma_meta": 1.0,
                "params": {"arch": "gated", "pca_per_stream": None, "dropout": 0.2, "weight_decay": 1e-3, "lr": 1e-2, "epochs": 40, "use_metadata": True}
            },
            {
                "hypothesis": "Early Concatenation with PCA dimensionality reduction (256 components, 95% variance)",
                "reasoning": "Compresses concatenated foundation representations to remove collinear feature dimensions.",
                "family": "early",
                "gamma_meta": 1.0,
                "params": {"c": 1.0, "l2_norm": False, "pca_dim": 256, "use_metadata": True}
            },
            {
                "hypothesis": "Stronger L2 regularization (C=0.1) on Calibrated Late Fusion",
                "reasoning": "Higher shrinkage reduces parameter variance across small patient training partitions.",
                "family": "late",
                "gamma_meta": 1.0,
                "params": {"strategy": "temperature", "c": 0.1, "tau": 1.5, "use_metadata": True}
            },
            {
                "hypothesis": "Milder L2 regularization (C=10.0) on Calibrated Late Fusion",
                "reasoning": "Lower regularization allows classifier to exploit subtle discriminating features.",
                "family": "late",
                "gamma_meta": 1.0,
                "params": {"strategy": "temperature", "c": 10.0, "tau": 1.5, "use_metadata": True}
            }
        ]

    def get_dataset(self, pooling: str = "concat_mean_max", gamma_meta: float = 1.0, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is None:
            seed = self.seed
        key = f"{seed}_{pooling}_{round(gamma_meta, 3)}"
        if key not in self.dataset_cache:
            self.dataset_cache[key] = load_multimodal_dataset(
                seed=seed,
                gamma_meta=gamma_meta,
                pooling=pooling
            )
        return self.dataset_cache[key]

    def run_stage1_pooling(self) -> str:
        """
        Stage 1: Decoupled Statistical Slide Representation Screen.
        Evaluates 5 pooling representations under a uniform baseline:
        - mean: Spatial mean across tiles
        - max: Element-wise maximum across tiles
        - concat: Dual [mean; max] multi-resolution representation
        - top20: Top-20% norm tile mean
        - mean_max_std: Tri-statistic [mean; max; std] (location + extrema + dispersion)
        """
        print("=" * 80, flush=True)
        print(f"STAGE 1: STATISTICAL POOLING REPRESENTATION SCREEN (Seed: {self.seed})", flush=True)
        print("=" * 80, flush=True)

        candidates = [
            {"trial": "T01", "name": "Spatial Mean", "pooling": "mean", "desc": "Spatial mean pooling across all tiles"},
            {"trial": "T02", "name": "Element Max", "pooling": "max", "desc": "Element-wise max pooling across all tiles"},
            {"trial": "T03", "name": "Concat [mean; max]", "pooling": "concat_mean_max", "desc": "Concatenated [mean; max] multi-resolution pooling"},
            {"trial": "T04", "name": "Top-20% Mean", "pooling": "top20", "desc": "Top-20% norm tile mean pooling"},
            {"trial": "T05", "name": "Concat [mean; max; std]", "pooling": "mean_max_std", "desc": "Tri-statistic [mean; max; std] (location + extrema + dispersion)"},
        ]

        base_cfg = {
            "family": "late",
            "params": {"strategy": "uniform", "c": 1.0, "use_metadata": True}
        }

        stage1_results = []
        for cand in candidates:
            t_start = time.time()
            ds = self.get_dataset(pooling=cand["pooling"], gamma_meta=1.0, seed=self.seed)
            folds = get_train_val_cv_folds(ds, n_splits=5, cv_seed=self.seed)

            eval_cfg = dict(base_cfg)
            eval_cfg["trial"] = cand["trial"]

            mean_auroc, fold_aurocs, _ = evaluate_candidate_cv(eval_cfg, ds, folds, gamma_meta=1.0)
            elapsed = time.time() - t_start

            record = {
                "trial": cand["trial"],
                "stage": "Stage 1 (Pooling Screen)",
                "hypothesis": cand["desc"],
                "family": "late",
                "params": eval_cfg["params"],
                "pooling": cand["pooling"],
                "gamma_meta": 1.0,
                "mean_cv_auroc": mean_auroc,
                "fold_cv_aurocs": fold_aurocs,
                "fold_std": float(np.std(fold_aurocs, ddof=1)),
                "decision": "EVALUATED",
                "time_sec": elapsed
            }
            stage1_results.append(record)
            self.trials_log.append(record)
            print(f"[{cand['trial']}] Pooling: {cand['name']:<24} | 5-Fold CV AUROC: {mean_auroc:.4f} (±{record['fold_std']:.4f}) in {elapsed:.2f}s", flush=True)

        best_cand = max(stage1_results, key=lambda x: x["mean_cv_auroc"])
        best_cand["decision"] = "STAGE1_WINNER"
        self.winning_pooling = best_cand["pooling"]

        print("-" * 80, flush=True)
        print(f"STAGE 1 WINNER: `{self.winning_pooling.upper()}` ({best_cand['hypothesis']}) with CV AUROC = {best_cand['mean_cv_auroc']:.4f}", flush=True)
        print(f"FREEZING REPRESENTATION R*: All subsequent trials will operate strictly on `{self.winning_pooling.upper()}`.", flush=True)
        print("=" * 80 + "\n", flush=True)
        return self.winning_pooling

    def run_stage2_reasoner(self) -> Dict[str, Any]:
        """
        Stage 2: Closed-Loop Autonomous Domain Reasoning on Frozen Representation R*.
        Conditions each candidate on history, enforces Paired 1.0-SE Guardrail,
        and triggers early stopping if 5 consecutive trials fail to improve by >= 1.0*SE.
        """
        print("=" * 80, flush=True)
        print(f"STAGE 2: CLOSED-LOOP AUTONOMOUS REASONING ON R*=`{self.winning_pooling.upper()}`", flush=True)
        print("=" * 80, flush=True)

        ds = self.get_dataset(pooling=self.winning_pooling, gamma_meta=1.0, seed=self.seed)
        folds = get_train_val_cv_folds(ds, n_splits=5, cv_seed=self.seed)

        # Establish Initial Baseline Champion on R*
        t0_start = time.time()
        init_cfg = {
            "trial": "T06",
            "hypothesis": f"Initial Baseline on R*=`{self.winning_pooling}`: Uniform Late Voting + Metadata",
            "family": "late",
            "params": {"strategy": "uniform", "c": 1.0, "use_metadata": True},
            "pooling": self.winning_pooling,
            "gamma_meta": 1.0
        }
        init_mean, init_folds, _ = evaluate_candidate_cv(init_cfg, ds, folds, gamma_meta=1.0)
        t0_elapsed = time.time() - t0_start

        self.champion = {
            "trial": "T06",
            "stage": "Stage 2 (Reasoner)",
            "config": {
                "family": init_cfg["family"],
                "params": init_cfg["params"]
            },
            "mean_auroc": init_mean,
            "fold_aurocs": init_folds,
            "pooling": self.winning_pooling,
            "gamma_meta": 1.0,
            "hypothesis": init_cfg["hypothesis"]
        }

        init_rec = {
            "trial": "T06",
            "stage": "Stage 2 (Reasoner)",
            "hypothesis": init_cfg["hypothesis"],
            "family": init_cfg["family"],
            "params": init_cfg["params"],
            "pooling": self.winning_pooling,
            "gamma_meta": 1.0,
            "mean_cv_auroc": init_mean,
            "fold_cv_aurocs": init_folds,
            "delta_vs_champion": 0.0,
            "se_delta": 0.0,
            "guardrail_threshold": 0.0,
            "decision": "INITIAL_CHAMPION",
            "reason": "Baseline configuration established on frozen representation R*",
            "time_sec": t0_elapsed
        }
        self.trials_log.append(init_rec)
        print(f"[T06] Initial Baseline Champion Established: CV AUROC = {init_mean:.4f}", flush=True)

        consecutive_failures = 0
        all_stage2_candidates = [init_rec]

        trial_idx = 7
        for cand_tmpl in self._domain_hypotheses:
            if trial_idx > (5 + self.max_stage2_trials):
                print(f"Max trial budget reached ({self.max_stage2_trials} trials). Concluding Stage 2.", flush=True)
                break

            t_id = f"T{trial_idx:02d}"
            trial_idx += 1

            gamma_cand = cand_tmpl.get("gamma_meta", 1.0)
            ds_cand = self.get_dataset(pooling=self.winning_pooling, gamma_meta=gamma_cand, seed=self.seed)
            folds_cand = get_train_val_cv_folds(ds_cand, n_splits=5, cv_seed=self.seed)

            eval_cfg = {
                "trial": t_id,
                "family": cand_tmpl["family"],
                "params": cand_tmpl["params"]
            }

            t_start = time.time()
            mean_auroc, fold_aurocs, _ = evaluate_candidate_cv(eval_cfg, ds_cand, folds_cand, gamma_meta=gamma_cand)
            elapsed = time.time() - t_start

            # Paired 1.0-SE Guardrail evaluation
            champ_folds = np.array(self.champion["fold_aurocs"], dtype=np.float64)
            cand_folds = np.array(fold_aurocs, dtype=np.float64)
            deltas = cand_folds - champ_folds

            delta_mean = float(np.mean(deltas))
            delta_std = float(np.std(deltas, ddof=1))
            se_delta = float(delta_std / math.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
            guardrail_threshold = self.k_se * se_delta

            if delta_mean >= guardrail_threshold and delta_mean > 0:
                decision = "ADOPT_CHAMPION"
                action_reason = f"Statistically meaningful improvement: Delta ({delta_mean:+.4f}) >= 1.0*SE ({guardrail_threshold:.4f})."
                self.champion = {
                    "trial": t_id,
                    "stage": "Stage 2 (Reasoner)",
                    "config": {
                        "family": cand_tmpl["family"],
                        "params": cand_tmpl["params"]
                    },
                    "mean_auroc": mean_auroc,
                    "fold_aurocs": fold_aurocs,
                    "pooling": self.winning_pooling,
                    "gamma_meta": gamma_cand,
                    "hypothesis": cand_tmpl["hypothesis"],
                    "reasoning": cand_tmpl["reasoning"]
                }
                consecutive_failures = 0
            else:
                decision = "REJECT"
                if delta_mean > 0:
                    action_reason = f"Marginal gain ({delta_mean:+.4f}) < 1.0*SE ({guardrail_threshold:.4f}). Rejected to avert Winner's Curse."
                else:
                    action_reason = f"Underperformed champion by {delta_mean:+.4f} (SE: {se_delta:.4f})."
                consecutive_failures += 1

            record = {
                "trial": t_id,
                "stage": "Stage 2 (Reasoner)",
                "hypothesis": cand_tmpl["hypothesis"],
                "reasoning": cand_tmpl["reasoning"],
                "family": cand_tmpl["family"],
                "params": cand_tmpl["params"],
                "pooling": self.winning_pooling,
                "gamma_meta": gamma_cand,
                "mean_cv_auroc": mean_auroc,
                "fold_cv_aurocs": fold_aurocs,
                "delta_vs_champion": delta_mean,
                "se_delta": se_delta,
                "guardrail_threshold": guardrail_threshold,
                "decision": decision,
                "reason": action_reason,
                "time_sec": elapsed
            }
            self.trials_log.append(record)
            all_stage2_candidates.append(record)

            print(f"[{t_id}] {cand_tmpl['family'].upper():<12} | gamma={gamma_cand:.1f} | CV AUROC: {mean_auroc:.4f} | Delta: {delta_mean:+.4f} (SE: {se_delta:.4f}) | {decision} | Failures: {consecutive_failures}/{self.max_consecutive_failures}", flush=True)

            if consecutive_failures >= self.max_consecutive_failures:
                print("=" * 80, flush=True)
                print(f"SE GUARDRAIL TRIGGERED: {self.max_consecutive_failures} consecutive trials failed to improve by >= 1.0*SE.", flush=True)
                print("Halting Stage 2 early to prevent Winner's Curse and validation split memorization.", flush=True)
                print("=" * 80, flush=True)
                break

        # Select Top-2 distinct finalists for Stage 3 stability validation
        unique_configs = []
        seen_keys = set()
        sorted_cands = sorted(all_stage2_candidates, key=lambda x: x["mean_cv_auroc"], reverse=True)

        for c in sorted_cands:
            key = (c["family"], c.get("gamma_meta", 1.0), json.dumps(c["params"], sort_keys=True))
            if key not in seen_keys:
                seen_keys.add(key)
                unique_configs.append(c)
            if len(unique_configs) >= 2:
                break

        self.finalists = unique_configs
        print("-" * 80, flush=True)
        print(f"STAGE 2 COMPLETE. Top Finalists Selected for Stage 3 Stability Verification:", flush=True)
        for i, fin in enumerate(self.finalists, 1):
            print(f"  Finalist {i}: [{fin['trial']}] {fin['hypothesis']} | CV AUROC: {fin['mean_cv_auroc']:.4f}", flush=True)
        print("=" * 80 + "\n", flush=True)
        return self.champion

    def run_stage3_stability(self) -> Dict[str, Any]:
        """
        Stage 3: Multi-Seed 15-Fold Risk-Adjusted Stability Utility Gate.
        Evaluates the top finalists across 15 folds (5 folds x 3 seeds) and ranks them by:
        U_stability = mu_CV - 1.0 * SE_CV
        """
        print("=" * 80, flush=True)
        print(f"STAGE 3: MULTI-SEED 15-FOLD RISK-ADJUSTED STABILITY UTILITY GATE", flush=True)
        print(f"Evaluation Grid: 3 seeds {self.stability_seeds} x 5 folds = 15 total folds per finalist", flush=True)
        print("=" * 80, flush=True)

        finalist_stability_scores = []
        trial_num = len(self.trials_log) + 1

        for idx, fin in enumerate(self.finalists, 1):
            t_id = f"T{trial_num:02d}"
            trial_num += 1

            all_15_fold_scores = []
            seed_means = []
            gamma_val = fin.get("gamma_meta", 1.0)

            t_start = time.time()
            for s in self.stability_seeds:
                ds_s = self.get_dataset(pooling=self.winning_pooling, gamma_meta=gamma_val, seed=s)
                folds_s = get_train_val_cv_folds(ds_s, n_splits=5, cv_seed=s)

                eval_cfg = {
                    "family": fin["family"],
                    "params": fin["params"]
                }
                mean_s, fold_aurocs_s, _ = evaluate_candidate_cv(eval_cfg, ds_s, folds_s, gamma_meta=gamma_val)
                all_15_fold_scores.extend(fold_aurocs_s)
                seed_means.append(mean_s)
            elapsed = time.time() - t_start

            grand_mean = float(np.mean(all_15_fold_scores))
            grand_std = float(np.std(all_15_fold_scores, ddof=1))
            grand_se = float(grand_std / math.sqrt(len(all_15_fold_scores)))
            stability_utility = grand_mean - (self.k_se * grand_se)

            stab_record = {
                "trial": t_id,
                "stage": "Stage 3 (Stability Gate)",
                "finalist_index": idx,
                "hypothesis": f"15-Fold Stability: {fin['hypothesis']}",
                "family": fin["family"],
                "params": fin["params"],
                "pooling": self.winning_pooling,
                "gamma_meta": gamma_val,
                "seed_means": seed_means,
                "grand_mean_auroc": grand_mean,
                "grand_std": grand_std,
                "grand_se": grand_se,
                "stability_utility": stability_utility,
                "mean_cv_auroc": grand_mean,
                "fold_cv_aurocs": all_15_fold_scores[:5],
                "time_sec": elapsed
            }
            finalist_stability_scores.append(stab_record)
            self.trials_log.append(stab_record)

            print(f"[{t_id}] Finalist {idx}: {fin['hypothesis']:<48} | 15-Fold Mean: {grand_mean:.4f} ± {grand_std:.4f} (SE: {grand_se:.4f}) | Stability Utility: {stability_utility:.4f}", flush=True)

        winner = max(finalist_stability_scores, key=lambda x: x["stability_utility"])
        self.champion = {
            "trial": winner["trial"],
            "stage": "Final Champion (Stage 3)",
            "config": {
                "family": winner["family"],
                "params": winner["params"]
            },
            "mean_auroc": winner["grand_mean_auroc"],
            "stability_utility": winner["stability_utility"],
            "grand_se": winner["grand_se"],
            "pooling": self.winning_pooling,
            "gamma_meta": winner["gamma_meta"]
        }

        print("-" * 80, flush=True)
        print(f"UNIFIED AGENT CHAMPION VALIDATED: [{self.champion['trial']}] with 15-Fold Grand CV AUROC = {self.champion['mean_auroc']:.4f} (Stability Utility: {self.champion['stability_utility']:.4f})", flush=True)
        print(f"Architecture: Family={self.champion['config']['family']}, Pooling={self.winning_pooling.upper()}, gamma_meta={self.champion['gamma_meta']}", flush=True)
        print(f"Params: {json.dumps(self.champion['config']['params'])}", flush=True)
        print("=" * 80 + "\n", flush=True)
        return self.champion

    def run_search(self) -> Dict[str, Any]:
        """Runs complete Agent v5 3-stage unified search pipeline."""
        t_total = time.time()
        self.run_stage1_pooling()
        self.run_stage2_reasoner()
        self.run_stage3_stability()
        total_time = time.time() - t_total
        print(f"Agent v5 Unified Agent Search completed in {total_time:.2f} seconds.", flush=True)
        return self.champion

    def save_logs(self, json_path: str, md_path: str) -> None:
        """Saves machine-readable JSON logs and human-readable Markdown summary."""
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        os.makedirs(os.path.dirname(md_path), exist_ok=True)

        summary_data = {
            "seed": self.seed,
            "agent_version": "v5",
            "winning_pooling": self.winning_pooling,
            "total_trials_run": len(self.trials_log),
            "champion": self.champion,
            "trials": self.trials_log
        }
        with open(json_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        md_lines = [
            f"# Phase 2 Agent v5 Decision Log (Unified Agent - Seed {self.seed})\n",
            f"**Architecture**: 3-Stage Unified Search (Screen -> Reason -> 15-Fold Stability Gate)  ",
            f"**Total Trials Run**: {len(self.trials_log)}  ",
            f"**Frozen Optimal Pooling (Stage 1)**: `{self.winning_pooling.upper()}`  ",
            f"**Final Champion (Stage 3)**: `{self.champion['trial']}` with **15-Fold Grand Mean CV AUROC: {self.champion['mean_auroc']:.4f}**  ",
            f"**Risk-Adjusted Stability Utility**: `{self.champion['stability_utility']:.4f}` (Grand Mean - 1.0*SE)  ",
            f"**Champion Family**: `{self.champion['config']['family']}` | **gamma_meta**: `{self.champion['gamma_meta']}`  ",
            f"**Champion Params**: `{json.dumps(self.champion['config']['params'])}`\n",
            "## Complete Trial Progression across All 3 Stages\n",
            "| Trial | Stage | Hypothesis | Family | Pooling | Gamma | CV AUROC | Delta vs Champ | SE(Delta) | Decision | Action / Reason |",
            "| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
        ]

        for t in self.trials_log:
            delta_str = f"{t.get('delta_vs_champion', 0.0):+.4f}" if "delta_vs_champion" in t else "-"
            se_str = f"{t.get('se_delta', 0.0):.4f}" if "se_delta" in t else "-"
            reason_str = t.get("reason", t.get("decision", "-"))
            gamma_str = f"{t.get('gamma_meta', 1.0):.1f}"
            md_lines.append(
                f"| **{t['trial']}** | {t['stage']} | {t['hypothesis']} | {t['family']} | "
                f"`{t['pooling']}` | {gamma_str} | **{t['mean_cv_auroc']:.4f}** | {delta_str} | {se_str} | "
                f"`{t.get('decision', 'EVAL')}` | {reason_str} |"
            )

        with open(md_path, "w") as f:
            f.write("\n".join(md_lines) + "\n")


# Backward-compatible alias
UnifiedProductionAgent = UnifiedAgent
