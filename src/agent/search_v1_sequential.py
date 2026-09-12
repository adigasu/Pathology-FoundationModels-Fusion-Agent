import os
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from src.fusion.data import load_multimodal_dataset, get_train_val_cv_folds, CLASS_NAMES
from src.fusion.models import EarlyFusionClassifier, LateFusionClassifier, IntermediateFusionClassifier
from src.eval.metrics import compute_multiclass_metrics
from src.eval.bootstrap import compute_patient_bootstrap_ci


def evaluate_candidate_cv(
    candidate_cfg: Dict[str, Any],
    dataset: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[float, List[float], List[Dict[str, Any]]]:
    """
    Evaluates candidate configuration on 5-fold Stratified Cross-Validation on Train+Val.
    Returns:
      mean_macro_auroc: float
      fold_aurocs: List[float]
      fold_metrics: List[Dict]
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
                c=params.get("c", 1.0),
                l2_norm_per_stream=params.get("l2_norm", False),
                pca_dim=params.get("pca_dim", None),
                use_metadata=params.get("use_metadata", True),
                class_weight=params.get("class_weight", None),
                random_state=42 + fold_idx
            )
        elif family == "late":
            model = LateFusionClassifier(
                strategy=params.get("strategy", "uniform"),
                c=params.get("c", 1.0),
                tau=params.get("tau", 1.5),
                use_metadata=params.get("use_metadata", True),
                class_weight=params.get("class_weight", None),
                random_state=42 + fold_idx
            )
        elif family == "intermediate":
            model = IntermediateFusionClassifier(
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

        model.fit(tr_feats, tr_meta, tr_y)
        pred_proba = model.predict_proba(val_feats, val_meta)
        m = compute_multiclass_metrics(val_y, pred_proba, class_names=CLASS_NAMES)
        fold_aurocs.append(m["macro_auroc"])
        fold_metrics.append(m)

    mean_auroc = float(np.mean(fold_aurocs))
    return mean_auroc, fold_aurocs, fold_metrics


class AutonomousFusionAgent:
    """
    Autonomous deterministic search agent governed by a hard 25-trial budget,
    paired standard error guardrails, touch-test-set-once rule, and zero leakage.
    """
    def __init__(self, seed: int = 42, k_se: float = 0.5, max_consecutive_failures: int = 5):
        self.seed = seed
        self.k_se = k_se
        self.max_consecutive_failures = max_consecutive_failures
        self.trials_log: List[Dict[str, Any]] = []
        self.champion: Optional[Dict[str, Any]] = None
        self.dataset_cache: Dict[str, Any] = {}

    def get_dataset(self, pooling: str = "mean", gamma_meta: float = 1.0) -> Dict[str, Any]:
        key = f"{pooling}_{gamma_meta}"
        if key not in self.dataset_cache:
            self.dataset_cache[key] = load_multimodal_dataset(
                seed=self.seed,
                gamma_meta=gamma_meta,
                pooling=pooling
            )
        return self.dataset_cache[key]

    def run_exploration(self, dataset: Dict[str, Any], folds: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Stage 1: Exploration (Trials T01 - T12)
        """
        exploration_configs = [
            # T01: Early raw 5632-d + L2 Logistic Regression + Metadata
            {"trial": "T01", "family": "early", "tier": "Early", "hypothesis": "Early Concatenation (Raw 5632-d) + L2 LogReg + Metadata",
             "params": {"c": 1.0, "l2_norm": False, "pca_dim": None, "use_metadata": True, "class_weight": None}},
            # T02: Early per-stream L2 norm
            {"trial": "T02", "family": "early", "tier": "Early", "hypothesis": "Per-stream L2-normalization before concatenation + Metadata",
             "params": {"c": 1.0, "l2_norm": True, "pca_dim": None, "use_metadata": True, "class_weight": None}},
            # T03: Early train-set PCA (256-d)
            {"trial": "T03", "family": "early", "tier": "Early", "hypothesis": "Train-set PCA (256-d, 95% variance) + L2 LogReg + Metadata",
             "params": {"c": 1.0, "l2_norm": False, "pca_dim": 256, "use_metadata": True, "class_weight": None}},
            # T04: Early ablation without metadata
            {"trial": "T04", "family": "early", "tier": "Early", "hypothesis": "Ablation: Top Early configuration without metadata",
             "params": {"c": 1.0, "l2_norm": True, "pca_dim": 256, "use_metadata": False, "class_weight": None}},
            # T05: Late uniform probability averaging
            {"trial": "T05", "family": "late", "tier": "Late", "hypothesis": "Uniform Softmax Probability Averaging (1/3 weight each) + Metadata head",
             "params": {"strategy": "uniform", "c": 1.0, "use_metadata": True, "class_weight": None}},
            # T06: Late temperature-scaled voting (tau=1.5)
            {"trial": "T06", "family": "late", "tier": "Late", "hypothesis": "Temperature-scaled soft voting (tau=1.5) + Metadata head",
             "params": {"strategy": "temperature", "c": 1.0, "tau": 1.5, "use_metadata": True, "class_weight": None}},
            # T07: Late simplex weights
            {"trial": "T07", "family": "late", "tier": "Late", "hypothesis": "Validation-optimized simplex weights (w1, w2, w3 >= 0, sum=1)",
             "params": {"strategy": "simplex", "c": 1.0, "use_metadata": True, "class_weight": None}},
            # T08: Late meta-stacking classifier
            {"trial": "T08", "family": "late", "tier": "Late", "hypothesis": "Meta-Stacking classifier (Logistic Regression on OOF probability vectors)",
             "params": {"strategy": "stacking", "c": 1.0, "use_metadata": True, "class_weight": None}},
            # T09: Intermediate stream-level gated fusion
            {"trial": "T09", "family": "intermediate", "tier": "Intermediate", "hypothesis": "Stream-level Gated Fusion (learnable alpha_m(x)) + Metadata",
             "params": {"arch": "gated", "dropout": 0.2, "epochs": 40, "use_metadata": True, "class_weight": None}},
            # T10: Intermediate Stream-ABMIL
            {"trial": "T10", "family": "intermediate", "tier": "Intermediate", "hypothesis": "Stream-ABMIL (multi-stream attention pooling across backbones)",
             "params": {"arch": "abmil", "epochs": 40, "use_metadata": True, "class_weight": None}},
            # T11: Intermediate PCA (128-d per stream) -> Gated Fusion
            {"trial": "T11", "family": "intermediate", "tier": "Intermediate", "hypothesis": "Train-set PCA (128-d per stream) -> Gated Fusion",
             "params": {"arch": "pca_gated", "pca_per_stream": 128, "epochs": 40, "use_metadata": True, "class_weight": None}},
            # T12: Intermediate ablation without metadata
            {"trial": "T12", "family": "intermediate", "tier": "Intermediate", "hypothesis": "Ablation: Top Intermediate configuration without metadata",
             "params": {"arch": "gated", "epochs": 40, "use_metadata": False, "class_weight": None}},
        ]

        for cfg in exploration_configs:
            t_start = time.time()
            mean_auroc, fold_aurocs, _ = evaluate_candidate_cv(cfg, dataset, folds)
            elapsed = time.time() - t_start

            # Guardrail evaluation
            if self.champion is None:
                decision = "INITIAL_CHAMPION"
                reason = "First baseline trial established as champion benchmark"
                delta_mean = 0.0
                se_delta = 0.0
                self.champion = {
                    "trial": cfg["trial"],
                    "config": cfg,
                    "mean_auroc": mean_auroc,
                    "fold_aurocs": fold_aurocs,
                    "pooling": "mean",
                    "gamma_meta": 1.0
                }
            else:
                # Paired fold-difference SE: SE(Delta) = std(cand_fold - champ_fold) / sqrt(K)
                deltas = np.array(fold_aurocs) - np.array(self.champion["fold_aurocs"])
                delta_mean = float(np.mean(deltas))
                se_delta = float(np.std(deltas, ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
                threshold = float(self.k_se * se_delta)

                if delta_mean >= threshold:
                    decision = "ADOPTED"
                    reason = f"Exceeded champion by delta={delta_mean:+.4f} >= {self.k_se:.1f}*SE ({threshold:.4f}, SE={se_delta:.4f})"
                    self.champion = {
                        "trial": cfg["trial"],
                        "config": cfg,
                        "mean_auroc": mean_auroc,
                        "fold_aurocs": fold_aurocs,
                        "pooling": "mean",
                        "gamma_meta": 1.0
                    }
                else:
                    decision = "REJECTED"
                    reason = f"Failed guardrail: delta={delta_mean:+.4f} < {self.k_se:.1f}*SE ({threshold:.4f}, SE={se_delta:.4f})"

            log_entry = {
                "trial": cfg["trial"],
                "stage": "Exploration",
                "tier": cfg["tier"],
                "hypothesis": cfg["hypothesis"],
                "family": cfg["family"],
                "params": cfg["params"],
                "mean_cv_auroc": mean_auroc,
                "fold_cv_aurocs": fold_aurocs,
                "delta_vs_champion": delta_mean,
                "se_delta": se_delta,
                "guardrail_threshold": threshold if 'threshold' in locals() else 0.0,
                "k_se": self.k_se,
                "paired_fold_deltas": deltas.tolist() if 'deltas' in locals() and isinstance(deltas, np.ndarray) else [],
                "decision": decision,
                "reason": reason,
                "champion_at_step": self.champion["trial"],
                "champion_mean_auroc": self.champion["mean_auroc"],
                "time_sec": elapsed
            }
            self.trials_log.append(log_entry)
            print(f"[{cfg['trial']}] {cfg['tier']:<12} | CV Macro AUROC: {mean_auroc:.4f} | Delta: {delta_mean:+.4f} | {decision:<10} | Champ: {self.champion['trial']} ({self.champion['mean_auroc']:.4f})", flush=True)

    def run_exploitation(self):
        """
        Stage 2: Focused Exploitation (Trials T13 - T25)
        Fine-tunes the winning paradigm from Stage 1 under SE Guardrails & Early Stopping.
        """
        # Determine winning family
        stage1_logs = [l for l in self.trials_log if l["stage"] == "Exploration"]
        family_scores = {"early": [], "late": [], "intermediate": []}
        for l in stage1_logs:
            family_scores[l["family"]].append(l["mean_cv_auroc"])

        avg_family_scores = {f: np.mean(scores) for f, scores in family_scores.items()}
        winning_family = max(avg_family_scores, key=avg_family_scores.get)
        print("=" * 75, flush=True)
        print(f"STAGE 1 COMPLETE. Winning Paradigm: {winning_family.upper()} (Mean CV: {avg_family_scores[winning_family]:.4f})", flush=True)
        print(f"Current Champion: {self.champion['trial']} (CV: {self.champion['mean_auroc']:.4f})", flush=True)
        print("=" * 75, flush=True)

        base_params = dict(self.champion["config"].get("params", {}))
        consecutive_failures = 0

        # Construct Stage 2 Search Plan (Structured exploitation across 4 dimensions: Pooling, Regularization/Strategy, Metadata Scale, Class Rebalancing)
        target_family = winning_family if avg_family_scores[winning_family] >= self.champion["mean_auroc"] else self.champion["config"]["family"]
        
        exploitation_trials = [
            # T13-T14: Pooling Strategy (Representation Capacity)
            {"trial": "T13", "category": "Pooling", "hypothesis": "Concatenated [mean; max] slide pooling representation", "pooling": "concat", "family": target_family},
            {"trial": "T14", "category": "Pooling", "hypothesis": "Slide max pooling representation", "pooling": "max", "family": target_family},
            # T15-T18: Voting Strategy & Regularization
            {"trial": "T15", "category": "Strategy", "hypothesis": "Temperature-scaled soft voting (tau=1.5)", "param_update": {"strategy": "temperature", "tau": 1.5, "c": 1.0}, "family": "late"},
            {"trial": "T16", "category": "Strategy", "hypothesis": "Calibrated temperature soft voting (tau=1.2)", "param_update": {"strategy": "temperature", "tau": 1.2, "c": 1.0}, "family": "late"},
            {"trial": "T17", "category": "Regularization", "hypothesis": "Regularization shrinkage C=0.5", "param_update": {"c": 0.5, "weight_decay": 2e-3}, "family": target_family},
            {"trial": "T18", "category": "Regularization", "hypothesis": "Regularization C=2.0", "param_update": {"c": 2.0, "weight_decay": 5e-4}, "family": target_family},
            # T19-T21: Metadata Scaling Factor gamma_meta
            {"trial": "T19", "category": "Metadata Scale", "hypothesis": "Metadata scaling factor gamma_meta=0.5", "gamma_meta": 0.5, "family": target_family},
            {"trial": "T20", "category": "Metadata Scale", "hypothesis": "Metadata scaling factor gamma_meta=0.2", "gamma_meta": 0.2, "family": target_family},
            {"trial": "T21", "category": "Metadata Scale", "hypothesis": "Metadata scaling factor gamma_meta=2.0", "gamma_meta": 2.0, "family": target_family},
            # T22-T25: Class-Prior Rebalancing & Refinement
            {"trial": "T22", "category": "Class Rebalance", "hypothesis": "Balanced class-prior loss weighting", "param_update": {"class_weight": "balanced"}, "family": target_family},
            {"trial": "T23", "category": "Class Rebalance", "hypothesis": "Uniform class weighting with optimal regularization", "param_update": {"class_weight": None, "c": 1.0}, "family": target_family},
            {"trial": "T24", "category": "Refinement", "hypothesis": "Simplex-optimized weights on concatenated representation", "param_update": {"strategy": "simplex"}, "family": "late"},
            {"trial": "T25", "category": "Refinement", "hypothesis": "Optimal temperature-scaled fusion + calibrated metadata", "param_update": {"strategy": "temperature", "tau": 1.5}, "gamma_meta": 1.0, "family": "late"},
        ]

        for plan in exploitation_trials:
            t_id = plan["trial"]
            t_start = time.time()

            # Set up pooling and gamma_meta
            curr_pooling = plan.get("pooling", self.champion.get("pooling", "mean"))
            curr_gamma = plan.get("gamma_meta", self.champion.get("gamma_meta", 1.0))

            ds = self.get_dataset(pooling=curr_pooling, gamma_meta=curr_gamma)
            folds = get_train_val_cv_folds(ds, n_splits=5, cv_seed=self.seed)

            # Build candidate config
            cand_params = dict(base_params)
            if "param_update" in plan:
                cand_params.update(plan["param_update"])

            cand_cfg = {
                "trial": t_id,
                "family": plan.get("family", self.champion["config"]["family"]),
                "tier": f"Exploitation ({plan['category']})",
                "hypothesis": plan["hypothesis"],
                "params": cand_params,
                "pooling": curr_pooling,
                "gamma_meta": curr_gamma
            }

            mean_auroc, fold_aurocs, _ = evaluate_candidate_cv(cand_cfg, ds, folds)
            elapsed = time.time() - t_start

            # Paired fold-difference SE Guardrail comparison: SE(Delta) = std(cand_fold - champ_fold) / sqrt(K)
            deltas = np.array(fold_aurocs) - np.array(self.champion["fold_aurocs"])
            delta_mean = float(np.mean(deltas))
            se_delta = float(np.std(deltas, ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
            threshold = float(self.k_se * se_delta)

            if delta_mean >= threshold:
                decision = "ADOPTED"
                reason = f"Exceeded champion by delta={delta_mean:+.4f} >= {self.k_se:.1f}*SE ({threshold:.4f}, SE={se_delta:.4f})"
                self.champion = {
                    "trial": t_id,
                    "config": cand_cfg,
                    "mean_auroc": mean_auroc,
                    "fold_aurocs": fold_aurocs,
                    "pooling": curr_pooling,
                    "gamma_meta": curr_gamma
                }
                base_params = dict(cand_params)
                consecutive_failures = 0
            else:
                decision = "REJECTED"
                reason = f"Failed guardrail: delta={delta_mean:+.4f} < {self.k_se:.1f}*SE ({threshold:.4f}, SE={se_delta:.4f})"
                consecutive_failures += 1

            log_entry = {
                "trial": t_id,
                "stage": "Exploitation",
                "tier": cand_cfg["tier"],
                "hypothesis": cand_cfg["hypothesis"],
                "family": cand_cfg["family"],
                "params": cand_cfg["params"],
                "pooling": curr_pooling,
                "gamma_meta": curr_gamma,
                "mean_cv_auroc": mean_auroc,
                "fold_cv_aurocs": fold_aurocs,
                "delta_vs_champion": delta_mean,
                "se_delta": se_delta,
                "guardrail_threshold": threshold if 'threshold' in locals() else 0.0,
                "k_se": self.k_se,
                "paired_fold_deltas": deltas.tolist() if 'deltas' in locals() and isinstance(deltas, np.ndarray) else [],
                "decision": decision,
                "reason": reason,
                "champion_at_step": self.champion["trial"],
                "champion_mean_auroc": self.champion["mean_auroc"],
                "consecutive_failures": consecutive_failures,
                "time_sec": elapsed
            }
            self.trials_log.append(log_entry)
            print(f"[{t_id}] {cand_cfg['tier']:<25} | CV AUROC: {mean_auroc:.4f} | Delta: {delta_mean:+.4f} (Thresh: {threshold:.4f}, SE: {se_delta:.4f}) | {decision:<10} | Champ: {self.champion['trial']} ({self.champion['mean_auroc']:.4f})", flush=True)

            # Check early stopping guardrail
            if consecutive_failures >= self.max_consecutive_failures:
                print("=" * 75, flush=True)
                print(f"SE GUARDRAIL TRIGGERED: {consecutive_failures} consecutive exploitation trials failed to exceed {self.k_se}*SE.", flush=True)
                print(f"Terminating exploitation search early to prevent Winner's Curse overfitting.", flush=True)
                print("=" * 75, flush=True)
                break

    def run_search(self) -> Dict[str, Any]:
        """
        Executes full 25-trial search loop.
        """
        print("=" * 75, flush=True)
        print(f"STARTING PHASE 2 AUTONOMOUS FUSION AGENT SEARCH (Seed: {self.seed})", flush=True)
        print("=" * 75, flush=True)
        ds_default = self.get_dataset(pooling="mean", gamma_meta=1.0)
        folds_default = get_train_val_cv_folds(ds_default, n_splits=5, cv_seed=self.seed)

        self.run_exploration(ds_default, folds_default)
        self.run_exploitation()

        print("=" * 75, flush=True)
        print(f"SEARCH COMPLETE! Final Champion: {self.champion['trial']} (5-fold CV Macro AUROC: {self.champion['mean_auroc']:.4f})", flush=True)
        print(f"Champion Config: {json.dumps(self.champion['config']['params'])}", flush=True)
        print("=" * 75, flush=True)
        return self.champion

    def save_logs(
        self,
        json_path: str = "artifacts/agent_decision_log.json",
        md_path: str = "artifacts/agent_search_summary.md"
    ):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        summary_record = {
            "seed": self.seed,
            "total_trials_run": len(self.trials_log),
            "champion": self.champion,
            "trials": self.trials_log
        }
        with open(json_path, "w") as f:
            json.dump(summary_record, f, indent=2)

        # Markdown report
        lines = [
            f"# Phase 2 Autonomous Agent Decision Log (Seed {self.seed})\n",
            f"**Total Trials Run**: {len(self.trials_log)}  ",
            f"**Final Champion**: `{self.champion['trial']}` with **5-fold CV Macro AUROC: {self.champion['mean_auroc']:.4f}**  ",
            f"**Champion Family**: `{self.champion['config']['family']}`  ",
            f"**Champion Params**: `{json.dumps(self.champion['config']['params'])}`  ",
            f"**Champion Pooling**: `{self.champion.get('pooling', 'mean')}` | **gamma_meta**: `{self.champion.get('gamma_meta', 1.0)}`\n",
            "## Trial Progression & SE Guardrail Decisions\n",
            "| Trial | Stage | Hypothesis | Family | CV AUROC | Delta vs Champ | SE(Delta) | Threshold (0.5×SE) | Decision | Action/Reason |",
            "| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
        ]

        for t in self.trials_log:
            lines.append(
                f"| **{t['trial']}** | {t['stage']} | {t['hypothesis']} | {t['family']} | "
                f"**{t['mean_cv_auroc']:.4f}** | {t['delta_vs_champion']:+.4f} | {t['se_delta']:.4f} | {t.get('guardrail_threshold', 0.0):.4f} | "
                f"`{t['decision']}` | {t['reason']} |"
            )

        with open(md_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Saved agent decision log to {json_path} and {md_path}", flush=True)
