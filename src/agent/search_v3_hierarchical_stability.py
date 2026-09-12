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




class HierarchicalStabilityAgent:
    """
    Search Agent v3: Hierarchical 3-Stage Stability Agent (User Proposed Architecture).
    - Stage 1 (Pooling/Representation): Evaluates Mean, Max, Top-5%, Top-10%, Top-20%, Mean+Max.
               Selects and FREEZES optimal slide representation R*.
    - Stage 2 (Fusion Architecture): Evaluates Early, L2-Early, PCA-Early, Late Uniform, Late Temp, Late Simplex
               strictly on top of R*. Identifies Top-2 finalists.
    - Stage 3 (Multi-Seed Stability & Freezing): Evaluates finalists across 5-fold CV x 3 seeds (15 folds)
               on Train+Val. Computes Mean, Variance, and SE, ranking by Stability Utility (Mean - 0.5*SE).
               Freezes the global Champion.
    - Final Evaluation: One-time untouched test evaluation on held-out test cohort with 1,000 bootstrap CIs.
    """
    def __init__(self, seed: int = 42, k_se: float = 0.5, stability_seeds: List[int] = (42, 1337, 2026)):
        self.seed = seed
        self.k_se = k_se
        self.stability_seeds = list(stability_seeds)
        self.trials_log: List[Dict[str, Any]] = []
        self.champion: Optional[Dict[str, Any]] = None
        self.dataset_cache: Dict[str, Any] = {}
        self.winning_pooling: str = "concat"
        self.finalists: List[Dict[str, Any]] = []

    def get_dataset(self, pooling: str = "mean", gamma_meta: float = 1.0, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is None:
            seed = self.seed
        key = f"{seed}_{pooling}_{gamma_meta}"
        if key not in self.dataset_cache:
            self.dataset_cache[key] = load_multimodal_dataset(
                seed=seed,
                gamma_meta=gamma_meta,
                pooling=pooling
            )
        return self.dataset_cache[key]

    def run_stage1_pooling(self):
        """
        Stage 1: Pooling / Slide Representation Search.
        Evaluates: Mean, Max, Top-5% mean, Top-10% mean, Top-20% mean, Mean + Max (concat).
        Baseline classifier: Uniform soft voting with C=1.0 + metadata.
        """
        print("=" * 75, flush=True)
        print(f"STAGE 1: SLIDE POOLING & REPRESENTATION SEARCH (Seed: {self.seed})", flush=True)
        print("=" * 75, flush=True)

        pooling_candidates = [
            {"trial": "T01", "name": "Mean", "pooling": "mean", "desc": "Spatial mean pooling across all tiles"},
            {"trial": "T02", "name": "Max", "pooling": "max", "desc": "Element-wise max pooling across all tiles"},
            {"trial": "T03", "name": "Top-5% mean", "pooling": "top5", "desc": "Top-5% norm tile mean pooling"},
            {"trial": "T04", "name": "Top-10% mean", "pooling": "top10", "desc": "Top-10% norm tile mean pooling"},
            {"trial": "T05", "name": "Top-20% mean", "pooling": "top20", "desc": "Top-20% norm tile mean pooling"},
            {"trial": "T06", "name": "Mean + Max", "pooling": "concat", "desc": "Concatenated [mean; max] multi-resolution pooling"},
        ]

        stage1_results = []
        base_cfg = {
            "family": "late",
            "params": {"strategy": "uniform", "c": 1.0, "use_metadata": True}
        }

        for cand in pooling_candidates:
            t_start = time.time()
            ds = self.get_dataset(seed=self.seed, pooling=cand["pooling"], gamma_meta=1.0)
            folds = get_train_val_cv_folds(ds, n_splits=5, cv_seed=self.seed)

            eval_cfg = dict(base_cfg)
            eval_cfg["trial"] = cand["trial"]

            mean_auroc, fold_aurocs, _ = evaluate_candidate_cv(eval_cfg, ds, folds)
            elapsed = time.time() - t_start

            record = {
                "trial": cand["trial"],
                "stage": "Stage 1 (Pooling)",
                "hypothesis": cand["desc"],
                "family": "late",
                "params": eval_cfg["params"],
                "pooling": cand["pooling"],
                "mean_cv_auroc": mean_auroc,
                "fold_cv_aurocs": fold_aurocs,
                "fold_std": float(np.std(fold_aurocs, ddof=1)),
                "time_sec": elapsed
            }
            stage1_results.append(record)
            self.trials_log.append(record)
            print(f"[{cand['trial']}] Pooling: {cand['name']:<15} | 5-Fold CV AUROC: {mean_auroc:.4f} (std: {record['fold_std']:.4f}) in {elapsed:.2f}s", flush=True)

        # Select winning pooling
        best_pool_cand = max(stage1_results, key=lambda x: x["mean_cv_auroc"])
        self.winning_pooling = best_pool_cand["pooling"]
        print("-" * 75, flush=True)
        print(f"STAGE 1 WINNER: `{self.winning_pooling.upper()}` ({best_pool_cand['hypothesis']}) with CV AUROC = {best_pool_cand['mean_cv_auroc']:.4f}", flush=True)
        print(f"FREEZING REPRESENTATION: All subsequent fusion and stability trials will use `{self.winning_pooling.upper()}`.", flush=True)
        print("=" * 75 + "\n", flush=True)

    def run_stage2_fusion(self):
        """
        Stage 2: Fusion Strategy Search on top of Frozen Representation R*.
        Evaluates:
        1. Early concatenation (Raw)
        2. L2 + concatenation
        3. PCA + concatenation
        4. Probability averaging (Uniform)
        5. Temperature-scaled probability fusion
        6. Weighted probability/logit fusion (Simplex)
        """
        print("=" * 75, flush=True)
        print(f"STAGE 2: FUSION STRATEGY SEARCH ON `{self.winning_pooling.upper()}` (Seed: {self.seed})", flush=True)
        print("=" * 75, flush=True)

        fusion_candidates = [
            {"trial": "T07", "name": "Early (Raw)", "family": "early", "params": {"c": 1.0, "use_metadata": True}, "desc": "Early concatenation of raw stream vectors"},
            {"trial": "T08", "name": "Early (L2-norm)", "family": "early", "params": {"l2_norm_per_stream": True, "c": 1.0, "use_metadata": True}, "desc": "Per-stream L2-normalization before concatenation"},
            {"trial": "T09", "name": "Early (PCA-256)", "family": "early", "params": {"pca_dim": 256, "c": 1.0, "use_metadata": True}, "desc": "Train-set PCA (256-d, 95% variance) dimensionality reduction"},
            {"trial": "T10", "name": "Late (Uniform)", "family": "late", "params": {"strategy": "uniform", "c": 1.0, "use_metadata": True}, "desc": "Uniform probability soft voting ensemble"},
            {"trial": "T11", "name": "Late (Temp tau=1.5)", "family": "late", "params": {"strategy": "temperature", "tau": 1.5, "c": 1.0, "use_metadata": True}, "desc": "Temperature-scaled calibrated soft voting"},
            {"trial": "T12", "name": "Late (Temp tau=1.2)", "family": "late", "params": {"strategy": "temperature", "tau": 1.2, "c": 1.0, "use_metadata": True}, "desc": "Calibrated temperature soft voting (tau=1.2)"},
            {"trial": "T13", "name": "Late (Simplex)", "family": "late", "params": {"strategy": "simplex", "c": 1.0, "use_metadata": True}, "desc": "Validation simplex-constrained convex weighting"},
            {"trial": "T14", "name": "Intermediate (Gated)", "family": "intermediate", "params": {"strategy": "gated", "hidden_dim": 256, "dropout": 0.2, "epochs": 25, "lr": 1e-3, "use_metadata": True}, "desc": "Stream-level Gated Fusion neural network"},
        ]

        ds = self.get_dataset(seed=self.seed, pooling=self.winning_pooling, gamma_meta=1.0)
        folds = get_train_val_cv_folds(ds, n_splits=5, cv_seed=self.seed)

        stage2_results = []
        for cand in fusion_candidates:
            t_start = time.time()
            eval_cfg = {
                "trial": cand["trial"],
                "family": cand["family"],
                "params": cand["params"]
            }

            mean_auroc, fold_aurocs, _ = evaluate_candidate_cv(eval_cfg, ds, folds)
            elapsed = time.time() - t_start

            record = {
                "trial": cand["trial"],
                "stage": "Stage 2 (Fusion)",
                "hypothesis": cand["desc"],
                "family": cand["family"],
                "params": cand["params"],
                "pooling": self.winning_pooling,
                "mean_cv_auroc": mean_auroc,
                "fold_cv_aurocs": fold_aurocs,
                "fold_std": float(np.std(fold_aurocs, ddof=1)),
                "time_sec": elapsed
            }
            stage2_results.append(record)
            self.trials_log.append(record)
            print(f"[{cand['trial']}] Fusion: {cand['name']:<20} | 5-Fold CV AUROC: {mean_auroc:.4f} (std: {record['fold_std']:.4f}) in {elapsed:.2f}s", flush=True)

        # Sort and pick top-2 finalists
        sorted_stage2 = sorted(stage2_results, key=lambda x: x["mean_cv_auroc"], reverse=True)
        self.finalists = sorted_stage2[:2]
        print("-" * 75, flush=True)
        print(f"STAGE 2 FINALISTS SELECTED FOR MULTI-SEED STABILITY VERIFICATION:")
        for idx, f in enumerate(self.finalists, 1):
            print(f"  Finalist {idx}: [{f['trial']}] {f['hypothesis']} ({f['family']}) | CV: {f['mean_cv_auroc']:.4f}", flush=True)
        print("=" * 75 + "\n", flush=True)

    def run_stage3_stability(self):
        """
        Stage 3: Multi-Seed Stability & Champion Freezing.
        Evaluates finalists across 5-fold CV x 3 seeds (15 evaluation folds total) on Train+Val.
        Computes Mean, Variance, and SE across all 15 folds.
        Ranks by Stability Utility: Utility = Mean - 0.5 * SE.
        Freezes the single global Champion.
        """
        print("=" * 75, flush=True)
        print(f"STAGE 3: MULTI-SEED STABILITY EVALUATION (15 Folds across Seeds {self.stability_seeds})", flush=True)
        print("=" * 75, flush=True)

        finalist_stability_scores = []
        trial_counter = 15

        for idx, fin in enumerate(self.finalists, 1):
            t_id = f"T{trial_counter:02d}"
            trial_counter += 1

            all_15_fold_scores = []
            seed_means = []

            for s in self.stability_seeds:
                ds_s = self.get_dataset(seed=s, pooling=self.winning_pooling, gamma_meta=1.0)
                folds_s = get_train_val_cv_folds(ds_s, n_splits=5, cv_seed=s)

                eval_cfg = {
                    "family": fin["family"],
                    "params": fin["params"]
                }
                mean_s, fold_aurocs_s, _ = evaluate_candidate_cv(eval_cfg, ds_s, folds_s)
                all_15_fold_scores.extend(fold_aurocs_s)
                seed_means.append(mean_s)

            grand_mean = float(np.mean(all_15_fold_scores))
            grand_std = float(np.std(all_15_fold_scores, ddof=1))
            grand_se = float(grand_std / np.sqrt(len(all_15_fold_scores)))
            stability_utility = grand_mean - (self.k_se * grand_se)

            stab_record = {
                "trial": t_id,
                "stage": "Stage 3 (Stability)",
                "finalist_index": idx,
                "hypothesis": f"Multi-seed stability: {fin['hypothesis']}",
                "family": fin["family"],
                "params": fin["params"],
                "pooling": self.winning_pooling,
                "seed_means": seed_means,
                "grand_mean_auroc": grand_mean,
                "grand_std": grand_std,
                "grand_se": grand_se,
                "stability_utility": stability_utility,
                "mean_cv_auroc": grand_mean,
                "fold_cv_aurocs": all_15_fold_scores[:5] # primary seed folds for compat
            }
            finalist_stability_scores.append(stab_record)
            self.trials_log.append(stab_record)

            print(f"[{t_id}] Finalist {idx}: {fin['hypothesis']:<45} | 15-Fold Mean: {grand_mean:.4f} ± {grand_std:.4f} (SE: {grand_se:.4f}) | Utility: {stability_utility:.4f}", flush=True)

        # Champion is the one with highest stability utility
        winner = max(finalist_stability_scores, key=lambda x: x["stability_utility"])
        self.champion = {
            "trial": winner["trial"],
            "config": {
                "family": winner["family"],
                "params": winner["params"],
            },
            "mean_auroc": winner["grand_mean_auroc"],
            "stability_utility": winner["stability_utility"],
            "grand_se": winner["grand_se"],
            "pooling": self.winning_pooling,
            "gamma_meta": 1.0
        }

        print("-" * 75, flush=True)
        print(f"GLOBAL CHAMPION FROZEN: [{self.champion['trial']}] with 15-Fold Mean CV AUROC = {self.champion['mean_auroc']:.4f} (Utility: {self.champion['stability_utility']:.4f})", flush=True)
        print(f"Champion Architecture: Family={self.champion['config']['family']}, Pooling={self.winning_pooling.upper()}, Params={json.dumps(self.champion['config']['params'])}", flush=True)
        print("=" * 75 + "\n", flush=True)

    def run_search(self) -> Dict[str, Any]:
        t0 = time.time()
        self.run_stage1_pooling()
        self.run_stage2_fusion()
        self.run_stage3_stability()
        print(f"3-Stage Hierarchical Stability Search completed in {time.time() - t0:.2f} seconds.", flush=True)
        return self.champion

    def save_logs(self, json_path: str, md_path: str):
        summary_record = {
            "seed": self.seed,
            "agent_version": "v3_hierarchical_stability",
            "winning_pooling": self.winning_pooling,
            "total_trials_run": len(self.trials_log),
            "champion": self.champion,
            "trials": self.trials_log
        }
        with open(json_path, "w") as f:
            json.dump(summary_record, f, indent=2)

        lines = [
            f"# Phase 2 Agent v3 Decision Log (Hierarchical Stability Agent)\n",
            f"**Architecture**: 3-Stage Hierarchical Stability Search  ",
            f"**Total Trials Run**: {len(self.trials_log)}  ",
            f"**Winning Pooling (Stage 1)**: `{self.winning_pooling.upper()}`  ",
            f"**Final Champion (Stage 3)**: `{self.champion['trial']}` with **15-Fold Grand Mean CV AUROC: {self.champion['mean_auroc']:.4f}**  ",
            f"**Stability Utility**: `{self.champion['stability_utility']:.4f}` (Mean - {self.k_se}*SE)  ",
            f"**Champion Family**: `{self.champion['config']['family']}`  ",
            f"**Champion Params**: `{json.dumps(self.champion['config']['params'])}`\n",
            "## Trial Progression across Stages\n",
            "| Trial | Stage | Strategy / Hypothesis | Family | Pooling | CV AUROC | Std / SE | Decision / Status |",
            "| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |"
        ]

        for t in self.trials_log:
            stage_str = t.get("stage", "")
            if "Stability" in stage_str:
                metric_str = f"**{t.get('grand_mean_auroc', 0.0):.4f}**"
                std_str = f"±{t.get('grand_std', 0.0):.4f} (SE: {t.get('grand_se', 0.0):.4f})"
                status = f"Champion (Utility: {t.get('stability_utility', 0.0):.4f})" if t["trial"] == self.champion["trial"] else "Runner-up"
            else:
                metric_str = f"{t.get('mean_cv_auroc', 0.0):.4f}"
                std_str = f"±{t.get('fold_std', 0.0):.4f}"
                status = "Winner" if t.get("pooling") == self.winning_pooling and "Stage 1" in stage_str else "Evaluated"

            lines.append(
                f"| **{t['trial']}** | {stage_str} | {t['hypothesis']} | {t['family']} | `{t['pooling']}` | "
                f"{metric_str} | {std_str} | {status} |"
            )

        with open(md_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Saved agent decision log to {json_path} and {md_path}", flush=True)


AutonomousFusionAgent = HierarchicalStabilityAgent
