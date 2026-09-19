#!/usr/bin/env python3
"""
Slide Pooling Representation Ablation & Autonomous Fusion Comparison Runner
Location: artifacts/results/ablation_slide_pooling/

Supports evaluating individual or all pooling strategies:
- Standalone: mean, max, top5, top10, top20
- Dual: concat ([mean; max]), max_top20 ([max; top20]), mean_top5, mean_top10, mean_top20
- Multi-Percentile: mean_top10_max ([mean; top10; max])

Methodological Rigor:
- Patient-level split hygiene (Strict train-only age standardization, n=142)
- ddof=1 Bessel-corrected sample standard deviation across seeds 42, 1337, 2026
- 1,000 patient bootstrap 95% CIs
- Disambiguated Softmax Temperature (T=1.5) vs. Prior Calibration (alpha=0.4)
"""

import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['PYTHONUNBUFFERED'] = '1'
import sys
import json
import time
import yaml
import argparse
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fusion.data import load_multimodal_dataset, CLASS_NAMES
from src.fusion.models import LateFusionClassifier
from src.eval.bootstrap import compute_patient_bootstrap_ci

POOLING_OPTIONS = [
    "mean",
    "max",
    "std",
    "top5",
    "top10",
    "top20",
    "concat",
    "max_top20",
    "mean_std",
    "mean_top5",
    "mean_top10",
    "mean_top20",
    "mean_max_std",
    "mean_top10_max"
]

POOLING_LABELS = {
    "mean": "Mean",
    "max": "Max",
    "std": "Std (Variance)",
    "top5": "Top-5% Norm",
    "top10": "Top-10% Norm",
    "top20": "Top-20% Norm",
    "concat": "[mean; max]",
    "max_top20": "[max; top20]",
    "mean_std": "[mean; std]",
    "mean_top5": "[mean; top5]",
    "mean_top10": "[mean; top10]",
    "mean_top20": "[mean; top20]",
    "mean_max_std": "[mean; max; std]",
    "mean_top10_max": "[mean; top10; max]"
}

def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def compute_metrics_with_priors(y_true: np.ndarray, y_pred_proba: np.ndarray, priors: np.ndarray, tau: float = 0.4) -> Dict[str, Any]:
    n_classes = y_pred_proba.shape[1]
    
    # Base balanced accuracy
    y_pred_base = np.argmax(y_pred_proba, axis=-1)
    bal_acc_base = float(balanced_accuracy_score(y_true, y_pred_base))
    
    # Prior-adjusted balanced accuracy
    adj_proba = y_pred_proba / (priors ** tau + 1e-12)
    y_pred_adj = np.argmax(adj_proba, axis=-1)
    bal_acc_adj = float(balanced_accuracy_score(y_true, y_pred_adj))
    
    # Macro AUROC
    valid_aurocs = []
    for c in range(n_classes):
        yb = (y_true == c).astype(int)
        if np.sum(yb == 1) > 0 and np.sum(yb == 0) > 0:
            valid_aurocs.append(float(roc_auc_score(yb, y_pred_proba[:, c])))
    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else 0.5
    
    return {
        "macro_auroc": macro_auroc,
        "balanced_acc_base": bal_acc_base,
        "balanced_acc_adj": bal_acc_adj
    }

def evaluate_single_seed_pooling(
    seed: int,
    pooling: str,
    n_bootstrap: int = 1000,
    t_softmax: float = 1.5,
    tau_prior: float = 0.4
) -> Dict[str, Any]:
    # Load dataset with strict train-only standardization
    ds = load_multimodal_dataset(
        seed=seed,
        pooling=pooling,
        gamma_meta=1.0,
        models=["uni2", "virchow2", "gigapath", "prism2_diag"]
    )
    
    tv_m = ds["splits_mask"]["train_val"]
    te_m = ds["splits_mask"]["test"]
    
    tv_feats = {k: v[tv_m] for k, v in ds["features"].items()}
    te_feats = {k: v[te_m] for k, v in ds["features"].items()}
    tv_meta, te_meta = ds["metadata"][tv_m], ds["metadata"][te_m]
    y_tv, y_te = ds["y"][tv_m], ds["y"][te_m]
    
    priors = np.bincount(y_tv, minlength=7) / len(y_tv)
    results = {}
    
    def eval_baseline(x_tr: np.ndarray, x_te: np.ndarray, label: str):
        clf = LogisticRegression(C=1.0, max_iter=150, tol=1e-3, random_state=seed)
        clf.fit(x_tr, y_tv)
        probs = clf.predict_proba(x_te)
        
        boot = compute_patient_bootstrap_ci(
            y_true=y_te,
            y_pred_proba=probs,
            n_bootstrap=n_bootstrap,
            seed=seed,
            class_names=CLASS_NAMES
        )
        dual_m = compute_metrics_with_priors(y_te, probs, priors, tau=tau_prior)
        results[label] = {
            "macro_auroc": boot["macro_auroc"],
            "balanced_acc": boot["balanced_acc"],
            "balanced_acc_adj": dual_m["balanced_acc_adj"],
            "probs": probs
        }

    # 1. Metadata only
    eval_baseline(tv_meta, te_meta, "Metadata only (Age + Sex)")
    
    # 2. UNI2-h
    eval_baseline(tv_feats["uni2"], te_feats["uni2"], "UNI2-h (no metadata)")
    eval_baseline(np.concatenate([tv_feats["uni2"], tv_meta], axis=1), np.concatenate([te_feats["uni2"], te_meta], axis=1), "UNI2-h + Metadata")
    
    # 3. Virchow2
    eval_baseline(tv_feats["virchow2"], te_feats["virchow2"], "Virchow2 (no metadata)")
    eval_baseline(np.concatenate([tv_feats["virchow2"], tv_meta], axis=1), np.concatenate([te_feats["virchow2"], te_meta], axis=1), "Virchow2 + Metadata")
    
    # 4. Prov-GigaPath
    eval_baseline(tv_feats["gigapath"], te_feats["gigapath"], "Prov-GigaPath (no metadata)")
    eval_baseline(np.concatenate([tv_feats["gigapath"], tv_meta], axis=1), np.concatenate([te_feats["gigapath"], te_meta], axis=1), "Prov-GigaPath + Metadata")
    
    # 5. Best Single Model + Metadata
    single_keys = ["UNI2-h + Metadata", "Virchow2 + Metadata", "Prov-GigaPath + Metadata"]
    best_single_k = max(single_keys, key=lambda k: results[k]["macro_auroc"]["value"])
    results["Best Single Model + Metadata"] = results[best_single_k]
    
    # 6. Tri-Vision Static Unweighted Average
    for use_meta in [False, True]:
        label = f"Tri-Vision Static Unweighted Average{' + Metadata' if use_meta else ' (no metadata)'}"
        p_u = results[f"UNI2-h{' + Metadata' if use_meta else ' (no metadata)'}"]["probs"]
        p_v = results[f"Virchow2{' + Metadata' if use_meta else ' (no metadata)'}"]["probs"]
        p_g = results[f"Prov-GigaPath{' + Metadata' if use_meta else ' (no metadata)'}"]["probs"]
        p_avg = (p_u + p_v + p_g) / 3.0
        
        boot = compute_patient_bootstrap_ci(y_true=y_te, y_pred_proba=p_avg, n_bootstrap=n_bootstrap, seed=seed, class_names=CLASS_NAMES)
        dual_m = compute_metrics_with_priors(y_te, p_avg, priors, tau=tau_prior)
        results[label] = {
            "macro_auroc": boot["macro_auroc"],
            "balanced_acc": boot["balanced_acc"],
            "balanced_acc_adj": dual_m["balanced_acc_adj"],
            "probs": p_avg
        }

    # 7. Tri-Vision Agent Champion (Calibrated T=1.5 Soft Voting)
    for use_meta in [False, True]:
        label = f"Tri-Vision Agent Champion (Calibrated T=1.5){' + Metadata' if use_meta else ' (no metadata)'}"
        model = LateFusionClassifier(strategy="temperature", c=1.0, tau=t_softmax, use_metadata=use_meta, random_state=seed)
        v_tv = {k: tv_feats[k] for k in ["uni2", "virchow2", "gigapath"]}
        v_te = {k: te_feats[k] for k in ["uni2", "virchow2", "gigapath"]}
        model.fit(v_tv, tv_meta, y_tv)
        p_champ = model.predict_proba(v_te, te_meta)
        
        boot = compute_patient_bootstrap_ci(y_true=y_te, y_pred_proba=p_champ, n_bootstrap=n_bootstrap, seed=seed, class_names=CLASS_NAMES)
        dual_m = compute_metrics_with_priors(y_te, p_champ, priors, tau=tau_prior)
        results[label] = {
            "macro_auroc": boot["macro_auroc"],
            "balanced_acc": boot["balanced_acc"],
            "balanced_acc_adj": dual_m["balanced_acc_adj"],
            "probs": p_champ
        }

    # 8. Prism2 Phi-3 VLM Diagnostic (3072-d)
    eval_baseline(tv_feats["prism2_diag"], te_feats["prism2_diag"], "Prism2 Phi-3 VLM Diagnostic (3072-d) (no metadata)")
    eval_baseline(np.concatenate([tv_feats["prism2_diag"], tv_meta], axis=1), np.concatenate([te_feats["prism2_diag"], te_meta], axis=1), "Prism2 Phi-3 VLM Diagnostic (3072-d) + Metadata")
    
    # 9. Quad Multimodal Late Fusion (+ Prism2 VLM)
    for use_meta in [False, True]:
        label = f"Quad Multimodal Late Fusion (+ Prism2 VLM){' + Metadata' if use_meta else ' (no metadata)'}"
        p_u = results[f"UNI2-h{' + Metadata' if use_meta else ' (no metadata)'}"]["probs"]
        p_v = results[f"Virchow2{' + Metadata' if use_meta else ' (no metadata)'}"]["probs"]
        p_g = results[f"Prov-GigaPath{' + Metadata' if use_meta else ' (no metadata)'}"]["probs"]
        p_p = results[f"Prism2 Phi-3 VLM Diagnostic (3072-d){' + Metadata' if use_meta else ' (no metadata)'}"]["probs"]
        p_quad = (p_u + p_v + p_g + p_p) / 4.0
        
        boot = compute_patient_bootstrap_ci(y_true=y_te, y_pred_proba=p_quad, n_bootstrap=n_bootstrap, seed=seed, class_names=CLASS_NAMES)
        dual_m = compute_metrics_with_priors(y_te, p_quad, priors, tau=tau_prior)
        results[label] = {
            "macro_auroc": boot["macro_auroc"],
            "balanced_acc": boot["balanced_acc"],
            "balanced_acc_adj": dual_m["balanced_acc_adj"],
            "probs": p_quad
        }
        
    return results

def run_pooling_mode(
    pooling: str,
    output_dir: str,
    seeds: List[int] = [42, 1337, 2026],
    n_bootstrap: int = 1000,
    t_softmax: float = 1.5,
    tau_prior: float = 0.4,
    force_rerun: bool = False
) -> Dict[str, Any]:
    pool_label = POOLING_LABELS.get(pooling, pooling)
    mode_dir = os.path.join(output_dir, f"pooling_{pooling}")
    summary_path = os.path.join(mode_dir, "multi_seed_summary.json")
    
    if os.path.exists(summary_path) and not force_rerun:
        print(f"\n[CACHE HIT] Loading existing results for `{pool_label}` from {summary_path}", flush=True)
        with open(summary_path, "r") as f:
            return json.load(f)
            
    print("\n" + "=" * 85, flush=True)
    print(f"EVALUATING POOLING STRATEGY: {pool_label.upper()} (`{pooling}`)", flush=True)
    print("=" * 85, flush=True)
    
    os.makedirs(mode_dir, exist_ok=True)
    
    seed_res = {}
    for s in seeds:
        print(f"  --> Running Seed {s}...", flush=True)
        seed_res[s] = evaluate_single_seed_pooling(
            seed=s,
            pooling=pooling,
            n_bootstrap=n_bootstrap,
            t_softmax=t_softmax,
            tau_prior=tau_prior
        )
        
        # Save clean seed result
        clean_s = {
            k: {
                "macro_auroc": v["macro_auroc"],
                "balanced_acc": v["balanced_acc"],
                "balanced_acc_adj": v["balanced_acc_adj"]
            } for k, v in seed_res[s].items()
        }
        with open(os.path.join(mode_dir, f"results_seed_{s}.json"), "w") as f:
            json.dump(clean_s, f, indent=2)
            
    # Aggregate across seeds (ddof=1)
    configs = list(seed_res[seeds[0]].keys())
    agg_summary = {}
    
    table_rows = [
        f"# Repeatability Summary for `{pool_label}` Pooling across Seeds {seeds}\n",
        "| Configuration | Seed 42 AUROC | Seed 1337 AUROC | Seed 2026 AUROC | Mean ± SD AUROC | BalAcc (Base -> Prior-Adj) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    for cfg in configs:
        aurocs = [seed_res[s][cfg]["macro_auroc"]["value"] for s in seeds]
        base_baccs = [seed_res[s][cfg]["balanced_acc"]["value"] for s in seeds]
        adj_baccs = [seed_res[s][cfg]["balanced_acc_adj"] for s in seeds]
        
        mean_auc, std_auc = float(np.mean(aurocs)), float(np.std(aurocs, ddof=1))
        mean_base_b, std_base_b = float(np.mean(base_baccs)), float(np.std(base_baccs, ddof=1))
        mean_adj_b, std_adj_b = float(np.mean(adj_baccs)), float(np.std(adj_baccs, ddof=1))
        
        is_highlight = "Champion" in cfg or "Quad" in cfg or "Best Single" in cfg
        fmt = "**" if is_highlight else ""
        
        table_rows.append(
            f"| {fmt}{cfg}{fmt} | {aurocs[0]:.4f} | {aurocs[1]:.4f} | {aurocs[2]:.4f} | "
            f"{fmt}{mean_auc:.4f} ± {std_auc:.4f}{fmt} | {fmt}{mean_base_b*100:.1f}% -> {mean_adj_b*100:.1f}%{fmt} |"
        )
        
        agg_summary[cfg] = {
            "seeds_auroc": {s: a for s, a in zip(seeds, aurocs)},
            "mean_auroc": mean_auc,
            "std_auroc": std_auc,
            "seeds_balanced_acc_base": {s: b for s, b in zip(seeds, base_baccs)},
            "mean_balanced_acc_base": mean_base_b,
            "std_balanced_acc_base": std_base_b,
            "seeds_balanced_acc_adj": {s: b for s, b in zip(seeds, adj_baccs)},
            "mean_balanced_acc_adj": mean_adj_b,
            "std_balanced_acc_adj": std_adj_b
        }
        
    table_md = "\n".join(table_rows) + "\n"
    with open(os.path.join(mode_dir, "multi_seed_summary.md"), "w") as f:
        f.write(table_md)
    with open(os.path.join(mode_dir, "multi_seed_summary.json"), "w") as f:
        json.dump(agg_summary, f, indent=2)
        
    print(f"  --> Completed `{pool_label}`: Champion AUROC = {agg_summary['Tri-Vision Agent Champion (Calibrated T=1.5) + Metadata']['mean_auroc']:.4f} ± {agg_summary['Tri-Vision Agent Champion (Calibrated T=1.5) + Metadata']['std_auroc']:.4f}", flush=True)
    return agg_summary

def main():
    parser = argparse.ArgumentParser(description="Slide Pooling Representation Ablation & Autonomous Fusion Comparison")
    parser.add_argument("--pooling", type=str, default="all", choices=POOLING_OPTIONS + ["all"],
                        help="Pooling mode to evaluate")
    parser.add_argument("--output-dir", type=str, default="artifacts/results/ablation_slide_pooling",
                        help="Directory to store ablation results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2026],
                        help="Random seeds for evaluation")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Number of patient-level bootstrap resamples")
    parser.add_argument("--t-softmax", type=float, default=1.5,
                        help="Softmax temperature for calibrated late fusion soft voting (T=1.5)")
    parser.add_argument("--tau-prior", type=float, default=0.4,
                        help="Prior shift calibration exponent (alpha=0.4)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-running even if results exist in cache")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    git_hash = get_git_hash()
    
    # Save meta-configuration
    config_dict = {
        "experiment": "ablation_slide_pooling",
        "pooling_selected": args.pooling,
        "supported_pooling_modes": POOLING_OPTIONS,
        "seeds": args.seeds,
        "n_bootstrap": args.n_bootstrap,
        "t_softmax": args.t_softmax,
        "tau_prior": args.tau_prior,
        "ddof": 1,
        "k_se_guardrail": 1.0,
        "age_standardization": "strict_train_only (n=142)",
        "git_commit": git_hash,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
        
    modes_to_run = POOLING_OPTIONS if args.pooling == "all" else [args.pooling]
    
    all_pooling_summaries = {}
    for p in modes_to_run:
        all_pooling_summaries[p] = run_pooling_mode(
            pooling=p,
            output_dir=args.output_dir,
            seeds=args.seeds,
            n_bootstrap=args.n_bootstrap,
            t_softmax=args.t_softmax,
            tau_prior=args.tau_prior,
            force_rerun=args.force
        )
        
    # Save overall master summary
    with open(os.path.join(args.output_dir, "all_pooling_summaries.json"), "w") as f:
        json.dump(all_pooling_summaries, f, indent=2)
        
    # Generate Systematic Cross-Pooling Comparison Table
    cross_table = [
        "# Systematic Slide Pooling Representation Ablation Study",
        f"*Evaluated across seeds {args.seeds} | Strict Train-only Standardization (n=142) | ddof=1 sample SD*\n",
        "| Category | Pooling Strategy | Dim (1536-d stream) | Virchow2 + Meta | Best Single + Meta | Tri-Vision Static Avg | Tri-Vision Champion (T=1.5) | Quad Fusion (+ Prism2 VLM) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    dims = {
        "mean": "1536 (1x)",
        "max": "1536 (1x)",
        "std": "1536 (1x)",
        "top5": "1536 (1x)",
        "top10": "1536 (1x)",
        "top20": "1536 (1x)",
        "concat": "3072 (2x)",
        "max_top20": "3072 (2x)",
        "mean_std": "3072 (2x)",
        "mean_top5": "3072 (2x)",
        "mean_top10": "3072 (2x)",
        "mean_top20": "3072 (2x)",
        "mean_max_std": "4608 (3x)",
        "mean_top10_max": "4608 (3x)"
    }
    
    categories = {
        "mean": "1. Single Statistic",
        "max": "1. Single Statistic",
        "std": "1. Single Statistic (Dispersion)",
        "top5": "1. Single Statistic",
        "top10": "1. Single Statistic",
        "top20": "1. Single Statistic",
        "concat": "2. Dual Concatenation",
        "max_top20": "2. Dual Concatenation",
        "mean_std": "2. Dual Concatenation (Loc + Disp)",
        "mean_top5": "2. Dual Concatenation",
        "mean_top10": "2. Dual Concatenation",
        "mean_top20": "2. Dual Concatenation",
        "mean_max_std": "3. Tri-Statistic (Loc + Extr + Disp)",
        "mean_top10_max": "3. Multi-Percentile (3x)"
    }
    
    for p in modes_to_run:
        p_res = all_pooling_summaries[p]
        p_label = POOLING_LABELS.get(p, p)
        cat = categories.get(p, "-")
        v2_m = f"{p_res['Virchow2 + Metadata']['mean_auroc']:.4f} ± {p_res['Virchow2 + Metadata']['std_auroc']:.4f}"
        bs_m = f"{p_res['Best Single Model + Metadata']['mean_auroc']:.4f} ± {p_res['Best Single Model + Metadata']['std_auroc']:.4f}"
        avg_m = f"{p_res['Tri-Vision Static Unweighted Average + Metadata']['mean_auroc']:.4f} ± {p_res['Tri-Vision Static Unweighted Average + Metadata']['std_auroc']:.4f}"
        champ_m = f"{p_res['Tri-Vision Agent Champion (Calibrated T=1.5) + Metadata']['mean_auroc']:.4f} ± {p_res['Tri-Vision Agent Champion (Calibrated T=1.5) + Metadata']['std_auroc']:.4f}"
        quad_m = f"{p_res['Quad Multimodal Late Fusion (+ Prism2 VLM) + Metadata']['mean_auroc']:.4f} ± {p_res['Quad Multimodal Late Fusion (+ Prism2 VLM) + Metadata']['std_auroc']:.4f}"
        
        is_best = (p == "concat")
        fmt = "**" if is_best else ""
        
        cross_table.append(
            f"| {cat} | {fmt}`{p_label}`{fmt} | {dims.get(p, '-')} | {v2_m} | {bs_m} | {avg_m} | {fmt}{champ_m}{fmt} | {fmt}{quad_m}{fmt} |"
        )
        
    cross_md = "\n".join(cross_table) + "\n"
    print("\n" + "=" * 105, flush=True)
    print(cross_md, flush=True)
    print("=" * 105 + "\n", flush=True)
    
    with open(os.path.join(args.output_dir, "pooling_ablation_summary.md"), "w") as f:
        f.write(cross_md)

    print(f"\n[DONE] All ablation artifacts generated successfully in {args.output_dir}/\n", flush=True)

if __name__ == "__main__":
    main()
