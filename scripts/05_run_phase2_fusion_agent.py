import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["PYTHONUNBUFFERED"] = "1"
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fusion.data import load_multimodal_dataset, CLASS_NAMES
from src.fusion.models import EarlyFusionClassifier, LateFusionClassifier, IntermediateFusionClassifier
from src.agent import AutonomousFusionAgent, get_agent_cls
from src.eval.bootstrap import compute_patient_bootstrap_ci
from sklearn.linear_model import LogisticRegression


def train_and_eval_baseline(
    feat_name: Optional[str],
    use_metadata: bool,
    dataset: Dict[str, Any],
    c: float = 1.0,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Fits single-model or metadata baseline on Train+Val (n=163) and evaluates on Test (n=41).
    Strict Zero Leakage.
    """
    tv_mask = dataset["splits_mask"]["train_val"]
    te_mask = dataset["splits_mask"]["test"]
    
    y_tv = dataset["y"][tv_mask]
    y_te = dataset["y"][te_mask]
    
    meta_tv = dataset["metadata"][tv_mask]
    meta_te = dataset["metadata"][te_mask]
    
    if feat_name is None:
        # Metadata only
        x_tv = meta_tv
        x_te = meta_te
    else:
        vis_tv = dataset["features"][feat_name][tv_mask]
        vis_te = dataset["features"][feat_name][te_mask]
        if use_metadata:
            x_tv = np.concatenate([vis_tv, meta_tv], axis=1)
            x_te = np.concatenate([vis_te, meta_te], axis=1)
        else:
            x_tv = vis_tv
            x_te = vis_te
            
    clf = LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=150, tol=1e-3, random_state=seed)
    clf.fit(x_tv, y_tv)
    pred_proba = clf.predict_proba(x_te)
    
    return compute_patient_bootstrap_ci(
        y_true=y_te,
        y_pred_proba=pred_proba,
        n_bootstrap=n_bootstrap,
        seed=seed,
        class_names=CLASS_NAMES
    )


def evaluate_champion_on_test(
    champion: Dict[str, Any],
    dataset: Dict[str, Any],
    use_metadata_override: Optional[bool] = None,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Fits champion configuration on Train+Val (n=163) and evaluates strictly once on Test (n=41).
    """
    tv_mask = dataset["splits_mask"]["train_val"]
    te_mask = dataset["splits_mask"]["test"]
    
    tv_feats = {k: v[tv_mask] for k, v in dataset["features"].items()}
    te_feats = {k: v[te_mask] for k, v in dataset["features"].items()}
    
    tv_meta = dataset["metadata"][tv_mask]
    te_meta = dataset["metadata"][te_mask]
    
    y_tv = dataset["y"][tv_mask]
    y_te = dataset["y"][te_mask]
    
    cfg = champion["config"]
    family = cfg["family"]
    params = dict(cfg.get("params", {}))
    
    if use_metadata_override is not None:
        params["use_metadata"] = use_metadata_override
        
    if family == "early":
        model = EarlyFusionClassifier(
            c=params.get("c", 1.0),
            l2_norm_per_stream=params.get("l2_norm", False),
            pca_dim=params.get("pca_dim", None),
            use_metadata=params.get("use_metadata", True),
            class_weight=params.get("class_weight", None),
            random_state=seed
        )
    elif family == "late":
        model = LateFusionClassifier(
            strategy=params.get("strategy", "uniform"),
            c=params.get("c", 1.0),
            tau=params.get("tau", 1.5),
            use_metadata=params.get("use_metadata", True),
            class_weight=params.get("class_weight", None),
            random_state=seed
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
            random_state=seed
        )
    else:
        raise ValueError(f"Unknown family: {family}")
        
    model.fit(tv_feats, tv_meta, y_tv)
    pred_proba = model.predict_proba(te_feats, te_meta)
    
    return compute_patient_bootstrap_ci(
        y_true=y_te,
        y_pred_proba=pred_proba,
        n_bootstrap=n_bootstrap,
        seed=seed,
        class_names=CLASS_NAMES
    )


def run_pipeline_for_seed(seed: int, k_se: float = 0.5, patience: int = 5, n_bootstrap: int = 1000, agent_version: str = "v3", output_dir: str = "artifacts") -> Dict[str, Any]:
    print("\n" + "#" * 80, flush=True)
    print(f"### EXECUTING PHASE 2 PIPELINE FOR SEED {seed} (Agent: {agent_version.upper()})", flush=True)
    print("#" * 80 + "\n", flush=True)
    
    # 1. Run Autonomous Search Agent
    agent_cls = get_agent_cls(agent_version)
    if agent_version == "v3":
        agent = agent_cls(seed=seed, k_se=k_se)
    else:
        agent = agent_cls(seed=seed, k_se=k_se, max_consecutive_failures=patience)
    champion = agent.run_search()
    
    # Save search logs
    agent.save_logs(
        json_path=os.path.join(output_dir, f"agent_decision_log_seed_{seed}.json"),
        md_path=os.path.join(output_dir, f"agent_search_summary_seed_{seed}.md")
    )
    if seed == 42:
        agent.save_logs(
            json_path=os.path.join(output_dir, "agent_decision_log.json"),
            md_path=os.path.join(output_dir, "agent_search_summary.md")
        )
        
    # 2. Load dataset for champion (using champion pooling & gamma_meta)
    champ_ds = agent.get_dataset(
        pooling=champion.get("pooling", "mean"),
        gamma_meta=champion.get("gamma_meta", 1.0)
    )
    
    # Standard dataset for single-model baselines (mean pooling, gamma_meta=1.0)
    std_ds = champ_ds
    
    print("\n" + "=" * 75, flush=True)
    print(f"EVALUATING TEST SET (n=41) & COMPUTING 1,000 BOOTSTRAP CIs (Seed {seed})", flush=True)
    print("=" * 75, flush=True)
    
    # Evaluate 10-Row Configurations
    results = {}
    
    # 1. Metadata only
    print("Evaluating: Metadata only (Age + Sex)...", flush=True)
    results["Metadata only (Age + Sex)"] = train_and_eval_baseline(
        feat_name=None, use_metadata=True, dataset=std_ds, seed=seed, n_bootstrap=n_bootstrap
    )
    
    # 2. UNI2-h alone (no metadata)
    print("Evaluating: UNI2-h alone (no metadata)...", flush=True)
    results["UNI2-h alone (no metadata)"] = train_and_eval_baseline(
        feat_name="uni2", use_metadata=False, dataset=std_ds, seed=seed, n_bootstrap=n_bootstrap
    )
    
    # 3. UNI2-h alone + Metadata
    print("Evaluating: UNI2-h alone + Metadata...", flush=True)
    results["UNI2-h alone + Metadata"] = train_and_eval_baseline(
        feat_name="uni2", use_metadata=True, dataset=std_ds, seed=seed, n_bootstrap=n_bootstrap
    )
    
    # 4. Virchow2 alone (no metadata)
    print("Evaluating: Virchow2 alone (no metadata)...", flush=True)
    results["Virchow2 alone (no metadata)"] = train_and_eval_baseline(
        feat_name="virchow2", use_metadata=False, dataset=std_ds, seed=seed, n_bootstrap=n_bootstrap
    )
    
    # 5. Virchow2 alone + Metadata
    print("Evaluating: Virchow2 alone + Metadata...", flush=True)
    results["Virchow2 alone + Metadata"] = train_and_eval_baseline(
        feat_name="virchow2", use_metadata=True, dataset=std_ds, seed=seed, n_bootstrap=n_bootstrap
    )
    
    # 6. Prov-GigaPath alone (no metadata)
    print("Evaluating: Prov-GigaPath alone (no metadata)...", flush=True)
    results["Prov-GigaPath alone (no metadata)"] = train_and_eval_baseline(
        feat_name="gigapath", use_metadata=False, dataset=std_ds, seed=seed, n_bootstrap=n_bootstrap
    )
    
    # 7. Prov-GigaPath alone + Metadata
    print("Evaluating: Prov-GigaPath alone + Metadata...", flush=True)
    results["Prov-GigaPath alone + Metadata"] = train_and_eval_baseline(
        feat_name="gigapath", use_metadata=True, dataset=std_ds, seed=seed, n_bootstrap=n_bootstrap
    )
    
    # 8. Best Single Model + Metadata
    single_model_keys = ["UNI2-h alone + Metadata", "Virchow2 alone + Metadata", "Prov-GigaPath alone + Metadata"]
    best_single_key = max(single_model_keys, key=lambda k: results[k]["macro_auroc"]["value"])
    results["Best Single Model + Metadata"] = results[best_single_key]
    print(f"Identified Best Single Model + Metadata: {best_single_key}", flush=True)
    
    # 9. Fused (Agent-Selected) (no metadata ablation)
    print("Evaluating: Fused (Agent-Selected) (no metadata)...", flush=True)
    results["Fused (Agent-Selected) (no metadata)"] = evaluate_champion_on_test(
        champion=champion, dataset=champ_ds, use_metadata_override=False, seed=seed, n_bootstrap=n_bootstrap
    )
    
    # 10. Fused (Agent-Selected) + Metadata (Champion)
    print("Evaluating: Fused (Agent-Selected) + Metadata (Champion)...", flush=True)
    results["Fused (Agent-Selected) + Metadata"] = evaluate_champion_on_test(
        champion=champion, dataset=champ_ds, use_metadata_override=True, seed=seed, n_bootstrap=n_bootstrap
    )
    
    # Format Comparison Table
    table_rows = [
        "| Configuration | Macro AUROC | 95% Bootstrap CI | Balanced Acc. | 95% Bootstrap CI |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]
    for name, res in results.items():
        auroc_val = res["macro_auroc"]["value"]
        auroc_ci = res["macro_auroc"]["ci_str"]
        bal_val = res["balanced_acc"]["value"]
        bal_ci = res["balanced_acc"]["ci_str"]
        is_champion = (name == "Fused (Agent-Selected) + Metadata")
        fmt = "**" if is_champion else ""
        table_rows.append(f"| {fmt}{name}{fmt} | {fmt}{auroc_val:.4f}{fmt} | {fmt}{auroc_ci}{fmt} | {fmt}{bal_val:.4f}{fmt} | {fmt}{bal_ci}{fmt} |")
        
    table_md = "\n".join(table_rows)
    print("\n" + "=" * 75, flush=True)
    print(f"REQUIRED COMPARISON TABLE (§5.3) - SEED {seed}", flush=True)
    print("=" * 75, flush=True)
    print(table_md, flush=True)
    print("=" * 75 + "\n", flush=True)
    
    # Save seed artifacts
    seed_summary = {
        "seed": seed,
        "champion_trial": champion["trial"],
        "champion_cv_auroc": champion["mean_auroc"],
        "champion_config": champion["config"],
        "results": results
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"phase2_results_seed_{seed}.json"), "w") as f:
        json.dump(seed_summary, f, indent=2)
    with open(os.path.join(output_dir, f"phase2_comparison_table_seed_{seed}.md"), "w") as f:
        f.write(f"# Phase 2 Model Comparison Table (Seed {seed})\n\n" + table_md + "\n")
        
    return seed_summary


def aggregate_multi_seed_results(seed_results: List[Dict[str, Any]], output_dir: str = "artifacts"):
    print("\n" + "=" * 80, flush=True)
    print("AGGREGATING MULTI-SEED EVALUATION ACROSS SEEDS (42, 1337, 2026)", flush=True)
    print("=" * 80 + "\n", flush=True)
    
    configs = list(seed_results[0]["results"].keys())
    multi_seed_table = [
        "| Configuration | Seed 42 AUROC | Seed 1337 AUROC | Seed 2026 AUROC | Mean ± SD AUROC | Mean ± SD Balanced Acc |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    summary_data = {}
    
    for cfg in configs:
        aurocs = [sr["results"][cfg]["macro_auroc"]["value"] for sr in seed_results]
        bal_accs = [sr["results"][cfg]["balanced_acc"]["value"] for sr in seed_results]
        
        mean_auc, std_auc = float(np.mean(aurocs)), float(np.std(aurocs, ddof=1))
        mean_bal, std_bal = float(np.mean(bal_accs)), float(np.std(bal_accs, ddof=1))
        
        is_champion = (cfg == "Fused (Agent-Selected) + Metadata")
        fmt = "**" if is_champion else ""
        
        multi_seed_table.append(
            f"| {fmt}{cfg}{fmt} | {aurocs[0]:.4f} | {aurocs[1]:.4f} | {aurocs[2]:.4f} | "
            f"{fmt}{mean_auc:.4f} ± {std_auc:.4f}{fmt} | {fmt}{mean_bal:.4f} ± {std_bal:.4f}{fmt} |"
        )
        
        summary_data[cfg] = {
            "seeds_auroc": {sr["seed"]: a for sr, a in zip(seed_results, aurocs)},
            "mean_auroc": mean_auc,
            "std_auroc": std_auc,
            "seeds_balanced_acc": {sr["seed"]: b for sr, b in zip(seed_results, bal_accs)},
            "mean_balanced_acc": mean_bal,
            "std_balanced_acc": std_bal
        }
        
    table_md = "\n".join(multi_seed_table)
    print(table_md, flush=True)
    print("\n" + "=" * 80 + "\n", flush=True)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "phase2_multi_seed_comparison.md"), "w") as f:
        f.write("# Phase 2 Multi-Seed Repeatability Summary (Seeds 42, 1337, 2026)\n\n" + table_md + "\n")
    with open(os.path.join(output_dir, "phase2_multi_seed_comparison.json"), "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Multi-seed summary saved to {output_dir}/phase2_multi_seed_comparison.md and .json", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Autonomous Multi-Foundation-Model Fusion Agent Runner")
    parser.add_argument("--seed", type=int, default=42, help="Primary random seed (default: 42)")
    parser.add_argument("--all-seeds", action="store_true", help="Run full multi-seed evaluation across seeds 42, 1337, 2026")
    parser.add_argument("--agent-version", type=str, default="v3", choices=["v1", "v2", "v3", "v4", "llm"], help="Agent search version: v1 (sequential), v2 (principled exploitation), v3 (3-stage hierarchical stability), v4/llm (LLM-powered autonomous fusion agent)")
    parser.add_argument("--k-se", type=float, default=0.5, help="SE Guardrail threshold factor (default: 0.5)")
    parser.add_argument("--patience", type=int, default=5, help="Max consecutive failures before early stopping (default: 5)")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap resamples (default: 1000)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name (saved under artifacts/results/<exp_name>)")
    args = parser.parse_args()
    
    # Determine destination directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.exp_name is not None:
        output_dir = os.path.join("artifacts", "results", args.exp_name)
    else:
        output_dir = "artifacts"

    os.makedirs(output_dir, exist_ok=True)

    # Save experiment configuration record
    try:
        import subprocess
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = "unknown"

    config_record = {
        "experiment_name": args.exp_name or os.path.basename(output_dir),
        "output_dir": output_dir,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_hash,
        "agent_version": args.agent_version,
        "k_se": args.k_se,
        "patience": args.patience,
        "n_bootstrap": args.n_bootstrap,
        "all_seeds": args.all_seeds,
        "seeds": [42, 1337, 2026] if args.all_seeds else [args.seed]
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_record, f, indent=2)
    try:
        import yaml
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(config_record, f, default_flow_style=False)
    except Exception:
        pass
    print(f"Configuration metadata saved to {output_dir}/config.yaml and config.json", flush=True)

    t_start = time.time()
    if args.all_seeds:
        seed_list = [42, 1337, 2026]
        all_res = []
        for s in seed_list:
            res = run_pipeline_for_seed(seed=s, k_se=args.k_se, patience=args.patience, n_bootstrap=args.n_bootstrap, agent_version=args.agent_version, output_dir=output_dir)
            all_res.append(res)
        aggregate_multi_seed_results(all_res, output_dir=output_dir)
    else:
        run_pipeline_for_seed(seed=args.seed, k_se=args.k_se, patience=args.patience, n_bootstrap=args.n_bootstrap, agent_version=args.agent_version, output_dir=output_dir)
        
    print(f"\nPhase 2 execution finished in {time.time() - t_start:.2f} seconds.", flush=True)

if __name__ == "__main__":
    main()
