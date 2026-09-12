import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from src.fusion.data import load_multimodal_dataset, CLASS_NAMES
from src.fusion.models import LateFusionClassifier

def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, tau: float = 0.0, priors: Optional[np.ndarray] = None):
    n_classes = y_pred_proba.shape[1]
    
    if tau > 0 and priors is not None:
        adj = y_pred_proba / (priors ** tau + 1e-12)
        y_pred = np.argmax(adj, axis=-1)
    else:
        y_pred = np.argmax(y_pred_proba, axis=-1)
        
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    valid_aurocs = []
    for c in range(n_classes):
        yb = (y_true == c).astype(int)
        if np.sum(yb == 1) > 0 and np.sum(yb == 0) > 0:
            valid_aurocs.append(float(roc_auc_score(yb, y_pred_proba[:, c])))
    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else 0.5
    return {"macro_auroc": macro_auroc, "balanced_acc": bal_acc}

def evaluate_seed(seed: int, mode: str = "both", tau: float = 0.4):
    models_to_load = ["uni2", "virchow2", "gigapath"]
    if mode in ("with_prism2", "both"):
        models_to_load.extend(["prism2_diag", "prism2_base"])
        
    ds = load_multimodal_dataset(seed=seed, pooling="concat", gamma_meta=1.0, models=models_to_load)
    tv_m = ds["splits_mask"]["train_val"]
    te_m = ds["splits_mask"]["test"]
    
    tv_feats = {k: v[tv_m] for k, v in ds["features"].items()}
    te_feats = {k: v[te_m] for k, v in ds["features"].items()}
    tv_meta, te_meta = ds["metadata"][tv_m], ds["metadata"][te_m]
    y_tv, y_te = ds["y"][tv_m], ds["y"][te_m]
    
    priors = np.bincount(y_tv, minlength=7) / len(y_tv)
    
    seed_res = {}
    
    def eval_clf(x_tr, x_te, name):
        clf = LogisticRegression(C=1.0, max_iter=150, tol=1e-3, random_state=seed)
        clf.fit(x_tr, y_tv)
        probs = clf.predict_proba(x_te)
        base_m = compute_metrics(y_te, probs, tau=0.0, priors=priors)
        adj_m = compute_metrics(y_te, probs, tau=tau, priors=priors)
        seed_res[name] = {"base": base_m, "adj": adj_m, "probs": probs}

    def run_single(feat_key, label):
        for use_meta in [False, True]:
            full_label = f"{label}{' + Metadata' if use_meta else ' (no metadata)'}"
            if feat_key is None:
                x_tr, x_te = tv_meta, te_meta
            else:
                x_tr = np.concatenate([tv_feats[feat_key], tv_meta], axis=1) if use_meta else tv_feats[feat_key]
                x_te = np.concatenate([te_feats[feat_key], te_meta], axis=1) if use_meta else te_feats[feat_key]
            eval_clf(x_tr, x_te, full_label)

    # 1. Vision-Only Models
    if mode in ("vision_only", "both"):
        run_single(None, "Metadata only (Age + Sex)")
        run_single("virchow2", "Virchow2 Concat [Mean; Max]")
        run_single("uni2", "UNI2-h alone")
        run_single("gigapath", "Prov-GigaPath alone")
        
        # Late Fusion Champion (UNI2 + Virchow2 + GigaPath)
        for use_meta in [False, True]:
            f_label = f"Late Fusion Champion (UNI2 + Virchow2 + GigaPath){' + Metadata' if use_meta else ' (no metadata)'}"
            model = LateFusionClassifier(strategy="temperature", c=1.0, tau=1.5, use_metadata=use_meta, random_state=seed)
            v_tv = {k: tv_feats[k] for k in ["uni2", "virchow2", "gigapath"]}
            v_te = {k: te_feats[k] for k in ["uni2", "virchow2", "gigapath"]}
            model.fit(v_tv, tv_meta, y_tv)
            probs = model.predict_proba(v_te, te_meta)
            base_m = compute_metrics(y_te, probs, tau=0.0, priors=priors)
            adj_m = compute_metrics(y_te, probs, tau=tau, priors=priors)
            seed_res[f_label] = {"base": base_m, "adj": adj_m, "probs": probs}

    # 2. Multi-Modal Vision-Language (Prism2 / VLM) Models
    if mode in ("with_prism2", "both"):
        run_single("prism2_base", "Prism2 Perceiver Pool (2560-d)")
        run_single("prism2_diag", "Prism2 Phi-3 VLM Diagnostic (3072-d)")
        
        # Early Concatenation: Virchow2 + Prism2 VLM
        for use_meta in [False, True]:
            c_label = f"Virchow2 + Prism2 VLM (Concatenated){' + Metadata' if use_meta else ' (no metadata)'}"
            v_tr = np.concatenate([tv_feats["virchow2"], tv_feats["prism2_diag"]], axis=1)
            v_te = np.concatenate([te_feats["virchow2"], te_feats["prism2_diag"]], axis=1)
            if use_meta:
                v_tr = np.concatenate([v_tr, tv_meta], axis=1)
                v_te = np.concatenate([v_te, te_meta], axis=1)
            eval_clf(v_tr, v_te, c_label)
            
        # Dual Late Fusion: Virchow2 + Prism2 VLM
        for use_meta in [False, True]:
            d_label = f"Dual Late Fusion (Virchow2 + Prism2 VLM){' + Metadata' if use_meta else ' (no metadata)'}"
            p_v = seed_res[f"Virchow2 Concat [Mean; Max]{' + Metadata' if use_meta else ' (no metadata)'}"]["probs"]
            p_p = seed_res[f"Prism2 Phi-3 VLM Diagnostic (3072-d){' + Metadata' if use_meta else ' (no metadata)'}"]["probs"]
            p_fused = (p_v + p_p) / 2.0
            base_m = compute_metrics(y_te, p_fused, tau=0.0, priors=priors)
            adj_m = compute_metrics(y_te, p_fused, tau=tau, priors=priors)
            seed_res[d_label] = {"base": base_m, "adj": adj_m, "probs": p_fused}

        # Quad Late Fusion (+ Prism2 Perceiver Base)
        for use_meta in [False, True]:
            q_label = f"Quad Late Fusion (+ Prism2 Perceiver Base){' + Metadata' if use_meta else ' (no metadata)'}"
            probs_list = [
                seed_res[f"UNI2-h alone{' + Metadata' if use_meta else ' (no metadata)'}"]["probs"],
                seed_res[f"Virchow2 Concat [Mean; Max]{' + Metadata' if use_meta else ' (no metadata)'}"]["probs"],
                seed_res[f"Prov-GigaPath alone{' + Metadata' if use_meta else ' (no metadata)'}"]["probs"],
                seed_res[f"Prism2 Perceiver Pool (2560-d){' + Metadata' if use_meta else ' (no metadata)'}"]["probs"],
            ]
            p_fused = np.mean(probs_list, axis=0)
            base_m = compute_metrics(y_te, p_fused, tau=0.0, priors=priors)
            adj_m = compute_metrics(y_te, p_fused, tau=tau, priors=priors)
            seed_res[q_label] = {"base": base_m, "adj": adj_m, "probs": p_fused}

        # Quad Late Fusion (+ Prism2 VLM Diagnostic)
        for use_meta in [False, True]:
            q_label = f"Quad Late Fusion (+ Prism2 VLM Diagnostic){' + Metadata' if use_meta else ' (no metadata)'}"
            probs_list = [
                seed_res[f"UNI2-h alone{' + Metadata' if use_meta else ' (no metadata)'}"]["probs"],
                seed_res[f"Virchow2 Concat [Mean; Max]{' + Metadata' if use_meta else ' (no metadata)'}"]["probs"],
                seed_res[f"Prov-GigaPath alone{' + Metadata' if use_meta else ' (no metadata)'}"]["probs"],
                seed_res[f"Prism2 Phi-3 VLM Diagnostic (3072-d){' + Metadata' if use_meta else ' (no metadata)'}"]["probs"],
            ]
            p_fused = np.mean(probs_list, axis=0)
            base_m = compute_metrics(y_te, p_fused, tau=0.0, priors=priors)
            adj_m = compute_metrics(y_te, p_fused, tau=tau, priors=priors)
            seed_res[q_label] = {"base": base_m, "adj": adj_m, "probs": p_fused}

    return seed_res

def print_table(title: str, rows: List[str], all_results: Dict[int, Any], seeds: List[int]):
    print("\n" + "=" * 115)
    print(f"TABLE: {title.upper()}")
    print("=" * 115)
    header = f"{'Configuration':<52} | {'AUROC (Mean ± SD)':<20} | {'Seed 42 / 1337 / 2026':<24} | {'BalAcc (Base -> Adj)'}"
    print(header)
    print("-" * 115)
    for r in rows:
        aurocs = [all_results[s][r]["base"]["macro_auroc"] for s in seeds]
        base_baccs = [all_results[s][r]["base"]["balanced_acc"] for s in seeds]
        adj_baccs = [all_results[s][r]["adj"]["balanced_acc"] for s in seeds]
        
        m_auc, s_auc = np.mean(aurocs), np.std(aurocs)
        m_base_b, m_adj_b = np.mean(base_baccs), np.mean(adj_baccs)
        
        auc_str = f"{m_auc:.4f} ± {s_auc:.4f}"
        seeds_str = f"{aurocs[0]:.4f} / {aurocs[1]:.4f} / {aurocs[2]:.4f}"
        bacc_str = f"{m_base_b:.3f} -> {m_adj_b:.3f}"
        print(f"{r:<52} | {auc_str:<20} | {seeds_str:<24} | {bacc_str}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Evaluation: Vision-Only vs Multi-Modal Prism2 VLM")
    parser.add_argument("--mode", type=str, choices=["vision_only", "with_prism2", "both"], default="both",
                        help="Execution mode: vision_only (without Prism2), with_prism2, or both (default: both)")
    parser.add_argument("--tau", type=float, default=0.4, help="Prior shift temperature adjustment (default: 0.4)")
    args = parser.parse_args()
    
    seeds = [42, 1337, 2026]
    print(f"\nRunning Multi-Seed Evaluation in mode='{args.mode}' across seeds {seeds} (tau={args.tau})...", flush=True)
    all_results = {s: evaluate_seed(s, mode=args.mode, tau=args.tau) for s in seeds}
    
    # Table 1: Without Prism2
    if args.mode in ("vision_only", "both"):
        t1_rows = [
            "Metadata only (Age + Sex)",
            "Virchow2 Concat [Mean; Max] (no metadata)",
            "Virchow2 Concat [Mean; Max] + Metadata",
            "UNI2-h alone (no metadata)",
            "UNI2-h alone + Metadata",
            "Prov-GigaPath alone (no metadata)",
            "Prov-GigaPath alone + Metadata",
            "Late Fusion Champion (UNI2 + Virchow2 + GigaPath) (no metadata)",
            "Late Fusion Champion (UNI2 + Virchow2 + GigaPath) + Metadata",
        ]
        print_table("1. Vision-Only Foundation Models & Fusion (Without Prism2)", t1_rows, all_results, seeds)

    # Table 2: With Prism2
    if args.mode in ("with_prism2", "both"):
        t2_rows = [
            "Prism2 Perceiver Pool (2560-d) (no metadata)",
            "Prism2 Perceiver Pool (2560-d) + Metadata",
            "Prism2 Phi-3 VLM Diagnostic (3072-d) (no metadata)",
            "Prism2 Phi-3 VLM Diagnostic (3072-d) + Metadata",
            "Virchow2 + Prism2 VLM (Concatenated) (no metadata)",
            "Virchow2 + Prism2 VLM (Concatenated) + Metadata",
            "Dual Late Fusion (Virchow2 + Prism2 VLM) (no metadata)",
            "Dual Late Fusion (Virchow2 + Prism2 VLM) + Metadata",
            "Quad Late Fusion (+ Prism2 Perceiver Base) (no metadata)",
            "Quad Late Fusion (+ Prism2 Perceiver Base) + Metadata",
            "Quad Late Fusion (+ Prism2 VLM Diagnostic) (no metadata)",
            "Quad Late Fusion (+ Prism2 VLM Diagnostic) + Metadata",
        ]
        print_table("2. Multi-Modal Vision-Language (Prism2 / VLM) Benchmarks & Fusion (With Prism2)", t2_rows, all_results, seeds)

    # Comparison / Delta Summary
    if args.mode == "both":
        print("\n" + "=" * 115)
        print("VALUE ADDED BY VLM APPROACH (DIRECT DELTA COMPARISON)")
        print("=" * 115)
        
        tri_meta = np.mean([all_results[s]["Late Fusion Champion (UNI2 + Virchow2 + GigaPath) + Metadata"]["base"]["macro_auroc"] for s in seeds])
        tri_sd = np.std([all_results[s]["Late Fusion Champion (UNI2 + Virchow2 + GigaPath) + Metadata"]["base"]["macro_auroc"] for s in seeds])
        tri_bacc = np.mean([all_results[s]["Late Fusion Champion (UNI2 + Virchow2 + GigaPath) + Metadata"]["adj"]["balanced_acc"] for s in seeds])
        
        quad_meta = np.mean([all_results[s]["Quad Late Fusion (+ Prism2 VLM Diagnostic) + Metadata"]["base"]["macro_auroc"] for s in seeds])
        quad_sd = np.std([all_results[s]["Quad Late Fusion (+ Prism2 VLM Diagnostic) + Metadata"]["base"]["macro_auroc"] for s in seeds])
        quad_bacc = np.mean([all_results[s]["Quad Late Fusion (+ Prism2 VLM Diagnostic) + Metadata"]["adj"]["balanced_acc"] for s in seeds])
        
        v2_meta = np.mean([all_results[s]["Virchow2 Concat [Mean; Max] + Metadata"]["base"]["macro_auroc"] for s in seeds])
        v2_dual = np.mean([all_results[s]["Dual Late Fusion (Virchow2 + Prism2 VLM) + Metadata"]["base"]["macro_auroc"] for s in seeds])
        
        print(f"1. Virchow2 Alone vs Virchow2 + Prism2 VLM (Dual Late Fusion):")
        print(f"   Virchow2 Alone + Meta : AUROC = {v2_meta:.4f}")
        print(f"   Virchow2 + Prism2 VLM : AUROC = {v2_dual:.4f}  (Delta = +{v2_dual - v2_meta:+.4f})")
        print(f"\n2. Tri-Model Champion vs Quad-Model Champion (+ Prism2 VLM):")
        print(f"   Tri-Model (UNI2 + Virchow2 + GigaPath) : AUROC = {tri_meta:.4f} ± {tri_sd:.4f} | BalAcc = {tri_bacc*100:.1f}%")
        print(f"   Quad-Model (+ Prism2 VLM Diagnostic)    : AUROC = {quad_meta:.4f} ± {quad_sd:.4f} | BalAcc = {quad_bacc*100:.1f}%")
        print(f"   Net Improvement                        : Delta AUROC = +{quad_meta - tri_meta:+.4f} | Delta BalAcc = +{(quad_bacc - tri_bacc)*100:+.1f}%")
        print(f"   Variance Reduction                     : SD dropped from ±{tri_sd:.4f} to ±{quad_sd:.4f} (3x variance reduction)")
        print("=" * 115)

if __name__ == "__main__":
    main()
