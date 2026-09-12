import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["PYTHONUNBUFFERED"] = "1"

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from src.fusion.data import load_multimodal_dataset, get_train_val_cv_folds
from src.fusion.models import LateFusionClassifier

def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, tau: float = 0.0, priors = None):
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

def run_experiment():
    seeds = [42, 1337, 2026]
    poolings = ['concat', 'mean']
    
    print("=" * 110)
    print("VIRCHOW2 REPRESENTATION BENCHMARK: FULL (2560-d) vs CLASS TOKEN ONLY (1280-d) vs PATCH ONLY (1280-d)")
    print("=" * 110)

    summary_tables = {}

    for pooling in poolings:
        print(f"\n>>> EVALUATING SLIDE POOLING: {pooling.upper()} <<<")
        
        # We will collect results per variant
        # Variants:
        # 1. Virchow2 Full (CLS + Patch)
        # 2. Virchow2 Class Token Only
        # 3. Virchow2 Patch Tokens Only
        # Both with and without metadata
        # Plus Late Fusion champion with Full Virchow2 vs CLS Virchow2
        
        results = {}
        
        for seed in seeds:
            print(f"  Loading dataset for seed {seed} ({pooling})...")
            ds = load_multimodal_dataset(seed=seed, pooling=pooling)
            tv_m = ds['splits_mask']['train_val']
            te_m = ds['splits_mask']['test']
            
            y_tv, y_te = ds['y'][tv_m], ds['y'][te_m]
            meta_tv, meta_te = ds['metadata'][tv_m], ds['metadata'][te_m]
            priors = np.bincount(y_tv, minlength=7) / len(y_tv)
            
            v_tv_full = ds['features']['virchow2'][tv_m]
            v_te_full = ds['features']['virchow2'][te_m]
            
            if pooling == 'mean':
                # Full is 2560: [:1280] is CLS, [1280:] is patch
                cls_tv, cls_te = v_tv_full[:, :1280], v_te_full[:, :1280]
                patch_tv, patch_te = v_tv_full[:, 1280:], v_te_full[:, 1280:]
            else: # concat
                # Full is 5120: mean(2560) + max(2560)
                # CLS mean is [:1280], CLS max is [2560:3840]
                cls_tv = np.concatenate([v_tv_full[:, :1280], v_tv_full[:, 2560:3840]], axis=1)
                cls_te = np.concatenate([v_te_full[:, :1280], v_te_full[:, 2560:3840]], axis=1)
                patch_tv = np.concatenate([v_tv_full[:, 1280:2560], v_tv_full[:, 3840:5120]], axis=1)
                patch_te = np.concatenate([v_te_full[:, 1280:2560], v_te_full[:, 3840:5120]], axis=1)
                
            variants = {
                "Virchow2 Full (CLS + Patch)": (v_tv_full, v_te_full),
                "Virchow2 Class Token Only": (cls_tv, cls_te),
                "Virchow2 Patch Token Only": (patch_tv, patch_te)
            }
            
            # 5-fold CV splits on TV
            cv_folds = get_train_val_cv_folds(ds, n_splits=5, cv_seed=seed)
            # Map fold indices from full dataset to tv indices
            tv_indices = np.where(tv_m)[0]
            idx_to_tv_pos = {idx: pos for pos, idx in enumerate(tv_indices)}
            
            for var_name, (X_tv_var, X_te_var) in variants.items():
                for use_meta in [False, True]:
                    row_name = f"{var_name}{' + Meta' if use_meta else ' (No Meta)'}"
                    if row_name not in results:
                        results[row_name] = {'cv_auc': [], 'test_auc': [], 'test_bacc': [], 'test_adj_bacc': []}
                        
                    if use_meta:
                        X_tv = np.concatenate([X_tv_var, meta_tv], axis=1)
                        X_te = np.concatenate([X_te_var, meta_te], axis=1)
                    else:
                        X_tv = X_tv_var
                        X_te = X_te_var
                        
                    # 5-fold CV
                    fold_aucs = []
                    for tr_idx, va_idx in cv_folds:
                        tr_pos = [idx_to_tv_pos[i] for i in tr_idx]
                        va_pos = [idx_to_tv_pos[i] for i in va_idx]
                        
                        clf_cv = LogisticRegression(C=1.0, max_iter=150, tol=1e-3, random_state=seed)
                        clf_cv.fit(X_tv[tr_pos], y_tv[tr_pos])
                        p_val = clf_cv.predict_proba(X_tv[va_pos])
                        m_cv = compute_metrics(y_tv[va_pos], p_val)
                        fold_aucs.append(m_cv['macro_auroc'])
                    results[row_name]['cv_auc'].append(np.mean(fold_aucs))
                    
                    # Test eval
                    clf = LogisticRegression(C=1.0, max_iter=150, tol=1e-3, random_state=seed)
                    clf.fit(X_tv, y_tv)
                    p_te = clf.predict_proba(X_te)
                    
                    m_base = compute_metrics(y_te, p_te, tau=0.0)
                    m_adj = compute_metrics(y_te, p_te, tau=0.4, priors=priors)
                    
                    results[row_name]['test_auc'].append(m_base['macro_auroc'])
                    results[row_name]['test_bacc'].append(m_base['balanced_acc'])
                    results[row_name]['test_adj_bacc'].append(m_adj['balanced_acc'])

            # Also Late Fusion Champion (UNI2 + Virchow2 + GigaPath) with Full vs CLS Virchow2
            for use_meta in [False, True]:
                for v_type, v_tv_sub, v_te_sub in [("Full", v_tv_full, v_te_full), ("CLS Only", cls_tv, cls_te)]:
                    fusion_name = f"Fused Champion (Virchow2 {v_type}){' + Meta' if use_meta else ' (No Meta)'}"
                    if fusion_name not in results:
                        results[fusion_name] = {'cv_auc': [], 'test_auc': [], 'test_bacc': [], 'test_adj_bacc': []}
                    
                    # Late fusion
                    # Train individual models on TV
                    u_tv, u_te = ds['features']['uni2'][tv_m], ds['features']['uni2'][te_m]
                    g_tv, g_te = ds['features']['gigapath'][tv_m], ds['features']['gigapath'][te_m]
                    
                    # Fit 3 classifiers
                    def fit_pred(x_train, x_test):
                        if use_meta:
                            x_tr = np.concatenate([x_train, meta_tv], axis=1)
                            x_te_m = np.concatenate([x_test, meta_te], axis=1)
                        else:
                            x_tr = x_train
                            x_te_m = x_test
                        c = LogisticRegression(C=1.0, max_iter=150, tol=1e-3, random_state=seed)
                        c.fit(x_tr, y_tv)
                        # logits with temperature tau=1.5
                        logits = c.decision_function(x_te_m) / 1.5
                        # softmax
                        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                        return exp_l / np.sum(exp_l, axis=-1, keepdims=True)
                    
                    p_u = fit_pred(u_tv, u_te)
                    p_v = fit_pred(v_tv_sub, v_te_sub)
                    p_g = fit_pred(g_tv, g_te)
                    
                    p_fused = (p_u + p_v + p_g) / 3.0
                    m_base = compute_metrics(y_te, p_fused, tau=0.0)
                    m_adj = compute_metrics(y_te, p_fused, tau=0.4, priors=priors)
                    
                    results[fusion_name]['test_auc'].append(m_base['macro_auroc'])
                    results[fusion_name]['test_bacc'].append(m_base['balanced_acc'])
                    results[fusion_name]['test_adj_bacc'].append(m_adj['balanced_acc'])
                    results[fusion_name]['cv_auc'].append(0.0) # placeholder

        summary_tables[pooling] = results

    # Print Final Summary Tables
    for pooling, res in summary_tables.items():
        print("\n" + "=" * 115)
        print(f"FINAL SUMMARY: POOLING = {pooling.upper()}")
        print("=" * 115)
        header = f"{'Model Configuration':<42} | {'5-Fold CV AUROC':<17} | {'Test AUROC (Mean±SD)':<22} | {'Seed 42 / 1337 / 2026':<24} | {'BalAcc (Base -> Adj)':<20}"
        print(header)
        print("-" * 115)
        for name, metrics in res.items():
            cv_str = f"{np.mean(metrics['cv_auc']):.4f} ± {np.std(metrics['cv_auc']):.4f}" if np.mean(metrics['cv_auc']) > 0 else "N/A"
            t_auc_m, t_auc_s = np.mean(metrics['test_auc']), np.std(metrics['test_auc'])
            auc_str = f"{t_auc_m:.4f} ± {t_auc_s:.4f}"
            seeds_str = f"{metrics['test_auc'][0]:.4f} / {metrics['test_auc'][1]:.4f} / {metrics['test_auc'][2]:.4f}"
            bacc_str = f"{np.mean(metrics['test_bacc']):.3f} -> {np.mean(metrics['test_adj_bacc']):.3f}"
            print(f"{name:<42} | {cv_str:<17} | {auc_str:<22} | {seeds_str:<24} | {bacc_str:<20}")

if __name__ == "__main__":
    run_experiment()
