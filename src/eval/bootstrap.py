import numpy as np
from typing import Dict, Any, List, Optional
from src.eval.metrics import compute_multiclass_metrics

def compute_patient_bootstrap_ci(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes 1,000-sample patient-level bootstrap empirical 95% Confidence Intervals (2.5% to 97.5%).
    Strict patient-level resampling with replacement.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    
    # Point estimates
    point_estimates = compute_multiclass_metrics(y_true, y_pred_proba, class_names=class_names)
    
    boot_macro_aurocs = []
    boot_bal_accs = []
    
    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        res_true = y_true[idx]
        res_proba = y_pred_proba[idx]
        
        m = compute_multiclass_metrics(res_true, res_proba, class_names=class_names)
        boot_macro_aurocs.append(m["macro_auroc"])
        boot_bal_accs.append(m["balanced_acc"])
        
    boot_macro_aurocs = np.array(boot_macro_aurocs)
    boot_bal_accs = np.array(boot_bal_accs)
    
    lower_pct = 100 * (alpha / 2.0)
    upper_pct = 100 * (1.0 - alpha / 2.0)
    
    auroc_ci_low = float(np.percentile(boot_macro_aurocs, lower_pct))
    auroc_ci_high = float(np.percentile(boot_macro_aurocs, upper_pct))
    
    bal_acc_ci_low = float(np.percentile(boot_bal_accs, lower_pct))
    bal_acc_ci_high = float(np.percentile(boot_bal_accs, upper_pct))
    
    return {
        "macro_auroc": {
            "value": point_estimates["macro_auroc"],
            "ci_lower": auroc_ci_low,
            "ci_upper": auroc_ci_high,
            "ci_str": f"[{auroc_ci_low:.3f}, {auroc_ci_high:.3f}]"
        },
        "balanced_acc": {
            "value": point_estimates["balanced_acc"],
            "ci_lower": bal_acc_ci_low,
            "ci_upper": bal_acc_ci_high,
            "ci_str": f"[{bal_acc_ci_low:.3f}, {bal_acc_ci_high:.3f}]"
        },
        "per_class_auroc": point_estimates["per_class_auroc"],
        "confusion_matrix": point_estimates["confusion_matrix"]
    }
