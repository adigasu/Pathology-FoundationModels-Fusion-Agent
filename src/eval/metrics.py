import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix

def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes Macro AUROC (OvR), Balanced Accuracy, Per-class AUROCs, and Confusion Matrix.
    Robust against edge-case bootstrap resamples where a class may be absent.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float32)
    
    n_classes = y_pred_proba.shape[1]
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]
        
    y_pred = np.argmax(y_pred_proba, axis=-1)
    
    # Balanced Accuracy
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    # Per-class AUROC (One-vs-Rest)
    per_class_auroc = {}
    valid_aurocs = []
    
    for c_idx in range(n_classes):
        c_name = class_names[c_idx] if c_idx < len(class_names) else f"Class_{c_idx}"
        y_binary = (y_true == c_idx).astype(int)
        
        # Check if both positive and negative cases exist
        if np.sum(y_binary == 1) > 0 and np.sum(y_binary == 0) > 0:
            try:
                score = float(roc_auc_score(y_binary, y_pred_proba[:, c_idx]))
                per_class_auroc[c_name] = score
                valid_aurocs.append(score)
            except Exception:
                per_class_auroc[c_name] = float("nan")
        else:
            per_class_auroc[c_name] = float("nan")
            
    macro_auroc = float(np.nanmean(valid_aurocs)) if len(valid_aurocs) > 0 else 0.5
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist()
    
    return {
        "macro_auroc": macro_auroc,
        "balanced_acc": bal_acc,
        "per_class_auroc": per_class_auroc,
        "confusion_matrix": conf_mat
    }
