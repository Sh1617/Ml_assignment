import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from typing import Dict, Any

def compute_binary_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Binary ROC AUC for SIIM-ISIC melanoma."""
    return roc_auc_score(y_true, y_pred_proba)

def compute_multiclass_logloss(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Multi-class log loss for Spooky/Whale."""
    pred_proba = np.clip(y_pred_proba, 1e-15, 1.0 - 1e-15)
    row_sums = pred_proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    pred_proba = pred_proba / row_sums
    return log_loss(y_true, pred_proba)

def compute_accuracy(y_true: np.ndarray, y_pred_labels: np.ndarray) -> float:
    """Accuracy for text normalization."""
    return accuracy_score(y_true, y_pred_labels)

def print_metrics(name: str, metrics: Dict[str, float]) -> None:
    print(f"\n=== {name} EVALUATION METRICS ===")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.6f}")
    print("=" * 50)
