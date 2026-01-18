"""
Evaluation metrics for binding site prediction.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_class: int = 2
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted class labels
        labels: Ground truth labels
        ignore_class: Class to ignore (padding)
        
    Returns:
        Dictionary with precision, recall, f1, accuracy
    """
    # Flatten if needed
    preds = predictions.view(-1).cpu().numpy()
    lbls = labels.view(-1).cpu().numpy()
    
    # Filter out ignore class
    mask = lbls != ignore_class
    preds = preds[mask]
    lbls = lbls[mask]
    
    # Binary classification (class 1 = positive)
    tp = np.sum((preds == 1) & (lbls == 1))
    fp = np.sum((preds == 1) & (lbls == 0))
    fn = np.sum((preds == 0) & (lbls == 1))
    tn = np.sum((preds == 0) & (lbls == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(preds) if len(preds) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def compute_auc(
    probabilities: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute AUC-ROC score.
    
    Args:
        probabilities: Predicted probabilities for positive class
        labels: Ground truth binary labels
        
    Returns:
        AUC-ROC score
    """
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels, probabilities)
    except ImportError:
        print("sklearn required for AUC computation")
        return 0.0
