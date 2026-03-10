"""
Evaluation metrics for recidivism prediction.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
from typing import Dict, List, Optional


def compute_standard_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """
    Compute standard classification metrics.

    Returns:
        {
            'auc': float,
            'brier_score': float,
            'avg_precision': float,
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1': float
        }
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'auc': float(roc_auc_score(y_true, y_prob)),
        'brier_score': float(brier_score_loss(y_true, y_prob)),
        'avg_precision': float(average_precision_score(y_true, y_prob)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.

    ECE measures how well predicted probabilities match observed frequencies.
    Lower is better. 0 = perfectly calibrated.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

    return float(ece)


def compute_uncertainty_metrics(
    y_true: np.ndarray,
    risk_scores: np.ndarray,
    uncertainty_scores: np.ndarray,
) -> Dict:
    """
    Compute metrics that evaluate uncertainty usefulness.

    Returns:
        {
            'uncertainty_error_correlation': float,
            'error_rate_low_uncertainty': float,
            'error_rate_high_uncertainty': float,
            'selective_prediction': List[Dict]
        }
    """
    errors = (np.round(risk_scores) != y_true).astype(float)

    # Pearson correlation between uncertainty and error
    corr = float(np.corrcoef(uncertainty_scores, errors)[0, 1])

    # High/low uncertainty split at median
    median_unc = np.median(uncertainty_scores)
    low_mask = uncertainty_scores <= median_unc
    high_mask = ~low_mask

    error_rate_low = float(errors[low_mask].mean()) if low_mask.sum() > 0 else float('nan')
    error_rate_high = float(errors[high_mask].mean()) if high_mask.sum() > 0 else float('nan')

    sp = selective_prediction_analysis(y_true, risk_scores, uncertainty_scores)

    return {
        'uncertainty_error_correlation': corr,
        'error_rate_low_uncertainty': error_rate_low,
        'error_rate_high_uncertainty': error_rate_high,
        'selective_prediction': sp,
    }


def selective_prediction_analysis(
    y_true: np.ndarray,
    risk_scores: np.ndarray,
    uncertainty_scores: np.ndarray,
    coverages: Optional[List[float]] = None,
) -> List[Dict]:
    """
    Compute risk-coverage tradeoff.

    At each coverage level, keep the (coverage * 100)% most confident predictions.
    """
    if coverages is None:
        coverages = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Sort by ascending uncertainty (most confident first)
    sorted_idx = np.argsort(uncertainty_scores)
    results = []

    for cov in coverages:
        n_keep = max(int(cov * len(y_true)), 2)
        idx = sorted_idx[:n_keep]
        y_sub = y_true[idx]
        p_sub = risk_scores[idx]

        try:
            auc = float(roc_auc_score(y_sub, p_sub))
        except ValueError:
            auc = float('nan')

        results.append({
            'coverage': float(cov),
            'n_samples': n_keep,
            'auc': auc,
            'accuracy': float(accuracy_score(y_sub, (p_sub >= 0.5).astype(int))),
            'brier_score': float(brier_score_loss(y_sub, p_sub)),
        })

    return results
