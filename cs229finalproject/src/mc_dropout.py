"""
Monte Carlo Dropout for uncertainty estimation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict


def enable_dropout(model: nn.Module) -> None:
    """
    Enable dropout layers during evaluation.
    By default, dropout is disabled during model.eval().
    This function keeps dropout ON for MC Dropout inference.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    X: np.ndarray,
    n_passes: int = 50,
    device: str = 'cpu',
) -> Dict:
    """
    Run Monte Carlo Dropout inference for MLP.

    Args:
        model: Trained PyTorch model with dropout layers
        X: Input features (n_samples, n_features)
        n_passes: Number of stochastic forward passes
        device: 'cpu' or 'cuda'

    Returns:
        {
            'risk_scores': np.ndarray (n_samples,) — mean predicted probability
            'uncertainty_scores': np.ndarray (n_samples,) — variance across passes
            'all_predictions': np.ndarray (n_samples, n_passes) — all predictions
        }
    """
    model.eval()
    enable_dropout(model)

    X_tensor = torch.FloatTensor(X).to(device)
    preds = []

    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)

    all_predictions = np.stack(preds, axis=1)   # (n_samples, n_passes)
    risk_scores = all_predictions.mean(axis=1)
    uncertainty_scores = all_predictions.var(axis=1)

    return {
        'risk_scores': risk_scores,
        'uncertainty_scores': uncertainty_scores,
        'all_predictions': all_predictions,
    }


def mc_dropout_predict_lstm(
    model: nn.Module,
    X: np.ndarray,
    n_passes: int = 50,
    device: str = 'cpu',
) -> Dict:
    """
    MC Dropout for LSTM — returns predictions at each time step.

    Args:
        model: Trained RecidivismLSTM
        X: (n_samples, seq_len, n_features)
        n_passes: Number of stochastic forward passes
        device: 'cpu' or 'cuda'

    Returns:
        {
            'risk_scores': np.ndarray (n_samples,) — final year mean probability
            'uncertainty_scores': np.ndarray (n_samples,) — final year variance
            'temporal_risk': np.ndarray (n_samples, seq_len) — mean at each step
            'temporal_uncertainty': np.ndarray (n_samples, seq_len) — var at each step
            'all_predictions': np.ndarray (n_samples, n_passes) — final year predictions
        }
    """
    model.eval()
    enable_dropout(model)

    X_tensor = torch.FloatTensor(X).to(device)
    preds = []   # each element: (n_samples, seq_len)

    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(X_tensor)                   # (n_samples, seq_len)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)

    all_preds = np.stack(preds, axis=2)               # (n_samples, seq_len, n_passes)
    temporal_risk = all_preds.mean(axis=2)            # (n_samples, seq_len)
    temporal_uncertainty = all_preds.var(axis=2)      # (n_samples, seq_len)

    risk_scores = temporal_risk[:, -1]
    uncertainty_scores = temporal_uncertainty[:, -1]
    all_predictions = all_preds[:, -1, :]             # (n_samples, n_passes)

    return {
        'risk_scores': risk_scores,
        'uncertainty_scores': uncertainty_scores,
        'temporal_risk': temporal_risk,
        'temporal_uncertainty': temporal_uncertainty,
        'all_predictions': all_predictions,
    }
