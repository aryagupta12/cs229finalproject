"""
Deep Ensemble training and inference for uncertainty estimation.

Trains N independent models with different random seeds.
Predictive uncertainty is the variance across member predictions.
Reference: Lakshminarayanan et al. (2017), "Simple and Scalable
           Predictive Uncertainty Estimation using Deep Ensembles."
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List

from src.models import RecidivismMLP, RecidivismLSTM
from src.train import train_pytorch_model


def train_ensemble_mlp_members(
    input_size: int,
    hidden_sizes: list,
    dropout: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_members: int = 5,
    base_seed: int = 100,
    **train_kwargs,
) -> List[nn.Module]:
    """
    Train N independent MLP models with different random seeds.

    Args:
        input_size: Number of input features
        hidden_sizes: Hidden layer sizes (same for all members)
        dropout: Dropout rate (same for all members)
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_members: Number of ensemble members
        base_seed: Seeds will be base_seed, base_seed+1, ..., base_seed+N-1
        **train_kwargs: Forwarded to train_pytorch_model (epochs, lr, etc.)

    Returns:
        List of N trained RecidivismMLP models in eval() mode
    """
    models = []
    for i in range(n_members):
        seed = base_seed + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        member = RecidivismMLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )

        print(f"    Member {i + 1}/{n_members} (seed={seed})...")
        result = train_pytorch_model(
            member, X_train, y_train, X_val, y_val,
            seed=seed,
            **train_kwargs,
        )
        models.append(result['model'])

    return models


def train_ensemble_lstm_members(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_members: int = 5,
    base_seed: int = 100,
    **train_kwargs,
) -> List[nn.Module]:
    """
    Train N independent LSTM models with different random seeds.

    Args:
        input_size: Number of input features per timestep
        hidden_size: LSTM hidden size (same for all members)
        num_layers: Number of LSTM layers (same for all members)
        dropout: Dropout rate (same for all members)
        X_train: (n_samples, seq_len, n_features)
        y_train: (n_samples, seq_len) — per-year labels
        X_val, y_val: Validation data in same format
        n_members: Number of ensemble members
        base_seed: Seeds will be base_seed, base_seed+1, ..., base_seed+N-1
        **train_kwargs: Forwarded to train_pytorch_model

    Returns:
        List of N trained RecidivismLSTM models in eval() mode
    """
    models = []
    for i in range(n_members):
        seed = base_seed + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        member = RecidivismLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        print(f"    Member {i + 1}/{n_members} (seed={seed})...")
        result = train_pytorch_model(
            member, X_train, y_train, X_val, y_val,
            seed=seed,
            lstm_mode=True,
            **train_kwargs,
        )
        models.append(result['model'])

    return models


def ensemble_predict_mlp(
    models: List[nn.Module],
    X: np.ndarray,
    device: str = 'cpu',
) -> Dict:
    """
    Aggregate deterministic predictions from an MLP ensemble.

    Args:
        models: List of N trained RecidivismMLP instances
        X: (n_samples, n_features)
        device: 'cpu' or 'cuda'

    Returns:
        {
            'risk_scores': np.ndarray (n_samples,) — mean predicted probability
            'uncertainty_scores': np.ndarray (n_samples,) — variance across members
            'all_predictions': np.ndarray (n_samples, N) — per-member probabilities
        }
    """
    X_tensor = torch.FloatTensor(X).to(device)
    preds = []

    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)

    all_predictions = np.stack(preds, axis=1)   # (n_samples, N)
    risk_scores = all_predictions.mean(axis=1)
    uncertainty_scores = all_predictions.var(axis=1)

    return {
        'risk_scores': risk_scores,
        'uncertainty_scores': uncertainty_scores,
        'all_predictions': all_predictions,
    }


def ensemble_predict_lstm(
    models: List[nn.Module],
    X: np.ndarray,
    device: str = 'cpu',
) -> Dict:
    """
    Aggregate LSTM ensemble predictions using the survival rule.

    Each member produces per-year hazard probabilities (p1, p2, p3).
    These are converted to a cumulative 3-year risk per member:
        P(recidivate within 3yr) = 1 - (1 - p1)(1 - p2)(1 - p3)
    Uncertainty is the variance of this 3-year risk across members.

    Args:
        models: List of N trained RecidivismLSTM instances
        X: (n_samples, seq_len, n_features)
        device: 'cpu' or 'cuda'

    Returns:
        {
            'risk_scores': np.ndarray (n_samples,) — mean 3-yr cumulative risk
            'uncertainty_scores': np.ndarray (n_samples,) — variance of 3-yr risk
            'temporal_risk': np.ndarray (n_samples, seq_len) — mean hazard per step
            'temporal_uncertainty': np.ndarray (n_samples, seq_len) — var per step
            'all_predictions': np.ndarray (n_samples, N) — per-member 3-yr risk
        }
    """
    X_tensor = torch.FloatTensor(X).to(device)
    year_preds = []   # each: (n_samples, seq_len)
    survival_preds = []  # each: (n_samples,)

    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(X_tensor)                         # (n_samples, seq_len)
            year_probs = torch.sigmoid(logits).cpu().numpy()
        year_preds.append(year_probs)

        # Survival rule per member
        survival = 1 - np.prod(1 - year_probs, axis=1)      # (n_samples,)
        survival_preds.append(survival)

    # (n_samples, seq_len, N)
    all_year_preds = np.stack(year_preds, axis=2)
    temporal_risk = all_year_preds.mean(axis=2)              # (n_samples, seq_len)
    temporal_uncertainty = all_year_preds.var(axis=2)        # (n_samples, seq_len)

    all_predictions = np.stack(survival_preds, axis=1)       # (n_samples, N)
    risk_scores = all_predictions.mean(axis=1)
    uncertainty_scores = all_predictions.var(axis=1)

    return {
        'risk_scores': risk_scores,
        'uncertainty_scores': uncertainty_scores,
        'temporal_risk': temporal_risk,
        'temporal_uncertainty': temporal_uncertainty,
        'all_predictions': all_predictions,
    }
