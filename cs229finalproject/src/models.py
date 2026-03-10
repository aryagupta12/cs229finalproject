"""
Model architectures for recidivism prediction.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from typing import Tuple

# ============================================
# Model 1: Logistic Regression (sklearn)
# ============================================

def create_logistic_regression(**kwargs) -> LogisticRegression:
    """Create sklearn logistic regression model."""
    return LogisticRegression(max_iter=1000, **kwargs)


# ============================================
# Model 2: MLP with Dropout (PyTorch)
# ============================================

class RecidivismMLP(nn.Module):
    """
    Multi-layer perceptron for recidivism prediction.

    Architecture:
    - Input → Dense(128) → BatchNorm → ReLU → Dropout(p)
    - → Dense(64) → BatchNorm → ReLU → Dropout(p)
    - → Dense(32) → BatchNorm → ReLU → Dropout(p)
    - → Dense(1)
    """

    def __init__(self, input_size: int, hidden_sizes: list = None, dropout: float = 0.3):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]

        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_size = h
        layers.append(nn.Linear(in_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# ============================================
# Model 3: LSTM with Dropout (PyTorch)
# ============================================

class RecidivismLSTM(nn.Module):
    """
    LSTM for sequential recidivism prediction.

    Architecture:
    - LSTM(input_size, hidden_size, num_layers, dropout)
    - Dropout(p)
    - Dense(hidden_size → 1) applied at each time step
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            (batch_size, seq_len) — logit at each time step
        """
        out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        out = self.dropout(out)
        logits = self.fc(out)           # (batch, seq_len, 1)
        return logits.squeeze(-1)       # (batch, seq_len)


def create_sequences_for_lstm(
    X: np.ndarray,
    y_year1: np.ndarray,
    y_year2: np.ndarray,
    y_year3: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert flat features to sequences for LSTM.

    Since features don't change over time, we repeat them and add time indicator.

    Args:
        X: (n_samples, n_features)
        y_year1, y_year2, y_year3: Labels for each year

    Returns:
        X_seq: (n_samples, 3, n_features + 1)  — +1 for time indicator
        y_seq: (n_samples, 3)  — label at each year
    """
    n_samples, n_features = X.shape
    time_indicators = np.array([1.0, 2.0, 3.0])  # year index (non-zero at all steps)

    X_seq = np.zeros((n_samples, 3, n_features + 1), dtype=np.float32)
    for t, ti in enumerate(time_indicators):
        X_seq[:, t, :n_features] = X
        X_seq[:, t, n_features] = ti

    y_seq = np.stack([y_year1, y_year2, y_year3], axis=1).astype(np.float32)
    return X_seq, y_seq
