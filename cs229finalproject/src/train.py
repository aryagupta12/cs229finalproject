"""
Training functions for PyTorch models.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm


def train_pytorch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 10,
    device: str = 'cpu',
    class_weight: Optional[float] = None,
    lstm_mode: bool = False,
    seed: int = 42,
    weight_decay: float = 0.0,
) -> Dict:
    """
    Train a PyTorch model with early stopping.

    Args:
        model: PyTorch model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Maximum epochs
        batch_size: Batch size
        learning_rate: Learning rate for Adam
        patience: Early stopping patience
        device: 'cpu' or 'cuda'
        class_weight: Positive class weight for imbalanced data (optional)
        lstm_mode: If True, y_train/y_val are (n, seq_len); use last timestep loss or all.

    Returns:
        {
            'model': trained model,
            'train_losses': List[float],
            'val_losses': List[float],
            'best_epoch': int
        }
    """
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if class_weight is not None:
        pos_weight = torch.tensor([class_weight], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.FloatTensor(y_val).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    best_epoch = 0

    pbar = tqdm(range(epochs), desc='Training', unit='epoch')
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)

            if lstm_mode:
                # Use all timesteps
                loss = criterion(logits, y_batch)
            else:
                loss = criterion(logits, y_batch)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)

        avg_train_loss = epoch_loss / len(X_t)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_loss = criterion(val_logits, y_v).item()
        val_losses.append(val_loss)

        pbar.set_postfix({'train_loss': f'{avg_train_loss:.4f}', 'val_loss': f'{val_loss:.4f}'})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\n  Early stopping at epoch {epoch + 1} (best epoch: {best_epoch + 1})')
                break

    model.load_state_dict(best_state)
    model.eval()

    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
    }


def compute_class_weight(y: np.ndarray) -> float:
    """
    Compute positive class weight for imbalanced data.

    weight = n_negative / n_positive
    """
    n_positive = float(y.sum())
    n_negative = len(y) - n_positive
    if n_positive == 0:
        return 1.0
    return n_negative / n_positive
