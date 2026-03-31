"""
PyTorch LSTM forecasting model backend for ML Forecast Lab.

Implements a multi-layer LSTM with temporal attention using PyTorch,
Adam optimisation, ReduceLROnPlateau scheduling, best-model
checkpointing, and early stopping.
"""

import logging
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import ForecastModel

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch is not installed. LSTMModel will not be functional.",
        ImportWarning,
    )


class _TemporalAttention(nn.Module):
    """Learnable attention over LSTM timesteps."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_vector = nn.Parameter(torch.randn(hidden_size))

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_size)
        scores = torch.tanh(self.attn_proj(lstm_out))  # (batch, seq_len, hidden)
        scores = (scores * self.attn_vector).sum(dim=-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden)
        return context


class _LSTMNet(nn.Module):
    """PyTorch LSTM with LayerNorm, temporal attention, and MLP head."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = _TemporalAttention(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.layer_norm(x)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        context = self.attention(lstm_out)  # (batch, hidden_size)
        return self.head(context).squeeze(-1)


class LSTMModel(ForecastModel):
    """
    PyTorch LSTM time-series forecasting model with temporal attention.

    Uses torch.nn.LSTM with autograd, temporal attention over all timesteps,
    Adam optimiser with ReduceLROnPlateau, best-model checkpointing,
    and early stopping on validation loss.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 15,
        lr_patience: int = 7,
        sequence_length: Optional[int] = None,
    ) -> None:
        """Initialise LSTM model."""
        super().__init__()
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lr_patience = lr_patience
        self.sequence_length = sequence_length

        self._model: Optional[_LSTMNet] = None
        self._training_history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    @property
    def name(self) -> str:
        return "lstm"

    def _reshape_to_sequences(self, X: np.ndarray) -> np.ndarray:
        """Reshape (n_samples, n_features) → (n_samples, seq_len, features_per_step)."""
        n_samples, n_features = X.shape
        seq_len = self.sequence_length or n_features
        if n_features % seq_len != 0 and seq_len == n_features:
            n_features_per_step = 1
        else:
            n_features_per_step = n_features // seq_len
        return X[:, :seq_len * n_features_per_step].reshape(n_samples, seq_len, n_features_per_step)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        """Train LSTM with PyTorch autograd, attention, and best-model checkpointing."""
        self._validate_X(X_train)
        y_train = self._validate_y(y_train)
        start_time = time.time()

        # Use pre-windowed sequence data if provided, otherwise reshape flat features
        sequence_data = kwargs.get("sequence_data")
        if sequence_data is not None:
            X_seq = sequence_data  # Already (n_samples, window_size, n_channels)
            logger.debug(f"Using pre-windowed sequence data: {X_seq.shape}")
        else:
            X_seq = self._reshape_to_sequences(X_train)
        _, seq_len, input_size = X_seq.shape

        # Extract sample weights
        sample_weight = kwargs.get("sample_weight")

        # Middle-out validation split — val from centre so model trains on recent data
        n_total = len(X_seq)
        val_split = kwargs.get("validation_split", 0.2)
        n_val = int(n_total * val_split)
        val_start = (n_total - n_val) // 2

        val_mask = np.zeros(n_total, dtype=bool)
        val_mask[val_start:val_start + n_val] = True
        train_mask = ~val_mask

        X_tr, X_val = X_seq[train_mask], X_seq[val_mask]
        y_tr, y_val = y_train[train_mask], y_train[val_mask]
        w_tr = sample_weight[train_mask] if sample_weight is not None else None

        # Convert to tensors
        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_t = torch.FloatTensor(y_tr)
        w_tr_t = torch.FloatTensor(w_tr) if w_tr is not None else None
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)

        # Create model
        self._model = _LSTMNet(input_size, self.hidden_size, self.num_layers, self.dropout)
        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode='min', factor=0.5, patience=self.lr_patience,
        )
        criterion = nn.HuberLoss(reduction='none')

        # Training loop with best-model checkpointing
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        self._training_history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            self._model.train()
            indices = torch.randperm(len(X_tr_t))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_tr_t), self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                X_batch = X_tr_t[batch_idx]
                y_batch = y_tr_t[batch_idx]

                optimiser.zero_grad()
                y_pred = self._model(X_batch)
                loss_per_sample = criterion(y_pred, y_batch)
                if w_tr_t is not None:
                    w_batch = w_tr_t[batch_idx]
                    loss = (loss_per_sample * w_batch).mean()
                else:
                    loss = loss_per_sample.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5.0)
                optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).mean().item()

            avg_loss = epoch_loss / max(n_batches, 1)
            self._training_history["train_loss"].append(avg_loss)
            self._training_history["val_loss"].append(val_loss)

            # LR scheduler step
            scheduler.step(val_loss)

            # Best-model checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(self._model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % max(1, self.epochs // 10) == 0:
                current_lr = optimiser.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs}: "
                    f"train_loss={avg_loss:.6f}, val_loss={val_loss:.6f}, lr={current_lr:.2e}"
                )

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_state is not None:
            self._model.load_state_dict(best_state)

        elapsed = time.time() - start_time
        self._is_fitted = True

        return {
            "time_seconds": elapsed,
            "epochs": epoch + 1,
            "best_val_loss": float(best_val_loss),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        self._validate_fitted()
        self._validate_X(X)

        X_seq = self._reshape_to_sequences(X)
        X_t = torch.FloatTensor(X_seq)

        self._model.eval()
        with torch.no_grad():
            predictions = self._model(X_t).numpy()

        # Clip to non-negative
        return np.clip(predictions, 0.0, None).astype(np.float32)

    def export_onnx(self, path: str) -> bool:
        """Export model to ONNX format."""
        if not self.is_fitted or self._model is None:
            return False
        try:
            dummy = torch.randn(1, self.sequence_length or 48, 1)
            torch.onnx.export(self._model, dummy, path, input_names=["input"], output_names=["output"])
            logger.info(f"Exported LSTM to ONNX: {path}")
            return True
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")
            return False

    def supports_hardware_accel(self) -> bool:
        return True

    def get_params(self) -> Dict[str, Any]:
        return deepcopy({
            "hidden_size": self.hidden_size, "num_layers": self.num_layers,
            "dropout": self.dropout, "learning_rate": self.learning_rate,
            "epochs": self.epochs, "batch_size": self.batch_size,
            "patience": self.patience, "lr_patience": self.lr_patience,
            "sequence_length": self.sequence_length,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"hidden_size", "num_layers", "dropout", "learning_rate",
                 "epochs", "batch_size", "patience", "lr_patience", "sequence_length"}
        for k, v in kwargs.items():
            if k not in valid:
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)

    def save(self, path: str) -> None:
        """Save model state dict."""
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Cannot save unfitted model")
        torch.save({"state_dict": self._model.state_dict(), "params": self.get_params()}, path)
        logger.info(f"Saved LSTM model to {path}")

    def load(self, path: str) -> None:
        """Load model state dict."""
        data = torch.load(path, map_location="cpu")
        self.set_params(**data["params"])
        self._is_fitted = True
        logger.info(f"Loaded LSTM model from {path}")
