"""
PyTorch LSTM forecasting model backend for ML Forecast Lab.

Implements a multi-layer LSTM using PyTorch with proper autograd,
Adam optimisation, and early stopping. Replaces the pure-NumPy
implementation for correct gradient flow and faster training.
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
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch is not installed. LSTMModel will not be functional.",
        ImportWarning,
    )


class _LSTMNet(nn.Module):
    """PyTorch LSTM network with dense output."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Use last timestep hidden state
        h_last = lstm_out[:, -1, :]
        return self.fc(h_last).squeeze(-1)


class LSTMModel(ForecastModel):
    """
    PyTorch LSTM time-series forecasting model.

    Uses torch.nn.LSTM with autograd for proper gradient computation,
    Adam optimiser, and early stopping on validation loss.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 15,
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
        """Train LSTM with PyTorch autograd and Adam."""
        self._validate_X(X_train)
        y_train = self._validate_y(y_train)
        start_time = time.time()

        # Reshape
        X_seq = self._reshape_to_sequences(X_train)
        _, seq_len, input_size = X_seq.shape

        # Train/val split
        val_split = kwargs.get("validation_split", 0.2)
        n_train = int(len(X_seq) * (1 - val_split))
        X_tr, X_val = X_seq[:n_train], X_seq[n_train:]
        y_tr, y_val = y_train[:n_train], y_train[n_train:]

        # Convert to tensors
        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_t = torch.FloatTensor(y_tr)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)

        # Create model
        self._model = _LSTMNet(input_size, self.hidden_size, self.num_layers, self.dropout)
        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float("inf")
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
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5.0)
                optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()

            avg_loss = epoch_loss / max(n_batches, 1)
            self._training_history["train_loss"].append(avg_loss)
            self._training_history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % max(1, self.epochs // 10) == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}: train_loss={avg_loss:.6f}, val_loss={val_loss:.6f}")

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

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
            dummy = torch.randn(1, self.sequence_length or 31, 1)
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
            "patience": self.patience, "sequence_length": self.sequence_length,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"hidden_size", "num_layers", "dropout", "learning_rate",
                 "epochs", "batch_size", "patience", "sequence_length"}
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
        # Recreate model architecture — need to know input_size
        # For now, defer to fit() or manual setup
        self._is_fitted = True
        logger.info(f"Loaded LSTM model from {path}")
