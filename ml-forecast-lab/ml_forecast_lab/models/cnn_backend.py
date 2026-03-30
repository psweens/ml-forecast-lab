"""
PyTorch 1D CNN forecasting model backend for ML Forecast Lab.

Implements a stack of 1D causal dilated convolutions (WaveNet-style)
using PyTorch with proper autograd, residual connections, and Adam
optimisation. Replaces the pure-NumPy implementation.
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
        "PyTorch is not installed. CNNModel will not be functional.",
        ImportWarning,
    )


class _CausalConv1d(nn.Module):
    """Causal 1D convolution with dilation — output depends only on past inputs."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x_padded = F.pad(x, (self.padding, 0))  # Left-pad for causality
        return self.conv(x_padded)


class _WaveNetBlock(nn.Module):
    """Single WaveNet-style block: causal dilated conv → ReLU → residual."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = _CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # 1x1 conv for residual if channel mismatch
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out + self.residual(x)


class _CNNNet(nn.Module):
    """PyTorch WaveNet-style CNN with global average pooling and dense output."""

    def __init__(self, input_size: int, n_filters: int, kernel_size: int,
                 n_layers: int, dilation_base: int, dropout: float):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch = input_size if i == 0 else n_filters
            dilation = dilation_base ** i
            layers.append(_WaveNetBlock(in_ch, n_filters, kernel_size, dilation, dropout))
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(n_filters, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size) → need (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        out = self.blocks(x)
        # Global average pooling over sequence dimension
        pooled = out.mean(dim=2)  # (batch, n_filters)
        return self.fc(pooled).squeeze(-1)


class CNNModel(ForecastModel):
    """
    PyTorch 1D Dilated Causal CNN for time-series forecasting.

    WaveNet-style architecture with residual connections, ReLU activation,
    dropout, and global average pooling. Uses PyTorch autograd for proper
    gradient computation.
    """

    def __init__(
        self,
        n_filters: int = 16,
        kernel_size: int = 3,
        n_layers: int = 3,
        dilation_base: int = 2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 15,
        dropout: float = 0.1,
    ) -> None:
        """Initialise CNN model."""
        super().__init__()
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed")

        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.dilation_base = dilation_base
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.dropout = dropout

        self._model: Optional[_CNNNet] = None
        self._input_size: Optional[int] = None
        self._sequence_length: Optional[int] = None
        self._training_history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    @property
    def name(self) -> str:
        return "cnn"

    def _reshape_to_sequences(self, X: np.ndarray) -> np.ndarray:
        """Reshape (n_samples, n_features) → (n_samples, seq_len, features_per_step)."""
        n_samples, n_features = X.shape
        seq_len = self._sequence_length or n_features
        if n_features % seq_len != 0 and seq_len == n_features:
            n_features_per_step = 1
        else:
            n_features_per_step = n_features // seq_len
        return X[:, :seq_len * n_features_per_step].reshape(n_samples, seq_len, n_features_per_step)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        """Train CNN with PyTorch autograd and Adam."""
        self._validate_X(X_train)
        y_train = self._validate_y(y_train)
        start_time = time.time()

        # Reshape
        X_seq = self._reshape_to_sequences(X_train)
        _, seq_len, input_size = X_seq.shape
        self._input_size = input_size
        self._sequence_length = seq_len

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
        self._model = _CNNNet(
            input_size, self.n_filters, self.kernel_size,
            self.n_layers, self.dilation_base, self.dropout,
        )
        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.HuberLoss(delta=1.0)

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

        return np.clip(predictions, 0.0, None).astype(np.float32)

    def export_onnx(self, path: str) -> bool:
        """Export model to ONNX format."""
        if not self.is_fitted or self._model is None:
            return False
        try:
            seq_len = self._sequence_length or 31
            input_size = self._input_size or 1
            dummy = torch.randn(1, seq_len, input_size)
            torch.onnx.export(self._model, dummy, path, input_names=["input"], output_names=["output"])
            logger.info(f"Exported CNN to ONNX: {path}")
            return True
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")
            return False

    def supports_hardware_accel(self) -> bool:
        return True

    def get_params(self) -> Dict[str, Any]:
        return deepcopy({
            "n_filters": self.n_filters, "kernel_size": self.kernel_size,
            "n_layers": self.n_layers, "dilation_base": self.dilation_base,
            "learning_rate": self.learning_rate, "epochs": self.epochs,
            "batch_size": self.batch_size, "patience": self.patience,
            "dropout": self.dropout,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"n_filters", "kernel_size", "n_layers", "dilation_base",
                 "learning_rate", "epochs", "batch_size", "patience", "dropout"}
        for k, v in kwargs.items():
            if k not in valid:
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)

    def save(self, path: str) -> None:
        """Save model state dict."""
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Cannot save unfitted model")
        torch.save({
            "state_dict": self._model.state_dict(),
            "params": self.get_params(),
            "input_size": self._input_size,
            "sequence_length": self._sequence_length,
        }, path)
        logger.info(f"Saved CNN model to {path}")

    def load(self, path: str) -> None:
        """Load model state dict."""
        data = torch.load(path, map_location="cpu")
        self.set_params(**data["params"])
        self._input_size = data.get("input_size")
        self._sequence_length = data.get("sequence_length")
        self._is_fitted = True
        logger.info(f"Loaded CNN model from {path}")
