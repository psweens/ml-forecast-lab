"""
PyTorch SparseTSF forecasting model backend for ML Forecast Lab.

Implements a period-based sparse cross-period linear architecture,
AdamW optimisation, CosineAnnealingLR scheduling, and best-model
checkpointing. Supports multi-horizon output via a shared encoder
and multi-output head.
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
        "PyTorch is not installed. SparseTSFModel will not be functional.",
        ImportWarning,
    )


class _SparseTSFNet(nn.Module):
    """SparseTSF: period-based sparse cross-period linear forecaster."""

    def __init__(self, seq_len: int, n_channels: int, period_len: int,
                 dropout: float, n_horizons: int = 1):
        super().__init__()
        self.n_horizons = n_horizons
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.period_len = period_len

        # Clamp period_len so at least one complete period fits
        if period_len > seq_len:
            period_len = max(1, seq_len)
            self.period_len = period_len

        # Number of complete periods that fit in the sequence
        self.sub_len = seq_len // period_len

        # Cross-period linear: maps sub_len -> 1 for each period position
        self.cross_period = nn.Linear(self.sub_len, 1)
        self.dropout = nn.Dropout(dropout)

        # Output head: flatten pooled representation and project to horizons
        self.head = nn.Linear(n_channels * period_len, n_horizons)

    def forward(self, x):
        # x: (batch, seq_len, n_channels)
        batch_size = x.size(0)

        # Use last (sub_len * period_len) timesteps to ensure complete periods
        effective_len = self.sub_len * self.period_len
        x = x[:, x.size(1) - effective_len:, :]  # (batch, sub_len * period_len, n_channels)

        # Transpose to channel-first: (batch, n_channels, sub_len * period_len)
        x = x.transpose(1, 2)

        # Reshape into periods: (batch, n_channels, sub_len, period_len)
        x = x.reshape(batch_size, self.n_channels, self.sub_len, self.period_len)

        # Transpose for cross-period linear: (batch, n_channels, period_len, sub_len)
        x = x.permute(0, 1, 3, 2)

        # Apply cross-period linear across the sub_len dimension
        # (batch, n_channels, period_len, sub_len) -> (batch, n_channels, period_len, 1)
        x = self.cross_period(x)
        x = x.squeeze(-1)  # (batch, n_channels, period_len)

        x = self.dropout(x)

        # Flatten and project to output
        x = x.reshape(batch_size, -1)  # (batch, n_channels * period_len)
        out = self.head(x)  # (batch, n_horizons)

        if self.n_horizons == 1:
            return out.squeeze(-1)  # (batch,) backward compat
        return out


class SparseTSFModel(ForecastModel):
    """
    PyTorch SparseTSF time-series forecasting model.

    Uses period-based sparse cross-period linear mapping,
    AdamW optimiser with CosineAnnealingLR, best-model checkpointing.
    Supports multi-horizon output.
    """

    def __init__(
        self,
        period_len: int = 48,
        dropout: float = 0.05,
        learning_rate: float = 5e-4,
        epochs: int = 100,
        batch_size: int = 64,
        sequence_length: Optional[int] = None,
        loss_fn: str = 'mse',
        patience: int = 20,
    ) -> None:
        super().__init__()
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed")

        self.period_len = period_len
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.loss_fn = loss_fn
        self.patience = patience

        self._model: Optional[_SparseTSFNet] = None
        self._input_size: Optional[int] = None
        self._seq_len: Optional[int] = None
        self._n_horizons: int = 1
        self._channel_mean: Optional[np.ndarray] = None
        self._channel_std: Optional[np.ndarray] = None
        self._y_mean = 0.0   # float or ndarray(n_horizons,)
        self._y_std = 1.0    # float or ndarray(n_horizons,)
        self._training_history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    @property
    def name(self) -> str:
        return "sparsetsf"

    @property
    def is_neural(self) -> bool:
        return True

    def _reshape_to_sequences(self, X: np.ndarray) -> np.ndarray:
        """Reshape (n_samples, n_features) -> (n_samples, seq_len, features_per_step)."""
        n_samples, n_features = X.shape
        seq_len = self.sequence_length or n_features
        if n_features % seq_len != 0 and seq_len == n_features:
            n_features_per_step = 1
        else:
            n_features_per_step = n_features // seq_len
        return X[:, :seq_len * n_features_per_step].reshape(n_samples, seq_len, n_features_per_step)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        """Train SparseTSF with PyTorch autograd and best-model checkpointing."""
        self._validate_X(X_train)
        y_train = self._validate_y(y_train)
        start_time = time.time()

        # Detect multi-horizon from y shape
        if y_train.ndim == 2 and y_train.shape[1] > 1:
            self._n_horizons = y_train.shape[1]
        else:
            self._n_horizons = 1

        # Use pre-windowed sequence data if provided, otherwise reshape flat features
        sequence_data = kwargs.get("sequence_data")
        if sequence_data is not None:
            X_seq = sequence_data  # Already (n_samples, window_size, n_channels)
            logger.debug(f"Using pre-windowed sequence data: {X_seq.shape}")
        else:
            X_seq = self._reshape_to_sequences(X_train)
        _, seq_len, input_size = X_seq.shape
        self._input_size = input_size
        self._seq_len = seq_len

        # Per-channel z-score standardisation (fitted on training data)
        # Capture last value of target channel BEFORE normalization
        # for residual prediction (model learns deltas, not absolute values).
        last_values = X_seq[:, -1, 0].astype(np.float32)

        self._channel_mean = X_seq.mean(axis=(0, 1))  # shape (n_channels,)
        self._channel_std = X_seq.std(axis=(0, 1))     # shape (n_channels,)
        self._channel_std[self._channel_std < 1e-8] = 1.0
        X_seq = (X_seq - self._channel_mean) / self._channel_std

        # Residual targets: only for single-horizon (1-step-ahead) models.
        # Multi-horizon models use absolute targets so the network must learn
        # horizon-specific temporal patterns rather than converging to zero-
        # residual predictions which produce flat forecasts at the last value.
        if self._n_horizons == 1:
            y_train = y_train - last_values
            self._residual_prediction = True
        else:
            self._residual_prediction = False

        # Target z-score normalisation -- per-horizon when multi-output
        if self._n_horizons > 1:
            self._y_mean = y_train.mean(axis=0)   # (n_horizons,)
            self._y_std = y_train.std(axis=0)     # (n_horizons,)
            self._y_std[self._y_std < 1e-8] = 1.0
        else:
            self._y_mean = float(y_train.mean())
            self._y_std = float(y_train.std())
            if self._y_std < 1e-8:
                self._y_std = 1.0
        y_train = (y_train - self._y_mean) / self._y_std

        # Extract sample weights
        sample_weight = kwargs.get("sample_weight")

        # Tail validation split -- val is the most recent slice; a purge gap
        # equal to the forecast horizon keeps train target windows from
        # overlapping val inputs, preventing temporal leakage.
        n_total = len(X_seq)
        val_split = kwargs.get("validation_split", 0.2)
        train_mask, val_mask = self._tail_val_split(
            n_total, val_split, gap=self._n_horizons,
        )

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
        self._model = _SparseTSFNet(
            seq_len, input_size, self.period_len, self.dropout,
            n_horizons=self._n_horizons,
        )
        optimiser = torch.optim.AdamW(self._model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.epochs, eta_min=1e-6)
        _loss_map = {'mse': nn.MSELoss, 'mae': nn.L1Loss, 'l1': nn.L1Loss, 'huber': nn.SmoothL1Loss}
        criterion = _loss_map.get(self.loss_fn, nn.MSELoss)(reduction='none')

        # Training loop -- cosine annealing + best-model checkpoint + early stopping
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
                    loss = self._weighted_mean_loss(loss_per_sample, w_batch)
                else:
                    loss = loss_per_sample.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5.0)
                optimiser.step()

                # Report unweighted per-sample MSE so train and val are on
                # the same scale (weighted `loss` above is used for backprop).
                epoch_loss += loss_per_sample.mean().item()
                n_batches += 1

            scheduler.step()

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).mean().item()

            avg_loss = epoch_loss / max(n_batches, 1)
            self._training_history["train_loss"].append(avg_loss)
            self._training_history["val_loss"].append(val_loss)

            self._emit_epoch(kwargs.get("epoch_callback"),
                model_name=self.name, epoch=epoch + 1, total_epochs=self.epochs,
                train_loss=avg_loss, val_loss=val_loss, lr=optimiser.param_groups[0]['lr'],
                patience_counter=patience_counter, patience_limit=self.patience,
                best_val_loss=best_val_loss)

            # Best-model checkpoint + early stopping
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
                logger.info(f"Early stopping at epoch {epoch + 1} (no improvement for {self.patience} epochs)")
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

    def predict_sequence(self, X: np.ndarray) -> np.ndarray:
        """Multi-horizon prediction from sliding-window input.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, window_size, n_channels)

        Returns
        -------
        np.ndarray, shape (n_samples, n_horizons) or (n_samples,) if single-horizon
        """
        self._validate_fitted()
        if self._model is None:
            raise RuntimeError("No model loaded")

        # Capture last value of target channel for residual reconstruction
        last_values = X[:, -1, 0].astype(np.float32)

        X_seq = X.copy()
        if self._channel_mean is not None and self._channel_std is not None:
            X_seq = (X_seq - self._channel_mean) / self._channel_std

        X_t = torch.FloatTensor(X_seq)
        self._model.eval()
        with torch.no_grad():
            predictions = self._model(X_t).numpy()

        # Denormalize predicted residuals
        predictions = predictions * self._y_std + self._y_mean

        # Add last value back (residual prediction reconstruction)
        if getattr(self, '_residual_prediction', False):
            if predictions.ndim == 2:
                predictions = predictions + last_values[:, None]
            else:
                predictions = predictions + last_values

        return np.clip(predictions, 0.0, None).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions (returns horizon-0 for multi-horizon models)."""
        self._validate_fitted()
        self._validate_X(X)

        X_seq = self._reshape_to_sequences(X)

        # Capture last value of target channel for residual reconstruction
        last_values = X_seq[:, -1, 0].astype(np.float32).copy()

        if self._channel_mean is not None and self._channel_std is not None:
            X_seq = (X_seq - self._channel_mean) / self._channel_std

        X_t = torch.FloatTensor(X_seq)

        self._model.eval()
        with torch.no_grad():
            predictions = self._model(X_t).numpy()

        # Denormalize back to original target scale
        predictions = predictions * self._y_std + self._y_mean

        # Add last value back (residual prediction reconstruction)
        if getattr(self, '_residual_prediction', False):
            if predictions.ndim == 2:
                predictions = predictions + last_values[:, None]
            else:
                predictions = predictions + last_values

        # Multi-horizon: return only first horizon for backward compat
        if predictions.ndim == 2:
            predictions = predictions[:, 0]

        return np.clip(predictions, 0.0, None).astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return deepcopy({
            "period_len": self.period_len, "dropout": self.dropout,
            "learning_rate": self.learning_rate, "epochs": self.epochs,
            "batch_size": self.batch_size, "sequence_length": self.sequence_length,
            "loss_fn": self.loss_fn,
            "patience": self.patience,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"period_len", "dropout", "learning_rate",
                 "epochs", "batch_size", "sequence_length", "loss_fn",
                 "patience"}
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
            "seq_len": self._seq_len,
            "n_horizons": self._n_horizons,
            "channel_mean": self._channel_mean,
            "channel_std": self._channel_std,
            "y_mean": self._y_mean,
            "y_std": self._y_std,
            "residual_prediction": getattr(self, '_residual_prediction', False),
        }, path)
        logger.info(f"Saved SparseTSF model to {path}")

    def load(self, path: str) -> None:
        """Load model state dict."""
        data = torch.load(path, map_location="cpu")
        self.set_params(**data["params"])
        self._channel_mean = data.get("channel_mean")
        self._channel_std = data.get("channel_std")

        # Backward compat: old checkpoints store scalar y_mean/y_std, no n_horizons
        self._n_horizons = data.get("n_horizons", 1)
        raw_y_mean = data.get("y_mean", 0.0)
        raw_y_std = data.get("y_std", 1.0)
        if isinstance(raw_y_mean, np.ndarray):
            self._y_mean = raw_y_mean
            self._y_std = raw_y_std
        else:
            self._y_mean = float(raw_y_mean)
            self._y_std = float(raw_y_std)
        self._residual_prediction = data.get("residual_prediction", False)

        # Reconstruct the nn.Module and load weights
        self._input_size = data.get("input_size")
        self._seq_len = data.get("seq_len")
        state_dict = data.get("state_dict")
        if state_dict is not None and self._input_size is not None and self._seq_len is not None:
            self._model = _SparseTSFNet(
                self._seq_len, self._input_size, self.period_len,
                self.dropout, n_horizons=self._n_horizons,
            )
            self._model.load_state_dict(state_dict)
            self._model.eval()

        self._is_fitted = True
        logger.info(f"Loaded SparseTSF model from {path}")
