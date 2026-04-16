"""
PyTorch N-HiTS forecasting model backend for ML Forecast Lab.

Implements N-HiTS (Neural Hierarchical Interpolation for Time Series)
with multi-rate signal sampling via MaxPool downsampling per stack,
doubly-residual stacking, AdamW optimisation, CosineAnnealingLR
scheduling, and best-model checkpointing. Supports multi-horizon output.
"""

import logging
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import ForecastModel, _build_activation, _resolve_sigmoid_scale

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch is not installed. NHiTSModel will not be functional.",
        ImportWarning,
    )


class _NHiTSBlock(nn.Module):
    """Single N-HiTS block: AvgPool1d downsampling -> FC stack -> backcast + forecast."""

    def __init__(self, seq_len: int, n_channels: int, hidden_size: int,
                 n_horizons: int, pool_kernel: int, n_fc_layers: int = 4):
        super().__init__()
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.pool_kernel = pool_kernel

        # Downsampled sequence length after pooling
        self.pooled_len = max(1, seq_len // pool_kernel)
        flat_input_size = self.pooled_len * n_channels

        # Pooling layer
        self.pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel) if pool_kernel > 1 else nn.Identity()

        # FC stack
        layers = []
        for i in range(n_fc_layers):
            in_dim = flat_input_size if i == 0 else hidden_size
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
        self.fc_stack = nn.Sequential(*layers)

        # Backcast projects to pooled_len * n_channels, then interpolated back
        self.backcast_proj = nn.Linear(hidden_size, self.pooled_len * n_channels)
        self.forecast_proj = nn.Linear(hidden_size, n_horizons)

    def forward(self, x):
        # x: (batch, seq_len, n_channels)
        batch_size = x.size(0)

        # Pool: need (batch, n_channels, seq_len) for AvgPool1d
        if self.pool_kernel > 1:
            x_perm = x.permute(0, 2, 1)                    # (batch, n_channels, seq_len)
            x_pooled = self.pool(x_perm)                    # (batch, n_channels, pooled_len)
            x_flat = x_pooled.reshape(batch_size, -1)       # (batch, pooled_len * n_channels)
        else:
            x_flat = x.reshape(batch_size, -1)              # (batch, seq_len * n_channels)

        h = self.fc_stack(x_flat)                           # (batch, hidden_size)
        forecast = self.forecast_proj(h)                    # (batch, n_horizons)

        # Backcast in pooled space, then interpolate back to original seq_len
        backcast_pooled = self.backcast_proj(h)              # (batch, pooled_len * n_channels)
        backcast_pooled = backcast_pooled.reshape(batch_size, self.n_channels, self.pooled_len)

        if self.pooled_len != self.seq_len:
            backcast = F.interpolate(
                backcast_pooled, size=self.seq_len, mode='linear', align_corners=False
            )  # (batch, n_channels, seq_len)
        else:
            backcast = backcast_pooled

        backcast = backcast.permute(0, 2, 1)                # (batch, seq_len, n_channels)
        return backcast, forecast


class _NHiTSNet(nn.Module):
    """N-HiTS with hierarchical pooling and doubly-residual stacking."""

    def __init__(self, seq_len: int, n_channels: int, hidden_size: int,
                 n_stacks: int, blocks_per_stack: int, pool_kernels: List[int],
                 n_fc_layers: int, n_horizons: int = 1,
                 output_activation: str = 'linear', sigmoid_scale: float = 1.0):
        super().__init__()
        self.n_horizons = n_horizons

        # Ensure pool_kernels list matches n_stacks (repeat last if too short).
        # Copy the input list first — callers may reuse the same list across
        # multiple model constructions and we must not mutate their reference.
        pool_kernels = list(pool_kernels)
        while len(pool_kernels) < n_stacks:
            pool_kernels.append(pool_kernels[-1])

        self.blocks = nn.ModuleList()
        for stack_idx in range(n_stacks):
            pk = min(pool_kernels[stack_idx], seq_len)  # clamp to seq_len
            for _ in range(blocks_per_stack):
                self.blocks.append(
                    _NHiTSBlock(seq_len, n_channels, hidden_size, n_horizons, pk, n_fc_layers)
                )
        self.activation = _build_activation(output_activation, scale=sigmoid_scale)

    def forward(self, x):
        # x: (batch, seq_len, n_channels)
        batch_size = x.size(0)
        residual = x
        forecast_sum = torch.zeros(batch_size, self.n_horizons, device=x.device)

        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_sum = forecast_sum + forecast

        out = self.activation(forecast_sum)
        if self.n_horizons == 1:
            return out.squeeze(-1)  # (batch,) backward compat
        return out


class NHiTSModel(ForecastModel):
    """
    PyTorch N-HiTS time-series forecasting model.

    Uses N-HiTS with hierarchical interpolation, multi-rate signal
    sampling via pooling, doubly-residual stacking, AdamW optimiser
    with CosineAnnealingLR, and best-model checkpointing on validation
    loss. Supports multi-horizon output.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        n_stacks: int = 3,
        blocks_per_stack: int = 1,
        pool_kernels: Optional[List[int]] = None,
        n_fc_layers: int = 4,
        learning_rate: float = 2e-4,
        epochs: int = 100,
        batch_size: int = 64,
        sequence_length: Optional[int] = None,
        loss_fn: str = 'mse',
        daily_loss_weight: float = 0.0,
        optimiser: str = 'adamw',
        patience: int = 20,
        output_activation: str = 'linear',
    ) -> None:
        super().__init__()
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed")

        self.hidden_size = hidden_size
        self.n_stacks = n_stacks
        self.blocks_per_stack = blocks_per_stack
        self.pool_kernels = pool_kernels if pool_kernels is not None else [8, 4, 1]
        self.n_fc_layers = n_fc_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.loss_fn = loss_fn
        self.daily_loss_weight = float(daily_loss_weight)
        self.optimiser = optimiser
        self.patience = patience
        self.output_activation = output_activation

        self._model: Optional[_NHiTSNet] = None
        self._input_size: Optional[int] = None
        self._seq_len: Optional[int] = None
        self._n_horizons: int = 1
        self._channel_mean: Optional[np.ndarray] = None
        self._channel_std: Optional[np.ndarray] = None
        self._sigmoid_scale: float = 1.0
        # Target z-score stats (only populated when output_activation == 'zscore').
        # Identity defaults so non-zscore paths are a safe no-op.
        self._y_mean: Any = 0.0
        self._y_std: Any = 1.0
        self._training_history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    @property
    def name(self) -> str:
        return "nhits"

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
        """Train N-HiTS with PyTorch autograd and best-model checkpointing."""
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
        self._channel_mean = X_seq.mean(axis=(0, 1))  # shape (n_channels,)
        self._channel_std = X_seq.std(axis=(0, 1))     # shape (n_channels,)
        self._channel_std[self._channel_std < 1e-8] = 1.0  # Avoid division by zero
        X_seq = (X_seq - self._channel_mean) / self._channel_std

        # Sigmoid activation needs a ceiling: use training-data maximum with
        # a 10% buffer so the network can reach observed extrema.
        if self.output_activation == 'sigmoid':
            self._sigmoid_scale = _resolve_sigmoid_scale(y_train)

        # Target z-score normalisation (output_activation == 'zscore'):
        # the head is linear, gradients stay O(1) regardless of target
        # magnitude, and predictions are denormalised back to physical
        # units at inference time. Per-horizon stats for multi-horizon so
        # each horizon column retains its own scale.
        if self.output_activation == 'zscore':
            if self._n_horizons > 1:
                y_mean = y_train.mean(axis=0)
                y_std = y_train.std(axis=0)
                y_std = np.where(y_std < 1e-8, 1.0, y_std)
                self._y_mean = y_mean.astype(np.float32)
                self._y_std = y_std.astype(np.float32)
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
        self._model = _NHiTSNet(
            seq_len, input_size, self.hidden_size,
            self.n_stacks, self.blocks_per_stack, self.pool_kernels,
            self.n_fc_layers, n_horizons=self._n_horizons,
            output_activation=self.output_activation,
            sigmoid_scale=self._sigmoid_scale,
        )
        optimiser = self._build_optimiser(
            self._model.parameters(), self.optimiser, self.learning_rate,
        )
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
                w_batch = w_tr_t[batch_idx] if w_tr_t is not None else None
                loss, loss_per_sample = self._composite_horizon_loss(
                    y_pred, y_batch, criterion, w_batch, self.daily_loss_weight,
                )
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
                val_loss_t, _ = self._composite_horizon_loss(
                    val_pred, y_val_t, criterion, None, self.daily_loss_weight,
                )
                val_loss = val_loss_t.item()

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

        X_seq = X.copy()
        if self._channel_mean is not None and self._channel_std is not None:
            X_seq = (X_seq - self._channel_mean) / self._channel_std

        X_t = torch.FloatTensor(X_seq)
        self._model.eval()
        with torch.no_grad():
            predictions = self._model(X_t).numpy()

        # Denormalise z-space predictions back to physical units. Floor at
        # zero because the linear head in z-space is unconstrained and
        # callers expect physically-valid (non-negative) forecasts.
        if self.output_activation == 'zscore':
            predictions = predictions * self._y_std + self._y_mean
            predictions = np.clip(predictions, 0.0, None)

        return predictions.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions (returns horizon-0 for multi-horizon models)."""
        self._validate_fitted()
        self._validate_X(X)

        X_seq = self._reshape_to_sequences(X)

        if self._channel_mean is not None and self._channel_std is not None:
            X_seq = (X_seq - self._channel_mean) / self._channel_std

        X_t = torch.FloatTensor(X_seq)

        self._model.eval()
        with torch.no_grad():
            predictions = self._model(X_t).numpy()

        # Denormalise z-space predictions *before* slicing to horizon-0 so the
        # per-horizon stats align with each column of the prediction array.
        if self.output_activation == 'zscore':
            predictions = predictions * self._y_std + self._y_mean
            predictions = np.clip(predictions, 0.0, None)

        # Multi-horizon: return only first horizon for backward compat
        if predictions.ndim == 2:
            predictions = predictions[:, 0]

        return predictions.astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return deepcopy({
            "hidden_size": self.hidden_size, "n_stacks": self.n_stacks,
            "blocks_per_stack": self.blocks_per_stack, "pool_kernels": self.pool_kernels,
            "n_fc_layers": self.n_fc_layers, "learning_rate": self.learning_rate,
            "epochs": self.epochs, "batch_size": self.batch_size,
            "sequence_length": self.sequence_length, "loss_fn": self.loss_fn,
            "daily_loss_weight": self.daily_loss_weight,
            "optimiser": self.optimiser,
            "patience": self.patience,
            "output_activation": self.output_activation,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"hidden_size", "n_stacks", "blocks_per_stack", "pool_kernels",
                 "n_fc_layers", "learning_rate", "epochs", "batch_size",
                 "sequence_length", "loss_fn", "daily_loss_weight", "optimiser", "patience",
                 "output_activation"}
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
            "sigmoid_scale": self._sigmoid_scale,
            "y_mean": self._y_mean,
            "y_std": self._y_std,
        }, path)
        logger.info(f"Saved N-HiTS model to {path}")

    def load(self, path: str) -> None:
        """Load model state dict."""
        data = torch.load(path, map_location="cpu")
        self.set_params(**data["params"])
        self._channel_mean = data.get("channel_mean")
        self._channel_std = data.get("channel_std")
        self._n_horizons = data.get("n_horizons", 1)
        self._sigmoid_scale = float(data.get("sigmoid_scale", 1.0))
        self._y_mean = data.get("y_mean", 0.0)
        self._y_std = data.get("y_std", 1.0)

        # Reconstruct the nn.Module and load weights
        self._input_size = data.get("input_size")
        self._seq_len = data.get("seq_len")
        state_dict = data.get("state_dict")
        if state_dict is not None and self._input_size is not None and self._seq_len is not None:
            self._model = _NHiTSNet(
                self._seq_len, self._input_size, self.hidden_size,
                self.n_stacks, self.blocks_per_stack, self.pool_kernels,
                self.n_fc_layers, n_horizons=self._n_horizons,
                output_activation=self.output_activation,
                sigmoid_scale=self._sigmoid_scale,
            )
            self._model.load_state_dict(state_dict)
            self._model.eval()

        self._is_fitted = True
        logger.info(f"Loaded N-HiTS model from {path}")
