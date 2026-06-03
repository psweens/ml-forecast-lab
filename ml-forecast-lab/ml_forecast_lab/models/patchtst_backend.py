"""
PyTorch PatchTST forecasting model backend for ML Forecast Lab.

Implements a channel-independent patch transformer architecture,
AdamW optimisation, CosineAnnealingLR scheduling, and best-model
checkpointing. Supports multi-horizon output via a shared encoder
and multi-output head.
"""

import logging
import math
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import ForecastModel, _build_activation, _resolve_sigmoid_scale, _RevIN

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch is not installed. PatchTSTModel will not be functional.",
        ImportWarning,
    )


class _PatchTSTNet(nn.Module):
    """PatchTST: channel-independent patch transformer for time-series."""

    def __init__(self, seq_len: int, n_channels: int, patch_len: int,
                 stride: int, d_model: int, n_heads: int,
                 n_encoder_layers: int, dropout: float, n_horizons: int = 1,
                 output_activation: str = 'linear', sigmoid_scale: float = 1.0,
                 use_revin: bool = True, target_channel: int = 0,
                 past_window_size: Optional[int] = None):
        super().__init__()
        self.use_revin = use_revin
        self.revin = _RevIN(n_channels, target_channel=target_channel, affine=True) if use_revin else None
        self.n_horizons = n_horizons
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.past_window_size = past_window_size

        # Number of patches
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Patch embedding: linear projection from patch_len to d_model
        self.patch_embed = nn.Linear(patch_len, d_model)

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Output head: from (n_channels * d_model) to n_horizons
        self.head = nn.Linear(n_channels * d_model, n_horizons)
        self.activation = _build_activation(output_activation, scale=sigmoid_scale)

    def forward(self, x):
        # x: (batch, seq_len, n_channels)
        batch_size = x.size(0)

        if self.revin is not None:
            x = self.revin.normalize(x, past_window_size=self.past_window_size)

        # Transpose to channel-first: (batch, n_channels, seq_len)
        x = x.transpose(1, 2)

        # Extract patches: unfold along the seq_len dimension
        # (batch, n_channels, n_patches, patch_len)
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)

        # Channel-independent: merge batch and channel dims
        # (batch * n_channels, n_patches, patch_len)
        x = x.reshape(batch_size * self.n_channels, self.n_patches, self.patch_len)

        # Linear patch embedding: (batch * n_channels, n_patches, d_model)
        x = self.patch_embed(x)

        # Add positional embedding
        x = x + self.pos_embed

        x = self.dropout(x)

        # PF3 (v2.37): patches whose start position is at or after
        # past_window_size are entirely in the zero-target future block.
        # Mask them out of the transformer's self-attention so the
        # encoder can't attend to them. Past-only training windows leave
        # the mask as None — behaviour unchanged. The mean pool below is
        # also restricted to unmasked patches.
        src_key_padding_mask = None
        n_past_patches = self.n_patches
        if (self.past_window_size is not None
                and self.past_window_size < x.size(1) * self.stride):
            # A patch starting at position s covers [s, s + patch_len).
            # We consider a patch "past" if its center is still in the
            # past, i.e. s + patch_len/2 <= past_window_size. Future
            # patches are masked.
            n_past_patches = max(
                1, (self.past_window_size - self.patch_len // 2) // self.stride + 1,
            )
            n_past_patches = min(n_past_patches, self.n_patches)
            if n_past_patches < self.n_patches:
                src_key_padding_mask = torch.zeros(
                    x.size(0), self.n_patches, dtype=torch.bool, device=x.device,
                )
                src_key_padding_mask[:, n_past_patches:] = True

        # Transformer encoder: (batch * n_channels, n_patches, d_model)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.layer_norm(x)

        # Mean pool over patches — when PF3 is active, only over past
        # patches so the pooled vector is not diluted by zero-output
        # transformer states from masked positions.
        if n_past_patches < self.n_patches:
            x = x[:, :n_past_patches, :].mean(dim=1)
        else:
            x = x.mean(dim=1)

        # Reshape back: (batch, n_channels * d_model)
        x = x.reshape(batch_size, self.n_channels * self.d_model)

        # Project to output: (batch, n_horizons)
        out = self.head(x)
        if self.revin is not None:
            out = self.revin.denormalize(out)
        out = self.activation(out)

        if self.n_horizons == 1:
            return out.squeeze(-1)  # (batch,) backward compat
        return out


class PatchTSTModel(ForecastModel):
    """
    PyTorch PatchTST time-series forecasting model.

    Uses channel-independent patch transformer with learnable positional
    embeddings, AdamW optimiser with CosineAnnealingLR, best-model
    checkpointing. Supports multi-horizon output.
    """

    def __init__(
        self,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 16,
        n_heads: int = 2,
        n_encoder_layers: int = 1,
        dropout: float = 0.2,
        learning_rate: float = 2e-4,
        epochs: int = 100,
        batch_size: int = 64,
        sequence_length: Optional[int] = None,
        loss_fn: str = 'mse',
        daily_loss_weight: float = 0.0,
        optimiser: str = 'adamw',
        patience: int = 20,
        output_activation: str = 'linear',
        use_revin: bool = True,
        target_channel: int = 0,
    ) -> None:
        super().__init__()
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed")

        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.loss_fn = loss_fn
        self.daily_loss_weight = float(daily_loss_weight)
        self.optimiser = optimiser
        self.patience = patience
        self.output_activation = output_activation
        self.use_revin = use_revin
        self.target_channel = target_channel

        self._model: Optional[_PatchTSTNet] = None
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
        # past_window_size enables PF1 (RevIN past-only stats); set per-fit
        # from kwargs, round-tripped in save/load. None means legacy
        # single-window path.
        self._past_window_size: Optional[int] = None
        self._training_history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    @property
    def name(self) -> str:
        return "patchtst"

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
        """Train PatchTST with PyTorch autograd and best-model checkpointing."""
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
        self._past_window_size = kwargs.get("past_window_size")

        if not self.use_revin:
            # Per-channel z-score standardisation (fitted on training data)
            self._channel_mean = X_seq.mean(axis=(0, 1))  # shape (n_channels,)
            self._channel_std = X_seq.std(axis=(0, 1))     # shape (n_channels,)
            self._channel_std[self._channel_std < 1e-8] = 1.0
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
        if self.output_activation == 'zscore' and not self.use_revin:
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
        _effective_activation = (
            'linear' if (self.use_revin and self.output_activation == 'zscore')
            else self.output_activation
        )
        self._model = _PatchTSTNet(
            seq_len, input_size, self.patch_len, self.stride,
            self.d_model, self.n_heads, self.n_encoder_layers,
            self.dropout, n_horizons=self._n_horizons,
            output_activation=_effective_activation,
            sigmoid_scale=self._sigmoid_scale,
            use_revin=self.use_revin,
            target_channel=self.target_channel,
            past_window_size=self._past_window_size,
        )
        optimiser = self._build_optimiser(
            self._model.parameters(), self.optimiser, self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.epochs, eta_min=1e-6)
        _loss_map = {'mse': nn.MSELoss, 'mae': nn.L1Loss, 'l1': nn.L1Loss, 'huber': nn.SmoothL1Loss}
        criterion = _loss_map.get(self.loss_fn, nn.MSELoss)(reduction='none')

        # Training loop -- cosine annealing + best-model checkpoint + early stopping
        best_val_loss = float("inf")
        # v2.40.12: best_val_loss tracks raw val_loss (for the
        # checkpoint); best_val_loss_smoothed + val_loss_ema drive the
        # stop decision via _step_early_stop (min_delta + EMA).
        best_val_loss_smoothed = float("inf")
        val_loss_ema: Optional[float] = None
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
            # Best-model checkpoint + early stopping
            # (v2.40.12: shared helper applies min_delta +
            # EMA-smoothed stop decision).
            es = self._step_early_stop(
                val_loss, best_val_loss, best_val_loss_smoothed,
                val_loss_ema, patience_counter,
                min_delta=getattr(self, 'min_delta', 1e-3),
                ema_alpha=getattr(self, 'ema_alpha', 0.3),
            )
            val_loss_ema = es['val_loss_ema']
            best_val_loss = es['best_val_loss']
            best_val_loss_smoothed = es['best_val_loss_smoothed']
            patience_counter = es['patience_counter']
            if es['checkpoint_best']:
                best_state = deepcopy(self._model.state_dict())

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
        if not self.use_revin and self._channel_mean is not None and self._channel_std is not None:
            X_seq = (X_seq - self._channel_mean) / self._channel_std

        X_t = torch.FloatTensor(X_seq)
        self._model.eval()
        with torch.no_grad():
            predictions = self._model(X_t).numpy()

        # Denormalise z-space predictions back to physical units. Floor at
        # zero because the linear head in z-space is unconstrained and
        # callers expect physically-valid (non-negative) forecasts.
        if self.output_activation == 'zscore' and not self.use_revin:
            predictions = predictions * self._y_std + self._y_mean
            predictions = np.clip(predictions, 0.0, None)

        return predictions.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions (returns horizon-0 for multi-horizon models)."""
        self._validate_fitted()
        self._validate_X(X)

        X_seq = self._reshape_to_sequences(X)

        if not self.use_revin and self._channel_mean is not None and self._channel_std is not None:
            X_seq = (X_seq - self._channel_mean) / self._channel_std

        X_t = torch.FloatTensor(X_seq)

        self._model.eval()
        with torch.no_grad():
            predictions = self._model(X_t).numpy()

        # Denormalise z-space predictions *before* slicing to horizon-0 so the
        # per-horizon stats align with each column of the prediction array.
        if self.output_activation == 'zscore' and not self.use_revin:
            predictions = predictions * self._y_std + self._y_mean
            predictions = np.clip(predictions, 0.0, None)

        # Multi-horizon: return only first horizon for backward compat
        if predictions.ndim == 2:
            predictions = predictions[:, 0]

        return predictions.astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return deepcopy({
            "patch_len": self.patch_len, "stride": self.stride,
            "d_model": self.d_model, "n_heads": self.n_heads,
            "n_encoder_layers": self.n_encoder_layers, "dropout": self.dropout,
            "learning_rate": self.learning_rate, "epochs": self.epochs,
            "batch_size": self.batch_size, "sequence_length": self.sequence_length,
            "loss_fn": self.loss_fn,
            "daily_loss_weight": self.daily_loss_weight,
            "optimiser": self.optimiser,
            "patience": self.patience,
            "output_activation": self.output_activation,
            "use_revin": self.use_revin,
            "target_channel": self.target_channel,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"patch_len", "stride", "d_model", "n_heads",
                 "n_encoder_layers", "dropout", "learning_rate",
                 "epochs", "batch_size", "sequence_length", "loss_fn",
                 "daily_loss_weight", "optimiser", "patience",
                 "output_activation",
                 "use_revin", "target_channel"}
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
            "past_window_size": self._past_window_size,
        }, path)
        logger.info(f"Saved PatchTST model to {path}")

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
        self._past_window_size = data.get("past_window_size")

        # Reconstruct the nn.Module and load weights
        self._input_size = data.get("input_size")
        self._seq_len = data.get("seq_len")
        state_dict = data.get("state_dict")
        if state_dict is not None and self._input_size is not None and self._seq_len is not None:
            _effective_activation = (
                'linear' if (self.use_revin and self.output_activation == 'zscore')
                else self.output_activation
            )
            self._model = _PatchTSTNet(
                self._seq_len, self._input_size, self.patch_len, self.stride,
                self.d_model, self.n_heads, self.n_encoder_layers,
                self.dropout, n_horizons=self._n_horizons,
                output_activation=_effective_activation,
                sigmoid_scale=self._sigmoid_scale,
                use_revin=self.use_revin,
                target_channel=self.target_channel,
                past_window_size=self._past_window_size,
            )
            self._model.load_state_dict(state_dict)
            self._model.eval()

        self._is_fitted = True
        logger.info(f"Loaded PatchTST model from {path}")
