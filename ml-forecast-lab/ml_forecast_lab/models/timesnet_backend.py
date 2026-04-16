"""
PyTorch TimesNet forecasting model backend for ML Forecast Lab.

Implements the TimesNet architecture with FFT-based period detection,
1D-to-2D reshape, and inception-style 2D convolution blocks. Uses AdamW
optimisation, CosineAnnealingLR scheduling, best-model checkpointing,
and multi-horizon output.
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
        "PyTorch is not installed. TimesNetModel will not be functional.",
        ImportWarning,
    )


class _InceptionBlock(nn.Module):
    """Three parallel Conv2d branches (1x1, 3x3, 5x5) -> concat -> 1x1 merge."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        # Merge 3 branches back to out_channels
        self.merge = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (batch, channels, H, W)
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        out = torch.cat([b1, b3, b5], dim=1)  # (batch, 3*out_channels, H, W)
        out = self.merge(out)  # (batch, out_channels, H, W)
        return F.relu(out)


class _TimesNetBlock(nn.Module):
    """FFT period detection -> reshape 1D to 2D -> inception Conv2d -> reshape back + residual."""

    def __init__(self, d_model: int, seq_len: int, top_k: int = 3):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.top_k = top_k
        self.inception = _InceptionBlock(d_model, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        #
        # Follows Wu et al. 2023 ("TimesNet: Temporal 2D-Variation Modeling for
        # General Time Series Analysis"): detect the top-k dominant periods via
        # FFT, apply the shared inception block to a 2D reshape for EACH period,
        # and aggregate results with softmax-weighted amplitudes as the
        # importance weights. Previously this block only used the median period
        # and discarded the rest, which collapsed to single-period 2D
        # convolution and defeated the paper's multi-scale contribution.
        batch_size = x.size(0)
        residual = x

        # FFT to find dominant periods.
        # Compute FFT along time dimension for each feature, average across features
        # (batch and channel dims) to get a single per-frequency amplitude.
        x_freq = torch.fft.rfft(x, dim=1)  # (batch, freq_bins, d_model)
        amp = x_freq.abs().mean(dim=(0, 2))  # (freq_bins,)

        # Exclude DC component (index 0) — it encodes the mean, not a period.
        amp = amp.clone()
        amp[0] = 0.0

        # Find top_k dominant frequencies.
        top_k = min(self.top_k, amp.numel() - 1)
        top_amps, top_indices = torch.topk(amp, top_k)

        # Convert (freq_idx, amp) pairs into (period, weight) pairs.
        # Drop any zero-freq indices and deduplicate periods (two nearby freq
        # bins can round to the same period at short seq_len); when we
        # deduplicate we sum the amplitudes so the merged period keeps the
        # combined energy.
        period_to_weight: Dict[int, torch.Tensor] = {}
        for idx, a in zip(top_indices.tolist(), top_amps):
            if idx <= 0:
                continue
            p = max(2, round(self.seq_len / idx))
            p = min(p, self.seq_len)
            if p in period_to_weight:
                period_to_weight[p] = period_to_weight[p] + a
            else:
                period_to_weight[p] = a

        if not period_to_weight:
            # Degenerate input (e.g. near-constant signal): fall back to a
            # single-period pass over the full sequence. Weight doesn't
            # matter because we only have one component.
            period_to_weight = {self.seq_len: top_amps.sum() + 1.0}

        periods = list(period_to_weight.keys())
        weights_raw = torch.stack([period_to_weight[p] for p in periods], dim=0)
        # Softmax-normalise amplitudes so the aggregation is a convex
        # combination. This matches the official TimesNet implementation.
        weights = torch.softmax(weights_raw, dim=0)

        # Apply the shared inception block once per period, aggregate
        # amplitude-weighted.
        out_accum = torch.zeros_like(x)
        for p, w in zip(periods, weights):
            # Pad sequence to multiple of p.
            padded_len = math.ceil(self.seq_len / p) * p
            if padded_len > self.seq_len:
                x_padded = F.pad(
                    x, (0, 0, 0, padded_len - self.seq_len),
                    mode='constant', value=0.0,
                )
            else:
                x_padded = x

            n_rows = padded_len // p

            # Reshape to 2D: (batch, d_model, n_rows, p).
            x_2d = x_padded.permute(0, 2, 1)  # (batch, d_model, padded_len)
            x_2d = x_2d.reshape(batch_size, self.d_model, n_rows, p)

            # Apply inception block.
            x_2d = self.inception(x_2d)  # (batch, d_model, n_rows, p)

            # Reshape back to (batch, padded_len, d_model) and trim.
            x_1d = x_2d.reshape(batch_size, self.d_model, padded_len)
            x_1d = x_1d.permute(0, 2, 1)  # (batch, padded_len, d_model)
            x_1d = x_1d[:, :self.seq_len, :]

            out_accum = out_accum + w * x_1d

        # Residual connection.
        return out_accum + residual


class _TimesNetNet(nn.Module):
    """TimesNet: input projection -> stacked TimesNetBlocks -> LayerNorm -> flatten -> head."""

    def __init__(self, seq_len: int, n_channels: int, d_model: int = 16,
                 n_layers: int = 2, top_k: int = 3, dropout: float = 0.2,
                 n_horizons: int = 1, output_activation: str = 'linear',
                 sigmoid_scale: float = 1.0,
                 use_revin: bool = True, target_channel: int = 0):
        super().__init__()
        self.use_revin = use_revin
        self.revin = _RevIN(n_channels, target_channel=target_channel, affine=True) if use_revin else None
        self.n_horizons = n_horizons
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.d_model = d_model

        # Input projection: (batch, seq_len, n_channels) -> (batch, seq_len, d_model)
        self.input_proj = nn.Linear(n_channels, d_model)

        # Stack of TimesNetBlocks with Dropout
        blocks = []
        for _ in range(n_layers):
            blocks.append(_TimesNetBlock(d_model, seq_len, top_k))
            blocks.append(nn.Dropout(dropout))
        self.blocks = nn.Sequential(*blocks)

        self.layer_norm = nn.LayerNorm(d_model)

        # Flatten (batch, seq_len, d_model) -> (batch, seq_len*d_model) -> head
        # Head hidden size must be >= n_horizons, otherwise multi-horizon
        # output gets bottlenecked and the model can only predict near-
        # constant values across all horizons (flat forecast).
        head_hidden = max(d_model, n_horizons)
        self.head = nn.Sequential(
            nn.Linear(seq_len * d_model, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_horizons),
        )
        self.activation = _build_activation(output_activation, scale=sigmoid_scale)

    def forward(self, x):
        # x: (batch, seq_len, n_channels)
        if self.revin is not None:
            x = self.revin.normalize(x)

        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # TimesNet blocks
        x = self.blocks(x)  # (batch, seq_len, d_model)

        # LayerNorm
        x = self.layer_norm(x)

        # Flatten and project to output
        x = x.reshape(x.size(0), -1)  # (batch, seq_len * d_model)
        out = self.head(x)  # (batch, n_horizons)
        if self.revin is not None:
            out = self.revin.denormalize(out)
        out = self.activation(out)

        if self.n_horizons == 1:
            return out.squeeze(-1)  # (batch,) backward compat
        return out


class TimesNetModel(ForecastModel):
    """
    PyTorch TimesNet time-series forecasting model.

    Uses FFT-based period detection to reshape 1D time series into 2D
    representations, then applies inception-style 2D convolutions to
    capture both intra-period and inter-period patterns. Uses AdamW
    with CosineAnnealingLR, best-model checkpointing. Supports
    multi-horizon output.
    """

    def __init__(
        self,
        d_model: int = 16,
        n_layers: int = 2,
        top_k: int = 3,
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

        self.d_model = d_model
        self.n_layers = n_layers
        self.top_k = top_k
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

        self._model: Optional[_TimesNetNet] = None
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
        return "timesnet"

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
        """Train TimesNet with PyTorch autograd and best-model checkpointing."""
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

        if not self.use_revin:
            # Per-channel z-score standardisation (fitted on training data)
            self._channel_mean = X_seq.mean(axis=(0, 1))  # shape (n_channels,)
            self._channel_std = X_seq.std(axis=(0, 1))     # shape (n_channels,)
            self._channel_std[self._channel_std < 1e-8] = 1.0
            X_seq = (X_seq - self._channel_mean) / self._channel_std

        # Resolve sigmoid scale from training targets (data-driven upper bound).
        if self.output_activation == 'sigmoid':
            self._sigmoid_scale = _resolve_sigmoid_scale(y_train)
        else:
            self._sigmoid_scale = 1.0

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
        self._model = _TimesNetNet(
            seq_len=seq_len,
            n_channels=input_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            top_k=self.top_k,
            dropout=self.dropout,
            n_horizons=self._n_horizons,
            output_activation=_effective_activation,
            sigmoid_scale=self._sigmoid_scale,
            use_revin=self.use_revin,
            target_channel=self.target_channel,
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
            "d_model": self.d_model, "n_layers": self.n_layers,
            "top_k": self.top_k,
            "dropout": self.dropout, "learning_rate": self.learning_rate,
            "epochs": self.epochs, "batch_size": self.batch_size,
            "sequence_length": self.sequence_length, "loss_fn": self.loss_fn,
            "daily_loss_weight": self.daily_loss_weight,
            "optimiser": self.optimiser,
            "patience": self.patience,
            "output_activation": self.output_activation,
            "use_revin": self.use_revin,
            "target_channel": self.target_channel,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"d_model", "n_layers", "top_k",
                 "dropout", "learning_rate", "epochs", "batch_size",
                 "sequence_length", "loss_fn", "daily_loss_weight", "optimiser", "patience", "output_activation",
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
        }, path)
        logger.info(f"Saved TimesNet model to {path}")

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
            _effective_activation = (
                'linear' if (self.use_revin and self.output_activation == 'zscore')
                else self.output_activation
            )
            self._model = _TimesNetNet(
                seq_len=self._seq_len,
                n_channels=self._input_size,
                d_model=self.d_model,
                n_layers=self.n_layers,
                top_k=self.top_k,
                dropout=self.dropout,
                n_horizons=self._n_horizons,
                output_activation=_effective_activation,
                sigmoid_scale=self._sigmoid_scale,
                use_revin=self.use_revin,
                target_channel=self.target_channel,
            )
            self._model.load_state_dict(state_dict)
            self._model.eval()

        self._is_fitted = True
        logger.info(f"Loaded TimesNet model from {path}")
