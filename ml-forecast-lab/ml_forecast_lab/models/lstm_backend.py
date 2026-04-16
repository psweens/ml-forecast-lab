"""
PyTorch LSTM forecasting model backend for ML Forecast Lab.

Implements a multi-layer LSTM with temporal attention using PyTorch,
AdamW optimisation, cosine-annealing learning-rate schedule, best-model
checkpointing, and early stopping. Supports multi-horizon output
via a shared encoder and multi-output head.
"""

import logging
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

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float, n_horizons: int = 1,
                 output_activation: str = 'linear', sigmoid_scale: float = 1.0,
                 use_revin: bool = True, target_channel: int = 0):
        super().__init__()
        self.n_horizons = n_horizons
        self.use_revin = use_revin
        # Reversible instance norm (Kim et al. 2022). Handles distribution
        # shift per-window — replaces the need for dataset-level target
        # z-scoring on non-stationary series.
        self.revin = _RevIN(input_size, target_channel=target_channel, affine=True) if use_revin else None
        self.layer_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = _TemporalAttention(hidden_size)
        head_hidden = max(hidden_size, n_horizons)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_horizons),
        )
        self.activation = _build_activation(output_activation, scale=sigmoid_scale)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        if self.revin is not None:
            x = self.revin.normalize(x)
        x = self.layer_norm(x)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        context = self.attention(lstm_out)  # (batch, hidden_size)
        out = self.head(context)  # (batch, n_horizons)
        if self.revin is not None:
            # Denormalise in z-space before the output activation so the
            # activation operates on physical-scale values (matters for
            # softplus / sigmoid / exp whose range constraints are only
            # meaningful in target space).
            if out.dim() == 1:
                out = self.revin.denormalize(out)
            else:
                out = self.revin.denormalize(out)
        out = self.activation(out)
        if self.n_horizons == 1:
            return out.squeeze(-1)  # (batch,) backward compat
        return out


class LSTMModel(ForecastModel):
    """
    PyTorch LSTM time-series forecasting model with temporal attention.

    Uses torch.nn.LSTM with autograd, temporal attention over all timesteps,
    Adam optimiser with ReduceLROnPlateau, best-model checkpointing,
    and early stopping on validation loss. Supports multi-horizon output.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 2e-4,
        epochs: int = 100,
        batch_size: int = 64,
        sequence_length: Optional[int] = None,
        loss_fn: str = 'mse',
        daily_loss_weight: float = 0.0,
        patience: int = 20,
        output_activation: str = 'linear',
        use_revin: bool = True,
        target_channel: int = 0,
    ) -> None:
        super().__init__()
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.loss_fn = loss_fn
        self.daily_loss_weight = float(daily_loss_weight)
        self.patience = patience
        self.output_activation = output_activation
        # RevIN (Kim et al. 2022) handles per-window distribution shift. When
        # on, it supersedes both the dataset-level channel normalisation and
        # the zscore output_activation path — RevIN owns the scale end to end.
        self.use_revin = use_revin
        self.target_channel = target_channel

        self._model: Optional[_LSTMNet] = None
        self._input_size: Optional[int] = None
        self._n_horizons: int = 1
        self._channel_mean: Optional[np.ndarray] = None
        self._channel_std: Optional[np.ndarray] = None
        self._sigmoid_scale: float = 1.0
        # Target z-score stats (only populated when output_activation == 'zscore'
        # AND use_revin is False — otherwise RevIN handles scale per-window).
        # Scalar for single-horizon, per-horizon ndarray for multi-horizon.
        # Defaults (0.0, 1.0) are identity — safe no-op for non-zscore paths.
        self._y_mean: Any = 0.0
        self._y_std: Any = 1.0
        self._training_history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    @property
    def name(self) -> str:
        return "lstm"

    @property
    def is_neural(self) -> bool:
        return True

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

        # Dataset-level channel normalisation is mutually exclusive with
        # RevIN: RevIN handles per-window instance-level normalisation inside
        # the network's forward pass, so applying a global z-score first
        # would double-normalise and wash out the instance signal RevIN
        # relies on.
        if not self.use_revin:
            # Per-channel z-score standardisation (fitted on training data)
            self._channel_mean = X_seq.mean(axis=(0, 1))  # shape (n_channels,)
            self._channel_std = X_seq.std(axis=(0, 1))     # shape (n_channels,)
            self._channel_std[self._channel_std < 1e-8] = 1.0  # Avoid division by zero
            X_seq = (X_seq - self._channel_mean) / self._channel_std

        # Sigmoid activation needs a ceiling: use training-data maximum with
        # a 10% buffer so the network can reach observed extrema.
        if self.output_activation == 'sigmoid':
            self._sigmoid_scale = _resolve_sigmoid_scale(y_train)

        # Target z-score normalisation (output_activation == 'zscore')
        # compute per-horizon (mean, std) from training targets and transform
        # y into z-space before training. The network predicts in z-space
        # through a linear head, and predictions are denormalised back to
        # physical units at inference time. Keeps gradient magnitudes O(1)
        # regardless of target scale, which is important for multi-horizon
        # MSE/Huber losses on raw targets with wide dynamic ranges.
        #
        # When use_revin is True this path is skipped — RevIN already
        # provides per-window scale normalisation and the two schemes
        # compose incorrectly.
        if self.output_activation == 'zscore' and not self.use_revin:
            if y_train.ndim == 2 and self._n_horizons > 1:
                y_mean = y_train.mean(axis=0)
                y_std = y_train.std(axis=0)
                # Guard constant-target horizons from div-by-zero
                y_std = np.where(y_std < 1e-8, 1.0, y_std)
                self._y_mean = y_mean.astype(np.float32)
                self._y_std = y_std.astype(np.float32)
            else:
                self._y_mean = float(y_train.mean())
                self._y_std = float(y_train.std())
                if self._y_std < 1e-8:
                    self._y_std = 1.0
            y_train = (y_train - self._y_mean) / self._y_std
            logger.info(
                f"[zscore] target normalised: "
                f"mean={np.asarray(self._y_mean).mean():.4f}, "
                f"std={np.asarray(self._y_std).mean():.4f}"
            )
        elif self.output_activation == 'zscore' and self.use_revin:
            logger.debug(
                "LSTM: output_activation='zscore' ignored because use_revin=True; "
                "RevIN owns the scale end to end."
            )

        # Extract sample weights
        sample_weight = kwargs.get("sample_weight")

        # Tail validation split — val is the most recent slice; a purge gap
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
        # When RevIN is on, the network owns the scale — treat any 'zscore'
        # activation request as 'linear' inside the forward path because
        # zscore's identity head is what RevIN expects anyway.
        _effective_activation = (
            'linear' if (self.use_revin and self.output_activation == 'zscore')
            else self.output_activation
        )
        self._model = _LSTMNet(
            input_size, self.hidden_size, self.num_layers, self.dropout,
            n_horizons=self._n_horizons,
            output_activation=_effective_activation,
            sigmoid_scale=self._sigmoid_scale,
            use_revin=self.use_revin,
            target_channel=self.target_channel,
        )
        optimiser = torch.optim.AdamW(self._model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.epochs, eta_min=1e-6)
        _loss_map = {'mse': nn.MSELoss, 'mae': nn.L1Loss, 'l1': nn.L1Loss, 'huber': nn.SmoothL1Loss}
        criterion = _loss_map.get(self.loss_fn, nn.MSELoss)(reduction='none')

        # Training loop — cosine annealing + best-model checkpoint + early stopping
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
        # Dataset-level channel normalisation only applies when RevIN is off —
        # otherwise RevIN handles per-window normalisation inside forward().
        if not self.use_revin and self._channel_mean is not None and self._channel_std is not None:
            X_seq = (X_seq - self._channel_mean) / self._channel_std

        X_t = torch.FloatTensor(X_seq)
        self._model.eval()
        with torch.no_grad():
            predictions = self._model(X_t).numpy()

        # Denormalise z-space predictions back to physical units. For
        # multi-horizon the stats are (n_horizons,) and broadcast across rows;
        # for single-horizon they're scalars. Floor at zero — the linear
        # head in z-space is unconstrained, and denormalising a slightly
        # negative z-prediction can produce values below zero for
        # non-negative physical targets. Skipped when use_revin is True:
        # the network already returns target-space predictions.
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

        # Denormalise z-space predictions *before* slicing to horizon-0, so
        # the per-horizon stats align with each column of the prediction array.
        # Floor at zero — see predict_sequence() for rationale.
        if self.output_activation == 'zscore' and not self.use_revin:
            predictions = predictions * self._y_std + self._y_mean
            predictions = np.clip(predictions, 0.0, None)

        # Multi-horizon: return only first horizon for backward compat
        if predictions.ndim == 2:
            predictions = predictions[:, 0]

        return predictions.astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return deepcopy({
            "hidden_size": self.hidden_size, "num_layers": self.num_layers,
            "dropout": self.dropout, "learning_rate": self.learning_rate,
            "epochs": self.epochs, "batch_size": self.batch_size,
            "sequence_length": self.sequence_length, "loss_fn": self.loss_fn,
            "daily_loss_weight": self.daily_loss_weight,
            "patience": self.patience,
            "output_activation": self.output_activation,
            "use_revin": self.use_revin,
            "target_channel": self.target_channel,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"hidden_size", "num_layers", "dropout", "learning_rate",
                 "epochs", "batch_size", "sequence_length", "loss_fn",
                 "daily_loss_weight",
                 "patience",
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
            "n_horizons": self._n_horizons,
            "channel_mean": self._channel_mean,
            "channel_std": self._channel_std,
            "sigmoid_scale": self._sigmoid_scale,
            # Target z-score stats (only meaningful when output_activation == 'zscore',
            # otherwise kept as identity defaults). Written unconditionally so old
            # checkpoints re-saved from a zscore run round-trip cleanly.
            "y_mean": self._y_mean,
            "y_std": self._y_std,
        }, path)
        logger.info(f"Saved LSTM model to {path}")

    def load(self, path: str) -> None:
        """Load model state dict."""
        data = torch.load(path, map_location="cpu")
        self.set_params(**data["params"])
        self._channel_mean = data.get("channel_mean")
        self._channel_std = data.get("channel_std")
        self._n_horizons = data.get("n_horizons", 1)
        self._sigmoid_scale = float(data.get("sigmoid_scale", 1.0))
        # Restore target z-score stats; fall back to identity (0, 1) for
        # checkpoints saved before this field existed.
        self._y_mean = data.get("y_mean", 0.0)
        self._y_std = data.get("y_std", 1.0)

        # Reconstruct the nn.Module and load weights
        self._input_size = data.get("input_size")
        state_dict = data.get("state_dict")
        if state_dict is not None and self._input_size is not None:
            _effective_activation = (
                'linear' if (self.use_revin and self.output_activation == 'zscore')
                else self.output_activation
            )
            self._model = _LSTMNet(
                self._input_size, self.hidden_size, self.num_layers, self.dropout,
                n_horizons=self._n_horizons,
                output_activation=_effective_activation,
                sigmoid_scale=self._sigmoid_scale,
                use_revin=self.use_revin,
                target_channel=self.target_channel,
            )
            self._model.load_state_dict(state_dict)
            self._model.eval()

        self._is_fitted = True
        logger.info(f"Loaded LSTM model from {path}")
