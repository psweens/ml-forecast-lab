"""
DLinear forecasting model backend for ML Forecast Lab.

Implements the Decomposition-Linear (DLinear) model from:
  "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., 2023)

Decomposes the input into trend (moving average) and seasonal (residual)
components, then applies a separate linear layer to each, summing the outputs.
Simple, fast, and surprisingly competitive on many benchmarks.
Supports multi-horizon output via multi-output linear heads.
"""

import logging
import time
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional

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
        "PyTorch is not installed. DLinearModel will not be functional.",
        ImportWarning,
    )


class _DLinearNet(nn.Module):
    """DLinear: decompose into trend + seasonal, one Linear each.

    When ``n_quantiles > 1`` the heads output ``n_horizons * n_quantiles``
    values per sample so the network produces a quantile band per horizon
    step. The forward returns shape ``(batch, n_horizons, n_quantiles)``
    in that mode; the existing ``(batch, n_horizons)`` shape is preserved
    when ``n_quantiles == 1``.
    """

    def __init__(self, seq_len: int, n_channels: int, kernel_size: int,
                 n_horizons: int = 1, output_activation: str = 'linear',
                 sigmoid_scale: float = 1.0,
                 use_revin: bool = True, target_channel: int = 0,
                 n_quantiles: int = 1,
                 past_window_size: Optional[int] = None):
        super().__init__()
        self.use_revin = use_revin
        # Reversible instance norm (Kim et al. 2022). Handles distribution
        # shift per-window — replaces the need for dataset-level channel
        # z-scoring on non-stationary series.
        self.revin = _RevIN(n_channels, target_channel=target_channel, affine=True) if use_revin else None
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.n_horizons = n_horizons
        self.n_quantiles = max(1, int(n_quantiles))
        self.output_activation = output_activation
        self.target_channel = target_channel
        self.n_channels = n_channels
        # PF7 (v2.37): when past_window_size < seq_len the future block's
        # target-channel slot is always zero. Excluding it from the head
        # input cuts head-input variance imbalance and lets the linear
        # head spend capacity on signal-bearing positions.
        if past_window_size is not None and past_window_size < seq_len:
            self.past_window_size = int(past_window_size)
            self._has_future = True
            future_len = seq_len - self.past_window_size
            flat = (self.past_window_size * n_channels
                    + future_len * (n_channels - 1))
        else:
            self.past_window_size = seq_len
            self._has_future = False
            flat = seq_len * n_channels
        pad = kernel_size // 2
        self.avg_pool = nn.AvgPool1d(kernel_size, stride=1, padding=pad, count_include_pad=False)
        out_dim = n_horizons * self.n_quantiles
        self.trend_linear = nn.Linear(flat, out_dim)
        self.seasonal_linear = nn.Linear(flat, out_dim)
        self.activation = _build_activation(output_activation, scale=sigmoid_scale)

    def _head_flatten(self, x_ch_first: "torch.Tensor") -> "torch.Tensor":
        """Flatten (B, n_channels, seq_len) → (B, flat) honouring PF7."""
        if not self._has_future:
            return x_ch_first.reshape(x_ch_first.shape[0], -1)
        past = x_ch_first[:, :, : self.past_window_size]
        future = x_ch_first[:, :, self.past_window_size :]
        keep = [c for c in range(self.n_channels) if c != self.target_channel]
        future_kept = future[:, keep, :]
        return torch.cat(
            [past.reshape(past.shape[0], -1),
             future_kept.reshape(future_kept.shape[0], -1)],
            dim=1,
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq_len, n_channels)
        if self.revin is not None:
            x = self.revin.normalize(x, past_window_size=self.past_window_size if self._has_future else None)
        x_t = x.permute(0, 2, 1)
        trend = self.avg_pool(x_t)[:, :, :self.seq_len]
        seasonal = x_t - trend
        trend_flat = self._head_flatten(trend)
        seasonal_flat = self._head_flatten(seasonal)
        out = self.trend_linear(trend_flat) + self.seasonal_linear(seasonal_flat)
        if self.n_quantiles > 1:
            out = out.view(-1, self.n_horizons, self.n_quantiles)
        if self.revin is not None:
            # Denormalise in z-space before the output activation so the
            # activation operates on physical-scale values (matters for
            # softplus / sigmoid / exp whose range constraints are only
            # meaningful in target space).
            out = self.revin.denormalize(out)
        out = self.activation(out)
        if self.n_quantiles > 1:
            return out  # (batch, n_horizons, n_quantiles)
        if self.n_horizons == 1:
            return out.squeeze(-1)
        return out


class DLinearModel(ForecastModel):
    """
    DLinear time-series forecasting model.

    Decomposes the input into trend and seasonal components via a moving
    average kernel, then projects each component to the forecast with a
    separate linear layer. Supports multi-horizon output.
    """

    def __init__(
        self,
        kernel_size: int = 13,
        learning_rate: float = 5e-4,
        epochs: int = 100,
        batch_size: int = 64,
        sequence_length: Optional[int] = None,
        loss_fn: str = 'huber',
        daily_loss_weight: float = 0.0,
        optimiser: str = 'adamw',
        patience: int = 20,
        output_activation: str = 'linear',
        use_revin: bool = True,
        target_channel: int = 0,
        quantiles: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed")
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.loss_fn = loss_fn
        self.daily_loss_weight = float(daily_loss_weight)
        self.optimiser = optimiser
        self.patience = patience
        self.output_activation = output_activation
        # RevIN (Kim et al. 2022) handles per-window distribution shift. When
        # on, it supersedes both the dataset-level channel normalisation and
        # the zscore output_activation path — RevIN owns the scale end to end.
        self.use_revin = use_revin
        self.target_channel = target_channel
        # Optional list of quantiles in (0, 1). When non-empty the network
        # grows a multi-quantile output head and trains with pinball loss.
        # The median quantile (or 0.5 if absent) is the point forecast used
        # everywhere downstream that expects a 1D prediction.
        self.quantiles: List[float] = list(quantiles) if quantiles else []

        self._model: Optional[_DLinearNet] = None
        self._seq_len: Optional[int] = None
        self._n_channels: Optional[int] = None
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

    @property
    def name(self) -> str:
        return "dlinear"

    @property
    def is_neural(self) -> bool:
        return True

    def _reshape_to_sequences(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        seq_len = self.sequence_length or n_features
        n_features_per_step = n_features // seq_len if n_features % seq_len == 0 else 1
        return X[:, :seq_len * n_features_per_step].reshape(n_samples, seq_len, n_features_per_step)

    def _build_model(self, seq_len: int, n_channels: int,
                     n_horizons: int = 1) -> "_DLinearNet":
        # When RevIN is on, the network owns the scale — treat any 'zscore'
        # activation request as 'linear' inside the forward path because
        # zscore's identity head is what RevIN expects anyway.
        _effective_activation = (
            'linear' if (self.use_revin and self.output_activation == 'zscore')
            else self.output_activation
        )
        return _DLinearNet(
            seq_len, n_channels, self.kernel_size,
            n_horizons=n_horizons,
            output_activation=_effective_activation,
            sigmoid_scale=self._sigmoid_scale,
            use_revin=self.use_revin,
            target_channel=self.target_channel,
            n_quantiles=max(1, len(self.quantiles)),
            past_window_size=self._past_window_size,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        """Train DLinear model."""
        self._validate_X(X_train)
        y_train = self._validate_y(y_train)
        start_time = time.time()

        # Detect multi-horizon from y shape
        if y_train.ndim == 2 and y_train.shape[1] > 1:
            self._n_horizons = y_train.shape[1]
        else:
            self._n_horizons = 1

        sequence_data = kwargs.get("sequence_data")
        if sequence_data is not None:
            X_seq = sequence_data
            logger.debug(f"Using pre-windowed sequence data: {X_seq.shape}")
        else:
            X_seq = self._reshape_to_sequences(X_train)

        _, seq_len, n_channels = X_seq.shape
        self._seq_len = seq_len
        self._n_channels = n_channels
        self._past_window_size = kwargs.get("past_window_size")

        # Dataset-level channel normalisation is mutually exclusive with
        # RevIN: RevIN handles per-window instance-level normalisation inside
        # the network's forward pass, so applying a global z-score first
        # would double-normalise and wash out the instance signal RevIN
        # relies on.
        if not self.use_revin:
            self._channel_mean = X_seq.mean(axis=(0, 1))
            self._channel_std = X_seq.std(axis=(0, 1))
            self._channel_std[self._channel_std < 1e-8] = 1.0
            X_seq = (X_seq - self._channel_mean) / self._channel_std

        # Sigmoid activation needs a ceiling: use training-data maximum with
        # a 10% buffer so the network can reach observed extrema. Other
        # activations leave the default scale (1.0) which _build_activation
        # ignores for non-sigmoid cases.
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

        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_t = torch.FloatTensor(y_tr)
        w_tr_t = torch.FloatTensor(w_tr) if w_tr is not None else None
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)

        self._model = self._build_model(seq_len, n_channels,
                                        n_horizons=self._n_horizons)
        optimiser = self._build_optimiser(
            self._model.parameters(), self.optimiser, self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.epochs, eta_min=1e-6)
        _loss_map = {'mse': nn.MSELoss, 'mae': nn.L1Loss, 'l1': nn.L1Loss, 'huber': nn.SmoothL1Loss}
        criterion = _loss_map.get(self.loss_fn, nn.SmoothL1Loss)(reduction='none')

        # Pinball criterion for multi-quantile mode. y_pred is
        # (batch, H, Q), y_true is (batch, H) broadcast to (batch, H, 1).
        # Returns per-sample mean across (H, Q) so the existing
        # composite_horizon_loss path keeps shape (batch,).
        quantile_tensor = (
            torch.tensor(self.quantiles, dtype=torch.float32)
            if self.quantiles else None
        )

        def _pinball(yp: "torch.Tensor", yt: "torch.Tensor") -> "torch.Tensor":
            if yp.dim() == 2:
                # Single-quantile fallback path: behave like criterion(yp, yt).
                return criterion(yp, yt)
            qs = quantile_tensor.to(yp.device).view(1, 1, -1)
            yt_b = yt.unsqueeze(-1) if yt.dim() == 2 else yt
            err = yt_b - yp
            loss = torch.maximum(qs * err, (qs - 1.0) * err)
            return loss.mean(dim=-1)  # collapse quantile axis → (batch, H)

        if self.quantiles:
            criterion = _pinball

        best_val_loss = float("inf")
        # v2.40.12: best_val_loss tracks raw val_loss (for the
        # checkpoint); best_val_loss_smoothed + val_loss_ema drive the
        # stop decision via _step_early_stop (min_delta + EMA).
        best_val_loss_smoothed = float("inf")
        val_loss_ema: Optional[float] = None
        best_state = None
        patience_counter = 0

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

            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_val_t)
                val_loss_t, _ = self._composite_horizon_loss(
                    val_pred, y_val_t, criterion, None, self.daily_loss_weight,
                )
                val_loss = val_loss_t.item()

            avg_loss = epoch_loss / max(n_batches, 1)

            self._emit_epoch(kwargs.get("epoch_callback"),
                model_name=self.name, epoch=epoch + 1, total_epochs=self.epochs,
                train_loss=avg_loss, val_loss=val_loss, lr=optimiser.param_groups[0]['lr'],
                patience_counter=patience_counter, patience_limit=self.patience,
                best_val_loss=best_val_loss)

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
                current_lr = optimiser.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs}: train={avg_loss:.6f}, val={val_loss:.6f}, lr={current_lr:.2e}"
                )

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch + 1} (no improvement for {self.patience} epochs)")
                break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        elapsed = time.time() - start_time
        self._is_fitted = True
        return {"time_seconds": elapsed, "epochs": epoch + 1, "best_val_loss": float(best_val_loss)}

    def predict_sequence(self, X: np.ndarray) -> np.ndarray:
        """Multi-horizon prediction from sliding-window input.

        When ``self.quantiles`` is non-empty the network produces a
        (n_samples, n_horizons, n_quantiles) tensor; this method returns
        only the median column so the existing single-prediction pipeline
        (metrics, scoring, conformal-residual logging) is unaffected. Use
        ``predict_quantiles`` to retrieve the full quantile band.
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

        if predictions.ndim == 3:
            # (n_samples, n_horizons, n_quantiles) → take the median column.
            qs = self.quantiles
            if 0.5 in qs:
                idx = qs.index(0.5)
            else:
                idx = min(range(len(qs)), key=lambda i: abs(qs[i] - 0.5))
            predictions = predictions[:, :, idx]

        if self.output_activation == 'zscore' and not self.use_revin:
            predictions = predictions * self._y_std + self._y_mean
            predictions = np.clip(predictions, 0.0, None)

        return predictions.astype(np.float32)

    def predict_quantiles(self, X: np.ndarray) -> np.ndarray:
        """Return the full quantile band (n_samples, n_horizons, n_quantiles).

        Raises ``RuntimeError`` if the model was not trained with a
        non-empty ``quantiles`` list — there is no band to return.
        """
        if not self.quantiles:
            raise RuntimeError(
                "predict_quantiles requires quantiles to be configured at "
                "construction; got an empty list"
            )
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

        if predictions.ndim != 3:
            raise RuntimeError(
                f"DLinear quantile head produced shape {predictions.shape}; "
                f"expected (batch, H, Q)"
            )
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
            "kernel_size": self.kernel_size, "learning_rate": self.learning_rate,
            "epochs": self.epochs, "batch_size": self.batch_size,
            "sequence_length": self.sequence_length, "loss_fn": self.loss_fn,
            "daily_loss_weight": self.daily_loss_weight,
            "optimiser": self.optimiser,
            "patience": self.patience,
            "output_activation": self.output_activation,
            "use_revin": self.use_revin,
            "target_channel": self.target_channel,
            "quantiles": list(self.quantiles),
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"kernel_size", "learning_rate", "epochs", "batch_size",
                 "sequence_length", "loss_fn", "daily_loss_weight", "optimiser", "patience",
                 "output_activation",
                 "use_revin", "target_channel", "quantiles"}
        for k, v in kwargs.items():
            if k not in valid:
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)

    def save(self, path: str) -> None:
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Cannot save unfitted model")
        torch.save({
            "state_dict": self._model.state_dict(),
            "params": self.get_params(),
            "seq_len": self._seq_len,
            "n_channels": self._n_channels,
            "n_horizons": self._n_horizons,
            "channel_mean": self._channel_mean,
            "channel_std": self._channel_std,
            "sigmoid_scale": self._sigmoid_scale,
            "y_mean": self._y_mean,
            "y_std": self._y_std,
            "past_window_size": self._past_window_size,
        }, path)
        logger.info(f"Saved DLinear model to {path}")

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        self.set_params(**data["params"])
        self._seq_len = data.get("seq_len")
        self._n_channels = data.get("n_channels")
        self._channel_mean = data.get("channel_mean")
        self._channel_std = data.get("channel_std")
        self._n_horizons = data.get("n_horizons", 1)
        self._sigmoid_scale = float(data.get("sigmoid_scale", 1.0))
        self._y_mean = data.get("y_mean", 0.0)
        self._y_std = data.get("y_std", 1.0)
        self._past_window_size = data.get("past_window_size")

        if self._seq_len is not None and self._n_channels is not None:
            self._model = self._build_model(self._seq_len, self._n_channels,
                                            n_horizons=self._n_horizons)
            self._model.load_state_dict(data["state_dict"])
            self._model.eval()
        self._is_fitted = True
        logger.info(f"Loaded DLinear model from {path}")
