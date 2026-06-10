"""
NLinear forecasting model backend for ML Forecast Lab.

Implements the Normalisation-Linear (NLinear) model from:
  "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., 2023)

NLinear subtracts the last value of the input window before applying a single
linear projection, then re-adds it. The subtraction acts as a per-window mean
shift, so the linear layer only has to model the residual dynamics around the
current level. This is the lighter, simpler companion to DLinear and remains a
top-tier baseline on many long-horizon benchmarks despite its trivial size.
"""

import logging
import time
import warnings
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

from .base import ForecastModel, _build_activation, _resolve_sigmoid_scale, _RevIN

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch is not installed. NLinearModel will not be functional.",
        ImportWarning,
    )


class _NLinearNet(nn.Module):
    """NLinear: subtract last value, single Linear, add back.

    v2.37 changes (driven by docs/investigations/2026-05-neural-pv.md):

    * **PF1**: when ``past_window_size`` is set, RevIN computes per-window
      mean/std over the past slice only — undoes the 50% mean bias that
      the future-position zero-target padding introduces in extended-
      window mode.
    * **PF2**: the "subtract the last value, re-add at the end" anchor
      uses ``x[:, past_window_size - 1, target_channel]`` instead of
      the literal last row. In extended mode the literal last row is a
      future position where the target channel is always zero, which
      makes the anchor trick a no-op.
    * **PF7**: the head's input omits the future-position target-channel
      slots (always zero, no signal). The flat input shape is therefore
      ``W*C + H*(C-1)`` rather than ``(W+H)*C``. Past-only training
      windows fall back to the original ``W*C`` shape — fully
      backwards-compatible with checkpoints that pre-date this change.
    """

    def __init__(self, seq_len: int, n_channels: int, n_horizons: int = 1,
                 output_activation: str = 'linear', sigmoid_scale: float = 1.0,
                 use_revin: bool = True, target_channel: int = 0,
                 past_window_size: Optional[int] = None,
                 seasonal_init_period: Optional[int] = None):
        super().__init__()
        self.use_revin = use_revin
        # RevIN composes with NLinear's last-value subtraction without
        # conflict because RevIN's stats are computed from the original
        # window — we only subtract the last value *after* RevIN normalises.
        self.revin = (
            _RevIN(n_channels, target_channel=target_channel, affine=True)
            if use_revin else None
        )
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.n_horizons = n_horizons
        self.target_channel = target_channel
        # past_window_size is the length of the past slice within seq_len.
        # None or == seq_len means "no future positions" (legacy path).
        if past_window_size is None or past_window_size >= seq_len:
            self.past_window_size = seq_len
            self._has_future = False
            flat = seq_len * n_channels
        else:
            self.past_window_size = int(past_window_size)
            self._has_future = True
            future_len = seq_len - self.past_window_size
            # Past block: every channel; future block: every channel except
            # the target (PF7) — those slots are always zero so they only
            # add input-imbalance noise to the linear head.
            flat = (self.past_window_size * n_channels
                    + future_len * (n_channels - 1))
        self.linear = nn.Linear(flat, n_horizons)
        # Seasonal-naive initialisation (v2.41.0). Start the head AT the
        # seasonal-naive solution — horizon h reads the target value at
        # "same time yesterday" with weight 1 — instead of random noise
        # around the anchor. Training then only has to learn corrections
        # on top of an already-sensible forecast. Without this, pure
        # per-interval loss needs many more epochs than the production
        # epoch budget to grow the daily amplitude from scratch, which
        # surfaced as the flat / strongly-under-peaked forecasts pinned
        # by tests/integration/test_pv_forecast_pipeline.py once the
        # (removed) cumulative-loss term stopped accelerating amplitude
        # convergence. Horizons beyond one period wrap to the same
        # time-of-day slot; horizons whose yesterday-slot falls outside
        # the past window keep the zero init (= flat-anchor persistence,
        # NLinear's natural baseline). Composes with RevIN + the anchor
        # subtract/re-add: at init the output is exactly the
        # denormalised yesterday value.
        if seasonal_init_period and seasonal_init_period >= 2:
            period = int(seasonal_init_period)
            with torch.no_grad():
                self.linear.weight.zero_()
                self.linear.bias.zero_()
                for h_idx in range(n_horizons):
                    q = (h_idx % period) + 1  # 1-based step within the day
                    p = self.past_window_size - 1 - period + q
                    if 0 <= p < self.past_window_size:
                        self.linear.weight[
                            h_idx, p * n_channels + target_channel
                        ] = 1.0
        self.activation = _build_activation(output_activation, scale=sigmoid_scale)

    def _head_input(self, x: "torch.Tensor") -> "torch.Tensor":
        """Reshape ``x`` into the flat tensor the head expects.

        Past block: take all channels. Future block: drop the target
        channel (PF7 — always zero anyway, removes head-input variance
        imbalance). For the legacy past-only path returns the same
        flatten as before.
        """
        if not self._has_future:
            return x.reshape(x.size(0), -1)
        past = x[:, : self.past_window_size, :]
        future = x[:, self.past_window_size :, :]
        # Drop the target channel from the future block.
        keep = [c for c in range(self.n_channels) if c != self.target_channel]
        future_kept = future[:, :, keep]
        flat_past = past.reshape(past.size(0), -1)
        flat_future = future_kept.reshape(future_kept.size(0), -1)
        return torch.cat([flat_past, flat_future], dim=1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq_len, n_channels)
        if self.revin is not None:
            x = self.revin.normalize(x, past_window_size=self.past_window_size)
        # PF2: anchor on the last PAST step, not the literal last row of
        # the (possibly extended) window. ``self.past_window_size - 1``
        # equals ``seq_len - 1`` on the legacy path, so this is fully
        # backwards-compatible.
        anchor_idx = self.past_window_size - 1
        last_val = x[:, anchor_idx : anchor_idx + 1, self.target_channel]   # (B, 1)
        # Subtract from every channel at every position (broadcasts).
        x_shifted = x - x[:, anchor_idx : anchor_idx + 1, :]
        flat = self._head_input(x_shifted)
        out = self.linear(flat)  # (batch, n_horizons)
        # Re-add the target's anchor value.
        out = out + last_val
        if self.revin is not None:
            out = self.revin.denormalize(out)
        out = self.activation(out)
        if self.n_horizons == 1:
            return out.squeeze(-1)
        return out


class NLinearModel(ForecastModel):
    """
    NLinear time-series forecasting model.

    Single linear layer over the input window after subtracting the most
    recent value, with the last value added back at the output. Tiny but
    competitive — published as DLinear's companion baseline.
    """

    def __init__(
        self,
        learning_rate: float = 5e-4,
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

        self._model: Optional[_NLinearNet] = None
        self._seq_len: Optional[int] = None
        self._n_channels: Optional[int] = None
        self._n_horizons: int = 1
        self._channel_mean: Optional[np.ndarray] = None
        self._channel_std: Optional[np.ndarray] = None
        self._sigmoid_scale: float = 1.0
        self._y_mean: Any = 0.0
        self._y_std: Any = 1.0
        # past_window_size enables PF1/PF2/PF7 (see _NLinearNet docstring).
        # Set per-fit from kwargs; round-tripped in save/load. None means
        # legacy single-window path.
        self._past_window_size: Optional[int] = None
        # Steps-per-day inferred at fit time for the seasonal-naive head
        # init. Not persisted: load() restores trained weights over the
        # init, so the value only matters for a fresh fit.
        self._seasonal_period: Optional[int] = None

    @property
    def name(self) -> str:
        return "nlinear"

    @property
    def is_neural(self) -> bool:
        return True

    def _reshape_to_sequences(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        seq_len = self.sequence_length or n_features
        n_features_per_step = (
            n_features // seq_len if n_features % seq_len == 0 else 1
        )
        return X[:, :seq_len * n_features_per_step].reshape(
            n_samples, seq_len, n_features_per_step
        )

    def _build_model(self, seq_len: int, n_channels: int,
                     n_horizons: int = 1) -> "_NLinearNet":
        _eff = ('linear' if (self.use_revin and self.output_activation == 'zscore')
                else self.output_activation)
        return _NLinearNet(
            seq_len, n_channels,
            n_horizons=n_horizons,
            output_activation=_eff,
            sigmoid_scale=self._sigmoid_scale,
            use_revin=self.use_revin,
            target_channel=self.target_channel,
            past_window_size=self._past_window_size,
            seasonal_init_period=self._seasonal_period,
        )

    @staticmethod
    def _infer_seasonal_period(X_seq: np.ndarray,
                               channel_names: Optional[list]) -> Optional[int]:
        """Infer steps-per-day from the deterministic hour_sin/hour_cos
        channels of the first training window.

        The sliding-window builder always appends these when
        ``add_temporal=True``. The hour angle is quantised to whole hours
        (``2π·hour/24``), so consecutive sub-hourly rows can share an
        angle — the per-row delta is 0 within an hour and 2π/24 at each
        hour boundary. Counting boundary crossings across the window
        therefore recovers steps-per-hour (rows per crossing) without the
        backend needing ``interval_minutes``: 30-min data shows ~half the
        rows crossing, 60-min data every row, 15-min data a quarter.
        Returns None when the temporal channels are absent or the window
        spans no hour boundary — the caller then skips the seasonal-naive
        init and keeps the zero/flat-anchor behaviour.
        """
        if not channel_names:
            return None
        try:
            sin_idx = channel_names.index('hour_sin')
            cos_idx = channel_names.index('hour_cos')
        except ValueError:
            return None
        if X_seq.shape[0] < 1 or X_seq.shape[1] < 3:
            return None
        theta = np.arctan2(
            X_seq[0, :, sin_idx].astype(np.float64),
            X_seq[0, :, cos_idx].astype(np.float64),
        )
        deltas = np.mod(np.diff(theta), 2 * np.pi)
        crossings = int(np.count_nonzero(deltas > 1e-6))
        if crossings < 1:
            return None
        steps_per_hour = int(round(len(deltas) / crossings))
        if steps_per_hour < 1:
            return None
        period = 24 * steps_per_hour
        # 15-min..1-hour sampling — anything outside is either sub-minute
        # data (init pointless) or coarser-than-hourly (mapping undefined).
        if not 4 <= period <= 288:
            return None
        return period

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            **kwargs: Any) -> Dict[str, Any]:
        self._validate_X(X_train)
        y_train = self._validate_y(y_train)
        start_time = time.time()

        if y_train.ndim == 2 and y_train.shape[1] > 1:
            self._n_horizons = y_train.shape[1]
        else:
            self._n_horizons = 1

        sequence_data = kwargs.get("sequence_data")
        if sequence_data is not None:
            X_seq = sequence_data
        else:
            X_seq = self._reshape_to_sequences(X_train)

        _, seq_len, n_channels = X_seq.shape
        self._seq_len = seq_len
        self._n_channels = n_channels
        # PF1/PF2/PF7 plumbing: training pipeline passes past_window_size
        # in seq_kwargs when extended_window is True. None means the
        # legacy single-window path; the _Net falls back to its v2.36
        # behaviour in that case.
        self._past_window_size = kwargs.get("past_window_size")
        # Steps-per-day for the seasonal-naive head init, recovered from
        # the temporal channels so no extra plumbing is needed from the
        # caller. None disables the init (see _NLinearNet).
        self._seasonal_period = self._infer_seasonal_period(
            X_seq, kwargs.get("channel_names"),
        )

        if not self.use_revin:
            self._channel_mean = X_seq.mean(axis=(0, 1))
            self._channel_std = X_seq.std(axis=(0, 1))
            self._channel_std[self._channel_std < 1e-8] = 1.0
            X_seq = (X_seq - self._channel_mean) / self._channel_std

        if self.output_activation == 'sigmoid':
            self._sigmoid_scale = _resolve_sigmoid_scale(y_train)

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

        sample_weight = kwargs.get("sample_weight")
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.epochs, eta_min=1e-6,
        )
        _loss_map = {'mse': nn.MSELoss, 'mae': nn.L1Loss,
                     'l1': nn.L1Loss, 'huber': nn.SmoothL1Loss}
        criterion = _loss_map.get(self.loss_fn, nn.MSELoss)(reduction='none')

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
                train_loss=avg_loss, val_loss=val_loss,
                lr=optimiser.param_groups[0]['lr'],
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

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        elapsed = time.time() - start_time
        self._is_fitted = True
        return {"time_seconds": elapsed, "epochs": epoch + 1,
                "best_val_loss": float(best_val_loss)}

    def predict_sequence(self, X: np.ndarray) -> np.ndarray:
        self._validate_fitted()
        if self._model is None:
            raise RuntimeError("No model loaded")
        X_seq = X.copy()
        if not self.use_revin and self._channel_mean is not None:
            X_seq = (X_seq - self._channel_mean) / self._channel_std
        X_t = torch.FloatTensor(X_seq)
        self._model.eval()
        with torch.no_grad():
            predictions = self._model(X_t).numpy()
        if self.output_activation == 'zscore' and not self.use_revin:
            predictions = predictions * self._y_std + self._y_mean
            predictions = np.clip(predictions, 0.0, None)
        return predictions.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._validate_fitted()
        self._validate_X(X)
        X_seq = self._reshape_to_sequences(X)
        if not self.use_revin and self._channel_mean is not None:
            X_seq = (X_seq - self._channel_mean) / self._channel_std
        X_t = torch.FloatTensor(X_seq)
        self._model.eval()
        with torch.no_grad():
            predictions = self._model(X_t).numpy()
        if self.output_activation == 'zscore' and not self.use_revin:
            predictions = predictions * self._y_std + self._y_mean
            predictions = np.clip(predictions, 0.0, None)
        if predictions.ndim == 2:
            predictions = predictions[:, 0]
        return predictions.astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return deepcopy({
            "learning_rate": self.learning_rate, "epochs": self.epochs,
            "batch_size": self.batch_size, "sequence_length": self.sequence_length,
            "loss_fn": self.loss_fn, "daily_loss_weight": self.daily_loss_weight,
            "optimiser": self.optimiser, "patience": self.patience,
            "output_activation": self.output_activation,
            "use_revin": self.use_revin, "target_channel": self.target_channel,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"learning_rate", "epochs", "batch_size", "sequence_length",
                 "loss_fn", "daily_loss_weight", "optimiser", "patience",
                 "output_activation", "use_revin", "target_channel"}
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
        logger.info(f"Saved NLinear model to {path}")

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
        # past_window_size absent on pre-v2.37 checkpoints — None means
        # legacy single-window path, which is the right behaviour for
        # those.
        self._past_window_size = data.get("past_window_size")
        if self._seq_len is not None and self._n_channels is not None:
            self._model = self._build_model(self._seq_len, self._n_channels,
                                            n_horizons=self._n_horizons)
            self._model.load_state_dict(data["state_dict"])
            self._model.eval()
        self._is_fitted = True
        logger.info(f"Loaded NLinear model from {path}")
