"""
CycleNet forecasting model backend for ML Forecast Lab.

Implements the Residual Cycle Forecasting architecture from:
  "CycleNet: Enhancing Time Series Forecasting through Modeling Periodic
  Patterns" (Lin et al., NeurIPS 2024 Spotlight,
  https://arxiv.org/abs/2409.18479)

RCF maintains a LEARNABLE cycle buffer ``Q`` of shape
``(cycle_len, n_channels)`` — the explicit periodic component of the
series, trained jointly with the forecaster. Each window subtracts the
cycle values aligned to its absolute position within the cycle, a
linear (or small MLP) head forecasts only the residual, and the cycle's
future segment is added back. For sensors dominated by a daily rhythm
(power, solar, temperature) this hands the model the periodic structure
outright instead of making it rediscover the cycle from lags.

Phase alignment: unlike every other neural backend, RCF needs the
ABSOLUTE position of each window inside the cycle — the same window
content at 06:00 and at 18:00 must read different rows of ``Q``. The
pipeline supplies this as ``window_step_index`` (epoch-anchored
grid-step numbers from ``features.grid_step_index``) through ``fit``
kwargs and ``predict_sequence``; the class attribute
``needs_window_step_index`` is what makes call sites pass it (via
``models.base.predict_sequence_with_context``). When the indices are
absent (harnesses fitting and predicting on the same synthetic
enumeration), the backend falls back to relative stride-1 indexing —
consistent within one fit/predict pair, but not across a cache reload,
which is why the fallback on a model fitted with real indices logs a
warning.

House adaptations (documented deviations from the reference code):
- The paper's per-channel output is replaced by the house single-target
  head: the residual window is flattened (PF7-style — future-block
  target slots excluded) into a linear or MLP projection to the horizon,
  and only the target channel's future cycle segment is added back.
- ``cycle_index`` is derived from real timestamps rather than dataset
  row positions — the upgrade the reference README itself recommends —
  so retention trimming and fold bridging cannot shift the phase.
- The house training kit shared by every neural backend applies: RevIN
  with past-only stats, AdamW + cosine annealing, composite-horizon
  loss, best-model checkpointing.
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
        "PyTorch is not installed. CycleNetModel will not be functional.",
        ImportWarning,
    )


class _RecurrentCycle(nn.Module):
    """The paper's learnable cycle buffer with modulo gather."""

    def __init__(self, cycle_len: int, n_channels: int):
        super().__init__()
        self.cycle_len = cycle_len
        # Zero-init (paper): the cycle starts as "no periodic component"
        # and is learned jointly with the residual forecaster.
        self.data = nn.Parameter(torch.zeros(cycle_len, n_channels))

    def forward(self, index: "torch.Tensor", length: int) -> "torch.Tensor":
        # index: (batch,) int64 — cycle position of each sample's first step
        gather_index = (
            index.view(-1, 1)
            + torch.arange(length, device=index.device).view(1, -1)
        ) % self.cycle_len
        return self.data[gather_index]  # (batch, length, n_channels)


class _CycleNetNet(nn.Module):
    """CycleNet: learnable-cycle removal, residual head, cycle add-back."""

    def __init__(self, seq_len: int, n_channels: int, cycle_len: int,
                 model_type: str, d_hidden: int, n_horizons: int = 1,
                 output_activation: str = 'linear', sigmoid_scale: float = 1.0,
                 use_revin: bool = True, target_channel: int = 0,
                 past_window_size: Optional[int] = None):
        super().__init__()
        self.use_revin = use_revin
        # Reversible instance norm (Kim et al. 2022). The cycle buffer
        # lives in RevIN-normalised space, same order as the reference:
        # norm -> remove cycle -> forecast residual -> add cycle -> denorm.
        self.revin = _RevIN(n_channels, target_channel=target_channel, affine=True) if use_revin else None
        self.n_horizons = n_horizons
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.cycle_len = max(1, int(cycle_len))
        self.target_channel = target_channel

        self.cycle_queue = _RecurrentCycle(self.cycle_len, n_channels)

        # PF7 (v2.37): in extended-window mode the future block's
        # target-channel slot is always zero — exclude it from the head
        # input (same treatment as DLinear).
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

        # Residual forecaster: the paper's two variants.
        if model_type == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(flat, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, n_horizons),
            )
        else:
            self.head = nn.Linear(flat, n_horizons)

        self.activation = _build_activation(output_activation, scale=sigmoid_scale)

    def _head_flatten(self, x: "torch.Tensor") -> "torch.Tensor":
        """Flatten (B, seq_len, n_channels) → (B, flat) honouring PF7."""
        if not self._has_future:
            return x.reshape(x.shape[0], -1)
        past = x[:, : self.past_window_size, :]
        future = x[:, self.past_window_size :, :]
        keep = [c for c in range(self.n_channels) if c != self.target_channel]
        future_kept = future[:, :, keep]
        return torch.cat(
            [past.reshape(past.shape[0], -1),
             future_kept.reshape(future_kept.shape[0], -1)],
            dim=1,
        )

    def forward(self, x, step_index):
        # x: (batch, seq_len, n_channels); step_index: (batch,) int64 —
        # epoch-anchored grid-step number of each window's first row.
        if self.revin is not None:
            x = self.revin.normalize(
                x,
                past_window_size=self.past_window_size if self._has_future else None,
            )

        cycle_index = torch.remainder(step_index, self.cycle_len)

        # Remove the cycle aligned to the window's absolute phase. The
        # extended window is contiguous in time, so one gather covers the
        # past block and the future-known block alike; future target
        # slots end up excluded by the PF7 flatten either way.
        residual = x - self.cycle_queue(cycle_index, self.seq_len)

        out = self.head(self._head_flatten(residual))  # (batch, n_horizons)

        # Add back the target channel's cycle for the horizon steps:
        # horizon h sits at step_index + past_window_size - 1 + h.
        future_index = torch.remainder(
            step_index + self.past_window_size, self.cycle_len,
        )
        future_cycle = self.cycle_queue(future_index, self.n_horizons)
        out = out + future_cycle[:, :, self.target_channel]

        if self.revin is not None:
            # Denormalise in z-space before the output activation so the
            # activation operates on physical-scale values (matters for
            # softplus / sigmoid / exp whose range constraints are only
            # meaningful in target space).
            out = self.revin.denormalize(out)
        out = self.activation(out)

        if self.n_horizons == 1:
            return out.squeeze(-1)  # (batch,) backward compat
        return out


class CycleNetModel(ForecastModel):
    """
    CycleNet time-series forecasting model.

    Learns the series' periodic component explicitly (a trainable cycle
    buffer) and forecasts only the residual with a linear or small MLP
    head. Supports multi-horizon output. ``cycle_len`` is a data-cadence
    property — steps per dominant cycle, e.g. 48 for a daily cycle at
    30-min sampling — not a hyperparameter to search over.
    """

    # Read by predict_sequence_with_context: this backend needs the
    # absolute grid position of every window it fits or predicts on.
    needs_window_step_index = True

    def __init__(
        self,
        cycle_len: int = 48,
        model_type: str = 'linear',
        d_hidden: int = 64,
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

        self.cycle_len = cycle_len
        self.model_type = model_type
        self.d_hidden = d_hidden
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

        self._model: Optional[_CycleNetNet] = None
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
        # True when fit received real epoch-anchored step indices; a
        # predict fallback to relative indexing on such a model would be
        # phase-misaligned, so it warns. Round-tripped in save/load.
        self._fit_had_step_index: bool = False

    @property
    def name(self) -> str:
        return "cyclenet"

    @property
    def is_neural(self) -> bool:
        return True

    def _reshape_to_sequences(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        seq_len = self.sequence_length or n_features
        n_features_per_step = n_features // seq_len if n_features % seq_len == 0 else 1
        return X[:, :seq_len * n_features_per_step].reshape(n_samples, seq_len, n_features_per_step)

    def _build_model(self, seq_len: int, n_channels: int,
                     n_horizons: int = 1) -> "_CycleNetNet":
        # When RevIN is on, the network owns the scale — treat any 'zscore'
        # activation request as 'linear' inside the forward path because
        # zscore's identity head is what RevIN expects anyway.
        _effective_activation = (
            'linear' if (self.use_revin and self.output_activation == 'zscore')
            else self.output_activation
        )
        return _CycleNetNet(
            seq_len, n_channels, self.cycle_len, self.model_type,
            self.d_hidden,
            n_horizons=n_horizons,
            output_activation=_effective_activation,
            sigmoid_scale=self._sigmoid_scale,
            use_revin=self.use_revin,
            target_channel=self.target_channel,
            past_window_size=self._past_window_size,
        )

    def _resolve_step_index(self, kwargs_value, n_samples: int,
                            during_fit: bool) -> np.ndarray:
        """Validate provided step indices or fall back to relative ones."""
        if kwargs_value is not None:
            steps = np.asarray(kwargs_value, dtype=np.int64)
            if len(steps) != n_samples:
                raise ValueError(
                    f"window_step_index has {len(steps)} entries for "
                    f"{n_samples} windows"
                )
            if during_fit:
                self._fit_had_step_index = True
            return steps
        # Relative stride-1 fallback: consistent only within harnesses
        # that fit and predict on the same synthetic enumeration.
        if not during_fit and self._fit_had_step_index:
            logger.warning(
                "cyclenet: predict_sequence called without "
                "window_step_index on a model fitted with real step "
                "indices — cycle phase is misaligned. Pass indices via "
                "predict_sequence_with_context."
            )
        if during_fit:
            self._fit_had_step_index = False
        return np.arange(n_samples, dtype=np.int64)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        """Train CycleNet with PyTorch autograd and best-model checkpointing."""
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

        step_index = self._resolve_step_index(
            kwargs.get("window_step_index"), len(X_seq), during_fit=True,
        )

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
        s_tr, s_val = step_index[train_mask], step_index[val_mask]
        w_tr = sample_weight[train_mask] if sample_weight is not None else None

        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_t = torch.FloatTensor(y_tr)
        s_tr_t = torch.LongTensor(s_tr)
        w_tr_t = torch.FloatTensor(w_tr) if w_tr is not None else None
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        s_val_t = torch.LongTensor(s_val)

        self._model = self._build_model(seq_len, n_channels,
                                        n_horizons=self._n_horizons)
        optimiser = self._build_optimiser(
            self._model.parameters(), self.optimiser, self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.epochs, eta_min=1e-6)
        _loss_map = {'mse': nn.MSELoss, 'mae': nn.L1Loss, 'l1': nn.L1Loss, 'huber': nn.SmoothL1Loss}
        criterion = _loss_map.get(self.loss_fn, nn.MSELoss)(reduction='none')

        best_val_loss = float("inf")
        # best_val_loss tracks raw val_loss (for the checkpoint);
        # best_val_loss_smoothed + val_loss_ema drive the stop decision
        # via _step_early_stop (min_delta + EMA).
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
                s_batch = s_tr_t[batch_idx]

                optimiser.zero_grad()
                y_pred = self._model(X_batch, s_batch)
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
                val_pred = self._model(X_val_t, s_val_t)
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

            # Best-model checkpoint + early stopping (shared helper applies
            # min_delta + EMA-smoothed stop decision).
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

    def predict_sequence(self, X: np.ndarray,
                         window_step_index: Optional[np.ndarray] = None) -> np.ndarray:
        """Multi-horizon prediction from sliding-window input.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, window_size, n_channels)
        window_step_index : np.ndarray of int64, optional
            Epoch-anchored grid-step number of each window's first row
            (``features.grid_step_index``). Callers pass it through
            ``models.base.predict_sequence_with_context``. Absent, the
            backend falls back to relative stride-1 indexing — only
            consistent for harnesses that fitted the same way.

        Returns
        -------
        np.ndarray, shape (n_samples, n_horizons) or (n_samples,) if single-horizon
        """
        self._validate_fitted()
        if self._model is None:
            raise RuntimeError("No model loaded")

        step_index = self._resolve_step_index(
            window_step_index, len(X), during_fit=False,
        )

        X_seq = X.copy()
        # Dataset-level channel normalisation only applies when RevIN is off —
        # otherwise RevIN handles per-window normalisation inside forward().
        if not self.use_revin and self._channel_mean is not None and self._channel_std is not None:
            X_seq = (X_seq - self._channel_mean) / self._channel_std

        X_t = torch.FloatTensor(X_seq)
        s_t = torch.LongTensor(step_index)
        self._model.eval()
        with torch.no_grad():
            predictions = self._model(X_t, s_t).numpy()

        # Denormalise z-space predictions back to physical units. Floor at
        # zero because the linear head in z-space is unconstrained and
        # callers expect physically-valid (non-negative) forecasts. Skipped
        # when use_revin is True: the network already returns target-space
        # predictions.
        if self.output_activation == 'zscore' and not self.use_revin:
            predictions = predictions * self._y_std + self._y_mean
            predictions = np.clip(predictions, 0.0, None)

        return predictions.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions (returns horizon-0 for multi-horizon models)."""
        self._validate_fitted()
        self._validate_X(X)
        X_seq = self._reshape_to_sequences(X)

        step_index = self._resolve_step_index(None, len(X_seq), during_fit=False)

        if not self.use_revin and self._channel_mean is not None and self._channel_std is not None:
            X_seq = (X_seq - self._channel_mean) / self._channel_std
        X_t = torch.FloatTensor(X_seq)
        s_t = torch.LongTensor(step_index)
        self._model.eval()
        with torch.no_grad():
            predictions = self._model(X_t, s_t).numpy()

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
            "cycle_len": self.cycle_len, "model_type": self.model_type,
            "d_hidden": self.d_hidden, "learning_rate": self.learning_rate,
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
        valid = {"cycle_len", "model_type", "d_hidden", "learning_rate",
                 "epochs", "batch_size", "sequence_length", "loss_fn",
                 "daily_loss_weight", "optimiser", "patience", "output_activation",
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
            "seq_len": self._seq_len,
            "n_channels": self._n_channels,
            "n_horizons": self._n_horizons,
            "channel_mean": self._channel_mean,
            "channel_std": self._channel_std,
            "sigmoid_scale": self._sigmoid_scale,
            "y_mean": self._y_mean,
            "y_std": self._y_std,
            "past_window_size": self._past_window_size,
            "fit_had_step_index": self._fit_had_step_index,
        }, path)
        logger.info(f"Saved CycleNet model to {path}")

    def load(self, path: str) -> None:
        """Load model state dict."""
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
        self._fit_had_step_index = bool(data.get("fit_had_step_index", False))

        if self._seq_len is not None and self._n_channels is not None:
            self._model = self._build_model(self._seq_len, self._n_channels,
                                            n_horizons=self._n_horizons)
            self._model.load_state_dict(data["state_dict"])
            self._model.eval()
        self._is_fitted = True
        logger.info(f"Loaded CycleNet model from {path}")
