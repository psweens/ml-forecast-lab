"""
xPatch forecasting model backend for ML Forecast Lab.

Implements the exponential-decomposition dual-stream architecture from:
  "xPatch: Dual-Stream Time Series Forecasting with Exponential
  Seasonal-Trend Decomposition" (Stitsyuk & Choi, AAAI 2025,
  https://arxiv.org/abs/2412.17323)

The input window is decomposed with an exponential moving average — the
EMA is the trend, the residual is the seasonality — and each component
gets its own stream: a patch-based CNN stream for the seasonal part and
a pure-MLP stream for the trend, fused by a final linear layer. The EMA
decomposition is what distinguishes it from the wired `dlinear`
(moving-average kernel): exponential weighting tracks level shifts
faster and is robust on short, noisy histories.

House adaptations (documented deviations from the reference code):
- The paper is channel-independent with a per-channel output. MLFL
  forecasts a single target from a multi-channel window, so the streams
  stay channel-independent but the per-channel horizon outputs are
  combined by a linear channel-mixing head into the target forecast —
  this is also how covariate channels (future-known rows of the
  extended window included) reach the prediction.
- The paper's arctan loss and sigmoid learning-rate schedule are
  replaced by the house training kit shared by every neural backend:
  RevIN with past-only stats, AdamW + cosine annealing,
  composite-horizon loss, best-model checkpointing.
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
        "PyTorch is not installed. XPatchModel will not be functional.",
        ImportWarning,
    )


class _XPatchNet(nn.Module):
    """xPatch: EMA decomposition, CNN seasonal stream, MLP trend stream."""

    def __init__(self, seq_len: int, n_channels: int, patch_len: int,
                 stride: int, ema_alpha: float, n_horizons: int = 1,
                 output_activation: str = 'linear', sigmoid_scale: float = 1.0,
                 use_revin: bool = True, target_channel: int = 0,
                 past_window_size: Optional[int] = None):
        super().__init__()
        self.use_revin = use_revin
        # Reversible instance norm (Kim et al. 2022). Handles distribution
        # shift per-window — replaces the need for dataset-level channel
        # z-scoring on non-stationary series.
        self.revin = _RevIN(n_channels, target_channel=target_channel, affine=True) if use_revin else None
        self.n_horizons = n_horizons
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.past_window_size = past_window_size
        # alpha at the boundaries degenerates the EMA (0: divide-by-zero
        # weights, 1: trend == input) — keep it strictly inside (0, 1).
        self.ema_alpha = min(max(float(ema_alpha), 0.01), 0.99)

        # Clamp patching so at least one full patch fits
        patch_len = max(1, min(patch_len, seq_len))
        stride = max(1, min(stride, patch_len))
        self.patch_len = patch_len
        self.stride = stride
        # End-replication padding of one stride, then unfold — computed on
        # the padded length so the layer sizes match the unfold output for
        # any (seq_len, patch_len, stride), not just the divisible case.
        self.patch_num = (seq_len + stride - patch_len) // stride + 1
        self.pad = nn.ReplicationPad1d((0, stride))

        H = max(1, n_horizons)
        dim = patch_len * patch_len

        # Seasonal (non-linear) stream: patch embedding, depthwise conv,
        # pointwise conv, flatten head.
        self.fc1 = nn.Linear(patch_len, dim)
        self.bn1 = nn.BatchNorm1d(self.patch_num)
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               kernel_size=patch_len, stride=patch_len,
                               groups=self.patch_num)
        self.bn2 = nn.BatchNorm1d(self.patch_num)
        self.fc2 = nn.Linear(dim, patch_len)
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.bn3 = nn.BatchNorm1d(self.patch_num)
        self.fc3 = nn.Linear(self.patch_num * patch_len, H * 2)
        self.fc4 = nn.Linear(H * 2, H)

        # Trend (linear) stream: MLP with average-pool bottlenecks. The
        # second pool halves H, which is meaningless at H == 1 — skip it
        # there and let the LayerNorm/fc7 operate on the un-pooled size.
        self.fc5 = nn.Linear(seq_len, H * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(H * 2)
        self.fc6 = nn.Linear(H * 2, H)
        self._trend_mid = H // 2 if H >= 2 else 1
        self.avgpool2 = nn.AvgPool1d(kernel_size=2) if H >= 2 else None
        self.ln2 = nn.LayerNorm(self._trend_mid)
        self.fc7 = nn.Linear(self._trend_mid, H)

        # Stream fusion, then the single-target channel-mixing head (house
        # adaptation, see module docstring).
        self.fc8 = nn.Linear(H * 2, H)
        self.channel_mix = nn.Linear(n_channels, 1)

        self.gelu = nn.GELU()
        self.activation = _build_activation(output_activation, scale=sigmoid_scale)

    def _ema(self, x: "torch.Tensor") -> "torch.Tensor":
        """Exponential moving average over the time axis.

        Closed-form O(1) version of s_t = α·x_t + (1-α)·s_{t-1} (paper's
        optimised implementation); double precision keeps the (1-α)^t
        weights from underflowing on long windows.
        """
        t = x.size(1)
        powers = torch.flip(
            torch.arange(t, dtype=torch.double, device=x.device), dims=(0,),
        )
        weights = torch.pow(1.0 - self.ema_alpha, powers)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.ema_alpha
        weights = weights.view(1, t, 1)
        divisor = divisor.view(1, t, 1)
        ema = torch.cumsum(x.double() * weights, dim=1) / divisor
        return ema.to(x.dtype)

    def forward(self, x):
        # x: (batch, seq_len, n_channels)
        batch_size = x.size(0)
        if self.revin is not None:
            x = self.revin.normalize(x, past_window_size=self.past_window_size)

        # EMA decomposition: the EMA is the trend, the residual the seasonality
        trend = self._ema(x)
        seasonal = x - trend

        # Channel-independent streams: (batch, seq_len, C) -> (batch*C, seq_len)
        s = seasonal.permute(0, 2, 1).reshape(batch_size * self.n_channels, self.seq_len)
        t = trend.permute(0, 2, 1).reshape(batch_size * self.n_channels, self.seq_len)

        # ---- Seasonal / non-linear stream ----
        s = self.pad(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: (batch*C, patch_num, patch_len)
        s = self.bn1(self.gelu(self.fc1(s)))
        res = s
        s = self.bn2(self.gelu(self.conv1(s)))
        s = s + self.fc2(res)
        s = self.bn3(self.gelu(self.conv2(s)))
        s = s.flatten(start_dim=-2)
        s = self.fc4(self.gelu(self.fc3(s)))  # (batch*C, H)

        # ---- Trend / linear stream ----
        t = self.ln1(self.avgpool1(self.fc5(t)))
        t = self.fc6(t)
        if self.avgpool2 is not None:
            t = self.avgpool2(t)
        t = self.fc7(self.ln2(t))  # (batch*C, H)

        # ---- Stream fusion + channel mixing ----
        y = self.fc8(torch.cat((s, t), dim=1))  # (batch*C, H)
        y = y.view(batch_size, self.n_channels, -1).permute(0, 2, 1)
        y = self.channel_mix(y).squeeze(-1)  # (batch, H)

        if self.revin is not None:
            # Denormalise in z-space before the output activation so the
            # activation operates on physical-scale values (matters for
            # softplus / sigmoid / exp whose range constraints are only
            # meaningful in target space).
            y = self.revin.denormalize(y)
        y = self.activation(y)

        if self.n_horizons == 1:
            return y.squeeze(-1)  # (batch,) backward compat
        return y


class XPatchModel(ForecastModel):
    """
    xPatch time-series forecasting model.

    Exponential-moving-average seasonal-trend decomposition with a
    patch-CNN stream for the seasonal component and an MLP stream for
    the trend, fused linearly. Supports multi-horizon output.
    """

    def __init__(
        self,
        patch_len: int = 8,
        stride: int = 4,
        ema_alpha: float = 0.3,
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
        self.ema_alpha = ema_alpha
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

        self._model: Optional[_XPatchNet] = None
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
        return "xpatch"

    @property
    def is_neural(self) -> bool:
        return True

    def _reshape_to_sequences(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        seq_len = self.sequence_length or n_features
        n_features_per_step = n_features // seq_len if n_features % seq_len == 0 else 1
        return X[:, :seq_len * n_features_per_step].reshape(n_samples, seq_len, n_features_per_step)

    def _build_model(self, seq_len: int, n_channels: int,
                     n_horizons: int = 1) -> "_XPatchNet":
        # When RevIN is on, the network owns the scale — treat any 'zscore'
        # activation request as 'linear' inside the forward path because
        # zscore's identity head is what RevIN expects anyway.
        _effective_activation = (
            'linear' if (self.use_revin and self.output_activation == 'zscore')
            else self.output_activation
        )
        return _XPatchNet(
            seq_len, n_channels, self.patch_len, self.stride, self.ema_alpha,
            n_horizons=n_horizons,
            output_activation=_effective_activation,
            sigmoid_scale=self._sigmoid_scale,
            use_revin=self.use_revin,
            target_channel=self.target_channel,
            past_window_size=self._past_window_size,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        """Train xPatch with PyTorch autograd and best-model checkpointing."""
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

            # Best-model checkpoint + early stopping (shared helper applies
            # min_delta + EMA-smoothed stop decision). The stop-EMA attr is
            # named ema_alpha_stop here — the house `ema_alpha` name is
            # taken by xPatch's decomposition smoothing factor and must not
            # leak into the early-stop smoothing.
            es = self._step_early_stop(
                val_loss, best_val_loss, best_val_loss_smoothed,
                val_loss_ema, patience_counter,
                min_delta=getattr(self, 'min_delta', 1e-3),
                ema_alpha=getattr(self, 'ema_alpha_stop', 0.3),
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
            "ema_alpha": self.ema_alpha, "learning_rate": self.learning_rate,
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
        valid = {"patch_len", "stride", "ema_alpha", "learning_rate",
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
        }, path)
        logger.info(f"Saved xPatch model to {path}")

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

        if self._seq_len is not None and self._n_channels is not None:
            self._model = self._build_model(self._seq_len, self._n_channels,
                                            n_horizons=self._n_horizons)
            self._model.load_state_dict(data["state_dict"])
            self._model.eval()
        self._is_fitted = True
        logger.info(f"Loaded xPatch model from {path}")
