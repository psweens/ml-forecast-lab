"""
FITS forecasting model backend for ML Forecast Lab.

Implements the FITS (Frequency-domain Interpolation for Time Series) model from:
  "FITS: Modeling Time Series with 10k Parameters" (Xu et al., ICLR 2024)
  https://openreview.net/forum?id=bWcnvZ3qMb

FITS works in the complex frequency domain:
  1. Real FFT of the input window.
  2. Low-pass filter — keep only the lowest k frequency bins (cutoff_ratio).
  3. A single complex linear layer maps low-freq input bins to low-freq
     output bins for the (input + horizon) length.
  4. Inverse RFFT, take the trailing horizon-many samples as the forecast.

The "10k parameters" claim comes from this single complex linear layer
operating on a small frequency cut-off rather than on raw time samples.
Training-data instance-norm (subtract per-window mean) handles distribution
shift.
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
        "PyTorch is not installed. FITSModel will not be functional.",
        ImportWarning,
    )


class _ComplexLinear(nn.Module):
    """A linear layer over complex-valued tensors, implemented as two real linears.

    PyTorch's ``nn.Linear`` doesn't support complex parameters directly on
    every backend, so we keep separate real and imaginary weight matrices and
    apply the standard complex matmul by parts:
        (a+ib) * (W_r + iW_i) = (a*W_r − b*W_i) + i(a*W_i + b*W_r).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.real = nn.Linear(in_features, out_features, bias=False)
        self.imag = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (..., in_features), complex64
        real = x.real
        imag = x.imag
        out_real = self.real(real) - self.imag(imag)
        out_imag = self.real(imag) + self.imag(real)
        return torch.complex(out_real, out_imag)


class _FITSNet(nn.Module):
    """FITS: low-pass complex linear in frequency domain."""

    def __init__(self, seq_len: int, n_channels: int, n_horizons: int,
                 cutoff_ratio: float = 0.25,
                 output_activation: str = 'linear', sigmoid_scale: float = 1.0,
                 use_revin: bool = True, target_channel: int = 0,
                 past_window_size: Optional[int] = None):
        super().__init__()
        self.use_revin = use_revin
        self.revin = (
            _RevIN(n_channels, target_channel=target_channel, affine=True)
            if use_revin else None
        )
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.n_horizons = n_horizons
        self.target_channel = target_channel
        self.past_window_size = past_window_size
        self.total_len = seq_len + n_horizons

        # Number of frequency bins in the rfft output of length seq_len /
        # total_len. Cutoff = how many low-freq bins we keep on each side.
        self.in_freq = seq_len // 2 + 1
        self.out_freq = self.total_len // 2 + 1
        self.cutoff_in = max(1, int(self.in_freq * cutoff_ratio))
        self.cutoff_out = max(1, int(self.out_freq * cutoff_ratio))

        # One complex linear per channel — channel-independent. This keeps
        # the model genuinely tiny (≪ a per-channel real linear over
        # raw time samples) and matches the original paper's setup.
        self.freq_linear = _ComplexLinear(self.cutoff_in, self.cutoff_out)
        self.activation = _build_activation(output_activation, scale=sigmoid_scale)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq_len, n_channels)
        if self.revin is not None:
            x = self.revin.normalize(x, past_window_size=self.past_window_size)
        # Per-window per-channel mean removal — FITS's standard preprocessing,
        # gives the linear layer a stationary input regardless of level.
        # When RevIN is on, this adds a small extra mean removal which is
        # harmless because RevIN already centred the window.
        means = x.mean(dim=1, keepdim=True)
        x_centred = x - means

        # (batch, seq_len, n_channels) → (batch, n_channels, seq_len) for FFT
        x_t = x_centred.permute(0, 2, 1)
        x_freq = torch.fft.rfft(x_t, dim=-1)  # (batch, n_channels, in_freq), complex

        # Low-pass: keep only low-frequency bins.
        x_freq_low = x_freq[..., :self.cutoff_in]
        # Apply complex linear to interpolate to (cutoff_out,) low-freq bins
        # of the longer (seq_len + n_horizons) signal.
        y_freq_low = self.freq_linear(x_freq_low)

        # Pad back up to out_freq bins with zeros for the high-frequency tail.
        pad_shape = list(y_freq_low.shape)
        pad_shape[-1] = self.out_freq - self.cutoff_out
        y_freq = torch.cat([
            y_freq_low,
            torch.zeros(pad_shape, dtype=y_freq_low.dtype, device=y_freq_low.device),
        ], dim=-1)

        # Inverse RFFT back to time domain at the longer length.
        y_t = torch.fft.irfft(y_freq, n=self.total_len, dim=-1)
        # (batch, n_channels, total_len) → (batch, total_len, n_channels)
        y = y_t.permute(0, 2, 1)
        # Re-add the per-window mean (broadcasts over total_len).
        y = y + means

        # Take only the trailing horizon-many samples of the target channel.
        y_target = y[:, -self.n_horizons:, self.target_channel]
        out = y_target  # (batch, n_horizons)
        if self.revin is not None:
            out = self.revin.denormalize(out)
        out = self.activation(out)
        if self.n_horizons == 1:
            return out.squeeze(-1)
        return out


class FITSModel(ForecastModel):
    """
    FITS time-series forecasting model.

    Frequency-domain low-pass complex linear interpolation. ~10k parameters
    in typical configurations, making it the lightest neural backend in the
    catalogue while remaining competitive on long-horizon benchmarks.
    """

    def __init__(
        self,
        cutoff_ratio: float = 0.25,
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
        self.cutoff_ratio = float(cutoff_ratio)
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

        self._model: Optional[_FITSNet] = None
        self._seq_len: Optional[int] = None
        self._n_channels: Optional[int] = None
        self._n_horizons: int = 1
        self._channel_mean: Optional[np.ndarray] = None
        self._channel_std: Optional[np.ndarray] = None
        self._sigmoid_scale: float = 1.0
        self._y_mean: Any = 0.0
        self._y_std: Any = 1.0
        # past_window_size enables PF1 (RevIN past-only stats); set per-fit
        # from kwargs, round-tripped in save/load. None means legacy
        # single-window path.
        self._past_window_size: Optional[int] = None

    @property
    def name(self) -> str:
        return "fits"

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

    def _build_model(self, seq_len: int, n_channels: int, n_horizons: int) -> "_FITSNet":
        _eff = ('linear' if (self.use_revin and self.output_activation == 'zscore')
                else self.output_activation)
        return _FITSNet(
            seq_len, n_channels, n_horizons,
            cutoff_ratio=self.cutoff_ratio,
            output_activation=_eff,
            sigmoid_scale=self._sigmoid_scale,
            use_revin=self.use_revin,
            target_channel=self.target_channel,
            past_window_size=self._past_window_size,
        )

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
        X_seq = sequence_data if sequence_data is not None else self._reshape_to_sequences(X_train)
        _, seq_len, n_channels = X_seq.shape
        self._seq_len = seq_len
        self._n_channels = n_channels
        self._past_window_size = kwargs.get("past_window_size")

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

        self._model = self._build_model(seq_len, n_channels, self._n_horizons)
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
            "cutoff_ratio": self.cutoff_ratio,
            "learning_rate": self.learning_rate, "epochs": self.epochs,
            "batch_size": self.batch_size, "sequence_length": self.sequence_length,
            "loss_fn": self.loss_fn, "daily_loss_weight": self.daily_loss_weight,
            "optimiser": self.optimiser, "patience": self.patience,
            "output_activation": self.output_activation,
            "use_revin": self.use_revin, "target_channel": self.target_channel,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"cutoff_ratio", "learning_rate", "epochs", "batch_size",
                 "sequence_length", "loss_fn", "daily_loss_weight",
                 "optimiser", "patience", "output_activation",
                 "use_revin", "target_channel"}
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
        logger.info(f"Saved FITS model to {path}")

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
                                            self._n_horizons)
            self._model.load_state_dict(data["state_dict"])
            self._model.eval()
        self._is_fitted = True
        logger.info(f"Loaded FITS model from {path}")
