"""
TimeMixer forecasting model backend for ML Forecast Lab.

Implements TimeMixer from:
  "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting"
  (Wang et al., ICLR 2024) https://openreview.net/forum?id=7oLshfEIC2

TimeMixer downsamples the input to multiple temporal scales, applies
season/trend decomposition at each scale, mixes information across scales
(Past-Decomposable-Mixing block), and uses a Future-Multipredictor that
projects each scale's representation to the forecast horizon and sums them.

This is a simplified, single-PDM-block implementation tailored to the
existing codebase conventions (RevIN, multi-horizon dense head, optional
covariates), keeping the core multiscale-mixing idea while avoiding the
parameter count of the published reference encoder stack.
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
        "PyTorch is not installed. TimeMixerModel will not be functional.",
        ImportWarning,
    )


def _moving_avg_decompose(x: "torch.Tensor", kernel_size: int):
    """Series decomposition into trend (moving avg) and season (residual).

    x: (batch, seq, channels). Returns (season, trend), same shape.
    """
    pad = kernel_size // 2
    # Channel-first for AvgPool1d.
    x_t = x.permute(0, 2, 1)
    trend_t = F.avg_pool1d(x_t, kernel_size=kernel_size, stride=1, padding=pad,
                           count_include_pad=False)
    trend_t = trend_t[..., :x_t.size(-1)]  # ensure same length
    trend = trend_t.permute(0, 2, 1)
    season = x - trend
    return season, trend


class _PDMBlock(nn.Module):
    """Past-Decomposable-Mixing block.

    Operates on a list of multi-scale representations [s_0, s_1, ..., s_{L-1}]
    where s_0 is the finest scale and s_{L-1} is the coarsest. For each scale,
    decomposes into season + trend and mixes:
      - season: bottom-up (fine → coarse) via a Linear that maps to the next
        scale's length.
      - trend: top-down (coarse → fine) via a Linear that maps from the
        coarser scale's length.
    Each scale is updated with the mixed components plus a residual.
    """

    def __init__(self, scale_lens: List[int], n_channels: int,
                 hidden_mult: int = 2, kernel_size: int = 25,
                 dropout: float = 0.1):
        super().__init__()
        self.scale_lens = scale_lens
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout(dropout)
        # season_linears[i] maps from scale_lens[i] to scale_lens[i+1]
        # trend_linears[i] maps from scale_lens[i+1] to scale_lens[i]
        self.season_linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(scale_lens[i], scale_lens[i] * hidden_mult),
                nn.GELU(),
                nn.Linear(scale_lens[i] * hidden_mult, scale_lens[i + 1]),
            )
            for i in range(len(scale_lens) - 1)
        ])
        self.trend_linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(scale_lens[i + 1], scale_lens[i + 1] * hidden_mult),
                nn.GELU(),
                nn.Linear(scale_lens[i + 1] * hidden_mult, scale_lens[i]),
            )
            for i in range(len(scale_lens) - 1)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(n_channels) for _ in scale_lens
        ])

    def forward(self, scales: List["torch.Tensor"]) -> List["torch.Tensor"]:
        # Decompose each scale.
        seasons, trends = [], []
        for s in scales:
            season, trend = _moving_avg_decompose(s, self.kernel_size)
            seasons.append(season)
            trends.append(trend)

        # Bottom-up season mixing: season[i+1] += linear(season[i])
        for i in range(len(scales) - 1):
            # (batch, seq_i, channels) → (batch, channels, seq_i)
            s_in = seasons[i].permute(0, 2, 1)
            s_out = self.season_linears[i](s_in)
            s_out = s_out.permute(0, 2, 1)
            seasons[i + 1] = seasons[i + 1] + s_out

        # Top-down trend mixing: trend[i] += linear(trend[i+1])
        for i in reversed(range(len(scales) - 1)):
            t_in = trends[i + 1].permute(0, 2, 1)
            t_out = self.trend_linears[i](t_in)
            t_out = t_out.permute(0, 2, 1)
            trends[i] = trends[i] + t_out

        # Recompose with residual + layer norm.
        out_scales = []
        for i, s in enumerate(scales):
            mixed = seasons[i] + trends[i]
            out_scales.append(self.layer_norms[i](s + self.dropout(mixed)))
        return out_scales


class _TimeMixerNet(nn.Module):
    """TimeMixer: PDM block + per-scale linear head (Future-Multipredictor)."""

    def __init__(self, seq_len: int, n_channels: int, n_horizons: int,
                 n_scales: int = 3, downsample: int = 2,
                 hidden_mult: int = 2, kernel_size: int = 25,
                 dropout: float = 0.1,
                 n_pdm_blocks: int = 1,
                 output_activation: str = 'linear',
                 sigmoid_scale: float = 1.0,
                 use_revin: bool = True, target_channel: int = 0,
                 past_window_size: Optional[int] = None):
        super().__init__()
        self.use_revin = use_revin
        self.revin = (
            _RevIN(n_channels, target_channel=target_channel, affine=True)
            if use_revin else None
        )
        self.n_scales = max(1, n_scales)
        self.downsample = max(2, downsample)
        self.target_channel = target_channel
        self.n_horizons = n_horizons
        self.past_window_size = past_window_size

        # Compute lengths at each scale, dropping any scale that would be
        # empty for the given sequence length.
        scale_lens = [max(1, seq_len // (self.downsample ** i))
                      for i in range(self.n_scales)]
        # Trim trailing identical lengths (happens when seq_len is small).
        unique_lens = []
        for L in scale_lens:
            if not unique_lens or unique_lens[-1] != L:
                unique_lens.append(L)
        self.scale_lens = unique_lens
        self.actual_n_scales = len(self.scale_lens)

        if self.actual_n_scales > 1:
            self.pdm_blocks = nn.ModuleList([
                _PDMBlock(self.scale_lens, n_channels,
                          hidden_mult=hidden_mult,
                          kernel_size=kernel_size, dropout=dropout)
                for _ in range(n_pdm_blocks)
            ])
        else:
            self.pdm_blocks = nn.ModuleList()

        # Future-Multipredictor: per-scale linear that flattens
        # (scale_len * n_channels) → n_horizons. Final forecast is the sum.
        self.predictors = nn.ModuleList([
            nn.Linear(L * n_channels, n_horizons) for L in self.scale_lens
        ])
        self.activation = _build_activation(output_activation, scale=sigmoid_scale)

    def _downsample_scales(self, x: "torch.Tensor") -> List["torch.Tensor"]:
        """x: (batch, seq, channels) → list of (batch, scale_len, channels)."""
        scales = [x]
        x_t = x.permute(0, 2, 1)
        cur_len = x.size(1)
        for L in self.scale_lens[1:]:
            stride = max(1, cur_len // L)
            pooled = F.avg_pool1d(x_t, kernel_size=stride, stride=stride)
            # Trim/pad to exactly L.
            if pooled.size(-1) > L:
                pooled = pooled[..., :L]
            elif pooled.size(-1) < L:
                pad = L - pooled.size(-1)
                pooled = F.pad(pooled, (0, pad))
            scales.append(pooled.permute(0, 2, 1))
        return scales

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq, channels)
        if self.revin is not None:
            x = self.revin.normalize(x, past_window_size=self.past_window_size)

        scales = self._downsample_scales(x)
        for block in self.pdm_blocks:
            scales = block(scales)

        # Sum per-scale predictions.
        out = None
        for predictor, s in zip(self.predictors, scales):
            flat = s.reshape(s.size(0), -1)
            pred = predictor(flat)  # (batch, n_horizons)
            out = pred if out is None else out + pred
        # Average rather than sum so the output magnitude is independent of
        # the number of scales (sum would blow up gradients early in training).
        out = out / float(len(scales))

        if self.revin is not None:
            out = self.revin.denormalize(out)
        out = self.activation(out)
        if self.n_horizons == 1:
            return out.squeeze(-1)
        return out


class TimeMixerModel(ForecastModel):
    """
    TimeMixer time-series forecasting model.

    Multiscale season/trend decomposition with cross-scale mixing
    (Past-Decomposable-Mixing) followed by a per-scale linear head whose
    outputs are averaged to produce the forecast.
    """

    def __init__(
        self,
        n_scales: int = 3,
        downsample: int = 2,
        hidden_mult: int = 2,
        decomp_kernel: int = 25,
        n_pdm_blocks: int = 1,
        dropout: float = 0.1,
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
        self.n_scales = n_scales
        self.downsample = downsample
        self.hidden_mult = hidden_mult
        self.decomp_kernel = decomp_kernel
        self.n_pdm_blocks = n_pdm_blocks
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

        self._model: Optional[_TimeMixerNet] = None
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
        return "timemixer"

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
                     n_horizons: int) -> "_TimeMixerNet":
        _eff = ('linear' if (self.use_revin and self.output_activation == 'zscore')
                else self.output_activation)
        return _TimeMixerNet(
            seq_len, n_channels, n_horizons,
            n_scales=self.n_scales,
            downsample=self.downsample,
            hidden_mult=self.hidden_mult,
            kernel_size=self.decomp_kernel,
            dropout=self.dropout,
            n_pdm_blocks=self.n_pdm_blocks,
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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(self._model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
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
            "n_scales": self.n_scales, "downsample": self.downsample,
            "hidden_mult": self.hidden_mult,
            "decomp_kernel": self.decomp_kernel,
            "n_pdm_blocks": self.n_pdm_blocks, "dropout": self.dropout,
            "learning_rate": self.learning_rate, "epochs": self.epochs,
            "batch_size": self.batch_size, "sequence_length": self.sequence_length,
            "loss_fn": self.loss_fn, "daily_loss_weight": self.daily_loss_weight,
            "optimiser": self.optimiser, "patience": self.patience,
            "output_activation": self.output_activation,
            "use_revin": self.use_revin, "target_channel": self.target_channel,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"n_scales", "downsample", "hidden_mult", "decomp_kernel",
                 "n_pdm_blocks", "dropout", "learning_rate", "epochs",
                 "batch_size", "sequence_length", "loss_fn",
                 "daily_loss_weight", "optimiser", "patience",
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
        logger.info(f"Saved TimeMixer model to {path}")

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
        logger.info(f"Loaded TimeMixer model from {path}")
