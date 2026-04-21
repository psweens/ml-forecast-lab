"""
Temporal Fusion Transformer (TFT) backend for ML Forecast Lab.

Implements a compact, single-target variant of the TFT from:
  "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series
  Forecasting" (Lim et al., 2021)
  https://arxiv.org/abs/1912.09363

The full reference TFT models known/observed/static covariates separately and
exposes per-variable selection weights for interpretability. This compact
variant keeps the architectural backbone — Variable Selection Network → LSTM
encoder → interpretable multi-head attention → gated residual output — while
treating the entire input tensor as a flat set of observed covariates plus
the target. That keeps it interoperable with the existing feature pipeline
(no covariate type metadata required) at the cost of skipping the static /
known-future split, which the smart-home use case rarely has anyway.

Multi-horizon outputs are produced directly by a per-horizon dense head over
the attention-pooled context vector.
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
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch is not installed. TFTModel will not be functional.",
        ImportWarning,
    )


class _GatedResidualNetwork(nn.Module):
    """TFT's GRN: dense → ELU → dense → GLU → residual + LayerNorm.

    A small but central building block in TFT. The GLU lets the network
    learn to skip the GRN entirely when its transformation isn't useful,
    which is the mechanism behind TFT's ability to suppress unhelpful
    variables.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        out_size = output_size or input_size
        self.input_size = input_size
        self.output_size = out_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size)
        # GLU: linear projection split into [a, b], output = a * sigmoid(b)
        self.gate = nn.Linear(out_size, out_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_size)
        self.skip = (nn.Linear(input_size, out_size)
                     if input_size != out_size else nn.Identity())

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        residual = self.skip(x)
        h = F.elu(self.fc1(x))
        h = self.fc2(h)
        h = self.dropout(h)
        gated = self.gate(h)
        a, b = gated.chunk(2, dim=-1)
        h = a * torch.sigmoid(b)
        return self.layer_norm(h + residual)


class _VariableSelectionNetwork(nn.Module):
    """Per-channel GRN + softmax selection weights, then weighted sum.

    Each channel gets its own GRN that produces a hidden_size embedding.
    The selection network (a separate GRN over the flattened inputs)
    produces softmax weights over channels, used to combine the per-channel
    embeddings into a single hidden_size vector.
    """

    def __init__(self, n_channels: int, seq_len: int, hidden_size: int,
                 dropout: float = 0.1):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        # Per-channel GRN: takes the time series of one channel and produces
        # a (seq_len, hidden_size) embedding.
        self.per_channel_grn = nn.ModuleList([
            _GatedResidualNetwork(1, hidden_size, hidden_size, dropout)
            for _ in range(n_channels)
        ])
        # Selection network: looks at all channels jointly to score them.
        self.selection_grn = _GatedResidualNetwork(
            n_channels, hidden_size, n_channels, dropout,
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq_len, n_channels)
        batch, seq_len, _ = x.shape
        # Per-channel embeddings: (batch, seq_len, n_channels, hidden_size)
        embeds = []
        for c in range(self.n_channels):
            xc = x[..., c:c + 1]  # (batch, seq_len, 1)
            embeds.append(self.per_channel_grn[c](xc))
        embeds = torch.stack(embeds, dim=2)  # (batch, seq_len, n_channels, hidden)

        # Selection weights from raw channel values, per timestep.
        sel = self.selection_grn(x)  # (batch, seq_len, n_channels)
        weights = F.softmax(sel, dim=-1).unsqueeze(-1)  # (batch, seq_len, n_channels, 1)
        # Weighted sum across channels → (batch, seq_len, hidden)
        return (embeds * weights).sum(dim=2)


class _TFTNet(nn.Module):
    """Compact TFT: VSN → LSTM → interpretable multi-head attention → head."""

    def __init__(self, seq_len: int, n_channels: int, n_horizons: int,
                 hidden_size: int = 32, n_heads: int = 4,
                 dropout: float = 0.1, n_lstm_layers: int = 1,
                 output_activation: str = 'linear', sigmoid_scale: float = 1.0,
                 use_revin: bool = True, target_channel: int = 0):
        super().__init__()
        self.use_revin = use_revin
        self.revin = (
            _RevIN(n_channels, target_channel=target_channel, affine=True)
            if use_revin else None
        )
        self.n_horizons = n_horizons

        self.vsn = _VariableSelectionNetwork(n_channels, seq_len,
                                             hidden_size, dropout)
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=n_lstm_layers, batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.post_lstm_grn = _GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout,
        )
        # Multi-head attention over the encoded sequence, queried by the
        # last timestep's representation.
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.position_grn = _GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout,
        )
        # Per-horizon output head — one Linear projecting the context vector
        # to a single scalar per horizon, stacked into (batch, n_horizons).
        self.output = nn.Linear(hidden_size, n_horizons)
        self.activation = _build_activation(output_activation, scale=sigmoid_scale)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq_len, n_channels)
        if self.revin is not None:
            x = self.revin.normalize(x)

        # Variable Selection — collapses channels to a single hidden stream.
        h = self.vsn(x)  # (batch, seq_len, hidden)

        # Temporal encoder.
        h_lstm, _ = self.lstm(h)  # (batch, seq_len, hidden)
        h_enc = self.post_lstm_grn(h_lstm + h)  # residual + GRN

        # Self-attention with the last step as query — pulls relevant history
        # forward into a single context vector.
        query = h_enc[:, -1:, :]  # (batch, 1, hidden)
        attn_out, _ = self.attention(query, h_enc, h_enc)
        attn_out = self.attn_norm(attn_out + query)
        ctx = self.position_grn(attn_out).squeeze(1)  # (batch, hidden)

        out = self.output(ctx)  # (batch, n_horizons)
        if self.revin is not None:
            out = self.revin.denormalize(out)
        out = self.activation(out)
        if self.n_horizons == 1:
            return out.squeeze(-1)
        return out


class TFTModel(ForecastModel):
    """
    Compact Temporal Fusion Transformer for time-series forecasting.

    Variable Selection Network → LSTM encoder → interpretable multi-head
    attention → gated residual output, with RevIN per-window normalisation
    and direct multi-horizon output. The static / known-future covariate
    branches of the reference TFT are not modelled — they're rare in HA
    sensor data and would require a covariate-type schema we don't track.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        n_heads: int = 4,
        n_lstm_layers: int = 1,
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
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_lstm_layers = n_lstm_layers
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

        self._model: Optional[_TFTNet] = None
        self._seq_len: Optional[int] = None
        self._n_channels: Optional[int] = None
        self._n_horizons: int = 1
        self._channel_mean: Optional[np.ndarray] = None
        self._channel_std: Optional[np.ndarray] = None
        self._sigmoid_scale: float = 1.0
        self._y_mean: Any = 0.0
        self._y_std: Any = 1.0

    @property
    def name(self) -> str:
        return "tft"

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
            n_samples, seq_len, n_features_per_step,
        )

    def _build_model(self, seq_len: int, n_channels: int,
                     n_horizons: int) -> "_TFTNet":
        # MultiheadAttention requires n_heads to divide hidden_size; clamp
        # n_heads down silently rather than crash on bad config combos.
        n_heads = self.n_heads
        while n_heads > 1 and (self.hidden_size % n_heads) != 0:
            n_heads -= 1
        _eff = ('linear' if (self.use_revin and self.output_activation == 'zscore')
                else self.output_activation)
        return _TFTNet(
            seq_len, n_channels, n_horizons,
            hidden_size=self.hidden_size,
            n_heads=n_heads,
            n_lstm_layers=self.n_lstm_layers,
            dropout=self.dropout,
            output_activation=_eff,
            sigmoid_scale=self._sigmoid_scale,
            use_revin=self.use_revin,
            target_channel=self.target_channel,
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
            "hidden_size": self.hidden_size, "n_heads": self.n_heads,
            "n_lstm_layers": self.n_lstm_layers, "dropout": self.dropout,
            "learning_rate": self.learning_rate, "epochs": self.epochs,
            "batch_size": self.batch_size, "sequence_length": self.sequence_length,
            "loss_fn": self.loss_fn, "daily_loss_weight": self.daily_loss_weight,
            "optimiser": self.optimiser, "patience": self.patience,
            "output_activation": self.output_activation,
            "use_revin": self.use_revin, "target_channel": self.target_channel,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"hidden_size", "n_heads", "n_lstm_layers", "dropout",
                 "learning_rate", "epochs", "batch_size", "sequence_length",
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
        }, path)
        logger.info(f"Saved TFT model to {path}")

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
        if self._seq_len is not None and self._n_channels is not None:
            self._model = self._build_model(self._seq_len, self._n_channels,
                                            self._n_horizons)
            self._model.load_state_dict(data["state_dict"])
            self._model.eval()
        self._is_fitted = True
        logger.info(f"Loaded TFT model from {path}")
