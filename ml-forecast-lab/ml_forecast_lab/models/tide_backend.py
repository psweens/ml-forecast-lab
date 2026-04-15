"""
PyTorch TiDE forecasting model backend for ML Forecast Lab.

Implements TiDE (Das et al. 2023, https://arxiv.org/abs/2304.08424)
with the paper's full architecture: feature projection for dynamic
covariates, dense MLP encoder, dense MLP decoder, per-horizon
temporal decoder, and a global linear residual from the past window
straight to the forecast. Adds Reversible Instance Normalization
(Kim et al. 2022) for non-stationary series, AdamW + CosineAnnealingLR,
and best-model checkpointing.

The future-covariate path is OPT-IN via the ``future_covariates`` kwarg
to ``fit()`` / ``predict()`` / ``predict_sequence()``. When provided,
the temporal decoder fuses the decoder state with per-horizon known
future values (weather forecast, calendar features, etc.) — this is
the mechanism from the paper that lets TiDE exploit information that
is unambiguously known before each forecast step is issued. When not
provided, TiDE degrades gracefully to a dense encoder-decoder on the
past window alone (backwards-compatible with pre-v2.15.0 behaviour).
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
        "PyTorch is not installed. TiDEModel will not be functional.",
        ImportWarning,
    )


class _ResidualBlock(nn.Module):
    """Residual block: Linear → ReLU → Dropout + skip + LayerNorm.

    The paper's building block for both the encoder stack and the
    temporal decoder. Skip path uses a Linear projection when the input
    and output dimensions differ (so the block can also be used for
    feature projection / bottlenecking).
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        residual = self.skip(x)
        out = self.linear(x)
        out = F.relu(out)
        out = self.dropout(out)
        return self.layer_norm(out + residual)


class _TiDENet(nn.Module):
    """TiDE: feature projection → encoder → decoder → temporal decoder + global residual.

    Shape reference
    ---------------
    - ``x``: ``(batch, seq_len, n_channels)`` — past input window.
    - ``future_covariates``: ``(batch, n_horizons, n_future_covariates)``
      or ``None`` if this TiDE was built without a future path.
    - Output: ``(batch, n_horizons)`` or ``(batch,)`` for single-horizon.
    """

    def __init__(
        self,
        seq_len: int,
        n_channels: int,
        hidden_size: int,
        encoder_layers: int,
        decoder_layers: int,
        dropout: float,
        n_horizons: int = 1,
        output_activation: str = 'linear',
        sigmoid_scale: float = 1.0,
        n_future_covariates: int = 0,
        feature_proj_size: int = 16,
        decoder_output_size: int = 16,
        temporal_hidden: int = 32,
        use_revin: bool = True,
        target_channel: int = 0,
    ):
        super().__init__()
        self.n_horizons = n_horizons
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.n_future_covariates = n_future_covariates
        self.decoder_output_size = decoder_output_size
        self.feature_proj_size = feature_proj_size
        self.use_revin = use_revin

        # RevIN handles per-window instance normalisation. Applied around
        # the whole network so the encoder / decoder / residual all see
        # normalised features and the head outputs denormalised predictions.
        self.revin = _RevIN(n_channels, target_channel=target_channel, affine=True) if use_revin else None

        # Feature projection for future covariates (opt-in). Projects each
        # horizon's future-known feature vector down to ``feature_proj_size``.
        # Shared across horizons (Linear applies on last dim).
        if n_future_covariates > 0:
            self.future_feat_proj = _ResidualBlock(
                n_future_covariates, feature_proj_size, dropout,
            )
        else:
            self.future_feat_proj = None

        # Encoder input = flattened past + flattened projected future covariates.
        past_flat_size = seq_len * n_channels
        future_flat_size = (n_horizons * feature_proj_size) if n_future_covariates > 0 else 0
        self.past_flat_size = past_flat_size
        self.encoder_input_proj = _ResidualBlock(
            past_flat_size + future_flat_size, hidden_size, dropout,
        )
        self.encoder = nn.Sequential(*[
            _ResidualBlock(hidden_size, hidden_size, dropout)
            for _ in range(encoder_layers)
        ])

        # Decoder produces per-horizon representation. In the paper, the
        # decoder head maps ``hidden_size → n_horizons * decoder_output_size``
        # and the tensor is reshaped to ``(batch, n_horizons, decoder_output_size)``.
        self.decoder = nn.Sequential(*[
            _ResidualBlock(hidden_size, hidden_size, dropout)
            for _ in range(decoder_layers)
        ])
        self.decoder_output_proj = nn.Linear(hidden_size, n_horizons * decoder_output_size)

        # Temporal decoder (per-horizon MLP). Fuses decoder state with the
        # projected future covariates for each horizon step independently.
        temporal_input_size = decoder_output_size + (feature_proj_size if n_future_covariates > 0 else 0)
        self.temporal_decoder = _ResidualBlock(temporal_input_size, temporal_hidden, dropout)
        self.temporal_head = nn.Linear(temporal_hidden, 1)

        # Global linear residual from past window straight to forecast —
        # gives the network an easy baseline "predict the recent level" path
        # and leaves the nonlinear branches to learn residuals on top.
        self.global_residual = nn.Linear(past_flat_size, n_horizons)

        self.activation = _build_activation(output_activation, scale=sigmoid_scale)

    def forward(self, x, future_covariates=None):
        # x: (batch, seq_len, n_channels)
        # future_covariates: (batch, n_horizons, n_future_covariates) or None
        if self.revin is not None:
            x = self.revin.normalize(x)

        batch_size = x.size(0)
        past_flat = x.reshape(batch_size, -1)  # (batch, seq_len * n_channels)

        # Project future covariates if the network was built for them AND the
        # caller provided them this pass.
        if self.future_feat_proj is not None and future_covariates is not None:
            # future_covariates: (batch, n_horizons, n_future_covariates)
            fcov_proj = self.future_feat_proj(future_covariates)  # (batch, n_horizons, feature_proj_size)
            fcov_flat = fcov_proj.reshape(batch_size, -1)  # (batch, n_horizons * feature_proj_size)
            encoder_in = torch.cat([past_flat, fcov_flat], dim=-1)
        elif self.future_feat_proj is not None:
            # Model was built with future path but caller didn't pass them —
            # feed zeros at inference so the shape still matches. This keeps
            # the model functional even when the external future-covariate
            # feed is temporarily unavailable.
            fcov_proj = None
            zeros = torch.zeros(
                batch_size, self.n_horizons * self.feature_proj_size,
                device=x.device, dtype=x.dtype,
            )
            encoder_in = torch.cat([past_flat, zeros], dim=-1)
        else:
            fcov_proj = None
            encoder_in = past_flat

        # Encode.
        encoded = self.encoder_input_proj(encoder_in)
        encoded = self.encoder(encoded)

        # Decode to per-horizon representation.
        decoded = self.decoder(encoded)
        decoded_per_h = self.decoder_output_proj(decoded).reshape(
            batch_size, self.n_horizons, self.decoder_output_size,
        )

        # Temporal decoder: per-horizon MLP combining decoder state + future cov.
        if fcov_proj is not None:
            temporal_in = torch.cat([decoded_per_h, fcov_proj], dim=-1)
        else:
            temporal_in = decoded_per_h
        forecast = self.temporal_decoder(temporal_in)  # (batch, n_horizons, temporal_hidden)
        forecast = self.temporal_head(forecast).squeeze(-1)  # (batch, n_horizons)

        # Global linear residual from the past window.
        forecast = forecast + self.global_residual(past_flat)

        # Denormalise in z-space before the output activation so the activation
        # operates on physical-scale values.
        if self.revin is not None:
            forecast = self.revin.denormalize(forecast)

        forecast = self.activation(forecast)

        if self.n_horizons == 1:
            return forecast.squeeze(-1)
        return forecast


class TiDEModel(ForecastModel):
    """
    PyTorch TiDE time-series forecasting model.

    Implements the full architecture from Das et al. 2023 including the
    temporal decoder and the global residual. Supports:
    - Multi-horizon direct forecasting
    - Future-known covariates via ``fit(..., future_covariates=...)`` and
      ``predict(..., future_covariates=...)``
    - RevIN instance normalisation (on by default) for non-stationary series
    - Standard output activations (linear / softplus / relu / exp / sigmoid)

    The future-covariate feature vector should contain ONLY values that are
    genuinely known at forecast issue time: calendar features (hour, day of
    week, day of year), public-holiday flag, and externally-forecast values
    like tomorrow's weather (a Solcast forecast, an Open-Meteo temperature
    forecast). Do NOT include lags or quantities that are derived from the
    future true value.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 2e-4,
        epochs: int = 100,
        batch_size: int = 64,
        sequence_length: Optional[int] = None,
        loss_fn: str = 'mse',
        patience: int = 20,
        output_activation: str = 'linear',
        use_revin: bool = True,
        target_channel: int = 0,
        feature_proj_size: int = 16,
        decoder_output_size: int = 16,
        temporal_hidden: int = 32,
    ) -> None:
        super().__init__()
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed")

        self.hidden_size = hidden_size
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.loss_fn = loss_fn
        self.patience = patience
        self.output_activation = output_activation
        # RevIN (Kim et al. 2022) for per-window distribution shift. Supersedes
        # dataset-level channel normalisation and the zscore output path.
        self.use_revin = use_revin
        self.target_channel = target_channel
        # Future-covariate architecture hyperparameters. Used only when the
        # caller passes future_covariates at fit-time.
        self.feature_proj_size = feature_proj_size
        self.decoder_output_size = decoder_output_size
        self.temporal_hidden = temporal_hidden

        self._model: Optional[_TiDENet] = None
        self._input_size: Optional[int] = None
        self._seq_len: Optional[int] = None
        self._n_horizons: int = 1
        self._n_future_covariates: int = 0
        self._channel_mean: Optional[np.ndarray] = None
        self._channel_std: Optional[np.ndarray] = None
        self._sigmoid_scale: float = 1.0
        # Target z-score stats (only populated when output_activation == 'zscore'
        # AND use_revin is False — otherwise RevIN handles scale per-window).
        self._y_mean: Any = 0.0
        self._y_std: Any = 1.0
        self._training_history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    @property
    def name(self) -> str:
        return "tide"

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

    def _validate_future_covariates(
        self,
        future_covariates: Optional[np.ndarray],
        n_samples: int,
    ) -> Optional[np.ndarray]:
        """Coerce + sanity-check the caller's future_covariates array.

        Expected shape ``(n_samples, n_horizons, n_future_covariates)``.
        Returns None if not provided.
        """
        if future_covariates is None:
            return None
        fcov = np.asarray(future_covariates, dtype=np.float32)
        if fcov.ndim != 3:
            raise ValueError(
                f"future_covariates must be 3D (n_samples, n_horizons, "
                f"n_future_covariates); got shape {fcov.shape}"
            )
        if fcov.shape[0] != n_samples:
            raise ValueError(
                f"future_covariates samples ({fcov.shape[0]}) does not match "
                f"X samples ({n_samples})"
            )
        if fcov.shape[1] != self._n_horizons:
            raise ValueError(
                f"future_covariates horizons ({fcov.shape[1]}) does not match "
                f"target n_horizons ({self._n_horizons})"
            )
        return fcov

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        """Train TiDE with PyTorch autograd and best-model checkpointing.

        Supported kwargs
        ----------------
        future_covariates : np.ndarray, shape (n_samples, n_horizons, n_future_covariates), optional
            Values KNOWN at forecast time for each horizon step. Typical
            contents: hour-of-day, day-of-week, public-holiday flag, externally
            forecast weather (Solcast p50 + p10 + p90, Open-Meteo temperature).
            Do NOT include lags or target-derived features.
        sequence_data : np.ndarray, shape (n_samples, window_size, n_channels), optional
            Pre-windowed input instead of reshaping the flat feature matrix.
        sample_weight : np.ndarray, shape (n_samples,), optional
        validation_split : float, default 0.2
        epoch_callback : callable, optional
        """
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

        # Future covariates (opt-in).
        future_covariates = self._validate_future_covariates(
            kwargs.get("future_covariates"), X_seq.shape[0],
        )
        if future_covariates is not None:
            self._n_future_covariates = int(future_covariates.shape[2])
            logger.info(
                f"TiDE: future-covariate path active "
                f"({self._n_future_covariates} features × {self._n_horizons} horizons)"
            )
        else:
            self._n_future_covariates = 0

        # Dataset-level channel normalisation is mutually exclusive with RevIN.
        if not self.use_revin:
            self._channel_mean = X_seq.mean(axis=(0, 1))  # shape (n_channels,)
            self._channel_std = X_seq.std(axis=(0, 1))     # shape (n_channels,)
            self._channel_std[self._channel_std < 1e-8] = 1.0
            X_seq = (X_seq - self._channel_mean) / self._channel_std

        # Resolve sigmoid scale from training targets (data-driven upper bound).
        if self.output_activation == 'sigmoid':
            self._sigmoid_scale = _resolve_sigmoid_scale(y_train)
        else:
            self._sigmoid_scale = 1.0

        # Target z-score normalisation only applies when RevIN is off.
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
        if future_covariates is not None:
            fcov_tr = future_covariates[train_mask]
            fcov_val = future_covariates[val_mask]
        else:
            fcov_tr = None
            fcov_val = None

        # Convert to tensors
        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_t = torch.FloatTensor(y_tr)
        w_tr_t = torch.FloatTensor(w_tr) if w_tr is not None else None
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        fcov_tr_t = torch.FloatTensor(fcov_tr) if fcov_tr is not None else None
        fcov_val_t = torch.FloatTensor(fcov_val) if fcov_val is not None else None

        # Create model
        _effective_activation = (
            'linear' if (self.use_revin and self.output_activation == 'zscore')
            else self.output_activation
        )
        self._model = _TiDENet(
            seq_len, input_size, self.hidden_size,
            self.encoder_layers, self.decoder_layers, self.dropout,
            n_horizons=self._n_horizons,
            output_activation=_effective_activation,
            sigmoid_scale=self._sigmoid_scale,
            n_future_covariates=self._n_future_covariates,
            feature_proj_size=self.feature_proj_size,
            decoder_output_size=self.decoder_output_size,
            temporal_hidden=self.temporal_hidden,
            use_revin=self.use_revin,
            target_channel=self.target_channel,
        )
        optimiser = torch.optim.AdamW(self._model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
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
                fcov_batch = fcov_tr_t[batch_idx] if fcov_tr_t is not None else None

                optimiser.zero_grad()
                y_pred = self._model(X_batch, future_covariates=fcov_batch)
                loss_per_sample = criterion(y_pred, y_batch)
                if w_tr_t is not None:
                    w_batch = w_tr_t[batch_idx]
                    loss = self._weighted_mean_loss(loss_per_sample, w_batch)
                else:
                    loss = loss_per_sample.mean()
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
                val_pred = self._model(X_val_t, future_covariates=fcov_val_t)
                val_loss = criterion(val_pred, y_val_t).mean().item()

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

    def predict_sequence(self, X: np.ndarray, future_covariates: Optional[np.ndarray] = None) -> np.ndarray:
        """Multi-horizon prediction from sliding-window input.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, window_size, n_channels)
        future_covariates : np.ndarray, shape (n_samples, n_horizons, n_future_covariates), optional
            If the model was trained with future covariates you must pass them
            here too; if you omit, the network falls back to zero-fed future
            features (the degraded but still-callable path).

        Returns
        -------
        np.ndarray, shape (n_samples, n_horizons) or (n_samples,) for single-horizon
        """
        self._validate_fitted()
        if self._model is None:
            raise RuntimeError("No model loaded")

        X_seq = X.copy()
        if not self.use_revin and self._channel_mean is not None and self._channel_std is not None:
            X_seq = (X_seq - self._channel_mean) / self._channel_std

        fcov = self._validate_future_covariates(future_covariates, X_seq.shape[0])

        X_t = torch.FloatTensor(X_seq)
        fcov_t = torch.FloatTensor(fcov) if fcov is not None else None
        self._model.eval()
        with torch.no_grad():
            predictions = self._model(X_t, future_covariates=fcov_t).numpy()

        if self.output_activation == 'zscore' and not self.use_revin:
            predictions = predictions * self._y_std + self._y_mean
            predictions = np.clip(predictions, 0.0, None)

        return predictions.astype(np.float32)

    def predict(self, X: np.ndarray, future_covariates: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate predictions (returns horizon-0 for multi-horizon models)."""
        self._validate_fitted()
        self._validate_X(X)

        X_seq = self._reshape_to_sequences(X)

        if not self.use_revin and self._channel_mean is not None and self._channel_std is not None:
            X_seq = (X_seq - self._channel_mean) / self._channel_std

        fcov = self._validate_future_covariates(future_covariates, X_seq.shape[0])

        X_t = torch.FloatTensor(X_seq)
        fcov_t = torch.FloatTensor(fcov) if fcov is not None else None

        self._model.eval()
        with torch.no_grad():
            predictions = self._model(X_t, future_covariates=fcov_t).numpy()

        if self.output_activation == 'zscore' and not self.use_revin:
            predictions = predictions * self._y_std + self._y_mean
            predictions = np.clip(predictions, 0.0, None)

        # Multi-horizon: return only first horizon for backward compat
        if predictions.ndim == 2:
            predictions = predictions[:, 0]

        return predictions.astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return deepcopy({
            "hidden_size": self.hidden_size, "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers, "dropout": self.dropout,
            "learning_rate": self.learning_rate, "epochs": self.epochs,
            "batch_size": self.batch_size, "sequence_length": self.sequence_length,
            "loss_fn": self.loss_fn,
            "patience": self.patience,
            "output_activation": self.output_activation,
            "use_revin": self.use_revin,
            "target_channel": self.target_channel,
            "feature_proj_size": self.feature_proj_size,
            "decoder_output_size": self.decoder_output_size,
            "temporal_hidden": self.temporal_hidden,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"hidden_size", "encoder_layers", "decoder_layers", "dropout",
                 "learning_rate", "epochs", "batch_size", "sequence_length", "loss_fn",
                 "patience", "output_activation",
                 "use_revin", "target_channel",
                 "feature_proj_size", "decoder_output_size", "temporal_hidden"}
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
            "n_future_covariates": self._n_future_covariates,
            "channel_mean": self._channel_mean,
            "channel_std": self._channel_std,
            "sigmoid_scale": self._sigmoid_scale,
            "y_mean": self._y_mean,
            "y_std": self._y_std,
        }, path)
        logger.info(f"Saved TiDE model to {path}")

    def load(self, path: str) -> None:
        """Load model state dict."""
        data = torch.load(path, map_location="cpu")
        self.set_params(**data["params"])
        self._channel_mean = data.get("channel_mean")
        self._channel_std = data.get("channel_std")

        self._n_horizons = data.get("n_horizons", 1)
        self._n_future_covariates = int(data.get("n_future_covariates", 0))
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
            self._model = _TiDENet(
                self._seq_len, self._input_size, self.hidden_size,
                self.encoder_layers, self.decoder_layers, self.dropout,
                n_horizons=self._n_horizons,
                output_activation=_effective_activation,
                sigmoid_scale=self._sigmoid_scale,
                n_future_covariates=self._n_future_covariates,
                feature_proj_size=self.feature_proj_size,
                decoder_output_size=self.decoder_output_size,
                temporal_hidden=self.temporal_hidden,
                use_revin=self.use_revin,
                target_channel=self.target_channel,
            )
            self._model.load_state_dict(state_dict)
            self._model.eval()

        self._is_fitted = True
        logger.info(f"Loaded TiDE model from {path}")
