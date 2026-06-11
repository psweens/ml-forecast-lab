"""
PyTorch TimeXer forecasting model backend for ML Forecast Lab.

Implements the TimeXer architecture (Wang et al. 2024, NeurIPS,
https://arxiv.org/abs/2402.19072): the target (endogenous) series is
embedded as patch tokens plus a learnable global token, every covariate
(exogenous) channel is embedded as a single variate token, and each
encoder layer runs self-attention over the endogenous tokens followed by
cross-attention from the global token to the exogenous variate tokens.
This makes it the only transformer in the catalogue designed explicitly
around exogenous variables — a natural fit for covariate-driven HA
targets (solar ~ irradiance, heating ~ outside temperature).

Uses AdamW optimisation, CosineAnnealingLR scheduling, and best-model
checkpointing. Supports multi-horizon output via a shared encoder and
multi-output head.
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
        "PyTorch is not installed. TimeXerModel will not be functional.",
        ImportWarning,
    )


class _TimeXerNet(nn.Module):
    """TimeXer: endogenous patch tokens + global token with cross-attention
    to exogenous variate tokens (Wang et al. 2024)."""

    def __init__(self, seq_len: int, n_channels: int, patch_len: int,
                 d_model: int, n_heads: int, n_encoder_layers: int,
                 dim_feedforward: int, dropout: float, n_horizons: int = 1,
                 output_activation: str = 'linear', sigmoid_scale: float = 1.0,
                 use_revin: bool = True, target_channel: int = 0,
                 past_window_size: Optional[int] = None):
        super().__init__()
        self.use_revin = use_revin
        self.revin = _RevIN(n_channels, target_channel=target_channel, affine=True) if use_revin else None
        self.n_horizons = n_horizons
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.d_model = d_model
        self.target_channel = target_channel
        self.past_window_size = past_window_size
        self.n_encoder_layers = n_encoder_layers

        # PF6-style slicing (same rationale as iTransformer): both the
        # endogenous patch embedder and the exogenous variate embedder
        # consume the PAST slice only when past_window_size < seq_len.
        # The target channel's future positions are zero placeholders in
        # extended-window mode, so patching across them would feed the
        # self-attention zero-target tokens with no signal.
        embed_len = (past_window_size
                     if past_window_size is not None and past_window_size < seq_len
                     else seq_len)
        self.embed_len = embed_len
        # Width of the future block (0 in legacy non-extended mode). Used
        # by the future_aux_head to consume user future covariates that
        # the past-only embedders ignore by design.
        self.future_window_size = (
            seq_len - past_window_size
            if past_window_size is not None and past_window_size < seq_len
            else 0
        )

        # Endogenous patching: non-overlapping patches per the paper.
        # Clamp so at least one full patch fits in the embed slice.
        if patch_len > embed_len:
            patch_len = max(1, embed_len)
        self.patch_len = patch_len
        self.n_patches = embed_len // patch_len

        # Patch embedding for the endogenous (target) series.
        self.patch_embed = nn.Linear(patch_len, d_model)

        # Learnable global token — the bridge between the endogenous
        # patches and the exogenous variate tokens.
        self.glb_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional embedding over patch tokens + global token.
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches + 1, d_model) * 0.02
        )

        # Exogenous variate embedding: each non-target channel's past
        # slice becomes one token (iTransformer-style series embedding).
        self.n_exog = n_channels - 1
        if self.n_exog > 0:
            self.exog_embed = nn.Linear(embed_len, d_model)
            self.exog_pos_embed = nn.Parameter(
                torch.randn(1, self.n_exog, d_model) * 0.02
            )

        # Per-layer modules. Following the reference implementation, each
        # layer is: endogenous self-attention -> global-token
        # cross-attention to exogenous tokens -> position-wise FFN, each
        # with residual + LayerNorm.
        self.self_attns = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.norms3 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(n_encoder_layers):
            self.self_attns.append(nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True,
            ))
            self.cross_attns.append(nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True,
            ))
            self.norms1.append(nn.LayerNorm(d_model))
            self.norms2.append(nn.LayerNorm(d_model))
            self.norms3.append(nn.LayerNorm(d_model))
            self.ffns.append(nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
            ))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # Output head: flatten endogenous tokens (patches + global).
        self.head = nn.Linear((self.n_patches + 1) * d_model, n_horizons)

        # Auxiliary future-feature head — same zero-init contract as the
        # nbeats / nhits / itransformer future heads: at step 0 the
        # output is exactly the past-only forecast, and the head only
        # learns to use the future-known covariate signal (Solcast,
        # weather forecasts) if it reduces residuals.
        if self.future_window_size > 0:
            future_flat = self.future_window_size * n_channels
            aux_hidden = max(d_model, n_horizons)
            self.future_aux_head = nn.Sequential(
                nn.Linear(future_flat, aux_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(aux_hidden, n_horizons),
            )
            nn.init.zeros_(self.future_aux_head[-1].weight)
            nn.init.zeros_(self.future_aux_head[-1].bias)
        else:
            self.future_aux_head = None

        self.activation = _build_activation(output_activation, scale=sigmoid_scale)

    def forward(self, x):
        # x: (batch, seq_len, n_channels)
        batch_size = x.size(0)
        if self.revin is not None:
            x = self.revin.normalize(x, past_window_size=self.past_window_size)

        # Snapshot the future block BEFORE slicing so the auxiliary head
        # can consume it (same pattern as iTransformer v2.37.7).
        future_block = None
        if (self.future_aux_head is not None
                and self.past_window_size is not None
                and self.past_window_size < x.size(1)):
            future_block = x[:, self.past_window_size:, :]

        # Past-only slice for both embedders (PF6 rationale).
        if (self.past_window_size is not None
                and self.past_window_size < x.size(1)):
            x = x[:, : self.past_window_size, :]

        # ---- Endogenous patch tokens ----
        endo = x[:, :, self.target_channel]  # (batch, embed_len)
        # Use the most recent n_patches * patch_len values so partial
        # leading patches are dropped, never trailing (recent) ones.
        effective_len = self.n_patches * self.patch_len
        endo = endo[:, endo.size(1) - effective_len:]
        endo = endo.reshape(batch_size, self.n_patches, self.patch_len)
        endo = self.patch_embed(endo)  # (batch, n_patches, d_model)

        # Append the learnable global token and add positions.
        glb = self.glb_token.expand(batch_size, -1, -1)
        endo = torch.cat([endo, glb], dim=1)  # (batch, n_patches + 1, d_model)
        endo = endo + self.pos_embed
        endo = self.dropout(endo)

        # ---- Exogenous variate tokens ----
        exog = None
        if self.n_exog > 0:
            idx = [c for c in range(self.n_channels) if c != self.target_channel]
            ex = x[:, :, idx]                  # (batch, embed_len, n_exog)
            ex = ex.permute(0, 2, 1)           # (batch, n_exog, embed_len)
            exog = self.exog_embed(ex)         # (batch, n_exog, d_model)
            exog = exog + self.exog_pos_embed
            exog = self.dropout(exog)

        # ---- Encoder layers ----
        for i in range(self.n_encoder_layers):
            # Self-attention over endogenous tokens (patches + global).
            attn_out, _ = self.self_attns[i](endo, endo, endo, need_weights=False)
            endo = self.norms1[i](endo + self.dropout(attn_out))

            # Cross-attention: the global token (query) attends to the
            # exogenous variate tokens (keys/values) — the paper's bridge
            # that injects covariate information into the endogenous
            # representation without exploding the attention cost.
            if exog is not None:
                glb_q = endo[:, -1:, :]
                cross_out, _ = self.cross_attns[i](
                    glb_q, exog, exog, need_weights=False,
                )
                glb_q = self.norms2[i](glb_q + self.dropout(cross_out))
                endo = torch.cat([endo[:, :-1, :], glb_q], dim=1)

            # Position-wise FFN.
            ffn_out = self.ffns[i](endo)
            endo = self.norms3[i](endo + self.dropout(ffn_out))

        endo = self.layer_norm(endo)

        # ---- Head ----
        flat = endo.reshape(batch_size, -1)
        out = self.head(flat)  # (batch, n_horizons)

        # Future-covariate adjustment (zero contribution at step 0).
        if future_block is not None:
            future_flat = future_block.reshape(batch_size, -1)
            out = out + self.future_aux_head(future_flat)

        if self.revin is not None:
            out = self.revin.denormalize(out)
        out = self.activation(out)

        if self.n_horizons == 1:
            return out.squeeze(-1)  # (batch,) backward compat
        return out


class TimeXerModel(ForecastModel):
    """
    PyTorch TimeXer time-series forecasting model.

    Endogenous patch tokens + learnable global token, with per-layer
    cross-attention from the global token to exogenous variate tokens
    (Wang et al. 2024). Uses AdamW with CosineAnnealingLR, best-model
    checkpointing. Supports multi-horizon output.
    """

    def __init__(
        self,
        patch_len: int = 8,
        d_model: int = 32,
        n_heads: int = 4,
        n_encoder_layers: int = 1,
        dim_feedforward: int = 64,
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

        self.patch_len = patch_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.dim_feedforward = dim_feedforward
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

        self._model: Optional[_TimeXerNet] = None
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
        # past_window_size enables PF1 (RevIN past-only stats); set per-fit
        # from kwargs, round-tripped in save/load. None means legacy
        # single-window path.
        self._past_window_size: Optional[int] = None
        self._training_history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    @property
    def name(self) -> str:
        return "timexer"

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
        """Train TimeXer with PyTorch autograd and best-model checkpointing."""
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
        self._past_window_size = kwargs.get("past_window_size")

        if not self.use_revin:
            # Per-channel z-score standardisation (fitted on training data)
            self._channel_mean = X_seq.mean(axis=(0, 1))  # shape (n_channels,)
            self._channel_std = X_seq.std(axis=(0, 1))     # shape (n_channels,)
            self._channel_std[self._channel_std < 1e-8] = 1.0
            X_seq = (X_seq - self._channel_mean) / self._channel_std

        # Sigmoid activation needs a ceiling: use training-data maximum with
        # a 10% buffer so the network can reach observed extrema.
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
        self._model = _TimeXerNet(
            seq_len=seq_len,
            n_channels=input_size,
            patch_len=self.patch_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_encoder_layers=self.n_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            n_horizons=self._n_horizons,
            output_activation=_effective_activation,
            sigmoid_scale=self._sigmoid_scale,
            use_revin=self.use_revin,
            target_channel=self.target_channel,
            past_window_size=self._past_window_size,
        )
        optimiser = self._build_optimiser(
            self._model.parameters(), self.optimiser, self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.epochs, eta_min=1e-6)
        _loss_map = {'mse': nn.MSELoss, 'mae': nn.L1Loss, 'l1': nn.L1Loss, 'huber': nn.SmoothL1Loss}
        criterion = _loss_map.get(self.loss_fn, nn.MSELoss)(reduction='none')

        # Training loop -- cosine annealing + best-model checkpoint + early stopping
        best_val_loss = float("inf")
        # best_val_loss tracks raw val_loss (for the checkpoint);
        # best_val_loss_smoothed + val_loss_ema drive the stop decision
        # via _step_early_stop (min_delta + EMA).
        best_val_loss_smoothed = float("inf")
        val_loss_ema: Optional[float] = None
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
            "patch_len": self.patch_len, "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_encoder_layers": self.n_encoder_layers,
            "dim_feedforward": self.dim_feedforward,
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
        valid = {"patch_len", "d_model", "n_heads", "n_encoder_layers",
                 "dim_feedforward", "dropout", "learning_rate", "epochs",
                 "batch_size", "sequence_length", "loss_fn",
                 "daily_loss_weight", "optimiser", "patience",
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
            "seq_len": self._seq_len,
            "n_horizons": self._n_horizons,
            "channel_mean": self._channel_mean,
            "channel_std": self._channel_std,
            "sigmoid_scale": self._sigmoid_scale,
            "y_mean": self._y_mean,
            "y_std": self._y_std,
            "past_window_size": self._past_window_size,
        }, path)
        logger.info(f"Saved TimeXer model to {path}")

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
        self._past_window_size = data.get("past_window_size")

        # Reconstruct the nn.Module and load weights
        self._input_size = data.get("input_size")
        self._seq_len = data.get("seq_len")
        state_dict = data.get("state_dict")
        if state_dict is not None and self._input_size is not None and self._seq_len is not None:
            _effective_activation = (
                'linear' if (self.use_revin and self.output_activation == 'zscore')
                else self.output_activation
            )
            self._model = _TimeXerNet(
                seq_len=self._seq_len,
                n_channels=self._input_size,
                patch_len=self.patch_len,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_encoder_layers=self.n_encoder_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                n_horizons=self._n_horizons,
                output_activation=_effective_activation,
                sigmoid_scale=self._sigmoid_scale,
                use_revin=self.use_revin,
                target_channel=self.target_channel,
                past_window_size=self._past_window_size,
            )
            self._model.load_state_dict(state_dict)
            self._model.eval()

        self._is_fitted = True
        logger.info(f"Loaded TimeXer model from {path}")
