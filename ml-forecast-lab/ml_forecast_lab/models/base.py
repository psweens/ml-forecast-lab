"""
Abstract base class and data structures for forecast models.

Defines the ForecastModel ABC that all concrete model implementations
must inherit from, along with the ModelResult dataclass for storing
training and inference outcomes.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output activation modules (neural backends)
# ---------------------------------------------------------------------------
#
# Each neural backend's forward() ends with one of these applied to the linear
# head's output, so predictions are emitted directly on the correct physical
# range — no post-hoc clipping required. The choice is driven by the
# experiment's ``output_activation`` setting (see ExperimentCfg).
#
# Torch is imported lazily so the base module stays importable in environments
# without PyTorch (e.g. tree-model-only deployments).


def _build_activation(name: str, scale: float = 1.0):
    """
    Return an ``nn.Module`` implementing the requested output activation.

    Parameters
    ----------
    name : str
        One of ``'linear'``, ``'softplus'``, ``'relu'``, ``'exp'``, ``'sigmoid'``,
        ``'zscore'``. ``'auto'`` must be resolved by the caller before reaching
        this function (it's a config-level alias, not a concrete activation).
        ``'zscore'`` uses an identity head because the z-scoring happens at the
        target-transform layer in the backend, not at the output activation.
    scale : float
        Only used for ``'sigmoid'``: upper bound of the sigmoid output range
        (``sigmoid(x) * scale``). Typically set from training-data maximum × a
        small buffer so the network can reach observed extrema.

    Returns
    -------
    torch.nn.Module
        Stateless for linear/softplus/relu/exp/zscore. Sigmoid stores ``scale``
        as a non-trainable buffer so it survives ``state_dict`` round-trips.

    Raises
    ------
    ValueError
        If ``name`` is not a recognised activation.
    RuntimeError
        If PyTorch is not installed.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise RuntimeError(
            'PyTorch is required to build output activations but is not installed'
        ) from e

    if name == 'linear' or name == 'zscore':
        # zscore: the network predicts in z-space with a linear head; the
        # backend denormalises predictions using the stored (mean, std) at
        # inference time. Identity here is correct.
        return nn.Identity()
    if name == 'softplus':
        # Smooth approximation of ReLU with non-zero gradient near zero,
        # range (0, ∞). Default beta=1.
        return nn.Softplus()
    if name == 'relu':
        # Hard non-negative output. Can produce dead units if many training
        # targets are exactly zero (e.g. overnight PV generation).
        return nn.ReLU()
    if name == 'exp':
        return _ExpActivation()
    if name == 'sigmoid':
        return _ScaledSigmoid(float(scale))
    raise ValueError(
        f"Unknown output_activation: {name!r} "
        f"(expected linear|softplus|relu|exp|sigmoid|zscore)"
    )


def _exp_activation_cls():
    """Lazy-define _ExpActivation so torch is only imported when needed."""
    global _ExpActivation, _ScaledSigmoid
    return _ExpActivation


try:
    import torch as _torch
    import torch.nn as _nn

    class _ExpActivation(_nn.Module):
        """
        Exponential output activation, torch.exp(x.clamp(max=20)).

        Useful for strictly-positive targets that vary by orders of magnitude
        (e.g. power with low overnight baseline + high midday peak). The
        ``clamp(max=20)`` prevents overflow to ``inf`` if the pre-activation
        drifts during early training (exp(20) ≈ 4.85e8 which is already far
        above any physically plausible forecast).
        """

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return _torch.exp(x.clamp(max=20.0))

    class _ScaledSigmoid(_nn.Module):
        """
        Sigmoid output with a learned-at-fit-time scale buffer.

        Output is ``sigmoid(x) * scale`` — bounded in (0, scale). The scale
        is stored as a non-trainable buffer so it's captured by
        ``state_dict()``, persisted by ``torch.save()``, and restored on
        load without needing to track it separately.

        Used for quantities with a hard physical ceiling (battery SoC,
        humidity percent, normalised capacity).
        """

        def __init__(self, scale: float = 1.0) -> None:
            super().__init__()
            self.register_buffer('scale', _torch.tensor(float(scale)))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return _torch.sigmoid(x) * self.scale

    class _RevIN(_nn.Module):
        """
        Reversible Instance Normalization (Kim et al. 2022,
        https://openreview.net/forum?id=cGDAkQo1C0p).

        Per-sample, per-channel mean/std are computed over the time axis of
        each input window independently, used to normalise the input, and
        reversed at the output using the target channel's statistics. This
        handles distribution shift between train and test on non-stationary
        series without requiring a retrain — each forecast window is
        automatically rescaled to its own instant level.

        The published architectures this codebase imitates (PatchTST,
        iTransformer, TimesNet, TiDE, TSMixer, SparseTSF, Crossformer) all
        ship with RevIN in their reference implementations. N-BEATS and
        N-HiTS handle instance-level normalisation architecturally via the
        doubly-residual backcast-subtraction stacking, so RevIN is not
        applied to them (it would double-normalise and conflict).

        Design notes
        ------------
        - Stats are computed with ``.detach()`` so the per-sample normalisation
          does not receive gradients — the network cannot game the stats.
        - Denormalisation uses the *target channel's* per-sample stats (default
          channel 0). This matches the common convention that the target's
          most-recent lag features live on channel 0 of the input window.
        - An optional learnable affine (per-channel ``γ`` and ``β``) is applied
          after normalisation; the target-channel affine is reversed before
          the per-sample scale is re-applied at denormalisation.
        - This module is stateful within one forward pass: ``normalize()``
          stashes ``_mean`` / ``_stdev`` as non-buffer tensors, which
          ``denormalize()`` then consumes. A fresh forward always recomputes.
        - RevIN is mutually exclusive with ``output_activation='zscore'``:
          the instance-level and dataset-level target normalisation schemes
          would compose incorrectly. Backends that see both active should
          treat zscore as ``linear`` and let RevIN own the scale.
        """

        def __init__(
            self,
            n_channels: int,
            target_channel: int = 0,
            eps: float = 1e-5,
            affine: bool = True,
        ) -> None:
            super().__init__()
            self.n_channels = int(n_channels)
            self.target_channel = int(target_channel)
            self.eps = float(eps)
            self.affine = bool(affine)
            if self.affine:
                self.affine_weight = _nn.Parameter(_torch.ones(self.n_channels))
                self.affine_bias = _nn.Parameter(_torch.zeros(self.n_channels))
            # Per-forward-pass state (not buffers — differ per batch).
            self._mean: Optional["torch.Tensor"] = None
            self._stdev: Optional["torch.Tensor"] = None

        def normalize(self, x: "torch.Tensor",
                       past_window_size: Optional[int] = None) -> "torch.Tensor":
            """
            Normalise a per-window input tensor.

            Parameters
            ----------
            x : torch.Tensor, shape (batch, seq_len, n_channels)
            past_window_size : int, optional
                When provided AND strictly less than ``x.size(1)``, compute
                per-window stats over the PAST slice
                ``x[:, :past_window_size, :]`` only. This is the v2.37 fix
                for the "RC1: RevIN bias from future-position zeros"
                degeneration: in extended-window mode the future positions
                leave the target channel at zero, which pulls the
                whole-window mean ~50% low and shifts the denormalised
                prediction by the same factor. Past-only stats avoid that
                bias while leaving the legacy single-window path
                completely unchanged.

                When omitted or equal to ``x.size(1)`` (the past-only
                training path), behaves exactly as the original RevIN.

            Returns
            -------
            torch.Tensor
                Input with per-sample per-channel zero-mean unit-variance
                (plus learnable affine if enabled).
            """
            if past_window_size is not None and past_window_size < x.size(1):
                src = x[:, :past_window_size, :]
            else:
                src = x
            # Detach so per-sample stats are constants w.r.t. autograd — the
            # network sees the normalised values but cannot pull gradient
            # through the normalisation itself.
            mean = src.mean(dim=1, keepdim=True).detach()
            var = src.var(dim=1, keepdim=True, unbiased=False).detach()
            stdev = _torch.sqrt(var + self.eps)
            self._mean = mean
            self._stdev = stdev
            x_norm = (x - mean) / stdev
            if self.affine:
                x_norm = x_norm * self.affine_weight + self.affine_bias
            return x_norm

        def denormalize(self, y: "torch.Tensor") -> "torch.Tensor":
            """
            Reverse the target-channel normalisation on a prediction tensor.

            Parameters
            ----------
            y : torch.Tensor, shape (batch,) or (batch, n_horizons)

            Returns
            -------
            torch.Tensor
                Same shape as input, rescaled to the original target scale
                using the target channel's per-sample stats from the most
                recent ``normalize()`` call.
            """
            if self._mean is None or self._stdev is None:
                raise RuntimeError(
                    "RevIN: denormalize() called before normalize() in this forward pass"
                )
            # (batch,) slices from (batch, 1, n_channels).
            mean_t = self._mean[:, 0, self.target_channel]
            stdev_t = self._stdev[:, 0, self.target_channel]
            # Reverse the target-channel affine (inputs went through
            # γ and β after normalise — undo them before re-scaling).
            if self.affine:
                w_t = self.affine_weight[self.target_channel]
                b_t = self.affine_bias[self.target_channel]
                y = (y - b_t) / (w_t + self.eps)
            if y.dim() == 1:
                return y * stdev_t + mean_t
            if y.dim() == 2:
                # Multi-horizon: broadcast over the horizon axis.
                return y * stdev_t.unsqueeze(-1) + mean_t.unsqueeze(-1)
            # Multi-horizon × multi-quantile: broadcast over both trailing
            # axes so DLinear's (batch, H, Q) quantile output denormalises
            # correctly.
            return y * stdev_t.view(-1, 1, 1) + mean_t.view(-1, 1, 1)

    def _past_only_score_mask(seq_len: int, past_window_size: Optional[int],
                                device: "torch.device") -> Optional["torch.Tensor"]:
        """Boolean key-padding mask masking out future positions.

        Returns shape ``(seq_len,)`` with True at positions to MASK OUT
        (i.e. the future positions ``[past_window_size:]``). Returns
        None when ``past_window_size`` is None or covers the whole
        window — matches what ``nn.MultiheadAttention`` and
        ``nn.TransformerEncoder`` expect as ``src_key_padding_mask``.

        Used by attention-bearing backends (LSTM/GRU temporal
        attention, CNN learnable pool, PatchTST/Crossformer transformer
        encoders, TFT multi-head attention) to ensure that the
        attention scores assigned to future positions — where the
        target channel is zero in extended-window mode — are pushed to
        ``-inf`` before softmax. Without this mask the LSTM in
        particular puts ~48% of its weight mass on future-position
        zero-target slots and learns to read absolute time from there,
        causing the v2.36 phase-inversion symptom.
        """
        if past_window_size is None or past_window_size >= seq_len:
            return None
        mask = _torch.zeros(seq_len, dtype=_torch.bool, device=device)
        mask[past_window_size:] = True
        return mask
except ImportError:
    # Torch not installed — activation factory will raise at call time.
    _ExpActivation = None  # type: ignore[assignment]
    _ScaledSigmoid = None  # type: ignore[assignment]
    _RevIN = None  # type: ignore[assignment]
    _past_only_score_mask = None  # type: ignore[assignment]


def _resolve_sigmoid_scale(y: np.ndarray, buffer: float = 1.1) -> float:
    """
    Compute the sigmoid upper bound from training targets.

    Uses ``max(|y|) * buffer`` so the network can reach observed extrema with a
    little headroom (sigmoid asymptotically approaches its bound but never
    touches it). Falls back to ``1.0`` for degenerate arrays (empty or all-zero)
    so the activation remains well-defined.
    """
    if y.size == 0:
        return 1.0
    y_max = float(np.max(np.abs(y)))
    if not np.isfinite(y_max) or y_max <= 0.0:
        return 1.0
    return y_max * float(buffer)


@dataclass
class ModelResult:
    """
    Container for model training and inference results.

    Attributes
    ----------
    model_name : str
        Identifier of the model backend (e.g. 'lightgbm', 'lstm').
    predictions : np.ndarray
        Forecast predictions of shape (n_samples, horizon) or (n_samples,)
        for single-step predictions.
    train_time_seconds : float
        Wall-clock training time in seconds.
    inference_time_seconds : float
        Wall-clock inference time in seconds (for predict or predict_multi).
    metrics : dict[str, float]
        Calculated metrics (e.g. {'mae': 0.15, 'rmse': 0.22}).
    fold_metrics : list[dict[str, float]]
        Per-fold metrics for cross-validation. Each entry is a dictionary
        of metric names to values.
    hyperparameters : dict
        Current hyperparameter configuration of the model.
    """

    model_name: str
    predictions: np.ndarray
    train_time_seconds: float
    inference_time_seconds: float
    metrics: Dict[str, float] = field(default_factory=dict)
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return detailed representation of result."""
        return (
            f'ModelResult(model_name={self.model_name!r}, '
            f'predictions.shape={self.predictions.shape}, '
            f'train_time={self.train_time_seconds:.2f}s, '
            f'inference_time={self.inference_time_seconds:.4f}s, '
            f'metrics={self.metrics})'
        )


class TrainingCancelled(Exception):
    """Cooperative training-cancellation signal (audit F10).

    Training runs in executor threads, which ``asyncio.Task.cancel()``
    cannot interrupt — before v2.41.0 the Stop button cancelled the
    coroutine while the thread kept saturating the CPU until the fit
    finished. Cancellation now flows through the epoch callback: the
    caller's callback raises this when its cancel event is set,
    ``_emit_epoch`` re-raises it (unlike ordinary callback errors,
    which are swallowed), and it propagates out of ``fit`` to the
    caller, stopping the thread at the next epoch boundary.
    """


class ForecastModel(ABC):
    """
    Abstract base class for all time-series forecast models.

    All concrete implementations must:
    1. Implement the abstract methods (name, fit, predict, etc.)
    2. Properly validate and type-hint inputs
    3. Support serialisation/deserialisation
    4. Provide logging for important events

    The model receives FLAT feature matrices of shape (n_samples, n_features)
    from the feature engineering pipeline. Models that require sequence data
    (e.g. LSTM, GRU) should reshape internally using utilities from the
    features module.
    """

    def __init__(self) -> None:
        """Initialise base model."""
        self._is_fitted = False
        self._fit_timestamp: Optional[float] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the model identifier string.

        Returns
        -------
        str
            Unique model name (e.g. 'lightgbm', 'lstm', 'xgboost').
            Used as the key in the model registry.
        """
        pass

    @property
    def is_fitted(self) -> bool:
        """
        Return whether the model has been trained.

        Returns
        -------
        bool
            True if fit() has been called successfully.
        """
        return self._is_fitted

    @property
    def is_neural(self) -> bool:
        """Whether this model requires sequence (sliding-window) input.

        Drives the benchmark/training pipelines: when True, the caller
        builds sliding-window features and uses ``predict_sequence``.
        Note that some non-neural models (seasonal_naive, arima, ets,
        theta) also return True here because they consume the target
        channel of each window — see ``model_family`` for the actual
        algorithmic family.
        """
        return False

    @property
    def model_family(self) -> str:
        """Algorithmic family for UI grouping and pipeline branching.

        One of:
          - ``'tree'``: gradient-boosted trees (lightgbm, xgboost, catboost).
          - ``'neural'``: deep-learning backends (lstm, gru, transformer, ...).
          - ``'classical'``: classical statistical models (arima, ets, theta).
          - ``'baseline'``: trivial reference rules (seasonal_naive).

        Defaults to ``'neural'`` when ``is_neural`` is True and ``'tree'``
        otherwise. Classical/baseline backends should override this
        explicitly so the leaderboard, model catalog, and per-family
        defaults can group them correctly.
        """
        return 'neural' if self.is_neural else 'tree'

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Train the model on the provided data.

        Parameters
        ----------
        X_train : np.ndarray
            Training features of shape (n_samples, n_features).
        y_train : np.ndarray
            Training targets of shape (n_samples,) or (n_samples, 1).
        **kwargs : Any
            Optional model-specific parameters (e.g. validation set,
            early stopping patience, sample weights).

        Returns
        -------
        dict[str, Any]
            Training metadata including at minimum:
            - 'time_seconds': Wall-clock training duration
            - 'epochs': Number of training epochs (for neural models)
            - Any other backend-specific metrics

        Notes
        -----
        Implementation should:
        1. Validate input shapes and types
        2. Log training progress
        3. Set self._is_fitted = True upon success
        4. Return comprehensive training metadata
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate single-step-ahead forecast.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples,) or (n_samples, 1).

        Raises
        ------
        RuntimeError
            If model has not been fitted.

        Notes
        -----
        For multi-step forecasting, use predict_multi() instead.
        """
        if not self.is_fitted:
            raise RuntimeError(f'{self.name} model must be fitted before prediction')

    def predict_sequence(self, X: np.ndarray) -> np.ndarray:
        """
        Generate multi-horizon forecasts from sliding-window input.

        Parameters
        ----------
        X : np.ndarray
            Sliding-window input of shape (n_samples, window_size, n_channels).

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples, n_horizons).

        Raises
        ------
        NotImplementedError
            If the model does not support direct multi-horizon prediction.
        """
        raise NotImplementedError(
            f"{self.name} does not support predict_sequence(). "
            f"Use predict() for single-step forecasting."
        )

    def predict_multi(
        self,
        X: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """
        Generate multi-step-ahead forecast.

        Default implementation performs recursive single-step predictions,
        re-feeding previous predictions as features. Subclasses can
        override for direct multi-output or attention-based approaches.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        horizon : int
            Number of steps to forecast ahead.

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples, horizon).

        Raises
        ------
        ValueError
            If horizon < 1.
        RuntimeError
            If model has not been fitted.

        Notes
        -----
        The recursive approach accumulates forecast error. For critical
        applications, consider direct multi-output models.
        """
        if not self.is_fitted:
            raise RuntimeError(f'{self.name} model must be fitted before prediction')
        if horizon < 1:
            raise ValueError(f'horizon must be >= 1, got {horizon}')

        n_samples, n_features = X.shape
        predictions = np.zeros((n_samples, horizon), dtype=np.float32)

        X_current = X.copy()
        for h in range(horizon):
            y_pred = self.predict(X_current)
            if y_pred.ndim > 1:
                y_pred = y_pred.ravel()
            predictions[:, h] = y_pred

            # For recursive forecasting, shift features and append predictions
            # This is a naive approach; sophisticated models should override
            if h < horizon - 1:
                X_current = X_current.copy()
                # Shift lagged features if they exist in position
                if n_features > 1:
                    X_current[:, :-1] = X_current[:, 1:]
                X_current[:, 0] = y_pred

        logger.debug(
            f'{self.name} produced {horizon}-step forecast '
            f'of shape {predictions.shape}'
        )
        return predictions

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Return all hyperparameters as a dictionary.

        Returns
        -------
        dict[str, Any]
            Hyperparameter names mapped to their current values.

        Notes
        -----
        Must return a deep copy to prevent external modification.
        """
        pass

    @abstractmethod
    def set_params(self, **kwargs: Any) -> None:
        """
        Update hyperparameters.

        Parameters
        ----------
        **kwargs : Any
            Hyperparameters to update. Invalid parameter names should
            raise ValueError.

        Raises
        ------
        ValueError
            If an unknown hyperparameter is provided.

        Notes
        -----
        Updates should not affect a fitted model; typically used before
        fitting. Implementations may log warnings if called on fitted models.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Serialise model state to disk.

        Parameters
        ----------
        path : str
            File path for saving. Should include appropriate file extension
            (e.g. '.pkl', '.joblib').

        Raises
        ------
        IOError
            If write fails.

        Notes
        -----
        Implementation should handle model-specific serialisation
        (e.g. LightGBM models may use native .txt format).
        The saved file must be loadable with load().
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Deserialise model state from disk.

        Parameters
        ----------
        path : str
            File path to the saved model.

        Raises
        ------
        IOError
            If read fails or file format is invalid.

        Notes
        -----
        After load(), the model should be in the same state as when save()
        was called, including self._is_fitted = True.
        """
        pass

    def _emit_epoch(self, callback: Any, **data: Any) -> None:
        """Invoke an epoch callback if provided, swallowing any errors.

        ``TrainingCancelled`` is the one exception allowed through —
        it's the cooperative stop signal, not a callback bug.
        """
        if callback is not None:
            try:
                callback(**data)
            except TrainingCancelled:
                raise
            except Exception:
                pass  # Never let callback errors break training

    @staticmethod
    def _tail_val_split(n_total: int, val_split: float, gap: int = 0):
        """
        Tail validation split with a purge gap.

        Val is the last `val_split` fraction of samples; train is everything
        before, minus a `gap` of samples immediately preceding val to prevent
        target leakage from train windows whose forecast horizon overlaps
        val inputs.

        Returns
        -------
        (train_mask, val_mask) : tuple[np.ndarray, np.ndarray]
            Boolean masks of length n_total.
        """
        n_val = max(1, int(n_total * val_split))
        val_start = n_total - n_val
        train_end = max(0, val_start - int(gap))
        val_mask = np.zeros(n_total, dtype=bool)
        val_mask[val_start:] = True
        train_mask = np.zeros(n_total, dtype=bool)
        train_mask[:train_end] = True
        return train_mask, val_mask

    @staticmethod
    def _weighted_mean_loss(loss_per_sample: "torch.Tensor",
                            w_batch: "torch.Tensor") -> "torch.Tensor":
        """
        Proper sample-weighted mean: sum(loss_i * w_i) / sum(w_i).

        Handles both single-horizon (1-D) and multi-horizon (2-D) loss
        tensors. For multi-horizon, averages over horizons per sample first
        so the scalar weighted mean is on the same scale as the unweighted
        mean (average per-sample MSE).
        """
        if loss_per_sample.ndim > 1:
            per_sample = loss_per_sample.mean(dim=tuple(range(1, loss_per_sample.ndim)))
        else:
            per_sample = loss_per_sample
        w_sum = w_batch.sum().clamp_min(1e-8)
        return (per_sample * w_batch).sum() / w_sum

    @staticmethod
    def _build_optimiser(
        params,
        name: str,
        lr: float,
        weight_decay: float = 1e-4,
    ):
        """
        Build a torch optimiser by name.

        Parameters
        ----------
        params : iterable of torch.nn.Parameter
            Typically ``model.parameters()``.
        name : {'adamw', 'adam'}
            Optimiser selection. Case-insensitive. ``'adamw'`` uses
            decoupled weight decay (Loshchilov & Hutter 2017) — the default
            and the one every published time-series transformer paper
            uses. ``'adam'`` is classic Adam with tied weight decay (decay
            divided by per-parameter adaptive LR, so frequently-updated
            parameters receive less effective regularisation).
        lr : float
            Initial learning rate. Passed through to the optimiser and then
            scheduled by the caller's ``CosineAnnealingLR``.
        weight_decay : float
            L2 regularisation strength. Default 1e-4 — the same value each
            backend was previously hardcoding. Kept consistent across both
            optimisers so the Adam vs AdamW comparison isolates the
            decoupling behaviour rather than confounding it with decay
            magnitude.

        Returns
        -------
        torch.optim.Optimizer

        Raises
        ------
        ValueError
            If ``name`` is not one of the supported values.
        """
        try:
            import torch.optim as optim
        except ImportError as e:
            raise RuntimeError(
                'PyTorch is required to build an optimiser but is not installed'
            ) from e
        key = str(name).lower()
        if key == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        if key == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        raise ValueError(
            f"Unknown optimiser: {name!r} (expected 'adamw' or 'adam')"
        )

    @staticmethod
    def _step_early_stop(
        val_loss: float,
        best_val_loss: float,
        best_val_loss_smoothed: float,
        val_loss_ema: Optional[float],
        patience_counter: int,
        *,
        min_delta: float = 1e-3,
        ema_alpha: float = 0.3,
    ) -> dict:
        """One-epoch early-stop bookkeeping with two refinements:

        1. ``min_delta``: improvements smaller than this relative
           fraction (default 0.1 %) don't reset the patience counter.
           Without this, the previous strict ``val_loss < best_val_loss``
           reset patience on micro-improvements, occasionally letting
           training run hours past where it should have stopped.

        2. ``ema_alpha``: the *stop decision* compares an EMA of
           val_loss (default α=0.3, ~3-4 epoch effective window) rather
           than the raw single-epoch value, so one noisy epoch doesn't
           reset patience and one lucky epoch doesn't extend training.
           The *best-model checkpoint* still tracks the raw val_loss so
           the weights returned to inference are the truly best ones.

        The two refinements are independent and can be disabled by
        ``min_delta=0`` / ``ema_alpha=1.0`` respectively — the latter
        recovers the pre-v2.40.12 behaviour byte-for-byte.

        Parameters
        ----------
        val_loss : float
            This epoch's raw val_loss.
        best_val_loss : float
            Best raw val_loss observed so far (drives the checkpoint).
        best_val_loss_smoothed : float
            Best EMA-smoothed val_loss observed so far (drives the
            stop decision).
        val_loss_ema : float or None
            Running EMA of val_loss. ``None`` on the first epoch — the
            helper seeds it with ``val_loss``.
        patience_counter : int
            Current patience counter.
        min_delta : float
            Minimum relative improvement required to reset patience.
            ``smoothed < best_smoothed * (1 - min_delta)``.
        ema_alpha : float
            EMA weight on the newest val_loss (1.0 = no smoothing).

        Returns
        -------
        dict
            ``val_loss_ema`` : updated EMA
            ``best_val_loss`` : updated raw best
            ``best_val_loss_smoothed`` : updated smoothed best
            ``patience_counter`` : 0 if improved (smoothed), else +1
            ``checkpoint_best`` : True if the raw val_loss is a new best
        """
        new_ema = (
            val_loss if val_loss_ema is None
            else ema_alpha * val_loss + (1.0 - ema_alpha) * val_loss_ema
        )
        is_new_raw_best = val_loss < best_val_loss
        new_best = val_loss if is_new_raw_best else best_val_loss

        # Stop decision: compare smoothed vs smoothed-best, with the
        # min_delta margin. The very first epoch (val_loss_ema was None)
        # always counts as an improvement so patience starts at 0.
        threshold = best_val_loss_smoothed * (1.0 - min_delta)
        if val_loss_ema is None or new_ema < threshold:
            new_best_smoothed = new_ema
            new_patience = 0
        else:
            new_best_smoothed = best_val_loss_smoothed
            new_patience = patience_counter + 1

        return {
            "val_loss_ema": new_ema,
            "best_val_loss": new_best,
            "best_val_loss_smoothed": new_best_smoothed,
            "patience_counter": new_patience,
            "checkpoint_best": is_new_raw_best,
        }

    # v2.40.14: composite-horizon loss collapsed to interval-only.
    # Old per-interval + cumulative-trajectory blend (legacy additive
    # ``L = L_interval + λ·L_daily`` and the convex EMA-normalised
    # ``L = (1-α)·L_interval/ema_i + α·L_daily/ema_d``) is gone after
    # the loss-comparison harness measured the cumulative term as a
    # net hurt in BOTH sparse-demand AND smooth-cumulative regimes,
    # with a structural gradient asymmetry (early horizon steps get H×
    # more cumsum-error terms than late ones → systematic
    # under-prediction at early horizon, exactly the failure mode the
    # term was supposed to fix). The α-cliff was the visible symptom;
    # the per-interval-bias jump on smooth-cumulative (-0.24 → -0.69 at
    # α=0.1) was the direct measurement of the gradient asymmetry.
    # Faster EMA decay softened but did not remove the cliff →
    # mechanism is in the loss, not the normaliser. See
    # ``scripts/LOSS_COMPARISON_FINDINGS.md`` for the full evidence
    # stack. Signature kept (``daily_weight`` ignored) so all 17
    # neural backends keep calling without per-backend edits.
    def _composite_horizon_loss(
        self,
        y_pred: "torch.Tensor",
        y_true: "torch.Tensor",
        criterion: "nn.Module",
        w_batch: Optional["torch.Tensor"],
        daily_weight: float,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """v2.40.14: per-interval loss only. The blended-with-cumulative
        path was removed (see method-block comment above for evidence
        + mechanism). ``daily_weight`` and ``self.loss_balance`` /
        ``self._loss_ema`` are ignored if set.

        Returns ``(scalar_loss, pre-reduction interval-loss tensor)`` —
        same signature backends and training-loop loggers already
        consume, so no per-backend changes are needed.
        """
        # DILATE (shape + time) loss for spiky targets. Selected per experiment
        # via loss_fn='dilate'; only meaningful for a multi-step horizon, so it
        # degrades to the point-loss path when y has <2 horizon columns. Handled
        # here at the shared chokepoint so every neural backend that calls
        # _composite_horizon_loss gets it with no per-backend edit.
        # `== 2`, not `>= 2`: a quantile backend emits (B, H, Q) and
        # dilate_per_sample only handles (B,) or (B, H). With `>= 2` the guard
        # passed for a 3-D tensor, dilate read the quantile axis as the horizon
        # and DLinear raised on its first batch — while also bypassing the
        # _pinball criterion that handles the 3-D case correctly.
        if (getattr(self, 'loss_fn', None) == 'dilate'
                and y_pred.dim() == 2 and y_pred.shape[-1] >= 2):
            from .dilate_loss import dilate_per_sample
            per_sample = dilate_per_sample(
                y_pred, y_true,
                alpha=float(getattr(self, 'dilate_alpha', 0.5)),
                gamma=float(getattr(self, 'dilate_gamma', 0.01)),
                band=getattr(self, 'dilate_band', None),
            )
            if w_batch is not None:
                loss = ForecastModel._weighted_mean_loss(per_sample, w_batch)
            else:
                loss = per_sample.mean()
            return loss, per_sample

        interval_per_sample = criterion(y_pred, y_true)
        if w_batch is not None:
            loss = ForecastModel._weighted_mean_loss(
                interval_per_sample, w_batch,
            )
        else:
            loss = interval_per_sample.mean()
        return loss, interval_per_sample

    def _validate_fitted(self) -> None:
        """
        Raise RuntimeError if model is not fitted.

        Utility method for subclasses to enforce fit() before operations.

        Raises
        ------
        RuntimeError
            If self._is_fitted is False.
        """
        if not self.is_fitted:
            raise RuntimeError(f'{self.name} model must be fitted before this operation')

    def _validate_X(self, X: np.ndarray) -> None:
        """
        Validate input feature array.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix to validate.

        Raises
        ------
        TypeError
            If X is not a 2D numpy array.
        ValueError
            If X is empty.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f'X must be a numpy array, got {type(X).__name__}')
        if X.ndim != 2:
            raise TypeError(f'X must be 2D, got shape {X.shape}')
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f'X cannot be empty, got shape {X.shape}')

    def _validate_y(self, y: np.ndarray) -> np.ndarray:
        """
        Validate and flatten target array.

        Parameters
        ----------
        y : np.ndarray
            Target array of shape (n_samples,) or (n_samples, 1).

        Returns
        -------
        np.ndarray
            Flattened target array of shape (n_samples,).

        Raises
        ------
        TypeError
            If y is not a numpy array.
        ValueError
            If shape does not match expected dimensions.
        """
        if not isinstance(y, np.ndarray):
            raise TypeError(f'y must be a numpy array, got {type(y).__name__}')
        if y.ndim == 1:
            return y
        elif y.ndim == 2 and y.shape[1] == 1:
            return y.ravel()
        elif y.ndim == 2 and y.shape[1] > 1:
            return y  # Multi-horizon targets, keep 2D
        else:
            raise ValueError(f'y must be 1D or 2D, got shape {y.shape}')

    def __repr__(self) -> str:
        """Return string representation of model."""
        fitted_str = 'fitted' if self.is_fitted else 'unfitted'
        return f'{self.__class__.__name__}({self.name!r}, {fitted_str})'
