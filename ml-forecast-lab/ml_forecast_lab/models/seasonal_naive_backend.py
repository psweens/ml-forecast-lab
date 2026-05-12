"""
Seasonal Naive baseline forecasting model backend for ML Forecast Lab.

Implements the seasonal-naive baseline that every forecasting paper requires:
  ŷ[t + h] = y[t + h - m]
where m is the seasonal period (e.g. 48 for half-hourly data with daily
seasonality, 24 for hourly daily, 7 for daily weekly).

This is the "fair comparison" reference — any learned model that fails to
beat seasonal-naive on its target series is providing no value over a
look-back-one-period rule. Including it as a registered backend means it
shows up in the Demšar ranking automatically and is benchmarked on the
same CV folds as every other model.

The implementation has no learnable parameters: ``fit`` only stores the
recent training tail to fall back on at inference time when the input
window is shorter than ``seasonal_period``. ``predict`` and
``predict_sequence`` simply look up the value from ``seasonal_period``
steps ago for each horizon.
"""

import logging
import pickle
import warnings
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

from .base import ForecastModel

logger = logging.getLogger(__name__)


class SeasonalNaiveModel(ForecastModel):
    """
    Seasonal-naive forecast: ŷ[t + h] = y[t + h - period].

    A no-training baseline that returns the value from one seasonal period
    ago for each horizon. Falls back to non-seasonal naive (last observed
    value) if the input window is shorter than the configured period.
    """

    def __init__(
        self,
        seasonal_period: int = 48,
        target_channel: int = 0,
        sequence_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        if seasonal_period < 1:
            raise ValueError(f"seasonal_period must be >= 1, got {seasonal_period}")
        self.seasonal_period = int(seasonal_period)
        self.target_channel = int(target_channel)
        self.sequence_length = sequence_length

        # Cached tail of the training series; used at inference time when the
        # caller provides only a flat feature row (no sequence) and we need
        # to back-fill enough history to look one period back.
        self._train_tail: Optional[np.ndarray] = None
        self._n_horizons: int = 1

    @property
    def name(self) -> str:
        return "seasonal_naive"

    @property
    def is_neural(self) -> bool:
        # Returning True here lets the benchmark pipeline pass sliding-window
        # sequence data the same way it does for the neural backends, which
        # is exactly what seasonal-naive needs to look one period back.
        return True

    @property
    def model_family(self) -> str:
        return 'baseline'

    def _reshape_to_sequences(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        seq_len = self.sequence_length or n_features
        n_features_per_step = (
            n_features // seq_len if n_features % seq_len == 0 else 1
        )
        return X[:, :seq_len * n_features_per_step].reshape(
            n_samples, seq_len, n_features_per_step,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            **kwargs: Any) -> Dict[str, Any]:
        """No training required — only cache the recent target tail."""
        self._validate_X(X_train)
        y_train = self._validate_y(y_train)

        if y_train.ndim == 2 and y_train.shape[1] > 1:
            self._n_horizons = y_train.shape[1]
            tail_source = y_train[:, 0]  # Use horizon-0 column as the series
        else:
            self._n_horizons = 1
            tail_source = y_train.ravel()

        # Keep the last 2 × period samples — enough to satisfy any look-back
        # query plus a buffer for multi-horizon recursion.
        keep = max(2 * self.seasonal_period, 1)
        self._train_tail = np.asarray(tail_source[-keep:], dtype=np.float32)

        self._is_fitted = True
        logger.info(
            f"SeasonalNaive fit (no training): period={self.seasonal_period}, "
            f"cached_tail={len(self._train_tail)}"
        )
        return {"time_seconds": 0.0, "epochs": 0, "best_val_loss": 0.0}

    def _per_window_predict(self, window: np.ndarray, n_horizons: int) -> np.ndarray:
        """Return n_horizons predictions for a single window.

        window: (seq_len, n_channels) — target lives on self.target_channel.
        """
        target_series = window[:, self.target_channel]
        seq_len = len(target_series)
        period = self.seasonal_period
        out = np.zeros(n_horizons, dtype=np.float32)

        for h in range(n_horizons):
            # Step in the future is t + h + 1 (1-indexed forecast horizon).
            # Look back exactly `period` steps from there.
            offset = h + 1 - period
            if offset >= 0:
                # Already past the window — use a previous prediction.
                # Out-of-window: fall back to recursion through previous
                # forecast steps (which themselves are seasonal-naive).
                # offset >= 0 means we're forecasting more than one period ahead.
                out[h] = out[offset]
            else:
                idx = seq_len + offset  # negative offset → look back from end
                if 0 <= idx < seq_len:
                    out[h] = target_series[idx]
                elif self._train_tail is not None and len(self._train_tail) >= -idx:
                    # Borrow from the cached training tail when the window
                    # alone isn't long enough.
                    out[h] = self._train_tail[idx]
                else:
                    # Last resort: most recent observation (non-seasonal naive).
                    out[h] = target_series[-1]
        return out

    def predict_sequence(self, X: np.ndarray) -> np.ndarray:
        """Multi-horizon prediction from sliding-window input."""
        self._validate_fitted()
        if X.ndim != 3:
            raise ValueError(f"Expected 3-D windowed input, got shape {X.shape}")
        n_samples = X.shape[0]
        out = np.zeros((n_samples, self._n_horizons), dtype=np.float32)
        for i in range(n_samples):
            out[i] = self._per_window_predict(X[i], self._n_horizons)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Single-step prediction (returns horizon-0)."""
        self._validate_fitted()
        self._validate_X(X)
        X_seq = self._reshape_to_sequences(X)
        out = np.zeros(X_seq.shape[0], dtype=np.float32)
        for i in range(X_seq.shape[0]):
            preds = self._per_window_predict(X_seq[i], max(self._n_horizons, 1))
            out[i] = preds[0]
        return out

    def get_params(self) -> Dict[str, Any]:
        return deepcopy({
            "seasonal_period": self.seasonal_period,
            "target_channel": self.target_channel,
            "sequence_length": self.sequence_length,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"seasonal_period", "target_channel", "sequence_length"}
        for k, v in kwargs.items():
            if k not in valid:
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)

    def save(self, path: str) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "params": self.get_params(),
                    "train_tail": self._train_tail,
                    "n_horizons": self._n_horizons,
                }, f)
            logger.info(f"Saved SeasonalNaive to {path}")
        except Exception as e:
            raise IOError(f"Failed to save model to {path}: {e}")

    def load(self, path: str) -> None:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.set_params(**data["params"])
            self._train_tail = data.get("train_tail")
            self._n_horizons = data.get("n_horizons", 1)
            self._is_fitted = True
            logger.info(f"Loaded SeasonalNaive from {path}")
        except Exception as e:
            raise IOError(f"Failed to load model from {path}: {e}")
