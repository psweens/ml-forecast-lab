"""
Daily-total × within-day-profile forecasting backend (hierarchical decomposition).

Motivation
----------
For an event-driven load (hot-water reheat energy, EV, appliances) the *within-
day timing* of the spikes is largely unpredictable, but the *daily total* — how
much energy the thing uses today — is far more predictable and is usually what
the user actually cares about. Forecasting at the level where the signal lives
and distributing it is the hierarchical-reconciliation idea (predict the
aggregate, then a shape, then make them coherent).

This backend implements a robust, dependency-light version of that:

    ŷ[t + h] = profile[t + h]  ×  (projected_daily_total / reference_daily_total)

where the **profile** is the most recent realised day's shape (look back one
seasonal period, exactly like Seasonal-Naive, which is phase-correct by
construction) and the **scale** reconciles that day's total to a projected
total estimated from the recent trailing daily totals. This scales the whole
day toward the projected total, so the model tracks day-level amplitude (big
vs small hot-water days) while distributing it by a realistic intra-day shape.

When the daily total is stable day-to-day the scale is ≈1 and the model reduces
to Seasonal-Naive; it only diverges when the level is trending (a run of big or
small hot-water days), which is exactly when a flat seasonal-naive under/over-
shoots. It competes in the benchmark like any other backend and — paired with
the peak-aware selection metric — is favoured when its scaled shape tracks the
peaks better than a flat baseline.

Why a positional (look-back-one-period) profile rather than an averaged
clock-of-day profile: the windowed backend interface exposes the target channel
and a sliding window but not absolute wall-clock time, so an averaged
interval-of-day profile cannot be phase-aligned reliably between fit and
predict. The one-period look-back is always phase-correct. This is the honest,
robust v1; a covariate-conditioned daily-total model is the natural follow-up.
"""

import logging
import pickle
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

from .base import ForecastModel

logger = logging.getLogger(__name__)


class DailyProfileModel(ForecastModel):
    """Total-reconciled seasonal-naive: recent day's shape × projected total."""

    def __init__(
        self,
        seasonal_period: int = 48,
        target_channel: int = 0,
        sequence_length: Optional[int] = None,
        level_days: int = 7,
        level_half_life_days: float = 3.0,
        scale_clip: float = 4.0,
    ) -> None:
        super().__init__()
        if seasonal_period < 1:
            raise ValueError(f"seasonal_period must be >= 1, got {seasonal_period}")
        if level_days < 1:
            raise ValueError(f"level_days must be >= 1, got {level_days}")
        if level_half_life_days <= 0:
            raise ValueError(
                f"level_half_life_days must be > 0, got {level_half_life_days}"
            )
        if scale_clip < 1.0:
            raise ValueError(f"scale_clip must be >= 1.0, got {scale_clip}")
        self.seasonal_period = int(seasonal_period)
        self.target_channel = int(target_channel)
        self.sequence_length = sequence_length
        # How many trailing complete days feed the projected-total estimate, and
        # the EWMA half-life (in days) used to weight them toward the recent end.
        self.level_days = int(level_days)
        self.level_half_life_days = float(level_half_life_days)
        # Cap on the reconciliation scale so a near-zero reference day can't
        # blow a forecast up (or collapse it to zero).
        self.scale_clip = float(scale_clip)

        self._past_window_size: Optional[int] = None
        self._train_tail: Optional[np.ndarray] = None
        self._n_horizons: int = 1

    @property
    def name(self) -> str:
        return "daily_profile"

    @property
    def is_neural(self) -> bool:
        # Mirror Seasonal-Naive: declaring neural makes the pipeline feed
        # sliding-window sequence data, which is what the period look-back needs.
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
        """No training — cache the recent target tail for level/look-back."""
        self._validate_X(X_train)
        y_train = self._validate_y(y_train)

        extended = bool(kwargs.get('extended_window', False))
        pw = kwargs.get('past_window_size')
        self._past_window_size = int(pw) if extended and pw is not None else None

        if y_train.ndim == 2 and y_train.shape[1] > 1:
            self._n_horizons = y_train.shape[1]
            tail_source = y_train[:, 0]
        else:
            self._n_horizons = 1
            tail_source = y_train.ravel()

        # Keep enough days to estimate the recent level plus the look-back.
        keep = max((self.level_days + 2) * self.seasonal_period, 2 * self.seasonal_period)
        self._train_tail = np.asarray(tail_source[-keep:], dtype=np.float32)

        self._is_fitted = True
        logger.info(
            f"DailyProfile fit (no training): period={self.seasonal_period}, "
            f"level_days={self.level_days}, cached_tail={len(self._train_tail)}"
        )
        return {"time_seconds": 0.0, "epochs": 0, "best_val_loss": 0.0}

    def _projected_total_and_reference(self, history: np.ndarray) -> tuple:
        """Estimate (projected_total, reference_total) from a 1-D past series.

        ``reference_total`` is the most recent complete day's sum (the day whose
        shape the look-back reuses). ``projected_total`` is an EWMA over the last
        ``level_days`` complete-day sums, weighted toward the recent end.
        """
        period = self.seasonal_period
        n = len(history)
        ref_total = float(np.sum(history[-period:])) if n >= period else float(np.sum(history))

        # Trailing complete-day totals, most-recent last.
        totals = []
        end = n
        while end - period >= 0 and len(totals) < self.level_days:
            totals.append(float(np.sum(history[end - period:end])))
            end -= period
        if not totals:
            return ref_total, ref_total
        totals = totals[::-1]  # oldest → newest

        # EWMA with the configured half-life (in days). Weight_i grows toward
        # the newest day so a recent regime shift is reflected quickly.
        decay = 0.5 ** (1.0 / max(self.level_half_life_days, 1e-6))
        k = len(totals)
        weights = np.array([decay ** (k - 1 - i) for i in range(k)], dtype=np.float64)
        proj_total = float(np.sum(weights * np.asarray(totals)) / np.sum(weights))
        return proj_total, ref_total

    def _per_window_predict(self, window: np.ndarray, n_horizons: int) -> np.ndarray:
        target_series = window[:, self.target_channel]
        if self._past_window_size is not None:
            past_len = min(self._past_window_size, len(target_series))
        else:
            past_len = len(target_series)
        past = target_series[:past_len]
        period = self.seasonal_period

        # Build the level-estimation history from the window's past plus the
        # cached training tail, so we have enough complete days even when a
        # single window is shorter than level_days × period.
        if self._train_tail is not None and len(self._train_tail) > 0:
            history = np.concatenate([self._train_tail, past])
        else:
            history = past

        proj_total, ref_total = self._projected_total_and_reference(history)
        if ref_total > 1e-9:
            scale = proj_total / ref_total
        else:
            scale = 1.0
        # Clamp so a tiny reference day can't explode or zero out the forecast.
        scale = float(np.clip(scale, 1.0 / self.scale_clip, self.scale_clip))

        out = np.zeros(n_horizons, dtype=np.float32)
        for h in range(n_horizons):
            offset = h + 1 - period
            if offset >= 0:
                # Beyond one period ahead — recurse on already-forecast values
                # (which are themselves scaled), keeping the day coherent.
                base = out[offset] / scale if scale != 0 else out[offset]
            else:
                idx = past_len + offset
                if 0 <= idx < past_len:
                    base = target_series[idx]
                elif self._train_tail is not None and len(self._train_tail) >= -idx:
                    base = self._train_tail[idx]
                elif past_len > 0:
                    base = target_series[past_len - 1]
                else:
                    base = 0.0
            out[h] = float(base) * scale
        return out

    def predict_sequence(self, X: np.ndarray) -> np.ndarray:
        self._validate_fitted()
        if X.ndim != 3:
            raise ValueError(f"Expected 3-D windowed input, got shape {X.shape}")
        n_samples = X.shape[0]
        out = np.zeros((n_samples, self._n_horizons), dtype=np.float32)
        for i in range(n_samples):
            out[i] = self._per_window_predict(X[i], self._n_horizons)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
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
            "level_days": self.level_days,
            "level_half_life_days": self.level_half_life_days,
            "scale_clip": self.scale_clip,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {
            "seasonal_period", "target_channel", "sequence_length",
            "level_days", "level_half_life_days", "scale_clip",
        }
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
                    "past_window_size": self._past_window_size,
                }, f)
            logger.info(f"Saved DailyProfile to {path}")
        except Exception as e:
            raise IOError(f"Failed to save model to {path}: {e}")

    def load(self, path: str) -> None:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.set_params(**data["params"])
            self._train_tail = data.get("train_tail")
            self._n_horizons = data.get("n_horizons", 1)
            self._past_window_size = data.get("past_window_size")
            self._is_fitted = True
            logger.info(f"Loaded DailyProfile from {path}")
        except Exception as e:
            raise IOError(f"Failed to load model from {path}: {e}")
