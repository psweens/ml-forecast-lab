"""
statsforecast classical baseline backends for ML Forecast Lab.

Wraps Nixtla's ``statsforecast`` library to expose three classical baselines
that the academic forecasting literature treats as mandatory comparison
references:

- ``arima`` — AutoARIMA (auto seasonal-ARIMA selection via AIC)
- ``ets``   — AutoETS (Hyndman exponential smoothing state-space model)
- ``theta`` — AutoTheta (decomposition-based forecasting; M3/M4 winner family)

statsforecast's implementations are numba-JIT-compiled and parallelisable, so
they remain genuinely lightweight even on the smaller HA series. They're
included as separate backends rather than a single switchable one so each
shows up independently in the Demšar ranking.

Each model treats the target series as univariate — covariate channels of
the input window are ignored. The fit consumes only the most recent
``train_history`` samples (default 1024) to keep AutoARIMA's order search
tractable on long histories.
"""

import logging
import pickle
import warnings
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

from .base import ForecastModel

logger = logging.getLogger(__name__)

try:
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta
    STATSFORECAST_AVAILABLE = True
except ImportError:
    STATSFORECAST_AVAILABLE = False
    AutoARIMA = AutoETS = AutoTheta = None  # type: ignore[assignment]
    warnings.warn(
        "statsforecast is not installed. ARIMA/ETS/Theta backends will not be "
        "functional. Install it with: pip install statsforecast",
        ImportWarning,
    )


class _StatsForecastBase(ForecastModel):
    """
    Common implementation for statsforecast univariate baselines.

    Subclasses set ``_model_kind`` and implement ``_make_model()``. The
    fit/predict pipeline mirrors the SeasonalNaive backend: only the
    target's last training-tail values are kept, and predictions are
    produced by calling the underlying statsforecast model's
    ``forecast(h=...)`` per inference row.
    """

    _model_kind: str = "stats"

    def __init__(
        self,
        seasonal_period: int = 48,
        train_history: int = 1024,
        target_channel: int = 0,
        sequence_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not STATSFORECAST_AVAILABLE:
            raise RuntimeError(
                "statsforecast is not installed. "
                "Install with: pip install statsforecast"
            )
        if seasonal_period < 1:
            raise ValueError(f"seasonal_period must be >= 1, got {seasonal_period}")
        self.seasonal_period = int(seasonal_period)
        self.train_history = int(train_history)
        self.target_channel = int(target_channel)
        self.sequence_length = sequence_length

        self._train_tail: Optional[np.ndarray] = None
        self._n_horizons: int = 1

    @property
    def is_neural(self) -> bool:
        # Same rationale as SeasonalNaive — we want the benchmark pipeline to
        # forward sliding-window sequence data so the model can pull the
        # target series out of channel ``target_channel``.
        return True

    def _make_model(self):  # pragma: no cover - subclass responsibility
        raise NotImplementedError

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
        """Cache the most recent target-series tail for later forecasting.

        statsforecast models are refit per inference window (each window
        provides its own context), so this method does not actually train
        anything — it only records ``n_horizons`` so ``predict_sequence``
        knows how far to forecast and stashes the training tail for
        fallback when input windows are too short.
        """
        self._validate_X(X_train)
        y_train = self._validate_y(y_train)
        if y_train.ndim == 2 and y_train.shape[1] > 1:
            self._n_horizons = y_train.shape[1]
            tail_source = y_train[:, 0]
        else:
            self._n_horizons = 1
            tail_source = y_train.ravel()

        keep = max(self.train_history, 4 * self.seasonal_period, 16)
        self._train_tail = np.asarray(tail_source[-keep:], dtype=np.float32)

        self._is_fitted = True
        logger.info(
            f"{self._model_kind} fit (tail-cache only): "
            f"period={self.seasonal_period}, tail={len(self._train_tail)}"
        )
        return {"time_seconds": 0.0, "epochs": 0, "best_val_loss": 0.0}

    def _forecast_single(self, history: np.ndarray, h: int) -> np.ndarray:
        """Fit the underlying statsforecast model on ``history`` and forecast h steps."""
        try:
            model = self._make_model()
            # statsforecast model.forecast returns dict with 'mean' key.
            res = model.forecast(y=np.asarray(history, dtype=np.float64), h=h)
            mean = res["mean"] if isinstance(res, dict) else res
            return np.asarray(mean, dtype=np.float32)
        except Exception as e:
            # Classical models can fail to converge on short or constant
            # series; fall back to seasonal-naive so the benchmark row is
            # still populated rather than crashing the whole pipeline.
            logger.warning(
                f"{self._model_kind} forecast failed ({e}); "
                f"falling back to seasonal-naive for this window"
            )
            out = np.zeros(h, dtype=np.float32)
            for k in range(h):
                idx = -self.seasonal_period + k
                if -idx <= len(history):
                    out[k] = history[idx]
                else:
                    out[k] = history[-1]
            return out

    def predict_sequence(self, X: np.ndarray) -> np.ndarray:
        self._validate_fitted()
        if X.ndim != 3:
            raise ValueError(f"Expected 3-D windowed input, got shape {X.shape}")
        n_samples = X.shape[0]
        out = np.zeros((n_samples, self._n_horizons), dtype=np.float32)
        for i in range(n_samples):
            target_window = X[i, :, self.target_channel]
            # Stitch the cached training tail in front of the window if the
            # window alone is shorter than two seasonal periods — gives the
            # auto-order search enough data to pick something sensible.
            if (self._train_tail is not None
                    and len(target_window) < 2 * self.seasonal_period):
                history = np.concatenate([self._train_tail, target_window])
            else:
                history = target_window
            out[i] = self._forecast_single(history, self._n_horizons)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._validate_fitted()
        self._validate_X(X)
        X_seq = self._reshape_to_sequences(X)
        out = np.zeros(X_seq.shape[0], dtype=np.float32)
        for i in range(X_seq.shape[0]):
            target_window = X_seq[i, :, self.target_channel]
            if (self._train_tail is not None
                    and len(target_window) < 2 * self.seasonal_period):
                history = np.concatenate([self._train_tail, target_window])
            else:
                history = target_window
            preds = self._forecast_single(history, max(self._n_horizons, 1))
            out[i] = preds[0]
        return out

    def get_params(self) -> Dict[str, Any]:
        return deepcopy({
            "seasonal_period": self.seasonal_period,
            "train_history": self.train_history,
            "target_channel": self.target_channel,
            "sequence_length": self.sequence_length,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"seasonal_period", "train_history",
                 "target_channel", "sequence_length"}
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
                    "model_kind": self._model_kind,
                }, f)
            logger.info(f"Saved {self._model_kind} to {path}")
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
            logger.info(f"Loaded {self._model_kind} from {path}")
        except Exception as e:
            raise IOError(f"Failed to load model from {path}: {e}")


class ARIMAModel(_StatsForecastBase):
    """AutoARIMA from statsforecast — auto seasonal-ARIMA via AIC search."""

    _model_kind = "arima"

    @property
    def name(self) -> str:
        return "arima"

    def _make_model(self):
        return AutoARIMA(season_length=self.seasonal_period)


class ETSModel(_StatsForecastBase):
    """AutoETS from statsforecast — Hyndman exponential smoothing state-space."""

    _model_kind = "ets"

    @property
    def name(self) -> str:
        return "ets"

    def _make_model(self):
        # 'ZZZ' lets AutoETS pick error/trend/seasonal types automatically.
        return AutoETS(season_length=self.seasonal_period, model='ZZZ')


class ThetaModel(_StatsForecastBase):
    """AutoTheta from statsforecast — Theta-method (M3/M4 winner family)."""

    _model_kind = "theta"

    @property
    def name(self) -> str:
        return "theta"

    def _make_model(self):
        return AutoTheta(season_length=self.seasonal_period)
