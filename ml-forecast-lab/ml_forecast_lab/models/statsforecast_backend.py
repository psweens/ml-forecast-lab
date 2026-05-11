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

    # Hard ceiling on the history length passed into the auto-search per
    # window, expressed as a multiplier on ``seasonal_period``. The stats
    # backends refit on every inference window, and AutoTheta in particular
    # is roughly O(n²) — feeding it the full 1024-sample ``train_history``
    # made each forecast take ~26 s on half-hourly daily data (so a single
    # CV fold of ~500 windows would not finish in any practical timeframe).
    # Empirically, a 4-period window (e.g. 192 samples for half-hourly
    # daily seasonality) keeps every backend below ~0.2 s per call:
    #   AutoARIMA constrained: ~0.5 s   AutoETS ZZA: ~0.16 s   AutoTheta: ~0.05 s
    # Forecast quality is indistinguishable from longer tails — the
    # auto-search converges on the same orders/parameters because the
    # seasonal structure is already fully expressed in 4 periods. Override
    # at the subclass level if a particular backend really benefits from
    # longer context (none of the current ones do).
    _max_history_periods: int = 4

    def __init__(
        self,
        seasonal_period: int = 48,
        train_history: int = 512,
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
        # Per-batch fallback bookkeeping. _forecast_single increments these
        # silently; predict_sequence / predict log a single summary line at
        # the end of each batch instead of one warning per failing window.
        self._fallback_count: int = 0
        self._fallback_first_error: Optional[str] = None

    @property
    def is_neural(self) -> bool:
        # Same rationale as SeasonalNaive — we want the benchmark pipeline to
        # forward sliding-window sequence data so the model can pull the
        # target series out of channel ``target_channel``.
        return True

    @property
    def model_family(self) -> str:
        return 'classical'

    def _make_model(self):  # pragma: no cover - subclass responsibility
        raise NotImplementedError

    def _cap_history(self, history: np.ndarray) -> np.ndarray:
        """Trim ``history`` to at most ``_max_history_periods`` seasons.

        The statsforecast auto-search runs once per inference window;
        cost scales super-linearly with history length (AutoTheta is
        roughly O(n²); AutoARIMA's stepwise grid scales with the number
        of candidate orders, which also grows with n). Capping the slice
        keeps each window's forecast below ~0.5 s.
        """
        cap = max(self._max_history_periods * self.seasonal_period, 32)
        if len(history) > cap:
            return history[-cap:]
        return history

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

        # Warm up statsforecast's numba JIT here so the first per-window
        # forecast call at inference time doesn't pay the ~25s compilation
        # cost (AutoTheta) / ~2s compilation cost (AutoARIMA) and trigger
        # the "pipeline is stuck" UX. JIT is module-level and persists for
        # the process lifetime, so subsequent fits reuse the compiled code
        # for free. Failures here are non-fatal — the worst case is the
        # first inference window pays the cost instead.
        self._warmup_jit()

        self._is_fitted = True
        logger.info(
            f"{self._model_kind} fit (tail-cache only): "
            f"period={self.seasonal_period}, tail={len(self._train_tail)}"
        )
        return {"time_seconds": 0.0, "epochs": 0, "best_val_loss": 0.0}

    def _warmup_jit(self) -> None:
        """Trigger numba JIT compilation up-front on a tiny synthetic series."""
        try:
            warmup_len = max(4 * self.seasonal_period, 32)
            t = np.linspace(0, 4 * np.pi, warmup_len)
            warmup_y = (np.sin(t) + 1.0).astype(np.float64)
            self._make_model().forecast(y=warmup_y, h=self.seasonal_period)
        except Exception as e:
            logger.debug(f"{self._model_kind} JIT warm-up skipped: {e}")

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
            # Don't log here — predict_sequence / predict aggregate and
            # emit a single summary at the end of each batch instead.
            self._fallback_count += 1
            if self._fallback_first_error is None:
                self._fallback_first_error = str(e)
            out = np.zeros(h, dtype=np.float32)
            for k in range(h):
                idx = -self.seasonal_period + k
                if -idx <= len(history):
                    out[k] = history[idx]
                else:
                    out[k] = history[-1]
            return out

    def _reset_fallback_counters(self) -> None:
        """Reset per-batch counters before predict_sequence / predict."""
        self._fallback_count = 0
        self._fallback_first_error = None

    def _log_fallback_summary(self, n_windows: int) -> None:
        """Emit a single summary line if any windows fell back this batch."""
        if self._fallback_count == 0:
            return
        logger.warning(
            f"{self._model_kind} fell back to seasonal-naive on "
            f"{self._fallback_count}/{n_windows} window(s); "
            f"first error: {self._fallback_first_error}"
        )

    def predict_sequence(self, X: np.ndarray) -> np.ndarray:
        self._validate_fitted()
        if X.ndim != 3:
            raise ValueError(f"Expected 3-D windowed input, got shape {X.shape}")
        n_samples = X.shape[0]
        out = np.zeros((n_samples, self._n_horizons), dtype=np.float32)
        self._reset_fallback_counters()
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
            out[i] = self._forecast_single(self._cap_history(history),
                                           self._n_horizons)
        self._log_fallback_summary(n_samples)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._validate_fitted()
        self._validate_X(X)
        X_seq = self._reshape_to_sequences(X)
        n_samples = X_seq.shape[0]
        out = np.zeros(n_samples, dtype=np.float32)
        self._reset_fallback_counters()
        for i in range(n_samples):
            target_window = X_seq[i, :, self.target_channel]
            if (self._train_tail is not None
                    and len(target_window) < 2 * self.seasonal_period):
                history = np.concatenate([self._train_tail, target_window])
            else:
                history = target_window
            preds = self._forecast_single(self._cap_history(history),
                                          max(self._n_horizons, 1))
            out[i] = preds[0]
        self._log_fallback_summary(n_samples)
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
        # The default AutoARIMA grid (max_p=5, max_q=5, max_P=2, max_Q=2,
        # nmodels=94, approximation=False) is impractically slow when
        # invoked per inference window — a single fit on half-hourly
        # data with season_length=48 routinely exceeds a minute, which
        # multiplied across CV windows is what made the benchmark
        # appear to hang on ARIMA before. The constraints below shrink
        # the candidate-order search and enable AIC approximation so
        # each per-window fit stays under ~1 s without materially
        # changing the chosen order on real HA series. The classical-
        # forecasting literature treats (p,q) ≤ (3,3) and (P,Q) ≤ (1,1)
        # as covering essentially all reasonable seasonal-ARIMA models
        # — relax these explicitly if you really need richer orders.
        return AutoARIMA(
            season_length=self.seasonal_period,
            max_p=2, max_q=2, max_P=1, max_Q=1,
            max_d=1, max_D=1,
            approximation=True,
            nmodels=10,
        )


class ETSModel(_StatsForecastBase):
    """AutoETS from statsforecast — Hyndman exponential smoothing state-space."""

    _model_kind = "ets"

    @property
    def name(self) -> str:
        return "ets"

    def _make_model(self):
        # 'ZZA' = auto error type, auto trend type, additive seasonality.
        # The fully-auto 'ZZZ' lets AutoETS pick *multiplicative* seasonality,
        # which is mathematically ill-defined on series containing zeros or
        # near-zero values — common for HA sensors (overnight energy demand,
        # solar at night, intermittent appliances). The optimiser bottoms out
        # with "Parameters out of range" and we fall back to seasonal-naive
        # for every single window. Locking seasonality to additive keeps the
        # auto-search useful while staying numerically well-defined on
        # zero-bearing data.
        return AutoETS(season_length=self.seasonal_period, model='ZZA')


class ThetaModel(_StatsForecastBase):
    """AutoTheta from statsforecast — Theta-method (M3/M4 winner family)."""

    _model_kind = "theta"

    @property
    def name(self) -> str:
        return "theta"

    def _make_model(self):
        return AutoTheta(season_length=self.seasonal_period)
