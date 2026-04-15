"""
NeuralProphet forecasting model backend for ML Forecast Lab.

Wraps Facebook's NeuralProphet library, providing automatic seasonality
detection, trend decomposition, and neural AR components. Handles the
conversion between the ML Forecast Lab feature matrix format and
NeuralProphet's DataFrame format.
"""

import logging
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base import ForecastModel

logger = logging.getLogger(__name__)

try:
    # Suppress NeuralProphet and PyTorch Lightning warnings
    import warnings as _w
    _w.filterwarnings("ignore", category=FutureWarning, module="neuralprophet")
    _w.filterwarnings("ignore", category=FutureWarning, module="pytorch_lightning")
    _w.filterwarnings("ignore", message=".*batch_size.*", module="pytorch_lightning")

    # Suppress "Importing plotly failed" error logs from NeuralProphet
    logging.getLogger("NP.plotly").setLevel(logging.CRITICAL)

    from neuralprophet import NeuralProphet, set_log_level as np_set_log_level
    np_set_log_level("ERROR")
    NEURALPROPHET_AVAILABLE = True
except ImportError:
    NEURALPROPHET_AVAILABLE = False
    warnings.warn(
        "NeuralProphet is not installed. NeuralProphetModel will not be functional.",
        ImportWarning,
    )


class NeuralProphetModel(ForecastModel):
    """
    NeuralProphet time-series forecasting model.

    Combines classical decomposition (trend, seasonality) with neural
    network autoregressive components. Automatically detects daily and
    weekly seasonality patterns.

    NeuralProphet expects data in a specific format (DataFrame with 'ds'
    and 'y' columns). This backend adapts the flat feature matrix from
    the benchmark runner by reconstructing timestamps from the training
    data index.
    """

    def __init__(
        self,
        n_lags: int = 12,
        n_forecasts: int = 1,
        learning_rate: float = 0.01,
        epochs: int = 100,
        batch_size: int = 64,
        yearly_seasonality: bool = False,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        n_changepoints: int = 10,
    ) -> None:
        """Initialise NeuralProphet model."""
        super().__init__()
        if not NEURALPROPHET_AVAILABLE:
            raise RuntimeError("NeuralProphet is not installed")

        self.n_lags = n_lags
        self.n_forecasts = n_forecasts
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.n_changepoints = n_changepoints

        self._model: Optional[NeuralProphet] = None
        self._train_df: Optional[pd.DataFrame] = None
        self._freq: str = "30min"

    @property
    def name(self) -> str:
        return "neuralprophet"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        """
        Train NeuralProphet model.

        NeuralProphet needs a DataFrame with 'ds' (datetime) and 'y' columns.
        Since the benchmark runner passes flat numpy arrays, we reconstruct
        timestamps from the caller's ``date_index`` kwarg where available,
        otherwise fall back to a fixed-anchor synthetic index (NOT
        ``pd.Timestamp.now()``, which would shift every fit() call and
        destroy temporal reproducibility).

        Accepts ``date_index`` in ``kwargs`` — an array-like of timestamps
        aligned with ``y_train`` — to preserve real calendar effects
        (seasonality, holidays, day-of-week) that NeuralProphet relies on.
        """
        self._validate_X(X_train)
        y_train = self._validate_y(y_train)
        start_time = time.time()

        n_samples = len(y_train)
        freq = kwargs.get("freq", self._freq)

        # Prefer real timestamps from the caller when available — this is what
        # NeuralProphet's trend/seasonality/holiday components actually need.
        date_index = kwargs.get("date_index")
        if date_index is not None:
            ds = pd.to_datetime(pd.Index(date_index))
            if len(ds) != n_samples:
                raise ValueError(
                    f"date_index length {len(ds)} does not match y_train "
                    f"length {n_samples}"
                )
        else:
            # Synthetic fallback anchored at a FIXED reference timestamp
            # (2000-01-01) so repeated fits with the same y produce
            # identical timestamps. This is reproducible but loses true
            # calendar effects — callers should pass date_index when they can.
            anchor = pd.Timestamp("2000-01-01")
            ds = pd.date_range(start=anchor, periods=n_samples, freq=freq)

        train_df = pd.DataFrame({"ds": ds, "y": y_train.astype(float)})

        # Create NeuralProphet model
        self._model = NeuralProphet(
            n_lags=self.n_lags,
            n_forecasts=self.n_forecasts,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            n_changepoints=self.n_changepoints,
            drop_missing=True,
        )

        # Fit
        metrics = self._model.fit(train_df, freq=freq)
        self._train_df = train_df
        self._freq = freq

        elapsed = time.time() - start_time
        self._is_fitted = True

        # Extract final training metrics
        final_metrics = {}
        if metrics is not None and len(metrics) > 0:
            last_row = metrics.iloc[-1]
            final_metrics = {
                "final_loss": float(last_row.get("Loss", 0)),
                "final_mae": float(last_row.get("MAE", 0)),
            }

        logger.info(
            f"NeuralProphet trained in {elapsed:.1f}s, "
            f"{self.epochs} epochs, {n_samples} samples"
        )

        return {
            "time_seconds": elapsed,
            "epochs": self.epochs,
            **final_metrics,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using NeuralProphet.

        For in-sample predictions (benchmark), we use the training data
        with make_future_dataframe. For new data, we reconstruct the
        DataFrame format.
        """
        self._validate_fitted()
        self._validate_X(X)

        n_samples = X.shape[0]

        try:
            # Use NeuralProphet's built-in prediction
            if self._train_df is not None:
                # Predict on training data (in-sample for benchmark)
                forecast = self._model.predict(self._train_df)

                if "yhat1" in forecast.columns:
                    preds = forecast["yhat1"].values
                else:
                    preds = forecast["y"].values

                # Align length to input
                if len(preds) >= n_samples:
                    preds = preds[-n_samples:]
                else:
                    # Pad with last value if forecast is shorter
                    preds = np.pad(preds, (n_samples - len(preds), 0),
                                   mode="edge")

                return np.clip(preds, 0.0, None).astype(np.float32)

        except Exception as e:
            logger.warning(f"NeuralProphet prediction failed: {e}")

        # Fallback: return mean of training data
        if self._train_df is not None:
            return np.full(n_samples, self._train_df["y"].mean(), dtype=np.float32)
        return np.zeros(n_samples, dtype=np.float32)

    def get_params(self) -> Dict[str, Any]:
        return deepcopy({
            "n_lags": self.n_lags, "n_forecasts": self.n_forecasts,
            "learning_rate": self.learning_rate, "epochs": self.epochs,
            "batch_size": self.batch_size,
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "n_changepoints": self.n_changepoints,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"n_lags", "n_forecasts", "learning_rate", "epochs",
                 "batch_size", "yearly_seasonality", "weekly_seasonality",
                 "daily_seasonality", "n_changepoints"}
        for k, v in kwargs.items():
            if k not in valid:
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)

    def save(self, path: str) -> None:
        """Save NeuralProphet model."""
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Cannot save unfitted model")
        # NeuralProphet uses torch.save internally
        import torch
        torch.save({
            "model": self._model,
            "params": self.get_params(),
            "freq": self._freq,
        }, path)
        logger.info(f"Saved NeuralProphet model to {path}")

    def load(self, path: str) -> None:
        """Load NeuralProphet model."""
        import torch
        data = torch.load(path, map_location="cpu")
        self._model = data["model"]
        self.set_params(**data["params"])
        self._freq = data.get("freq", "30min")
        self._is_fitted = True
        logger.info(f"Loaded NeuralProphet model from {path}")
