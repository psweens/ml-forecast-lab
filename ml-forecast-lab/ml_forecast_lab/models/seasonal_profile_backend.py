"""
Seasonal profile forecasting backend for ML Forecast Lab.

The forecast for an interval is the mean of the target at the same time of
day on the same kind of day — weekday or weekend — over the training
window:

    ŷ[t] = mean(y[s] : slot_of_day(s) = slot_of_day(t), weekend(s) = weekend(t))

Why it is a backend and not only a reference line. On an intermittent
target — mostly near zero, with events at habitual times of day — whether
an event falls in a given interval is close to unpredictable, and what can
be known in advance is how often and how large events are at that time of
day. That is this profile. With a few weeks of history a learned model has
to recover the same table from a few dozen examples per interval while
also fitting lags that carry almost nothing, and on such targets it
typically ends with a shallower profile and a lower level. When no learned
model beats the profile the profile is the right model to publish, and the
leaderboard can only say so if it is on it.

It differs from Seasonal Naive, which repeats the most recent day (one
sample per interval, so every event of yesterday is forecast again today),
and from Daily Profile, which scales yesterday's shape to a projected
total.

The backend is tabular: it reads ``slot_of_day`` (or ``hour_of_day`` on an
hourly grid) and ``is_weekend`` from the feature matrix, which are computed
in the configured timezone, and ignores every other column. Its recursive
multi-step forecast is therefore the same as its one-step forecast. A slot
with no examples for the day type falls back to the slot across both day
types, then to the overall mean.
"""

import logging
import pickle
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

from .base import ForecastModel

logger = logging.getLogger(__name__)


class SeasonalProfileModel(ForecastModel):
    """Mean of the target by time of day and weekday/weekend."""

    SLOT_COLUMNS = ("slot_of_day", "hour_of_day")
    DAY_TYPE_COLUMN = "is_weekend"

    def __init__(self, split_weekend: bool = True) -> None:
        super().__init__()
        self.split_weekend = bool(split_weekend)
        self.feature_names_: Optional[list] = None
        self._slot_col: Optional[int] = None
        self._day_col: Optional[int] = None
        self._by_slot_day: Dict[tuple, float] = {}
        self._by_slot: Dict[int, float] = {}
        self._overall: float = 0.0
        self.training_metadata: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "seasonal_profile"

    @property
    def model_family(self) -> str:
        return "baseline"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        self._validate_X(X_train)
        y = self._validate_y(y_train)
        if len(X_train) != len(y):
            raise ValueError("X and y must have the same number of samples")
        names = kwargs.get("feature_names")
        if names is None:
            raise ValueError(
                "seasonal_profile needs feature_names to find its time-of-day "
                "column"
            )
        names = list(names)
        slot_name = next((c for c in self.SLOT_COLUMNS if c in names), None)
        if slot_name is None:
            raise ValueError(
                f"seasonal_profile needs one of {self.SLOT_COLUMNS} among the "
                f"features"
            )
        self.feature_names_ = names
        self._slot_col = names.index(slot_name)
        self._day_col = (
            names.index(self.DAY_TYPE_COLUMN)
            if self.split_weekend and self.DAY_TYPE_COLUMN in names else None
        )

        slots = np.rint(X_train[:, self._slot_col]).astype(int)
        days = (
            np.rint(X_train[:, self._day_col]).astype(int)
            if self._day_col is not None else np.zeros(len(y), dtype=int)
        )
        ok = np.isfinite(y)
        self._overall = float(np.mean(y[ok])) if ok.any() else 0.0
        self._by_slot = {
            int(s): float(np.mean(y[ok & (slots == s)]))
            for s in np.unique(slots[ok])
        }
        self._by_slot_day = {}
        for s in np.unique(slots[ok]):
            for d in np.unique(days[ok & (slots == s)]):
                sel = ok & (slots == s) & (days == d)
                self._by_slot_day[(int(s), int(d))] = float(np.mean(y[sel]))

        self.training_metadata = {
            "slot_column": slot_name,
            "split_weekend": self._day_col is not None,
            "n_slots": len(self._by_slot),
            "n_groups": len(self._by_slot_day),
        }
        self._is_fitted = True
        logger.info(
            f"SeasonalProfile fit: {len(self._by_slot_day)} groups over "
            f"{len(self._by_slot)} slots ({slot_name}"
            f"{' × weekend' if self._day_col is not None else ''})"
        )
        return {"time_seconds": 0.0, **self.training_metadata}

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._validate_fitted()
        self._validate_X(X)
        if X.shape[1] != len(self.feature_names_):
            raise ValueError(
                f"Expected {len(self.feature_names_)} features, but got {X.shape[1]}"
            )
        slots = np.rint(X[:, self._slot_col]).astype(int)
        days = (
            np.rint(X[:, self._day_col]).astype(int)
            if self._day_col is not None else np.zeros(len(X), dtype=int)
        )
        out = np.empty(len(X), dtype=np.float32)
        for i, (s, d) in enumerate(zip(slots.tolist(), days.tolist())):
            out[i] = self._by_slot_day.get(
                (s, d), self._by_slot.get(s, self._overall),
            )
        return out

    def get_params(self) -> Dict[str, Any]:
        return deepcopy({"split_weekend": self.split_weekend})

    def set_params(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            # Experiment-level settings applied to every backend (loss,
            # patience) mean nothing to a model with no optimisation.
            if key in ("loss_fn", "patience"):
                continue
            if key != "split_weekend":
                raise ValueError(f"Unknown parameter: {key}")
            self.split_weekend = bool(value)

    def save(self, path: str) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "feature_names": self.feature_names_,
                    "slot_col": self._slot_col,
                    "day_col": self._day_col,
                    "by_slot_day": self._by_slot_day,
                    "by_slot": self._by_slot,
                    "overall": self._overall,
                    "training_metadata": self.training_metadata,
                    "hyperparameters": self.get_params(),
                }, f)
        except Exception as e:
            raise IOError(f"Failed to save model to {path}: {e}")

    def load(self, path: str) -> None:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.feature_names_ = data["feature_names"]
            self._slot_col = data["slot_col"]
            self._day_col = data["day_col"]
            self._by_slot_day = data["by_slot_day"]
            self._by_slot = data["by_slot"]
            self._overall = data["overall"]
            self.training_metadata = data.get("training_metadata", {})
            self.set_params(**data.get("hyperparameters", {}))
            self._is_fitted = True
        except Exception as e:
            raise IOError(f"Failed to load model from {path}: {e}")
