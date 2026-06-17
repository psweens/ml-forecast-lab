"""Unit tests for the DailyProfile (total-reconciled seasonal-naive) backend.

The invariants that matter:
  * with a stable daily level the scale is ~1 and it reproduces Seasonal-Naive
    (so it never *hurts* on a flat-level series);
  * with a trending level it scales the recent day's shape toward the projected
    total (the day-level amplitude it adds over Seasonal-Naive);
  * the scale is clamped so a near-zero reference day can't blow up.
"""

import numpy as np
import pytest

from ml_forecast_lab.models.daily_profile_backend import DailyProfileModel
from ml_forecast_lab.models.seasonal_naive_backend import SeasonalNaiveModel


PERIOD = 4
DAY = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)  # one day's shape, sum 10


def _fit(model, series, n_horizons=PERIOD):
    """Fit a windowed baseline on a 1-D series via a dummy windowed y_train."""
    n = len(series)
    # y_train[:, 0] is the contiguous series the tail is cached from.
    y_train = np.zeros((n, n_horizons), dtype=np.float32)
    y_train[:, 0] = series
    X_train = np.zeros((n, 8), dtype=np.float32)
    model.fit(X_train, y_train)
    return model


def _window(series_tail):
    """One (1, seq_len, 1) window from a 1-D tail."""
    return np.asarray(series_tail, dtype=np.float32).reshape(1, -1, 1)


def test_reduces_to_seasonal_naive_on_flat_level():
    series = np.tile(DAY, 8)  # 8 identical days → stable total
    dp = _fit(DailyProfileModel(seasonal_period=PERIOD, level_days=5), series)
    sn = _fit(SeasonalNaiveModel(seasonal_period=PERIOD), series)

    win = _window(np.tile(DAY, 3))  # 3-day window
    dp_pred = dp.predict_sequence(win)
    sn_pred = sn.predict_sequence(win)
    np.testing.assert_allclose(dp_pred, sn_pred, rtol=1e-5, atol=1e-5)


def test_scales_up_when_reference_day_below_recent_level():
    # Typical level is ~20/day; the window ends on smaller-than-typical days,
    # so the projected total (toward the bigger recent level) exceeds the
    # reference (look-back) day → the seasonal-naive shape is scaled UP.
    big = np.tile(DAY * 2.0, 8)  # typical daily total ~20
    dp = _fit(DailyProfileModel(seasonal_period=PERIOD, level_days=6,
                                level_half_life_days=3.0), big)
    sn = _fit(SeasonalNaiveModel(seasonal_period=PERIOD), big)

    win = _window(np.tile(DAY, 3))  # window ends on small days (sum 10)
    dp_pred = dp.predict_sequence(win)[0]
    sn_pred = sn.predict_sequence(win)[0]
    # Same shape, scaled up: the day's total is pulled toward the recent level.
    nz = sn_pred > 0
    assert np.any(nz)
    assert np.all(dp_pred[nz] >= sn_pred[nz] - 1e-6)
    assert dp_pred[nz].sum() > sn_pred[nz].sum() + 1e-6


def test_scale_is_clamped_on_near_zero_reference_day():
    # A near-zero recent day (reference) followed by a normal projection must
    # not blow the forecast up beyond the configured clip.
    series = np.concatenate([
        np.tile(DAY, 5),                 # normal days
        np.zeros(PERIOD, dtype=np.float32) + 1e-4,  # a near-zero "reference" day
    ])
    dp = _fit(DailyProfileModel(seasonal_period=PERIOD, scale_clip=3.0), series)
    win = _window(np.concatenate([np.tile(DAY, 2),
                                  np.zeros(PERIOD) + 1e-4]))
    pred = dp.predict_sequence(win)[0]
    assert np.all(np.isfinite(pred))
    # Bounded by scale_clip × the look-back magnitude (max day value 4).
    assert pred.max() <= 4.0 * 3.0 + 1e-3


def test_registered_and_save_load_roundtrip(tmp_path):
    from ml_forecast_lab.models.registry import get_registry
    # Register like the app does and create through the registry.
    reg = get_registry()
    reg.register("daily_profile", DailyProfileModel)
    model = reg.create("daily_profile", seasonal_period=PERIOD)
    assert model.name == "daily_profile"
    assert model.is_neural is True
    assert model.model_family == "baseline"

    series = np.tile(DAY, 8)
    _fit(model, series)
    win = _window(np.tile(DAY, 3))
    before = model.predict_sequence(win)

    path = str(tmp_path / "dp.pkl")
    model.save(path)
    fresh = DailyProfileModel()
    fresh.load(path)
    after = fresh.predict_sequence(win)
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)
