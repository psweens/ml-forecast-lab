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


class TestOutageAmplification:
    """A zero-filled outage must not inflate the forecast above anything the
    sensor has recorded.

    `scale = proj_total / ref_total`, and `ref_total` sums the most recent
    period. Zero-filling part of that day — a documented config via
    `idle_value: 0` for EV chargers and solar pumps — depresses the denominator
    while the surviving samples keep full magnitude, so the ratio inflates and
    the shipped `scale_clip` of 4.0 lets it reach 4x. Nothing downstream
    catches it: `_publish_forecast_sensors` has no upper clamp, and the
    log-inversion clamp only runs when `log_transform` is on.
    """

    @staticmethod
    def _peaked_day(period=48, peak=10.0):
        """A tidy daily shape: quiet overnight, a broad daytime hump."""
        t = np.arange(period)
        shape = np.clip(np.sin((t - 12) / period * np.pi * 2), 0, None)
        return shape / shape.sum() * peak * period / 10.0

    def _fit_model(self, days=14, period=48, zero_hours=0):
        day = self._peaked_day(period)
        series = np.tile(day, days).astype(np.float64)
        if zero_hours:
            n_zero = int(zero_hours * period / 24)
            series[-n_zero:] = 0.0            # outage at the tail of the window
        m = DailyProfileModel(seasonal_period=period)
        X = series.reshape(-1, 1)
        m.fit(X, series, n_horizons=period)
        return m, series

    @pytest.mark.parametrize("zero_hours", [0, 7, 10, 12, 16])
    def test_forecast_never_exceeds_the_observed_maximum(self, zero_hours):
        m, series = self._fit_model(zero_hours=zero_hours)
        window = series[-48:].reshape(-1, 1)
        out = m._per_window_predict(window, 48)
        observed_max = float(np.max(series))
        assert float(np.max(out)) <= observed_max + 1e-6, (
            f"with {zero_hours}h zero-filled the forecast peaked at "
            f"{float(np.max(out)):.3f}, above the observed maximum "
            f"{observed_max:.3f} — a value the sensor has never reported"
        )

    def test_signed_series_is_not_clipped(self):
        """The ceiling applies to non-negative sensors only; a signed sensor
        (net grid flow, say) must pass through untouched."""
        period = 48
        day = self._peaked_day(period) - 2.0        # straddles zero
        series = np.tile(day, 14).astype(np.float64)
        m = DailyProfileModel(seasonal_period=period)
        m.fit(series.reshape(-1, 1), series, n_horizons=period)
        out = m._per_window_predict(series[-48:].reshape(-1, 1), 48)
        assert np.isfinite(out).all()
