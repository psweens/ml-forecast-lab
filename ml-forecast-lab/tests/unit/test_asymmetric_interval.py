"""
Asymmetric prediction intervals for non-negative targets (v2.58.0).

Pins the defect behind the 2026-11 intermittent-demand investigation: the
published band was the point forecast ± one quantile of past absolute
errors, clipped at zero. On a non-negative right-skewed target that band
spends its lower half on values that cannot occur and stops short of the
large ones. The asymmetric band takes separate lower and upper offsets
from past signed errors made when the forecast was of similar size.

- ``level_conditioned_offsets`` groups errors by forecast size and falls
  back to pooled quantiles, or to nothing, when there are too few;
- ``interval_method: auto`` resolves by whether the target is non-negative;
- ``_conformal_bands`` asks for signed errors only when it will use them,
  contains the point forecast, clips at zero only where needed, and keeps
  the symmetric band until enough errors are calibrated;
- the calibration query returns the signed pairs on request and is
  otherwise unchanged, as is its replay key.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab import replay
from ml_forecast_lab.benchmark.horizon import level_conditioned_offsets
from ml_forecast_lab.config import ExperimentCfg
from ml_forecast_lab.db import HistoryDB
from ml_forecast_lab.main import MLForecastLabApp, _interval_is_asymmetric

INTERVAL = 30
LEVELS = [0.1, 0.9]


def _skewed_pairs(n=2000, seed=0):
    """Forecasts low at 'night' and high at 'peak'; errors small and
    one-sided for the first, large and right-skewed for the second."""
    rng = np.random.default_rng(seed)
    peak = rng.random(n) < 0.25
    predicted = np.where(peak, 3.0, 0.3) + rng.normal(0, 0.01, n)
    actual = np.where(
        peak,
        np.where(rng.random(n) < 0.5, rng.lognormal(1.6, 0.6, n), 0.2),
        0.1 * rng.poisson(2.0, n) + 0.1,
    )
    return predicted, actual - predicted


class TestLevelConditionedOffsets:
    def test_offsets_follow_the_size_of_the_forecast(self):
        predicted, residual = _skewed_pairs()
        off = level_conditioned_offsets(predicted, residual, np.array([0.3, 3.0]), LEVELS)
        night, peak = off
        assert (night[1] - night[0]) < 1.0
        assert (peak[1] - peak[0]) > 3.0
        # Right-skew at the peak: further to the upper edge than the lower.
        assert peak[1] > -peak[0]

    def test_pooled_quantiles_would_miss_both(self):
        predicted, residual = _skewed_pairs()
        pooled = np.quantile(residual, LEVELS)
        off = level_conditioned_offsets(predicted, residual, np.array([0.3, 3.0]), LEVELS)
        assert off[0, 1] < pooled[1] < off[1, 1]

    def test_too_few_errors_gives_nothing(self):
        assert level_conditioned_offsets(
            np.ones(10), np.zeros(10), np.array([1.0]), LEVELS,
        ) is None

    def test_a_constant_forecast_uses_the_pooled_quantiles(self):
        rng = np.random.default_rng(1)
        residual = rng.normal(0, 1, 500)
        off = level_conditioned_offsets(np.ones(500), residual, np.array([1.0, 1.0]), LEVELS)
        np.testing.assert_allclose(off[0], np.quantile(residual, LEVELS))
        np.testing.assert_allclose(off[0], off[1])

    def test_non_finite_pairs_are_ignored(self):
        predicted, residual = _skewed_pairs()
        dirty = residual.copy()
        dirty[::7] = np.nan
        off = level_conditioned_offsets(predicted, dirty, np.array([0.3]), LEVELS)
        assert np.isfinite(off).all()


def _exp(**kw):
    params = dict(name="e", target_entity="sensor.x", interval_minutes=INTERVAL,
                  future_periods=4, mode="production")
    params.update(kw)
    return ExperimentCfg(**params)


class TestMethodResolution:
    @pytest.mark.parametrize("kw,expected", [
        (dict(), False),
        (dict(source_is_cumulative=True), True),
        (dict(target_is_nonnegative=True), True),
        (dict(source_is_cumulative=True, interval_method="symmetric"), False),
        (dict(interval_method="asymmetric"), True),
    ])
    def test_auto_follows_non_negativity(self, kw, expected):
        assert _interval_is_asymmetric(_exp(**kw)) is expected

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="interval_method"):
            _exp(interval_method="lopsided")


class _DB:
    def __init__(self, response):
        self.calls, self.response = [], response

    def actuals_table_for(self, exp_cfg):
        return "t"

    def get_conformal_quantiles(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _future(n=4):
    start = pd.Timestamp(datetime.utcnow().replace(second=0, microsecond=0))
    return pd.DatetimeIndex(
        [start + pd.Timedelta(minutes=INTERVAL * (i + 1)) for i in range(n)]
    )


def _bands(exp, response, y_pred):
    app = MLForecastLabApp()
    app.history_db = _DB(response)
    out = asyncio.run(app._conformal_bands(
        exp, np.asarray(y_pred, dtype=np.float32), _future(len(y_pred)), "lightgbm", "v1",
    ))
    return out, app.history_db.calls


def _response(pairs=True):
    predicted, residual = _skewed_pairs()
    resp = {"quantiles": {}, "fallback_quantile": 1.0, "total_samples": len(predicted)}
    if pairs:
        resp["signed_residuals"] = np.column_stack([predicted, residual]).tolist()
    return resp


class TestBandConstruction:
    Y = [0.3, 3.0, 0.3, 3.0]

    def test_non_negative_target_gets_an_asymmetric_band(self):
        bands, calls = _bands(_exp(source_is_cumulative=True, reset_daily=True), _response(), self.Y)
        assert calls[0].get("signed") is True
        up = bands.upper - np.asarray(self.Y)
        down = np.asarray(self.Y) - bands.lower
        assert up[1] > down[1] + 0.5              # peak: skewed upwards
        assert (bands.upper - bands.lower)[1] > 3 * (bands.upper - bands.lower)[0]

    def test_band_contains_the_forecast_and_is_never_negative(self):
        bands, _ = _bands(_exp(source_is_cumulative=True, reset_daily=True), _response(), self.Y)
        assert (bands.lower <= np.asarray(self.Y, dtype=np.float32) + 1e-6).all()
        assert (bands.upper >= np.asarray(self.Y, dtype=np.float32) - 1e-6).all()
        assert (bands.lower >= 0).all()
        np.testing.assert_allclose(bands.q_vec, (bands.upper - bands.lower) / 2, rtol=1e-6)

    def test_signed_target_keeps_the_symmetric_band_and_its_query(self):
        bands, calls = _bands(_exp(), _response(pairs=False), self.Y)
        assert "signed" not in calls[0]
        np.testing.assert_allclose(bands.upper, np.asarray(self.Y) + 1.0, rtol=1e-6)
        np.testing.assert_allclose(bands.lower, np.asarray(self.Y) - 1.0, rtol=1e-6)

    def test_symmetric_can_be_forced_on_a_non_negative_target(self):
        exp = _exp(source_is_cumulative=True, reset_daily=True, interval_method="symmetric")
        bands, calls = _bands(exp, _response(pairs=False), self.Y)
        assert "signed" not in calls[0]
        np.testing.assert_allclose(bands.upper, np.asarray(self.Y) + 1.0, rtol=1e-6)
        np.testing.assert_allclose(bands.lower, np.maximum(np.asarray(self.Y) - 1.0, 0.0), atol=1e-6)

    def test_too_few_pairs_falls_back_to_the_symmetric_band(self):
        resp = {"quantiles": {}, "fallback_quantile": 1.0, "total_samples": 12,
                "signed_residuals": [[1.0, 0.1]] * 12}
        bands, _ = _bands(_exp(source_is_cumulative=True, reset_daily=True), resp, self.Y)
        np.testing.assert_allclose(bands.upper, np.asarray(self.Y) + 1.0, rtol=1e-6)


@pytest.fixture
def db_with_log(tmp_path):
    db = HistoryDB(str(tmp_path / "h.db"))
    db.ensure_forecast_log_table()
    now = pd.Timestamp(datetime.utcnow()).floor("30min")
    grid = pd.date_range(now - pd.Timedelta(days=3), now - pd.Timedelta(hours=1), freq="30min")
    rng = np.random.default_rng(0)
    actual = pd.Series(rng.gamma(1.0, 1.0, len(grid)), index=grid)
    table = db.safe_table_name("sensor.x")
    db.store_history(table, pd.DataFrame({"ds": grid, "value": actual.values}))
    issued = grid[0] - pd.Timedelta(minutes=30)
    db.log_forecast(
        "e", issued.to_pydatetime(), grid, np.full(len(grid), 0.8),
        model_name="lightgbm", model_version="v1",
    )
    return db, table, actual


class TestCalibrationQuery:
    def test_signed_pairs_are_returned_on_request_only(self, db_with_log):
        db, table, actual = db_with_log
        plain = db.get_conformal_quantiles("e", table, level=0.8, model_name="lightgbm")
        signed = db.get_conformal_quantiles(
            "e", table, level=0.8, model_name="lightgbm", signed=True,
        )
        assert "signed_residuals" not in plain
        pairs = np.asarray(signed.pop("signed_residuals"))
        assert signed == plain
        assert len(pairs) == plain["total_samples"] > 30
        np.testing.assert_allclose(pairs[:, 0], 0.8)
        assert (pairs[:, 1] < 0).any() and (pairs[:, 1] > 0).any()
        # Each pair is actual − predicted for a logged target.
        assert set(np.round(pairs[:, 1], 6)) <= set(np.round(actual.values - 0.8, 6))


class TestReplayKey:
    def test_symmetric_key_is_unchanged_and_signed_is_distinct(self):
        base = dict(level=0.8, model_name="lightgbm", model_version="v1",
                    interval_minutes=30)
        plain = replay._conformal_key("e", "t", **base)
        assert replay._conformal_key("e", "t", signed=False, **base) == plain
        assert "signed" not in plain
        assert replay._conformal_key("e", "t", signed=True, **base) != plain

    def test_signed_pairs_survive_the_bundle_codec(self):
        resp = {"quantiles": {30: 1.0}, "fallback_quantile": 1.0,
                "sample_counts": {30: 5}, "total_samples": 5, "level": 0.8,
                "signed_residuals": [[0.5, -0.25], [1.5, 2.0]]}
        assert replay._decode_conformal(replay._encode_conformal(resp)) == resp
