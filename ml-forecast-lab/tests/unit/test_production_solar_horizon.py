"""Production-inference tree forecast held solar features flat over the horizon.

Found against v2.52.3. ``_run_production_inference`` built each recursive
step's feature row from ``last_cov_vals`` for every column of the fetched
frame, ``sun_elevation`` and ``clear_sky_ghi`` included. Neither has a
covariate config entry, so neither reached ``future_cov_values``, and every
horizon step carried the value at ``last_ts``: a forecast made at night told
the tree it stayed night for the whole day ahead, and the
``<col>_x_hour_sin/cos`` interactions inherited the same frozen value.
``_forecast_with_cached`` already computed both from pvlib for the forecast
grid; both loops now share ``_future_solar_frame``.

The defect was latent: ``_run_production_inference`` is reached only through
``update_experiment(..., is_lab_mode=False)`` from ``_run_update_cycle``,
which nothing schedules — production retrains go through
``_retrain_and_cache``. These tests pin the path so it cannot drift from the
cached one if it is reconnected.

Both columns are deterministic given site and timestamp; each step must carry
its own pvlib value, matching the cached path row for row.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab import main as main_mod
from ml_forecast_lab.config import AppConfig, ExperimentCfg
from ml_forecast_lab.covariates import CovariateResolver
from ml_forecast_lab.db import HistoryDB
from ml_forecast_lab.main import MLForecastLabApp
from ml_forecast_lab.solar_physics import compute_solar_features

pytest.importorskip("lightgbm")
pytest.importorskip("pvlib")

from ml_forecast_lab.models.lightgbm_backend import LightGBMModel  # noqa: E402

LAT, LON = 52.2, 0.12
INTERVAL = 30
FUTURE_PERIODS = 48  # 24 h: elevation crosses the horizon at this latitude
SOLAR_COLS = ("sun_elevation", "clear_sky_ghi")


def _run(coro):
    return asyncio.run(coro)


class _StubHA:
    def __init__(self, rows_by_entity):
        self.rows_by_entity = rows_by_entity

    async def get_history(self, entity_id, start, end, include_attributes=False):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if s.tzinfo is None:
            s = s.tz_localize("UTC")
        if e.tzinfo is None:
            e = e.tz_localize("UTC")
        return [
            r for r in self.rows_by_entity.get(entity_id, [])
            if s <= pd.Timestamp(r["last_changed"]) <= e
        ]

    async def get_state(self, entity_id, default=None, attribute=None):
        return default

    async def get_config(self):
        return {"latitude": LAT, "longitude": LON, "time_zone": "UTC"}


def _recorder_rows(entity_now, days, cadence_min=15,
                   value=lambda i: i % 37 + 1.0):
    ts = pd.date_range(
        entity_now - timedelta(days=days), entity_now,
        freq=f"{cadence_min}min", tz="UTC",
    )
    return [
        {"last_changed": t.isoformat(), "state": f"{value(i):.4f}"}
        for i, t in enumerate(ts)
    ]


class _Registry:
    def create(self, name):
        return LightGBMModel()


def _make_app(tmp_db, exp_cfg, monkeypatch):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    app = MLForecastLabApp()
    app.history_db = HistoryDB(tmp_db)
    app.ha_interface = _StubHA({
        exp_cfg.target_entity: _recorder_rows(now, days=11),
    })
    app.config = AppConfig(experiments=[exp_cfg])
    app.covariate_resolver = CovariateResolver(
        app.ha_interface,
        history_db=app.history_db,
        retention_provider=app._retention_days_for_table,
    )
    app.model_registry = _Registry()

    published: list[dict] = []

    async def _publish(**kw):
        published.append(kw)

    async def _noop(*a, **kw):
        return None

    monkeypatch.setattr(app, "_publish_forecast_sensors", _publish)
    monkeypatch.setattr(app, "_publish_lifecycle_sensor", _noop)
    monkeypatch.setattr(app, "_persist_cached_model", lambda *a, **kw: None)

    rows: list[dict] = []
    real_guard = main_mod._nan_to_num_guarded

    def _guard(X, label, cols=None, *a, **kw):
        if "recursive forecast row" in label and cols is not None:
            rows.append(dict(zip(cols, np.asarray(X, dtype=float).ravel())))
        return real_guard(X, label, cols, *a, **kw)

    monkeypatch.setattr(main_mod, "_nan_to_num_guarded", _guard)
    return app, published, rows


def _exp(**kw):
    params = dict(
        name="pv", target_entity="sensor.pv_power", days_history=10,
        interval_minutes=INTERVAL, future_periods=FUTURE_PERIODS,
        models_enabled=["lightgbm"], production_model="lightgbm",
        include_sun_elevation=True, include_clear_sky_irradiance=True,
    )
    params.update(kw)
    return ExperimentCfg(**params)


def _expected_solar(ds_future, cols):
    return compute_solar_features(
        pd.DatetimeIndex(ds_future), latitude=LAT, longitude=LON,
        include_elevation="sun_elevation" in cols,
        include_clear_sky="clear_sky_ghi" in cols,
    )


def _assert_rows_follow_pvlib(rows, ds_future, cols):
    assert len(rows) == FUTURE_PERIODS
    expected = _expected_solar(ds_future, cols)
    for col in cols:
        got = np.array([r[col] for r in rows])
        want = expected[col].to_numpy(dtype=float)
        assert np.ptp(want) > 1.0, f"{col} must vary over the horizon"
        np.testing.assert_allclose(
            got, want.astype(np.float32), rtol=1e-5, atol=1e-3,
            err_msg=f"{col} does not advance per horizon step",
        )
        hour_sin = np.array([r["hour_sin"] for r in rows])
        hour_cos = np.array([r["hour_cos"] for r in rows])
        np.testing.assert_allclose(
            [r[f"{col}_x_hour_sin"] for r in rows], got * hour_sin,
            rtol=1e-4, atol=1e-3,
        )
        np.testing.assert_allclose(
            [r[f"{col}_x_hour_cos"] for r in rows], got * hour_cos,
            rtol=1e-4, atol=1e-3,
        )


class TestProductionInferenceSolarHorizon:
    @pytest.mark.parametrize("cols", [
        ("sun_elevation", "clear_sky_ghi"),
        ("sun_elevation",),
        ("clear_sky_ghi",),
    ])
    def test_solar_features_advance_per_step(self, tmp_db, monkeypatch, cols):
        exp = _exp(
            include_sun_elevation="sun_elevation" in cols,
            include_clear_sky_irradiance="clear_sky_ghi" in cols,
        )
        app, published, rows = _make_app(tmp_db, exp, monkeypatch)
        _run(app._run_production_inference(exp))

        assert len(published) == 1
        for col in SOLAR_COLS:
            assert (col in rows[0]) == (col in cols)
        _assert_rows_follow_pvlib(rows, published[0]["ds_future"], cols)

    def test_matches_cached_forecast_rows(self, tmp_db, monkeypatch):
        """The post-retrain forecast and the next cached tick see the same
        solar inputs for the same horizon."""
        exp = _exp()
        app, published, rows = _make_app(tmp_db, exp, monkeypatch)
        _run(app._run_production_inference(exp))
        prod_rows = list(rows)
        rows.clear()
        _run(app._retrain_and_cache(exp))

        assert list(published[0]["ds_future"]) == list(published[1]["ds_future"])
        for col in SOLAR_COLS:
            np.testing.assert_array_equal(
                [r[col] for r in prod_rows], [r[col] for r in rows],
            )
