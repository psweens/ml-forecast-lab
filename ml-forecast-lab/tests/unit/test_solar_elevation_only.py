"""v2.52.3: tree recursive forecast with ``sun_elevation`` but no ``clear_sky_ghi``.

``compute_solar_features`` emits only the columns it is asked for. Both tree
recursive-forecast loops — ``_compute_cached_forecast`` (behind
``_forecast_with_cached``) and ``_run_production_inference`` — read
``clear_sky_ghi`` out of their future solar frame to decide whether to push
0 into the lag buffer at night. An experiment with
``include_sun_elevation: true`` and ``include_clear_sky_irradiance: false``
handed the cached path a frame holding ``sun_elevation`` alone, the lookup
raised ``KeyError``, and every forecast tick — including the immediate
post-retrain one — failed.

The gate must follow training. ``build_features`` gates lags on clear-sky GHI
only when that column is in the frame, so an elevation-only model was fitted
on ungated lags and its recursive buffer must stay ungated: each step's
prediction becomes the next step's ``y_lag_1`` verbatim, night or day. The
clear-sky cases pin the other half — with GHI present, night steps still push
0 — so the fix cannot pass by disabling the gate outright.
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
FUTURE_PERIODS = 48  # 24 h: always spans a night at this latitude


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
        # _get_site_location reads these; without them no solar column is
        # computed and the test would pass vacuously.
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
    """Hands out LightGBM models whose single-row predictions are recorded,
    so each recursive step's output can be matched to the next step's lag."""

    def __init__(self):
        self.step_preds: list[float] = []

    def create(self, name):
        model = LightGBMModel()
        inner = model.predict

        def predict(X):
            y = inner(X)
            if np.asarray(X).shape[0] == 1:
                self.step_preds.append(float(np.ravel(y)[0]))
            return y

        model.predict = predict
        return model


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

    # Every recursive step routes its feature row through this guard, with
    # the column names — the one place both paths expose the row by name.
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
        include_sun_elevation=True, include_clear_sky_irradiance=False,
    )
    params.update(kw)
    return ExperimentCfg(**params)


def _horizon_solar(ds_future):
    return compute_solar_features(
        pd.DatetimeIndex(ds_future), latitude=LAT, longitude=LON,
        include_elevation=True, include_clear_sky=True,
    )


def _assert_lag_chain(rows, preds, ds_future, gated):
    """Step s+1's y_lag_1 is step s's prediction — or 0 after a night step
    when the clear-sky gate applies."""
    assert len(rows) == FUTURE_PERIODS
    assert len(preds) == FUTURE_PERIODS
    solar = _horizon_solar(ds_future)
    night = solar["clear_sky_ghi"].to_numpy() <= 0
    assert night[:-1].any() and (~night[:-1]).any(), (
        "the horizon must span both night and day or the gate is untested"
    )
    night_preds = [preds[s] for s in range(FUTURE_PERIODS - 1) if night[s]]
    assert any(abs(p) > 1e-3 for p in night_preds), (
        "a night prediction must be non-zero or gated and ungated agree"
    )
    for s in range(FUTURE_PERIODS - 1):
        expected = 0.0 if (gated and night[s]) else preds[s]
        got = rows[s + 1]["y_lag_1"]
        assert got == pytest.approx(expected, rel=1e-6, abs=1e-6), (
            f"step {s + 1}: y_lag_1={got}, expected {expected} "
            f"({'night' if night[s] else 'day'}, gated={gated})"
        )


class TestCachedForecast:
    def test_elevation_only_forecast_completes(self, tmp_db, monkeypatch):
        exp = _exp()
        app, published, _ = _make_app(tmp_db, exp, monkeypatch)

        # _retrain_and_cache runs the post-retrain _forecast_with_cached →
        # _compute_cached_forecast, which is where the KeyError surfaced.
        _run(app._retrain_and_cache(exp))

        cache = app._cached_models[exp.name]
        assert not cache["is_neural"]
        assert "sun_elevation" in cache["feature_cols"]
        assert "clear_sky_ghi" not in cache["feature_cols"]
        assert len(published) == 1
        y_pred = np.asarray(published[0]["y_pred"])
        assert y_pred.shape == (FUTURE_PERIODS,)
        assert np.isfinite(y_pred).all()

    def test_elevation_only_lag_buffer_is_not_gated(self, tmp_db, monkeypatch):
        exp = _exp()
        app, published, rows = _make_app(tmp_db, exp, monkeypatch)
        _run(app._retrain_and_cache(exp))

        preds = app.model_registry.step_preds
        _assert_lag_chain(
            rows, preds, published[0]["ds_future"], gated=False,
        )

    def test_clear_sky_still_gates_the_lag_buffer(self, tmp_db, monkeypatch):
        exp = _exp(include_clear_sky_irradiance=True)
        app, published, rows = _make_app(tmp_db, exp, monkeypatch)
        _run(app._retrain_and_cache(exp))

        assert "clear_sky_ghi" in app._cached_models[exp.name]["feature_cols"]
        preds = app.model_registry.step_preds
        _assert_lag_chain(
            rows, preds, published[0]["ds_future"], gated=True,
        )


class TestProductionInference:
    def test_elevation_only_lag_buffer_is_not_gated(self, tmp_db, monkeypatch):
        exp = _exp()
        app, published, rows = _make_app(tmp_db, exp, monkeypatch)
        _run(app._run_production_inference(exp))

        assert len(published) == 1
        assert "sun_elevation" in rows[0]
        assert "clear_sky_ghi" not in rows[0]
        preds = app.model_registry.step_preds
        _assert_lag_chain(
            rows, preds, published[0]["ds_future"], gated=False,
        )

    def test_clear_sky_still_gates_the_lag_buffer(self, tmp_db, monkeypatch):
        exp = _exp(include_clear_sky_irradiance=True)
        app, published, rows = _make_app(tmp_db, exp, monkeypatch)
        _run(app._run_production_inference(exp))

        assert "clear_sky_ghi" in rows[0]
        preds = app.model_registry.step_preds
        _assert_lag_chain(
            rows, preds, published[0]["ds_future"], gated=True,
        )
