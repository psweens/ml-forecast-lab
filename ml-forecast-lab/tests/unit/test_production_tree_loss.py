"""
Production retrain applies the experiment's loss to tree backends.

Pins a defect found in the 2026-11 intermittent-demand investigation: the
benchmark and tuning paths gave tree backends the experiment's ``loss_fn``
while ``_retrain_and_cache``, ``_run_production_inference`` and the holdout
refit applied it to neural backends only. An experiment configured with
``loss_fn: tweedie`` was ranked with a Tweedie tree and published forecasts
from a Huber one.
"""
from __future__ import annotations

import asyncio
import importlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab import main as M
from ml_forecast_lab.config import AppConfig, ExperimentCfg
from ml_forecast_lab.covariates import CovariateResolver
from ml_forecast_lab.db import HistoryDB
from ml_forecast_lab.main import MLForecastLabApp, _apply_experiment_loss
from ml_forecast_lab.models.registry import ModelRegistry
from tests.synthetic import intermittent_demand as gen

pytest.importorskip("lightgbm")

NOW = datetime(2026, 11, 10, 19, 30, tzinfo=timezone.utc)


class _StubHA:
    def __init__(self, rows):
        self.rows = rows

    async def get_history(self, entity_id, start, end, include_attributes=False):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        s = s.tz_localize("UTC") if s.tzinfo is None else s
        e = e.tz_localize("UTC") if e.tzinfo is None else e
        return [r for r in self.rows if s <= pd.Timestamp(r["last_changed"]) <= e]

    async def get_state(self, entity_id, default=None, attribute=None):
        return default


def _retrain(tmp_path, monkeypatch, **exp_kw):
    y, _ = gen.slot_truth("2026-10-27", 16, seed=0)
    rows = gen.recorder_rows(y[y.index < pd.Timestamp(NOW)])
    exp = ExperimentCfg(
        name="demand", target_entity="sensor.demand_today", mode="production",
        interval_minutes=30, future_periods=12, days_history=14,
        source_is_cumulative=True, reset_daily=True,
        models_enabled=["lightgbm"], production_model="lightgbm",
        max_age=36500, **exp_kw,
    )
    monkeypatch.setattr(
        MLForecastLabApp, "_cached_model_dir",
        staticmethod(lambda n: tmp_path / "models" / n),
    )
    orig = MLForecastLabApp._prepare_training_frame

    async def _at_now(self, exp_cfg, now=None, _o=orig, **kw):
        return await _o(self, exp_cfg, now=NOW, **kw)

    monkeypatch.setattr(MLForecastLabApp, "_prepare_training_frame", _at_now)
    app = MLForecastLabApp()
    app.history_db = HistoryDB(str(tmp_path / "h.db"))
    ha = _StubHA(rows)
    app.ha_interface = ha
    app.config = AppConfig(
        experiments=[exp], model_overrides={"lightgbm": {"n_estimators": 20}},
    )
    app.covariate_resolver = CovariateResolver(ha, history_db=None)
    reg = ModelRegistry()
    reg.register("lightgbm", importlib.import_module(
        "ml_forecast_lab.models.lightgbm_backend").LightGBMModel)
    app.model_registry = reg

    async def _noop(*a, **k):
        return True

    app._publish_forecast_sensors = _noop
    app._publish_lifecycle_sensor = _noop
    app._forecast_with_cached = _noop
    np.random.seed(0)
    asyncio.run(app._retrain_and_cache(exp))
    return app._cached_models[exp.name]["model"]


def _objective(model) -> str:
    return model.model.params["objective"]


class TestProductionTreeLoss:
    def test_default_experiment_keeps_the_default_objective(self, tmp_path, monkeypatch):
        model = _retrain(tmp_path, monkeypatch)
        assert model.loss_fn == "huber"
        assert _objective(model) == "huber"

    def test_experiment_loss_reaches_the_production_tree(self, tmp_path, monkeypatch):
        model = _retrain(tmp_path, monkeypatch, loss_fn="tweedie")
        assert model.loss_fn == "tweedie"
        assert _objective(model) == "tweedie"

    def test_model_params_override_wins(self, tmp_path, monkeypatch):
        model = _retrain(
            tmp_path, monkeypatch, loss_fn="tweedie",
            model_params={"lightgbm": {"loss_fn": "mse"}},
        )
        assert model.loss_fn == "mse"
        assert _objective(model) == "regression"


class TestHelper:
    class _Tree:
        is_neural = False
        loss_fn = "huber"

        def set_params(self, **kw):
            self.__dict__.update(kw)

    class _NoLoss:
        is_neural = False

    def test_applies_to_any_backend_with_a_loss(self):
        m = self._Tree()
        _apply_experiment_loss(m, ExperimentCfg(name="e", target_entity="sensor.x", loss_fn="mse"))
        assert m.loss_fn == "mse"

    def test_skips_when_overridden_or_unsupported(self):
        m = self._Tree()
        exp = ExperimentCfg(name="e", target_entity="sensor.x", loss_fn="mse")
        _apply_experiment_loss(m, exp, {"loss_fn": "tweedie"})
        assert m.loss_fn == "huber"
        _apply_experiment_loss(self._NoLoss(), exp)  # must not raise


def test_no_production_path_gates_the_loss_on_is_neural():
    src = Path(M.__file__).read_text(encoding="utf-8")
    assert "is_neural and hasattr(model, 'loss_fn')" not in src
    assert "is_neural and hasattr(m, 'loss_fn')" not in src
    assert src.count("_apply_experiment_loss(") >= 4  # definition + three call sites
