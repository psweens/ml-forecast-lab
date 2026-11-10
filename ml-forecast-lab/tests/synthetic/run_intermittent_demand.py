"""
Intermittent-demand reproduction — production retrain and cached forecast
on the synthetic counter from ``intermittent_demand.py``.

Runs the app's own ``_retrain_and_cache`` and ``_compute_cached_forecast``
against a stub history source, then reports for each configuration:

  - target construction: level of the training target against ground truth
  - the trained model: best iteration, objective, top feature gains
  - the published point forecast: range, daily total, and correlation with
    the generator's expected profile
  - one-step validation error of constant forecasts, to show how the
    ranking metrics order a degenerate forecast

Findings are recorded in
``docs/investigations/2026-11-intermittent-demand-flat-forecast.md``.

    cd ml-forecast-lab
    python tests/synthetic/run_intermittent_demand.py
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml_forecast_lab.config import AppConfig, ExperimentCfg  # noqa: E402
from ml_forecast_lab.covariates import CovariateResolver  # noqa: E402
from ml_forecast_lab.db import HistoryDB  # noqa: E402
from ml_forecast_lab.main import MLForecastLabApp  # noqa: E402
from ml_forecast_lab.models.registry import ModelRegistry  # noqa: E402
from tests.synthetic import intermittent_demand as gen  # noqa: E402

# Evening issue time, 16 days after a daylight-saving change, so a 40-day
# window holds both UTC offsets.
NOW = datetime(2026, 11, 10, 19, 30, tzinfo=timezone.utc)
ENTITY = "sensor.demand_today"


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


def run_case(label, exp_kw=None, overrides=None, days=40, seed=0):
    start = (pd.Timestamp(NOW) - pd.Timedelta(days=days + 1)).floor("D")
    y, _ = gen.slot_truth(start.tz_localize(None), days + 4, seed=seed)
    rows = gen.recorder_rows(y[y.index < pd.Timestamp(NOW)])
    kw = dict(
        name="demand", target_entity=ENTITY, mode="production",
        interval_minutes=30, future_periods=96, days_history=days,
        source_is_cumulative=True, reset_daily=True,
        models_enabled=["lightgbm"], production_model="lightgbm",
        max_age=36500,
    )
    kw.update(exp_kw or {})
    exp = ExperimentCfg(**kw)

    tmp = Path(tempfile.mkdtemp(prefix="mlfl_intermittent_"))
    MLForecastLabApp._cached_model_dir = staticmethod(lambda n: tmp / "models" / n)
    app = MLForecastLabApp()
    app.history_db = HistoryDB(str(tmp / "h.db"))
    ha = _StubHA(rows)
    app.ha_interface = ha
    app.config = AppConfig(experiments=[exp], model_overrides=overrides or {})
    app.config.timezone = gen.TZ
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
    orig = MLForecastLabApp._prepare_training_frame

    async def _at_now(exp_cfg, now=None, **k):
        return await orig(app, exp_cfg, now=NOW, **k)

    app._prepare_training_frame = _at_now

    np.random.seed(0)
    asyncio.run(app._retrain_and_cache(exp))
    cache = app._cached_models[exp.name]
    fc = asyncio.run(app._compute_cached_forecast(cache, now=NOW))

    model = cache["model"]
    target = cache["combined"]["target"]
    truth = y.copy()
    truth.index = truth.index.tz_localize(None)
    common = target.index.intersection(truth.index)
    y_pred = np.asarray(fc.y_pred, dtype=float)
    expected = gen.expected_demand(
        pd.DatetimeIndex(fc.ds_future).tz_localize("UTC")
    ).to_numpy()
    # The production retrain passes no feature names to fit(), so the
    # stored importances are keyed feature_<i>; label them from the cache.
    cols = cache["feature_cols"]
    imp = sorted(
        (
            (cols[int(k.split("_")[1])] if k.startswith("feature_") else k, v)
            for k, v in model.training_metadata["feature_importances"].items()
        ),
        key=lambda kv: -kv[1],
    )
    calendar = sum(v for k, v in imp if not k.startswith("y_"))

    print(f"\n=== {label} ===")
    print(
        f"target: {len(target)} rows, mean {target.mean():.3f} "
        f"(truth {truth.loc[common].mean():.3f}, "
        f"{(target.loc[common].sum() / truth.loc[common].sum() - 1) * 100:+.1f} %)"
    )
    print(
        f"model: objective loss_fn={model.loss_fn!r} (experiment loss_fn="
        f"{exp.loss_fn!r}), best_iteration="
        f"{model.training_metadata.get('best_iteration')}, "
        f"calendar gain share {calendar:.2f}"
    )
    print("  top gain:", [(k, round(v, 3)) for k, v in imp[:6]])
    print(
        f"forecast: range [{y_pred.min():.3f}, {y_pred.max():.3f}], "
        f"day-1 total {y_pred[:48].sum():.1f} "
        f"(expected {expected[:48].sum():.1f}, "
        f"{(y_pred[:48].sum() / expected[:48].sum() - 1) * 100:+.0f} %), "
        f"expected range [{expected.min():.2f}, {expected.max():.2f}], "
        f"profile corr {np.corrcoef(y_pred, expected)[0, 1]:.2f}, "
        f"amplitude ratio {y_pred.std() / expected.std():.2f}"
    )
    return dict(target=target, truth=truth, forecast=y_pred, expected=expected)


def constant_scores(target: pd.Series) -> None:
    k = int(len(target) * 0.8)
    train, val = target.to_numpy()[:k], target.to_numpy()[k:]
    print("\n=== constant forecasts, last 20 % of the window ===")
    for name, c in (("zero", 0.0), ("median", float(np.median(train))),
                    ("mean", float(train.mean()))):
        print(
            f"  {name:6s} {c:5.2f}: MAE {np.abs(val - c).mean():.3f}  "
            f"RMSE {np.sqrt(((val - c) ** 2).mean()):.3f}"
        )


def main() -> None:
    logging.basicConfig(level=logging.ERROR)
    base = run_case("defaults (Huber)")
    run_case("experiment loss_fn: tweedie", exp_kw=dict(loss_fn="tweedie"))
    run_case("model_params loss_fn: mse",
             exp_kw=dict(model_params={"lightgbm": {"loss_fn": "mse"}}))
    run_case("model_params loss_fn: tweedie",
             exp_kw=dict(model_params={"lightgbm": {"loss_fn": "tweedie"}}))
    run_case("model_params learning_rate: 0.001",
             exp_kw=dict(model_params={"lightgbm": {"learning_rate": 0.001}}))
    constant_scores(base["target"])


if __name__ == "__main__":
    main()
