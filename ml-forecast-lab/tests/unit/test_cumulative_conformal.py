"""v2.52.4: band calibration, band coverage and the accuracy sensor scored a
cumulative target's per-interval forecast against its running counter.

For ``source_is_cumulative`` experiments ``forecast_log.predicted`` (and the
logged ``upper``/``lower``) are per-interval deltas, while the actuals table
holds the raw counter. ``get_conformal_quantiles`` and
``get_forecast_coverage`` joined the raw grid, and the published
``_forecast_accuracy`` sensor called ``get_forecast_accuracy`` in its default
raw mode. The residual was therefore |delta − running total|: on a
daily-reset kWh counter the 80% band came out tens of kWh wide around a
fraction-of-a-kWh forecast, coverage read whatever the counter level
happened to produce, and the accuracy state measured the counter level, not
the model. The web Accuracy tab already used increment mode.

All three now compare against the actuals increments, which follow the
training label's rule (pinned in ``test_increment_actuals.py``), and the
non-cumulative path is unchanged. The replay key for
``get_conformal_quantiles`` carries the new flag only when it is set, so
bundles recorded before it existed keep their keys.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.config import ExperimentCfg
from ml_forecast_lab.db import HistoryDB
from ml_forecast_lab.main import MLForecastLabApp

EXP = "demand"
ENTITY = "sensor.demand_today"
INTERVAL = 30
LEVEL = 0.8


def _grid_end() -> datetime:
    # Whole hour, comfortably in the past so every target is <= now.
    return datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)


def _daily_counter(days: int = 4, quiet: tuple[int, int] = (100, 103)):
    """A daily-reset counter on the 30-min grid, plus its true increments.

    Resets at UTC midnight (the reset row carries that interval's
    increment). ``quiet`` (positional, half-open) is a stretch with no use,
    which a change-only recorder leaves without rows.
    """
    end = _grid_end()
    idx = pd.date_range(end - timedelta(days=days), end, freq=f"{INTERVAL}min")
    inc = np.array([0.2 + 0.1 * (i % 5) for i in range(len(idx))])
    inc[quiet[0]:quiet[1]] = 0.0
    counter = np.empty(len(idx))
    total = 0.0
    for i, ts in enumerate(idx):
        if i > 0 and ts.normalize() != idx[i - 1].normalize():
            total = 0.0
        total += inc[i]
        counter[i] = total
    keep = np.ones(len(idx), dtype=bool)
    keep[quiet[0]:quiet[1]] = False
    return idx, inc, counter, keep


def _scored(idx):
    """Bins with a scored increment: every bin but the window's first,
    quiet and reset bins included."""
    return np.arange(len(idx)) >= 1


@pytest.fixture
def cumulative_db(tmp_path):
    """Counter actuals plus one h=1 forecast per bin: predicted = true
    increment + a known residual, with a ±0.1 band around the true
    increment. Returns what the expected-value maths needs."""
    db = HistoryDB(tmp_path / "history.db")
    db.ensure_forecast_log_table()
    table = db.safe_table_name(ENTITY)
    idx, inc, counter, keep = _daily_counter()
    db.store_history(table, pd.DataFrame({"ds": idx[keep], "value": counter[keep]}))

    rng = np.random.default_rng(11)
    resid = rng.normal(0, 0.05, len(idx))
    for i in range(1, len(idx)):
        target = idx[i].to_pydatetime()
        db.log_forecast(
            EXP, target - timedelta(minutes=INTERVAL), [target],
            [float(inc[i] + resid[i])], "lightgbm",
            upper_bounds=[float(inc[i] + 0.1)],
            lower_bounds=[float(inc[i] - 0.1)],
            model_version="v1",
        )
    return db, table, idx, inc, counter, keep, resid


class TestConformalQuantiles:
    def test_cumulative_residuals_are_per_interval(self, cumulative_db):
        db, table, idx, _inc, _counter, keep, resid = cumulative_db
        cq = db.get_conformal_quantiles(
            EXP, table, level=LEVEL, model_name="lightgbm",
            model_version="v1", interval_minutes=INTERVAL,
            source_is_cumulative=True,
        )
        kept = _scored(idx)
        expected = float(np.quantile(np.abs(resid[kept]), LEVEL))
        # Quiet bins score as zero use, reset bins as their own use.
        assert cq["total_samples"] == int(kept.sum())
        assert cq["fallback_quantile"] == pytest.approx(expected, rel=1e-6)
        assert cq["quantiles"][INTERVAL] == pytest.approx(expected, rel=1e-6)
        # A delta-space band is a fraction of a kWh, not the daily total.
        assert cq["fallback_quantile"] < 0.2

    def test_raw_mode_is_unchanged(self, cumulative_db):
        """Default (non-cumulative) calls keep the pre-v2.52.4 semantics:
        |predicted − raw grid value| over every bin with an actual."""
        db, table, idx, inc, counter, keep, resid = cumulative_db
        cq = db.get_conformal_quantiles(
            EXP, table, level=LEVEL, model_name="lightgbm",
            model_version="v1", interval_minutes=INTERVAL,
        )
        present = keep.copy()
        present[0] = False  # no forecast targets the first bin
        raw_resid = np.abs(inc[present] + resid[present] - counter[present])
        assert cq["total_samples"] == int(present.sum())
        assert cq["fallback_quantile"] == pytest.approx(
            float(np.quantile(raw_resid, LEVEL)), rel=1e-6,
        )


class TestCoverage:
    def test_cumulative_band_is_tested_against_deltas(self, cumulative_db):
        db, table, idx, _inc, _counter, keep, _resid = cumulative_db
        cov = db.get_forecast_coverage(
            EXP, table, interval_minutes=INTERVAL,
            model_name="lightgbm", model_version="v1",
            source_is_cumulative=True,
        )
        kept = int(_scored(idx).sum())
        # The ±0.1 band brackets every true increment, quiet and reset
        # bins included.
        assert cov["overall"] == {"coverage": 1.0, "n": kept}
        assert cov["by_lead"]["coverage"] == [1.0]
        assert cov["by_lead"]["n"] == [kept]
        assert sum(cov["by_hour_of_day"]["n"]) == kept

    def test_window_first_bin_is_not_scored(self, cumulative_db):
        """The window's first bin has no increment (no earlier reading);
        a banded forecast for it is left out, not counted as a miss."""
        db, table, idx, _inc, _counter, _keep, _resid = cumulative_db
        first = idx[0].to_pydatetime()
        db.log_forecast(
            EXP, first - timedelta(minutes=INTERVAL), [first], [0.5],
            "lightgbm", upper_bounds=[0.6], lower_bounds=[0.4],
            model_version="v1",
        )
        cov = db.get_forecast_coverage(
            EXP, table, interval_minutes=INTERVAL,
            model_name="lightgbm", model_version="v1",
            source_is_cumulative=True,
        )
        kept = int(_scored(idx).sum())
        assert cov["overall"] == {"coverage": 1.0, "n": kept}
        assert cov["by_lead"]["n"] == [kept]
        assert sum(cov["by_hour_of_day"]["n"]) == kept

    def test_raw_mode_is_unchanged(self, cumulative_db):
        db, table, idx, inc, counter, keep, _resid = cumulative_db
        cov = db.get_forecast_coverage(
            EXP, table, interval_minutes=INTERVAL,
            model_name="lightgbm", model_version="v1",
        )
        present = keep.copy()
        present[0] = False
        inside = np.abs(counter[present] - inc[present]) <= 0.1
        assert cov["overall"]["n"] == int(present.sum())
        assert cov["overall"]["coverage"] == pytest.approx(
            round(float(inside.mean()), 4),
        )


class _RecordingDB:
    """Stands in for HistoryDB on the band path: records each call."""

    def __init__(self, responses):
        self.calls: list[dict] = []
        self._responses = list(responses)

    def safe_table_name(self, entity_id):
        return HistoryDB.safe_table_name(self, entity_id)

    def get_conformal_quantiles(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def _exp(**kw):
    params = dict(
        name=EXP, target_entity=ENTITY, interval_minutes=INTERVAL,
        future_periods=3, publish_prefix="mlfl_", publish_name=EXP,
        mode="production", units="kWh",
    )
    params.update(kw)
    return ExperimentCfg(**params)


def _future(n=3):
    start = pd.Timestamp(datetime.utcnow().replace(second=0, microsecond=0))
    return pd.DatetimeIndex([start + pd.Timedelta(minutes=INTERVAL * (i + 1)) for i in range(n)])


class TestBandWiring:
    @pytest.mark.parametrize("cumulative", [True, False])
    def test_pinned_and_pooled_queries_carry_the_flag(self, cumulative):
        app = MLForecastLabApp()
        app.history_db = _RecordingDB([
            {"fallback_quantile": None, "total_samples": 3},  # pinned: too few
            {"quantiles": {}, "fallback_quantile": 0.5, "total_samples": 40},
        ])
        exp = _exp(source_is_cumulative=cumulative, reset_daily=cumulative)
        bands = asyncio.run(app._conformal_bands(
            exp, np.array([1.0, 1.0, 1.0], dtype=np.float32), _future(),
            "lightgbm", "v2",
        ))
        assert bands is not None and bands.pooled_versions
        assert [c["model_version"] for c in app.history_db.calls] == ["v2", None]
        assert [c["source_is_cumulative"] for c in app.history_db.calls] == [cumulative] * 2


class _StubHA:
    def __init__(self):
        self.captured: dict[str, tuple] = {}

    async def set_state(self, entity_id, state, attributes=None):
        self.captured[entity_id] = (state, attributes or {})
        return True

    async def get_state(self, entity_id, default=None, attribute=None):
        return default


class TestPublishedSensors:
    def _publish(self, db, exp):
        app = MLForecastLabApp()
        app.ha_interface = _StubHA()
        app.history_db = db
        app._cached_models = {exp.name: {"model_version": "v1"}}
        y = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        asyncio.run(app._publish_forecast_sensors(
            exp_cfg=exp, y_pred=y, ds_future=_future(),
            model_name="lightgbm", last_trained_iso="2026-10-01T00:00:00Z",
        ))
        return app.ha_interface.captured

    def test_cumulative_accuracy_and_bands_are_per_interval(self, cumulative_db):
        db, _table, idx, _inc, _counter, keep, resid = cumulative_db
        exp = _exp(source_is_cumulative=True, reset_daily=True)
        captured = self._publish(db, exp)

        kept = _scored(idx)
        acc_state, acc_attrs = captured[f"sensor.mlfl_{EXP}_forecast_accuracy"]
        assert acc_attrs["status"] == "ready"
        assert float(acc_state) == pytest.approx(
            float(np.mean(np.abs(resid[kept]))), abs=1e-4,
        )

        q = float(np.quantile(np.abs(resid[kept]), LEVEL))
        upper, _ = captured[f"sensor.mlfl_{EXP}_upper_80"]
        assert float(upper) == pytest.approx(0.5 + q, abs=1e-3)

    def test_non_cumulative_accuracy_stays_raw(self, tmp_path, monkeypatch):
        db = HistoryDB(tmp_path / "history.db")
        modes = []
        real = db.get_forecast_accuracy

        def spy(*args, **kwargs):
            modes.append(kwargs.get("evaluation_mode", "raw"))
            return real(*args, **kwargs)

        monkeypatch.setattr(db, "get_forecast_accuracy", spy)
        self._publish(db, _exp(source_is_cumulative=False))
        assert modes == ["raw"]


class TestWebWiring:
    """The Accuracy tab passes the flag positionally; pin that it binds to
    ``source_is_cumulative`` in the real signatures."""

    @pytest.mark.parametrize(
        "method", ["get_forecast_coverage", "get_conformal_quantiles"],
    )
    def test_positional_flag_binds_to_source_is_cumulative(self, method):
        import ast
        import inspect
        from pathlib import Path

        import ml_forecast_lab.web.app as web_app

        tree = ast.parse(Path(web_app.__file__).read_text())
        params = list(inspect.signature(getattr(HistoryDB, method)).parameters)[1:]
        bound = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and node.args
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "to_thread"):
                continue
            target = node.args[0]
            if not (isinstance(target, ast.Attribute) and target.attr == method):
                continue
            args = [ast.unparse(a) for a in node.args[1:]]
            if "bool(exp_cfg.source_is_cumulative)" in args:
                bound.append(params[args.index("bool(exp_cfg.source_is_cumulative)")])
        assert bound and set(bound) == {"source_is_cumulative"}


class TestReplayKey:
    def test_flag_absent_or_false_keeps_the_pre_v2_52_4_key(self):
        from ml_forecast_lab.replay import _conformal_key

        legacy = json.dumps([
            "get_conformal_quantiles",
            json.dumps({
                "actuals_table": "sensor_x", "experiment": EXP,
                "interval_minutes": 30, "level": 0.8, "max_age_days": 14,
                "min_samples": 10, "model_name": "lightgbm",
                "model_version": "v1",
            }, sort_keys=True),
        ])
        kw = dict(level=0.8, model_name="lightgbm", model_version="v1",
                  interval_minutes=30)
        assert _conformal_key(EXP, "sensor_x", **kw) == legacy
        assert _conformal_key(
            EXP, "sensor_x", source_is_cumulative=False, **kw) == legacy
        assert _conformal_key(
            EXP, "sensor_x", source_is_cumulative=True, **kw) != legacy
        assert '\\"source_is_cumulative\\": true' in _conformal_key(
            EXP, "sensor_x", source_is_cumulative=True, **kw)
