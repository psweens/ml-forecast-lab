"""v2.52.4: the ``_forecast_accuracy`` sensor froze on a hindcast lead bucket.

The published state was ``lead_time_curve["mae"][0]``, the most negative
lead bucket. A forecast tick whose fresh fetch fails falls back to the
cached frame and logs targets that are already in the past (leads of minus
several hours). Those targets have actuals, so they join, and their bucket
sorts first: one stale tick pinned the state to that bucket's handful of
samples, identical on every publish, until the rows aged out of the 30-day
window. Every experiment issued during the same outage froze together,
cumulative or not.

The state is now the first bucket with a non-negative lead (bucket 0 holds
leads in ``(-interval, interval)``). The hindcast bucket stays in the
``lead_hours`` / ``mae`` attributes; it no longer becomes the state.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.config import ExperimentCfg
from ml_forecast_lab.db import HistoryDB
from ml_forecast_lab.main import MLForecastLabApp

EXP = "solar"
ENTITY = "sensor.solar_power"
INTERVAL = 30
STEP1_ERR = 0.05
HINDCAST_ERR = 3.0


class _StubHA:
    def __init__(self):
        self.captured: dict[str, tuple] = {}

    async def set_state(self, entity_id, state, attributes=None):
        self.captured[entity_id] = (state, attributes or {})
        return True

    async def get_state(self, entity_id, default=None, attribute=None):
        return default


def _exp():
    return ExperimentCfg(
        name=EXP, target_entity=ENTITY, interval_minutes=INTERVAL,
        future_periods=3, publish_prefix="mlfl_", publish_name=EXP,
        mode="production", units="kW",
    )


def _publish(db):
    app = MLForecastLabApp()
    app.ha_interface = _StubHA()
    app.history_db = db
    app._cached_models = {EXP: {"model_version": "v1"}}
    start = pd.Timestamp(datetime.utcnow().replace(second=0, microsecond=0))
    ds_future = pd.DatetimeIndex(
        [start + pd.Timedelta(minutes=INTERVAL * (i + 1)) for i in range(3)]
    )
    asyncio.run(app._publish_forecast_sensors(
        exp_cfg=_exp(), y_pred=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        ds_future=ds_future, model_name="lightgbm",
        last_trained_iso="2026-10-01T00:00:00Z",
    ))
    return app.ha_interface.captured[f"sensor.mlfl_{EXP}_forecast_accuracy"]


@pytest.fixture
def db_with_actuals(tmp_path):
    """Two days of raw actuals on the grid; returns (db, grid)."""
    db = HistoryDB(tmp_path / "history.db")
    db.ensure_forecast_log_table()
    end = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    grid = pd.date_range(end - timedelta(days=2), end, freq=f"{INTERVAL}min")
    db.store_history(
        db.safe_table_name(ENTITY),
        pd.DataFrame({"ds": grid, "value": np.ones(len(grid))}),
    )
    return db, grid


def _log_step1(db, grid):
    """Live ticks 7 min after each grid point: step 1 has lead 23 min."""
    for ts in grid[1:]:
        target = ts.to_pydatetime()
        db.log_forecast(
            EXP, target - timedelta(minutes=INTERVAL - 7), [target],
            [1.0 + STEP1_ERR], "lightgbm", model_version="v1",
        )


def _log_hindcast(db, grid, hours_stale=9):
    """One stale-frame tick: issued at the last grid point, targets taken
    from a frame ``hours_stale`` hours old."""
    issued = grid[-1].to_pydatetime() + timedelta(minutes=7)
    targets = [
        t.to_pydatetime()
        for t in grid[-2 * hours_stale - 1:-2 * hours_stale + 2]
    ]
    db.log_forecast(
        EXP, issued, targets, [1.0 + HINDCAST_ERR] * len(targets),
        "lightgbm", model_version="v1",
    )


class TestAccuracyState:
    def test_hindcast_bucket_does_not_become_the_state(self, db_with_actuals):
        db, grid = db_with_actuals
        _log_step1(db, grid)
        _log_hindcast(db, grid)

        state, attrs = _publish(db)
        assert attrs["status"] == "ready"
        # The hindcast bucket is still reported in the curve...
        assert attrs["lead_hours"][0] < 0
        assert attrs["mae"][0] == pytest.approx(HINDCAST_ERR, abs=1e-6)
        # ...but the state is the bucket-0 (next-interval) error.
        assert float(state) == pytest.approx(STEP1_ERR, abs=1e-4)

    def test_state_without_hindcast_rows_is_unchanged(self, db_with_actuals):
        db, grid = db_with_actuals
        _log_step1(db, grid)

        state, attrs = _publish(db)
        assert attrs["status"] == "ready"
        assert attrs["lead_hours"][0] == 0
        assert float(state) == pytest.approx(attrs["mae"][0], abs=1e-4)
        assert float(state) == pytest.approx(STEP1_ERR, abs=1e-4)

    def test_only_hindcast_rows_stays_accumulating(self, db_with_actuals):
        db, grid = db_with_actuals
        _log_hindcast(db, grid)

        state, attrs = _publish(db)
        assert attrs["lead_hours"] and attrs["lead_hours"][0] < 0
        assert attrs["status"] == "accumulating"
        assert state == "0"


def test_accuracy_tab_headline_skips_hindcast_buckets():
    """Source contract: the Accuracy tab's headline error picks the first
    non-negative lead bucket, as the sensor state does."""
    html = (
        Path(__file__).resolve().parents[2]
        / "ml_forecast_lab" / "web" / "templates" / "experiment.html"
    ).read_text()
    start = html.index("--- Accuracy chip + headline MAE ---")
    block = html[start:html.index("--- Overall headline ---", start)]
    assert "if (leads[i] < 0) continue;" in block
    assert block.index("leads[i] < 0") < block.index("ns[i] >= LEADBUCKET_MIN_N")
    assert "headlineMae = maes[first];" in block
    assert "maes[0]" not in block
