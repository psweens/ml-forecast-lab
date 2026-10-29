"""v2.55.0: an experiment with ``load_subtract`` is scored on what it forecasts.

``load_subtract`` took the subtracted loads out of the target only on the
training and inference path. Everything that compares the forecast with
what happened read the raw target instead, which still contains them:

- the ``_cumulative`` sensor was seeded with the target's own reading, so
  the forecast of today's remaining use after subtraction was added to a
  running total that included the subtracted loads;
- the ``_forecast_accuracy`` sensor, the Forecast Accuracy and Forecast
  Comparison tabs, the trajectory and evolution views, and the conformal
  band calibration all joined the forecast log against the raw target's
  table, so each "error" was mostly the size of the subtracted loads.

The fetch path now keeps the target after subtraction in a per-experiment
table (``HistoryDB.net_actuals_table``): a running total since local midnight
per grid bin for a cumulative target, the bin's value otherwise, NULL where
the label is unmeasured. ``HistoryDB.actuals_table_for`` routes every reader
to it, and the seed is today's use after subtraction. Experiments without
``load_subtract`` read exactly what they read before.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.config import AppConfig, ExperimentCfg, SubtractCfg
from ml_forecast_lab.covariates import CovariateResolver
from ml_forecast_lab.db import NET_ACTUALS_PREFIX, HistoryDB
from ml_forecast_lab.main import MLForecastLabApp

INTERVAL = 30
LOAD = "sensor.house_load_today"
EV = "sensor.ev_energy_today"
GROSS_PER_BIN = 1.0     # kWh per 30-minute bin on the house meter
EV_PER_BIN = 0.4        # of which the subtracted load
NET_PER_BIN = GROSS_PER_BIN - EV_PER_BIN


def _run(coro):
    return asyncio.run(coro)


def _exp(subtract: bool = True, **kw) -> ExperimentCfg:
    params = dict(
        name="house_load", target_entity=LOAD, days_history=4,
        interval_minutes=INTERVAL, models_enabled=["lightgbm"],
        source_is_cumulative=True, reset_daily=True,
        future_periods=4, publish_prefix="mlfl_", publish_name="house_load",
        mode="production", units="kWh",
    )
    if subtract:
        params["load_subtract"] = [
            SubtractCfg(entity_id=EV, source="cumulative_daily", on_missing="zero"),
        ]
    params.update(kw)
    return ExperimentCfg(**params)


def _daily_counter_rows(now: datetime, days: int, per_bin: float, cadence_min: int = 15):
    """A UTC-midnight-reset counter rising ``per_bin`` every 30 minutes,
    read every ``cadence_min`` minutes."""
    ts = pd.date_range(now - timedelta(days=days), now, freq=f"{cadence_min}min", tz="UTC")
    rows = []
    for t in ts:
        since_midnight = (t - t.normalize()).total_seconds() / 60.0
        rows.append({
            "last_changed": t.isoformat(),
            "state": f"{per_bin * since_midnight / INTERVAL:.4f}",
        })
    return rows


class _StubHA:
    """History from recorded rows; ``get_state`` returns the configured
    live reading; ``set_state`` captures what is published."""

    def __init__(self, rows_by_entity, states=None):
        self.rows_by_entity = rows_by_entity
        self.states = states or {}
        self.captured: dict[str, tuple] = {}
        self.state_reads: list[str] = []

    async def get_history(self, entity_id, start, end, include_attributes=False):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        s = s.tz_localize("UTC") if s.tzinfo is None else s
        e = e.tz_localize("UTC") if e.tzinfo is None else e
        return [
            r for r in self.rows_by_entity.get(entity_id, [])
            if s <= pd.Timestamp(r["last_changed"]) <= e
        ]

    async def get_state(self, entity_id, default=None, attribute=None):
        self.state_reads.append(entity_id)
        return self.states.get(entity_id, default)

    async def set_state(self, entity_id, state, attributes=None):
        self.captured[entity_id] = (state, attributes or {})
        return True


def _make_app(tmp_db, exp_cfg, ha, tz="UTC"):
    app = MLForecastLabApp()
    app.history_db = HistoryDB(tmp_db)
    app.ha_interface = ha
    app.config = AppConfig(experiments=[exp_cfg], timezone=tz)
    app.covariate_resolver = CovariateResolver(
        app.ha_interface,
        history_db=app.history_db,
        retention_provider=app._retention_days_for_table,
    )
    return app


def _now() -> datetime:
    # Mid-afternoon UTC, so "today" has a known number of whole bins.
    return datetime(2026, 10, 20, 14, 0, tzinfo=timezone.utc)


def _fetched_app(tmp_db, tz="UTC", subtract=True):
    now = _now()
    exp = _exp(subtract=subtract)
    ha = _StubHA({
        LOAD: _daily_counter_rows(now, days=5, per_bin=GROSS_PER_BIN),
        EV: _daily_counter_rows(now, days=5, per_bin=EV_PER_BIN),
    })
    app = _make_app(tmp_db, exp, ha, tz=tz)
    _run(app._fetch_and_preprocess(exp, now=now))
    return app, exp, now


# ---------------------------------------------------------------------
# Routing: which table each experiment is scored against
# ---------------------------------------------------------------------

class TestActualsTableRouting:
    def test_without_load_subtract_the_raw_target_is_read(self, tmp_db):
        db = HistoryDB(tmp_db)
        exp = _exp(subtract=False)
        assert db.actuals_table_for(exp) == db.safe_table_name(LOAD)

    def test_with_load_subtract_the_net_table_is_read(self, tmp_db):
        db = HistoryDB(tmp_db)
        exp = _exp()
        table = db.actuals_table_for(exp)
        assert table == db.net_actuals_table("house_load")
        assert table.startswith(NET_ACTUALS_PREFIX)
        assert table != db.safe_table_name(LOAD)

    def test_every_reader_routes_through_actuals_table_for(self):
        """Source contract: no scoring, band or comparison path may read
        the raw target's table directly — that is the defect."""
        root = Path(__file__).resolve().parents[2]
        main_src = (root / "ml_forecast_lab" / "main.py").read_text()
        web_src = (root / "ml_forecast_lab" / "web" / "app.py").read_text()
        assert "db.safe_table_name(exp_cfg.target_entity)" not in web_src
        assert web_src.count("db.actuals_table_for(exp_cfg)") == 4
        for fn in ("def _conformal_bands", "def _publish_forecast_sensors"):
            body = main_src[main_src.index(fn):]
            body = body[: body.index("\n    async def ", 1)]
            assert "actuals_table_for(exp_cfg)" in body, fn
            assert "safe_table_name(\n            exp_cfg.target_entity" not in body, fn

    def test_retention_covers_the_net_table(self, tmp_db):
        app = _make_app(tmp_db, _exp(), _StubHA({}))
        tables = app._retention_tables_for_experiment(_exp())
        assert app.history_db.net_actuals_table("house_load") in tables


# ---------------------------------------------------------------------
# The fetch path records the target after subtraction
# ---------------------------------------------------------------------

class TestFetchRecordsNetActuals:
    def test_todays_use_is_net_of_the_subtracted_load(self, tmp_db):
        app, exp, now = _fetched_app(tmp_db)
        bins_today = (now - now.replace(hour=0)).total_seconds() / 60 / INTERVAL
        rec = app._net_today["house_load"]
        assert rec["date"] == now.date()
        # The live bin is partial, so allow one bin either way.
        assert rec["value"] == pytest.approx(NET_PER_BIN * bins_today, abs=NET_PER_BIN)
        assert rec["value"] < GROSS_PER_BIN * bins_today * 0.7

    def test_net_table_holds_a_daily_running_total(self, tmp_db):
        app, exp, now = _fetched_app(tmp_db)
        hist = app.history_db.get_history(app.history_db.net_actuals_table("house_load"))
        assert not hist.empty
        s = hist.set_index("ds")["y"].dropna()
        # Within a day the total rises by the net use per bin...
        inc = s.diff().dropna()
        same_day = s.index[1:].normalize() == s.index[:-1].normalize()
        steady = inc[same_day][2:-2]
        assert steady.median() == pytest.approx(NET_PER_BIN, rel=0.05)
        # ...and restarts at midnight.
        first_bins = s[s.index == s.index.normalize()]
        assert (first_bins <= GROSS_PER_BIN).all()

    def test_running_total_restarts_at_local_midnight(self, tmp_db):
        """The day boundary is the configured timezone's, as for the
        cumulative sensor and the daily views."""
        app, exp, now = _fetched_app(tmp_db, tz="Europe/London")
        hist = app.history_db.get_history(app.history_db.net_actuals_table("house_load"))
        s = hist.set_index("ds")["y"].dropna()
        # 20 Oct 2026 is BST: local midnight is 23:00 UTC.
        at_23 = s[s.index == pd.Timestamp("2026-10-18 23:00:00")]
        before = s[s.index == pd.Timestamp("2026-10-18 22:30:00")]
        assert len(at_23) == 1 and len(before) == 1
        assert at_23.iloc[0] < before.iloc[0]

    def test_no_table_without_load_subtract(self, tmp_db):
        app, exp, now = _fetched_app(tmp_db, subtract=False)
        assert "house_load" not in app._net_today
        hist = app.history_db.get_history(app.history_db.net_actuals_table("house_load"))
        assert hist.empty

    def test_a_repeat_fetch_rewrites_only_the_tail(self, tmp_db):
        app, exp, now = _fetched_app(tmp_db)
        written: list[int] = []
        inner = app.history_db.upsert_history

        def spy(table, df):
            written.append(len(df))
            return inner(table, df)

        app.history_db.upsert_history = spy
        _run(app._fetch_and_preprocess(exp, now=now))
        assert written, "a second fetch must still refresh today's total"
        steps_per_day = 24 * 60 // INTERVAL
        assert written[0] <= (app._NET_ACTUALS_REWRITE_DAYS + 1) * steps_per_day


# ---------------------------------------------------------------------
# Upsert and unmeasured bins
# ---------------------------------------------------------------------

def _increments(db: HistoryDB, table: str) -> pd.Series:
    cur = db.conn.cursor()
    assert db._materialise_actuals_grid(cur, table, INTERVAL * 60, increment=True)
    rows = cur.execute(
        "SELECT grid_dt, value FROM _mlfl_actuals_vals_tmp ORDER BY grid_dt"
    ).fetchall()
    return pd.Series(
        [v for _, v in rows], index=pd.to_datetime([g for g, _ in rows]),
        dtype="float64",
    )


class TestNetTableStorage:
    def test_upsert_overwrites_an_existing_bin(self, tmp_db):
        db = HistoryDB(tmp_db)
        table = db.net_actuals_table("house_load")
        ts = pd.Timestamp("2026-10-20 10:00:00")
        db.upsert_history(table, pd.DataFrame({"ds": [ts], "value": [1.0]}))
        db.upsert_history(table, pd.DataFrame({"ds": [ts], "value": [2.5]}))
        hist = db.get_history(table)
        assert len(hist) == 1 and hist["y"].iloc[0] == pytest.approx(2.5)

    def test_an_unmeasured_bin_is_unknown_not_zero_use(self, tmp_db):
        db = HistoryDB(tmp_db)
        table = db.net_actuals_table("house_load")
        grid = pd.date_range("2026-10-20 08:00", periods=6, freq=f"{INTERVAL}min")
        running = [0.6, 1.2, np.nan, 1.8, 2.4, 3.0]
        db.upsert_history(table, pd.DataFrame({"ds": grid, "value": running}))
        inc = _increments(db, table)
        assert pd.isna(inc[grid[2]]), "an unmeasured label must not score as 0"
        assert inc[grid[3]] == pytest.approx(0.6)
        assert inc[grid[4]] == pytest.approx(0.6)

    def test_a_raw_counter_quiet_bin_is_still_zero_use(self, tmp_db):
        """Unchanged for raw tables: a change-only recorder leaves a quiet
        bin without readings, and that bin is zero use."""
        db = HistoryDB(tmp_db)
        table = db.safe_table_name(LOAD)
        grid = pd.date_range("2026-10-20 08:00", periods=6, freq=f"{INTERVAL}min")
        values = [0.6, 1.2, 1.8, 2.4]
        keep = [grid[0], grid[1], grid[3], grid[4]]
        db.store_history(table, pd.DataFrame({"ds": keep, "value": values}))
        inc = _increments(db, table)
        assert inc[grid[2]] == pytest.approx(0.0)


# ---------------------------------------------------------------------
# Scoring and the seed
# ---------------------------------------------------------------------

class TestScoredOnTheNetTarget:
    def test_accuracy_against_the_net_table_measures_the_forecast(self, tmp_db):
        app, exp, now = _fetched_app(tmp_db)
        db = app.history_db
        db.ensure_forecast_log_table()
        # A perfect forecast of the net use, issued a bin ahead.
        targets = pd.date_range(
            pd.Timestamp(now.replace(tzinfo=None)) - timedelta(hours=20),
            pd.Timestamp(now.replace(tzinfo=None)) - timedelta(hours=2),
            freq=f"{INTERVAL}min",
        )
        for t in targets:
            t = t.to_pydatetime()
            db.log_forecast(
                exp.name, t - timedelta(minutes=INTERVAL - 5), [t],
                [NET_PER_BIN], "lightgbm", model_version="v1",
            )

        def mean_abs_error(table):
            acc = db.get_forecast_accuracy(
                exp.name, table, 30, INTERVAL, evaluation_mode="increment",
            )
            maes = [m for m in acc["lead_time_curve"]["mae"] if m is not None]
            assert maes, acc
            return float(np.mean(maes))

        assert mean_abs_error(db.actuals_table_for(exp)) == pytest.approx(0.0, abs=0.05)
        assert mean_abs_error(db.safe_table_name(LOAD)) == pytest.approx(
            EV_PER_BIN, abs=0.05,
        ), "the raw target's table scores the subtracted load as error"

    def test_cumulative_sensor_is_seeded_with_net_use(self, tmp_db):
        exp = _exp()
        gross_reading = "28.0"
        ha = _StubHA({}, states={LOAD: gross_reading})
        app = _make_app(tmp_db, exp, ha)
        app._cached_models = {exp.name: {"model_version": "v1"}}
        issued = datetime.now(timezone.utc)
        app._net_today[exp.name] = {"date": issued.date(), "value": 11.5}
        start = pd.Timestamp(issued.replace(tzinfo=None, second=0, microsecond=0))
        ds_future = pd.DatetimeIndex(
            [start + pd.Timedelta(minutes=INTERVAL * (i + 1)) for i in range(4)]
        )
        _run(app._publish_forecast_sensors(
            exp_cfg=exp, y_pred=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            ds_future=ds_future, model_name="lightgbm",
            last_trained_iso="2026-10-01T00:00:00Z",
        ))
        state, attrs = ha.captured["sensor.mlfl_house_load_cumulative"]
        assert attrs["seeded_with"] == pytest.approx(11.5)
        assert LOAD not in ha.state_reads, (
            "the raw target's reading includes the subtracted loads"
        )

    def test_without_load_subtract_the_seed_is_the_targets_reading(self, tmp_db):
        exp = _exp(subtract=False)
        ha = _StubHA({}, states={LOAD: "28.0"})
        app = _make_app(tmp_db, exp, ha)
        app._cached_models = {exp.name: {"model_version": "v1"}}
        start = pd.Timestamp(datetime.utcnow().replace(second=0, microsecond=0))
        ds_future = pd.DatetimeIndex(
            [start + pd.Timedelta(minutes=INTERVAL * (i + 1)) for i in range(4)]
        )
        _run(app._publish_forecast_sensors(
            exp_cfg=exp, y_pred=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            ds_future=ds_future, model_name="lightgbm",
            last_trained_iso="2026-10-01T00:00:00Z",
        ))
        _state, attrs = ha.captured["sensor.mlfl_house_load_cumulative"]
        assert attrs["seeded_with"] == pytest.approx(28.0)


# ---------------------------------------------------------------------
# Replay doubles
# ---------------------------------------------------------------------

class TestReplayDoubles:
    def test_capture_and_replay_suppress_the_upsert(self, tmp_db):
        from ml_forecast_lab.replay import RecordingHistoryDB, ReplayHistoryDB, _CallQueue

        inner = HistoryDB(tmp_db)
        rec = RecordingHistoryDB(inner, log=None)
        rep = ReplayHistoryDB(_CallQueue([]))
        frame = pd.DataFrame({"ds": [pd.Timestamp("2026-10-20 10:00")], "value": [1.0]})
        for db in (rec, rep):
            assert db.upsert_history("net_actuals_house_load", frame) == 0
            assert db.actuals_table_for(_exp()) == inner.net_actuals_table("house_load")
            assert db.actuals_table_for(_exp(subtract=False)) == inner.safe_table_name(LOAD)
        assert inner.get_history(inner.net_actuals_table("house_load")).empty
