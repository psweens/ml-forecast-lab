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
table (``HistoryDB.net_actuals_table``): each grid bin's value — its use, for
a cumulative target — NULL where the label is unmeasured, and only for bins
the fetch measured in full, so a restart's rewrite never replaces a good row
with a worse one. ``HistoryDB.actuals_table_for`` routes every reader to it;
the readers take the rows as they are, so no day boundary has to be guessed
from the data, and a bin with a NULL value or without a row is unknown. The
seed is today's use after subtraction. Experiments without ``load_subtract``
read exactly what they read before.
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

    def test_net_table_holds_each_bins_use(self, tmp_db):
        app, exp, now = _fetched_app(tmp_db)
        hist = app.history_db.get_history(app.history_db.net_actuals_table("house_load"))
        assert not hist.empty
        s = hist.set_index("ds")["y"].dropna()
        assert s.median() == pytest.approx(NET_PER_BIN, rel=0.05)
        assert s.max() == pytest.approx(NET_PER_BIN, rel=0.05), (
            "a bin's own use, not a running total"
        )

    def test_the_windows_first_local_day_is_not_written(self, tmp_db):
        """The window opens mid-day: its first reading has no rise and the
        deficit carried from earlier that day is missing, so those bins
        would be short — and a restart rewrites the window over rows an
        earlier run wrote whole."""
        app, exp, now = _fetched_app(tmp_db, tz="Europe/London")
        hist = app.history_db.get_history(app.history_db.net_actuals_table("house_load"))
        # The window opens 16 Oct 14:00 UTC (15:00 BST); the next local
        # midnight is 23:00 UTC.
        assert now - timedelta(days=exp.days_history) == datetime(
            2026, 10, 16, 14, 0, tzinfo=timezone.utc,
        )
        assert hist["ds"].min() == pd.Timestamp("2026-10-16 23:00:00")

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
# What a rewrite may replace
# ---------------------------------------------------------------------

BOILER = "sensor.boiler_energy_today"
BOILER_PER_BIN = 0.1
# HA's recorder keeps the subtracted sensor's history from here on; the
# app's own cache still holds the target's whole window.
RECORDER_FLOOR = pd.Timestamp("2026-10-19 06:00", tz="UTC")


def _since_floor(rows):
    return [r for r in rows if pd.Timestamp(r["last_changed"]) >= RECORDER_FLOOR]


class TestNetTableRewrites:
    def test_nothing_is_written_before_a_subtracted_sensors_history(self, tmp_db):
        """There the target would be written with nothing taken off."""
        now = _now()
        exp = _exp()
        ha = _StubHA({
            LOAD: _daily_counter_rows(now, days=5, per_bin=GROSS_PER_BIN),
            EV: _since_floor(_daily_counter_rows(now, days=5, per_bin=EV_PER_BIN)),
        })
        app = _make_app(tmp_db, exp, ha)
        _run(app._fetch_and_preprocess(exp, now=now))
        s = app.history_db.get_history(
            app.history_db.net_actuals_table(exp.name),
        ).set_index("ds")["y"].dropna()
        # The sensor's first bin is partial, so the next local midnight.
        assert s.index.min() == pd.Timestamp("2026-10-20 00:00:00")
        assert s.max() == pytest.approx(NET_PER_BIN, rel=0.05)

    def test_a_restart_keeps_rows_the_recorder_no_longer_covers(self, tmp_db):
        now = _now()
        exp = _exp()
        load = _daily_counter_rows(now, days=5, per_bin=GROSS_PER_BIN)
        ev = _daily_counter_rows(now, days=5, per_bin=EV_PER_BIN)
        app = _make_app(tmp_db, exp, _StubHA({LOAD: load, EV: ev}))
        _run(app._fetch_and_preprocess(exp, now=now))
        table = app.history_db.net_actuals_table(exp.name)
        before = app.history_db.get_history(table).set_index("ds")["y"]

        # A new run on the same database, after HA's recorder purged the
        # older history: the target comes from the app's cache.
        later = _make_app(
            tmp_db, exp, _StubHA({LOAD: _since_floor(load), EV: _since_floor(ev)}),
        )
        _run(later._fetch_and_preprocess(exp, now=now))
        after = later.history_db.get_history(table).set_index("ds")["y"]
        old = before.index[before.index < pd.Timestamp("2026-10-20 00:00:00")]
        assert len(old) > 48
        pd.testing.assert_series_equal(after.loc[old], before.loc[old])

    def test_a_failed_subtract_fetch_leaves_the_table_alone(self, tmp_db):
        now = _now()
        exp = _exp(load_subtract=[
            SubtractCfg(entity_id=EV, source="cumulative_daily", on_missing="zero"),
            SubtractCfg(entity_id=BOILER, source="cumulative_daily", on_missing="zero"),
        ])
        ha = _StubHA({
            LOAD: _daily_counter_rows(now, days=5, per_bin=GROSS_PER_BIN),
            EV: _daily_counter_rows(now, days=5, per_bin=EV_PER_BIN),
            BOILER: _daily_counter_rows(now, days=5, per_bin=BOILER_PER_BIN),
        })
        app = _make_app(tmp_db, exp, ha)
        _run(app._fetch_and_preprocess(exp, now=now))
        table = app.history_db.net_actuals_table(exp.name)
        before = app.history_db.get_history(table)
        assert before["y"].median() == pytest.approx(
            GROSS_PER_BIN - EV_PER_BIN - BOILER_PER_BIN, rel=0.05,
        )

        inner = ha.get_history

        async def ev_unavailable(entity_id, *args, **kwargs):
            if entity_id == EV:
                raise RuntimeError("HA unavailable")
            return await inner(entity_id, *args, **kwargs)

        ha.get_history = ev_unavailable
        app._net_actuals_backfilled.clear()   # as after a restart
        _run(app._fetch_and_preprocess(exp, now=now))
        pd.testing.assert_frame_equal(app.history_db.get_history(table), before)


# ---------------------------------------------------------------------
# Upsert and unmeasured bins
# ---------------------------------------------------------------------

def _increments(db: HistoryDB, table: str, interval_sec: int = INTERVAL * 60) -> pd.Series:
    cur = db.conn.cursor()
    assert db._materialise_actuals_grid(cur, table, interval_sec, increment=True)
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
        per_bin = [0.6, 0.6, np.nan, 0.6, 0.6, 0.6]
        db.upsert_history(table, pd.DataFrame({"ds": grid, "value": per_bin}))
        inc = _increments(db, table)
        assert pd.isna(inc[grid[2]]), "an unmeasured label must not score as 0"
        assert inc[grid[3]] == pytest.approx(0.6)
        assert inc[grid[4]] == pytest.approx(0.6)

    def test_a_bin_without_a_row_is_unknown_not_zero_use(self, tmp_db):
        """A run that stops writing for longer than the rewritten tail
        leaves bins without rows."""
        db = HistoryDB(tmp_db)
        table = db.net_actuals_table("house_load")
        grid = pd.date_range("2026-10-20 08:00", periods=6, freq=f"{INTERVAL}min")
        keep = [grid[0], grid[1], grid[3], grid[4]]
        db.upsert_history(table, pd.DataFrame({"ds": keep, "value": [0.6] * 4}))
        inc = _increments(db, table)
        assert grid[2] not in inc.index
        assert inc[grid[3]] == pytest.approx(0.6)

    def test_rows_are_read_as_each_bins_use(self, tmp_db):
        """With a daily interval each bin is its own day. Read as a counter,
        a day using at least 90% of the day before was not taken for a
        restart and scored the difference (11.9 kWh as 1.9)."""
        db = HistoryDB(tmp_db)
        table = db.net_actuals_table("house_load")
        days = pd.date_range("2026-10-01", periods=5, freq="1D")
        use = [10.0, 11.9, 9.5, 0.2, 0.19]
        db.upsert_history(table, pd.DataFrame({"ds": days, "value": use}))
        inc = _increments(db, table, interval_sec=86400)
        assert inc.reindex(days).to_numpy() == pytest.approx(use)

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

    def test_comparison_reads_the_net_rows_as_they_are(self, tmp_db):
        db = HistoryDB(tmp_db)
        db.ensure_forecast_log_table()
        db.ensure_external_forecast_log_table()
        table = db.net_actuals_table("house_load")
        grid = pd.date_range("2026-10-18 00:00", periods=96, freq=f"{INTERVAL}min")
        use = 0.3 + 0.2 * np.sin(np.arange(96) / 4.0)
        db.upsert_history(table, pd.DataFrame({"ds": grid, "value": use}))
        for t, u in zip(grid, use):
            t = t.to_pydatetime()
            db.log_forecast("house_load", t - timedelta(minutes=INTERVAL), [t],
                            [u + 0.01], "lightgbm", model_version="v1")
        ext = db.safe_table_name("sensor.ext_forecast")
        db.store_history(ext, pd.DataFrame({"ds": grid, "value": use + 0.2}))
        res = db.get_external_forecast_comparison(
            "house_load", table,
            [{"entity": "sensor.ext_forecast", "mode": "state", "table": ext,
              "scale": None, "is_cumulative": False, "label": "ext"}],
            3650, INTERVAL, "increment",
        )
        h = res["comparisons"][0]["head_to_head"]
        assert h["app"]["mae"] == pytest.approx(0.01, abs=1e-4)
        assert h["external"]["mae"] == pytest.approx(0.2, abs=1e-4)

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
# Daily-cumulative accuracy
# ---------------------------------------------------------------------

def _perfect_daily_forecasts(db, table, values):
    """Net rows over the last three days and, every hour, an exact
    forecast of the next 12 bins."""
    end = pd.Timestamp(datetime.utcnow().replace(minute=0, second=0, microsecond=0))
    grid = pd.date_range(end - pd.Timedelta(days=3), end - pd.Timedelta(hours=1),
                         freq=f"{INTERVAL}min")
    values = values(len(grid))
    db.upsert_history(table, pd.DataFrame({"ds": grid, "value": values}))
    db.ensure_forecast_log_table()
    exact = pd.Series(values, index=grid).fillna(NET_PER_BIN)
    for issued in grid[::2][:-12]:
        targets = grid[grid > issued][:12]
        db.log_forecast(
            "house_load", issued.to_pydatetime(),
            [t.to_pydatetime() for t in targets], list(exact[targets]),
            "lightgbm", model_version="v1",
        )


def _daily_cumulative(db, table):
    acc = db.get_forecast_accuracy(
        "house_load", table, 30, INTERVAL, evaluation_mode="daily_cumulative",
        day_offset_hours=1.0, cumulative_source=True,
    )
    curve = acc["lead_time_curve"]
    maes = [m for m in curve["mae"] if m is not None]
    assert maes, acc
    return max(maes), sum(curve["sample_count"])


class TestDailyCumulativeOnNetRows:
    def test_a_perfect_forecast_scores_zero_in_any_time_zone(self, tmp_db):
        """HA's time zone splits the days (here UTC+1). The rows carry no
        day boundary of their own, so the configured timezone can't
        disagree with it."""
        db = HistoryDB(tmp_db)
        table = db.net_actuals_table("house_load")
        _perfect_daily_forecasts(db, table, lambda n: np.full(n, NET_PER_BIN))
        worst, _n = _daily_cumulative(db, table)
        assert worst == pytest.approx(0.0, abs=1e-6)

    def test_an_unmeasured_bin_makes_the_rest_of_its_day_unknown(self, tmp_db):
        """Summing past it would score a perfect forecast as over-predicting
        by the unmeasured bin's use."""
        db = HistoryDB(tmp_db)
        table = db.net_actuals_table("house_load")

        def with_a_hole(n):
            v = np.full(n, NET_PER_BIN)
            v[n // 2] = np.nan
            return v

        _perfect_daily_forecasts(db, table, with_a_hole)
        worst, n_hole = _daily_cumulative(db, table)
        assert worst == pytest.approx(0.0, abs=1e-6)

        full = HistoryDB(str(Path(tmp_db).with_name("full.db")))
        _perfect_daily_forecasts(full, table, lambda n: np.full(n, NET_PER_BIN))
        _worst, n_full = _daily_cumulative(full, table)
        assert n_hole < n_full


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
