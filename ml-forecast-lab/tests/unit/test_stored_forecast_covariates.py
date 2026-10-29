"""
v2.57.0: covariates that learn from the forecast as issued.

A ``role: future`` covariate trained on its sensor's history learns from
what the sensor measured, then forecasts from what its forecast says. With
``history_source: stored_forecast`` each grid bin's history is instead the
value from the latest stored trajectory issued at or before the bin's start,
so training sees what forecasting sees. ``future_extra_entities`` joins a
forecast published across several sensors (e.g. one per day), which a single
sensor could not cover: past its last entry the horizon held that entry,
so a day-split forecast read the next day as its final value.

Pinned here:
  - config: the new fields, their validation and normalisation;
  - ``stored_forecast_on_grid``: latest issuance at or before each bin,
    interpolation within a trajectory, no fill outside its span;
  - ``HistoryDB``: storing with content de-duplication, reading across the
    covariate's and the Forecast Comparison's stored trajectories, and the
    backfill/prune maintenance;
  - ``CovariateResolver``: extra sensors merged (main sensor wins), the live
    trajectory stored only for stored-forecast covariates and only from its
    own points, history routed to the stored forecasts;
  - ``MLForecastLabApp``: the training frame carries the stored forecast, the
    retention is the longest ``days_history`` sharing the forecast, and the
    comparison capture no longer logs values beyond a source's own span;
  - replay: the history read is recorded and served back, the store write
    suppressed.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
import yaml

from ml_forecast_lab.config import (
    AppConfig, CovariateCfg, ExperimentCfg, ExternalForecastCfg,
    STORED_FORECAST_PREFIX, load_config, stored_forecast_source,
)
from ml_forecast_lab.covariates import CovariateResolver, align_future_points
from ml_forecast_lab.db import HistoryDB, stored_forecast_on_grid


INTERVAL = 30
FREQ = "30min"
SRC = "sensor.fc|forecast|power"


def _run(coro):
    return asyncio.run(coro)


def _rows(*trajs):
    """(issued, {target: value}) pairs → the rows frame the grid reader takes."""
    out = []
    for issued, traj in trajs:
        for t, v in traj.items():
            out.append((pd.Timestamp(issued), pd.Timestamp(t), float(v)))
    return pd.DataFrame(out, columns=["issued_at", "target_dt", "value"])


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        c = CovariateCfg(entity="sensor.fc", role="future")
        assert c.history_source == "sensor"
        assert c.future_extra_entities == []

    def test_validation(self):
        with pytest.raises(ValueError):
            CovariateCfg(entity="sensor.fc", role="future", history_source="guess")
        # A lagged covariate has no forecast to store.
        with pytest.raises(ValueError):
            CovariateCfg(entity="sensor.fc", role="lagged", history_source="stored_forecast")
        CovariateCfg(entity="sensor.fc", role="both", history_source="stored_forecast")

    def test_extra_entities_normalised(self):
        c = CovariateCfg(entity="sensor.fc", role="future",
                         future_extra_entities="sensor.fc_tomorrow")
        assert c.future_extra_entities == ["sensor.fc_tomorrow"]
        c = CovariateCfg(entity="sensor.fc", role="future",
                         future_extra_entities=[" sensor.a ", "", "sensor.fc", "sensor.b"])
        assert c.future_extra_entities == ["sensor.a", "sensor.b"]

    def test_yaml_roundtrip(self, tmp_path):
        p = tmp_path / "mlfl.yaml"
        p.write_text(yaml.dump({"experiments": [{
            "name": "e", "target_entity": "sensor.t", "covariates": [{
                "entity": "sensor.fc", "role": "future",
                "future_attribute": "forecast", "future_value_key": "power",
                "future_extra_entities": ["sensor.fc_tomorrow"],
                "history_source": "stored_forecast",
            }],
        }]}))
        c = load_config(p).experiments[0].covariates[0]
        assert c.history_source == "stored_forecast"
        assert c.future_extra_entities == ["sensor.fc_tomorrow"]

    def test_source_matches_the_comparison_key(self):
        # The covariate reads trajectories the comparison tab stored for the
        # same entity, attribute and value key.
        ext = ExternalForecastCfg(entity_id="sensor.fc", mode="attribute",
                                  attribute="forecast", value_key="power")
        assert stored_forecast_source("sensor.fc", "forecast", "power") == ext.source_key
        assert stored_forecast_source("sensor.fc", None, None) == "sensor.fc|forecast|"


# ---------------------------------------------------------------------
# The grid reader
# ---------------------------------------------------------------------

class TestOnGrid:
    GRID = pd.date_range("2026-10-01 06:00", "2026-10-01 12:00", freq=FREQ)

    def test_latest_issuance_at_or_before_each_bin(self):
        rows = _rows(
            ("2026-10-01 05:00", {f"2026-10-01 {h:02d}:00": 1.0 for h in range(5, 13)}),
            ("2026-10-01 09:00", {f"2026-10-01 {h:02d}:00": 2.0 for h in range(9, 13)}),
        )
        s = stored_forecast_on_grid(rows, self.GRID)
        assert (s[:"2026-10-01 08:30"] == 1.0).all()
        # The 09:00 bin starts when the second trajectory was issued.
        assert (s["2026-10-01 09:00":] == 2.0).all()

    def test_interpolates_within_a_trajectory(self):
        rows = _rows(("2026-10-01 05:00", {"2026-10-01 06:00": 0.0,
                                           "2026-10-01 07:00": 2.0}))
        s = stored_forecast_on_grid(rows, self.GRID)
        assert s["2026-10-01 06:30"] == pytest.approx(1.0)

    def test_no_value_outside_the_chosen_trajectory(self):
        # The later trajectory covers only 09:00-10:00; 10:30 onwards is not
        # filled from it, nor from the older trajectory that did cover it.
        rows = _rows(
            ("2026-10-01 05:00", {f"2026-10-01 {h:02d}:00": 1.0 for h in range(5, 13)}),
            ("2026-10-01 09:00", {"2026-10-01 09:00": 2.0, "2026-10-01 10:00": 2.0}),
        )
        s = stored_forecast_on_grid(rows, self.GRID)
        assert s["2026-10-01 10:00"] == 2.0
        assert s["2026-10-01 10:30":].isna().all()

    def test_bins_before_any_issuance_are_unmeasured(self):
        rows = _rows(("2026-10-01 08:00", {f"2026-10-01 {h:02d}:00": 3.0 for h in range(6, 13)}))
        s = stored_forecast_on_grid(rows, self.GRID)
        assert s[:"2026-10-01 07:30"].isna().all()
        assert (s["2026-10-01 08:00":] == 3.0).all()

    def test_duplicate_rows_keep_the_first(self):
        rows = pd.concat([
            _rows(("2026-10-01 05:00", {"2026-10-01 06:00": 1.0, "2026-10-01 12:00": 1.0})),
            _rows(("2026-10-01 05:00", {"2026-10-01 06:00": 9.0, "2026-10-01 12:00": 9.0})),
        ], ignore_index=True)
        assert (stored_forecast_on_grid(rows, self.GRID) == 1.0).all()

    def test_empty(self):
        assert stored_forecast_on_grid(pd.DataFrame(columns=["issued_at", "target_dt", "value"]),
                                       self.GRID).isna().all()


def _reference_on_grid(rows, grid):
    """The grid rule written the slow, obvious way."""
    out = []
    for t in grid:
        past = rows[rows["issued_at"] <= t]
        if past.empty:
            out.append(np.nan)
            continue
        traj = past[past["issued_at"] == past["issued_at"].max()]
        traj = traj.drop_duplicates("target_dt").sort_values("target_dt")
        x = traj["target_dt"].values.astype("int64")
        out.append(float(np.interp(t.value, x, traj["value"].values,
                                   left=np.nan, right=np.nan)))
    return pd.Series(out, index=grid)


def _random_store(seed, n_issuances=40, step_min=30):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-10-01")
    gaps = rng.integers(10, 240, n_issuances)
    issued = t0 + pd.to_timedelta(np.cumsum(gaps), unit="m")
    out = []
    for iss in issued:
        first = iss.floor(f"{step_min}min") - pd.Timedelta(minutes=step_min)
        n = int(rng.integers(2, 40))
        for k in range(n):
            out.append((iss, first + pd.Timedelta(minutes=step_min * k), float(rng.normal())))
    return pd.DataFrame(out, columns=["issued_at", "target_dt", "value"])


class TestOnGridEquivalence:
    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("step_min", [30, 60])
    def test_matches_the_reference(self, seed, step_min):
        rows = _random_store(seed, step_min=step_min)
        grid = pd.date_range(rows["target_dt"].min() - pd.Timedelta(hours=2),
                             rows["target_dt"].max() + pd.Timedelta(hours=2), freq=FREQ)
        got = stored_forecast_on_grid(rows, grid)
        want = _reference_on_grid(rows, grid)
        assert got.isna().equals(want.isna())
        assert np.allclose(got.dropna(), want.dropna())


# ---------------------------------------------------------------------
# HistoryDB
# ---------------------------------------------------------------------

@pytest.fixture
def db(tmp_db):
    h = HistoryDB(tmp_db)
    h.ensure_forecast_log_table()
    h.ensure_external_forecast_log_table()
    yield h
    h.close()


def _targets(start, n, step=INTERVAL):
    return [datetime.fromisoformat(start) + timedelta(minutes=step * i) for i in range(n)]


class TestStore:
    def test_content_dedup(self, db):
        t = _targets("2026-10-01 06:00", 4)
        assert db.store_covariate_forecast(SRC, datetime(2026, 10, 1, 5), t, [1, 2, 3, 4]) == 4
        # Same values, later capture: same issuance, not stored.
        assert db.store_covariate_forecast(SRC, datetime(2026, 10, 1, 5, 30), t, [1, 2, 3, 4]) == 0
        assert db.store_covariate_forecast(SRC, datetime(2026, 10, 1, 6), t, [1, 2, 3, 5]) == 4
        cur = db.conn.cursor()
        cur.execute("SELECT DISTINCT experiment, source FROM external_forecast_log")
        assert [tuple(r) for r in cur.fetchall()] == [("", STORED_FORECAST_PREFIX + SRC)]

    def test_history_reads_both_stores(self, db):
        # Comparison rows from another experiment cover 06:00-07:30; the
        # covariate's own trajectory, issued later, takes over from 08:00.
        db.log_external_forecast("other", SRC, datetime(2026, 10, 1, 5),
                                 _targets("2026-10-01 06:00", 6), [1.0] * 6)
        db.store_covariate_forecast(SRC, datetime(2026, 10, 1, 8),
                                    _targets("2026-10-01 08:00", 4), [2.0] * 4)
        df = db.get_stored_forecast_history(
            SRC, datetime(2026, 10, 1, 6), datetime(2026, 10, 1, 9, 30), FREQ)
        s = pd.Series(df["y"].values, index=pd.DatetimeIndex(df["ds"]))
        assert list(df.columns) == ["ds", "y"]
        assert (s[:"2026-10-01 07:30"] == 1.0).all()
        assert (s["2026-10-01 08:00":] == 2.0).all()
        # Another source is never read.
        db.log_external_forecast("other", "sensor.x|forecast|power", datetime(2026, 10, 1, 9),
                                 _targets("2026-10-01 09:00", 2), [7.0] * 2)
        df2 = db.get_stored_forecast_history(
            SRC, datetime(2026, 10, 1, 6), datetime(2026, 10, 1, 9, 30), FREQ)
        assert 7.0 not in set(df2["y"])

    def test_covariate_rows_win_an_exact_duplicate(self, db):
        t = _targets("2026-10-01 06:00", 2)
        db.log_external_forecast("other", SRC, datetime(2026, 10, 1, 5), t, [9.0, 9.0])
        db.store_covariate_forecast(SRC, datetime(2026, 10, 1, 5), t, [1.0, 1.0])
        df = db.get_stored_forecast_history(
            SRC, datetime(2026, 10, 1, 6), datetime(2026, 10, 1, 6, 30), FREQ)
        assert set(df["y"]) == {1.0}

    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("step_min", [30, 60])
    def test_reading_only_the_usable_rows_changes_nothing(self, db, seed, step_min):
        # The reader skips each trajectory's targets past the next issuance;
        # the result must equal the rule over every stored row. Half the
        # trajectories come from the comparison's store.
        rows = _random_store(seed, step_min=step_min)
        for i, (iss, part) in enumerate(rows.groupby("issued_at")):
            targets = [t.to_pydatetime() for t in part["target_dt"]]
            if i % 2:
                db.log_external_forecast("other", SRC, iss.to_pydatetime(), targets,
                                         list(part["value"]))
            else:
                db.log_external_forecast("", STORED_FORECAST_PREFIX + SRC, iss.to_pydatetime(),
                                         targets, list(part["value"]))
        start, end = rows["target_dt"].min(), rows["target_dt"].max()
        df = db.get_stored_forecast_history(SRC, start.to_pydatetime(), end.to_pydatetime(), FREQ)
        got = pd.Series(df["y"].values, index=pd.DatetimeIndex(df["ds"]))
        grid = pd.date_range(start.floor(FREQ), end.floor(FREQ), freq=FREQ)
        stored = rows.assign(value=rows["value"].round(6))
        want = _reference_on_grid(stored, grid).dropna()
        assert list(got.index) == list(want.index)
        assert np.allclose(got.values, want.values, atol=1e-6)

    def test_maintenance_backfills_then_prunes(self, db):
        now = datetime(2026, 10, 30, 12)
        # Comparison rows: 40, 20 and 1 day(s) old; the covariate started
        # storing its own 5 days ago.
        for age in (40, 20, 1):
            iss = now - timedelta(days=age)
            db.log_external_forecast("other", SRC, iss, [iss + timedelta(hours=1)], [float(age)])
        db.store_covariate_forecast(SRC, now - timedelta(days=5),
                                    [now - timedelta(days=5) + timedelta(hours=1)], [5.0])
        res = db.maintain_stored_forecasts(SRC, keep_days=30, now=now)
        # 20-day-old row copied (older than the covariate's first own row,
        # inside 30 days); the 1-day-old one is not; the 40-day-old one is out
        # of retention.
        assert res == {"copied": 1, "pruned": 0}
        cur = db.conn.cursor()
        cur.execute("SELECT value FROM external_forecast_log WHERE experiment = '' "
                    "ORDER BY issued_at")
        assert [r[0] for r in cur.fetchall()] == [20.0, 5.0]
        # Idempotent.
        assert db.maintain_stored_forecasts(SRC, keep_days=30, now=now)["copied"] == 0
        # The comparison's own rows are left to that tab.
        cur.execute("SELECT COUNT(*) FROM external_forecast_log WHERE experiment = 'other'")
        assert cur.fetchone()[0] == 3
        # Shorter retention prunes the covariate's rows only.
        res = db.maintain_stored_forecasts(SRC, keep_days=10, now=now)
        assert res["pruned"] == 1
        cur.execute("SELECT COUNT(*) FROM external_forecast_log WHERE experiment = 'other'")
        assert cur.fetchone()[0] == 3


# ---------------------------------------------------------------------
# CovariateResolver
# ---------------------------------------------------------------------

class _AttrHA:
    """Serves forecast attributes; history is never asked for here."""

    def __init__(self, attrs):
        self.attrs = attrs
        self.asked = []

    async def get_state(self, entity_id, default=None, attribute=None):
        self.asked.append((entity_id, attribute))
        return self.attrs.get((entity_id, attribute), default)

    async def get_history(self, *a, **k):  # pragma: no cover - not reached
        raise AssertionError("history must come from the stored forecasts")


class _RecordingDB:
    def __init__(self):
        self.stored = []
        self.reads = []

    def store_covariate_forecast(self, source, issued_at, targets, values):
        self.stored.append((source, issued_at, targets, values))
        return len(targets)

    def get_stored_forecast_history(self, source, start, end, freq):
        self.reads.append((source, start, end, freq))
        return pd.DataFrame({"ds": pd.to_datetime(["2026-10-01 06:00", "2026-10-01 06:30"]),
                             "y": [1.5, 2.5]})


def _entries(start, n, value, step=INTERVAL):
    t0 = pd.Timestamp(start, tz="UTC")
    return [{"period_start": (t0 + pd.Timedelta(minutes=step * i)).isoformat(),
             "power": value(i)} for i in range(n)]


class TestResolver:
    def test_extra_sensor_continues_the_forecast(self):
        # Today's sensor ends at 23:30; tomorrow's carries on. The main
        # sensor's value wins where they overlap.
        ha = _AttrHA({
            ("sensor.fc", "forecast"): _entries("2026-10-01 22:00", 4, lambda i: 1.0),
            ("sensor.fc_tomorrow", "forecast"): _entries("2026-10-01 23:30", 4, lambda i: 5.0),
        })
        r = CovariateResolver(ha)
        cfg = {"entity_id": "sensor.fc", "future_attribute": "forecast",
               "future_value_key": "power", "future_extra_entities": ["sensor.fc_tomorrow"]}
        idx = pd.date_range("2026-10-01 22:00", periods=6, freq=FREQ, tz="UTC")
        out = _run(r.fetch_future(cfg, idx))
        assert list(out.values) == [1.0, 1.0, 1.0, 1.0, 5.0, 5.0]

    def test_without_the_extra_sensor_the_horizon_holds_the_last_entry(self):
        # The behaviour a day-split forecast had before.
        ha = _AttrHA({("sensor.fc", "forecast"): _entries("2026-10-01 22:00", 4, lambda i: 1.0)})
        idx = pd.date_range("2026-10-01 22:00", periods=6, freq=FREQ, tz="UTC")
        out = _run(CovariateResolver(ha).fetch_future(
            {"entity_id": "sensor.fc", "future_value_key": "power"}, idx))
        assert list(out.values) == [1.0] * 6

    def test_points_are_naive_utc_and_unfilled(self):
        ha = _AttrHA({("sensor.fc", "forecast"): [
            {"period_start": "2026-10-01T10:00:00+01:00", "power": 2.0},
            {"period_start": "2026-10-01T10:30:00+01:00", "power": 3.0},
        ]})
        pts = _run(CovariateResolver(ha).fetch_future_points(
            {"entity_id": "sensor.fc", "future_value_key": "power"}))
        assert list(pts.index) == [pd.Timestamp("2026-10-01 09:00"), pd.Timestamp("2026-10-01 09:30")]
        assert pts.index.tz is None

    def test_live_forecast_stored_only_for_stored_forecast_covariates(self):
        now = pd.Timestamp(datetime.utcnow()).floor(FREQ)
        entries = _entries(str(now - pd.Timedelta(hours=4)), 16, lambda i: float(i))
        ha = _AttrHA({("sensor.fc", "forecast"): entries})
        db = _RecordingDB()
        r = CovariateResolver(ha, history_db=db)
        idx = pd.date_range(now, periods=4, freq=FREQ, tz="UTC")
        base = {"entity_id": "sensor.fc", "future_attribute": "forecast",
                "future_value_key": "power"}
        _run(r.fetch_future(base, idx))
        assert db.stored == []
        _run(r.fetch_future(dict(base, history_source="stored_forecast",
                                 stored_source=SRC), idx))
        assert len(db.stored) == 1
        source, issued, targets, values = db.stored[0]
        assert source == SRC
        # The source's own points only (16 entries from 4 h ago), from two
        # hours before the capture; not the 4-bin horizon it was resolved to.
        cutoff = pd.Timestamp(issued) - pd.Timedelta(hours=2)
        all_points = [pd.Timestamp(e["period_start"]).tz_convert(None) for e in entries]
        assert targets == [t.to_pydatetime() for t in all_points if t >= cutoff]
        assert max(targets) == (now + pd.Timedelta(hours=3, minutes=30)).to_pydatetime()
        assert len(targets) > 4

    def test_history_comes_from_the_stored_forecasts(self):
        db = _RecordingDB()
        r = CovariateResolver(_AttrHA({}), history_db=db)
        cfg = {"entity_id": "sensor.fc", "name": "fc", "future_attribute": "forecast",
               "future_value_key": "power", "history_source": "stored_forecast"}
        s = _run(r.fetch_history(cfg, datetime(2026, 10, 1, 6), datetime(2026, 10, 1, 7), FREQ))
        assert db.reads and db.reads[0][0] == SRC
        assert list(s.values) == [1.5, 2.5] and s.name == "fc"

    def test_stored_forecast_history_without_a_database_is_empty(self):
        r = CovariateResolver(_AttrHA({}))
        s = _run(r.fetch_history(
            {"entity_id": "sensor.fc", "history_source": "stored_forecast"},
            datetime(2026, 10, 1, 6), datetime(2026, 10, 1, 7), FREQ))
        assert s.empty


# ---------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------

class _StubHA(_AttrHA):
    def __init__(self, rows_by_entity, attrs=None):
        super().__init__(attrs or {})
        self.rows_by_entity = rows_by_entity

    async def get_history(self, entity_id, start, end, include_attributes=False):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        s = s.tz_localize("UTC") if s.tzinfo is None else s
        e = e.tz_localize("UTC") if e.tzinfo is None else e
        return [r for r in self.rows_by_entity.get(entity_id, [])
                if s <= pd.Timestamp(r["last_changed"]) <= e]


def _recorder_rows(entity_now, days, cadence_min=15):
    ts = pd.date_range(entity_now - timedelta(days=days), entity_now,
                       freq=f"{cadence_min}min", tz="UTC")
    return [{"last_changed": t.isoformat(), "state": f"{i % 37 + 1.0:.4f}"}
            for i, t in enumerate(ts)]


def _make_app(tmp_db, experiments, ha):
    from ml_forecast_lab.main import MLForecastLabApp
    app = MLForecastLabApp()
    app.history_db = HistoryDB(tmp_db)
    app.history_db.ensure_external_forecast_log_table()
    app.ha_interface = ha
    app.config = AppConfig(experiments=experiments)
    app.covariate_resolver = CovariateResolver(
        ha, history_db=app.history_db,
        retention_provider=app._retention_days_for_table,
    )
    return app


def _exp(name="e", days=5, covariates=None, **kw):
    return ExperimentCfg(name=name, target_entity="sensor.load", days_history=days,
                         interval_minutes=INTERVAL, models_enabled=["lightgbm"],
                         covariates=covariates or [], **kw)


def _stored_cov(**kw):
    params = dict(entity="sensor.fc", role="future", future_attribute="forecast",
                  future_value_key="power", history_source="stored_forecast")
    params.update(kw)
    return CovariateCfg(**params)


class TestApp:
    def test_training_frame_carries_the_stored_forecast(self, tmp_db):
        from ml_forecast_lab.main import _cov_column_name
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        cov = _stored_cov()
        exp = _exp(covariates=[cov])
        app = _make_app(tmp_db, [exp], _StubHA({"sensor.load": _recorder_rows(now, days=6)}))
        # One stored trajectory, issued 2 days ago, covering the next 12 h.
        naive_now = now.replace(tzinfo=None)
        issued = (pd.Timestamp(naive_now) - pd.Timedelta(days=2)).floor(FREQ).to_pydatetime()
        targets = [issued + timedelta(minutes=INTERVAL * i) for i in range(24)]
        app.history_db.store_covariate_forecast(SRC, issued, targets, [float(i) for i in range(24)])

        result = _run(app._fetch_and_preprocess(exp))
        col = result[_cov_column_name(cov, all_covs=exp.covariates)]
        idx = pd.DatetimeIndex(result.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        got = pd.Series(col.values, index=idx)
        want = pd.Series([float(i) for i in range(24)], index=pd.DatetimeIndex(targets))
        common = want.index.intersection(got.index)
        assert len(common) >= 20
        assert np.allclose(got[common].values, want[common].values)
        # Before the trajectory was issued there is no forecast to learn from.
        assert got[got.index < pd.Timestamp(issued)].isna().all()

    def test_retention_is_the_longest_days_history_sharing_the_forecast(self, tmp_db, monkeypatch):
        a = _exp("a", days=10, covariates=[_stored_cov()])
        b = _exp("b", days=30, covariates=[_stored_cov()])
        c = _exp("c", days=90, covariates=[_stored_cov(entity="sensor.other")])
        d = _exp("d", days=365, covariates=[_stored_cov(history_source="sensor")])
        app = _make_app(tmp_db, [a, b, c, d], _StubHA({}))
        calls = []
        monkeypatch.setattr(app.history_db, "maintain_stored_forecasts",
                            lambda source, keep: calls.append((source, keep)) or {})
        _run(app._maintain_stored_forecasts(a))
        assert calls == [(SRC, 31)]
        calls.clear()
        _run(app._maintain_stored_forecasts(d))  # no stored-forecast covariate
        assert calls == []

    def test_comparison_capture_logs_only_the_sources_own_span(self, tmp_db):
        # The source covers two hours; the experiment's horizon is six. Hours
        # three to six used to be logged as the last value held forward.
        now = pd.Timestamp(datetime.utcnow()).floor(FREQ)
        ext = ExternalForecastCfg(entity_id="sensor.fc", mode="attribute",
                                  attribute="forecast", value_key="power")
        exp = _exp(external_forecasts=[ext], mode="production")
        ha = _StubHA({}, {("sensor.fc", "forecast"): _entries(str(now), 5, lambda i: 1.0 + i)})
        app = _make_app(tmp_db, [exp], ha)
        ds_future = pd.date_range(now, periods=12, freq=FREQ, tz="UTC")
        _run(app._capture_external_forecast(exp, now.to_pydatetime(), ds_future))
        cur = app.history_db.conn.cursor()
        cur.execute("SELECT target_dt, value FROM external_forecast_log ORDER BY target_dt")
        rows = cur.fetchall()
        assert [r[1] for r in rows] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert rows[-1][0] == (now + pd.Timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")

    def test_within_points_span(self):
        from ml_forecast_lab.main import _within_points_span
        pts = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-10-01 10:00", "2026-10-01 11:00"]))
        series = align_future_points(
            pts, pd.date_range("2026-10-01 09:00", periods=6, freq=FREQ, tz="UTC"))
        kept = _within_points_span(series, pts)
        assert list(kept.values) == [1.0, 1.5, 2.0]
        assert _within_points_span(series, None).empty


# ---------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------

class TestReplay:
    def test_history_read_round_trips_and_writes_are_suppressed(self, db):
        from ml_forecast_lab.replay import (
            RecordingHistoryDB, ReplayHistoryDB, _CallLog, _CallQueue,
        )
        db.store_covariate_forecast(SRC, datetime(2026, 10, 1, 5),
                                    _targets("2026-10-01 06:00", 4), [1.0, 2.0, 3.0, 4.0])
        log = _CallLog()
        rec = RecordingHistoryDB(db, log)
        start, end = datetime(2026, 10, 1, 6), datetime(2026, 10, 1, 7, 30)
        live = rec.get_stored_forecast_history(SRC, start, end, FREQ)
        assert list(live["y"]) == [1.0, 2.0, 3.0, 4.0]
        assert rec.store_covariate_forecast(SRC, datetime(2026, 10, 1, 6), [], []) == 0
        replayed = ReplayHistoryDB(_CallQueue(log.calls)).get_stored_forecast_history(
            SRC, start, end, FREQ)
        pd.testing.assert_frame_equal(replayed.reset_index(drop=True),
                                      live.reset_index(drop=True))
        assert ReplayHistoryDB(_CallQueue([])).store_covariate_forecast(SRC, None, [], []) == 0
