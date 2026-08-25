"""Incremental covariate-history caching (v2.39.4).

The target series is cached incrementally in main._fetch_and_preprocess;
covariates were re-fetched for the full days_history window every cycle.
CovariateResolver now caches raw observations in SQLite and fetches only
the delta when a history_db is supplied. These tests lock in:

  - delta-only fetch on the second cycle (the speed win),
  - resample equivalence vs a full-window fetch (correctness),
  - per-(entity, attribute) cache keys,
  - graceful degradation when the cache misbehaves,
  - unchanged full-fetch behaviour when no history_db is injected.

v2.50.0 adds:

  - the cache outliving the HA recorder's own retention window,
  - pruning at the largest max_age across every experiment sharing a table,
  - a one-shot backfill when days_history is widened past the cached span.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.covariates import CovariateResolver
from ml_forecast_lab.db import HistoryDB


def _run(coro):
    return asyncio.run(coro)


class _RecordingIface:
    """Mock HAInterface whose get_history serves synthetic rows from a
    fixed full series, sliced to the requested [start, end] window, and
    records every call's window so tests can assert delta behaviour.

    ``retention_days`` additionally clips the served window the way a real
    HA recorder does, so a test can prove the SQLite cache outlives it."""

    def __init__(self, full_rows, retention_days=None):
        self.full_rows = full_rows
        self.retention_days = retention_days
        self.calls = []  # list of (start, end, include_attributes)

    async def get_history(self, entity_id, start, end, include_attributes=False):
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        if s.tzinfo is None:
            s = s.tz_localize("UTC")
        if e.tzinfo is None:
            e = e.tz_localize("UTC")
        if self.retention_days is not None:
            s = max(s, e - pd.Timedelta(days=self.retention_days))
        self.calls.append((s, e, include_attributes))
        out = []
        for r in self.full_rows:
            ts = pd.Timestamp(r["last_changed"])
            if s <= ts <= e:
                out.append(r)
        return out

    def window(self, i):
        return self.calls[i][1] - self.calls[i][0]


class _RecordingDB:
    """Wraps a real HistoryDB and records every cleanup boundary."""

    def __init__(self, inner):
        self.inner = inner
        self.cleanups = []  # list of (table, oldest_datetime)

    def safe_table_name(self, entity_id):
        return self.inner.safe_table_name(entity_id)

    def get_history(self, table):
        return self.inner.get_history(table)

    def store_history(self, table, df):
        return self.inner.store_history(table, df)

    def cleanup(self, table, oldest):
        self.cleanups.append((table, oldest))
        return self.inner.cleanup(table, oldest)


def _numeric_rows(start, end, cadence_min=15):
    ts = pd.date_range(start, end, freq=f"{cadence_min}min", tz="UTC")
    rng = np.random.default_rng(0)
    return [
        {"last_changed": t.isoformat(), "state": f"{rng.random() * 100:.3f}"}
        for t in ts
    ]


@pytest.fixture
def db(tmp_db):
    return HistoryDB(tmp_db)


def test_second_cycle_fetches_only_delta(db):
    """The win: with a history_db, cycle 2 fetches from the last cached
    observation, not the full days_history window."""
    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    rows = _numeric_rows(start, now + pd.Timedelta(hours=1))
    iface = _RecordingIface(rows)
    resolver = CovariateResolver(iface, history_db=db)
    cov = {"entity_id": "sensor.x", "name": "x"}

    # Cycle 1: cold cache → full-window fetch.
    s1 = _run(resolver.fetch_history(cov, start, now, "30min"))
    assert not s1.empty
    first_window = iface.calls[0][1] - iface.calls[0][0]
    assert first_window >= pd.Timedelta(days=13)  # ~full days_history

    # Cycle 2, 30 minutes later: should fetch only the recent delta.
    now2 = now + pd.Timedelta(minutes=30)
    s2 = _run(resolver.fetch_history(cov, start, now2, "30min"))
    assert not s2.empty
    second_window = iface.calls[1][1] - iface.calls[1][0]
    assert second_window <= pd.Timedelta(hours=2), (
        f"cycle 2 should fetch a small delta, not the full window; "
        f"got {second_window}"
    )


def test_cached_result_matches_full_fetch(db):
    """Correctness: the resampled series built from cache+delta must be
    identical to one built from a single full-window fetch."""
    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    start = datetime(2026, 5, 8, tzinfo=timezone.utc)
    rows = _numeric_rows(start, now + pd.Timedelta(hours=1))

    # Cached resolver: two cycles (cold, then delta).
    cached_resolver = CovariateResolver(_RecordingIface(rows), history_db=db)
    cov = {"entity_id": "sensor.x", "name": "x"}
    mid = now - pd.Timedelta(hours=6)
    _run(cached_resolver.fetch_history(cov, start, mid, "30min"))
    cached_series = _run(cached_resolver.fetch_history(cov, start, now, "30min"))

    # Reference resolver: NO db → single full-window fetch at `now`.
    ref_resolver = CovariateResolver(_RecordingIface(rows), history_db=None)
    ref_series = _run(ref_resolver.fetch_history(cov, start, now, "30min"))

    merged = pd.concat([cached_series, ref_series], axis=1).dropna()
    assert len(merged) > 0
    np.testing.assert_allclose(
        merged.iloc[:, 0].values, merged.iloc[:, 1].values, rtol=1e-9,
        err_msg="cache+delta resample diverged from full-window resample",
    )


def test_distinct_attribute_keys_use_separate_caches(db):
    """Two covariates on the same weather entity reading different
    attributes must not share a cache table (else one would shadow the
    other's values)."""
    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    start = datetime(2026, 5, 14, tzinfo=timezone.utc)
    ts = pd.date_range(start, now, freq="15min", tz="UTC")
    rows = [
        {"last_changed": t.isoformat(), "state": "partlycloudy",
         "attributes": {"temperature": 10.0 + i, "cloud_coverage": 90.0 - i}}
        for i, t in enumerate(ts)
    ]
    resolver = CovariateResolver(_RecordingIface(rows), history_db=db)

    temp = _run(resolver.fetch_history(
        {"entity_id": "weather.x", "name": "t", "future_value_key": "temperature"},
        start, now, "30min"))
    cloud = _run(resolver.fetch_history(
        {"entity_id": "weather.x", "name": "c", "future_value_key": "cloud_coverage"},
        start, now, "30min"))

    # Distinct tables, distinct value ranges (temp ~10-100, cloud ~0-90).
    t_table = resolver._cov_cache_table("weather.x", "temperature")
    c_table = resolver._cov_cache_table("weather.x", "cloud_coverage")
    assert t_table != c_table
    assert temp.mean() != pytest.approx(cloud.mean())


def test_cache_error_degrades_to_full_fetch():
    """A misbehaving history_db must not break the fetch — it falls back
    to the full-window path and still returns data."""
    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    start = datetime(2026, 5, 14, tzinfo=timezone.utc)
    rows = _numeric_rows(start, now)

    class _BrokenDB:
        def safe_table_name(self, e):
            return "cov_x"

        def get_history(self, t):
            raise RuntimeError("db down")

        def store_history(self, t, df):
            raise RuntimeError("db down")

        def cleanup(self, t, oldest):
            raise RuntimeError("db down")

    resolver = CovariateResolver(_RecordingIface(rows), history_db=_BrokenDB())
    s = _run(resolver.fetch_history({"entity_id": "sensor.x", "name": "x"}, start, now, "30min"))
    assert not s.empty, "cache failure must degrade to a working full fetch"


def test_no_history_db_keeps_full_fetch_each_call():
    """Back-compat: without a history_db every call fetches the full
    window (the pre-v2.39.4 behaviour relied on by existing callers)."""
    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    start = datetime(2026, 5, 8, tzinfo=timezone.utc)
    rows = _numeric_rows(start, now)
    iface = _RecordingIface(rows)
    resolver = CovariateResolver(iface, history_db=None)
    cov = {"entity_id": "sensor.x", "name": "x"}

    _run(resolver.fetch_history(cov, start, now, "30min"))
    _run(resolver.fetch_history(cov, start, now, "30min"))
    # Both calls span the full window — no delta narrowing without a db.
    for s, e, _ in iface.calls:
        assert (e - s) >= pd.Timedelta(days=6)


# ---------------------------------------------------------------------
# v2.50.0: retention, and outliving the recorder
# ---------------------------------------------------------------------


def test_cache_outlives_recorder_retention(db):
    """The point of caching: history accumulates past the window HA is
    willing to serve. With a 1-day recorder, three cycles two days apart
    must still yield a span no single HA fetch could ever return."""
    start_wall = datetime(2026, 5, 1, tzinfo=timezone.utc)
    rows = _numeric_rows(start_wall, datetime(2026, 5, 20, tzinfo=timezone.utc))
    iface = _RecordingIface(rows, retention_days=1)
    resolver = CovariateResolver(iface, history_db=db)
    cov = {"entity_id": "sensor.x", "name": "x"}
    table = resolver._cov_cache_table("sensor.x", None)

    spans, row_counts = [], []
    for offset_days in (0, 2, 4):
        now = datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc) + timedelta(
            days=offset_days,
        )
        series = _run(resolver.fetch_history(
            cov, now - timedelta(days=14), now, "30min",
        ))
        spans.append(series.index[-1] - series.index[0])
        row_counts.append(len(db.get_history(table)))

    assert spans[-1] >= pd.Timedelta(days=4), (
        f"cached span should outgrow the 1-day recorder window; "
        f"got {spans[-1]}"
    )
    assert row_counts == sorted(row_counts) and row_counts[-1] > row_counts[0], (
        f"cached rows should accumulate across cycles; got {row_counts}"
    )
    for i in range(len(iface.calls)):
        assert iface.window(i) <= pd.Timedelta(days=1, minutes=1), (
            "precondition: HA never served more than its retention window"
        )


def test_prune_uses_the_retention_provider(db):
    """A shared table must be pruned at the largest max_age among the
    experiments referencing it, not at a per-resolver default."""
    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    start = now - timedelta(days=14)
    rec = _RecordingDB(db)
    resolver = CovariateResolver(
        _RecordingIface(_numeric_rows(start, now)),
        history_db=rec,
        retention_provider=lambda table: 400,
    )

    _run(resolver.fetch_history({"entity_id": "sensor.x", "name": "x"},
                                start, now, "30min"))

    assert len(rec.cleanups) == 1
    table, oldest = rec.cleanups[0]
    age_days = (datetime.now(timezone.utc) - oldest).days
    assert 399 <= age_days <= 401, (
        f"expected a ~400-day prune boundary, got {age_days} days"
    )


def test_prune_boundary_is_per_table(db):
    """Two entities with different retentions must not share a boundary."""
    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    start = now - timedelta(days=14)
    rec = _RecordingDB(db)
    per_table = {"cov_sensor_short": 30}
    resolver = CovariateResolver(
        _RecordingIface(_numeric_rows(start, now)),
        history_db=rec,
        retention_provider=lambda t: per_table.get(t, 365),
    )

    _run(resolver.fetch_history({"entity_id": "sensor.short", "name": "s"},
                                start, now, "30min"))
    _run(resolver.fetch_history({"entity_id": "sensor.long", "name": "l"},
                                start, now, "30min"))

    ages = {
        table: (datetime.now(timezone.utc) - oldest).days
        for table, oldest in rec.cleanups
    }
    assert 29 <= ages["cov_sensor_short"] <= 31
    assert 364 <= ages["cov_sensor_long"] <= 366


def test_no_retention_provider_falls_back_to_the_default(db):
    """Back-compat: the pre-v2.50.0 constructor still prunes at
    cache_max_age_days."""
    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    start = now - timedelta(days=14)
    rec = _RecordingDB(db)
    resolver = CovariateResolver(
        _RecordingIface(_numeric_rows(start, now)), history_db=rec,
    )

    _run(resolver.fetch_history({"entity_id": "sensor.x", "name": "x"},
                                start, now, "30min"))

    age_days = (datetime.now(timezone.utc) - rec.cleanups[0][1]).days
    assert 364 <= age_days <= 366


def test_retention_provider_error_still_prunes(db):
    """A provider that raises must not skip the prune or break the fetch."""
    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    start = now - timedelta(days=14)
    rec = _RecordingDB(db)

    def _boom(table):
        raise RuntimeError("config exploded")

    resolver = CovariateResolver(
        _RecordingIface(_numeric_rows(start, now)),
        history_db=rec, retention_provider=_boom,
    )

    series = _run(resolver.fetch_history(
        {"entity_id": "sensor.x", "name": "x"}, start, now, "30min",
    ))

    assert not series.empty
    assert len(rec.cleanups) == 1
    age_days = (datetime.now(timezone.utc) - rec.cleanups[0][1]).days
    assert 364 <= age_days <= 366


@pytest.mark.parametrize("bad_days", [0, -10])
def test_non_positive_retention_is_clamped(db, bad_days):
    """A 0 or negative retention would put the cut-off in the future and
    delete the entire table on the next prune."""
    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    start = now - timedelta(days=14)
    rec = _RecordingDB(db)
    resolver = CovariateResolver(
        _RecordingIface(_numeric_rows(start, now)),
        history_db=rec, retention_provider=lambda t: bad_days,
    )

    _run(resolver.fetch_history({"entity_id": "sensor.x", "name": "x"},
                                start, now, "30min"))

    assert rec.cleanups[0][1] < datetime.now(timezone.utc), (
        "the prune boundary must never be in the future"
    )


# ---------------------------------------------------------------------
# v2.50.0: widening days_history
# ---------------------------------------------------------------------


def test_widened_window_is_backfilled_once(db):
    """A delta fetch anchored on the newest cached row only ever extends
    forwards, so raising days_history would otherwise cap the covariate at
    the old window for good — even with the rows still in the recorder."""
    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    rows = _numeric_rows(now - timedelta(days=30), now)
    iface = _RecordingIface(rows)
    resolver = CovariateResolver(iface, history_db=db)
    cov = {"entity_id": "sensor.x", "name": "x"}

    # Cycle 1: a 2-day window.
    narrow = _run(resolver.fetch_history(
        cov, now - timedelta(days=2), now, "30min",
    ))
    assert narrow.index[-1] - narrow.index[0] <= pd.Timedelta(days=2, hours=1)

    # Cycle 2: the user widens days_history to 10.
    wide = _run(resolver.fetch_history(
        cov, now - timedelta(days=10), now, "30min",
    ))

    assert iface.window(1) >= pd.Timedelta(days=9), (
        "the widened cycle must re-ask HA for the whole window"
    )
    assert wide.index[-1] - wide.index[0] >= pd.Timedelta(days=9), (
        f"the widened window should now be covered; "
        f"got {wide.index[-1] - wide.index[0]}"
    )

    # Cycle 3: back to a delta — the backfill is one-shot, not per-cycle.
    _run(resolver.fetch_history(
        cov, now - timedelta(days=10), now + timedelta(minutes=30), "30min",
    ))
    assert iface.window(2) <= pd.Timedelta(hours=2), (
        f"the backfill must not repeat every cycle; got {iface.window(2)}"
    )


def test_young_entity_does_not_refetch_every_cycle(db):
    """A covariate whose history genuinely starts inside the window must
    not trigger a full fetch forever — the cache can never reach `start`
    for it, so only the one-shot guard stops an endless full fetch."""
    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    rows = _numeric_rows(now - timedelta(days=2), now)  # only 2 days exist
    iface = _RecordingIface(rows)
    resolver = CovariateResolver(iface, history_db=db)
    cov = {"entity_id": "sensor.young", "name": "y"}

    for i in range(3):
        _run(resolver.fetch_history(
            cov,
            now - timedelta(days=14),
            now + timedelta(minutes=30 * i),
            "30min",
        ))

    assert iface.window(0) >= pd.Timedelta(days=13)
    for i in (1, 2):
        assert iface.window(i) <= pd.Timedelta(hours=2), (
            f"cycle {i + 1} should be a delta, not a full window; "
            f"got {iface.window(i)}"
        )


def test_unwidened_window_never_backfills(db):
    """A steady-state experiment must see no extra fetches at all."""
    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    start = now - timedelta(days=14)
    iface = _RecordingIface(_numeric_rows(start, now))
    resolver = CovariateResolver(iface, history_db=db)
    cov = {"entity_id": "sensor.x", "name": "x"}

    _run(resolver.fetch_history(cov, start, now, "30min"))
    _run(resolver.fetch_history(cov, start, now + timedelta(minutes=30), "30min"))
    _run(resolver.fetch_history(cov, start, now + timedelta(minutes=60), "30min"))

    assert iface.window(0) >= pd.Timedelta(days=13)
    assert iface.window(1) <= pd.Timedelta(hours=2)
    assert iface.window(2) <= pd.Timedelta(hours=2)


def test_prune_of_a_never_created_table_is_quiet(db, caplog):
    """The prune runs once per covariate per cycle. On an entity that has
    never cached anything it used to log an ERROR with a traceback."""
    import logging

    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    resolver = CovariateResolver(_RecordingIface([]), history_db=db)

    with caplog.at_level(logging.ERROR, logger="ml_forecast_lab.db"):
        _run(resolver.fetch_history(
            {"entity_id": "sensor.nothing", "name": "n"},
            now - timedelta(days=14), now, "30min",
        ))

    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
