"""
Target construction for a daily-resetting intermittent-demand counter.

Written for the 2026-11 intermittent-demand investigation, which had to
rule target construction out as the source of a 20 % level shortfall. These
tests pin what was verified there: converting the counter to per-interval
demand and summing onto the grid preserves the counter's totals

- through resets at local midnight in both UTC offsets of a zone with
  daylight saving (the reset is at 23:00 UTC in summer time, 00:00 UTC
  otherwise), including the 25-hour day of the change;
- through outages — a missed peak, a missed reset, repeated short ones —
  which move demand into the interval where readings resume, not lose it;
- at a two-minute update cadence where a fifth of demand arrives in single
  increments many times the typical one, which a cap derived from typical
  rates must not clip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.ha_interface import normalise_history
from ml_forecast_lab.preprocessing import (
    clip_outliers, cumulative_to_interval, resample_to_grid,
)
from tests.synthetic import intermittent_demand as gen

INTERVAL = 30


def _grid(rows):
    df = normalise_history(rows)
    df["ds"] = df["ds"].dt.tz_localize(None)
    s = df.set_index("ds").sort_index()["value"]
    iv = cumulative_to_interval(s, INTERVAL, reset_daily=True)
    return resample_to_grid(
        iv, f"{INTERVAL}min", method="sum", gap_handling="interpolate",
    )


def _truth(start="2026-10-12", days=28, seed=0):
    y, _ = gen.slot_truth(start, days, seed=seed)
    return y


def _naive(y):
    out = y.copy()
    out.index = out.index.tz_localize(None)
    return out


def _local_day_totals(series_utc_naive):
    loc = series_utc_naive.index.tz_localize("UTC").tz_convert(gen.TZ)
    return series_utc_naive.groupby(loc.date).sum()


def T(s):
    return pd.Timestamp(s, tz="UTC")


class TestResets:
    def test_totals_survive_resets_in_both_utc_offsets(self):
        y = _truth()                       # spans the 2026-10-25 change
        g, truth = _grid(gen.recorder_rows(y)), _naive(y)
        common = g.index.intersection(truth.index)
        change = pd.Timestamp("2026-10-25 01:00")
        # 1.5 %, not 1 %: this generator places readings at arbitrary
        # offsets, so a few events land within seconds of a background
        # tick. The default increment cap scales with the time since the
        # previous reading and trims those (about 1 % of demand here). At a
        # regular polling cadence nothing is trimmed — see
        # TestIncrementCapAtRealisticCadence.
        for part in (common[common < change], common[common >= change]):
            assert g.loc[part].sum() == pytest.approx(truth.loc[part].sum(), rel=0.015)

    def test_each_local_day_matches_the_counter(self):
        y = _truth()
        g, truth = _grid(gen.recorder_rows(y)), _naive(y)
        common = g.index.intersection(truth.index)
        got, want = _local_day_totals(g.loc[common]), _local_day_totals(truth.loc[common])
        # The first and last local days are partial on a UTC-cut window.
        got, want = got.iloc[1:-1], want.iloc[1:-1]
        assert len(want) >= 25
        rel = ((got - want) / want).abs()
        assert rel.median() < 0.005
        assert rel.max() < 0.10

    def test_reset_is_not_scored_as_demand(self):
        y = _truth(days=6)
        g = _grid(gen.recorder_rows(y))
        # No interval can exceed the largest true slot by more than the
        # handful of increments that straddle a slot boundary.
        assert g.max() <= y.max() + 1.0
        assert (g >= 0).all()


class TestOutages:
    CASES = {
        "peak": [(T("2026-11-03 17:10"), T("2026-11-03 20:20"))],
        "across_reset": [(T("2026-11-04 21:40"), T("2026-11-05 07:40"))],
        "repeated": [
            (T(f"2026-11-0{d} 18:05"), T(f"2026-11-0{d} 19:25")) for d in range(1, 6)
        ],
    }

    @pytest.mark.parametrize("case", sorted(CASES))
    def test_an_outage_moves_demand_but_does_not_lose_it(self, case):
        y = _truth()
        g = _grid(gen.recorder_rows(y, outages=self.CASES[case]))
        truth = _naive(y)
        common = g.index.intersection(truth.index)
        assert g.loc[common].sum() == pytest.approx(truth.loc[common].sum(), rel=0.015)

    def test_demand_missed_in_an_outage_lands_where_readings_resume(self):
        y = _truth()
        start, end = self.CASES["peak"][0]
        g = _grid(gen.recorder_rows(y, outages=[(start, end)]))
        truth = _naive(y)
        inside = (truth.index >= start.tz_localize(None).floor("30min")) & (
            truth.index <= end.tz_localize(None).floor("30min")
        )
        assert g.reindex(truth.index)[inside].sum() == pytest.approx(
            truth[inside].sum(), rel=0.05,
        )


def _two_minute_counter(days=10, seed=3):
    """Readings every ~2 minutes while the counter moves: background 0.1
    ticks, and events of a few minutes whose steps reach several units."""
    rng = np.random.default_rng(seed)
    y, _ = gen.slot_truth("2026-11-01", days, seed=seed)
    rows, total, day = [], 0.0, None
    for t, v in y.items():
        local_day = t.tz_convert(gen.TZ).date()
        if day is not None and local_day != day:
            total = 0.0
            rows.append((t + pd.Timedelta(seconds=1), 0.0))
        day = local_day
        if v <= 0:
            continue
        if v < 1.0:
            steps = [gen.STEP] * int(round(v / gen.STEP))
            offs = np.sort(rng.choice(np.arange(1, 14), size=len(steps), replace=False)) * 120
        else:
            k = int(rng.integers(1, 4))            # 1-3 large steps
            cuts = np.sort(rng.uniform(0, v, size=k - 1))
            steps = list(np.diff(np.concatenate([[0.0], cuts, [v]])))
            first = int(rng.integers(1, 10))
            offs = (first + np.arange(k)) * 120
        for off, inc in zip(offs, steps):
            total = round(total + float(inc), 2)
            rows.append((t + pd.Timedelta(seconds=float(off)), total))
    return y, [
        {"last_changed": ts.isoformat(), "state": f"{val:.2f}"} for ts, val in rows
    ]


class TestIncrementCapAtRealisticCadence:
    def test_large_single_increments_are_not_clipped(self):
        y, rows = _two_minute_counter()
        inc = pd.Series([float(r["state"]) for r in rows]).diff()
        pos = inc[inc > 0]
        assert pos[pos >= 3].sum() / pos.sum() > 0.15, "fixture must carry large steps"
        g, truth = _grid(rows), _naive(y)
        common = g.index.intersection(truth.index)
        assert g.loc[common].sum() == pytest.approx(truth.loc[common].sum(), rel=0.01)

    def test_outlier_clip_costs_under_one_percent(self):
        y, rows = _two_minute_counter()
        g = _grid(rows)
        clipped = clip_outliers(g, quantile=0.999, positive_only=True)
        assert clipped.sum() == pytest.approx(g.sum(), rel=0.01)
