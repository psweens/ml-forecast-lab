"""v2.55.0: cumulative labels keep the use a counter recorded.

Two conversions threw real use away before the model saw it.

1. **The default increment cap was a size.** With ``max_increment`` unset,
   ``cumulative_to_interval`` capped every reading's rise at the 95th
   percentile of rises, so by construction the largest 5% were cut. Where a
   sensor updates slowly or irregularly, a genuine large rise builds up
   between readings and was cut back to a typical one. Replaying one
   install's history, daily label totals were 1.2 kWh/day low on a
   house-load counter and 10.6 points/day low on a hot-water demand
   counter. The default cap is now a rate — twice the counter's 99th
   percentile rate of rise, times the time since the previous reading — so
   a rise that took longer may be larger, and a glitch is still caught. The
   same conversion runs on every ``load_subtract`` sensor, which cannot set
   its own cap.

   Restart detection changed with it. Every drop used to be a restart, so a
   dip (12.3 → 12.2) scored the whole reading as one interval's use, which
   only the old tight cap hid. A restart is now a drop below 90% of the
   previous reading, as in the analytics and HA's ``total_increasing``, and
   a dip scores nothing until the counter passes the level already scored.
   That level, not the raw reading, is what a dip is measured from, so a
   capped spike that comes back down no longer zeroes the labels that
   follow, and a dip deeper than an interval's plausible use is a
   re-baseline. A lifetime counter's drop that its next rise undoes (a
   brief 0) is a glitch and is ignored, and the default cap's rate estimate
   leaves glitch recoveries and spikes out, so they can't set it. With
   ``reset_daily``, a small rise across UTC midnight also counted as a
   restart, so in summer time a local-midnight counter's first hour of each
   day was scored twice; only a drop across midnight is a restart now.

2. **Subtraction clipped each row at zero.** When a subtracted sensor and
   the parent meter recorded the same energy in different rows, or the
   parent stalled and caught up later, the row where the subtracted energy
   landed went negative and the deficit was thrown away, so that energy was
   never subtracted. For cumulative sources the deficit is now carried into
   the following rows of the same local day.

Models trained on the old labels are retrained (``LABEL_RULE_VERSION``) for
the experiments these touch; other experiments keep their cached models.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.config import SubtractCfg
from ml_forecast_lab.db import HistoryDB
from ml_forecast_lab.preprocessing import (
    COUNTER_RESTART_RATIO,
    RATE_CAP_MIN_SAMPLES,
    _carry_deficit,
    _score_from_accepted_level,
    apply_load_subtract,
    cumulative_to_interval,
)

INTERVAL = 30


def _counter(values, start="2026-10-20 08:00", step_min=2.0, times=None):
    if times is None:
        idx = pd.date_range(start, periods=len(values), freq=f"{int(step_min * 60)}s")
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(times))
    return pd.Series(np.asarray(values, dtype=float), index=idx)


# ---------------------------------------------------------------------
# Restarts and dips
# ---------------------------------------------------------------------

class TestRestartsAndDips:
    def test_a_dip_and_its_recovery_net_to_the_counters_rise(self):
        s = _counter([10.0, 10.1, 10.2, 10.1, 10.2, 10.3])
        inc = cumulative_to_interval(s, INTERVAL, max_increment=1e9)
        np.testing.assert_allclose(inc.to_numpy(), [0, 0.1, 0.1, 0, 0, 0.1], atol=1e-12)

    def test_a_dip_is_not_scored_as_a_restart(self):
        """Pre-2.55.0 the dip row scored 10.1 — the whole reading."""
        s = _counter([10.0, 10.2, 10.1, 10.3])
        inc = cumulative_to_interval(s, INTERVAL, max_increment=1e9)
        assert inc.sum() == pytest.approx(0.3)

    def test_a_restart_scores_the_new_reading(self):
        s = _counter([24.6, 24.8, 0.2, 0.4], step_min=10)
        inc = cumulative_to_interval(s, INTERVAL, reset_daily=True, max_increment=1e9)
        np.testing.assert_allclose(inc.to_numpy(), [0, 0.2, 0.2, 0.2], atol=1e-12)

    def test_a_dip_after_a_restart_is_measured_from_the_new_high(self):
        s = _counter([24.8, 0.2, 0.5, 0.45, 0.6], step_min=10)
        inc = cumulative_to_interval(s, INTERVAL, max_increment=1e9)
        np.testing.assert_allclose(inc.to_numpy(), [0, 0.2, 0.3, 0, 0.1], atol=1e-12)

    def test_summer_time_hour_is_not_scored_twice(self):
        """A counter resetting at local midnight in BST restarts at 23:00
        UTC. Pre-2.55.0 the first reading after UTC midnight, an hour into
        the day, rose slightly, counted as a restart, and scored the hour's
        use a second time."""
        # 0.4 kW, with 8 kW from 17:00 to 20:00 local: the evening peak sets
        # the rate cap well above the night-time hour's use, so the cap
        # alone cannot hide a double count.
        idx = pd.date_range("2026-07-01 12:00", "2026-07-04 12:00", freq="2min")
        local = idx.tz_localize("UTC").tz_convert("Europe/London")
        hours = np.asarray((local - local.normalize()).total_seconds() / 3600)
        kw = np.where((hours >= 17) & (hours < 20), 8.0, 0.4)
        day = np.asarray(local.date)
        energy = pd.Series(kw * 2 / 60, index=idx).groupby(day).cumsum()
        s = pd.Series(np.floor(energy.to_numpy() * 100) / 100, index=idx)
        by_day = s.groupby(day)
        recorded = float((by_day.last() - by_day.first()).sum()
                         + by_day.first().iloc[1:].sum())
        inc = cumulative_to_interval(s, INTERVAL, reset_daily=True)
        # The old midnight rule with the rate cap scored 0.4 kWh/day extra.
        assert inc.sum() == pytest.approx(recorded, abs=0.02)
        utc_midnight = (inc.index.hour == 0) & (inc.index.minute == 0)
        assert (inc[utc_midnight] < 0.05).all()

    def test_a_small_drop_across_midnight_is_a_restart(self):
        """A daily counter that read 0.5 the day before restarts and reads
        0.48 by its first reading — above 90% of 0.5, but a restart."""
        times = ["2026-12-01 23:40", "2026-12-01 23:58", "2026-12-02 00:30",
                 "2026-12-02 00:40"]
        s = _counter([0.4, 0.5, 0.48, 0.6], times=times)
        inc = cumulative_to_interval(s, INTERVAL, reset_daily=True, max_increment=1e9)
        np.testing.assert_allclose(inc.to_numpy(), [0, 0.1, 0.48, 0.12], atol=1e-12)

    def test_restart_threshold_matches_the_analytics(self):
        assert COUNTER_RESTART_RATIO == HistoryDB._COUNTER_RESTART_RATIO

    def test_a_monotonic_counter_is_unchanged(self):
        rng = np.random.default_rng(3)
        s = _counter(np.cumsum(rng.uniform(0, 0.3, 200)) + 5.0)
        inc = cumulative_to_interval(s, INTERVAL, max_increment=1e9)
        np.testing.assert_allclose(inc.to_numpy(), s.diff().fillna(0.0).to_numpy())


# ---------------------------------------------------------------------
# The default cap
# ---------------------------------------------------------------------

def _irregular_house_meter(seed=7, hours=48):
    """A 0.1 kWh-resolution counter polled every 1–3 minutes, with an
    occasional 8–20 minute silence (never long enough to be a multi-interval
    gap), recording 1 kW all day and 12 kW from 18:00 to midnight."""
    rng = np.random.default_rng(seed)
    t = pd.Timestamp("2026-10-18 00:00")
    end = t + pd.Timedelta(hours=hours)
    energy = 0.0
    times, values = [t], [0.0]
    while t < end:
        minutes = rng.uniform(8, 20) if rng.random() < 0.08 else rng.uniform(1, 3)
        kw = 12.0 if t.hour >= 18 else 1.0
        energy += kw * minutes / 60.0
        t = t + pd.Timedelta(minutes=minutes)
        times.append(t)
        values.append(np.floor(energy * 10) / 10)
    return _counter(values, times=times)


class TestRateBasedCap:
    def test_slow_updates_keep_their_whole_rise(self):
        s = _irregular_house_meter()
        inc = cumulative_to_interval(s, INTERVAL)
        assert inc.sum() == pytest.approx(s.iloc[-1] - s.iloc[0])

    def test_the_old_size_cap_loses_use_on_the_same_counter(self):
        """Why the default changed: the 95th-percentile size cap, applied to
        the same counter, loses a material share of its use."""
        s = _irregular_house_meter()
        rises = s.diff()
        p95 = float(rises[rises > 0].quantile(0.95))
        old = cumulative_to_interval(s, INTERVAL, max_increment=p95)
        assert old.sum() < 0.9 * (s.iloc[-1] - s.iloc[0])

    def test_a_glitch_recovery_scores_only_the_use_since(self):
        s = _irregular_house_meter()
        i = len(s) // 2
        t = s.index[i]
        recovery_at = t + pd.Timedelta(seconds=40)
        assert recovery_at < s.index[i + 1]
        glitch = pd.concat([
            s.iloc[:i],
            pd.Series([0.0, s.iloc[i - 1]], index=[t, recovery_at]),
            s.iloc[i + 1:],
        ])
        inc = cumulative_to_interval(glitch, INTERVAL)
        # The recovery would re-add the whole counter (tens of kWh).
        assert s.iloc[i - 1] > 10.0
        assert inc.loc[recovery_at] == pytest.approx(0.0)
        assert inc.sum() == pytest.approx(s.iloc[-1] - s.iloc[0])

    def test_a_daily_counters_glitch_is_still_capped(self):
        """Daily counters keep every drop below 90% as a restart, so the
        recovery is a rise from 0 — kept out of the rate estimate, so the
        cap still holds it."""
        s = _irregular_house_meter()
        i = len(s) // 2
        t = s.index[i]
        recovery_at = t + pd.Timedelta(seconds=40)
        glitch = pd.concat([
            s.iloc[:i],
            pd.Series([0.0, s.iloc[i - 1]], index=[t, recovery_at]),
            s.iloc[i + 1:],
        ])
        inc = cumulative_to_interval(glitch, INTERVAL, reset_daily=True)
        assert inc.loc[recovery_at] < 1.0

    def test_a_rise_after_a_silence_is_kept_at_the_usual_rate(self):
        """A change-only counter that goes quiet for 25 minutes, still inside
        one interval's tolerance, then records the 6 points used meanwhile."""
        times = list(pd.date_range("2026-10-20 06:00", periods=30, freq="2min"))
        times.append(times[-1] + pd.Timedelta(minutes=25))
        times += list(pd.date_range(times[-1] + pd.Timedelta(minutes=2), periods=10, freq="2min"))
        vals = np.r_[np.arange(30) * 0.5, 14.5 + 6.0, 20.5 + 0.5 * np.arange(1, 11)]
        s = _counter(vals, times=times)
        inc = cumulative_to_interval(s, INTERVAL)
        assert inc.sum() == pytest.approx(s.iloc[-1] - s.iloc[0])
        old = cumulative_to_interval(s, INTERVAL, max_increment=0.5)
        assert old.sum() == pytest.approx(inc.sum() - 5.5)

    def test_an_explicit_max_increment_still_caps_each_reading(self):
        s = _counter([0.0, 0.1, 0.9, 1.0])
        inc = cumulative_to_interval(s, INTERVAL, max_increment=0.5)
        assert inc.iloc[2] == pytest.approx(0.5)

    def test_too_few_rates_fall_back_to_the_size_cap(self):
        s = _counter([0.0, 0.1, 0.2, 5.0, 5.1])
        assert len(s) - 1 < RATE_CAP_MIN_SAMPLES
        inc = cumulative_to_interval(s, INTERVAL)
        rises = s.diff()
        assert inc.iloc[3] == pytest.approx(float(rises[rises > 0].quantile(0.95)))

    def test_a_multi_interval_gap_is_still_exempt(self):
        times = ["2026-10-20 06:00", "2026-10-20 06:02", "2026-10-20 06:04",
                 "2026-10-20 09:00"]
        s = _counter([0.0, 0.1, 0.2, 30.0], times=times)
        inc = cumulative_to_interval(s, INTERVAL)
        assert inc.iloc[3] == pytest.approx(29.8)


# ---------------------------------------------------------------------
# Glitches and re-baselines
# ---------------------------------------------------------------------

def _lifetime_meter(days=28, step_min=30, seed=0):
    """A 12,345 kWh lifetime counter read every ``step_min`` minutes,
    using 0.2–1.0 kWh per half hour."""
    idx = pd.date_range("2026-10-01", periods=int(days * 24 * 60 / step_min),
                        freq=f"{step_min}min")
    use = np.random.default_rng(seed).uniform(0.2, 1.0, len(idx)) * step_min / 30
    return pd.Series(12345.0 + np.cumsum(use), index=idx)


def _with_zero_glitches(s, every):
    """Each ``every`` readings, the counter reads 0 and then its old value
    again before its next reading."""
    step = s.index[1] - s.index[0]
    extra = {}
    for i in range(every // 2, len(s) - 1, every):
        extra[s.index[i] + step / 3] = 0.0
        extra[s.index[i] + 2 * step / 3] = s.iloc[i]
    return pd.concat([s, pd.Series(extra)]).sort_index()


class TestGlitchesAndRebaselines:
    def test_a_drop_to_zero_and_back_is_ignored_on_a_short_history(self):
        """51 readings: the recovery's own rate was the 99th percentile, so
        the cap let the whole counter through as one interval's use."""
        s = _lifetime_meter(days=51 / 48)
        inc = cumulative_to_interval(_with_zero_glitches(s, len(s)), INTERVAL)
        assert inc.sum() == pytest.approx(s.iloc[-1] - s.iloc[0])
        assert inc.max() < 2.0

    def test_daily_glitches_do_not_lift_the_cap(self):
        s = _lifetime_meter(days=14, step_min=15)
        inc = cumulative_to_interval(_with_zero_glitches(s, 96), INTERVAL)
        assert inc.sum() == pytest.approx(s.iloc[-1] - s.iloc[0])
        assert inc.max() < 2.0

    def test_a_daily_reset_is_never_taken_for_a_glitch(self):
        """One reading after a daily reset can carry most of the day's use,
        e.g. a charger that reports its session when it ends."""
        day1 = pd.date_range("2026-10-20 00:00", periods=48, freq="30min")
        times = list(day1) + [pd.Timestamp("2026-10-21 00:01"),
                              pd.Timestamp("2026-10-21 06:00")]
        vals = list(0.4 * np.arange(1, 49)) + [0.0, 18.0]
        inc = cumulative_to_interval(_counter(vals, times=times), INTERVAL,
                                     reset_daily=True)
        assert inc.iloc[-1] == pytest.approx(18.0)

    @pytest.mark.parametrize("max_increment", [None, 2.0])
    def test_a_capped_spike_that_returns_keeps_later_labels(self, max_increment):
        """One corrupted reading (+1000 kWh, under the 10% restart drop):
        measured from the raw spike, every later reading scored 0 until
        the counter passed it — 20 of 28 days."""
        s = _lifetime_meter()
        g = s.copy()
        g.iloc[7 * 48] += 1000.0
        inc = cumulative_to_interval(g, INTERVAL, max_increment=max_increment)
        daily = inc.groupby(inc.index.date).sum()
        assert (daily > 5.0).all()
        assert inc.sum() == pytest.approx(s.iloc[-1] - s.iloc[0], abs=2.0)

    @pytest.mark.parametrize("max_increment", [None, 2.0])
    def test_a_small_step_down_rebaselines_the_counter(self, max_increment):
        """A meter re-baselined 5% lower (617 kWh) scored 0 until it climbed
        back past the old reading."""
        s = _lifetime_meter()
        g = s.copy()
        g.iloc[7 * 48:] -= 0.05 * s.iloc[7 * 48]
        inc = cumulative_to_interval(g, INTERVAL, max_increment=max_increment)
        daily = inc.groupby(inc.index.date).sum()
        assert (daily > 5.0).all()
        assert inc.sum() == pytest.approx(s.iloc[-1] - s.iloc[0], abs=2.0)

    def test_a_stale_reading_then_a_gap_scores_only_the_use(self):
        """A 5% stale reading, then nothing for 2 h: the return is a gap row,
        exempt from the cap, so a re-baseline at the stale value scored the
        dip's depth (617 kWh) as one bin's use."""
        s = _lifetime_meter()
        i = 7 * 48
        g = s.copy()
        g.iloc[i] = 0.95 * s.iloc[i]
        g = g.drop(g.index[i + 1:i + 5])
        inc = cumulative_to_interval(g, INTERVAL)
        assert inc.max() < 5.0
        assert inc.sum() == pytest.approx(s.iloc[-1] - s.iloc[0], abs=2.0)

    def test_a_stale_reading_then_a_gap_with_an_explicit_cap(self):
        times = ["2026-10-20 00:00", "2026-10-20 00:30", "2026-10-20 01:00",
                 "2026-10-20 03:00"]
        s = _counter([10000.0, 10000.5, 9500.0, 10001.0], times=times)
        inc = cumulative_to_interval(s, INTERVAL, max_increment=2.0)
        assert inc.sum() == pytest.approx(1.0)

    def test_without_caps_or_deep_dips_the_scan_is_the_running_high_rule(self):
        """The scan only runs when it can differ from the vectorised rule;
        where it can't, it must agree exactly."""
        rng = np.random.default_rng(3)
        v = 100.0 + np.cumsum(rng.uniform(0.0, 1.0, 500))
        v[rng.choice(500, 40, replace=False)] -= 0.3   # measurement dips
        s = pd.Series(v)
        high = s.cummax().shift(1)
        expected = (s - high).clip(lower=0.0).fillna(0.0).to_numpy()
        n = len(v)
        got = _score_from_accepted_level(
            v.tolist(), [False] * n, [False] * n, [np.inf] * n, np.inf,
        )
        assert np.asarray(got) == pytest.approx(expected)


# ---------------------------------------------------------------------
# Deficit carry in load_subtract
# ---------------------------------------------------------------------

def _days(idx):
    return np.asarray(idx.date)


_SUB = {"entity_id": "sensor.immersion", "on_missing": "zero",
        "max_fraction_of_load": 10.0}


class TestDeficitCarry:
    def test_a_deficit_comes_off_the_next_rows_of_the_day(self):
        idx = pd.date_range("2026-10-20 10:00", periods=5, freq="30min")
        out, carried, dropped = _carry_deficit(
            pd.Series([1.0, -0.6, 0.4, 0.5, 0.3], index=idx), _days(idx),
        )
        np.testing.assert_allclose(out.to_numpy(), [1.0, 0.0, 0.0, 0.3, 0.3])
        assert carried == pytest.approx(0.6)
        assert dropped == pytest.approx(0.0)

    def test_totals_are_preserved_up_to_what_is_left_at_day_end(self):
        rng = np.random.default_rng(1)
        idx = pd.date_range("2026-10-20 00:00", periods=144, freq="30min")
        vals = pd.Series(rng.normal(0.1, 0.5, len(idx)), index=idx)
        out, carried, dropped = _carry_deficit(vals, _days(idx))
        assert (out >= 0).all()
        assert out.sum() == pytest.approx(vals.sum() + dropped)
        assert carried + dropped == pytest.approx(float(-vals[vals < 0].sum()))
        for day in np.unique(_days(idx)):
            mask = _days(idx) == day
            assert out[mask].sum() >= vals[mask].sum() - 1e-9

    def test_a_deficit_does_not_cross_midnight(self):
        idx = pd.date_range("2026-10-20 23:00", periods=3, freq="30min")
        out, carried, dropped = _carry_deficit(
            pd.Series([0.2, -0.5, 0.4], index=idx), _days(idx),
        )
        np.testing.assert_allclose(out.to_numpy(), [0.2, 0.0, 0.4])
        assert dropped == pytest.approx(0.5)
        assert carried == pytest.approx(0.0)

    def test_unmeasured_rows_pass_through(self):
        idx = pd.date_range("2026-10-20 10:00", periods=4, freq="30min")
        out, carried, _ = _carry_deficit(
            pd.Series([-0.3, np.nan, 0.2, 0.5], index=idx), _days(idx),
        )
        assert np.isnan(out.iloc[1])
        np.testing.assert_allclose(out.dropna().to_numpy(), [0.0, 0.0, 0.4])
        assert carried == pytest.approx(0.3)

    def test_a_late_subtracted_sensor_still_comes_off_the_day(self):
        """The subtracted sensor reports 30 minutes after the parent meter:
        clipping lost 1.15 kWh of its energy; carrying keeps the day's net
        exact."""
        idx = pd.date_range("2026-10-20 00:00", periods=48, freq="30min")
        load = pd.Series(0.5, index=idx)
        load.iloc[20:24] += 1.65
        sub = pd.Series(0.0, index=idx)
        sub.iloc[21:25] = 1.65
        carried, audit = apply_load_subtract(
            load, [(_SUB, sub)], carry_deficit_days=_days(idx),
        )
        clipped, clip_audit = apply_load_subtract(load, [(_SUB, sub)])
        assert carried.sum() == pytest.approx(load.sum() - sub.sum())
        assert clipped.sum() == pytest.approx(carried.sum() + 1.15)
        assert audit["deficit_carried_kwh"] == pytest.approx(1.15)
        assert audit["deficit_dropped_kwh"] == pytest.approx(0.0)
        assert clip_audit["deficit_carried_kwh"] == 0.0
        assert (carried >= 0).all()

    def test_dropped_rows_neither_create_nor_absorb_a_deficit(self):
        idx = pd.date_range("2026-10-20 10:00", periods=4, freq="30min")
        load = pd.Series([0.2, 0.3, 0.6, 0.6], index=idx)
        sub = pd.Series([0.5, np.nan, 0.1, 0.1], index=idx)
        cfg = {**_SUB, "on_missing": "drop"}
        out, audit = apply_load_subtract(
            load, [(cfg, sub)], carry_deficit_days=_days(idx),
        )
        assert list(out.index) == [idx[0], idx[2], idx[3]]
        np.testing.assert_allclose(out.to_numpy(), [0.0, 0.2, 0.5])
        assert audit["deficit_carried_kwh"] == pytest.approx(0.3)

    def test_without_day_labels_rows_are_clipped_as_before(self):
        idx = pd.date_range("2026-10-20 10:00", periods=3, freq="30min")
        load = pd.Series([1.0, 0.2, 1.0], index=idx)
        sub = pd.Series([0.0, 0.5, 0.0], index=idx)
        out, audit = apply_load_subtract(load, [(_SUB, sub)])
        np.testing.assert_allclose(out.to_numpy(), [1.0, 0.0, 1.0])
        assert audit["n_clipped_rows"] == 1

    def test_the_fetch_path_carries_for_cumulative_targets_only(self):
        src = (Path(__file__).resolve().parents[2]
               / "ml_forecast_lab" / "main.py").read_text()
        start = src.index("async def _fetch_and_preprocess(")
        nxt = re.compile(r"\n    (?:async )?def ").search(src, start + 1)
        body = src[start: nxt.start() if nxt else len(src)]
        assert re.search(
            r"apply_load_subtract\(\s*series, subtract_inputs,\s*"
            r"carry_deficit_days=\(\s*self\._local_dates\(grid_index\)\s*"
            r"if exp_cfg\.source_is_cumulative else None",
            body,
        )

    def test_day_labels_are_local_dates(self):
        from ml_forecast_lab.config import AppConfig
        from ml_forecast_lab.main import MLForecastLabApp
        app = MLForecastLabApp()
        app.config = AppConfig(experiments=[], timezone="Europe/London")
        # 23:30 UTC on 20 Oct is 00:30 BST on the 21st; 23:30 UTC on 1 Nov
        # is still the 1st (GMT).
        idx = pd.DatetimeIndex(["2026-10-20 23:30", "2026-11-01 23:30"])
        days = app._local_dates(idx)
        assert [str(d) for d in days] == ["2026-10-21", "2026-11-01"]


# ---------------------------------------------------------------------
# Cached models trained on the old labels
# ---------------------------------------------------------------------

class TestLabelRuleGate:
    @pytest.fixture
    def trained(self, tmp_db, tmp_path, monkeypatch):
        from tests.unit import test_cache_roundtrip as rt
        from ml_forecast_lab.main import MLForecastLabApp

        root = tmp_path / "models"
        monkeypatch.setattr(
            MLForecastLabApp, "_cached_model_dir",
            staticmethod(lambda exp_name: root / exp_name),
        )
        exp = rt._exp("lightgbm")
        app, _ = rt._make_app(tmp_db, exp)
        asyncio.run(app._retrain_and_cache(exp))
        return app, exp, root / exp.name / "cache_meta.json"

    @staticmethod
    def _as_old_rule(meta_file):
        meta = json.loads(meta_file.read_text())
        meta.pop("label_rule")
        meta_file.write_text(json.dumps(meta))

    @staticmethod
    def _restored(app, exp):
        app._cached_models.clear()
        app._restore_cached_models()
        return exp.name in app._cached_models

    def test_new_caches_record_the_label_rule(self, trained):
        from ml_forecast_lab.main import LABEL_RULE_VERSION
        _app, _exp, meta_file = trained
        assert json.loads(meta_file.read_text())["label_rule"] == LABEL_RULE_VERSION

    @pytest.mark.parametrize("field, value", [
        ("source_is_cumulative", True),
        ("load_subtract", [SubtractCfg(entity_id="sensor.ev")]),
    ])
    def test_an_affected_experiment_retrains_from_an_old_rule_cache(
        self, trained, field, value,
    ):
        app, exp, meta_file = trained
        setattr(exp, field, value)
        assert self._restored(app, exp), "a current-rule cache restores"
        self._as_old_rule(meta_file)
        assert not self._restored(app, exp)

    def test_other_experiments_keep_an_old_rule_cache(self, trained):
        app, exp, meta_file = trained
        self._as_old_rule(meta_file)
        assert self._restored(app, exp)

    def test_rollback_refuses_an_old_rule_generation(self, trained):
        app, exp, meta_file = trained
        asyncio.run(app._retrain_and_cache(exp))   # first one → previous/
        prev_meta = meta_file.parent / "previous" / "cache_meta.json"
        self._as_old_rule(prev_meta)
        exp.source_is_cumulative = True
        live_version = app._cached_models[exp.name]["model_version"]

        ok, msg = app._rollback_cached_model(exp.name)
        assert not ok and "earlier release" in msg
        assert app._cached_models[exp.name]["model_version"] == live_version
        assert "label_rule" in json.loads(meta_file.read_text()), "disk not swapped"

        exp.source_is_cumulative = False
        ok, msg = app._rollback_cached_model(exp.name)
        assert ok, msg
