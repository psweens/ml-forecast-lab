"""v2.52.4: increment-mode actuals follow the training label's rule.

A cumulative experiment is trained on ``cumulative_to_interval`` followed by
a sum resample: each bin's label is the sum of reading-to-reading
increments recorded in it, an empty bin is a zero increment, and since
v2.40.5 the change after a quiet stretch counts in full in the bin where it
was recorded. The analytics' increment actuals (``_mlfl_actuals_vals_tmp``)
instead differenced bin means and set a bin to NULL unless the previous bin
had a reading. Against a counter that only logs changes, as HA's recorder
does, that:

- never scored an empty (zero-use) bin, since it had no grid row;
- dropped the first bin after every quiet stretch, often the day's main
  draw, and the bin containing the daily reset;
- compared multi-reading bins on a half-bin-shifted difference of means.

Bands were then calibrated on a small, biased subset: about three times too
wide for a change-only demand counter in simulation. The actuals now follow
training per reading: each reading's rise over the previous one, or the
reading itself after a restart (a drop below 90% of the previous reading),
summed per bin, with zero for a bin with no readings. A reset inside a bin
keeps the use recorded before it. They equal the training label except
where training caps a spike (``max_increment``), relabels a small rise
across UTC midnight as a reset (``reset_daily``), or treats a dip as a
reset; the helper below disables the first two. A dip (a drop to no less
than 90%) is negative use here, which the recovery cancels, so it cannot
score a lifetime counter's whole reading as one interval's use.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.db import HistoryDB
from ml_forecast_lab.preprocessing import cumulative_to_interval, resample_to_grid

INTERVAL = 30


def change_only_counter(days: int = 3, seed: int = 5):
    """Readings of a daily-reset counter as a change-only recorder stores
    them: two readings in a bin with use (40% then 100% of its increment),
    none in a quiet bin, and a reading of the reset value after UTC
    midnight. Returns (readings, grid, true per-bin increments)."""
    rng = np.random.default_rng(seed)
    end = datetime(2026, 9, 20)
    grid = pd.date_range(end - timedelta(days=days), end, freq=f"{INTERVAL}min")[:-1]
    inc = np.where(rng.random(len(grid)) < 0.35, rng.uniform(0.1, 1.5, len(grid)), 0.0).round(3)
    inc[0] = inc[-1] = 0.5      # the window opens and closes on a reading
    inc[40:52] = 0.0            # a six-hour quiet stretch
    inc[96] = 0.7               # use in a reset bin (09-19 00:00)
    rows, total, last = [], 0.0, None
    for i, ts in enumerate(grid):
        if i and ts.normalize() != grid[i - 1].normalize():
            total = 0.0
        for frac, minute in ((0.4, 5), (1.0, 20)):
            value = total + frac * inc[i]
            if value != last:
                rows.append((ts + timedelta(minutes=minute), value))
                last = value
        total += inc[i]
    readings = pd.DataFrame(rows, columns=["ds", "value"])
    return readings, grid, inc


def training_label(readings: pd.DataFrame, interval: int = INTERVAL) -> pd.Series:
    """The label the pipeline trains on, with the spike cap disabled."""
    series = readings.set_index("ds")["value"]
    per_reading = cumulative_to_interval(series, interval, max_increment=1e9)
    return resample_to_grid(per_reading, freq=f"{interval}min", method="sum")


def increment_actuals(db: HistoryDB, table: str, interval: int = INTERVAL) -> pd.Series:
    cur = db.conn.cursor()
    assert db._materialise_actuals_grid(cur, table, interval * 60, increment=True)
    rows = cur.execute(
        "SELECT grid_dt, value FROM _mlfl_actuals_vals_tmp ORDER BY grid_dt"
    ).fetchall()
    return pd.Series(
        [v for _, v in rows], index=pd.to_datetime([g for g, _ in rows]),
        dtype="float64",
    )


@pytest.fixture
def counter_db(tmp_path):
    db = HistoryDB(tmp_path / "history.db")
    table = db.safe_table_name("sensor.demand_today")
    readings, grid, inc = change_only_counter()
    db.store_history(table, readings)
    return db, table, readings, grid, inc


class TestIncrementActuals:
    def test_matches_the_training_label(self, counter_db):
        db, table, readings, grid, inc = counter_db
        actuals = increment_actuals(db, table)
        label = training_label(readings)

        assert list(actuals.index) == list(grid)
        # The window's first bin has no prior reading; training zeroes it.
        assert np.isnan(actuals.iloc[0])
        np.testing.assert_allclose(actuals.iloc[1:], label.iloc[1:], atol=1e-9)
        np.testing.assert_allclose(actuals.iloc[1:], inc[1:], atol=1e-9)

    def test_quiet_bins_are_zero_and_the_next_draw_counts(self, counter_db):
        db, table, _readings, grid, inc = counter_db
        actuals = increment_actuals(db, table)
        assert (actuals.iloc[40:52] == 0.0).all()
        first_draw = next(i for i in range(52, len(grid)) if inc[i] > 0)
        assert actuals.iloc[first_draw] == pytest.approx(inc[first_draw])

    def test_reset_bin_counts_its_own_use(self, counter_db):
        db, table, _readings, grid, inc = counter_db
        actuals = increment_actuals(db, table)
        midnights = [i for i in range(1, len(grid)) if grid[i].hour == 0 and grid[i].minute == 0]
        assert midnights
        for i in midnights:
            assert actuals.iloc[i] == pytest.approx(inc[i], abs=1e-9)

    def test_raw_grid_is_unchanged(self, counter_db):
        db, table, readings, _grid, _inc = counter_db
        cur = db.conn.cursor()
        assert db._materialise_actuals_grid(cur, table, INTERVAL * 60)
        rows = cur.execute(
            "SELECT grid_dt, value FROM _mlfl_actuals_grid_tmp ORDER BY grid_dt"
        ).fetchall()
        expected = readings.set_index("ds")["value"].resample(f"{INTERVAL}min").mean().dropna()
        assert [g for g, _ in rows] == [t.strftime("%Y-%m-%d %H:%M:%S") for t in expected.index]
        np.testing.assert_allclose([v for _, v in rows], expected.values, atol=1e-9)

    def test_perfect_forecast_scores_zero(self, counter_db):
        """A model that forecasts the training label exactly gets a zero
        band, full coverage and zero error, quiet bins included."""
        db, table, readings, grid, inc = counter_db
        db.ensure_forecast_log_table()
        for i in range(1, len(grid)):
            target = grid[i].to_pydatetime()
            db.log_forecast(
                "demand", target - timedelta(minutes=INTERVAL), [target],
                [float(inc[i])], "lightgbm",
                upper_bounds=[float(inc[i]) + 1e-6],
                lower_bounds=[float(inc[i]) - 1e-6],
                model_version="v1",
            )
        kw = dict(model_name="lightgbm", model_version="v1",
                  interval_minutes=INTERVAL)
        cq = db.get_conformal_quantiles(
            "demand", table, level=0.8, max_age_days=3650,
            source_is_cumulative=True, **kw,
        )
        assert cq["total_samples"] == len(grid) - 1
        assert cq["fallback_quantile"] == pytest.approx(0.0, abs=1e-9)

        cov = db.get_forecast_coverage(
            "demand", table, max_age_days=3650, source_is_cumulative=True, **kw,
        )
        assert cov["overall"] == {"coverage": 1.0, "n": len(grid) - 1}

        acc = db.get_forecast_accuracy(
            "demand", table, max_age_days=3650, interval_minutes=INTERVAL,
            evaluation_mode="increment", model_name="lightgbm",
        )
        assert sum(acc["lead_time_curve"]["sample_count"]) == len(grid) - 1
        assert max(acc["lead_time_curve"]["mae"]) == pytest.approx(0.0, abs=1e-9)


class TestRestartsAndDips:
    def test_reset_inside_a_bin_keeps_the_use_before_it(self, tmp_path):
        """60-min bins with the daily reset at :30 (a UTC+5:30 local
        midnight): the bin holds use before and after the reset."""
        db = HistoryDB(tmp_path / "history.db")
        table = db.safe_table_name("sensor.demand_today")
        idx = pd.date_range("2026-09-17 00:00", "2026-09-20 00:00", freq="10min")
        rng = np.random.default_rng(3)
        step = rng.uniform(0.0, 0.2, len(idx)).round(3)
        values, total = [], 0.0
        for ts, d in zip(idx, step):
            if ts.hour == 18 and ts.minute == 30:
                total = 0.0
            total += d
            values.append(total)
        readings = pd.DataFrame({"ds": idx, "value": values})
        db.store_history(table, readings)

        actuals = increment_actuals(db, table, interval=60)
        label = training_label(readings, interval=60)
        np.testing.assert_allclose(actuals.iloc[1:], label.iloc[1:].values, atol=1e-9)
        reset_bins = [t for t in actuals.index if t.hour == 18]
        assert reset_bins and all(actuals[t] > 0.1 for t in reset_bins)

    def test_small_dip_is_not_a_restart(self, tmp_path):
        """A lifetime counter that logs one reading 0.05 below the last,
        under the previous bin's last reading, then recovers."""
        db = HistoryDB(tmp_path / "history.db")
        table = db.safe_table_name("sensor.energy_total")
        idx = pd.date_range("2026-09-17 00:00", periods=48 * 3, freq="10min")
        values = 12345.0 + 0.01 * np.arange(len(idx))
        values[50] = values[49] - 0.05           # 08:20, last reading of the 08:00 bin
        db.store_history(table, pd.DataFrame({"ds": idx, "value": values}))

        actuals = increment_actuals(db, table)
        assert actuals.iloc[1:].abs().max() < 0.1
        # Use is conserved across the dip and its recovery.
        assert actuals.iloc[1:].sum() == pytest.approx(
            values[-1] - values[2], abs=1e-6,
        )

    def test_counter_restart_counts_the_new_reading(self, tmp_path):
        db = HistoryDB(tmp_path / "history.db")
        table = db.safe_table_name("sensor.energy_total")
        idx = pd.date_range("2026-09-17 00:00", periods=12, freq="30min")
        values = [100.0, 100.5, 101.0, 101.5, 102.0, 0.3, 0.8, 1.3, 1.8, 2.3, 2.8, 3.3]
        db.store_history(table, pd.DataFrame({"ds": idx, "value": values}))

        actuals = increment_actuals(db, table)
        np.testing.assert_allclose(actuals.iloc[1:], [0.5] * 4 + [0.3] + [0.5] * 6)
