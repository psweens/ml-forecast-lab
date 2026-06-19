"""Tests for SQLite database module."""

from datetime import datetime

import pandas as pd
import pytest

from ml_forecast_lab.db import HistoryDB


class TestSafeTableName:
    def test_normal_entity_id(self, tmp_db):
        db = HistoryDB(tmp_db)
        assert db.safe_table_name("sensor.temperature") == "sensor_temperature"

    def test_special_characters(self, tmp_db):
        db = HistoryDB(tmp_db)
        result = db.safe_table_name("sensor.my-entity.with.dots")
        assert result == "sensor_my_entity_with_dots"

    def test_leading_digit(self, tmp_db):
        db = HistoryDB(tmp_db)
        result = db.safe_table_name("123sensor")
        assert result[0] == "_", "Leading digit should be replaced with _"

    def test_rejects_empty_after_sanitisation(self, tmp_db):
        db = HistoryDB(tmp_db)
        with pytest.raises(ValueError, match="Invalid table name"):
            db.safe_table_name("")


class TestStoreAndRetrieve:
    """Round-trip the actual public API: store_history / get_history / cleanup."""

    def test_roundtrip(self, tmp_db):
        db = HistoryDB(tmp_db)
        table = db.safe_table_name("sensor.test_entity")
        idx = pd.date_range("2024-01-01", periods=10, freq="30min")
        data = pd.DataFrame({
            "ds": [t.isoformat() for t in idx],
            "value": range(10),
        })
        inserted = db.store_history(table, data)
        assert inserted == 10
        result = db.get_history(table)
        assert len(result) == 10

    def test_deduplication(self, tmp_db):
        """`INSERT OR IGNORE` collapses identical (ds, value) rows."""
        db = HistoryDB(tmp_db)
        table = db.safe_table_name("sensor.dedup_test")
        data = pd.DataFrame({
            "ds": ["2024-01-01T00:00:00", "2024-01-01T00:00:00"],
            "value": [1, 2],
        })
        db.store_history(table, data)
        result = db.get_history(table)
        # Composite key is (ds, value), so two distinct values at the same ts
        # are kept — but a true exact-duplicate insert is ignored.
        db.store_history(table, data)
        result_after = db.get_history(table)
        assert len(result_after) == len(result), "Re-inserting must not duplicate"

    def test_cleanup(self, tmp_db):
        db = HistoryDB(tmp_db)
        table = db.safe_table_name("sensor.cleanup_test")
        data = pd.DataFrame({
            "ds": ["2024-01-01T00:00:00", "2024-06-01T00:00:00"],
            "value": [1, 2],
        })
        db.store_history(table, data)
        db.cleanup(table, datetime(2024, 3, 1))
        result = db.get_history(table)
        assert len(result) == 1
        assert result.iloc[0]["y"] == 2


class TestForecastCoverage:
    """v2.39.3 bug 15 + T2: get_forecast_coverage's worst_bucket and the
    TEMP-table actuals_grid refactor must work end-to-end."""

    def _seed_coverage(self, db, *, in_band_frac_hour13=0.5, in_band_frac_hour19=0.99):
        """Seed forecast_log + actuals to give two distinct hour-of-day
        buckets — hour 13 under-covered (50%) and hour 19 over-covered
        (99%). For nominal=0.8, |dev(hour13)|=0.30 and
        |dev(hour19)|=0.19 — hour 13 should win 'worst' by |deviation|.

        v2.40.7: anchor all timestamps to ``now − 35 days`` rather than
        a hard-coded 2026-04-01 so the 30-day seed always lands inside
        ``max_age_days=60``. The old form drifted out of the window as
        wall-clock advanced past the fixture date.
        """
        from datetime import datetime as _dt, timedelta as _td

        db.ensure_forecast_log_table()
        table = db.safe_table_name("sensor.pv_power")
        # Floor to midnight so the hour-of-day buckets line up cleanly.
        anchor = (_dt.utcnow() - _td(days=35)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        # 30 days of synthetic actuals at 30-min freq.
        ds = pd.date_range(anchor, periods=30 * 48, freq="30min", tz="UTC")
        actuals = pd.DataFrame({
            "ds": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in ds],
            "value": [10.0] * len(ds),
        })
        db.store_history(table, actuals)

        # Forecast log: 30 rows per hour-of-day for both buckets, all
        # issued at the same instant for simplicity. Construct upper/
        # lower bounds so the actual (10.0) falls inside the requested
        # fraction of the time.
        issued = anchor
        for hour, frac in [(13, in_band_frac_hour13), (19, in_band_frac_hour19)]:
            targets, preds, ups, lows = [], [], [], []
            for day in range(30):
                target = anchor + _td(days=day, hours=hour)
                in_band = (day / 30) < frac
                # Centre on the actual; either tight (in-band) or
                # off-centre (out-of-band)
                if in_band:
                    ups.append(11.0); lows.append(9.0)
                else:
                    ups.append(5.0); lows.append(4.0)
                targets.append(target)
                preds.append(10.0)
            db.log_forecast(
                experiment="pv",
                issued_at=issued,
                targets=targets,
                predictions=preds,
                model_name="m1",
                upper_bounds=ups,
                lower_bounds=lows,
                model_version="v1",
            )

    def test_worst_bucket_uses_max_abs_deviation(self, tmp_db):
        """v2.39.3 bug 15: worst_bucket must be selected by max|deviation|
        from the nominal, NOT min(coverage). Pre-v2.39.3 an over-covered
        bucket (99% on nominal 80% → |dev|=0.19) lost to an under-covered
        one (50% → |dev|=0.30) only because under-cover is numerically
        smaller — but the OPPOSITE example (75% vs 99%) was wrong:
        pre-fix picked 75% (|dev|=0.05) over 99% (|dev|=0.19)."""
        db = HistoryDB(tmp_db)
        # 75% in hour 13, 99% in hour 19 — hour 19's |dev| (0.19) is
        # bigger than hour 13's (0.05) under nominal 0.8.
        self._seed_coverage(
            db, in_band_frac_hour13=0.75, in_band_frac_hour19=0.99,
        )
        result = db.get_forecast_coverage(
            experiment="pv",
            actuals_table=db.safe_table_name("sensor.pv_power"),
            interval_minutes=30,
            max_age_days=60,
            tz="UTC",
            nominal=0.8,
        )
        worst = result.get("worst_bucket")
        assert worst is not None
        # hour 19 has the largest |dev|; pre-fix would have returned hour 13
        assert worst["label"] == "hour 19", (
            f"worst_bucket should be the one with largest |coverage-nominal|, "
            f"got {worst}"
        )

    def test_worst_bucket_respects_caller_nominal(self, tmp_db):
        """Pass nominal=0.5 and the candidate distances flip: hour 13
        (50% → |dev|=0.0) becomes 'best', hour 19 (99% → |dev|=0.49)
        becomes worst."""
        db = HistoryDB(tmp_db)
        self._seed_coverage(db)
        result = db.get_forecast_coverage(
            experiment="pv",
            actuals_table=db.safe_table_name("sensor.pv_power"),
            interval_minutes=30,
            max_age_days=60,
            tz="UTC",
            nominal=0.5,
        )
        worst = result.get("worst_bucket")
        assert worst is not None
        assert worst["label"] == "hour 19"


class TestForecastAccuracyIncrementMode:
    """v2.40.7 regression: increment mode must NOT double-difference the
    forecast.

    ``forecast_log.predicted`` is logged from ``y_pred.tolist()`` in
    ``main.py:5068``, and ``y_pred`` is already the per-interval delta
    for cumulative-source experiments (the HA ``_cumulative`` sensor is
    derived by cumsum-ing ``y_pred`` at ``main.py:5194-5208`` but is
    never logged). The increment-mode actuals CTE diffs the raw
    cumulative correctly. The forecast CTE must therefore PASS
    ``predicted`` THROUGH UNCHANGED — taking a LAG diff on it produces a
    second-difference and compares apples to oranges. The trajectory
    function (db.py:1040-1053, 1084) gets this right; the accuracy
    function must mirror it.
    """

    def _seed_perfect_cumulative_forecast(self, db):
        """Seed actuals = cumulative counter 10 → 12 → 15 → 16 over four
        consecutive 30-min grid points, and one forecast issuance whose
        predictions are the corresponding per-interval deltas [2, 3, 1]
        at the trailing three grid points. This is what a PERFECT model
        would log on a cumulative-source experiment.
        """
        from datetime import datetime as _dt, timedelta as _td

        db.ensure_forecast_log_table()
        table = db.safe_table_name("sensor.demand_today")

        # Use recent timestamps so max_age_days=30 covers them.
        base = (_dt.utcnow() - _td(days=2)).replace(
            minute=0, second=0, microsecond=0,
        )
        grid_times = [base + _td(minutes=30 * i) for i in range(4)]

        # Cumulative actuals: 10, 12, 15, 16
        actuals = pd.DataFrame({
            "ds": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in grid_times],
            "value": [10.0, 12.0, 15.0, 16.0],
        })
        db.store_history(table, actuals)

        # Forecast: issued at grid_times[0], predicts the per-interval
        # delta at the next three grid points. Perfect model → predicted
        # deltas equal actual deltas.
        issued = grid_times[0]
        targets = grid_times[1:]                  # T1, T2, T3
        predictions = [2.0, 3.0, 1.0]             # actuals[i+1]-actuals[i]
        db.log_forecast(
            experiment="demand",
            issued_at=issued,
            targets=targets,
            predictions=predictions,
            model_name="m1",
            model_version="v1",
        )
        return table

    def test_increment_mode_perfect_cumulative_forecast_scores_zero(self, tmp_db):
        """A perfect cumulative-source prediction must score MAE ≈ 0 in
        increment mode. Before the fix the forecast CTE took a SECOND
        LAG diff, scoring the perfect model with MAE = 2.0 (and
        silently discarding the negative-second-difference row via the
        ``fv.value >= 0`` filter).
        """
        db = HistoryDB(tmp_db)
        table = self._seed_perfect_cumulative_forecast(db)

        result = db.get_forecast_accuracy(
            experiment="demand",
            actuals_table=table,
            max_age_days=30,
            interval_minutes=30,
            evaluation_mode="increment",
        )

        curve = result.get("lead_time_curve", {})
        maes = curve.get("mae") or []
        ns = curve.get("sample_count") or []
        assert maes, (
            f"lead_time_curve is empty — perfect forecast should produce "
            f"at least one bucket. Got: {curve}"
        )

        # Sample-weighted mean MAE across buckets — for a perfect model
        # this must be 0. Before the fix it was 2.0 (the typical
        # per-interval demand) and several rows were silently dropped.
        total_n = sum(ns)
        assert total_n >= 2, (
            f"Expected ≥ 2 matched rows for a 3-point perfect forecast, "
            f"got n_total={total_n}. The ``fv.value >= 0`` filter is "
            f"likely dropping legitimate rows because of the spurious "
            f"second-difference."
        )
        weighted_mae = sum(m * n for m, n in zip(maes, ns)) / total_n
        assert weighted_mae == pytest.approx(0.0, abs=1e-9), (
            f"Perfect cumulative forecast scored MAE={weighted_mae} in "
            f"increment mode (expected 0). The forecast is being "
            f"second-differenced while the actual is first-differenced "
            f"— predicted and actual are in different spaces."
        )

    def test_increment_mode_revision_tile_also_correct(self, tmp_db):
        """The revision-improvement tile (``first_forecast_mae`` /
        ``latest_forecast_mae``) reuses the same forecast_vals CTE, so
        it must also be zero for a perfect forecast in increment mode.
        Locks in the fact that the same fix lands on both.
        """
        db = HistoryDB(tmp_db)
        table = self._seed_perfect_cumulative_forecast(db)

        result = db.get_forecast_accuracy(
            experiment="demand",
            actuals_table=table,
            max_age_days=30,
            interval_minutes=30,
            evaluation_mode="increment",
        )

        rev = result.get("revision_improvement") or {}
        # When there's only one issuance per target (no revisions), the
        # tile may report zero samples and skip — accept either an empty
        # tile or one whose MAEs are zero.
        if rev.get("sample_count"):
            assert rev["first_forecast_mae"] == pytest.approx(0.0, abs=1e-9)
            assert rev["latest_forecast_mae"] == pytest.approx(0.0, abs=1e-9)


class TestForecastAccuracyDailyCumulativeMode:
    """v2.40.9: daily_cumulative evaluation mode for daily-reset
    cumulative sensors.

    Question the mode answers: "how close is the forecast's predicted
    cumulative reading at target_dt to the actual cumulative reading at
    target_dt?" — i.e. accuracy of the running daily total rather than
    accuracy of each per-interval delta.

    For each forecast row the predicted_cumulative at target_dt is:

        seed + Σ (per-interval predictions within target_dt's local day,
                  from the first target onward, up to and including
                  target_dt)

    where ``seed = actual_cumulative_at(issued_at)`` when target_dt is
    in the SAME local day as issued_at, otherwise ``seed = 0`` (the
    midnight reset makes the prior day's accumulation irrelevant).

    Compared against ``actual_cumulative_at(target_dt)`` — which for a
    daily-reset sensor is the raw ``_today`` reading at that moment.
    """

    def _seed_same_day_perfect_forecast(self, db):
        """One forecast issuance entirely within a single day.

        Cumulative actuals at hourly bins:
          09:00 = 5, 09:30 = 8, 10:00 = 12, 10:30 = 14, 11:00 = 18

        Forecast issued at 09:00 (cumulative reading = 5), predicting
        the next four per-interval deltas perfectly: [3, 4, 2, 4].

        Predicted cumulatives = seed + cumsum = 5 + [3, 7, 9, 13]
                              = [8, 12, 14, 18]
        Actual cumulatives    = [8, 12, 14, 18]
        → MAE = 0 at every lead.
        """
        from datetime import datetime as _dt, timedelta as _td

        db.ensure_forecast_log_table()
        table = db.safe_table_name("sensor.demand_today")

        # Use a recent calendar day so max_age_days=30 covers it.
        base_day = (_dt.utcnow() - _td(days=2)).replace(
            hour=9, minute=0, second=0, microsecond=0,
        )
        grid_times = [base_day + _td(minutes=30 * i) for i in range(5)]

        actuals = pd.DataFrame({
            "ds": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in grid_times],
            "value": [5.0, 8.0, 12.0, 14.0, 18.0],
        })
        db.store_history(table, actuals)

        issued = grid_times[0]
        targets = grid_times[1:]
        predictions = [3.0, 4.0, 2.0, 4.0]
        db.log_forecast(
            experiment="demand",
            issued_at=issued,
            targets=targets,
            predictions=predictions,
            model_name="m1",
            model_version="v1",
        )
        return table

    def test_daily_cumulative_perfect_forecast_scores_zero(self, tmp_db):
        """Perfect cumulative-source forecast must score MAE ≈ 0 in
        daily_cumulative mode — the running-total prediction matches
        the running-total actual at every lead."""
        db = HistoryDB(tmp_db)
        table = self._seed_same_day_perfect_forecast(db)

        result = db.get_forecast_accuracy(
            experiment="demand",
            actuals_table=table,
            max_age_days=30,
            interval_minutes=30,
            evaluation_mode="daily_cumulative",
        )

        curve = result.get("lead_time_curve", {})
        maes = curve.get("mae") or []
        ns = curve.get("sample_count") or []
        assert maes, f"Empty lead_time_curve; got: {curve}"
        total_n = sum(ns)
        assert total_n >= 4, (
            f"Expected ≥ 4 matched rows for 4 perfect predictions, "
            f"got {total_n}."
        )
        weighted_mae = sum(m * n for m, n in zip(maes, ns)) / total_n
        assert weighted_mae == pytest.approx(0.0, abs=1e-9), (
            f"Perfect daily-cumulative forecast scored MAE={weighted_mae} "
            f"(expected 0). Predicted cumulative does not match actual "
            f"cumulative at the target."
        )

    def test_daily_cumulative_off_by_one_per_interval_accumulates(self, tmp_db):
        """A model that over-predicts each interval delta by 1 unit
        should produce a cumulative error that grows linearly with lead
        — at lead k the cumulative error is k * (per-interval error).

        Locks in: cumulative-space errors integrate per-interval errors,
        so the chart slopes upward and the headline MAE on lead 30 is
        very different from MAE at end-of-day. This is intentional —
        cumulative is a DIFFERENT metric, not strictly comparable to
        per-interval MAE.
        """
        from datetime import datetime as _dt, timedelta as _td

        db = HistoryDB(tmp_db)
        db.ensure_forecast_log_table()
        table = db.safe_table_name("sensor.demand_today")

        base = (_dt.utcnow() - _td(days=2)).replace(
            hour=9, minute=0, second=0, microsecond=0,
        )
        grid_times = [base + _td(minutes=30 * i) for i in range(5)]
        # Actual cumulative: 0, 2, 4, 6, 8 (constant +2/bin)
        actuals_vals = [0.0, 2.0, 4.0, 6.0, 8.0]
        actuals = pd.DataFrame({
            "ds": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in grid_times],
            "value": actuals_vals,
        })
        db.store_history(table, actuals)

        # Model predicts +3/bin (off by +1 each). Seed=0 at issuance.
        # Predicted cumulatives at the four targets:
        #   lead 30: 0 + 3 = 3   (actual=2, err=1)
        #   lead 60: 0 + 6 = 6   (actual=4, err=2)
        #   lead 90: 0 + 9 = 9   (actual=6, err=3)
        #   lead120: 0 + 12 = 12 (actual=8, err=4)
        db.log_forecast(
            experiment="demand",
            issued_at=grid_times[0],
            targets=grid_times[1:],
            predictions=[3.0, 3.0, 3.0, 3.0],
            model_name="m1",
            model_version="v1",
        )

        result = db.get_forecast_accuracy(
            experiment="demand",
            actuals_table=table,
            max_age_days=30,
            interval_minutes=30,
            evaluation_mode="daily_cumulative",
        )
        curve = result["lead_time_curve"]
        leads = curve["lead_minutes"]
        maes = curve["mae"]
        # Each lead bucket should hold one row.
        assert curve["sample_count"] == [1, 1, 1, 1]
        # Errors grow linearly with lead.
        by_lead = dict(zip(leads, maes))
        assert by_lead[30] == pytest.approx(1.0, abs=1e-9)
        assert by_lead[60] == pytest.approx(2.0, abs=1e-9)
        assert by_lead[90] == pytest.approx(3.0, abs=1e-9)
        assert by_lead[120] == pytest.approx(4.0, abs=1e-9)

    def test_daily_cumulative_day_offset_hours_shifts_bucket(self, tmp_db):
        """v2.40.10 regression: ``day_offset_hours`` must shift the
        day-bucketing key so the cumulative sensor's local-midnight
        reset is respected.

        Setup: a BST-style sensor (HA-local = UTC+1). The actual
        ``_today`` counter sits at 30 throughout the late evening, then
        resets to 0 at LOCAL midnight = 23:00 UTC.

        Forecast issued at 21:00 UTC = 22:00 local, predicting forward
        through 23:30 UTC (= 00:30 next local day):
          - Without offset (UTC bucketing): "last same-day" target =
            23:30 UTC. Actual at 23:30 UTC reads 0 (post-reset). The
            End-of-day card thinks the daily total was 0, badly off.
          - With ``day_offset_hours=1.0``: target_day for 23:30 UTC is
            "next local day". Only 22:00 and 22:30 UTC are same-day-as-
            issuance. The "last same-day" target is then 22:30 UTC,
            where actual = 30 (pre-reset) — the correct daily total.
        """
        from datetime import datetime as _dt, timedelta as _td

        db = HistoryDB(tmp_db)
        db.ensure_forecast_log_table()
        table = db.safe_table_name("sensor.demand_today")

        # Anchor recently so max_age_days=30 covers it. Pick a base
        # date and snap to 21:00 UTC.
        base = (_dt.utcnow() - _td(days=2)).replace(
            hour=21, minute=0, second=0, microsecond=0,
        )
        # Actuals at 21:00, 21:30, 22:00, 22:30 UTC = 30
        # Reset at 23:00 UTC (local midnight in UTC+1)
        # Actuals at 23:00, 23:30 UTC = 0 (post-reset, no new draws yet)
        grid_times = [base + _td(minutes=30 * i) for i in range(6)]
        actual_vals = [30.0, 30.0, 30.0, 30.0, 0.0, 0.0]
        actuals = pd.DataFrame({
            "ds": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in grid_times],
            "value": actual_vals,
        })
        db.store_history(table, actuals)

        # Forecast: zero per-interval demand (sensor was idle); predicts
        # the cumulative will stay at 30 same-local-day, then drop to 0
        # after reset.
        db.log_forecast(
            experiment="demand",
            issued_at=grid_times[0],
            targets=grid_times[1:],
            predictions=[0.0, 0.0, 0.0, 0.0, 0.0],
            model_name="m1",
            model_version="v1",
        )

        # Without offset (UTC bucketing): the "last same-day" target is
        # 23:30 UTC (in same UTC date as 21:00). Actual there is 0
        # (post-reset). So the avg actual day-total reads ~0.
        result_utc = db.get_forecast_accuracy(
            experiment="demand", actuals_table=table,
            max_age_days=30, interval_minutes=30,
            evaluation_mode="daily_cumulative",
        )
        eod_utc = result_utc.get("end_of_day", {})
        assert eod_utc.get("sample_count", 0) > 0
        # UTC bucketing: avg actual is the post-reset value → near 0.
        # (This documents the broken pre-fix behaviour.)
        assert eod_utc["mean_actual"] == pytest.approx(0.0, abs=1e-9), (
            f"Expected UTC bucketing to read post-reset 0; got "
            f"{eod_utc['mean_actual']}"
        )

        # With offset = +1.0 hours: the local day "rolls over" at 23:00
        # UTC. Targets at 23:00 and 23:30 UTC are in the NEXT local
        # day, so the "last same-day" target is 22:30 UTC. Actual at
        # 22:30 = 30 — the correct daily total.
        result_local = db.get_forecast_accuracy(
            experiment="demand", actuals_table=table,
            max_age_days=30, interval_minutes=30,
            evaluation_mode="daily_cumulative",
            day_offset_hours=1.0,
        )
        eod_local = result_local.get("end_of_day", {})
        assert eod_local.get("sample_count", 0) > 0
        assert eod_local["mean_actual"] == pytest.approx(30.0, abs=1e-9), (
            f"With day_offset_hours=1.0 the last same-day target should "
            f"be 22:30 UTC (pre-reset, value=30), but mean_actual "
            f"reads {eod_local['mean_actual']}."
        )

    def test_daily_cumulative_cross_midnight_resets_seed(self, tmp_db):
        """A forecast issued late on day D predicting targets into day
        D+1 must reset the cumulative seed to 0 at the midnight
        boundary — the daily-reset sensor restarts from zero, so the
        prior day's cumulation must NOT carry over.
        """
        from datetime import datetime as _dt, timedelta as _td

        db = HistoryDB(tmp_db)
        db.ensure_forecast_log_table()
        table = db.safe_table_name("sensor.demand_today")

        # Build two grid days with a clean midnight boundary. The
        # day-D end:    22:30 = 30, 23:00 = 32, 23:30 = 33
        # day-D+1:      00:00 = 0,  00:30 = 4,  01:00 = 7
        # (D+1 actuals reset to 0 at midnight then accumulate.)
        d0 = (_dt.utcnow() - _td(days=2)).replace(
            hour=22, minute=30, second=0, microsecond=0,
        )
        grid_d0 = [d0 + _td(minutes=30 * i) for i in range(3)]   # 22:30, 23:00, 23:30
        grid_d1 = [d0 + _td(minutes=30 * (i + 3)) for i in range(3)]  # 00:00, 00:30, 01:00
        all_times = grid_d0 + grid_d1
        all_vals = [30.0, 32.0, 33.0, 0.0, 4.0, 7.0]
        actuals = pd.DataFrame({
            "ds": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in all_times],
            "value": all_vals,
        })
        db.store_history(table, actuals)

        # Forecast issued at 22:30 (seed = 30) predicting all 5 forward
        # deltas perfectly:
        #   23:00 same day  → cum = 30 + 2 = 32
        #   23:30 same day  → cum = 30 + 3 = 33
        #   00:00 new day   → seed resets to 0, cum = 0 + 0 = 0  (delta=0 at midnight)
        #   00:30 new day   → cum = 0 + 4 = 4
        #   01:00 new day   → cum = 0 + 7 = 7
        # Actual deltas: 2, 1, -33 (reset; SQL adjacency guard nulls),
        #                4, 3. The midnight reset row is dropped by the
        #                actuals adjacency guard (>= 0 filter).
        deltas = [2.0, 1.0, 0.0, 4.0, 3.0]
        db.log_forecast(
            experiment="demand",
            issued_at=grid_d0[0],
            targets=all_times[1:],
            predictions=deltas,
            model_name="m1",
            model_version="v1",
        )

        result = db.get_forecast_accuracy(
            experiment="demand",
            actuals_table=table,
            max_age_days=30,
            interval_minutes=30,
            evaluation_mode="daily_cumulative",
        )
        curve = result["lead_time_curve"]
        maes = curve["mae"]
        ns = curve["sample_count"]
        assert maes, "Empty lead_time_curve on cross-midnight forecast"
        total_n = sum(ns)
        weighted_mae = sum(m * n for m, n in zip(maes, ns)) / total_n
        assert weighted_mae == pytest.approx(0.0, abs=1e-9), (
            f"Cross-midnight forecast scored MAE={weighted_mae}; expected "
            f"0. The seed likely failed to reset at the midnight "
            f"boundary, so day-D+1 cumulatives carried day-D's offset."
        )


class TestForecastLogBlowupGuard:
    """The write-path guard in log_forecast keeps a log-inversion blow-up out
    of forecast_log entirely — the single source every analytics tab reads, so
    a ~1e30 / inf value can't corrupt any of them (Accuracy, Comparison,
    trajectory, evolution, stability)."""

    def _targets(self, issued, n):
        from datetime import timedelta as _td
        return [issued + _td(minutes=30 * (i + 1)) for i in range(n)]

    def test_blowup_predictions_are_dropped_before_insert(self, tmp_db):
        from datetime import datetime as _dt
        db = HistoryDB(tmp_db)
        db.ensure_forecast_log_table()
        issued = _dt(2024, 1, 1, 0, 0, 0)
        # 2 good values + inf + 5e30 + nan → only the 2 good survive.
        preds = [1.0, float("inf"), 5e30, float("nan"), 2.0]
        n = db.log_forecast(
            experiment="pv", issued_at=issued,
            targets=self._targets(issued, 5), predictions=preds,
            model_name="m1", model_version="v1",
        )
        assert n == 2
        cur = db.conn.cursor()
        cur.execute(
            "SELECT predicted FROM forecast_log WHERE experiment='pv' "
            "ORDER BY target_dt"
        )
        vals = [r[0] for r in cur.fetchall()]
        assert vals == [1.0, 2.0]
        assert all(abs(v) < 1e12 for v in vals)

    def test_blowup_bound_nulled_but_point_kept(self, tmp_db):
        from datetime import datetime as _dt
        db = HistoryDB(tmp_db)
        db.ensure_forecast_log_table()
        issued = _dt(2024, 1, 1, 0, 0, 0)
        db.log_forecast(
            experiment="pv", issued_at=issued,
            targets=self._targets(issued, 1), predictions=[10.0],
            model_name="m1", upper_bounds=[1e30], lower_bounds=[9.0],
            model_version="v1",
        )
        cur = db.conn.cursor()
        cur.execute(
            "SELECT predicted, upper, lower FROM forecast_log "
            "WHERE experiment='pv'"
        )
        pred, upper, lower = cur.fetchone()
        assert pred == 10.0       # point forecast kept
        assert upper is None      # absurd band nulled
        assert lower == 9.0       # sane band kept

    def test_normal_forecast_is_untouched(self, tmp_db):
        from datetime import datetime as _dt
        db = HistoryDB(tmp_db)
        db.ensure_forecast_log_table()
        issued = _dt(2024, 1, 1, 0, 0, 0)
        preds = [0.0, 5.0, 12.5, 100.0]
        n = db.log_forecast(
            experiment="pv", issued_at=issued,
            targets=self._targets(issued, 4), predictions=preds,
            model_name="m1", model_version="v1",
        )
        assert n == 4

