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

