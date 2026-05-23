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
        |dev(hour19)|=0.19 — hour 13 should win 'worst' by |deviation|."""
        from datetime import datetime as _dt, timedelta as _td

        db.ensure_forecast_log_table()
        table = db.safe_table_name("sensor.pv_power")
        # 30 days of synthetic actuals at 30-min freq.
        ds = pd.date_range("2026-04-01", periods=30 * 48, freq="30min", tz="UTC")
        actuals = pd.DataFrame({
            "ds": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in ds],
            "value": [10.0] * len(ds),
        })
        db.store_history(table, actuals)

        # Forecast log: 30 rows per hour-of-day for both buckets, all
        # issued at the same instant for simplicity. Construct upper/
        # lower bounds so the actual (10.0) falls inside the requested
        # fraction of the time.
        issued = _dt(2026, 4, 1)
        for hour, frac in [(13, in_band_frac_hour13), (19, in_band_frac_hour19)]:
            targets, preds, ups, lows = [], [], [], []
            for day in range(30):
                target = _dt(2026, 4, 1) + _td(days=day, hours=hour)
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
