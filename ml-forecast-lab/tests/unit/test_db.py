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
