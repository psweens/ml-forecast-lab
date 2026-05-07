"""Tests for SQLite database module."""

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
    def test_roundtrip(self, tmp_db):
        db = HistoryDB(tmp_db)
        entity = "sensor.test_entity"
        idx = pd.date_range("2024-01-01", periods=10, freq="30min")
        data = pd.DataFrame({
            "ds": [t.isoformat() for t in idx],
            "value": range(10),
        })
        db.store(entity, data)
        result = db.load(entity)
        assert len(result) == 10

    def test_deduplication(self, tmp_db):
        db = HistoryDB(tmp_db)
        entity = "sensor.dedup_test"
        data = pd.DataFrame({
            "ds": ["2024-01-01T00:00:00", "2024-01-01T00:00:00"],
            "value": [1, 2],
        })
        db.store(entity, data)
        result = db.load(entity)
        assert len(result) == 1  # INSERT OR IGNORE deduplicates

    def test_cleanup(self, tmp_db):
        db = HistoryDB(tmp_db)
        entity = "sensor.cleanup_test"
        data = pd.DataFrame({
            "ds": ["2024-01-01T00:00:00", "2024-06-01T00:00:00"],
            "value": [1, 2],
        })
        db.store(entity, data)
        db.cleanup(entity, "2024-03-01T00:00:00")
        result = db.load(entity)
        assert len(result) == 1
        assert result.iloc[0]["value"] == 2
