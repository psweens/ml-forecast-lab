"""
v2.56.0: the unit of an external forecast's values.

Pins the defect where the Forecast Comparison tab read an attribute-mode
forecast in its sensor's HA unit. A forecast sensor's state can be the day's
total in kWh while each trajectory entry is average power in kW; such entries
were scored as kWh per interval: ×2 on a 30-minute grid, inside the 4×
scale-mismatch guard. These external forecasts then showed about double their
error and a large positive bias, and the app appeared to beat them.

  - ``ExternalForecastCfg.unit`` (canonical spelling, validation, tolerant
    YAML load) and ``set_experiment_external_forecast_unit``.
  - ``suspect_external_unit``: the scale-ratio test for a power/energy mix-up.
  - ``get_external_forecast_comparison``: a sensor-unit forecast that matches
    the mix-up is flagged with a suggested unit and kept out of the matched-
    lead ranking; an explicit unit is trusted and scores correctly.
"""

import logging
import math
from datetime import timedelta

import pandas as pd
import pytest
import yaml

from ml_forecast_lab.config import (
    ExternalForecastCfg, EXTERNAL_FORECAST_UNITS, load_config,
    add_experiment_external_forecast, set_experiment_external_forecast_unit,
)
from ml_forecast_lab.db import HistoryDB, suspect_external_unit


INTERVAL = 30
GENEROUS_WINDOW = 3650


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

class TestUnitConfig:
    def test_default_none_and_canonical_spelling(self):
        assert ExternalForecastCfg(entity_id="sensor.a").unit is None
        assert ExternalForecastCfg(entity_id="sensor.a", unit="kw").unit == "kW"
        assert ExternalForecastCfg(entity_id="sensor.a", unit=" MWH ").unit == "MWh"
        assert ExternalForecastCfg(entity_id="sensor.a", unit="").unit is None

    def test_rejects_other_units(self):
        with pytest.raises(ValueError):
            ExternalForecastCfg(entity_id="sensor.a", unit="°C")
        assert set(EXTERNAL_FORECAST_UNITS) == {"W", "kW", "MW", "Wh", "kWh", "MWh"}

    def test_yaml_unknown_unit_falls_back_to_sensor(self, tmp_path, caplog):
        # A typo must not stop the whole config loading.
        p = tmp_path / "mlfl.yaml"
        p.write_text(yaml.dump({"experiments": [{
            "name": "e", "target_entity": "sensor.t", "external_forecasts": [
                {"entity_id": "sensor.a", "mode": "attribute", "unit": "kWatts"},
                {"entity_id": "sensor.b", "mode": "attribute", "unit": "kw"},
            ],
        }]}))
        with caplog.at_level(logging.WARNING, logger="ml_forecast_lab.config"):
            exts = load_config(p).experiments[0].external_forecasts
        assert [e.unit for e in exts] == [None, "kW"]
        assert any("using the sensor unit" in r.getMessage() for r in caplog.records)

    def test_add_helper_stores_canonical_unit(self, tmp_path):
        p = tmp_path / "mlfl.yaml"
        p.write_text(yaml.dump({"experiments": [{"name": "e", "target_entity": "sensor.t"}]}))
        assert add_experiment_external_forecast(
            p, "e", {"entity_id": "sensor.s", "mode": "attribute", "unit": "kw"}) is True
        raw = yaml.safe_load(p.read_text())["experiments"][0]["external_forecasts"][0]
        assert raw["unit"] == "kW"
        with pytest.raises(ValueError):
            add_experiment_external_forecast(
                p, "e", {"entity_id": "sensor.other", "unit": "litres"})

    def test_set_unit_by_source_key(self, tmp_path):
        # Two forecasts read from one entity: only the addressed one changes,
        # and clearing removes the key.
        p = tmp_path / "mlfl.yaml"
        base = {"entity_id": "sensor.pv_forecast", "mode": "attribute",
                "attribute": "forecast"}
        p.write_text(yaml.dump({"experiments": [{
            "name": "e", "target_entity": "sensor.t", "external_forecasts": [
                dict(base, value_key="power"),
                dict(base, value_key="power_p10", label="P10"),
            ],
        }]}))
        exts = load_config(p).experiments[0].external_forecasts
        src10 = exts[1].source_key
        assert set_experiment_external_forecast_unit(p, "e", src10, "kw") is True
        exts = load_config(p).experiments[0].external_forecasts
        assert [e.unit for e in exts] == [None, "kW"]
        assert exts[1].label == "P10"   # other fields kept
        assert set_experiment_external_forecast_unit(p, "e", src10, None) is True
        raw = yaml.safe_load(p.read_text())["experiments"][0]["external_forecasts"][1]
        assert "unit" not in raw
        assert set_experiment_external_forecast_unit(p, "e", "sensor.nope", "kW") is False
        assert set_experiment_external_forecast_unit(p, "missing", src10, "kW") is False
        with pytest.raises(ValueError):
            set_experiment_external_forecast_unit(p, "e", src10, "volts")


# ---------------------------------------------------------------------
# The mix-up test
# ---------------------------------------------------------------------

class TestSuspectExternalUnit:
    IH = INTERVAL / 60.0

    def _s(self, ratio, dim="energy", base=1.0, mode="attribute", cum=False,
           explicit=False, ih=None):
        return suspect_external_unit(mode, dim, base, cum, explicit,
                                     self.IH if ih is None else ih, ratio)

    def test_power_entries_on_an_energy_sensor(self):
        # Doubled on a 30-minute grid, give or take a forecast's own bias.
        assert self._s(1.8) == "kW"
        assert self._s(2.3) == "kW"
        assert self._s(2.0, base=0.001) == "W"
        assert self._s(2.0, base=1000.0) == "MW"

    def test_biased_but_correctly_read_forecast_is_not_flagged(self):
        # Below the midpoint sqrt(2) the ratio is nearer 1 than 2.
        for r in (0.5, 0.9, 1.0, 1.2, 1.4):
            assert self._s(r) is None, r

    def test_energy_entries_on_a_power_sensor(self):
        # Energy per interval read as power is scaled by interval_hours.
        assert self._s(0.5, dim="power") == "kWh"
        assert self._s(0.6, dim="power", base=0.001) == "Wh"
        assert self._s(0.75, dim="power") is None
        assert self._s(1.8, dim="power") is None

    def test_interval_length_sets_the_factor(self):
        # 15-minute grid: the mix-up is ×4, midpoint ×2.
        assert self._s(3.6, ih=0.25) == "kW"
        assert self._s(1.9, ih=0.25) is None
        # Hourly grid: both readings coincide, nothing to flag.
        assert self._s(2.0, ih=1.0) is None
        # Two-hour grid: power read as energy now comes out small.
        assert self._s(0.5, ih=2.0) == "kW"

    def test_only_unset_attribute_units_on_non_cumulative_series(self):
        assert self._s(2.0, explicit=True) is None
        assert self._s(2.0, mode="state") is None
        assert self._s(2.0, cum=True) is None
        assert self._s(2.0, dim="other") is None
        assert self._s(2.0, dim="unknown") is None
        assert self._s(None) is None
        assert self._s(0.0) is None


# ---------------------------------------------------------------------
# Comparison end to end
# ---------------------------------------------------------------------

@pytest.fixture
def db(tmp_db):
    h = HistoryDB(tmp_db)
    h.ensure_forecast_log_table()
    h.ensure_external_forecast_log_table()
    yield h
    h.close()


def _seed_pv(db, n=96, ext_factor=1.0):
    """kW target with a daylight curve, a perfect-ish app forecast, and a
    trajectory external whose entries are average power in kW, issued every
    2 h for the next 8 h. ``ext_factor`` scales the external's error-free
    values to model a biased forecast."""
    grid = list(pd.date_range("2024-06-15 00:00", periods=n, freq=f"{INTERVAL}min"))
    power = [max(0.0, math.sin((i % 48 - 12) / 7.5)) * 3.0 for i in range(n)]
    ttbl = db.safe_table_name("sensor.pv_power")
    db.store_history(ttbl, pd.DataFrame({"ds": grid, "value": power}))
    for i, t in enumerate(grid):
        db.log_forecast(experiment="e", issued_at=t - timedelta(minutes=INTERVAL),
                        targets=[t], predictions=[power[i] + 0.1],
                        model_name="lgb", model_version="v1")
    src = "sensor.pv_forecast|forecast|power"
    for k in range(0, n, 4):
        issued = grid[k] - timedelta(minutes=INTERVAL)
        targets = grid[k:k + 16]
        db.log_external_forecast("e", src, issued, targets,
                                 [power[k + j] * ext_factor + 0.05 for j in range(len(targets))])
    return ttbl, src, power


def _spec(src, unit, explicit):
    return {"entity": "sensor.pv_forecast", "source": src, "mode": "attribute",
            "table": None, "scale": None, "is_cumulative": False,
            "label": "PV forecast", "unit": unit, "unit_explicit": explicit}


class TestComparisonUnits:
    def _run(self, db, ttbl, spec):
        return db.get_external_forecast_comparison(
            "e", ttbl, [spec], GENEROUS_WINDOW, INTERVAL, "raw",
            None, None, "per_interval", "kW")

    def test_sensor_unit_kwh_on_kw_entries_is_flagged(self, db):
        ttbl, src, _ = _seed_pv(db)
        res = self._run(db, ttbl, _spec(src, "kWh", False))
        c = res["comparisons"][0]
        # Doubled, but inside the 4× guard: the defect's signature.
        assert 1.8 < c["scale_ratio"] < 2.2, c["scale_ratio"]
        assert c["scale_mismatch"] is False
        assert c["unit_suspect"] is True
        assert c["suggested_unit"] == "kW"
        assert c["unit"] == "kWh" and c["unit_from"] == "sensor"
        assert c["source"] == src
        # Kept out of the matched-lead ranking, with the reason.
        sk = res["skill"]
        assert sk["available"] is False
        assert {"label": "PV forecast", "reason": "unit looks wrong"} in sk["excluded"]

    def test_explicit_kw_scores_like_for_like(self, db):
        ttbl, src, _ = _seed_pv(db)
        res = self._run(db, ttbl, _spec(src, "kW", True))
        c = res["comparisons"][0]
        assert 0.9 < c["scale_ratio"] < 1.1, c["scale_ratio"]
        assert c["unit_suspect"] is False and c["suggested_unit"] is None
        assert c["unit_from"] == "setting"
        # The external's per-interval error is its +0.05 kW offset, not the
        # doubled curve.
        assert abs(c["head_to_head"]["external"]["mae"] - 0.05) < 0.01
        sk = res["skill"]
        assert sk["available"] is True
        assert [r["label"] for r in sk["rows"]] == ["ML Forecast Lab", "PV forecast"]

    def test_explicit_unit_is_trusted_even_when_the_ratio_looks_wrong(self, db):
        # A forecast that really is twice the actuals, with its unit set, is a
        # bad forecast, not a unit problem: scored, not excluded.
        ttbl, src, _ = _seed_pv(db, ext_factor=2.0)
        c = self._run(db, ttbl, _spec(src, "kW", True))["comparisons"][0]
        assert c["scale_ratio"] > 1.8
        assert c["unit_suspect"] is False
        assert c["head_to_head"]["winner"] == "app"

    def test_unchanged_when_the_sensor_unit_is_right(self, db):
        # kW sensor unit, kW entries: same numbers as before, no flag.
        ttbl, src, _ = _seed_pv(db)
        explicit = self._run(db, ttbl, _spec(src, "kW", True))["comparisons"][0]
        sensor = self._run(db, ttbl, _spec(src, "kW", False))["comparisons"][0]
        assert sensor["unit_suspect"] is False
        assert sensor["head_to_head"] == explicit["head_to_head"]
        assert sensor["scale_ratio"] == explicit["scale_ratio"]
