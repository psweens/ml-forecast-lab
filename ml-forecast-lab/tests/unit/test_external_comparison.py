"""
Regression suite for the External Comparison feature (multi-source).

Pins:
  - ExternalForecastCfg / external_forecasts parsing, the legacy flat-key
    migration, the per-experiment cap, and the add/remove helpers.
  - HistoryDB.log_external_forecast (source column, lead computation,
    non-finite skip).
  - HistoryDB.get_external_forecast_comparison across several externals at
    once (state + attribute modes, raw / increment spaces, scale, timing,
    combined lead-time, empty states).

Every test seeds a small, deterministic actuals table + forecast_log
(+ external_forecast_log for attribute mode). No HA, no models.
"""

import math
from datetime import datetime, timedelta

import pandas as pd
import pytest
import yaml

from ml_forecast_lab.config import (
    ExternalForecastCfg, MAX_EXTERNAL_FORECASTS, load_config,
    add_experiment_external_forecast, remove_experiment_external_forecast,
)
from ml_forecast_lab.db import HistoryDB


GENEROUS_WINDOW = 3650
INTERVAL = 30


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

class TestExternalForecastCfg:
    def test_defaults_and_validation(self):
        e = ExternalForecastCfg(entity_id="sensor.a")
        assert e.mode == "state" and e.attribute == "forecast"
        assert e.scale is None and e.is_cumulative is None
        with pytest.raises(ValueError):
            ExternalForecastCfg(entity_id="sensor.a", mode="bogus")
        with pytest.raises(ValueError):
            ExternalForecastCfg(entity_id="")

    def test_list_roundtrip(self, tmp_path):
        p = tmp_path / "mlfl.yaml"
        p.write_text(yaml.dump({"experiments": [{
            "name": "e", "target_entity": "sensor.t", "external_forecasts": [
                {"entity_id": "sensor.a", "mode": "state", "scale": 0.001},
                {"entity_id": "sensor.b", "mode": "attribute",
                 "attribute": "detailedForecast", "value_key": "pv_estimate"},
            ],
        }]}))
        e = load_config(p).experiments[0]
        assert len(e.external_forecasts) == 2
        assert e.external_forecasts[0].entity_id == "sensor.a"
        assert e.external_forecasts[0].scale == 0.001
        assert e.external_forecasts[1].mode == "attribute"
        assert e.external_forecasts[1].value_key == "pv_estimate"

    def test_legacy_scalar_migration(self, tmp_path):
        p = tmp_path / "mlfl.yaml"
        p.write_text(yaml.dump({"experiments": [{
            "name": "e", "target_entity": "sensor.t",
            "external_forecast_entity": "sensor.legacy",
            "external_forecast_mode": "attribute",
            "external_forecast_scale": 2.0,
            "external_forecast_label": "Old",
        }]}))
        e = load_config(p).experiments[0]
        assert len(e.external_forecasts) == 1
        ext = e.external_forecasts[0]
        assert ext.entity_id == "sensor.legacy" and ext.mode == "attribute"
        assert ext.scale == 2.0 and ext.label == "Old"
        # disk migrated to the list form, flat keys gone, and stable on reload
        raw = yaml.safe_load(p.read_text())["experiments"][0]
        assert "external_forecast_entity" not in raw
        assert raw["external_forecasts"][0]["entity_id"] == "sensor.legacy"
        assert len(load_config(p).experiments[0].external_forecasts) == 1

    def test_cap_enforced(self, tmp_path):
        p = tmp_path / "mlfl.yaml"
        p.write_text(yaml.dump({"experiments": [{
            "name": "e", "target_entity": "sensor.t",
            "external_forecasts": [{"entity_id": f"sensor.s{i}"} for i in range(7)],
        }]}))
        e = load_config(p).experiments[0]
        assert len(e.external_forecasts) == MAX_EXTERNAL_FORECASTS

    def test_add_remove_helpers(self, tmp_path):
        p = tmp_path / "mlfl.yaml"
        p.write_text(yaml.dump({"experiments": [{"name": "e", "target_entity": "sensor.t"}]}))
        assert add_experiment_external_forecast(p, "e", {"entity_id": "sensor.x", "mode": "state"}) is True
        assert add_experiment_external_forecast(p, "e", {"entity_id": "sensor.x"}) is False  # dup
        for i in range(4):
            assert add_experiment_external_forecast(p, "e", {"entity_id": f"sensor.y{i}"}) is True
        assert add_experiment_external_forecast(p, "e", {"entity_id": "sensor.z"}) is False  # cap
        assert len(load_config(p).experiments[0].external_forecasts) == MAX_EXTERNAL_FORECASTS
        assert remove_experiment_external_forecast(p, "e", "sensor.x") is True
        assert len(load_config(p).experiments[0].external_forecasts) == 4
        with pytest.raises(ValueError):
            add_experiment_external_forecast(p, "e", {"entity_id": "sensor.bad", "mode": "nope"})


# ---------------------------------------------------------------------
# DB fixtures & helpers
# ---------------------------------------------------------------------

@pytest.fixture
def db(tmp_db):
    h = HistoryDB(tmp_db)
    h.ensure_forecast_log_table()
    h.ensure_external_forecast_log_table()
    yield h
    h.close()


def _grid(n=48, start="2024-06-15 00:00"):
    return list(pd.date_range(start, periods=n, freq=f"{INTERVAL}min"))


def _actual_curve(n=48):
    return [100.0 + 50.0 * math.sin(i / 4.0) + 60.0 for i in range(n)]


def _spec(entity, mode="state", table=None, scale=None, is_cum=False, label=None):
    return {"entity": entity, "mode": mode, "table": table, "scale": scale,
            "is_cumulative": is_cum, "label": label or entity.split(".")[-1]}


# ---------------------------------------------------------------------
# log_external_forecast
# ---------------------------------------------------------------------

class TestLogExternalForecast:
    def test_source_lead_and_count(self, db):
        issued = datetime(2024, 6, 15, 0, 0, 0)
        targets = [issued + timedelta(minutes=INTERVAL * (i + 1)) for i in range(4)]
        n = db.log_external_forecast("e", "sensor.solcast", issued, targets, [1.0, 2.0, 3.0, 4.0])
        assert n == 4
        cur = db.conn.cursor()
        cur.execute("SELECT source, lead_minutes, value FROM external_forecast_log ORDER BY lead_minutes")
        rows = cur.fetchall()
        assert all(r[0] == "sensor.solcast" for r in rows)
        assert [r[1] for r in rows] == [30, 60, 90, 120]

    def test_skips_non_finite(self, db):
        issued = datetime(2024, 6, 15, 0, 0, 0)
        targets = [issued + timedelta(minutes=INTERVAL * (i + 1)) for i in range(3)]
        assert db.log_external_forecast("e", "s", issued, targets, [1.0, float("nan"), None]) == 1


# ---------------------------------------------------------------------
# get_external_forecast_comparison (multi-source)
# ---------------------------------------------------------------------

class TestComparisonMulti:
    def _seed(self, db):
        grid = _grid()
        actual = _actual_curve()
        ttbl = db.safe_table_name("sensor.load_w")
        db.store_history(ttbl, pd.DataFrame({"ds": grid, "value": actual}))
        # app: latest per target, lead 30, MAE ~4
        for i, t in enumerate(grid):
            db.log_forecast(experiment="e", issued_at=t - timedelta(minutes=INTERVAL),
                            targets=[t], predictions=[actual[i] + 4.0],
                            model_name="lgb", model_version="v1")
        # external 1 (state): MAE ~12 (worse than app)
        e1 = db.safe_table_name("sensor.ext_state")
        db.store_history(e1, pd.DataFrame({"ds": grid, "value": [actual[i] + 12.0 for i in range(48)]}))
        # external 2 (attribute trajectory): MAE ~2 (better than app)
        for issue_idx in range(0, 48, 4):
            issued = grid[issue_idx] - timedelta(minutes=INTERVAL)
            targets = grid[issue_idx:issue_idx + 8]
            if targets:
                db.log_external_forecast("e", "sensor.solcast", issued, targets,
                                         [actual[issue_idx + k] + 2.0 for k in range(len(targets))])
        return ttbl, e1, actual

    def test_two_externals_ranked(self, db):
        ttbl, e1, _ = self._seed(db)
        specs = [
            _spec("sensor.ext_state", "state", e1, label="Crude"),
            _spec("sensor.solcast", "attribute", None, label="Solcast"),
        ]
        res = db.get_external_forecast_comparison("e", ttbl, specs, GENEROUS_WINDOW, INTERVAL, "raw")
        assert res["configured"] is True
        assert len(res["comparisons"]) == 2
        assert len(res["overlay"]["externals"]) == 2
        by = {c["label"]: c for c in res["comparisons"]}
        assert by["Crude"]["head_to_head"]["winner"] == "app"          # 12 vs 4
        assert by["Solcast"]["head_to_head"]["winner"] == "external"   # 2 vs 4
        assert abs(by["Crude"]["head_to_head"]["external"]["mae"] - 12.0) < 1.0
        assert abs(by["Solcast"]["head_to_head"]["external"]["mae"] - 2.0) < 0.5
        # timing: state contemporaneous, attribute lead-matched
        assert by["Crude"]["timing"]["external_contemporaneous"] is True
        assert by["Solcast"]["timing"]["external_contemporaneous"] is False
        assert by["Solcast"]["timing"]["external_median_lead_minutes"] is not None
        # combined lead-time has only the attribute external + the app curve
        lt = res["lead_time"]
        assert lt is not None and len(lt["externals"]) == 1
        assert lt["externals"][0]["label"] == "Solcast"
        assert lt["app_mae"] and lt["externals"][0]["mae"]
        # overlay arrays all aligned to the shared ds
        n = len(res["overlay"]["ds"])
        assert len(res["overlay"]["actual"]) == n and len(res["overlay"]["app"]) == n
        assert all(len(x["values"]) == n for x in res["overlay"]["externals"])

    def test_scale_applied(self, db):
        grid = _grid(); actual = _actual_curve()
        ttbl = db.safe_table_name("sensor.load_w")
        db.store_history(ttbl, pd.DataFrame({"ds": grid, "value": actual}))
        for i, t in enumerate(grid):
            db.log_forecast(experiment="e", issued_at=t - timedelta(minutes=INTERVAL),
                            targets=[t], predictions=[actual[i]], model_name="lgb", model_version="v1")
        e1 = db.safe_table_name("sensor.ext_milli")
        db.store_history(e1, pd.DataFrame({"ds": grid, "value": [(actual[i] + 10.0) * 1000.0 for i in range(48)]}))
        res = db.get_external_forecast_comparison(
            "e", ttbl, [_spec("sensor.ext_milli", "state", e1, scale=0.001)],
            GENEROUS_WINDOW, INTERVAL, "raw")
        assert abs(res["comparisons"][0]["head_to_head"]["external"]["mae"] - 10.0) < 1.0

    def test_increment_cumulative_external(self, db):
        grid = _grid(); actual = _actual_curve()
        cum, s = [], 0.0
        for v in actual:
            s += v / 100.0
            cum.append(s)
        ttbl = db.safe_table_name("sensor.energy_today")
        db.store_history(ttbl, pd.DataFrame({"ds": grid, "value": cum}))
        for i, t in enumerate(grid):
            delta = cum[i] - (cum[i - 1] if i > 0 else 0.0)
            db.log_forecast(experiment="e", issued_at=t - timedelta(minutes=INTERVAL),
                            targets=[t], predictions=[delta + 0.01], model_name="lgb", model_version="v1")
        e1 = db.safe_table_name("sensor.ext_energy")
        db.store_history(e1, pd.DataFrame({"ds": grid, "value": [cum[i] + 0.2 * (i + 1) for i in range(48)]}))
        res = db.get_external_forecast_comparison(
            "e", ttbl, [_spec("sensor.ext_energy", "state", e1, is_cum=True)],
            GENEROUS_WINDOW, INTERVAL, "increment")
        h = res["comparisons"][0]["head_to_head"]
        assert h is not None and h["app"]["mae"] < h["external"]["mae"]
        assert h["winner"] == "app"

    def test_scale_mismatch_flagged(self, db):
        # External on a ~15x larger scale than the target (e.g. a cumulative
        # kWh sensor vs instantaneous power) must be flagged, not silently
        # declared "96% worse".
        grid = _grid(); actual = _actual_curve()
        ttbl = db.safe_table_name("sensor.pv_power")
        db.store_history(ttbl, pd.DataFrame({"ds": grid, "value": actual}))
        for i, t in enumerate(grid):
            db.log_forecast(experiment="e", issued_at=t - timedelta(minutes=INTERVAL),
                            targets=[t], predictions=[actual[i]], model_name="lgb", model_version="v1")
        e1 = db.safe_table_name("sensor.pv_today")
        db.store_history(e1, pd.DataFrame({"ds": grid, "value": [actual[i] * 15.0 for i in range(48)]}))
        res = db.get_external_forecast_comparison(
            "e", ttbl, [_spec("sensor.pv_today", "state", e1)],
            GENEROUS_WINDOW, INTERVAL, "raw")
        c = res["comparisons"][0]
        assert c["scale_mismatch"] is True
        assert c["scale_ratio"] is not None and c["scale_ratio"] > 4.0

    def test_comparable_scale_not_flagged(self, db):
        grid = _grid(); actual = _actual_curve()
        ttbl = db.safe_table_name("sensor.pv_power")
        db.store_history(ttbl, pd.DataFrame({"ds": grid, "value": actual}))
        for i, t in enumerate(grid):
            db.log_forecast(experiment="e", issued_at=t - timedelta(minutes=INTERVAL),
                            targets=[t], predictions=[actual[i]], model_name="lgb", model_version="v1")
        e1 = db.safe_table_name("sensor.other_pv")
        db.store_history(e1, pd.DataFrame({"ds": grid, "value": [actual[i] + 0.5 for i in range(48)]}))
        res = db.get_external_forecast_comparison(
            "e", ttbl, [_spec("sensor.other_pv", "state", e1)],
            GENEROUS_WINDOW, INTERVAL, "raw")
        assert res["comparisons"][0]["scale_mismatch"] is False

    def test_delete_source(self, db):
        ttbl, e1, _ = self._seed(db)
        assert db.delete_external_forecast_source("e", "sensor.solcast") > 0
        cur = db.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM external_forecast_log WHERE source='sensor.solcast'")
        assert cur.fetchone()[0] == 0


class TestComparisonEmptyStates:
    def test_missing_actuals_table(self, db):
        res = db.get_external_forecast_comparison(
            "e", db.safe_table_name("sensor.nope"),
            [_spec("sensor.x", "attribute", None)], GENEROUS_WINDOW, INTERVAL, "raw")
        assert res.get("empty_reason") == "no_actuals"
        assert res["comparisons"] == []

    def test_no_forecasts_yet(self, db):
        grid = _grid()
        ttbl = db.safe_table_name("sensor.load_w")
        db.store_history(ttbl, pd.DataFrame({"ds": grid, "value": _actual_curve()}))
        res = db.get_external_forecast_comparison(
            "e", ttbl, [_spec("sensor.ext", "state", db.safe_table_name("sensor.ext"))],
            GENEROUS_WINDOW, INTERVAL, "raw")
        # actual exists but no app/external rows → head_to_head is None
        assert res["comparisons"][0]["head_to_head"] is None
        assert res["comparisons"][0]["n"] == 0
