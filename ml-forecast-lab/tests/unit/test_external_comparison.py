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

    def test_same_entity_multiple_forecasts(self, tmp_path):
        # The same entity can be added more than once with different
        # attribute / value_key (e.g. Solcast percentiles), but an exact
        # duplicate is still rejected — and each has a distinct source key.
        p = tmp_path / "mlfl.yaml"
        p.write_text(yaml.dump({"experiments": [{"name": "e", "target_entity": "sensor.t"}]}))
        base = {"entity_id": "sensor.solcast", "mode": "attribute", "attribute": "detailedForecast"}
        assert add_experiment_external_forecast(p, "e", dict(base, value_key="pv_estimate")) is True
        assert add_experiment_external_forecast(p, "e", dict(base, value_key="pv_estimate10")) is True
        assert add_experiment_external_forecast(p, "e", dict(base, value_key="pv_estimate90")) is True
        # exact duplicate rejected
        assert add_experiment_external_forecast(p, "e", dict(base, value_key="pv_estimate")) is False
        exts = load_config(p).experiments[0].external_forecasts
        assert len(exts) == 3
        keys = {x.source_key for x in exts}
        assert len(keys) == 3   # all distinct
        # remove just the 10% one, by full identity
        assert remove_experiment_external_forecast(
            p, "e", "sensor.solcast", mode="attribute",
            attribute="detailedForecast", value_key="pv_estimate10") is True
        left = load_config(p).experiments[0].external_forecasts
        assert len(left) == 2
        assert all(x.value_key != "pv_estimate10" for x in left)

    def test_source_key_distinguishes_attribute_forecasts(self):
        a = ExternalForecastCfg(entity_id="sensor.s", mode="attribute",
                                attribute="detailedForecast", value_key="pv_estimate")
        b = ExternalForecastCfg(entity_id="sensor.s", mode="attribute",
                                attribute="detailedForecast", value_key="pv_estimate90")
        c = ExternalForecastCfg(entity_id="sensor.s", mode="state")
        assert a.source_key != b.source_key
        assert c.source_key == "sensor.s"


# ---------------------------------------------------------------------
# DB fixtures & helpers
# ---------------------------------------------------------------------

class TestExternalRetentionSetting:
    """v2.44.x: external_forecast_retention_days (global, System tab)."""

    def test_default_is_60(self, tmp_path):
        p = tmp_path / "mlfl.yaml"
        p.write_text(yaml.dump({"experiments": [
            {"name": "e", "target_entity": "sensor.t"}]}))
        assert load_config(p).external_forecast_retention_days == 60

    def test_reads_from_yaml(self, tmp_path):
        p = tmp_path / "mlfl.yaml"
        p.write_text(yaml.dump({
            "external_forecast_retention_days": 30,
            "experiments": [{"name": "e", "target_entity": "sensor.t"}],
        }))
        assert load_config(p).external_forecast_retention_days == 30

    def test_rejects_below_one(self):
        from ml_forecast_lab.config import AppConfig
        with pytest.raises(ValueError):
            AppConfig(external_forecast_retention_days=0,
                      experiments=[ExternalForecastCfg.__new__(ExternalForecastCfg)])


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

    def test_last_trajectory_returns_latest_issuance(self, db):
        # get_last_external_trajectory backs content-change detection: it must
        # return only the most-recently-issued snapshot's {target: value}.
        t0 = datetime(2024, 6, 15, 6, 0, 0)
        targets = [t0 + timedelta(minutes=INTERVAL * (i + 1)) for i in range(3)]
        db.log_external_forecast("e", "sensor.s", t0, targets, [1.0, 2.0, 3.0])
        t1 = datetime(2024, 6, 15, 8, 24, 0)   # later issuance, changed values
        db.log_external_forecast("e", "sensor.s", t1, targets, [1.5, 2.5, 3.5])
        traj = db.get_last_external_trajectory("e", "sensor.s")
        assert len(traj) == 3
        assert sorted(traj.values()) == [1.5, 2.5, 3.5]   # latest issuance only
        assert db.get_last_external_trajectory("e", "missing") == {}

    def test_last_issued_at_drives_dedup(self, db):
        # None before anything is logged.
        assert db.get_last_external_issued_at("e", "sensor.solcast") is None
        i1 = datetime(2024, 6, 15, 8, 0, 0)
        i2 = datetime(2024, 6, 15, 10, 0, 0)
        targets1 = [i1 + timedelta(minutes=INTERVAL * (k + 1)) for k in range(3)]
        targets2 = [i2 + timedelta(minutes=INTERVAL * (k + 1)) for k in range(3)]
        db.log_external_forecast("e", "sensor.solcast", i1, targets1, [1.0, 2.0, 3.0])
        assert db.get_last_external_issued_at("e", "sensor.solcast") == i1
        db.log_external_forecast("e", "sensor.solcast", i2, targets2, [1.0, 2.0, 3.0])
        # Tracks the most recent issue (the de-dup high-water mark).
        assert db.get_last_external_issued_at("e", "sensor.solcast") == i2
        # Per-source, not cross-contaminated.
        assert db.get_last_external_issued_at("e", "sensor.other") is None

    def test_lead_reflects_source_issue_time_not_capture(self, db):
        # A trajectory stamped with a 2h-old issue time yields ~2h+ leads,
        # not the ~15 min you'd get from a capture-time stamp. This is the
        # whole point of using the source's last_updated as issued_at.
        issued_2h_ago = datetime(2024, 6, 15, 8, 0, 0)
        # Targets are "now" (10:00) onward — 2h+ after the source issued.
        targets = [datetime(2024, 6, 15, 10, 0, 0) + timedelta(minutes=INTERVAL * k)
                   for k in range(3)]
        db.log_external_forecast("e", "sensor.solcast", issued_2h_ago, targets, [1.0, 2.0, 3.0])
        cur = db.conn.cursor()
        cur.execute("SELECT MIN(lead_minutes) FROM external_forecast_log WHERE source='sensor.solcast'")
        assert cur.fetchone()[0] == 120  # 10:00 − 08:00 = 2h, not ~15 min


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

    def test_unit_aware_power_target_vs_cumulative_energy(self, db):
        # Target = instantaneous power (kW); external = cumulative daily
        # energy (kWh) that is the true integral of the power. Unit-aware
        # conversion must make them line up (no scale mismatch) in BOTH the
        # per-interval (kW) and cumulative (kWh) views, and auto-detect the
        # cumulative shape.
        ih = INTERVAL / 60.0
        grid = _grid()
        power = [max(0.0, math.sin((i - 10) / 6.0)) * 3.0 for i in range(48)]
        ttbl = db.safe_table_name("sensor.pv_power")
        db.store_history(ttbl, pd.DataFrame({"ds": grid, "value": power}))
        for i, t in enumerate(grid):
            db.log_forecast(experiment="e", issued_at=t - timedelta(minutes=INTERVAL),
                            targets=[t], predictions=[power[i]], model_name="lgb", model_version="v1")
        cum, s = [], 0.0
        for i in range(48):
            s += power[i] * ih
            cum.append(s)
        e1 = db.safe_table_name("sensor.pv_today")
        db.store_history(e1, pd.DataFrame({"ds": grid, "value": cum}))
        spec = {"entity": "sensor.pv_today", "mode": "state", "table": e1,
                "scale": None, "is_cumulative": None, "label": "PV Today", "unit": "kWh"}

        # Per-interval view (target unit kW).
        res = db.get_external_forecast_comparison(
            "e", ttbl, [spec], GENEROUS_WINDOW, INTERVAL, "raw",
            None, None, "per_interval", "kW")
        c = res["comparisons"][0]
        assert res["unit_aware"] is True
        assert c["auto_cumulative"] is True          # detected cumulative shape
        assert c["scale_mismatch"] is False, c["scale_ratio"]
        assert c["head_to_head"]["external"]["mae"] < 0.2   # ≈ the power curve
        assert res["display_unit"] == "kW"

        # Cumulative view → both in kWh, still aligned.
        res2 = db.get_external_forecast_comparison(
            "e", ttbl, [spec], GENEROUS_WINDOW, INTERVAL, "raw",
            None, None, "cumulative", "kW")
        c2 = res2["comparisons"][0]
        assert res2["display_unit"] == "kWh"
        assert c2["scale_mismatch"] is False
        assert c2["head_to_head"]["external"]["mae"] < 0.6

    def test_unit_aware_base_scale_wh_vs_kw(self, db):
        # External in Wh (cumulative) vs kW target — base-unit scaling (Wh→kWh)
        # must be handled automatically.
        ih = INTERVAL / 60.0
        grid = _grid()
        power = [max(0.0, math.sin((i - 10) / 6.0)) * 2.0 for i in range(48)]
        ttbl = db.safe_table_name("sensor.p2")
        db.store_history(ttbl, pd.DataFrame({"ds": grid, "value": power}))
        for i, t in enumerate(grid):
            db.log_forecast(experiment="e", issued_at=t - timedelta(minutes=INTERVAL),
                            targets=[t], predictions=[power[i]], model_name="lgb", model_version="v1")
        cum, s = [], 0.0
        for i in range(48):
            s += power[i] * ih * 1000.0   # Wh
            cum.append(s)
        e1 = db.safe_table_name("sensor.e_wh")
        db.store_history(e1, pd.DataFrame({"ds": grid, "value": cum}))
        spec = {"entity": "sensor.e_wh", "mode": "state", "table": e1,
                "scale": None, "is_cumulative": None, "label": "Wh", "unit": "Wh"}
        res = db.get_external_forecast_comparison(
            "e", ttbl, [spec], GENEROUS_WINDOW, INTERVAL, "raw",
            None, None, "per_interval", "kW")
        c = res["comparisons"][0]
        assert c["scale_mismatch"] is False, c["scale_ratio"]
        assert c["head_to_head"]["external"]["mae"] < 0.2

    def test_same_entity_two_sources_scored_independently(self, db):
        # Two attribute forecasts from ONE entity (distinct source keys) must
        # be scored as separate comparisons.
        grid = _grid(); actual = _actual_curve()
        ttbl = db.safe_table_name("sensor.pv")
        db.store_history(ttbl, pd.DataFrame({"ds": grid, "value": actual}))
        for i, t in enumerate(grid):
            db.log_forecast(experiment="e", issued_at=t - timedelta(minutes=INTERVAL),
                            targets=[t], predictions=[actual[i] + 4.0], model_name="lgb", model_version="v1")
        for issue_idx in range(0, 48, 4):
            issued = grid[issue_idx] - timedelta(minutes=INTERVAL)
            targets = grid[issue_idx:issue_idx + 8]
            if not targets:
                continue
            db.log_external_forecast("e", "sensor.s|detailedForecast|pv_estimate", issued, targets,
                                     [actual[issue_idx + k] + 2.0 for k in range(len(targets))])
            db.log_external_forecast("e", "sensor.s|detailedForecast|pv_estimate90", issued, targets,
                                     [actual[issue_idx + k] + 9.0 for k in range(len(targets))])
        specs = [
            {"entity": "sensor.s", "source": "sensor.s|detailedForecast|pv_estimate",
             "mode": "attribute", "table": None, "scale": None, "is_cumulative": False, "label": "p50"},
            {"entity": "sensor.s", "source": "sensor.s|detailedForecast|pv_estimate90",
             "mode": "attribute", "table": None, "scale": None, "is_cumulative": False, "label": "p90"},
        ]
        res = db.get_external_forecast_comparison("e", ttbl, specs, GENEROUS_WINDOW, INTERVAL, "raw")
        assert len(res["comparisons"]) == 2
        by = {c["label"]: c for c in res["comparisons"]}
        assert abs(by["p50"]["head_to_head"]["external"]["mae"] - 2.0) < 0.5
        assert abs(by["p90"]["head_to_head"]["external"]["mae"] - 9.0) < 0.5

    def test_delete_source(self, db):
        ttbl, e1, _ = self._seed(db)
        assert db.delete_external_forecast_source("e", "sensor.solcast") > 0
        cur = db.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM external_forecast_log WHERE source='sensor.solcast'")
        assert cur.fetchone()[0] == 0

    def test_actual_carried_forward_to_forecast_horizon(self, db):
        # HA's recorder dedups unchanged states, so a cumulative daily total
        # whose generation has stopped (a PV "energy today" sensor on a cloudy
        # afternoon) writes no new rows — its cached series ends at the last
        # change, mid-window. The forecast is logged live and runs further. The
        # overlay must hold the last actual flat up to the forecast horizon so
        # the actual line doesn't "just stop" mid-day while the forecast
        # continues; in cumulative view that means the running total stays flat.
        grid = _grid(48)
        # Cumulative daily energy that plateaus after bin 40 (generation stops).
        cum, s = [], 0.0
        for i in range(48):
            s += (1.0 if i <= 40 else 0.0)        # flat from bin 41 on
            cum.append(s)
        ttbl = db.safe_table_name("sensor.pv_today_kwh")
        # The recorder only logs while the value changes → cache stops at bin 40.
        db.store_history(ttbl, pd.DataFrame({"ds": grid[:41], "value": cum[:41]}))
        # App forecast is logged live for the whole window, out to bin 47.
        for i, t in enumerate(grid):
            delta = cum[i] - (cum[i - 1] if i > 0 else 0.0)
            db.log_forecast(experiment="e", issued_at=t - timedelta(minutes=INTERVAL),
                            targets=[t], predictions=[delta + 0.01],
                            model_name="lgb", model_version="v1")
        specs = [_spec("sensor.dummy", "state", db.safe_table_name("sensor.dummy"))]
        res = db.get_external_forecast_comparison(
            "e", ttbl, specs, GENEROUS_WINDOW, INTERVAL, "increment",
            None, None, "cumulative", "kWh")
        ds = res["overlay"]["ds"]
        actual = res["overlay"]["actual"]
        app = res["overlay"]["app"]
        last = lambda a: max(i for i, v in enumerate(a) if v is not None)
        # Actual now reaches the same horizon as the live forecast (bin 47),
        # not the last recorder write (bin 40).
        assert last(actual) == last(app)
        # The carried tail holds flat at the value of the LAST REAL bin (40),
        # not merely flat at whatever the tail happens to hold — the original
        # assertion compared actual[last] against itself and passed on any
        # value at all. Anchor on the display-space value at bin 40, since in
        # cumulative view the plotted series is re-derived, not the raw total.
        last_real = float(actual[40])
        carried = [actual[i] for i in range(41, len(actual)) if actual[i] is not None]
        assert carried, "expected a carried tail past the last recorder write"
        assert all(abs(v - last_real) < 1e-6 for v in carried), (
            f"carried tail should hold {last_real}, got {carried[:5]}"
        )

    def test_carried_points_are_display_only_and_never_scored(self, db):
        """The held values are inferred, not observed. They belong on the chart
        and nowhere else.

        If they reach the scoring path the add-on ends up grading its own
        forecast against a flat line it invented — and because the held points
        are non-NaN they also defeat the dropna() filters that previously
        excluded exactly those bins. On a sensor that has genuinely died, that
        is a leaderboard computed almost entirely from fabricated actuals.
        """
        grid = _grid(48)
        cum, s = [], 0.0
        for i in range(48):
            s += (1.0 if i <= 40 else 0.0)
            cum.append(s)
        ttbl = db.safe_table_name("sensor.pv_today_kwh")
        db.store_history(ttbl, pd.DataFrame({"ds": grid[:41], "value": cum[:41]}))
        for i, t_ in enumerate(grid):
            delta = cum[i] - (cum[i - 1] if i > 0 else 0.0)
            db.log_forecast(experiment="e", issued_at=t_ - timedelta(minutes=INTERVAL),
                            targets=[t_], predictions=[delta + 0.01],
                            model_name="lgb", model_version="v1")
        specs = [_spec("sensor.dummy", "state", db.safe_table_name("sensor.dummy"))]
        res = db.get_external_forecast_comparison(
            "e", ttbl, specs, GENEROUS_WINDOW, INTERVAL, "increment",
            None, None, "cumulative", "kWh")

        # The overlay is extended...
        actual = res["overlay"]["actual"]
        last = lambda a: max(i for i, v in enumerate(a) if v is not None)
        assert last(actual) == last(res["overlay"]["app"])

        # ...but scoring saw only the 41 real bins, not the 48 displayed ones.
        scored_n = (res.get("app_self") or {}).get("n")
        assert scored_n is not None
        assert scored_n <= 41, (
            f"scored {scored_n} bins from 41 real readings — carried points "
            f"leaked into the metrics"
        )

    def test_actual_not_extended_when_already_current(self, db):
        # When the actuals already reach the forecast horizon, carry-forward is
        # a no-op: it must not fabricate a flat tail beyond the real data, and
        # in particular must not extend a stale series across the (huge) window.
        ttbl, e1, _ = self._seed(db)   # actuals + app share the same 48-bin grid
        specs = [_spec("sensor.ext_state", "state", e1, label="Crude")]
        res = db.get_external_forecast_comparison(
            "e", ttbl, specs, GENEROUS_WINDOW, INTERVAL, "raw")
        actual = res["overlay"]["actual"]
        app = res["overlay"]["app"]
        last = lambda a: max(i for i, v in enumerate(a) if v is not None)
        assert last(actual) == last(app)            # no tail past the forecast
        assert len(res["overlay"]["ds"]) <= 48 + 1  # window not blown up


class TestComparisonBaselineAndWarmup:
    """v2.44.x additions: %-of-typical + daily metrics, and the 7-day
    warming-up gate."""

    def _seed_days(self, db, n_days=10, interval=INTERVAL, mae_app=4.0, mae_ext=12.0):
        """Seed n_days of a repeating diurnal actual + app forecast (lead 1)
        + one state-mode external. Returns (ttbl, e1_table, n_days)."""
        per_day = 24 * 60 // interval
        grid = list(pd.date_range("2024-06-01 00:00", periods=per_day * n_days,
                                   freq=f"{interval}min"))
        # Repeating daily shape so 'same time yesterday' is a good naive.
        actual = [100.0 + 50.0 * math.sin((i % per_day) / 4.0) + 60.0 for i in range(len(grid))]
        ttbl = db.safe_table_name("sensor.load_w")
        db.store_history(ttbl, pd.DataFrame({"ds": grid, "value": actual}))
        for i, t in enumerate(grid):
            db.log_forecast(experiment="e", issued_at=t - timedelta(minutes=interval),
                            targets=[t], predictions=[actual[i] + mae_app],
                            model_name="lgb", model_version="v1")
        e1 = db.safe_table_name("sensor.ext_state")
        db.store_history(e1, pd.DataFrame({"ds": grid,
                                           "value": [actual[i] + mae_ext for i in range(len(grid))]}))
        return ttbl, e1, n_days

    def test_no_seasonal_naive_baseline_emitted(self, db):
        # Seasonal Naive was removed (v2.44.5): the response must not carry a
        # baseline row or overlay line, and the only ranked row is the one
        # configured external (plus the app reference, which is separate).
        ttbl, e1, _ = self._seed_days(db, n_days=10)
        res = db.get_external_forecast_comparison(
            "e", ttbl, [_spec("sensor.ext_state", "state", e1, label="Crude")],
            GENEROUS_WINDOW, INTERVAL, "raw")
        assert "baseline" not in res
        assert "baseline" not in res["overlay"]
        assert len(res["comparisons"]) == 1

    def test_corrupt_forecast_value_is_dropped(self, db):
        # A log-transform inversion overflow (~1e30) logged among normal
        # forecasts must be dropped — otherwise one point dwarfs every real
        # value, flattens the charts and explodes the MAE/ranking.
        ttbl, e1, _ = self._seed_days(db, n_days=8)
        bad_t = pd.Timestamp("2024-06-03 12:00")
        # Simulate a HISTORICAL corrupt row written before the log_forecast
        # write-guard existed. The guard now drops such a value at write time,
        # so insert directly to bypass it — the read-side corruption filter's
        # remaining job is exactly cleaning up legacy rows like this. Stamped
        # LATER than the seeded row for that grid so, without the read filter,
        # groupby-last would pick the corrupt value.
        _ts = bad_t.strftime("%Y-%m-%d %H:%M:%S")
        _cur = db.conn.cursor()
        _cur.execute(
            "INSERT INTO forecast_log (experiment, model_name, issued_at, "
            "target_dt, lead_minutes, predicted, forecast_type, upper, lower, "
            "model_version) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("e", "lgb", _ts, _ts, 0, 5e30, "cached", None, None, "v1"),
        )
        db.conn.commit()
        res = db.get_external_forecast_comparison(
            "e", ttbl, [_spec("sensor.ext_state", "state", e1, label="Crude")],
            GENEROUS_WINDOW, INTERVAL, "raw")
        # App metrics stay finite and physical (the 5e30 point is gone).
        assert res["app_self"] is not None
        assert math.isfinite(res["app_self"]["mae"])
        assert res["app_self"]["mae"] < 1e6
        # The overlay app line carries no absurd value either.
        app_overlay = [v for v in res["overlay"]["app"] if v is not None]
        assert app_overlay and all(abs(v) < 1e9 for v in app_overlay)
        # ...but the blowup is SURFACED (not silently hidden) so the UI can
        # flag it — the latest-per-target / h=1 views would otherwise mask it.
        assert "corrupt" in res
        assert res["corrupt"]["app"]["count"] == 1
        assert res["corrupt"]["app"]["max_value"] >= 5e30 - 1
        assert res["corrupt"]["app"]["last_ts"]

    def test_pct_of_typical_and_daily_metrics(self, db):
        ttbl, e1, _ = self._seed_days(db, n_days=8, mae_app=4.0, mae_ext=12.0)
        res = db.get_external_forecast_comparison(
            "e", ttbl, [_spec("sensor.ext_state", "state", e1, label="Crude")],
            GENEROUS_WINDOW, INTERVAL, "raw")
        c = res["comparisons"][0]
        # %-of-typical present on head-to-head and on the standalone metrics
        assert c["head_to_head"]["app"]["pct"] is not None
        assert c["head_to_head"]["external"]["pct"] is not None
        assert c["metrics"]["pct"] is not None
        # daily MAE present and external (+12/bin) much worse than app (+4/bin)
        assert c["head_to_head"]["daily"]["app"]["mae"] < c["head_to_head"]["daily"]["external"]["mae"]
        assert res["typical"] > 0
        # app reference row carries the same metric family
        assert res["app_self"] is not None
        assert res["app_self"]["pct"] is not None
        assert res["app_self"]["daily"] is not None

    def test_warming_up_flag_below_threshold(self, db):
        # 4 distinct days < 7-day threshold → warming.
        ttbl, e1, _ = self._seed_days(db, n_days=4)
        res = db.get_external_forecast_comparison(
            "e", ttbl, [_spec("sensor.ext_state", "state", e1)],
            GENEROUS_WINDOW, INTERVAL, "raw")
        assert res["warmup_days"] == 7
        assert res["warming_up"] is True
        assert res["comparisons"][0]["warming"] is True
        assert res["comparisons"][0]["days_logged"] == 4
        assert res["app_days_logged"] == 4

    def test_not_warming_above_threshold(self, db):
        # 9 distinct days ≥ 7 → not warming.
        ttbl, e1, _ = self._seed_days(db, n_days=9)
        res = db.get_external_forecast_comparison(
            "e", ttbl, [_spec("sensor.ext_state", "state", e1)],
            GENEROUS_WINDOW, INTERVAL, "raw")
        assert res["warming_up"] is False
        assert res["comparisons"][0]["warming"] is False
        assert res["comparisons"][0]["days_logged"] == 9


class TestComparisonSkill:
    """v2.46: 'Same lead time' (skill) scoring — match forecasters at a common
    lead band so update frequency doesn't masquerade as accuracy."""

    def _seed_multilead(self, db, exp="e"):
        grid = _grid(); actual = _actual_curve()
        ttbl = db.safe_table_name("sensor.pv")
        db.store_history(ttbl, pd.DataFrame({"ds": grid, "value": actual}))
        for issue_idx in range(0, 48, 4):
            issued = grid[issue_idx] - timedelta(minutes=INTERVAL)
            targets = grid[issue_idx:issue_idx + 8]
            if not targets:
                continue
            db.log_forecast(experiment=exp, issued_at=issued, targets=targets,
                            predictions=[actual[issue_idx + k] + 3.0 for k in range(len(targets))],
                            model_name="lgb", model_version="v1")
            db.log_external_forecast(exp, "sensor.solcast", issued, targets,
                                     [actual[issue_idx + k] + 10.0 for k in range(len(targets))])
        return ttbl, actual, grid

    def test_skill_band_ranking_and_exclusions(self, db):
        ttbl, actual, grid = self._seed_multilead(db)
        e_state = db.safe_table_name("sensor.crude")
        db.store_history(e_state, pd.DataFrame({"ds": grid, "value": [actual[i] + 5.0 for i in range(48)]}))
        specs = [
            {"entity": "sensor.solcast", "source": "sensor.solcast", "mode": "attribute",
             "table": None, "scale": None, "is_cumulative": False, "label": "Solcast"},
            _spec("sensor.crude", "state", e_state, label="Crude"),
        ]
        res = db.get_external_forecast_comparison("e", ttbl, specs, GENEROUS_WINDOW, INTERVAL, "raw")
        sk = res["skill"]
        assert sk and sk["available"] is True, sk
        lo, hi = sk["lead_band_minutes"]
        assert 0 <= lo <= hi
        by = {r["label"]: r for r in sk["rows"]}
        assert "ML Forecast Lab" in by and "Solcast" in by
        # At matched lead the app (offset 3) beats the external (offset 10).
        assert by["ML Forecast Lab"]["mae"] < by["Solcast"]["mae"]
        # Full metric set is present at matched lead (not just MAE).
        for r in sk["rows"]:
            for k in ("mae", "rmse", "bias", "pct", "n"):
                assert k in r, k
            assert r["rmse"] >= r["mae"]              # RMSE ≥ MAE always
        # Constant-offset forecasts → bias ≈ the offset, RMSE ≈ MAE.
        assert abs(by["Solcast"]["bias"] - 10.0) < 0.5
        assert abs(by["ML Forecast Lab"]["bias"] - 3.0) < 0.5
        # The state-mode source can't do equal-lead → excluded as nowcast-only.
        reasons = {e["label"]: e["reason"] for e in sk["excluded"]}
        assert reasons.get("Crude", "").startswith("state-mode")

    def test_skill_unavailable_without_trajectory(self, db):
        grid = _grid(); actual = _actual_curve()
        ttbl = db.safe_table_name("sensor.pv")
        db.store_history(ttbl, pd.DataFrame({"ds": grid, "value": actual}))
        for i, t in enumerate(grid):
            db.log_forecast(experiment="e", issued_at=t - timedelta(minutes=INTERVAL),
                            targets=[t], predictions=[actual[i]], model_name="lgb", model_version="v1")
        e_state = db.safe_table_name("sensor.crude")
        db.store_history(e_state, pd.DataFrame({"ds": grid, "value": [actual[i] + 2.0 for i in range(48)]}))
        res = db.get_external_forecast_comparison(
            "e", ttbl, [_spec("sensor.crude", "state", e_state)], GENEROUS_WINDOW, INTERVAL, "raw")
        sk = res["skill"]
        assert sk is None or sk.get("available") is False


class TestExternalDedupeMigration:
    """v2.46.2: one-time collapse of trajectories that were re-logged unchanged
    every forecast cycle (before capture-time content-change detection). Each
    run of identical issuances folds into its earliest, restoring true leads."""

    def _seed_dupes(self, db, source="sensor.solcast"):
        # Three targets, fixed across issuances so they fully overlap.
        T = [datetime(2024, 6, 15, 8, 0) + timedelta(hours=h) for h in range(3)]
        # 06:00 issuance, then two unchanged re-logs (06:15 / 06:30).
        for m in (0, 15, 30):
            db.log_external_forecast(
                "e", source, datetime(2024, 6, 15, 6, m), T, [2.0, 3.0, 4.0])
        # 07:00 genuine refresh (values changed), re-logged unchanged at 07:15.
        for m in (0, 15):
            db.log_external_forecast(
                "e", source, datetime(2024, 6, 15, 7, m), T, [2.5, 3.0, 4.0])
        return T

    def _issuances(self, db, source="sensor.solcast"):
        cur = db.conn.cursor()
        cur.execute(
            "SELECT DISTINCT issued_at FROM external_forecast_log "
            "WHERE source = ? ORDER BY issued_at", (source,))
        return [r[0] for r in cur.fetchall()]

    def test_helper_collapses_identical_runs_to_earliest(self, db):
        T = self._seed_dupes(db)
        assert len(self._issuances(db)) == 5
        # A neighbouring source with a single issuance must be left untouched.
        db.log_external_forecast("e", "sensor.other",
                                 datetime(2024, 6, 15, 6, 0), T, [9.0, 9.0, 9.0])
        removed = db._collapse_external_duplicates(db.conn.cursor())
        db.conn.commit()
        # 2 dupes in the 06:00 run + 1 in the 07:00 run, 3 rows each.
        assert removed == 9
        assert self._issuances(db) == [
            "2024-06-15 06:00:00", "2024-06-15 07:00:00"]
        assert len(self._issuances(db, "sensor.other")) == 1
        # The surviving earliest issuance carries the true (long) lead: 06:00
        # → 08:00 is 120 min, not the ~15 min the per-cycle re-logs implied.
        cur = db.conn.cursor()
        cur.execute("SELECT MIN(lead_minutes) FROM external_forecast_log "
                    "WHERE source='sensor.solcast' "
                    "AND issued_at='2024-06-15 06:00:00'")
        assert cur.fetchone()[0] == 120
        # The latest kept trajectory is the genuine 07:00 refresh.
        traj = db.get_last_external_trajectory("e", "sensor.solcast")
        assert sorted(traj.values()) == [2.5, 3.0, 4.0]

    def test_migration_is_gated_and_runs_once(self, db):
        # Seed first (the table-ensure inside log_external_forecast records the
        # migration as a no-op on the empty table); then simulate a pre-fix
        # install by clearing the marker so the populated data is migrated.
        self._seed_dupes(db)
        assert len(self._issuances(db)) == 5
        cur = db.conn.cursor()
        cur.execute("DELETE FROM schema_versions WHERE version = 2")
        db.conn.commit()
        assert 2 not in db._applied_versions()
        db.ensure_external_forecast_log_table()      # triggers the migration
        assert 2 in db._applied_versions()
        assert len(self._issuances(db)) == 2
        # Idempotent: with the marker set, a further ensure changes nothing.
        db.ensure_external_forecast_log_table()
        assert len(self._issuances(db)) == 2


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
