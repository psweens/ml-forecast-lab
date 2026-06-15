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


class TestComparisonBaselineAndWarmup:
    """v2.44.x additions: Seasonal Naive baseline row, %-of-typical + daily
    metrics, and the 7-day warming-up gate."""

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

    def test_seasonal_naive_baseline_present(self, db):
        ttbl, e1, _ = self._seed_days(db, n_days=10)
        res = db.get_external_forecast_comparison(
            "e", ttbl, [_spec("sensor.ext_state", "state", e1, label="Crude")],
            GENEROUS_WINDOW, INTERVAL, "raw")
        base = res["baseline"]
        assert base is not None, "Seasonal Naive baseline should be computed"
        assert base["label"] == "Seasonal Naive"
        assert base["is_baseline"] is True
        # repeating-day actual → 'same time yesterday' is near-perfect
        assert base["metrics"]["mae"] < 1.0
        # baseline is NOT counted as a configured competitor row
        assert len(res["comparisons"]) == 1
        # overlay carries a baseline line aligned to ds
        assert res["overlay"]["baseline"] is not None
        assert len(res["overlay"]["baseline"]) == len(res["overlay"]["ds"])
        # naive should beat the +12 external on its head-to-head vs app? no —
        # naive vs APP: app MAE ~4, naive MAE ~0 → naive wins (external='naive')
        assert base["head_to_head"]["winner"] == "external"

    def test_single_day_has_no_naive_baseline(self, db):
        # < 1 day of data → 'same time yesterday' has no overlap → no baseline.
        ttbl, e1, _ = self._seed_days(db, n_days=1)
        res = db.get_external_forecast_comparison(
            "e", ttbl, [_spec("sensor.ext_state", "state", e1)],
            GENEROUS_WINDOW, INTERVAL, "raw")
        assert res["baseline"] is None
        assert res["overlay"]["baseline"] is None

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
