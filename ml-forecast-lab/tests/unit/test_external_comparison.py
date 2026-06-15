"""
Regression suite for the External Comparison feature.

Pins:
  - ExperimentCfg parsing of the external_forecast_* fields (defaults,
    round-trip, invalid-mode rejection).
  - HistoryDB.log_external_forecast (lead computation, non-finite skip).
  - HistoryDB.get_external_forecast_comparison across the two ingestion
    modes (state / attribute) and both evaluation spaces (raw / increment),
    including the scale multiplier and the per-lead-time curve.

Every test seeds a small, fully-deterministic actuals table + forecast_log
(+ external_forecast_log for attribute mode) and asserts a property of the
head-to-head / overlay / lead-time result. No HA, no models — only the
config dataclass and the SQL / aggregation layer.
"""

import math
from datetime import datetime, timedelta

import pandas as pd
import pytest
import yaml

from ml_forecast_lab.config import ExperimentCfg, load_config
from ml_forecast_lab.db import HistoryDB


# Fixed past dates + a generous window so the wall-clock cutoff
# (now - max_age_days) and the target_dt <= now guard never flake.
GENEROUS_WINDOW = 3650
INTERVAL = 30


# ---------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------

class TestExternalForecastConfig:
    def test_defaults(self):
        cfg = ExperimentCfg(name="t", target_entity="sensor.t")
        assert cfg.external_forecast_entity is None
        assert cfg.external_forecast_mode == "state"
        assert cfg.external_forecast_attribute == "forecast"
        assert cfg.external_forecast_value_key is None
        assert cfg.external_forecast_scale is None
        assert cfg.external_forecast_is_cumulative is None
        assert cfg.external_forecast_label is None

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            ExperimentCfg(
                name="t", target_entity="sensor.t",
                external_forecast_mode="bogus",
            )

    def test_roundtrips_through_yaml(self, tmp_path):
        config_data = {
            "experiments": [{
                "name": "t", "target_entity": "sensor.load_w",
                "external_forecast_entity": "sensor.solcast",
                "external_forecast_mode": "attribute",
                "external_forecast_attribute": "detailedForecast",
                "external_forecast_value_key": "pv_estimate",
                "external_forecast_scale": 0.001,
                "external_forecast_is_cumulative": False,
                "external_forecast_label": "Solcast",
            }],
        }
        p = tmp_path / "mlfl.yaml"
        p.write_text(yaml.dump(config_data))
        e = load_config(p).experiments[0]
        assert e.external_forecast_entity == "sensor.solcast"
        assert e.external_forecast_mode == "attribute"
        assert e.external_forecast_attribute == "detailedForecast"
        assert e.external_forecast_value_key == "pv_estimate"
        assert e.external_forecast_scale == 0.001
        assert e.external_forecast_is_cumulative is False
        assert e.external_forecast_label == "Solcast"

    def test_bad_mode_skips_experiment_not_crash(self, tmp_path):
        p = tmp_path / "mlfl.yaml"
        p.write_text(yaml.dump({"experiments": [
            {"name": "t", "target_entity": "sensor.t",
             "external_forecast_mode": "nope"},
        ]}))
        # One bad experiment must not kill config loading.
        assert load_config(p).experiments == []


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
    # Smooth, strictly positive so cumulative deltas stay non-negative.
    return [100.0 + 50.0 * math.sin(i / 4.0) + 60.0 for i in range(n)]


# ---------------------------------------------------------------------
# log_external_forecast
# ---------------------------------------------------------------------

class TestLogExternalForecast:
    def test_lead_minutes_and_count(self, db):
        issued = datetime(2024, 6, 15, 0, 0, 0)
        targets = [issued + timedelta(minutes=INTERVAL * (i + 1)) for i in range(4)]
        n = db.log_external_forecast("e", issued, targets, [1.0, 2.0, 3.0, 4.0])
        assert n == 4
        cur = db.conn.cursor()
        cur.execute(
            "SELECT lead_minutes, value FROM external_forecast_log "
            "ORDER BY lead_minutes"
        )
        rows = cur.fetchall()
        assert [r[0] for r in rows] == [30, 60, 90, 120]
        assert [r[1] for r in rows] == [1.0, 2.0, 3.0, 4.0]

    def test_skips_non_finite(self, db):
        issued = datetime(2024, 6, 15, 0, 0, 0)
        targets = [issued + timedelta(minutes=INTERVAL * (i + 1)) for i in range(3)]
        n = db.log_external_forecast(
            "e", issued, targets, [1.0, float("nan"), None],
        )
        assert n == 1


# ---------------------------------------------------------------------
# get_external_forecast_comparison
# ---------------------------------------------------------------------

class TestComparisonStateRaw:
    def _seed(self, db, app_offset, ext_offset, ext_mul=1.0):
        grid = _grid()
        actual = _actual_curve()
        ttbl = db.safe_table_name("sensor.load_w")
        etbl = db.safe_table_name("sensor.ext_forecast")
        db.store_history(ttbl, pd.DataFrame({"ds": grid, "value": actual}))
        for i, t in enumerate(grid):
            issued = t - timedelta(minutes=INTERVAL)
            db.log_forecast(
                experiment="e", issued_at=issued, targets=[t],
                predictions=[actual[i] + app_offset], model_name="lgb",
                model_version="v1",
            )
        db.store_history(etbl, pd.DataFrame({
            "ds": grid, "value": [(actual[i] + ext_offset) * ext_mul for i in range(len(grid))],
        }))
        return ttbl, etbl

    def test_app_wins_and_metrics(self, db):
        ttbl, etbl = self._seed(db, app_offset=5.0, ext_offset=20.0)
        res = db.get_external_forecast_comparison(
            "e", ttbl, etbl, "state", GENEROUS_WINDOW, INTERVAL, "raw",
            None, False,
        )
        assert res["configured"] is True
        h = res["head_to_head"]
        assert h["winner"] == "app"
        assert abs(h["app"]["mae"] - 5.0) < 0.5
        assert abs(h["external"]["mae"] - 20.0) < 1.0
        assert h["n"] >= 40
        # state mode carries no lead-time dimension
        assert res["lead_time"] is None
        assert len(res["overlay"]["ds"]) >= 40
        # timing transparency: state mode is a contemporaneous snapshot,
        # the app forecast carries a ~30-min lead, external updates ~30min.
        t = res["timing"]
        assert t["external_contemporaneous"] is True
        assert t["external_median_lead_minutes"] is None
        assert abs(t["app_median_lead_minutes"] - 30.0) < 1e-6
        assert abs(t["external_update_minutes"] - 30.0) < 1e-6
        assert t["external_stale"] is False
        assert t["external_points"] >= 40 and t["grid_points"] >= 40

    def test_scale_applied_to_external(self, db):
        # External stored in *1000 units; scale=0.001 should recover it so
        # external MAE lands at ~20, not ~20000.
        ttbl, etbl = self._seed(db, app_offset=5.0, ext_offset=20.0, ext_mul=1000.0)
        res = db.get_external_forecast_comparison(
            "e", ttbl, etbl, "state", GENEROUS_WINDOW, INTERVAL, "raw",
            0.001, False,
        )
        assert abs(res["head_to_head"]["external"]["mae"] - 20.0) < 1.0

    def test_overlay_within_window_and_ordered(self, db):
        ttbl, etbl = self._seed(db, app_offset=5.0, ext_offset=20.0)
        res = db.get_external_forecast_comparison(
            "e", ttbl, etbl, "state", GENEROUS_WINDOW, INTERVAL, "raw",
            None, False,
        )
        ds = res["overlay"]["ds"]
        parsed = [pd.Timestamp(s) for s in ds]
        assert parsed == sorted(parsed)
        assert len(res["overlay"]["actual"]) == len(ds)
        assert len(res["overlay"]["app"]) == len(ds)
        assert len(res["overlay"]["external"]) == len(ds)


class TestComparisonAttributeLeadTime:
    def test_lead_time_curve_app_better_each_horizon(self, db):
        grid = _grid()
        actual = _actual_curve()
        ttbl = db.safe_table_name("sensor.solar_w")
        db.store_history(ttbl, pd.DataFrame({"ds": grid, "value": actual}))
        # Multi-lead app + external trajectories every 4 steps.
        for issue_idx in range(0, 48, 4):
            issued = grid[issue_idx] - timedelta(minutes=INTERVAL)
            targets = grid[issue_idx:issue_idx + 8]
            if not targets:
                continue
            db.log_forecast(
                experiment="e", issued_at=issued, targets=targets,
                predictions=[actual[issue_idx + k] + 3.0 for k in range(len(targets))],
                model_name="lgb", model_version="v1",
            )
            db.log_external_forecast(
                "e", issued, targets,
                [actual[issue_idx + k] + 15.0 for k in range(len(targets))],
            )
        res = db.get_external_forecast_comparison(
            "e", ttbl, None, "attribute", GENEROUS_WINDOW, INTERVAL, "raw",
            None, False,
        )
        h = res["head_to_head"]
        assert h["winner"] == "app"
        lt = res["lead_time"]
        assert lt is not None and len(lt["lead_minutes"]) >= 1
        for am, em in zip(lt["app_mae"], lt["external_mae"]):
            if am is not None and em is not None:
                assert am < em
        # attribute mode is lead-matched: both sides expose a median lead.
        t = res["timing"]
        assert t["external_contemporaneous"] is False
        assert t["external_median_lead_minutes"] is not None
        assert t["app_median_lead_minutes"] is not None


class TestComparisonIncrement:
    def test_cumulative_state_external_diffed(self, db):
        grid = _grid()
        actual = _actual_curve()
        # Cumulative actuals (monotonic running total of the curve/100).
        cum, s = [], 0.0
        for v in actual:
            s += v / 100.0
            cum.append(s)
        ttbl = db.safe_table_name("sensor.energy_today")
        etbl = db.safe_table_name("sensor.ext_energy")
        db.store_history(ttbl, pd.DataFrame({"ds": grid, "value": cum}))
        # App logs per-interval deltas (forecast_log convention for cumulative).
        for i, t in enumerate(grid):
            issued = t - timedelta(minutes=INTERVAL)
            delta = cum[i] - (cum[i - 1] if i > 0 else 0.0)
            db.log_forecast(
                experiment="e", issued_at=issued, targets=[t],
                predictions=[delta + 0.01], model_name="lgb", model_version="v1",
            )
        # External is a cumulative sensor too — drifts +0.2 per step.
        ext_cum = [cum[i] + 0.2 * (i + 1) for i in range(len(grid))]
        db.store_history(etbl, pd.DataFrame({"ds": grid, "value": ext_cum}))
        res = db.get_external_forecast_comparison(
            "e", ttbl, etbl, "state", GENEROUS_WINDOW, INTERVAL, "increment",
            None, True,  # external_is_cumulative
        )
        h = res["head_to_head"]
        assert h is not None
        # Per-interval app error ~0.01; external ~0.2 (the per-step drift).
        assert h["app"]["mae"] < h["external"]["mae"]
        assert h["winner"] == "app"


class TestComparisonEmptyStates:
    def test_missing_actuals_table(self, db):
        res = db.get_external_forecast_comparison(
            "e", db.safe_table_name("sensor.nope"), None, "attribute",
            GENEROUS_WINDOW, INTERVAL, "raw", None, False,
        )
        assert res.get("empty_reason") == "no_actuals"
        assert res["head_to_head"] is None

    def test_no_forecasts_yet(self, db):
        grid = _grid()
        ttbl = db.safe_table_name("sensor.load_w")
        db.store_history(ttbl, pd.DataFrame({"ds": grid, "value": _actual_curve()}))
        res = db.get_external_forecast_comparison(
            "e", ttbl, db.safe_table_name("sensor.ext"), "state",
            GENEROUS_WINDOW, INTERVAL, "raw", None, False,
        )
        # Actuals exist but no app/external rows → no head-to-head.
        assert res["head_to_head"] is None
        assert res["counts"]["common"] == 0
