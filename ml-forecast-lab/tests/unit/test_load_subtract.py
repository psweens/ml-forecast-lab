"""Tests for load_subtract: SubtractCfg validation, YAML plumbing, and the
``apply_load_subtract`` robustness checklist.

Each test in ``TestApplyLoadSubtract`` exercises one row of the robustness
checklist — see the docstring of each test for the specific concern.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import yaml

from ml_forecast_lab.config import (
    SubtractCfg,
    add_experiment_load_subtract,
    clear_experiment_load_subtract,
    load_config,
    remove_experiment_load_subtract,
)
from ml_forecast_lab.preprocessing import (
    LoadSubtractError,
    apply_load_subtract,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def load_48h() -> pd.Series:
    """48 h of half-hourly synthetic load in kWh per interval.

    Baseline ~0.2 kWh/interval with a mid-day bump. Deterministic, no noise.
    """
    idx = pd.date_range("2026-01-01", periods=96, freq="30min")
    baseline = np.full(96, 0.2)
    # Mid-day bump (hours 10-14) to give a realistic daily curve.
    bump_idx = np.arange(20, 28)
    baseline[bump_idx] += np.sin(np.linspace(0, np.pi, len(bump_idx))) * 0.3
    return pd.Series(baseline, index=idx, name="load")


@pytest.fixture
def load_48h_with_ev(load_48h) -> pd.Series:
    """Whole-house load that INCLUDES EV charging — the realistic input
    shape. Charging window hours 2-5 on day 1 adds 1.4 kWh/interval on top
    of baseline, matching how the GivTCP load sensor reports it in reality."""
    s = load_48h.copy()
    s.iloc[4:10] += 1.4  # 6 half-hour slots × 1.4 kWh ≈ 8.4 kWh EV session
    return s


@pytest.fixture
def ev_only(load_48h) -> pd.Series:
    """EV-only contribution: same index as load, zero except during the
    charging window. Safe to subtract — strictly ≤ the load it was added to."""
    ev = pd.Series(0.0, index=load_48h.index)
    ev.iloc[4:10] = 1.4
    return ev


# ---------------------------------------------------------------------------
# SubtractCfg validation
# ---------------------------------------------------------------------------


class TestSubtractCfgValidation:
    def test_defaults(self):
        cfg = SubtractCfg(entity_id="sensor.ev_today")
        assert cfg.source == "auto"
        assert cfg.on_missing == "zero"
        assert cfg.scale is None
        assert cfg.max_fraction_of_load == 1.0
        assert cfg.max_fraction_violation_pct == 5.0

    @pytest.mark.parametrize(
        "source",
        ["auto", "cumulative_daily", "cumulative_monotonic", "interval"],
    )
    def test_valid_source(self, source):
        cfg = SubtractCfg(entity_id="sensor.x", source=source)
        assert cfg.source == source

    def test_invalid_source(self):
        with pytest.raises(ValueError, match="source must be one of"):
            SubtractCfg(entity_id="sensor.x", source="cumulative")

    @pytest.mark.parametrize("on_missing", ["zero", "drop", "error"])
    def test_valid_on_missing(self, on_missing):
        cfg = SubtractCfg(entity_id="sensor.x", on_missing=on_missing)
        assert cfg.on_missing == on_missing

    def test_invalid_on_missing(self):
        with pytest.raises(ValueError, match="on_missing must be one of"):
            SubtractCfg(entity_id="sensor.x", on_missing="fill")

    def test_negative_max_fraction(self):
        with pytest.raises(ValueError, match="max_fraction_of_load"):
            SubtractCfg(entity_id="sensor.x", max_fraction_of_load=-0.1)

    def test_violation_pct_out_of_range(self):
        with pytest.raises(ValueError, match="max_fraction_violation_pct"):
            SubtractCfg(entity_id="sensor.x", max_fraction_violation_pct=150)

    def test_empty_entity_id(self):
        with pytest.raises(ValueError, match="entity_id"):
            SubtractCfg(entity_id="")


# ---------------------------------------------------------------------------
# YAML round-trip and legacy `subtract` deprecation
# ---------------------------------------------------------------------------


class TestYamlRoundTrip:
    def test_load_config_parses_load_subtract(self, tmp_path):
        cfg_path = tmp_path / "mlfl.yaml"
        cfg_path.write_text(yaml.dump({
            "experiments": [{
                "name": "house_load",
                "target_entity": "sensor.load",
                "load_subtract": [
                    {
                        "entity_id": "sensor.ev_today",
                        "source": "cumulative_daily",
                        "on_missing": "zero",
                    },
                    {
                        "entity_id": "sensor.iboost_today",
                        "source": "cumulative_daily",
                        "scale": 0.001,  # Wh → kWh
                    },
                ],
            }],
        }))
        app = load_config(cfg_path)
        exp = app.experiments[0]
        assert len(exp.load_subtract) == 2
        assert isinstance(exp.load_subtract[0], SubtractCfg)
        assert exp.load_subtract[0].entity_id == "sensor.ev_today"
        assert exp.load_subtract[1].scale == 0.001

    def test_bare_string_entries_tolerated_with_warning(self, tmp_path, caplog):
        cfg_path = tmp_path / "mlfl.yaml"
        cfg_path.write_text(yaml.dump({
            "experiments": [{
                "name": "exp",
                "target_entity": "sensor.load",
                "load_subtract": ["sensor.ev"],
            }],
        }))
        with caplog.at_level(logging.WARNING):
            app = load_config(cfg_path)
        assert len(app.experiments[0].load_subtract) == 1
        assert app.experiments[0].load_subtract[0].entity_id == "sensor.ev"
        assert any("bare string" in r.message for r in caplog.records)

    def test_legacy_subtract_field_warns(self, tmp_path, caplog):
        """Legacy `subtract: [str]` stub must log a deprecation warning so
        users migrating don't silently lose the behaviour they expected."""
        cfg_path = tmp_path / "mlfl.yaml"
        cfg_path.write_text(yaml.dump({
            "experiments": [{
                "name": "exp",
                "target_entity": "sensor.load",
                "subtract": ["sensor.old"],
            }],
        }))
        with caplog.at_level(logging.WARNING):
            load_config(cfg_path)
        assert any(
            "deprecated" in r.message and "load_subtract" in r.message
            for r in caplog.records
        )

    def test_add_and_remove_helpers(self, tmp_path):
        cfg_path = tmp_path / "mlfl.yaml"
        cfg_path.write_text(yaml.dump({
            "experiments": [{"name": "e", "target_entity": "sensor.x"}],
        }))

        # Add
        added = add_experiment_load_subtract(
            cfg_path, "e",
            {"entity_id": "sensor.ev", "source": "cumulative_daily"},
        )
        assert added is True

        # Duplicate rejected
        dup = add_experiment_load_subtract(
            cfg_path, "e", {"entity_id": "sensor.ev"},
        )
        assert dup is False

        # Invalid entry rejected by SubtractCfg validation
        with pytest.raises(ValueError):
            add_experiment_load_subtract(
                cfg_path, "e",
                {"entity_id": "sensor.y", "source": "bogus"},
            )

        # Remove by short name
        removed = remove_experiment_load_subtract(cfg_path, "e", "ev")
        assert removed is True

        # Add two and clear
        add_experiment_load_subtract(
            cfg_path, "e", {"entity_id": "sensor.a"},
        )
        add_experiment_load_subtract(
            cfg_path, "e", {"entity_id": "sensor.b"},
        )
        assert clear_experiment_load_subtract(cfg_path, "e") == 2


# ---------------------------------------------------------------------------
# apply_load_subtract — robustness checklist
# ---------------------------------------------------------------------------


def _cfg(entity_id: str, **overrides) -> dict:
    """Helper: build a dict representation of SubtractCfg with overrides."""
    return asdict(SubtractCfg(entity_id=entity_id, **overrides))


class TestApplyLoadSubtract:
    # -- No subtract -----------------------------------------------------

    def test_empty_subtracts_returns_copy(self, load_48h):
        """Empty subtract list → load returned unchanged (but as a copy)."""
        adj, audit = apply_load_subtract(load_48h, [])
        pd.testing.assert_series_equal(adj, load_48h)
        assert adj is not load_48h
        assert audit["per_sensor"] == []
        assert audit["n_clipped_rows"] == 0

    # -- Time alignment --------------------------------------------------

    def test_perfect_alignment(self, load_48h_with_ev, ev_only, load_48h):
        """Subtract aligned exactly to load grid, subtract ≤ load everywhere:
        standard subtraction with no clipping and no missing rows.

        Uses the realistic shape: whole-house load (``load_48h_with_ev``)
        includes the EV charging contribution, so subtracting ``ev_only``
        returns the baseline ``load_48h``."""
        adj, audit = apply_load_subtract(
            load_48h_with_ev, [(_cfg("sensor.ev"), ev_only)],
        )
        pd.testing.assert_series_equal(
            adj, load_48h.astype("float64"),
            check_names=False,
        )
        assert audit["n_clipped_rows"] == 0
        assert audit["per_sensor"][0]["rows_missing"] == 0
        assert audit["per_sensor"][0]["violation_rows"] == 0

    # -- on_missing: zero / drop / error --------------------------------

    def test_on_missing_zero_fills_gaps(self, load_48h):
        """on_missing='zero': NaN rows → filled with 0, not silently dropped."""
        # EV series covers only first 12 hours
        short_ev = pd.Series(0.1, index=load_48h.index[:24])
        adj, audit = apply_load_subtract(
            load_48h, [(_cfg("sensor.ev", on_missing="zero"), short_ev)],
        )
        # First 24 rows: load - 0.1; remaining 72 rows: unchanged.
        np.testing.assert_allclose(
            adj.iloc[:24].values,
            (load_48h.iloc[:24] - 0.1).clip(lower=0).values,
        )
        np.testing.assert_allclose(
            adj.iloc[24:].values, load_48h.iloc[24:].values,
        )
        assert audit["per_sensor"][0]["rows_missing"] == 72
        # Leading-gap fields stay None when the sensor covers the start of
        # the window; the trailing gap is now reported separately.
        assert audit["per_sensor"][0]["gap_start"] is None
        assert audit["per_sensor"][0]["trailing_gap_start"] is not None
        assert audit["per_sensor"][0]["trailing_gap_end"] == load_48h.index[-1].isoformat()

    def test_on_missing_drop_removes_rows(self, load_48h):
        """on_missing='drop': NaN rows → dropped from the adjusted series."""
        short_ev = pd.Series(0.1, index=load_48h.index[:24])
        adj, audit = apply_load_subtract(
            load_48h, [(_cfg("sensor.ev", on_missing="drop"), short_ev)],
        )
        # Only first 24 rows should remain.
        assert len(adj) == 24
        assert audit["n_rows"] == 24
        assert audit["per_sensor"][0]["rows_dropped"] == 72

    def test_on_missing_error_raises(self, load_48h):
        """on_missing='error': any gap → ValueError, no silent fill."""
        short_ev = pd.Series(0.1, index=load_48h.index[:24])
        with pytest.raises(ValueError, match="on_missing='error'"):
            apply_load_subtract(
                load_48h, [(_cfg("sensor.ev", on_missing="error"), short_ev)],
            )

    # -- History coverage gap (leading gap detected) --------------------

    def test_leading_gap_detected(self, load_48h):
        """Sensor that starts mid-window (new EV install): leading gap_start
        and gap_end must be populated in the audit."""
        late_ev = pd.Series(0.1, index=load_48h.index[48:])  # only day 2
        _, audit = apply_load_subtract(
            load_48h, [(_cfg("sensor.ev"), late_ev)],
        )
        rec = audit["per_sensor"][0]
        assert rec["gap_start"] is not None
        assert rec["gap_end"] is not None
        # gap ends the timestamp before first present row
        assert rec["gap_end"] == load_48h.index[47].isoformat()
        # Trailing-gap fields stay None because the sensor reaches the end.
        assert rec["trailing_gap_start"] is None
        assert rec["trailing_gap_end"] is None

    def test_trailing_gap_detected(self, load_48h):
        """Sensor that stopped reporting before the load did: the trailing
        gap window must be recorded in the audit so users can see the sensor
        went stale. Previously the docstring promised this but the detector
        only captured leading gaps."""
        early_ev = pd.Series(0.1, index=load_48h.index[:48])  # only day 1
        _, audit = apply_load_subtract(
            load_48h, [(_cfg("sensor.ev"), early_ev)],
        )
        rec = audit["per_sensor"][0]
        # Leading gap absent (sensor covers the start)
        assert rec["gap_start"] is None
        assert rec["gap_end"] is None
        # Trailing gap starts one row after the last present timestamp
        assert rec["trailing_gap_start"] == load_48h.index[48].isoformat()
        assert rec["trailing_gap_end"] == load_48h.index[-1].isoformat()

    def test_leading_and_trailing_gaps_both_detected(self, load_48h):
        """Sensor that only covers the middle of the window: both gap
        windows must be reported."""
        middle = pd.Series(0.1, index=load_48h.index[24:72])
        _, audit = apply_load_subtract(
            load_48h, [(_cfg("sensor.x"), middle)],
        )
        rec = audit["per_sensor"][0]
        assert rec["gap_start"] == load_48h.index[0].isoformat()
        assert rec["gap_end"] == load_48h.index[23].isoformat()
        assert rec["trailing_gap_start"] == load_48h.index[72].isoformat()
        assert rec["trailing_gap_end"] == load_48h.index[-1].isoformat()

    # -- Negative clip ---------------------------------------------------

    def test_negative_result_clipped_and_counted(self, load_48h):
        """When subtract > load on a row, result clipped to 0 and counted."""
        over = load_48h * 2.0  # always twice the load
        adj, audit = apply_load_subtract(
            # huge max_fraction so the fraction guard doesn't fire first
            load_48h,
            [(_cfg("sensor.x", max_fraction_of_load=10.0,
                   max_fraction_violation_pct=100.0), over)],
        )
        assert (adj >= 0).all()
        assert audit["n_clipped_rows"] == len(load_48h)
        assert audit["clipped_pct"] == pytest.approx(100.0)

    # -- Fraction guard (fail-fast) -------------------------------------

    def test_fraction_guard_fires_on_unit_bug(self, load_48h):
        """Simulate a Wh-vs-kWh unit bug: subtract is 1000× the load.
        Guard must raise with a diagnostic message, not silently clip."""
        wh_bug = load_48h * 1000.0
        with pytest.raises(LoadSubtractError, match="exceeded"):
            apply_load_subtract(
                load_48h, [(_cfg("sensor.bug"), wh_bug)],
            )

    def test_fraction_guard_tolerates_noise_band(self, load_48h):
        """Tiny per-row exceedances under the violation percentage should
        NOT raise — only clip. Simulates measurement-noise band, not a unit
        bug."""
        # Make 3 out of 96 rows slightly exceed load (≈3.1% — under the 5%
        # default threshold).
        tiny_over = pd.Series(0.0, index=load_48h.index)
        tiny_over.iloc[0:3] = load_48h.iloc[0:3] * 1.01
        adj, audit = apply_load_subtract(
            load_48h, [(_cfg("sensor.noisy"), tiny_over)],
        )
        assert audit["per_sensor"][0]["violation_rows"] == 3
        assert audit["n_clipped_rows"] == 3

    # -- Unit scale ------------------------------------------------------

    def test_scale_applied_before_subtraction(self, load_48h):
        """cfg.scale multiplies the subtract series before subtraction
        (the unit-fix path)."""
        ev_wh = pd.Series(100.0, index=load_48h.index)  # 100 Wh per interval
        adj, audit = apply_load_subtract(
            load_48h,
            [(_cfg("sensor.ev", scale=0.001,       # Wh → kWh
                   max_fraction_of_load=1.0), ev_wh)],
        )
        # Effective subtract is 0.1 kWh / interval
        expected = (load_48h - 0.1).clip(lower=0)
        pd.testing.assert_series_equal(
            adj, expected.astype("float64"), check_names=False,
        )

    # -- Timezone mismatch ----------------------------------------------

    def test_tz_mismatch_raises(self, load_48h):
        """tz-naive load + tz-aware subtract → immediate error, before any
        arithmetic happens (avoids pandas silently producing NaN rows)."""
        aware = pd.Series(
            0.1,
            index=pd.date_range(
                "2026-01-01", periods=96, freq="30min", tz="UTC",
            ),
        )
        with pytest.raises(ValueError, match="tz-aware"):
            apply_load_subtract(
                load_48h, [(_cfg("sensor.ev"), aware)],
            )

    # -- Multiple subtract sensors compose correctly --------------------

    def test_multiple_subtracts_summed(self, load_48h):
        """Two subtract sensors: result = load - a - b (clipped)."""
        a = pd.Series(0.05, index=load_48h.index)
        b = pd.Series(0.03, index=load_48h.index)
        adj, audit = apply_load_subtract(
            load_48h,
            [(_cfg("sensor.a"), a), (_cfg("sensor.b"), b)],
        )
        expected = (load_48h - 0.08).clip(lower=0)
        pd.testing.assert_series_equal(
            adj, expected.astype("float64"), check_names=False,
        )
        assert len(audit["per_sensor"]) == 2
