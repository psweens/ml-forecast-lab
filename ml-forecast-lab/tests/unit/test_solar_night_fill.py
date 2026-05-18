"""Regression tests for the solar night-time NaN→0 fill.

HA's recorder is delta-storage based: when a PV sensor sits at 0 W
from sunset to sunrise it records one transition and then nothing,
or reports ``unavailable`` (parsed as NaN) while the inverter sleeps.
The default ``gap_handling='interpolate'`` only fills gaps up to
``gap_max_minutes`` (90), so 10-14h nights stay NaN and the
downstream ``result.dropna()`` deletes every night-time row.

These tests pin the fill behaviour so a future refactor can't silently
re-introduce the bug. The user's v2.37.2 debug bundle showed only 3
of 2088 training rows had ``sun_elevation < 0`` — the model trained on
a daytime-only window and predicted 0.3-0.7 kW at 23:00 with the daily
peak phase-shifted to 18:00. This fix restores the full daily curve
in the training set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_forecast_lab.config import ExperimentCfg
from ml_forecast_lab.main import _apply_idle_value_fill, _apply_solar_night_fill


def _make_solar_result(
    n_days: int = 5,
    interval_minutes: int = 30,
    drop_night: bool = True,
) -> pd.DataFrame:
    """Build a ``result`` dataframe matching what
    ``_fetch_and_preprocess`` produces right before the dropna call.
    Daytime rows have realistic PV values; night rows are NaN to
    simulate the HA delta-storage / gap_handling=interpolate skew.
    """
    idx = pd.date_range(
        "2026-05-01 00:00",
        periods=n_days * (24 * 60 // interval_minutes),
        freq=f"{interval_minutes}min",
        tz=None,
    )
    hour = idx.hour + idx.minute / 60.0
    # sun_elev as a single source of truth — peaks at hour 13, zero at
    # hours 6 and 20. Derive y and ghi from it so all three are
    # internally consistent (matches pvlib's deterministic computation
    # in the real addon).
    sun_elev = 50 * np.sin(np.pi * (hour - 6) / 14)
    sun_elev = np.where((hour < 6) | (hour > 20), -10.0, sun_elev)
    sun_up = sun_elev > 0

    daylight_factor = np.clip(sun_elev / 50.0, 0, None)
    y = 3.5 * daylight_factor  # peak ~3.5 kW
    y = pd.Series(y, index=idx, name="y")
    ghi = 800 * daylight_factor  # 0 wherever sun is at-or-below horizon

    # Simulate HA delta-storage drop: every night-time slot becomes
    # NaN in y, mimicking what the user's bundle showed (hours 0-3
    # and 21-23 entirely missing).
    if drop_night:
        y[~sun_up] = np.nan

    result = pd.DataFrame({
        "y": y,
        "clear_sky_ghi": ghi,
        "sun_elevation": sun_elev,
    }, index=idx)
    return result


def test_fill_skipped_when_target_is_not_nonnegative():
    """Signed targets (net grid flow, temperature delta) keep the
    original drop-on-NaN behaviour. Gate must short-circuit."""
    result = _make_solar_result()
    exp = ExperimentCfg(name="grid", target_entity="x", target_is_nonnegative=False)
    n_before = result["y"].isna().sum()
    n_filled = _apply_idle_value_fill(result, exp)
    assert n_filled == 0
    assert result["y"].isna().sum() == n_before


def test_fill_skipped_when_no_solar_features():
    """Without ``clear_sky_ghi`` or ``sun_elevation`` the helper has
    no way to determine night-vs-day — must skip rather than guess."""
    result = _make_solar_result()
    result = result.drop(columns=["clear_sky_ghi", "sun_elevation"])
    exp = ExperimentCfg(name="pv", target_entity="x", target_is_nonnegative=True)
    n_filled = _apply_idle_value_fill(result, exp)
    assert n_filled == 0


def test_fill_uses_clear_sky_ghi_preferentially():
    """When both ``clear_sky_ghi`` and ``sun_elevation`` are present
    the helper uses ``clear_sky_ghi <= 0`` (matches features.py
    physics gate)."""
    result = _make_solar_result()
    exp = ExperimentCfg(name="pv", target_entity="x", target_is_nonnegative=True)
    n_filled = _apply_idle_value_fill(result, exp)
    assert n_filled > 0
    night_idx = result["clear_sky_ghi"] <= 0
    # Every night row that was NaN should now be 0
    assert (result.loc[night_idx, "y"] == 0.0).all()
    # Daytime rows untouched
    day_idx = result["clear_sky_ghi"] > 0
    assert result.loc[day_idx, "y"].notna().all()


def test_fill_falls_back_to_sun_elevation():
    """When only ``sun_elevation`` is present, falls back to that
    with the standard -0.833° astronomical horizon."""
    result = _make_solar_result()
    result = result.drop(columns=["clear_sky_ghi"])
    exp = ExperimentCfg(name="pv", target_entity="x", target_is_nonnegative=True)
    n_filled = _apply_idle_value_fill(result, exp)
    assert n_filled > 0
    night_idx = result["sun_elevation"] < -0.833
    assert (result.loc[night_idx, "y"] == 0.0).all()


def test_fill_preserves_daytime_nan():
    """A sensor outage during daylight should NOT be filled with 0.
    Daytime NaN must propagate to dropna so genuine sensor failures
    aren't silently masked."""
    result = _make_solar_result()
    # Punch a hole at midday: clear_sky_ghi is high here, so y NaN is
    # a real outage that the fill must leave alone.
    midday = (result.index.hour == 12) & (result.index.day == 3)
    result.loc[midday, "y"] = np.nan
    exp = ExperimentCfg(name="pv", target_entity="x", target_is_nonnegative=True)
    _apply_idle_value_fill(result, exp)
    # Midday NaN remains NaN (will be dropped by the dropna step)
    assert result.loc[midday, "y"].isna().all()


def test_fill_is_idempotent():
    """Running twice produces the same result; no double-fill."""
    result = _make_solar_result()
    exp = ExperimentCfg(name="pv", target_entity="x", target_is_nonnegative=True)
    n1 = _apply_idle_value_fill(result, exp)
    n2 = _apply_idle_value_fill(result, exp)
    assert n1 > 0
    assert n2 == 0  # nothing left to fill


def test_fill_restores_full_daily_coverage():
    """After fill + dropna, every hour-of-day must be represented.
    This is the regression-shape pinning: without the fill, hours
    21-03 are completely absent."""
    result = _make_solar_result(n_days=10)
    exp = ExperimentCfg(name="pv", target_entity="x", target_is_nonnegative=True)
    _apply_idle_value_fill(result, exp)
    surviving = result.dropna()
    hours_present = set(surviving.index.hour.unique())
    assert hours_present == set(range(24)), (
        f"Missing hours: {set(range(24)) - hours_present}. "
        f"This is the user's v2.37.2 bug — only daylight hours present."
    )


def test_fill_log_transformed_y_is_still_zero():
    """When ``log_transform=True`` the y series is in log space, but
    log(1+0) = 0 so writing 0.0 is correct without any inverse."""
    result = _make_solar_result()
    # log-transform the surviving daytime values, leave NaN alone
    day = result["y"].notna()
    result.loc[day, "y"] = np.log1p(result.loc[day, "y"])
    exp = ExperimentCfg(
        name="pv", target_entity="x",
        target_is_nonnegative=True, log_transform=True,
    )
    _apply_idle_value_fill(result, exp)
    night_idx = result["clear_sky_ghi"] <= 0
    assert (result.loc[night_idx, "y"] == 0.0).all()
    # Spot check expm1 inverse still gives physical 0
    assert np.expm1(0.0) == 0.0


# ----------------------------------------------------------------------
# v2.37.4 — generalised idle_value path (non-solar non-negative targets)
# ----------------------------------------------------------------------

def _make_ev_charger_result(n_days: int = 5) -> pd.DataFrame:
    """Synthesise an EV charger power series: 0 most of the time, with
    a few daily sessions. No solar physics features. Simulates HA's
    delta-storage drop by setting all 'idle' rows to NaN."""
    idx = pd.date_range(
        "2026-05-01 00:00",
        periods=n_days * 48,
        freq="30min",
        tz=None,
    )
    rng = np.random.default_rng(0)
    # Active charging in a random 4-hr block per day, ~7 kW
    active = np.zeros(len(idx), dtype=bool)
    for d in range(n_days):
        start = d * 48 + rng.integers(20, 36)
        active[start:start + 8] = True
    y = np.where(active, 7.0 + rng.normal(0, 0.3, len(idx)), np.nan)
    return pd.DataFrame({"y": pd.Series(y, index=idx)}, index=idx)


def test_idle_value_zero_fills_all_ev_idle_gaps():
    """EV charger with idle_value=0: every NaN slot (idle) becomes 0,
    so the model sees the full 24-hour duty cycle instead of training
    on the active-only ~17% of intervals."""
    result = _make_ev_charger_result()
    nan_before = result["y"].isna().sum()
    assert nan_before > 0, "fixture should have idle NaN gaps"

    exp = ExperimentCfg(
        name="ev_power", target_entity="sensor.ev_charger_power",
        target_is_nonnegative=True, idle_value=0.0,
    )
    n_filled = _apply_idle_value_fill(result, exp)
    assert n_filled == nan_before
    assert result["y"].isna().sum() == 0
    # All idle slots are exactly 0; active slots still ~7 kW
    assert (result["y"].between(0.0, 8.0)).all()


def test_idle_value_none_keeps_drop_behaviour_for_non_solar():
    """Default idle_value=None on a non-solar target: no fill applied,
    NaN rows still get dropped downstream (preserves the v2.37.3
    behaviour for users who haven't opted in)."""
    result = _make_ev_charger_result()
    nan_before = result["y"].isna().sum()

    exp = ExperimentCfg(
        name="ev_power", target_entity="sensor.ev_charger_power",
        target_is_nonnegative=True,  # idle_value omitted → None
    )
    n_filled = _apply_idle_value_fill(result, exp)
    assert n_filled == 0
    assert result["y"].isna().sum() == nan_before


def test_idle_value_overrides_solar_default_zero():
    """Solar target with idle_value=0.005 (measurable inverter
    standby): night-time slots get 0.005 instead of the default 0."""
    result = _make_solar_result()
    exp = ExperimentCfg(
        name="pv", target_entity="x",
        target_is_nonnegative=True, idle_value=0.005,
    )
    _apply_idle_value_fill(result, exp)
    night_idx = result["clear_sky_ghi"] <= 0
    assert (result.loc[night_idx, "y"] == 0.005).all()


def test_idle_value_does_not_fill_when_target_not_nonneg():
    """Even with idle_value set, target_is_nonnegative=False still
    blocks the fill — signed targets keep drop-on-NaN."""
    result = _make_ev_charger_result()
    exp = ExperimentCfg(
        name="grid_flow", target_entity="x",
        target_is_nonnegative=False, idle_value=0.0,
    )
    n_filled = _apply_idle_value_fill(result, exp)
    assert n_filled == 0


def test_idle_value_negative_allowed():
    """A negative idle_value is valid (e.g. a sensor that reads -5
    when idle but should still be treated as below-active). Field
    validator must not reject it."""
    result = _make_ev_charger_result()
    exp = ExperimentCfg(
        name="weird", target_entity="x",
        target_is_nonnegative=True, idle_value=-5.0,
    )
    _apply_idle_value_fill(result, exp)
    # Every idle slot becomes -5; check at least one row was filled.
    assert (result["y"] == -5.0).any()


def test_solar_night_fill_alias_still_callable():
    """Back-compat: _apply_solar_night_fill is an alias for the
    renamed helper. Pin so future cleanups don't break pinned imports
    or v2.37.3 docs/tests."""
    assert _apply_solar_night_fill is _apply_idle_value_fill
