"""Auto-inherited forecast units (sensor unit_of_measurement).

When an experiment leaves ``units`` blank, the published HA sensors must
inherit the target sensor's own ``unit_of_measurement`` from HA. An
explicit ``units`` in the experiment config always wins. These tests
drive the real ``_publish_forecast_sensors`` against a stub HA interface
and assert the ``unit_of_measurement`` that lands on every sensor.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.main import MLForecastLabApp
from ml_forecast_lab.config import ExperimentCfg


class _StubHA:
    """Captures set_state payloads; serves a (mutable) source unit.

    ``raise_on_unit`` makes the unit lookup raise to simulate a transient
    HA API failure (``HAInterface.get_state`` would itself swallow this and
    return ``default``, so the stub mirrors the value path: None).
    """

    def __init__(self, source_unit):
        self._source_unit = source_unit
        self.raise_on_unit = False
        self.captured: list = []
        self.get_state_calls = 0

    async def set_state(self, entity_id, state, attributes=None):
        self.captured.append((entity_id, state, attributes or {}))
        return True

    async def get_state(self, entity_id, default=None, attribute=None):
        if attribute == "unit_of_measurement":
            self.get_state_calls += 1
            if self.raise_on_unit:
                raise RuntimeError("simulated transient HA failure")
            return self._source_unit if self._source_unit is not None else default
        return default


def _make_app(source_unit):
    app = MLForecastLabApp()
    app.ha_interface = _StubHA(source_unit)
    app.history_db = None
    app._cached_models = {}
    return app


def _exp(units=""):
    return ExperimentCfg(
        name="solar", target_entity="sensor.pv_power", units=units,
        interval_minutes=30, future_periods=3,
        publish_prefix="mlfl_", publish_name="solar", mode="production",
    )


def _publish(app, exp):
    # Reset so the return reflects only THIS publish cycle's sensors (the
    # stub accumulates set_state payloads across cycles).
    app.ha_interface.captured = []
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ds = pd.DatetimeIndex([
        pd.Timestamp("2026-06-12 12:00") + pd.Timedelta(minutes=30 * (i + 1))
        for i in range(3)
    ])
    asyncio.run(app._publish_forecast_sensors(
        exp_cfg=exp, y_pred=y, ds_future=ds,
        model_name="nlinear", last_trained_iso="2026-06-12T00:00:00Z",
    ))
    return [a.get("unit_of_measurement") for _, _, a in app.ha_interface.captured]


def test_explicit_units_win_and_skip_ha_lookup():
    app = _make_app(source_unit="W")  # would inherit W, but config says kW
    units = _publish(app, _exp(units="kW"))
    assert units, "expected at least one sensor published"
    assert all(u == "kW" for u in units)
    # Explicit config must not trigger a source-unit lookup.
    assert app.ha_interface.get_state_calls == 0


def test_blank_units_inherit_source_sensor_unit():
    app = _make_app(source_unit="W")
    units = _publish(app, _exp(units=""))
    assert units and all(u == "W" for u in units), (
        f"blank units should inherit 'W' from the source sensor; got {units}"
    )


def test_blank_units_with_unitless_source_stays_empty():
    app = _make_app(source_unit=None)  # source has no unit_of_measurement
    units = _publish(app, _exp(units=""))
    assert units and all(u == "" for u in units), (
        f"unitless source must yield empty unit, not crash; got {units}"
    )


def test_inherited_unit_is_cached_across_cycles():
    app = _make_app(source_unit="kWh")
    _publish(app, _exp(units=""))
    _publish(app, _exp(units=""))
    _publish(app, _exp(units=""))
    # One lookup total despite three publish cycles.
    assert app.ha_interface.get_state_calls == 1
    assert app._source_unit_cache.get("sensor.pv_power") == "kWh"


def test_retrain_then_transient_empty_keeps_last_good_unit():
    """Regression: a retrain invalidates the unit cache; if the post-retrain
    re-resolve comes back empty (HA hiccup / source momentarily
    ``unavailable``), the publish must keep the last-known unit instead of
    flipping HA's unit_of_measurement to ''."""
    app = _make_app(source_unit="W")
    units = _publish(app, _exp(units=""))
    assert all(u == "W" for u in units)

    # Simulate the retrain cache invalidation (main.py:_retrain_single).
    app._source_unit_cache.pop("sensor.pv_power", None)
    # Source now returns no unit on the fresh lookup (transient miss).
    app.ha_interface._source_unit = None

    units2 = _publish(app, _exp(units=""))
    assert all(u == "W" for u in units2), (
        f"transient empty after retrain must keep last-known 'W'; got {units2}"
    )


def test_retrain_then_fetch_raises_keeps_last_good_unit():
    """Same regression but the lookup raises (transient API error)."""
    app = _make_app(source_unit="kWh")
    assert all(u == "kWh" for u in _publish(app, _exp(units="")))
    app._source_unit_cache.pop("sensor.pv_power", None)
    app.ha_interface.raise_on_unit = True
    units = _publish(app, _exp(units=""))
    assert all(u == "kWh" for u in units), (
        f"a raising lookup after retrain must keep 'kWh'; got {units}"
    )


def test_retrain_picks_up_legitimate_unit_change():
    """The retrain re-resolve must still adopt a genuine source-unit change."""
    app = _make_app(source_unit="W")
    assert all(u == "W" for u in _publish(app, _exp(units="")))
    app._source_unit_cache.pop("sensor.pv_power", None)
    app.ha_interface._source_unit = "kW"  # source genuinely changed
    units = _publish(app, _exp(units=""))
    assert all(u == "kW" for u in units), (
        f"a real source-unit change must be adopted; got {units}"
    )


def test_cold_start_unitless_source_stays_empty_without_last_good():
    """No last-known unit yet + unitless source → empty, as before."""
    app = _make_app(source_unit=None)
    units = _publish(app, _exp(units=""))
    assert all(u == "" for u in units)
    assert app._source_unit_last_good.get("sensor.pv_power") is None
