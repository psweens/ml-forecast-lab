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
    """Captures set_state payloads; serves a fixed source unit."""

    def __init__(self, source_unit):
        self._source_unit = source_unit
        self.captured: list = []
        self.get_state_calls = 0

    async def set_state(self, entity_id, state, attributes=None):
        self.captured.append((entity_id, state, attributes or {}))
        return True

    async def get_state(self, entity_id, default=None, attribute=None):
        if attribute == "unit_of_measurement":
            self.get_state_calls += 1
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
