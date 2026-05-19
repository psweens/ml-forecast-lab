"""Regression tests for the v2.38.4 attribute-history path.

HA's `/api/history/period` returns each state-change with the full
state object including the .attributes dict (modulo
``minimal_response``). For entities whose .state is categorical
(``weather.metoffice → "partlycloudy"``) the useful numeric metrics
live in .attributes (``temperature``, ``cloud_coverage``, etc.).

Until v2.38.4, ``normalise_history`` only read the .state field, so
weather entities produced 0 numeric history values and v2.38.3's
empty-column guard had to zero-fill the past. v2.38.4 makes
``normalise_history`` and ``get_history`` capable of reading from
attributes; ``fetch_history`` routes weather-domain entities with
a future_value_key through that path automatically.

These tests pin:
- normalise_history extracts attribute_key from per-record attributes
- normalise_history handles missing attributes / non-numeric values
  gracefully (NaN, not crash)
- get_history's ``include_attributes`` flag controls the HA query
  (no minimal_response)
- fetch_history auto-routes weather.* + future_value_key through
  the attribute path
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.covariates import CovariateResolver
from ml_forecast_lab.ha_interface import normalise_history


def _run(coro):
    return asyncio.run(coro)


def test_normalise_history_state_path_unchanged_when_no_attribute_key():
    """Default call (no attribute_key) reads .state. Pin so the
    common numeric-sensor path can't regress."""
    raw = [
        {"last_changed": "2026-05-01T12:00:00+00:00", "state": "14.5"},
        {"last_changed": "2026-05-01T13:00:00+00:00", "state": "15.2"},
    ]
    df = normalise_history(raw)
    assert list(df.columns) == ["ds", "value"]
    assert df["value"].tolist() == [14.5, 15.2]


def test_normalise_history_reads_from_attribute_when_key_set():
    """v2.38.4: attribute_key=foo pulls record['attributes']['foo']
    instead of record['state']. Simulates an HA weather entity
    history payload."""
    raw = [
        {
            "last_changed": "2026-05-01T12:00:00+00:00",
            "state": "partlycloudy",
            "attributes": {"temperature": 14.5, "cloud_coverage": 30},
        },
        {
            "last_changed": "2026-05-01T13:00:00+00:00",
            "state": "cloudy",
            "attributes": {"temperature": 15.2, "cloud_coverage": 75},
        },
    ]
    df_temp = normalise_history(raw, attribute_key="temperature")
    assert df_temp["value"].tolist() == [14.5, 15.2]
    df_cloud = normalise_history(raw, attribute_key="cloud_coverage")
    assert df_cloud["value"].tolist() == [30.0, 75.0]


def test_normalise_history_handles_missing_attribute():
    """A record without the requested attribute key produces NaN —
    typical for transient 'unavailable' state changes that strip
    the attribute dict."""
    raw = [
        {
            "last_changed": "2026-05-01T12:00:00+00:00",
            "state": "partlycloudy",
            "attributes": {"temperature": 14.5},
        },
        {
            "last_changed": "2026-05-01T12:30:00+00:00",
            "state": "unavailable",
            "attributes": {},
        },
        {
            "last_changed": "2026-05-01T13:00:00+00:00",
            "state": "cloudy",
            "attributes": {"temperature": 15.2},
        },
    ]
    df = normalise_history(raw, attribute_key="temperature")
    # Three rows kept (timestamps preserved); middle row NaN'd
    assert len(df) == 3
    vals = df["value"].tolist()
    assert vals[0] == 14.5
    assert pd.isna(vals[1])
    assert vals[2] == 15.2


def test_get_history_include_attributes_flag_passes_to_params():
    """``include_attributes=True`` must drop ``minimal_response`` from
    the HA history query so the response carries attribute dicts.
    Without this the API would still strip them server-side."""
    from ml_forecast_lab.ha_interface import HAInterface
    from datetime import datetime, timezone

    # Build a bare instance via __new__ to skip __init__'s session
    # construction (no event loop in this sync test).
    iface = HAInterface.__new__(HAInterface)
    iface.api_call = AsyncMock(return_value=[[]])

    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    end = datetime(2026, 5, 2, tzinfo=timezone.utc)

    # Default — minimal_response should be set
    _run(iface.get_history("weather.test", start, end))
    params = iface.api_call.call_args.kwargs["params"]
    assert "minimal_response" in params

    # include_attributes=True — minimal_response should be absent
    iface.api_call = AsyncMock(return_value=[[]])
    _run(iface.get_history("weather.test", start, end, include_attributes=True))
    params = iface.api_call.call_args.kwargs["params"]
    assert "minimal_response" not in params


def test_fetch_history_routes_weather_entity_with_value_key_through_attribute_path():
    """``weather.*`` entity + ``future_value_key`` set: the resolver
    must call get_history with ``include_attributes=True`` AND pass
    the value_key as the attribute_key to normalise_history. This
    is the core v2.38.4 contract — weather entities now produce
    real historical numeric data instead of 0% coverage."""
    iface = MagicMock()
    iface.get_history = AsyncMock(return_value=[
        {
            "last_changed": "2026-05-01T12:00:00+00:00",
            "state": "partlycloudy",
            "attributes": {"cloud_coverage": 30},
        },
        {
            "last_changed": "2026-05-01T12:30:00+00:00",
            "state": "cloudy",
            "attributes": {"cloud_coverage": 75},
        },
    ])
    resolver = CovariateResolver(iface)

    cov_cfg = {
        "entity_id": "weather.met_office_balsham",
        "name": "met_office_balsham__cloud_coverage",
        "future_value_key": "cloud_coverage",
    }
    from datetime import datetime, timezone
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    end = datetime(2026, 5, 2, tzinfo=timezone.utc)
    result = _run(resolver.fetch_history(cov_cfg, start, end, freq="30min"))

    # get_history was called with include_attributes=True
    iface.get_history.assert_called_once()
    call_args = iface.get_history.call_args
    assert call_args.kwargs.get("include_attributes") is True

    # Result is populated — not all-NaN like the v2.38.3 fall-back
    assert not result.isna().all()


def test_fetch_history_keeps_state_path_for_normal_sensor():
    """A regular numeric sensor (``sensor.outdoor_temp``) shouldn't
    invoke the attribute path even if future_value_key happens to
    be set. Back-compat for the legacy numeric-sensor case."""
    iface = MagicMock()
    iface.get_history = AsyncMock(return_value=[
        {"last_changed": "2026-05-01T12:00:00+00:00", "state": "14.5"},
    ])
    resolver = CovariateResolver(iface)

    cov_cfg = {
        "entity_id": "sensor.outdoor_temp",
        "name": "outdoor_temp",
        "future_value_key": "value",  # set but irrelevant for non-weather
    }
    from datetime import datetime, timezone
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    end = datetime(2026, 5, 2, tzinfo=timezone.utc)
    _run(resolver.fetch_history(cov_cfg, start, end, freq="30min"))

    # get_history called with include_attributes=False (the default)
    assert iface.get_history.call_args.kwargs.get("include_attributes") is False


def test_fetch_history_keeps_state_path_for_weather_without_value_key():
    """A weather entity without ``future_value_key`` (someone added
    it as role:lagged without specifying a metric) still uses the
    state path. We can't guess which attribute they want, so let
    fetch_history return 0% coverage and v2.38.3's empty-column
    guard handle it from there."""
    iface = MagicMock()
    iface.get_history = AsyncMock(return_value=[])
    resolver = CovariateResolver(iface)

    cov_cfg = {
        "entity_id": "weather.met_office_balsham",
        "name": "met_office_balsham",
    }
    from datetime import datetime, timezone
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    end = datetime(2026, 5, 2, tzinfo=timezone.utc)
    _run(resolver.fetch_history(cov_cfg, start, end, freq="30min"))

    assert iface.get_history.call_args.kwargs.get("include_attributes") is False
