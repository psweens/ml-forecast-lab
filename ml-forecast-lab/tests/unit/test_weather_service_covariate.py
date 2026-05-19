"""Regression tests for the v2.37.8 weather.get_forecasts service-API
path in ``CovariateResolver.fetch_future``.

HA 2023.9+ weather integrations (Met Office DataHub, OpenWeatherMap,
AccuWeather, modern met.no) moved forecasts off the ``forecast`` state
attribute and behind a separate service call. Without service-API
support, the v2.37.5+ future-covariate wiring returned NaN for all
of them — the user's debug bundle showed an empty
``cov_for_inference`` dict and the model fell back to past-only
signal.

These tests pin the contract that:

1. When ``entity_id`` is ``weather.*`` AND ``future_attribute`` is one
   of ``hourly`` / ``daily`` / ``twice_daily``, the resolver calls
   ``api_call("POST", "/api/services/weather/get_forecasts?return_response", ...)``
   instead of fetching a state attribute.
2. The response shape ``{service_response: {entity_id: {forecast: [...]}}, ...}``
   is parsed correctly and aligned to the future_index.
3. Legacy callers (``future_attribute: forecast`` / ``detailedForecast``)
   still use the attribute path — no regression for Solcast / met.no.
4. A missing forecast / failed call returns NaN cleanly (no crash).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.covariates import CovariateResolver


def _run(coro):
    """Run an async coroutine to completion. Used in lieu of
    ``pytest.mark.asyncio`` since the addon's test stack ships only
    pytest-anyio and our tests don't need fancy async fixtures."""
    return asyncio.run(coro)


def _make_resolver(api_response=None, attr_response=None):
    """Build a CovariateResolver with a mocked HA interface. Returns
    (resolver, mock_iface) so tests can assert on which method got
    called."""
    iface = MagicMock()
    iface.api_call = AsyncMock(return_value=api_response)
    iface.get_state = AsyncMock(return_value=attr_response)
    resolver = CovariateResolver(iface)
    return resolver, iface


@pytest.fixture
def future_index():
    return pd.date_range(
        "2026-05-19 12:00", periods=12, freq="30min", tz=None,
    )


def test_weather_service_path_called_for_modern_weather_entity(future_index):
    """``weather.met_office_balsham`` + ``future_attribute: hourly``
    must call the service API, not get_state."""
    api_response = {
        "service_response": {
            "weather.met_office_balsham": {
                "forecast": [
                    {"datetime": "2026-05-19T12:00:00+00:00",
                     "temperature": 14.5, "cloud_coverage": 30},
                    {"datetime": "2026-05-19T13:00:00+00:00",
                     "temperature": 15.1, "cloud_coverage": 40},
                    {"datetime": "2026-05-19T14:00:00+00:00",
                     "temperature": 15.8, "cloud_coverage": 50},
                ]
            }
        }
    }
    resolver, iface = _make_resolver(api_response=api_response)

    cov_cfg = {
        "entity_id": "weather.met_office_balsham",
        "name": "met_office_balsham",
        "future_attribute": "hourly",
        "future_value_key": "cloud_coverage",
    }
    result = _run(resolver.fetch_future(cov_cfg, future_index))

    # Service API was called with the right URL + body
    iface.api_call.assert_called_once()
    call_args = iface.api_call.call_args
    assert call_args.args[0] == "POST"
    assert "weather/get_forecasts" in call_args.args[1]
    assert "return_response" in call_args.args[1]
    assert call_args.kwargs["json_data"]["entity_id"] == "weather.met_office_balsham"
    assert call_args.kwargs["json_data"]["type"] == "hourly"

    # State attribute path was NOT called
    iface.get_state.assert_not_called()

    # Result aligned to future_index, populated with cloud_coverage values
    assert not result.isna().all()
    assert len(result) == len(future_index)


def test_weather_service_path_handles_daily_and_twice_daily(future_index):
    """All three forecast types (``hourly``, ``daily``, ``twice_daily``)
    route through the service API."""
    api_response = {
        "service_response": {
            "weather.test": {"forecast": [
                {"datetime": "2026-05-19T12:00:00+00:00", "temperature": 10},
            ]}
        }
    }
    for forecast_type in ("hourly", "daily", "twice_daily"):
        resolver, iface = _make_resolver(api_response=api_response)
        cov_cfg = {
            "entity_id": "weather.test",
            "future_attribute": forecast_type,
            "future_value_key": "temperature",
        }
        _run(resolver.fetch_future(cov_cfg, future_index))
        iface.api_call.assert_called_once()
        assert iface.api_call.call_args.kwargs["json_data"]["type"] == forecast_type
        iface.get_state.assert_not_called()


def test_legacy_attribute_path_unchanged_for_solcast(future_index):
    """``sensor.solcast_pv_forecast`` + ``future_attribute:
    detailedForecast`` must still hit get_state, not the service API.
    Backwards-compatibility pin."""
    attr_response = [
        {"period_start": "2026-05-19T12:00:00+00:00", "pv_estimate": 2.1},
        {"period_start": "2026-05-19T12:30:00+00:00", "pv_estimate": 2.5},
        {"period_start": "2026-05-19T13:00:00+00:00", "pv_estimate": 2.8},
    ]
    resolver, iface = _make_resolver(attr_response=attr_response)

    cov_cfg = {
        "entity_id": "sensor.solcast_pv_forecast_forecast_today",
        "future_attribute": "detailedForecast",
        "future_value_key": "pv_estimate",
    }
    result = _run(resolver.fetch_future(cov_cfg, future_index))

    iface.get_state.assert_called_once()
    iface.api_call.assert_not_called()
    assert not result.isna().all()


def test_legacy_path_for_weather_entity_with_attribute_forecast(future_index):
    """A `weather.*` entity with ``future_attribute: forecast`` (the
    legacy default, used by pre-2023.9 met.no integrations) still
    routes through get_state — only ``hourly`` / ``daily`` /
    ``twice_daily`` trigger the service path. Lets older HA installs
    keep working."""
    attr_response = [
        {"datetime": "2026-05-19T12:00:00+00:00", "temperature": 14.5},
    ]
    resolver, iface = _make_resolver(attr_response=attr_response)

    cov_cfg = {
        "entity_id": "weather.legacy_metno",
        "future_attribute": "forecast",
        "future_value_key": "temperature",
    }
    _run(resolver.fetch_future(cov_cfg, future_index))
    iface.get_state.assert_called_once()
    iface.api_call.assert_not_called()


def test_weather_service_missing_response_returns_nan(future_index):
    """If the service API returns an empty / malformed body, the
    resolver returns an all-NaN series — never crashes the retrain."""
    resolver, iface = _make_resolver(api_response={"service_response": {}})
    cov_cfg = {
        "entity_id": "weather.broken",
        "future_attribute": "hourly",
        "future_value_key": "temperature",
    }
    result = _run(resolver.fetch_future(cov_cfg, future_index))
    assert result.isna().all()


def test_weather_service_call_failure_returns_nan(future_index):
    """If the service API call itself raises, the resolver swallows
    the exception and returns NaN — same defensive pattern as the
    attribute path."""
    iface = MagicMock()
    iface.api_call = AsyncMock(side_effect=RuntimeError("connection refused"))
    iface.get_state = AsyncMock()
    resolver = CovariateResolver(iface)
    cov_cfg = {
        "entity_id": "weather.unreachable",
        "future_attribute": "hourly",
        "future_value_key": "temperature",
    }
    result = _run(resolver.fetch_future(cov_cfg, future_index))
    assert result.isna().all()
    # Service was attempted, get_state was not (different from
    # the attribute-path failure mode).
    iface.api_call.assert_called_once()
    iface.get_state.assert_not_called()
