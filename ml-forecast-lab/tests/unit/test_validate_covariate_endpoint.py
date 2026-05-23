"""Tests for the v2.38.7 covariate-validation classifier.

The new ``/api/covariates/validate`` endpoint is a thin transport
wrapper around ``classify_covariate_state``, a pure function that
takes a raw HA state object and decides ok / partial / broken.
Testing the pure function avoids any FastAPI / aiohttp test
dependencies (httpx isn't installed in this env) and pins the
exact contract the UI chip relies on.
"""

from __future__ import annotations

import pytest

from ml_forecast_lab.web.app import classify_covariate_state


def _state(value, attrs=None):
    return {
        "state": value,
        "last_changed": "2026-05-19T12:00:00+00:00",
        "attributes": attrs or {},
    }


def test_numeric_state_no_future_attr_is_ok():
    """``sensor.outdoor_temp`` with state '14.5' and no
    future_attribute → ok. Bread-and-butter case."""
    out = classify_covariate_state(
        entity_id="sensor.outdoor_temp",
        state_obj=_state("14.5"),
    )
    assert out["ok"] is True
    assert out["status"] == "ok"
    assert out["state_value"] == 14.5


def test_categorical_state_no_attribute_is_broken():
    """``weather.met_office_balsham`` with state='partlycloudy' and
    no future_attribute → broken. Surfaces at config time the trap
    that v2.38.4 fixed in the back-end (weather-entity history
    silently producing 0% coverage)."""
    out = classify_covariate_state(
        entity_id="weather.met_office_balsham",
        state_obj=_state("partlycloudy", attrs={"temperature": 14.5}),
    )
    assert out["ok"] is False
    assert out["status"] == "broken"


def test_weather_entity_with_value_key_attribute_path_is_ok():
    """``weather.met_office_balsham`` + ``future_value_key=temperature``
    routes lagged history through the v2.38.4 attribute path. The
    classifier should reflect that and return ok."""
    out = classify_covariate_state(
        entity_id="weather.met_office_balsham",
        state_obj=_state("partlycloudy",
                         attrs={"temperature": 14.5, "cloud_coverage": 30}),
        future_value_key="temperature",
    )
    assert out["ok"] is True
    assert out["status"] == "ok"


def test_predbat_rates_flat_dict_attribute_is_ok():
    """``predbat.rates`` with state='22.29' and a flat-dict
    ``attributes.results`` of {iso_ts: float} (the user's actual
    config) → ok with a first-entry preview."""
    out = classify_covariate_state(
        entity_id="predbat.rates",
        state_obj=_state("22.29", attrs={
            "results": {
                "2026-05-19T00:00:00+00:00": 22.29,
                "2026-05-19T00:30:00+00:00": 21.21,
                "2026-05-19T01:00:00+00:00": 20.55,
            },
        }),
        future_attribute="results",
    )
    assert out["ok"] is True
    assert out["status"] == "ok"
    assert out["attribute_preview"] == 22.29


def test_future_attribute_missing_is_partial():
    """Lagged numeric but the requested future_attribute doesn't
    exist → partial. Chart will get history but future block is NaN."""
    out = classify_covariate_state(
        entity_id="predbat.rates",
        state_obj=_state("22.29", attrs={}),
        future_attribute="results",
    )
    assert out["ok"] is True
    assert out["status"] == "partial"
    assert "didn't parse" in out["message"]


def test_future_attribute_unparseable_is_partial():
    """``attributes.results`` is a scalar (not list-of-dict / dict-of-dt)
    → parser returns None → partial."""
    out = classify_covariate_state(
        entity_id="predbat.rates",
        state_obj=_state("22.29", attrs={"results": 5.0}),
        future_attribute="results",
    )
    assert out["ok"] is True
    assert out["status"] == "partial"


def test_value_key_wrong_for_list_attribute_is_partial_then_ok_with_key():
    """The user's original concern: ``attr.results`` is a list of
    dicts and ``future_value_key`` is wrong / missing. Should be
    partial. Setting the right key flips it to ok."""
    state = _state("22.29", attrs={
        "forecast": [
            {"time": "2026-05-19T00:00:00", "rate": 22.29},
            {"time": "2026-05-19T00:30:00", "rate": 21.21},
        ],
    })
    # rate isn't in the auto-detect VAL_KEYS, parser returns None.
    out = classify_covariate_state(
        entity_id="predbat.rates",
        state_obj=state,
        future_attribute="forecast",
    )
    assert out["status"] == "partial"

    # With the explicit key, parses cleanly.
    out2 = classify_covariate_state(
        entity_id="predbat.rates",
        state_obj=state,
        future_attribute="forecast",
        future_value_key="rate",
    )
    assert out2["status"] == "ok"
    assert out2["attribute_preview"] == 22.29


def test_unavailable_state_no_attribute_is_broken():
    """An entity that exists but is currently 'unavailable' has no
    usable lagged data. Without a future_attribute fallback it's
    broken."""
    out = classify_covariate_state(
        entity_id="sensor.broken",
        state_obj=_state("unavailable"),
    )
    assert out["ok"] is False
    assert out["status"] == "broken"


def test_weather_service_future_attribute_is_ok():
    """HA 2023.9+ weather entities expose hourly/daily/twice_daily
    forecasts via the ``weather.get_forecasts`` service call, not as
    state attributes. The state-only validator shouldn't false-flag
    them as partial when ``attrs.get('hourly')`` returns None — the
    resolver fetches via the service path."""
    out = classify_covariate_state(
        entity_id="weather.met_office_balsham",
        state_obj=_state("partlycloudy", attrs={"uv_index": 4.2}),
        future_attribute="hourly",
        future_value_key="uv_index",
    )
    assert out["ok"] is True
    assert out["status"] == "ok"
    assert "weather.get_forecasts(hourly)" in out["message"]


def test_weather_service_future_with_no_lagged_path_returns_partial():
    """Same as above but without a weather-attr-history path for
    lagged: the future side is service-fetched (ok) but the lagged
    channel will be empty because state is categorical and no
    future_value_key was set to route through the attribute path.
    v2.39.3: surface this as ``partial`` (per the docstring contract)
    rather than misleadingly green-chipping a row whose past channel
    the empty-column guard will end up zero-filling at train time."""
    out = classify_covariate_state(
        entity_id="weather.met_office_balsham",
        state_obj=_state("partlycloudy", attrs={}),
        future_attribute="daily",
    )
    assert out["ok"] is True
    assert out["status"] == "partial"
    assert "lagged history will be empty" in out["message"]


def test_partial_message_describes_weather_attr_lagged_path():
    """When state is categorical but lagged history pulls via the
    weather-attr-history path, the partial message must NOT say
    ``last=None`` — that was misleading. It should describe the
    actual lagged source (the attribute name + current value)."""
    out = classify_covariate_state(
        entity_id="weather.met_office_balsham",
        state_obj=_state("partlycloudy", attrs={
            "uv_index": 4.2,
            # Non-service future attr that won't parse (scalar):
            "forecast_summary": 5.0,
        }),
        future_attribute="forecast_summary",
        future_value_key="uv_index",
    )
    assert out["status"] == "partial"
    assert "last=None" not in out["message"]
    assert "uv_index" in out["message"]


def test_weather_string_numeric_attribute_no_longer_false_broken():
    """v2.39.3 bug 10: weather integrations frequently store attribute
    numerics as strings (OpenWeatherMap returns ``temperature: '16.5'``).
    The production resolver tolerates strings via ``state_to_float``;
    the validator must too — pre-v2.39.3 it used ``isinstance(int, float)``
    and red-chipped working covariates."""
    out = classify_covariate_state(
        entity_id="weather.openweathermap",
        state_obj=_state("partlycloudy", attrs={"temperature": "16.5"}),
        future_value_key="temperature",
    )
    assert out["ok"] is True
    assert out["status"] != "broken", (
        "string-numeric attributes are parseable via state_to_float and "
        "must not be flagged broken"
    )


def test_weather_service_legacy_attribute_still_parses():
    """Non-service future attributes on a weather entity (e.g. the
    legacy ``forecast`` attribute Solcast/custom integrations still
    expose) must continue to be parsed from state — the service
    short-circuit only applies to {hourly, daily, twice_daily}."""
    out = classify_covariate_state(
        entity_id="weather.met_office_balsham",
        state_obj=_state("partlycloudy", attrs={
            "temperature": 14.5,
            "forecast": [
                {"datetime": "2026-05-19T00:00:00", "temperature": 14.5},
                {"datetime": "2026-05-19T01:00:00", "temperature": 13.8},
            ],
        }),
        future_attribute="forecast",
        future_value_key="temperature",
    )
    assert out["ok"] is True
    assert out["status"] == "ok"
    assert out["attribute_preview"] == 14.5
