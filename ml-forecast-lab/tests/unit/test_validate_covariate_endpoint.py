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
