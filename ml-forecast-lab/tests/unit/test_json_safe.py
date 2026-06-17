"""Regression tests for the JSON-response NaN/Inf sanitiser.

The Forecast Comparison tab computes floats (MAE, bias, ratios, per-bin
means over possibly-empty groups) that can come out NaN. Starlette's
``JSONResponse`` serialises with ``json.dumps(allow_nan=True)``, so a NaN
is emitted as the bare token ``NaN`` — valid Python, invalid JSON. Safari's
``response.json()`` then throws ``SyntaxError: The string did not match the
expected pattern.`` and the whole tab fails to load. ``_json_safe`` scrubs
non-finite floats to ``None`` at the response boundary; these tests pin
that so the bug can't silently return.
"""

from __future__ import annotations

import json
import math

import pytest
from starlette.responses import JSONResponse

from ml_forecast_lab.web.app import SafeJSONResponse, _json_safe


def _is_strict_json(obj) -> bool:
    """True iff obj serialises as spec-valid JSON (no NaN/Inf tokens)."""
    try:
        json.dumps(obj, allow_nan=False)
        return True
    except ValueError:
        return False


def test_nan_becomes_none():
    assert _json_safe(float("nan")) is None


def test_pos_and_neg_inf_become_none():
    assert _json_safe(float("inf")) is None
    assert _json_safe(float("-inf")) is None


def test_finite_floats_pass_through_unchanged():
    assert _json_safe(0.0) == 0.0
    assert _json_safe(-3.14) == -3.14
    assert _json_safe(1e9) == 1e9


def test_non_floats_pass_through():
    assert _json_safe("hi") == "hi"
    assert _json_safe(7) == 7
    assert _json_safe(None) is None
    assert _json_safe(True) is True


def test_nested_structures_are_scrubbed():
    raw = {
        "overlay": {"actual": [1.0, float("nan"), 3.0], "app": [float("inf"), 2.0]},
        "comparisons": [
            {"mae": float("nan"), "bias": -0.5, "label": "Solcast"},
            {"mae": 0.2, "rmse": float("-inf")},
        ],
        "days": 30,
        "units": "kW",
    }
    assert not _is_strict_json(raw), "fixture should start invalid"

    safe = _json_safe(raw)
    assert _is_strict_json(safe), "sanitised payload must be spec-valid JSON"
    assert safe["overlay"]["actual"] == [1.0, None, 3.0]
    assert safe["overlay"]["app"] == [None, 2.0]
    assert safe["comparisons"][0]["mae"] is None
    assert safe["comparisons"][0]["bias"] == -0.5
    assert safe["comparisons"][1]["rmse"] is None
    # Non-float scalars are untouched.
    assert safe["days"] == 30
    assert safe["units"] == "kW"


def test_tuples_become_lists_and_are_scrubbed():
    safe = _json_safe((1.0, float("nan")))
    assert safe == [1.0, None]


def test_round_trips_through_strict_parser():
    """End-to-end: a payload with NaN survives a strict dumps/loads cycle
    after sanitising — exactly what the browser's JSON parser does."""
    payload = _json_safe({"x": [float("nan"), 1.5], "y": float("inf")})
    reparsed = json.loads(json.dumps(payload, allow_nan=False))
    assert reparsed == {"x": [None, 1.5], "y": None}
    assert all(
        v is None or math.isfinite(v)
        for v in [reparsed["x"][0], reparsed["x"][1], reparsed["y"]]
        if isinstance(v, float)
    )


# ----------------------------------------------------------------------
# SafeJSONResponse — the response class used by the data endpoints.
# ----------------------------------------------------------------------

def test_plain_jsonresponse_render_rejects_nan():
    """Pin the failure we are fixing: Starlette renders with
    allow_nan=False, so a NaN raises at render time (after the endpoint
    has already returned) → unhandled 500 with a non-JSON body."""
    with pytest.raises(ValueError):
        JSONResponse(content={"x": float("nan")}).render({"x": float("nan")})


def test_safe_jsonresponse_render_survives_nan_and_inf():
    payload = {
        "overlay": {"actual": [1.0, float("nan")], "app": [float("inf")]},
        "mae": float("-inf"),
        "label": "Solcast",
        "n": 12,
    }
    body = SafeJSONResponse(content=payload).render(payload)
    parsed = json.loads(body)  # strict parse, mirrors the browser
    assert parsed["overlay"]["actual"] == [1.0, None]
    assert parsed["overlay"]["app"] == [None]
    assert parsed["mae"] is None
    assert parsed["label"] == "Solcast"
    assert parsed["n"] == 12


def test_safe_jsonresponse_finite_payload_unchanged():
    payload = {"a": 1.5, "b": [0.0, -2.0], "c": "ok", "d": None}
    parsed = json.loads(SafeJSONResponse(content=payload).render(payload))
    assert parsed == payload


def test_full_request_cycle_nan_does_not_500():
    """End-to-end through the ASGI stack: a route returning NaN via the
    module's NaN-safe JSONResponse returns 200 with valid JSON (null),
    while the unpatched Starlette response 500s during render — which is
    what produced the client-side SyntaxError on WebKit."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from ml_forecast_lab.web.app import JSONResponse as SafeResponse

    app = FastAPI()

    @app.get("/safe")
    def _safe():
        return SafeResponse(content={"mae": float("nan"), "n": 3})

    @app.get("/unsafe")
    def _unsafe():
        return JSONResponse(content={"mae": float("nan"), "n": 3})

    client = TestClient(app, raise_server_exceptions=False)

    r = client.get("/safe")
    assert r.status_code == 200
    assert r.json() == {"mae": None, "n": 3}

    # The base Starlette response raises during render -> 500 (the bug).
    assert client.get("/unsafe").status_code == 500
