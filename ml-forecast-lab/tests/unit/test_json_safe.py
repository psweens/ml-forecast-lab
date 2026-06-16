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

from ml_forecast_lab.web.app import _json_safe


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
