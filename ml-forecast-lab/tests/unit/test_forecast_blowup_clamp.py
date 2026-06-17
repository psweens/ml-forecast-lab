"""Regression tests for the publish-boundary forecast blow-up clamp.

When ``log_transform`` is on, predictions are inverted with ``np.expm1``.
A diverged log-space value (~70) explodes to ~1e30; with only a lower
(>= 0) clamp at the publish boundary, that garbage was published to Home
Assistant and logged verbatim. ``_clamp_forecast_blowup`` caps the inverted
forecast to a generous multiple (10×) of the largest observed value so a
blow-up can never be published.
"""

from __future__ import annotations

import numpy as np

from ml_forecast_lab.main import (
    FORECAST_BLOWUP_CAP_FACTOR,
    _clamp_forecast_blowup,
)


def test_blowup_is_capped_to_factor_times_ref():
    y = np.array([1.0, 2.0, 5e30], dtype=np.float32)
    out, n, cap = _clamp_forecast_blowup(y, ref_max_display=3.0)
    assert n == 1
    assert cap == FORECAST_BLOWUP_CAP_FACTOR * 3.0  # 30.0
    assert out[2] == np.float32(30.0)
    assert out[0] == np.float32(1.0) and out[1] == np.float32(2.0)


def test_plausible_forecast_is_untouched():
    # Anything up to the (generous) cap is kept — only blow-ups are clamped.
    y = np.array([0.0, 5.0, 30.0], dtype=np.float32)
    out, n, cap = _clamp_forecast_blowup(y, ref_max_display=3.0)
    assert n == 0
    assert list(out) == [0.0, 5.0, 30.0]


def test_no_cap_without_a_reference():
    y = np.array([1.0, 5e30], dtype=np.float32)
    out, n, cap = _clamp_forecast_blowup(y, ref_max_display=None)
    assert n == 0 and cap is None
    assert out[1] == np.float32(5e30)  # untouched (can't judge without a ref)


def test_no_cap_for_degenerate_reference():
    y = np.array([1.0, 1e30], dtype=np.float32)
    for bad in (0.0, -2.0, float("nan"), float("inf")):
        _, n, cap = _clamp_forecast_blowup(y, ref_max_display=bad)
        assert n == 0 and cap is None


def test_non_finite_predictions_not_miscounted():
    # inf/nan predictions aren't counted as "clamped" (isfinite gate); the
    # finite blow-up still is.
    y = np.array([np.inf, np.nan, 5e30], dtype=np.float32)
    out, n, cap = _clamp_forecast_blowup(y, ref_max_display=2.0)
    assert n == 1  # only the finite 5e30
    assert out[2] == np.float32(20.0)
