"""Holdout display assembly from windowed multi-horizon predictions (v2.40.4).

Neural backends predict the holdout via sliding windows, so the trailing
max_horizon-1 points have no h=1 window. _holdout_display_from_windows fills
them from the last window's h=2..H outputs so neural lines span the whole
holdout (the "LSTM/CNN not as far along as LightGBM" artifact), instead of
stopping ~future_periods points short.
"""
import numpy as np

from ml_forecast_lab.main import _holdout_display_from_windows


def test_tail_filled_from_last_window_multihorizon():
    y_p = np.array([
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33],
    ], dtype=np.float32)  # 3 windows, H=4
    out = _holdout_display_from_windows(y_p, target_len=6)  # 3 + (4-1)
    # head = h=1 column; tail = last window's h=2..H
    np.testing.assert_array_equal(out, [10, 20, 30, 31, 32, 33])
    assert not np.isnan(out).any(), "neural line should span the full holdout"


def test_no_tail_when_lengths_match():
    y_p = np.array([[10, 11], [20, 21], [30, 31]], dtype=np.float32)
    out = _holdout_display_from_windows(y_p, target_len=3)
    np.testing.assert_array_equal(out, [10, 20, 30])


def test_one_dimensional_prediction_padded_with_nan():
    y_p = np.array([5, 6, 7], dtype=np.float32)
    out = _holdout_display_from_windows(y_p, target_len=5)
    np.testing.assert_array_equal(out[:3], [5, 6, 7])
    assert np.isnan(out[3]) and np.isnan(out[4])


def test_single_horizon_tail_stays_nan():
    """H=1 has no h>=2 outputs to fill the tail with — it stays NaN
    (the chart shows the gap rather than fabricating values)."""
    y_p = np.array([[10], [20], [30]], dtype=np.float32)
    out = _holdout_display_from_windows(y_p, target_len=5)
    np.testing.assert_array_equal(out[:3], [10, 20, 30])
    assert np.isnan(out[3]) and np.isnan(out[4])


def test_tail_longer_than_horizon_partial_fill():
    """Defensive: if the gap somehow exceeds H-1, fill what we can, NaN the
    rest — never index past the last window's horizon."""
    y_p = np.array([[10, 11, 12]], dtype=np.float32)  # 1 window, H=3
    out = _holdout_display_from_windows(y_p, target_len=6)
    # out[0]=h1; out[1:3]=last[1:3]; out[3:] no data -> NaN
    np.testing.assert_array_equal(out[:3], [10, 11, 12])
    assert np.isnan(out[3:]).all()
