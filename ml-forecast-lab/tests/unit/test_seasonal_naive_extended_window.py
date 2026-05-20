"""Regression tests for the v2.38.6 SeasonalNaive extended-window fix.

Bug: when any ``role: future`` covariate was configured, the
benchmark pipeline called ``create_sliding_windows`` with
``future_features_df`` set, which appends ``max(horizon_steps)``
future positions to every window. The target channel's slot in
those future positions is always a zero placeholder (only the
future-known covariate channels get populated by name match).

Pre-v2.38.6 ``SeasonalNaive._per_window_predict`` used
``seq_len = len(target_series)`` for its lookback indexing, so
every seasonal reference indexed into the zero-padded future
block. The result: SeasonalNaive returned 0 for every horizon,
showing as a flat-zero line on the holdout chart while every
other model produced a normal daily PV bell curve. Users
reported "is this a bigger bug?" — and yes it was.

The v2.38.6 fix: capture ``past_window_size`` from fit kwargs
when ``extended_window=True``, then confine all index arithmetic
in ``_per_window_predict`` to ``[0, past_window_size)`` so
lookbacks always land in real past data.

These tests pin:
- Legacy past-only mode unchanged (back-compat).
- Extended-window mode produces non-zero predictions matching the
  past block's daily cycle.
- The exact "user PV chart" reproducer: a 48-period daily cycle
  in the past block, future block all zero, should produce a
  bell-curve prediction not a flat-zero line.
"""

from __future__ import annotations

import numpy as np

from ml_forecast_lab.models.seasonal_naive_backend import SeasonalNaiveModel


def _bell_curve(n_period: int = 48) -> np.ndarray:
    """48-step half-sine over [0, π) — peaks at midday, zero at the ends."""
    t = np.linspace(0.0, np.pi, n_period, endpoint=False)
    return np.maximum(np.sin(t), 0.0).astype(np.float32)


def test_legacy_past_only_mode_unchanged():
    """Without ``extended_window``, predict_sequence behaves as before:
    look back one period from the end of the window."""
    period = 48
    past = _bell_curve(period)  # one full day
    # Window: past only, 1 channel
    window = past.reshape(period, 1)
    X = window[np.newaxis, :, :]  # (1, 48, 1)

    model = SeasonalNaiveModel(seasonal_period=period)
    # y_train shape (n, n_horizons) — set 48 horizons so per-window predict
    # produces 48 outputs and we can compare to the bell curve.
    n_horizons = period
    y_train = np.tile(past, (10, 1)).astype(np.float32)
    model.fit(np.zeros((10, period), dtype=np.float32), y_train)
    preds = model.predict_sequence(X)

    assert preds.shape == (1, period)
    # Predictions must NOT be all-zero (the bug symptom)
    assert preds.sum() > 0
    # Sanity: predictions should resemble the past bell curve since the
    # daily cycle repeats exactly.
    assert preds.max() > 0.5


def test_extended_window_mode_produces_nonzero_predictions():
    """v2.38.6: with ``extended_window=True`` and ``past_window_size``
    set, lookbacks must confine to the past block. Without the fix,
    every lookback would index into the zero-padded future block and
    every prediction would be 0 — the exact "flat blue line" bug
    the user reported on the holdout chart."""
    period = 48
    past_window_size = period  # 48 past steps
    max_horizon = 96  # typical user config
    n_channels = 1

    # Past block: a real PV-like bell curve. Future block: zeros (the
    # target channel never carries future values).
    past = _bell_curve(period)
    future = np.zeros(max_horizon, dtype=np.float32)
    window_target = np.concatenate([past, future])  # length 144
    window = window_target.reshape(-1, n_channels)
    X = window[np.newaxis, :, :]  # (1, 144, 1)

    model = SeasonalNaiveModel(seasonal_period=period)
    n_horizons = max_horizon
    y_train = np.tile(past, (10, 2)).astype(np.float32)  # (10, 96)
    # Mimic the holdout-neural path's fit kwargs
    model.fit(
        np.zeros((10, period), dtype=np.float32), y_train,
        extended_window=True, past_window_size=past_window_size,
    )
    preds = model.predict_sequence(X)

    assert preds.shape == (1, max_horizon)
    # The headline assertion: predictions must NOT be all zero. Pre-fix
    # they were exactly 0.0 across the whole horizon.
    assert preds.sum() > 0, (
        "SeasonalNaive returned all-zero predictions — the extended-window "
        "regression has reappeared. Check that past_window_size is being "
        "honoured in _per_window_predict."
    )
    # And the bell curve should be visible in the first 48-step horizon
    # (one period out → repeats the past bell)
    assert preds[0, :period].max() > 0.5


def test_extended_window_recursion_propagates_real_values_not_zeros():
    """Horizons beyond ``period`` use the recursion branch
    ``out[h] = out[offset]``. Pre-fix that recursion fed 0 forward
    from h=0..47 (all zeros from the future-block lookup), so
    h=48..95 were also 0. The fix's first 48 horizons read real
    past values; the recursion then propagates those real values
    into h=48..95."""
    period = 48
    past = _bell_curve(period)
    future = np.zeros(period * 2, dtype=np.float32)
    window = np.concatenate([past, future]).reshape(-1, 1)
    X = window[np.newaxis, :, :]  # (1, 144, 1)

    model = SeasonalNaiveModel(seasonal_period=period)
    y_train = np.tile(past, (10, 2)).astype(np.float32)
    model.fit(
        np.zeros((10, period), dtype=np.float32), y_train,
        extended_window=True, past_window_size=period,
    )
    preds = model.predict_sequence(X)[0]  # (96,)

    # Horizons 0..47 read from past directly. Horizons 48..95 recurse
    # through ``out[h + 1 - period]``. The headline assertion is that
    # neither slice is all-zero — the recursion can't propagate
    # zeros from a broken past lookup (the v2.38.6 bug).
    assert preds[:period].sum() > 0
    assert preds[period:].sum() > 0
    # The recursion shifts by one position (the formula in the model
    # uses ``offset = h + 1 - period``, not ``h - period``), so
    # ``preds[period+k] == preds[k+1]`` for k in [0, period-1).
    np.testing.assert_array_equal(preds[period:2 * period - 1], preds[1:period])
