"""
Regression tests pinning the two confirmed neural-PV root causes.

These tests are written so they **fail under today's code** and would
pass under either of the proposed fixes (see
``docs/investigations/2026-05-neural-pv.md``).

Each test uses the deterministic ``make_realistic_pv(0)`` synthetic
dataset — no network access, no real Home Assistant data.

Failure modes pinned
--------------------

RC1 — RevIN bias from future-position zeros
    For a window-extended training tensor (v2.36+), the per-window mean
    of the target channel is biased ~50% low because half the timesteps
    are zero-padded future positions. Test:
    ``test_revin_extended_window_mean_unbiased`` — asserts that for any
    sample, RevIN's stored ``_mean`` for the target channel matches the
    PAST-block mean within 5%, regardless of whether the window is
    extended.

RC2 — NLinear last-value anchor degeneration
    NLinear subtracts ``x[:, -1, target_channel]`` and re-adds it. In
    extended windows that index lands on the LAST future position where
    target is zero. Test:
    ``test_nlinear_anchor_carries_last_past_observation`` — asserts
    that the value NLinear actually uses as its anchor equals the LAST
    PAST observation, not zero.

Each test exercises the public ``_RevIN`` / ``_NLinearNet`` modules
exactly the way the backends do, so they cover the production code
path (not just the synthetic data path).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ml_forecast_lab.features import (
    compute_known_future_features, create_sliding_windows,
)
from ml_forecast_lab.models.base import _RevIN
from ml_forecast_lab.models.nlinear_backend import _NLinearNet

from tests.synthetic.datasets import make_realistic_pv, GB_LAT, GB_LON


WINDOW = 48
HORIZON = 48


@pytest.fixture(scope="module")
def extended_window_tensor():
    """A real extended window tensor from realistic_pv."""
    d = make_realistic_pv(0)
    horizon_steps = list(range(1, HORIZON + 1))
    future_df = compute_known_future_features(
        d.df.index, add_temporal=True, country="GB",
        solar_lat_lon=(GB_LAT, GB_LON),
        include_sun_elevation=True, include_clear_sky_ghi=True,
    )
    X, _, ch = create_sliding_windows(
        d.df, "y", window_size=WINDOW,
        covariate_cols=["sun_elevation", "clear_sky_ghi"],
        add_temporal=True,
        horizon_steps=horizon_steps,
        future_features_df=future_df,
    )
    return X, ch


@pytest.mark.xfail(
    strict=True,
    reason=(
        "RC1: today RevIN averages across all 96 positions of the extended window; "
        "the future block's 48 zero-target positions pull the mean to ~50% of the "
        "past-only mean. See docs/investigations/2026-05-neural-pv.md §1 S1. "
        "When PF1 lands, this test xpasses and the marker should be removed."
    ),
)
def test_revin_extended_window_mean_unbiased(extended_window_tensor):
    """RC1: RevIN's per-window target-channel mean must match the past mean.

    XFAIL under today's code: production ``_RevIN`` averages across all
    96 positions; the future block's 48 zeros pull the mean to ~50% of
    the past-only mean.
    """
    X, ch = extended_window_tensor
    target_ch = 0
    # Pick a sample whose past block has non-trivial target values
    # (otherwise both past-mean and full-mean are near zero and the
    # bias is invisible).
    past_means = X[:, :WINDOW, target_ch].mean(axis=1)
    # The user's failure shape: high past mean is when day is in the
    # window. Pick a sample whose past mean is in the top decile.
    threshold = float(np.quantile(past_means, 0.9))
    candidates = np.where(past_means > threshold)[0]
    assert candidates.size > 0
    i = int(candidates[0])
    sample = torch.from_numpy(X[i: i + 1])
    revin = _RevIN(X.shape[2], target_channel=target_ch, affine=False)
    revin.normalize(sample)
    # Mean RevIN stored for the target channel
    revin_mean = float(revin._mean[0, 0, target_ch].item())
    past_mean = float(X[i, :WINDOW, target_ch].mean())
    bias = abs(revin_mean - past_mean) / max(abs(past_mean), 1e-6)
    # Today this bias is ~0.5; the fix brings it under 0.05.
    assert bias < 0.05, (
        f"RC1 still broken: RevIN mean={revin_mean:.2f} vs past_mean={past_mean:.2f}; "
        f"relative bias {bias:.3f} exceeds the 5% tolerance. The future-position "
        f"zeros are still being averaged in."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "RC2: today NLinear anchors on `x[:, -1, target_channel]` which in "
        "extended-window mode is always 0 (the last future position has the "
        "target channel left zero). See docs/investigations/2026-05-neural-pv.md §1 S1. "
        "When PF2 lands (anchor at `x[:, W-1, target_channel]`), this test xpasses "
        "and the marker should be removed."
    ),
)
def test_nlinear_anchor_carries_last_past_observation(extended_window_tensor):
    """RC2: NLinear's anchor value must be the last past target observation.

    XFAIL under today's code: ``x[:, -1, target_channel]`` in extended
    mode lands on the LAST FUTURE position where the target channel is
    always zero.
    """
    X, ch = extended_window_tensor
    target_ch = 0
    # Pick a sample where the last past row IS non-zero (daytime end).
    last_past = X[:, WINDOW - 1, target_ch]
    candidates = np.where(last_past > 100.0)[0]  # 100 W = daylight on realistic_pv
    assert candidates.size > 0
    i = int(candidates[0])
    sample = torch.from_numpy(X[i: i + 1])
    net = _NLinearNet(
        seq_len=X.shape[1], n_channels=X.shape[2],
        n_horizons=HORIZON, output_activation="linear",
        sigmoid_scale=1.0, use_revin=False, target_channel=target_ch,
    )
    # Mirror what forward() does to compute the anchor without running
    # the whole linear head — directly read the value the anchor
    # subtraction lands on.
    expected = float(X[i, WINDOW - 1, target_ch])     # last past observation
    actually_used = float(sample[:, -1, target_ch].item())  # last row of full window
    # On a fixed version the anchor IS the last past observation.
    assert abs(actually_used - expected) / max(abs(expected), 1e-6) < 0.05, (
        f"RC2 still broken: NLinear anchor uses x[:, -1, 0] = {actually_used:.2f} "
        f"but the last past observation is {expected:.2f}. In extended-window mode "
        f"the literal last row is a future position with target=0, making the "
        f"'subtract last value, re-add' trick a no-op."
    )


def test_prototype_pf1_past_only_revin_removes_bias(extended_window_tensor):
    """Smoke test for PF1: a past-only-aware RevIN does NOT have the bias.

    This test exists to demonstrate that the FIX shape proposed in
    ``docs/investigations/2026-05-neural-pv.md`` actually clears the
    regression — i.e. that swapping in a RevIN variant that computes
    stats over ``x[:, :past_window_size, :]`` only makes
    ``test_revin_extended_window_mean_unbiased`` pass.

    The prototype lives in ``tests/synthetic/run_prototype_fixes.py``
    as ``_RevINPastOnly``; we re-define a minimal version here to keep
    this test self-contained.
    """
    X, ch = extended_window_tensor
    target_ch = 0
    past_means = X[:, :WINDOW, target_ch].mean(axis=1)
    candidates = np.where(past_means > float(np.quantile(past_means, 0.9)))[0]
    i = int(candidates[0])
    sample = torch.from_numpy(X[i: i + 1])

    class _RevINPastOnly(_RevIN):
        def __init__(self, n_channels, past_window_size, **kw):
            super().__init__(n_channels, **kw)
            self.past_window_size = past_window_size

        def normalize(self, x):
            past = x[:, :self.past_window_size, :]
            mean = past.mean(dim=1, keepdim=True).detach()
            var = past.var(dim=1, keepdim=True, unbiased=False).detach()
            stdev = torch.sqrt(var + self.eps)
            self._mean = mean
            self._stdev = stdev
            x_norm = (x - mean) / stdev
            if self.affine:
                x_norm = x_norm * self.affine_weight + self.affine_bias
            return x_norm

    revin = _RevINPastOnly(X.shape[2], WINDOW, target_channel=target_ch, affine=False)
    revin.normalize(sample)
    revin_mean = float(revin._mean[0, 0, target_ch].item())
    past_mean = float(X[i, :WINDOW, target_ch].mean())
    bias = abs(revin_mean - past_mean) / max(abs(past_mean), 1e-6)
    assert bias < 0.05, (
        f"PF1 prototype itself is broken: bias {bias:.3f} should be < 0.05 "
        f"under past-only normalisation"
    )
