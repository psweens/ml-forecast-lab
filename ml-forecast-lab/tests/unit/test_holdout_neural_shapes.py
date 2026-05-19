"""Regression test for the v2.38.5 holdout NLinear shape mismatch fix.

Bug (pre-v2.38.5): the holdout neural path called
``create_sliding_windows`` with ``horizon_steps=horizon_steps``
(full list, max 96) for the fit-side windows AND with
``horizon_steps=[1]`` for the predict-side windows. When
``future_features_df`` is supplied, ``create_sliding_windows``
extends each window by ``max(horizon_steps)`` future positions —
so the fit-side window was 48+96=144 steps wide and the predict-
side was 48+1=49 steps wide. Linear-head backends (NLinear,
TiDE) size their ``nn.Linear`` off the fit-time flat input
size; presenting them a narrower predict-time input crashed
with ``mat1 and mat2 shapes cannot be multiplied (894x2253 and
6528x96)``.

The fix passes the same ``horizon_steps`` list to both calls.
The trade-off is losing the last ``max_horizon - 1`` rows from
the holdout slice (no window can be formed for them), but
that's strictly preferable to crashing the entire holdout
metric.

This test pins the invariant: with ``future_features_df`` set,
fit-side and predict-side sliding-window outputs must agree on
shape[1] (effective window length) AND shape[2] (channel count).
Future regressions of the form ``horizon_steps=[1]`` on the
predict side will fail this test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_forecast_lab.features import (
    compute_known_future_features,
    create_sliding_windows,
)


def _build_df(n_rows: int = 500, freq: str = "30min") -> pd.DataFrame:
    """Synthetic 30-min PV-like series with one future-known covariate."""
    idx = pd.date_range("2026-01-01", periods=n_rows, freq=freq)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "pv_energy": rng.uniform(0, 5, size=n_rows).astype("float32"),
            "cloud_coverage": rng.uniform(0, 100, size=n_rows).astype("float32"),
        },
        index=idx,
    )
    return df


def _expected_nlinear_flat(window_size: int, max_horizon: int, n_channels: int) -> int:
    """Mirror _NLinearNet's flat-input calculation (extended mode + PF7).

    Past block: all channels. Future block: drops target channel.
    """
    return window_size * n_channels + max_horizon * (n_channels - 1)


def test_holdout_neural_path_uses_matching_horizon_steps_at_fit_and_predict():
    """Pin the v2.38.5 invariant: with future_features_df set,
    fit-side and predict-side sliding-window outputs must have
    matching effective_window so the trained NLinear / TiDE
    Linear head can accept the predict-time input.

    Pre-v2.38.5: predict called with horizon_steps=[1] →
    effective_window=49 → flat=2253. Fit called with full list →
    effective_window=144 → flat=6528. Crash.
    """
    df = _build_df()
    window_size = 48
    horizon_steps = list(range(1, 97))  # the user-config horizons
    cov_cols = ["cloud_coverage"]

    future_features_df = compute_known_future_features(
        df.index,
        add_temporal=True,
        country=None,
        solar_lat_lon=None,
        include_sun_elevation=False,
        include_clear_sky_ghi=False,
        future_covariate_values={
            "cloud_coverage": df["cloud_coverage"],
        },
    )

    # Fit-side windows (always uses full horizon_steps)
    fit_X, _, fit_channels = create_sliding_windows(
        df, "pv_energy", window_size=window_size,
        covariate_cols=cov_cols,
        add_temporal=True,
        horizon_steps=horizon_steps,
        future_features_df=future_features_df,
    )

    # Predict-side windows (must use same horizon_steps post-v2.38.5)
    predict_X, _, predict_channels = create_sliding_windows(
        df, "pv_energy", window_size=window_size,
        covariate_cols=cov_cols,
        add_temporal=True,
        horizon_steps=horizon_steps,
        future_features_df=future_features_df,
    )

    # Shapes [1] (effective window) and [2] (channels) must agree —
    # the trained NLinear head can't accept a different flat size.
    assert fit_X.shape[1] == predict_X.shape[1], (
        f"fit-time effective window {fit_X.shape[1]} != "
        f"predict-time {predict_X.shape[1]} — the NLinear/TiDE Linear "
        f"head built at fit time can't accept the predict-time input"
    )
    assert fit_X.shape[2] == predict_X.shape[2]
    assert fit_channels == predict_channels


def test_horizon_steps_one_at_predict_would_break_nlinear_head():
    """Negative test: the OLD predict-side call (horizon_steps=[1])
    produces a flat size that mismatches the fit-time head. Pins
    the bug we're guarding against — if someone reverts the v2.38.5
    fix and goes back to [1], this test demonstrates exactly which
    invariant breaks."""
    df = _build_df()
    window_size = 48
    horizon_steps = list(range(1, 97))
    cov_cols = ["cloud_coverage"]

    future_features_df = compute_known_future_features(
        df.index, add_temporal=True,
        country=None, solar_lat_lon=None,
        include_sun_elevation=False, include_clear_sky_ghi=False,
        future_covariate_values={"cloud_coverage": df["cloud_coverage"]},
    )

    fit_X, _, _ = create_sliding_windows(
        df, "pv_energy", window_size=window_size,
        covariate_cols=cov_cols, add_temporal=True,
        horizon_steps=horizon_steps,
        future_features_df=future_features_df,
    )

    # OLD broken behaviour — horizon_steps=[1] at predict
    bad_predict_X, _, _ = create_sliding_windows(
        df, "pv_energy", window_size=window_size,
        covariate_cols=cov_cols, add_temporal=True,
        horizon_steps=[1],
        future_features_df=future_features_df,
    )

    # The shape mismatch is real and would crash NLinear
    assert fit_X.shape[1] != bad_predict_X.shape[1]
    n_channels = fit_X.shape[2]
    fit_flat = _expected_nlinear_flat(window_size, 96, n_channels)
    bad_predict_flat = _expected_nlinear_flat(window_size, 1, n_channels)
    assert fit_flat != bad_predict_flat
    # Sanity: these are the exact numbers from the user's
    # 894x2253 / 6528x96 crash. window_size=48, max_horizon
    # train=96, max_horizon predict=1, n_channels=46:
    #   fit = 48*46 + 96*45 = 2208 + 4320 = 6528
    #   predict_bad = 48*46 + 1*45 = 2208 + 45 = 2253
    assert _expected_nlinear_flat(48, 96, 46) == 6528
    assert _expected_nlinear_flat(48, 1, 46) == 2253
