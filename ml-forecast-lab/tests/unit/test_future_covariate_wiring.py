"""Regression tests for the future-covariate wiring (v2.37.5).

Pins the contract that user-configured ``role: future`` covariates
reach the neural extended-window head at horizon positions, not
just as past-window lags. Closes the gap that left NLinear / DLinear
/ TSMixer / TiDE information-starved versus the tree path (which
always saw future covariate values via the recursive
``future_cov_values`` dict).

These tests exercise the public feature plumbing
(``compute_known_future_features`` + ``create_sliding_windows`` +
``build_inference_window``) since that's the contract surface the
neural path consumes — they cover the wiring without needing the
full ForecastService / HA stack.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_forecast_lab.features import (
    build_inference_window,
    compute_known_future_features,
    create_sliding_windows,
)


def _make_combined_with_future_cov(n_rows: int = 200) -> pd.DataFrame:
    """Build a combined dataframe with a target plus a future-known
    covariate (simulating e.g. a Solcast PV forecast aligned to the
    target index)."""
    idx = pd.date_range(
        "2026-05-01 00:00", periods=n_rows, freq="30min", tz=None
    )
    rng = np.random.default_rng(0)
    # Smooth daily-cycle target
    hour = idx.hour + idx.minute / 60.0
    target = np.clip(np.sin(np.pi * (hour - 6) / 14), 0, None) * 3.0
    target = target + rng.normal(0, 0.05, n_rows)
    # Solcast-like covariate: similar shape but slightly noisier
    solcast = np.clip(np.sin(np.pi * (hour - 6) / 14), 0, None) * 3.2
    solcast = solcast + rng.normal(0, 0.1, n_rows)
    return pd.DataFrame({
        "target": target,
        "solcast_pv_forecast": solcast,
    }, index=idx)


def test_future_covariate_reaches_horizon_position_in_training_window():
    """A future-known covariate passed via
    ``compute_known_future_features.future_covariate_values`` must
    populate the corresponding channel at future positions of every
    training window — not just the past positions where the lagged
    history sits."""
    df = _make_combined_with_future_cov()
    horizon_steps = list(range(1, 5))  # 4 horizons

    future_features_df = compute_known_future_features(
        df.index,
        add_temporal=True,
        future_covariate_values={"solcast_pv_forecast": df["solcast_pv_forecast"]},
    )
    assert "solcast_pv_forecast" in future_features_df.columns

    seq_X, seq_y, channel_names = create_sliding_windows(
        df, "target", window_size=48,
        covariate_cols=["solcast_pv_forecast"],
        add_temporal=True,
        horizon_steps=horizon_steps,
        future_features_df=future_features_df,
    )
    # 48 past + 4 future = 52 steps
    assert seq_X.shape[1] == 48 + 4
    solcast_ch = channel_names.index("solcast_pv_forecast")

    # For window i, the future-block-row j corresponds to absolute
    # row (i + 48 + j) in df. Pick a sample with non-trivial covariate
    # values (mid-dataset, mid-day-ish) and verify the future positions
    # carry the actual covariate observations from df.
    i = 50
    for j in range(4):
        absolute_row = i + 48 + j
        expected = float(df.iloc[absolute_row]["solcast_pv_forecast"])
        actual = float(seq_X[i, 48 + j, solcast_ch])
        # Float32 round-trip — tolerate 1e-5
        assert abs(actual - expected) < 1e-5, (
            f"Window {i} future-position {j}: solcast channel reads "
            f"{actual}, expected {expected}. Future-covariate wiring "
            f"is dropping values somewhere."
        )


def test_future_covariate_reaches_horizon_position_in_inference_window():
    """At inference, ``compute_known_future_features`` is called with
    a future_index disjoint from the past data — the values come from
    HA's forecast attribute (fetched separately). The wiring must
    place those values in the future positions of the inference
    window."""
    df = _make_combined_with_future_cov()
    # 96 future timestamps starting immediately after df ends
    future_index = pd.date_range(
        start=df.index[-1] + pd.Timedelta(minutes=30),
        periods=96, freq="30min", tz=None,
    )
    # Synthesise a "Solcast forecast" for the future window
    future_hour = future_index.hour + future_index.minute / 60.0
    future_solcast = np.clip(
        np.sin(np.pi * (future_hour - 6) / 14), 0, None
    ) * 2.8
    future_solcast_series = pd.Series(future_solcast, index=future_index)

    future_features_df = compute_known_future_features(
        future_index,
        add_temporal=True,
        future_covariate_values={"solcast_pv_forecast": future_solcast_series},
    )
    seq_X, channel_names = build_inference_window(
        df, "target", window_size=48,
        covariate_cols=["solcast_pv_forecast"],
        add_temporal=True,
        future_features_df=future_features_df,
    )
    # Shape: (1, 48 + 96, n_channels)
    assert seq_X.shape == (1, 144, len(channel_names))
    solcast_ch = channel_names.index("solcast_pv_forecast")

    # Future positions of the inference window must mirror the
    # forecast values
    for j in range(96):
        actual = float(seq_X[0, 48 + j, solcast_ch])
        expected = float(future_solcast_series.iloc[j])
        assert abs(actual - expected) < 1e-5, (
            f"Inference future-position {j}: solcast={actual}, "
            f"expected={expected}"
        )


def test_no_future_covariate_keeps_zero_at_horizon():
    """Sanity check: when future_covariate_values is not passed
    (current v2.37.4 behaviour), the future positions for that
    channel are zero — matches the pre-wiring contract."""
    df = _make_combined_with_future_cov()
    future_features_df = compute_known_future_features(
        df.index, add_temporal=True,
        # No future_covariate_values argument!
    )
    assert "solcast_pv_forecast" not in future_features_df.columns

    seq_X, _, channel_names = create_sliding_windows(
        df, "target", window_size=48,
        covariate_cols=["solcast_pv_forecast"],
        add_temporal=True,
        horizon_steps=list(range(1, 5)),
        future_features_df=future_features_df,
    )
    solcast_ch = channel_names.index("solcast_pv_forecast")
    # Future positions of solcast channel must all be zero —
    # build_inference_window / create_sliding_windows initialise
    # future_block / future_data to zeros and only populate columns
    # that appear in future_features_df.
    assert (seq_X[:, 48:, solcast_ch] == 0.0).all()


def test_future_covariate_aligned_via_reindex_and_ffill():
    """The ``compute_known_future_features`` helper reindexes the
    incoming series to the future_index and ffill/bfill-fills any
    misalignment. A covariate series with sparse coverage should
    still end up dense at future positions."""
    future_index = pd.date_range(
        "2026-05-01 00:00", periods=48, freq="30min", tz=None
    )
    # Only 8 of 48 timestamps have observations — like HA's
    # delta-storage giving us sparse points
    sparse_idx = future_index[::6]
    sparse_series = pd.Series(
        np.arange(len(sparse_idx), dtype=float), index=sparse_idx
    )
    fdf = compute_known_future_features(
        future_index, add_temporal=False,
        future_covariate_values={"sparse": sparse_series},
    )
    # No NaN anywhere — ffill+bfill should have filled the gaps
    assert fdf["sparse"].notna().all()
    # First sparse point is at index 0 with value 0 — bfill
    # propagates that value back to any leading positions
    assert fdf["sparse"].iloc[0] == 0.0
    # Last value is 7 (8 sparse points: 0..7)
    assert fdf["sparse"].iloc[-1] == 7.0


def test_collect_train_future_covariates_helper():
    """The ``_collect_train_future_covariates`` helper is shared by
    every training-side caller (production cache, benchmark holdout,
    legacy production inference). It must return a dict mapping
    cov_name → series for ONLY role in {future, both} covariates,
    and skip ones not yet present in the combined dataframe."""
    from ml_forecast_lab.main import _collect_train_future_covariates
    from ml_forecast_lab.config import CovariateCfg, ExperimentCfg

    idx = pd.date_range("2026-05-01", periods=20, freq="30min", tz=None)
    combined = pd.DataFrame({
        "target": np.arange(20, dtype=float),
        "solcast_pv": np.arange(20, dtype=float) * 2,
        "load_today": np.arange(20, dtype=float) * 3,
    }, index=idx)

    exp = ExperimentCfg(
        name="x", target_entity="t",
        covariates=[
            CovariateCfg(entity="sensor.solcast_pv", role="future"),
            CovariateCfg(entity="sensor.load_today", role="lagged"),
            CovariateCfg(entity="sensor.battery_flow", role="both"),  # not in combined
        ],
    )
    out = _collect_train_future_covariates(combined, exp)
    # future + both that ARE in combined → included
    assert set(out.keys()) == {"solcast_pv"}
    # lagged → excluded
    assert "load_today" not in out
    # both but not in combined → excluded (skipped silently)
    assert "battery_flow" not in out


def test_cov_column_name_single_entity_uses_bare_name():
    """When an entity appears only once in the experiment, the
    column name is the bare entity_id last-component — preserves
    cache-meta channel parity for existing pre-v2.38.2 experiments."""
    from ml_forecast_lab.main import _cov_column_name
    from ml_forecast_lab.config import CovariateCfg, ExperimentCfg

    cov = CovariateCfg(
        entity="sensor.solcast_pv_forecast",
        role="future", future_value_key="pv_estimate",
    )
    exp = ExperimentCfg(name="x", target_entity="t", covariates=[cov])
    assert _cov_column_name(cov, all_covs=exp.covariates) == "solcast_pv_forecast"


def test_cov_column_name_multiple_same_entity_disambiguates_by_value_key():
    """Same entity configured for two metrics (e.g. cloud_coverage AND
    temperature from one weather entity) gets value_key-suffixed
    column names so they don't collide in the dataframe."""
    from ml_forecast_lab.main import _cov_column_name
    from ml_forecast_lab.config import CovariateCfg, ExperimentCfg

    cov_a = CovariateCfg(
        entity="weather.met_office_balsham", role="future",
        future_attribute="hourly", future_value_key="cloud_coverage",
    )
    cov_b = CovariateCfg(
        entity="weather.met_office_balsham", role="future",
        future_attribute="hourly", future_value_key="temperature",
    )
    exp = ExperimentCfg(name="x", target_entity="t", covariates=[cov_a, cov_b])
    assert _cov_column_name(cov_a, all_covs=exp.covariates) == (
        "met_office_balsham__cloud_coverage"
    )
    assert _cov_column_name(cov_b, all_covs=exp.covariates) == (
        "met_office_balsham__temperature"
    )


def test_cov_column_name_no_all_covs_keeps_bare_name():
    """Helper called without ``all_covs`` keeps the bare name —
    callers that don't know the full covariate set get the legacy
    behaviour. Backwards-compatible for any external usage."""
    from ml_forecast_lab.main import _cov_column_name
    from ml_forecast_lab.config import CovariateCfg

    cov = CovariateCfg(
        entity="weather.met_office_balsham", role="future",
        future_value_key="cloud_coverage",
    )
    assert _cov_column_name(cov) == "met_office_balsham"


def test_same_covariate_allows_different_value_keys():
    """Two covariates sharing entity + role + future_attribute but
    differing on ``future_value_key`` must NOT be flagged as
    duplicates — that's the legitimate use case the v2.38.2 dedup
    relaxation enables."""
    from ml_forecast_lab.config import _same_covariate

    a = {
        "entity": "weather.met_office_balsham",
        "role": "future",
        "future_attribute": "hourly",
        "future_value_key": "cloud_coverage",
    }
    b = dict(a, future_value_key="temperature")
    assert _same_covariate(a, b) is False


def test_same_covariate_flags_identical_configs():
    """The dedup relaxation must still catch genuine duplicates —
    same entity, role, and full future-value source. Without this
    a user clicking Add twice would silently double-register the
    same covariate."""
    from ml_forecast_lab.config import _same_covariate

    a = {
        "entity": "weather.met_office_balsham",
        "role": "future",
        "future_attribute": "hourly",
        "future_value_key": "cloud_coverage",
    }
    assert _same_covariate(a, dict(a)) is True


def test_collect_train_future_covariates_no_covariates():
    """No covariates configured → empty dict, no error."""
    from ml_forecast_lab.main import _collect_train_future_covariates
    from ml_forecast_lab.config import ExperimentCfg

    idx = pd.date_range("2026-05-01", periods=10, freq="30min", tz=None)
    combined = pd.DataFrame({"target": np.zeros(10)}, index=idx)
    exp = ExperimentCfg(name="x", target_entity="t")
    assert _collect_train_future_covariates(combined, exp) == {}


def test_no_warning_now_that_v237_7_fixed_broken_backends(caplog):
    """v2.37.6 emitted a warning when nbeats / nhits / itransformer
    were combined with a future covariate (they sliced the future
    block off). v2.37.7 added auxiliary future-feature heads to all
    three, so they now consume future covariates and the warning
    has been removed. Pin so a future regression that re-introduces
    the past-only behaviour can't silently slip past CI."""
    import logging
    from ml_forecast_lab.config import CovariateCfg, ExperimentCfg

    with caplog.at_level(logging.WARNING):
        ExperimentCfg(
            name="optimised_solar", target_entity="predbat.pv_power",
            models_enabled=["nlinear", "nbeats", "nhits", "itransformer"],
            covariates=[
                CovariateCfg(entity="sensor.solcast_pv", role="future"),
            ],
        )
    # Critical assertion: the v2.37.6 warning string must not appear.
    # If a future PR re-adds it, this test fails and forces a
    # documentation update to explain the new known-broken state.
    assert "slice to past-window only" not in caplog.text


def test_multiple_future_covariates_each_reach_own_channel():
    """Two future covariates (e.g. Solcast PV + met.no temperature)
    must each land in their own channel at horizon positions."""
    df = _make_combined_with_future_cov()
    rng = np.random.default_rng(1)
    df["temperature_forecast"] = 15 + 10 * np.sin(
        2 * np.pi * np.arange(len(df)) / 48
    ) + rng.normal(0, 0.5, len(df))

    future_features_df = compute_known_future_features(
        df.index, add_temporal=True,
        future_covariate_values={
            "solcast_pv_forecast": df["solcast_pv_forecast"],
            "temperature_forecast": df["temperature_forecast"],
        },
    )
    seq_X, _, channel_names = create_sliding_windows(
        df, "target", window_size=48,
        covariate_cols=["solcast_pv_forecast", "temperature_forecast"],
        add_temporal=True,
        horizon_steps=list(range(1, 5)),
        future_features_df=future_features_df,
    )
    solcast_ch = channel_names.index("solcast_pv_forecast")
    temp_ch = channel_names.index("temperature_forecast")

    i = 50
    for j in range(4):
        absolute_row = i + 48 + j
        assert abs(
            float(seq_X[i, 48 + j, solcast_ch])
            - float(df.iloc[absolute_row]["solcast_pv_forecast"])
        ) < 1e-5
        assert abs(
            float(seq_X[i, 48 + j, temp_ch])
            - float(df.iloc[absolute_row]["temperature_forecast"])
        ) < 1e-5
