"""Tests for features module."""

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.features import (
    build_features,
    create_sliding_windows,
    is_holiday,
)


class TestBuildFeatures:
    def test_output_shape(self, synthetic_df):
        result = build_features(synthetic_df, target_col="y", interval_minutes=30)
        # Should have same number of rows as input
        assert len(result) == len(synthetic_df)
        # Should have temporal + lag + rolling features
        assert len(result.columns) > 10

    def test_has_temporal_features(self, synthetic_df):
        result = build_features(synthetic_df, target_col="y", interval_minutes=30)
        for col in ['hour_of_day', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos']:
            assert col in result.columns, f"Missing temporal feature: {col}"

    def test_has_lag_features(self, synthetic_df):
        result = build_features(synthetic_df, target_col="y", interval_minutes=30)
        lag_cols = [c for c in result.columns if c.startswith('y_lag_')]
        assert len(lag_cols) >= 1

    def test_has_rolling_features(self, synthetic_df):
        result = build_features(synthetic_df, target_col="y", interval_minutes=30)
        rolling_cols = [c for c in result.columns if c.startswith('y_rolling_')]
        assert len(rolling_cols) >= 3

    def test_no_target_leakage_in_features(self):
        """Feature row at t must not depend on target[t]."""
        idx = pd.date_range('2024-01-01', periods=200, freq='30min')
        df = pd.DataFrame({'y': np.linspace(0, 100, 200)}, index=idx)

        baseline = build_features(df, target_col='y', interval_minutes=30, n_lags=6)

        # Perturb a single target row and confirm that the feature row at the
        # same timestamp is unchanged. Any feature column that depends on
        # target[t] would shift here — including unshifted rolling stats.
        probe_idx = 150
        perturbed = df.copy()
        perturbed.iloc[probe_idx, 0] += 1000.0
        perturbed_features = build_features(
            perturbed, target_col='y', interval_minutes=30, n_lags=6
        )

        for col in baseline.columns:
            base_val = baseline.iloc[probe_idx][col]
            new_val = perturbed_features.iloc[probe_idx][col]
            if pd.isna(base_val) and pd.isna(new_val):
                continue
            assert base_val == new_val, (
                f"feature '{col}' at row {probe_idx} changed from {base_val!r} "
                f"to {new_val!r} when target[{probe_idx}] was perturbed — "
                f"this is a look-ahead leak"
            )


class TestCreateSlidingWindows:
    def test_output_shapes(self, synthetic_df):
        df = synthetic_df.copy()
        df = df.rename(columns={"y": "target"})
        X, y, channels = create_sliding_windows(
            df, "target", window_size=24,
            covariate_cols=["current_charge", "external_temperature"],
            add_temporal=True,
        )
        assert X.ndim == 3
        assert X.shape[0] == len(df) - 24  # n_samples
        assert X.shape[1] == 24  # window_size
        # target + 2 covariates + 5 temporal features = 8 channels
        assert X.shape[2] == 8
        assert len(y) == X.shape[0]
        assert len(channels) == 8

    def test_channel_names(self, synthetic_df):
        df = synthetic_df.rename(columns={"y": "target"})
        _, _, channels = create_sliding_windows(
            df, "target", window_size=12,
            covariate_cols=["current_charge"],
            add_temporal=True,
        )
        assert channels[0] == "target"
        assert "current_charge" in channels
        assert "hour_sin" in channels
        assert "is_weekend" in channels

    def test_no_temporal(self, synthetic_df):
        df = synthetic_df.rename(columns={"y": "target"})
        _, _, channels = create_sliding_windows(
            df, "target", window_size=12,
            covariate_cols=None, add_temporal=False,
        )
        assert channels == ["target"]

    def test_target_is_next_step(self, synthetic_df):
        """y[i] should be the target value at position i + window_size."""
        df = synthetic_df.rename(columns={"y": "target"})
        X, y, _ = create_sliding_windows(
            df, "target", window_size=12,
            covariate_cols=None, add_temporal=False,
        )
        target_vals = df["target"].values
        for i in range(min(5, len(y))):
            np.testing.assert_almost_equal(y[i], target_vals[i + 12])


class TestIsHoliday:
    def test_christmas_gb(self):
        dt = pd.Timestamp("2024-12-25")
        assert is_holiday(dt, "GB") is True

    def test_random_day_gb(self):
        dt = pd.Timestamp("2024-03-15")
        assert is_holiday(dt, "GB") is False

    def test_none_country(self):
        dt = pd.Timestamp("2024-12-25")
        assert is_holiday(dt, None) is False
