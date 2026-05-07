"""Tests for preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.preprocessing import (
    clip_outliers,
    cumulative_to_interval,
    resample_to_grid,
    apply_transform,
    invert_transform,
)


class TestCumulativeToInterval:
    def test_basic_conversion(self, synthetic_cumulative_series):
        result = cumulative_to_interval(
            synthetic_cumulative_series, interval_minutes=30, reset_daily=True
        )
        assert len(result) == len(synthetic_cumulative_series)
        assert (result >= 0).all(), "Interval values should be non-negative"
        assert result.iloc[0] == 0, "First value should be 0"

    def test_preserves_index(self, synthetic_cumulative_series):
        result = cumulative_to_interval(
            synthetic_cumulative_series, interval_minutes=30, reset_daily=True
        )
        pd.testing.assert_index_equal(result.index, synthetic_cumulative_series.index)

    def test_rejects_non_series(self):
        with pytest.raises(TypeError):
            cumulative_to_interval([1, 2, 3], interval_minutes=30)

    def test_rejects_non_datetime_index(self):
        s = pd.Series([1, 2, 3], index=[0, 1, 2])
        with pytest.raises(TypeError):
            cumulative_to_interval(s, interval_minutes=30)

    def test_max_increment_caps_spikes(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="30min")
        s = pd.Series([0, 1, 2, 102, 103], index=idx)  # Spike at position 3
        result = cumulative_to_interval(s, interval_minutes=30, max_increment=10)
        assert result.iloc[3] <= 10


class TestResampleToGrid:
    def test_mean_method(self, synthetic_interval_series):
        result = resample_to_grid(synthetic_interval_series, freq="1h", method="mean")
        assert len(result) < len(synthetic_interval_series)

    def test_sum_method(self, synthetic_interval_series):
        result = resample_to_grid(synthetic_interval_series, freq="1h", method="sum")
        # Sum of 2 x 30-min values should be ~2x the mean
        mean_result = resample_to_grid(synthetic_interval_series, freq="1h", method="mean")
        np.testing.assert_allclose(result.values, mean_result.values * 2, atol=0.01)

    def test_no_nans_in_output(self, synthetic_interval_series):
        result = resample_to_grid(synthetic_interval_series, freq="30min", method="mean")
        assert not result.isna().any()


class TestClipOutliers:
    def test_clips_at_quantile(self):
        idx = pd.date_range("2024-01-01", periods=1000, freq="30min")
        rng = np.random.default_rng(42)
        values = rng.normal(5, 1, 1000)
        values[0] = 100  # Extreme outlier
        s = pd.Series(values, index=idx)
        result = clip_outliers(s, quantile=0.99)
        assert result.max() < 100

    def test_positive_only(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="30min")
        rng = np.random.default_rng(42)
        values = rng.normal(5, 1, 100)
        values[0] = -10
        s = pd.Series(values, index=idx)
        result = clip_outliers(s, positive_only=True)
        assert (result >= 0).all()

    def test_default_quantile_is_995(self):
        """Verify default changed from 0.95 to 0.995."""
        import inspect
        sig = inspect.signature(clip_outliers)
        assert sig.parameters['quantile'].default == 0.995


class TestTransformRoundtrip:
    def test_log_roundtrip(self, synthetic_interval_series):
        transformed = apply_transform(synthetic_interval_series, 'log')
        recovered = invert_transform(transformed, 'log')
        np.testing.assert_allclose(
            recovered.values, synthetic_interval_series.values, atol=1e-6
        )

    def test_sqrt_roundtrip(self, synthetic_interval_series):
        transformed = apply_transform(synthetic_interval_series, 'sqrt')
        recovered = invert_transform(transformed, 'sqrt')
        np.testing.assert_allclose(
            recovered.values, synthetic_interval_series.values, atol=1e-6
        )

    def test_shifted_log_roundtrip(self, synthetic_interval_series):
        transformed = apply_transform(synthetic_interval_series, 'shifted_log')
        recovered = invert_transform(transformed, 'shifted_log')
        np.testing.assert_allclose(
            recovered.values, synthetic_interval_series.values, atol=1e-6
        )

    def test_box_cox_alias(self, synthetic_interval_series):
        """box_cox should still work as alias for shifted_log."""
        transformed = apply_transform(synthetic_interval_series, 'box_cox')
        assert transformed.attrs.get('transform') == 'shifted_log'
        recovered = invert_transform(transformed, 'shifted_log')
        np.testing.assert_allclose(
            recovered.values, synthetic_interval_series.values, atol=1e-6
        )
