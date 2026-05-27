"""Tests for preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.preprocessing import (
    align_series,
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

    def test_quiet_period_gap_demand_is_preserved(self):
        """v2.40.5 regression: HA's recorder stores only state changes, so a
        daily-reset demand counter has NO rows during quiet periods (overnight,
        between draw-offs). The draw-off that ends a quiet period spans >1.5
        intervals — its increment must be KEPT (real demand), not dropped, or
        the daily total under-counts (it used to ~halve)."""
        # Sparse, change-only cumulative for one day at a 10-min interval:
        #   06:00 reset to 0, a morning draw at 06:00->06:10 (5%),
        #   then a 7-hour quiet gap (no rows), then an evening draw to 45%.
        idx = pd.DatetimeIndex([
            "2026-05-01 06:00", "2026-05-01 06:10",   # morning draw: 0 -> 5
            "2026-05-01 13:10", "2026-05-01 13:20",   # evening draw after 7h gap
        ])
        cumulative = pd.Series([0.0, 5.0, 40.0, 45.0], index=idx)
        result = cumulative_to_interval(
            cumulative, interval_minutes=10, reset_daily=True,
            max_increment=100,
        )
        # Daily total of the increments must equal the day's cumulative peak
        # (45), NOT half of it. The 13:10 row spans a 6h50m gap (>1.5 intervals)
        # and carries 35 units of real demand — it must be kept.
        assert result.sum() == pytest.approx(45.0), (
            f"gap-spanning demand was lost; got total {result.sum()} vs 45"
        )

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

    def test_default_quantile_is_999(self):
        """0.999 is the post-H-2 default: 0.5% top trim was clipping
        legitimate sensor peaks on spiky targets (rainfall, EV draw)."""
        import inspect
        sig = inspect.signature(clip_outliers)
        assert sig.parameters['quantile'].default == 0.999


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


class TestAlignSeries:
    """``align_series`` claims to support 'left' and 'right' joins, but the
    previous implementation forwarded the method straight to ``pd.concat``
    which only accepts 'inner'/'outer'. These tests pin the documented
    behaviour."""

    def _two_offset_series(self):
        idx_a = pd.date_range("2024-01-01", periods=4, freq="1h")
        idx_b = pd.date_range("2024-01-01 02:00", periods=4, freq="1h")
        a = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx_a, name="a")
        b = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx_b, name="b")
        return a, b

    def test_inner_intersects_indices(self):
        a, b = self._two_offset_series()
        out_a, out_b = align_series([a, b], method="inner")
        # Inner join keeps only the two overlapping timestamps
        assert len(out_a) == 2
        assert len(out_b) == 2
        assert not out_a.isna().any()
        assert not out_b.isna().any()

    def test_outer_unions_indices(self):
        a, b = self._two_offset_series()
        out_a, out_b = align_series([a, b], method="outer")
        # Outer join keeps all 6 distinct timestamps
        assert len(out_a) == 6
        # NaNs appear where each series didn't originally cover that timestamp
        assert out_a.isna().sum() == 2
        assert out_b.isna().sum() == 2

    def test_left_anchors_to_first(self):
        a, b = self._two_offset_series()
        out_a, out_b = align_series([a, b], method="left")
        # Left join uses a's index as the anchor
        pd.testing.assert_index_equal(out_a.index, a.index)
        pd.testing.assert_index_equal(out_b.index, a.index)
        # a's values must be unchanged
        np.testing.assert_array_equal(out_a.values, a.values)
        # b is reindexed onto a's index — NaN where it didn't originally cover
        assert out_b.isna().sum() == 2

    def test_right_anchors_to_last(self):
        a, b = self._two_offset_series()
        out_a, out_b = align_series([a, b], method="right")
        pd.testing.assert_index_equal(out_a.index, b.index)
        pd.testing.assert_index_equal(out_b.index, b.index)
        np.testing.assert_array_equal(out_b.values, b.values)
        assert out_a.isna().sum() == 2

    def test_unknown_method_raises(self):
        a, b = self._two_offset_series()
        with pytest.raises(ValueError, match="method must be"):
            align_series([a, b], method="garbage")

    def test_single_series_returns_copy(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="1h")
        a = pd.Series([1.0, 2.0, 3.0], index=idx)
        out = align_series([a])
        assert len(out) == 1
        # Must be a copy — mutating the result should not touch the input
        out[0].iloc[0] = 99.0
        assert a.iloc[0] == 1.0
