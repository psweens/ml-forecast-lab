"""Tests for preprocessing module."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.preprocessing import (
    daily_autocorrelation,
    spikiness,
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

    def test_interpolate_does_not_backfill_interior_gaps(self):
        """``gap_handling='interpolate'`` must only back-fill the LEADING
        run of NaNs (before the first observation). A long interior gap
        (e.g. an overnight PV blackout) must stay NaN beyond the
        interpolation horizon — back-filling it with the next observation
        would plant a future value into the past (lookahead leakage) and,
        for solar, paint non-zero generation into the small hours."""
        idx = pd.date_range("2024-01-01 06:00", periods=48, freq="30min")
        s = pd.Series(np.nan, index=idx)
        # Observations only at the two ends; a 10-hour hole in the middle.
        s.iloc[0:4] = [1.0, 2.0, 3.0, 4.0]
        s.iloc[-4:] = [5.0, 6.0, 7.0, 8.0]
        result = resample_to_grid(
            s.dropna(), freq="30min", method="mean",
            gap_handling="interpolate", gap_max_minutes=90,
        )
        result = result.reindex(idx)
        # The deep interior gap (beyond the 90-min / 3-step horizon) is
        # NOT filled with the trailing block's values.
        interior = result.iloc[7:-7]
        assert interior.isna().any(), (
            "interior gap was back-filled — interpolate must leave long "
            "gaps as NaN, not inherit the next observation"
        )
        assert result.iloc[-1] == 8.0  # trailing observation preserved


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


# --------------------------------------------------------------------------- #
# Data-shape diagnostics (Data Sanity Check)
# --------------------------------------------------------------------------- #
class TestSpikiness:
    """Peak-to-mean ratio. Must separate a mostly-off spike train from a smooth
    daytime bump — the two look similar to a standard deviation, since both
    spend about half the day near zero, which is why this is reported."""

    @staticmethod
    def _grid(n, minutes=30):
        return pd.date_range("2024-06-15", periods=n, freq=f"{minutes}min")

    def test_spike_train_scores_higher_than_a_smooth_bump(self):
        n = 48 * 14
        spiky = np.zeros(n)
        spiky[::16] = 10.0                        # brief, tall
        t = np.arange(n)
        smooth = np.clip(np.sin((t % 48) / 48 * 2 * np.pi - np.pi / 2), 0, None) * 3.0
        s_spiky = spikiness(pd.Series(spiky, index=self._grid(n)))
        s_smooth = spikiness(pd.Series(smooth, index=self._grid(n)))
        assert s_spiky > s_smooth * 2, (
            f"spike train {s_spiky:.2f} should clearly exceed smooth bump "
            f"{s_smooth:.2f}"
        )

    def test_flat_series_is_about_one(self):
        n = 48 * 3
        s = spikiness(pd.Series(np.full(n, 4.0), index=self._grid(n)))
        assert s == pytest.approx(1.0, abs=0.05)

    def test_returns_none_when_unmeasurable(self):
        assert spikiness(pd.Series([], dtype=float)) is None
        assert spikiness(pd.Series([1.0])) is None


class TestDailyAutocorrelation:
    """The daily lag must stay a real 24 hours when the recorder has holes.

    Measuring it positionally on a gap-compacted array shifts every sample
    after a hole, so the lag drifts and the correlation collapses — the fixture
    below reads 0.99 gapless, 0.06 at a 5% hole rate and goes negative at 10%
    under that approach.
    """

    @staticmethod
    def _tank(days=30, per_day=48, seed=1):
        idx = pd.date_range("2024-06-15", periods=days * per_day, freq="30min")
        rng = np.random.default_rng(seed)
        y = np.zeros(days * per_day)
        for d in range(days):
            for slot in (12, 36):
                y[d * per_day + slot] = rng.uniform(8, 12)
        return pd.Series(y + rng.normal(0, 0.05, y.size).clip(0), index=idx)

    @pytest.mark.parametrize("hole_pct", [0.0, 0.02, 0.05, 0.10, 0.25])
    def test_rhythm_survives_recorder_gaps(self, hole_pct):
        s = self._tank()
        if hole_pct:
            rng = np.random.default_rng(99)
            s = s.copy()
            s.iloc[rng.choice(s.size, int(s.size * hole_pct), replace=False)] = np.nan
        assert daily_autocorrelation(s, 30) > 0.9

    def test_does_not_invent_a_rhythm(self):
        """Gap tolerance must not become a free pass, or every sensor would
        look seasonal."""
        idx = pd.date_range("2024-06-15", periods=30 * 48, freq="30min")
        noise = pd.Series(np.random.default_rng(3).normal(5, 1, 30 * 48), index=idx)
        assert abs(daily_autocorrelation(noise, 30)) < 0.1

    def test_lag_follows_the_interval(self):
        """A day is 96 bins at 15 minutes, 24 at 60 — the same physical rhythm
        must read the same at any resolution."""
        for per_day, minutes in ((96, 15), (48, 30), (24, 60)):
            idx = pd.date_range("2024-06-15", periods=30 * per_day,
                                freq=f"{minutes}min")
            t = np.arange(len(idx))
            y = np.sin((t % per_day) / per_day * 2 * np.pi)
            r = daily_autocorrelation(pd.Series(y, index=idx), minutes)
            assert r > 0.95, f"{minutes}-minute grid read {r}"

    def test_returns_none_below_two_days(self):
        idx = pd.date_range("2024-06-15", periods=48, freq="30min")
        assert daily_autocorrelation(pd.Series(np.arange(48.0), index=idx), 30) is None


class TestSpikinessSeparatesRealShapes:
    """The reported bands are only worth showing if they separate the shapes a
    household actually has. Measured on 30 days at 30-minute resolution."""

    @staticmethod
    def _idx():
        return pd.date_range("2024-06-15", periods=30 * 48, freq="30min")

    def test_intermittent_loads_score_far_above_continuous_ones(self):
        idx = self._idx()
        rng = np.random.default_rng(1)
        t = np.arange(len(idx))

        heat_pump = 1.2 + 0.8 * np.sin(t / 48 * 2 * np.pi) + rng.normal(0, .15, len(idx))
        solar = np.clip(np.sin((t % 48) / 48 * 2 * np.pi - np.pi / 2), 0, None) * 3
        hot_water = np.zeros(len(idx))
        for d in range(30):
            for s0 in (12, 36):
                hot_water[d * 48 + s0:d * 48 + s0 + 2] = 3.0

        sp_hp = spikiness(pd.Series(heat_pump, index=idx))
        sp_solar = spikiness(pd.Series(solar, index=idx))
        sp_hw = spikiness(pd.Series(hot_water, index=idx))

        assert sp_hp < 3, f"a modulating heat pump should read low; got {sp_hp:.2f}"
        assert sp_hw >= 8, f"a twice-daily reheat should read high; got {sp_hw:.2f}"
        assert sp_hw > sp_solar * 2, (
            f"hot water {sp_hw:.2f} must clearly separate from solar {sp_solar:.2f}"
        )

    def test_solar_does_not_change_band_between_summer_and_winter(self):
        """The same array reads ~3.7 in summer and ~5.8 in winter. Both must
        land in the same band, or one sensor would change its description with
        the season while nothing about it had changed."""
        idx = self._idx()
        rng = np.random.default_rng(1)
        t = np.arange(len(idx))
        summer = np.clip(np.sin((t % 48) / 48 * 2 * np.pi - np.pi / 2), 0, None) * 3 \
            * rng.uniform(.6, 1, len(idx))
        winter = np.clip(np.sin((t % 48) / 48 * 2 * np.pi - np.pi / 2) * 3 - 1.6, 0, None) \
            * rng.uniform(.5, 1, len(idx))
        band = lambda s: (s >= 8) + (s >= 3) + (s >= 1.8)   # matches the UI bands
        assert band(spikiness(pd.Series(summer, index=idx))) == \
               band(spikiness(pd.Series(winter, index=idx)))


class TestSpikinessIsReciprocalDutyCycle:
    """The reported bands are only transferable if the number means something
    physical rather than matching the fixtures it was tuned on.

    For a non-negative load resting near zero, spikiness is 1 / duty-cycle. So
    the band boundary at 8 is "on about an eighth of the time" — roughly three
    hours a day — not a round number someone liked.
    """

    @staticmethod
    def _idx(n=60 * 48):
        return pd.date_range("2024-06-15", periods=n, freq="30min")

    @pytest.mark.parametrize("duty", [0.04, 0.0833, 0.125, 0.20, 0.33, 0.50, 0.75])
    def test_tracks_the_reciprocal_of_duty_cycle(self, duty):
        idx = self._idx()
        rng = np.random.default_rng(0)
        y = np.zeros(len(idx))
        y[rng.random(len(idx)) < duty] = 3.0
        s = spikiness(pd.Series(y, index=idx))
        assert s * duty == pytest.approx(1.0, abs=0.15), (
            f"at a {duty:.1%} duty cycle spikiness read {s:.2f}; "
            f"expected ~{1/duty:.1f}"
        )

    @pytest.mark.parametrize("amp", [0.5, 3.0, 7.0, 3000.0])
    def test_is_invariant_to_amplitude(self, amp):
        """Units must not move it, or a band means different things for a
        sensor in W and the same sensor in kW."""
        idx = self._idx()
        rng = np.random.default_rng(0)
        y = np.zeros(len(idx))
        y[rng.random(len(idx)) < 0.0833] = amp
        assert spikiness(pd.Series(y, index=idx)) == pytest.approx(12.0, rel=0.20)

    def test_a_standing_baseline_dilutes_it(self):
        """Documented limitation, pinned so it is not mistaken for a bug: the
        ratio is measured against zero, so a constant floor pulls it down."""
        idx = self._idx()
        rng = np.random.default_rng(0)
        on = rng.random(len(idx)) < 0.0833
        readings = []
        for base in (0.0, 1.0, 3.0):
            y = np.full(len(idx), base)
            y[on] += 3.0
            readings.append(spikiness(pd.Series(y, index=idx)))
        assert readings[0] > readings[1] > readings[2], readings
        assert readings[0] >= 8 and readings[2] < 3, (
            f"expected the same load to fall out of the top band once it sits "
            f"on a baseline; got {readings}"
        )


class TestDailyAutocorrelationIsNotFooledByTrend:
    """A raw lag-24h correlation cannot tell a daily rhythm from a trend: any
    slowly-varying signal correlates with itself at every lag.

    Measured on the undetrended form, a pure linear ramp with no cycle at all
    read 1.000 and a random walk 0.968 — both reported as a strong daily
    pattern, pushing the user toward seasonal metrics and calendar features
    that cannot help. Subtracting a one-day centred rolling mean removes
    anything slower than a day while leaving the within-day shape intact.
    """

    @staticmethod
    def _idx(n=60 * 48):
        return pd.date_range("2024-06-15", periods=n, freq="30min")

    @pytest.mark.parametrize("name,build", [
        ("linear trend", lambda t, n, rng: t / n),
        ("slow drift + noise", lambda t, n, rng: t / n * 3 + rng.normal(0, .05, n)),
        ("random walk", lambda t, n, rng: np.cumsum(rng.normal(0, .1, n))),
        ("white noise", lambda t, n, rng: rng.normal(0, 1, n)),
    ])
    def test_signals_without_a_daily_cycle_read_near_zero(self, name, build):
        idx = self._idx()
        n = len(idx)
        r = daily_autocorrelation(
            pd.Series(build(np.arange(n), n, np.random.default_rng(0)), index=idx), 30
        )
        assert abs(r) < 0.2, f"{name} reported a daily rhythm of {r:.3f}"

    def test_a_genuine_rhythm_survives_a_strong_trend(self):
        """Removing the trend must not remove the signal with it — a sensor
        whose usage is climbing still has its daily shape."""
        idx = self._idx()
        t = np.arange(len(idx))
        y = np.sin(t / 48 * 2 * np.pi) + t / len(idx) * 6      # trend dominates in amplitude
        assert daily_autocorrelation(pd.Series(y, index=idx), 30) > 0.9

    def test_a_weekly_only_load_is_not_called_daily(self):
        """An EV charged three evenings a week has no daily pattern —
        yesterday tells you nothing about today."""
        idx = self._idx()
        y = np.zeros(len(idx))
        for d in range(60):
            if d % 7 in (1, 3, 5):
                y[d * 48 + 36:d * 48 + 44] = 7.0
        assert daily_autocorrelation(pd.Series(y, index=idx), 30) < 0.2

    def test_a_weekday_only_rhythm_still_counts_as_daily(self):
        """Five days in seven repeating IS a daily pattern worth exploiting."""
        idx = self._idx()
        y = np.zeros(len(idx))
        for d in range(60):
            if d % 7 < 5:
                for s0 in (12, 36):
                    y[d * 48 + s0:d * 48 + s0 + 2] = 3.0
        assert daily_autocorrelation(pd.Series(y, index=idx), 30) > 0.7


class TestDataReportContract:
    """Every field the Data Sanity Check UI reads must exist in the response.

    `compute_data_report` builds an explicit allow-list rather than spreading
    the per-entity analysis, so a statistic added to `_analyse_entity_history`
    is computed, stored in `target[...]`, and then silently dropped on the way
    out. The UI guards each row with `!= null`, so the row just does not
    appear — no error, no log line, nothing to notice.

    That is exactly how `spikiness` and `daily_autocorr` shipped invisible:
    both were unit-tested, and so was the UI, but nothing tested the wiring
    between them.
    """

    @staticmethod
    def _paths():
        root = Path(__file__).resolve().parents[2]
        return (root / "ml_forecast_lab" / "web" / "templates" / "experiment.html",
                root / "ml_forecast_lab" / "main.py")

    def test_every_field_the_ui_reads_is_returned(self):
        import re
        html_path, main_path = self._paths()

        js = html_path.read_text()
        i = js.index("window.runDataReport")
        j = js.index("window.", i + 20)
        read = set(re.findall(r'\brep\.([a-zA-Z_][a-zA-Z0-9_]*)', js[i:j]))

        src = main_path.read_text()
        k = src.index("async def compute_data_report")
        end = src.index("\n    async def ", k + 10)
        provided = set(re.findall(r'^\s+"([a-z_0-9]+)":', src[k:end], re.M))

        missing = sorted(read - provided)
        assert not missing, (
            f"the Data Sanity Check UI reads {missing} but compute_data_report "
            f"does not return them — those rows will silently never render"
        )

    def test_the_shape_diagnostics_specifically_are_wired(self):
        """Named explicitly, so a regression names itself rather than showing
        up as an anonymous set difference."""
        import re
        _, main_path = self._paths()
        src = main_path.read_text()
        k = src.index("async def compute_data_report")
        end = src.index("\n    async def ", k + 10)
        block = src[k:end]
        for f in ("spikiness", "daily_autocorr"):
            assert re.search(rf'^\s+"{f}":', block, re.M), (
                f"{f} is computed in _analyse_entity_history but not surfaced "
                f"by compute_data_report"
            )
