"""
Calendar features in the configured timezone (v2.58.0).

Pins a defect found in the 2026-11 intermittent-demand investigation:
hour-of-day, day-of-week and the weekend flag were read off the tz-naive
UTC index. In a zone with daylight saving, a training window that spans a
change held two profiles one hour apart under the same ``hour_of_day``, and
the day rolled over at UTC midnight rather than local midnight.

- with a calendar zone set, a habit at a fixed local time has one
  ``hour_of_day`` / ``slot_of_day`` on both sides of a change;
- day-of-week and weekend flip at local midnight;
- every path that builds a row — feature frame, future-known features,
  sliding-window channels, the recursive forecast row — uses the same
  calendar;
- with no zone (or UTC) every pre-existing column is exactly as before;
- sub-hourly frames gain ``slot_of_day``; hourly frames gain nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab import features as F
from ml_forecast_lab.features import (
    build_features, calendar_row, compute_known_future_features,
    create_sliding_windows, recursive_tree_forecast, set_calendar_timezone,
)

TZ = "Europe/London"          # clocks went back on 2026-10-25 at 01:00 UTC


@pytest.fixture(autouse=True)
def _reset_calendar():
    set_calendar_timezone(None)
    yield
    set_calendar_timezone(None)


def _frame(start="2026-10-19", days=14, freq="30min"):
    idx = pd.date_range(start, periods=days * (48 if freq == "30min" else 24), freq=freq)
    local = idx.tz_localize("UTC").tz_convert(TZ)
    # An event every day at 08:00 local.
    y = np.where((local.hour == 8) & (local.minute == 0), 5.0, 0.2)
    return pd.DataFrame({"y": y}, index=idx), local


class TestLocalCalendar:
    def test_a_local_habit_has_one_hour_across_a_clock_change(self):
        df, _ = _frame()
        utc = build_features(df, "y", 30)
        assert set(utc.loc[df["y"] == 5.0, "hour_of_day"]) == {7, 8}
        set_calendar_timezone(TZ)
        loc = build_features(df, "y", 30)
        assert set(loc.loc[df["y"] == 5.0, "hour_of_day"]) == {8}
        assert set(loc.loc[df["y"] == 5.0, "slot_of_day"]) == {16}

    def test_day_boundaries_fall_at_local_midnight(self):
        # Saturday 2026-10-17 23:30 UTC is Sunday 00:30 in summer time.
        idx = pd.date_range("2026-10-17 22:00", periods=6, freq="30min")
        df = pd.DataFrame({"y": np.arange(6.0)}, index=idx)
        set_calendar_timezone(TZ)
        f = build_features(df, "y", 30, n_lags=1)
        assert f["day_of_week"].tolist() == [5, 5, 6, 6, 6, 6]
        assert f["day_of_month"].tolist() == [17, 17, 18, 18, 18, 18]
        assert f["slot_of_day"].tolist() == [46, 47, 0, 1, 2, 3]

    def test_the_repeated_hour_is_one_hour_twice(self):
        # 00:00-02:00 UTC on the change day is 01:00, 01:30, 01:00, 01:30 local.
        idx = pd.date_range("2026-10-25 00:00", periods=4, freq="30min")
        df = pd.DataFrame({"y": np.arange(4.0)}, index=idx)
        set_calendar_timezone(TZ)
        f = build_features(df, "y", 30, n_lags=1)
        assert f["hour_of_day"].tolist() == [1, 1, 1, 1]
        assert f["slot_of_day"].tolist() == [2, 3, 2, 3]


class TestEveryPathAgrees:
    def test_recursive_row_matches_the_frame(self):
        set_calendar_timezone(TZ)
        df, _ = _frame(days=6)
        feats = build_features(df, "y", 30)
        cols = list(feats.columns)
        seen = {}

        def predict(X):
            seen.setdefault("row", X[0].copy())
            return np.array([0.0])

        recursive_tree_forecast(predict, cols, df["y"].iloc[:-1], 1, 30)
        np.testing.assert_allclose(
            seen["row"], feats.iloc[-1].to_numpy(dtype=np.float32), rtol=1e-5, atol=1e-6,
        )

    def test_calendar_row_matches_the_frame_on_both_sides_of_the_change(self):
        set_calendar_timezone(TZ)
        df, _ = _frame()
        feats = build_features(df, "y", 30)
        for ts in (df.index[100], df.index[-100]):
            row = calendar_row(ts, 30)
            for k, v in row.items():
                assert feats.at[ts, k] == pytest.approx(v), k

    def test_future_known_features_and_window_channels_use_it(self):
        df, _ = _frame()
        work = df.rename(columns={"y": "target"})
        fut_utc = compute_known_future_features(df.index)
        _, _, names = create_sliding_windows(work, "target", window_size=12, horizon_steps=[1])
        X_utc, _, _ = create_sliding_windows(work, "target", window_size=12, horizon_steps=[1])
        set_calendar_timezone(TZ)
        fut_loc = compute_known_future_features(df.index)
        X_loc, _, _ = create_sliding_windows(work, "target", window_size=12, horizon_steps=[1])
        summer = df.index < pd.Timestamp("2026-10-25 01:00")
        assert not np.allclose(fut_utc["hour_sin"][summer], fut_loc["hour_sin"][summer])
        np.testing.assert_allclose(fut_utc["hour_sin"][~summer], fut_loc["hour_sin"][~summer])
        h = names.index("hour_sin")
        assert not np.allclose(X_utc[:100, :, h], X_loc[:100, :, h])
        np.testing.assert_allclose(X_utc[-100:, :, h], X_loc[-100:, :, h])
        # Frame and future-known features agree with each other.
        feats = build_features(df, "y", 30)
        np.testing.assert_allclose(fut_loc["hour_sin"], feats["hour_sin"], rtol=1e-6)


class TestUnchangedWithoutAZone:
    @pytest.mark.parametrize("tz", [None, "UTC", "utc", ""])
    def test_utc_is_the_previous_behaviour(self, tz):
        df, _ = _frame()
        set_calendar_timezone(tz)
        assert F.get_calendar_timezone() is None
        f = build_features(df, "y", 30)
        np.testing.assert_array_equal(f["hour_of_day"], df.index.hour)
        np.testing.assert_array_equal(f["day_of_week"], df.index.dayofweek)
        np.testing.assert_array_equal(f["is_weekend"], (df.index.dayofweek >= 5).astype(int))
        np.testing.assert_array_equal(f["month"], df.index.month)
        np.testing.assert_array_equal(f["day_of_month"], df.index.day)

    def test_hourly_frames_gain_no_column(self):
        df, _ = _frame(freq="60min")
        assert "slot_of_day" not in build_features(df, "y", 60).columns
        assert "slot_of_day" not in calendar_row(df.index[0], 60)

    def test_sub_hourly_frames_gain_slot_of_day_only(self):
        df, _ = _frame()
        f = build_features(df, "y", 30)
        np.testing.assert_array_equal(
            f["slot_of_day"], (df.index.hour * 60 + df.index.minute) // 30,
        )
        assert "slot_of_day" in F.ENGINEERED_TEMPORAL_COLUMNS


class TestAppSetsTheZone:
    def test_pipeline_start_sets_it_and_the_cache_schema_moved(self):
        from pathlib import Path
        from ml_forecast_lab import main as M
        src = Path(M.__file__).read_text(encoding="utf-8")
        assert 'set_calendar_timezone(getattr(self.config, "timezone", None))' in src
        assert M.CACHE_SCHEMA_VERSION == 4
        # No path computes calendar features off the raw index any more.
        assert "ts.hour" not in src and "idx.hour" not in src
