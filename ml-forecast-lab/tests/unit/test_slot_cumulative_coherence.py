"""
The cumulative forecast is the running sum of the slot forecast.

Written for the 2026-11 intermittent-demand investigation. For a
daily-resetting cumulative target the app publishes a per-interval forecast
and a cumulative one; an automation that reads "how much by 18:00" from the
second and "how much between 17:00 and 18:00" from the first must get
answers that agree. These tests pin that the cumulative curve is built from
the published slot forecast and nothing else:

- within a local day each cumulative point is the previous one plus that
  interval's forecast, starting from today's reading of the counter;
- the curve restarts from zero at local midnight, on both sides of a clock
  change, not at UTC midnight;
- the sensor's state is the projected end-of-today total;
- a non-resetting target gets the plain running sum across the horizon.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.config import AppConfig, ExperimentCfg
from ml_forecast_lab.db import HistoryDB
from ml_forecast_lab.main import MLForecastLabApp

TZ = "Europe/London"
EXP = "demand"
ENTITY = "sensor.demand_today"


class _StubHA:
    def __init__(self, seed):
        self.captured, self.seed = {}, seed

    async def set_state(self, entity_id, state, attributes=None):
        self.captured[entity_id] = (state, attributes or {})
        return True

    async def get_state(self, entity_id, default=None, attribute=None):
        return self.seed if entity_id == ENTITY else default


def _publish(tmp_path, first_utc, periods=96, seed=12.5, **exp_kw):
    exp = ExperimentCfg(
        name=EXP, target_entity=ENTITY, interval_minutes=30,
        future_periods=periods, publish_prefix="mlfl_", publish_name=EXP,
        mode="production", units="%", **exp_kw,
    )
    app = MLForecastLabApp()
    app.ha_interface = _StubHA(str(seed))
    app.history_db = HistoryDB(str(tmp_path / "h.db"))
    app.config = AppConfig(experiments=[exp])
    app.config.timezone = TZ
    app._cached_models = {exp.name: {"model_version": "v1"}}
    rng = np.random.default_rng(0)
    y = np.round(rng.gamma(0.6, 2.0, periods), 3).astype(np.float32)
    ds = pd.date_range(first_utc, periods=periods, freq="30min")
    asyncio.run(app._publish_forecast_sensors(
        exp_cfg=exp, y_pred=y, ds_future=ds, model_name="lightgbm",
        last_trained_iso="2026-10-01T00:00:00Z",
    ))
    return app.ha_interface.captured, y, ds


def _curve(captured):
    state, attrs = captured[f"sensor.mlfl_{EXP}_cumulative"]
    ts = pd.DatetimeIndex([pd.Timestamp(p["datetime"]) for p in attrs["forecast"]])
    return state, attrs, ts, np.array([p["value"] for p in attrs["forecast"]])


def _slots(captured):
    _, attrs = captured[f"sensor.mlfl_{EXP}_forecast"]
    return np.array([p["value"] for p in attrs["forecast"]], dtype=float)


class TestDailyResettingTarget:
    KW = dict(source_is_cumulative=True, reset_daily=True)

    @pytest.fixture
    def published(self, tmp_path, monkeypatch):
        # Issued at 19:30 local on a winter-time day.
        import ml_forecast_lab.main as M

        class _Clock(M.datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 11, 10, 19, 30, tzinfo=M.timezone.utc)

        monkeypatch.setattr(M, "datetime", _Clock)
        return _publish(tmp_path, "2026-11-10 20:00", **self.KW)

    def test_each_point_adds_that_intervals_forecast(self, published):
        captured, y, ds = published
        _, attrs, ts, cum = _curve(captured)
        slots = _slots(captured)
        np.testing.assert_allclose(slots, y, atol=1e-3)
        local_day = ts.tz_convert(TZ).date
        for i in range(1, len(cum)):
            if local_day[i] == local_day[i - 1]:
                assert cum[i] - cum[i - 1] == pytest.approx(slots[i], abs=2e-4)

    def test_today_starts_from_the_counter_and_later_days_from_zero(self, published):
        captured, y, ds = published
        _, attrs, ts, cum = _curve(captured)
        slots = _slots(captured)
        local_day = ts.tz_convert(TZ).date
        assert attrs["seeded_with"] == pytest.approx(12.5)
        assert cum[0] == pytest.approx(12.5 + slots[0], abs=2e-4)
        firsts = [i for i in range(1, len(cum)) if local_day[i] != local_day[i - 1]]
        assert len(firsts) == 2
        for i in firsts:
            assert cum[i] == pytest.approx(slots[i], abs=2e-4)

    def test_state_is_the_projected_end_of_today_total(self, published):
        captured, y, ds = published
        state, attrs, ts, cum = _curve(captured)
        slots = _slots(captured)
        today = ts.tz_convert(TZ).date == ts.tz_convert(TZ).date[0]
        assert float(state) == pytest.approx(12.5 + slots[today].sum(), abs=1e-3)
        assert attrs["end_of_today_value"] == pytest.approx(float(state))

    def test_day_totals_equal_the_sum_of_the_days_slots(self, published):
        captured, y, ds = published
        _, _, ts, cum = _curve(captured)
        slots = _slots(captured)
        local_day = pd.Series(ts.tz_convert(TZ).date)
        tomorrow = local_day.unique()[1]
        sel = (local_day == tomorrow).to_numpy()
        assert sel.sum() == 48
        assert cum[sel][-1] == pytest.approx(slots[sel].sum(), abs=1e-3)


class TestResetFollowsLocalMidnight:
    def test_summer_time_reset_is_at_2300_utc(self, tmp_path):
        captured, y, ds = _publish(
            tmp_path, "2026-10-20 18:00", periods=24,
            source_is_cumulative=True, reset_daily=True,
        )
        _, _, ts, cum = _curve(captured)
        slots = _slots(captured)
        # 23:00 UTC is local midnight in summer time: the curve restarts there.
        i = list(ts.tz_convert("UTC").strftime("%H:%M")).index("23:00")
        assert cum[i] == pytest.approx(slots[i], abs=2e-4)
        assert cum[i - 1] > cum[i] or slots[i] > cum[i - 1]

    def test_the_25_hour_day_is_one_day(self, tmp_path):
        # Local day 2026-10-25 runs 23:00 UTC on the 24th to 00:00 UTC on the 26th.
        captured, y, ds = _publish(
            tmp_path, "2026-10-24 22:00", periods=60,
            source_is_cumulative=True, reset_daily=True,
        )
        _, _, ts, cum = _curve(captured)
        slots = _slots(captured)
        local_day = pd.Series(ts.tz_convert(TZ).date)
        long_day = pd.Timestamp("2026-10-25").date()
        sel = (local_day == long_day).to_numpy()
        assert sel.sum() == 50
        assert cum[sel][-1] == pytest.approx(slots[sel].sum(), abs=1e-3)


class TestNonResettingTarget:
    def test_plain_running_sum_across_the_horizon(self, tmp_path):
        captured, y, ds = _publish(tmp_path, "2026-11-10 20:00", periods=12)
        _, _, _, cum = _curve(captured)
        np.testing.assert_allclose(cum, np.cumsum(_slots(captured)), atol=1e-3)
