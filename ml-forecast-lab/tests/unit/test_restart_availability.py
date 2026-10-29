"""v2.52.7: forecast sensors come back promptly after a restart.

Defects pinned (all live up to v2.52.6):
- HA does not restore states POSTed over its REST API, and nothing
  re-posted them: after an HA restart the sensors were missing until each
  experiment's next forecast, up to a full cadence; after the app started,
  the first forecast waited a full cadence too.
- A retrain or forecast that failed because HA was unreachable (the app
  started before Core, or Core was restarting) waited for its normal
  cadence; with no restored model that meant no forecasts for up to
  ``retrain_every_hours``.
- A retrain replacing a pinned ``production_model`` deleted the old model
  first, so a failed retrain left nothing to forecast with.
- ``_cumulative`` (signed predictions) and the bands (before calibration)
  were omitted rather than posted, so an HA restart removed them.
- The heartbeat (``sensor.mlfl_last_run``) was only posted by a retrain,
  and ``_last_benchmark`` never by a Training-tab benchmark.
- Sensors of renamed, removed or lab experiments were never removed, and a
  restored model without a saved benchmark showed "Pending".

Review-driven constraints, also pinned: only the restart check posts the
heartbeat (any other path could re-create it after an HA restart and hide
the loss), and ``_last_<event>`` sensors are never re-created (that fires
state-triggered automations for an event that didn't happen).
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
import pytest

import ml_forecast_lab.ha_interface as hai
from ml_forecast_lab.config import ExperimentCfg
from ml_forecast_lab.ha_interface import (
    HAInterface, HAResponseError, HAUnavailableError,
)
from ml_forecast_lab.main import (
    HA_RETRY_MINUTES, HEARTBEAT_ENTITY, STARTUP_FORECAST_DELAY_SECONDS,
    MLForecastLabApp, _ha_unavailable,
)

ENTITY = "sensor.pv_power"


# --- HAInterface ---------------------------------------------------------

class _FakeResp:
    def __init__(self, status, body="", json_body=None):
        self.status = status
        self._body = body
        self._json = json_body
        self.headers = {"content-type": "application/json" if json_body is not None else "text/plain"}

    async def text(self):
        return self._body

    async def json(self):
        return self._json

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Answers each request with the next item: a status code, or an
    exception instance to raise."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0
        self.closed = False

    def request(self, method, url, **kw):
        self.calls += 1
        a = self.answers.pop(0)
        if isinstance(a, BaseException):
            raise a
        if a == 200:
            return _FakeResp(200, json_body={"state": "1"})
        return _FakeResp(a, body="Entity not found." if a == 404 else "bad gateway")


def _iface(answers, monkeypatch):
    monkeypatch.setattr(hai, "RETRY_BACKOFF_BASE", 0.0)
    sess = _FakeSession(answers)
    return HAInterface(ha_url="http://ha", ha_key="x", session=sess), sess


class TestHAErrors:
    def test_exhausted_5xx_raises_unavailable(self, monkeypatch):
        iface, sess = _iface([502, 502, 502], monkeypatch)
        with pytest.raises(HAUnavailableError) as ei:
            asyncio.run(iface.api_call("GET", "/api/states/x"))
        assert isinstance(ei.value, RuntimeError)
        assert "failed after 3 attempts" in str(ei.value)
        assert sess.calls == 3

    def test_exhausted_connection_errors_raise_unavailable(self, monkeypatch):
        errs = [aiohttp.ClientConnectionError("refused") for _ in range(3)]
        iface, _ = _iface(errs, monkeypatch)
        with pytest.raises(HAUnavailableError):
            asyncio.run(iface.api_call("GET", "/api/states/x"))

    def test_4xx_raises_response_error_with_status(self, monkeypatch):
        iface, sess = _iface([404], monkeypatch)
        with pytest.raises(HAResponseError) as ei:
            asyncio.run(iface.api_call("GET", "/api/states/x"))
        assert ei.value.status == 404
        assert str(ei.value).startswith("HA API error 404:")
        assert not isinstance(ei.value, HAUnavailableError)
        assert sess.calls == 1

    @pytest.mark.parametrize("answer, expected", [
        (200, True), (404, False), (502, None),
        (aiohttp.ClientConnectionError("refused"), None),
    ])
    def test_delete_state(self, monkeypatch, answer, expected):
        iface, _ = _iface([answer] * 3, monkeypatch)
        assert asyncio.run(iface.delete_state("sensor.x")) is expected

    def test_delete_state_raises_on_other_4xx(self, monkeypatch):
        iface, _ = _iface([405], monkeypatch)
        with pytest.raises(HAResponseError):
            asyncio.run(iface.delete_state("sensor.x"))

    @pytest.mark.parametrize("answer, expected", [
        (200, True), (404, False), (401, None), (502, None),
        (aiohttp.ClientConnectionError("refused"), None),
    ])
    def test_entity_exists_makes_one_attempt(self, monkeypatch, answer, expected):
        iface, sess = _iface([answer], monkeypatch)
        assert asyncio.run(iface.entity_exists("sensor.x")) is expected
        assert sess.calls == 1

    def test_unavailable_found_through_the_cause_chain(self):
        try:
            try:
                raise HAUnavailableError("down")
            except HAUnavailableError as inner:
                raise RuntimeError("fetch failed") from inner
        except RuntimeError as outer:
            assert _ha_unavailable(outer)
        assert not _ha_unavailable(RuntimeError("HA API error 401: no"))
        assert not _ha_unavailable(None)


# --- app harness ---------------------------------------------------------

class _StubHA:
    def __init__(self):
        self.captured: list = []
        self.fail_posts = False
        self.exists: dict = {}  # entity_id -> True / False / None
        self.deleted: list = []
        self.delete_result = True  # or None (unreachable) / an exception

    async def delete_state(self, entity_id):
        if isinstance(self.delete_result, BaseException):
            raise self.delete_result
        if self.delete_result is not None:
            self.deleted.append(entity_id)
        return self.delete_result

    async def set_state(self, entity_id, state, attributes=None):
        if self.fail_posts:
            return False
        self.captured.append((entity_id, state, attributes or {}))
        return True

    async def get_state_object(self, entity_id):
        return {"state": "1.0", "attributes": {"unit_of_measurement": "kW"}}

    async def get_state(self, entity_id, default=None, attribute=None):
        return default

    async def entity_exists(self, entity_id):
        v = self.exists.get(entity_id, True)
        return v.pop(0) if isinstance(v, list) else v

    def posted(self, suffix=""):
        return {eid: (s, a) for eid, s, a in self.captured if eid.endswith(suffix)}


def _exp(name="solar", **kw):
    base = dict(
        name=name, target_entity=ENTITY, units="kW", interval_minutes=30,
        future_periods=3, publish_prefix="mlfl_", publish_name=name,
        mode="production",
    )
    base.update(kw)
    return ExperimentCfg(**base)


def _make_app(*exps, forecast_every_minutes=30, retrain_every_hours=24.0):
    app = MLForecastLabApp()
    app.ha_interface = _StubHA()
    app.history_db = None
    app.web_app = None
    app._cached_models = {}
    app.config = types.SimpleNamespace(
        experiments=list(exps),
        forecast_every_minutes=forecast_every_minutes,
        retrain_every_hours=retrain_every_hours,
    )
    return app


def _publish(app, exp, y=(1.0, 2.0, 3.0)):
    ds = pd.DatetimeIndex([
        pd.Timestamp("2026-10-25 12:00") + pd.Timedelta(minutes=30 * (i + 1))
        for i in range(len(y))
    ])
    return asyncio.run(app._publish_forecast_sensors(
        exp_cfg=exp, y_pred=np.array(y, dtype=np.float32), ds_future=ds,
        model_name="lightgbm", last_trained_iso="2026-10-25T00:00:00+00:00",
    ))


def _minutes_from_now(when):
    return (when - datetime.now(timezone.utc)).total_seconds() / 60


# --- retrain retries -----------------------------------------------------

class TestRetrainRetry:
    def _failing_app(self, exc, **kw):
        exp = _exp(**kw)
        app = _make_app(exp)
        far = datetime.now(timezone.utc) + timedelta(hours=24)
        app._next_retrain_per_exp[exp.name] = far

        async def boom(_cfg):
            raise exc

        app._retrain_and_cache = boom
        return app, exp, far

    def test_ha_unavailable_retries_with_backoff(self):
        app, exp, _ = self._failing_app(
            HAUnavailableError("HA GET /api/history/period failed after 3 attempts")
        )
        for expected in HA_RETRY_MINUTES + (HA_RETRY_MINUTES[-1],):
            app._next_retrain_per_exp[exp.name] = datetime.now(timezone.utc) + timedelta(hours=24)
            asyncio.run(app._retrain_single(exp))
            assert _minutes_from_now(app._next_retrain_per_exp[exp.name]) == pytest.approx(expected, abs=0.1)

    def test_retry_never_later_than_the_retrain_cadence(self):
        app, exp, _ = self._failing_app(HAUnavailableError("down"), retrain_every_hours=0.05)
        for _ in range(3):
            app._next_retrain_per_exp[exp.name] = datetime.now(timezone.utc) + timedelta(hours=24)
            asyncio.run(app._retrain_single(exp))
        assert _minutes_from_now(app._next_retrain_per_exp[exp.name]) == pytest.approx(3, abs=0.1)

    def test_other_failures_keep_the_normal_schedule(self):
        app, exp, far = self._failing_app(ValueError("not enough data"))
        asyncio.run(app._retrain_single(exp))
        assert app._next_retrain_per_exp[exp.name] == far
        assert exp.name not in app._retrain_ha_failures

    def test_success_resets_the_backoff(self):
        app, exp, _ = self._failing_app(HAUnavailableError("down"))
        asyncio.run(app._retrain_single(exp))
        assert app._retrain_ha_failures[exp.name] == 1

        async def ok(_cfg):
            return None

        app._retrain_and_cache = ok
        asyncio.run(app._retrain_single(exp))
        assert exp.name not in app._retrain_ha_failures

    def test_failed_replacement_keeps_the_previous_model(self, caplog):
        app, exp, _ = self._failing_app(HAUnavailableError("down"), production_model="xgboost")
        old = {"model_name": "lightgbm", "trained_at": datetime.now(timezone.utc)}
        app._cached_models[exp.name] = old
        with caplog.at_level(logging.WARNING, logger="ml_forecast_lab.main"):
            asyncio.run(app._retrain_single(exp))
        assert app._cached_models[exp.name] is old
        assert "Keeping the previous lightgbm for solar" in caplog.text

    def test_successful_replacement_is_not_overwritten(self):
        app, exp, _ = self._failing_app(HAUnavailableError("down"), production_model="xgboost")
        app._cached_models[exp.name] = {"model_name": "lightgbm"}
        new = {"model_name": "xgboost"}

        async def train(_cfg):
            app._cached_models[exp.name] = new

        app._retrain_and_cache = train
        asyncio.run(app._retrain_single(exp))
        assert app._cached_models[exp.name] is new


# --- forecast retries ----------------------------------------------------

class TestForecastRetry:
    def _app(self, outcome, **kw):
        exp = _exp()
        app = _make_app(exp, **kw)
        app._cached_models[exp.name] = {"model_name": "lightgbm"}
        far = datetime.now(timezone.utc) + timedelta(minutes=30)
        app._next_forecast_per_exp[exp.name] = far

        async def forecast(_name, now=None):
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        app._forecast_with_cached = forecast
        return app, exp, far, None

    @pytest.mark.parametrize("outcome", ["no_frame", "publish_failed", HAUnavailableError("down")])
    def test_failed_tick_retries_within_minutes(self, outcome):
        app, exp, _, calls = self._app(outcome)
        asyncio.run(app._forecast_single(exp))
        assert _minutes_from_now(app._next_forecast_per_exp[exp.name]) == pytest.approx(2, abs=0.1)
        assert app._last_success_at is None
        assert app._forecast_running[exp.name] is False

    @pytest.mark.parametrize("outcome", ["channel_mismatch", "insufficient_data", ValueError("bug")])
    def test_other_outcomes_keep_the_cadence(self, outcome):
        app, exp, far, _ = self._app(outcome)
        asyncio.run(app._forecast_single(exp))
        assert app._next_forecast_per_exp[exp.name] == far

    def test_retry_capped_at_the_forecast_cadence(self):
        app, exp, _, _ = self._app("no_frame", forecast_every_minutes=3)
        for _ in range(3):
            app._next_forecast_per_exp[exp.name] = datetime.now(timezone.utc) + timedelta(minutes=30)
            asyncio.run(app._forecast_single(exp))
        assert _minutes_from_now(app._next_forecast_per_exp[exp.name]) == pytest.approx(3, abs=0.1)

    def test_success_notes_the_time_but_posts_no_heartbeat(self):
        app, exp, _, _ = self._app("ok")
        app._forecast_failures[exp.name] = 2
        asyncio.run(app._forecast_single(exp))
        assert app._last_success_at is not None
        assert HEARTBEAT_ENTITY not in app.ha_interface.posted()
        assert exp.name not in app._forecast_failures


# --- startup timers ------------------------------------------------------

class TestStartupTimers:
    def test_restored_models_forecast_shortly_after_start(self):
        now = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
        fresh, stale, none, lab = _exp("fresh"), _exp("stale"), _exp("none"), _exp("lab", mode="lab")
        app = _make_app(fresh, stale, none, lab)
        app._cached_models = {
            "fresh": {"trained_at": now - timedelta(hours=2)},
            "stale": {"trained_at": now - timedelta(hours=30)},
        }
        app._init_timers(now)
        delay = timedelta(seconds=STARTUP_FORECAST_DELAY_SECONDS)
        assert app._next_forecast_per_exp["fresh"] == now + delay
        assert app._next_forecast_per_exp["stale"] == now + delay
        assert app._next_retrain_per_exp["stale"] == now
        # No model: the retrain's own forecast publishes.
        assert app._next_forecast_per_exp["none"] == now + timedelta(minutes=30)
        assert app._next_retrain_per_exp["none"] == now
        assert app._next_forecast_per_exp["lab"] == now + timedelta(minutes=30)

    def test_startup_delay_never_exceeds_the_cadence(self):
        now = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
        app = _make_app(_exp(), forecast_every_minutes=1)
        app._cached_models = {"solar": {"trained_at": now}}
        app._init_timers(now)
        assert app._next_forecast_per_exp["solar"] == now + timedelta(minutes=1)


# --- sensors without a value ---------------------------------------------

class TestUnknownSensors:
    def test_signed_predictions_post_cumulative_as_unknown(self):
        app = _make_app()
        assert _publish(app, _exp(), y=(1.0, -0.01, 2.0)) is True
        state, attrs = app.ha_interface.posted("_cumulative")["sensor.mlfl_solar_cumulative"]
        assert state == "unknown"
        assert attrs["status"] == "signed_predictions"
        assert attrs["unit_of_measurement"] == "kW"

    def test_uncalibrated_bands_post_unknown_at_the_configured_level(self):
        app = _make_app()
        _publish(app, _exp(conformal_coverage=0.9))
        posted = app.ha_interface.posted()
        for side in ("upper", "lower"):
            state, attrs = posted[f"sensor.mlfl_solar_{side}_90"]
            assert state == "unknown"
            assert attrs["status"] == "no_conformal_bands"
            assert attrs["interval_level"] == 0.9
        assert "sensor.mlfl_solar_upper_80" not in posted

    def test_non_negative_forecast_posts_cumulative_value(self):
        app = _make_app()
        _publish(app, _exp())
        state, attrs = app.ha_interface.posted()["sensor.mlfl_solar_cumulative"]
        assert float(state) == pytest.approx(6.0)
        assert "status" not in attrs

    def test_payloads_kept_and_failure_reported(self):
        app = _make_app()
        exp = _exp()
        assert _publish(app, exp) is True
        assert {eid for eid, _, _ in app._last_payloads["solar"]} >= {
            "sensor.mlfl_solar_forecast", "sensor.mlfl_solar_cumulative",
        }
        app.ha_interface.fail_posts = True
        assert _publish(app, exp) is False


# --- HA restart detection ------------------------------------------------

class TestRepostAfterHARestart:
    def _app(self):
        solar, lab = _exp(), _exp("lab", mode="lab")
        app = _make_app(solar, lab)
        _publish(app, solar)
        asyncio.run(app._publish_lifecycle_sensor(solar, "retrain", "completed"))
        app._last_payloads["lab"] = [("sensor.mlfl_lab_forecast", "1", {})]
        app._last_payloads["gone"] = [("sensor.mlfl_gone_forecast", "1", {})]
        app._note_success()
        app.ha_interface.captured = []
        return app

    def _check(self, app, exists):
        app.ha_interface.exists[HEARTBEAT_ENTITY] = exists
        asyncio.run(app._check_ha_states())

    def test_heartbeat_posted_by_the_check_with_the_last_success_time(self):
        app = self._app()
        self._check(app, False)  # first check after start
        state, _ = app.ha_interface.posted()[HEARTBEAT_ENTITY]
        assert state == app._last_success_at.isoformat()
        assert app._ha_states_seen is True
        app.ha_interface.captured = []
        self._check(app, True)   # nothing new to report
        assert app.ha_interface.captured == []

    def test_lost_states_are_reposted_once(self, caplog):
        app = self._app()
        self._check(app, True)
        app.ha_interface.captured = []
        with caplog.at_level(logging.INFO, logger="ml_forecast_lab.main"):
            self._check(app, False)
        posted = app.ha_interface.posted()
        assert "sensor.mlfl_solar_forecast" in posted
        assert HEARTBEAT_ENTITY in posted
        # Only current production experiments, and never a lifecycle sensor.
        assert "sensor.mlfl_lab_forecast" not in posted
        assert "sensor.mlfl_gone_forecast" not in posted
        assert "sensor.mlfl_solar_last_retrain" not in posted
        assert "no longer had the app's sensors" in caplog.text
        app.ha_interface.captured = []
        self._check(app, True)
        assert app.ha_interface.captured == []

    def test_other_experiments_success_cannot_hide_the_restart(self):
        # Another experiment forecasting between HA's return and the check
        # only notes the time; the heartbeat stays missing until the check.
        app = self._app()
        self._check(app, True)
        app._note_success()
        app.ha_interface.captured = []
        self._check(app, False)
        assert "sensor.mlfl_solar_forecast" in app.ha_interface.posted()

    def test_second_restart_right_after_a_repost_is_caught(self):
        app = self._app()
        self._check(app, True)
        self._check(app, False)
        app.ha_interface.captured = []
        self._check(app, False)  # HA restarted again before any check saw it
        assert "sensor.mlfl_solar_forecast" in app.ha_interface.posted()

    def test_repost_does_not_advance_the_heartbeat(self):
        app = self._app()
        self._check(app, True)
        before = app._last_success_at
        app.ha_interface.captured = []
        self._check(app, False)
        assert app.ha_interface.posted()[HEARTBEAT_ENTITY][0] == before.isoformat()

    def test_no_repost_before_the_heartbeat_was_seen(self):
        app = self._app()
        self._check(app, False)
        assert "sensor.mlfl_solar_forecast" not in app.ha_interface.posted()

    def test_no_heartbeat_before_any_success(self):
        app = self._app()
        app._last_success_at = None
        self._check(app, False)
        assert app.ha_interface.captured == []

    def test_unreachable_ha_changes_nothing(self):
        app = self._app()
        app._ha_states_seen = True
        self._check(app, None)
        assert app._ha_states_seen is True and app.ha_interface.captured == []

    def test_running_forecast_is_not_overwritten(self):
        app = self._app()
        app._ha_states_seen = True
        app._forecast_running["solar"] = True
        self._check(app, False)
        assert "sensor.mlfl_solar_forecast" not in app.ha_interface.posted()

    def test_only_the_check_posts_the_heartbeat(self):
        src = inspect.getsource(MLForecastLabApp)
        assert src.count("self._post_heartbeat(") == 1
        assert "publish_heartbeat" not in src


class TestTrainingTabBenchmarkSensor:
    def test_completed_benchmark_posts_last_benchmark(self):
        exp = _exp()
        app = _make_app(exp)
        app.web_app = types.SimpleNamespace(state=types.SimpleNamespace(
            appstate=types.SimpleNamespace(benchmark_results={
                "solar": types.SimpleNamespace(best_model_name="nhits"),
            })))
        started = datetime.now(timezone.utc) - timedelta(seconds=90)
        asyncio.run(app._publish_benchmark_outcome(exp, started, True))
        state, attrs = app.ha_interface.posted()["sensor.mlfl_solar_last_benchmark"]
        assert attrs["outcome"] == "completed" and attrs["winner"] == "nhits"
        assert attrs["duration_seconds"] == pytest.approx(90, abs=2)

    def test_benchmark_that_ran_nothing_is_reported_failed(self):
        exp = _exp()
        app = _make_app(exp)
        asyncio.run(app._publish_benchmark_outcome(exp, datetime.now(timezone.utc), False))
        _, attrs = app.ha_interface.posted()["sensor.mlfl_solar_last_benchmark"]
        assert attrs["outcome"] == "failed"
        assert "no measured target data" in attrs["error"]
        assert "winner" not in attrs

    def test_benchmark_trigger_posts_on_completion_and_failure(self):
        src = inspect.getsource(MLForecastLabApp.start_web_server)
        trigger = src[src.index("async def _benchmark_trigger"):src.index("benchmark_callback = _benchmark_trigger")]
        assert "ran = await self._run_benchmark(exp_cfg)" in trigger
        assert "self._publish_benchmark_outcome(" in trigger
        assert '"benchmark", "failed"' in trigger

    def test_benchmark_trigger_logs_an_experiment_missing_from_config(self):
        # v2.55.1: a queued benchmark for a card whose mlfl.yaml entry is gone
        # used to return without a log line.
        src = inspect.getsource(MLForecastLabApp.start_web_server)
        trigger = src[src.index("async def _benchmark_trigger"):src.index("benchmark_callback = _benchmark_trigger")]
        assert "not found in config" in trigger

    def test_run_benchmark_reports_whether_it_ran(self):
        src = inspect.getsource(MLForecastLabApp._run_benchmark)
        assert "if df is None:" in src and "return False" in src
        assert src.rstrip().endswith("return True")


# --- dashboard card after a restart --------------------------------------

def _status(**kw):
    base = dict(best_model=None, model_version=None,
                last_benchmark_status="pending", last_benchmark_timestamp=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


class TestRestoredModelsOnTheDashboard:
    def _app(self, st):
        app = _make_app(_exp())
        self.trained = datetime(2026, 10, 25, 3, 0, tzinfo=timezone.utc)
        app._cached_models["solar"] = {
            "model_name": "lightgbm", "model_version": "v7", "trained_at": self.trained,
        }
        app.web_app = types.SimpleNamespace(state=types.SimpleNamespace(
            appstate=types.SimpleNamespace(experiment_statuses={"solar": st})))
        return app

    def test_pending_card_shows_the_restored_model(self):
        st = _status()
        self._app(st)._sync_restored_models_to_ui()
        assert st.best_model == "lightgbm" and st.model_version == "v7"
        assert st.last_benchmark_status == "completed"
        assert st.last_benchmark_timestamp == self.trained.isoformat()

    def test_saved_benchmark_values_are_kept(self):
        st = _status(best_model="nhits", last_benchmark_status="failed",
                     last_benchmark_timestamp="2026-10-20T09:15:00+00:00")
        self._app(st)._sync_restored_models_to_ui()
        assert st.best_model == "nhits"
        assert st.last_benchmark_status == "failed"
        assert st.last_benchmark_timestamp == "2026-10-20T09:15:00+00:00"
        # v7 belongs to lightgbm, not the saved winner: it would mis-filter
        # the Accuracy tab, which defaults to model_version for best_model.
        assert st.model_version is None

    def test_model_version_set_when_the_winner_is_deployed(self):
        st = _status(best_model="lightgbm", last_benchmark_status="completed")
        self._app(st)._sync_restored_models_to_ui()
        assert st.model_version == "v7"

    def test_run_syncs_after_the_web_app_starts(self):
        src = inspect.getsource(MLForecastLabApp.run)
        assert src.index("await self.start_web_server()") < src.index(
            "self._sync_restored_models_to_ui()") < src.index("await self.main_loop()")


# --- sensors the app no longer posts --------------------------------------

class TestOrphanedSensors:
    def _app(self, tmp_path=None, **kw):
        exp = _exp(**kw)
        app = _make_app(exp)
        if tmp_path is not None:
            app._published_entities_path = tmp_path / "published_entities.json"
        _publish(app, exp)
        asyncio.run(app._publish_lifecycle_sensor(exp, "retrain", "completed"))
        return app, exp

    def _check(self, app):
        asyncio.run(app._retire_orphaned_sensors())

    def test_posted_ids_are_recorded_and_persisted(self, tmp_path):
        import json
        app, _ = self._app(tmp_path)
        recorded = json.loads((tmp_path / "published_entities.json").read_text())
        assert recorded["sensor.mlfl_solar_forecast"] == "solar"
        assert recorded["sensor.mlfl_solar_last_retrain"] == "solar"
        app2 = _make_app(_exp())
        app2._published_entities_path = tmp_path / "published_entities.json"
        app2._restore_published_entities()
        assert app2._published_entities == recorded

    def test_renamed_sensors_removed_on_the_second_check(self, caplog):
        app, exp = self._app()
        app.config.experiments = [_exp(publish_name="pv")]
        self._check(app)
        assert app.ha_interface.deleted == []
        with caplog.at_level(logging.INFO, logger="ml_forecast_lab.main"):
            self._check(app)
        assert "sensor.mlfl_solar_forecast" in app.ha_interface.deleted
        assert "sensor.mlfl_solar_last_retrain" in app.ha_interface.deleted
        assert not any(e.startswith("sensor.mlfl_pv") for e in app.ha_interface.deleted)
        assert not any(k.startswith("sensor.mlfl_solar") for k in app._published_entities)
        assert "sensor(s) the app no longer posts" in caplog.text

    def test_lab_experiment_sensors_removed(self):
        app, _ = self._app()
        app.config.experiments = [_exp(mode="lab"), _exp("other")]
        self._check(app)
        self._check(app)
        assert "sensor.mlfl_solar_forecast" in app.ha_interface.deleted

    def test_changed_coverage_removes_only_the_old_bands(self):
        app, _ = self._app()
        app.config.experiments = [_exp(conformal_coverage=0.9)]
        self._check(app)
        self._check(app)
        assert set(app.ha_interface.deleted) == {
            "sensor.mlfl_solar_upper_80", "sensor.mlfl_solar_lower_80",
        }

    def test_ids_reused_by_another_experiment_are_kept(self):
        app, _ = self._app()
        app.config.experiments = [_exp("renamed", publish_name="solar")]
        self._check(app)
        self._check(app)
        assert app.ha_interface.deleted == []

    def test_unexpected_in_one_check_only_is_kept(self):
        app, exp = self._app()
        app.config.experiments = [_exp(publish_name="pv")]
        self._check(app)
        app.config.experiments = [exp]
        self._check(app)
        self._check(app)
        assert app.ha_interface.deleted == []

    @pytest.mark.parametrize("setup", ["stub", "empty"])
    def test_no_cleanup_without_a_real_config(self, setup):
        app, _ = self._app()
        if setup == "stub":
            app._config_is_stub = True
            app.config.experiments = [_exp("example")]
        else:
            app.config.experiments = []
        self._check(app)
        self._check(app)
        assert app.ha_interface.deleted == []

    def test_unreachable_ha_retries_later(self):
        app, _ = self._app()
        app.config.experiments = [_exp(publish_name="pv")]
        app.ha_interface.delete_result = None
        self._check(app)
        self._check(app)
        assert "sensor.mlfl_solar_forecast" in app._published_entities
        app.ha_interface.delete_result = True
        self._check(app)
        assert "sensor.mlfl_solar_forecast" in app.ha_interface.deleted

    def test_refused_delete_disables_cleanup(self, caplog):
        app, _ = self._app()
        app.config.experiments = [_exp(publish_name="pv")]
        app.ha_interface.delete_result = HAResponseError(405, "HA API error 405: no")
        self._check(app)
        with caplog.at_level(logging.WARNING, logger="ml_forecast_lab.main"):
            self._check(app)
        assert app._orphan_cleanup_disabled
        assert "refused to remove" in caplog.text

    def test_repost_skips_ids_the_experiment_no_longer_posts(self):
        app, _ = self._app()
        app.config.experiments = [_exp(conformal_coverage=0.9)]
        app._ha_states_seen = True
        app._note_success()
        app.ha_interface.exists[HEARTBEAT_ENTITY] = False
        app.ha_interface.delete_result = None
        app.ha_interface.captured = []
        asyncio.run(app._check_ha_states())
        posted = app.ha_interface.posted()
        assert "sensor.mlfl_solar_forecast" in posted
        assert "sensor.mlfl_solar_upper_80" not in posted

    def test_renamed_sensors_kept_until_the_retrain_moves_them(self):
        # Forecasts publish with the exp_cfg cached at the last retrain.
        app, exp = self._app()
        app._cached_models["solar"] = {"model_name": "lightgbm", "exp_cfg": exp}
        app.config.experiments = [_exp(publish_name="pv", conformal_coverage=0.9)]
        self._check(app)
        self._check(app)
        assert not any("_forecast" in e or "_80" in e for e in app.ha_interface.deleted)
        # The retrain caches the new config: the old ids are now orphans.
        app._cached_models["solar"]["exp_cfg"] = app.config.experiments[0]
        self._check(app)
        self._check(app)
        assert "sensor.mlfl_solar_forecast" in app.ha_interface.deleted
        assert "sensor.mlfl_solar_upper_80" in app.ha_interface.deleted

    def test_lab_experiments_keep_their_lifecycle_sensors(self):
        app, exp = self._app()
        asyncio.run(app._publish_lifecycle_sensor(exp, "benchmark", "completed"))
        app.config.experiments = [_exp(mode="lab")]
        self._check(app)
        self._check(app)
        assert "sensor.mlfl_solar_forecast" in app.ha_interface.deleted
        assert "sensor.mlfl_solar_last_benchmark" not in app.ha_interface.deleted
        assert "sensor.mlfl_solar_last_retrain" not in app.ha_interface.deleted

    def test_unparseable_config_on_first_load_is_treated_as_stub(self, tmp_path):
        bad = tmp_path / "mlfl.yaml"
        bad.write_text("experiments:\n\t- name: solar\n")
        app, _ = self._app()
        app.config = None
        asyncio.run(app.load_config(config_path=bad))
        assert app._config_is_stub is True
        self._check(app)
        self._check(app)
        assert app.ha_interface.deleted == []

    def test_run_restores_the_record_before_main_loop(self):
        src = inspect.getsource(MLForecastLabApp.run)
        assert src.index("self._restore_published_entities()") < src.index("await self.main_loop()")


# --- wiring --------------------------------------------------------------

def test_app_starts_after_home_assistant():
    config = (Path(__file__).resolve().parents[2] / "config.yaml").read_text()
    assert "\nstartup: application\n" in config


def test_main_loop_wiring():
    src = inspect.getsource(MLForecastLabApp.main_loop)
    assert "self._init_timers(" in src
    assert "self._spawn(self._check_ha_states())" in src
