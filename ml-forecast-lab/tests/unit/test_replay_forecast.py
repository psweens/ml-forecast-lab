"""Replay bundles, forecast stage (v2.52.2).

A training-frame replay cannot reproduce a wrong published forecast: the
forecast is built from a separate fresh fetch, future covariates (attribute
forecasts and the ``weather.get_forecasts`` service), the cached model and a
conformal band read from the forecast log — none of which the frame stage
records. The forecast stage bundles the model saved from memory, records that
I/O, and replays ``_compute_cached_forecast`` + ``_conformal_bands``.

These tests pin, with real LightGBM and NLinear models: tree and neural
forecasts (point, raw output, inputs, band) replay bit-identically; capture
has no side effects on the live app; a change to inference is reported at the
forecast stage with its inputs; a request missing from the bundle fails the
run even where the pipeline swallows the error; the lead-bucket quantile keys
survive JSON; the forecast stage needs --trust-model; the saved model is
checked against the live one; and the compute/publish split publishes exactly
what it computes.
"""
from __future__ import annotations

import asyncio
import io
import json
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab import main as main_mod
from ml_forecast_lab import replay
from ml_forecast_lab.config import AppConfig, CovariateCfg, ExperimentCfg
from ml_forecast_lab.covariates import CovariateResolver
from ml_forecast_lab.db import HistoryDB
from ml_forecast_lab.main import MLForecastLabApp
from ml_forecast_lab.models.lightgbm_backend import LightGBMModel
from ml_forecast_lab.models.registry import ModelRegistry

_now = datetime.now(timezone.utc)
NOW = _now.replace(minute=30 if _now.minute >= 30 else 0, second=0, microsecond=0)
WEATHER = "weather.home"


# ---- harness (after test_replay.py / test_covariate_coverage.py) ----

def _rows(days, cadence_min=15, value=lambda i: i % 37 + 1.0, attrs=None):
    ts = pd.date_range(NOW - timedelta(days=days), NOW,
                       freq=f"{cadence_min}min", tz="UTC")
    out = []
    for i, t in enumerate(ts):
        r = {"last_changed": t.isoformat(), "state": f"{value(i):.4f}"}
        if attrs is not None:
            r["state"] = "cloudy"
            r["attributes"] = attrs(i)
        out.append(r)
    return out


def _future(step_min, n, key, value):
    ts = pd.date_range(NOW, periods=n, freq=f"{step_min}min", tz="UTC")
    return [{"datetime": t.isoformat(), key: value(i)} for i, t in enumerate(ts)]


class _StubHA:
    def __init__(self):
        self.rows = {
            "sensor.load": _rows(7),
            "sensor.temp": _rows(7, value=lambda i: (i % 17) * 0.5 + 3.0),
            "sensor.solcast": _rows(7, value=lambda i: float(max(0, 20 - abs(i % 96 - 48)))),
            WEATHER: _rows(7, attrs=lambda i: {"temperature": 10.0 + (i % 24) * 0.3}),
        }
        self.attributes = {
            ("sensor.solcast", "forecast"): _future(
                30, 96, "pv_estimate", lambda i: float(max(0, 18 - abs(i - 24)))),
        }
        self.weather = _future(60, 48, "temperature", lambda i: 12.0 + (i % 12) * 0.25)
        self.set_calls = []
        self.api_calls = []

    async def get_history(self, entity_id, start, end, include_attributes=False):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        s = s.tz_localize("UTC") if s.tzinfo is None else s
        e = e.tz_localize("UTC") if e.tzinfo is None else e
        return [r for r in self.rows.get(entity_id, [])
                if s <= pd.Timestamp(r["last_changed"]) <= e]

    async def get_state(self, entity_id, default=None, attribute=None):
        if attribute is not None:
            return self.attributes.get((entity_id, attribute), default)
        return default

    async def get_config(self):
        return {"latitude": 52.2, "longitude": 0.12, "location_name": "Home"}

    async def api_call(self, method, endpoint, params=None, json_data=None, **kw):
        self.api_calls.append((method, endpoint, json_data))
        entity = (json_data or {}).get("entity_id")
        return {"service_response": {entity: {"forecast": list(self.weather)}},
                "context": {"id": "x"}}

    async def set_state(self, entity_id, state, attrs=None):
        self.set_calls.append(entity_id)
        return True


def _registry():
    from ml_forecast_lab.models.nlinear_backend import NLinearModel
    reg = ModelRegistry()
    reg.register("lightgbm", LightGBMModel)
    reg.register("nlinear", NLinearModel)
    return reg


def _exp(model, **kw):
    covs = [
        CovariateCfg(entity="sensor.temp", role="lagged"),
        CovariateCfg(entity="sensor.solcast", role="both"),
    ]
    if model == "nlinear":
        covs.append(CovariateCfg(entity=WEATHER, role="both",
                                 future_attribute="hourly",
                                 future_value_key="temperature"))
    base = dict(
        name=f"fc_{model}", target_entity="sensor.load", days_history=6,
        interval_minutes=30, future_periods=12, mode="production",
        production_model=model, models_enabled=[model], covariates=covs,
        model_params={"nlinear": {"epochs": 2, "patience": 1}},
    )
    base.update(kw)
    return ExperimentCfg(**base)


def _seed_residuals(app, exp):
    """Forecast-log history whose residuals grow with lead time, so the
    per-lead quantiles differ and a lost int key would show."""
    db = app.history_db
    cache = app._cached_models[exp.name]
    table = db.safe_table_name(exp.target_entity)
    act = db.get_history(table).set_index("ds")["y"]
    rng = np.random.default_rng(3)
    issue_times = [t for t in act.index
                   if t.minute in (0, 30) and t < pd.Timestamp(NOW).tz_localize(None)
                   - pd.Timedelta(hours=2)][-60:]
    for t in issue_times:
        targets = [t + pd.Timedelta(minutes=30), t + pd.Timedelta(minutes=60)]
        preds = [float(act[targets[0]]) + rng.normal(0, 1),
                 float(act[targets[1]]) + rng.normal(0, 6)]
        db.log_forecast(exp.name, t.to_pydatetime(),
                        [x.to_pydatetime() for x in targets], preds,
                        cache["model_name"], model_version=cache["model_version"])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(MLForecastLabApp, "_cached_model_dir",
                        staticmethod(lambda n: tmp_path / "models" / n))
    monkeypatch.setattr(main_mod, "build_model_registry", _registry)

    def make(model, seed=True, **kw):
        exp = _exp(model, **kw)
        ha = _StubHA()
        app = MLForecastLabApp()
        app.history_db = HistoryDB(str(tmp_path / f"{model}.db"))
        app.history_db.ensure_forecast_log_table()
        app.ha_interface = ha
        app.config = AppConfig(experiments=[exp])
        app.covariate_resolver = CovariateResolver(
            ha, history_db=app.history_db,
            retention_provider=app._retention_days_for_table)
        app.model_registry = _registry()
        app.published = []

        async def _publish(**kwargs):
            app.published.append(kwargs)
        app._publish_forecast_sensors = _publish
        asyncio.run(app._retrain_and_cache(exp))
        assert exp.name in app._cached_models, "retrain did not cache a model"
        if seed:
            _seed_residuals(app, exp)
        return app, exp, ha
    return make


def _capture(app, exp):
    return asyncio.run(replay.capture_bundle(app, exp, now=NOW))


def _write(tmp_path, payload, name="bundle.zip"):
    p = tmp_path / name
    p.write_bytes(payload)
    return p


def _replay(path, **kw):
    kw.setdefault("trust_model", True)
    return asyncio.run(replay.replay_bundle(path, **kw))


def _zip_json(payload, name):
    return json.loads(zipfile.ZipFile(io.BytesIO(payload)).read(name))


def _extract(payload, tmp_path, name="bundle_dir"):
    d = tmp_path / name
    zipfile.ZipFile(io.BytesIO(payload)).extractall(d)
    return d


# ---- round trips ----

class TestForecastRoundTrip:
    def test_tree_forecast_replays_identically(self, setup, tmp_path):
        app, exp, _ = setup("lightgbm")
        payload = _capture(app, exp)
        report = _zip_json(payload, "forecast/report.json")
        assert report["status"] == "ok" and report["path"] == "tree"
        assert report["used_fresh_frame"] is True
        assert "solcast" in " ".join(report["future_covariates"])
        assert report["bands"] is not None, "seeded residuals should give a band"
        assert report["model_roundtrip"]["status"] == "equal"
        manifest = _zip_json(payload, "manifest.json")
        assert manifest["forecast"]["disk"]["consistent"] is True
        assert manifest["capture_misses"] == [] and manifest["forecast"]["misses"] == []

        res = _replay(_write(tmp_path, payload))
        assert res.matches, res.render()
        assert res.stages["forecast"] == []
        assert res.unused_calls == []

    def test_neural_forecast_with_weather_service_replays_identically(self, setup, tmp_path):
        app, exp, ha = setup("nlinear")
        payload = _capture(app, exp)
        report = _zip_json(payload, "forecast/report.json")
        assert report["status"] == "ok" and report["path"] == "neural"
        # Column names come from the entity id: weather.home -> "home".
        assert set(report["future_covariates"]) >= {"home", "solcast"}, (
            report["future_covariates"])
        calls = _zip_json(payload, "forecast/calls.json")
        assert any('"api_call"' in c["key"] for c in calls)

        res = _replay(_write(tmp_path, payload))
        assert res.matches, res.render()
        assert res.stages["forecast"] == []

    def test_lead_bucket_quantiles_survive_the_bundle(self, setup, tmp_path):
        app, exp, _ = setup("lightgbm")
        payload = _capture(app, exp)
        parts = replay._npz_to_parts(
            zipfile.ZipFile(io.BytesIO(payload)).read("forecast/expected.npz"))
        q = parts["q_vec"]
        assert len(set(q[:2].tolist())) == 2, (
            "leads 30 and 60 have different residual scales, so their "
            "quantiles differ — equal values mean the int keys were lost "
            "and every lead fell back to the pooled quantile")
        assert _replay(_write(tmp_path, payload)).matches


# ---- capture discipline ----

class TestCaptureSideEffects:
    def test_capture_leaves_the_live_app_untouched(self, setup, tmp_path):
        app, exp, ha = setup("lightgbm")
        entry = app._cached_models[exp.name]
        model_dir = MLForecastLabApp._cached_model_dir(exp.name)
        before_files = {p.name: p.read_bytes() for p in model_dir.iterdir() if p.is_file()}
        n_log = app.history_db.conn.execute("SELECT COUNT(*) FROM forecast_log").fetchone()[0]
        n_pub, n_set = len(app.published), len(ha.set_calls)

        _capture(app, exp)

        assert app._cached_models[exp.name] is entry
        assert {p.name: p.read_bytes() for p in model_dir.iterdir()
                if p.is_file()} == before_files
        assert app.history_db.conn.execute(
            "SELECT COUNT(*) FROM forecast_log").fetchone()[0] == n_log
        assert len(app.published) == n_pub and len(ha.set_calls) == n_set

    def test_forecast_uses_the_trained_config_not_a_later_edit(self, setup, tmp_path):
        app, exp, _ = setup("lightgbm")
        edited = _exp("lightgbm", days_history=5)
        app.config = AppConfig(experiments=[edited])
        payload = _capture(app, edited)
        manifest = _zip_json(payload, "manifest.json")
        assert manifest["experiment_config"]["days_history"] == 5
        assert manifest["forecast"]["experiment_config"]["days_history"] == 6
        assert _replay(_write(tmp_path, payload)).matches

    def test_no_cached_model_records_an_absent_stage(self, tmp_path):
        from ml_forecast_lab.replay import _capture_forecast
        app = MLForecastLabApp()
        summary = asyncio.run(_capture_forecast(app, "nope", NOW, {}))
        assert summary["status"] == "absent"


class TestModelRoundTrip:
    def test_a_lossy_save_is_reported(self, setup, tmp_path):
        app, exp, _ = setup("lightgbm")

        class _Lossy(LightGBMModel):
            def predict(self, X, **kw):
                return super().predict(X, **kw) + 1.0

        app.model_registry.register("lightgbm", _Lossy)
        payload = _capture(app, exp)
        rt = _zip_json(payload, "forecast/report.json")["model_roundtrip"]
        assert rt["status"] == "differs", rt


# ---- detection and strictness ----

class TestDetectsChange:
    def test_inference_change_is_reported_with_its_inputs(self, setup, tmp_path, monkeypatch):
        app, exp, _ = setup("lightgbm")
        payload = _capture(app, exp)
        real = main_mod._nan_to_num_guarded

        def shifted(X, *a, **kw):
            out = real(X, *a, **kw)
            if a and a[0] == "cached recursive forecast row":
                out = out + np.float32(0.5)
            return out

        monkeypatch.setattr(main_mod, "_nan_to_num_guarded", shifted)
        res = _replay(_write(tmp_path, payload))
        assert res.stages["frame"] == [] and res.stages["windows"] == []
        diffs = res.stages["forecast"]
        assert any(d.startswith("forecast.X_rows") for d in diffs), res.render()
        assert any(d.startswith("forecast.y_pred") for d in diffs), res.render()


class TestStrictness:
    def test_a_swallowed_miss_still_fails_the_run(self, setup, tmp_path):
        """The band helper's caller catches every error, so a missing
        conformal record would otherwise replay as "no band"."""
        app, exp, _ = setup("lightgbm")
        d = _extract(_capture(app, exp), tmp_path)
        calls = json.loads((d / "forecast/calls.json").read_text())
        (d / "forecast/calls.json").write_text(json.dumps(
            [c for c in calls if "get_conformal_quantiles" not in c["key"]]))
        res = _replay(d)
        assert res.misses and not res.matches, res.render()
        assert replay.main([str(d), "--trust-model"]) == 2

    def test_missing_weather_service_record_fails_the_run(self, setup, tmp_path):
        app, exp, _ = setup("nlinear")
        d = _extract(_capture(app, exp), tmp_path)
        calls = json.loads((d / "forecast/calls.json").read_text())
        (d / "forecast/calls.json").write_text(json.dumps(
            [c for c in calls if '"api_call"' not in c["key"]]))
        assert replay.main([str(d), "--trust-model"]) == 2

    def test_api_call_outside_the_allowlist_is_a_capture_miss(self):
        log = replay._CallLog()
        rec = replay.RecordingHA(_StubHA(), log)
        with pytest.raises(PermissionError):
            asyncio.run(rec.api_call("POST", "/api/states/sensor.x", json_data={}))
        assert log.misses == ["HA.api_call POST /api/states/sensor.x"]

    def test_unknown_method_is_a_miss_but_dunders_are_not(self):
        log = replay._CallLog()
        rec = replay.RecordingHistoryDB(HistoryDB(":memory:"), log)
        with pytest.raises(AttributeError):
            rec.log_forecast
        with pytest.raises(AttributeError):
            rec.__deepcopy__
        assert log.misses == ["HistoryDB.log_forecast"]


class TestTrustAndCompatibility:
    def test_forecast_stage_needs_trust_model(self, setup, tmp_path, capsys):
        app, exp, _ = setup("lightgbm")
        p = _write(tmp_path, _capture(app, exp))
        res = _replay(p, trust_model=False)
        assert "forecast" not in res.stages
        assert any("--trust-model" in n for n in res.notes)
        assert replay.main([str(p)]) == 0
        assert "--trust-model" in capsys.readouterr().out

    def test_schema_mismatch_is_not_comparable(self, setup, tmp_path):
        app, exp, _ = setup("lightgbm")
        d = _extract(_capture(app, exp), tmp_path)
        meta = json.loads((d / "forecast/meta.json").read_text())
        meta["schema_version"] = 99
        (d / "forecast/meta.json").write_text(json.dumps(meta))
        res = _replay(d)
        assert "forecast" not in res.stages
        assert any("not comparable" in n for n in res.notes)

    def test_large_responses_are_stored_once(self, setup, tmp_path):
        app, exp, _ = setup("lightgbm")
        z = zipfile.ZipFile(io.BytesIO(_capture(app, exp)))
        refs = [c["ref"] for name in ("calls.json", "forecast/calls.json")
                for c in json.loads(z.read(name)) if "ref" in c]
        blobs = [n for n in z.namelist() if n.startswith("responses/")]
        assert len(refs) > len(blobs), (
            "the forecast stage re-fetches the frame stage's history; each "
            "distinct response should be stored once")


class TestConformalCodec:
    def test_int_keys_round_trip(self):
        resp = {"quantiles": {30: 1.5, 60: 4.25}, "fallback_quantile": 3.0,
                "sample_counts": {30: 12, 60: 11}, "total_samples": 23}
        enc = replay._encode_conformal(resp)
        dec = replay._decode_conformal(json.loads(json.dumps(enc)))
        assert dec == resp
        assert all(isinstance(k, int) for k in dec["quantiles"])


# ---- the compute/publish split ----

class TestComputePublishSplit:
    def test_what_is_published_is_what_is_computed(self, setup):
        app, exp, _ = setup("lightgbm", seed=False)
        app.published.clear()
        asyncio.run(app._forecast_with_cached(exp.name, now=NOW))
        (pub,) = app.published
        fc = asyncio.run(app._compute_cached_forecast(
            app._cached_models[exp.name], now=NOW))
        assert fc.status == "ok"
        assert np.array_equal(pub["y_pred"], fc.y_pred)
        assert pub["ds_future"].equals(fc.ds_future)
        assert pub["model_name"] == fc.model_name == "lightgbm"
        assert pub["last_trained_iso"] == fc.last_trained.isoformat()
