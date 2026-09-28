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
        # Recorder timestamps carry microseconds; every 5th lands on a whole
        # second, as ~1 row in a million does on a real sensor. A cache table
        # mixing both shapes is what broke an ISO-string codec.
        t = t + pd.Timedelta(microseconds=0 if i % 5 == 0 else (i * 7919) % 999_983)
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
        self.published = getattr(self, "published", {})
        self.published[entity_id] = (state, attrs or {})
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
    raw = db.get_history(table).set_index("ds")["y"]
    # Bucketed onto the interval grid, as the conformal query's actuals are.
    act = raw.groupby(raw.index.floor("30min")).mean()
    rng = np.random.default_rng(3)
    issue_times = [t for t in act.index
                   if t < pd.Timestamp(NOW).tz_localize(None)
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
        assert report["model_roundtrip"]["status"] == "equal", report["model_roundtrip"]
        assert report["bands"] is not None

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
        """Replay compares _compute_cached_forecast + _conformal_bands; this
        pins that HA receives exactly those values. log_transform makes the
        raw output differ from the published one, so publishing the wrong
        array cannot pass."""
        app, exp, ha = setup("lightgbm", log_transform=True)
        del app._publish_forecast_sensors          # the real publisher
        cache = app._cached_models[exp.name]
        run = asyncio.run(replay._run_forecast(app, cache, NOW))
        fc, bands = run["fc"], run["bands"]
        assert fc.status == "ok" and bands is not None
        assert not np.allclose(fc.y_pred, fc.y_pred_raw)

        ha.published = {}
        asyncio.run(app._forecast_with_cached(exp.name, now=NOW))
        base = f"sensor.{exp.publish_prefix}{exp.publish_name or exp.name}"

        def values(suffix):
            return [p["value"] for p in ha.published[base + suffix][1]["forecast"]]

        pct = int(round(bands.level * 100))
        assert values("_forecast") == [round(float(v), 4) for v in fc.y_pred]
        assert values(f"_upper_{pct}") == [round(float(v), 4) for v in bands.upper]
        assert values(f"_lower_{pct}") == [round(float(v), 4) for v in bands.lower]


# ---- review findings (v2.52.2) ----

class TestHistoryCodec:
    def test_mixed_timestamp_precision_round_trips(self, tmp_path):
        db = HistoryDB(str(tmp_path / "c.db"))
        table = db.safe_table_name("sensor.x")
        ds = pd.to_datetime(["2026-09-21 11:00:03.123456", "2026-09-21 11:05:00",
                             "2026-09-21 11:10:07.654321"], format="ISO8601")
        db.store_history(table, pd.DataFrame({"ds": ds, "value": [1.0, None, 3.0]}))
        live = db.get_history(table)
        rec = replay.RecordingHistoryDB(db, replay._CallLog()).get_history(table)
        pd.testing.assert_frame_equal(rec, live)

    def test_format_1_iso_rows_still_decode(self):
        payload = {"rows": [["2026-09-21T11:00:03.123456", 1.0],
                            ["2026-09-21T11:05:00", 2.0]], "y_dtype": "float64"}
        df = replay._rows_to_history_frame(payload)
        assert list(df["ds"]) == [pd.Timestamp("2026-09-21 11:00:03.123456"),
                                  pd.Timestamp("2026-09-21 11:05:00")]


class TestRecordedErrors:
    @pytest.mark.parametrize("exc", [OSError("disk I/O error"), KeyError("x"),
                                     type("OperationalError", (Exception,), {})("locked")])
    def test_replayed_error_renders_like_the_original(self, exc):
        log = replay._CallLog()
        log.add("db", "k", error=exc)
        q = replay._CallQueue(json.loads(json.dumps(log.calls)))
        with pytest.raises(Exception) as got:
            q.pop("db", "k")
        assert f"{type(got.value).__name__}: {got.value}" == f"{type(exc).__name__}: {exc}"
        if type(exc).__module__ == "builtins":
            assert isinstance(got.value, type(exc))

    def test_a_failure_at_capture_replays_identically(self, setup, tmp_path):
        import sqlite3
        app, exp, _ = setup("lightgbm")

        def locked(*a, **kw):
            raise sqlite3.OperationalError("database is locked")
        app.history_db.get_conformal_quantiles = locked
        payload = _capture(app, exp)
        report = _zip_json(payload, "forecast/report.json")
        assert report["bands_error"] == "OperationalError: database is locked"
        res = _replay(_write(tmp_path, payload))
        assert res.matches, res.render()


class TestHistoryTrim:
    def test_rows_older_than_the_window_are_not_bundled(self, setup, tmp_path):
        app, exp, _ = setup("lightgbm")
        table = app.history_db.safe_table_name(exp.target_entity)
        old = pd.date_range(pd.Timestamp(NOW).tz_localize(None) - pd.Timedelta(days=40),
                            periods=48, freq="30min")
        app.history_db.store_history(table, pd.DataFrame({"ds": old, "value": 9.0}))
        payload = _capture(app, exp)
        floor = (pd.Timestamp(NOW) - pd.Timedelta(days=exp.days_history + 1)).tz_localize(None)
        bundle = replay._Bundle(_write(tmp_path, payload))
        for name in ("calls.json", "forecast/calls.json"):
            for c in bundle.calls(name):
                if c["key"].startswith('["get_history"') and c["target"] == "db":
                    df = replay._rows_to_history_frame(c["response"])
                    assert df.empty or df["ds"].min() >= floor, name
        assert _replay(bundle._files and _write(tmp_path, payload, "b2.zip")).matches


class TestCrossHostPolicy:
    def test_policy_by_backend_family(self):
        assert replay._comparison_policy("nlinear", same_host=True) == "exact"
        assert replay._comparison_policy("lightgbm", same_host=False) == "exact"
        assert replay._comparison_policy("nlinear", same_host=False) == "relative"
        assert replay._comparison_policy("arima", same_host=False) == "informational"

    def test_tolerance_is_scaled_by_the_raw_output_when_clamped_to_zero(self):
        exp = {"y_pred": np.zeros(4, np.float32),
               "model_output": np.array([-3.0, -1.0, 5.0, 2.0], np.float32)}
        tol = replay._array_tolerance("relative", exp, "y_pred")
        assert tol == pytest.approx(replay._NEURAL_REL_TOL * 5.0)
        assert replay._array_tolerance("exact", exp, "y_pred") == 0.0

    def test_cpu_identity_is_part_of_the_host(self):
        a = replay._host_fingerprint()
        b = json.loads(json.dumps(a))
        assert replay._same_host(a, b)
        b["cpu"]["model"] = "some other CPU"
        assert not replay._same_host(a, b)

    def _perturbed(self, payload, tmp_path, scale, other_host, name):
        d = _extract(payload, tmp_path, name)
        if other_host:
            m = json.loads((d / "manifest.json").read_text())
            m["host"]["machine"] = "elsewhere"
            (d / "manifest.json").write_text(json.dumps(m))
        parts = replay._npz_to_parts((d / "forecast/expected.npz").read_bytes())
        y = parts["y_pred"].astype(np.float64)
        parts["y_pred"] = (y + scale * max(1.0, float(np.abs(y).max()))).astype(np.float32)
        (d / "forecast/expected.npz").write_bytes(replay._arrays_to_npz(parts))
        return _replay(d)

    def test_neural_across_hosts_tolerates_drift_but_not_change(self, setup, tmp_path):
        app, exp, _ = setup("nlinear")
        payload = _capture(app, exp)
        small = self._perturbed(payload, tmp_path, 1e-6, True, "small")
        assert small.stages["forecast"] == [], small.render()
        assert any("different host" in n for n in small.notes)
        large = self._perturbed(payload, tmp_path, 1e-2, True, "large")
        assert any(d.startswith("forecast.y_pred") for d in large.stages["forecast"])

    def test_same_host_rejects_one_ulp(self, setup, tmp_path):
        app, exp, _ = setup("lightgbm")
        d = _extract(_capture(app, exp), tmp_path)
        parts = replay._npz_to_parts((d / "forecast/expected.npz").read_bytes())
        parts["y_pred"] = parts["y_pred"].copy()
        parts["y_pred"][0] = np.nextafter(parts["y_pred"][0], np.float32(np.inf))
        (d / "forecast/expected.npz").write_bytes(replay._arrays_to_npz(parts))
        res = _replay(d)
        assert any(x.startswith("forecast.y_pred") for x in res.stages["forecast"])


class TestBundleSafety:
    @pytest.mark.parametrize("name", ["../escape.bin", "/etc/escape.bin",
                                      "a/../../escape.bin", "a\\..\\escape.bin"])
    def test_escaping_entry_names_are_refused(self, tmp_path, name):
        with pytest.raises(ValueError):
            replay._safe_member_path(tmp_path.resolve(), name)

    def test_zip_slip_bundle_is_refused(self, setup, tmp_path):
        app, exp, _ = setup("lightgbm")
        payload = _capture(app, exp)
        buf = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(payload)) as src, \
                zipfile.ZipFile(buf, "w") as dst:
            for n in src.namelist():
                dst.writestr(n, src.read(n))
            dst.writestr("forecast/model/../../../escaped.txt", b"x")
        with pytest.raises(ValueError):
            _replay(_write(tmp_path, buf.getvalue(), "slip.zip"))
        assert not list(tmp_path.parent.glob("escaped.txt"))

    def test_capture_never_saves_the_live_model_object(self, setup, monkeypatch):
        app, exp, _ = setup("lightgbm")
        live = app._cached_models[exp.name]["model"]
        saved_from = []
        real = LightGBMModel.save

        def spy(self, path):
            saved_from.append(id(self))
            return real(self, path)
        monkeypatch.setattr(LightGBMModel, "save", spy)
        _capture(app, exp)
        assert saved_from and id(live) not in saved_from


class TestDiskCheck:
    def test_stale_model_on_disk_is_reported(self, setup, tmp_path):
        app, exp, _ = setup("lightgbm")
        meta_file = MLForecastLabApp._cached_model_dir(exp.name) / "cache_meta.json"
        meta = json.loads(meta_file.read_text())
        meta["model_version"] = "2020-01-01T00:00:00Z"
        meta_file.write_text(json.dumps(meta))
        payload = _capture(app, exp)
        disk = _zip_json(payload, "manifest.json")["forecast"]["disk"]
        assert disk["consistent"] is False and disk["mismatched_fields"] == ["model_version"]
        res = _replay(_write(tmp_path, payload))
        assert any("restart would have served a different model" in n for n in res.notes)

    def test_disk_is_checked_even_when_the_save_fails(self, setup, monkeypatch):
        app, exp, _ = setup("lightgbm")

        def broken(self, path):
            raise OSError("no space left on device")
        monkeypatch.setattr(LightGBMModel, "save", broken)
        fsum = _zip_json(_capture(app, exp), "manifest.json")["forecast"]
        assert fsum["status"] == "model_save_failed"
        assert fsum["disk"]["consistent"] is True
