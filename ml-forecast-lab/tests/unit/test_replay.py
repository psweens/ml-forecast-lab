"""Replay bundles (v2.52.2).

The per-retrain debug dump captured the supervised frame *after*
preprocessing, feature construction and missingness resolution, and nothing
at all of what the forecast fetched, so a production failure could not be
re-run: the stages where regressions have historically lived were upstream
of the capture. A replay bundle records the pipeline's inputs at its I/O
boundary (HA responses, history-cache reads, config, wall clock) and replay
drives the production methods with them.

These tests pin: a capture replays bit-identically through every stage;
capture leaves the user's cache untouched; a pipeline change is reported at
the stage and column it affects; and a request the bundle never recorded
fails loudly instead of degrading into a data gap.
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

from ml_forecast_lab import replay
from ml_forecast_lab.config import AppConfig, CovariateCfg, ExperimentCfg
from ml_forecast_lab.covariates import CovariateResolver
from ml_forecast_lab.db import HistoryDB
from ml_forecast_lab.main import MLForecastLabApp

NOW = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


# ---- harness (after test_covariate_coverage.py) ----

class _StubHA:
    def __init__(self, rows_by_entity, config=None):
        self.rows_by_entity = rows_by_entity
        self.config = config or {"latitude": 52.2, "longitude": 0.12,
                                 "location_name": "Home", "components": ["x"]}

    async def get_history(self, entity_id, start, end, include_attributes=False):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if s.tzinfo is None:
            s = s.tz_localize("UTC")
        if e.tzinfo is None:
            e = e.tz_localize("UTC")
        return [
            r for r in self.rows_by_entity.get(entity_id, [])
            if s <= pd.Timestamp(r["last_changed"]) <= e
        ]

    async def get_state(self, entity_id, default=None, attribute=None):
        return default

    async def get_config(self):
        return dict(self.config)


def _recorder_rows(days, cadence_min=15, value=lambda i: i % 37 + 1.0,
                   drop=None):
    ts = pd.date_range(NOW - timedelta(days=days), NOW,
                       freq=f"{cadence_min}min", tz="UTC")
    return [
        {"last_changed": t.isoformat(), "state": f"{value(i):.4f}"}
        for i, t in enumerate(ts)
        if not (drop and drop[0] <= t < drop[1])
    ]


class _NeuralRegistry:
    class _M:
        is_neural = True

    def create(self, name):
        return self._M()


def _exp(**kw):
    base = dict(
        name="replay",
        target_entity="sensor.load",
        days_history=6,
        interval_minutes=30,
        covariates=[CovariateCfg(entity="sensor.temp", role="lagged")],
        models_enabled=["lightgbm"],
    )
    base.update(kw)
    return ExperimentCfg(**base)


def _rows(gap=True):
    drop = (pd.Timestamp(NOW - timedelta(days=3)),
            pd.Timestamp(NOW - timedelta(days=3) + timedelta(hours=5))) if gap else None
    return {
        "sensor.load": _recorder_rows(7, drop=drop),
        "sensor.temp": _recorder_rows(7, value=lambda i: (i % 17) * 0.5 + 3.0),
    }


def _make_app(tmp_db, exp_cfg, rows, neural=False, precache_days=0):
    app = MLForecastLabApp()
    app.history_db = HistoryDB(tmp_db)
    app.ha_interface = _StubHA(rows)
    app.config = AppConfig(experiments=[exp_cfg])
    app.covariate_resolver = CovariateResolver(
        app.ha_interface, history_db=app.history_db,
        retention_provider=app._retention_days_for_table,
    )
    if neural:
        app.model_registry = _NeuralRegistry()
    if precache_days:
        # Older rows only in the cache: replay must serve them from the
        # recorded cache read, not from HA.
        cached = pd.DataFrame({
            "ds": pd.date_range(NOW - timedelta(days=6), NOW - timedelta(days=precache_days),
                                freq="30min").tz_localize(None),
            "value": 5.0,
        })
        app.history_db.store_history(
            app.history_db.safe_table_name(exp_cfg.target_entity), cached)
    return app


def _capture(app, exp):
    return asyncio.run(replay.capture_bundle(app, exp, now=NOW))


def _replay(tmp_path, payload, until="windows"):
    p = tmp_path / "bundle.zip"
    p.write_bytes(payload)
    return asyncio.run(replay.replay_bundle(p, until=until))


# ---- tests ----

class TestRoundTrip:
    def test_tabular_capture_replays_identically(self, tmp_db, tmp_path):
        exp = _exp()
        payload = _capture(_make_app(tmp_db, exp, _rows(), precache_days=2), exp)
        res = _replay(tmp_path, payload)
        assert res.matches, res.render()
        assert set(res.stages) == {"grid", "frame", "windows"}
        assert res.unused_calls == []

    def test_neural_windows_replay_identically(self, tmp_db, tmp_path):
        exp = _exp()
        payload = _capture(_make_app(tmp_db, exp, _rows(), neural=True), exp)
        report = json.loads(zipfile.ZipFile(io.BytesIO(payload)).read("expected/report.json"))
        assert report["windows"] is not None
        assert report["windows"]["seq_X_shape"][0] > 0
        res = _replay(tmp_path, payload)
        assert res.matches, res.render()

    def test_grid_only_replay(self, tmp_db, tmp_path):
        exp = _exp()
        payload = _capture(_make_app(tmp_db, exp, _rows()), exp)
        res = _replay(tmp_path, payload, until="grid")
        assert list(res.stages) == ["grid"] and res.matches


class TestCaptureSideEffects:
    def test_capture_does_not_write_the_cache(self, tmp_db):
        exp = _exp()
        app = _make_app(tmp_db, exp, _rows(), precache_days=2)
        table = app.history_db.safe_table_name(exp.target_entity)
        before = len(app.history_db.get_history(table))
        _capture(app, exp)
        assert len(app.history_db.get_history(table)) == before

    def test_ha_config_is_trimmed_to_coordinates(self, tmp_db):
        exp = _exp()
        payload = _capture(_make_app(tmp_db, exp, _rows(), neural=True), exp)
        calls = json.loads(zipfile.ZipFile(io.BytesIO(payload)).read("calls.json"))
        cfg_calls = [c for c in calls if '"get_config"' in c["key"]]
        assert cfg_calls, "neural window build reads the site location"
        assert set(cfg_calls[0]["response"]) == {"latitude", "longitude"}

    def test_manifest_carries_full_experiment_config(self, tmp_db):
        exp = _exp()
        payload = _capture(_make_app(tmp_db, exp, _rows()), exp)
        manifest = json.loads(zipfile.ZipFile(io.BytesIO(payload)).read("manifest.json"))
        assert manifest["captured_at"] == pd.Timestamp(NOW).isoformat()
        assert manifest["experiment_config"]["covariates"][0]["entity"] == "sensor.temp"
        assert manifest["experiment_config"]["gap_handling"] == exp.gap_handling


class TestDetectsChange:
    def test_feature_change_reported_at_frame_stage(self, tmp_db, tmp_path, monkeypatch):
        exp = _exp()
        payload = _capture(_make_app(tmp_db, exp, _rows()), exp)

        import ml_forecast_lab.features as features
        real = features.build_features

        def perturbed(*a, **kw):
            out = real(*a, **kw)
            col = next(c for c in out.columns if c.startswith("hour"))
            out[col] = out[col] + 1.0
            return out

        monkeypatch.setattr(features, "build_features", perturbed)
        res = _replay(tmp_path, payload, until="frame")
        assert res.stages["grid"] == []
        assert not res.matches
        assert any(d.startswith("frame.hour") for d in res.stages["frame"]), res.render()

    def test_changed_windows_are_reported(self, tmp_db, tmp_path, monkeypatch):
        exp = _exp()
        payload = _capture(_make_app(tmp_db, exp, _rows(), neural=True), exp)

        import ml_forecast_lab.features as features
        real = features.create_sliding_windows

        def shifted(*a, **kw):
            X, y, names, kept = real(*a, **kw)
            return X * 2.0, y, names, kept

        monkeypatch.setattr(features, "create_sliding_windows", shifted)
        res = _replay(tmp_path, payload)
        assert res.stages["frame"] == []
        assert res.stages["windows"], res.render()


class TestStrictness:
    def _extract(self, payload, tmp_path):
        d = tmp_path / "bundle"
        zipfile.ZipFile(io.BytesIO(payload)).extractall(d)
        return d

    def test_unrecorded_request_raises(self, tmp_db, tmp_path):
        exp = _exp()
        d = self._extract(_capture(_make_app(tmp_db, exp, _rows()), exp), tmp_path)
        m = json.loads((d / "manifest.json").read_text())
        m["captured_at"] = pd.Timestamp(NOW + timedelta(minutes=1)).isoformat()
        (d / "manifest.json").write_text(json.dumps(m))
        with pytest.raises(replay.UnrecordedCall):
            asyncio.run(replay.replay_bundle(d))

    def test_cli_exit_codes(self, tmp_db, tmp_path, capsys):
        exp = _exp()
        payload = _capture(_make_app(tmp_db, exp, _rows()), exp)
        p = tmp_path / "b.zip"
        p.write_bytes(payload)
        assert replay.main([str(p)]) == 0
        assert "identical" in capsys.readouterr().out
        d = self._extract(payload, tmp_path)
        (d / "calls.json").write_text("[]")
        assert replay.main([str(d)]) == 2


class TestFrameStorage:
    def test_npz_round_trip_is_exact(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="30min")
        df = pd.DataFrame({
            "a": [0.1, np.nan, 1e-300, -2.5, 3.0],
            "b": np.array([1, 0, 1, 1, 0], dtype=np.int8),
            "c": [True, False, True, False, True],
        }, index=idx)
        parts = replay._npz_to_parts(replay._frame_to_npz(df))
        assert replay._compare_frames("f", parts, replay._frame_parts(df)) == []
        df2 = df.copy()
        df2.iloc[2, 0] = 1e-299
        diffs = replay._compare_frames("f", parts, replay._frame_parts(df2))
        assert len(diffs) == 1 and diffs[0].startswith("f.a: 1 cell(s) differ from 2026-01-01T01:00:00")
