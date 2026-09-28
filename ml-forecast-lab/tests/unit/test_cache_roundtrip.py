"""Production model cache round trip (v2.52.2).

``_persist_cached_model`` writes the retrained champion to disk and
``_restore_cached_models`` / ``_rollback_cached_model`` read it back after a
restart. Three defects broke that round trip; all are fixed and pinned here.

A. **XGBoost metadata went to a sidecar the persist never renamed.**
   ``save`` wrote ``path + ".metadata.json"`` beside the model, but the
   persist saves to ``model.bin.tmp`` and renames only that file, so the
   sidecar stayed at ``model.bin.tmp.metadata.json``. ``load`` found no
   metadata, left ``feature_names_=None`` and still reported success, so the
   retrain was deferred and every forecast tick raised ``TypeError`` until
   the next scheduled retrain. ``previous/`` never held the sidecar either.
   The metadata now travels inside the model file.

A0. **All-zero XGBoost importances made save raise.** When the trees grow
   no splits every importance is zero, the normalisation is skipped and
   the values stay np.float32, which json.dump rejects. The persist then
   aborted before its rename and disk kept the previous generation —
   possibly a different backend, which restore served without comment.
   Importances are now plain floats, and restore warns when it loads a
   model other than ``production_model``.

B. **torch >= 2.6 refused every checkpoint carrying numpy stats.**
   ``torch.load`` defaults to ``weights_only=True`` there, which rejects
   ``numpy.core.multiarray._reconstruct``. N-BEATS / NHiTS always store
   numpy channel statistics, as does every other backend with
   ``use_revin=False``, so their restore always failed and every restart
   forced a retrain. Loads now go through ``load_torch_checkpoint``, which
   stays ``weights_only`` with exactly those numpy globals allowlisted.

Every pipeline test drives the real retrain -> persist -> restore -> forecast
path and asserts the post-restore forecast is bit-identical to the one the
live model published.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import math
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.config import AppConfig, ExperimentCfg
from ml_forecast_lab.covariates import CovariateResolver
from ml_forecast_lab.db import HistoryDB
from ml_forecast_lab.main import MLForecastLabApp
from ml_forecast_lab.models.registry import ModelRegistry

INTERVAL = 30

_BACKENDS = {
    "lightgbm": ("lightgbm_backend", "LightGBMModel"),
    "xgboost": ("xgboost_backend", "XGBoostModel"),
    "nbeats": ("nbeats_backend", "NBeatsModel"),
    "nhits": ("nhits_backend", "NHiTSModel"),
    "lstm": ("lstm_backend", "LSTMModel"),
    "nlinear": ("nlinear_backend", "NLinearModel"),
}

# Small enough to train in about a second. The fitted values are irrelevant;
# only whether restore reproduces what the live model published. XGBoost
# trains with mse: on this target its default huber objective grows no
# splits at all, and two constant forecasts agreeing proves nothing about
# the round trip.
_FAST = {
    "lightgbm": {"n_estimators": 30},
    "xgboost": {"n_estimators": 30, "loss_fn": "mse"},
    "nbeats": {"epochs": 2, "hidden_size": 16, "n_stacks": 1, "blocks_per_stack": 1},
    "nhits": {"epochs": 2, "hidden_size": 16, "n_stacks": 1, "blocks_per_stack": 1},
    "lstm": {"epochs": 2, "hidden_size": 8},
    "nlinear": {"epochs": 2},
}


# ---------------------------------------------------------------------
# Harness (from test_missingness_masking.py)
# ---------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


class _StubHA:
    def __init__(self, rows_by_entity):
        self.rows_by_entity = rows_by_entity

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


def _daily(i: int) -> float:
    return 20.0 + 10.0 * math.sin(2 * math.pi * i / 96) + (i % 7) * 0.3


def _recorder_rows(entity_now, days, cadence_min=15, value=_daily):
    ts = pd.date_range(
        entity_now - timedelta(days=days), entity_now,
        freq=f"{cadence_min}min", tz="UTC",
    )
    return [
        {"last_changed": t.isoformat(), "state": f"{value(i):.4f}"}
        for i, t in enumerate(ts)
    ]


def _registry() -> ModelRegistry:
    reg = ModelRegistry()
    for name, (module, cls_name) in _BACKENDS.items():
        mod = importlib.import_module(f"ml_forecast_lab.models.{module}")
        reg.register(name, getattr(mod, cls_name))
    return reg


def _exp(backend: str) -> ExperimentCfg:
    return ExperimentCfg(
        name="roundtrip", target_entity="sensor.load", mode="production",
        interval_minutes=INTERVAL, future_periods=24,
        # History spans less than the fetch window, so the retrain and the
        # post-restore forecast see identical rows whatever the wall clock
        # does between them.
        days_history=10, models_enabled=[backend], production_model=backend,
    )


def _make_app(tmp_db, exp_cfg, overrides=None, value=_daily):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    app = MLForecastLabApp()
    app.history_db = HistoryDB(tmp_db)
    app.ha_interface = _StubHA({
        "sensor.load": _recorder_rows(now, days=8, value=value),
    })
    app.config = AppConfig(
        experiments=[exp_cfg], model_overrides=overrides or dict(_FAST),
    )
    app.covariate_resolver = CovariateResolver(
        app.ha_interface,
        history_db=app.history_db,
        retention_provider=app._retention_days_for_table,
    )
    app.model_registry = _registry()

    published = []

    async def _capture(**kw):
        published.append((kw["model_name"], np.array(kw["y_pred"]), kw["ds_future"]))

    app._publish_forecast_sensors = _capture
    return app, published


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "models"
    monkeypatch.setattr(
        MLForecastLabApp, "_cached_model_dir",
        staticmethod(lambda exp_name: root / exp_name),
    )
    return root


def _restart(app, exp_name):
    """What an add-on restart leaves: nothing in memory, the disk intact."""
    app._cached_models.clear()
    app._restore_cached_models()
    assert exp_name in app._cached_models, "restore must succeed"
    _run(app._forecast_with_cached(exp_name))


def _assert_identical(before, after):
    name_b, y_b, ds_b = before
    name_a, y_a, ds_a = after
    assert name_a == name_b
    assert ds_a.equals(ds_b)
    np.testing.assert_array_equal(y_a, y_b)


# ---------------------------------------------------------------------
# The round trip, per backend
# ---------------------------------------------------------------------

_CASES = [
    pytest.param("lightgbm", {}, id="lightgbm"),
    pytest.param("xgboost", {}, id="xgboost-A"),
    pytest.param("nbeats", {}, id="nbeats-B"),
    pytest.param("nhits", {}, id="nhits-B"),
    pytest.param(
        "lstm", {"use_revin": False, "output_activation": "zscore"},
        id="lstm-no-revin-zscore-B",
    ),
    pytest.param("nlinear", {}, id="nlinear"),
]


class TestRestoreReproducesTheLiveForecast:
    @pytest.mark.parametrize("backend, extra", _CASES)
    def test_forecast_after_restart_is_bit_identical(
        self, tmp_db, cache_root, backend, extra,
    ):
        exp = _exp(backend)
        overrides = dict(_FAST)
        overrides[backend] = {**_FAST[backend], **extra}
        app, published = _make_app(tmp_db, exp, overrides)

        _run(app._retrain_and_cache(exp))
        assert len(published) == 1, "retrain publishes its forecast"
        assert np.ptp(published[0][1]) > 0, (
            "a constant forecast would make the comparison below vacuous"
        )

        _restart(app, exp.name)
        assert len(published) == 2
        _assert_identical(published[0], published[1])

    @pytest.mark.parametrize("backend, extra", _CASES)
    def test_save_writes_exactly_the_file_it_was_given(
        self, tmp_db, cache_root, backend, extra,
    ):
        """The persist renames, archives and rolls back ``model.bin`` by
        name. Anything a backend writes beside it is lost on the first of
        those — which is how defect A happened."""
        exp = _exp(backend)
        overrides = dict(_FAST)
        overrides[backend] = {**_FAST[backend], **extra}
        app, _ = _make_app(tmp_db, exp, overrides)

        _run(app._retrain_and_cache(exp))

        model_dir = cache_root / exp.name
        assert sorted(p.name for p in model_dir.iterdir()) == [
            "cache_meta.json", "model.bin",
        ]


# ---------------------------------------------------------------------
# XGBoost (A, A0)
# ---------------------------------------------------------------------

class TestXGBoost:
    def test_restored_champion_has_its_feature_names(self, tmp_db, cache_root):
        exp = _exp("xgboost")
        app, _ = _make_app(tmp_db, exp)
        _run(app._retrain_and_cache(exp))
        live_names = list(app._cached_models[exp.name]["model"].feature_names_)

        app._cached_models.clear()
        app._restore_cached_models()

        restored = app._cached_models[exp.name]["model"]
        assert restored.feature_names_ == live_names
        assert restored.training_metadata["num_features"] == len(live_names)

    def test_rollback_generation_is_complete(self, tmp_db, cache_root):
        """``previous/`` is copied file by file. An XGBoost champion archived
        there must come back whole, not as a booster without metadata."""
        exp = _exp("xgboost")
        app, published = _make_app(tmp_db, exp)
        _run(app._retrain_and_cache(exp))           # generation 1: xgboost
        exp.production_model = "lightgbm"
        _run(app._retrain_and_cache(exp))           # generation 2: lightgbm
        assert published[1][0] == "lightgbm"

        ok, msg = app._rollback_cached_model(exp.name)
        assert ok, msg
        _run(app._forecast_with_cached(exp.name))

        _assert_identical(published[0], published[-1])

    def test_all_zero_importances_still_persist(self, tmp_db, cache_root):
        """Trees that grow no splits (here: a constant target) score every
        importance zero, and the normalisation is skipped. Those values
        stayed np.float32 and json.dump raised mid-save (A0)."""
        exp = _exp("xgboost")
        app, published = _make_app(tmp_db, exp, value=lambda i: 5.0)

        _run(app._retrain_and_cache(exp))
        live = app._cached_models[exp.name]["model"]
        importances = live.training_metadata["feature_importances"]
        assert importances and all(v == 0.0 for v in importances.values())
        assert all(type(v) is float for v in importances.values())

        meta = json.loads((cache_root / exp.name / "cache_meta.json").read_text())
        assert meta["model_name"] == "xgboost"
        _restart(app, exp.name)
        _assert_identical(published[0], published[1])

    def test_cache_written_before_the_fix_still_restores(self, tmp_db, cache_root):
        """A cache persisted by v2.52.1 or earlier holds a bare booster:
        the sidecar was stranded at ``model.bin.tmp.metadata.json``. The
        booster is intact, so restore must serve it, not a model whose
        predict raises."""
        exp = _exp("xgboost")
        app, published = _make_app(tmp_db, exp)
        _run(app._retrain_and_cache(exp))

        model_dir = cache_root / exp.name
        live = app._cached_models[exp.name]["model"]
        legacy_tmp = model_dir / "model.bin.tmp"
        live.model.save_model(str(legacy_tmp))     # the pre-fix main file
        legacy_tmp.replace(model_dir / "model.bin")

        _restart(app, exp.name)
        _assert_identical(published[0], published[1])


# ---------------------------------------------------------------------
# Restore names a champion it did not expect
# ---------------------------------------------------------------------

class TestChampionMismatch:
    def test_restoring_another_backend_is_warned_about(
        self, tmp_db, cache_root, caplog,
    ):
        exp = _exp("lightgbm")
        app, _ = _make_app(tmp_db, exp)
        _run(app._retrain_and_cache(exp))
        exp.production_model = "xgboost"

        app._cached_models.clear()
        with caplog.at_level(logging.WARNING, logger="ml_forecast_lab.main"):
            app._restore_cached_models()

        hits = [r for r in caplog.records if "production_model is xgboost" in r.message]
        assert len(hits) == 1
        assert "is lightgbm" in hits[0].message
        # Still served: a rollback to another backend produces exactly this
        # state, and rejecting it would undo the rollback on every restart.
        assert app._cached_models[exp.name]["model_name"] == "lightgbm"

    def test_matching_champion_restores_quietly(self, tmp_db, cache_root, caplog):
        exp = _exp("lightgbm")
        app, _ = _make_app(tmp_db, exp)
        _run(app._retrain_and_cache(exp))

        app._cached_models.clear()
        with caplog.at_level(logging.WARNING, logger="ml_forecast_lab.main"):
            app._restore_cached_models()

        assert exp.name in app._cached_models
        assert "production_model is" not in caplog.text


# ---------------------------------------------------------------------
# torch checkpoint loading (B)
# ---------------------------------------------------------------------

def _torch_checkpoint_backends() -> list:
    """Every wired backend whose save() writes a torch checkpoint."""
    import ml_forecast_lab.models as models
    out = []
    for cls_name in models.WIRED_BACKENDS:
        cls = getattr(models, cls_name, None)
        if cls is None:
            continue
        if "torch.save(" in Path(inspect.getsourcefile(cls)).read_text():
            out.append(cls)
    return out


class TestTorchCheckpointLoading:
    def _stats_checkpoint(self, path):
        import torch
        torch.save({
            "state_dict": {"w": torch.ones(3)},
            "channel_mean": np.arange(4, dtype=np.float32),
            "channel_std": np.ones(4, dtype=np.float64),
            "y_mean": np.float32(1.5),
            "flag": np.bool_(True),
        }, path)

    def test_numpy_stats_load_under_weights_only(self, tmp_path):
        from ml_forecast_lab.models.base import load_torch_checkpoint
        path = str(tmp_path / "ckpt.bin")
        self._stats_checkpoint(path)

        data = load_torch_checkpoint(path)

        np.testing.assert_array_equal(data["channel_mean"], np.arange(4, dtype=np.float32))
        assert data["channel_mean"].dtype == np.float32
        assert data["channel_std"].dtype == np.float64
        assert data["y_mean"] == np.float32(1.5)

    def test_a_plain_weights_only_load_refuses_them(self, tmp_path):
        """Guards the guard: without the allowlist the same checkpoint is
        refused, so the test above does exercise the allowlist. A torch
        that accepts numpy by default leaves nothing to guard."""
        import torch
        path = str(tmp_path / "ckpt.bin")
        self._stats_checkpoint(path)
        try:
            torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            return
        pytest.skip("this torch accepts numpy under weights_only by default")

    def test_loading_stays_weights_only(self, tmp_path):
        """The allowlist is the numpy stats and nothing else — an arbitrary
        global in a checkpoint is still refused."""
        import torch
        from ml_forecast_lab.models.base import load_torch_checkpoint
        if getattr(torch.serialization, "safe_globals", None) is None:
            pytest.skip("torch < 2.5 keeps its own weights_only=False default")
        path = str(tmp_path / "ckpt.bin")
        torch.save({"payload": datetime(2026, 1, 1)}, path)
        with pytest.raises(pickle.UnpicklingError):
            load_torch_checkpoint(path)

    @pytest.mark.parametrize("cls", _torch_checkpoint_backends(), ids=lambda c: c.__name__)
    def test_every_neural_checkpoint_round_trips_its_numpy_stats(self, cls, tmp_path):
        """The pipeline cases above cover one backend per checkpoint shape;
        this covers them all. use_revin=False puts numpy channel stats into
        every checkpoint, and a z-score head adds the target stats."""
        rng = np.random.default_rng(0)
        seq = rng.random((64, 24, 3)).astype(np.float32)
        y = (rng.random((64, 8)) * 20).astype(np.float32)
        X_flat = rng.random((64, 10)).astype(np.float32)

        model = cls()
        params = model.get_params()
        wanted = {
            "epochs": 1, "patience": 1, "output_activation": "zscore",
            "use_revin": False, "cycle_len": 12,
        }
        model.set_params(**{k: v for k, v in wanted.items() if k in params})
        model.fit(X_flat, y, sequence_data=seq, past_window_size=16)
        before = model.predict_sequence(seq[:4])

        path = str(tmp_path / "model.bin")
        model.save(path)
        import torch
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert any(isinstance(v, np.ndarray) for v in raw.values()), (
            "no numpy in the checkpoint — this case no longer exercises B"
        )

        restored = cls()
        restored.load(path)
        np.testing.assert_array_equal(restored.predict_sequence(seq[:4]), before)

    def test_no_backend_calls_torch_load_directly(self):
        """Source contract: every neural backend loads through the helper,
        so a future backend cannot reintroduce B by copying an old load()."""
        models_dir = Path(__file__).resolve().parents[2] / "ml_forecast_lab" / "models"
        offenders = [
            p.name for p in sorted(models_dir.glob("*.py"))
            if p.name != "base.py" and "torch.load(" in p.read_text()
        ]
        assert offenders == []
