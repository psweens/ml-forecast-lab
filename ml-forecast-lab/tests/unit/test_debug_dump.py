"""Regression tests for the debug-bundle dumper.

The dumper is a diagnostic surface — when a user reports a regression
that synthetic tests don't reproduce, we ask them to enable the toggle,
retrain, and ship the resulting bundle. If the bundle is empty, missing
files, or contains the wrong shapes, the diagnostic path itself is the
bug. These tests pin the bundle contract so future refactors of
_retrain_and_cache or _forecast_with_cached can't silently degrade it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.config import ExperimentCfg
from ml_forecast_lab.debug_dump import KEEP_LAST_N, DebugDumper


@pytest.fixture
def dumper(tmp_path: Path) -> DebugDumper:
    cfg = tmp_path / "mlfl.yaml"
    cfg.touch()
    d = DebugDumper.from_config_path(cfg)
    assert d is not None
    return d


@pytest.fixture
def fake_combined() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=50, freq="30min", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "target": rng.random(50),
            "hour_sin": np.sin(np.arange(50) * 0.1),
            "hour_cos": np.cos(np.arange(50) * 0.1),
        },
        index=idx,
    )


@pytest.fixture
def fake_exp() -> ExperimentCfg:
    return ExperimentCfg(
        name="optimised_solar",
        target_entity="predbat.pv_power",
        log_transform=True,
        target_is_nonnegative=True,
        debug_save_training_dumps=True,
    )


def test_training_dump_writes_all_expected_files(
    dumper: DebugDumper, fake_combined: pd.DataFrame, fake_exp: ExperimentCfg
):
    out = dumper.dump_training(
        exp_name=fake_exp.name,
        model_name="nlinear",
        exp_cfg=fake_exp,
        combined=fake_combined,
        feature_cols=["hour_sin", "hour_cos"],
        target_stats={"mean": 0.5, "min": 0.0, "max": 1.0, "zeros": 0, "n_samples": 50},
        seq_X=np.random.default_rng(1).random((30, 48, 3)).astype(np.float32),
        seq_y=np.random.default_rng(2).random((30, 96)).astype(np.float32),
        channel_names=["target", "hour_sin", "hour_cos"],
        seq_kwargs={
            "extended_window": True,
            "past_window_size": 48,
            "future_feature_cols": ["hour_sin", "hour_cos"],
        },
        model_params={"learning_rate": 0.0005, "output_activation": "softplus"},
        rows_before_dropna=80,
        rows_after_dropna=50,
    )
    assert out is not None
    files = {p.name for p in out.iterdir()}
    assert {"meta.json", "training.parquet", "sliding_window.npz"} <= files


def test_training_meta_captures_pf1_pf10_relevant_fields(
    dumper: DebugDumper, fake_combined: pd.DataFrame, fake_exp: ExperimentCfg
):
    out = dumper.dump_training(
        exp_name=fake_exp.name,
        model_name="nlinear",
        exp_cfg=fake_exp,
        combined=fake_combined,
        feature_cols=["hour_sin"],
        target_stats={"mean": 0.5},
        model_params={"output_activation": "softplus", "use_revin": True},
        seq_kwargs={"extended_window": True, "past_window_size": 48},
    )
    meta = json.loads((out / "meta.json").read_text())
    ec = meta["experiment_config"]
    assert ec["log_transform"] is True
    assert ec["target_is_nonnegative"] is True
    assert ec["target_entity"] == "predbat.pv_power"
    assert meta["model_params"]["output_activation"] == "softplus"
    assert meta["model_params"]["use_revin"] is True
    assert meta["seq_kwargs"]["extended_window"] is True
    assert meta["seq_kwargs"]["past_window_size"] == 48


def test_forecast_dump_appends_to_pending_training_dir(
    dumper: DebugDumper, fake_combined: pd.DataFrame, fake_exp: ExperimentCfg
):
    train_out = dumper.dump_training(
        exp_name=fake_exp.name, model_name="nlinear", exp_cfg=fake_exp,
        combined=fake_combined, feature_cols=["hour_sin"], target_stats={},
    )
    ds_future = pd.date_range("2026-01-03", periods=96, freq="30min", tz="UTC")
    fc_out = dumper.dump_forecast(
        exp_name=fake_exp.name,
        y_pred_raw=np.linspace(0, 1, 96, dtype=np.float32),
        y_pred_physical=np.linspace(0, 5, 96, dtype=np.float32),
        ds_future=ds_future,
        model_version="v1",
        log_transform_applied=True,
    )
    assert fc_out == train_out, "forecast dump must reuse training dir"
    assert (train_out / "forecast.parquet").exists()

    meta = json.loads((train_out / "meta.json").read_text())
    assert meta["forecast"]["n_points"] == 96
    assert meta["forecast"]["range_physical"] == [0.0, 5.0]
    assert meta["forecast"]["log_transform_applied"] is True


def test_forecast_without_pending_training_is_noop(dumper: DebugDumper):
    ds_future = pd.date_range("2026-01-03", periods=96, freq="30min", tz="UTC")
    out = dumper.dump_forecast(
        exp_name="never_trained",
        y_pred_raw=None,
        y_pred_physical=np.zeros(96, dtype=np.float32),
        ds_future=ds_future,
    )
    assert out is None


def test_rotation_keeps_last_n(
    dumper: DebugDumper, fake_combined: pd.DataFrame, fake_exp: ExperimentCfg
):
    from unittest.mock import patch
    import datetime as _dt

    real_dt = _dt.datetime
    for i in range(KEEP_LAST_N + 3):
        fake_time = real_dt(2026, 1, 1, 0, i, 0, tzinfo=_dt.timezone.utc)
        with patch("ml_forecast_lab.debug_dump.datetime") as mock_dt:
            mock_dt.now.return_value = fake_time
            mock_dt.side_effect = lambda *a, **k: real_dt(*a, **k)
            dumper.dump_training(
                exp_name=fake_exp.name, model_name="lgbm", exp_cfg=fake_exp,
                combined=fake_combined, feature_cols=["hour_sin"], target_stats={},
            )
        dumper._pending_dirs.pop(fake_exp.name, None)
    dirs = sorted((dumper.root / fake_exp.name).iterdir())
    assert len(dirs) == KEEP_LAST_N
    assert dirs[0].name == "20260101T000300Z"
    assert dirs[-1].name == "20260101T000700Z"


def test_dumper_handles_parquet_engine_missing(
    dumper: DebugDumper, fake_exp: ExperimentCfg
):
    idx = pd.date_range("2026-01-01", periods=10, freq="30min", tz="UTC")
    combined = pd.DataFrame({"target": np.zeros(10, dtype=object)}, index=idx)
    combined.loc[idx[0], "target"] = object()
    out = dumper.dump_training(
        exp_name=fake_exp.name, model_name="x", exp_cfg=fake_exp,
        combined=combined, feature_cols=[], target_stats={},
    )
    assert out is not None
    csv_path = out / "training.csv"
    pq_path = out / "training.parquet"
    assert csv_path.exists() or pq_path.exists()


def test_dump_when_no_config_path_returns_none():
    assert DebugDumper.from_config_path(None) is None
