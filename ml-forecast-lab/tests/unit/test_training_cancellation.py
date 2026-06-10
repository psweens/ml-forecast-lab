"""Cooperative stop-training (audit F10): a set cancel_event must stop
run_single_model at the next epoch/fold boundary instead of letting the
executor thread train to completion."""

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.benchmark.runner import BenchmarkRunner
from ml_forecast_lab.models.base import TrainingCancelled
import threading


class _SlowFakeModel:
    """Minimal neural-shaped model whose fit loops 'epochs' via the
    epoch callback, mimicking the backends' _emit_epoch contract."""

    name = "slowfake"
    is_neural = False

    def __init__(self):
        self.epochs_run = 0
        self.fits = 0

    def fit(self, X, y, epoch_callback=None, **kw):
        self.fits += 1
        for epoch in range(50):
            self.epochs_run += 1
            if epoch_callback is not None:
                # mirror ForecastModel._emit_epoch semantics
                try:
                    epoch_callback(epoch=epoch)
                except TrainingCancelled:
                    raise
                except Exception:
                    pass

    def predict(self, X):
        return np.zeros(len(X))


def _runner_and_df():
    n = 300
    idx = pd.date_range("2026-01-01", periods=n, freq="30min")
    df = pd.DataFrame({"target": np.random.default_rng(0).normal(size=n)},
                      index=idx)
    cfg = {"name": "t", "cv_strategy": "walk_forward", "cv_folds": 3,
           "production_metric": "mae", "metrics": ["mae"],
           "interval_minutes": 30, "recency_half_life_days": 0.0}
    runner = BenchmarkRunner(cfg, lambda d, c, purpose="train":
                             np.zeros((len(d), 1), dtype=np.float32))
    folds = runner._prepare_train_test_splits(df)
    return runner, df, folds


def test_pre_set_cancel_event_stops_before_first_fold():
    runner, df, folds = _runner_and_df()
    model = _SlowFakeModel()
    ev = threading.Event()
    ev.set()
    with pytest.raises(TrainingCancelled):
        runner.run_single_model(df, model, folds, cancel_event=ev)
    assert model.fits == 0


def test_cancel_mid_training_stops_at_epoch_boundary():
    runner, df, folds = _runner_and_df()
    model = _SlowFakeModel()
    ev = threading.Event()

    fired = {"n": 0}

    def cb(**data):
        fired["n"] += 1
        if fired["n"] >= 3:
            ev.set()

    with pytest.raises(TrainingCancelled):
        runner.run_single_model(df, model, folds,
                                epoch_callback=cb, cancel_event=ev)
    # stopped well short of 3 folds x 50 epochs
    assert model.epochs_run < 10


def test_no_cancel_event_runs_to_completion():
    runner, df, folds = _runner_and_df()
    model = _SlowFakeModel()
    result = runner.run_single_model(df, model, folds)
    assert model.fits == len(folds)
