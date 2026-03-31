"""Tests for benchmark runner."""

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.benchmark.runner import BenchmarkRunner
from ml_forecast_lab.benchmark.metrics import get_metric_registry


def _make_experiment_cfg(**overrides):
    """Create a minimal experiment config dict."""
    cfg = {
        "name": "test",
        "cv_strategy": "walk_forward",
        "cv_folds": 3,
        "production_metric": "mae",
        "metrics": ["mae", "rmse"],
        "interval_minutes": 30,
    }
    cfg.update(overrides)
    return cfg


def _make_feature_builder():
    """Feature builder that extracts all non-target columns."""
    def builder(df_sub, config, purpose="train"):
        cols = [c for c in df_sub.columns if c != "target"]
        X = df_sub[cols].values.astype(np.float32)
        return np.nan_to_num(X, nan=0.0)
    return builder


class TestCVSplits:
    def test_walk_forward_no_overlap(self):
        """Train and test indices must not overlap in any fold."""
        cfg = _make_experiment_cfg(cv_folds=5)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        idx = pd.date_range("2024-01-01", periods=300, freq="30min")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "feature_1": rng.random(300),
            "target": rng.random(300),
        }, index=idx)

        splits = runner._prepare_train_test_splits(df)
        assert len(splits) == 5

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Fold {fold_idx}: train/test overlap"

    def test_walk_forward_train_grows(self):
        """Walk-forward: each fold should have a larger training set."""
        cfg = _make_experiment_cfg(cv_folds=3)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        idx = pd.date_range("2024-01-01", periods=200, freq="30min")
        df = pd.DataFrame({
            "feature_1": np.random.default_rng(42).random(200),
            "target": np.random.default_rng(42).random(200),
        }, index=idx)

        splits = runner._prepare_train_test_splits(df)
        train_sizes = [len(train) for train, _ in splits]
        assert train_sizes == sorted(train_sizes), "Training set should grow"

    def test_test_always_after_train(self):
        """Test indices should always be after training indices (no look-ahead)."""
        cfg = _make_experiment_cfg(cv_folds=3)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        idx = pd.date_range("2024-01-01", periods=200, freq="30min")
        df = pd.DataFrame({
            "feature_1": np.random.default_rng(42).random(200),
            "target": np.random.default_rng(42).random(200),
        }, index=idx)

        splits = runner._prepare_train_test_splits(df)
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            assert train_idx.max() < test_idx.min(), \
                f"Fold {fold_idx}: test starts before train ends"


class TestMeanRankScoring:
    def test_mean_rank_computed(self):
        """Run a minimal benchmark and verify mean_rank is in metrics."""
        from ml_forecast_lab.models.lightgbm_backend import LightGBMModel

        cfg = _make_experiment_cfg(cv_folds=2)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=200, freq="30min")
        df = pd.DataFrame({
            "feature_1": rng.random(200),
            "feature_2": rng.random(200),
            "target": rng.random(200),
        }, index=idx)

        models = {"lightgbm": LightGBMModel()}
        result = runner.run_benchmark(df, models)
        assert "mean_rank" in result.model_results["lightgbm"].metrics
