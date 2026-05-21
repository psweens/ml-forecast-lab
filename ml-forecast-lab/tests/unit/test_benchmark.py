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

    def test_sliding_window_indices_in_range(self):
        """Regression: sliding_window previously produced test indices past
        the end of the dataframe for every fold past fold 0, which then
        crashed ``df.iloc[test_idx]`` with IndexError. All folds must now
        keep their indices in ``[0, n_samples)``."""
        cfg = _make_experiment_cfg(cv_strategy="sliding_window", cv_folds=5)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        idx = pd.date_range("2024-01-01", periods=300, freq="30min")
        df = pd.DataFrame({
            "feature_1": np.arange(300, dtype=float),
            "target": np.arange(300, dtype=float),
        }, index=idx)

        splits = runner._prepare_train_test_splits(df)
        assert len(splits) == 5

        n = len(df)
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            assert train_idx.min() >= 0 and train_idx.max() < n, \
                f"Fold {fold_idx}: train indices out of range"
            assert test_idx.min() >= 0 and test_idx.max() < n, \
                f"Fold {fold_idx}: test indices out of range"
            assert len(set(train_idx) & set(test_idx)) == 0, \
                f"Fold {fold_idx}: train/test overlap"
            assert train_idx.max() < test_idx.min(), \
                f"Fold {fold_idx}: test starts before train ends"

    def test_sliding_window_slides_forward(self):
        """Sliding window's train start must advance fold-by-fold so each
        fold uses a different slice of history (otherwise it degenerates
        into evaluating the same fit n times)."""
        cfg = _make_experiment_cfg(cv_strategy="sliding_window", cv_folds=4)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        idx = pd.date_range("2024-01-01", periods=400, freq="30min")
        df = pd.DataFrame({
            "target": np.arange(400, dtype=float),
        }, index=idx)

        splits = runner._prepare_train_test_splits(df)
        train_starts = [train.min() for train, _ in splits]
        assert train_starts == sorted(train_starts)
        assert train_starts[0] < train_starts[-1], \
            "train_start must advance across folds"

    def test_sliding_window_last_fold_reaches_end(self):
        """The final sliding fold should evaluate the most-recent rows, so
        leaderboard rankings reflect current-time performance."""
        cfg = _make_experiment_cfg(cv_strategy="sliding_window", cv_folds=5)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        idx = pd.date_range("2024-01-01", periods=300, freq="30min")
        df = pd.DataFrame({"target": np.arange(300, dtype=float)}, index=idx)

        splits = runner._prepare_train_test_splits(df)
        last_train, last_test = splits[-1]
        assert last_test.max() == len(df) - 1

    def test_walk_forward_honours_embargo(self):
        """``cv_embargo_periods`` must produce a gap of that size between
        train_end and test_start. Previously the runner ignored the field
        entirely, so a documented config knob silently leaked across the
        train/test boundary."""
        cfg = _make_experiment_cfg(cv_folds=5, cv_embargo_periods=12)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        idx = pd.date_range("2024-01-01", periods=300, freq="30min")
        df = pd.DataFrame({"target": np.arange(300, dtype=float)}, index=idx)

        splits = runner._prepare_train_test_splits(df)
        assert len(splits) >= 1
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            gap = int(test_idx.min()) - int(train_idx.max()) - 1
            assert gap >= 12, (
                f"Fold {fold_idx}: gap={gap}, expected ≥12 from "
                f"cv_embargo_periods=12"
            )

    def test_sliding_window_honours_embargo(self):
        """Sliding-window CV must also apply ``cv_embargo_periods``."""
        cfg = _make_experiment_cfg(
            cv_strategy="sliding_window", cv_folds=4, cv_embargo_periods=8,
        )
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        idx = pd.date_range("2024-01-01", periods=300, freq="30min")
        df = pd.DataFrame({"target": np.arange(300, dtype=float)}, index=idx)

        splits = runner._prepare_train_test_splits(df)
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            gap = int(test_idx.min()) - int(train_idx.max()) - 1
            assert gap >= 8, (
                f"Fold {fold_idx}: gap={gap}, expected ≥8 from "
                f"cv_embargo_periods=8"
            )

    def test_walk_forward_runs_end_to_end(self):
        """Regression: the runner must not raise IndexError when running a
        full benchmark with a non-trivial CV configuration."""
        cfg = _make_experiment_cfg(cv_folds=4)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        idx = pd.date_range("2024-01-01", periods=240, freq="30min")
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "feature_1": rng.random(240),
            "target": rng.random(240),
        }, index=idx)

        from ml_forecast_lab.models.lightgbm_backend import LightGBMModel
        result = runner.run_benchmark(df, {"lightgbm": LightGBMModel()})
        # Every fold must have actually produced metrics (no silent skips).
        fold_metrics = result.model_results["lightgbm"].fold_metrics
        assert all(fm for fm in fold_metrics), \
            f"Some folds produced empty metrics: {fold_metrics}"

    def test_sliding_window_runs_end_to_end(self):
        """Regression for the sliding_window IndexError: a full benchmark
        with ``cv_strategy='sliding_window'`` must run without raising and
        must produce a metric for every fold."""
        cfg = _make_experiment_cfg(cv_strategy="sliding_window", cv_folds=3)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        idx = pd.date_range("2024-01-01", periods=240, freq="30min")
        rng = np.random.default_rng(11)
        df = pd.DataFrame({
            "feature_1": rng.random(240),
            "target": rng.random(240),
        }, index=idx)

        from ml_forecast_lab.models.lightgbm_backend import LightGBMModel
        result = runner.run_benchmark(df, {"lightgbm": LightGBMModel()})
        fold_metrics = result.model_results["lightgbm"].fold_metrics
        assert all(fm for fm in fold_metrics), \
            f"Some folds produced empty metrics: {fold_metrics}"


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

    def test_mean_rank_bootstrap_ci_populated(self):
        """``_compute_composite_ranks`` must populate ``mean_rank_low`` and
        ``mean_rank_high`` on every rankable model when n_folds >= 2. With
        N=1 the CI is meaningless and must be omitted."""
        from ml_forecast_lab.benchmark.runner import ModelResult

        cfg = _make_experiment_cfg(cv_folds=3)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        # Two synthetic models with hand-crafted per-fold metrics so the
        # rank order is deterministic and the bootstrap operates on a
        # known input. Model A wins on folds 0,1; model B wins on fold 2.
        a = ModelResult(model_name="A")
        b = ModelResult(model_name="B")
        a.fold_metrics = [{"mae": 1.0, "rmse": 1.0}, {"mae": 1.0, "rmse": 1.0}, {"mae": 5.0, "rmse": 5.0}]
        b.fold_metrics = [{"mae": 2.0, "rmse": 2.0}, {"mae": 2.0, "rmse": 2.0}, {"mae": 1.0, "rmse": 1.0}]

        means, ranks, cis, dnc = runner._compute_composite_ranks(
            {"A": a, "B": b}, metric_source="fold_metrics",
            bootstrap_iters=200, bootstrap_seed=0,
        )
        assert dnc == [], "Neither model failed; DNC must be empty"
        assert ranks == {"A": 1, "B": 2}, (
            f"A wins 2 of 3 folds, expected #1; got {ranks}"
        )
        assert "A" in cis and "B" in cis, "Bootstrap CIs must be populated"
        a_low, a_high = cis["A"]
        # Mean rank of A across (1,1,2) is 4/3 ≈ 1.33; CI should span at
        # least the rank-1 floor and not exceed rank-2.
        assert 1.0 <= a_low <= a_high <= 2.0, (
            f"A's CI {cis['A']} should sit in [1, 2]"
        )
        # CI should be wider than zero for both models (varying fold ranks)
        assert a_high > a_low, "Bootstrap CI must have non-zero width"

    def test_did_not_complete_excluded_from_rank(self):
        """A model that errored on at least one fold (empty {} entry)
        must NOT appear in the ranked pool — otherwise it takes a
        last-place phantom slot off every survivor, inflating the
        leaderboard's apparent dominance.
        """
        from ml_forecast_lab.benchmark.runner import ModelResult

        cfg = _make_experiment_cfg(cv_folds=3)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        complete = ModelResult(model_name="complete")
        partial = ModelResult(model_name="partial")
        complete.fold_metrics = [{"mae": 1.0, "rmse": 1.0}] * 3
        # Partial completed only 2 of 3 folds (one empty {} → DNC)
        partial.fold_metrics = [{"mae": 2.0, "rmse": 2.0}, {}, {"mae": 2.0, "rmse": 2.0}]

        means, ranks, cis, dnc = runner._compute_composite_ranks(
            {"complete": complete, "partial": partial},
            metric_source="fold_metrics",
            bootstrap_iters=50,
        )
        assert "partial" in dnc, "Partial model must be in DNC"
        assert "partial" not in ranks, (
            "DNC model must be excluded from the rankable pool — otherwise "
            "the surviving model gets a free last-place 'win' against it"
        )
        assert ranks == {"complete": 1}

    def test_all_models_fail_returns_empty_rank(self):
        """Edge case: every model errored on at least one fold. Nothing
        should be ranked, and all should appear in DNC."""
        from ml_forecast_lab.benchmark.runner import ModelResult

        cfg = _make_experiment_cfg(cv_folds=3)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        a = ModelResult(model_name="a")
        b = ModelResult(model_name="b")
        a.fold_metrics = [{}, {"mae": 1.0, "rmse": 1.0}, {}]
        b.fold_metrics = []  # Outer-level failure: zero folds

        means, ranks, cis, dnc = runner._compute_composite_ranks(
            {"a": a, "b": b}, metric_source="fold_metrics",
            bootstrap_iters=50,
        )
        assert ranks == {}
        assert means == {}
        assert sorted(dnc) == ["a", "b"]
