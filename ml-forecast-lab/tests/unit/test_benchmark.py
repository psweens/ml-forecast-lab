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

    def test_baseline_included_in_rank_pool(self):
        """A completed baseline (e.g. force-run seasonal_naive) must be ranked
        alongside the others and take a contiguous integer rank — it is the
        reference the leaderboard measures against, and hiding it would leave
        a gap in the rank sequence. Regression guard for the results tables
        dropping seasonal_naive."""
        from ml_forecast_lab.benchmark.runner import ModelResult

        cfg = _make_experiment_cfg(cv_folds=3)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        good = ModelResult(model_name="good")
        naive = ModelResult(model_name="seasonal_naive")
        good.fold_metrics = [{"mae": 1.0, "rmse": 1.0}] * 3
        naive.fold_metrics = [{"mae": 3.0, "rmse": 3.0}] * 3

        means, ranks, cis, dnc = runner._compute_composite_ranks(
            {"good": good, "seasonal_naive": naive},
            metric_source="fold_metrics", bootstrap_iters=50,
        )
        assert dnc == []
        # Baseline is ranked (not dropped) and ranks are contiguous 1..N.
        assert ranks == {"good": 1, "seasonal_naive": 2}
        assert sorted(ranks.values()) == [1, 2]

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

    def test_skipped_metric_sentinel_does_not_mark_model_as_dnc(self):
        """v2.39.3 bug 7: the daily-rank path emits ``{'__skipped__': True}``
        for folds whose test/train spans <2 distinct dates (legitimate, not
        a failure). The completeness check must distinguish that sentinel
        from empty {} (real fold-level error) — pre-v2.39.3 it conflated
        both as DNC, silently dropping models that fitted and predicted
        successfully from the daily leaderboard."""
        from ml_forecast_lab.benchmark.runner import ModelResult

        cfg = _make_experiment_cfg(cv_folds=3)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        a = ModelResult(model_name="a")
        b = ModelResult(model_name="b")
        # Both models computed daily metrics on folds 0 and 2; fold 1 was
        # too short for daily totals (sentinel emitted). Neither errored.
        a.daily_fold_metrics = [
            {"mae": 1.0, "rmse": 1.0},
            {"__skipped__": True},
            {"mae": 1.0, "rmse": 1.0},
        ]
        b.daily_fold_metrics = [
            {"mae": 2.0, "rmse": 2.0},
            {"__skipped__": True},
            {"mae": 2.0, "rmse": 2.0},
        ]

        means, ranks, cis, dnc = runner._compute_composite_ranks(
            {"a": a, "b": b}, metric_source="daily_fold_metrics",
            bootstrap_iters=50,
        )
        assert dnc == [], (
            "Sentinel-skipped folds must NOT mark a model as DNC — only "
            f"empty {{}} does. Got dnc={dnc}"
        )
        assert ranks == {"a": 1, "b": 2}, (
            f"Both models ranked normally on the folds where the metric was "
            f"computable; got ranks={ranks}"
        )

    def test_all_folds_skipped_demotes_to_dnc_not_fake_integer_ranks(self):
        """v2.39.3 follow-up: when every fold's daily metric is the
        ``__skipped__`` sentinel (e.g. test span <2 distinct dates on
        every walk-forward fold), the model is complete (no real
        failures) but has NO ranked folds. The pre-fix would emit
        integer ranks {a: 1, b: 2} in dict-insertion order — meaningless.
        These models must surface as DNC for this metric_source."""
        from ml_forecast_lab.benchmark.runner import ModelResult

        cfg = _make_experiment_cfg(cv_folds=3)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        a = ModelResult(model_name="a")
        b = ModelResult(model_name="b")
        a.daily_fold_metrics = [{"__skipped__": True}] * 3
        b.daily_fold_metrics = [{"__skipped__": True}] * 3

        means, ranks, cis, dnc = runner._compute_composite_ranks(
            {"a": a, "b": b}, metric_source="daily_fold_metrics",
        )
        assert means == {}
        assert ranks == {}
        assert sorted(dnc) == ["a", "b"], (
            "All-sentinel models must surface as DNC, not get fake "
            f"integer ranks; got dnc={dnc}, ranks={ranks}"
        )

    def test_zero_folds_returns_empty_not_insertion_order_ranks(self):
        """Edge case from Angle B/C: when every model has zero
        fold-metric entries (e.g. an exotic test fixture or upstream
        failure), pre-fix the completeness check passed vacuously
        (``len([]) == 0``), every model's mean_rank ended up inf, and
        integer ranks were assigned in dict-insertion order (1, 2, 3,
        ...). Output is now empty so callers don't downstream-trust
        meaningless ranks."""
        from ml_forecast_lab.benchmark.runner import ModelResult

        cfg = _make_experiment_cfg(cv_folds=2)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        a = ModelResult(model_name="a")
        b = ModelResult(model_name="b")
        # Both have zero entries — n_folds resolves to 0.
        a.fold_metrics = []
        b.fold_metrics = []

        means, ranks, cis, dnc = runner._compute_composite_ranks(
            {"a": a, "b": b}, metric_source="fold_metrics",
        )
        assert means == {}
        assert ranks == {}
        assert sorted(dnc) == ["a", "b"]

    def test_daily_dnc_separated_from_interval_dnc_in_result(self):
        """v2.39.3 visualisation fix: a model ranked in the per-interval
        leaderboard but excluded from the daily ranking (e.g. all daily
        folds skipped) must land in result.did_not_complete_DAILY, NOT
        result.did_not_complete — otherwise the UI shows a per-interval-
        ranked model under the main 'Did not complete' section, which is
        contradictory."""
        cfg = _make_experiment_cfg(cv_folds=4)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        idx = pd.date_range("2024-01-01", periods=240, freq="30min")
        rng = np.random.default_rng(11)
        df = pd.DataFrame({
            "feature_1": rng.random(240),
            "target": rng.random(240),
        }, index=idx)

        from ml_forecast_lab.models.lightgbm_backend import LightGBMModel
        result = runner.run_benchmark(df, {"lightgbm": LightGBMModel()})

        # Whatever the daily-rank outcome, the two DNC lists must be
        # disjoint and a model present in did_not_complete_daily must NOT
        # also be in did_not_complete (that's the contradiction we fixed).
        assert hasattr(result, "did_not_complete_daily")
        overlap = set(result.did_not_complete) & set(result.did_not_complete_daily)
        assert not overlap, (
            f"daily-DNC and interval-DNC lists must be disjoint; overlap={overlap}"
        )

    def test_bootstrap_ci_is_paired_across_models(self):
        """v2.39.3 bug 5: bootstrap iterations must apply the SAME resampled
        fold IDs to every model, not draw independent IDs per model. With
        paired resampling, when one model's mean-rank is computed for
        iteration k, the other model's mean-rank for iteration k MUST be
        computable from the same fold-id columns (i.e. they share the bootstrap
        index matrix). Independent draws inflate marginal CIs and the UI's
        'T#1 (tied within fold noise)' chip over-reports ties.

        Operationally: if every fold's rank-1 model is the SAME model, the
        paired bootstrap of (rank_A - rank_B) is always negative, so the
        leader's CI on (rank_A - rank_B) sits strictly below 0 — non-tie.
        Independent draws would produce a near-zero spread around 0.
        """
        from ml_forecast_lab.benchmark.runner import ModelResult
        import numpy as np

        cfg = _make_experiment_cfg(cv_folds=5)
        runner = BenchmarkRunner(cfg, _make_feature_builder())

        # Model A wins on every fold by a wide margin; B is always second.
        # A's rank vector is [1,1,1,1,1], B's is [2,2,2,2,2]. Under PAIRED
        # bootstrap every iteration is (rank_A=1, rank_B=2) so A's mean
        # rank CI collapses to [1.0, 1.0]. Under INDEPENDENT bootstrap A
        # still gets [1.0, 1.0] because its values are constant — so the
        # paired-vs-independent difference is invisible on constant vectors.
        # Use mixed ranks so the pairing is observable:
        a = ModelResult(model_name="a")
        b = ModelResult(model_name="b")
        # On 3 folds A wins, on 2 folds B wins — but A is overall better
        a.fold_metrics = [
            {"mae": 1.0, "rmse": 1.0},  # A wins
            {"mae": 1.0, "rmse": 1.0},
            {"mae": 5.0, "rmse": 5.0},  # B wins
            {"mae": 1.0, "rmse": 1.0},
            {"mae": 5.0, "rmse": 5.0},
        ]
        b.fold_metrics = [
            {"mae": 5.0, "rmse": 5.0},
            {"mae": 5.0, "rmse": 5.0},
            {"mae": 1.0, "rmse": 1.0},
            {"mae": 5.0, "rmse": 5.0},
            {"mae": 1.0, "rmse": 1.0},
        ]
        means, ranks, cis, _ = runner._compute_composite_ranks(
            {"a": a, "b": b}, metric_source="fold_metrics",
            bootstrap_iters=2000, bootstrap_seed=42,
        )
        # Under paired bootstrap every resample sums the per-fold ranks of A
        # and B at the SAME fold-ids, so the sum (rank_A + rank_B) at every
        # iteration is exactly 3 (one is 1, the other 2). That means
        # mean_A + mean_B is exactly 3.0 (no variance), which is a structural
        # property of paired ranks that the bootstrap preserves.
        a_low, a_high = cis["a"]
        b_low, b_high = cis["b"]
        # Pairing constraint: (a_low + b_high) and (a_high + b_low) both
        # equal 3.0 within float precision.
        assert abs((a_low + b_high) - 3.0) < 1e-9, (
            f"Paired bootstrap should preserve rank-sum=3 invariant: "
            f"a_low={a_low}, b_high={b_high}"
        )
        assert abs((a_high + b_low) - 3.0) < 1e-9, (
            f"Paired bootstrap should preserve rank-sum=3 invariant: "
            f"a_high={a_high}, b_low={b_low}"
        )


# --------------------------------------------------------------------------- #
# Peak-aware metric + selection ("stop flattening my spikes")
# --------------------------------------------------------------------------- #
def _spiky_load(days=30, per_day=48, reheats=(14, 34, 44), seed=0):
    """A realistic intermittent load: mostly off, a few reheats a day at
    varying heights. ~94% zeros — the regime real 30-minute hot-water, EV and
    appliance sensors actually sit in."""
    rng = np.random.default_rng(seed)
    n = days * per_day
    y = np.zeros(n)
    spikes = []
    for d in range(days):
        for slot in reheats:
            i = d * per_day + slot
            y[i] = float(rng.uniform(6, 14))
            spikes.append(i)
    return y, np.array(spikes)


def test_peak_weighted_prefers_peak_hitter_where_mae_prefers_undershooter():
    """The discriminating case: MAE crowns the model that undershoots the
    peaks (because peaks are rare), while peak_weighted_mae crowns the one
    that actually reaches them. This is the whole point of the metric.

    Deliberately run at ~94% zeros rather than a short toy series: below
    ~90% the plain p90 still separates from the median and the metric works
    either way, so a small fixture cannot detect the degeneracy this guards.
    """
    reg = get_metric_registry()
    yt, spikes = _spiky_load()
    base = np.setdiff1d(np.arange(yt.size), spikes)

    undershooter = np.zeros(yt.size)
    undershooter[spikes] = yt[spikes] * 0.5      # misses every peak by half
    peak_hitter = np.zeros(yt.size)
    peak_hitter[spikes] = yt[spikes]             # nails the peaks
    peak_hitter[base[:700]] = 0.9                # pays baseline noise to do it

    assert reg.compute("mae", yt, undershooter) < reg.compute("mae", yt, peak_hitter)
    assert (reg.compute("peak_weighted_mae", yt, peak_hitter)
            < reg.compute("peak_weighted_mae", yt, undershooter))


def test_peak_weighted_mae_does_not_degenerate_on_zero_heavy_target():
    """Regression guard. On a zero-heavy target the p90 sits inside the flat
    baseline, so the naive `p90 - median` scale collapses and the metric
    silently becomes plain MAE — on exactly the sensors it exists for. It
    must anchor the scale on the active part instead."""
    reg = get_metric_registry()
    yt, spikes = _spiky_load()
    assert (yt == 0).mean() > 0.9, "fixture must be in the degenerate regime"

    yp = np.zeros(yt.size)
    yp[spikes] = yt[spikes] * 0.5

    assert reg.compute("peak_weighted_mae", yt, yp) != pytest.approx(
        reg.compute("mae", yt, yp)
    ), "peak_weighted_mae collapsed to plain MAE on a spiky target"


def test_peak_weighted_mae_degenerate_falls_back_to_mae():
    """A genuinely constant target has no peaks to weight — MAE is honest."""
    reg = get_metric_registry()
    yt = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    yp = np.array([2.0, 4.0, 3.0, 3.0, 1.0])
    assert reg.compute("peak_weighted_mae", yt, yp) == pytest.approx(
        reg.compute("mae", yt, yp)
    )


def test_peak_weighted_mae_falls_back_when_too_few_active_intervals():
    """Too thin to estimate a peak scale from — fall back rather than weight
    on a two-sample quantile."""
    reg = get_metric_registry()
    yt = np.zeros(50)
    yt[[3, 20]] = 10.0
    yp = np.zeros(50)
    yp[[3, 20]] = 5.0
    assert reg.compute("peak_weighted_mae", yt, yp) == pytest.approx(
        reg.compute("mae", yt, yp)
    )


def test_pinball_q90_penalises_undershoot_more_than_overshoot():
    reg = get_metric_registry()
    yt = np.array([10.0, 10.0, 10.0, 10.0])
    under = np.array([8.0, 8.0, 8.0, 8.0])
    over = np.array([12.0, 12.0, 12.0, 12.0])
    assert reg.compute("pinball_q90", yt, under) > reg.compute("pinball_q90", yt, over)


def test_production_metric_weighted_makes_peak_champion_win():
    """With production_metric=peak_weighted_mae, the model that wins on peaks
    takes the crown even though it loses on mae/rmse — i.e. selection is no
    longer dominated by the L1/L2 metrics that favour the smoother."""
    from ml_forecast_lab.benchmark.runner import ModelResult

    cfg = _make_experiment_cfg(
        cv_folds=2, metrics=["mae", "rmse"],
        production_metric="peak_weighted_mae",
    )
    runner = BenchmarkRunner(cfg, _make_feature_builder())
    smoother = ModelResult(model_name="smoother")
    chaser = ModelResult(model_name="chaser")
    smoother.fold_metrics = [{"mae": 1.0, "rmse": 1.0, "peak_weighted_mae": 5.0}] * 2
    chaser.fold_metrics = [{"mae": 2.0, "rmse": 2.0, "peak_weighted_mae": 1.0}] * 2

    _, ranks, _, dnc = runner._compute_composite_ranks(
        {"smoother": smoother, "chaser": chaser},
        metric_source="fold_metrics", bootstrap_iters=10,
    )
    assert dnc == []
    assert ranks["chaser"] == 1, (
        f"peak-weighted selection metric should crown the peak-chaser; got {ranks}"
    )


def test_two_metric_weighting_is_neutral_keeps_mae_winner():
    """Control: with only 2 ranking metrics the production weight collapses to
    1 (no over-weighting), so a normal benchmark still picks the mae winner —
    the weighting must not invert ordinary results."""
    from ml_forecast_lab.benchmark.runner import ModelResult

    cfg = _make_experiment_cfg(
        cv_folds=2, metrics=["mae", "rmse"], production_metric="mae",
    )
    runner = BenchmarkRunner(cfg, _make_feature_builder())
    a = ModelResult(model_name="a")
    b = ModelResult(model_name="b")
    a.fold_metrics = [{"mae": 1.0, "rmse": 1.0}] * 2
    b.fold_metrics = [{"mae": 2.0, "rmse": 2.0}] * 2
    _, ranks, _, _ = runner._compute_composite_ranks(
        {"a": a, "b": b}, metric_source="fold_metrics", bootstrap_iters=10,
    )
    assert ranks["a"] == 1
