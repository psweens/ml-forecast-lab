"""
Model selection on the published forecast (v2.58.0).

Pins the selection defect behind the 2026-11 intermittent-demand
investigation. The leaderboard scored every backend one step ahead with
measured lags and ranked on MAE-type metrics; on a non-negative target that
sits near zero most of the time, that prefers an under-levelled,
near-constant forecast — and the forecast scored was not the one published.

- the runner collects each backend's full-horizon forecast inside its test
  folds (recursive for a tabular backend) and scores it with the reference
  forecasts;
- selection is by CRPS with a period-total guardrail when
  ``production_metric`` is ``crps``, or when the target is non-negative and
  the metric is median-seeking; otherwise the one-step ranking stands and
  the scores are informational;
- a backend that could not be scored leaves selection on the one-step rank;
- the guardrail removes a forecast whose totals are badly wrong even when
  its CRPS is the lowest;
- frames without a DatetimeIndex behave exactly as before.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.benchmark import horizon as H
from ml_forecast_lab.benchmark.runner import BenchmarkRunner, ModelResult
from ml_forecast_lab.features import build_features, rebuild_fold_features
from ml_forecast_lab.models.base import ForecastModel
from tests.synthetic import intermittent_demand as gen

INTERVAL = 30
HORIZON = 48


def _frame(days=24, seed=0):
    y, _ = gen.slot_truth("2026-11-02", days, seed=seed)
    y.index = y.index.tz_localize(None)
    df = pd.DataFrame({"y": y})
    feats = build_features(df, "y", INTERVAL)
    return feats.assign(target=df["y"]).dropna()


def _feature_builder(df_sub, config, purpose="train"):
    out = rebuild_fold_features(df_sub.copy(), "target", INTERVAL, [6, 24, 72], 48)
    cols = [c for c in out.columns if c != "target"]
    return np.nan_to_num(out[cols].values.astype(np.float32))


class _Tabular(ForecastModel):
    """Predicts ``scale`` × the mean target seen at fit for each hour."""

    is_neural = False

    def __init__(self, name, scale=1.0, flat=None):
        super().__init__()
        self._name, self.scale, self.flat = name, scale, flat
        self.by_hour, self.hour_col = {}, 0

    @property
    def name(self):
        return self._name

    def fit(self, X, y, **kw):
        names = list(kw.get("feature_names") or [])
        self.hour_col = names.index("hour_of_day")
        hours = X[:, self.hour_col].astype(int)
        self.by_hour = {h: float(y[hours == h].mean()) for h in np.unique(hours)}
        self._is_fitted = True
        return {}

    def predict(self, X):
        if self.flat is not None:
            return np.full(len(X), self.flat, dtype=np.float32)
        hours = X[:, self.hour_col].astype(int)
        return np.array(
            [self.scale * self.by_hour.get(h, 0.0) for h in hours], dtype=np.float32,
        )

    def get_params(self):
        return {}

    def set_params(self, **kw):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


def _runner(**cfg):
    base = dict(
        name="e", interval_minutes=INTERVAL, future_periods=HORIZON,
        cv_strategy="walk_forward", cv_folds=4, production_metric="seasonal_mase",
        metrics=["mae", "rmse", "mase", "seasonal_mase"], timezone=gen.TZ,
        source_is_cumulative=True,
    )
    base.update(cfg)
    return BenchmarkRunner(base, _feature_builder)


def _models():
    return {
        "profile": _Tabular("profile"),
        "near_zero": _Tabular("near_zero", flat=0.2),
    }


class TestCollection:
    def test_tabular_backend_gets_full_horizon_forecasts_per_fold(self):
        df, runner = _frame(), _runner()
        folds = runner._prepare_train_test_splits(df)
        mr = runner.run_single_model(df, _Tabular("profile"), folds)
        assert mr.horizon_folds, "no published forecasts were collected"
        for ts, mat in mr.horizon_folds:
            assert mat.shape == (len(ts), HORIZON)
            assert np.isfinite(mat).all()
            assert len(ts) <= 24

    def test_no_datetime_index_collects_nothing(self):
        df, runner = _frame().reset_index(drop=True), _runner()
        folds = runner._prepare_train_test_splits(df)
        mr = runner.run_single_model(df, _Tabular("profile"), folds)
        assert mr.horizon_folds == []
        assert runner.score_published_forecasts(df, {"profile": mr}) is None


class TestWhenItSelects:
    @pytest.mark.parametrize("cfg,expected", [
        (dict(production_metric="crps", source_is_cumulative=False), True),
        (dict(production_metric="seasonal_mase"), True),
        (dict(production_metric="mae", source_is_cumulative=False,
              target_is_nonnegative=True), True),
        (dict(production_metric="rmse"), False),
        (dict(production_metric="seasonal_mase", source_is_cumulative=False), False),
    ])
    def test_rule(self, cfg, expected):
        assert _runner(**cfg).selects_on_published_forecast() is expected


class TestSelection:
    @pytest.fixture(scope="class")
    def frame(self):
        return _frame()

    def test_one_step_mae_prefers_the_near_zero_forecast(self, frame):
        result = _runner(source_is_cumulative=False).run_benchmark(frame, _models())
        m = result.model_results
        assert m["near_zero"].metrics["mae"] < m["profile"].metrics["mae"]
        assert result.best_model == "near_zero"
        assert result.metric_used == "seasonal_mase"
        # Scored and reported all the same.
        assert {r["name"] for r in result.published_scores} >= {"profile", "near_zero"}

    def test_published_forecast_selection_picks_the_levelled_model(self, frame):
        result = _runner().run_benchmark(frame, _models())
        assert result.best_model == "profile"
        assert result.metric_used == "crps"
        assert result.rankings == {"profile": 1, "near_zero": 2}
        m = result.model_results
        assert m["profile"].metrics["crps"] < m["near_zero"].metrics["crps"]
        assert abs(m["profile"].metrics["published_bias_pct"]) < 30
        assert m["near_zero"].metrics["published_bias_pct"] < -70
        assert result.best_metric_value == pytest.approx(m["profile"].metrics["crps"])

    def test_reference_forecasts_are_scored_alongside(self, frame):
        result = _runner().run_benchmark(frame, _models())
        names = [r["name"] for r in result.published_scores]
        for ref in ("baseline_zero", "baseline_mean", "baseline_profile_mean",
                    "baseline_profile_median", "baseline_previous_day"):
            assert ref in names
        refs = [r for r in result.published_scores if not r["candidate"]]
        assert len(refs) == 5
        assert result.published_evaluation_days > 0

    def test_unscored_backend_leaves_the_one_step_ranking(self, frame):
        runner = _runner()
        folds = runner._prepare_train_test_splits(frame)
        results = {n: runner.run_single_model(frame, m, folds) for n, m in _models().items()}
        results["profile"].horizon_folds = []
        published = runner.score_published_forecasts(frame, results, tz=gen.TZ)
        assert published["winner"] is None
        assert published["unscored"] == ["profile"]
        assert runner.rank_by_published_scores(published, {"profile": 1, "near_zero": 2}) is None

    def test_excluded_backend_is_scored_but_cannot_win(self, frame):
        runner = _runner()
        folds = runner._prepare_train_test_splits(frame)
        results = {n: runner.run_single_model(frame, m, folds) for n, m in _models().items()}
        published = runner.score_published_forecasts(
            frame, results, tz=gen.TZ, exclude=["profile"],
        )
        assert published["winner"] == "near_zero"
        row = next(r for r in published["rows"] if r["name"] == "profile")
        assert np.isfinite(row["crps"]) and not row["candidate"]


class TestGuardrail:
    def _evaluation(self, totals, crps):
        scores = {}
        for name in totals:
            scores[name] = H.HorizonScores(
                name=name, n_origins=10, n_points=100, mae=1.0, rmse=1.0,
                bias_pct=0.0, crps=crps[name], coverage=0.8, interval_width=1.0,
                amplitude_ratio=1.0, period_total_rmse=totals[name],
            )
        return H.HorizonEvaluation(scores, 20.0, True, 0.8)

    def test_lowest_crps_loses_when_its_totals_are_badly_wrong(self):
        ev = self._evaluation(
            totals={"baseline_mean": 10.0, "a": 13.0, "b": 11.0},
            crps={"baseline_mean": 1.0, "a": 0.80, "b": 0.85},
        )
        winner, rows = H.select_by_crps(ev, ["a", "b"])
        assert winner == "b"
        by = {r["name"]: r for r in rows}
        assert not by["a"]["eligible"] and by["b"]["eligible"]

    def test_tolerance_is_a_quarter_over_the_constant_mean(self):
        assert H.PERIOD_TOTAL_TOLERANCE == 1.25
        ev = self._evaluation(
            totals={"baseline_mean": 10.0, "a": 12.5}, crps={"baseline_mean": 1.0, "a": 0.9},
        )
        assert H.select_by_crps(ev, ["a"])[1][0]["eligible"]

    def test_a_winner_is_named_even_if_every_candidate_fails(self):
        ev = self._evaluation(
            totals={"baseline_mean": 10.0, "a": 30.0, "b": 40.0},
            crps={"baseline_mean": 1.0, "a": 0.9, "b": 0.8},
        )
        winner, rows = H.select_by_crps(ev, ["a", "b"])
        assert winner == "b"
        assert not any(r["eligible"] for r in rows if r["candidate"])

    def test_ranking_puts_eligible_backends_first(self):
        published = {"winner": "b", "rows": [
            {"name": "a", "crps": 0.80, "eligible": False, "candidate": True},
            {"name": "b", "crps": 0.85, "eligible": True, "candidate": True},
            {"name": "baseline_mean", "crps": 0.5, "eligible": True, "candidate": False},
        ]}
        assert BenchmarkRunner.rank_by_published_scores(
            published, {"a": 1, "b": 2},
        ) == {"b": 1, "a": 2}


class TestTuningObjective:
    def test_objective_is_crps_and_penalises_a_guardrail_failure(self):
        frame, runner = _frame(), _runner()
        folds = runner._prepare_train_test_splits(frame)
        good = runner.run_single_model(frame, _Tabular("profile"), folds)
        bad = runner.run_single_model(frame, _Tabular("near_zero", flat=0.2), folds)
        g = runner.published_objective(frame, good, tz=gen.TZ)
        b = runner.published_objective(frame, bad, tz=gen.TZ)
        assert g == pytest.approx(good.metrics["crps"])
        assert b > bad.metrics["crps"], "a failed guardrail must cost more than its CRPS"
        assert g < b

    def test_unscorable_model_has_no_objective(self):
        frame, runner = _frame().reset_index(drop=True), _runner()
        assert runner.published_objective(frame, ModelResult(model_name="m")) is None
