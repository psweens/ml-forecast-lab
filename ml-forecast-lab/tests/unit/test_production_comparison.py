"""
Comparison of the published-forecast leader with the production model
(v2.58.0).

The user decides which model publishes forecasts by promoting it; a later
benchmark only informs that decision. On a target that sits near zero most
of the time, CRPS separates models by a few per cent, so a different model
topping the table is not by itself a reason to switch. The Results tab
therefore says how the top model compares with the promoted one on a
paired comparison over the same issue times. An earlier cut acted on that
comparison — keeping the previous winner as "Best" unless beaten clearly —
which contradicted the table the user was choosing from. Pins:

- nothing to compare without a top model, without a production model, or
  when the production model was not scored;
- ``same`` when the production model tops the table;
- otherwise the improvement is the top model's CRPS reduction relative to
  the production model's, over shared issue times only, with the paired
  day-block verdict;
- the benchmark computes it for display and never changes the best model
  with it.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.benchmark import horizon as H


def _scores(name, crps):
    times = pd.date_range("2026-11-01", periods=len(crps), freq="6h")
    crps = np.asarray(crps, float)
    return H.HorizonScores(
        name=name, n_origins=len(crps), n_points=len(crps) * 48,
        mae=1.0, rmse=1.0, bias_pct=0.0, crps=float(np.mean(crps)),
        coverage=0.8, interval_width=1.0, amplitude_ratio=1.0,
        period_total_rmse=10.0, origin_crps=crps, origin_times=times,
    )


def _setup(top_crps, prod_crps):
    ev = H.HorizonEvaluation(
        scores={"top": _scores("top", top_crps), "prod": _scores("prod", prod_crps)},
        evaluation_days=len(top_crps) / 4, enough_evidence=True, coverage_level=0.8,
    )
    rows = [{"name": n, "crps": sc.crps, "eligible": True, "candidate": True}
            for n, sc in ev.scores.items()]
    return ev, rows


BASE = 1.0 + np.random.default_rng(1).normal(0.0, 0.3, 80)  # 20 days × 4


class TestNothingToCompare:
    def test_no_production_model(self):
        ev, rows = _setup(BASE, BASE)
        assert H.compare_with_production(ev, rows, "top", None) is None

    def test_no_top_model(self):
        ev, rows = _setup(BASE, BASE)
        assert H.compare_with_production(ev, rows, None, "prod") is None

    def test_production_model_not_scored(self):
        ev, rows = _setup(BASE, BASE)
        rows = [r for r in rows if r["name"] != "prod"]
        assert H.compare_with_production(ev, rows, "top", "prod") is None


class TestComparison:
    def test_production_model_on_top(self):
        ev, rows = _setup(BASE, BASE)
        c = H.compare_with_production(ev, rows, "prod", "prod")
        assert c["same"] is True

    def test_consistent_improvement(self):
        ev, rows = _setup(BASE * 0.95, BASE)
        c = H.compare_with_production(ev, rows, "top", "prod")
        assert not c["same"]
        assert c["improvement"] == pytest.approx(0.05)
        assert c["verdict"] == "better"
        assert c["n_days"] == 20

    def test_noisy_difference_is_indistinguishable(self):
        rng = np.random.default_rng(2)
        ev, rows = _setup(0.97 + rng.normal(0, 0.4, 12), 1.0 + rng.normal(0, 0.4, 12))
        c = H.compare_with_production(ev, rows, "top", "prod")
        assert c["verdict"] == "indistinguishable"

    def test_only_shared_issue_times_are_compared(self):
        ev, rows = _setup(BASE * 0.5, BASE)
        half = len(BASE) // 2
        prod = ev.scores["prod"]
        prod.origin_crps, prod.origin_times = prod.origin_crps[half:], prod.origin_times[half:]
        ev.scores["top"].origin_crps = np.r_[BASE[:half] * 0.5, BASE[half:]]
        c = H.compare_with_production(ev, rows, "top", "prod")
        assert c["improvement"] == pytest.approx(0.0, abs=1e-12)
        assert c["n_days"] == half // 4


class TestWiring:
    def test_benchmark_shows_it_and_never_acts_on_it(self):
        from ml_forecast_lab import main
        src = inspect.getsource(main.MLForecastLabApp._run_benchmark)
        assert "compare_with_production(" in src
        assert "production=exp_cfg.production_model or None" in src
        # The best model is the table's winner; the comparison is display-only.
        assert 'best_model_name = published["winner"]' in src
        assert "promotion_decision" not in src
        assert "best_model_name = _incumbent" not in src
