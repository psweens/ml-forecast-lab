"""
Promotion gate for selection on the published forecast (v2.58.0).

Slot-level CRPS separates forecasts of a non-negative target by a few per
cent at most, and selection on it re-ran on every benchmark, so the
champion could change with sampling noise alone — each change resetting
the published band's calibration. Pins:

- with no incumbent, or the incumbent winning again, the CRPS winner is
  promoted;
- an incumbent that was not scored, or now fails the period-total
  guardrail, is replaced;
- otherwise the challenger replaces it only on a paired day-block win of
  at least ``PROMOTION_MIN_IMPROVEMENT``; a marginal or noisy advantage
  keeps the incumbent;
- the comparison uses only issue times both were scored on;
- the benchmark wires the decision in and keeps the incumbent when told to.
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


def _setup(challenger_crps, incumbent_crps, incumbent_eligible=True,
           incumbent_candidate=True):
    ev = H.HorizonEvaluation(
        scores={"new": _scores("new", challenger_crps),
                "old": _scores("old", incumbent_crps)},
        evaluation_days=len(challenger_crps) / 4, enough_evidence=True,
        coverage_level=0.8,
    )
    rows = [
        {"name": "new", "crps": float(np.mean(challenger_crps)),
         "eligible": True, "candidate": True},
        {"name": "old", "crps": float(np.mean(incumbent_crps)),
         "eligible": incumbent_eligible, "candidate": incumbent_candidate},
    ]
    return ev, rows


def _noise(n, seed, scale):
    return np.random.default_rng(seed).normal(0.0, scale, n)


BASE = 1.0 + _noise(80, 1, 0.3)  # 20 days of four issue times


class TestDecision:
    def test_no_incumbent_promotes(self):
        ev, rows = _setup(BASE, BASE)
        d = H.promotion_decision(ev, rows, challenger="new", incumbent=None)
        assert d["promote"] and d["paired"] is None

    def test_incumbent_winning_again_promotes(self):
        ev, rows = _setup(BASE, BASE)
        assert H.promotion_decision(ev, rows, "new", "new")["promote"]

    def test_unscored_incumbent_is_replaced(self):
        ev, rows = _setup(BASE, BASE)
        rows = [r for r in rows if r["name"] != "old"]
        d = H.promotion_decision(ev, rows, "new", "old")
        assert d["promote"] and "not scored" in d["reason"]

    def test_excluded_incumbent_is_replaced(self):
        ev, rows = _setup(BASE, BASE, incumbent_candidate=False)
        assert H.promotion_decision(ev, rows, "new", "old")["promote"]

    def test_incumbent_failing_the_guardrail_is_replaced(self):
        ev, rows = _setup(BASE, BASE, incumbent_eligible=False)
        d = H.promotion_decision(ev, rows, "new", "old")
        assert d["promote"] and "guardrail" in d["reason"]

    def test_clear_paired_win_promotes(self):
        # 5 % lower on every issue time, with the shared noise cancelling.
        ev, rows = _setup(BASE * 0.95, BASE)
        d = H.promotion_decision(ev, rows, "new", "old")
        assert d["promote"]
        assert d["paired"]["verdict"] == "better"

    def test_significant_but_marginal_win_keeps_the_incumbent(self):
        # Consistently lower, but by 0.5 %: significant, below the margin.
        ev, rows = _setup(BASE * 0.995, BASE)
        d = H.promotion_decision(ev, rows, "new", "old")
        assert d["paired"]["verdict"] == "better"
        assert not d["promote"]
        assert d["reason"].startswith("kept old")

    def test_noisy_win_keeps_the_incumbent(self):
        # Lower on average by 3 %, but with independent per-origin noise
        # far larger than that over a few days.
        n = 12
        ev, rows = _setup(0.97 + _noise(n, 2, 0.4), 1.0 + _noise(n, 3, 0.4))
        d = H.promotion_decision(ev, rows, "new", "old")
        assert d["paired"]["verdict"] != "better"
        assert not d["promote"]

    def test_worse_challenger_keeps_the_incumbent(self):
        ev, rows = _setup(BASE * 1.05, BASE)
        d = H.promotion_decision(ev, rows, "new", "old")
        assert not d["promote"] and "higher" in d["reason"]

    def test_only_shared_issue_times_are_compared(self):
        ev, rows = _setup(BASE * 0.95, BASE)
        # The incumbent was scored on the second half only, where the
        # challenger is no better.
        old = ev.scores["old"]
        half = len(BASE) // 2
        old.origin_crps, old.origin_times = old.origin_crps[half:], old.origin_times[half:]
        ev.scores["new"].origin_crps = np.r_[BASE[:half] * 0.5, BASE[half:]]
        d = H.promotion_decision(ev, rows, "new", "old")
        assert not d["promote"]
        assert d["paired"]["n_days"] == half // 4


class TestWiring:
    def test_benchmark_applies_the_decision(self):
        from ml_forecast_lab import main
        src = inspect.getsource(main.MLForecastLabApp._run_benchmark)
        assert "promotion_decision(" in src
        assert 'selection_metric", None) == "crps"' in src
        assert "best_model_name = _incumbent" in src
