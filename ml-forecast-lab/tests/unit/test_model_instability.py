"""Unstable-model flag (v2.39.5).

The composite mean rank is outlier-robust — a model that is strong on
most folds but catastrophic on one can out-rank a consistently mediocre
model, because the blow-up only costs it one last-place finish. The
leaderboard now flags such models so the rank can't make a blow-up-prone
model look like a solid mid-pack pick. These tests cover the detector.
"""
from ml_forecast_lab.main import _assess_model_instability


def _folds(metric, values):
    return [{metric: v} for v in values]


def test_blowup_fold_flagged():
    """N-HiTS-like: great on most folds, catastrophic on one. The worst
    fold dwarfs the median → flagged with the blow-up reason."""
    folds = _folds("mase", [1.1, 0.9, 1.2, 1.0, 46000.0])
    unstable, reason = _assess_model_instability(folds, ["mase"])
    assert unstable is True
    assert "blew up" in reason
    assert "mase" in reason


def test_consistent_model_not_flagged():
    """DLinear-like: tight, low-variance folds → stable."""
    folds = _folds("mase", [1.10, 1.20, 1.05, 1.15, 1.18])
    unstable, reason = _assess_model_instability(folds, ["mase"])
    assert unstable is False
    assert reason is None


def test_high_dispersion_without_single_blowup_flagged():
    """std >= mean (CV >= 1.0) but the worst fold is < 10x the median →
    flagged with the dispersion reason rather than the blow-up reason."""
    # [1,1,1,1,8]: mean 2.4, std 2.8 (CV 1.17 >= 1.0); max/median = 8/1 = 8 < 10
    folds = _folds("mase", [1.0, 1.0, 1.0, 1.0, 8.0])
    unstable, reason = _assess_model_instability(folds, ["mase"])
    assert unstable is True
    assert "swings" in reason  # CV branch, not the blow-up branch


def test_single_fold_not_judged():
    """A flag off one fold would be meaningless → never unstable."""
    unstable, reason = _assess_model_instability(_folds("mase", [9999.0]), ["mase"])
    assert unstable is False
    assert reason is None


def test_all_zero_error_not_flagged():
    """Degenerate all-zero error (mean <= 0) must not divide-by-zero or
    flag — there's nothing unstable about a perfect-on-every-fold model."""
    unstable, reason = _assess_model_instability(_folds("mase", [0.0, 0.0, 0.0]), ["mase"])
    assert unstable is False
    assert reason is None


def test_falls_back_when_primary_metric_absent():
    """If the production metric has no fold data, fall through to the next
    candidate (mase, then mae)."""
    folds = [{"mae": v} for v in [1.0, 1.0, 1.0, 1.0, 200.0]]
    # 'seasonal_mase' absent from every fold → assessed on 'mae'
    unstable, reason = _assess_model_instability(
        folds, ["seasonal_mase", "mase", "mae"],
    )
    assert unstable is True
    assert "mae" in reason


def test_empty_and_nan_folds_ignored():
    """Empty {} (errored fold) and NaN/inf values are skipped, not counted
    as zeros that would distort the spread."""
    import numpy as np
    folds = [
        {"mase": 1.0}, {}, {"mase": float("nan")},
        {"mase": 1.1}, {"mase": float("inf")}, {"mase": 1.05},
    ]
    unstable, reason = _assess_model_instability(folds, ["mase"])
    # Three finite, tight values → stable; inf/nan/{} didn't blow it up.
    assert unstable is False
