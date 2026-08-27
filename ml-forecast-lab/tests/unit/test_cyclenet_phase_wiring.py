"""v2.52.0: cyclenet's cycle phase is plumbed, absolute, and gated.

CycleNet's Residual Cycle Forecasting is the first backend that needs the
ABSOLUTE grid position of every window it sees — the same window content
at 06:00 and 18:00 must read different rows of the learned cycle. Three
things keep that honest, and this module pins each:

1. ``features.grid_step_index`` anchors positions to the epoch, so two
   frame slices covering the same wall-clock rows agree — retention
   trimming, fold bridging, and cache reload cannot rotate the phase.
2. ``models.base.predict_sequence_with_context`` forwards the indices
   only to backends declaring ``needs_window_step_index``; every other
   backend keeps its bare ``predict_sequence(X)`` signature.
3. Source contracts: the benchmark runner and main.py call sites really
   do pass ``window_step_index`` — unit tests cannot reach that wiring,
   so the snippets are pinned directly (expect to update these when
   refactoring those lines).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.features import grid_step_index
from ml_forecast_lab.models.base import predict_sequence_with_context

APP_DIR = Path(__file__).resolve().parents[2]


class TestGridStepIndex:
    def test_epoch_anchoring_is_slice_invariant(self):
        # The same wall-clock row must get the same step number no matter
        # which frame slice it was computed from.
        full = pd.date_range("2026-01-01", periods=200, freq="30min")
        trimmed = full[37:]
        steps_full = grid_step_index(full, np.arange(len(full)))
        steps_trimmed = grid_step_index(trimmed, np.arange(len(trimmed)))
        assert np.array_equal(steps_full[37:], steps_trimmed)

    def test_consecutive_rows_differ_by_one(self):
        idx = pd.date_range("2026-01-01", periods=50, freq="30min")
        steps = grid_step_index(idx, np.arange(50))
        assert np.all(np.diff(steps) == 1)

    def test_daily_cycle_phase_from_timestamps(self):
        # At 30-min sampling, step mod 48 must advance with time of day.
        idx = pd.date_range("2026-01-01 00:00", periods=48, freq="30min")
        steps = grid_step_index(idx, np.arange(48))
        phases = steps % 48
        # 48 consecutive half-hours cover each phase exactly once.
        assert len(np.unique(phases)) == 48

    def test_positions_select_rows(self):
        idx = pd.date_range("2026-01-01", periods=100, freq="30min")
        all_steps = grid_step_index(idx, np.arange(100))
        kept = np.array([3, 17, 42])
        assert np.array_equal(grid_step_index(idx, kept), all_steps[kept])

    def test_non_datetime_index_returns_none(self):
        assert grid_step_index(pd.RangeIndex(50), np.arange(50)) is None

    def test_too_short_index_returns_none(self):
        idx = pd.date_range("2026-01-01", periods=1, freq="30min")
        assert grid_step_index(idx, [0]) is None


class _BareModel:
    """predict_sequence with the bare (X) signature — the 31 other backends."""
    def predict_sequence(self, X):
        return ("bare", X)


class _PhaseAwareModel:
    needs_window_step_index = True

    def predict_sequence(self, X, window_step_index=None):
        return ("aware", window_step_index)


class TestPredictSequenceWithContext:
    def test_bare_backend_never_receives_kwarg(self):
        tag, _ = predict_sequence_with_context(
            _BareModel(), np.zeros((2, 4, 1)), np.array([0, 1]),
        )
        assert tag == "bare"

    def test_phase_aware_backend_receives_indices(self):
        steps = np.array([10, 11])
        tag, got = predict_sequence_with_context(
            _PhaseAwareModel(), np.zeros((2, 4, 1)), steps,
        )
        assert tag == "aware"
        assert got is steps

    def test_none_indices_degrade_to_bare_call(self):
        tag, got = predict_sequence_with_context(
            _PhaseAwareModel(), np.zeros((2, 4, 1)), None,
        )
        assert tag == "aware"
        assert got is None


class TestSourceContracts:
    """Pin the call-site wiring that unit tests cannot execute."""

    def _src(self, rel: str) -> str:
        return (APP_DIR / rel).read_text(encoding="utf-8")

    def test_runner_passes_fit_and_predict_indices(self):
        src = self._src("ml_forecast_lab/benchmark/runner.py")
        assert "sequence_kwargs['window_step_index']" in src, (
            "benchmark runner no longer passes window_step_index to fit()"
        )
        # Train diagnostics + both test-window branches go through the shim.
        assert src.count("predict_sequence_with_context(") >= 3, (
            "benchmark runner predict paths bypass "
            "predict_sequence_with_context"
        )

    def test_runner_precomputed_fold_carries_indices(self):
        src = self._src("ml_forecast_lab/benchmark/runner.py")
        assert "pc_fold.get('window_step_index')" in src, (
            "precomputed-fold path drops window_step_index"
        )

    def test_main_wires_every_window_path(self):
        src = self._src("ml_forecast_lab/main.py")
        # Fit-side: benchmark-holdout, retrain-legacy, retrain-and-cache,
        # tuning precompute, tuning holdout, covariate analysis.
        assert src.count("window_step_index") >= 8, (
            "main.py window paths lost their window_step_index wiring"
        )
        # Predict-side: holdout chart, post-retrain forecast, cached tick,
        # tuning holdout, covariate analysis.
        assert src.count("predict_sequence_with_context(") >= 5, (
            "main.py predict paths bypass predict_sequence_with_context"
        )

    def test_inference_window_anchor_arithmetic_is_pinned(self):
        # The two single-window production paths anchor the window's first
        # row at (tail length - past window size); a drift here rotates the
        # cycle phase of every published cyclenet forecast.
        src = self._src("ml_forecast_lab/main.py")
        assert "[len(_frame_tail_prod) - past_window_size_prod]" in src
        assert "[len(_win_src) - window_size]" in src
