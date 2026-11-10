"""
Rolling-origin evaluation of the published forecast (benchmark.horizon).

Pins the defect behind the 2026-11 intermittent-demand investigation: the
leaderboard scored tree backends one step ahead with measured lags and
ranked them on MAE-type metrics, so on a non-negative right-skewed target
an under-levelled, near-constant forecast outranked a correct one. These
tests hold the replacement to its contract:

- the recursive forecast row matches ``build_features`` column for column;
- CRPS from a quantile grid is a proper score (degenerate → absolute error;
  the true distribution beats a shifted one);
- the reference forecasts use local time and never read the future;
- an under-levelled forecaster ranks below the seasonal-mean profile, which
  MAE gets the wrong way round;
- period totals are scored from the issue time to local midnight;
- labels are never imputed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.benchmark import horizon as H
from ml_forecast_lab.features import build_features, recursive_tree_forecast
from tests.synthetic import intermittent_demand as gen

INTERVAL = 30


def _series(days=30, seed=0, start="2026-10-05"):
    y, _ = gen.slot_truth(start, days, seed=seed)
    y.index = y.index.tz_localize(None)
    return y


class TestRecursiveRowParity:
    def test_first_step_row_equals_build_features(self):
        y = _series(days=6)
        history, nxt = y.iloc[:-1], y.iloc[-1:]
        feats = build_features(pd.DataFrame({"y": y}), "y", INTERVAL)
        cols = list(feats.columns)
        expected = feats.iloc[-1].to_numpy(dtype=np.float32)
        seen = {}

        def predict(X):
            seen.setdefault("row", X[0].copy())
            return np.array([0.0])

        out = recursive_tree_forecast(predict, cols, history, 1, INTERVAL)
        assert out.index[0] == nxt.index[0]
        np.testing.assert_allclose(seen["row"], expected, rtol=1e-5, atol=1e-6)

    def test_predictions_feed_back_as_lags(self):
        y = _series(days=4)
        cols = list(build_features(pd.DataFrame({"y": y}), "y", INTERVAL).columns)
        lag1 = cols.index("y_lag_1")
        rows = []

        def predict(X):
            rows.append(X[0].copy())
            return np.array([7.0 + len(rows)])

        recursive_tree_forecast(predict, cols, y, 3, INTERVAL)
        assert rows[0][lag1] == pytest.approx(y.iloc[-1])
        assert rows[1][lag1] == pytest.approx(8.0)
        assert rows[2][lag1] == pytest.approx(9.0)


class TestScoringRule:
    def test_degenerate_distribution_scores_absolute_error(self):
        y = np.array([0.0, 2.0, 5.0])
        q = np.tile(np.array([1.0, 1.0, 1.0])[:, None], (1, len(H.QUANTILE_LEVELS)))
        np.testing.assert_allclose(H.crps_from_quantiles(y, q), np.abs(y - 1.0))

    def test_true_distribution_beats_shifted_and_point(self):
        rng = np.random.default_rng(0)
        y = rng.lognormal(0.0, 1.0, size=20000)
        true_q = np.quantile(rng.lognormal(0.0, 1.0, size=200000), H.QUANTILE_LEVELS)
        tile = lambda q: np.tile(q, (len(y), 1))
        good = H.crps_from_quantiles(y, tile(true_q)).mean()
        shifted = H.crps_from_quantiles(y, tile(true_q * 0.5)).mean()
        point = H.crps_from_quantiles(y, tile(np.full_like(true_q, np.median(y)))).mean()
        assert good < shifted
        assert good < point


class TestBaselines:
    def test_profile_groups_by_local_time_across_a_clock_change(self):
        # Constant 5.0 in the local 08:00 slot, 0 elsewhere, over a window
        # that spans the 2026-10-25 change: in UTC the slot moves an hour.
        idx = pd.date_range("2026-10-12", periods=28 * 48, freq="30min")
        loc = idx.tz_localize("UTC").tz_convert(gen.TZ)
        y = pd.Series(np.where((loc.hour == 8) & (loc.minute == 0), 5.0, 0.0), index=idx)
        fut = pd.date_range(idx[-1] + pd.Timedelta(minutes=30), periods=48, freq="30min")
        f_loc = fut.tz_localize("UTC").tz_convert(gen.TZ)
        local = H.SeasonalProfileBaseline(INTERVAL, gen.TZ, "mean").predict(y, fut)
        assert local[(f_loc.hour == 8) & (f_loc.minute == 0)] == pytest.approx(5.0)
        assert local.sum() == pytest.approx(5.0)
        utc = H.SeasonalProfileBaseline(INTERVAL, None, "mean").predict(y, fut)
        assert utc.max() < 5.0, "a UTC profile smears the slot across the change"

    def test_previous_day_repeats_the_last_measured_day(self):
        y = _series(days=3)
        fut = pd.date_range(y.index[-1] + pd.Timedelta(minutes=30), periods=96, freq="30min")
        out = H.PreviousDayBaseline(INTERVAL).predict(y, fut)
        np.testing.assert_allclose(out[:48], y.to_numpy()[-48:])
        np.testing.assert_allclose(out[48:], y.to_numpy()[-48:])

    def test_default_set(self):
        names = [b.name for b in H.default_baselines(INTERVAL, gen.TZ)]
        assert names == [
            "baseline_zero", "baseline_mean", "baseline_profile_mean",
            "baseline_profile_median", "baseline_previous_day",
        ]


class _Scaled(H.Forecaster):
    """The seasonal-mean profile times a constant: a known level error."""

    def __init__(self, factor, name):
        self.inner = H.SeasonalProfileBaseline(INTERVAL, gen.TZ, "mean")
        self.factor, self.name = factor, name

    def predict(self, history, future_index):
        return self.factor * self.inner.predict(history, future_index)


class _Flat(H.Forecaster):
    def __init__(self, level, name):
        self.level, self.name = level, name

    def predict(self, history, future_index):
        return np.full(len(future_index), self.level)


def _evaluate(y, forecasters, **kw):
    params = dict(
        interval_minutes=INTERVAL, horizon=96, first_origin=12 * 48,
        origin_stride=7, refit_every=7, tz=gen.TZ, nonnegative=True,
    )
    params.update(kw)
    return H.evaluate_forecasters(y, forecasters, **params)


class TestRanking:
    @pytest.fixture(scope="class")
    def evaluation(self):
        y = _series(days=40)
        fcs = H.default_baselines(INTERVAL, gen.TZ) + [
            _Flat(0.2, "flat_near_median"),
            _Scaled(1.0, "profile_point_only"),
        ]
        return _evaluate(y, fcs)

    def test_mae_prefers_the_degenerate_forecast(self, evaluation):
        s = evaluation.scores
        assert s["flat_near_median"].mae < s["baseline_profile_mean"].mae

    def test_crps_and_rmse_rank_it_below_the_seasonal_mean(self, evaluation):
        s = evaluation.scores
        assert s["flat_near_median"].crps > s["baseline_profile_mean"].crps
        assert s["flat_near_median"].rmse > s["baseline_profile_mean"].rmse
        assert evaluation.table().index[0].startswith("baseline_profile")

    def test_level_and_shape_are_reported(self, evaluation):
        s = evaluation.scores
        # A profile fitted on a couple of weeks of a target whose daily
        # totals vary by a third carries real level noise; it is still far
        # from the flat forecast's.
        assert abs(s["baseline_profile_mean"].bias_pct) < 25
        assert s["flat_near_median"].bias_pct < -70
        assert s["flat_near_median"].amplitude_ratio == pytest.approx(0.0)
        assert s["baseline_profile_mean"].amplitude_ratio > 0.3

    def test_paired_comparison_reaches_a_verdict(self, evaluation):
        s = evaluation.scores
        d = H.paired_difference(
            s["baseline_zero"].origin_crps, s["baseline_profile_mean"].origin_crps,
            s["baseline_zero"].origin_times,
        )
        assert d["verdict"] == "worse" and d["lo"] > 0
        same = H.paired_difference(
            s["baseline_profile_mean"].origin_crps,
            s["baseline_profile_mean"].origin_crps,
            s["baseline_profile_mean"].origin_times,
        )
        assert same["verdict"] == "indistinguishable"

    def test_every_forecaster_is_scored_on_the_same_cells(self, evaluation):
        counts = {s.n_points for s in evaluation.scores.values()}
        assert len(counts) == 1

    def test_evidence_flag_follows_the_window(self, evaluation):
        assert evaluation.enough_evidence
        short = _evaluate(_series(days=16), [H.ZeroBaseline()], first_origin=10 * 48)
        assert not short.enough_evidence


class TestPeriodTotals:
    def test_remaining_total_error_falls_with_issue_time(self):
        y = _series(days=40)
        ev = _evaluate(y, [H.SeasonalProfileBaseline(INTERVAL, gen.TZ, "mean")])
        pt = ev.scores["baseline_profile_mean"].period_total
        early = pt[pt.issue_hour < 6]["rmse"].mean()
        late = pt[pt.issue_hour >= 18]["rmse"].mean()
        assert late < 0.6 * early

    def test_total_runs_to_local_midnight(self):
        # 1.0 in every slot: the remaining total is the number of slots left
        # in the local day, and a zero forecast misses exactly that.
        # Window chosen clear of a clock change, so every local day has 48.
        idx = pd.date_range("2026-11-02", periods=30 * 48, freq="30min")
        y = pd.Series(1.0, index=idx)
        ev = _evaluate(y, [H.ZeroBaseline()], first_origin=12 * 48, nonnegative=True)
        pt = ev.scores["baseline_zero"].period_total
        expected = (24 - pt["issue_hour"]) * 2
        np.testing.assert_allclose(-pt["bias"], expected)


class TestLabelsAreNeverImputed:
    def test_unmeasured_targets_are_left_out_of_every_score(self):
        y = _series(days=40)
        gap = y.copy()
        gap.iloc[30 * 48:31 * 48] = np.nan
        full = _evaluate(y, [H.ZeroBaseline()])
        holed = _evaluate(gap, [H.ZeroBaseline()])
        assert holed.scores["baseline_zero"].n_points < full.scores["baseline_zero"].n_points
        assert np.isfinite(holed.scores["baseline_zero"].crps)
