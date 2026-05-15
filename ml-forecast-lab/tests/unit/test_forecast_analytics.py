"""
Regression suite for forecast analytics queries in HistoryDB.

Pins the behaviour of get_forecast_accuracy, get_forecast_coverage,
get_forecast_trajectory, get_forecast_stability so the categories of
bugs fixed in 2.22.0 – 2.22.2 stay fixed:

  - Model-filter contamination (accuracy / coverage / trajectory /
    stability mixing rotated-out models)
  - Increment-mode LAG diffs across actuals gaps
  - Trajectory cumulative-space mismatch (delta prediction vs raw
    cumulative actual on the same axis)
  - Daily-total stability inflated by partial-coverage issuances
  - Stability CV edge cases at mean ≈ 0

Each test seeds a small, fully-deterministic forecast_log + actuals
table, exercises one function, and asserts the property that would
fail under the old query. No HA, no training, no models — only the
SQL / aggregation layer is tested.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from ml_forecast_lab.db import HistoryDB


# ---------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------

# All tests use fixed past dates. Passing a generous window to every
# query makes the cutoff far enough in the past that wall-clock drift
# during CI doesn't flake tests. 10 years is plenty.
GENEROUS_WINDOW = 3650


def _log_cycle(db, experiment, issued_at, targets, predictions,
               model_name="lgb", upper=None, lower=None, model_version=None):
    """Thin wrapper over HistoryDB.log_forecast with list-of-datetimes."""
    return db.log_forecast(
        experiment=experiment,
        issued_at=issued_at,
        targets=targets,
        predictions=predictions,
        model_name=model_name,
        upper_bounds=upper,
        lower_bounds=lower,
        model_version=model_version,
    )


@pytest.fixture
def db(tmp_db):
    """Empty HistoryDB with forecast_log table pre-ensured."""
    h = HistoryDB(tmp_db)
    h.ensure_forecast_log_table()
    yield h
    h.close()


@pytest.fixture
def actuals_monotonic(db):
    """
    Cumulative-style actuals: value[i] = i at 30-min intervals for 4 days.
    Deltas are a constant 1.0 per interval, so increment mode should see
    zero model error on a model that always predicts 1.0.
    """
    table = db.safe_table_name("sensor.mono")
    idx = pd.date_range("2024-06-15 00:00", periods=48 * 4, freq="30min")
    db.store_history(table, pd.DataFrame({
        "ds": idx,
        "value": [float(i) for i in range(len(idx))],
    }))
    return table


@pytest.fixture
def actuals_with_gap(db):
    """
    Cumulative-style actuals with a 2-hour hole in the middle of Day 2.
    Forcing the increment-mode LAG to span a gap without the adjacency
    guard would produce a spurious 4-bin-worth delta at the first row
    after the hole.
    """
    table = db.safe_table_name("sensor.mono_gap")
    idx = pd.date_range("2024-06-15 00:00", periods=48 * 2, freq="30min")
    df = pd.DataFrame({
        "ds": idx, "value": [float(i) for i in range(len(idx))],
    })
    # Drop rows covering 04:00 → 05:30 on Day 2 (4 rows).
    mask = ~((df["ds"] >= "2024-06-16 04:00") &
             (df["ds"] <  "2024-06-16 06:00"))
    db.store_history(table, df[mask])
    return table


def _targets_30min(issued_at, horizon=6):
    """N 30-min targets after issued_at."""
    return [issued_at + timedelta(minutes=30 * (i + 1))
            for i in range(horizon)]


# ---------------------------------------------------------------------
# Accuracy / lead-time curve
# ---------------------------------------------------------------------

class TestAccuracyLeadTime:
    def test_raw_mode_returns_buckets_and_counts(self, db, actuals_monotonic):
        issued = datetime(2024, 6, 15, 8, 0)
        targets = _targets_30min(issued, 4)
        # _targets_30min starts at issued + 30min. For issued 08:00
        # the targets are 08:30/09:00/09:30/10:00 → seeded actual values
        # 17/18/19/20 (row indices).
        preds = [17.0, 18.0, 19.0, 20.0]
        _log_cycle(db, "exp", issued, targets, preds)
        r = db.get_forecast_accuracy(
            "exp", actuals_monotonic, max_age_days=GENEROUS_WINDOW,
            evaluation_mode="raw", model_name="lgb",
        )
        ltc = r["lead_time_curve"]
        assert ltc["lead_minutes"] == [30, 60, 90, 120]
        assert ltc["sample_count"] == [1, 1, 1, 1]
        # Perfect forecast → zero MAE at every lead.
        assert all(m == 0 for m in ltc["mae"])

    def test_increment_mode_drops_first_target_per_cycle(self, db, actuals_monotonic):
        # Within each issuance the LAG at the first target is NULL, so
        # increment mode cannot evaluate lead=30 (the first bucket).
        issued = datetime(2024, 6, 15, 8, 0)
        targets = _targets_30min(issued, 4)
        preds = [17.0, 18.0, 19.0, 20.0]  # match seeded actuals
        _log_cycle(db, "exp", issued, targets, preds)
        r = db.get_forecast_accuracy(
            "exp", actuals_monotonic, max_age_days=GENEROUS_WINDOW,
            evaluation_mode="increment", model_name="lgb",
        )
        ltc = r["lead_time_curve"]
        # First lead bucket should be absent; later buckets present.
        assert 30 not in ltc["lead_minutes"]
        assert 60 in ltc["lead_minutes"]
        # Predictions match actual deltas exactly (constant 1.0) so MAE ≈ 0.
        assert all(m < 1e-6 for m in ltc["mae"])

    def test_increment_mode_nulls_delta_on_actuals_gap(self, db, actuals_with_gap):
        """
        With a 2-hour hole in actuals, the LAG at the first post-gap bin
        spans 4 intervals. Without the adjacency guard, that bin's
        "delta" = value[06:00] − value[03:30] = 5, compared against a
        forecast's per-bin delta of 1 → MAE of 4 at that lead bucket.
        The guard nulls the row out; MAE stays ~0.
        """
        issued = datetime(2024, 6, 16, 3, 0)
        # Predict per-interval demand = 1.0 (matches the seeded monotonic
        # increments when no gap is in play). Horizon spans the gap.
        targets = _targets_30min(issued, 8)
        preds = [float(i) for i in range(
            targets[0].hour * 2 + targets[0].minute // 30 + 48,  # absolute offset
            targets[0].hour * 2 + targets[0].minute // 30 + 48 + 8,
        )]
        _log_cycle(db, "exp", issued, targets, preds)
        r = db.get_forecast_accuracy(
            "exp", actuals_with_gap, max_age_days=GENEROUS_WINDOW,
            evaluation_mode="increment", model_name="lgb",
        )
        ltc = r["lead_time_curve"]
        # Any non-zero MAE would mean a gap-spanning delta leaked through.
        for m in ltc["mae"]:
            assert m < 1e-6, f"Gap-spanning delta leaked into MAE: {ltc['mae']}"

    def test_model_filter_excludes_other_models(self, db, actuals_monotonic):
        issued = datetime(2024, 6, 15, 8, 0)
        targets = _targets_30min(issued, 4)
        _log_cycle(db, "exp", issued, targets, [17.0, 18.0, 19.0, 20.0], "lgb")
        # Tinker-era model with wildly different predictions
        _log_cycle(db, "exp", issued, targets, [999, 999, 999, 999], "tinker")
        r_all = db.get_forecast_accuracy(
            "exp", actuals_monotonic, max_age_days=GENEROUS_WINDOW,
            evaluation_mode="raw", model_name=None,
        )
        r_lgb = db.get_forecast_accuracy(
            "exp", actuals_monotonic, max_age_days=GENEROUS_WINDOW,
            evaluation_mode="raw", model_name="lgb",
        )
        # lgb sample counts should be 1-per-bucket, "all" should be 2.
        assert r_lgb["lead_time_curve"]["sample_count"] == [1, 1, 1, 1]
        assert r_all["lead_time_curve"]["sample_count"] == [2, 2, 2, 2]
        # lgb MAE ≈ 0, all-model MAE non-trivial (tinker poisons it).
        assert all(m < 1e-6 for m in r_lgb["lead_time_curve"]["mae"])
        assert max(r_all["lead_time_curve"]["mae"]) > 100

    def test_typical_interval_demand_mode_aware(self, db, actuals_monotonic):
        # Raw typical = mean |value| across the actuals in the window.
        # The monotonic 0..191 series has mean 95.5.
        # Increment typical = mean |delta| across actuals_grid, with
        # adjacency guard (every delta is +1) → 1.0.
        issued = datetime(2024, 6, 15, 8, 0)
        targets = _targets_30min(issued, 2)
        _log_cycle(db, "exp", issued, targets, [17.0, 18.0])
        r_raw = db.get_forecast_accuracy(
            "exp", actuals_monotonic, max_age_days=GENEROUS_WINDOW,
            evaluation_mode="raw", model_name="lgb",
        )
        r_inc = db.get_forecast_accuracy(
            "exp", actuals_monotonic, max_age_days=GENEROUS_WINDOW,
            evaluation_mode="increment", model_name="lgb",
        )
        assert r_raw["typical_interval_demand"] == pytest.approx(95.5, rel=1e-3)
        assert r_inc["typical_interval_demand"] == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------
# Revision improvement
# ---------------------------------------------------------------------

class TestRevisionImprovement:
    def test_requires_two_distinct_issuances_per_target(self, db, actuals_monotonic):
        # One issuance only → no revision info
        issued = datetime(2024, 6, 15, 8, 0)
        targets = _targets_30min(issued, 3)
        _log_cycle(db, "exp", issued, targets, [16.0, 17.0, 18.0])
        r = db.get_forecast_accuracy(
            "exp", actuals_monotonic, max_age_days=GENEROUS_WINDOW,
            evaluation_mode="raw", model_name="lgb",
        )
        assert r["revision_improvement"] == {}

    def test_identical_first_last_still_counted(self, db, actuals_monotonic):
        # Two issuances of the same target with IDENTICAL predictions
        # used to be filtered out by the old `first_pred != last_pred`
        # guard (v2.21.0 era). The count-based filter keeps them.
        t = datetime(2024, 6, 15, 9, 0)
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0), [t], [18.0])
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 30), [t], [18.0])
        r = db.get_forecast_accuracy(
            "exp", actuals_monotonic, max_age_days=GENEROUS_WINDOW,
            evaluation_mode="raw", model_name="lgb",
        )
        rev = r["revision_improvement"]
        assert rev.get("sample_count") == 1
        # Both forecasts are perfect → both MAEs = 0; improvement = 0%.
        assert rev["first_forecast_mae"] == 0
        assert rev["latest_forecast_mae"] == 0

    def test_improvement_sign(self, db, actuals_monotonic):
        # Same target, first forecast poor, latest forecast perfect →
        # improvement_pct should be +100.
        t = datetime(2024, 6, 15, 9, 0)
        actual_at_t = 18.0  # monotonic seed: row index 18
        _log_cycle(db, "exp", datetime(2024, 6, 15, 7, 0), [t], [actual_at_t + 5])  # first: off by 5
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 30), [t], [actual_at_t])     # last: perfect
        r = db.get_forecast_accuracy(
            "exp", actuals_monotonic, max_age_days=GENEROUS_WINDOW,
            evaluation_mode="raw", model_name="lgb",
        )
        rev = r["revision_improvement"]
        assert rev["first_forecast_mae"] == 5.0
        assert rev["latest_forecast_mae"] == 0.0
        assert rev["improvement_pct"] == 100.0


# ---------------------------------------------------------------------
# probe_forecast_rows — fast existence check used by the web layer to
# pick the narrowest (model, version) filter that has data before
# running the expensive accuracy query.
# ---------------------------------------------------------------------

class TestProbeForecastRows:
    def test_strict_filter_matches_only_logged_combo(self, db, actuals_monotonic):
        issued = datetime(2024, 6, 15, 8, 0)
        t = _targets_30min(issued, 1)
        _log_cycle(db, "exp", issued, t, [18.0],
                   model_name="lgb", model_version="v1")
        assert db.probe_forecast_rows(
            "exp", "lgb", "v1", max_age_days=GENEROUS_WINDOW,
        ) is True
        assert db.probe_forecast_rows(
            "exp", "lgb", "v2", max_age_days=GENEROUS_WINDOW,
        ) is False
        assert db.probe_forecast_rows(
            "exp", "xgb", None, max_age_days=GENEROUS_WINDOW,
        ) is False
        # Widening to all models + all versions picks the row up.
        assert db.probe_forecast_rows(
            "exp", None, None, max_age_days=GENEROUS_WINDOW,
        ) is True

    def test_cutoff_excludes_stale_cycles(self, db, actuals_monotonic):
        # Row older than the window should not register.
        issued = datetime.utcnow() - timedelta(days=60)
        t = [issued + timedelta(minutes=30)]
        _log_cycle(db, "exp", issued, t, [0.0], model_name="lgb")
        assert db.probe_forecast_rows("exp", "lgb", None, max_age_days=30) is False
        assert db.probe_forecast_rows(
            "exp", "lgb", None, max_age_days=GENEROUS_WINDOW,
        ) is True

    def test_future_only_targets_do_not_register(self, db, actuals_monotonic):
        # Reproduces the v2.34.0 retrain bug: a freshly-retrained cohort
        # has only future-targeting predictions, no actuals to join. The
        # probe must return False so the widening ladder falls back to
        # older versions of the same model.
        issued = datetime.utcnow() - timedelta(seconds=30)
        future_targets = [
            issued + timedelta(minutes=30 * (i + 1)) for i in range(48)
        ]
        _log_cycle(
            db, "exp", issued, future_targets, [1.0] * 48,
            model_name="lgb", model_version="v_new",
        )
        # Strict (model + version) — all rows target the future → False.
        assert db.probe_forecast_rows(
            "exp", "lgb", "v_new", max_age_days=30,
        ) is False
        # Adding an older cycle with elapsed targets to a different
        # version makes the model-only probe register, while strict on
        # the new version still returns False.
        old_issued = datetime.utcnow() - timedelta(days=2)
        _log_cycle(
            db, "exp", old_issued,
            [old_issued + timedelta(minutes=30)],
            [1.0], model_name="lgb", model_version="v_old",
        )
        assert db.probe_forecast_rows(
            "exp", "lgb", "v_new", max_age_days=30,
        ) is False
        assert db.probe_forecast_rows(
            "exp", "lgb", None, max_age_days=30,
        ) is True


# ---------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------

class TestCoverage:
    def test_ignores_rows_without_bands(self, db, actuals_monotonic):
        # Legacy point-only forecasts (upper/lower = None) must be
        # excluded so coverage isn't pulled toward 0.
        t = _targets_30min(datetime(2024, 6, 15, 8, 0), 2)
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0), t, [16.0, 17.0])  # no bands
        cov = db.get_forecast_coverage(
            "exp", actuals_monotonic, max_age_days=GENEROUS_WINDOW,
            model_name="lgb",
        )
        assert cov["overall"] == {}

    def test_overall_coverage_in_band(self, db, actuals_monotonic):
        # Band that straddles actual → 100% coverage.
        t = _targets_30min(datetime(2024, 6, 15, 8, 0), 3)
        preds = [16.0, 17.0, 18.0]
        _log_cycle(
            db, "exp", datetime(2024, 6, 15, 8, 0),
            t, preds,
            upper=[p + 1 for p in preds],
            lower=[p - 1 for p in preds],
        )
        cov = db.get_forecast_coverage(
            "exp", actuals_monotonic, max_age_days=GENEROUS_WINDOW,
            model_name="lgb",
        )
        assert cov["overall"]["n"] == 3
        assert cov["overall"]["coverage"] == 1.0

    def test_model_filter_affects_counts(self, db, actuals_monotonic):
        # v2.34.0: each model_name produces its own coverage value.
        # When the caller passes model_name=None, the query no longer
        # POOLS coverage across models (which produced a number
        # that didn't correspond to any actually-published band).
        # Instead, the SQL partitions per cohort and the result
        # picks the single dominant cohort by sample count then
        # by most recent model_version. With both cohorts at n=2 and
        # no model_version pinned, the LIMIT 1 picks ONE of the two
        # — either is correct semantically; what matters is that
        # the value is one of the per-cohort coverages and never
        # something like 0.5 (the pre-fix pooled mean).
        t = _targets_30min(datetime(2024, 6, 15, 8, 0), 2)
        # lgb: band covers actual
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0),
                   t, [16.0, 17.0],
                   upper=[17.0, 18.0], lower=[15.0, 16.0], model_name="lgb")
        # tinker: band misses actual
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0),
                   t, [100.0, 100.0],
                   upper=[101.0, 101.0], lower=[99.0, 99.0], model_name="tinker")
        lgb = db.get_forecast_coverage(
            "exp", actuals_monotonic, max_age_days=GENEROUS_WINDOW, model_name="lgb")
        tinker = db.get_forecast_coverage(
            "exp", actuals_monotonic, max_age_days=GENEROUS_WINDOW, model_name="tinker")
        mixed = db.get_forecast_coverage(
            "exp", actuals_monotonic, max_age_days=GENEROUS_WINDOW, model_name=None)
        assert lgb["overall"]["coverage"] == 1.0
        assert tinker["overall"]["coverage"] == 0.0
        # Mixed result is the dominant cohort's value — never the pooled mean.
        assert mixed["overall"]["coverage"] in (0.0, 1.0)

    def test_mixed_filter_picks_dominant_cohort_by_count(self, db, actuals_monotonic):
        # When two cohorts have unequal sample counts, the larger
        # cohort wins. This is the v2.34.0 invariant that prevents
        # cross-cohort pooling silently producing meaningless numbers.
        t = _targets_30min(datetime(2024, 6, 15, 8, 0), 3)
        # lgb: 3 rows, band covers
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0),
                   t, [16.0, 17.0, 18.0],
                   upper=[17.0, 18.0, 19.0], lower=[15.0, 16.0, 17.0],
                   model_name="lgb")
        # tinker: 2 rows, band misses
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0),
                   t[:2], [100.0, 100.0],
                   upper=[101.0, 101.0], lower=[99.0, 99.0],
                   model_name="tinker")
        mixed = db.get_forecast_coverage(
            "exp", actuals_monotonic, max_age_days=GENEROUS_WINDOW, model_name=None)
        # lgb has more samples → wins. Coverage = lgb's 1.0, not pooled.
        assert mixed["overall"]["coverage"] == 1.0
        assert mixed["overall"]["n"] == 3


# ---------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------

class TestTrajectory:
    def test_raw_space_for_instantaneous_source(self, db, actuals_monotonic):
        t = datetime(2024, 6, 15, 9, 0)  # actual = 18.0 on the seeded series
        _log_cycle(db, "exp", datetime(2024, 6, 15, 7, 0), [t], [17.5])
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0), [t], [18.0])
        r = db.get_forecast_trajectory(
            "exp", actuals_monotonic,
            interval_minutes=30, max_age_days=GENEROUS_WINDOW,
            source_is_cumulative=False, model_name="lgb",
        )
        assert r["actual_space"] == "raw"
        assert r["actual"] == 18.0

    def test_delta_space_for_cumulative_source(self, db, actuals_monotonic):
        # Same target; cumulative source → actual should be the
        # per-interval delta (= 1.0, since seeded with +1/bin).
        t = datetime(2024, 6, 15, 9, 0)
        _log_cycle(db, "exp", datetime(2024, 6, 15, 7, 0), [t], [1.1])
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0), [t], [1.0])
        r = db.get_forecast_trajectory(
            "exp", actuals_monotonic,
            interval_minutes=30, max_age_days=GENEROUS_WINDOW,
            source_is_cumulative=True, model_name="lgb",
        )
        assert r["actual_space"] == "delta"
        assert r["actual"] == pytest.approx(1.0, abs=1e-6)
        # target_meta.max_abs_error is also in the delta space now.
        m = next(m for m in r["target_meta"] if m["target_dt"].startswith("2024-06-15 09"))
        assert m["max_abs_error"] == pytest.approx(0.1, abs=1e-6)

    def test_delta_space_gap_nulls_actual(self, db, actuals_with_gap):
        # A target right after the 04:00-05:30 hole has no adjacent
        # predecessor → actual is null (matches the gap guard on the
        # accuracy increment query).
        t = datetime(2024, 6, 16, 6, 0)
        _log_cycle(db, "exp", datetime(2024, 6, 16, 3, 0), [t], [1.0])
        _log_cycle(db, "exp", datetime(2024, 6, 16, 3, 30), [t], [1.0])
        r = db.get_forecast_trajectory(
            "exp", actuals_with_gap,
            interval_minutes=30, max_age_days=GENEROUS_WINDOW,
            source_is_cumulative=True, model_name="lgb",
        )
        # The target shouldn't appear among available_targets because
        # av.value IS NULL fails the candidate-query join filter.
        assert not any(t.strftime("%Y-%m-%d %H:%M:%S") in x
                       for x in r.get("available_targets", []))

    def test_model_filter_excludes_other_models(self, db, actuals_monotonic):
        t = datetime(2024, 6, 15, 9, 0)
        _log_cycle(db, "exp", datetime(2024, 6, 15, 7, 0), [t], [17.5], model_name="lgb")
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0), [t], [17.5], model_name="lgb")
        _log_cycle(db, "exp", datetime(2024, 6, 15, 7, 0), [t], [999], model_name="tinker")
        r_lgb = db.get_forecast_trajectory(
            "exp", actuals_monotonic,
            interval_minutes=30, max_age_days=GENEROUS_WINDOW,
            source_is_cumulative=False, model_name="lgb",
        )
        # Only the lgb rows surface in forecasts[]
        assert all(f["model_name"] == "lgb" for f in r_lgb["forecasts"])

    def test_target_meta_sort_by_worst(self, db, actuals_monotonic):
        # Two targets, one much worse than the other → the UI sort
        # helper relies on target_meta.max_abs_error being populated
        # and larger for the worse target.
        t_good = datetime(2024, 6, 15, 8, 30)   # actual = 17
        t_bad  = datetime(2024, 6, 15, 10, 0)    # actual = 20
        for iss_h in (7, 7.5):
            iss = datetime(2024, 6, 15, int(iss_h),
                           30 if iss_h % 1 else 0)
            _log_cycle(db, "exp", iss,
                       [t_good, t_bad], [17.0, 5.0])  # bad pred on t_bad
        r = db.get_forecast_trajectory(
            "exp", actuals_monotonic,
            interval_minutes=30, max_age_days=GENEROUS_WINDOW,
            source_is_cumulative=False, model_name="lgb",
        )
        by_t = {m["target_dt"]: m for m in r["target_meta"]}
        good = by_t[t_good.strftime("%Y-%m-%d %H:%M:%S")]
        bad = by_t[t_bad.strftime("%Y-%m-%d %H:%M:%S")]
        assert good["max_abs_error"] == 0
        assert bad["max_abs_error"] == pytest.approx(15.0, abs=1e-6)


# ---------------------------------------------------------------------
# Evolution (fan chart actuals)
# ---------------------------------------------------------------------

class TestEvolutionActuals:
    def test_cumulative_source_returns_deltas(self, db, actuals_monotonic):
        # Predictions for a cumulative source are stored in delta space.
        # If the evolution endpoint returns raw cumulative actuals, the
        # fan chart plots a 0 → N climb against tiny delta predictions —
        # the bug the user reported on v2.34.1. With source_is_cumulative
        # the actuals must arrive pre-diffed.
        issued = datetime(2024, 6, 15, 8, 0)
        targets = _targets_30min(issued, 4)
        _log_cycle(db, "exp", issued, targets, [1.0, 1.0, 1.0, 1.0])
        r = db.get_forecast_evolution(
            "exp", actuals_monotonic,
            n_cycles=12, interval_minutes=30,
            source_is_cumulative=True,
        )
        vals = r["actuals"]["values"]
        # Monotonic fixture increments by 1 per 30-min bin → delta = 1.0.
        assert all(abs(v - 1.0) < 1e-9 for v in vals)

    def test_raw_source_returns_raw_values(self, db, actuals_monotonic):
        # Non-cumulative source: actuals should pass through untouched.
        issued = datetime(2024, 6, 15, 8, 0)
        targets = _targets_30min(issued, 4)
        _log_cycle(db, "exp", issued, targets, [16.0, 17.0, 18.0, 19.0])
        r = db.get_forecast_evolution(
            "exp", actuals_monotonic,
            n_cycles=12, interval_minutes=30,
            source_is_cumulative=False,
        )
        vals = r["actuals"]["values"]
        # Monotonic seeded value[i] = i. Targets 08:30..10:00 → row
        # indices 17..20.
        assert vals[:4] == [17.0, 18.0, 19.0, 20.0]

    def test_cumulative_source_clamps_negative_resets(self, db):
        # Daily-reset sensor: cumulative climbs through the day then
        # snaps back to 0 at midnight. Naive diff would emit a large
        # negative spike at the reset; the clamp should pin that to 0.
        table = db.safe_table_name("sensor.reset")
        idx = pd.date_range("2024-06-15 22:00", periods=6, freq="30min")
        # Values: 5, 6, 7, 0, 1, 2  (reset at index 3)
        db.store_history(table, pd.DataFrame({
            "ds": idx,
            "value": [5.0, 6.0, 7.0, 0.0, 1.0, 2.0],
        }))
        issued = datetime(2024, 6, 15, 21, 30)
        targets = [idx[i].to_pydatetime() for i in range(6)]
        _log_cycle(db, "exp", issued, targets, [1.0] * 6)
        r = db.get_forecast_evolution(
            "exp", table,
            n_cycles=12, interval_minutes=30,
            source_is_cumulative=True,
        )
        vals = r["actuals"]["values"]
        # First entry is NULL (no prior bin) and excluded. Then deltas:
        # 1.0, 1.0, max(0, -7) = 0, 1.0, 1.0
        assert vals == [1.0, 1.0, 0.0, 1.0, 1.0]


# ---------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------

class TestStability:
    def test_per_timestep_cv_zero_when_flat(self, db):
        # Two cycles that predict the same value for the same target
        # → std = 0, cv = 0, not skipped.
        t = datetime(2024, 6, 15, 9, 0)
        _log_cycle(db, "exp", datetime(2024, 6, 15, 7, 0), [t], [5.0])
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0), [t], [5.0])
        s = db.get_forecast_stability(
            "exp", max_age_days=GENEROUS_WINDOW, source_is_cumulative=False,
            model_name="lgb",
        )
        assert s["summary"]["steps_analysed"] == 1
        assert s["per_timestep"]["cv_pct"] == [0.0]

    def test_per_timestep_skips_zero_mean_nonzero_std(self, db):
        # Two cycles predicting ±1 around 0 for the same target — the
        # mean is ~0, std is ~1, and CV is ill-defined. Old code
        # reported CV=0 (falsely "stable"); new code skips that row.
        t = datetime(2024, 6, 15, 9, 0)
        _log_cycle(db, "exp", datetime(2024, 6, 15, 7, 0), [t], [+1.0])
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0), [t], [-1.0])
        s = db.get_forecast_stability(
            "exp", max_age_days=GENEROUS_WINDOW, source_is_cumulative=False,
            model_name="lgb",
        )
        # The target is dropped from per_timestep rather than reported
        # as a false zero-CV entry.
        assert s["summary"]["steps_analysed"] == 0
        assert s["per_timestep"]["target_dt"] == []

    def test_model_filter_excludes_other_models(self, db):
        # v2.34.0: when model_name=None the query no longer pools
        # cross-model cycles into a single inflated CV. The SQL
        # partitions by cohort and picks ONE winner per target_dt
        # (here both cohorts have n_cycles=2 so tie-break runs by
        # most recent model_version — both NULL, so SQLite's row
        # ordering decides). What matters is the result equals ONE
        # of the per-model CVs, never the pre-fix pooled value of
        # >20% that mixed lgb's tight predictions with tinker's wild
        # ones.
        t = datetime(2024, 6, 15, 9, 0)
        _log_cycle(db, "exp", datetime(2024, 6, 15, 7, 0), [t], [5.0], "lgb")
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0), [t], [5.1], "lgb")
        _log_cycle(db, "exp", datetime(2024, 6, 15, 7, 0), [t], [100], "tinker")
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0), [t], [200], "tinker")
        mixed = db.get_forecast_stability(
            "exp", max_age_days=GENEROUS_WINDOW, source_is_cumulative=False,
            model_name=None,
        )
        lgb = db.get_forecast_stability(
            "exp", max_age_days=GENEROUS_WINDOW, source_is_cumulative=False,
            model_name="lgb",
        )
        tinker = db.get_forecast_stability(
            "exp", max_age_days=GENEROUS_WINDOW, source_is_cumulative=False,
            model_name="tinker",
        )
        # Per-cohort values still bracket the truth.
        assert lgb["summary"]["median_step_cv_pct"] < 5
        assert tinker["summary"]["median_step_cv_pct"] > 20
        # Mixed: must equal ONE cohort's value, never the pooled mean.
        mixed_cv = mixed["summary"]["median_step_cv_pct"]
        assert mixed_cv == pytest.approx(lgb["summary"]["median_step_cv_pct"]) \
            or mixed_cv == pytest.approx(tinker["summary"]["median_step_cv_pct"])

    def test_mixed_filter_picks_dominant_cohort_by_count(self, db):
        # v2.34.0 invariant: when cohorts differ in cycle count, the
        # larger cohort wins. Prevents cross-cohort pooling silently
        # producing a misleading "run-to-run swing".
        t = datetime(2024, 6, 15, 9, 0)
        # lgb has 3 cycles (more data) — should win
        _log_cycle(db, "exp", datetime(2024, 6, 15, 6, 0), [t], [5.0], "lgb")
        _log_cycle(db, "exp", datetime(2024, 6, 15, 7, 0), [t], [5.1], "lgb")
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0), [t], [5.2], "lgb")
        # tinker has 2 cycles
        _log_cycle(db, "exp", datetime(2024, 6, 15, 7, 0), [t], [100], "tinker")
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0), [t], [200], "tinker")
        mixed = db.get_forecast_stability(
            "exp", max_age_days=GENEROUS_WINDOW, source_is_cumulative=False,
            model_name=None,
        )
        # Should match lgb's CV (low), not the pooled mean (huge).
        assert mixed["summary"]["median_step_cv_pct"] < 5

    def test_daily_total_gates_on_full_coverage(self, db):
        """
        This is the v2.22.2 fix: a partial-coverage issuance must not
        be pooled with full-coverage issuances when computing daily-
        total spread.

        Three cycles predicting the same "day D":
          - Cycle A: full 48 bins at 1.00/bin → total 48.00
          - Cycle C: full 48 bins at 1.05/bin → total 50.40
          - Cycle B: partial 23 bins at 1.00/bin → total 23.00

        Without gating, std across {48, 50.4, 23} is huge (~12 units,
        ~30% CV). With gating, only {48, 50.4} compare → CV ~2.5%.
        """
        day_targets_full = [
            datetime(2024, 6, 15, 0, 30) + timedelta(minutes=30 * i)
            for i in range(48)
        ]
        _log_cycle(db, "exp", datetime(2024, 6, 14, 20, 0),
                   day_targets_full, [1.00] * 48, "lgb")
        _log_cycle(db, "exp", datetime(2024, 6, 14,  5, 0),
                   day_targets_full, [1.05] * 48, "lgb")
        day_targets_partial = [t for t in day_targets_full
                               if (t.hour, t.minute) >= (12, 30)]
        _log_cycle(db, "exp", datetime(2024, 6, 15, 12, 0),
                   day_targets_partial,
                   [1.00] * len(day_targets_partial), "lgb")

        s = db.get_forecast_stability(
            "exp", max_age_days=GENEROUS_WINDOW,
            source_is_cumulative=True, model_name="lgb",
        )
        day_row = next(d for d in s["daily_totals"] if d["day"] == "2024-06-15")
        # Only the two full-coverage cycles count.
        assert day_row["n_cycles"] == 2
        # CV of {48.00, 50.40} is 2.44%.
        assert day_row["cv_pct"] == pytest.approx(2.44, abs=0.05)

    def test_cleanup_removes_pre_retrain_rows(self, db):
        # Verifies the promotion-time cleanup path: rows issued before a
        # retrain timestamp are removed so the stability metric doesn't
        # pool predictions from two weight regimes under the same
        # model_name.
        t = datetime(2024, 6, 15, 9, 0)
        # "Old weights" era — wildly different predictions
        _log_cycle(db, "exp", datetime(2024, 6, 15, 6, 0), [t], [100.0], "lgb")
        _log_cycle(db, "exp", datetime(2024, 6, 15, 7, 0), [t], [200.0], "lgb")
        # Retrain boundary
        retrain_at = datetime(2024, 6, 15, 8, 0)
        # "New weights" era — tight predictions
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 30), [t], [5.0], "lgb")
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 45), [t], [5.1], "lgb")

        before = db.get_forecast_stability(
            "exp", max_age_days=GENEROUS_WINDOW, source_is_cumulative=False,
            model_name="lgb",
        )
        assert before["summary"]["median_step_cv_pct"] > 50

        deleted = db.cleanup_forecast_log("exp", retrain_at)
        assert deleted == 2

        after = db.get_forecast_stability(
            "exp", max_age_days=GENEROUS_WINDOW, source_is_cumulative=False,
            model_name="lgb",
        )
        assert after["summary"]["median_step_cv_pct"] < 5

    def test_daily_total_drops_day_with_single_full_coverage(self, db):
        # If only one cycle fully covers a day (others partial),
        # HAVING n_cycles >= 2 should drop the day entirely — fail-
        # closed rather than report a spread of zero against nothing.
        day_targets_full = [
            datetime(2024, 6, 15, 0, 30) + timedelta(minutes=30 * i)
            for i in range(48)
        ]
        _log_cycle(db, "exp", datetime(2024, 6, 14, 20, 0),
                   day_targets_full, [1.0] * 48, "lgb")
        partial = day_targets_full[20:]
        _log_cycle(db, "exp", datetime(2024, 6, 15, 10, 0),
                   partial, [1.0] * len(partial), "lgb")
        s = db.get_forecast_stability(
            "exp", max_age_days=GENEROUS_WINDOW,
            source_is_cumulative=True, model_name="lgb",
        )
        assert all(d["day"] != "2024-06-15" for d in s["daily_totals"])


# ---------------------------------------------------------------------
# Model version filtering (v2.24.0)
# ---------------------------------------------------------------------

class TestModelVersion:
    """
    `model_version` segregates weight regimes of a model that keeps the
    same name across retrains. Without it, the stability / accuracy /
    coverage / trajectory queries silently pool predictions from v1 and
    v2 under a shared model_name, which is the "I retrained and now
    stability looks terrible" pattern.
    """

    def test_migration_adds_column_to_legacy_table(self, tmp_db):
        # Simulate a pre-2.24 DB: create the table without model_version.
        import sqlite3 as _sql
        conn = _sql.connect(tmp_db)
        conn.execute("""
            CREATE TABLE forecast_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment TEXT NOT NULL,
                model_name TEXT NOT NULL,
                issued_at TEXT NOT NULL,
                target_dt TEXT NOT NULL,
                lead_minutes INTEGER NOT NULL,
                predicted REAL NOT NULL,
                forecast_type TEXT NOT NULL DEFAULT 'cached',
                upper REAL,
                lower REAL
            )
        """)
        conn.execute(
            "INSERT INTO forecast_log "
            "(experiment, model_name, issued_at, target_dt, lead_minutes, predicted) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("exp", "lgb", "2024-06-15 08:00:00", "2024-06-15 08:30:00", 30, 1.0),
        )
        conn.commit()
        conn.close()

        # Open through HistoryDB — ensure_forecast_log_table should
        # ALTER the column in, preserving the legacy row as NULL.
        h = HistoryDB(tmp_db)
        h.ensure_forecast_log_table()
        cur = h.conn.cursor()
        cur.execute("PRAGMA table_info(forecast_log)")
        cols = {row[1] for row in cur.fetchall()}
        assert "model_version" in cols
        cur.execute("SELECT model_version FROM forecast_log")
        assert cur.fetchone()[0] is None  # legacy row: null version
        h.close()

    def test_log_forecast_stamps_version(self, db):
        t = [datetime(2024, 6, 15, 9, 0)]
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0),
                   t, [1.0], "lgb", model_version="2024-06-15T07:00:00Z")
        cur = db.conn.cursor()
        cur.execute("SELECT model_version FROM forecast_log")
        assert cur.fetchone()[0] == "2024-06-15T07:00:00Z"

    def test_version_filter_segregates_weight_regimes(self, db):
        """
        Same model_name, two weight regimes: v1 predicts ~100, v2
        predicts ~5.

        v2.34.0: The SQL now self-protects against cross-cohort
        pooling. Even with `model_version=None` (which previously
        pooled v1 and v2 into a single astronomical CV), the result
        is now one cohort's CV — never the pooled mean. v2-only
        still collapses to the real single-regime disagreement.
        """
        t = datetime(2024, 6, 15, 9, 0)
        # v1: old weights, noisy-around-100 predictions
        _log_cycle(db, "exp", datetime(2024, 6, 15, 5, 0), [t], [100.0],
                   "lgb", model_version="v1")
        _log_cycle(db, "exp", datetime(2024, 6, 15, 6, 0), [t], [110.0],
                   "lgb", model_version="v1")
        # v2: new weights, tight-around-5 predictions
        _log_cycle(db, "exp", datetime(2024, 6, 15, 7, 0), [t], [5.0],
                   "lgb", model_version="v2")
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0), [t], [5.1],
                   "lgb", model_version="v2")

        mixed = db.get_forecast_stability(
            "exp", max_age_days=GENEROUS_WINDOW, source_is_cumulative=False,
            model_name="lgb", model_version=None,
        )
        v2_only = db.get_forecast_stability(
            "exp", max_age_days=GENEROUS_WINDOW, source_is_cumulative=False,
            model_name="lgb", model_version="v2",
        )
        v1_only = db.get_forecast_stability(
            "exp", max_age_days=GENEROUS_WINDOW, source_is_cumulative=False,
            model_name="lgb", model_version="v1",
        )
        # Per-version values still bracket the true regimes.
        assert v2_only["summary"]["median_step_cv_pct"] < 5
        assert v1_only["summary"]["median_step_cv_pct"] > 4   # ~5% of 105
        # Mixed: cohort partitioning forces selection of ONE regime.
        # With both at n_cycles=2 the tie-break picks the newer
        # model_version (v2 > v1), so mixed should equal v2_only.
        mixed_cv = mixed["summary"]["median_step_cv_pct"]
        assert mixed_cv == pytest.approx(v2_only["summary"]["median_step_cv_pct"])
        # Critically: never the pre-fix pooled value of >50.
        assert mixed_cv < 10

    def test_version_filter_excludes_null_legacy_rows(self, db):
        """
        Legacy rows (no version) should NOT pool into a version-
        filtered query — that's the whole point of the column.
        """
        t = datetime(2024, 6, 15, 9, 0)
        # Two legacy rows (no version)
        _log_cycle(db, "exp", datetime(2024, 6, 15, 5, 0), [t], [100.0], "lgb")
        _log_cycle(db, "exp", datetime(2024, 6, 15, 6, 0), [t], [200.0], "lgb")
        # Two versioned rows
        _log_cycle(db, "exp", datetime(2024, 6, 15, 7, 0), [t], [5.0],
                   "lgb", model_version="v2")
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0), [t], [5.1],
                   "lgb", model_version="v2")

        v2_only = db.get_forecast_stability(
            "exp", max_age_days=GENEROUS_WINDOW, source_is_cumulative=False,
            model_name="lgb", model_version="v2",
        )
        # Only 2 v2 cycles should contribute (NULL legacy rows excluded).
        assert v2_only["summary"]["total_cycles"] == 2
        assert v2_only["summary"]["median_step_cv_pct"] < 5

    def test_ha_local_day_bucketing_shifts_day_labels(self, db):
        """
        `day_offset_hours` shifts target_dt by the HA-local offset
        before taking the YYYY-MM-DD prefix, so daily-total buckets
        align with the HA-local day (when the `_today` sensor
        actually resets) rather than UTC. In BST (UTC+1) a day spans
        UTC 23:00 prev-day → UTC 22:30 this-day.

        Test: 48 half-hour targets covering BST Apr 16 exactly. With
        offset=+1h they sit in a single "2024-04-16" bucket — the
        physically-correct one. With offset=0 they straddle two UTC
        days ("2024-04-15" and "2024-04-16") because the first two
        bins spill into UTC Apr 15. The semantic difference is what
        the fix addresses — a viewer in one timezone looking at a sensor
        hosted in another (e.g. California viewer, UK-hosted HA) now sees
        "Apr 16" meaning BST Apr 16.
        """
        # BST Apr 16 00:00→23:30 = UTC Apr 15 23:00 → UTC Apr 16 22:30
        day_targets = [
            datetime(2024, 4, 15, 23, 0) + timedelta(minutes=30 * i)
            for i in range(48)
        ]
        _log_cycle(db, "exp", datetime(2024, 4, 15, 12, 0),
                   day_targets, [1.00] * 48, "lgb")
        _log_cycle(db, "exp", datetime(2024, 4, 15, 13, 0),
                   day_targets, [1.10] * 48, "lgb")

        naive = db.get_forecast_stability(
            "exp", max_age_days=GENEROUS_WINDOW,
            source_is_cumulative=True, model_name="lgb",
        )
        naive_days = {d["day"] for d in naive["daily_totals"]}
        # Naive UTC bucketing splits the physical-BST-day-16 across
        # "2024-04-15" (2 bins) and "2024-04-16" (46 bins).
        assert "2024-04-15" in naive_days
        assert "2024-04-16" in naive_days

        shifted = db.get_forecast_stability(
            "exp", max_age_days=GENEROUS_WINDOW,
            source_is_cumulative=True, model_name="lgb",
            day_offset_hours=1.0,
        )
        shifted_days = {d["day"] for d in shifted["daily_totals"]}
        # BST bucketing unifies the physical day into one bucket and
        # eliminates the spurious Apr-15 tail.
        assert shifted_days == {"2024-04-16"}
        apr16 = next(d for d in shifted["daily_totals"] if d["day"] == "2024-04-16")
        assert apr16["n_cycles"] == 2
        # 48-bin daily total: {48.00, 52.80} → CV ≈ 4.76%.
        assert apr16["cv_pct"] == pytest.approx(4.76, abs=0.1)

    def test_ha_local_day_bucketing_ignored_when_offset_zero(self, db):
        """Offset=0 or None should be a no-op — preserves the
        existing UTC-day bucketing on deployments without HA TZ info."""
        day_targets = [
            datetime(2024, 6, 15, 0, 30) + timedelta(minutes=30 * i)
            for i in range(48)
        ]
        _log_cycle(db, "exp", datetime(2024, 6, 14, 20, 0),
                   day_targets, [1.0] * 48, "lgb")
        _log_cycle(db, "exp", datetime(2024, 6, 14, 21, 0),
                   day_targets, [1.05] * 48, "lgb")
        ref = db.get_forecast_stability(
            "exp", max_age_days=GENEROUS_WINDOW,
            source_is_cumulative=True, model_name="lgb",
        )
        zero = db.get_forecast_stability(
            "exp", max_age_days=GENEROUS_WINDOW,
            source_is_cumulative=True, model_name="lgb",
            day_offset_hours=0.0,
        )
        assert [d["day"] for d in ref["daily_totals"]] == \
               [d["day"] for d in zero["daily_totals"]]
        assert [d["cv_pct"] for d in ref["daily_totals"]] == \
               [d["cv_pct"] for d in zero["daily_totals"]]

    def test_accuracy_coverage_trajectory_all_honour_version(
        self, db, actuals_monotonic
    ):
        # Verify the filter propagates through all four analytics
        # queries, not just stability. Setup: v1 predictions far from
        # actual; v2 predictions spot-on.
        issued1 = datetime(2024, 6, 15, 7, 0)
        issued2 = datetime(2024, 6, 15, 7, 30)
        targets = [datetime(2024, 6, 15, 8, 30),
                   datetime(2024, 6, 15, 9, 0)]  # actuals 17.0, 18.0
        # v1: wildly off
        _log_cycle(db, "exp", issued1, targets, [1.0, 1.0], "lgb",
                   upper=[2.0, 2.0], lower=[0.0, 0.0], model_version="v1")
        _log_cycle(db, "exp", issued2, targets, [1.0, 1.0], "lgb",
                   upper=[2.0, 2.0], lower=[0.0, 0.0], model_version="v1")
        # v2: perfect
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 0),
                   targets, [17.0, 18.0], "lgb",
                   upper=[18.0, 19.0], lower=[16.0, 17.0], model_version="v2")
        _log_cycle(db, "exp", datetime(2024, 6, 15, 8, 15),
                   targets, [17.0, 18.0], "lgb",
                   upper=[18.0, 19.0], lower=[16.0, 17.0], model_version="v2")

        acc_v2 = db.get_forecast_accuracy(
            "exp", actuals_monotonic, max_age_days=GENEROUS_WINDOW,
            evaluation_mode="raw", model_name="lgb", model_version="v2",
        )
        # v2 is perfect → MAE ≈ 0 at every bucket.
        assert all(m < 1e-6 for m in acc_v2["lead_time_curve"]["mae"])

        cov_v2 = db.get_forecast_coverage(
            "exp", actuals_monotonic, max_age_days=GENEROUS_WINDOW,
            model_name="lgb", model_version="v2",
        )
        # v2 bands cover the actual; v1 bands don't.
        assert cov_v2["overall"]["coverage"] == 1.0
        cov_v1 = db.get_forecast_coverage(
            "exp", actuals_monotonic, max_age_days=GENEROUS_WINDOW,
            model_name="lgb", model_version="v1",
        )
        assert cov_v1["overall"]["coverage"] == 0.0

        traj_v2 = db.get_forecast_trajectory(
            "exp", actuals_monotonic,
            interval_minutes=30, max_age_days=GENEROUS_WINDOW,
            source_is_cumulative=False,
            model_name="lgb", model_version="v2",
        )
        # Only v2 forecast rows should feed the chosen target.
        assert all(f["predicted"] in (17.0, 18.0) for f in traj_v2["forecasts"])
