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
               model_name="lgb", upper=None, lower=None):
    """Thin wrapper over HistoryDB.log_forecast with list-of-datetimes."""
    return db.log_forecast(
        experiment=experiment,
        issued_at=issued_at,
        targets=targets,
        predictions=predictions,
        model_name=model_name,
        upper_bounds=upper,
        lower_bounds=lower,
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
        assert mixed["overall"]["coverage"] == 0.5


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
        # Same target, two distinct cycles each. lgb predicts tight;
        # tinker predicts wildly. Stability should switch from
        # artificially high (mixed) to low (filtered).
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
        assert mixed["summary"]["median_step_cv_pct"] > 20
        assert lgb["summary"]["median_step_cv_pct"] < 5

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
