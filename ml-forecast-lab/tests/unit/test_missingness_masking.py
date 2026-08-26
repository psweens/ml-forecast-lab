"""Missingness: masking, indicators and pipeline ordering (v2.51.0).

Three defects, all fixed here, all pinned below.

1. **Rows were dropped before features were built.** `_fetch_and_preprocess`
   ended with `result.dropna()` and every `build_features` call ran after
   it. Lags are positional — `target.shift(lag)` — so any gap beyond
   `gap_max_minutes` punched a hole in the index first and `y_lag_48`
   silently stopped meaning "24 hours ago". This needed no covariate to
   trigger; the default `gap_handling='interpolate'` leaves gaps over 90
   minutes as NaN, so it was live for any experiment with an outage.

2. **Masking was not expressible.** `resample_to_grid` has offered
   `gap_handling='mask'` for releases, but `np.nan_to_num(X, nan=0.0)` sat
   at every feature-matrix boundary, so a masked gap arrived at the model
   as 0.0 — worse than the fill, because zero is physically meaningful and
   at the extreme of most sensor ranges.

3. **Unbounded filling fabricated data.** Covariate alignment ended with
   `.ffill().bfill()`, propagating the oldest available value backwards
   across the whole window.

The contract now: a missing **label** means the row is not a supervised
sample and is excluded, never imputed. A missing **feature** is masked,
flagged and causally imputed, so its row survives.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.config import AppConfig, CovariateCfg, ExperimentCfg
from ml_forecast_lab.covariates import CovariateResolver
from ml_forecast_lab.db import HistoryDB
from ml_forecast_lab.features import (
    MISSING_SUFFIX,
    TARGET_MISSING_COLUMN,
    build_features,
    default_lag_windows,
    feature_warmup_rows,
    neural_covariate_columns,
    rebuild_fold_features,
)
from ml_forecast_lab.main import (
    MLForecastLabApp,
    _align_covariate_to_grid,
    _inference_indicator_map,
    _nan_to_num_guarded,
    _seed_lag_buffer,
    _supervised_frame,
)
from ml_forecast_lab import features as features_mod
from ml_forecast_lab import preprocessing as preprocessing_mod
from ml_forecast_lab.preprocessing import causal_impute, resolve_missingness

INTERVAL = 30
FREQ = "30min"
STEPS_PER_DAY = 48


# ---------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------

def _grid(n: int = 48 * 20, start: str = "2026-05-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=FREQ)


def _target(index: pd.DatetimeIndex, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    base = 20 + 10 * np.sin(np.arange(len(index)) / 12.0)
    return pd.Series(base + rng.normal(0, 0.2, len(index)), index=index)


def _assemble(df: pd.DataFrame, **kwargs):
    """`build_features` + the join every call site performs, undropped."""
    feats = build_features(df, "y", interval_minutes=INTERVAL, **kwargs)
    combined = feats.copy()
    combined["target"] = df["y"]
    for col in [c for c in df.columns if c != "y"]:
        combined[col] = df[col]
    return combined


def _resolved(df: pd.DataFrame, **kwargs):
    combined = _assemble(df)
    warmup = feature_warmup_rows(
        len(df), INTERVAL, ghi_gated="clear_sky_ghi" in df.columns,
    )
    return resolve_missingness(combined, "target", warmup, **kwargs)


# ---------------------------------------------------------------------
# Ordering: lags must be true time offsets
# ---------------------------------------------------------------------

class TestOrdering:
    """The headline defect. Features are built on the complete grid, so a
    hole in the target no longer redefines what a lag means."""

    def test_lag_48_is_24_hours_earlier_across_a_six_hour_gap(self):
        idx = _grid()
        y = _target(idx)
        y.iloc[200:212] = np.nan          # 6 hours at 30-min intervals
        out, _ = _resolved(pd.DataFrame({"y": y}))

        checked = 0
        for ts in out.index:
            prior = ts - pd.Timedelta(hours=24)
            if prior not in idx or pd.isna(y.loc[prior]):
                continue
            checked += 1
            assert out.loc[ts, "y_lag_48"] == pytest.approx(y.loc[prior]), (
                f"y_lag_48 at {ts} must be the value at {prior}"
            )
        assert checked > 300, "the check itself must actually exercise rows"

    def test_positional_lags_would_have_been_wrong(self):
        """Guards the guard: on the pre-fix ordering — drop first, then
        build — the same frame produces a materially wrong y_lag_48. If
        this ever stops failing, the test above has stopped proving
        anything."""
        idx = _grid()
        y = _target(idx)
        y.iloc[200:212] = np.nan
        punctured = pd.DataFrame({"y": y}).dropna()
        legacy = _assemble(punctured).dropna()

        wrong = sum(
            1 for ts in legacy.index
            if (ts - pd.Timedelta(hours=24)) in idx
            and not pd.isna(y.loc[ts - pd.Timedelta(hours=24)])
            and not np.isclose(
                legacy.loc[ts, "y_lag_48"], y.loc[ts - pd.Timedelta(hours=24)],
            )
        )
        assert wrong > 0, "the old ordering is supposed to be broken here"

    @pytest.mark.parametrize("gap", [(0, 3), (100, 130), (900, 950)])
    def test_index_handed_to_build_features_is_contiguous(self, gap):
        idx = _grid()
        y = _target(idx)
        y.iloc[gap[0]:gap[1]] = np.nan
        df = pd.DataFrame({"y": y})
        assert len(set(np.diff(df.index.values))) == 1, (
            "the grid must stay unbroken at interval_minutes in every gap "
            "scenario — that is the whole precondition for shift() being a "
            "time offset"
        )


# ---------------------------------------------------------------------
# Labels are never imputed
# ---------------------------------------------------------------------

class TestLabelsAreNeverImputed:
    """Imputing a label teaches the model something false and then scores
    it against that fabrication. Imputed values are also smooth, so every
    backend is flattered and the composite ranking starts rewarding
    whichever model best reproduces the imputation scheme."""

    def test_rows_without_a_measured_label_are_excluded(self):
        idx = _grid()
        y = _target(idx)
        y.iloc[300:320] = np.nan
        out, report = _resolved(pd.DataFrame({"y": y}))

        gap_index = idx[300:320]
        assert not any(ts in out.index for ts in gap_index)
        assert report["label_gap_rows"] == 20
        assert out["target"].notna().all()

    def test_no_label_is_ever_invented(self):
        idx = _grid()
        y = _target(idx)
        y.iloc[300:320] = np.nan
        out, _ = _resolved(pd.DataFrame({"y": y}))
        # Every surviving label is a value that was actually measured.
        for ts, val in out["target"].items():
            assert val == pytest.approx(y.loc[ts])

    def test_target_column_never_gets_an_indicator(self):
        idx = _grid()
        y = _target(idx)
        y.iloc[300:320] = np.nan
        out, report = _resolved(pd.DataFrame({"y": y}))
        assert f"target{MISSING_SUFFIX}" not in out.columns
        assert "target" not in report["imputed_cells"]

    def test_a_lag_drawn_from_a_label_gap_survives_flagged(self):
        """The second row of the design's table, which matters more than it
        looks: a row 24 hours after a target gap has y_lag_48 = NaN. That
        is a *feature*, so it is flagged and imputed rather than deleting
        an otherwise perfectly good supervised row."""
        idx = _grid()
        y = _target(idx)
        y.iloc[300:312] = np.nan
        out, _ = _resolved(pd.DataFrame({"y": y}))

        affected = idx[348]               # 300 + 48 → its y_lag_48 is in the gap
        assert affected in out.index, "the row must survive"
        assert out.loc[affected, TARGET_MISSING_COLUMN] == 1.0
        assert not np.isnan(out.loc[affected, "y_lag_48"])


# ---------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------

class TestIndicators:
    def test_a_gap_free_experiment_gains_no_columns(self):
        idx = _grid()
        df = pd.DataFrame({"y": _target(idx), "temp": _target(idx, seed=3)})
        before = _assemble(df)
        out, report = _resolved(df)

        assert report["indicator_cols"] == []
        assert list(out.columns) == list(before.columns)

    def test_only_covariates_with_gaps_get_a_companion(self):
        idx = _grid()
        clean = _target(idx, seed=3)
        gappy = _target(idx, seed=4)
        gappy.iloc[400:500] = np.nan
        out, report = _resolved(
            pd.DataFrame({"y": _target(idx), "clean": clean, "gappy": gappy})
        )

        assert f"gappy{MISSING_SUFFIX}" in report["indicator_cols"]
        assert f"clean{MISSING_SUFFIX}" not in out.columns

    def test_indicator_is_one_where_masked_and_zero_elsewhere(self):
        idx = _grid()
        cov = _target(idx, seed=4)
        cov.iloc[400:500] = np.nan
        out, _ = _resolved(pd.DataFrame({"y": _target(idx), "cov": cov}))

        flag = out[f"cov{MISSING_SUFFIX}"]
        masked = [ts for ts in idx[400:500] if ts in out.index]
        assert (flag.loc[masked] == 1.0).all()
        assert flag.drop(index=masked).eq(0.0).all()
        assert set(np.unique(flag.to_numpy())) <= {0.0, 1.0}

    def test_target_derived_features_share_one_aggregate_indicator(self):
        """One target gap makes every lag, rolling statistic and diff gappy
        at once. Per-column companions would be ~24 near-constant columns
        whose *set* changes every cycle as the window slides over the gap —
        and an unstable column list breaks a cached model outright."""
        idx = _grid()
        y = _target(idx)
        y.iloc[300:312] = np.nan
        out, report = _resolved(pd.DataFrame({"y": y}))

        assert report["indicator_cols"] == [TARGET_MISSING_COLUMN]
        assert len(report["imputed_cells"]) > 10, (
            "many target-derived columns really were imputed"
        )
        assert not any(
            c.startswith("y_lag_") and c.endswith(MISSING_SUFFIX)
            for c in out.columns
        )

    def test_interactions_inherit_the_base_covariate_indicator(self):
        """`cov_x_hour_sin` is NaN exactly where `cov` is, so a separate
        companion would triple the column cost and say nothing new."""
        idx = _grid()
        cov = _target(idx, seed=4)
        cov.iloc[400:500] = np.nan
        out, report = _resolved(pd.DataFrame({"y": _target(idx), "cov": cov}))

        assert f"cov_x_hour_sin{MISSING_SUFFIX}" not in out.columns
        assert f"cov_x_hour_cos{MISSING_SUFFIX}" not in out.columns
        # And the interaction stays exactly base x factor at every row.
        assert np.allclose(
            out["cov_x_hour_sin"], out["cov"] * out["hour_sin"],
        )

    def test_indicators_are_created_after_build_features_not_before(self):
        """`resolve_missingness` runs downstream of `build_features`, so an
        indicator never reaches the interaction loop and no suffix rule is
        needed to keep it out. That matters because a real entity can be
        called `binary_sensor.pump_missing`: filtering interactions by the
        `_missing` suffix would silently strip a genuine covariate's
        interaction terms and break the gap-free parity guarantee."""
        idx = _grid()
        cov = _target(idx, seed=4)
        cov.iloc[400:500] = np.nan
        combined, report = _resolved(pd.DataFrame({"y": _target(idx), "cov": cov}))
        assert f"cov{MISSING_SUFFIX}" in report["indicator_cols"]
        # The frame build_features saw had no indicator in it.
        feats = build_features(
            pd.DataFrame({"y": _target(idx), "cov": cov}), "y",
            interval_minutes=INTERVAL,
        )
        assert not any(c.endswith(MISSING_SUFFIX) for c in feats.columns)

    def test_a_covariate_named_missing_keeps_its_interactions(self):
        idx = _grid()
        df = pd.DataFrame({
            "y": _target(idx),
            f"pump{MISSING_SUFFIX}": _target(idx, seed=7),
        })
        feats = build_features(df, "y", interval_minutes=INTERVAL)
        assert f"pump{MISSING_SUFFIX}_x_hour_sin" in feats.columns
        assert f"pump{MISSING_SUFFIX}_x_hour_cos" in feats.columns

    def test_a_pinned_name_never_overwrites_a_real_column(self):
        """The forecast path pins the indicator set. If that set were ever
        derived from a name suffix it would pick up a covariate called
        `pump_missing` and zero the live channel on every cycle while the
        model trained on its real values."""
        idx = _grid()
        pump = _target(idx, seed=7)
        df = pd.DataFrame({"y": _target(idx), f"pump{MISSING_SUFFIX}": pump})
        out, _ = _resolved(
            df, required_indicators=[f"pump{MISSING_SUFFIX}"],
        )
        assert np.allclose(
            out[f"pump{MISSING_SUFFIX}"], pump.loc[out.index],
        ), "a real covariate must survive being named like an indicator"

    def test_a_covariate_named_like_an_interaction_is_not_rebuilt(self):
        """`foo_x_hour_sin` as an actual sensor, alongside a covariate
        `foo`. Inferring parentage from the name would replace every one of
        its measurements with foo * hour_sin, not just its gaps."""
        idx = _grid()
        real = _target(idx, seed=8)
        real.iloc[500] = np.nan
        df = pd.DataFrame({
            "y": _target(idx), "foo": _target(idx, seed=9),
            "foo_x_hour_sin": real,
        })
        combined = _assemble(df)
        out, report = resolve_missingness(
            combined, "target",
            feature_warmup_rows(len(df), INTERVAL),
            source_columns=["foo", "foo_x_hour_sin"],
        )
        kept = [ts for ts in out.index if ts != idx[500]]
        assert np.allclose(
            out.loc[kept, "foo_x_hour_sin"], real.loc[kept],
        )
        assert f"foo_x_hour_sin{MISSING_SUFFIX}" in report["indicator_cols"]

    def test_an_indicator_name_collision_does_not_clobber_a_covariate(self):
        idx = _grid()
        cov = _target(idx, seed=4)
        cov.iloc[400:500] = np.nan
        decoy = _target(idx, seed=9)
        out, report = _resolved(pd.DataFrame({
            "y": _target(idx), "cov": cov, f"cov{MISSING_SUFFIX}": decoy,
        }))
        # The real covariate keeps its values; the flag takes another name.
        assert np.allclose(
            out[f"cov{MISSING_SUFFIX}"], decoy.loc[out.index],
        )
        assert f"cov{MISSING_SUFFIX}_flag" in report["indicator_cols"]

    def test_a_column_masked_only_inside_warmup_gains_nothing(self):
        """A constant-zero indicator is pure noise, and emitting one would
        also break the gap-free column-count guarantee for any experiment
        whose covariate merely starts a few rows late."""
        idx = _grid()
        cov = _target(idx, seed=4)
        cov.iloc[:10] = np.nan
        out, report = _resolved(pd.DataFrame({"y": _target(idx), "cov": cov}))
        assert report["indicator_cols"] == []
        assert f"cov{MISSING_SUFFIX}" not in out.columns


# ---------------------------------------------------------------------
# Imputation must be causal
# ---------------------------------------------------------------------

class TestCausalImputation:
    def test_imputed_value_is_independent_of_the_future(self):
        """The CV harness splits *after* preprocessing, so a statistic over
        the whole window leaks the scored fold into training."""
        idx = _grid()
        base = _target(idx, seed=5)
        base.iloc[200] = np.nan

        a, _ = causal_impute(base)
        perturbed = base.copy()
        perturbed.iloc[400:] += 1000.0     # rewrite the entire future
        b, _ = causal_impute(perturbed)

        assert a.iloc[200] == pytest.approx(b.iloc[200])
        assert np.allclose(
            a.iloc[:400].to_numpy(), b.iloc[:400].to_numpy(), equal_nan=True,
        )

    def test_imputed_value_is_the_median_of_strictly_prior_observations(self):
        s = pd.Series([1.0, 5.0, 3.0, np.nan, 100.0], index=_grid(5))
        filled, mask = causal_impute(s)
        assert filled.iloc[3] == pytest.approx(3.0)   # median(1, 5, 3)
        assert mask.tolist() == [False, False, False, True, False]

    def test_whole_frame_imputation_is_independent_of_the_future(self):
        idx = _grid()
        cov = _target(idx, seed=4)
        cov.iloc[400:420] = np.nan
        df = pd.DataFrame({"y": _target(idx), "cov": cov})
        out_a, _ = _resolved(df)

        df_b = df.copy()
        df_b.iloc[600:, :] += 500.0
        out_b, _ = _resolved(df_b)

        early = out_a.index[out_a.index < idx[600]]
        assert np.allclose(
            out_a.loc[early, "cov"].to_numpy(),
            out_b.loc[early, "cov"].to_numpy(),
        )

    @pytest.mark.parametrize("seed", range(25))
    def test_matches_a_naive_reference_implementation(self, seed):
        """The vectorised version only walks the prefix up to the last gap,
        because the expanding median is most of the cost of the whole
        missingness step and it runs on every forecast tick. This pins it
        against the obvious row-by-row definition."""
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 80))
        s = pd.Series(rng.normal(size=n))
        s[rng.random(n) < 0.3] = np.nan
        if s.notna().sum() == 0:
            pytest.skip("no observations to impute from")

        got, mask = causal_impute(s)

        want = s.copy()
        for i in range(n):
            if pd.isna(s.iloc[i]):
                prior = s.iloc[:i].dropna()
                want.iloc[i] = (
                    prior.median() if len(prior) else s.dropna().iloc[0]
                )
        assert np.allclose(got.to_numpy(), want.to_numpy())
        assert (mask.to_numpy() == s.isna().to_numpy()).all()

    def test_leading_gap_takes_the_first_observation_and_is_flagged(self):
        """A deliberate, documented exception: a leading gap has no prior
        observation, so it takes a single scalar from the boundary. The
        flag is 1 across the whole leading region, so the model has an
        explicit signal to discount the column there."""
        idx = _grid()
        cov = _target(idx, seed=4)
        cov.iloc[:200] = np.nan
        out, _ = _resolved(pd.DataFrame({"y": _target(idx), "cov": cov}))

        lead = [ts for ts in idx[:200] if ts in out.index]
        assert lead, "some leading rows must survive the warm-up"
        first_observed = float(cov.dropna().iloc[0])
        assert np.allclose(out.loc[lead, "cov"], first_observed)
        assert (out.loc[lead, f"cov{MISSING_SUFFIX}"] == 1.0).all()

    def test_a_column_with_no_observations_is_imputed_at_zero_and_flagged(self):
        idx = _grid()
        out, report = _resolved(pd.DataFrame({
            "y": _target(idx), "dead": pd.Series(np.nan, index=idx),
        }))
        assert report["empty_cols"] == ["dead"]
        assert (out["dead"] == 0.0).all()
        assert (out[f"dead{MISSING_SUFFIX}"] == 1.0).all()

    def test_no_nan_survives(self):
        idx = _grid()
        y = _target(idx)
        y.iloc[300:312] = np.nan
        cov = _target(idx, seed=4)
        cov.iloc[400:500] = np.nan
        out, _ = _resolved(pd.DataFrame({"y": y, "cov": cov}))
        assert not out.isna().to_numpy().any()


# ---------------------------------------------------------------------
# Warm-up arithmetic
# ---------------------------------------------------------------------

class TestWarmupRows:
    """Warm-up is not missing data — it is not enough history yet — so it
    is dropped rather than imputed. If this count drifts from what
    `build_features` actually leaves undefined, a gap-free experiment
    silently changes its training-row count."""

    @pytest.mark.parametrize("interval", [5, 15, 30, 60, 7, 1440])
    @pytest.mark.parametrize("gated", [False, True])
    def test_matches_build_features_exactly(self, interval, gated):
        n = max(60, int(48 * 20 * 30 / max(interval, 1)))
        n = min(n, 3000)
        idx = pd.date_range("2026-05-01", periods=n, freq=f"{interval}min")
        df = pd.DataFrame({"y": _target(idx)})
        if gated:
            steps = max(1, 1440 // interval)
            df["clear_sky_ghi"] = np.clip(
                np.sin(np.arange(n) / steps * 2 * np.pi - 0.5), 0, None,
            ) * 800

        feats = build_features(df, "y", interval_minutes=interval)
        any_nan = feats.isna().any(axis=1).to_numpy()
        observed = int(np.argmin(any_nan)) if not any_nan.all() else n

        assert feature_warmup_rows(n, interval, ghi_gated=gated) == observed
        assert not any_nan[observed:].any(), (
            "everything after warm-up must be complete on a gap-free frame, "
            "or warm-up is not what is being measured"
        )

    def test_the_clear_sky_gate_shortens_warmup(self):
        """`_gate_by_past_ghi` writes 0.0 wherever the shifted GHI is not
        positive, and a shift that reaches off the front of the series is
        NaN — so `NaN > 0` is False and every warm-up cell of every gated
        column becomes 0.0. Counting those columns anyway would delete real
        supervised rows from every PV experiment."""
        assert feature_warmup_rows(960, 30, ghi_gated=True) == 72
        assert feature_warmup_rows(960, 30, ghi_gated=False) == 96

    def test_warmup_rows_are_dropped_not_imputed(self):
        idx = _grid()
        df = pd.DataFrame({"y": _target(idx)})
        out, report = _resolved(df)
        assert report["warmup_rows"] == 96
        assert out.index[0] == idx[96]

    def test_lag_windows_default_is_shared_with_build_features(self):
        for interval in (5, 30, 60):
            idx = pd.date_range("2026-05-01", periods=2000, freq=f"{interval}min")
            feats = build_features(
                pd.DataFrame({"y": _target(idx)}), "y",
                interval_minutes=interval,
            )
            for w in default_lag_windows(interval):
                assert f"y_rolling_mean_{w}" in feats.columns


# ---------------------------------------------------------------------
# Column-set stability across a sliding window
# ---------------------------------------------------------------------

class TestIndicatorPinning:
    """Which columns have gaps is a property of the *window*. Between a
    retrain and the next forecast the window slides, so letting the data
    decide at inference changes the matrix's shape out from under a cached
    model — a silently different column list for tree backends, and a
    channel-name mismatch that refuses the forecast for sequence ones."""

    def test_pinned_indicators_are_emitted_even_when_clean(self):
        idx = _grid()
        clean = pd.DataFrame({"y": _target(idx), "cov": _target(idx, seed=4)})
        out, report = _resolved(
            clean, required_indicators=[f"cov{MISSING_SUFFIX}"],
        )
        assert report["indicator_cols"] == [f"cov{MISSING_SUFFIX}"]
        assert (out[f"cov{MISSING_SUFFIX}"] == 0.0).all()

    def test_pinned_set_suppresses_unrequested_indicators(self):
        idx = _grid()
        cov = _target(idx, seed=4)
        cov.iloc[400:500] = np.nan
        out, report = _resolved(
            pd.DataFrame({"y": _target(idx), "cov": cov}),
            required_indicators=[],
        )
        assert report["indicator_cols"] == []
        assert report["dropped_indicators"] == [f"cov{MISSING_SUFFIX}"]
        # Still imputed — just not reported to the model this cycle.
        assert not out["cov"].isna().any()

    def test_a_cache_from_before_this_change_still_matches(self):
        """A model cached by the previous release has no indicators in its
        `feature_cols`. The next forecast tick must reproduce exactly that
        column list, not add two columns the model was never fitted on."""
        idx = _grid()
        y = _target(idx)
        y.iloc[300:312] = np.nan
        cov = _target(idx, seed=4)
        cov.iloc[400:500] = np.nan
        df = pd.DataFrame({"y": y, "cov": cov})

        current, _ = _resolved(df)
        legacy_cols = [
            c for c in current.columns
            if c != "target" and not c.endswith(MISSING_SUFFIX)
        ]
        replay, report = _resolved(
            df,
            required_indicators=[
                c for c in legacy_cols if c.endswith(MISSING_SUFFIX)
            ],
        )
        assert [c for c in replay.columns if c != "target"] == legacy_cols
        assert set(report["dropped_indicators"]) == {
            f"cov{MISSING_SUFFIX}", TARGET_MISSING_COLUMN,
        }
        assert not replay.isna().to_numpy().any(), (
            "suppressing the flag must not suppress the imputation"
        )

    def test_column_list_is_identical_with_and_without_the_gap(self):
        idx = _grid()
        gappy = _target(idx, seed=4)
        gappy.iloc[400:500] = np.nan
        trained, report = _resolved(
            pd.DataFrame({"y": _target(idx), "cov": gappy})
        )
        later, _ = _resolved(
            pd.DataFrame({"y": _target(idx), "cov": _target(idx, seed=4)}),
            required_indicators=report["indicator_cols"],
        )
        assert list(trained.columns) == list(later.columns)


class TestLeadingTargetGap:
    """Warm-up is counted from the first *measured* label. Counting it from
    grid row 0 leaves the first genuinely supervised row with every
    target-derived feature still unfilled, and `causal_impute`'s
    leading-gap branch then fills them from the first observed value — for
    `y_lag_k` that value IS a later label, so the row would be handed its
    own answer as a feature."""

    def test_no_row_is_given_its_own_label_as_a_lag(self):
        idx = _grid(1000)
        y = _target(idx)
        y.iloc[:400] = np.nan
        out, report = _resolved(pd.DataFrame({"y": y}))

        assert not out.empty
        lag_cols = [c for c in out.columns if c.startswith("y_lag_")]
        for col in lag_cols:
            assert not np.isclose(
                out[col].to_numpy(), out["target"].to_numpy(),
            ).any(), f"{col} equals its own label on some row"

    def test_rolling_features_do_not_summarise_their_own_future(self):
        idx = _grid(1000)
        y = _target(idx)
        y.iloc[:400] = np.nan
        out, _ = _resolved(pd.DataFrame({"y": y}))
        for w in default_lag_windows(INTERVAL):
            col = f"y_rolling_mean_{w}"
            future_mean = float(y.dropna().iloc[:w].mean())
            assert not np.isclose(
                out[col].to_numpy(), future_mean,
            ).all(), f"{col} is a constant drawn from its own future"

    def test_warmup_is_measured_from_the_first_measured_label(self):
        idx = _grid(1000)
        y = _target(idx)
        y.iloc[:400] = np.nan
        out, report = _resolved(pd.DataFrame({"y": y}))
        assert report["warmup_rows"] == 400 + 96
        assert out.index[0] == idx[496]

    def test_a_gap_free_frame_is_unaffected_by_the_anchor(self):
        idx = _grid()
        out, report = _resolved(pd.DataFrame({"y": _target(idx)}))
        assert report["warmup_rows"] == 96
        assert out.index[0] == idx[96]


class TestBinaryCovariates:
    """A binary covariate is a step function — upstream it is resampled
    with last().ffill() under a recorder where "no row" means "did not
    move". An expanding median returns 0.5 whenever an even number of
    prior observations splits evenly: a value that appears in no observed
    row of that channel."""

    def test_a_binary_gap_is_held_not_averaged(self):
        idx = _grid()
        cov = pd.Series(
            np.tile([0.0, 0.0, 1.0, 1.0], len(idx) // 4), index=idx,
        )
        cov.iloc[400:500] = np.nan
        out, _ = _resolved(pd.DataFrame({"y": _target(idx), "cov": cov}))
        assert set(np.unique(out["cov"].to_numpy())) <= {0.0, 1.0}

    def test_a_continuous_covariate_still_uses_the_median(self):
        s = pd.Series([1.0, 5.0, 3.0, np.nan, 100.0], index=_grid(5))
        filled, _ = causal_impute(s)
        assert filled.iloc[3] == pytest.approx(3.0)

    def test_hold_mode_carries_the_last_prior_observation(self):
        s = pd.Series([0.0, 1.0, np.nan, np.nan, 0.0], index=_grid(5))
        filled, mask = causal_impute(s, method="hold")
        assert filled.tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]
        assert mask.tolist() == [False, False, True, True, False]


class TestLoadSubtractKeepsTheGrid:
    """`load_subtract` with `on_missing: drop` deletes rows outright, after
    resampling. A deleted row carries no NaN, so the whole missingness
    machinery would report a clean frame while `target.shift(48)` silently
    went back to meaning "48 surviving rows"."""

    def test_dropped_rows_come_back_as_unmeasured(self):
        from ml_forecast_lab.preprocessing import apply_load_subtract

        idx = _grid(500)
        base = pd.Series(np.arange(500, dtype=float) + 1000.0, index=idx)
        sub = pd.Series(np.ones(500), index=idx)
        sub.iloc[200:220] = np.nan
        adjusted, _audit = apply_load_subtract(
            base, [({"entity_id": "sensor.s", "on_missing": "drop"}, sub)],
        )
        assert len(adjusted) < len(idx), (
            "on_missing=drop is supposed to delete rows — if this stops "
            "being true the guard below is testing nothing"
        )
        restored = adjusted.reindex(idx)
        assert len(restored) == len(idx)
        assert int(restored.isna().sum()) == len(idx) - len(adjusted)

    def test_the_pipeline_restores_the_grid(self):
        root = Path(__file__).resolve().parents[2]
        src = (root / "ml_forecast_lab" / "main.py").read_text()
        body = src[src.index("def _fetch_and_preprocess"):]
        body = body[: body.index("\n    def ", 1)]
        assert "series = series.reindex(grid_index)" in body


class TestInsufficientDataGuard:
    def test_the_benchmark_guard_counts_measured_labels(self):
        """`len(df)` used to be the supervised row count. It is now the
        grid size, so a target measured for 9% of its window would pass a
        guard that exists to stop exactly that."""
        root = Path(__file__).resolve().parents[2]
        src = (root / "ml_forecast_lab" / "main.py").read_text()
        assert '_measured = int(df["y"].notna().sum())' in src
        assert "if len(df) < exp_cfg.cv_folds * 10:" not in src
        assert "Insufficient data for benchmark after row selection" in src


# ---------------------------------------------------------------------
# Covariate alignment is bounded by the entity's own cadence
# ---------------------------------------------------------------------

class TestBoundedAlignment:
    def test_a_healthy_hourly_covariate_on_a_half_hourly_grid_is_not_masked(self):
        """A threshold that fires on correct setups is one people learn to
        ignore — and here it would mask a perfectly good covariate out of
        the model, not merely warn about it."""
        grid = _grid(48 * 5)
        hourly = pd.Series(
            np.arange(len(grid[::2])) * 1.0, index=grid[::2],
        )
        aligned = _align_covariate_to_grid(hourly, grid, INTERVAL)
        assert aligned.notna().all()

    def test_a_covariate_that_starts_late_is_masked_not_backfilled(self):
        grid = _grid(48 * 5)
        late = pd.Series(np.arange(100) * 1.0, index=grid[-100:])
        aligned = _align_covariate_to_grid(late, grid, INTERVAL)
        assert aligned.iloc[:100].isna().all()
        assert aligned.iloc[-100:].notna().all()

    def test_a_long_outage_is_masked_not_held_forever(self):
        grid = _grid(48 * 10)
        obs = pd.Series(np.arange(len(grid)) * 1.0, index=grid)
        obs.iloc[100:300] = np.nan
        aligned = _align_covariate_to_grid(obs, grid, INTERVAL)
        assert aligned.iloc[150:290].isna().all(), (
            "the value must not be held across a 4-day outage"
        )
        assert aligned.iloc[:100].notna().all()


# ---------------------------------------------------------------------
# The nan_to_num backstop
# ---------------------------------------------------------------------

class TestNanToNumBackstop:
    """With missingness resolved upstream, nothing should reach these. The
    calls stay — a NaN reaching a backend is worse than a zero — but
    silently absorbing one is how the v2.27.10 covariate regression
    survived a whole release."""

    def test_a_firing_backstop_warns_and_names_the_column(self, caplog):
        X = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
        with caplog.at_level(logging.WARNING, logger="ml_forecast_lab.main"):
            out = _nan_to_num_guarded(X, "unit test", ["a", "b"])
        assert out[0, 1] == 0.0
        assert "nan_to_num backstop fired at unit test" in caplog.text
        assert "b:1" in caplog.text

    def test_a_clean_matrix_is_silent(self, caplog):
        X = np.array([[1.0, 2.0]], dtype=np.float32)
        with caplog.at_level(logging.DEBUG, logger="ml_forecast_lab.main"):
            _nan_to_num_guarded(X, "unit test", ["a", "b"])
        assert "backstop fired" not in caplog.text

    def test_expected_sites_log_below_warning(self, caplog):
        """The per-fold closures recompute rolling statistics inside a CV
        fold, so they produce warm-up NaN by construction once per fold per
        model. A warning there is noise, not a signal."""
        X = np.array([[np.nan]], dtype=np.float32)
        with caplog.at_level(logging.WARNING, logger="ml_forecast_lab.main"):
            _nan_to_num_guarded(X, "fold", ["a"], expected=True)
        assert caplog.text == ""

    def test_every_boundary_uses_the_guarded_helper(self):
        """Source contract: a bare np.nan_to_num anywhere else is a
        boundary that can swallow a bug silently."""
        root = Path(__file__).resolve().parents[2]
        src = (root / "ml_forecast_lab" / "main.py").read_text()
        # The only bare call left is the one inside the helper itself.
        assert src.count("np.nan_to_num(") == 1


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------

class TestInferenceContract:
    def test_indicator_map_excludes_the_aggregate_target_flag(self):
        cols = ["y_lag_1", f"cov{MISSING_SUFFIX}", TARGET_MISSING_COLUMN, "cov"]
        assert _inference_indicator_map(cols) == {f"cov{MISSING_SUFFIX}": "cov"}

    def test_the_forecast_flag_answers_the_training_question(self):
        """Training flags 1 when no observation fell within the entity's
        cadence of that grid point. A forecast row has no observation for
        any covariate, so the faithful answer is the one the last frame row
        gave. Forcing 1 whenever no *future source* exists answers a
        different question, and answers it for every horizon step —
        including step 1, thirty minutes after a reading training would
        have flagged 0."""
        idx = _grid()
        cov = _target(idx, seed=4)
        cov.iloc[400:500] = np.nan          # one ordinary outage, long past
        out, report = _resolved(pd.DataFrame({"y": _target(idx), "cov": cov}))
        flag = f"cov{MISSING_SUFFIX}"

        assert flag in report["indicator_cols"]
        share_flagged = float(out[flag].mean())
        assert share_flagged < 0.2, "the covariate is healthy on most rows"
        # What inference carries forward for a lagged-only covariate.
        carried = float(out[flag].iloc[-1])
        assert carried == 0.0, (
            "the covariate was measured at the last row, so every forecast "
            "step must be told 0 — the value 97% of training rows carried"
        )

    def test_no_inference_path_hardcodes_the_flag(self):
        root = Path(__file__).resolve().parents[2]
        src = (root / "ml_forecast_lab" / "main.py").read_text()
        assert "row[flag_col] = 0.0 if fresh.get(base) else 1.0" not in src
        assert src.count("else last_flag_vals.get(flag_col, 0.0)") == 2

    def test_indicator_provenance_comes_from_the_cache(self):
        """Re-deriving the pinned set from the `_missing` suffix picks up a
        covariate genuinely named `pump_missing` and zeroes a live channel
        on every forecast. The trained set is stored instead."""
        root = Path(__file__).resolve().parents[2]
        src = (root / "ml_forecast_lab" / "main.py").read_text()
        assert 'required_indicators = list(cache.get("missing_indicators") or [])' in src
        assert (
            "required_indicators = [\n                c for c in feature_cols "
            "if c.endswith(MISSING_SUFFIX)\n            ]"
        ) not in src
        # Written on retrain, persisted, and restored.
        assert '"missing_indicators": list(' in src
        assert src.count('meta.get("missing_indicators")') == 2

    def test_lag_buffer_is_seeded_on_true_time_offsets(self):
        """`buf[-k]` is only "k intervals ago" if the buffer holds
        consecutive grid slots. Seeding from the supervised frame breaks
        that the moment the target has a recent outage."""
        idx = _grid(200)
        y = _target(idx)
        y.iloc[150:160] = np.nan
        values, imputed = _seed_lag_buffer(y, None, idx[-1], 97)

        assert len(values) == 97
        assert not np.isnan(values).any()
        # Position of the outage inside the buffer, counted from the end.
        offsets = [len(idx) - 1 - i for i in range(150, 160)]
        for off in offsets:
            assert imputed[-(off + 1)] is True
        assert sum(imputed) == 10

    def test_lag_buffer_falls_back_when_no_grid_is_available(self):
        values, imputed = _seed_lag_buffer(
            None, np.arange(10, dtype=float), None, 4,
        )
        assert values == [6.0, 7.0, 8.0, 9.0]
        assert imputed == [False] * 4

    def test_gap_free_seed_matches_the_legacy_tail(self):
        idx = _grid(200)
        y = _target(idx)
        values, imputed = _seed_lag_buffer(y, None, idx[-1], 97)
        assert np.allclose(values, y.to_numpy()[-97:])
        assert not any(imputed)


class TestWindowFrame:
    """Sequence models window over consecutive rows, so they get their own
    frame: the complete grid from the warm-up anchor onward, every feature
    imputed, the target causally imputed, and a per-row label mask. Window
    contents are features and may be imputed; the horizon values a window
    is trained or scored against are labels and never are."""

    def test_gap_free_window_frame_is_the_supervised_frame(self):
        idx = _grid()
        out, report = _resolved(
            pd.DataFrame({"y": _target(idx), "cov": _target(idx, seed=3)})
        )
        assert report["window_frame"].equals(out)
        assert bool(report["window_label_mask"].all())

    def test_window_frame_is_contiguous_and_complete(self):
        idx = _grid()
        y = _target(idx)
        y.iloc[600:612] = np.nan
        cov = _target(idx, seed=4)
        cov.iloc[300:500] = np.nan
        out, report = _resolved(pd.DataFrame({"y": y, "cov": cov}))
        win = report["window_frame"]

        assert len(set(np.diff(win.index.values))) == 1
        assert not win.isna().to_numpy().any()
        assert len(win) == len(out) + 12, (
            "the label-gap rows survive as window inputs"
        )

    def test_window_target_flag_is_per_row_not_aggregate(self):
        """The supervised frame's y_missing is the OR over target-derived
        features — the honest flag for a row of lags. A window's target
        channel is raw y, so its honest flag is "the y at THIS row is
        invented", which is a strict subset."""
        idx = _grid()
        y = _target(idx)
        y.iloc[600:612] = np.nan
        out, report = _resolved(pd.DataFrame({"y": y}))
        win = report["window_frame"]

        assert int(win[TARGET_MISSING_COLUMN].sum()) == 12
        assert int(out[TARGET_MISSING_COLUMN].sum()) > 12
        mask = report["window_label_mask"]
        assert (
            win[TARGET_MISSING_COLUMN].to_numpy()
            == (~mask).to_numpy().astype(np.float32)
        ).all()

    def test_window_labels_are_never_imputed_values(self):
        """The end-to-end guarantee: build real training windows from the
        frame and check every label in seq_y is a measured y value."""
        idx = _grid()
        y = _target(idx)
        y.iloc[600:612] = np.nan
        _, report = _resolved(pd.DataFrame({"y": y}))
        win = report["window_frame"]
        lm = report["window_label_mask"].to_numpy()

        from ml_forecast_lab.features import create_sliding_windows
        H = list(range(1, 49))
        _, seq_y, _, kept = create_sliding_windows(
            win, "target", window_size=48, add_temporal=True,
            horizon_steps=H, label_mask=lm,
        )
        assert len(kept) > 0
        for row, k in enumerate(kept):
            for h in (1, 24, 48):
                ts = win.index[k + 48 + h - 1]
                assert not pd.isna(y.loc[ts]), "label row must be measured"
                assert seq_y[row, h - 1] == pytest.approx(y.loc[ts])

    def test_every_kept_window_is_a_true_time_span(self):
        idx = _grid()
        y = _target(idx)
        y.iloc[600:612] = np.nan
        _, report = _resolved(pd.DataFrame({"y": y}))
        win = report["window_frame"]
        lm = report["window_label_mask"].to_numpy()

        from ml_forecast_lab.features import create_sliding_windows
        _, _, _, kept = create_sliding_windows(
            win, "target", window_size=48, add_temporal=True,
            horizon_steps=[1], label_mask=lm,
        )
        step = pd.Timedelta(minutes=INTERVAL)
        assert all(
            win.index[k + 47] - win.index[k] == 47 * step for k in kept
        )

    def test_the_supervised_frame_would_have_warped_windows(self):
        """Guards the guard: windowing the supervised frame across the
        same gap produces windows whose 48 rows span more than 24 hours.
        If this stops failing, the tests above prove nothing."""
        idx = _grid()
        y = _target(idx)
        y.iloc[600:612] = np.nan
        out, _ = _resolved(pd.DataFrame({"y": y}))
        step = pd.Timedelta(minutes=INTERVAL)
        warped = sum(
            1 for i in range(len(out) - 48)
            if out.index[i + 47] - out.index[i] != 47 * step
        )
        assert warped > 0

    def test_trailing_gap_reaches_the_inference_window(self):
        """A recent outage: the frame's tail up to the last measured label
        is contiguous grid rows with imputed, flagged y — the sequence
        analogue of the recursive path's _seed_lag_buffer."""
        idx = _grid()
        y = _target(idx)
        y.iloc[-30:-10] = np.nan
        out, report = _resolved(pd.DataFrame({"y": y}))
        win = report["window_frame"]
        last_ts = out.index[-1]
        tail = win.loc[:last_ts].iloc[-48:]
        assert len(tail) == 48
        assert (
            tail.index[-1] - tail.index[0]
            == 47 * pd.Timedelta(minutes=INTERVAL)
        )
        assert int(tail[TARGET_MISSING_COLUMN].sum()) == 20


class TestEvaluationWindowMask:
    def test_eval_windows_require_only_the_scored_label(self):
        """Training windows need every horizon label measured — they all
        enter the loss. Evaluation windows are scored at h=1 only, and
        requiring all 48 would discard scoreable predictions near a gap."""
        idx = _grid()
        y = _target(idx)
        y.iloc[600:612] = np.nan
        _, report = _resolved(pd.DataFrame({"y": y}))
        win = report["window_frame"]
        lm = report["window_label_mask"].to_numpy()

        from ml_forecast_lab.features import create_sliding_windows
        H = list(range(1, 49))
        _, _, _, kept_strict = create_sliding_windows(
            win, "target", window_size=48, add_temporal=True,
            horizon_steps=H, label_mask=lm,
        )
        _, _, _, kept_eval = create_sliding_windows(
            win, "target", window_size=48, add_temporal=True,
            horizon_steps=H, label_mask=lm, mask_horizons=[1],
        )
        assert len(kept_eval) > len(kept_strict)
        # And each eval window's h=1 label is measured.
        assert all(lm[k + 48] for k in kept_eval)


# ---------------------------------------------------------------------
# CV fold feature rebuild
# ---------------------------------------------------------------------

class TestFoldFeatureRebuild:
    """The benchmark rebuilds rolling statistics and periodic lags per
    fold. Those shifts are positional too, so fixing lag construction
    upstream and leaving this alone would fix training and then score it
    with the broken features."""

    def test_gap_free_fold_is_unchanged(self):
        idx = _grid(400)
        frame = pd.DataFrame({"target": _target(idx)}, index=idx)
        legacy = frame.copy()
        t = legacy["target"]
        shifted = t.shift(1)
        for w in default_lag_windows(INTERVAL):
            legacy[f"y_rolling_mean_{w}"] = shifted.rolling(w).mean()
            legacy[f"y_rolling_std_{w}"] = shifted.rolling(w).std()
            legacy[f"y_rolling_max_{w}"] = shifted.rolling(w).max()
        for d in (1, 2):
            legacy[f"y_lag_{STEPS_PER_DAY * d}"] = t.shift(STEPS_PER_DAY * d)
        legacy["y_diff_1"] = t.shift(1) - t.shift(2)

        rebuilt = rebuild_fold_features(
            frame.copy(), "target", INTERVAL,
            default_lag_windows(INTERVAL), STEPS_PER_DAY,
        )
        pd.testing.assert_frame_equal(
            rebuilt[legacy.columns], legacy, check_exact=False,
        )

    def test_punctured_fold_gets_true_offsets(self):
        idx = _grid(400)
        full = _target(idx)
        keep = idx.delete(range(150, 170))
        frame = pd.DataFrame({"target": full.loc[keep]}, index=keep)

        rebuilt = rebuild_fold_features(
            frame.copy(), "target", INTERVAL,
            default_lag_windows(INTERVAL), STEPS_PER_DAY,
        )
        ts = keep[-1]
        want = full.loc[ts - pd.Timedelta(hours=24)]
        assert rebuilt.loc[ts, "y_lag_48"] == pytest.approx(want)

    def test_range_index_falls_back_without_raising(self):
        frame = pd.DataFrame({"target": np.arange(300, dtype=float)})
        rebuilt = rebuild_fold_features(
            frame.copy(), "target", INTERVAL,
            default_lag_windows(INTERVAL), STEPS_PER_DAY,
        )
        assert rebuilt.loc[100, "y_lag_48"] == pytest.approx(52.0)


# ---------------------------------------------------------------------
# Neural channel selection
# ---------------------------------------------------------------------

class TestNeuralChannels:
    def test_every_indicator_is_a_channel_including_the_target_flag(self):
        """Sequence models window over raw y, and the window frame's y is
        causally imputed across label gaps — so the per-row y_missing is
        exactly the channel that tells the model which of its y inputs
        were invented. Engineered lags stay out: the window already IS
        the history they summarise."""
        cols = [
            "target", "hour_sin", "is_holiday", "y_lag_1", "y_lag_1_missing",
            "y_rolling_mean_6", "cov", f"cov{MISSING_SUFFIX}",
            TARGET_MISSING_COLUMN,
        ]
        out = neural_covariate_columns(cols)
        assert f"cov{MISSING_SUFFIX}" in out
        assert TARGET_MISSING_COLUMN in out
        assert "y_lag_1" not in out and "y_lag_1_missing" not in out
        assert "hour_sin" not in out and "target" not in out
        # Unchanged for the columns that existed before this spec.
        assert "y_rolling_mean_6" in out and "cov" in out

    def test_channel_list_is_unchanged_for_pre_existing_columns(self):
        """The cached channel order is compared name-by-name at forecast
        time and a mismatch refuses the publish outright, so this filter
        must reproduce the previous release's answer exactly on any column
        set that release could produce."""
        def legacy(cols, target_col="target"):
            engineered = {
                "hour_of_day", "day_of_week", "is_weekend", "month",
                "day_of_month", "hour_sin", "hour_cos", "dow_sin",
                "dow_cos", "is_holiday",
            }
            engineered.update(c for c in cols if c.startswith("y_lag_"))
            return [
                c for c in cols
                if c not in engineered and c != target_col
            ]

        idx = _grid()
        combined = _assemble(
            pd.DataFrame({"y": _target(idx), "cov": _target(idx, seed=3)})
        )
        cols = list(combined.columns)
        assert legacy(cols) == neural_covariate_columns(cols)

    def test_the_two_modules_agree_on_the_indicator_names(self):
        assert features_mod.MISSING_SUFFIX == preprocessing_mod.MISSING_SUFFIX
        assert (
            features_mod.TARGET_MISSING_COLUMN
            == preprocessing_mod.TARGET_MISSING_COLUMN
        )


# ---------------------------------------------------------------------
# End to end, through the real fetch path
# ---------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


class _StubHA:
    def __init__(self, rows_by_entity):
        self.rows_by_entity = rows_by_entity

    async def get_history(self, entity_id, start, end, include_attributes=False):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if s.tzinfo is None:
            s = s.tz_localize("UTC")
        if e.tzinfo is None:
            e = e.tz_localize("UTC")
        return [
            r for r in self.rows_by_entity.get(entity_id, [])
            if s <= pd.Timestamp(r["last_changed"]) <= e
        ]

    async def get_state(self, entity_id, default=None, attribute=None):
        return default


def _recorder_rows(entity_now, days, cadence_min=15, skip=None,
                   value=lambda i: i % 37 + 1.0):
    ts = pd.date_range(
        entity_now - timedelta(days=days), entity_now,
        freq=f"{cadence_min}min", tz="UTC",
    )
    rows = []
    for i, t in enumerate(ts):
        if skip is not None and skip[0] <= t <= skip[1]:
            continue
        rows.append({"last_changed": t.isoformat(), "state": f"{value(i):.4f}"})
    return rows


def _make_app(tmp_db, exp_cfg, rows_by_entity):
    app = MLForecastLabApp()
    app.history_db = HistoryDB(tmp_db)
    app.ha_interface = _StubHA(rows_by_entity)
    app.config = AppConfig(experiments=[exp_cfg])
    app.covariate_resolver = CovariateResolver(
        app.ha_interface,
        history_db=app.history_db,
        retention_provider=app._retention_days_for_table,
    )
    return app


class TestEndToEnd:
    @staticmethod
    def _exp(**kw):
        params = dict(
            name="missing", target_entity="sensor.load", days_history=10,
            interval_minutes=INTERVAL, models_enabled=["lightgbm"],
        )
        params.update(kw)
        return ExperimentCfg(**params)

    def test_preprocess_returns_the_complete_grid(self, tmp_db):
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        gap = (now - timedelta(days=4), now - timedelta(days=4) + timedelta(hours=6))
        exp = self._exp()
        app = _make_app(tmp_db, exp, {
            "sensor.load": _recorder_rows(now, days=11, skip=gap),
        })
        result = _run(app._fetch_and_preprocess(exp))

        assert result is not None
        assert len(set(np.diff(result.index.values))) == 1, (
            "the frame handed to build_features must be an unbroken grid"
        )
        assert result["y"].isna().any(), "the outage must still be visible"

    def test_a_target_gap_is_warned_about_by_name(self, tmp_db, caplog):
        """A holed target is a broken experiment, whereas a holed covariate
        is merely a weak one — the two deserve different volume."""
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        gap = (now - timedelta(days=4), now - timedelta(days=4) + timedelta(hours=6))
        exp = self._exp()
        app = _make_app(tmp_db, exp, {
            "sensor.load": _recorder_rows(now, days=11, skip=gap),
        })
        with caplog.at_level(logging.WARNING, logger="ml_forecast_lab.main"):
            _run(app._fetch_and_preprocess(exp))

        hits = [
            r for r in caplog.records
            if "unmeasured row" in r.message and r.levelno >= logging.WARNING
        ]
        assert len(hits) == 1
        assert "cannot be supervised samples" in hits[0].message

    def test_an_entirely_unusable_target_skips_the_cycle(self, tmp_db, caplog):
        """The old guard was `len(result) == 0`, which only worked because
        the frame had just been dropna'd. It now counts measured labels
        instead, so an experiment with nothing to learn from is still
        skipped rather than trained on invented rows."""
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        exp = self._exp()
        app = _make_app(tmp_db, exp, {
            "sensor.load": [
                {"last_changed": (now - timedelta(days=5)).isoformat(),
                 "state": "unavailable"},
            ],
        })
        with caplog.at_level(logging.WARNING, logger="ml_forecast_lab.main"):
            result = _run(app._fetch_and_preprocess(exp))

        assert result is None
        assert "No measured target rows" in caplog.text

    def test_the_zero_row_guard_reads_labels_not_frame_length(self):
        """Source contract. `len(result) == 0` is now always False by the
        time it would be checked — the grid is non-empty whenever any
        observation exists — so the guard has to count measured labels."""
        root = Path(__file__).resolve().parents[2]
        src = (root / "ml_forecast_lab" / "main.py").read_text()
        body = src[src.index("def _fetch_and_preprocess"):]
        body = body[: body.index("\n    def ", 1)]
        assert "if supervised_rows == 0:" in body
        assert "if len(result) == 0:" not in body
        assert "result = result.dropna()" not in body

    def test_supervised_frame_survives_a_covariate_gap(self, tmp_db):
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        cov_gap = (now - timedelta(days=6), now - timedelta(days=3))
        exp = self._exp(
            covariates=[CovariateCfg(entity="sensor.temp", role="lagged")],
        )
        app = _make_app(tmp_db, exp, {
            "sensor.load": _recorder_rows(now, days=11),
            "sensor.temp": _recorder_rows(now, days=11, skip=cov_gap),
        })
        df = _run(app._fetch_and_preprocess(exp))
        feats = build_features(df, "y", interval_minutes=INTERVAL)
        combined, report = _supervised_frame(df, feats, exp)

        assert report["label_gap_rows"] == 0, (
            "a covariate gap must never cost a supervised row"
        )
        assert f"temp{MISSING_SUFFIX}" in report["indicator_cols"]
        assert not combined.isna().to_numpy().any()
        assert combined[f"temp{MISSING_SUFFIX}"].max() == 1.0
