"""True covariate coverage, measured before the gap fill (v2.50.0).

`_fetch_and_preprocess` forward- and back-fills every covariate onto the
target grid so a gap can never delete a training row. The manifest then
counted non-NaN values *after* that fill, which is 100% by construction —
so a covariate with ten days of history against a two-year window claimed
full coverage on about 1.4% real data.

These tests pin the replacement measurement, and — just as importantly —
pin that the fill itself is untouched.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.covariates import CovariateResolver
from ml_forecast_lab.main import (
    COV_COVERAGE_ALERT_PCT,
    COV_COVERAGE_WARN_PCT,
    _covariate_grid_coverage,
)

INTERVAL = 30
FREQ = "30min"


def _grid(days: int = 14, start: str = "2026-05-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=days * (1440 // INTERVAL) + 1, freq=FREQ)


def _resampled(index: pd.DatetimeIndex, binary: bool = False) -> pd.Series:
    """Push a synthetic covariate through the real resampler, so these
    tests measure what `fetch_history` actually hands to the merge."""
    resolver = CovariateResolver(None)
    if binary:
        values = (np.arange(len(index)) % 2) * 1.0
    else:
        values = np.arange(len(index)) * 1.0
    return resolver._resample_covariate(
        pd.Series(values, index=index), FREQ, is_binary=binary or None,
    )


def _pct(observed: int, grid: pd.DatetimeIndex) -> float:
    return 100.0 * observed / len(grid)


class TestHeadlineCase:
    """The bug the spec is about."""

    def test_short_history_against_long_window_reports_true_coverage(self):
        grid = _grid(days=730, start="2024-01-01")
        cov = _resampled(
            pd.date_range(grid[-1] - pd.Timedelta(days=10), grid[-1], freq=FREQ)
        )

        observed, filled = _covariate_grid_coverage(cov, grid, INTERVAL)

        assert 1.3 <= _pct(observed, grid) <= 1.5, (
            f"ten days against a two-year window should read ~1.4%, "
            f"got {_pct(observed, grid):.2f}%"
        )
        assert observed + filled == len(grid)

    def test_the_old_measurement_would_have_said_one_hundred_percent(self):
        """Pins the defect itself, so a regression is unmistakable."""
        grid = _grid(days=730, start="2024-01-01")
        cov = _resampled(
            pd.date_range(grid[-1] - pd.Timedelta(days=10), grid[-1], freq=FREQ)
        )

        post_fill = cov.reindex(grid, method="ffill").ffill().bfill()

        assert int(post_fill.notna().sum()) == len(grid), (
            "the post-fill column is non-NaN everywhere — which is exactly "
            "why counting it reported full coverage"
        )


class TestHealthyCovariatesAreNotFlagged:
    """A diagnostic that cries wolf is a diagnostic nobody reads."""

    @pytest.mark.parametrize("cadence", ["30min", "1h", "3h", "6h"])
    def test_healthy_cadence_reads_full_coverage(self, cadence):
        grid = _grid()
        cov = _resampled(pd.date_range(grid[0], grid[-1], freq=cadence))

        observed, filled = _covariate_grid_coverage(cov, grid, INTERVAL)

        assert _pct(observed, grid) == pytest.approx(100.0), (
            f"a healthy {cadence} covariate on a {INTERVAL}min grid must not "
            f"be flagged; got {_pct(observed, grid):.1f}%"
        )
        assert filled == 0

    def test_jittery_on_change_sensor_is_not_flagged(self):
        """HA's recorder is delta-storage, so update spacing is irregular."""
        rng = np.random.default_rng(1)
        grid = _grid()
        offsets = np.cumsum(rng.integers(20, 75, 700))
        idx = pd.DatetimeIndex(grid[0] + pd.to_timedelta(offsets, unit="m"))
        idx = idx[idx <= grid[-1]]
        cov = _resampled(idx)

        observed, _ = _covariate_grid_coverage(cov, grid, INTERVAL)

        assert _pct(observed, grid) >= COV_COVERAGE_WARN_PCT

    def test_non_divisor_interval_is_phase_agnostic(self):
        """interval_minutes is validated only as >= 1, so 50 is reachable.

        Target and covariate are resampled independently, each anchored to
        midnight of its own first day, so for an interval that does not
        divide a day the two grids share no label at all. A bare reindex
        scores a dense, healthy covariate at exactly zero.
        """
        grid = pd.Series(
            1.0, index=pd.date_range("2026-05-01", "2026-05-15", freq="1min"),
        ).resample("50min").mean().index
        cov = pd.Series(
            1.0, index=pd.date_range("2026-05-08 00:07", "2026-05-15", freq="1min"),
        ).resample("50min").mean()

        observed, _ = _covariate_grid_coverage(cov, grid, 50)

        assert int(cov.reindex(grid).notna().sum()) == 0, (
            "precondition: exact-label alignment finds nothing here"
        )
        assert observed >= 0.45 * len(grid), (
            f"a dense covariate covering half the window should read ~50%, "
            f"got {_pct(observed, grid):.1f}%"
        )


class TestRealShortfallsAreCaught:
    def test_covariate_that_stopped_reporting(self):
        grid = _grid()
        cov = _resampled(
            pd.date_range(grid[0], grid[-1] - pd.Timedelta(days=4), freq=FREQ)
        )

        observed, _ = _covariate_grid_coverage(cov, grid, INTERVAL)

        assert 70.0 <= _pct(observed, grid) <= 74.0

    def test_interior_outage(self):
        grid = _grid()
        idx = pd.date_range(
            grid[0], grid[0] + pd.Timedelta(days=5), freq=FREQ,
        ).append(
            pd.date_range(grid[0] + pd.Timedelta(days=8), grid[-1], freq=FREQ)
        )
        cov = _resampled(idx)

        observed, _ = _covariate_grid_coverage(cov, grid, INTERVAL)

        assert 77.0 <= _pct(observed, grid) <= 81.0

    def test_covariate_installed_part_way_through_the_window(self):
        grid = _grid()
        cov = _resampled(
            pd.date_range(grid[-1] - pd.Timedelta(days=3), grid[-1], freq=FREQ)
        )

        observed, _ = _covariate_grid_coverage(cov, grid, INTERVAL)

        assert _pct(observed, grid) < COV_COVERAGE_ALERT_PCT
        assert 20.0 <= _pct(observed, grid) <= 23.0


class TestBinarySemantics:
    """A binary covariate is resampled upstream with last().ffill(), so
    `observed_count` measures span. That is the right reading for a step
    function under a delta-storage recorder: "no row" means "did not
    move", not "unknown"."""

    def test_healthy_binary_reads_full_coverage(self):
        grid = _grid()
        cov = _resampled(
            pd.date_range(grid[0], grid[-1], freq="6h"), binary=True,
        )

        observed, _ = _covariate_grid_coverage(cov, grid, INTERVAL)

        assert _pct(observed, grid) == pytest.approx(100.0)

    def test_binary_installed_part_way_is_still_caught(self):
        grid = _grid()
        cov = _resampled(
            pd.date_range(grid[-1] - pd.Timedelta(days=3), grid[-1], freq="6h"),
            binary=True,
        )

        observed, _ = _covariate_grid_coverage(cov, grid, INTERVAL)

        assert _pct(observed, grid) < COV_COVERAGE_ALERT_PCT


class TestEdgeCases:
    def test_empty_series(self):
        grid = _grid()
        assert _covariate_grid_coverage(
            pd.Series(dtype=float), grid, INTERVAL,
        ) == (0, len(grid))

    def test_all_nan_series(self):
        grid = _grid()
        cov = pd.Series(np.nan, index=grid)
        assert _covariate_grid_coverage(cov, grid, INTERVAL) == (0, len(grid))

    def test_none_series(self):
        grid = _grid()
        assert _covariate_grid_coverage(None, grid, INTERVAL) == (0, len(grid))

    def test_empty_grid(self):
        cov = _resampled(pd.date_range("2026-05-01", periods=10, freq=FREQ))
        assert _covariate_grid_coverage(
            cov, pd.DatetimeIndex([]), INTERVAL,
        ) == (0, 0)

    def test_single_observation(self):
        grid = _grid()
        cov = pd.Series([1.0], index=[grid[0]])
        observed, filled = _covariate_grid_coverage(cov, grid, INTERVAL)
        assert observed + filled == len(grid)
        assert observed < len(grid)

    def test_zero_interval_does_not_divide_by_zero(self):
        grid = _grid()
        cov = _resampled(pd.date_range(grid[0], grid[-1], freq=FREQ))
        observed, filled = _covariate_grid_coverage(cov, grid, 0)
        assert observed + filled == len(grid)

    @pytest.mark.parametrize("days_present", [1, 3, 7, 14])
    def test_counts_always_partition_the_grid(self, days_present):
        grid = _grid()
        cov = _resampled(
            pd.date_range(
                grid[-1] - pd.Timedelta(days=days_present), grid[-1], freq=FREQ,
            )
        )
        observed, filled = _covariate_grid_coverage(cov, grid, INTERVAL)
        assert observed + filled == len(grid)
        assert observed >= 0 and filled >= 0

    @pytest.mark.parametrize("cov_tz,grid_tz", [("UTC", None), (None, "UTC")])
    def test_timezone_mismatch_does_not_read_as_zero(self, cov_tz, grid_tz):
        """A tz mismatch used to make reindex raise and the count fall
        through to 0% — an alert about a covariate that is fully healthy."""
        base = pd.date_range("2026-05-01", periods=100, freq=FREQ)
        grid = base.tz_localize(grid_tz) if grid_tz else base
        cov_index = base.tz_localize(cov_tz) if cov_tz else base
        cov = pd.Series(np.arange(100.0), index=cov_index)

        observed, filled = _covariate_grid_coverage(cov, grid, INTERVAL)

        assert observed == 100 and filled == 0

    def test_duplicate_labels_do_not_raise(self):
        """`reindex` refuses an axis with duplicate labels. This is a
        diagnostic — it must never be what takes a covariate out of
        training."""
        grid = _grid(days=2)
        cov = pd.Series(
            [1.0, 2.0, 3.0], index=[grid[0], grid[0], grid[1]],
        )

        observed, filled = _covariate_grid_coverage(cov, grid, INTERVAL)

        assert observed + filled == len(grid)

    def test_unsorted_indexes_do_not_raise(self):
        grid = _grid(days=2)
        cov = pd.Series(np.arange(len(grid)) * 1.0, index=grid[::-1])

        observed, filled = _covariate_grid_coverage(cov, grid, INTERVAL)

        assert observed + filled == len(grid)
        assert observed == len(grid)

    @pytest.mark.parametrize("bad", [None, float("nan"), -5])
    def test_unusable_interval_does_not_raise(self, bad):
        grid = _grid(days=2)
        cov = _resampled(pd.date_range(grid[0], grid[-1], freq=FREQ))

        observed, filled = _covariate_grid_coverage(cov, grid, bad)

        assert observed + filled == len(grid)

    def test_measurement_does_not_mutate_its_input(self):
        grid = _grid()
        cov = _resampled(
            pd.date_range(grid[-1] - pd.Timedelta(days=3), grid[-1], freq=FREQ)
        )
        before = cov.copy(deep=True)

        _covariate_grid_coverage(cov, grid, INTERVAL)

        pd.testing.assert_series_equal(cov, before)

    def test_measurement_does_not_mutate_a_tz_aware_input(self):
        """The tz-alignment branch is the one that touches `.index`."""
        grid = _grid(days=2)
        cov = pd.Series(
            np.arange(len(grid)) * 1.0, index=grid.tz_localize("UTC"),
        )
        before = cov.copy(deep=True)

        _covariate_grid_coverage(cov, grid, INTERVAL)

        pd.testing.assert_series_equal(cov, before)
        assert cov.index.tz is not None


class TestFillBehaviourUnchanged:
    """The spec is explicit: filling, dropping and feature construction do
    not change — only the measurement does. This is the cheap direct guard
    on that, following the source-contract precedent in
    tests/unit/test_preprocessing.py."""

    @staticmethod
    def _main_source() -> str:
        root = Path(__file__).resolve().parents[2]
        return (root / "ml_forecast_lab" / "main.py").read_text()

    def test_covariate_fill_expressions_are_verbatim(self):
        src = self._main_source()
        assert 'cov_series.reindex(result.index, method="ffill")' in src
        assert "cov_aligned = cov_aligned.ffill().bfill()" in src

    def test_measurement_happens_before_the_fill(self):
        src = self._main_source()
        measure = src.index("observed_count, filled_count = _covariate_grid_coverage")
        fill = src.index('cov_aligned = cov_series.reindex(result.index, method="ffill")')
        assert measure < fill, (
            "coverage must be measured before the fill, or it measures the fill"
        )

    def test_thresholds_are_fixed_constants(self):
        assert COV_COVERAGE_WARN_PCT == 90.0
        assert COV_COVERAGE_ALERT_PCT == 50.0

    def test_thresholds_are_not_read_from_config(self):
        """They are diagnostic thresholds, not tuning knobs — a
        configurable threshold is a configurable way to silence the
        warning."""
        root = Path(__file__).resolve().parents[2]
        cfg = (root / "ml_forecast_lab" / "config.py").read_text()
        assert not re.search(r"coverage_warn|coverage_alert", cfg, re.I)


# ---------------------------------------------------------------------
# End-to-end: the training frame itself must not move
# ---------------------------------------------------------------------

import asyncio  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

from ml_forecast_lab.config import AppConfig, CovariateCfg, ExperimentCfg  # noqa: E402
from ml_forecast_lab.db import HistoryDB  # noqa: E402
from ml_forecast_lab.main import MLForecastLabApp  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


class _StubHA:
    """Serves synthetic recorder rows per entity, sliced to the requested
    window — enough of HAInterface for `_fetch_and_preprocess`."""

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


def _recorder_rows(entity_now, days, cadence_min=15, value=lambda i: i % 37 + 1.0):
    ts = pd.date_range(
        entity_now - timedelta(days=days), entity_now,
        freq=f"{cadence_min}min", tz="UTC",
    )
    return [
        {"last_changed": t.isoformat(), "state": f"{value(i):.4f}"}
        for i, t in enumerate(ts)
    ]


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


class TestTrainingFrameUnchanged:
    """The spec's regression guard: for an experiment with a complete
    covariate, the training rows must be exactly what they were before the
    measurement was added — and caching must not change them either."""

    @staticmethod
    def _exp():
        return ExperimentCfg(
            name="parity",
            target_entity="sensor.load",
            days_history=5,
            interval_minutes=30,
            covariates=[CovariateCfg(entity="sensor.temp", role="lagged")],
            models_enabled=["lightgbm"],
        )

    def _rows(self):
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        return {
            "sensor.load": _recorder_rows(now, days=6),
            "sensor.temp": _recorder_rows(
                now, days=6, value=lambda i: (i % 17) * 0.5 + 3.0,
            ),
        }

    def test_complete_covariate_produces_a_full_frame(self, tmp_db):
        exp = self._exp()
        app = _make_app(tmp_db, exp, self._rows())

        result = _run(app._fetch_and_preprocess(exp))

        assert result is not None and len(result) > 200
        assert "temp" in result.columns
        assert int(result["temp"].isna().sum()) == 0
        expected_span = timedelta(days=exp.days_history)
        assert (result.index[-1] - result.index[0]) <= expected_span

    def test_merged_values_are_exactly_the_untouched_fill(self, tmp_db):
        """The covariate column must equal reindex(ffill) + ffill + bfill —
        the measurement must not have altered a single value."""
        exp = self._exp()
        app = _make_app(tmp_db, exp, self._rows())

        result = _run(app._fetch_and_preprocess(exp))

        raw = _run(app.covariate_resolver.fetch_history(
            {"entity_id": "sensor.temp", "name": "temp"},
            result.index[0].tz_localize("UTC"),
            datetime.now(timezone.utc),
            "30min",
        ))
        expected = raw.reindex(result.index, method="ffill").ffill().bfill()

        pd.testing.assert_series_equal(
            result["temp"], expected.loc[result.index], check_names=False,
        )

    def test_warm_cache_yields_an_identical_frame(self, tmp_db):
        """Caching is a fetch optimisation, not a data change."""
        exp = self._exp()
        app = _make_app(tmp_db, exp, self._rows())

        cold = _run(app._fetch_and_preprocess(exp))
        warm = _run(app._fetch_and_preprocess(exp))

        assert len(cold) == len(warm)
        common = cold.index.intersection(warm.index)
        assert len(common) >= len(cold) - 2  # at most one grid step of drift
        pd.testing.assert_frame_equal(
            cold.loc[common], warm.loc[common], check_freq=False,
        )

    def test_manifest_reports_full_coverage_for_a_complete_covariate(
        self, tmp_db, caplog,
    ):
        import logging

        exp = self._exp()
        app = _make_app(tmp_db, exp, self._rows())

        with caplog.at_level(logging.INFO, logger="ml_forecast_lab.main"):
            _run(app._fetch_and_preprocess(exp))

        assert "(100.0%)" in caplog.text
        assert [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and "coverage below" in r.message
        ] == []

    def test_manifest_alerts_on_a_covariate_with_a_short_history(
        self, tmp_db, caplog,
    ):
        """The end-to-end version of the headline bug: a covariate whose
        recorder history covers a fraction of the target window."""
        import logging

        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        exp = ExperimentCfg(
            name="short_cov",
            target_entity="sensor.load",
            days_history=10,
            interval_minutes=30,
            covariates=[CovariateCfg(entity="sensor.new", role="lagged")],
            models_enabled=["lightgbm"],
        )
        app = _make_app(tmp_db, exp, {
            "sensor.load": _recorder_rows(now, days=11),
            "sensor.new": _recorder_rows(now, days=1),
        })

        with caplog.at_level(logging.INFO, logger="ml_forecast_lab.main"):
            result = _run(app._fetch_and_preprocess(exp))

        assert result is not None and len(result) > 400, (
            "the fill is unchanged, so no training row is lost"
        )
        assert int(result["new"].isna().sum()) == 0
        assert "⛔ sensor.new" in caplog.text
        assert "(100.0%)" not in caplog.text.split("sensor.new")[1][:120]
        alerts = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "coverage below" in r.message
        ]
        assert len(alerts) == 1
        assert "sensor.new" in alerts[0].message
