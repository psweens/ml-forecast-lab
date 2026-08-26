"""The covariate manifest reports real coverage, with traffic lights.

`_log_covariate_manifest` is a pure synchronous method on the app, so it
can be driven directly with a hand-built `cov_stats` list. The whole block
is emitted as one `logger.info("\\n".join(lines))`, so assertions go
against `caplog.text` rather than per-record.
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab import main as main_mod
from ml_forecast_lab.config import CovariateCfg, ExperimentCfg
from ml_forecast_lab.main import MLForecastLabApp

GRID = 200
NOW = datetime(2026, 5, 5, 0, 0)


def _frame(*names) -> pd.DataFrame:
    idx = pd.date_range("2026-05-01", periods=GRID, freq="30min")
    rng = np.random.default_rng(7)
    data = {"y": np.arange(GRID) * 1.0}
    for n in names:
        data[n] = rng.normal(size=GRID)
    return pd.DataFrame(data, index=idx)


def _exp(n_covs: int = 1) -> ExperimentCfg:
    return ExperimentCfg(
        name="agile",
        target_entity="sensor.rate",
        interval_minutes=30,
        covariates=[
            CovariateCfg(entity=f"sensor.c{i}", role="lagged")
            for i in range(n_covs)
        ],
    )


def _stat(name="c0", entity="sensor.c0", observed=GRID, **extra) -> dict:
    stat = {
        "entity": entity,
        "name": name,
        "role": "lagged",
        "raw_count": observed,
        "aligned_count": GRID,
        "observed_count": observed,
        "filled_count": GRID - observed,
        "grid_points": GRID,
        "last_ts": None,
        "ok": True,
    }
    stat.update(extra)
    return stat


def _emit(app, caplog, cov_stats, result=None, exp=None,
          before=GRID, after=GRID, nan_counts=None):
    """``before``/``after`` are grid rows and supervised rows respectively.

    Covariate gaps no longer delete rows — they are masked, flagged and
    imputed — so the only thing that separates the two numbers is a target
    with no measurement, and ``nan_counts`` reports masked covariate cells
    rather than rows about to be dropped.
    """
    result = result if result is not None else _frame("c0")
    with caplog.at_level(logging.INFO, logger="ml_forecast_lab.main"):
        app._log_covariate_manifest(
            exp_cfg=exp or _exp(),
            cov_stats=cov_stats,
            result=result,
            now=NOW,
            grid_rows=before,
            supervised_rows=after,
            masked_counts=nan_counts if nan_counts is not None else {"c0": 0},
        )
    return caplog.text


@pytest.fixture
def app():
    return MLForecastLabApp()


class TestCoverageIsReportedHonestly:
    def test_mostly_filled_covariate_is_not_reported_as_full(self, app, caplog):
        text = _emit(app, caplog, [_stat(observed=3)])

        assert "obs=3/200" in text
        assert "(1.5%)" in text
        assert "100.0%" not in text

    def test_masked_count_is_shown(self, app, caplog):
        text = _emit(app, caplog, [_stat(observed=150)])

        assert "masked=50" in text

    def test_full_coverage_omits_the_masked_field(self, app, caplog):
        text = _emit(app, caplog, [_stat(observed=GRID)])

        assert "obs=200/200 (100.0%)" in text
        assert "masked=" not in text


class TestTrafficLights:
    def test_healthy_covariate_is_clean(self, app, caplog):
        text = _emit(app, caplog, [_stat(observed=GRID)])

        assert "✓ sensor.c0" in text
        assert "coverage" not in text
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_warn_threshold_marks_but_does_not_escalate(self, app, caplog):
        text = _emit(app, caplog, [_stat(observed=170)])  # 85%

        assert "⚠ sensor.c0" in text
        assert "coverage 85.0% < 90%" in text
        assert "⛔" not in text
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_alert_threshold_marks_and_escalates(self, app, caplog):
        text = _emit(app, caplog, [_stat(observed=60)])  # 30%

        assert "⛔ sensor.c0" in text
        assert "coverage 30.0% < 50%" in text
        warnings = [
            r for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warnings) == 1
        assert "sensor.c0 (30.0%)" in warnings[0].message

    def test_escalation_names_every_alerting_covariate(self, app, caplog):
        stats = [
            _stat(name="c0", entity="sensor.c0", observed=20),
            _stat(name="c1", entity="sensor.c1", observed=GRID),
            _stat(name="c2", entity="sensor.c2", observed=40),
        ]
        text = _emit(app, caplog, stats, result=_frame("c0", "c1", "c2"),
                     exp=_exp(3), nan_counts={"c0": 0})

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "2 covariate(s)" in warnings[0].message
        assert "sensor.c0" in warnings[0].message
        assert "sensor.c2" in warnings[0].message
        assert "sensor.c1" not in warnings[0].message
        assert "⚠ sensor.c1" not in text

    @pytest.mark.parametrize("observed,expected", [
        (180, "ok"),      # exactly 90.0%
        (179, "warn"),    # 89.5%
        (100, "warn"),    # exactly 50.0%
        (99, "alert"),    # 49.5%
    ])
    def test_thresholds_are_strict_less_than(self, app, caplog, observed, expected):
        text = _emit(app, caplog, [_stat(observed=observed)])

        if expected == "ok":
            assert "✓ sensor.c0" in text
            assert "coverage " not in text
        elif expected == "warn":
            assert "⚠ sensor.c0" in text
            assert "< 90%" in text
        else:
            assert "⛔ sensor.c0" in text
            assert "< 50%" in text

    def test_no_uppercase_level_tokens_inside_the_info_block(self, app, caplog):
        """The Logs tab colourises by naive substring, so an INFO manifest
        line containing 'WARNING' or 'ERROR' would re-tint the whole block."""
        _emit(app, caplog, [_stat(observed=10)])

        block = [
            r.message for r in caplog.records if r.levelno == logging.INFO
        ]
        assert block
        for msg in block:
            assert "WARNING" not in msg
            assert "ERROR" not in msg


class TestRobustness:
    def test_supervised_row_line_names_the_most_masked_covariate(
        self, app, caplog,
    ):
        """Rows are lost only to the target now — a covariate gap is masked
        and imputed, never a deletion — so the line reports supervised rows
        and reports the covariate's masked cells as a separate quantity."""
        text = _emit(
            app, caplog, [_stat(observed=GRID)],
            before=100, after=60, nan_counts={"c0": 40},
        )

        assert "supervised: 60 of 100 grid rows" in text
        assert "40 unmeasured target" in text
        assert "most-masked covariate: c0 40 cells" in text

    def test_failed_fetch_entry_does_not_raise(self, app, caplog):
        text = _emit(app, caplog, [{
            "entity": "sensor.broken", "name": "broken",
            "role": "lagged", "ok": False, "error": "boom",
        }])

        assert "✗ sensor.broken" in text
        assert "fetch failed: boom" in text

    def test_no_history_entry_is_visible(self, app, caplog):
        text = _emit(app, caplog, [
            _stat(observed=0, no_history=True, last_ts=None),
        ])

        assert "no history returned" in text
        assert "obs=0/200 (0.0%)" in text

    def test_column_fate_is_reported(self, app, caplog):
        text = _emit(app, caplog, [
            _stat(observed=0, fate="dropped", no_history=False),
        ])

        assert "column dropped" in text

    def test_entry_without_coverage_keys_falls_back(self, app, caplog):
        """A stats entry that predates the coverage keys must render a line
        rather than crash the cycle."""
        text = _emit(app, caplog, [{
            "entity": "sensor.legacy", "name": "c0", "role": "lagged",
            "raw_count": 10, "aligned_count": 200, "last_ts": None,
            "ok": True,
        }])

        assert "sensor.legacy" in text
        assert "cov~" in text

    def test_physics_entry_reads_full_coverage(self, app, caplog):
        idx = pd.date_range("2026-05-01", periods=GRID, freq="30min")
        result = pd.DataFrame({
            "y": np.arange(GRID) * 1.0,
            "sun_elevation": np.sin(np.arange(GRID) / 30.0),
        }, index=idx)
        text = _emit(app, caplog, [{
            "entity": "sun_elevation", "name": "sun_elevation",
            "role": "physics", "raw_count": GRID, "aligned_count": GRID,
            "observed_count": GRID, "filled_count": 0, "grid_points": GRID,
            "last_ts": None, "ok": True,
        }], result=result, exp=_exp(0), nan_counts={"sun_elevation": 0})

        assert "✓ sun_elevation [physics]  obs=200/200 (100.0%)" in text

    def test_header_counts_physics_not_failures(self, app, caplog):
        """A failed covariate also occupies a cov_stats row; it must not be
        reported as a physics feature."""
        text = _emit(app, caplog, [
            _stat(observed=GRID),
            {"entity": "sensor.broken", "name": "broken", "role": "lagged",
             "ok": False, "error": "boom"},
        ])

        assert "Covariate manifest for agile (1 configured):" in text
        assert "physics" not in text

    def test_zero_grid_points_does_not_divide_by_zero(self, app, caplog):
        text = _emit(
            app, caplog,
            [_stat(observed=0, grid_points=0, filled_count=0)],
            before=0, after=0, nan_counts={},
        )

        assert "sensor.c0" in text


class TestConstants:
    def test_thresholds_are_module_level(self):
        assert main_mod.COV_COVERAGE_WARN_PCT == 90.0
        assert main_mod.COV_COVERAGE_ALERT_PCT == 50.0
