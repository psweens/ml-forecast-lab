"""Pin the conformal band's quantile-level semantics (audit F2).

For a symmetric band built from ABSOLUTE residuals, realised coverage is
P(|y − ŷ| ≤ q̂) — i.e. exactly the quantile level used. The pre-v2.41.0
code applied the two-sided signed-residual rule (1 − α/2) to absolute
residuals, so nominal-80% bands realised ~90% coverage and were ~1.5×
wider than calibrated. These tests build a forecast_log with known
residuals and assert the returned quantile is the `level`-quantile of
|residual| and that the implied band covers ≈ `level` of the
calibration sample.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.db import HistoryDB


EXP = "conformal_exp"
ENTITY = "sensor.conformal_target"
INTERVAL_MIN = 30


@pytest.fixture
def seeded_db(tmp_path):
    """HistoryDB with ~14 days of actuals and one forecast per interval,
    each with a single h=1 target, so every residual lands in the same
    lead bucket and the quantile maths is directly checkable."""
    db = HistoryDB(tmp_path / "history.db")
    db.ensure_forecast_log_table()

    table = db.safe_table_name(ENTITY)
    rng = np.random.default_rng(7)
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=13)
    idx = pd.date_range(start, now, freq=f"{INTERVAL_MIN}min")
    actual = 100.0 + rng.normal(0, 10, len(idx))
    db.store_history(table, pd.DataFrame({"ds": idx, "value": actual}))

    residuals = []
    for i, ts in enumerate(idx[:-1]):
        target = idx[i + 1]
        resid = float(rng.normal(0, 10))
        residuals.append(abs(resid))
        db.log_forecast(
            EXP,
            ts.to_pydatetime(),
            [target.to_pydatetime()],
            [float(actual[i + 1]) + resid],
            "lightgbm",
            model_version="v1",
        )
    return db, table, np.array(residuals)


def test_quantile_is_level_quantile_of_abs_residuals(seeded_db):
    db, table, abs_resid = seeded_db
    for level in (0.8, 0.9):
        cq = db.get_conformal_quantiles(
            EXP, table, level=level,
            model_name="lightgbm", model_version="v1",
            interval_minutes=INTERVAL_MIN,
        )
        assert cq["total_samples"] > 100
        got = cq["fallback_quantile"]
        expected = float(np.quantile(abs_resid, level))
        wrong_old = float(np.quantile(abs_resid, 1 - (1 - level) / 2))
        # Within sampling tolerance of the correct quantile…
        assert got == pytest.approx(expected, rel=0.08), (
            f"level={level}: expected the {level:.0%} quantile of "
            f"|residual| ({expected:.2f}), got {got:.2f}"
        )
        # …and clearly NOT the old (1 − α/2) quantile.
        assert abs(got - wrong_old) > abs(got - expected), (
            f"level={level}: quantile {got:.2f} is closer to the "
            f"pre-v2.41.0 (1−α/2) value {wrong_old:.2f} than to the "
            f"correct {expected:.2f}"
        )


def test_band_realises_nominal_coverage(seeded_db):
    """The band pred ± q̂ must cover ≈ `level` of the calibration sample."""
    db, table, abs_resid = seeded_db
    level = 0.8
    cq = db.get_conformal_quantiles(
        EXP, table, level=level,
        model_name="lightgbm", model_version="v1",
        interval_minutes=INTERVAL_MIN,
    )
    q = cq["fallback_quantile"]
    realised = float(np.mean(abs_resid <= q))
    assert realised == pytest.approx(level, abs=0.05), (
        f"band ±{q:.2f} realises {realised:.1%} coverage on the "
        f"calibration residuals; nominal is {level:.0%}"
    )
