"""Unit tests for the Smart Setup auto-resolver (ml_forecast_lab.auto_config).

These lock down the data → persona → settings mapping that powers the
Settings-tab "Automatic" tier: a smooth solar-like cycle must NOT be treated
like a spiky hot-water load, and the spiky load must resolve to the
peak-preserving settings (Tweedie loss, outlier clipping off).
"""

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab import auto_config as ac


INTERVAL = 30
STEPS_PER_DAY = 1440 // INTERVAL  # 48
DAYS = 21


def _index(n):
    return pd.date_range("2024-01-01", periods=n, freq="30min")


def _smooth_solar_like(rng):
    """Daytime half-sine bump, zero at night, height varies day to day."""
    vals = []
    for d in range(DAYS):
        height = 5.0 + rng.uniform(-1.5, 1.5)  # cloud-driven day-to-day change
        for step in range(STEPS_PER_DAY):
            hour = step * INTERVAL / 60.0
            if 7.0 <= hour <= 19.0:
                phase = (hour - 7.0) / 12.0
                vals.append(max(0.0, height * np.sin(np.pi * phase)))
            else:
                vals.append(0.0)
    return pd.Series(vals, index=_index(len(vals)))


def _bursty_load(rng):
    """Mostly zero with a few sharp, randomly-timed spikes per day."""
    n = DAYS * STEPS_PER_DAY
    vals = np.zeros(n)
    for d in range(DAYS):
        for _ in range(rng.integers(1, 4)):
            pos = d * STEPS_PER_DAY + rng.integers(0, STEPS_PER_DAY)
            vals[pos] = rng.uniform(2.0, 10.0)
    return pd.Series(vals, index=_index(n))


# --------------------------------------------------------------------------- #
# Characterisation / persona
# --------------------------------------------------------------------------- #
def test_smooth_cycle_persona():
    rng = np.random.default_rng(0)
    prof = ac.characterize(_smooth_solar_like(rng), INTERVAL)
    assert prof.persona == "smooth_cycle"
    assert prof.daily_autocorr >= ac._SEASONAL
    assert prof.nonneg is True


def test_bursty_persona():
    rng = np.random.default_rng(1)
    prof = ac.characterize(_bursty_load(rng), INTERVAL)
    assert prof.persona == "bursty"
    assert prof.zero_fraction >= ac._INTERMITTENT
    assert prof.spikiness >= ac._SPIKY


def test_counts_persona():
    rng = np.random.default_rng(2)
    n = DAYS * STEPS_PER_DAY
    vals = rng.integers(0, 4, size=n).astype(float)
    prof = ac.characterize(pd.Series(vals, index=_index(n)), INTERVAL)
    assert prof.persona == "counts"
    assert prof.integerish is True


def test_empty_series_is_safe():
    prof = ac.characterize(pd.Series([], dtype=float), INTERVAL)
    assert prof.persona == "general"
    assert prof.n == 0


# --------------------------------------------------------------------------- #
# Resolution mapping
# --------------------------------------------------------------------------- #
def test_bursty_resolves_to_peak_preserving_settings():
    rng = np.random.default_rng(3)
    prof = ac.characterize(_bursty_load(rng), INTERVAL)
    res = ac.resolve(prof)
    assert res["loss_fn"].value == "tweedie"
    assert res["outlier_method"].value == "off"
    # every resolution carries a human-readable reason
    assert res["loss_fn"].reason
    assert res["outlier_method"].reason


def test_smooth_resolves_to_gentle_settings():
    rng = np.random.default_rng(4)
    prof = ac.characterize(_smooth_solar_like(rng), INTERVAL)
    res = ac.resolve(prof)
    assert res["loss_fn"].value == "huber"
    assert res["outlier_method"].value in ("quantile", "mad")
    assert res["production_metric"].value == "seasonal_mase"
    assert res["log_transform"].value is False


def test_guided_priority_overrides_persona():
    rng = np.random.default_rng(5)
    prof = ac.characterize(_smooth_solar_like(rng), INTERVAL)
    res = ac.resolve(prof, answers={"priority": "peaks"})
    # Even on a smooth signal, asking for peaks flips to peak-preserving.
    assert res["loss_fn"].value == "tweedie"
    assert res["outlier_method"].value == "off"


# --------------------------------------------------------------------------- #
# Report + pinning
# --------------------------------------------------------------------------- #
def test_report_respects_pinned_value():
    rng = np.random.default_rng(6)
    series = _bursty_load(rng)
    report = ac.resolve_settings_report(
        series, INTERVAL, pinned={"loss_fn": "mae", "outlier_method": ac.AUTO,
                                  "production_metric": ac.AUTO},
    )
    assert report["fields"]["loss_fn"]["source"] == "pinned"
    assert report["fields"]["loss_fn"]["value"] == "mae"
    assert report["fields"]["outlier_method"]["source"] == "automatic"
    assert report["n_pinned"] == 1


def test_report_has_persona_and_hints():
    rng = np.random.default_rng(7)
    report = ac.resolve_settings_report(_bursty_load(rng), INTERVAL)
    assert report["profile"]["persona"] == "bursty"
    assert isinstance(report["hints"], list) and report["hints"]


# --------------------------------------------------------------------------- #
# apply_to_experiment — in-place sentinel resolution + drift re-resolution
# --------------------------------------------------------------------------- #
class _FakeCfg:
    """Minimal stand-in for ExperimentCfg with the fields we resolve."""
    def __init__(self, **kw):
        self.name = "t"
        self.interval_minutes = INTERVAL
        self.source_is_cumulative = False
        self.loss_fn = kw.get("loss_fn", ac.AUTO)
        self.outlier_method = kw.get("outlier_method", ac.AUTO)
        self.production_metric = kw.get("production_metric", ac.AUTO)


def test_apply_resolves_auto_fields_in_place():
    rng = np.random.default_rng(8)
    cfg = _FakeCfg()
    report = ac.apply_to_experiment(cfg, _bursty_load(rng))
    # 'auto' sentinels are replaced with concrete resolved values...
    assert cfg.loss_fn == "tweedie"
    assert cfg.outlier_method == "off"
    assert cfg.production_metric in ("mae", "seasonal_mase")
    # ...and the report is available for the UI preview.
    assert report["profile"]["persona"] == "bursty"
    assert getattr(cfg, "_auto_resolution") is report


def test_apply_leaves_pinned_fields_untouched():
    rng = np.random.default_rng(9)
    cfg = _FakeCfg(loss_fn="mse")  # pinned
    ac.apply_to_experiment(cfg, _bursty_load(rng))
    assert cfg.loss_fn == "mse"           # pin survives
    assert cfg.outlier_method == "off"    # still auto-resolved


def test_apply_reresolves_each_call_via_sentinel_memory():
    # First on a bursty series → tweedie/off. Then the SAME cfg object on a
    # smooth series must re-resolve (the sentinel is remembered), not stay
    # frozen at the first cycle's concrete value.
    cfg = _FakeCfg()
    ac.apply_to_experiment(cfg, _bursty_load(np.random.default_rng(10)))
    assert cfg.loss_fn == "tweedie"
    ac.apply_to_experiment(cfg, _smooth_solar_like(np.random.default_rng(11)))
    assert cfg.loss_fn == "huber"
    assert cfg.outlier_method in ("quantile", "mad")


def test_apply_is_safe_on_garbage_series():
    cfg = _FakeCfg()
    # All-NaN series must not raise and must leave usable concrete values.
    ac.apply_to_experiment(cfg, pd.Series([np.nan] * 10, index=_index(10)))
    assert cfg.loss_fn in ("huber", "mse", "mae", "tweedie")
    assert cfg.outlier_method in ("quantile", "mad", "off")


# --------------------------------------------------------------------------- #
# Model-dependent loss — the resolved loss carries a per-family breakdown so
# the UI can show that the same intent means different things to each backend.
# --------------------------------------------------------------------------- #
def test_loss_resolution_is_model_dependent():
    rng = np.random.default_rng(12)
    prof = ac.characterize(_bursty_load(rng), INTERVAL)
    res = ac.resolve(prof)["loss_fn"]
    # Single applied value stays Tweedie (what trees + the config plumbing use)…
    assert res.value == "tweedie"
    # …but the breakdown spells out that neural backends fall back to Huber and
    # classical / zero-shot backends don't use a point loss at all.
    assert res.detail is not None
    fams = res.detail
    tree = fams[ac._LOSS_FAMILY_LABELS["tree"]]
    neural = fams[ac._LOSS_FAMILY_LABELS["neural"]]
    assert tree == "tweedie"
    assert neural == "huber"          # no native torch Tweedie loss
    assert "objective" in fams[ac._LOSS_FAMILY_LABELS["classical"]].lower()
    assert fams[ac._LOSS_FAMILY_LABELS["foundation"]]  # non-empty note
    # And it survives serialisation for the web preview.
    assert "detail" in res.to_dict()


def test_loss_family_helper_maps_tweedie_to_huber_for_neural():
    d = ac._loss_by_family("tweedie")
    assert d[ac._LOSS_FAMILY_LABELS["tree"]] == "tweedie"
    assert d[ac._LOSS_FAMILY_LABELS["neural"]] == "huber"
    # A non-Tweedie intent applies uniformly to trees and neural.
    d2 = ac._loss_by_family("huber")
    assert d2[ac._LOSS_FAMILY_LABELS["tree"]] == "huber"
    assert d2[ac._LOSS_FAMILY_LABELS["neural"]] == "huber"
