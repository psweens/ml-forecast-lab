"""
Automatic, data-driven resolution of experiment parameters.

This is the engine behind the **Smart Setup** tier model on the experiment
Settings tab. A non-ML user should never have to know what ``tweedie`` or
``huber`` means: they leave data-dependent settings on **Automatic**, the
app characterises their sensor's history, and picks a sensible value *with a
plain-English reason*. Any setting can still be pinned to an explicit value
(an override) or reset back to Automatic.

Design notes
------------
* The three "managed" settings (``loss_fn``, ``outlier_method``,
  ``production_metric``) use the literal sentinel ``"auto"`` as their value.
  When a setting holds the sentinel its concrete value is resolved at
  training time from the data — so the choice tracks the sensor as its
  behaviour drifts, rather than being frozen at config time. This mirrors
  the pre-existing ``output_activation: 'auto'`` / ``outlier_lower: 'auto'``
  convention already in the codebase.
* Resolution is **pure** (no I/O): :func:`characterize` turns a series into a
  :class:`DataProfile`; :func:`resolve` turns a profile (plus optional guided
  answers) into a ``{field: Resolution}`` map. The orchestrator wires the
  data in and applies the result; the web layer renders it.
* Everything is wrapped defensively by callers: a profiling failure must
  never break a training run, so unresolved sentinels fall back to the
  conservative defaults (``huber`` / ``quantile`` / ``seasonal_mase``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The sentinel that means "resolve this from the data at training time".
AUTO = "auto"

# Settings this module owns end-to-end (sentinel → resolved value at runtime).
MANAGED_FIELDS = ("loss_fn", "outlier_method", "production_metric")

# Conservative fallbacks used when profiling fails or data is too thin.
SAFE_DEFAULTS = {
    "loss_fn": "huber",
    "outlier_method": "quantile",
    "production_metric": "seasonal_mase",
}


# --------------------------------------------------------------------------- #
# Personas — the user-facing "what kind of thing is this sensor?" categories.
# Each is a bundle of parameter intent; the resolver maps a DataProfile onto
# exactly one of these, and the UI shows the label/blurb so the user can
# sanity-check (and correct) the auto-detected type.
# --------------------------------------------------------------------------- #
PERSONAS: Dict[str, Dict[str, str]] = {
    "smooth_cycle": {
        "label": "Smooth daily cycle",
        "blurb": "Follows a repeating, smooth daily shape (solar generation, "
                 "temperature, outdoor light). The timing is predictable; only "
                 "the height changes day to day.",
    },
    "bursty": {
        "label": "Bursts on and off",
        "blurb": "Mostly idle with occasional sharp spikes at hard-to-predict "
                 "times (hot-water reheat, EV charging, kettle, pumps). The "
                 "spikes are the signal — not noise.",
    },
    "baseline_plus_spikes": {
        "label": "Steady baseline with occasional spikes",
        "blurb": "A continuous base level (whole-home load) with intermittent "
                 "large draws layered on top.",
    },
    "counts": {
        "label": "Counts or occupancy",
        "blurb": "Small whole-number values (people present, door opens, "
                 "events per interval).",
    },
    "general": {
        "label": "General signal",
        "blurb": "No strong structure detected — using safe, robust defaults.",
    },
}


@dataclass
class DataProfile:
    """Quantitative fingerprint of a target series, plus the chosen persona."""

    n: int
    nonneg: bool
    zero_fraction: float
    spikiness: float          # positive-conditional peak ratio (p99 / median⁺)
    cv: float                 # coefficient of variation
    daily_autocorr: float     # autocorrelation at the daily lag (−1..1)
    span_orders: float        # orders of magnitude spanned by positive values
    integerish: bool
    max_value: float
    persona: str = "general"

    @property
    def persona_label(self) -> str:
        return PERSONAS.get(self.persona, PERSONAS["general"])["label"]

    @property
    def persona_blurb(self) -> str:
        return PERSONAS.get(self.persona, PERSONAS["general"])["blurb"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "nonneg": self.nonneg,
            "zero_fraction": round(self.zero_fraction, 3),
            "spikiness": round(self.spikiness, 2),
            "cv": round(self.cv, 2),
            "daily_autocorr": round(self.daily_autocorr, 2),
            "span_orders": round(self.span_orders, 2),
            "integerish": self.integerish,
            "max_value": round(self.max_value, 4),
            "persona": self.persona,
            "persona_label": self.persona_label,
            "persona_blurb": self.persona_blurb,
        }


@dataclass
class Resolution:
    """A resolved (or pinned) setting value with a human-readable reason."""

    value: Any
    reason: str
    source: str = "automatic"   # "automatic" | "pinned"

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "reason": self.reason, "source": self.source}


# Thresholds — deliberately gentle. These are the knobs a maintainer would
# tune against real household sensors; named constants keep them honest.
_SPIKY = 3.5            # p99 / mean above this ⇒ "sharp" / spike-driven
_SMOOTH_SPIKY_CEIL = 6.0  # a phase-locked cycle still counts as smooth below this
_SEASONAL = 0.45        # daily autocorrelation above this ⇒ strong daily cycle
_INTERMITTENT = 0.40    # zero-fraction above this ⇒ "mostly off"


def characterize(
    series: pd.Series,
    interval_minutes: int,
    source_is_cumulative: bool = False,
) -> DataProfile:
    """Build a :class:`DataProfile` from a per-interval target series.

    ``series`` is expected to be the per-interval signal the model actually
    trains on (i.e. *after* any cumulative→interval conversion), and *before*
    outlier clipping / log transform so the fingerprint reflects the true
    sensor rather than a pre-squashed version of it.
    """
    y = pd.to_numeric(pd.Series(series), errors="coerce").dropna().to_numpy(dtype=float)
    n = int(y.size)
    if n == 0:
        return DataProfile(0, True, 0.0, 0.0, 0.0, 0.0, 0.0, False, 0.0, "general")

    abs_y = np.abs(y)
    scale = float(np.median(abs_y[abs_y > 0])) if np.any(abs_y > 0) else 0.0
    eps = max(1e-9, 1e-3 * scale)

    nonneg = bool(np.nanmin(y) >= -eps)
    zero_fraction = float(np.mean(abs_y <= eps))

    positives = y[y > eps]
    p99 = float(np.quantile(y, 0.99))
    mean_abs = float(np.mean(abs_y))
    # Peak-to-mean ratio: how concentrated the mass is. A mostly-off spike
    # train (hot water, EV) has a tiny mean and a large p99 ⇒ high; a smooth
    # daytime bump (solar) spreads its mass ⇒ low — even though both spend
    # ~half the day at zero. This is what separates "sharp" from "smooth".
    spikiness = p99 / (mean_abs + eps)

    cv = float(np.std(y) / (mean_abs + eps))
    max_value = float(np.max(y))

    # Orders of magnitude spanned by the positive mass (drives log-transform).
    if positives.size:
        lo = float(np.quantile(positives, 0.05))
        hi = float(np.quantile(positives, 0.99))
        span_orders = float(np.log10(hi / lo)) if lo > 0 and hi > lo else 0.0
    else:
        span_orders = 0.0

    # Integer-valued, small-range ⇒ counts/occupancy.
    integerish = bool(np.all(np.abs(y - np.round(y)) < 1e-6))

    # Autocorrelation at the daily lag separates a smooth, phase-locked daily
    # cycle (solar: high) from an intermittent spike train whose timing wanders
    # (hot water: low) — even when both have a similar zero-fraction.
    steps_per_day = max(1, int(round(1440 / max(interval_minutes, 1))))
    daily_autocorr = _autocorr(y, steps_per_day)

    profile = DataProfile(
        n=n,
        nonneg=nonneg,
        zero_fraction=zero_fraction,
        spikiness=float(spikiness),
        cv=cv,
        daily_autocorr=daily_autocorr,
        span_orders=span_orders,
        integerish=integerish,
        max_value=max_value,
    )
    profile.persona = _pick_persona(profile)
    return profile


def _autocorr(y: np.ndarray, lag: int) -> float:
    """Pearson autocorrelation of ``y`` at ``lag`` (0.0 if undeterminable)."""
    if lag < 1 or y.size <= 2 * lag:
        return 0.0
    a = y[:-lag]
    b = y[lag:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if np.isnan(r) else r


def _pick_persona(p: DataProfile) -> str:
    if p.integerish and p.nonneg and 0 < p.max_value <= 50 and p.zero_fraction > 0.05:
        return "counts"
    # Smooth, phase-locked daily cycle (solar, temperature). High daily
    # autocorrelation AND not extremely sharp — night-time zeros are fine.
    if p.daily_autocorr >= _SEASONAL and p.spikiness < _SMOOTH_SPIKY_CEIL:
        return "smooth_cycle"
    # Sharp + mostly-off ⇒ bursty event load (hot water, EV, appliances).
    if p.spikiness >= _SPIKY and p.zero_fraction >= _INTERMITTENT:
        return "bursty"
    # Sharp but with a continuous base level ⇒ whole-home style.
    if p.spikiness >= _SPIKY:
        return "baseline_plus_spikes"
    return "general"


def resolve(
    profile: DataProfile,
    answers: Optional[Dict[str, str]] = None,
) -> Dict[str, Resolution]:
    """Map a :class:`DataProfile` (+ optional guided answers) to settings.

    Returns a map covering every managed field plus a few **advisory** fields
    (``log_transform``, ``outlier_lower``) the UI surfaces as recommendations.
    Guided ``answers`` (e.g. ``{"priority": "peaks"}``) take precedence over
    the persona default for the fields they touch.
    """
    answers = answers or {}
    persona = profile.persona
    out: Dict[str, Resolution] = {}

    # ---- loss_fn ----------------------------------------------------------
    if persona in ("bursty", "counts"):
        out["loss_fn"] = Resolution(
            "tweedie",
            f"{int(profile.zero_fraction * 100)}% of readings are ~zero with "
            f"sharp spikes — Tweedie models the off-state and the spike size "
            f"together, where plain MAE/MSE would flatten the peaks.",
        )
    elif persona == "smooth_cycle":
        out["loss_fn"] = Resolution(
            "huber", "Smooth, well-behaved signal — Huber is a robust default."
        )
    elif persona == "baseline_plus_spikes":
        out["loss_fn"] = Resolution(
            "huber", "Continuous baseline with occasional spikes — Huber stays "
            "robust to the spikes without ignoring the baseline.",
        )
    else:
        out["loss_fn"] = Resolution("huber", "Robust general-purpose default.")

    # ---- outlier_method ---------------------------------------------------
    # Persona-keyed, not raw-spikiness-keyed: a smooth daytime bump has a
    # high peak-to-mean ratio too (half the day is zero), but its peaks are
    # smooth and benefit from a gentle trim — unlike a true event load whose
    # sharp spikes ARE the signal and must not be clipped.
    if persona in ("bursty", "baseline_plus_spikes", "counts"):
        out["outlier_method"] = Resolution(
            "off", "The peaks are real signal, not noise — clipping outliers "
            "would erase exactly what you want to predict.",
        )
    elif persona == "smooth_cycle":
        out["outlier_method"] = Resolution(
            "quantile", "Smooth signal — a gentle top-quantile trim removes "
            "sensor glitches without touching the shape.",
        )
    else:
        out["outlier_method"] = Resolution(
            "mad", "Robust (median-based) clipping — trims glitches while "
            "tolerating a heavy tail.",
        )

    # ---- production_metric (champion selection) ---------------------------
    if profile.daily_autocorr >= 0.3:
        out["production_metric"] = Resolution(
            "seasonal_mase", "Strong daily pattern — scoring against "
            "same-time-yesterday is the meaningful skill measure.",
        )
    else:
        out["production_metric"] = Resolution(
            "mae", "No strong daily cycle — plain MAE is the clearest "
            "head-to-head.",
        )

    # ---- advisory: log_transform -----------------------------------------
    if (profile.zero_fraction < 0.02 and profile.span_orders >= 2.5
            and profile.nonneg and profile.spikiness < _SPIKY):
        out["log_transform"] = Resolution(
            True, "Strictly positive and spanning several orders of magnitude "
            "— a log transform balances small and large values.",
        )
    else:
        out["log_transform"] = Resolution(
            False, "Off — a log transform would compress the peaks (and "
            "near-zero values make it ill-behaved).",
        )

    # ---- advisory: outlier_lower -----------------------------------------
    out["outlier_lower"] = Resolution(
        "zero" if profile.nonneg else "symmetric",
        "Floor at zero for a non-negative quantity." if profile.nonneg
        else "Two-sided trim for a signed signal.",
    )

    # ---- guided answers override the persona defaults --------------------
    priority = answers.get("priority")
    if priority == "peaks":
        out["loss_fn"] = Resolution(
            "tweedie", "You chose to prioritise catching the peaks — Tweedie "
            "lets the model reach toward the spikes.", source="automatic",
        )
        out["outlier_method"] = Resolution(
            "off", "Prioritising peaks — outlier clipping is disabled so real "
            "spikes survive.", source="automatic",
        )
    elif priority == "average":
        out["loss_fn"] = Resolution(
            "huber", "You chose accuracy-on-average — Huber targets the "
            "typical value and resists being pulled by spikes.",
        )
        out["outlier_method"] = Resolution(
            "quantile", "Accuracy-on-average — a gentle trim steadies the fit.",
        )
    elif priority == "total":
        out["production_metric"] = Resolution(
            "seasonal_mase", "You care about the running/daily total — this "
            "scores the cumulative-shape skill best of the available metrics.",
        )

    return out


def resolve_settings_report(
    series: pd.Series,
    interval_minutes: int,
    *,
    source_is_cumulative: bool = False,
    pinned: Optional[Dict[str, Any]] = None,
    answers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """High-level helper: characterise → resolve → apply pins → report.

    ``pinned`` maps a managed field to its current configured value; any field
    whose configured value is **not** the ``"auto"`` sentinel is reported as
    ``source="pinned"`` and keeps the user's value. The returned dict is the
    JSON payload the Settings tab renders and the orchestrator applies.
    """
    pinned = pinned or {}
    profile = characterize(series, interval_minutes, source_is_cumulative)
    resolved = resolve(profile, answers=answers)

    fields: Dict[str, Any] = {}
    for fname, res in resolved.items():
        configured = pinned.get(fname, AUTO)
        if fname in MANAGED_FIELDS and configured != AUTO and configured is not None:
            fields[fname] = Resolution(
                configured, "You pinned this value.", source="pinned"
            ).to_dict()
        else:
            fields[fname] = res.to_dict()

    n_pinned = sum(1 for f in MANAGED_FIELDS if pinned.get(f, AUTO) not in (AUTO, None))
    return {
        "profile": profile.to_dict(),
        "fields": fields,
        "hints": _covariate_hints(profile),
        "n_managed": len(MANAGED_FIELDS),
        "n_pinned": n_pinned,
    }


def _covariate_hints(profile: DataProfile) -> List[str]:
    """Plain-language, advisory suggestions about inputs the model can't see."""
    hints: List[str] = []
    if profile.persona == "smooth_cycle" and profile.nonneg:
        hints.append(
            "If this is solar generation, enabling the built-in sun-elevation "
            "/ clear-sky-irradiance covariates usually helps a lot."
        )
    if profile.persona in ("bursty", "baseline_plus_spikes"):
        hints.append(
            "The spike timing is driven by things the model can't see yet — "
            "adding covariates (occupancy, appliance/EV state, a heating "
            "schedule) is the biggest accuracy lever here."
        )
    if profile.persona == "baseline_plus_spikes":
        hints.append(
            "If big switchable loads (EV, immersion) sit on top of a baseline, "
            "Load-subtract lets the model learn the clean baseline separately."
        )
    return hints


def apply_to_experiment(exp_cfg, series: pd.Series) -> Optional[Dict[str, Any]]:
    """Resolve the ``"auto"`` settings on ``exp_cfg`` in place, from ``series``.

    Called from the preprocessing chokepoint with the pre-clip per-interval
    series. For each managed field left on the ``"auto"`` sentinel, the
    concrete resolved value is written onto ``exp_cfg`` so all existing
    downstream readers (the ``clip_outliers`` call, model ``loss_fn``
    propagation, the benchmark ranking metric) pick it up unchanged.

    The original sentinels are remembered on ``exp_cfg._auto_sentinels`` so
    the resolution re-runs every cycle (tracking data drift) rather than
    freezing after the first call. The full report is stashed on
    ``exp_cfg._auto_resolution`` for the Settings-tab preview. Returns that
    report (or ``None`` when nothing is on Automatic).
    """
    # Remember which fields were on the sentinel *before* we mutate them.
    sentinels = getattr(exp_cfg, "_auto_sentinels", None)
    if sentinels is None:
        sentinels = {f: getattr(exp_cfg, f, AUTO) for f in MANAGED_FIELDS}
        try:
            exp_cfg._auto_sentinels = sentinels
        except Exception:
            pass

    pinned = {f: sentinels.get(f, AUTO) for f in MANAGED_FIELDS}
    any_auto = any(v == AUTO for v in pinned.values())

    try:
        report = resolve_settings_report(
            series,
            getattr(exp_cfg, "interval_minutes", 30),
            source_is_cumulative=getattr(exp_cfg, "source_is_cumulative", False),
            pinned=pinned,
        )
    except Exception as e:  # never let profiling break a training run
        logger.warning("auto_config resolution failed (%s); using safe defaults", e)
        report = None
        for fname in MANAGED_FIELDS:
            if pinned.get(fname) == AUTO:
                _safe_set(exp_cfg, fname, SAFE_DEFAULTS[fname])
        return None

    # Apply the resolved concrete value to every field still on the sentinel.
    for fname in MANAGED_FIELDS:
        if pinned.get(fname) == AUTO:
            resolved_value = report["fields"].get(fname, {}).get(
                "value", SAFE_DEFAULTS[fname]
            )
            _safe_set(exp_cfg, fname, resolved_value)

    try:
        exp_cfg._auto_resolution = report
    except Exception:
        pass
    return report if any_auto or report else report


def _safe_set(exp_cfg, field_name: str, value: Any) -> None:
    try:
        setattr(exp_cfg, field_name, value)
    except Exception:
        pass
