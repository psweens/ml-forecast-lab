"""
Deterministic synthetic datasets for the neural-PV investigation.

Each generator returns a (df, meta) pair where df has a DatetimeIndex,
columns ``y`` (target) and any covariates the synthetic scenario wants
to expose (sun_elevation, clear_sky_ghi computed via pvlib so we have
real physics-grade timestamps to compare against).

Datasets:
    pure_pv         deterministic noiseless PV bell curve
    cloudy_pv       same shape with smooth multiplicative cloud noise
    ev_mixergy      Mixergy-like daily cycle (no solar physics) — negative control

The shape used for solar power is ``max(0, sin(2pi(h-6)/12))**2`` * envelope,
zero at night, peak at 12:00 local.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


GB_LAT = 52.0
GB_LON = -1.0


def _solar_covariates(idx: pd.DatetimeIndex) -> pd.DataFrame:
    from ml_forecast_lab.solar_physics import compute_solar_features

    df = compute_solar_features(
        idx,
        latitude=GB_LAT,
        longitude=GB_LON,
        include_elevation=True,
        include_clear_sky=True,
    )
    df = df.fillna(0.0)
    return df


def _pv_shape(idx: pd.DatetimeIndex, scale: float = 3.0) -> np.ndarray:
    """Deterministic noiseless PV-like daily bell curve.

    Peaks at solar noon (12:00 local) and is zero between 18:00 and 06:00.
    Modulated with a yearly envelope that stays above 0.4 so winter days
    have real (smaller) generation rather than collapsing to zero — the
    seasonal floor matters for this investigation because the holdout
    must contain non-trivial signal.
    """
    h = idx.hour + idx.minute / 60.0
    daylight = np.maximum(0.0, np.sin(2 * np.pi * (h - 6) / 24)) ** 2
    day_of_year = idx.dayofyear.values
    envelope = 0.4 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25))
    return scale * envelope * daylight.values


@dataclass
class SyntheticData:
    name: str
    df: pd.DataFrame
    meta: Dict[str, str]


def make_pure_pv(seed: int = 0) -> SyntheticData:
    """Pure deterministic PV with real solar covariates. No noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=365 * 48, freq="30min", tz="UTC")
    y = _pv_shape(idx)
    cov = _solar_covariates(idx)
    df = pd.DataFrame({"y": y.astype(np.float32)}, index=idx)
    for c in cov.columns:
        df[c] = cov[c].astype(np.float32)
    # Tiny float-32 noise so std() never collapses to 0 in any constant window.
    df["y"] = df["y"].values + rng.normal(0, 1e-4, size=len(df)).astype(np.float32)
    return SyntheticData(
        name="pure_pv",
        df=df,
        meta={
            "description": "deterministic PV bell curve + pvlib solar covariates",
            "n_rows": str(len(df)),
            "freq": "30min",
            "n_days": "365",
        },
    )


def make_cloudy_pv(seed: int = 0) -> SyntheticData:
    """PV with smooth multiplicative cloud noise in [0.3, 1.2]."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=365 * 48, freq="30min", tz="UTC")
    y = _pv_shape(idx)
    # Smooth cloud cover: AR(1) random walk passed through a sigmoid.
    raw = np.zeros(len(idx))
    for t in range(1, len(idx)):
        raw[t] = 0.95 * raw[t - 1] + rng.normal(0, 0.4)
    cloud = 0.3 + 0.9 * (1.0 / (1 + np.exp(-raw)))  # ~ [0.3, 1.2]
    y_cloudy = y * cloud
    cov = _solar_covariates(idx)
    df = pd.DataFrame({"y": y_cloudy.astype(np.float32)}, index=idx)
    for c in cov.columns:
        df[c] = cov[c].astype(np.float32)
    return SyntheticData(
        name="cloudy_pv",
        df=df,
        meta={
            "description": "PV with multiplicative cloud noise in [0.3, 1.2]",
            "n_rows": str(len(df)),
            "freq": "30min",
            "n_days": "365",
        },
    )


def make_ev_mixergy(seed: int = 0) -> SyntheticData:
    """Mixergy-like non-physics daily cycle: EV charging schedule.

    Peaks at ~23:00 + small bump around 07:00, zero during work hours.
    Negative control — neural backends are reported to work fine on
    targets shaped like this.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=365 * 48, freq="30min", tz="UTC")
    h = idx.hour + idx.minute / 60.0
    night_charge = np.exp(-((h - 1) ** 2) / 4.0)  # peak ~01:00, wide
    morning = 0.4 * np.exp(-((h - 7) ** 2) / 1.0)  # peak 07:00, narrow
    base = night_charge + morning
    day_of_year = idx.dayofyear.values
    # Slight weekly seasonality.
    weekday_modifier = 1.0 - 0.2 * (idx.dayofweek.values >= 5)
    y = (3.0 * base * weekday_modifier).astype(np.float32)
    y = y + rng.normal(0, 0.05, size=len(y)).astype(np.float32)
    df = pd.DataFrame({"y": y}, index=idx)
    # No physics covariates for the negative control — this is the cleanest
    # case for showing that the bug is solar-specific.
    return SyntheticData(
        name="ev_mixergy",
        df=df,
        meta={
            "description": "EV-charging-like non-physics daily cycle (negative control)",
            "n_rows": str(len(df)),
            "freq": "30min",
            "n_days": "365",
        },
    )


def make_realistic_pv(seed: int = 0) -> SyntheticData:
    """Sensor-grade noisy PV — aims to reproduce production failure mode.

    Differences from cloudy_pv:
      * Watt-scale (~4 kW peak) so the absolute RevIN mean bias is larger.
      * Quantised to nearest integer Watt (real predbat sensors do this).
      * Intermittent sensor zeros: 0.5% of daytime samples randomly set to 0.
      * Heavier cloud bursts (correlated AR(1) with deeper attenuation).
      * The covariates exposed are pvlib sun_elevation / clear_sky_ghi so
        the training signal-to-noise ratio per covariate roughly matches
        production.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=365 * 48, freq="30min", tz="UTC")
    y_clean = _pv_shape(idx, scale=4500.0)   # W
    # Heavier cloud noise.
    raw = np.zeros(len(idx))
    for t in range(1, len(idx)):
        raw[t] = 0.92 * raw[t - 1] + rng.normal(0, 0.55)
    cloud = 0.15 + 1.05 * (1.0 / (1 + np.exp(-raw)))  # ~ [0.15, 1.2]
    y = y_clean * cloud
    # Daytime sensor dropouts ~0.5%
    daytime = (y_clean > 50)
    dropout_mask = (rng.random(len(idx)) < 0.005) & daytime
    y[dropout_mask] = 0.0
    # Quantise to integer watts.
    y = np.round(y).astype(np.float32)
    cov = _solar_covariates(idx)
    df = pd.DataFrame({"y": y}, index=idx)
    for c in cov.columns:
        df[c] = cov[c].astype(np.float32)
    return SyntheticData(
        name="realistic_pv",
        df=df,
        meta={
            "description": "Watt-scale PV with sensor zeros + integer quantisation + AR(1) clouds",
            "n_rows": str(len(df)),
            "freq": "30min",
            "n_days": "365",
        },
    )


def all_datasets() -> Tuple[SyntheticData, SyntheticData, SyntheticData, SyntheticData]:
    return (make_pure_pv(0), make_cloudy_pv(0), make_realistic_pv(0), make_ev_mixergy(0))
