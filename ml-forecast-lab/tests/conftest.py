"""Shared fixtures for ML Forecast Lab tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_cumulative_series():
    """Cumulative demand series that resets daily (like sensor.energy_today)."""
    idx = pd.date_range("2024-01-01", periods=48 * 14, freq="30min")  # 14 days
    daily_curve = np.sin(np.linspace(0, np.pi, 48)) * 3  # 0-3% per step, peaks midday
    values = np.tile(daily_curve, 14)
    cumulative = np.cumsum(values.reshape(14, 48), axis=1).ravel()
    return pd.Series(cumulative, index=idx, name="demand_today")


@pytest.fixture
def synthetic_interval_series():
    """Interval (non-cumulative) demand series at 30-min resolution."""
    idx = pd.date_range("2024-01-01", periods=48 * 14, freq="30min")
    rng = np.random.default_rng(42)
    daily_curve = np.sin(np.linspace(0, np.pi, 48)) * 1.5 + 0.5
    values = np.tile(daily_curve, 14) + rng.normal(0, 0.1, len(idx))
    values = np.clip(values, 0, None)
    return pd.Series(values, index=idx, name="demand_interval")


@pytest.fixture
def synthetic_df():
    """DataFrame with target + covariates for feature building."""
    idx = pd.date_range("2024-01-01", periods=48 * 14, freq="30min")
    rng = np.random.default_rng(42)
    daily_curve = np.sin(np.linspace(0, np.pi, 48)) * 1.5 + 0.5
    y = np.tile(daily_curve, 14) + rng.normal(0, 0.1, len(idx))
    y = np.clip(y, 0, None)
    charge = 50 + 30 * np.sin(np.linspace(0, 2 * np.pi * 14, len(idx))) + rng.normal(0, 2, len(idx))
    temp = 10 + 5 * np.sin(np.linspace(0, 2 * np.pi * 14, len(idx))) + rng.normal(0, 1, len(idx))
    return pd.DataFrame({
        "y": y,
        "current_charge": charge / 100,  # Scaled 0-1
        "external_temperature": temp,
    }, index=idx)


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite database path."""
    return str(tmp_path / "test_history.db")
