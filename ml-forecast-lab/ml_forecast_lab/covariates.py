"""
Covariate resolution and resampling for ML Forecast Lab.

Handles fetching historical and future covariate data from Home Assistant,
with intelligent binary detection and adaptive resampling strategies.
"""

import logging
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from .ha_interface import HAInterface, normalise_history, state_to_float

logger = logging.getLogger(__name__)


class CovariateResolver:
    """Fetch and resample covariates from Home Assistant."""

    def __init__(
        self,
        iface: HAInterface,
        covariate_configs: Optional[list[dict]] = None,
    ):
        """
        Initialise covariate resolver.

        Args:
            iface: HAInterface instance
            covariate_configs: List of covariate configuration dicts:
                - entity_id: HA entity ID
                - name: Covariate name (optional, defaults to entity_id)
                - binary: Explicit binary flag (optional, overrides auto-detection)
                - constant_value: Value to use for future (optional, for non-forecasted covariates)
        """
        self.iface = iface
        self.covariate_configs = covariate_configs or []
        logger.info(f"CovariateResolver initialised with {len(self.covariate_configs)} covariates")

    def _detect_binary(self, series: pd.Series) -> bool:
        """
        Auto-detect if series is binary (0/1 or True/False).

        Args:
            series: Pandas Series of numeric values

        Returns:
            True if series contains mostly 0, 1, or NaN
        """
        valid = series.dropna()
        if len(valid) < 2:
            return False

        unique_vals = valid.unique()
        # Binary if at most 2 unique non-NaN values in {0, 1}
        return len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0})

    def _resample_covariate(
        self,
        series: pd.Series,
        freq: str,
        is_binary: Optional[bool] = None,
    ) -> pd.Series:
        """
        Resample covariate to desired frequency.

        Binary covariates use forward fill (step function).
        Continuous covariates use mean aggregation.

        Args:
            series: Indexed by datetime, values are numeric
            freq: Pandas frequency string (e.g. '1H', '1D')
            is_binary: Explicit binary flag; if None, auto-detect

        Returns:
            Resampled Series with freq index
        """
        if series.empty:
            return series

        # Auto-detect binary if not specified
        if is_binary is None:
            is_binary = self._detect_binary(series)

        resampler = series.resample(freq)

        if is_binary:
            # Forward fill for binary (step function)
            resampled = resampler.last().fillna(method="ffill")
        else:
            # Mean for continuous
            resampled = resampler.mean()

        return resampled

    async def fetch_history(
        self,
        cov_cfg: dict,
        start: datetime,
        end: datetime,
        freq: str,
    ) -> pd.Series:
        """
        Fetch and resample historical covariate data.

        Args:
            cov_cfg: Covariate config dict (entity_id, name, binary, etc.)
            start: Start datetime
            end: End datetime
            freq: Resample frequency (e.g. '1H')

        Returns:
            Resampled Series indexed by datetime, with name from config
        """
        entity_id = cov_cfg.get("entity_id")
        if not entity_id:
            raise ValueError("cov_cfg missing 'entity_id'")

        name = cov_cfg.get("name", entity_id)
        binary_flag = cov_cfg.get("binary")

        logger.info(f"Fetching covariate history: {entity_id}")

        try:
            raw = await self.iface.get_history(entity_id, start, end)
            df = normalise_history(raw)

            if df.empty:
                logger.warning(f"No history for {entity_id}")
                return pd.Series(dtype=float, name=name)

            # Normalise to tz-naive for consistent resampling
            if hasattr(df["ds"].dtype, "tz") and df["ds"].dt.tz is not None:
                df["ds"] = df["ds"].dt.tz_localize(None)

            # Set datetime index
            df.set_index("ds", inplace=True)

            # Resample
            resampled = self._resample_covariate(
                df["value"], freq, is_binary=binary_flag
            )
            resampled.name = name

            logger.debug(
                f"Resampled {entity_id}: {len(df)} raw -> {len(resampled)} points"
            )
            return resampled

        except Exception as e:
            logger.error(f"Error fetching covariate {entity_id}: {e}", exc_info=True)
            return pd.Series(dtype=float, name=name)

    async def fetch_future(
        self,
        cov_cfg: dict,
        future_index: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Fetch or generate future covariate values.

        Strategies:
        1. If 'constant_value' in config, use that for all future periods
        2. If entity has forecast attribute, fetch and align to future_index
        3. Otherwise, forward fill from latest historical value

        Args:
            cov_cfg: Covariate config dict
            future_index: DatetimeIndex of future forecast periods

        Returns:
            Series indexed by future_index with cov values (or NaN if unavailable)
        """
        entity_id = cov_cfg.get("entity_id")
        name = cov_cfg.get("name", entity_id)

        # Constant value strategy
        if "constant_value" in cov_cfg:
            val = cov_cfg["constant_value"]
            logger.debug(f"Using constant value {val} for {entity_id}")
            return pd.Series(val, index=future_index, name=name)

        # Try to fetch forecast from entity attribute
        try:
            forecast_attr = await self.iface.get_state(
                entity_id, attribute="forecast"
            )
            if forecast_attr:
                logger.debug(f"Found forecast attribute for {entity_id}")
                # Assume forecast_attr is structured; extract values aligned to future_index
                # This is application-specific; for now, return NaN
                return pd.Series(np.nan, index=future_index, name=name)
        except Exception:
            pass

        # Default: return NaN (forecaster should handle missing covariates)
        logger.info(f"No future covariate data for {entity_id}, using NaN")
        return pd.Series(np.nan, index=future_index, name=name)
