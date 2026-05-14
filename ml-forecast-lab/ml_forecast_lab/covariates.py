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
            # Forward fill for binary (step function). pandas 2.x removed the
            # ``method=`` kwarg from Series.fillna in favour of the explicit
            # ``ffill`` method.
            resampled = resampler.last().ffill()
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
        Fetch or generate future covariate values from an HA entity attribute.

        Resolution order:
        1. ``constant_value`` in config — broadcast across the horizon.
        2. The HA attribute named by ``future_attribute`` (default ``forecast``).
           Two shapes are accepted:
           - list of dicts with a datetime-like key and a value key. The
             datetime key is auto-detected among {datetime, period_start,
             period_end, time, dt}; the value key is taken from
             ``future_value_key`` if set, otherwise auto-detected among
             {value, pv_estimate, state, temperature, cloud_coverage,
             wind_speed, precipitation}.
           - flat dict mapping iso-datetime → value.
        3. NaN fallback when neither path produces aligned data.

        The resulting series is reindexed to ``future_index`` with linear
        interpolation between known points and forward+backward fill at
        the edges so the model never sees NaN inside the supplied horizon.
        """
        entity_id = cov_cfg.get("entity_id")
        name = cov_cfg.get("name", entity_id)

        if "constant_value" in cov_cfg:
            return pd.Series(
                cov_cfg["constant_value"], index=future_index, name=name,
            )

        attr_name = cov_cfg.get("future_attribute", "forecast")
        value_key = cov_cfg.get("future_value_key")

        try:
            attr = await self.iface.get_state(entity_id, attribute=attr_name)
        except Exception as e:
            logger.warning(
                "fetch_future: %s attribute fetch failed: %s", entity_id, e,
            )
            return pd.Series(np.nan, index=future_index, name=name)

        parsed = _parse_forecast_attribute(attr, value_key=value_key)
        if parsed is None or parsed.empty:
            logger.debug(
                "fetch_future: %s attribute %r empty / unparseable; "
                "returning NaN", entity_id, attr_name,
            )
            return pd.Series(np.nan, index=future_index, name=name)

        # Align timezone of the parsed series to the requested future_index
        # so reindex doesn't drop everything.
        target_tz = future_index.tz
        parsed_tz = parsed.index.tz
        if target_tz is None and parsed_tz is not None:
            parsed.index = parsed.index.tz_convert(None)
        elif target_tz is not None and parsed_tz is None:
            parsed.index = parsed.index.tz_localize('UTC').tz_convert(target_tz)
        elif target_tz is not None and parsed_tz is not None and target_tz != parsed_tz:
            parsed.index = parsed.index.tz_convert(target_tz)

        # Interpolate to the request grid: union, interpolate, then reindex.
        union = parsed.index.union(future_index).sort_values()
        on_union = parsed.reindex(union).interpolate(method='time', limit_direction='both')
        aligned = on_union.reindex(future_index).ffill().bfill()
        aligned.name = name
        return aligned


def _parse_forecast_attribute(
    attr: Any, value_key: Optional[str] = None,
) -> Optional[pd.Series]:
    """Best-effort parse of a Home Assistant 'forecast'-style attribute.

    Returns a datetime-indexed Series of floats, or None if the shape is
    not understood. Two layouts are supported:

    - ``list[dict]`` with a datetime-ish key and at least one numeric value
      key (Met.no weather forecast, Solcast detailedForecast, ...).
    - ``dict[str, float]`` mapping ISO datetimes to floats
      (Forecast.Solar's detailedForecast).
    """
    DT_KEYS = ('datetime', 'period_start', 'period_end', 'time', 'dt', 'start')
    VAL_KEYS = (
        'value', 'pv_estimate', 'state',
        'temperature', 'cloud_coverage', 'cloud_cover',
        'wind_speed', 'precipitation', 'humidity',
    )

    if isinstance(attr, dict):
        rows = []
        for k, v in attr.items():
            try:
                ts = pd.to_datetime(k)
                val = float(v)
            except (TypeError, ValueError):
                continue
            rows.append((ts, val))
        if not rows:
            return None
        rows.sort(key=lambda r: r[0])
        idx = pd.DatetimeIndex([r[0] for r in rows])
        return pd.Series([r[1] for r in rows], index=idx, dtype='float64')

    if isinstance(attr, list) and attr and isinstance(attr[0], dict):
        sample = attr[0]
        dt_key = next((k for k in DT_KEYS if k in sample), None)
        if dt_key is None:
            return None
        if value_key is not None and value_key in sample:
            vkey = value_key
        else:
            vkey = next((k for k in VAL_KEYS if k in sample), None)
        if vkey is None:
            return None
        rows = []
        for entry in attr:
            try:
                ts = pd.to_datetime(entry[dt_key])
                val = float(entry[vkey])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append((ts, val))
        if not rows:
            return None
        rows.sort(key=lambda r: r[0])
        idx = pd.DatetimeIndex([r[0] for r in rows])
        return pd.Series([r[1] for r in rows], index=idx, dtype='float64')

    return None
