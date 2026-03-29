"""
Forecast publishing to Home Assistant.

Publishes forecast results (point, interval, and cumulative) as entities
with configurable naming and aggregation strategies.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from .ha_interface import HAInterface

logger = logging.getLogger(__name__)


def make_entity_name(
    publish_prefix: str,
    experiment_name: str,
    suffix: str,
) -> str:
    """
    Construct entity name for forecast entity.

    Args:
        publish_prefix: Prefix (default 'mlfl_')
        experiment_name: Experiment identifier
        suffix: Entity suffix (e.g. 'point', 'upper_95')

    Returns:
        Entity name suitable for sensor.<name>
    """
    return f"{publish_prefix}{experiment_name}_{suffix}".lower().replace("-", "_")


def dict_from_series(
    series: pd.Series,
    max_points: int = 100,
) -> dict[str, Any]:
    """
    Serialise Series to dict for HA attribute.

    Args:
        series: Pandas Series (index is timestamps or labels)
        max_points: Limit output points (sample if exceeded)

    Returns:
        Dict with 'timestamps' and 'values' lists
    """
    if series.empty:
        return {"timestamps": [], "values": []}

    s = series.copy()
    if len(s) > max_points:
        # Sample evenly
        indices = np.linspace(0, len(s) - 1, max_points, dtype=int)
        s = s.iloc[indices]

    result = {
        "timestamps": (
            s.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
            if hasattr(s.index, "strftime")
            else [str(i) for i in s.index]
        ),
        "values": s.fillna(np.nan).tolist(),
    }

    return result


def daily_cumulative_series(
    forecast_series: pd.Series,
    reference_date: Optional[datetime] = None,
) -> pd.Series:
    """
    Group forecast by date and cumulate within each day.

    Useful for energy forecasts where you want daily totals.

    Args:
        forecast_series: Series indexed by datetime, values are increments
        reference_date: Date to group by (defaults to today UTC)

    Returns:
        Series with same index, cumulative within each day
    """
    if forecast_series.empty:
        return forecast_series

    if reference_date is None:
        reference_date = datetime.now(timezone.utc).date()

    s = forecast_series.copy()

    # Ensure datetime index
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)

    # Group by date and cumulate
    daily_groups = s.groupby(s.index.date)
    cumulative = pd.concat(
        [group.cumsum() for _, group in daily_groups]
    )

    return cumulative.sort_index()


def energy_already_used_today(
    iface: HAInterface,
    entity_id: str,
) -> float:
    """
    Fetch total energy used/produced so far today (synchronous call).

    Note: Async caller should use asyncio.run() or similar.

    Args:
        iface: HAInterface (assumes sync state fetch available)
        entity_id: Energy sensor entity ID

    Returns:
        Energy value, or 0.0 if unavailable
    """
    # This is a placeholder; actual implementation depends on whether
    # we have a sync wrapper or async context
    logger.debug(f"Fetching current state for {entity_id}")
    return 0.0


async def publish_forecasts(
    experiment_cfg: dict,
    iface: HAInterface,
    app_config: dict,
    ds_future: pd.DatetimeIndex,
    yhat_interval: pd.DataFrame,
    yhat_level: float,
    metrics: Optional[dict] = None,
    hist_cum_df: Optional[pd.DataFrame] = None,
) -> bool:
    """
    Publish forecast results to Home Assistant.

    Creates the following entities:
    - {prefix}{exp_name}_point: Point forecast
    - {prefix}{exp_name}_upper_{level}: Upper interval bound
    - {prefix}{exp_name}_lower_{level}: Lower interval bound
    - {prefix}{exp_name}_cumulative: Cumulative forecast (sum)
    - {prefix}{exp_name}_daily_cumulative: Daily cumulative with offset
    - {prefix}{exp_name}_horizon_*: Scalar entities for key horizons

    Args:
        experiment_cfg: Experiment config with keys:
            - name: Experiment identifier
            - publish_prefix: Entity prefix (default 'mlfl_')
            - publish_entity_id: Override sensor entity name
            - horizons_to_publish: List of horizon strings (e.g. ['+2h', '+8h'])
        iface: HAInterface for publishing
        app_config: Application config (unused in this version)
        ds_future: DatetimeIndex of forecast periods
        yhat_interval: DataFrame with columns ['ds', 'yhat', 'upper', 'lower']
                       or at minimum ['ds', 'yhat']
        yhat_level: Confidence level (e.g. 0.95)
        metrics: Optional dict of forecast metrics to publish
        hist_cum_df: Optional historical cumulative for curve visualization

    Returns:
        True if all publishes succeeded
    """
    exp_name = experiment_cfg.get("name", "forecast")
    prefix = experiment_cfg.get("publish_prefix", "mlfl_")
    base_entity_id = experiment_cfg.get(
        "publish_entity_id", f"sensor.{make_entity_name(prefix, exp_name, 'point')}"
    )

    logger.info(f"Publishing forecasts for {exp_name}")

    # Ensure yhat_interval has required columns
    if "ds" not in yhat_interval.columns:
        yhat_interval = yhat_interval.reset_index()
        yhat_interval.rename(columns={"index": "ds"}, inplace=True)

    success = True

    # 1. Point forecast
    if "yhat" in yhat_interval.columns:
        try:
            yhat_series = yhat_interval.set_index("ds")["yhat"]
            state = str(float(yhat_series.iloc[-1]))
            attrs = {
                "unit_of_measurement": "unknown",
                "icon": "mdi:chart-line",
                "friendly_name": f"{exp_name} Point Forecast",
                "forecast": dict_from_series(yhat_series, max_points=100),
            }
            if metrics:
                attrs["metrics"] = metrics

            ok = await iface.set_state(base_entity_id, state, attributes=attrs)
            success = success and ok
            logger.info(f"Published point forecast to {base_entity_id}")
        except Exception as e:
            logger.error(f"Error publishing point forecast: {e}")
            success = False

    # 2. Interval bounds
    level_pct = int(yhat_level * 100)

    if "upper" in yhat_interval.columns:
        try:
            upper_series = yhat_interval.set_index("ds")["upper"]
            state = str(float(upper_series.iloc[-1]))
            entity = f"{base_entity_id.replace('_point', '')}_upper_{level_pct}"
            attrs = {
                "friendly_name": f"{exp_name} Upper {level_pct}%",
                "forecast": dict_from_series(upper_series, max_points=100),
            }
            ok = await iface.set_state(entity, state, attributes=attrs)
            success = success and ok
            logger.debug(f"Published upper bound to {entity}")
        except Exception as e:
            logger.error(f"Error publishing upper bound: {e}")
            success = False

    if "lower" in yhat_interval.columns:
        try:
            lower_series = yhat_interval.set_index("ds")["lower"]
            state = str(float(lower_series.iloc[-1]))
            entity = f"{base_entity_id.replace('_point', '')}_lower_{level_pct}"
            attrs = {
                "friendly_name": f"{exp_name} Lower {level_pct}%",
                "forecast": dict_from_series(lower_series, max_points=100),
            }
            ok = await iface.set_state(entity, state, attributes=attrs)
            success = success and ok
            logger.debug(f"Published lower bound to {entity}")
        except Exception as e:
            logger.error(f"Error publishing lower bound: {e}")
            success = False

    # 3. Cumulative forecast
    if "yhat" in yhat_interval.columns:
        try:
            yhat_series = yhat_interval.set_index("ds")["yhat"]
            cumulative = yhat_series.cumsum()
            state = str(float(cumulative.iloc[-1]))
            entity = f"{base_entity_id.replace('_point', '')}_cumulative"
            attrs = {
                "friendly_name": f"{exp_name} Cumulative Forecast",
                "cumulative": dict_from_series(cumulative, max_points=100),
            }
            ok = await iface.set_state(entity, state, attributes=attrs)
            success = success and ok
            logger.debug(f"Published cumulative to {entity}")
        except Exception as e:
            logger.error(f"Error publishing cumulative: {e}")
            success = False

    # 4. Daily cumulative with offset
    try:
        yhat_series = yhat_interval.set_index("ds")["yhat"]
        daily_cum = daily_cumulative_series(yhat_series)
        state = str(float(daily_cum.iloc[-1]))
        entity = f"{base_entity_id.replace('_point', '')}_daily_cumulative"
        attrs = {
            "friendly_name": f"{exp_name} Daily Cumulative",
            "cumulative": dict_from_series(daily_cum, max_points=100),
        }
        ok = await iface.set_state(entity, state, attributes=attrs)
        success = success and ok
        logger.debug(f"Published daily cumulative to {entity}")
    except Exception as e:
        logger.error(f"Error publishing daily cumulative: {e}")
        success = False

    # 5. Horizon scalar entities
    horizons = experiment_cfg.get("horizons_to_publish", [])
    for horizon_str in horizons:
        try:
            # Parse horizon string (e.g. '+2h' -> 2 hours)
            horizon_str_clean = horizon_str.strip()
            if horizon_str_clean.startswith("+"):
                horizon_str_clean = horizon_str_clean[1:]

            # Simple parsing: '2h', '30m', etc.
            multiplier = 1
            if horizon_str_clean.endswith("h"):
                multiplier = 60  # Convert to minutes
                value = int(horizon_str_clean[:-1])
            elif horizon_str_clean.endswith("m"):
                value = int(horizon_str_clean[:-1])
            elif horizon_str_clean.endswith("d"):
                multiplier = 24 * 60  # Convert to minutes
                value = int(horizon_str_clean[:-1])
            else:
                value = int(horizon_str_clean)

            minutes_ahead = value * multiplier

            # Find forecast point at this horizon
            if len(yhat_interval) > 0:
                idx = min(minutes_ahead // 60, len(yhat_interval) - 1)
                if idx >= 0 and idx < len(yhat_interval):
                    forecast_val = yhat_interval.iloc[idx].get("yhat", np.nan)
                    state = str(float(forecast_val))
                    entity = f"{base_entity_id.replace('_point', '')}_horizon_{horizon_str_clean}"
                    attrs = {
                        "friendly_name": f"{exp_name} Horizon {horizon_str}",
                        "horizon": horizon_str,
                    }
                    ok = await iface.set_state(entity, state, attributes=attrs)
                    success = success and ok
                    logger.debug(f"Published horizon {horizon_str} to {entity}")
        except Exception as e:
            logger.error(f"Error publishing horizon {horizon_str}: {e}")
            success = False

    # 6. Prediction curve (current + historical)
    if hist_cum_df is not None and not hist_cum_df.empty:
        try:
            yhat_series = yhat_interval.set_index("ds")["yhat"]
            curve = pd.concat([hist_cum_df.set_index("ds")["value"], yhat_series])
            entity = f"{base_entity_id.replace('_point', '')}_curve"
            attrs = {
                "friendly_name": f"{exp_name} Prediction Curve",
                "curve": dict_from_series(curve, max_points=150),
            }
            ok = await iface.set_state(entity, "active", attributes=attrs)
            success = success and ok
            logger.debug(f"Published curve to {entity}")
        except Exception as e:
            logger.error(f"Error publishing curve: {e}")
            success = False

    if success:
        logger.info(f"Successfully published all forecasts for {exp_name}")
    else:
        logger.warning(f"Some forecast publishes failed for {exp_name}")

    return success
