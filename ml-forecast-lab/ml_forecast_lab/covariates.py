"""
Covariate resolution and resampling for ML Forecast Lab.

Handles fetching historical and future covariate data from Home Assistant,
with intelligent binary detection and adaptive resampling strategies.
"""

import logging
import re
from datetime import datetime, timedelta, timezone
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
        history_db: Optional[Any] = None,
        cache_max_age_days: int = 365,
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
            history_db: Optional ``HistoryDB``. When supplied, covariate
                history is cached in SQLite and only the delta since the
                last cached observation is fetched from HA on each call —
                mirroring the incremental caching the target series
                already uses. Every forecast cycle re-fetches the full
                ``days_history`` window per covariate otherwise; on a
                covariate-heavy experiment that is the dominant per-cycle
                HA recorder + network cost. When ``None`` (e.g. unit
                tests), behaviour is the original full-window fetch.
            cache_max_age_days: Rolling-window bound for the per-covariate
                cache tables, pruned after each fetch.
        """
        self.iface = iface
        self.covariate_configs = covariate_configs or []
        self.history_db = history_db
        self.cache_max_age_days = cache_max_age_days
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

        # Attribute-history path (v2.38.4+): when ``future_value_key``
        # is set, pull historical numerics from
        # ``record.attributes[future_value_key]`` instead of the
        # entity's ``.state``. Originally weather-only; v2.39.3
        # generalises so two configs of the same non-weather entity
        # with distinct ``future_value_key`` values resolve to
        # different signals (otherwise both fall back to ``.state``
        # and the model trains on two columns of identical data).
        attribute_key = None
        value_key_for_history = cov_cfg.get("future_value_key")
        if value_key_for_history:
            attribute_key = value_key_for_history

        logger.info(
            f"Fetching covariate history: {entity_id}"
            + (f" (attribute={attribute_key})" if attribute_key else "")
        )

        try:
            df = await self._fetch_raw_history(
                entity_id, start, end, attribute_key,
            )

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

    def _cov_cache_table(
        self, entity_id: str, attribute_key: Optional[str],
    ) -> str:
        """SQLite cache-table name for a covariate's raw observations.

        Namespaced under ``cov_`` so it never collides with a target
        series cached under ``safe_table_name(entity_id)``, and keyed by
        ``attribute_key`` so two covariates on the same entity reading
        different attributes (e.g. ``temperature`` vs ``cloud_coverage``)
        cache independently.
        """
        raw_key = f"cov_{entity_id}"
        if attribute_key:
            raw_key = f"{raw_key}__{attribute_key}"
        return self.history_db.safe_table_name(raw_key)

    async def _fetch_raw_history(
        self,
        entity_id: str,
        start: datetime,
        end: datetime,
        attribute_key: Optional[str],
    ) -> pd.DataFrame:
        """Return raw ``[ds, value]`` observations in ``[start, end]``.

        When a ``history_db`` is configured this reads the cached rows,
        fetches only the delta newer than the latest cached observation
        from HA, persists the new rows, and returns the merged frame —
        the same incremental pattern the target series uses in
        ``main._fetch_and_preprocess``. Resampling happens in the caller
        on the merged raw rows, so the result is identical to a
        full-window fetch (modulo recorder restatements, which match the
        target's ``INSERT OR IGNORE`` semantics). Any cache error
        degrades to a full-window fetch so caching can never break a
        forecast cycle.
        """
        include_attrs = attribute_key is not None

        # No DB → original full-window behaviour.
        if self.history_db is None:
            raw = await self.iface.get_history(
                entity_id, start, end, include_attributes=include_attrs,
            )
            return normalise_history(raw, attribute_key=attribute_key)

        start_naive = pd.Timestamp(start)
        if start_naive.tzinfo is not None:
            start_naive = start_naive.tz_localize(None)

        # --- Read cache (rows within the requested window) ---
        cached = pd.DataFrame(columns=["ds", "value"])
        table: Optional[str] = None
        try:
            table = self._cov_cache_table(entity_id, attribute_key)
            c = self.history_db.get_history(table)  # columns [ds, y]
            if not c.empty:
                c = c.rename(columns={"y": "value"})
                c = c[c["ds"] >= start_naive]
                if len(c) > 0:
                    cached = c[["ds", "value"]]
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Covariate cache read failed for {entity_id}: {e}")
            cached = pd.DataFrame(columns=["ds", "value"])

        # --- Determine the delta fetch start ---
        if len(cached) > 0:
            last_cached = cached["ds"].max()
            fetch_start = pd.Timestamp(last_cached)
            if fetch_start.tzinfo is None:
                fetch_start = fetch_start.tz_localize("UTC")
            fetch_start = fetch_start.to_pydatetime()
        else:
            fetch_start = start

        raw = await self.iface.get_history(
            entity_id, fetch_start, end, include_attributes=include_attrs,
        )
        new_df = normalise_history(raw, attribute_key=attribute_key)
        if (
            not new_df.empty
            and hasattr(new_df["ds"].dtype, "tz")
            and new_df["ds"].dt.tz is not None
        ):
            new_df["ds"] = new_df["ds"].dt.tz_localize(None)

        # --- Persist new rows + prune the rolling window ---
        if table is not None:
            try:
                if not new_df.empty:
                    self.history_db.store_history(table, new_df)
                oldest = datetime.now(timezone.utc) - timedelta(
                    days=self.cache_max_age_days,
                )
                self.history_db.cleanup(table, oldest)
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Covariate cache write failed for {entity_id}: {e}")

        # --- Merge cached + delta ---
        if len(cached) > 0 and not new_df.empty:
            merged = pd.concat(
                [cached, new_df[["ds", "value"]]], ignore_index=True,
            )
            merged = (
                merged.drop_duplicates(subset=["ds"], keep="last")
                .sort_values("ds")
                .reset_index(drop=True)
            )
            return merged
        if len(cached) > 0:
            return cached.sort_values("ds").reset_index(drop=True)
        return new_df

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

        # HA 2023.9+ weather entities (Met Office DataHub, OpenWeatherMap,
        # AccuWeather, met.no, etc.) moved the forecast OUT of state
        # attributes and into a separate ``weather.get_forecasts``
        # service call. When the user picks ``future_attribute`` =
        # ``hourly`` / ``daily`` / ``twice_daily`` (the three forecast
        # types the service supports) and the entity is in the weather
        # domain, fetch via the service API instead of the state
        # attribute. Falls through to the legacy attribute path for
        # any other future_attribute (so Solcast's ``detailedForecast``,
        # Forecast.Solar, custom integrations still work unchanged).
        WEATHER_SERVICE_TYPES = {"hourly", "daily", "twice_daily"}
        is_weather_service = (
            isinstance(entity_id, str)
            and entity_id.startswith("weather.")
            and attr_name in WEATHER_SERVICE_TYPES
        )
        if is_weather_service:
            try:
                resp = await self.iface.api_call(
                    "POST",
                    "/api/services/weather/get_forecasts?return_response",
                    json_data={"entity_id": entity_id, "type": attr_name},
                )
            except Exception as e:
                logger.warning(
                    "fetch_future: %s weather.get_forecasts(type=%s) failed: %s",
                    entity_id, attr_name, e,
                )
                return pd.Series(np.nan, index=future_index, name=name)

            # HA returns ``{service_response: {entity_id: {forecast: [...]}},
            # context: ...}`` — the per-entity forecast array sits two
            # levels deep. Defensive lookup so a schema tweak doesn't
            # crash the resolver.
            service_resp = (resp or {}).get("service_response") or {}
            entity_block = service_resp.get(entity_id) or {}
            forecast_list = entity_block.get("forecast")
            if not forecast_list:
                logger.debug(
                    "fetch_future: %s weather.get_forecasts(type=%s) "
                    "returned no forecast array; returning NaN",
                    entity_id, attr_name,
                )
                return pd.Series(np.nan, index=future_index, name=name)
            # Same parser as the attribute path — the forecast array
            # is a list[dict] with ``datetime`` + numeric keys, exactly
            # the legacy attribute schema. Reuse _parse_forecast_attribute
            # so the auto-detect / value_key logic matches.
            parsed = _parse_forecast_attribute(
                forecast_list, value_key=value_key,
            )
            if parsed is None or parsed.empty:
                return pd.Series(np.nan, index=future_index, name=name)
            # Skip the attribute fetch below — we already have the
            # parsed series. Jump straight to alignment by routing
            # through the same downstream code path.
            # Skip the attribute fetch below — we already have the
            # parsed series. Jump to the alignment block.
        else:
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
