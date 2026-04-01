"""
Home Assistant async API client for ML Forecast Lab.

Provides a clean, type-safe interface to Home Assistant's REST API with
robust error handling, timestamp parsing, and data normalisation.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_timestamp(ts_string: str) -> datetime:
    """
    Parse ISO8601 timestamp string from Home Assistant.

    Handles variants:
    - With/without fractional seconds (e.g. .123456)
    - With/without colon in UTC offset (e.g. +00:00 or +0000)
    - Z suffix for UTC

    Args:
        ts_string: ISO8601 timestamp string from HA

    Returns:
        datetime in UTC
    """
    if not ts_string:
        raise ValueError("Empty timestamp string")

    ts_clean = ts_string.strip()

    # Replace 'Z' with '+00:00'
    if ts_clean.endswith("Z"):
        ts_clean = ts_clean[:-1] + "+00:00"

    # Handle fractional seconds with colon offset (e.g. 2024-01-15T10:30:45.123456+01:00)
    try:
        return datetime.fromisoformat(ts_clean)
    except ValueError:
        pass

    # Handle fractional seconds with no colon (e.g. 2024-01-15T10:30:45.123456+0100)
    if "+" in ts_clean or (ts_clean.count("-") > 2):
        # Find offset
        offset_idx = max(ts_clean.rfind("+"), ts_clean.rfind("-", 10))
        if offset_idx > 0:
            main_part = ts_clean[:offset_idx]
            offset_part = ts_clean[offset_idx:]

            # Add colon to offset if missing
            if len(offset_part) == 5 and ":" not in offset_part:  # e.g. '+0100'
                offset_part = offset_part[:3] + ":" + offset_part[3:]

            ts_clean = main_part + offset_part
            try:
                return datetime.fromisoformat(ts_clean)
            except ValueError:
                pass

    # Last resort: try parsing without offset info
    try:
        return datetime.fromisoformat(ts_clean)
    except ValueError as e:
        raise ValueError(f"Cannot parse timestamp '{ts_string}'") from e


def ensure_utc(dt: datetime) -> datetime:
    """
    Ensure datetime is in UTC, converting if necessary.

    Args:
        dt: datetime object (naive or with timezone)

    Returns:
        datetime in UTC timezone
    """
    if dt.tzinfo is None:
        # Assume naive datetimes are UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def state_to_float(x: Any) -> Optional[float]:
    """
    Convert Home Assistant state value to float.

    Handles:
    - Numeric strings and numbers
    - Boolean strings: on/off, true/false, home/not_home, open/closed
    - Special values: unknown, unavailable -> None
    - NaN/None -> None

    Args:
        x: State value from HA

    Returns:
        float or None if unavailable
    """
    if x is None:
        return None

    if isinstance(x, bool):
        return float(x)

    if isinstance(x, (int, float)):
        if np.isnan(x) if isinstance(x, float) else False:
            return None
        return float(x)

    if not isinstance(x, str):
        return None

    x_lower = x.lower().strip()

    # Special values
    if x_lower in ("unknown", "unavailable", "none", ""):
        return None

    # Boolean strings
    if x_lower in ("on", "true", "yes", "home", "open"):
        return 1.0
    if x_lower in ("off", "false", "no", "not_home", "closed"):
        return 0.0

    # Try numeric conversion
    try:
        return float(x)
    except ValueError:
        return None


def normalise_history(raw_records: list[dict]) -> pd.DataFrame:
    """
    Convert Home Assistant history response to clean DataFrame.

    Args:
        raw_records: List of dicts from HA history API with keys:
                     'state', 'last_changed', 'attributes', etc.

    Returns:
        DataFrame with columns ['ds', 'value']:
        - ds: datetime in UTC
        - value: float (or NaN if unavailable)
    """
    if not raw_records:
        return pd.DataFrame(columns=["ds", "value"])

    rows = []
    for record in raw_records:
        try:
            ts = parse_timestamp(record["last_changed"])
            ts_utc = ensure_utc(ts)
            val = state_to_float(record["state"])

            rows.append({"ds": ts_utc, "value": val})
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping malformed history record: {e}")
            continue

    if not rows:
        return pd.DataFrame(columns=["ds", "value"])

    df = pd.DataFrame(rows)
    df["ds"] = pd.to_datetime(df["ds"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df.sort_values("ds").reset_index(drop=True)


class HAInterface:
    """Async Home Assistant API client."""

    def __init__(
        self,
        ha_url: Optional[str] = None,
        ha_key: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        """
        Initialise HA interface.

        Args:
            ha_url: Base URL (defaults to http://supervisor/core)
            ha_key: Bearer token (defaults to SUPERVISOR_TOKEN env var)
            session: Existing aiohttp session; create new if None
        """
        self.ha_url = ha_url or "http://supervisor/core"
        self.ha_key = ha_key or os.environ.get("SUPERVISOR_TOKEN", "")
        self.session = session
        self._owns_session = session is None

        if self._owns_session:
            self.session = aiohttp.ClientSession()

        logger.info(f"HAInterface initialised with URL: {self.ha_url}")

    async def api_call(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        json_data: Optional[dict] = None,
    ) -> Any:
        """
        Generic HTTP call to HA API with error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g. '/api/history/period')
            params: URL parameters
            json_data: JSON request body

        Returns:
            Response JSON or text

        Raises:
            RuntimeError: For HTTP errors or connection issues
        """
        if not self.session or self.session.closed:
            raise RuntimeError("Session not initialised or closed")

        url = f"{self.ha_url}{endpoint}"
        headers = {"Authorization": f"Bearer {self.ha_key}"}

        try:
            async with self.session.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status in (200, 201):
                    if "application/json" in resp.headers.get("content-type", ""):
                        return await resp.json()
                    return await resp.text()

                error_text = await resp.text()
                raise RuntimeError(
                    f"HA API error {resp.status}: {error_text[:200]}"
                )
        except aiohttp.ClientError as e:
            raise RuntimeError(f"HTTP request failed: {e}") from e

    async def get_history(
        self,
        entity_id: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        """
        Fetch raw history records for an entity.

        Args:
            entity_id: Entity ID (e.g. 'sensor.temperature')
            start: Start datetime (converted to ISO8601)
            end: End datetime (converted to ISO8601)

        Returns:
            List of history dicts with 'state', 'last_changed', etc.
        """
        start_iso = start.isoformat()
        end_iso = end.isoformat()

        endpoint = f"/api/history/period/{start_iso}"
        params = {
            "end_time": end_iso,
            "filter_entity_id": entity_id,
            "minimal_response": "",
        }

        result = await self.api_call("GET", endpoint, params=params)

        if isinstance(result, list) and len(result) > 0:
            return result[0]
        return []

    async def get_state(
        self,
        entity_id: str,
        default: Any = None,
        attribute: Optional[str] = None,
    ) -> Any:
        """
        Get current state or attribute of an entity.

        Args:
            entity_id: Entity ID
            default: Value to return if entity not found
            attribute: Specific attribute to fetch (e.g. 'brightness');
                      if None, returns 'state' field

        Returns:
            State/attribute value or default
        """
        endpoint = f"/api/states/{entity_id}"

        try:
            state_obj = await self.api_call("GET", endpoint)
            if attribute:
                return state_obj.get("attributes", {}).get(attribute, default)
            return state_obj.get("state", default)
        except RuntimeError:
            logger.warning(f"Entity not found: {entity_id}")
            return default

    async def set_state(
        self,
        entity_id: str,
        state: str,
        attributes: Optional[dict] = None,
    ) -> bool:
        """
        Publish state change to HA.

        Args:
            entity_id: Entity ID
            state: New state value
            attributes: Optional state attributes dict

        Returns:
            True if successful
        """
        endpoint = f"/api/states/{entity_id}"
        json_data = {"state": state}
        if attributes:
            json_data["attributes"] = attributes

        try:
            await self.api_call("POST", endpoint, json_data=json_data)
            logger.debug(f"State set: {entity_id} -> {state}")
            return True
        except RuntimeError as e:
            logger.error(f"Failed to set state {entity_id}: {e}")
            return False

    async def close(self) -> None:
        """Close HTTP session if we own it."""
        if self._owns_session and self.session and not self.session.closed:
            await self.session.close()
            logger.info("HAInterface session closed")
