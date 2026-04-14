"""
Deterministic solar physics covariates.

Provides sun elevation angle and clear-sky global horizontal irradiance (GHI)
as computed features for any timestamp at a given (latitude, longitude).

These features are deterministic — the same timestamp + location always yields
the same value — so they are equally valid for historical training data and
future forecast horizons with zero forecast error.

Requires pvlib.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def compute_solar_features(
    index: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    include_elevation: bool = False,
    include_clear_sky: bool = False,
    altitude: float = 0.0,
) -> pd.DataFrame:
    """
    Compute deterministic solar covariates for a datetime index.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Timestamps at which to compute features. May be tz-aware or tz-naive;
        naive indices are assumed to be UTC.
    latitude : float
        Site latitude in degrees (positive north).
    longitude : float
        Site longitude in degrees (positive east).
    include_elevation : bool
        If True, include a 'sun_elevation' column (degrees above horizon,
        negative at night).
    include_clear_sky : bool
        If True, include a 'clear_sky_ghi' column (W/m², Ineichen model).
    altitude : float
        Site altitude in metres (minor effect on clear-sky GHI; default 0).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by the input index, containing the requested
        columns. Empty DataFrame if neither feature is requested.
    """
    if not (include_elevation or include_clear_sky):
        return pd.DataFrame(index=index)

    try:
        import pvlib
    except ImportError as e:
        raise RuntimeError(
            "pvlib is required for solar_physics features but is not "
            "installed. Add 'pvlib>=0.10.0' to requirements.txt."
        ) from e

    if len(index) == 0:
        return pd.DataFrame(index=index)

    # pvlib requires a tz-aware index; assume naive = UTC.
    if index.tz is None:
        times = index.tz_localize("UTC")
    else:
        times = index

    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
    )

    out = pd.DataFrame(index=index)

    if include_elevation:
        sol_pos = location.get_solarposition(times)
        # Use 'apparent_elevation' which accounts for atmospheric refraction
        # (the sun appears slightly above its true geometric position near
        # the horizon). Clipped at -90..+90.
        out["sun_elevation"] = sol_pos["apparent_elevation"].values

    if include_clear_sky:
        cs = location.get_clearsky(times, model="ineichen")
        out["clear_sky_ghi"] = cs["ghi"].values

    return out
