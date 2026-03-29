"""
Unified preprocessing pipeline for ML Forecast Lab.

Consolidates cumulative-to-interval conversion, resampling, outlier handling,
and transformations into a single coherent set of functions.
"""

import logging
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cumulative_to_interval(
    series: pd.Series,
    interval_minutes: int,
    reset_daily: bool = False,
    max_increment: Optional[float] = None,
    method: str = 'diff',
) -> pd.Series:
    """
    Convert cumulative sensor readings to interval values.

    Handles both daily-reset and non-daily cumulative sensors with intelligent
    gap filling, negative difference handling, and spike capping.

    Parameters
    ----------
    series : pd.Series
        Cumulative sensor values with DatetimeIndex.
    interval_minutes : int
        Expected sampling interval in minutes.
    reset_daily : bool, default False
        If True, sensor resets daily (e.g. 'today' energy meters).
    max_increment : float, optional
        Maximum allowed increment; larger values indicate anomalies or resets.
        If None, computed as 95th percentile of observed increments.
    method : str, default 'diff'
        Conversion method; currently only 'diff' is supported.

    Returns
    -------
    pd.Series
        Interval values (non-negative) with same index as input.

    Notes
    -----
    The function:
    1. Detects and handles resets (where cumulative value decreases)
    2. Performs gap-aware interpolation for missing intervals
    3. Caps spikes exceeding max_increment
    4. Handles negative differences intelligently
    5. Preserves timezone information if present

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range('2024-01-01', periods=10, freq='30min')
    >>> cumul = pd.Series([100, 102, 105, 108, 110, 112, 115, 118, 120, 122],
    ...                    index=idx)
    >>> result = cumulative_to_interval(cumul, interval_minutes=30)
    """
    if not isinstance(series, pd.Series):
        raise TypeError('series must be a pandas Series')
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError('series must have DatetimeIndex')
    if interval_minutes < 1:
        raise ValueError(f'interval_minutes must be >= 1, got {interval_minutes}')

    series = series.copy()
    series = series.dropna()

    if len(series) < 2:
        logger.warning('Series has fewer than 2 values; returning zeros')
        return pd.Series(
            np.zeros(len(series)), index=series.index, dtype='float64'
        )

    # Calculate raw differences
    diffs = series.diff()

    # Detect resets (negative differences)
    resets = diffs < 0
    if reset_daily:
        # For daily-reset sensors, also flag suspected resets based on time
        midnight_crosses = (
            series.index.normalize() != series.index.shift(1).normalize()
        )
        resets = resets | (midnight_crosses & (diffs < 0.1 * series.abs().mean()))

    # Estimate max_increment if not provided
    if max_increment is None:
        positive_diffs = diffs[diffs > 0]
        if len(positive_diffs) > 0:
            max_increment = positive_diffs.quantile(0.95)
        else:
            max_increment = 1.0
            logger.warning(
                'No positive differences found; using default max_increment=1.0'
            )

    logger.debug(f'Using max_increment={max_increment:.2f}')

    # Handle resets: assume the reset value is the interval
    diffs_adj = diffs.copy()
    diffs_adj[resets] = series[resets]

    # Cap spikes
    spike_mask = diffs_adj > max_increment
    if spike_mask.any():
        logger.debug(f'Found {spike_mask.sum()} spike(s) exceeding max_increment')
        diffs_adj[spike_mask] = np.clip(
            diffs_adj[spike_mask], 0, max_increment
        )

    # Handle negative differences (after reset handling)
    diffs_adj[diffs_adj < 0] = 0

    # Gap-aware interpolation: scale by actual interval
    time_diffs = series.index.to_series().diff().dt.total_seconds() / 60.0
    gap_scale = time_diffs / interval_minutes
    gap_scale = gap_scale.clip(lower=1.0)  # Never scale down
    diffs_adj = diffs_adj / gap_scale

    # Ensure no NaNs at start
    diffs_adj.iloc[0] = 0

    return diffs_adj.astype('float64')


def resample_to_grid(
    series: pd.Series,
    freq: str,
    method: Literal['mean', 'sum', 'max', 'min', 'last', 'forward_fill'] = 'mean',
) -> pd.Series:
    """
    Resample time series to a regular frequency grid.

    Fills gaps using specified method and forward-fills remaining NaNs.

    Parameters
    ----------
    series : pd.Series
        Time series with DatetimeIndex.
    freq : str
        Resampling frequency (e.g. '30min', '1H', '1D').
    method : str, default 'mean'
        Aggregation method: 'mean', 'sum', 'max', 'min', 'last', or 'forward_fill'.

    Returns
    -------
    pd.Series
        Resampled series on regular grid.

    Notes
    -----
    Always applies forward-fill after aggregation to handle sparse data gracefully.
    """
    if not isinstance(series, pd.Series):
        raise TypeError('series must be a pandas Series')
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError('series must have DatetimeIndex')

    series = series.copy().dropna()

    if len(series) == 0:
        logger.warning('Empty series after dropna')
        return pd.Series([], dtype='float64')

    # Resample
    if method == 'forward_fill':
        resampled = series.resample(freq).last()
    else:
        resampled = getattr(series.resample(freq), method)()

    # Forward-fill gaps
    resampled = resampled.ffill()

    # Back-fill any leading NaNs
    resampled = resampled.bfill()

    return resampled.astype('float64')


def clip_outliers(
    series: pd.Series,
    quantile: float = 0.95,
    positive_only: bool = False,
) -> pd.Series:
    """
    Clip extreme outliers using quantile-based bounds.

    Parameters
    ----------
    series : pd.Series
        Input series to process.
    quantile : float, default 0.95
        Quantile for upper bound; lower bound is 1 - quantile.
    positive_only : bool, default False
        If True, only clip upper tail (assume data is non-negative).

    Returns
    -------
    pd.Series
        Series with outliers clipped to bounds.

    Notes
    -----
    Uses symmetric quantiles: lower = (1-q), upper = q.
    If positive_only=True, lower bound is 0.
    """
    if not 0 < quantile < 1:
        raise ValueError(f'quantile must be in (0, 1), got {quantile}')

    series = series.copy()
    non_null = series.dropna()

    if len(non_null) < 2:
        logger.warning('Insufficient data for outlier clipping')
        return series

    if positive_only:
        lower = 0.0
        upper = non_null.quantile(quantile)
    else:
        lower = non_null.quantile(1 - quantile)
        upper = non_null.quantile(quantile)

    clipped = series.clip(lower, upper)
    n_clipped = (clipped != series).sum()
    if n_clipped > 0:
        logger.debug(f'Clipped {n_clipped} outlier values')

    return clipped


def apply_log_transform(series: pd.Series, shift: float = 1.0) -> pd.Series:
    """
    Apply log transformation with shift for non-positive values.

    Parameters
    ----------
    series : pd.Series
        Input series.
    shift : float, default 1.0
        Shift applied before log: log(x + shift).

    Returns
    -------
    pd.Series
        Log-transformed series.

    Notes
    -----
    Stores shift value as attribute for later inversion.
    """
    if shift < 0:
        raise ValueError(f'shift must be >= 0, got {shift}')

    series = series.copy()
    transformed = np.log(series + shift)
    transformed.attrs['log_shift'] = shift

    return transformed


def invert_log_transform(
    series: pd.Series,
    shift: Optional[float] = None,
) -> pd.Series:
    """
    Invert log transformation.

    Parameters
    ----------
    series : pd.Series
        Log-transformed series.
    shift : float, optional
        Shift value used during transformation.
        If None, attempts to read from series.attrs['log_shift'].

    Returns
    -------
    pd.Series
        Original-scale series.
    """
    if shift is None:
        shift = series.attrs.get('log_shift', 1.0)

    series = series.copy()
    inverted = np.exp(series) - shift

    return inverted


def subtract_series(
    base: pd.Series,
    subtract: pd.Series,
    fill_method: str = 'ffill',
) -> pd.Series:
    """
    Vectorised subtraction of two time series with reindexing.

    Parameters
    ----------
    base : pd.Series
        Primary series.
    subtract : pd.Series
        Series to subtract from base.
    fill_method : str, default 'ffill'
        How to fill misaligned indices: 'ffill' (forward-fill), 'bfill', or 'interpolate'.

    Returns
    -------
    pd.Series
        Result of base - subtract, aligned to base's index.

    Notes
    -----
    Automatically reindexes subtract to match base, filling gaps as specified.
    """
    if not isinstance(base, pd.Series) or not isinstance(subtract, pd.Series):
        raise TypeError('Both arguments must be pandas Series')

    result = base.copy()

    # Reindex subtract to base
    subtract_aligned = subtract.reindex(base.index)

    # Fill missing values
    if fill_method == 'ffill':
        subtract_aligned = subtract_aligned.ffill()
    elif fill_method == 'bfill':
        subtract_aligned = subtract_aligned.bfill()
    elif fill_method == 'interpolate':
        subtract_aligned = subtract_aligned.interpolate(method='linear')
    else:
        raise ValueError(
            f"fill_method must be 'ffill', 'bfill', or 'interpolate', got {fill_method!r}"
        )

    result = result - subtract_aligned

    return result.ffill()


def power_to_energy(
    series: pd.Series,
    interval_minutes: int,
    units: str = 'W',
) -> pd.Series:
    """
    Convert power (W/kW) to energy (Wh/kWh) over interval.

    Parameters
    ----------
    series : pd.Series
        Power values (in watts or kilowatts).
    interval_minutes : int
        Sampling interval in minutes.
    units : str, default 'W'
        Input units: 'W' (watts) or 'kW' (kilowatts).

    Returns
    -------
    pd.Series
        Energy in Wh (if W input) or kWh (if kW input).

    Notes
    -----
    Energy = Power * Time. Assumes constant power over interval.

    Examples
    --------
    >>> idx = pd.date_range('2024-01-01', periods=48, freq='30min')
    >>> power_w = pd.Series(np.full(48, 1000.0), index=idx)
    >>> energy_wh = power_to_energy(power_w, 30, units='W')
    >>> assert (energy_wh == 500.0).all()  # 1000 W * 0.5 hours
    """
    if units not in {'W', 'kW'}:
        raise ValueError(f"units must be 'W' or 'kW', got {units!r}")
    if interval_minutes < 1:
        raise ValueError(f'interval_minutes must be >= 1, got {interval_minutes}')

    series = series.copy()
    hours = interval_minutes / 60.0

    energy = series * hours

    return energy.astype('float64')


def apply_transform(
    series: pd.Series,
    transform: Optional[str],
) -> pd.Series:
    """
    Apply optional transformation to series.

    Parameters
    ----------
    series : pd.Series
        Input series.
    transform : str, optional
        Transformation type: 'log', 'sqrt', 'box_cox', or None.

    Returns
    -------
    pd.Series
        Transformed series (or original if transform is None).

    Notes
    -----
    Stores transformation metadata in series.attrs for later inversion.
    """
    if transform is None:
        return series.copy()

    series = series.copy()

    if transform == 'log':
        # Shift before log if necessary
        min_val = series.min()
        shift = 0.0 if min_val >= 0 else abs(min_val) + 1.0
        series = np.log(series + shift)
        series.attrs['transform'] = 'log'
        series.attrs['transform_shift'] = shift

    elif transform == 'sqrt':
        # Clip negative values
        series = series.clip(lower=0)
        series = np.sqrt(series)
        series.attrs['transform'] = 'sqrt'

    elif transform == 'box_cox':
        # Simplified Box-Cox: shift to positive and use log
        min_val = series.min()
        shift = 1.0 if min_val >= 0 else abs(min_val) + 1.0
        series = np.log(series + shift)
        series.attrs['transform'] = 'box_cox'
        series.attrs['transform_shift'] = shift

    else:
        raise ValueError(
            f"transform must be 'log', 'sqrt', 'box_cox', or None, got {transform!r}"
        )

    return series


def invert_transform(
    series: pd.Series,
    transform: Optional[str],
) -> pd.Series:
    """
    Invert a transformation applied to series.

    Parameters
    ----------
    series : pd.Series
        Transformed series (should have transform metadata if auto-inverting).
    transform : str, optional
        Transformation type. If None, attempts to read from series.attrs.

    Returns
    -------
    pd.Series
        Original-scale series.
    """
    if transform is None:
        transform = series.attrs.get('transform')

    if transform is None:
        return series.copy()

    series = series.copy()

    if transform == 'log':
        shift = series.attrs.get('transform_shift', 0.0)
        series = np.exp(series) - shift

    elif transform == 'sqrt':
        series = series ** 2

    elif transform == 'box_cox':
        shift = series.attrs.get('transform_shift', 1.0)
        series = np.exp(series) - shift

    else:
        raise ValueError(
            f"transform must be 'log', 'sqrt', 'box_cox', or None, got {transform!r}"
        )

    return series


def align_series(
    series_list: list[pd.Series],
    method: str = 'inner',
) -> list[pd.Series]:
    """
    Align multiple series to a common index.

    Parameters
    ----------
    series_list : list of pd.Series
        Series to align.
    method : str, default 'inner'
        Join method: 'inner', 'outer', 'left', 'right'.

    Returns
    -------
    list of pd.Series
        Aligned series.

    Notes
    -----
    Useful for combining target and covariates with potentially different timestamps.
    """
    if not series_list:
        return []

    if len(series_list) == 1:
        return series_list.copy()

    # Concatenate with join method
    combined = pd.concat(series_list, axis=1, join=method)

    # Split back
    return [combined.iloc[:, i] for i in range(len(series_list))]
