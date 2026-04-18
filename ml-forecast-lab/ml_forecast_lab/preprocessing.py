"""
Unified preprocessing pipeline for ML Forecast Lab.

Consolidates cumulative-to-interval conversion, resampling, outlier handling,
and transformations into a single coherent set of functions.
"""

import logging
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LoadSubtractError(ValueError):
    """Raised when load-subtract fails a fail-fast robustness check.

    The message includes the offending entity_id, the observed violation
    rate, and the threshold that was exceeded — enough to diagnose a unit
    mismatch or a double-counted signal without re-running training.
    """


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
        current_dates = series.index.normalize()
        previous_dates = pd.Series(current_dates, index=series.index).shift(1)
        midnight_crosses = current_dates != previous_dates
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
    quantile: float = 0.995,
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


def apply_load_subtract(
    load: pd.Series,
    subtracts: Sequence[Tuple[Mapping[str, Any], pd.Series]],
) -> Tuple[pd.Series, dict]:
    """Subtract one or more sensor signals from a load series, robustly.

    Each ``subtracts`` entry is ``(cfg, series)`` where ``cfg`` is a mapping
    exposing ``SubtractCfg`` fields (``entity_id``, ``on_missing``, ``scale``,
    ``max_fraction_of_load``, ``max_fraction_violation_pct``) and ``series``
    is an already-preprocessed pandas ``Series`` on the same grid as
    ``load`` (i.e. ``cumulative_to_interval`` / ``resample_to_grid`` already
    applied by the caller, which owns cumulative semantics via
    ``cfg.source``).

    This function is ONLY responsible for subtracting — the fetch / cumulative
    interpretation happens upstream. That separation keeps ``preprocessing.py``
    independent of the HA client and lets tests pass synthetic frames.

    Robustness checklist (each item returns audit data rather than silent
    behaviour):

    - **on_missing**: ``zero`` fills gap rows with 0.0; ``drop`` drops the
      parent-load row for that timestamp; ``error`` raises
      ``LoadSubtractError``. Never silently coerce NaN to 0 unless you ask.
    - **scale**: applied before subtraction to fix unit mismatches (Wh→kWh).
    - **history coverage gap**: if the subtract series is shorter than the
      load, the first/last gap windows are recorded in ``audit['per_sensor']``
      with ``gap_start``/``gap_end``.
    - **negative clip**: after subtraction, the result is ``clip(lower=0)``
      and the number of clipped rows is counted. > 5 % → warning.
    - **fraction guard**: per-row ``(sum_subtract / load)`` is checked against
      ``max_fraction_of_load``. If the violation rate exceeds
      ``max_fraction_violation_pct`` for ANY sensor, raises
      ``LoadSubtractError`` with a diagnostic message. This is the unit-bug
      canary.
    - **timezone**: load and subtract series must have matching index
      tz-awareness (both naive or both aware). Mismatch raises before
      anything else runs.

    Parameters
    ----------
    load : pd.Series
        Target load, already on the experiment's interval grid. Must have a
        ``DatetimeIndex``.
    subtracts : sequence of (cfg_mapping, pd.Series)
        One entry per subtract sensor.

    Returns
    -------
    (adjusted_load, audit) : (pd.Series, dict)
        ``adjusted_load`` is ``load - Σ subtracts`` clipped to [0, ∞), reindexed
        by ``load.index`` (minus any rows dropped via ``on_missing='drop'``).

        ``audit`` has shape::

            {
                "n_rows": int,                 # rows after any drops
                "n_clipped_rows": int,         # rows where raw result < 0
                "clipped_pct": float,          # 100 * n_clipped_rows / n_rows
                "load_total_kwh": float,       # Σ load on the kept index
                "subtract_total_kwh": float,   # Σ (Σ subtracts) on the kept index
                "per_sensor": [
                    {
                        "entity_id": str,
                        "rows_present": int,       # non-NaN after reindex
                        "rows_missing": int,       # NaN after reindex
                        "rows_dropped": int,       # how many load rows removed
                        "mean_kwh": float,         # mean of post-scale values
                        "sum_kwh": float,
                        "max_fraction": float,     # max (this_sensor / load)
                        "violation_rows": int,     # rows > max_fraction_of_load
                        "violation_pct": float,
                        "gap_start": str | None,   # ISO ts of first missing window
                        "gap_end": str | None,
                    },
                    ...
                ],
            }

    Raises
    ------
    TypeError
        If ``load`` is not a ``pd.Series`` with a ``DatetimeIndex``.
    ValueError
        If a subtract series has mismatched tz-awareness, or a sensor is
        missing but ``on_missing='error'``.
    LoadSubtractError
        If any sensor exceeds its ``max_fraction_violation_pct`` threshold.
    """
    if not isinstance(load, pd.Series):
        raise TypeError("load must be a pandas Series")
    if not isinstance(load.index, pd.DatetimeIndex):
        raise TypeError("load must have a DatetimeIndex")

    if not subtracts:
        # Nothing to do. Return a copy so callers can't accidentally alias.
        return load.copy(), {
            "n_rows": len(load),
            "n_clipped_rows": 0,
            "clipped_pct": 0.0,
            "load_total_kwh": float(load.sum()) if len(load) else 0.0,
            "subtract_total_kwh": 0.0,
            "per_sensor": [],
        }

    load_tz_aware = load.index.tz is not None

    # --- Stage 1: align each subtract series, apply scale + on_missing. -----
    #
    # We build two parallel structures:
    #   - scaled_series[i]: the per-sensor series reindexed to load.index
    #     with scale applied and NaN handled per on_missing.
    #   - per_sensor_audit[i]: the diagnostic record we'll return.
    #
    # We also accumulate a boolean mask of rows to DROP (only used when any
    # sensor has on_missing='drop'). This mask is applied once at the end
    # so multiple 'drop' sensors compose correctly.
    scaled_series: list[pd.Series] = []
    per_sensor_audit: list[dict] = []
    drop_mask = pd.Series(False, index=load.index)

    for cfg, raw in subtracts:
        if not isinstance(raw, pd.Series):
            raise TypeError(
                f"subtract series for {cfg.get('entity_id', '?')} is not "
                f"a pandas Series"
            )
        if not isinstance(raw.index, pd.DatetimeIndex):
            raise TypeError(
                f"subtract series for {cfg.get('entity_id', '?')} must "
                f"have a DatetimeIndex"
            )
        raw_tz_aware = raw.index.tz is not None
        if raw_tz_aware != load_tz_aware:
            raise ValueError(
                f"subtract series for {cfg.get('entity_id', '?')} has "
                f"tz-{'aware' if raw_tz_aware else 'naive'} index but load "
                f"is tz-{'aware' if load_tz_aware else 'naive'}; normalise "
                f"upstream"
            )

        entity_id = cfg.get("entity_id", "?")
        on_missing = cfg.get("on_missing", "zero")
        scale = cfg.get("scale")

        # Reindex onto the load grid. Anything outside the raw series'
        # covered range is NaN — that's the signal we act on next.
        aligned = raw.reindex(load.index)
        if scale is not None:
            aligned = aligned * scale

        missing_mask = aligned.isna()
        rows_missing = int(missing_mask.sum())
        rows_present = len(aligned) - rows_missing

        # Detect history-coverage gap windows. We only report the FIRST
        # contiguous leading gap and the last trailing gap, because the
        # common case is "the sensor didn't exist before install date".
        # Interior gaps roll up into rows_missing without per-window detail.
        gap_start: Optional[str] = None
        gap_end: Optional[str] = None
        if rows_missing > 0:
            leading = missing_mask.values.argmax() if missing_mask.iloc[0] else -1
            if leading >= 0:
                # find end of leading gap (first non-missing)
                not_missing = (~missing_mask).values
                if not_missing.any():
                    first_present = int(not_missing.argmax())
                    gap_start = load.index[0].isoformat()
                    gap_end = load.index[first_present - 1].isoformat()
                else:
                    # All missing. Whole window is a gap.
                    gap_start = load.index[0].isoformat()
                    gap_end = load.index[-1].isoformat()

        # Apply on_missing policy.
        rows_dropped = 0
        if rows_missing > 0:
            if on_missing == "zero":
                aligned = aligned.fillna(0.0)
            elif on_missing == "drop":
                # Mark these rows for removal at the end.
                drop_mask = drop_mask | missing_mask
                rows_dropped = rows_missing
                aligned = aligned.fillna(0.0)  # placeholder; row will be dropped
            elif on_missing == "error":
                first_missing_ts = load.index[missing_mask.values.argmax()]
                raise ValueError(
                    f"load_subtract[{entity_id}]: {rows_missing} missing row(s) "
                    f"with on_missing='error' (first at {first_missing_ts})"
                )
            else:
                raise ValueError(
                    f"load_subtract[{entity_id}]: unknown on_missing "
                    f"{on_missing!r}"
                )

        scaled_series.append(aligned)
        per_sensor_audit.append({
            "entity_id": entity_id,
            "rows_present": rows_present,
            "rows_missing": rows_missing,
            "rows_dropped": rows_dropped,
            "mean_kwh": float(aligned.mean()) if len(aligned) else 0.0,
            "sum_kwh": float(aligned.sum()) if len(aligned) else 0.0,
            "max_fraction": 0.0,      # filled in below
            "violation_rows": 0,       # filled in below
            "violation_pct": 0.0,      # filled in below
            "gap_start": gap_start,
            "gap_end": gap_end,
        })

    # --- Stage 2: per-sensor fraction-of-load guard (pre-drop). -------------
    #
    # Evaluate each sensor's ratio to load on rows where load > 0. Division by
    # zero/near-zero is the common noise case (e.g. sleeping household), not a
    # unit bug, so we exclude near-zero load rows from the guard rather than
    # letting them dominate the violation rate.
    #
    # We check BEFORE applying the drop mask so the guard sees the full signal
    # — dropping rows then checking would hide the bug we're trying to catch.
    NEAR_ZERO_LOAD = 1e-9
    valid_load_mask = load.abs() > NEAR_ZERO_LOAD
    n_valid = int(valid_load_mask.sum())

    for cfg, aligned, audit in zip(
        (c for c, _ in subtracts), scaled_series, per_sensor_audit
    ):
        max_fraction = cfg.get("max_fraction_of_load", 1.0)
        max_violation_pct = cfg.get("max_fraction_violation_pct", 5.0)

        if n_valid == 0:
            continue  # can't evaluate a ratio; let downstream handle it

        ratio = (aligned[valid_load_mask].abs()
                 / load[valid_load_mask].abs())
        # NaN means load was non-zero but aligned was NaN; treat conservatively
        # as zero ratio (after on_missing handling, NaN shouldn't appear for
        # 'zero'/'drop'; 'error' would have already raised).
        ratio = ratio.fillna(0.0)
        audit["max_fraction"] = float(ratio.max()) if len(ratio) else 0.0

        violations = ratio > max_fraction
        n_violations = int(violations.sum())
        violation_pct = 100.0 * n_violations / n_valid
        audit["violation_rows"] = n_violations
        audit["violation_pct"] = violation_pct

        if violation_pct > max_violation_pct:
            # Find the worst offender for the diagnostic message.
            worst_idx = ratio.idxmax()
            worst_ratio = float(ratio.max())
            worst_load = float(load.loc[worst_idx])
            worst_sub = float(aligned.loc[worst_idx])
            raise LoadSubtractError(
                f"load_subtract[{audit['entity_id']}]: subtract exceeded "
                f"{max_fraction:.2f}× load on "
                f"{violation_pct:.2f}% of rows (threshold "
                f"{max_violation_pct:.2f}%). "
                f"Worst: {worst_idx} subtract={worst_sub:.3f} "
                f"load={worst_load:.3f} ratio={worst_ratio:.2f}. "
                f"Likely a unit mismatch (Wh vs kWh?) or the subtract sensor "
                f"measures a superset of the parent load."
            )

    # --- Stage 3: sum subtracts, subtract from load, clip, apply drops. -----
    total_subtract = sum(scaled_series, start=pd.Series(0.0, index=load.index))
    raw_adjusted = load - total_subtract

    # Negative-clip audit: rows where subtracts exceed load (usually small
    # measurement-noise band, but the count is a useful health metric).
    clipped_mask = raw_adjusted < 0
    n_clipped = int(clipped_mask.sum())
    adjusted = raw_adjusted.clip(lower=0.0)

    # Apply any 'drop' masks last so clip/fraction-guard see the full signal.
    if drop_mask.any():
        adjusted = adjusted[~drop_mask]
        kept_load = load[~drop_mask]
        kept_subtract = total_subtract[~drop_mask]
    else:
        kept_load = load
        kept_subtract = total_subtract

    n_rows = len(adjusted)
    clipped_pct = (100.0 * n_clipped / n_rows) if n_rows else 0.0

    if n_clipped > 0:
        # Noise band is usually < 1%. 5% is the warn threshold — any higher
        # almost certainly means a miscalibrated subtract rather than jitter.
        log = logger.warning if clipped_pct > 5.0 else logger.info
        log(
            f"load_subtract: clipped {n_clipped}/{n_rows} rows ({clipped_pct:.2f}%) "
            f"to zero after subtraction"
        )

    audit = {
        "n_rows": n_rows,
        "n_clipped_rows": n_clipped,
        "clipped_pct": clipped_pct,
        "load_total_kwh": float(kept_load.sum()) if len(kept_load) else 0.0,
        "subtract_total_kwh": (
            float(kept_subtract.sum()) if len(kept_subtract) else 0.0
        ),
        "per_sensor": per_sensor_audit,
    }

    return adjusted.astype("float64"), audit


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

    elif transform in ('shifted_log', 'box_cox'):
        # Shifted log: shift to positive and use log
        # ('box_cox' kept as alias for backward compatibility)
        min_val = series.min()
        shift = 1.0 if min_val >= 0 else abs(min_val) + 1.0
        series = np.log(series + shift)
        series.attrs['transform'] = 'shifted_log'
        series.attrs['transform_shift'] = shift

    else:
        raise ValueError(
            f"transform must be 'log', 'sqrt', 'shifted_log', or None, got {transform!r}"
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

    elif transform in ('shifted_log', 'box_cox'):
        shift = series.attrs.get('transform_shift', 1.0)
        series = np.exp(series) - shift

    else:
        raise ValueError(
            f"transform must be 'log', 'sqrt', 'shifted_log', or None, got {transform!r}"
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
