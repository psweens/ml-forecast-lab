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

    # Identify rows where the time gap exceeds one interval. These hold the
    # cumulative delta over multiple intervals — the per-row spike cap and
    # the gap-scale division below would silently distort them. Multi-
    # interval gaps are dropped to NaN downstream so resample_to_grid sees
    # them as missing rather than as either a fake spike or an under-
    # reported single bucket.
    time_diffs = series.index.to_series().diff().dt.total_seconds() / 60.0
    gap_scale = time_diffs / interval_minutes
    multi_interval_gap = gap_scale > 1.5

    # Cap spikes — but skip gap rows so a legitimate 4 h outage carrying
    # several intervals' worth of energy isn't clamped to max_increment.
    spike_mask = (diffs_adj > max_increment) & ~multi_interval_gap
    if spike_mask.any():
        logger.debug(f'Found {spike_mask.sum()} spike(s) exceeding max_increment')
        diffs_adj[spike_mask] = np.clip(
            diffs_adj[spike_mask], 0, max_increment
        )

    # Handle negative differences (after reset handling)
    diffs_adj[diffs_adj < 0] = 0

    # Multi-interval gaps — for a CUMULATIVE source, the value rising across
    # a gap is REAL demand that accumulated while the recorder logged no
    # rows. HA's recorder uses minimal_response (stores only state CHANGES),
    # so quiet periods — overnight, between hot-water draw-offs — leave gaps
    # with no rows; the draw-off that ends a quiet period then spans >1.5
    # intervals. The previous behaviour DROPPED that delta to NaN, and the
    # downstream sum-resample counts NaN as 0 — silently discarding the
    # demand and undercounting the daily total. For a daily-reset counter
    # like ``sensor.x_demand_today`` the post-quiet draw (e.g. the morning
    # shower after the overnight reset) carries much of the day, so dropping
    # it roughly halved the total.
    #
    # We now KEEP the full delta, attributed to the row where the change was
    # recorded — i.e. when the draw actually happened. That preserves the
    # daily total exactly, and is a better attribution than spreading it
    # across the quiet period, when no demand occurred. A genuine multi-hour
    # recorder OUTAGE puts the whole gap's demand in one bucket; the
    # downstream outlier clip handles any pathological spike. The
    # ``multi_interval_gap`` mask is still used above to EXEMPT these rows
    # from the per-row spike cap so a legitimate large draw isn't clamped to
    # ``max_increment``.
    if multi_interval_gap.any():
        logger.info(
            'cumulative_to_interval: %d row(s) span >1.5 intervals; '
            'keeping the accumulated delta (real demand recorded after a '
            'quiet period) so the daily total is preserved.',
            int(multi_interval_gap.sum()),
        )

    # Ensure no NaNs at start (no diff for the first observation)
    diffs_adj.iloc[0] = 0

    return diffs_adj.astype('float64')


def resample_to_grid(
    series: pd.Series,
    freq: str,
    method: Literal['mean', 'sum', 'max', 'min', 'last', 'forward_fill'] = 'mean',
    gap_handling: str = 'ffill',
    gap_max_minutes: int = 90,
) -> pd.Series:
    """
    Resample time series to a regular frequency grid with controlled gap fill.

    Parameters
    ----------
    gap_handling : {'ffill', 'interpolate', 'mask'}, default 'ffill'
        How to fill the gaps that resampling introduces. ``ffill`` is the
        legacy behaviour (propagate the last observed value across any gap).
        ``interpolate`` linearly interpolates gaps up to ``gap_max_minutes``
        and leaves longer gaps as NaN. ``mask`` leaves every gap as NaN so
        downstream dropna removes the rows entirely.
    gap_max_minutes : int, default 90
        Maximum interpolation horizon when ``gap_handling='interpolate'``.
    """
    if not isinstance(series, pd.Series):
        raise TypeError('series must be a pandas Series')
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError('series must have DatetimeIndex')

    series = series.copy().dropna()
    if len(series) == 0:
        logger.warning('Empty series after dropna')
        return pd.Series([], dtype='float64')

    if method == 'forward_fill':
        resampled = series.resample(freq).last()
    else:
        resampled = getattr(series.resample(freq), method)()

    if gap_handling == 'mask':
        return resampled.astype('float64')

    if gap_handling == 'interpolate':
        # Linear interpolation, but capped at gap_max_minutes so a multi-hour
        # outage isn't smoothed into a synthetic ramp.
        try:
            inferred = pd.Timedelta(freq)
            interval_min = max(1, int(inferred.total_seconds() // 60))
        except Exception:
            interval_min = 30
        max_steps = max(1, int(gap_max_minutes // interval_min))
        resampled = resampled.interpolate(
            method='linear', limit=max_steps, limit_direction='forward'
        )
        # Leading NaNs (before the first observation) still need a value;
        # back-fill ONLY those so the dataframe construction doesn't drop
        # them. A blanket ``bfill()`` would also reach backwards across
        # every interior gap that exceeds ``max_steps`` and fill it with
        # the next observation — e.g. an overnight PV gap would inherit
        # the following morning's value, planting non-zero readings into
        # the small hours (lookahead leakage). Restricting the back-fill
        # to the head leaves interior gaps as NaN so downstream dropna /
        # idle-value handling deals with them.
        first_valid = resampled.first_valid_index()
        if first_valid is not None:
            head = resampled.index <= first_valid
            resampled.loc[head] = resampled.loc[head].bfill()
        return resampled.astype('float64')

    # Default / legacy: forward-fill, then back-fill leading NaNs.
    resampled = resampled.ffill()
    resampled = resampled.bfill()
    return resampled.astype('float64')


def clip_outliers(
    series: pd.Series,
    quantile: float = 0.999,
    positive_only: bool = False,
    method: str = 'quantile',
    lower_bound: str = 'auto',
) -> pd.Series:
    """
    Clip extreme outliers.

    Parameters
    ----------
    series : pd.Series
        Input series to process.
    quantile : float, default 0.999
        Upper-tail quantile when ``method='quantile'``.
    positive_only : bool, default False
        Legacy alias for ``lower_bound='zero'`` when ``lower_bound='auto'``.
    method : {'quantile', 'mad', 'off'}, default 'quantile'
        ``mad`` uses median ± 3.5 · MAD (Iglewicz-Hoaglin). ``off`` returns
        the series unchanged.
    lower_bound : {'auto', 'zero', 'symmetric', 'off'}, default 'auto'
        Lower-bound policy for ``method='quantile'``. ``auto`` uses zero
        when ``positive_only`` is True else the symmetric (1-q) quantile.
    """
    if method == 'off':
        return series.copy()
    if method not in ('quantile', 'mad'):
        raise ValueError(
            f"method must be 'quantile', 'mad', or 'off', got {method!r}"
        )
    if method == 'quantile' and not 0 < quantile < 1:
        raise ValueError(f'quantile must be in (0, 1), got {quantile}')

    series = series.copy()
    non_null = series.dropna()
    if len(non_null) < 2:
        logger.warning('Insufficient data for outlier clipping')
        return series

    if method == 'mad':
        median = float(non_null.median())
        mad = float((non_null - median).abs().median())
        if mad == 0:
            return series
        k = 3.5
        lower = median - k * mad
        upper = median + k * mad
        if positive_only:
            lower = max(lower, 0.0)
    else:
        upper = non_null.quantile(quantile)
        if lower_bound == 'off':
            lower = float(non_null.min())
        elif lower_bound == 'zero':
            lower = 0.0
        elif lower_bound == 'symmetric':
            lower = non_null.quantile(1 - quantile)
        else:  # 'auto'
            lower = 0.0 if positive_only else non_null.quantile(1 - quantile)

    clipped = series.clip(lower, upper)
    n_clipped = (clipped != series).sum()
    if n_clipped > 0:
        logger.debug(f'Clipped {n_clipped} outlier values')
    return clipped


def apply_log_transform(
    series: pd.Series, shift: Optional[float] = None
) -> pd.Series:
    """
    Apply ``log(x + shift)`` to a series.

    When ``shift`` is None the shift is derived from the data:
    ``max(1.0, abs(min(series)) + 1.0)`` so the transform is well-defined
    even on signed targets, while remaining identical to the legacy
    ``shift=1.0`` path for the non-negative HA targets the rest of the
    pipeline produces. The chosen shift is stored on ``series.attrs`` so
    ``invert_log_transform`` can read it back without the caller having
    to thread it through.
    """
    series = series.copy()
    if shift is None:
        try:
            min_val = float(series.min())
        except (TypeError, ValueError):
            min_val = 0.0
        shift = 1.0 if min_val >= 0 else abs(min_val) + 1.0
    if shift < 0:
        raise ValueError(f'shift must be >= 0, got {shift}')

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
      load, the *leading* gap window is recorded in ``audit['per_sensor']``
      under ``gap_start``/``gap_end``, and the *trailing* gap window (sensor
      stopped reporting before the load did) is recorded under
      ``trailing_gap_start``/``trailing_gap_end``. Interior gaps roll up
      into ``rows_missing`` without per-window detail.
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
                        "gap_start": str | None,   # ISO ts of leading-gap start
                        "gap_end": str | None,     # ISO ts of leading-gap end
                        "trailing_gap_start": str | None,  # ISO ts of trailing-gap start
                        "trailing_gap_end": str | None,    # ISO ts of trailing-gap end
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

        # Detect history-coverage gap windows. We report the FIRST
        # contiguous leading gap (sensor didn't exist before install date)
        # and the LAST contiguous trailing gap (sensor stopped reporting),
        # because those are the two cases that benefit from per-window
        # diagnostics. Interior gaps roll up into rows_missing without
        # per-window detail.
        gap_start: Optional[str] = None
        gap_end: Optional[str] = None
        trailing_gap_start: Optional[str] = None
        trailing_gap_end: Optional[str] = None
        if rows_missing > 0:
            not_missing = (~missing_mask).values
            if not not_missing.any():
                # All missing. Whole window is a gap — only record it as the
                # leading gap to avoid double-counting via trailing fields.
                gap_start = load.index[0].isoformat()
                gap_end = load.index[-1].isoformat()
            else:
                first_present = int(not_missing.argmax())
                # Leading gap: zero or more rows of NaN before the first
                # non-missing row.
                if missing_mask.iloc[0]:
                    gap_start = load.index[0].isoformat()
                    gap_end = load.index[first_present - 1].isoformat()
                # Trailing gap: zero or more rows of NaN after the last
                # non-missing row. argmax on the reversed array finds the
                # last True (since argmax returns the first True index).
                if missing_mask.iloc[-1]:
                    last_present = len(missing_mask) - 1 - int(not_missing[::-1].argmax())
                    trailing_gap_start = load.index[last_present + 1].isoformat()
                    trailing_gap_end = load.index[-1].isoformat()

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
            "trailing_gap_start": trailing_gap_start,
            "trailing_gap_end": trailing_gap_end,
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
        # Shift before log if necessary. log(0) = -inf, so any data with
        # exact zeros (night-time PV, off-state load, midnight cumulative
        # resets) needs a small positive shift even when min_val == 0.
        min_val = series.min()
        if min_val > 0:
            shift = 0.0
        elif min_val == 0:
            shift = 1.0
        else:
            shift = abs(min_val) + 1.0
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


# A diverged model can emit a large value in log space; np.expm1 then explodes
# it (expm1(70) ≈ 2.5e30, and it overflows float32 to inf beyond ~89). Real
# demand / PV is physically bounded, so a forecast beyond a generous multiple
# of the largest value ever observed is a divergence, not a forecast — cap it.
# The cap is intentionally loose (10× the historical max) so it never touches a
# plausible forecast, only a blow-up.
#
# This lives here (not in main.py) so BOTH the production publish path AND the
# benchmark runner — which can't import main without a circular import — share
# the exact same rule. That keeps every analysis surface (leaderboard CV
# metrics, holdout chart, published sensors) clamping a log-inversion blow-up
# identically, instead of only the publish boundary being guarded.
FORECAST_BLOWUP_CAP_FACTOR = 10.0


def clamp_forecast_blowup(
    values: "np.ndarray",
    ref_max_display: Optional[float],
    factor: float = FORECAST_BLOWUP_CAP_FACTOR,
) -> Tuple["np.ndarray", int, Optional[float]]:
    """Cap a display-space forecast to ``factor`` × the largest observed value.

    A ``log_transform`` inversion (``np.expm1``) of a diverged log-space
    prediction explodes to ~1e30 (or ``inf``). Such a value is not a forecast —
    it would dominate any MAE/RMSE it enters and flatten any chart. This caps it
    to a generous multiple of the reference scale so a blow-up can never reach a
    metric, a chart, or a published sensor, while a plausible forecast is never
    touched.

    Parameters
    ----------
    values : array-like
        Forecast values in display (un-transformed) units.
    ref_max_display : float or None
        Largest observed magnitude in display units (e.g. the fold's / training
        actuals' max). ``None`` / non-finite / ``<= 0`` disables the cap.
    factor : float, default ``FORECAST_BLOWUP_CAP_FACTOR``
        Multiplier applied to ``ref_max_display`` to form the cap.

    Returns
    -------
    (clamped, n_clamped, cap) : (np.ndarray float32, int, float | None)
        ``cap`` is ``None`` when no reference was usable.
    """
    y = np.asarray(values, dtype=np.float32)
    if ref_max_display is None:
        return y, 0, None
    try:
        ref = float(ref_max_display)
    except (TypeError, ValueError):
        return y, 0, None
    if not np.isfinite(ref) or ref <= 0:
        return y, 0, None
    cap = float(factor) * ref
    over = np.isfinite(y) & (y > cap)
    n = int(np.count_nonzero(over))
    if n:
        y = np.minimum(y, np.float32(cap))
    return y.astype(np.float32), n, cap


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
        Join method:
        - ``'inner'``: intersection of all indices (no NaNs introduced).
        - ``'outer'``: union of all indices (NaNs where a series doesn't cover a
          timestamp).
        - ``'left'``: anchor on the first series' index; other series are
          reindexed (and may pick up NaNs at unmatched timestamps).
        - ``'right'``: anchor on the last series' index; other series are
          reindexed.

    Returns
    -------
    list of pd.Series
        Aligned series, in the same order as ``series_list``.

    Notes
    -----
    Useful for combining target and covariates with potentially different
    timestamps. ``'left'`` and ``'right'`` are implemented via
    ``Series.reindex`` rather than ``pd.concat`` because ``pd.concat`` only
    supports ``'inner'``/``'outer'`` joins.
    """
    if not series_list:
        return []

    if len(series_list) == 1:
        return [series_list[0].copy()]

    if method in ('inner', 'outer'):
        combined = pd.concat(series_list, axis=1, join=method)
        return [combined.iloc[:, i] for i in range(len(series_list))]

    if method == 'left':
        anchor = series_list[0].index
        return [series_list[0].copy()] + [s.reindex(anchor) for s in series_list[1:]]

    if method == 'right':
        anchor = series_list[-1].index
        return [s.reindex(anchor) for s in series_list[:-1]] + [series_list[-1].copy()]

    raise ValueError(
        f"method must be 'inner', 'outer', 'left', or 'right', got {method!r}"
    )


# --------------------------------------------------------------------------- #
# Data-shape diagnostics
# --------------------------------------------------------------------------- #
# Two descriptive statistics for the Data Sanity Check. Everything else that
# panel reports (min / median / max / std, zero runs, coverage, gaps) can be
# read off the raw recorder rows; these two need the series on its regular
# interval grid, and neither is derivable from the others.
#
# They exist to answer two questions a user otherwise has to guess at:
# "does my sensor have a daily rhythm?" (which decides whether seasonal_mase
# and calendar features are worth anything) and "is this a spiky load?" (which
# decides whether to reach for peak_weighted_mae or the DILATE loss).

def spikiness(series: "pd.Series") -> Optional[float]:
    """Peak-to-mean ratio: ``p99 / mean(|y|)``.

    How concentrated the mass is. A mostly-off spike train (hot water, EV) has
    a small mean and a large p99, so it scores high; a smooth daytime bump
    spreads its mass and scores low — even though both spend about half the
    day near zero. Standard deviation does not separate those cases, which is
    why this is not derivable from what the panel already reports.

    **What the number means.** For a non-negative load that rests near zero,
    this is the reciprocal of the duty cycle — verified against synthetic loads
    from 2% to 75% on-time, where ``spikiness × duty`` stays within 0.79-1.05,
    and it is invariant to amplitude, units and interval. So a reading of 8
    means "on about an eighth of the time", i.e. roughly three hours a day.
    That is what makes the reported bands transferable rather than tuned to one
    person's sensors.

    **Where it stops meaning that.** It measures mass concentrated *relative to
    zero*, not variability, so a standing baseline dilutes it. The same 3 kW
    load at an 8.3% duty reads 11.3 on no baseline, 5.99 on 0.3 kW, 3.17 on
    1 kW and 1.84 on 3 kW. A signal that never approaches zero — a tank
    temperature, a charge percentage — reads 1.2-2.0 however much it moves, so
    the number is uninformative there rather than wrong. A small standby draw
    is harmless: a CT clamp reading 0.5 W against 3 kW peaks is 0.017% of the
    peak and does not move the ratio.

    Returns ``None`` when there is nothing to measure.
    """
    y = pd.to_numeric(pd.Series(series), errors="coerce").dropna().to_numpy(dtype=float)
    if y.size < 2:
        return None
    abs_y = np.abs(y)
    mean_abs = float(np.mean(abs_y))
    # Guard a near-zero mean without letting the guard dominate the ratio.
    scale = float(np.median(abs_y[abs_y > 0])) if np.any(abs_y > 0) else 0.0
    eps = max(1e-9, 1e-3 * scale)
    if mean_abs + eps <= 0:
        return None
    val = float(np.quantile(y, 0.99)) / (mean_abs + eps)
    return val if np.isfinite(val) else None


def daily_autocorrelation(
    on_grid: "pd.Series", interval_minutes: int
) -> Optional[float]:
    """Correlation between the series and itself one day earlier (-1..1).

    Takes the series **with its gaps** — on the regular interval grid, not
    compacted. Dropping NaN first shifts every sample after a hole, so a
    positional lag stops being 24 hours the moment the recorder misses a
    reading, and the error compounds with each subsequent gap. Measured on a
    synthetic tank with a genuine 0.99 daily rhythm, a 5% hole rate reads 0.06
    and 10% goes negative.

    ``Series.corr`` uses pairwise-complete observations, so gaps cost only the
    pairs they touch and the lag stays a true 24 hours.

    Returns ``None`` when it cannot be determined — fewer than two full days,
    or either side constant.
    """
    lag = max(1, int(round(24 * 60 / max(1, int(interval_minutes)))))
    s = pd.to_numeric(pd.Series(on_grid), errors="coerce")
    if lag < 1 or s.size <= 2 * lag:
        return None

    # Remove everything slower than a day before correlating.
    #
    # A raw lag-24h correlation cannot tell a daily rhythm from a trend: any
    # slowly-varying signal correlates with itself at every lag. Measured on
    # the raw form — a pure linear ramp with no cycle at all read 1.000, a
    # random walk 0.968, and a purely WEEKLY rhythm 0.627, all of which would
    # be reported as a daily pattern and push the user toward seasonal metrics
    # and calendar features that cannot help them.
    #
    # Subtracting a one-day centred rolling mean is a high-pass filter at
    # exactly the scale of interest: a full daily cycle averages to ~0 over a
    # day so it survives intact, while a trend, a random walk or a weekly
    # component is tracked by the rolling mean and cancels. `min_periods` lets
    # the filter tolerate the recorder gaps this function exists to survive.
    baseline = s.rolling(lag, center=True, min_periods=max(2, lag // 2)).mean()
    resid = s - baseline

    # A signal with no within-day structure leaves a residual that is pure
    # numerical dust; correlating that yields a meaningless number. Compare
    # against the original scale rather than an absolute floor, so this holds
    # for a sensor in watts and the same sensor in kilowatts.
    ref = float(s.std(skipna=True))
    if not np.isfinite(ref) or ref <= 0:
        return None
    if float(resid.std(skipna=True) or 0.0) < 1e-6 * ref:
        return 0.0

    paired = pd.concat([resid, resid.shift(lag)], axis=1).dropna()
    if len(paired) < 2:
        return None
    a, b = paired.iloc[:, 0], paired.iloc[:, 1]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    r = a.corr(b)
    return None if pd.isna(r) else float(r)


# --- Missingness resolution ------------------------------------------------
#
# Everything below decides what a model is shown where data is absent. Two
# rules, and the split between them is the whole design:
#
#   * A missing **label** means the row cannot be a supervised sample. It is
#     excluded, never imputed. Imputing a label teaches the model something
#     false and then scores it against that fabrication — and because imputed
#     values are smooth, they are unusually easy to predict, so every backend
#     is flattered and the composite ranking starts rewarding whichever model
#     best reproduces the imputation scheme. Conformal bands would calibrate
#     on fabricated residuals for the same reason.
#   * A missing **feature** is masked, flagged and imputed. Imputing a feature
#     adds noise the model can learn to discount, which is what the flag is
#     for.


# Suffix for the companion indicator column. Mirrors
# ``features.MISSING_SUFFIX``; both are pinned equal by a unit test.
MISSING_SUFFIX = '_missing'

# Feature columns ``build_features`` derives from the target itself. Their
# gaps all trace back to one cause — a hole in ``y`` — so they share a single
# indicator instead of each carrying its own. One target gap otherwise emits
# roughly two dozen companions, most of them a handful of non-zero cells in
# tens of thousands of rows, and the *set* of them changes every cycle as the
# history window slides over the gap. A feature matrix whose column list is
# unstable between a retrain and the next forecast is worse than a coarse
# flag: the cached channel order stops matching and the forecast is refused.
TARGET_FEATURE_PREFIXES = ('y_lag_', 'y_rolling_')
TARGET_FEATURE_NAMES = ('y_diff_1',)
TARGET_MISSING_COLUMN = 'y' + MISSING_SUFFIX


def is_target_derived_feature(column: str, target_col: str = 'target') -> bool:
    """Whether ``column`` is a feature computed from the target's own history."""
    if column == target_col:
        return False
    return (
        column.startswith(TARGET_FEATURE_PREFIXES)
        or column in TARGET_FEATURE_NAMES
    )



def is_binary_column(series: pd.Series) -> bool:
    """Whether every observed value of ``series`` is 0 or 1.

    A binary covariate is a step function: upstream it is resampled with
    ``last().ffill()`` under a recorder where "no row" means "did not move".
    An expanding median over such a column returns 0.5 whenever an even
    number of prior observations splits evenly — a value the model never
    sees in any observed row of that channel, and meaningless for a state
    that is either on or off.
    """
    obs = series.dropna()
    if obs.empty:
        return False
    return bool(np.isin(obs.to_numpy(), (0.0, 1.0)).all())


def causal_impute(
    series: pd.Series,
    method: str = 'median',
) -> Tuple[pd.Series, pd.Series]:
    """
    Fill a feature column's gaps using only observations that precede them.

    ``method='hold'`` carries the last prior observation forward instead of
    taking an expanding median — equally leak-free, and the only in-domain
    answer for a binary channel.

    Returns ``(filled, imputed_mask)`` where ``imputed_mask`` is True exactly
    where a value was invented.

    Imputing with a statistic computed over the whole window would leak
    test-fold information into training, because the CV harness splits
    *after* preprocessing: a median taken over all of 2026 is partly a
    summary of the fold being scored. An expanding median over strictly
    prior observations is leak-free by construction and, unlike a per-fold
    statistic, gives the same answer no matter how the folds are drawn.

    Leading gaps have no prior observation, so they take the first observed
    value — a single scalar drawn from the boundary. That is a small, real
    leak and it is accepted deliberately: the flag is 1 across the whole
    leading region, so the model has an explicit signal to discount the
    column there. It is recorded here so it is not mistaken for an oversight.

    A column with no observations at all cannot be imputed from itself; it
    is filled with 0.0 and flagged everywhere, which is what the caller
    wants for a covariate that returned no usable history.
    """
    missing = series.isna()
    if not missing.any():
        return series, missing

    observed = series.dropna()
    if observed.empty:
        # Nothing to impute from. 0.0 with the flag raised for every row is
        # honest: the column carries no information and says so.
        return series.fillna(0.0), missing

    # ``expanding().median()`` at row t includes row t; the shift makes it
    # strictly prior. NaNs are skipped, so the statistic is over observed
    # values only.
    #
    # Only the prefix up to the last gap is ever read, and the expanding
    # median is the expensive part of the whole missingness step — on a
    # two-year half-hourly window it is most of the cost, and it runs on
    # every forecast tick. A gap early in a long window then costs a
    # fraction of the full pass instead of all of it.
    last_gap = int(np.flatnonzero(missing.to_numpy())[-1])
    head = series.iloc[: last_gap + 1]
    if method == 'hold':
        prior = head.ffill().shift(1)
    else:
        prior = head.expanding().median().shift(1)
    filled = series.copy()
    filled.iloc[: last_gap + 1] = head.fillna(prior)
    # Leading region — no prior observation existed.
    filled = filled.fillna(float(observed.iloc[0]))
    return filled, missing


def _interaction_base(
    column: str, frame_columns, exclude=(),
) -> Optional[Tuple[str, str]]:
    """Split ``<base>_x_hour_sin`` into ``(base, 'hour_sin')`` when both exist.

    ``build_features`` emits one interaction pair per covariate. Their NaNs
    are exactly the base column's NaNs, so a separate indicator for each
    would triple the column cost of a gappy covariate and say nothing new —
    and the honest imputed value for the interaction is the imputed base
    times the (always known) time term, not a median of the product.
    """
    if column in exclude:
        # A covariate of the source frame, not something build_features
        # produced. Rebuilding it as base x factor would overwrite every
        # one of its real measurements, not just its gaps.
        return None
    for factor in ('hour_sin', 'hour_cos'):
        suffix = f'_x_{factor}'
        if column.endswith(suffix) and factor in frame_columns:
            base = column[: -len(suffix)]
            if base in frame_columns:
                return base, factor
    return None


def resolve_missingness(
    frame: pd.DataFrame,
    target_col: str,
    warmup_rows: int,
    missing_suffix: str = MISSING_SUFFIX,
    required_indicators: Optional[Sequence[str]] = None,
    source_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Turn a feature frame built on a complete grid into supervised rows.

    Expects ``frame`` to span an unbroken time grid — features built before
    any row was deleted, so ``shift(k)`` is a true time offset. Performs, in
    order:

    1. **Impute features** over the whole grid, causally (see
       :func:`causal_impute`), recording which cells were invented.
    2. **Drop warm-up rows** — the leading rows whose features are
       structurally undefined for want of history, not missing from the
       data. See :func:`ml_forecast_lab.features.feature_warmup_rows`.
    3. **Drop label gaps** — rows whose target is NaN. These can never be
       supervised samples and their labels are never invented.
    4. **Emit indicators** — a ``<name>_missing`` companion column, 1 where
       the cell was imputed and 0 elsewhere, for each covariate that still
       has an imputed cell among the surviving rows, plus a single
       ``y_missing`` covering every target-derived feature. A covariate with
       no gaps gains no companion, so a complete experiment gains no columns
       at all and its matrix is unchanged.

    Parameters
    ----------
    frame : pd.DataFrame
        Features plus ``target_col`` plus covariates, on a complete grid.
    target_col : str
        The label column. Never imputed, never flagged.
    warmup_rows : int
        Leading rows to drop before anything else is judged.
    missing_suffix : str, default '_missing'
        Suffix for the companion indicator columns.
    required_indicators : sequence of str, optional
        Emit exactly these indicator columns, in this order, instead of the
        ones the data happens to call for — zero-filled where nothing was
        imputed. Inference passes the set the trained model was fitted on,
        because which columns have gaps is a property of the *window*: a
        history window that slides past an outage between a retrain and the
        next forecast would otherwise change the feature matrix's shape out
        from under a cached model. Anything measured but not required is
        still imputed, just not reported to the model, and is named in the
        report's ``dropped_indicators``.
    source_columns : sequence of str, optional
        Columns that came from the data rather than from
        ``build_features`` — the covariates. Named so that a covariate
        called e.g. ``foo_x_hour_sin`` is not mistaken for the interaction
        term of a covariate called ``foo`` and overwritten with the
        product of the two.

    Returns
    -------
    (pd.DataFrame, dict)
        The supervised frame, and a report with ``grid_rows``,
        ``warmup_rows``, ``label_gap_rows``, ``supervised_rows``,
        ``imputed_cells`` (per column, counted over surviving rows),
        ``indicator_cols``, ``dropped_indicators``, ``empty_cols``,
        ``label_gap_spans``, and — for the sequence-model path —
        ``window_frame`` and ``window_label_mask``. The window frame is
        the complete grid from the warm-up anchor onward, fully imputed
        (target included, causally), so windows built over it are true
        time spans; its label mask is True exactly where the label was
        measured. Its ``y_missing`` column, when emitted, is per-row —
        "the y at this row is invented" — where the supervised frame's is
        the aggregate over target-derived features, because each frame's
        consumer reads the target differently.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('frame must be a pandas DataFrame')
    if target_col not in frame.columns:
        raise ValueError(f'target_col {target_col!r} not in frame.columns')

    n_rows = len(frame)
    warmup_rows = int(max(0, min(int(warmup_rows), n_rows)))

    report: dict = {
        'grid_rows': n_rows,
        'warmup_rows': warmup_rows,
        'label_gap_rows': 0,
        'supervised_rows': 0,
        'imputed_cells': {},
        'indicator_cols': [],
        'dropped_indicators': [],
        'empty_cols': [],
        'label_gap_spans': [],
    }
    if n_rows == 0:
        empty = frame.copy()
        report['window_frame'] = empty
        report['window_label_mask'] = pd.Series(dtype=bool)
        return empty, report

    out = frame.copy()
    columns = list(out.columns)
    frame_columns = set(columns)
    source_columns = set(source_columns or ())

    # --- 1. Impute features causally, over the whole grid ------------------
    #
    # Over the whole grid, not over the surviving rows: a row that cannot be
    # a supervised sample (its label is missing) still holds perfectly good
    # *feature* observations, and dropping it first would hide them from the
    # expanding statistic. Strictly-prior is preserved either way.
    masks: dict = {}
    deferred_interactions: list = []
    for col in columns:
        if col == target_col:
            continue
        series = out[col]
        if not series.isna().any():
            continue
        interaction = _interaction_base(col, frame_columns, source_columns)
        if interaction is not None:
            # Handled after the base column is imputed.
            deferred_interactions.append((col, interaction))
            continue
        if series.notna().sum() == 0:
            report['empty_cols'].append(col)
        filled, mask = causal_impute(
            series, method='hold' if is_binary_column(series) else 'median',
        )
        out[col] = filled
        masks[col] = mask

    for col, (base, factor) in deferred_interactions:
        # Rebuild from the imputed base rather than imputing the product, so
        # the interaction stays exactly base x factor at every row — the same
        # identity build_features guarantees.
        out[col] = out[base] * out[factor]
        if out[col].isna().any():
            filled, mask = causal_impute(out[col])
            out[col] = filled
            masks[col] = mask

    # --- 2 & 3. Select supervised rows ------------------------------------
    #
    # Warm-up is counted from the first *measured* label, not from row 0.
    # A window that opens with a target outage has no history behind it
    # either, so the rows just after that outage are warm-up in exactly the
    # sense this drops for — and treating them as a gap instead would send
    # every target-derived feature down `causal_impute`'s leading branch,
    # which fills from the first observed value. For `y_lag_k` that value
    # IS a later label, so the first supervised row would be handed its own
    # answer as a feature.
    label_missing = out[target_col].isna().to_numpy()
    keep = np.ones(n_rows, dtype=bool)
    measured = np.flatnonzero(~label_missing)
    first_measured = int(measured[0]) if measured.size else n_rows
    if warmup_rows or first_measured:
        keep[: min(n_rows, first_measured + warmup_rows)] = False
    report['warmup_rows'] = int(min(n_rows, first_measured + warmup_rows))

    report['label_gap_rows'] = int((label_missing & keep).sum())
    report['label_gap_spans'] = _describe_gap_spans(
        out.index, label_missing & keep,
    )
    keep &= ~label_missing

    # The pre-selection frame, kept for the window frame below. `.loc[keep]`
    # allocates a new object, so this is a reference, not a copy.
    grid_imputed = out
    anchor = report['warmup_rows']

    # Explicit copy: with `grid_imputed` still holding the parent, a bare
    # .loc[keep] would be a pandas child view and every indicator write
    # below would raise SettingWithCopyWarning.
    out = out.loc[keep].copy()
    report['supervised_rows'] = len(out)

    # --- 4. Emit indicators for surviving gaps ----------------------------
    #
    # Decided after the row selection, not before: a column whose only NaNs
    # sat in the warm-up region has nothing left to flag, and a constant-zero
    # indicator is pure noise. This is also what keeps a gap-free experiment
    # bit-identical — no surviving imputed cell, no new column.
    n_keep = int(keep.sum())
    candidates: dict = {}
    grid_candidates: dict = {}
    target_flag = np.zeros(n_keep, dtype=bool)
    for col in columns:
        mask = masks.get(col)
        if mask is None:
            continue
        grid_mask = mask.to_numpy()
        surviving = grid_mask[keep]
        n_imputed = int(surviving.sum())
        if n_imputed == 0:
            continue
        report['imputed_cells'][col] = n_imputed
        if is_target_derived_feature(col, target_col):
            # Folded into the one aggregate flag — see TARGET_FEATURE_PREFIXES.
            target_flag |= surviving
            continue
        name = f'{col}{missing_suffix}'
        if name in frame_columns:
            # A real covariate already owns that name. Never silently
            # overwrite it — the collision is astronomically unlikely but
            # a clobbered covariate would be invisible.
            name = f'{col}{missing_suffix}_flag'
            logger.warning(
                "Indicator column for %r collided with an existing column; "
                "using %r instead.", col, name,
            )
        candidates[name] = surviving
        grid_candidates[name] = grid_mask

    # The window frame's target flag is per-row — "the y at THIS row is
    # invented" — because a sequence model consumes raw y directly, not
    # the lag columns the supervised frame's aggregate flag describes. A
    # label gap that survives into the window span therefore also forces
    # the indicator into existence, even when (as with a trailing outage)
    # no surviving row carries a flagged lag.
    if target_flag.any() or bool((label_missing[anchor:]).any()):
        candidates[TARGET_MISSING_COLUMN] = target_flag
        grid_candidates[TARGET_MISSING_COLUMN] = label_missing

    if required_indicators is None:
        emit = list(candidates)
    else:
        emit = list(required_indicators)
        report['dropped_indicators'] = [
            name for name in candidates if name not in set(emit)
        ]
        if report['dropped_indicators']:
            logger.warning(
                "Missingness indicators %s were measured but are not in the "
                "trained model's feature set; their gaps are imputed but "
                "unflagged for this cycle.", report['dropped_indicators'],
            )

    zeros = np.zeros(n_keep, dtype=np.float32)
    for name in emit:
        values = candidates.get(name)
        if values is None and name in frame_columns:
            # A pinned name that this frame did not measure a gap for AND
            # that is a real column of the input. Zero-filling it would
            # overwrite live measurements with a constant the model never
            # trained on — the exact train/serve skew the pin exists to
            # prevent. The column is already present and correct, so leave
            # it alone.
            logger.warning(
                "Indicator %r is also a data column in this frame; leaving "
                "its measured values in place rather than zero-filling it.",
                name,
            )
            report['indicator_cols'].append(name)
            continue
        out[name] = zeros if values is None else values.astype(np.float32)
        report['indicator_cols'].append(name)

    # --- 5. Window frame for sequence backends ----------------------------
    #
    # Sequence models window over consecutive rows of raw y, so handing
    # them the supervised frame silently time-warps every window that
    # spans a label gap: 48 rows of "24 hours" quietly become 24 hours
    # plus the outage. They get the complete grid instead — warm-up
    # trimmed, every feature imputed, and the target causally imputed the
    # same way the recursive lag buffer is seeded — with a per-row label
    # mask so the window builder can refuse to read an invented value as
    # a *label*. Window contents are features and may be imputed; the
    # horizon values a window is scored against are labels and never are.
    #
    # On a gap-free frame this is row-for-row identical to the supervised
    # frame, which is what keeps windows bit-identical there.
    win = grid_imputed.iloc[anchor:].copy()
    if label_missing.any():
        y_input, _ = causal_impute(frame[target_col])
        win[target_col] = y_input.iloc[anchor:]
    win_zeros = np.zeros(len(win), dtype=np.float32)
    for name in report['indicator_cols']:
        grid_vals = grid_candidates.get(name)
        if grid_vals is None:
            if name in frame_columns:
                # Same rule as above: a real data column keeps its values.
                continue
            win[name] = win_zeros
        else:
            win[name] = grid_vals[anchor:].astype(np.float32)
    report['window_frame'] = win
    report['window_label_mask'] = pd.Series(
        ~label_missing[anchor:], index=win.index,
    )

    return out, report


def _describe_gap_spans(index, mask, limit: int = 3) -> list:
    """Summarise a boolean mask as contiguous runs, longest first.

    Returns at most ``limit`` entries of ``(start, end, n_rows)``. Used to
    make the target-gap warning name the actual outage rather than just
    counting rows.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    spans = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 1))
    spans.sort(key=lambda s: s[1] - s[0], reverse=True)
    out = []
    for lo, hi in spans[:limit]:
        try:
            out.append((index[lo], index[hi], hi - lo + 1))
        except Exception:  # pragma: no cover - non-indexable index
            out.append((lo, hi, hi - lo + 1))
    return out
