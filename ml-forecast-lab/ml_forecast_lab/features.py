"""
Temporal feature engineering for ML Forecast Lab.

Provides unified feature generation across all model backends with support for
temporal features, lag features, rolling statistics, and holiday indicators.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Holiday detection using the holidays library (accurate movable dates)
try:
    import holidays as _holidays_lib
    _HOLIDAYS_AVAILABLE = True
except ImportError:
    _holidays_lib = None
    _HOLIDAYS_AVAILABLE = False
    logger.debug("holidays library not installed; holiday features disabled")

# Cache for holiday year lookups
_holiday_cache: dict = {}


def is_holiday(date: pd.Timestamp, country: Optional[str]) -> bool:
    """
    Check if date is a holiday.

    Parameters
    ----------
    date : pd.Timestamp
        Date to check.
    country : str, optional
        Country code ('GB', 'US', 'DE', etc.). If None, returns False.

    Returns
    -------
    bool
        Whether date is a holiday for the given country.

    Notes
    -----
    Uses the `holidays` library for accurate movable holiday dates.
    Falls back gracefully if the library is not installed.
    """
    if country is None or not _HOLIDAYS_AVAILABLE:
        return False

    cache_key = (country, date.year)
    if cache_key not in _holiday_cache:
        try:
            _holiday_cache[cache_key] = _holidays_lib.country_holidays(
                country, years=date.year
            )
        except Exception:
            return False

    return date.date() in _holiday_cache[cache_key]


# Suffix used for the companion indicator column that marks which cells of a
# feature column were imputed rather than observed. Emitted only for columns
# that actually have gaps, so a complete experiment gains no columns at all.
# See ``preprocessing.resolve_missingness``.
MISSING_SUFFIX = '_missing'


def default_lag_windows(interval_minutes: int) -> List[int]:
    """
    Rolling-statistic window sizes, in rows, for a given sampling interval.

    Pick rolling windows by their hour-of-history meaning rather than by row
    count. At 30-min interval this resolves to [6, 24, 72] — the legacy
    default — and at 5-min interval it scales to [36, 144, 432] so the
    longest window still spans 36 h. Without this the model never sees the
    daily seasonality on small intervals.
    """
    steps_per_hour = max(1, 60 // max(interval_minutes, 1))
    return [
        max(2, 3 * steps_per_hour),    # ~3 h
        max(3, 12 * steps_per_hour),   # ~12 h
        max(4, 36 * steps_per_hour),   # ~36 h
    ]


def feature_warmup_rows(
    n_rows: int,
    interval_minutes: int,
    n_lags: int = 12,
    lag_windows: Optional[List[int]] = None,
    ghi_gated: bool = False,
) -> int:
    """
    Leading rows of a :func:`build_features` frame that can never be complete.

    The longest-reaching feature decides it: ``y_lag_k`` is first defined at
    row ``k``, a rolling statistic over ``w`` rows of ``target.shift(1)`` is
    first defined at row ``w``, and ``y_diff_1`` at row 2. Every row before
    the maximum of those has a structurally undefined feature — not a gap in
    the data, just not enough history yet.

    That distinction is the whole point of this function. Once missing
    *features* are imputed rather than dropped (see
    ``preprocessing.resolve_missingness``), something still has to delete
    the warm-up rows, or a frame that used to start at row ``k`` would
    suddenly start at row 0 with ``k`` rows of invented lags — and the
    training-row count would change for experiments that have no gaps at
    all. Warm-up is dropped; everything after it is masked, flagged and
    imputed.

    ``n_rows`` is needed because :func:`build_features` only emits the
    periodic lags when they fit inside the frame.

    Parameters
    ----------
    ghi_gated : bool, default False
        Whether ``build_features`` will apply the clear-sky gate, i.e.
        whether ``clear_sky_ghi`` is a column of the frame. It changes the
        answer: ``_gate_by_past_ghi`` writes ``0.0`` wherever the shifted
        GHI is not positive, and a shifted GHI that reaches off the front
        of the series is NaN, so ``NaN > 0`` is False and every warm-up
        cell of every gated column becomes ``0.0`` rather than NaN. The
        gated columns — all the lags and ``y_diff_1`` — therefore impose no
        warm-up at all on a solar experiment, and only the ungated rolling
        statistics do. Counting them anyway silently deletes real
        supervised rows from every PV experiment.

    Returns
    -------
    int
        Row count, clamped to ``[0, n_rows]``.
    """
    if n_rows <= 0:
        return 0
    if lag_windows is None:
        lag_windows = default_lag_windows(interval_minutes)

    # Rolling statistics over target.shift(1): first valid row is the window.
    # Never gated, so they always count.
    reach = 0
    for window in lag_windows:
        reach = max(reach, int(window))

    if not ghi_gated:
        # y_lag_1..n_lags, and y_diff_1's target.shift(2).
        reach = max(reach, int(n_lags), 2)
        # Periodic lags — emitted only when they fit (mirrors build_features).
        steps_per_day = max(1, 1440 // max(interval_minutes, 1))
        for d in (1, 2):
            lag_steps = steps_per_day * d
            if lag_steps <= n_rows:
                reach = max(reach, lag_steps)

    return int(min(reach, n_rows))


# Feature columns that ``build_features`` derives from the calendar rather
# than from a covariate. Sequence backends receive these as their own
# channels via ``add_temporal=True``, so they must not be offered a second
# time as covariate channels.
ENGINEERED_TEMPORAL_COLUMNS = frozenset({
    'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_holiday',
})

# The single indicator that covers every target-derived feature — the lags,
# the rolling statistics and y_diff_1 — rather than one companion per
# column. See ``preprocessing.resolve_missingness`` for why it is aggregated.
TARGET_MISSING_COLUMN = 'y' + MISSING_SUFFIX


def neural_covariate_columns(
    columns,
    target_col: str = 'target',
) -> List[str]:
    """
    The columns a sequence backend should receive as extra input channels.

    Everything that is not a calendar feature, not a target-derived lag,
    and not the target itself. Sequence models window over the raw target,
    so the engineered lags say nothing they cannot already see. The
    missingness indicators DO stay — ``TARGET_MISSING_COLUMN`` included:
    windows are built over the window frame, whose target is causally
    imputed across label gaps, and its per-row ``y_missing`` is the
    channel that tells the model which of those y inputs were invented.
    A masked covariate rides along for the same reason.

    Kept in one place because the same filter is needed at five call sites
    (benchmark, holdout, production training, cached forecast, CV runner)
    and the cached channel order must agree between them, or a forecast is
    refused for a channel-name mismatch.
    """
    out: List[str] = []
    for c in columns:
        if c == target_col:
            continue
        if c in ENGINEERED_TEMPORAL_COLUMNS:
            continue
        if c.startswith('y_lag_'):
            continue
        out.append(c)
    return out


def build_features(
    df: pd.DataFrame,
    target_col: str,
    interval_minutes: int,
    n_lags: int = 12,
    lag_windows: Optional[List[int]] = None,
    country: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build temporal and lag features for forecasting.

    Generates comprehensive feature set including temporal cyclical features,
    lag features, rolling statistics, and holiday indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with DatetimeIndex and target_col.
    target_col : str
        Name of target column to use for lags and rolling stats.
    interval_minutes : int
        Sampling interval in minutes (used to determine time-based features).
    n_lags : int, default 12
        Number of lag features to create (y_lag_1, ..., y_lag_N).
    lag_windows : list of int, optional
        Window sizes (in lags) for rolling statistics.
        Default: [6, 24, 72] (e.g. at 30-min intervals: 3h, 12h, 36h).
    country : str, optional
        Country code for holiday features.

    Returns
    -------
    pd.DataFrame
        Feature matrix with temporal, lag, and statistical features.

    Notes
    -----
    All lag features are shifted by 1 to prevent look-ahead bias.
    Circular encoding (sin/cos) is applied to cyclical features.
    Rolling statistics use the target column.

    Features created:
    - Temporal: hour_of_day, day_of_week, is_weekend, month, day_of_month,
                hour_sin, hour_cos, dow_sin, dow_cos
    - Lag: y_lag_1, ..., y_lag_N
    - Periodic lag: y_lag_{steps_per_day}, y_lag_{steps_per_day*2} (same time yesterday/2d ago)
    - Rolling: y_rolling_mean_{window}, y_rolling_std_{window}, y_rolling_max_{window}
    - Rate of change: y_diff_1 (first difference of consecutive lags)
    - Interactions: {covariate}_x_hour_sin, {covariate}_x_hour_cos
    - Holiday: is_holiday (if country specified)

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range('2024-01-01', periods=1000, freq='30min')
    >>> df = pd.DataFrame({'y': np.random.randn(1000)}, index=idx)
    >>> features = build_features(df, 'y', interval_minutes=30, n_lags=12)
    >>> assert 'y_lag_1' in features.columns
    >>> assert 'hour_sin' in features.columns
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame')
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError('df must have DatetimeIndex')
    if target_col not in df.columns:
        raise ValueError(f'target_col {target_col!r} not in df.columns')
    if n_lags < 1:
        raise ValueError(f'n_lags must be >= 1, got {n_lags}')
    if interval_minutes < 1:
        raise ValueError(f'interval_minutes must be >= 1, got {interval_minutes}')

    if lag_windows is None:
        lag_windows = default_lag_windows(interval_minutes)

    features = pd.DataFrame(index=df.index)

    # Temporal features
    features['hour_of_day'] = df.index.hour
    features['day_of_week'] = df.index.dayofweek
    features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    features['month'] = df.index.month
    features['day_of_month'] = df.index.day

    # Circular encoding for hour (0-23)
    hour_rad = 2 * np.pi * features['hour_of_day'] / 24
    features['hour_sin'] = np.sin(hour_rad)
    features['hour_cos'] = np.cos(hour_rad)

    # Circular encoding for day of week (0-6)
    dow_rad = 2 * np.pi * features['day_of_week'] / 7
    features['dow_sin'] = np.sin(dow_rad)
    features['dow_cos'] = np.cos(dow_rad)

    # Lag features.
    #
    # Physics gate for solar-driven targets: when `clear_sky_ghi` is
    # present in the dataframe, the lag at row t of column y_lag_k is
    # target[t-k] — which at night is already 0 for a clean sensor and
    # so is unchanged by the gate. The gate matters at INFERENCE time,
    # where the recursive multi-step forecaster in _forecast_with_cached
    # feeds each step's prediction forward as lag_1 of the next step.
    # A sunset step biased ~150 W leaks into the next step's lag_1 even
    # though the corresponding past `clear_sky_ghi` is 0, producing a
    # feature vector (ghi=0, lag_1=150) that training never saw; the
    # tree then lands on a daytime leaf and predicts non-zero across
    # the whole night. By encoding the "night lag = 0" invariant here
    # and mirroring it in the inference path, both sides of the model
    # contract agree: gated lags are always 0 when the past was night,
    # regardless of whether the past value came from ground truth or a
    # recursive prediction.
    target = df[target_col]
    ghi_col = df['clear_sky_ghi'] if 'clear_sky_ghi' in df.columns else None

    def _gate_by_past_ghi(shifted: pd.Series, lag: int) -> pd.Series:
        if ghi_col is None:
            return shifted
        ghi_shifted = ghi_col.shift(lag)
        return shifted.where(ghi_shifted > 0, 0.0)

    for lag in range(1, n_lags + 1):
        features[f'y_lag_{lag}'] = _gate_by_past_ghi(target.shift(lag), lag)

    # Rolling statistics — shift by 1 before rolling so the feature at row t
    # spans target[t-w..t-1] (strictly past). Without the shift, pandas
    # rolling is right-closed and includes target[t], which is the value
    # being predicted: a hard look-ahead leak for tree backends and a
    # train/inference distribution mismatch for the recursive forecast
    # path that uses buf[-w:] (strictly past).
    shifted_target = target.shift(1)
    for window in lag_windows:
        features[f'y_rolling_mean_{window}'] = shifted_target.rolling(window=window).mean()
        features[f'y_rolling_std_{window}'] = shifted_target.rolling(window=window).std()
        features[f'y_rolling_max_{window}'] = shifted_target.rolling(window=window).max()

    # Periodic lags — "same time yesterday/2-days-ago"
    steps_per_day = max(1, 1440 // interval_minutes)  # e.g. 48 for 30-min
    for d in [1, 2]:
        lag_steps = steps_per_day * d
        if lag_steps <= len(target):
            features[f'y_lag_{lag_steps}'] = _gate_by_past_ghi(
                target.shift(lag_steps), lag_steps
            )

    # Rate of change — first difference of consecutive lags.
    #
    # Gating each shift independently used to create a synthetic discontinuity
    # at every dusk-to-night transition: at row t with ghi(t-1)=0 (just past
    # sunset) but ghi(t-2)>0 (last daytime sample), the gated lag_1 collapses
    # to 0 while gated lag_2 retains the full daytime value, producing
    # y_diff_1 ≈ -daytime instead of the physically expected small dip.
    # We gate the entire diff by the lag_1 mask so both terms vanish
    # together at night (correct) and the actual target.shift(1)-shift(2)
    # is preserved at every other timestep — including dusk, where the
    # true daytime-to-night slope is captured by the un-gated values.
    raw_diff = target.shift(1) - target.shift(2)
    if ghi_col is None:
        features['y_diff_1'] = raw_diff
    else:
        features['y_diff_1'] = raw_diff.where(ghi_col.shift(1) > 0, 0.0)

    # Interaction features — covariate × time-of-day.
    #
    # Every non-target column gets one, including a covariate whose name
    # happens to end in `_missing`. Missingness indicators are created
    # downstream by ``preprocessing.resolve_missingness``, after this runs,
    # so they never reach here — and excluding columns by name suffix would
    # silently strip a real covariate's interactions instead.
    cov_cols = [c for c in df.columns if c != target_col]
    for col in cov_cols:
        features[f'{col}_x_hour_sin'] = df[col] * features['hour_sin']
        features[f'{col}_x_hour_cos'] = df[col] * features['hour_cos']

    # Holiday indicator
    if country is not None:
        features['is_holiday'] = df.index.map(
            lambda d: int(is_holiday(d, country))
        )

    logger.debug(
        f'Built {len(features.columns)} features: {list(features.columns)}'
    )

    return features


def rebuild_fold_features(
    df_out: pd.DataFrame,
    target_col: str,
    interval_minutes: int,
    rolling_windows: List[int],
    steps_per_day: int,
) -> pd.DataFrame:
    """
    Recompute a CV fold's rolling statistics and periodic lags, on time.

    The benchmark and tuning harnesses rebuild these columns per fold rather
    than reusing the ones :func:`build_features` produced. ``shift`` and
    ``rolling`` are positional, so on a frame whose rows are the *surviving*
    supervised samples, "48 rows back" is 48 rows back — not 24 hours back —
    the moment the target has an outage anywhere inside the fold. Fixing lag
    construction upstream and leaving this alone would fix training and then
    score it with the broken features.

    So the recompute happens on the fold's own complete grid: reindex to
    ``interval_minutes``, causally impute the target for *feature*
    construction only, compute, and select back onto the rows the fold
    actually holds. The labels are untouched — nothing fabricated is ever
    scored — and on a fold with no gaps the reindex is a no-op, so the
    features are bit-identical to what this produced before.

    Falls back to the plain positional recompute when the frame has no
    DatetimeIndex (test fixtures pass a RangeIndex).

    Mutates and returns ``df_out``.
    """
    from ml_forecast_lab.preprocessing import causal_impute

    target = df_out[target_col]
    index = df_out.index
    grid = None
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        try:
            grid = pd.date_range(
                index[0], index[-1], freq=f'{max(1, int(interval_minutes))}min',
            )
        except Exception:
            grid = None

    if grid is not None and len(grid) > len(index):
        on_grid, _ = causal_impute(target.reindex(grid))
    else:
        on_grid = target
        grid = index

    shifted = on_grid.shift(1)
    for window in rolling_windows:
        df_out[f'y_rolling_mean_{window}'] = (
            shifted.rolling(window=window).mean().reindex(index)
        )
        df_out[f'y_rolling_std_{window}'] = (
            shifted.rolling(window=window).std().reindex(index)
        )
        df_out[f'y_rolling_max_{window}'] = (
            shifted.rolling(window=window).max().reindex(index)
        )
    for d in (1, 2):
        lag_steps = steps_per_day * d
        # Guard against len(target), not len(grid): matches what
        # build_features emitted, so the fold never invents a column the
        # trained feature list does not have.
        if lag_steps <= len(target):
            df_out[f'y_lag_{lag_steps}'] = (
                on_grid.shift(lag_steps).reindex(index)
            )
    df_out['y_diff_1'] = (
        (on_grid.shift(1) - on_grid.shift(2)).reindex(index)
    )
    return df_out


def compute_known_future_features(
    future_index: pd.DatetimeIndex,
    add_temporal: bool = True,
    country: Optional[str] = None,
    solar_lat_lon: Optional[Tuple[float, float]] = None,
    include_sun_elevation: bool = False,
    include_clear_sky_ghi: bool = False,
    future_covariate_values: Optional[Dict[str, pd.Series]] = None,
) -> pd.DataFrame:
    """
    Compute features that are deterministically known for future timestamps.

    A single linear projection from a past-window encoder onto multiple
    horizons cannot disambiguate "horizon h" from "absolute hour at h" —
    the same h-slot in training spans every absolute hour depending on the
    window's end time, so a fixed weight per h is forced into a phase-
    smeared compromise. The model can do better when each horizon position
    in its input carries its own time-anchored features. Tree models get
    this naturally (one row per horizon during recursive inference);
    neural sliding-window models do not, unless the window is extended
    with future positions populated by *this* function.

    Parameters
    ----------
    future_index : pd.DatetimeIndex
        Timestamps for which to compute features. May be tz-aware or
        tz-naive (naive is treated as UTC by the solar helper).
    add_temporal : bool, default True
        Include hour_sin, hour_cos, dow_sin, dow_cos, is_weekend.
    country : str, optional
        Country code for is_holiday. Omitted if None.
    solar_lat_lon : (lat, lon), optional
        Required for sun_elevation / clear_sky_ghi.
    include_sun_elevation : bool, default False
        Include 'sun_elevation' column (requires solar_lat_lon).
    include_clear_sky_ghi : bool, default False
        Include 'clear_sky_ghi' column (requires solar_lat_lon).
    future_covariate_values : dict[str, pd.Series], optional
        Forecast-style covariates whose future values are known (e.g.
        Solcast PV-forecast, weather forecasts). Series are reindexed to
        future_index and ffill/bfill-aligned.

    Returns
    -------
    pd.DataFrame
        Indexed by future_index. Columns are a subset of the channel
        names that ``create_sliding_windows`` / ``build_inference_window``
        produce, so that callers can populate future-position rows in an
        extended window by channel-name match.
    """
    out = pd.DataFrame(index=future_index)
    if len(future_index) == 0:
        return out

    if add_temporal:
        hour_rad = 2 * np.pi * future_index.hour / 24
        dow_rad = 2 * np.pi * future_index.dayofweek / 7
        out['hour_sin'] = np.sin(hour_rad).astype(np.float32)
        out['hour_cos'] = np.cos(hour_rad).astype(np.float32)
        out['dow_sin'] = np.sin(dow_rad).astype(np.float32)
        out['dow_cos'] = np.cos(dow_rad).astype(np.float32)
        out['is_weekend'] = (future_index.dayofweek >= 5).astype(np.float32)

    if country is not None and _HOLIDAYS_AVAILABLE:
        out['is_holiday'] = np.array(
            [int(is_holiday(d, country)) for d in future_index],
            dtype=np.float32,
        )

    if solar_lat_lon is not None and (include_sun_elevation or include_clear_sky_ghi):
        try:
            from ml_forecast_lab.solar_physics import compute_solar_features
            solar_df = compute_solar_features(
                future_index,
                latitude=solar_lat_lon[0],
                longitude=solar_lat_lon[1],
                include_elevation=include_sun_elevation,
                include_clear_sky=include_clear_sky_ghi,
            )
            for col in solar_df.columns:
                out[col] = solar_df[col].values.astype(np.float32)
        except Exception as e:
            logger.warning(
                f"Failed to compute future solar features for "
                f"{len(future_index)} timestamps: {e}"
            )

    if future_covariate_values:
        for name, series in future_covariate_values.items():
            if series is None or len(series) == 0:
                continue
            aligned = series.reindex(future_index).ffill().bfill()
            out[name] = aligned.values.astype(np.float32)

    return out


def prepare_train_test(
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    cv_strategy: str = 'walk_forward',
    n_folds: int = 5,
    embargo_periods: int = 2,
    test_size: Optional[float] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create train/test splits for cross-validation.

    Generates indices for either walk-forward or sliding-window cross-validation
    with temporal embargo to prevent look-ahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with DatetimeIndex.
    features_df : pd.DataFrame
        Features frame (must have same length and index as df).
    cv_strategy : str, default 'walk_forward'
        Strategy: 'walk_forward' (expanding window) or 'sliding_window' (fixed).
    n_folds : int, default 5
        Number of folds.
    embargo_periods : int, default 2
        Gap between training and test sets (in periods).
    test_size : float, optional
        Test set size as fraction of data. If None, uses (1 - 1/n_folds).

    Returns
    -------
    list of (train_idx, test_idx) tuples
        Indices for each CV fold.

    Notes
    -----
    Walk-forward CV:
        - Fold 1: train on [0:T1], test on [T1:T2]
        - Fold 2: train on [0:T2], test on [T2:T3]
        - ...
        Sizes expand over time, mimicking real-world retraining scenarios.

    Sliding-window CV:
        - Fold 1: train on [0:T1], test on [T1:T2]
        - Fold 2: train on [Δ:T1+Δ], test on [T1+Δ:T2+Δ]
        - ...
        Window size stays constant, sliding forward.

    Embargo:
        Removes embargo_periods between train and test to avoid temporal leakage.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range('2024-01-01', periods=1000, freq='30min')
    >>> df = pd.DataFrame({'y': range(1000)}, index=idx)
    >>> features_df = pd.DataFrame(index=idx)
    >>> splits = prepare_train_test(df, features_df, 'walk_forward', n_folds=5)
    >>> assert len(splits) == 5
    >>> for train_idx, test_idx in splits:
    ...     assert len(train_idx) > 0 and len(test_idx) > 0
    """
    if len(df) != len(features_df):
        raise ValueError('df and features_df must have same length')
    if n_folds < 2:
        raise ValueError(f'n_folds must be >= 2, got {n_folds}')
    if embargo_periods < 0:
        raise ValueError(f'embargo_periods must be >= 0, got {embargo_periods}')

    n = len(df)

    if test_size is None:
        test_size = 1 / (n_folds + 1)  # More conservative default

    test_points = max(1, int(n * test_size))

    # Validate that we have enough data for the requested configuration
    min_samples_needed = test_points * (n_folds + 1)
    if min_samples_needed > n:
        test_points = max(1, n // (n_folds + 2))
        logger.warning(
            f'Adjusted test_points to {test_points} for available data ({n} samples)'
        )

    splits = []

    if cv_strategy == 'walk_forward':
        # Expanding window: train set grows, test set fixed size
        for fold in range(n_folds):
            test_start = n - test_points * (n_folds - fold)
            test_end = test_start + test_points
            train_end = max(test_start - embargo_periods, 0)

            if test_end > n:
                test_end = n
            if train_end < 0:
                train_end = 0

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

    elif cv_strategy == 'sliding_window':
        # Fixed window: train and test window size constant, slides forward
        train_size = n - test_points * (n_folds + 1)
        train_size = max(train_size, int(n / (2 * n_folds)))

        for fold in range(n_folds):
            stride = (n - train_size - test_points) // max(n_folds - 1, 1)
            train_start = fold * stride
            train_end = train_start + train_size
            test_start = min(train_end + embargo_periods, n - test_points)
            test_end = test_start + test_points

            if test_end > n:
                test_end = n

            if train_end <= train_start or test_end <= test_start:
                continue

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            splits.append((train_idx, test_idx))

    else:
        raise ValueError(
            f"cv_strategy must be 'walk_forward' or 'sliding_window', got {cv_strategy!r}"
        )

    if not splits:
        raise ValueError(
            'Could not create any CV splits; check n_folds, test_size, embargo_periods'
        )

    logger.info(f'Created {len(splits)} {cv_strategy} CV splits')
    return splits


def reshape_for_sequence(
    X: np.ndarray,
    n_lags: int,
) -> np.ndarray:
    """
    Reshape flat features into 3D array for recurrent/convolutional models.

    Converts (n_samples, n_features) into (n_samples, n_lags, features_per_lag)
    suitable for LSTM, GRU, or 1D CNN.

    Parameters
    ----------
    X : np.ndarray
        2D feature array of shape (n_samples, n_features).
    n_lags : int
        Number of lagged timesteps per sample.

    Returns
    -------
    np.ndarray
        3D array of shape (n_samples, n_lags, features_per_lag).

    Notes
    -----
    Assumes the first n_lags columns are lag features (y_lag_1, ..., y_lag_n_lags).
    The remaining columns are repeated for each lag step (representing current-time features).

    Examples
    --------
    >>> X = np.random.randn(100, 15)  # 100 samples, 15 features
    >>> X_seq = reshape_for_sequence(X, n_lags=5)
    >>> assert X_seq.shape == (100, 5, 3)  # Each lag sees 3 features
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError('X must be a 2D numpy array')
    if n_lags < 1:
        raise ValueError(f'n_lags must be >= 1, got {n_lags}')

    n_samples, n_features = X.shape

    # Assume first n_lags columns are lag features
    lag_features = X[:, :n_lags]
    other_features = X[:, n_lags:]

    n_other = other_features.shape[1]
    features_per_lag = 1 + n_other

    # Reshape: (n_samples, n_lags, features_per_lag)
    X_seq = np.zeros((n_samples, n_lags, features_per_lag), dtype=X.dtype)

    # Fill with lag features (+ other features repeated)
    for t in range(n_lags):
        X_seq[:, t, 0] = lag_features[:, t]
        if n_other > 0:
            X_seq[:, t, 1:] = other_features

    logger.debug(
        f'Reshaped features from {X.shape} to {X_seq.shape} for sequence models'
    )

    return X_seq


def grid_step_index(index, positions=None) -> Optional[np.ndarray]:
    """Epoch-anchored grid-step number of selected rows of a regular grid.

    Returns ``index.asi8 // step_ns`` (int64) at ``positions`` — the number
    of whole grid steps between the Unix epoch and each row. Because the
    anchor is the epoch rather than the frame, two windows covering the
    same wall-clock rows get identical values no matter which frame slice
    they were built from: retention trimming, fold bridging, and cache
    reload cannot shift it. That is what makes it a safe absolute phase
    for cycle-aware backends (``cyclenet``): ``step mod cycle_len`` is the
    row's stable position within a daily/weekly cycle.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The window source frame's index (a complete regular grid).
    positions : array-like of int, optional
        Row positions of each window's FIRST timestep — ``kept`` from
        ``create_sliding_windows`` in label-mask mode, or
        ``np.arange(n_windows)`` otherwise. When omitted, every row's
        step number is returned.

    Returns
    -------
    np.ndarray of int64, or None when the index is not a DatetimeIndex
    with at least two rows (no grid step to infer) — callers skip the
    kwarg and phase-aware backends fall back to relative indexing.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return None
    step_ns = int((index[1] - index[0]).value)
    if step_ns <= 0:
        return None
    vals = index.asi8
    if positions is not None:
        vals = vals[np.asarray(positions, dtype=int)]
    return (vals // step_ns).astype(np.int64)


def create_sliding_windows(
    df: pd.DataFrame,
    target_col: str,
    window_size: int = 48,
    covariate_cols: Optional[List[str]] = None,
    add_temporal: bool = True,
    horizon_steps: Optional[List[int]] = None,
    future_features_df: Optional[pd.DataFrame] = None,
    label_mask=None,
    mask_horizons: Optional[List[int]] = None,
):
    """
    Create sliding window sequences from raw time series for LSTM/CNN.

    Creates (n_samples, window_size, n_channels) arrays where each sample
    is a window of raw values plus optional temporal features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex containing target and optional covariates.
    target_col : str
        Name of target column.
    window_size : int
        Number of timesteps per window (default 48 = 24h at 30-min intervals).
    covariate_cols : list of str, optional
        Additional columns to include as channels.
    add_temporal : bool, default True
        Add temporal features (hour_sin, hour_cos, dow_sin, dow_cos, is_weekend)
        as extra channels.
    horizon_steps : list of int, optional
        Steps-ahead for each forecast horizon. E.g. [4, 16, 24, 48] for
        2h/8h/12h/24h at 30-min intervals. When provided, y becomes 2D
        with shape (n_samples, len(horizon_steps)). horizon_steps=[1] is
        equivalent to the default single-step-ahead behaviour.
    future_features_df : pd.DataFrame, optional
        Indexed by ``df.index`` (or a superset thereof) and containing
        deterministically-known feature values for the future positions
        of each window. When provided, each output window is EXTENDED
        with ``max(horizon_steps)`` future positions appended in time
        order; columns of ``future_features_df`` whose names match
        channels in the window are populated at the future positions,
        all other channels at future positions are zero. The resulting
        per-sample shape becomes
        ``(window_size + max(horizon_steps), n_channels)``.

        This gives the model per-horizon, time-anchored signal directly
        in its input — closing the asymmetry where tree models get
        future temporal/solar features per recursive step but neural
        multi-head models historically only saw the past window and
        had to phase-disambiguate horizons from a single linear
        projection.

    label_mask : array-like of bool, optional
        Aligned with ``df``'s rows; True where the row's target value was
        actually measured. When provided, ``df`` is expected to be the
        *window frame* — a complete time grid whose target has been
        causally imputed so window INPUTS are unbroken time spans — and a
        window is kept only when every label row it is scored against is
        True. Window contents are features and may be imputed; the values
        a window is trained or scored against are labels and never are.
        The return value grows a fourth element: the kept sample indices
        into the unfiltered window enumeration, so callers can align
        per-window quantities (sample weights, label timestamps).
    mask_horizons : list of int, optional
        Which horizon steps' labels must be measured for a window to
        survive ``label_mask``. Defaults to every step in
        ``horizon_steps``: at training time all of them enter the loss.
        Evaluation windows are scored only at h=1, so requiring all 48
        would discard scoreable predictions — those callers pass ``[1]``.

    Returns
    -------
    X : np.ndarray
        Shape (n_samples, window_size, n_channels) — or, when
        ``future_features_df`` is provided,
        (n_samples, window_size + max(horizon_steps), n_channels).
    y : np.ndarray
        Shape (n_samples,) when horizon_steps is None, or
        (n_samples, n_horizons) when horizon_steps is provided.
    channel_names : list of str
        Names of the channels in X.
    kept : np.ndarray, only when ``label_mask`` is provided
        Original sample indices of the surviving windows.
    """
    # Build channel list
    channel_names = [target_col]
    work_df = df[[target_col]].copy()

    if covariate_cols:
        for c in covariate_cols:
            if c in df.columns:
                channel_names.append(c)
                work_df[c] = df[c]

    if add_temporal and isinstance(df.index, pd.DatetimeIndex):
        hour_rad = 2 * np.pi * df.index.hour / 24
        dow_rad = 2 * np.pi * df.index.dayofweek / 7
        work_df['hour_sin'] = np.sin(hour_rad)
        work_df['hour_cos'] = np.cos(hour_rad)
        work_df['dow_sin'] = np.sin(dow_rad)
        work_df['dow_cos'] = np.cos(dow_rad)
        work_df['is_weekend'] = (df.index.dayofweek >= 5).astype(np.float32)
        channel_names.extend(['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend'])

    data = work_df.values.astype(np.float32)
    n_total = len(data)
    n_channels = len(channel_names)

    if n_total <= window_size:
        raise ValueError(f"Need more than {window_size} samples, got {n_total}")

    if horizon_steps is not None:
        max_horizon = max(horizon_steps)
        n_samples = n_total - window_size - max_horizon + 1
        if n_samples <= 0:
            raise ValueError(
                f"Not enough data for window_size={window_size} and "
                f"max_horizon={max_horizon}: need > {window_size + max_horizon - 1} "
                f"samples, got {n_total}"
            )
        n_horizons = len(horizon_steps)
    else:
        n_samples = n_total - window_size
        n_horizons = 0

    # Window-extension mode: append max_horizon future positions whose
    # channels are populated from future_features_df by name match.
    # Channel order is preserved; channels not present in future_features_df
    # are left as zero in the future positions.
    if future_features_df is not None:
        if horizon_steps is None:
            raise ValueError(
                "future_features_df requires horizon_steps to be set so the "
                "number of appended future positions is well-defined"
            )
        extend_by = max_horizon
        # Pre-compute the future-channel slice aligned with df.index. This
        # lets us index by row position rather than re-aligning per sample.
        future_aligned = future_features_df.reindex(df.index)
        future_data = np.zeros((n_total, n_channels), dtype=np.float32)
        for ch_idx, ch_name in enumerate(channel_names):
            if ch_name in future_aligned.columns:
                vals = future_aligned[ch_name].values
                # NaN at rows outside future_features_df's original index
                # → leave as zero (no future-known signal there).
                mask = ~np.isnan(vals.astype(np.float64))
                future_data[mask, ch_idx] = vals[mask].astype(np.float32)
        effective_window = window_size + extend_by
    else:
        extend_by = 0
        future_data = None
        effective_window = window_size

    X = np.zeros((n_samples, effective_window, n_channels), dtype=np.float32)
    if n_horizons > 0:
        y = np.zeros((n_samples, n_horizons), dtype=np.float32)
    else:
        y = np.zeros(n_samples, dtype=np.float32)

    target_idx = 0  # target_col is always first
    for i in range(n_samples):
        X[i, :window_size] = data[i:i + window_size]
        if future_data is not None:
            X[i, window_size:] = future_data[i + window_size : i + window_size + extend_by]
        if n_horizons > 0:
            for h_idx, h in enumerate(horizon_steps):
                y[i, h_idx] = data[i + window_size + h - 1, target_idx]
        else:
            y[i] = data[i + window_size, target_idx]

    if label_mask is not None:
        lm = np.asarray(label_mask, dtype=bool)
        if len(lm) != n_total:
            raise ValueError(
                f'label_mask has {len(lm)} rows but df has {n_total}'
            )
        if horizon_steps is not None:
            steps = mask_horizons if mask_horizons is not None else horizon_steps
            label_pos = np.add.outer(
                np.arange(n_samples) + window_size,
                np.asarray(steps, dtype=int) - 1,
            )
            valid = lm[label_pos].all(axis=1)
        else:
            valid = lm[np.arange(n_samples) + window_size]
        kept = np.flatnonzero(valid)
        n_dropped = n_samples - len(kept)
        if n_dropped:
            logger.info(
                f'Dropped {n_dropped} of {n_samples} windows whose label '
                f'rows were not measured; {len(kept)} remain.'
            )
        X = X[valid]
        y = y[valid]
        n_samples = len(kept)

    horizon_info = f", horizons={horizon_steps}" if horizon_steps else ""
    extend_info = f" +{extend_by} future positions" if extend_by else ""
    logger.debug(
        f"Created {n_samples} sliding windows: "
        f"({effective_window} steps{extend_info} × {n_channels} channels: "
        f"{channel_names}{horizon_info})"
    )

    if label_mask is not None:
        return X, y, channel_names, kept
    return X, y, channel_names


def build_inference_window(
    df: pd.DataFrame,
    target_col: str,
    window_size: int,
    covariate_cols: Optional[List[str]] = None,
    add_temporal: bool = True,
    future_features_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a single inference window from the tail of a DataFrame.

    Counterpart to ``create_sliding_windows`` for the production forecast
    path. Returns a ``(1, window_size, n_channels)`` tensor whose window
    ends at ``df.index[-1]`` (i.e. the most recent timestamp).

    ``create_sliding_windows`` cannot be used for inference window
    construction because it reserves the row after the window for the
    ``h=1`` y-label, even when ``horizon_steps=[1]`` is requested only
    to make the call valid. The consequence is that its last possible
    window ends at ``df.index[-2]`` — a half-hour misalignment at 30-min
    sampling that publishes every prediction one interval later than the
    model intended, surfacing as visible time-of-day shifts in the
    forecast.

    Channel ordering matches ``create_sliding_windows`` exactly so the
    cached training ``channel_names`` are directly comparable by the
    parity guard in ``_forecast_with_cached``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex containing target and optional
        covariates. Must have at least ``window_size`` rows.
    target_col : str
        Name of target column. Becomes channel 0.
    window_size : int
        Number of timesteps in the window.
    covariate_cols : list of str, optional
        Additional columns to include as channels (channels 1..N+0,
        in iteration order — same as ``create_sliding_windows``).
    add_temporal : bool, default True
        Append temporal features (hour_sin, hour_cos, dow_sin, dow_cos,
        is_weekend) as the final channels.
    future_features_df : pd.DataFrame, optional
        Indexed by the future timestamps following ``df.index[-1]`` (i.e.
        the rows for which the model will predict). When provided, the
        returned tensor is EXTENDED with ``len(future_features_df)``
        future positions appended after the past window; columns of
        ``future_features_df`` whose names match channels in the window
        are populated at those future positions, all other channels at
        future positions are zero. The resulting shape is
        ``(1, window_size + n_horizons, n_channels)``.

        Must be channel-compatible with the matching ``create_sliding_windows``
        call that produced the cached training tensor — that is, the same
        ``future_features_df`` columns must be available at inference
        time as were available at training time, otherwise the future
        positions of the inference window will silently disagree with
        what the model learned to read.

    Returns
    -------
    X : np.ndarray
        Shape ``(1, window_size, n_channels)``, or
        ``(1, window_size + n_horizons, n_channels)`` when
        ``future_features_df`` is provided. dtype float32.
    channel_names : list of str
        Per-channel labels in the same order as the third dimension of X.
    """
    if len(df) < window_size:
        raise ValueError(
            f"Need at least {window_size} rows for an inference window, "
            f"got {len(df)}"
        )

    tail = df.iloc[-window_size:]

    channel_names = [target_col]
    work_df = tail[[target_col]].copy()

    if covariate_cols:
        for c in covariate_cols:
            if c in tail.columns:
                channel_names.append(c)
                work_df[c] = tail[c]

    if add_temporal and isinstance(tail.index, pd.DatetimeIndex):
        hour_rad = 2 * np.pi * tail.index.hour / 24
        dow_rad = 2 * np.pi * tail.index.dayofweek / 7
        work_df['hour_sin'] = np.sin(hour_rad)
        work_df['hour_cos'] = np.cos(hour_rad)
        work_df['dow_sin'] = np.sin(dow_rad)
        work_df['dow_cos'] = np.cos(dow_rad)
        work_df['is_weekend'] = (tail.index.dayofweek >= 5).astype(np.float32)
        channel_names.extend(
            ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']
        )

    n_channels = len(channel_names)
    past_data = work_df.values.astype(np.float32)

    if future_features_df is not None and len(future_features_df) > 0:
        n_horizons = len(future_features_df)
        future_block = np.zeros((n_horizons, n_channels), dtype=np.float32)
        for ch_idx, ch_name in enumerate(channel_names):
            if ch_name in future_features_df.columns:
                vals = future_features_df[ch_name].values
                # NaN-safe copy; leave NaN slots as zero (no known signal).
                mask = ~np.isnan(vals.astype(np.float64))
                future_block[mask, ch_idx] = vals[mask].astype(np.float32)
        combined_block = np.concatenate([past_data, future_block], axis=0)
        X = combined_block.reshape(1, window_size + n_horizons, n_channels)
    else:
        X = past_data.reshape(1, window_size, n_channels)
    return X, channel_names


