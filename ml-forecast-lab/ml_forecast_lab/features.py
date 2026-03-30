"""
Temporal feature engineering for ML Forecast Lab.

Provides unified feature generation across all model backends with support for
temporal features, lag features, rolling statistics, and holiday indicators.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Simple holiday definitions (no heavy dependencies)
HOLIDAYS = {
    'GB': {  # United Kingdom bank holidays (approximate)
        (1, 1): 'New Year',
        (4, 9): 'Easter Monday',
        (5, 5): 'Early May Bank Holiday',
        (5, 26): 'Spring Bank Holiday',
        (8, 25): 'Summer Bank Holiday',
        (12, 25): 'Christmas Day',
        (12, 26): 'Boxing Day',
    },
    'US': {  # United States federal holidays
        (1, 1): 'New Year',
        (7, 4): 'Independence Day',
        (11, 24): 'Thanksgiving',
        (12, 25): 'Christmas',
    },
    'DE': {  # Germany
        (1, 1): 'New Year',
        (12, 25): 'Christmas Day',
        (12, 26): 'Boxing Day',
    },
}


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
    Uses simple (month, day) matching. Does not account for movable holidays
    like Easter (fixed to approximate date). For production use, consider
    a dedicated library like `holidays`.
    """
    if country is None or country not in HOLIDAYS:
        return False

    month_day = (date.month, date.day)
    return month_day in HOLIDAYS[country]


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
    - Rolling: y_rolling_mean_{window}, y_rolling_std_{window}, y_rolling_max_{window}
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
        lag_windows = [6, 24, 72]

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

    # Lag features
    target = df[target_col]
    for lag in range(1, n_lags + 1):
        features[f'y_lag_{lag}'] = target.shift(lag)

    # Rolling statistics
    for window in lag_windows:
        features[f'y_rolling_mean_{window}'] = target.rolling(window=window).mean()
        features[f'y_rolling_std_{window}'] = target.rolling(window=window).std()
        features[f'y_rolling_max_{window}'] = target.rolling(window=window).max()

    # Holiday indicator
    if country is not None:
        features['is_holiday'] = df.index.map(
            lambda d: int(is_holiday(d, country))
        )

    logger.debug(
        f'Built {len(features.columns)} features: {list(features.columns)}'
    )

    return features


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


def create_sliding_windows(
    df: pd.DataFrame,
    target_col: str,
    window_size: int = 48,
    covariate_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences from raw time series for LSTM/CNN.

    Instead of pre-computed features, this creates (n_samples, window_size, n_channels)
    arrays where each sample is a window of raw values.

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

    Returns
    -------
    X : np.ndarray
        Shape (n_samples, window_size, n_channels) where n_channels = 1 + len(covariate_cols).
    y : np.ndarray
        Shape (n_samples,) — the target value at the step after each window.
    """
    cols = [target_col]
    if covariate_cols:
        cols += [c for c in covariate_cols if c in df.columns]

    data = df[cols].values.astype(np.float32)
    n_total = len(data)
    n_channels = len(cols)

    if n_total <= window_size:
        raise ValueError(f"Need more than {window_size} samples, got {n_total}")

    n_samples = n_total - window_size
    X = np.zeros((n_samples, window_size, n_channels), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    target_idx = 0  # target_col is always first
    for i in range(n_samples):
        X[i] = data[i:i + window_size]
        y[i] = data[i + window_size, target_idx]

    logger.debug(
        f"Created {n_samples} sliding windows: "
        f"({window_size} steps × {n_channels} channels)"
    )

    return X, y


def create_forecast_features(
    last_timestamp: pd.Timestamp,
    interval_minutes: int,
    horizons_minutes: List[int],
    n_lags: int,
    lag_values: np.ndarray,
    country: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create features for forecasting at future horizons.

    Generates a feature matrix for multiple lookahead steps (e.g. predictions
    for t+2h, t+8h, t+12h) based on the most recent lag values and future timestamps.

    Parameters
    ----------
    last_timestamp : pd.Timestamp
        Most recent timestamp in the data.
    interval_minutes : int
        Sampling interval in minutes.
    horizons_minutes : list of int
        Prediction horizons (e.g. [120, 480, 720] for 2h, 8h, 12h).
    n_lags : int
        Number of lag features expected.
    lag_values : np.ndarray
        Most recent lag values of shape (n_lags,).
    country : str, optional
        Country code for holiday features.

    Returns
    -------
    pd.DataFrame
        Feature matrix for forecasting with one row per horizon.

    Notes
    -----
    Creates temporal features for each future timestamp and fills lag features
    with the most recent available values. Rolling statistics are set to NaN
    (model should handle gracefully via imputation or separate output head).
    """
    if len(lag_values) != n_lags:
        raise ValueError(
            f'lag_values length ({len(lag_values)}) must match n_lags ({n_lags})'
        )

    forecast_timestamps = [
        last_timestamp + pd.Timedelta(minutes=h) for h in horizons_minutes
    ]

    forecast_df = pd.DataFrame(index=forecast_timestamps)

    # Temporal features
    forecast_df['hour_of_day'] = forecast_df.index.hour
    forecast_df['day_of_week'] = forecast_df.index.dayofweek
    forecast_df['is_weekend'] = (forecast_df.index.dayofweek >= 5).astype(int)
    forecast_df['month'] = forecast_df.index.month
    forecast_df['day_of_month'] = forecast_df.index.day

    # Circular encodings
    hour_rad = 2 * np.pi * forecast_df['hour_of_day'] / 24
    forecast_df['hour_sin'] = np.sin(hour_rad)
    forecast_df['hour_cos'] = np.cos(hour_rad)

    dow_rad = 2 * np.pi * forecast_df['day_of_week'] / 7
    forecast_df['dow_sin'] = np.sin(dow_rad)
    forecast_df['dow_cos'] = np.cos(dow_rad)

    # Lag features (use most recent values)
    for i, lag_val in enumerate(lag_values, start=1):
        forecast_df[f'y_lag_{i}'] = lag_val

    # Rolling statistics (set to NaN; model should impute)
    rolling_windows = [6, 24, 72]
    for window in rolling_windows:
        forecast_df[f'y_rolling_mean_{window}'] = np.nan
        forecast_df[f'y_rolling_std_{window}'] = np.nan
        forecast_df[f'y_rolling_max_{window}'] = np.nan

    # Holiday indicator
    if country is not None:
        forecast_df['is_holiday'] = forecast_df.index.map(
            lambda d: int(is_holiday(d, country))
        )

    return forecast_df
