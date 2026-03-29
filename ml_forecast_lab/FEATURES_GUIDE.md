# Features Module Guide

## Overview

The `features` module provides unified temporal feature engineering shared across all model backends (LightGBM, XGBoost, LSTM, CNN). It generates lag features, rolling statistics, temporal cyclical encodings, and prepares cross-validation splits with proper temporal embargo to prevent look-ahead bias.

## Core Functions

### Feature Building

#### `build_features(df, target_col, interval_minutes, n_lags=12, lag_windows=None, country=None)`

Build comprehensive feature matrix from time series data.

**Parameters:**
- **df** (pd.DataFrame): Input data with DatetimeIndex.
- **target_col** (str): Name of target column (for lags and rolling stats).
- **interval_minutes** (int): Sampling interval in minutes.
- **n_lags** (int, default 12): Number of lag features to create.
- **lag_windows** (list[int], optional): Window sizes for rolling stats. Default: [6, 24, 72].
- **country** (str, optional): Country code for holiday features ('GB', 'US', 'DE').

**Returns:**
- pd.DataFrame: Feature matrix aligned with input.

**Features Created:**

| Category | Features |
|----------|----------|
| **Temporal** | hour_of_day, day_of_week, is_weekend, month, day_of_month |
| **Cyclical** | hour_sin, hour_cos, dow_sin, dow_cos (circular encoding) |
| **Lag** | y_lag_1, y_lag_2, ..., y_lag_N |
| **Rolling** | y_rolling_mean_{window}, y_rolling_std_{window}, y_rolling_max_{window} |
| **Holiday** | is_holiday (if country specified) |

**Example - Solar forecasting:**

```python
import pandas as pd
import numpy as np
from ml_forecast_lab.features import build_features

# Create time series data
idx = pd.date_range('2024-01-01', periods=1000, freq='30min')
df = pd.DataFrame({
    'generation_w': np.sin(np.arange(1000) * 2*np.pi/48) + np.random.randn(1000)*0.1,
    'cloud_cover': np.random.rand(1000) * 100,
}, index=idx)

# Build features
features = build_features(
    df,
    target_col='generation_w',
    interval_minutes=30,
    n_lags=12,  # ~6 hours of history
    lag_windows=[6, 24, 72],  # 3h, 12h, 36h rolling windows
    country='GB'  # Include UK holidays
)

print(features.shape)  # (1000, 31) - temporal + lags + rolling + holiday
print(features.columns)
# Index(['hour_of_day', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
#        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
#        'y_lag_1', 'y_lag_2', ..., 'y_lag_12',
#        'y_rolling_mean_6', 'y_rolling_std_6', 'y_rolling_max_6',
#        'y_rolling_mean_24', 'y_rolling_std_24', 'y_rolling_max_24',
#        'y_rolling_mean_72', 'y_rolling_std_72', 'y_rolling_max_72',
#        'is_holiday'], ...)
```

**Key Details:**

1. **Cyclical Encoding**: Hour and day-of-week are encoded as sine/cosine to preserve circularity (23:00 is close to 00:00, Sunday is close to Monday).
2. **Lag Prevention**: All lags are shifted by 1 period to prevent information leakage into the training set.
3. **Rolling Stats**: Computed over configurable windows (default 6, 24, 72 lags = 3h, 12h, 36h at 30-min intervals).
4. **Holidays**: Simple dictionary-based lookup (no heavy dependency).

---

### Cross-Validation Splitting

#### `prepare_train_test(df, features_df, cv_strategy='walk_forward', n_folds=5, embargo_periods=2, test_size=None)`

Create train/test indices for temporal cross-validation with embargo.

**Parameters:**
- **df** (pd.DataFrame): Data frame with DatetimeIndex.
- **features_df** (pd.DataFrame): Features frame (must align with df).
- **cv_strategy** (str): 'walk_forward' or 'sliding_window'.
- **n_folds** (int): Number of folds.
- **embargo_periods** (int): Gap between train and test (in periods) to prevent leakage.
- **test_size** (float, optional): Test set size as fraction. Default: 1/(n_folds+1).

**Returns:**
- list of (train_idx, test_idx): Indices for each fold.

**Strategies:**

**Walk-Forward CV** (Expanding window):
```
Fold 1: [===TRAIN===|embargo|===TEST===]
Fold 2: [====TRAIN====|embargo|===TEST===]
Fold 3: [=====TRAIN=====|embargo|===TEST===]
```
- Training set expands over time
- Mimics real-world retraining scenarios
- More data in later folds (realistic for production)
- Recommended for time series forecasting

**Sliding-Window CV** (Fixed window):
```
Fold 1: [===TRAIN===|embargo|===TEST===]
Fold 2:    [===TRAIN===|embargo|===TEST===]
Fold 3:       [===TRAIN===|embargo|===TEST===]
```
- Window size stays constant
- Less training data per fold
- Better for stationary data or benchmarking
- Allows more folds on fixed data size

**Embargo:**
- Removes periods between train and test to prevent temporal leakage
- Crucial for time series (future information leaks backward)
- Default 2 periods handles feedback loops and dependencies

**Example:**

```python
from ml_forecast_lab.features import prepare_train_test

# Create data
idx = pd.date_range('2024-01-01', periods=500, freq='30min')
df = pd.DataFrame({'y': range(500)}, index=idx)
features_df = pd.DataFrame(index=idx)

# Walk-forward CV
splits_wf = prepare_train_test(
    df,
    features_df,
    cv_strategy='walk_forward',
    n_folds=5,
    embargo_periods=2,
    test_size=0.2
)

# Sliding-window CV
splits_sw = prepare_train_test(
    df,
    features_df,
    cv_strategy='sliding_window',
    n_folds=3,
    embargo_periods=2
)

# Use splits
for fold_idx, (train_idx, test_idx) in enumerate(splits_wf):
    X_train, y_train = features_df.iloc[train_idx], df['y'].iloc[train_idx]
    X_test, y_test = features_df.iloc[test_idx], df['y'].iloc[test_idx]

    print(f'Fold {fold_idx+1}: train={len(train_idx)}, test={len(test_idx)}')
    # Model training and evaluation...
```

**Output for walk_forward (5 folds, 500 samples):**
```
Fold 1: train=80, test=82
Fold 2: train=162, test=82
Fold 3: train=244, test=82
Fold 4: train=326, test=82
Fold 5: train=408, test=82
```

---

### Sequence Reshaping

#### `reshape_for_sequence(X, n_lags)`

Reshape flat features into 3D array for RNNs (LSTM) and 1D CNNs.

**Parameters:**
- **X** (np.ndarray): 2D feature array (n_samples, n_features).
- **n_lags** (int): Number of lag features per sample.

**Returns:**
- np.ndarray: 3D array (n_samples, n_lags, features_per_lag).

**Purpose:**
- RNNs and CNNs require sequential input: (batch, timesteps, features)
- Lag features become the timestep dimension
- Other features are repeated at each timestep

**Example:**

```python
import numpy as np
from ml_forecast_lab.features import reshape_for_sequence

# Feature matrix: 100 samples, 15 features
# (y_lag_1, y_lag_2, ..., y_lag_12, hour_of_day, day_of_week, is_weekend)
X = np.random.randn(100, 15)

# Reshape for LSTM: 12 timesteps, 4 features per timestep
# y_lag_t + (hour_of_day, day_of_week, is_weekend)
X_seq = reshape_for_sequence(X, n_lags=12)

print(X_seq.shape)  # (100, 12, 4)

# For CNN: same shape, but will apply 1D convolutions across timesteps
# For LSTM: feed directly to LSTM layer
```

**Usage with models:**

```python
# LightGBM / XGBoost - use flat X
model_gb = LGBMRegressor()
model_gb.fit(X, y)

# LSTM - use reshaped X_seq
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model_lstm = Sequential([
    LSTM(64, input_shape=(12, 4)),
    Dense(32, activation='relu'),
    Dense(1)
])
model_lstm.fit(X_seq, y)

# 1D CNN - use reshaped X_seq
model_cnn = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(12, 4)),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'),
    Dense(1)
])
model_cnn.fit(X_seq, y)
```

---

### Forecast Features

#### `create_forecast_features(last_timestamp, interval_minutes, horizons_minutes, n_lags, lag_values, country=None)`

Create feature matrix for forecasting at future horizons.

**Parameters:**
- **last_timestamp** (pd.Timestamp): Most recent timestamp in training data.
- **interval_minutes** (int): Sampling interval in minutes.
- **horizons_minutes** (list[int]): Prediction lookaheads (e.g. [120, 480, 1440]).
- **n_lags** (int): Number of lag features.
- **lag_values** (np.ndarray): Most recent lag values (shape: (n_lags,)).
- **country** (str, optional): Country code for holiday features.

**Returns:**
- pd.DataFrame: Feature matrix with one row per horizon.

**Purpose:**
- Generate features for out-of-sample prediction
- Lag features are filled with most recent values
- Temporal features computed for future timestamps
- Rolling statistics set to NaN (model should handle via imputation)

**Example:**

```python
import pandas as pd
import numpy as np
from ml_forecast_lab.features import create_forecast_features

# Last timestamp in training data
last_ts = pd.Timestamp('2024-01-15 12:00:00')

# Most recent lag values (12 lags of target)
last_lags = np.array([1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

# Create features for 3 forecast horizons
forecast_features = create_forecast_features(
    last_timestamp=last_ts,
    interval_minutes=30,
    horizons_minutes=[120, 480, 1440],  # 2h, 8h, 24h ahead
    n_lags=12,
    lag_values=last_lags,
    country='GB'
)

print(forecast_features)
#                                  hour_of_day  day_of_week  is_weekend  ...
# 2024-01-15 14:00:00                      14            0           0  ...
# 2024-01-15 20:00:00                      20            0           0  ...
# 2024-01-16 12:00:00                      12            1           0  ...

# Make predictions
predictions = model.predict(forecast_features)
# Result: 3 forecasts (one per horizon)
```

**Key Details:**
- Lag features are constant across horizons (most recent values available at forecast time)
- Temporal features differ (hour, day reflect the future timestamp)
- Rolling statistics are NaN (model should impute or learn to ignore)
- Holiday features reflect future dates

---

### Holiday Detection

#### `is_holiday(date, country)`

Check if date is a holiday for given country.

**Parameters:**
- **date** (pd.Timestamp): Date to check.
- **country** (str, optional): Country code ('GB', 'US', 'DE'). If None, returns False.

**Returns:**
- bool: Whether date is a holiday.

**Supported Countries:**
- **'GB'**: UK bank holidays (New Year, Easter Monday, May Day, etc.)
- **'US'**: US federal holidays (New Year, Independence Day, Thanksgiving, Christmas)
- **'DE'**: German holidays (New Year, Christmas, Boxing Day)

**Example:**

```python
from ml_forecast_lab.features import is_holiday
import pandas as pd

christmas_2024 = pd.Timestamp('2024-12-25')
boxing_2024 = pd.Timestamp('2024-12-26')
random_day = pd.Timestamp('2024-06-15')

print(is_holiday(christmas_2024, 'GB'))  # True
print(is_holiday(boxing_2024, 'GB'))     # True
print(is_holiday(random_day, 'GB'))      # False
```

**Limitations:**
- Uses simple (month, day) matching
- Does NOT handle movable holidays (Easter varies each year)
- For production use, consider the `holidays` package

---

## Usage Patterns

### Pattern 1: Complete ML pipeline

```python
import pandas as pd
import numpy as np
from ml_forecast_lab.preprocessing import cumulative_to_interval, resample_to_grid
from ml_forecast_lab.features import (
    build_features,
    prepare_train_test,
    reshape_for_sequence,
    create_forecast_features
)
from sklearn.preprocessing import StandardScaler

# 1. Get and clean data
sensor_history = fetch_sensor_history('sensor.solar_generation_w', days=30)
clean = resample_to_grid(sensor_history, freq='30min', method='mean')

# 2. Build features
X = build_features(
    pd.DataFrame({'target': clean}, index=clean.index),
    'target',
    interval_minutes=30,
    n_lags=12,
    country='GB'
)

# 3. Create CV splits
splits = prepare_train_test(
    pd.DataFrame({'y': clean}, index=clean.index),
    X,
    cv_strategy='walk_forward',
    n_folds=5,
    embargo_periods=2
)

# 4. Train models
for fold, (train_idx, test_idx) in enumerate(splits):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = clean.iloc[train_idx], clean.iloc[test_idx]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LGBMRegressor()
    model.fit(X_train_scaled, y_train)

    # Evaluate
    pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, pred)
    print(f'Fold {fold+1} MAE: {mae:.2f}')

# 5. Forecast
last_ts = clean.index[-1]
last_lags = clean.tail(12).values[::-1]  # Last 12 values in lag order
forecast_X = create_forecast_features(
    last_ts,
    interval_minutes=30,
    horizons_minutes=[120, 480, 1440],
    n_lags=12,
    lag_values=last_lags,
    country='GB'
)
forecast_X_scaled = scaler.transform(forecast_X)
forecast = model.predict(forecast_X_scaled)
print(f'Forecast: {forecast}')  # [value_at_2h, value_at_8h, value_at_24h]
```

### Pattern 2: Neural network with sequence features

```python
from ml_forecast_lab.features import reshape_for_sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Build features (as above)
X = build_features(...)
splits = prepare_train_test(...)

# Train LSTM on reshaped features
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(12, 4)),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(loss='mae', optimizer='adam')

for fold, (train_idx, test_idx) in enumerate(splits):
    X_train_flat = X.iloc[train_idx]
    X_test_flat = X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Reshape for LSTM
    X_train_seq = reshape_for_sequence(X_train_flat.values, n_lags=12)
    X_test_seq = reshape_for_sequence(X_test_flat.values, n_lags=12)

    # Train
    model.fit(X_train_seq, y_train, epochs=20, batch_size=32)
    pred = model.predict(X_test_seq)
    mae = mean_absolute_error(y_test, pred)
    print(f'Fold {fold+1} MAE: {mae:.2f}')
```

---

## Best Practices

1. **Always use embargo**: Set `cv_embargo_periods >= 2` to prevent temporal leakage.
2. **Choose CV strategy carefully**:
   - Use `'walk_forward'` for realistic evaluation
   - Use `'sliding_window'` for stationary data
3. **Scale features**: Apply StandardScaler or similar AFTER creating splits (fit on train only).
4. **Handle NaN from rolling stats**: Use forward-fill or group-specific statistics.
5. **Validate lag order**: Ensure y_lag_1 is most recent (one period back).
6. **Test holidays parameter**: Verify that `is_holiday()` returns expected results for your country.

---

## Performance Notes

- `build_features`: O(n) for lags + rolling windows
- `prepare_train_test`: O(n) split creation
- `reshape_for_sequence`: O(n_samples * n_features) for reshaping
- `create_forecast_features`: O(n_horizons) (typically very small)
- All functions use pandas/numpy (no sklearn dependencies)
