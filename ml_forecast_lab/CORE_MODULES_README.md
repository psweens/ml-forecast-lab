# ML Forecast Lab - Core Modules

This document provides an overview of the three core modules that form the foundation of ML Forecast Lab: configuration, preprocessing, and feature engineering.

## Module Overview

### 1. **config.py** - Experiment Configuration

Dataclasses and YAML loader for declarative experiment setup.

**Key Classes:**
- `CovariateCfg`: Single external feature specification
- `ExperimentCfg`: Complete forecasting experiment for one sensor
- `AppConfig`: Application-level settings
- `load_config(path)`: Load configuration from YAML file

**Key Features:**
- Flexible covariate roles (future, lagged, both)
- Cumulative sensor handling with daily reset support
- Multiple cross-validation strategies (walk-forward, sliding-window)
- Multi-horizon forecasting (2h, 8h, 24h, etc.)
- Custom metric support
- Production model auto-selection
- Home Assistant entity publishing

**Example:**

```python
from ml_forecast_lab.config import load_config
cfg = load_config('config.yaml')

for exp in cfg.experiments:
    print(f'{exp.name}: {exp.target_entity}')
    print(f'  CV: {exp.cv_strategy} ({exp.cv_folds} folds)')
    print(f'  Models: {exp.models_enabled}')
```

**See:** `CONFIG_GUIDE.md` for detailed documentation and examples.

---

### 2. **preprocessing.py** - Data Cleaning and Transformation

Unified pipeline for time series preprocessing consolidating four separate cumulative-to-interval paths into one.

**Core Functions:**
- `cumulative_to_interval()`: Convert cumulative sensors (handles daily resets, gaps, spikes)
- `resample_to_grid()`: Resample to regular frequency
- `clip_outliers()`: Quantile-based outlier clipping
- `apply_transform()` / `invert_transform()`: Log, sqrt, Box-Cox
- `subtract_series()`: Vectorised subtraction with alignment
- `power_to_energy()`: W/kW to Wh/kWh conversion
- `align_series()`: Multi-series alignment

**Key Features:**
- **Single unified function** for all cumulative patterns:
  - Daily-reset sensors (energy today)
  - Non-daily cumulative (meters)
  - Gap-aware interpolation
  - Negative difference handling
  - Spike capping based on max_increment
- Gap filling and resampling with configurable methods
- Transformation with stored metadata for inversion
- Logging and error handling throughout

**Example:**

```python
from ml_forecast_lab.preprocessing import (
    cumulative_to_interval,
    resample_to_grid,
    clip_outliers
)

# Convert daily-reset sensor
intervals = cumulative_to_interval(
    cumul_series,
    interval_minutes=30,
    reset_daily=True,
    max_increment=10000.0
)

# Resample to regular grid
regular = resample_to_grid(intervals, freq='30min', method='mean')

# Remove outliers
clean = clip_outliers(regular, quantile=0.99, positive_only=True)
```

**See:** `PREPROCESSING_GUIDE.md` for detailed documentation and patterns.

---

### 3. **features.py** - Feature Engineering

Temporal features, lag generation, and CV split creation shared across all model backends.

**Core Functions:**
- `build_features()`: Generate temporal + lag + rolling + holiday features
- `prepare_train_test()`: Create CV splits with temporal embargo
- `reshape_for_sequence()`: Reshape for RNN/CNN models
- `create_forecast_features()`: Generate features for out-of-sample prediction
- `is_holiday()`: Simple holiday detection

**Key Features:**
- **Temporal features**: hour_of_day, day_of_week, is_weekend, month, day_of_month
- **Cyclical encoding**: hour_sin/cos, dow_sin/cos (circular preservation)
- **Lag features**: Configurable number of lags with proper shift
- **Rolling statistics**: mean, std, max over configurable windows
- **Holiday indicators**: Built-in lookup for GB, US, DE
- **Two CV strategies**:
  - Walk-forward: Expanding window (production-realistic)
  - Sliding-window: Fixed window (stationary data)
- **Temporal embargo**: Configurable gap to prevent leakage
- **Model-agnostic**: Works with GB/XGB (flat), LSTM/CNN (sequence)

**Example:**

```python
from ml_forecast_lab.features import (
    build_features,
    prepare_train_test,
    reshape_for_sequence
)

# Build features
features = build_features(
    df,
    target_col='y',
    interval_minutes=30,
    n_lags=12,
    lag_windows=[6, 24, 72],
    country='GB'
)

# Create CV splits
splits = prepare_train_test(
    df,
    features,
    cv_strategy='walk_forward',
    n_folds=5,
    embargo_periods=2
)

# For neural networks, reshape features
X_seq = reshape_for_sequence(X_train.values, n_lags=12)
model_lstm.fit(X_seq, y_train)
```

**See:** `FEATURES_GUIDE.md` for detailed documentation and patterns.

---

## Quick Start

### 1. Configure Experiment

Create `config.yaml`:

```yaml
update_every_minutes: 5
timezone: Europe/London
experiments:
  - name: solar_forecast
    target_entity: sensor.solar_generation_w
    source_is_cumulative: true
    reset_daily: true
    days_history: 30
    interval_minutes: 30
    horizons_minutes: [120, 480, 1440]
    units: W
    country: GB
    models_enabled: [lightgbm, xgboost, lstm]
    cv_strategy: walk_forward
    cv_folds: 5
    metrics: [mae, rmse, mape]
    covariates:
      - entity: sensor.cloud_cover_percent
        role: future
```

### 2. Load Configuration

```python
from ml_forecast_lab.config import load_config
cfg = load_config('config.yaml')
exp = cfg.experiments[0]
```

### 3. Fetch and Clean Data

```python
from ml_forecast_lab.preprocessing import (
    cumulative_to_interval,
    resample_to_grid,
    clip_outliers
)

# Get data from Home Assistant
data = fetch_ha_history(exp.target_entity, days=exp.days_history)

# Convert cumulative to intervals
if exp.source_is_cumulative:
    intervals = cumulative_to_interval(
        data,
        interval_minutes=exp.interval_minutes,
        reset_daily=exp.reset_daily,
        max_increment=exp.max_increment
    )
else:
    intervals = data

# Resample to grid
regular = resample_to_grid(intervals, freq=f'{exp.interval_minutes}min')

# Clean outliers
clean = clip_outliers(regular, quantile=0.99)
```

### 4. Build Features

```python
from ml_forecast_lab.features import build_features

df = pd.DataFrame({'target': clean}, index=clean.index)
features = build_features(
    df,
    target_col='target',
    interval_minutes=exp.interval_minutes,
    n_lags=12,
    lag_windows=[6, 24, 72],
    country=exp.country
)
```

### 5. Create CV Splits

```python
from ml_forecast_lab.features import prepare_train_test

splits = prepare_train_test(
    df,
    features,
    cv_strategy=exp.cv_strategy,
    n_folds=exp.cv_folds,
    embargo_periods=exp.cv_embargo_periods
)

for fold, (train_idx, test_idx) in enumerate(splits):
    X_train = features.iloc[train_idx]
    y_train = df['target'].iloc[train_idx]
    X_test = features.iloc[test_idx]
    y_test = df['target'].iloc[test_idx]

    # Train and evaluate model...
```

### 6. Generate Forecasts

```python
from ml_forecast_lab.features import create_forecast_features

last_ts = clean.index[-1]
last_lags = clean.tail(12).values[::-1]

forecast_features = create_forecast_features(
    last_timestamp=last_ts,
    interval_minutes=exp.interval_minutes,
    horizons_minutes=exp.horizons_minutes,
    n_lags=12,
    lag_values=last_lags,
    country=exp.country
)

predictions = model.predict(forecast_features)
```

---

## Design Principles

1. **Unified APIs**: One function per concept, not four
   - `cumulative_to_interval()` handles all cumulative cases
   - `build_features()` works for all models (GB/XGB/LSTM/CNN)
   - `prepare_train_test()` supports all CV strategies

2. **Production-Ready**:
   - Full error handling and validation
   - Comprehensive logging
   - Type hints throughout
   - British spelling in comments (as per specification)

3. **No Heavy Dependencies**:
   - numpy and pandas only (no sklearn, no TensorFlow)
   - Composable with any ML framework

4. **Temporal Correctness**:
   - Proper lag shifting (no look-ahead bias)
   - Embargo periods prevent temporal leakage
   - Circular encoding for cyclical features
   - Timezone awareness

5. **Flexibility**:
   - Configurable windows for all statistics
   - Multiple transformation options
   - Custom metrics support
   - Pluggable model backends

---

## File Locations

```
/sessions/serene-determined-ride/mnt/PredAI/ml-forecast-lab/ml_forecast_lab/

├── config.py                      # Configuration dataclasses and YAML loader
├── preprocessing.py               # Data cleaning and transformation
├── features.py                    # Feature engineering and CV
├── __init__.py                    # Module exports

├── CONFIG_GUIDE.md                # Detailed config documentation
├── PREPROCESSING_GUIDE.md         # Preprocessing examples and patterns
├── FEATURES_GUIDE.md              # Feature engineering guide
└── CORE_MODULES_README.md         # This file
```

---

## Testing

All modules have been tested for:
- Correct dataclass validation
- Proper YAML loading
- Feature generation (31 features across all categories)
- CV split creation (walk-forward and sliding-window)
- Sequence reshaping for neural models
- Transform/invert round-trips
- Error handling and edge cases

```python
# Run tests
cd /sessions/serene-determined-ride/mnt/PredAI/ml-forecast-lab
python3 -m pytest tests/  # If test suite exists
```

---

## Dependencies

**Required (for core modules):**
- numpy
- pandas
- pyyaml (for YAML loading)

**Optional (for different model backends):**
- lightgbm (for LightGBM models)
- xgboost (for XGBoost models)
- tensorflow (for LSTM/CNN models)
- scikit-learn (for metrics, optional)

---

## Integration with Home Assistant

These modules are designed to integrate with Home Assistant:

1. **Entity IDs**: All sensor references use HA format (sensor.xxx)
2. **Publishing**: Config supports publishing forecasts as HA entities
3. **History API**: Ready for integration with HA history retrieval
4. **Timezone**: Config includes timezone for proper temporal handling

---

## Future Extensions

Potential additions (without modifying core APIs):

- ARIMA/SARIMA support in features module
- Deep learning backends (Transformer, etc.)
- Probabilistic forecasting (quantiles, intervals)
- Adaptive retraining (drift detection)
- Hierarchical forecasting
- Bayesian optimisation for hyperparameters

---

## References

- `CONFIG_GUIDE.md`: Complete configuration documentation
- `PREPROCESSING_GUIDE.md`: Data cleaning patterns and examples
- `FEATURES_GUIDE.md`: Feature engineering and CV strategies
- `__init__.py`: Module exports and API

---

## Version

**ML Forecast Lab Core Modules v0.1.0**

Created: March 2026
