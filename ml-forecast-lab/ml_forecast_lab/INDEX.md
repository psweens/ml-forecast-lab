# ML Forecast Lab Core Modules - Index

## Quick Navigation

### Getting Started
- **New to ML Forecast Lab?** Start with [CORE_MODULES_README.md](CORE_MODULES_README.md)
- **Want to configure experiments?** See [CONFIG_GUIDE.md](CONFIG_GUIDE.md)
- **Need to preprocess data?** Check [PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md)
- **Building features?** Read [FEATURES_GUIDE.md](FEATURES_GUIDE.md)
- **Full creation details?** View [CREATION_REPORT.md](CREATION_REPORT.md)

### Core Modules

#### config.py
**Configuration and experiment setup**
- `CovariateCfg` - External feature specification
- `ExperimentCfg` - Complete forecasting experiment
- `AppConfig` - Application settings
- `load_config(path)` - YAML file loading

**Import:**
```python
from ml_forecast_lab.config import CovariateCfg, ExperimentCfg, AppConfig, load_config
```

**Use when:** Setting up experiments, configuring sensor predictions, defining cross-validation strategies.

---

#### preprocessing.py
**Data cleaning and transformation**
- `cumulative_to_interval()` - UNIFIED cumulative conversion (daily-reset, gaps, spikes)
- `resample_to_grid()` - Regular frequency resampling
- `clip_outliers()` - Quantile-based outlier handling
- `apply_transform()` / `invert_transform()` - Log, sqrt, Box-Cox
- `subtract_series()` - Net metering (import - export)
- `power_to_energy()` - W/kW to Wh/kWh conversion
- `align_series()` - Multi-series index alignment

**Import:**
```python
from ml_forecast_lab.preprocessing import cumulative_to_interval, resample_to_grid, clip_outliers, ...
```

**Use when:** Cleaning sensor data, handling cumulative meters, converting units, removing outliers.

---

#### features.py
**Feature engineering and cross-validation**
- `build_features()` - Generate 31 temporal/lag/rolling/holiday features
- `prepare_train_test()` - Walk-forward and sliding-window CV with embargo
- `reshape_for_sequence()` - Convert to 3D for RNN/CNN models
- `create_forecast_features()` - Generate features for out-of-sample prediction
- `is_holiday()` - Holiday detection (GB, US, DE)

**Import:**
```python
from ml_forecast_lab.features import build_features, prepare_train_test, reshape_for_sequence, ...
```

**Use when:** Creating ML features, setting up cross-validation, forecasting at future horizons.

---

### Documentation

| Document | Size | Purpose |
|----------|------|---------|
| [CORE_MODULES_README.md](CORE_MODULES_README.md) | 8 KB | High-level overview, quick start guide |
| [CONFIG_GUIDE.md](CONFIG_GUIDE.md) | 8.6 KB | Configuration reference and examples |
| [PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md) | 12 KB | Data cleaning patterns and examples |
| [FEATURES_GUIDE.md](FEATURES_GUIDE.md) | 15 KB | Feature engineering and CV strategies |
| [CREATION_REPORT.md](CREATION_REPORT.md) | 10 KB | Detailed creation and testing report |

---

### Common Tasks

#### I want to...

**Configure an experiment**
1. Read: [CONFIG_GUIDE.md](CONFIG_GUIDE.md)
2. Use: `config.py` with `load_config()`
3. Example: Load YAML and iterate over experiments

**Process a sensor reading**
1. Read: [PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md)
2. Use: `preprocessing.py` functions
3. Example: `cumulative_to_interval()` for daily-reset energy meters

**Create ML training data**
1. Read: [FEATURES_GUIDE.md](FEATURES_GUIDE.md)
2. Use: `features.py` functions
3. Example: `build_features()` and `prepare_train_test()`

**Train models for LSTM/CNN**
1. Read: [FEATURES_GUIDE.md](FEATURES_GUIDE.md) - Sequence Reshaping section
2. Use: `reshape_for_sequence()` to convert flat features to 3D
3. Example: Reshape before fitting neural model

**Generate production forecasts**
1. Read: [FEATURES_GUIDE.md](FEATURES_GUIDE.md) - Forecast Features section
2. Use: `create_forecast_features()` for future horizons
3. Example: Create features for t+2h, t+8h, t+24h predictions

---

### API Reference

#### Configuration Classes

```python
# Define a covariate
cov = CovariateCfg(
    entity='sensor.temperature',
    role='lagged',  # 'future', 'lagged', 'both'
    scale=0.1,
    transform='log',  # None, 'log', 'sqrt', 'box_cox'
    aggregation='mean'  # 'mean', 'sum', 'max', 'min', 'last'
)

# Define an experiment
exp = ExperimentCfg(
    name='solar_forecast',
    target_entity='sensor.solar_w',
    source_is_cumulative=True,
    reset_daily=True,
    days_history=30,
    interval_minutes=30,
    horizons_minutes=[120, 480, 1440],
    cv_strategy='walk_forward',  # 'walk_forward' or 'sliding_window'
    cv_folds=5,
    metrics=['mae', 'rmse', 'mape']
)

# Load from YAML
cfg = load_config('config.yaml')
```

#### Preprocessing Functions

```python
# Convert cumulative to intervals
intervals = cumulative_to_interval(
    series,
    interval_minutes=30,
    reset_daily=True,      # Daily-reset sensor?
    max_increment=10000.0  # Max spike threshold
)

# Resample to regular grid
regular = resample_to_grid(series, freq='30min', method='mean')

# Remove outliers
clean = clip_outliers(series, quantile=0.95, positive_only=True)

# Transform and invert
log_series = apply_log_transform(series, shift=1.0)
original = invert_log_transform(log_series)
```

#### Feature Engineering Functions

```python
# Build 31 features
features = build_features(
    df,
    target_col='target',
    interval_minutes=30,
    n_lags=12,
    lag_windows=[6, 24, 72],
    country='GB'  # For holidays
)

# Create CV splits
splits = prepare_train_test(
    df, features,
    cv_strategy='walk_forward',
    n_folds=5,
    embargo_periods=2
)

# Reshape for neural models
X_seq = reshape_for_sequence(X.values, n_lags=12)

# Generate forecast features
forecast_features = create_forecast_features(
    last_timestamp=df.index[-1],
    interval_minutes=30,
    horizons_minutes=[120, 480, 1440],
    n_lags=12,
    lag_values=last_12_values,
    country='GB'
)
```

---

### Dependencies

**Required:**
- numpy
- pandas
- pyyaml

**Optional (for model backends):**
- lightgbm
- xgboost
- tensorflow
- scikit-learn

---

### Version

**ML Forecast Lab Core Modules v0.1.0**

Created: 2026-03-29  
Status: Production-Ready

---

### Support

- Questions about configuration? See [CONFIG_GUIDE.md](CONFIG_GUIDE.md)
- Issues with data preprocessing? See [PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md)
- Feature engineering help? See [FEATURES_GUIDE.md](FEATURES_GUIDE.md)
- General overview? See [CORE_MODULES_README.md](CORE_MODULES_README.md)
- Implementation details? See [CREATION_REPORT.md](CREATION_REPORT.md)

---

### Files at a Glance

```
Core Code (41.8 KB):
├── config.py (9.1 KB) - 3 classes, 1 function
├── preprocessing.py (15 KB) - 11 functions
├── features.py (16 KB) - 6 functions
└── __init__.py (1.8 KB) - Exports

Documentation (43.6 KB):
├── CORE_MODULES_README.md - Start here
├── CONFIG_GUIDE.md - Configuration reference
├── PREPROCESSING_GUIDE.md - Data cleaning guide
├── FEATURES_GUIDE.md - Feature engineering guide
├── CREATION_REPORT.md - Full details
└── INDEX.md (this file)
```

Total: 85.4 KB of code and documentation
