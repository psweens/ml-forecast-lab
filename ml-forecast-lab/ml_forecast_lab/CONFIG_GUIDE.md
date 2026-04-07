# Configuration Module Guide

## Overview

The `config` module provides dataclasses and YAML loading for ML Forecast Lab experiment configuration. It enables declarative setup of time series forecasting experiments with flexible covariate handling and cross-validation strategies.

## Core Classes

### `CovariateCfg`

Specifies a single external feature (covariate) for an experiment.

#### Fields

- **entity** (str): Home Assistant sensor entity_id (e.g. `sensor.temperature`).
- **role** (str): Feature availability:
  - `'future'`: Known in advance (e.g. weather forecasts, scheduled events)
  - `'lagged'`: Historical only, cannot be used for forecasting
  - `'both'`: Can be used both historically and for forecasting
- **scale** (float, optional): Multiplicative scaling factor. If None, no scaling applied.
- **transform** (str, optional): Transformation method:
  - `None`: No transformation
  - `'log'`: Log transformation (after shift for non-positive values)
  - `'sqrt'`: Square root (negative values clipped to 0)
  - `'box_cox'`: Simplified Box-Cox transformation
- **aggregation** (str): Resampling method when aligning to the experiment frequency:
  - `'mean'`: Average over interval
  - `'sum'`: Total over interval
  - `'max'`: Maximum value
  - `'min'`: Minimum value
  - `'last'`: Last observed value
- **is_binary** (bool): Whether this is a 0/1 indicator feature.

#### Example

```python
from ml_forecast_lab.config import CovariateCfg

# Solar-aware covariate
irradiance = CovariateCfg(
    entity='sensor.solar_irradiance_w_m2',
    role='future',
    scale=1.0,
    aggregation='mean'
)

# Temperature (lagged only, log-transformed)
temperature = CovariateCfg(
    entity='sensor.ambient_temperature_c',
    role='lagged',
    scale=0.1,
    transform='log',
    aggregation='mean'
)
```

### `ExperimentCfg`

Defines a complete forecasting experiment for one sensor.

#### Core Fields

- **name** (str): Unique experiment identifier (e.g. `'solar_forecast'`).
- **target_entity** (str): Home Assistant sensor to predict.
- **covariates** (list[CovariateCfg]): External features.
- **days_history** (int, default 14): Training data window in days.
- **interval_minutes** (int, default 30): Sampling frequency.
- **horizons_minutes** (list[int]): Prediction lookaheads (e.g. [120, 480, 1440] = 2h, 8h, 24h).

#### Cumulative Handling

For sensors reporting cumulative values (e.g. energy meters):

- **source_is_cumulative** (bool): Whether sensor is cumulative.
- **reset_daily** (bool): If True, sensor resets daily (e.g. 'energy today').
- **max_increment** (float, optional): Maximum valid change per interval. Exceeds indicate anomalies.

#### Feature Engineering

- **log_transform** (bool): Apply log to target before modelling.
- **subtract** (list[str]): Entity IDs to subtract (e.g. net import = import - export).
- **country** (str, optional): Country code for holiday features ('GB', 'US', 'DE').
- **units** (str): Target units (e.g. 'W', 'kWh', 'L').
- **output_units** (str, optional): Convert output to different units.

#### Cross-Validation

- **cv_strategy** (str, default 'walk_forward'):
  - `'walk_forward'`: Expanding window (expanding train, fixed test)
  - `'sliding_window'`: Fixed window size (train and test slide together)
- **cv_folds** (int, default 5): Number of folds.
- **cv_embargo_periods** (int, default 2): Gap between train and test to prevent leakage.

#### Model Selection

- **models_enabled** (list[str]): Models to train (e.g. `['lightgbm', 'xgboost', 'lstm', 'cnn']`).
- **metrics** (list[str]): Metrics to compute (e.g. `['mae', 'rmse', 'mape']`).
- **custom_metrics** (dict[str, str], optional): Custom metrics as `{name: 'Python expression'}`.
- **production_model** (str, optional): Fixed model to use in production (None = auto-select).
- **production_metric** (str, default 'mae'): Metric for auto-selection.

#### Publishing

- **publish_prefix** (str, default 'mlfl_'): Entity name prefix.
- **publish_interval** (bool): Publish interval values (if cumulative input).
- **publish_cumulative** (bool): Publish reconstructed cumulative.
- **publish_daily_cumulative** (bool): Publish daily totals.

#### Example

```python
from ml_forecast_lab.config import ExperimentCfg, CovariateCfg

exp = ExperimentCfg(
    name='solar_forecast',
    target_entity='sensor.solar_generation_w',

    # Cumulative handling
    source_is_cumulative=True,
    reset_daily=True,
    max_increment=10000.0,  # Max watts in 30min

    # Data window
    days_history=30,
    interval_minutes=30,
    horizons_minutes=[120, 480, 1440],  # 2h, 8h, 24h

    # Features
    country='GB',
    units='W',
    log_transform=False,
    subtract=['sensor.solar_export_w'],

    # Models
    models_enabled=['lightgbm', 'xgboost', 'lstm'],
    cv_strategy='walk_forward',
    cv_folds=5,
    cv_embargo_periods=2,
    metrics=['mae', 'rmse', 'mape'],
    production_metric='mae',

    # Covariates
    covariates=[
        CovariateCfg(
            entity='sensor.cloud_cover_percent',
            role='future',
            aggregation='mean'
        ),
        CovariateCfg(
            entity='sensor.outdoor_temperature_c',
            role='lagged',
            scale=0.1,
            aggregation='mean'
        ),
    ],

    # Publishing
    publish_prefix='mlfl_',
    publish_interval=True,
)
```

### `AppConfig`

Top-level application configuration.

#### Fields

- **update_every_minutes** (int, default 5): How often to run inference.
- **timezone** (str, default 'UTC'): Timezone for temporal features.
- **experiments** (list[ExperimentCfg]): Experiment configurations.

#### Example

```python
from ml_forecast_lab.config import AppConfig, ExperimentCfg

app = AppConfig(
    update_every_minutes=5,
    timezone='Europe/London',
    experiments=[exp1, exp2, exp3]
)
```

## YAML Configuration

Load configuration from YAML using `load_config()`:

```python
from pathlib import Path
from ml_forecast_lab.config import load_config

cfg = load_config(Path('config.yaml'))
```

### YAML Schema

```yaml
# Application settings
update_every_minutes: 5
timezone: Europe/London

# Experiments
experiments:
  - name: solar_forecast
    target_entity: sensor.solar_generation_w

    # Cumulative settings
    source_is_cumulative: true
    reset_daily: true
    max_increment: 10000.0

    # Data window
    days_history: 30
    interval_minutes: 30
    horizons_minutes: [120, 480, 1440]

    # Features
    units: W
    output_units: null
    log_transform: false
    country: GB
    subtract: []

    # Models and CV
    models_enabled: [lightgbm, xgboost, lstm, cnn]
    cv_strategy: walk_forward
    cv_folds: 5
    cv_embargo_periods: 2

    # Metrics
    metrics: [mae, rmse, mape]
    custom_metrics: null
    production_model: null
    production_metric: mae

    # Publishing
    publish_prefix: mlfl_
    publish_interval: true
    publish_cumulative: false
    publish_daily_cumulative: false

    # Covariates
    covariates:
      - entity: sensor.cloud_cover_percent
        role: future
        scale: null
        transform: null
        aggregation: mean
        is_binary: false

      - entity: sensor.outdoor_temperature_c
        role: lagged
        scale: 0.1
        transform: null
        aggregation: mean
        is_binary: false
```

## Validation

All dataclasses validate their arguments:

```python
# This raises ValueError: cv_strategy must be one of {'walk_forward', 'sliding_window'}
exp = ExperimentCfg(
    name='test',
    target_entity='sensor.test',
    cv_strategy='invalid'  # ERROR
)

# This raises ValueError: cv_folds must be >= 2
exp = ExperimentCfg(
    name='test',
    target_entity='sensor.test',
    cv_folds=1  # ERROR
)
```

## Best Practices

1. **Use sensible defaults**: Most fields have production-ready defaults.
2. **Specify country for holidays**: Include `country` for better temporal features.
3. **Choose CV strategy carefully**:
   - Use `'walk_forward'` for realistic, expanding-window evaluation (production-like).
   - Use `'sliding_window'` for stationary data or benchmark comparisons.
4. **Set embargo_periods**: Use `cv_embargo_periods >= 2` to prevent temporal leakage.
5. **Configure publish entities**: Enable publishing for Home Assistant integration.
6. **Document custom metrics**: Use clear expression strings in `custom_metrics`.

## Error Handling

```python
from ml_forecast_lab.config import load_config
from pathlib import Path

try:
    cfg = load_config('config.yaml')
except FileNotFoundError:
    print('Config file not found')
except ValueError as e:
    print(f'Invalid configuration: {e}')
```
