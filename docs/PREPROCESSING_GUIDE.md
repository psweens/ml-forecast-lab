# Preprocessing Module Guide

## Overview

The `preprocessing` module provides a unified pipeline for time series data cleaning and transformation. It consolidates four separate cumulative-to-interval paths into one flexible function, plus utilities for resampling, outlier handling, and feature transformations.

## Key Principle

**One function, many paths**: `cumulative_to_interval()` handles daily-reset sensors, non-daily cumulative sensors, gap-aware interpolation, negative-difference handling, and spike capping all in one place.

## Core Functions

### Cumulative-to-Interval Conversion

#### `cumulative_to_interval(series, interval_minutes, reset_daily=False, max_increment=None, method='diff')`

Convert cumulative sensor readings to interval values with robust anomaly handling.

**Parameters:**
- **series** (pd.Series): Cumulative values with DatetimeIndex.
- **interval_minutes** (int): Expected sampling interval in minutes.
- **reset_daily** (bool): If True, sensor resets daily (e.g. 'energy today').
- **max_increment** (float, optional): Maximum allowed increment. Larger values indicate resets or anomalies. If None, computed as 95th percentile.
- **method** (str): Conversion method (currently only 'diff' supported).

**Returns:**
- pd.Series: Interval values (non-negative) with same index.

**What it does:**

1. **Detects resets**: Identifies where cumulative value decreases (negative diff).
2. **Gap-aware interpolation**: Scales differences by actual sample interval (handles missing data).
3. **Spike capping**: Clips outliers exceeding max_increment.
4. **Negative handling**: Treats negative differences as resets or anomalies.

**Example - Daily reset sensor (energy today):**

```python
import pandas as pd
import numpy as np
from ml_forecast_lab.preprocessing import cumulative_to_interval

# Daily reset sensor: increases during day, resets at midnight
idx = pd.date_range('2024-01-01', periods=48, freq='30min')
# Day 1: 0 -> 100 (50 W continuous)
# Day 2: 0 (reset) -> 100
cumul = pd.Series(
    [0, 50, 100, 150, 200] + [0, 50, 100, 150, 200] * 9,
    index=idx[:50]
)

# Convert with daily reset awareness
interval = cumulative_to_interval(
    cumul,
    interval_minutes=30,
    reset_daily=True,
    max_increment=250.0
)

# Result: ~50 W at each 30-min interval (after reset at midnight)
assert (interval.iloc[1:5] > 40).all()  # Intervals are ~50
assert interval.iloc[5] > 0  # Reset detected
```

**Example - Non-daily cumulative (electricity meter):**

```python
# Non-reset cumulative: monotonically increasing
idx = pd.date_range('2024-01-01', periods=100, freq='30min')
cumul = pd.Series(np.linspace(1000, 1500, 100), index=idx)

# Convert without daily reset
interval = cumulative_to_interval(
    cumul,
    interval_minutes=30,
    reset_daily=False,
    max_increment=100.0
)

# All intervals are ~5 (constant consumption)
assert (interval[1:] == interval.iloc[1]).all()
```

**Example - Handling anomalies:**

```python
# Meter with reset and spikes
cumul = pd.Series(
    [0, 10, 20, 30, 40, 0, 10, 100, 20, 30],  # Spike at index 7
    index=pd.date_range('2024-01-01', periods=10, freq='1h')
)

interval = cumulative_to_interval(
    cumul,
    interval_minutes=60,
    reset_daily=True,
    max_increment=15.0  # Cap spikes > 15
)

# Spike is capped: 100 becomes 15
assert interval.iloc[7] <= 15.0
```

---

### Resampling

#### `resample_to_grid(series, freq, method='mean')`

Resample time series to regular frequency grid with gap filling.

**Parameters:**
- **series** (pd.Series): Input series with DatetimeIndex.
- **freq** (str): Target frequency (e.g. '30min', '1h', '1D').
- **method** (str): Aggregation method: 'mean', 'sum', 'max', 'min', 'last', 'forward_fill'.

**Returns:**
- pd.Series: Regularly sampled series.

**Key behaviour:**
- Aggregates sparse data to regular grid
- Forward-fills remaining gaps
- Back-fills any leading NaNs

**Example:**

```python
from ml_forecast_lab.preprocessing import resample_to_grid

# Irregular sampling
idx = pd.DatetimeIndex([
    '2024-01-01 00:00', '2024-01-01 00:15', '2024-01-01 00:32',
    '2024-01-01 01:05', '2024-01-01 01:27'
])
series = pd.Series([10, 12, 11, 15, 14], index=idx)

# Resample to hourly
hourly = resample_to_grid(series, freq='1h', method='mean')
# Result has 2 rows (hour 0 and hour 1) with no gaps
```

---

### Outlier Handling

#### `clip_outliers(series, quantile=0.95, positive_only=False)`

Clip extreme values using quantile-based bounds.

**Parameters:**
- **series** (pd.Series): Input data.
- **quantile** (float, default 0.95): Quantile for bounds (lower=1-q, upper=q).
- **positive_only** (bool): If True, only clip upper tail (assume non-negative).

**Returns:**
- pd.Series: Clipped series.

**Example:**

```python
from ml_forecast_lab.preprocessing import clip_outliers

data = pd.Series([1, 2, 3, 4, 100])  # Last value is outlier

clipped = clip_outliers(data, quantile=0.95)
# Upper bound is ~4.0, so 100 is clipped to ~4.0

clipped_pos = clip_outliers(data, quantile=0.95, positive_only=True)
# Positive-only ignores lower tail
```

---

### Transformations

#### `apply_transform(series, transform)`

Apply optional transformation: 'log', 'sqrt', 'box_cox'.

**Parameters:**
- **series** (pd.Series): Input series.
- **transform** (str, optional): Transformation type or None.

**Returns:**
- pd.Series: Transformed series with metadata for inversion.

**Example:**

```python
from ml_forecast_lab.preprocessing import apply_transform, invert_transform

series = pd.Series([1, 2, 4, 8, 16])

# Log transform
log_series = apply_transform(series, 'log')
# Result: [0, 0.693, 1.386, 2.079, 2.773]

# Invert
original = invert_transform(log_series, 'log')
# Result: [1, 2, 4, 8, 16]
```

#### `apply_log_transform(series, shift=1.0)` / `invert_log_transform(series, shift=None)`

Specialised log transformation with shift for non-positive values.

**Example:**

```python
from ml_forecast_lab.preprocessing import apply_log_transform, invert_log_transform

series = pd.Series([0, 1, 2, 3])  # Has zero

log_series = apply_log_transform(series, shift=1.0)
# log(0 + 1) = 0, log(1 + 1) = 0.693, ...

original = invert_log_transform(log_series)
# Recovers original with same shift
```

---

### Series Operations

#### `subtract_series(base, subtract, fill_method='ffill')`

Vectorised subtraction with automatic reindexing and gap filling.

**Parameters:**
- **base** (pd.Series): Primary series.
- **subtract** (pd.Series): Series to subtract (may have different index).
- **fill_method** (str): How to fill misaligned indices: 'ffill', 'bfill', 'interpolate'.

**Returns:**
- pd.Series: Result of base - subtract.

**Use case:** Calculating net import/export:

```python
from ml_forecast_lab.preprocessing import subtract_series

grid_import = pd.Series([100, 110, 120], index=pd.date_range('2024-01-01', periods=3, freq='1h'))
solar_export = pd.Series([30, 40], index=pd.date_range('2024-01-01', periods=2, freq='1h'))

net = subtract_series(grid_import, solar_export)
# Result: [70, 70, 120] (gaps forward-filled)
```

---

### Unit Conversion

#### `power_to_energy(series, interval_minutes, units='W')`

Convert power (W/kW) to energy (Wh/kWh) over interval.

**Parameters:**
- **series** (pd.Series): Power values in watts or kilowatts.
- **interval_minutes** (int): Sampling interval in minutes.
- **units** (str): Input units: 'W' or 'kW'.

**Returns:**
- pd.Series: Energy in Wh (if W) or kWh (if kW).

**Formula:** Energy = Power × (interval_minutes / 60)

**Example:**

```python
from ml_forecast_lab.preprocessing import power_to_energy

# 1000 W for 30 minutes = 500 Wh
power = pd.Series([1000.0] * 48, index=pd.date_range('2024-01-01', periods=48, freq='30min'))
energy = power_to_energy(power, interval_minutes=30, units='W')

assert (energy == 500.0).all()  # 1000 W * 0.5 hours
```

---

### Index Alignment

#### `align_series(series_list, method='inner')`

Align multiple series to common index (useful for combining target and covariates).

**Parameters:**
- **series_list** (list[pd.Series]): Series with potentially different indices.
- **method** (str): Join method: 'inner', 'outer', 'left', 'right'.

**Returns:**
- list[pd.Series]: Aligned series.

**Example:**

```python
from ml_forecast_lab.preprocessing import align_series

target = pd.Series([1, 2, 3], index=pd.date_range('2024-01-01', periods=3, freq='D'))
covariate1 = pd.Series([10, 20], index=pd.date_range('2024-01-01', periods=2, freq='D'))
covariate2 = pd.Series([100, 200, 300], index=pd.date_range('2024-01-01', periods=3, freq='D'))

# Inner join: only 2024-01-01, 2024-01-02 (intersection)
aligned = align_series([target, covariate1, covariate2], method='inner')
assert len(aligned[0]) == 2
```

---

## Usage Patterns

### Pattern 1: Cumulative daily-reset sensor (energy today)

```python
from ml_forecast_lab.preprocessing import (
    cumulative_to_interval,
    resample_to_grid,
    clip_outliers
)

# Get daily-reset energy readings from Home Assistant history
cumul = fetch_sensor_history('sensor.energy_today_kwh')

# Convert to intervals
intervals = cumulative_to_interval(
    cumul,
    interval_minutes=30,
    reset_daily=True,
    max_increment=10.0  # Max 10 kWh in 30min
)

# Resample to regular grid (fill gaps from missing data)
regular = resample_to_grid(intervals, freq='30min', method='mean')

# Remove outliers
clean = clip_outliers(regular, quantile=0.99, positive_only=True)
```

### Pattern 2: Non-cumulative with transformation

```python
from ml_forecast_lab.preprocessing import apply_log_transform

# Power sensor (not cumulative)
power = fetch_sensor_history('sensor.power_w')

# Log transform for normalisation
log_power = apply_log_transform(power, shift=1.0)

# Model on log scale
predictions_log = model.predict(log_power)

# Invert for interpretation
predictions_w = invert_log_transform(predictions_log)
```

### Pattern 3: Net metering (import - export)

```python
from ml_forecast_lab.preprocessing import (
    cumulative_to_interval,
    subtract_series,
    clip_outliers
)

# Grid import and solar export (both cumulative)
import_cumul = fetch_sensor_history('sensor.grid_import_kwh')
export_cumul = fetch_sensor_history('sensor.solar_export_kwh')

# Convert both to intervals
import_intervals = cumulative_to_interval(import_cumul, 30, reset_daily=True)
export_intervals = cumulative_to_interval(export_cumul, 30, reset_daily=True)

# Net import (can be negative = net export)
net = subtract_series(import_intervals, export_intervals)

# Remove outliers
clean_net = clip_outliers(net, quantile=0.99, positive_only=False)
```

---

## Error Handling

```python
from ml_forecast_lab.preprocessing import cumulative_to_interval

try:
    result = cumulative_to_interval(
        series,
        interval_minutes=-1  # ERROR: must be >= 1
    )
except ValueError as e:
    print(f'Invalid parameter: {e}')

try:
    result = cumulative_to_interval(
        series,
        interval_minutes=30,
        quantile=1.5  # ERROR: must be in (0, 1)
    )
except ValueError as e:
    print(f'Invalid parameter: {e}')
```

---

## Logging

All functions use Python's standard `logging` module:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('ml_forecast_lab.preprocessing')

# Now you'll see debug messages from preprocessing functions
```

---

## Performance Notes

- **cumulative_to_interval**: O(n) with two passes over data
- **resample_to_grid**: O(n) resampling + O(n) gap-filling
- **clip_outliers**: O(n) for quantile computation
- **apply_transform**: O(n) element-wise
- All functions preserve index and dtype information
