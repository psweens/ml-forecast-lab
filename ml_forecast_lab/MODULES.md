# ML Forecast Lab Modules

Production-ready Home Assistant integration modules for time-series forecasting.

## ha_interface.py

Async Home Assistant REST API client with robust timestamp parsing and state conversion.

**Classes:**
- `HAInterface`: Main API client
  - `__init__(ha_url, ha_key, session)`: Initialise with HA URL and auth token
  - `async api_call(method, endpoint, params, json_data)`: Generic HTTP call with error handling
  - `async get_history(entity_id, start, end)`: Fetch sensor history
  - `async get_state(entity_id, default, attribute)`: Get current state or attribute
  - `async set_state(entity_id, state, attributes)`: Publish state
  - `async close()`: Cleanup session

**Functions:**
- `parse_timestamp(ts_string)`: Robust ISO8601 parser (handles HA variants with/without fractional seconds, with/without colon offset)
- `ensure_utc(dt)`: Convert datetime to UTC
- `state_to_float(x)`: Convert HA state to float (handles on/off, true/false, home/not_home, unknown/unavailable)
- `normalise_history(raw_records)`: Convert HA history response to clean DataFrame with columns ['ds', 'value']

## db.py

SQLite history cache with efficient bulk insert and cleanup.

**Classes:**
- `HistoryDB`: SQLite database for entity history
  - `__init__(path)`: Initialise database
  - `safe_table_name(entity_id)`: Convert entity ID to SQL-safe table name
  - `ensure_table(table_name)`: Create table if needed
  - `store_history(table_name, df)`: Bulk INSERT OR IGNORE (using executemany, not row iteration)
  - `get_history(table_name)`: Retrieve all history as DataFrame with ['ds', 'y']
  - `cleanup(table_name, oldest_datetime)`: Delete records before date
  - `close()`: Close database connection

## covariates.py

Covariate fetching and resampling with intelligent binary detection.

**Classes:**
- `CovariateResolver`: Fetch and process covariate data
  - `__init__(iface, covariate_configs)`: Initialise resolver
  - `async fetch_history(cov_cfg, start, end, freq)`: Fetch and resample historical covariate
  - `async fetch_future(cov_cfg, future_index)`: Get future covariate values (from forecast attribute or constant fill)
  - `_detect_binary(series)`: Auto-detect binary covariates (0/1 values only)
  - `_resample_covariate(series, freq, is_binary)`: Resample with appropriate aggregation (forward-fill for binary, mean for continuous)

Binary detection and resampling logic is centralised in private methods (no duplication).

## publishing.py

Forecast publishing to Home Assistant with multiple entity types and aggregation strategies.

**Functions:**
- `async publish_forecasts(experiment_cfg, iface, app_config, ds_future, yhat_interval, yhat_level, metrics, hist_cum_df)`: Publish all forecast entities:
  - Point forecast
  - Interval bounds (upper/lower at specified confidence level)
  - Cumulative forecast
  - Daily cumulative with offset
  - Horizon scalar entities (+2h, +8h, etc.)
  - Prediction curve (historical + forecast)

- `make_entity_name(publish_prefix, experiment_name, suffix)`: Construct entity names
- `dict_from_series(series, max_points)`: Serialise Series to dict for HA attribute
- `daily_cumulative_series(forecast_series, reference_date)`: Group by date and cumulate within each day
- `energy_already_used_today(iface, entity_id)`: Fetch energy used so far today

## Key Features

- **British spelling** in all comments and docstrings
- **Proper type hints** throughout
- **Minimal imports**: aiohttp, sqlite3, numpy, pandas, datetime
- **Async/await** for all I/O operations
- **Comprehensive logging** at module level
- **Error handling** with graceful degradation
- **Production-ready** code with no external dependencies beyond standard ML stack

## Configuration Examples

### Covariate Config
```python
covariate_configs = [
    {
        "entity_id": "sensor.temperature",
        "name": "temperature",
        "binary": False,
    },
    {
        "entity_id": "binary_sensor.cloud_coverage",
        "name": "cloudy",
        "binary": True,
    },
    {
        "entity_id": "sensor.day_of_week",
        "name": "day_of_week",
        "constant_value": 3,  # Use constant for future
    },
]
```

### Experiment Config
```python
experiment_cfg = {
    "name": "solar_forecast",
    "publish_prefix": "mlfl_",
    "publish_entity_id": "sensor.mlfl_solar_point",
    "horizons_to_publish": ["+2h", "+4h", "+8h", "+12h"],
}
```

## Dependencies

- `aiohttp`: Async HTTP client
- `sqlite3`: Built-in database
- `numpy`: Numeric operations
- `pandas`: DataFrames
- `logging`: Standard logging
