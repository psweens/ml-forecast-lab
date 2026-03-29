# ML Forecast Lab

Multi-model machine learning forecasting and benchmarking for Home Assistant.

ML Forecast Lab lets you train, compare, and deploy time-series forecasting models for any Home Assistant sensor. It brings academic-standard evaluation (walk-forward cross-validation, Diebold-Mariano tests, model confidence sets) to the HA ecosystem, so you can make informed decisions about which model actually works best for your data before deploying it.

## Features

**Four model backends**, benchmarked on identical data splits:

| Model | Type | Notes |
|-------|------|-------|
| LightGBM | Gradient boosting | Fast training, recursive multi-step prediction, early stopping |
| XGBoost | Gradient boosting | Alternative GBDT implementation |
| LSTM | Recurrent neural network | Pure NumPy — no PyTorch/TensorFlow dependency |
| CNN | 1D dilated causal convolution | WaveNet-style architecture, pure NumPy, residual connections |

The neural backends are implemented entirely in NumPy to keep the Docker image small enough for Raspberry Pi deployment (~50 MB vs ~800 MB+ for PyTorch/TensorFlow).

**Rigorous evaluation:**

- Walk-forward and sliding-window cross-validation with configurable embargo gaps to prevent data leakage
- 8 standard metrics: MAE, RMSE, MAPE, sMAPE, MASE, R², pinball loss, coverage
- Custom metrics via sandboxed Python expressions
- Diebold-Mariano statistical test for pairwise model comparison
- Model Confidence Sets to identify statistically indistinguishable best models

**Two operational modes:**

- **Lab mode** — trains all enabled models, benchmarks with CV, displays results in the web UI. No forecasts published to HA. Use this for model selection.
- **Production mode** — trains only the promoted (best) model on full history, publishes forecasts as HA sensor entities on a configurable cycle.

**26 auto-generated features** including temporal encodings (hour, day-of-week, month), cyclical sin/cos transforms, configurable lag features, rolling statistics (mean, std, max), and holiday indicators.

**Built-in web UI** (FastAPI + HTMX + Plotly) on port 5052 with experiment dashboard, model comparison tables, forecast charts, and one-click model promotion.

## Installation

### As a custom repository (recommended)

1. In Home Assistant, go to **Settings → Add-ons → Add-on store**
2. Click **⋮** (top right) → **Repositories**
3. Add: `https://github.com/psweens/ml-forecast-lab`
4. ML Forecast Lab will appear in the store — click **Install**

The first build takes 10-15 minutes on a Raspberry Pi 5 (compiling LightGBM and XGBoost for ARM).

### Supported architectures

- `aarch64` (Raspberry Pi 4/5, etc.)
- `amd64` (x86-64 servers)
- `armv7`

## Configuration

Create `/config/mlfl.yaml` (or `/addon_configs/ml_forecast_lab/mlfl.yaml`) with your experiments. Example for a Mixergy hot water cylinder:

```yaml
global:
  timezone: "Europe/London"
  update_every_minutes: 360
  hailo_enabled: false

experiments:
  - name: mixergy_demand
    target_entity: sensor.mixergy_demand_today
    mode: lab
    source_is_cumulative: true
    reset_daily: true
    interval_minutes: 30
    max_increment: 5.0
    days_history: 30
    max_age: 90
    horizons_minutes: [120, 480, 720, 1440]
    future_periods: 48

    covariates:
      - entity: sensor.mixergy_current_charge
        role: lagged
        aggregation: mean
        scaling: standard
      - entity: sensor.external_temperature
        role: concurrent
        aggregation: mean
        scaling: standard

    models_enabled:
      - lightgbm
      - xgboost
      - lstm
      - cnn

    cv_strategy: walk_forward
    cv_folds: 5
    cv_embargo_periods: 2
```

### Configuration reference

| Field | Description | Default |
|-------|-------------|---------|
| `target_entity` | HA sensor entity to forecast | *required* |
| `mode` | `lab` (benchmarking) or `production` (deploy best model) | `lab` |
| `source_is_cumulative` | Set `true` if sensor reports running totals | `false` |
| `reset_daily` | Cumulative sensor resets at midnight | `false` |
| `interval_minutes` | Resampling grid interval | `30` |
| `days_history` | Days of history to fetch for training | `30` |
| `horizons_minutes` | Forecast horizons to evaluate | `[120, 480, 1440]` |
| `models_enabled` | Which backends to benchmark | all four |
| `cv_strategy` | `walk_forward` or `sliding_window` | `walk_forward` |
| `cv_folds` | Number of CV folds | `5` |
| `cv_embargo_periods` | Gap between train/test splits | `2` |

## Web UI

Once running, access the dashboard at:

```
http://homeassistant.local:5052
```

Or via the **Open Web UI** button on the add-on page.

The dashboard shows all configured experiments with status badges, and each experiment page displays model comparison tables with metrics, interactive forecast charts (via Plotly), and a button to promote the best model to production.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Health check and experiment counts |
| `/api/models` | GET | List available model backends |
| `/experiment/{name}/results` | GET | Latest benchmark results (JSON) |
| `/experiment/{name}/forecast` | GET | Forecast data for charting |
| `/experiment/{name}/run-benchmark` | POST | Trigger benchmark run |
| `/experiment/{name}/promote/{model}` | POST | Promote model to production |

## Architecture

```
mlfl.yaml
    │
    ▼
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ HA Interface │────▶│ History Database │────▶│ Feature Engineer │
│  (API client)│     │    (SQLite)      │     │  (26 features)   │
└──────────────┘     └─────────────────┘     └────────┬─────────┘
                                                       │
                           ┌───────────────────────────┤
                           ▼                           ▼
                  ┌─────────────────┐        ┌──────────────────┐
                  │  Cross-Validator │        │  Model Registry  │
                  │ (walk-fwd / SW) │        │ LGB│XGB│LSTM│CNN │
                  └────────┬────────┘        └──────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Benchmarker   │
                  │ metrics + stats │
                  └────────┬────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Web UI  │ │ Forecast │ │   DM /   │
        │ (5052)   │ │ Publisher│ │   MCS    │
        └──────────┘ └──────────┘ └──────────┘
```

## Multiple experiments

ML Forecast Lab supports multiple simultaneous experiments — one per sensor. Add additional entries under `experiments:` in your config:

```yaml
experiments:
  - name: mixergy_demand
    target_entity: sensor.mixergy_demand_today
    # ...

  - name: solar_generation
    target_entity: sensor.solar_power
    mode: lab
    interval_minutes: 15
    # ...
```

Each experiment trains and evaluates models independently.

## Hailo AI HAT support (optional)

For Raspberry Pi users with a Hailo-8L AI HAT, ML Forecast Lab can export trained neural models (LSTM, CNN) to ONNX and compile them to HEF format for NPU-accelerated inference. Training always runs on CPU; only inference is offloaded.

Set `hailo_enabled: true` in the global config to enable.

## Dependencies

numpy, pandas, LightGBM, XGBoost, scikit-learn, scipy, ONNX, FastAPI, uvicorn, Jinja2, Plotly.

## Licence

MIT

## Credits

Built by [Paul Sweeney](https://github.com/psweens), University of Cambridge.
