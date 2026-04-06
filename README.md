# ML Forecast Lab

Multi-model machine learning forecasting and benchmarking for Home Assistant.

ML Forecast Lab lets you train, compare, and deploy time-series forecasting models for any Home Assistant sensor. It brings academic-standard evaluation (walk-forward cross-validation, Diebold-Mariano tests, model confidence sets) to the HA ecosystem, so you can make informed decisions about which model actually works best for your data before deploying it.

## Features

**15 model backends**, benchmarked on identical data splits:

| Family | Models |
|--------|--------|
| Gradient boosting | LightGBM, XGBoost |
| Recurrent | LSTM |
| Convolutional | CNN (WaveNet-style dilated causal), TimesNet |
| Linear / MLP | DLinear, TSMixer, TiDE, SparseTSF |
| N-BEATS family | N-BEATS, N-HiTS |
| Transformer | PatchTST, iTransformer, Crossformer |
| Probabilistic | NeuralProphet |

All neural backends use PyTorch and support multi-horizon dense outputs with residual prediction (predict deltas, not absolute values) for stable forecasts.

**Rigorous evaluation:**

- Walk-forward and sliding-window cross-validation with configurable embargo gaps to prevent data leakage
- Multiple metrics: MAE, RMSE, MAPE, sMAPE, MASE, R², pinball loss, coverage
- Composite Demšar (2006) ranking across CV folds for fair multi-metric comparison
- Custom metrics via sandboxed Python expressions
- Diebold-Mariano statistical test for pairwise model comparison

**Decoupled training and inference cycles:**

- **Retrain cycle** (configurable, default 24h) — trains all enabled models from scratch and refreshes the in-memory cache.
- **Forecast cycle** (configurable, default 30min) — runs the cached model for sub-second inference and publishes forecasts as HA sensors.

**Two operational modes:**

- **Lab mode** — trains all enabled models, benchmarks with CV, displays results in the web UI. No forecasts published to HA. Use this for model selection.
- **Production mode** — trains only the selected model on full history, publishes forecasts as HA sensor entities on the forecast cycle.

**Hyperparameter tuning** — Bayesian optimisation (Optuna TPE) per-model with composite-rank trial selection. Default vs tuned holdout comparison plot included.

**Ensembles** — combine multiple models with simple averaging, inverse-metric weighting, or stacking. Re-runnable without retraining.

**Covariate analysis** — automatically test every covariate combination across all enabled models to discover which external features genuinely improve forecasts.

**Auto-generated features** include temporal encodings (hour, day-of-week, month), cyclical sin/cos transforms, configurable lag features, rolling statistics, holiday indicators, and per-covariate aggregations / scalings.

**Built-in web UI** (FastAPI + Jinja2 + Plotly) on port 5052 with experiment dashboard, tabbed experiment view (Models, Training, Results, Predictions, Generalisation, Features, Ensemble, Covariate Analysis, Tuning), live retraining progress, and live forecast/retrain countdowns.

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
  hailo_enabled: false        # Set true if you have a Hailo AI HAT (Raspberry Pi)

experiments:
  - name: mixergy_demand
    target_entity: sensor.mixergy_demand_today
    mode: lab                 # 'lab' for benchmarking, 'production' to deploy
    source_is_cumulative: true
    reset_daily: true
    interval_minutes: 30
    max_increment: 5.0
    days_history: 30
    max_age: 90
    future_periods: 48        # 48 steps × 30min = 24h horizon

    # Per-experiment scheduling (optional — defaults are 30min / 24h)
    forecast_every_minutes: 30
    retrain_every_hours: 24

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
      # 11 more neural backends available: dlinear, nbeats, nhits, tide,
      # tsmixer, sparsetsf, patchtst, itransformer, crossformer, timesnet,
      # neuralprophet

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
| `future_periods` | Number of future steps to forecast | `48` |
| `forecast_every_minutes` | How often to run cached-model inference | `30` |
| `retrain_every_hours` | How often to retrain the model from scratch | `24` |
| `models_enabled` | Which backends to benchmark (15 available) | LightGBM, XGBoost |
| `cv_strategy` | `walk_forward` or `sliding_window` | `walk_forward` |
| `cv_folds` | Number of CV folds | `5` |
| `cv_embargo_periods` | Gap between train/test splits | `2` |

## Web UI

Once running, access the dashboard at:

```
http://homeassistant.local:5052
```

Or via the **Open Web UI** button on the add-on page.

The dashboard shows all configured experiments with status badges and live "Next Forecast" / "Next Retrain" countdowns. Each experiment page contains tabs for model comparison (Demšar composite ranking across CV folds), holdout predictions on a Plotly chart, generalisation diagnostics, feature importance, ensembles, hyperparameter tuning (Optuna TPE), and covariate analysis.

## API

The web UI exposes a JSON API for status, benchmark results, forecasts, model selection, ensembles, hyperparameter tuning, and covariate analysis. Endpoints are not yet stabilised — browse the FastAPI auto-docs at `/docs` for the current surface.

## Architecture

```
                       mlfl.yaml
                           │
                           ▼
          ┌──────────────┐     ┌──────────────────┐
          │ HA Interface │────▶│  History Database│
          │  (API client)│     │     (SQLite)     │
          └──────────────┘     └────────┬─────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │ Feature Engineer │
                              │ lags + temporal +│
                              │   covariates     │
                              └────────┬─────────┘
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
       ┌─────────────────┐   ┌─────────────────┐    ┌──────────────────┐
       │ Cross-Validator │   │ Model Registry  │    │  Hailo Wrapper   │
       │ walk-fwd or SW  │   │ 15 backends     │    │ ONNX → NPU       │
       └────────┬────────┘   │ tree + neural   │    │ + CPU fallback   │
                │            └─────────────────┘    └──────────────────┘
                ▼
      ┌──────────────────┐         ┌──────────────────┐
      │   Benchmarker    │────────▶│  Model Cache     │
      │ Demšar ranking   │         │ (per experiment) │
      └────────┬─────────┘         └────────┬─────────┘
               │                            │
               │                            ▼
               │                  ┌──────────────────┐
               │                  │ Forecast Cycle   │
               │                  │ (every 30 min)   │
               │                  └────────┬─────────┘
               ▼                            │
      ┌──────────────────┐                  ▼
      │     Web UI       │         ┌──────────────────┐
      │  (port 5052)     │         │ HA Sensor        │
      └──────────────────┘         │ Publisher        │
                                   └──────────────────┘
```

The retrain cycle (every ~24h) trains all enabled models from scratch and refreshes the cache. The forecast cycle (every ~30min) uses the cached model for fast inference without retraining.

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

For Raspberry Pi users with a Hailo-8L AI HAT, ML Forecast Lab can export trained neural models to ONNX and run inference on the NPU. Training always runs on CPU; only the forecast cycle is offloaded. Supported across all 13 neural backends (LSTM, CNN, DLinear, N-BEATS, N-HiTS, TiDE, TSMixer, SparseTSF, PatchTST, iTransformer, Crossformer, TimesNet, NeuralProphet).

After every retrain a CPU-vs-NPU validation test runs on a sample of the training data. If the hardware is missing or the outputs diverge from CPU by more than 1%, the add-on falls back to CPU inference and logs a warning — Hailo never silently breaks forecasts.

Set `hailo_enabled: true` in the global config to enable.

## Dependencies

Core: numpy, pandas, scikit-learn, scipy, FastAPI, uvicorn, Jinja2, Plotly.
Tree models: LightGBM, XGBoost.
Neural models: PyTorch (LSTM, CNN, DLinear, N-BEATS, N-HiTS, TiDE, TSMixer, SparseTSF, PatchTST, iTransformer, Crossformer, TimesNet), NeuralProphet.
Hyperparameter tuning: Optuna.
Hardware acceleration (optional): ONNX, hailort (Hailo NPU).

## Licence

MIT

## Credits

Built by [Paul Sweeney](https://github.com/psweens), University of Cambridge.
