# ML Forecast Lab

Multi-model machine learning forecasting and benchmarking for Home Assistant.

ML Forecast Lab lets you train, compare, and deploy time-series forecasting models for any Home Assistant sensor. It brings academic-standard evaluation (walk-forward cross-validation, Diebold-Mariano tests, model confidence sets) to the HA ecosystem, so you can make informed decisions about which model actually works best for your data before deploying it.

The intended workflow is **benchmark once, run forever**. Lab mode does the heavy lifting up front — trains every enabled backend, ranks them by composite Demšar score on your actual data, and shows you which one wins. Production mode is then set-and-forget: the chosen model retrains on a schedule and publishes forecasts back to HA as sensors. You don't need to babysit it after the initial benchmark.

## How it differs from existing forecasting add-ons

The HA forecasting space has good narrow tools — [EMHASS](https://github.com/davidusb-geek/emhass) for energy management, [Solar Forecast ML](https://github.com/Zara-Toorox/Solar-Forecast-ML) for PV production, [predbat](https://github.com/springfall2008/batpred) for battery optimisation. ML Forecast Lab fills a different niche: **the general-purpose case**, where you want to forecast *any* sensor (water demand, occupancy, grid prices, EV charge state, room temperature, anything) and you want the add-on to actually compare modern model architectures on your data rather than picking one for you.

If your forecasting problem fits cleanly into a domain-specific tool's wheelhouse, use that. If it doesn't — or you want side-by-side benchmarks of 24 backends including the recent transformer / N-HiTS / TFT / TimeMixer architectures — that's what this is for.

## Features

**24 model backends**, benchmarked on identical data splits:

| Family | Models |
|--------|--------|
| Gradient boosting | LightGBM, XGBoost, CatBoost |
| Recurrent | LSTM, GRU |
| Convolutional | CNN (WaveNet-style dilated causal), TimesNet |
| Linear / MLP | DLinear, NLinear, TSMixer, TimeMixer, TiDE, SparseTSF |
| Frequency-domain | FITS (~10k parameters) |
| N-BEATS family | N-BEATS, N-HiTS |
| Transformer | PatchTST, iTransformer, Crossformer, TFT |
| Classical | AutoARIMA, AutoETS, AutoTheta (via statsforecast) |
| Baseline | Seasonal Naive |

All neural backends use PyTorch and support multi-horizon dense outputs with residual prediction (predict deltas, not absolute values) for stable forecasts. Loss function (MSE / MAE / Huber) is configurable per experiment.

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

**Uncertainty intervals** — conformal prediction 80% bands published as companion `_upper_80` / `_lower_80` HA sensors, calibrated on held-out residuals.

**Forecast accuracy tracking** — every prediction is logged with its model version and compared against actuals as ground truth arrives. Three-layer diagnostic UI shows bias, trajectory drift, and per-horizon error; the log auto-clears on retrain so a new model's accuracy isn't contaminated by the old one's.

**Load subtract** — optionally subtract one HA sensor from another before modelling (e.g., net-of-solar demand) with a robustness layer to handle missing covariate data and an audit log of subtraction events.

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

## Quick start

End-to-end first-run flow once the add-on is installed:

1. **Create `mlfl.yaml`** at `/addon_configs/ml_forecast_lab/mlfl.yaml` with one experiment targeting one HA sensor. The minimal viable config is just `name`, `target_entity`, and `models_enabled` — see [Configuration](#configuration) below for the full schema.
2. **Start the add-on.** Open the Web UI (`http://homeassistant.local:5052` or via the **Open Web UI** button). Your experiment appears on the dashboard in **lab mode** with status *pending*.
3. **Run the benchmark.** Click into the experiment → **Run Pipeline**. Every enabled model trains on your sensor's history with walk-forward CV (typically a few minutes for tree models, longer for neural). Results land in the **Models** tab, ranked by composite Demšar score across MAE / RMSE / MASE.
4. **Pick a model.** The top-ranked one is auto-selected, but you can override from the **Models** tab. Click **Promote to Production** on your choice.
5. **Production mode is on.** The add-on now retrains the chosen model on the configured schedule (default: every 24h) and publishes forecasts as `sensor.mlfl_<experiment_name>` companion sensors with `_lower_80` / `_upper_80` conformal bands. The forecast cycle (default: every 30min) just runs the cached model — sub-second inference, no retrain.
6. **Watch accuracy in production.** The **Forecast Accuracy** tab compares each logged prediction against the actual once it arrives. Bias, per-horizon error, and trajectory drift are tracked automatically. Re-benchmark whenever you want to swap to a different model.

That's the whole loop: benchmark once, set the winner, walk away. Re-benchmark only when your sensor's behaviour changes (new equipment, seasonal drift) or when you want to try newer model backends.

## Configuration

Create `/config/mlfl.yaml` (or `/addon_configs/ml_forecast_lab/mlfl.yaml`) with your experiments. Example for a Mixergy hot water cylinder:

```yaml
global:
  timezone: "Europe/London"

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
      # 20 more backends available: catboost, gru, dlinear, nlinear, fits,
      # nbeats, nhits, tide, tsmixer, timemixer, sparsetsf, patchtst,
      # itransformer, crossformer, timesnet, tft, seasonal_naive,
      # arima, ets, theta

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
| `models_enabled` | Which backends to benchmark (24 available) | LightGBM, XGBoost |
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
                ▼                      ▼                      │
       ┌─────────────────┐   ┌─────────────────┐               │
       │ Cross-Validator │   │ Model Registry  │               │
       │ walk-fwd or SW  │   │ 24 backends     │               │
       └────────┬────────┘   │ tree+neural+cls │               │
                │            └─────────────────┘               │
                ▼                                              │
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

## Dependencies

Core: numpy, pandas, scikit-learn, scipy, FastAPI, uvicorn, Jinja2, Plotly.
Tree models: LightGBM, XGBoost, CatBoost.
Neural models: PyTorch (LSTM, GRU, CNN, DLinear, NLinear, FITS, N-BEATS, N-HiTS, TiDE, TSMixer, TimeMixer, SparseTSF, PatchTST, iTransformer, Crossformer, TimesNet, TFT).
Classical baselines: statsforecast (AutoARIMA, AutoETS, AutoTheta).
Hyperparameter tuning: Optuna.

## Troubleshooting

**Benchmark fails with "not enough data."** The add-on needs at least `cv_folds × interval` worth of history for walk-forward CV. With the defaults (5 folds, 30-min interval) you need roughly 30 days of data in HA's recorder. If your recorder retention is the HA default of 10 days, either bump the retention in `configuration.yaml` or wait — the add-on will start working once enough history accumulates.

**Sensors don't appear after promoting to production.** Check the add-on log for HA REST errors. The most common cause is missing `homeassistant_api: true` (already set in `config.yaml`) or HA's auth token not being available — restarting the add-on usually resolves it. Sensors are published as `sensor.mlfl_<experiment_name>` plus `_lower_80` / `_upper_80` for the conformal bands.

**Forecasts look flat / collapse to zero overnight on solar targets.** v2.27.8 added a physics gate that forces solar forecasts to zero at night (gated by past `clear_sky_ghi`, with `pvlib`'s Ineichen model). The gate only applies if the experiment opts into solar features via `include_sun_elevation: true` or `include_clear_sky_irradiance: true`. Latitude / longitude are pulled from HA's own config — no need to set them in `mlfl.yaml`.

**Web UI shows "Database not available" on the Forecast Accuracy tab.** This usually means the SQLite forecast log was wiped (e.g. fresh install) — accuracy tracking populates as production forecasts run and their actuals arrive. Wait one or two cycles after promoting to production.

**ARM build is slow on first install.** First build takes 10-15 minutes on a Pi 5 because LightGBM, XGBoost, and PyTorch all need to compile native extensions for `aarch64`. Subsequent updates use the cached image.

For anything else, check the add-on log (visible in the Web UI's **Logs** tab or via the HA add-on page) — every phase tags its log lines (`[BENCH]`, `[MODEL]`, `[WEB]`, `[HA]`, `[PREP]`) so you can `grep` for the subsystem you're debugging.

## Development

Tests run locally without needing an HA instance:

```bash
cd ml-forecast-lab
pip install -r requirements.txt -r tests/requirements-dev.txt
pytest tests/                # 185 tests, ~10s
pytest tests/smoke/          # 61 smoke tests, ~2s — release gate
pytest tests/unit/           # 124 unit tests
```

Both suites are wired into GitHub Actions on every PR and main push (`tests.yml`). The smoke suite boots the FastAPI app against a tmp `mlfl.yaml` and walks the eight golden user flows (page renders, experiment CRUD, model param round-trip, settings persistence, promote/mode-toggle, analytics empty-states, HA picker fallback) without needing trained models — designed as a fast release gate that catches UI/API regressions before they ship.

Documentation guides for the internal modules live in [`docs/`](docs/): [`CONFIG_GUIDE.md`](docs/CONFIG_GUIDE.md), [`PREPROCESSING_GUIDE.md`](docs/PREPROCESSING_GUIDE.md), [`FEATURES_GUIDE.md`](docs/FEATURES_GUIDE.md).

## Licence

MIT

## Credits

Created by [Dr Paul W. Sweeney](https://github.com/psweens), University of Cambridge.
