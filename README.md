<div align="center">

<img src="logo.png" alt="ML Forecast Lab" width="180">

# ML Forecast Lab

**Multi-model machine learning forecasting for Home Assistant.**

Train, benchmark, and deploy time-series models for any HA sensor — with academic-standard evaluation built in.

[![Latest release][release-shield]][release-link]
[![Licence][licence-shield]][licence-link]
[![Tests][tests-shield]][tests-link]
[![Home Assistant add-on][ha-shield]][ha-link]
[![Architectures][arch-shield]][release-link]

[Install](#installation) · [Quick start](#quick-start) · [Configuration](#configuration) · [Documentation](#documentation) · [Troubleshooting](#troubleshooting)

</div>

---

## About

ML Forecast Lab does the heavy lifting once — trains every enabled backend on your sensor's history, ranks them on identical CV folds with a composite Demšar score across MAE / RMSE / MASE, and shows you which one wins. You promote the winner to **production**, and the add-on retrains it on schedule and publishes forecasts back to Home Assistant as companion sensors with calibrated 80% conformal bands.

The intended workflow is **benchmark once, run forever**. After the initial benchmark, production mode is set-and-forget; re-benchmark only when your sensor's behaviour drifts or you want to try newer model architectures.

## Installation

### Open in Home Assistant

[![Open your Home Assistant instance and show the add add-on repository dialog with this repository pre-filled.][openhainstall-shield]][openhainstall-link]

### Manual

1. Go to **Settings → Add-ons → Add-on store** in Home Assistant
2. Click **⋮** (top right) → **Repositories**
3. Add `https://github.com/psweens/ml-forecast-lab`
4. ML Forecast Lab will appear in the store — click **Install**

> [!NOTE]
> First build takes 10–15 minutes on a Raspberry Pi 5 — LightGBM, XGBoost, and PyTorch all compile native extensions for `aarch64`. Subsequent updates use the cached image.

**Supported architectures:** `aarch64` (Raspberry Pi 4/5), `amd64` (x86-64 servers), `armv7`.

## Screenshots

<!-- Replace these placeholders with real captures from your running instance. -->
<!-- The dashboard screenshot belongs at the top — that's the first thing users see in the store. -->

<!-- TODO: dashboard screenshot (overview of experiments + status) -->
<!-- TODO: experiment detail screenshot (Models tab with rank table) -->
<!-- TODO: forecast accuracy screenshot (per-horizon error chart) -->

## Quick start

End-to-end first-run flow once installed:

1. **Create `mlfl.yaml`** at `/addon_configs/ml_forecast_lab/mlfl.yaml` with one experiment targeting one HA sensor. The minimal viable config is just `name`, `target_entity`, and `models_enabled` — see [Configuration](#configuration) below.
2. **Start the add-on.** Open the Web UI (`http://homeassistant.local:5052` or via **Open Web UI** on the add-on page). Your experiment appears on the dashboard in **lab mode**.
3. **Run the benchmark.** Click into the experiment → **Run Pipeline**. Every enabled model trains on your sensor's history with walk-forward CV. Results land in the **Models** tab, ranked by composite Demšar score.
4. **Pick a model.** The top-ranked one is auto-selected; override from the **Models** tab if you want. Click **Promote to Production**.
5. **You're live.** The add-on retrains the chosen model on the configured schedule (default: 24h) and publishes `sensor.mlfl_<experiment_name>` companion sensors with `_lower_80` / `_upper_80` conformal bands. Forecast cycles (default: 30min) just run the cached model — sub-second inference, no retrain.
6. **Watch it.** The **Forecast Accuracy** tab compares each logged prediction against the actual once it arrives. Bias, per-horizon error, and trajectory drift are tracked automatically.

## Features

<table>
<tr>
<td valign="top" width="50%">

**24 model backends, benchmarked on identical splits.** Tree (LightGBM, XGBoost, CatBoost), recurrent (LSTM, GRU), convolutional (CNN, TimesNet), linear/MLP (DLinear, NLinear, TSMixer, TimeMixer, TiDE, SparseTSF), N-BEATS family (N-BEATS, N-HiTS), transformers (PatchTST, iTransformer, Crossformer, TFT), classical (AutoARIMA, AutoETS, AutoTheta), frequency-domain (FITS), and Seasonal Naive baseline. See [docs/MODEL_GUIDE.md](docs/MODEL_GUIDE.md) for picking the right ones for your data.

**Rigorous evaluation.** Walk-forward and sliding-window CV with embargo gaps. MAE, RMSE, MAPE, sMAPE, MASE, R², pinball loss, coverage. Composite Demšar (2006) ranking across folds for fair multi-metric comparison. Custom metrics via sandboxed Python expressions. Diebold-Mariano for pairwise model comparison.

**Hyperparameter tuning.** Bayesian optimisation (Optuna TPE) per-model with composite-rank trial selection. Default vs tuned holdout comparison plot.

</td>
<td valign="top" width="50%">

**Decoupled retrain and forecast cycles.** Retrain (default 24h) trains all enabled models from scratch and refreshes the cache. Forecast (default 30min) runs the cached model for sub-second inference. Each cycle is configurable per-experiment.

**Conformal uncertainty bands.** 80% prediction intervals published as companion `_upper_80` / `_lower_80` HA sensors, calibrated on held-out residuals.

**Forecast accuracy tracking.** Every prediction logged with model version; compared against actuals as ground truth arrives. Bias, trajectory drift, and per-horizon error in a three-layer diagnostic UI.

**Load subtract.** Subtract one HA sensor from another before modelling (e.g. net-of-solar demand) with a robustness layer for missing covariate data.

**Covariate analysis.** Automatically test every covariate combination across enabled models to discover which features genuinely improve forecasts.

</td>
</tr>
</table>

<details>
<summary><b>Auto-generated features and built-in physics</b></summary>

Temporal encodings (hour, day-of-week, month), cyclical sin/cos transforms, configurable lag features, rolling statistics, holiday indicators (per-country), and per-covariate aggregations / scalings.

For solar PV targets, optional deterministic features computed via `pvlib`'s Ineichen clear-sky model: sun elevation (degrees above horizon, negative at night) and theoretical maximum solar GHI (W/m²). Latitude / longitude are pulled from your HA installation's config — no extra setup.

</details>

<details>
<summary><b>Architecture</b></summary>

```
                     mlfl.yaml
                         │
                         ▼
         ┌──────────────┐    ┌──────────────────┐
         │ HA Interface │───▶│ History Database │
         │ (API client) │    │     (SQLite)     │
         └──────────────┘    └────────┬─────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │ Feature Engineer │
                            │ lags + temporal +│
                            │   covariates     │
                            └────────┬─────────┘
                                     │
                ┌────────────────────┼─────────────────────┐
                ▼                    ▼                     │
       ┌─────────────────┐   ┌─────────────────┐           │
       │ Cross-Validator │   │ Model Registry  │           │
       │ walk-fwd or SW  │   │ 24 backends     │           │
       └────────┬────────┘   │ tree+neural+cls │           │
                │            └─────────────────┘           │
                ▼                                          │
      ┌──────────────────┐         ┌──────────────────┐
      │   Benchmarker    │────────▶│   Model Cache    │
      │ Demšar ranking   │         │ (per experiment) │
      └────────┬─────────┘         └────────┬─────────┘
               │                            │
               ▼                            ▼
      ┌──────────────────┐         ┌──────────────────┐
      │      Web UI      │         │ Forecast Cycle   │
      │  (port 5052)     │         │ (every 30 min)   │
      └──────────────────┘         └────────┬─────────┘
                                            ▼
                                   ┌──────────────────┐
                                   │   HA Sensor      │
                                   │   Publisher      │
                                   └──────────────────┘
```

The retrain cycle (every ~24h) trains all enabled models from scratch and refreshes the cache. The forecast cycle (every ~30min) uses the cached model for fast inference without retraining.

</details>

## Configuration

Place `mlfl.yaml` at `/addon_configs/ml_forecast_lab/mlfl.yaml` (or `/config/mlfl.yaml` as fallback). Minimal example:

```yaml
timezone: "Europe/London"

experiments:
  - name: mixergy_demand
    target_entity: sensor.mixergy_demand_today
    mode: lab                  # 'lab' for benchmarking, 'production' to deploy
    source_is_cumulative: true
    reset_daily: true
    interval_minutes: 30
    days_history: 30
    future_periods: 48         # 48 steps × 30min = 24h horizon

    covariates:
      - entity: sensor.mixergy_current_charge
        role: lagged
        aggregation: mean
      - entity: sensor.external_temperature
        role: lagged
        aggregation: mean

    models_enabled:
      - lightgbm
      - xgboost
      - lstm
      - cnn

    cv_strategy: walk_forward
    cv_folds: 5
    cv_embargo_periods: 2
```

| Field | Description | Default |
|---|---|---|
| `target_entity` | HA sensor entity to forecast | *required* |
| `mode` | `lab` (benchmarking) or `production` (deploy best model) | `lab` |
| `source_is_cumulative` | `true` if sensor reports running totals | `false` |
| `reset_daily` | Cumulative sensor resets at midnight | `false` |
| `interval_minutes` | Resampling grid interval | `30` |
| `days_history` | Days of history to fetch for training | `30` |
| `future_periods` | Number of future steps to forecast | `48` |
| `forecast_every_minutes` | Inference cadence | `30` |
| `retrain_every_hours` | Retrain cadence | `24` |
| `models_enabled` | Backends to benchmark (24 available) | LightGBM, XGBoost |
| `cv_strategy` | `walk_forward` or `sliding_window` | `walk_forward` |
| `cv_folds` | Number of CV folds | `5` |

Full schema in [`docs/CONFIG_GUIDE.md`](docs/CONFIG_GUIDE.md).

## Documentation

- [`docs/MODEL_GUIDE.md`](docs/MODEL_GUIDE.md) — practical "which of the 24 backends should I enable?" with starter sets keyed to data volume, target shape, and Pi compute budget.
- [`docs/CONFIG_GUIDE.md`](docs/CONFIG_GUIDE.md) — full schema reference for `mlfl.yaml`.
- [`docs/PREPROCESSING_GUIDE.md`](docs/PREPROCESSING_GUIDE.md) — how cumulative-source handling, log transforms, and load subtract work.
- [`docs/FEATURES_GUIDE.md`](docs/FEATURES_GUIDE.md) — feature-engineering reference (lags, temporal encodings, covariate roles, rolling statistics).

## Troubleshooting

<details>
<summary><b>Benchmark fails with "not enough data"</b></summary>

The add-on needs at least `cv_folds × interval` worth of history for walk-forward CV. With the defaults (5 folds, 30-min interval) you need roughly 30 days of data in HA's recorder. If your recorder retention is the HA default of 10 days, either bump retention in `configuration.yaml` or wait — the add-on works once enough history accumulates.

</details>

<details>
<summary><b>Sensors don't appear after promoting to production</b></summary>

Check the add-on log for HA REST errors. The most common cause is missing `homeassistant_api: true` (already set in `config.yaml`) or HA's auth token not being available — restarting the add-on usually resolves it. Sensors are published as `sensor.mlfl_<experiment_name>` plus `_lower_80` / `_upper_80` for the conformal bands.

</details>

<details>
<summary><b>Forecasts look flat / collapse to zero overnight on solar targets</b></summary>

v2.27.8 added a physics gate that forces solar forecasts to zero at night (gated by past `clear_sky_ghi`, with `pvlib`'s Ineichen model). The gate only applies if the experiment opts into solar features via `include_sun_elevation: true` or `include_clear_sky_irradiance: true`. Latitude / longitude are pulled from HA's own config — no need to set them in `mlfl.yaml`.

</details>

<details>
<summary><b>Forecast Accuracy tab shows "Database not available"</b></summary>

This usually means the SQLite forecast log was wiped (e.g. fresh install) — accuracy tracking populates as production forecasts run and their actuals arrive. Wait one or two cycles after promoting to production.

</details>

<details>
<summary><b>ARM build is slow on first install</b></summary>

First build takes 10–15 minutes on a Pi 5 because LightGBM, XGBoost, and PyTorch all need to compile native extensions for `aarch64`. Subsequent updates use the cached image.

</details>

For anything else, check the add-on log (visible in the Web UI's **Logs** tab or via the HA add-on page) — every phase tags its log lines (`[BENCH]`, `[MODEL]`, `[WEB]`, `[HA]`, `[PREP]`) so you can `grep` for the subsystem you're debugging.

## Development

Tests run locally without an HA instance:

```bash
cd ml-forecast-lab
pip install -r requirements.txt -r tests/requirements-dev.txt
pytest tests/                # 185 tests, ~10s
pytest tests/smoke/          # 61 smoke tests, ~2s — release gate
pytest tests/unit/           # 124 unit tests
```

Both suites are wired into GitHub Actions on every PR and main push (`tests.yml`). The smoke suite boots the FastAPI app against a tmp `mlfl.yaml` and walks the eight golden user flows without needing trained models — designed as a fast release gate that catches UI/API regressions before they ship.

## Licence

[MIT](LICENSE) © Dr Paul W. Sweeney, University of Cambridge.

<!-- Badges -->
[release-shield]: https://img.shields.io/github/v/release/psweens/ml-forecast-lab?style=flat-square&color=41bdf5
[release-link]: https://github.com/psweens/ml-forecast-lab/releases/latest
[licence-shield]: https://img.shields.io/github/license/psweens/ml-forecast-lab?style=flat-square&color=41bdf5
[licence-link]: LICENSE
[tests-shield]: https://img.shields.io/github/actions/workflow/status/psweens/ml-forecast-lab/tests.yml?branch=main&style=flat-square&label=tests&color=41bdf5
[tests-link]: https://github.com/psweens/ml-forecast-lab/actions/workflows/tests.yml
[ha-shield]: https://img.shields.io/badge/Home%20Assistant-add--on-41bdf5?style=flat-square&logo=home-assistant&logoColor=white
[ha-link]: https://www.home-assistant.io/
[arch-shield]: https://img.shields.io/badge/arch-aarch64%20%7C%20amd64%20%7C%20armv7-41bdf5?style=flat-square
[openhainstall-shield]: https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg
[openhainstall-link]: https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fpsweens%2Fml-forecast-lab
