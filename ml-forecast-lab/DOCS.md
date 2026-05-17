# ML Forecast Lab — Documentation

This is the full reference for users running the add-on. For a quick install and a first forecast, see [README.md](README.md). For picking which of the 24 backends to enable, see [`docs/MODEL_GUIDE.md`](https://github.com/psweens/ml-forecast-lab/blob/main/docs/MODEL_GUIDE.md) in the repository.

## Contents

1. [Where things live](#where-things-live)
2. [Configuration reference](#configuration-reference)
3. [Published Home Assistant sensors](#published-home-assistant-sensors)
4. [Web UI tour](#web-ui-tour)
5. [Operations](#operations)
6. [Troubleshooting](#troubleshooting)
7. [Upgrading and version compatibility](#upgrading-and-version-compatibility)

---

## Where things live

| Path on the HA host | Contents | Edit? |
|---|---|---|
| `/addon_configs/ml_forecast_lab/mlfl.yaml` | **Your experiment configuration.** Canonical path. | Yes — this is the one file you edit. |
| `/config/mlfl.yaml` | Legacy fallback path; still loaded if the canonical path is missing. | Yes, but prefer the canonical path. |
| `/share/ml_forecast_lab/` *(or `/data/ml_forecast_lab/` inside the container)* | Model cache, SQLite history, forecast log, retrain rollbacks. Managed by the add-on. | No. |

The add-on searches these in order: explicit `--config-path` (development only) → `/addon_configs/ml_forecast_lab/mlfl.yaml` → `/config/mlfl.yaml` → the bundled example. If nothing is found a stub config is created.

---

## Configuration reference

`mlfl.yaml` has two levels: **global** settings at the top and a list of **experiments** underneath. Every field has a default; the only required keys per experiment are `name` and `target_entity`.

### Global settings

| Key | Type | Default | What it does |
|---|---|---|---|
| `timezone` | string | `UTC` | IANA timezone for temporal features (hour, day-of-week, holiday). Set it to your home's timezone. |
| `forecast_every_minutes` | int | `30` | How often the production cycle runs inference and publishes sensor updates. |
| `retrain_every_hours` | float | `24.0` | How often the production model retrains from scratch on fresh history. |
| `update_every_minutes` | int | `5` | Legacy alias for `forecast_every_minutes`. Kept for backwards compatibility. |
| `cpu_cores` | int | `0` (= all) | Caps `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `torch.set_num_threads` so training doesn't saturate every core. |
| `nice_priority` | int 0–19 | `10` | Process `nice` value for training. Higher numbers = lower priority, so the Pi stays responsive during a benchmark. |
| `model_overrides` | mapping | `{}` | Global per-model hyperparameter overrides; keys are model registry names (e.g. `lightgbm`, `lstm`). |
| `experiments` | list | required | One entry per sensor you want to forecast. |

### Experiment essentials

| Key | Type | Default | What it does |
|---|---|---|---|
| `name` | string | **required** | Unique identifier. Must match `[a-z][a-z0-9_]{0,63}`. Used in entity names and on disk. |
| `target_entity` | string | **required** | HA sensor entity to forecast. |
| `mode` | `lab` \| `production` | `lab` | `lab` benchmarks every enabled model and shows results in the UI. `production` trains only the promoted model and publishes sensors. |
| `interval_minutes` | int | `30` | Resampling grid. Must match what your sensor can reasonably support after gap-filling. |
| `days_history` | int | `14` | Days of HA recorder history to fetch for training. |
| `future_periods` | int | `48` | Number of future steps to forecast (e.g. 48 × 30 min = 24 h horizon). |
| `forecast_every_minutes` | int | inherits global | Per-experiment override of the inference cadence. |
| `retrain_every_hours` | float | inherits global | Per-experiment override of the retrain cadence. |
| `country` | ISO code | unset | Two-letter country code for holiday features (`GB`, `US`, `DE`, …). Unset = no holiday feature. |
| `units` | string | `""` | Target units (`W`, `kWh`, `%`, …). Shown in the UI and on published sensors. |
| `output_units` | string | unset | Optional unit conversion at publish time (e.g. train on Wh, publish kWh). |
| `max_age` | int (days) | `365` | Cap on rows kept in the SQLite actuals cache. The cache is always-on (v2.33.1+); older rows are pruned each cycle. |
| `publish_prefix` | string | `mlfl_` | Prefix for every companion sensor. Change only if you have a naming clash. |
| `publish_name` | string | inherits `name` | Override the experiment's name when constructing companion sensor IDs. |

### Cumulative-source handling

Many HA sensors report running totals (`*_today` energy, daily heat). Configure these explicitly so the add-on can extract per-interval increments correctly.

| Key | Type | Default | What it does |
|---|---|---|---|
| `source_is_cumulative` | bool | `false` | `true` if the target sensor reports a running total. |
| `reset_daily` | bool | `false` | `true` if the cumulative sensor resets at local midnight. |
| `max_increment` | float | unset (= auto) | Caps a single-interval delta. Useful to bound abnormal jumps (e.g. EV plug-ins). If unset, the 95th percentile of observed increments is used. |

### Covariates

External signals that improve the forecast. Add as many as you want.

```yaml
covariates:
  - entity: sensor.outside_temperature
    role: lagged
    aggregation: mean

  - entity: weather.home
    role: future
    future_attribute: forecast
    future_value_key: temperature
    aggregation: mean
```

| Key | Type | Default | What it does |
|---|---|---|---|
| `entity` | string | **required** | HA sensor or weather entity. |
| `role` | `lagged` \| `future` | `lagged` | `lagged` = historical only (used as a lag feature). `future` = the entity exposes a known-future forecast attribute, available at every horizon step. |
| `future_attribute` | string | `forecast` | For `role: future`: the entity attribute carrying the known-future series (`forecast` for Met.no, `detailedForecast` for Solcast). |
| `future_value_key` | string | unset | For `role: future`: the key inside each forecast entry that contains the value (`temperature`, `pv_estimate`, …). If unset, common keys are tried in order. |
| `scale` | float | unset | Multiplicative pre-scaling (e.g. percent → fraction: `0.01`). |
| `transform` | `log` \| `sqrt` \| `box_cox` | unset | Per-covariate transform before model input. |
| `aggregation` | `mean` \| `sum` \| `max` \| `min` \| `last` | `mean` | Resampling method when aligning to `interval_minutes`. |
| `is_binary` | bool | `false` | Marks 0/1 indicators (holiday flag, occupancy). Disables the rolling-statistic features for that signal. |

### Load subtract

Subtract one or more sensors from the target before training. Use when you want to model the *baseline* signal without contributions from a known disturbance (e.g. household load excluding EV charging and solar-divert dumps).

```yaml
load_subtract:
  - entity_id: sensor.ev_energy_today
    source: cumulative_daily
    on_missing: zero
    scale: 1.0
    max_fraction_of_load: 0.8
```

| Key | Type | Default | What it does |
|---|---|---|---|
| `entity_id` | string | **required** | HA sensor to subtract. |
| `source` | `cumulative_daily` \| `cumulative_monotonic` \| `interval` \| `auto` | `auto` | Cumulative semantics of the subtract sensor. Be explicit when the subtract sensor's semantics differ from the target. |
| `on_missing` | `zero` \| `drop` \| `error` | `zero` | What to do with gap / `unavailable` rows. `zero` is common for EVs (missing usually means "wasn't on"). `drop` is safer if missing might mean "we don't know". `error` makes data-pipeline bugs loud. |
| `scale` | float | unset | Multiplier — use to fix unit mismatches (e.g. Wh → kWh: `0.001`). |
| `max_fraction_of_load` | float | `1.0` | Per-row ceiling on `subtract / load`. Rows exceeding it count as violations. |
| `max_fraction_violation_pct` | float | `5.0` | Maximum percentage of rows allowed to violate `max_fraction_of_load` before the load-subtract step raises. Acts as a fail-fast guard for unit bugs. |

**Note:** the older `subtract: [entity_id]` field is a deprecated stub — it loads without error but does not affect training. Migrate any existing config to `load_subtract`.

### Solar-physics features

Two zero-cost deterministic covariates for solar PV (or any sun-driven target). Latitude / longitude are pulled from your HA installation's config — you don't need to set them.

| Key | Type | Default | What it does |
|---|---|---|---|
| `include_sun_elevation` | bool | `false` | Adds the sun's angle above the horizon (degrees, negative at night) as a covariate. Strong physical signal for diurnal patterns. |
| `include_clear_sky_irradiance` | bool | `false` | Adds the theoretical clear-sky GHI (W/m²) from `pvlib`'s Ineichen model. Zero at night. Turns "predict solar generation" into "predict cloud-cover attenuation". |

When `include_clear_sky_irradiance` is on, the add-on also gates production forecasts to zero at night based on past `clear_sky_ghi` — preventing the "small positive forecast at 3 a.m." pattern.

### Cross-validation

| Key | Type | Default | What it does |
|---|---|---|---|
| `cv_strategy` | `walk_forward` \| `sliding_window` | `walk_forward` | Walk-forward expands the training window each fold (closer to real-world retraining). Sliding-window keeps the training window fixed. |
| `cv_folds` | int 2–20 | `5` | Number of folds. With short histories, raising this leaves too few rows per test slice. |
| `cv_embargo_periods` | int | `2` | Gap (in periods) between train and test in each fold. Prevents rolling-window features from leaking across the boundary. |

### Models and training

| Key | Type | Default | What it does |
|---|---|---|---|
| `models_enabled` | list | `[lightgbm, xgboost, lstm, cnn]` | Backends to train. Names match the registry slugs in `docs/MODEL_GUIDE.md` (`seasonal_naive`, `lightgbm`, `xgboost`, `catboost`, `lstm`, `gru`, `cnn`, `dlinear`, `nlinear`, `tsmixer`, `timemixer`, `tide`, `sparsetsf`, `fits`, `nbeats`, `nhits`, `patchtst`, `itransformer`, `crossformer`, `timesnet`, `tft`, `arima`, `ets`, `theta`). |
| `model_params` | mapping | `{}` | Per-experiment hyperparameter overrides; keys are model names. Takes precedence over global `model_overrides`. Easier path: tune in the UI and use **Apply Tuned Params, Promote & Retrain**. |
| `loss_fn` | `mse` \| `mae` \| `huber` \| `tweedie` | `huber` | Training loss for neural models. `huber` is quadratic near zero and linear in the tails, which is right for the spiky near-zero HA signals most users forecast. `tweedie` is honoured only by tree backends (LightGBM / XGBoost / CatBoost). |
| `optimiser` | `adam` \| `adamw` | `adamw` | Neural optimiser. `adamw` (decoupled weight decay) matches every published time-series transformer paper; `adam` is the classic. Ignored by tree models. |
| `output_activation` | `auto` \| `linear` \| `softplus` \| `relu` \| `exp` \| `sigmoid` \| `zscore` | `auto` | Output-head activation for PyTorch neural backends. `auto` picks `softplus` for cumulative sources and `linear` otherwise (and `zscore` for LSTM). Override for niche cases — `sigmoid` for hard-bounded quantities (battery SOC, humidity %), `linear` for signed targets (temperature delta). Tree models ignore this. |
| `use_revin` | bool | `true` | Reversible Instance Normalisation. Per-window normalisation at the network's input + reversal at the output. Matches the published transformer / MLP-mixer reference implementations. Tree models, N-BEATS, and N-HiTS ignore this. |
| `daily_loss_weight` | float | `0.0` | Weight λ for an auxiliary cumulative-trajectory loss term during neural training. `0.0` disables it (interval loss only). Try `0.1–1.0` if cumulative-curve shape is the metric your downstream automation cares about. |
| `recency_half_life_days` | float | `0.0` | Exponential recency weighting for training samples. `0` = uniform (default; the right choice for stable household sensors). Set to e.g. `7` if your series recently entered a new regime (heat pump install, schedule change). |
| `quantiles` | list[float] | `[]` | Multi-quantile training. Empty = point forecast wrapped in a conformal band (recommended). Non-empty (e.g. `[0.1, 0.5, 0.9]`) routes the DLinear backend through a pinball-loss head; other backends still use the point + conformal path. |
| `future_covariate_features` | list[string] | `[]` | Names of feature columns that the TiDE backend should route through its known-future temporal-decoder path. Calendar features and externally-forecast weather. Do not include lags of the target. |

### Pre-processing pipeline

| Key | Type | Default | What it does |
|---|---|---|---|
| `gap_handling` | `interpolate` \| `ffill` \| `mask` | `interpolate` | What to do with gaps after resampling. `interpolate` linear-fills short gaps and leaves long ones as NaN. `ffill` propagates the last value (legacy). `mask` leaves every gap as NaN so the row is dropped downstream. |
| `gap_max_minutes` | int | `90` | Maximum gap that `interpolate` will bridge. Longer gaps fall through to NaN. |
| `outlier_method` | `quantile` \| `mad` \| `off` | `quantile` | Outlier-clipping strategy. `quantile` clips the upper tail; `mad` uses the Iglewicz-Hoaglin robust bound (more forgiving on heavy-tailed legitimate data like rainfall); `off` disables clipping. |
| `outlier_quantile` | float | `0.999` | Upper-tail quantile for `outlier_method: quantile`. Lower this if your target has a clean upper bound. |
| `outlier_lower` | `auto` \| `zero` \| `symmetric` \| `off` | `auto` | Lower-bound rule. `auto` clips at zero for cumulative sources, symmetric quantile otherwise. `zero` for non-negative quantities, `symmetric` for two-sided signals. |
| `log_transform` | bool | `false` | Apply log to the target before modelling. Useful when the target spans orders of magnitude. |

### Metrics and model ranking

| Key | Type | Default | What it does |
|---|---|---|---|
| `metrics` | list | `[mae, rmse, mase, seasonal_mase]` | Metrics to compute during benchmarking. Available: `mae`, `rmse`, `mape`, `smape`, `mase`, `seasonal_mase`, `r2`, `pinball`, `coverage`. |
| `custom_metrics` | mapping | unset | `{name: 'python expression'}` evaluated in a sandbox (`asteval`) with `y_true`, `y_pred`, `np` in scope. |
| `production_metric` | string | `seasonal_mase` | Metric used to auto-select the best model when `production_model` is unset. `seasonal_mase` (scaled by the same-time-yesterday baseline) is the right comparison for daily-seasonal HA sensors. |
| `production_model` | string | unset | Pin a specific model name. Unset = auto-pick by `production_metric`. |
| `selected_model` | string | unset | Which model the Results-tab UI highlights by default. The `/select-model` click in the UI persists here. |

### Conformal prediction and stability

| Key | Type | Default | What it does |
|---|---|---|---|
| `conformal_coverage` | float in (0, 1) | `0.8` | Nominal coverage of the prediction interval. Default 0.8 publishes `_upper_80` / `_lower_80` companion sensors. Raise to 0.9 if downstream automations need wider safety margins; lower to 0.5 for diagnostic plots. |
| `stability_focus` | `per_moment` \| `daily_total` | `per_moment` | Which stability metric drives the Forecast Accuracy verdict chip. `per_moment` is right when downstream consumers care about *when* something happens (HVAC pre-heat, battery dispatch). `daily_total` (cumulative sources only) is right when only the daily integral matters. |
| `clear_forecast_log_on_retrain` | bool | `true` | Whether to prune forecast-log rows older than the latest retrain when a champion is promoted. Keeps stability metrics honest — set `false` only if you want to preserve full history for offline analysis. |

---

## Published Home Assistant sensors

When an experiment is in `mode: production`, the add-on publishes the following sensors. The placeholder `<name>` is the experiment's `publish_name` (or `name` if unset) prefixed by `publish_prefix`.

| Entity | Value | Notes |
|---|---|---|
| `sensor.mlfl_<name>_forecast` | The next interval's forecast value. | Attributes include the full future curve as a list of `{datetime, value}` pairs. |
| `sensor.mlfl_<name>_interval` | The next interval's value, expressed as a per-interval increment. | Published only when `source_is_cumulative` is true (otherwise identical to `_forecast`). |
| `sensor.mlfl_<name>_cumulative` | The integrated forecast curve. Resets at local midnight when `source_is_cumulative` and `reset_daily` are both true; otherwise a `cumsum` anchored at zero. | Useful for daily-budget automations (EV planning, hot-water tank pre-heat). |
| `sensor.mlfl_<name>_upper_<pct>` | Upper conformal band at the `<pct>` coverage level (default `80`). | Renamed to match `conformal_coverage` — e.g. `_upper_90` if you set `0.9`. Appears once enough residuals have been calibrated; cold-start may take ~10 forecast cycles. |
| `sensor.mlfl_<name>_lower_<pct>` | Lower conformal band. | As above. |
| `sensor.mlfl_<name>_forecast_accuracy` | Running accuracy summary (bias, MAE, coverage). | Updated whenever a logged prediction's actual arrives. |
| `sensor.mlfl_<name>_last_benchmark` | ISO timestamp of the most recent benchmark completion. | `device_class: timestamp`. Attributes include outcome, duration, winner, and a truncated error string when the cycle failed — convenient triggers for HA automations. |
| `sensor.mlfl_<name>_last_retrain` | ISO timestamp of the most recent retrain. | Same shape as `_last_benchmark`. |

---

## Web UI tour

Open the UI via the add-on's **Open Web UI** button (HA ingress).

- **Dashboard.** One card per experiment. Click into a card to drill in. The grid refreshes via HTMX without losing scroll position or expanded panels.
- **Experiment page tabs.**
  - **Settings.** All knobs above as a form. Editing here writes back to `mlfl.yaml` atomically. Includes a pre-flight **Data sanity check** that reports rows fetched vs expected, biggest gap, recorder freshness, missing-value rate, and (for cumulative sensors) max-increment hits — run this before a benchmark to catch a 14-day flatline before you spend an hour training.
  - **Models.** Per-model toggle, default vs tuned MAE, **Promote to Production** button. Quick-preset chips (Fast / Balanced / Thorough) flip groups of toggles to match the starter sets in `docs/MODEL_GUIDE.md`.
  - **Tuning.** Bayesian optimisation (Optuna TPE) per model with default vs tuned holdout comparison. **Tune All Enabled** sweeps every enabled backend sequentially.
  - **Results.** Composite Demšar rank across MAE / RMSE / MASE, the always-on "vs Seasonal Naive" skill chip, a pairwise model-comparison matrix (paired-t test on per-fold MAE), the training-window vs test-window drift verdict (PSI), and a "Compare with previous run" strip — the last five benchmarks are retained and diff-able.
  - **Forecast Accuracy.** Three-layer diagnostic: verdict chip, per-horizon error chart, retrain-history chips (filter the chart to a specific `(model_name, model_version)` cohort). Conformal-band calibration countdown surfaces "Calibrating · N of 10 residuals" rather than a silent blank.
  - **Predictions** and **Covariate Analysis.** Forecast-trace overlay and an automatic search across covariate combinations to identify which signals genuinely improve forecasts.
- **System page.** CPU-core / nice-priority controls (actually applied) and a global "Run all benchmarks" trigger.

---

## Operations

### Logs

The add-on log is visible from the HA add-on page or via the Web UI's **Logs** tab. Every line carries a short phase tag in square brackets — useful for `grep`-ing:

| Tag | Subsystem |
|---|---|
| `[APP]` | Top-level lifecycle. |
| `[CFG]` | Config loading and atomic writes. |
| `[HA]` | Home Assistant REST / WebSocket. |
| `[DB]` | SQLite cache and forecast log. |
| `[PREP]` | Pre-processing pipeline (cumulative → interval, gap fill, outlier clip). |
| `[FEAT]` | Feature engineering (lags, temporal encodings, rolling stats). |
| `[COV]` | Covariate resolution (history fetch, future-attribute parsing). |
| `[SOLAR]` | Sun-elevation and clear-sky physics. |
| `[MODEL]` | Per-backend training and prediction. |
| `[BENCH]` | Benchmark orchestration. |
| `[TRAIN]` | Production retrain cycles. |
| `[PUB]` | Sensor publication back to HA. |
| `[WEB]` | FastAPI request handling. |

**Persistent log files** (v2.37+) live in `/data/ml_forecast_lab/logs/` inside the container:

* `mlfl.log` — size-rotated, 10 MB × 5 backups (≈50 MB total). Always tails the most recent activity. Open it with the **File editor** add-on or via Samba.
* `mlfl-daily.log` + `mlfl-daily.log.YYYY-MM-DD` — one file per UTC day, kept for 14 days. Easier to grep against a specific date when investigating an issue.
* Both files use the same `[PHASE] level [module] message` format as the console log, so they're directly greppable.
* Suppress the daily archive by setting `MLFL_DAILY_LOG_KEEP=0` in the add-on environment (the size-rotated log keeps working).
* Disk footprint is bounded — ~50 MB live + (typical INFO-level daily volume × 14 days).

### Backing up trained models

`/share/ml_forecast_lab/` contains the model cache, SQLite database, and forecast log. The HA **Backups** add-on already picks this up if you have backups configured for add-on data. To export by hand: download the directory through the Samba / SSH add-on.

### Rolling back a bad retrain

Every retrain archives the previous champion under `<model_dir>/previous/` before overwriting. From the production experiment header, click **Roll back** — the current model and the previous model swap atomically and the live cache rehydrates. Single-generation cap (one rollback deep) keeps SD-card writes bounded.

### Resetting an experiment

To wipe accuracy history and start fresh:

1. From the experiment header, click **Delete experiment**, or delete the entry from `mlfl.yaml`.
2. Restart the add-on. The next benchmark starts from zero.

The SQLite database survives the experiment deletion — only the per-experiment cached model and forecast-log rows are removed.

### Updating the add-on

Updates are delivered via the HA add-on store like any other add-on. Read the [CHANGELOG](CHANGELOG.md) before upgrading — the changelog calls out behaviour changes that require config edits.

---

## Troubleshooting

### "Not enough data" when starting a benchmark

The add-on needs roughly `cv_folds × interval_minutes × test_size` worth of history per fold. With defaults (5 folds, 30-min interval), aim for **at least 30 days** in HA's recorder. If your recorder retention is the HA default of 10 days, either raise `recorder.purge_keep_days` in `configuration.yaml` or wait — the add-on will pick up once enough history has accumulated.

### Companion sensors don't appear after promotion

Check the add-on log for `[HA]` errors. The most common causes:

- `homeassistant_api: true` got disabled in `config.yaml` (it shouldn't have been; the add-on declares it). Restart the add-on to refresh the token.
- The experiment is still in `mode: lab`. Promotion in the UI also flips the mode; if you changed YAML by hand, make sure `mode: production` is set.
- A sensor name collision. The first publish logs `Publishing forecast for <name>: base=sensor.mlfl_<name>` — check the actual entity ID and search HA's developer tools.

### Conformal bands missing from the published sensors

`_upper_<pct>` / `_lower_<pct>` only publish once enough residuals have been calibrated (default 10 deployed predictions whose actuals have arrived). Until then the Forecast Accuracy tab shows a countdown ("Calibrating · 4 of 10 residuals · ~3 h to bands"). After ~3 forecast cycles in production the bands should appear.

If they vanished after a retrain: v2.24.0 had a regression where the conformal query filtered too strictly to the fresh model version. Upgrade to v2.27 or later, which falls back to pooled residuals during cold-start.

### Solar forecasts go positive at night

You're not opted into the physics gate. Set:

```yaml
include_sun_elevation: true
include_clear_sky_irradiance: true
```

on the solar experiment. The gate zeros production forecasts whenever past `clear_sky_ghi` indicates night — and the new covariates also improve daytime accuracy substantially.

### "Database not available" on the Forecast Accuracy tab

The SQLite forecast log is empty — either a fresh install, or `clear_forecast_log_on_retrain` just pruned everything. Wait one or two production cycles after a promote / retrain.

### Forecasts collapse to a flat line

Three usual causes:

1. The target has a long stretch of identical values that the model is overfitting to. The Settings-tab Data sanity check surfaces zero-run length and missing-value rate.
2. The covariates with `role: future` are returning `unavailable`. The `[COV]` log lines show how each covariate resolved per cycle. A weather entity that briefly stops exposing `forecast` will starve future covariates of values and the model defaults to a flat extrapolation.
3. `recency_half_life_days` is too small for your data. Try setting it back to `0` (uniform weighting) on stable targets.

### Out-of-memory on the Pi

Disable the heaviest backends (`tft`, `crossformer`, `timesnet`, `patchtst`) and rerun the benchmark. The Quick-preset **Fast** chip on the Models tab does this in one click. If OOMs persist, lower `cpu_cores` to `2` — concurrent backend training is more memory-hungry than CPU-bound.

### Add-on takes 10+ minutes to start the first time

Expected. LightGBM, XGBoost, and PyTorch all compile native extensions for `aarch64` on first install. Subsequent updates use the cached image and start in seconds.

### Custom-metric expression errors

`custom_metrics` is evaluated by `asteval` (a Python-subset sandbox). Available names are `y_true`, `y_pred`, and `np`. Errors land in the `[BENCH]` log lines with the offending expression quoted. Common gotchas: passing a list comprehension (asteval doesn't support all forms), or calling functions outside the allow-list.

For anything else, the `[BENCH]`, `[MODEL]`, `[WEB]`, `[HA]`, `[PREP]`, `[FEAT]`, `[COV]`, `[CFG]`, `[DB]` tags should let you narrow the failure to a subsystem in a few `grep`s.

---

## Upgrading and version compatibility

- **Backwards compatibility.** `mlfl.yaml` is auto-migrated where possible: deprecated fields (`horizons_minutes`) are silently stripped, and the legacy `subtract: [entity_id]` list is loaded but ignored with a deprecation warning in the log. Migrate to `load_subtract` with explicit `source` / `on_missing` per sensor.
- **Default changes between versions.** Several knobs have changed their defaults in recent releases — most notably `production_metric` (`mae` → `seasonal_mase`), `outlier_quantile` (`0.995` → `0.999`), and `recency_half_life_days` (`7` → `0`). The CHANGELOG calls these out per version.
- **2.30.0 ingress-only.** Direct port 5052 exposure was removed. The web UI is now reached exclusively through HA's authenticated ingress proxy. If you were proxying directly to the port, switch to ingress.

See [CHANGELOG.md](CHANGELOG.md) for the full release history.
