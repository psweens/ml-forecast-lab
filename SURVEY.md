# SURVEY.md — ml-forecast-lab

Product/UX reconnaissance ahead of an improvements proposal targeted at HA power users on a Raspberry Pi 5. British English throughout. File:line references are absolute paths from the repo root.

The earlier security-audit framing of this file has been replaced; for that material, see git history.

---

## 1. Repository tree (one-line role per file)

```
ml-forecast-lab/                                          ← HA add-on repository root
├── repository.yaml                                       ← HA add-on repository manifest
├── README.md                                             ← user-facing overview, install + quick start
├── LICENSE                                               ← MIT (Dr Paul W. Sweeney)
├── icon.png, logo.png                                    ← add-on artwork (also copied into web/static/)
├── AUDIT_PROMPT.md                                       ← scratch brief for a security audit pass
├── docs/
│   ├── CONFIG_GUIDE.md                                   ← user docs: full mlfl.yaml schema reference
│   ├── FEATURES_GUIDE.md                                 ← user docs: feature engineering
│   ├── MODEL_GUIDE.md                                    ← user docs: which of the 24 backends to enable
│   └── PREPROCESSING_GUIDE.md                            ← user docs: cumulative/log/load-subtract
├── .github/workflows/
│   ├── tests.yml                                         ← CI: pytest unit + smoke
│   ├── validate.yml                                      ← CI: HA add-on lint
│   └── release.yml                                       ← CI: tag → multi-arch image publish
└── ml-forecast-lab/                                      ← the add-on itself
    ├── config.yaml                                       ← HA manifest: slug=ml_forecast_lab, ingress:5052, homeassistant_api:true
    ├── build.yaml                                        ← base image map per arch (ubuntu-base 9.0.5)
    ├── Dockerfile                                        ← two-stage builder venv → runtime; apt + pip only
    ├── requirements.txt                                  ← Python deps (CPU-only torch, lightgbm, optuna, pvlib, fastapi…)
    ├── mlfl.yaml                                         ← bundled example user config, copied to /addon_configs on first boot
    ├── CHANGELOG.md                                      ← long, detailed (~2 900 lines; notes Hailo removal)
    ├── MODELS_CREATED.txt                                ← scratch notes from earlier model-addition work
    ├── translations/en.yaml                              ← HA add-on store UI strings
    ├── rootfs/etc/s6-overlay/s6-rc.d/
    │   ├── init-mlforecastlab/{run,up,type}              ← s6 longrun service; bashio runs python3 -m ml_forecast_lab
    │   └── user/contents.d/init-mlforecastlab            ← marker enabling the service
    ├── ml_forecast_lab/                                  ← Python package
    │   ├── __init__.py                                   ← re-exports public API; sets __version__
    │   ├── __main__.py                                   ← logging setup, asyncio.run(main); fallback stub FastAPI on import failure
    │   ├── main.py                  (5241 ln)            ← MLForecastLabApp orchestrator: schedules forecast/retrain/tuning/cov-analysis cycles
    │   ├── config.py                (1185 ln)            ← dataclasses AppConfig/ExperimentCfg/CovariateCfg/SubtractCfg + YAML I/O
    │   ├── db.py                    (1823 ln)            ← HistoryDB (SQLite WAL): history, forecast_log, benchmark_results, schema_versions
    │   ├── ha_interface.py          (424 ln)             ← async aiohttp client for HA REST API (history, set_state, config)
    │   ├── covariates.py            (201 ln)             ← CovariateResolver: covariate history+future fetch, binary auto-detect
    │   ├── preprocessing.py         (892 ln)             ← cumulative→interval, resample, clip, log, load-subtract
    │   ├── features.py              (544 ln)             ← lags (with night-time GHI gate), rolling stats, cyclic encodings, holidays
    │   ├── solar_physics.py         (97 ln)              ← pvlib Ineichen clear-sky GHI + apparent solar elevation
    │   ├── training_events.py       (123 ln)             ← thread↔asyncio TrainingEventBus for the SSE training stream
    │   ├── dashboard.py             (250 ln)             ← generates a Lovelace YAML (ApexCharts cards) into /addon_configs
    │   ├── web/
    │   │   ├── __init__.py
    │   │   ├── app.py               (3229 ln)            ← FastAPI app (create_app), routes, AppState, SSE, MODEL_CATALOG, MODEL_PARAM_SCHEMA
    │   │   ├── static/{style.css,htmx.min.js,plotly-basic.min.js,icon.png}  ← vendored assets, no build step
    │   │   └── templates/
    │   │       ├── base.html        (141 ln)             ← shell: nav, toast, confirm modal, info-tip JS
    │   │       ├── dashboard.html   (416 ln)             ← experiment cards + "New experiment" modal
    │   │       ├── experiment.html  (3736 ln)            ← the bulk of the UI: 10 in-page tabs (single template)
    │   │       ├── training.html    (629 ln)             ← stand-alone /training redirect target with global training feed
    │   │       ├── models.html      (244 ln)             ← per-model hyperparameter config cards
    │   │       ├── logs.html        (129 ln)             ← live log tail (3 s poll) with level filter
    │   │       └── system.html      (224 ln)             ← system info, global settings, experiment summary
    │   ├── benchmark/
    │   │   ├── __init__.py
    │   │   ├── runner.py            (940 ln)             ← BenchmarkRunner: walk-forward / sliding-window CV, composite Demšar rank, daily metrics
    │   │   ├── metrics.py           (549 ln)             ← MAE/RMSE/MAPE/sMAPE/MASE/R² + asteval-sandboxed custom metrics
    │   │   └── comparison.py        (336 ln)             ← Diebold-Mariano + cross-model comparison utilities
    │   └── models/
    │       ├── __init__.py                               ← optional dynamic import of every backend
    │       ├── registry.py                               ← ModelRegistry (no entry-points)
    │       ├── base.py              (916 ln)             ← ForecastModel ABC, RevIN, composite horizon loss, optimiser builder
    │       └── {lightgbm,xgboost,catboost,lstm,gru,cnn,tft,tide,tsmixer,timemixer,timesnet,
    │            patchtst,itransformer,crossformer,nhits,nbeats,nlinear,dlinear,sparsetsf,
    │            fits,seasonal_naive,statsforecast}_backend.py
    └── tests/
        ├── conftest.py, __init__.py, requirements-dev.txt
        ├── dryrun_pipeline.py                            ← manual smoke harness
        ├── unit/test_{config,db,features,preprocessing,benchmark,models,forecast_analytics,load_subtract}.py
        └── smoke/test_{pages,settings,harness,promote_flow,tuning_guard,model_config,
                         experiment_lifecycle,analytics_empty_state,ha_entities}.py
```

Python LOC ≈ 28 000. Frontend is server-rendered Jinja + HTMX + Plotly with **no build step or JS bundler**; the only sizeable vendored asset is `plotly-basic.min.js`.

---

## 2. Stated purpose

From `README.md` (lines 5–28) and `mlfl.yaml` headers (lines 1–16):

> *"Multi-model machine learning forecasting for Home Assistant. Train, benchmark, and deploy time-series models for any HA sensor — with academic-standard evaluation built in."*

The intended workflow is **"benchmark once, run forever"** (README:27):

1. Define an experiment in `mlfl.yaml` (or the New Experiment modal) targeting one HA sensor.
2. Run a benchmark in **lab mode**: every enabled backend trains on the sensor's history, CV-ranked by a Demšar (2006) composite over MAE / RMSE / MASE.
3. Promote the winner to **production mode**. The add-on retrains on a schedule (24 h default) and publishes companion forecast sensors back to HA every cycle (30 min default), with conformal 80 % uncertainty bands.
4. The Forecast Accuracy tab continuously logs predictions and compares them with actuals as they arrive, surfacing bias, lead-time error, calibration, and run-to-run stability.

The product positioning, per the README, is "an academic-standard evaluation harness wrapped in an HA add-on" — the user is not expected to know ML, only to be a competent HA power user.

---

## 3. Current feature inventory — what a user can do today, end to end

Grouped by user goal, with the entry points the user actually clicks.

### 3a. Set up a forecast

- **Add the repository, install the add-on** (README:32–45). First build is 10–15 min on a Pi 5 — LightGBM / XGBoost / torch wheels exist for ARM64 but some compile.
- **Create an experiment two ways:**
  - Edit `/addon_configs/<hash>_ml_forecast_lab/mlfl.yaml` directly. Bundled example is `ml-forecast-lab/mlfl.yaml`.
  - Click **+ New Experiment** on the dashboard (`web/templates/dashboard.html:30, 183-224`). The modal takes a friendly name, target entity (with debounced HA entity search), cumulative and reset-daily toggles, and posts to `/api/experiments/create` (`web/app.py:1683`). Anything else — covariates, models, CV — defaults until the user opens the new experiment's Settings tab.
- **Configure the experiment** in the per-experiment Settings tab (`experiment.html:84-404`): target shape (cumulative, reset-daily, max-increment), data window (days_history, interval, future_periods), retrain/forecast cadence overrides, log transform, CV strategy/folds, recency half-life, production metric (MAE/RMSE/MASE), neural loss/optimiser/output activation, daily cumulative-loss toggle, solar physics toggles (sun elevation, clear-sky GHI), per-experiment covariates (with HA entity search), per-experiment load-subtract list. All edits autosave via `/api/experiment-settings`.
- **Enable/disable model backends** per experiment from the Models tab (`experiment.html:407-451`) with toggle switches; per-tenant config persists to `mlfl.yaml`. Global hyperparameters live on the separate `/models` page (`web/templates/models.html`) with debounced autosave per field.

### 3b. Run the benchmark

- One-click **Run Pipeline** on the experiment page (`experiment.html:41`) or **Run All Benchmarks** on the dashboard (`dashboard.html:24-29`). Both post to `/experiment/{name}/run-pipeline` / `/api/benchmarks/run-all` (`web/app.py:3184, 1144`).
- The add-on serialises training behind `_training_lock` (`main.py:463`) and a de-duped FIFO `_retrain_queue`. Dashboard shows "Queued (#N)" badges and lets users **cancel queued items** or **Stop Training** (`dashboard.html:138-149`, `web/app.py:1630`).
- During a run, a **Training** tab appears with live progress (`experiment.html:454-513`): current model, fold, epoch, validation loss, best loss, patience, model picker with per-model loss-curve chart, and a collapsible event log. Backed by Server-Sent Events at `/experiment/{name}/training-stream` (`web/app.py:3129`).
- A separate `/training` page (`templates/training.html`) is the canonical "live training" route; the dashboard auto-refreshes every 10 s while training is in flight, 60 s otherwise (`dashboard.html:314`).

### 3c. Decide which model to deploy

- **Results tab** (`experiment.html:517-662`): per-interval (h=1) accuracy table and an optional Daily Cumulative accuracy table — both with MAE/RMSE/MASE means ± std across folds, train time, and Demšar rank. The top row is highlighted; a radio button per row lets the user override the auto-selected winner (`POST /experiment/{name}/select-model`).
- **Predictions tab** (`experiment.html:664-697`): holdout predictions (last 20 % of data) with every model overlaid against the actual, plus residuals.
- **Generalisation tab** (`experiment.html:700-759`): train-vs-test MAE/RMSE per model with colour-coded gap (red = strong overfit, orange = moderate, green = OK), plus fold-stability line charts.
- **Features tab** (`experiment.html:763-772`): feature-importance bar chart for tree models only; users are redirected to Covariate Analysis for neural models.
- **Covariate Analysis tab** (`experiment.html:946-1080`): trigger a leave-one-out test across enabled models. Surfaces a matrix of MAE/RMSE/MASE deltas vs "all covariates", a textual recommendation block, a per-covariate one-click **Remove** button, and an **Apply Best & Retrain** macro that rewrites `mlfl.yaml` and kicks off a fresh retrain.
- **Tuning tab** (`experiment.html:776-942`): pick a model, a trial count (5–200), and a strategy (Optuna TPE or random). Live progress with completed trials and the current best composite. When finished, shows the best params alongside defaults, an Apply button that promotes + retrains, a holdout chart of default vs tuned (interval or cumulative view), and a sortable trial-details table.
- **Promote**: button labelled "Publish *<model>*" on the experiment page (`experiment.html:51-54`). It calls `/experiment/{name}/promote/{model_name}` and flips the experiment to production mode.

### 3d. Production: live forecasts in HA

- Once promoted, the add-on retrains on `retrain_every_hours` and runs inference on `forecast_every_minutes` (both configurable globally and per-experiment).
- Sensors published per `_publish_forecast_sensors` (`main.py:3049-3683`):
  - `sensor.{publish_prefix}{publish_name}` — main forecast state with `forecast` attribute (list of `{datetime, value}` over the horizon).
  - `sensor.{prefix}{name}_cumulative` — running total (daily-reset when source is cumulative + reset_daily, otherwise cumsum from zero).
  - `sensor.{prefix}{name}_interval` — per-interval increments (only when source is cumulative).
  - `sensor.{prefix}{name}_curve` — combined "actual + future" curve for the Lovelace ApexCharts card.
  - `sensor.{prefix}{name}_upper_80` / `_lower_80` — conformal 80 % bands when the residual buffer is calibrated. Cold-start with no `forecast_log` rows: bands omitted, point-only forecast surfaces in the UI.
  - `sensor.mlfl_last_run` — heartbeat sensor for the whole add-on.
- **One-click HA Lovelace dashboard**: `/dashboard_yaml` (`web/app.py:3091`) serves the auto-generated `mlfl_dashboard.yaml` written to `/addon_configs/.../mlfl_dashboard.yaml` (`main.py:5238`, `dashboard.py:182`). One view per experiment: forecast chart, prediction curve, optional cumulative chart, markdown back-link to the lab.

### 3e. Trust the forecast

- **Forecast Accuracy tab** (production only; `experiment.html:1095-1461`). Three-layer diagnostic:
  - **Verdict card**: traffic-light chips for Accuracy / Calibration / Stability, plus headline numbers — typical next-step MAE, 80 % band coverage, run-to-run swing.
  - **Drivers** mid-layer: lead-time error chart (toggle for RMSE & bias) and a "does re-forecasting help?" card comparing first-issued vs latest-issued forecasts for the same future moment.
  - **Diagnostic tools** accordion: raw forecast-log inspector (with a "View raw JSON" debug button), forecast-convergence fan chart (configurable 6/12/24/48 cycles, interval/cumulative view for cumulative targets), per-target trajectory chart (shows every forecast ever issued of a chosen future moment), and a run-to-run disagreement panel with per-moment and daily-total swing tiles.

### 3f. Operations

- **Logs tab** (`logs.html`): live `tail`-style view (3 s poll of `/api/log?lines=500`), level filter (All / Info / Warnings / Errors), colourised by subsystem tag (`[BENCH]`, `[MODEL]`, `[HA]`, `[PREP]`), with download-full-log link.
- **System tab** (`system.html`): version, CPU model and cores, memory and disk usage, paths to config/log/db; global settings form (training CPU cores, process priority, timezone); per-experiment summary cards collapsing to a "Configure →" link.

---

## 4. UI surfaces — every page, panel, and modal

All under HA ingress at port 5052. Direct port also exposed via `config.yaml`. No auth at the FastAPI layer.

### 4a. Top-level pages (nav)

| Route | Template | User task |
|---|---|---|
| `/` | `dashboard.html` | See every experiment at a glance; trigger benchmarks; switch modes; stop training |
| `/experiment/{name}` | `experiment.html` | Configure, run, evaluate, promote, monitor a single experiment |
| `/models` | `models.html` | Edit default hyperparameters per backend (24 cards) |
| `/log` | `logs.html` | Live log tail with level filter |
| `/system` | `system.html` | Health, hardware, global settings, experiment summary |
| `/training` | `training.html` | Global "watch live training" route — used when the dashboard auto-refreshes mid-run |
| `/settings`, `/status` | – | Both 302-redirect to `/system` (`web/app.py:2644, 2744`) — historic URLs kept for compatibility |

### 4b. Modals & one-off panels

- **New Experiment modal** — `dashboard.html:183-224`. Name + entity + cumulative/reset toggles + helper text pointing at the Settings tab for the rest.
- **Confirm modal** — global, in `base.html:43-52`, used by every destructive button (run-all, retrain, stop-training, reset-params, promote, delete experiment, etc.).
- **Toast notifications** — global, top-right, four-level `info/success/warning/error` (`base.html:88-101`).
- **Info-tips** — `<span class="info-tip">?<span class="tip-text">…</span></span>` pattern used 80+ times across `experiment.html` to explain every setting and metric. Position-corrected by JS in `base.html:122-137`.

### 4c. Experiment page tabs (single-page tab-strip, no client routing)

The tabs are rendered conditionally; what the user sees depends on whether a benchmark exists and whether they're in lab or production.

| Tab | Visible when | User task |
|---|---|---|
| Settings | Always | Edit every per-experiment knob: target, data window, cadences, CV, optimiser/loss/activation, daily-loss toggle, solar physics, covariates, load-subtract |
| Models | Always | Enable/disable backends for this experiment (tree models above, neural below); link to `/models` for hyperparameters |
| Training | `is_running` or recent SSE history exists | Live: current model/fold/epoch/val-loss, per-model loss curve, event log |
| Results | `benchmark_result` exists | Per-interval table, optional Daily Cumulative table, per-fold metrics under a `<details>` |
| Predictions | `benchmark_result` | Holdout overlay chart + residuals |
| Generalisation | `benchmark_result` | Train-vs-test gap table, fold stability charts for MAE/RMSE/MASE |
| Features | Any tree-based feature importance present | Per-model importance bar chart |
| Covariate Analysis | `benchmark_result` | LOO covariate test, recommendations, one-click "Apply Best & Retrain" |
| Tuning | `benchmark_result` | Optuna TPE / random; "Apply Tuned Params, Promote & Retrain" |
| Forecast Accuracy | `mode == 'production'` | Verdict, lead-time error, convergence fan, trajectory, run-to-run swing |

### 4d. Page-level controls visible everywhere

- Top nav: Dashboard / Models / Logs / System with hamburger collapse below ~720 px (`base.html:14-30`).
- Dashboard auto-refresh: 10 s while any training is running, 60 s otherwise (`dashboard.html:314`).
- Experiment-page live updates: SSE for training; polling for benchmark results (`base.html:60-85`).

---

## 5. ML pipeline shape — what is exposed and what is hidden

The README diagram (lines 110–151) summarises the flow; the table below maps each stage to user-facing surfaces.

| Stage | Code | Exposed to user | Hidden from user |
|---|---|---|---|
| **Ingest** | `ha_interface.py:309 get_history` → SQLite cache then delta-fetch in `main.py:878 _fetch_and_preprocess` | Target entity, `days_history`, `max_age`, cumulative/reset_daily flags. "Recorder gap" warnings appear in the log (`main.py:1030`). `Database not available` empty state on Forecast Accuracy when SQLite forecast_log is empty | The carry-forward synthetic-sample logic (`main.py:983-1036`) when HA recorder dedupes static values; max-increment clipping count; the SQLite cache itself is invisible |
| **Preprocess** | `preprocessing.py`: `cumulative_to_interval`, `resample_to_grid`, `clip_outliers`, `apply_log_transform`, `apply_load_subtract` | Cumulative source toggle, reset-daily toggle, max-increment, log-transform toggle, load-subtract list with per-entry source semantics / missing-policy / scale / max-fraction | Outlier clipping bounds; the audit dictionary returned by `apply_load_subtract` is logged but not surfaced in the UI |
| **Feature engineering** | `features.py:build_features`; `solar_physics.py` | Solar physics toggles, holiday country code (`country`), covariates with role (`lagged`/`future`/`both`) + aggregation + binary flag + scale | Lag count `n_lags=12`, rolling windows `[6, 24, 72]`, cyclic encodings, the night-time GHI lag gate (`features.py:178`), interaction features `{covariate}_x_hour_*` |
| **Train (benchmark)** | `benchmark/runner.py:730 run_benchmark` per fold per model | CV strategy, folds, embargo, recency half-life, loss function, optimiser, output activation, daily cumulative loss toggle, per-model hyperparameters | Per-fold split sizes, scaler choice, RevIN, early-stopping patience, the composite-rank computation in `_compute_composite_ranks` |
| **Evaluate** | `benchmark/metrics.py` | MAE / RMSE / MAPE / sMAPE / MASE / R² values, ± std across folds, per-fold metrics in a `<details>`, custom Python-expression metrics (asteval-sandboxed) | Diebold-Mariano pairwise comparison (`comparison.py`) is computed but not surfaced anywhere in the UI |
| **Inference (production)** | `main.py:2153 _run_production_inference` and `_forecast_with_cached` | Forecast & conformal-band sensors in HA; the auto-Lovelace dashboard at `/dashboard_yaml` | The recursive feature builder, the GHI gate at inference, the conformal-quantile pooling fallback when no quantiles are calibrated for the current `model_version` |
| **Persist** | `db.py` (forecast_log, history_*, benchmark_results), `main.py:_persist_cached_model` (`models/<exp>/model.bin` + `cache_meta.json`) | Disk-usage stat in System page | Cache schema, `cache_meta.json` contents, schema_versions table |
| **Present** | `web/templates/experiment.html`, `dashboard.py` Lovelace generator, `_publish_forecast_sensors` HA sensors | All tabs above, the Lovelace YAML download | The internal model_version tagging on each forecast_log row, the issued_at timestamp |

Notable structural choices:

- **Benchmark vs production are explicit modes per experiment** (lab/production). The mode badge appears everywhere; the visible tab set differs between modes (Forecast Accuracy only in production; Tuning/Covariate Analysis only after a lab benchmark).
- **Forecast and retrain cycles are decoupled** (README:79-82). The user sees both cadences and next-run countdowns on the dashboard card.
- **Single training operation across the whole add-on at a time** (`_training_lock`, `main.py:463`); UI shows queued counts but no estimate of when a queued item starts.

---

## 6. Model lifecycle — what the user can actually do

| Action | How | Notes |
|---|---|---|
| **Add a backend** | Not user-facing. New backends are Python files added to `ml_forecast_lab/models/` and registered against `ModelRegistry`. The UI catalog is a hard-coded literal at `web/app.py:751-824` |
| **Enable / disable a backend for an experiment** | Models tab on the experiment page, toggle switches (`experiment.html:407-451`) → persists `models_enabled` to `mlfl.yaml` |
| **Edit default hyperparameters** | `/models` page, debounced autosave (`models.html:232-242`). 600 ms debounce; "N overrides" badge per card |
| **Override params per experiment** | Only via tuning (apply tuned params) or by hand-editing `mlfl.yaml`. No per-experiment param form in the UI |
| **Train (benchmark all enabled models)** | "Run Pipeline" or "Run All Benchmarks". Walks `_training_lock`-serialised queue |
| **Retrain the production model** | "Retrain" button on dashboard card (`dashboard.html:151-157`) or `POST /experiment/{name}/retrain`. Used after settings/covariate changes |
| **Compare models** | Results / Predictions / Generalisation / Features tabs; ranks already computed |
| **Promote a model** | "Publish *<name>*" button on the experiment page or per-row "Promote" actions inside Results. Flips mode to production, persists `production_model: <name>` to YAML |
| **Auto-select on next benchmark** | Set `production_model: null` in YAML (the bundled example does this) — the highest-ranked model wins each cycle |
| **Tune** | Tuning tab: TPE or random search, 5–200 trials, scoped to one model; "Apply Tuned Params, Promote & Retrain" is a single button that writes YAML and kicks a retrain |
| **Apply data-driven covariate config** | Covariate Analysis tab → "Apply Best & Retrain" — removes covariates the analysis flags as harmful and retrains |
| **Stop in-flight training** | "Stop Training" on dashboard card or experiment page → cancels after current epoch via `_stop_training_trigger` (`main.py:581`) |
| **Delete an experiment** | `POST /api/experiments/{name}/delete` (`web/app.py:1726`). No UI button — only the API endpoint exists |
| **Delete a model's cached weights** | No UI. Files in `/data/ml_forecast_lab/models/<exp>/` are user-deletable from the shell only |
| **Roll back / switch versions** | No version history. `model_version` is tagged per forecast_log row (`web/app.py:106`), and `clear_forecast_log_on_retrain` clears old rows by default (`main.py:1556`). No "previous champion" archive |
| **Export / share a trained model** | No export. No ONNX path, no quantisation, no portable artefact. Pickle/torch state dicts only |

---

## 7. Observability surface — what users see about quality, drift, errors

### 7a. Quality (per-benchmark)

- Per-model **MAE / RMSE / MASE means ± std** across CV folds, per-fold breakdown under a `<details>` accordion, and a Demšar composite rank (Results tab).
- Per-day Daily Cumulative MAE/RMSE/MASE rankings as a parallel table (informational only — does not drive Promote).
- **Train-vs-test gap** with traffic-light colouring (Generalisation tab).
- **Holdout predictions overlay** (last 20 %) and residual plots (Predictions tab).
- **Feature importance** for LightGBM/XGBoost only (Features tab).
- **Diebold-Mariano** pairwise tests computed in `comparison.py:336` but not surfaced.

### 7b. Quality (live)

- Forecast Accuracy verdict card: typical next-step MAE, 80 % band coverage, run-to-run swing.
- Lead-time error chart (MAE by lead time; optional RMSE/bias overlay).
- First-vs-latest forecast comparison ("does re-forecasting help?").

### 7c. Drift / stability

- **Forecast convergence fan** (last N runs over a future window) — shows whether the band narrows toward truth.
- **Per-target trajectory chart** — every forecast issued for one future moment over time; oscillation = unstable model.
- **Run-to-run disagreement panel** — per-moment swing (median CoV) and daily-total swing if applicable.
- `model_version` tagging on every forecast_log row (`main.py:2830`) lets analytics queries separate pre- and post-retrain cohorts. The user only sees this indirectly (the Forecast Accuracy chart and verdict respect the current version where possible).
- **No first-class data-drift check** on the input target distribution — no PSI, KS, or covariate drift surfaces.

### 7d. Errors

- Per-experiment `last_error` chip on dashboard cards and a compact run-info bar on the experiment header (`dashboard.html:73-77`, `experiment.html:26-37`).
- Toast notifications for action failures (mode toggle, retrain, stop, tuning apply, etc.).
- Live log tail with level filter; download full log button.
- Some failures only surface in logs (e.g. `load_subtract` audit, conformal-quantile fallback path, carry-forward warning).

---

## 8. Hardware assumptions in code — Pi 5 compatibility audit

### 8a. Architecture and base image

- `config.yaml:10-13` declares `arch: [aarch64, amd64, armv7]`. The Pi 5 hits the **aarch64** path.
- `build.yaml` resolves the base image per arch (Ubuntu-base 9.0.5 from `ghcr.io/hassio-addons`).
- `Dockerfile` is a two-stage builder→runtime image; build deps include `gcc`, `g++`, `gfortran`, `cmake`. Runtime keeps only `libgomp1` and Python. **First build on a Pi 5 takes 10–15 min** (README:42) because LightGBM, XGBoost and torch native extensions compile.

### 8b. CPU / GPU assumptions

- **No CUDA dependency** anywhere. `requirements.txt:8` pins `torch>=2.0.0` with no cu* extras. All neural backends are CPU-only.
- **Every `torch.load` uses `map_location="cpu"`** (all 17 neural backends — `models/{lstm,cnn,...}_backend.py`). No `to("cuda")`, no `torch.device`, no `cuda.is_available()` branching anywhere in the codebase.
- **No `torch.set_num_threads` / `set_num_interop_threads` calls** — torch picks its own intra-op pool, which on a Pi 5 (4 cores) typically means saturating all cores during training.
- **`cpu_cores` and `nice_priority` settings exist in `AppConfig`** (`config.py:571, 574`) and the `system.html` form writes them to YAML, **but nothing in the codebase actually applies them.** No `os.nice()` call, no `sched_setaffinity`, no torch thread-count setter. The settings are inert — a real UX trap given the form's prominence on the System page.

### 8c. Memory

- The orchestrator is memory-aware: cgroup v1/v2 detection (`main.py:4247-4302`), per-process RSS via `/proc/self/status`, and a memory-pressure abort path in tuning (`main.py:4501-4504`).
- Tuning batch size is hard-capped: `TUNING_NEURAL_BATCH_SIZE = 16` "halved from default 64 to reduce peak memory" (`main.py:4208`).
- The full benchmark path is not similarly throttled — peak memory during benchmark training is bounded only by the chosen backend's defaults.
- Single shared SQLite connection with `check_same_thread=False` (`db.py:56`) and an RLock; WAL mode enabled. Acceptable for Pi-class hardware but **all DB activity is single-threaded**, so heavy accuracy scans can stall UI requests.

### 8d. Disk / SD card

- All writeable state lives under `/data/ml_forecast_lab/` (HA per-add-on volume): SQLite history.db, model bins, logs.
- Log rotation: `RotatingFileHandler`, 5 MB × 5 files.
- **No VACUUM** is scheduled; no WAL checkpoint logic. On a heavily used add-on with months of forecast_log writes, history.db can grow to several hundred MB without compaction.
- `homeassistant_config:ro` mount means the add-on never writes to `/config` (good for SD-card longevity).
- The Lovelace YAML is written once at startup (`main.py:5238`); no recurring disk writes from that path.

### 8e. Network

- Only outbound HTTP is to the HA Supervisor proxy at `http://supervisor/core/api` (`ha_interface.py:309-410`). No external SaaS, no model downloads, no telemetry, no licence checks.
- `holidays` and `pvlib` are vendored — no online lookups.

### 8f. UI weight on a 2019-era laptop over HA ingress

- HTMX vendored (~14 KB gzipped) + Plotly-basic (~700 KB gzipped). Plotly-basic is the heaviest single asset; experiment.html re-renders 5+ Plotly charts on the Forecast Accuracy tab alone.
- `experiment.html` is 3 736 lines, all inline. No code-split, no lazy tab loading — tabs are pre-rendered server-side, displayed/hidden via CSS. Initial HTML payload for a production experiment with full diagnostics can run to several hundred KB before charts populate.
- Dashboard auto-reloads every 10 s during training (full page reload, not partial) — noticeable on a slow ingress connection.

---

## 9. Hailo NPU integration — status

**There is no Hailo (or Coral / Edge-TPU) integration in the current code.** The CHANGELOG (`ml-forecast-lab/CHANGELOG.md:2910-2954`) documents an earlier integration that was **removed entirely** in an earlier release. Quoting the relevant entry:

> *"The Hailo integration has been removed entirely. After investigation we found … Hailo's Data Flow Compiler (DFC) is x86-64 Linux only … `HailoAcceleratedModel(model, hef_path=onnx_path)` where the class actually fell back to the CPU PyTorch model (both sides were CPU), so `hailo_active=True` was set in the dashboard while the NPU was idle. No forecasts were ever actually accelerated."*

The removal took out:

- `ml_forecast_lab/models/hailo_runtime.py` (entire file)
- `hailo_enabled` from `AppConfig`, `hailo_active` from `ExperimentStatus`
- Hailo branch in `_retrain_and_cache`, the `is_hailo` / `hailo_accelerated` cached-model fields
- Hailo checkbox from the System page and matching JS
- Hailo badge from dashboard experiment cards
- `python3-hailort` apt install + `--system-site-packages` venv tweak
- `/dev/hailo0` device mapping + `SYS_RAWIO` capability from the add-on manifest
- Hailo section from README + CONFIG_GUIDE

**Current state for Pi 5 + Hailo-8/8L users:** zero acceleration paths in this add-on. Every prediction runs on the four ARM cores. Anything that would put this back — ONNX export from the neural backends, a Hailo runtime wrapper, device mapping in `config.yaml` — would be a re-introduction, not an extension, and would have to confront the same compilation problem (Hailo's DFC is x86-64 only).

The only model-export-adjacent code in the repo is the deprecated `export_onnx` stub mentioned in `MODELS_CREATED.txt` ("Returns False (not recommended)").

---

## 10. Things that look like features but aren't (discoverability traps)

These will inform the improvements proposal — they're cases where something exists in the code or UI but doesn't currently deliver to the user.

- **Training CPU cores / process priority settings** in `/system` (`system.html:62-100`) write to `mlfl.yaml` but are never applied (§8b). A user who turns "Training CPU cores" down to 1 will see no change in load.
- **Future-role covariates** are advertised in `mlfl.yaml` (line 150) and the Settings UI dropdown (`experiment.html:297`), but `CovariateResolver.fetch_future` (`covariates.py:159`) returns NaN as a placeholder — future covariates aren't actually wired into the feature build for inference.
- **Diebold-Mariano pairwise model tests** are computed in `benchmark/comparison.py` but never rendered in the UI.
- **Daily Cumulative ranking** is presented as a parallel ranking table (Results tab) but explicitly does not drive Promote / Tuning / live forecasting — flagged in the info-tip but a likely cause of "which rank should I trust?" confusion.
- **Custom metrics** (asteval-sandboxed Python expressions in YAML) compute and store values but have no first-class UI surface beyond appearing alongside built-in metrics.
- **Delete experiment** has an API endpoint (`/api/experiments/{name}/delete`) but no UI button.
- **`mlfl_dashboard.yaml`** auto-generation produces a Lovelace dashboard but the download path (`/dashboard_yaml`) is not linked from any UI page — users have to know it exists.
- **Direct port 5052** is exposed alongside ingress (`config.yaml:17` + the bound port). Users sometimes hit `http://homeassistant.local:5052` directly (this URL is even hardcoded in `dashboard.py:124`); the lack of auth at this layer is a discoverability *and* security concern that surfaces as UX.

---

Survey ends. Awaiting confirmation before drafting `IMPROVEMENTS.md`.
