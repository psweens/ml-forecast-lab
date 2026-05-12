# SURVEY.md — ml-forecast-lab

A reconnaissance of the repository prior to writing a tailored audit prompt. British English throughout. File:line references are absolute paths from the repo root.

---

## 1. Repository tree (one-line role per file)

```
ml-forecast-lab/                                          ← repo root (HA add-on repo)
├── repository.yaml                                       ← HA add-on repository manifest
├── README.md                                             ← user-facing overview
├── LICENSE                                               ← MIT/Apache (not inspected)
├── icon.png, logo.png                                    ← add-on artwork
├── docs/
│   ├── CONFIG_GUIDE.md                                   ← user docs: configuring experiments
│   ├── FEATURES_GUIDE.md                                 ← user docs: feature engineering
│   ├── MODEL_GUIDE.md                                    ← user docs: model selection
│   └── PREPROCESSING_GUIDE.md                            ← user docs: preprocessing pipeline
├── .github/workflows/
│   ├── tests.yml                                         ← CI: pytest matrix
│   ├── validate.yml                                      ← CI: HA add-on lint
│   └── release.yml                                       ← CI: tag → image publish
└── ml-forecast-lab/                                      ← the actual add-on
    ├── config.yaml                                       ← HA add-on manifest (slug=ml_forecast_lab, ingress, port 5052)
    ├── build.yaml                                        ← base image map per arch (ubuntu-base 9.0.5)
    ├── Dockerfile                                        ← multi-stage build (builder venv → runtime)
    ├── requirements.txt                                  ← Python deps (numpy/pandas/torch/fastapi/optuna/pvlib/asteval…)
    ├── mlfl.yaml                                         ← example user config copied to /addon_configs on first boot
    ├── CHANGELOG.md
    ├── MODELS_CREATED.txt                                ← apparent scratch/notes file
    ├── translations/en.yaml                              ← HA add-on UI strings
    ├── rootfs/etc/s6-overlay/s6-rc.d/
    │   ├── init-mlforecastlab/{run,up,type}              ← s6 longrun service (type=longrun, up→run, run = bashio script)
    │   └── user/contents.d/init-mlforecastlab            ← marker enabling the service in the user bundle
    ├── ml_forecast_lab/                                  ← Python package (entry: python -m ml_forecast_lab)
    │   ├── __init__.py                                   ← re-exports public API (config, preprocessing, features, db, HA…)
    │   ├── __main__.py                                   ← logging setup + asyncio.run(main()); fallback stub FastAPI on import failure
    │   ├── main.py             (5241 ln)                 ← MLForecastLabApp orchestrator: schedules forecast/retrain cycles
    │   ├── config.py           (1185 ln)                 ← dataclasses (AppConfig/ExperimentCfg/CovariateCfg/SubtractCfg) + YAML load/save
    │   ├── db.py               (1823 ln)                 ← HistoryDB: SQLite (WAL) for history, forecast_log, benchmark_results
    │   ├── ha_interface.py     (424 ln)                  ← async aiohttp client for HA REST API
    │   ├── covariates.py       (201 ln)                  ← CovariateResolver: history+future fetch, binary auto-detect
    │   ├── preprocessing.py    (892 ln)                  ← cumulative→interval, resample, clip, log/sqrt/box-cox, load-subtract
    │   ├── features.py         (544 ln)                  ← lags, rolling stats, holiday flags, solar night-gate, sliding windows
    │   ├── solar_physics.py    (97 ln)                   ← pvlib clear-sky GHI + apparent elevation
    │   ├── training_events.py  (123 ln)                  ← thread→asyncio event bus (SSE training stream)
    │   ├── dashboard.py        (250 ln)                  ← generates ApexCharts Lovelace YAML
    │   ├── web/
    │   │   ├── __init__.py
    │   │   ├── app.py          (3229 ln)                 ← FastAPI app (create_app), 50+ routes, AppState, SSE, callbacks
    │   │   ├── static/{style.css,htmx.min.js,plotly-basic.min.js}
    │   │   └── templates/{base,dashboard,experiment,training,models,logs,system}.html
    │   ├── benchmark/
    │   │   ├── __init__.py
    │   │   ├── runner.py       (940 ln)                  ← walk-forward / sliding-window CV, composite rank, daily-cumulative metrics
    │   │   ├── metrics.py      (549 ln)                  ← MAE/RMSE/MAPE/sMAPE/MASE/R² + asteval-sandboxed custom metrics
    │   │   └── comparison.py   (336 ln)                  ← cross-model comparison utilities
    │   └── models/
    │       ├── __init__.py                               ← optional dynamic import of every backend
    │       ├── registry.py                               ← ModelRegistry (static dict, no entry-points)
    │       ├── base.py         (916 ln)                  ← ForecastModel ABC + RevIN + composite horizon loss + optimiser builder
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

Total Python LOC ≈ 28 138.

---

## 2. Detected stack

| Layer | Choice |
|---|---|
| Base image | `ghcr.io/hassio-addons/ubuntu-base:9.0.5` (aarch64/amd64/armv7) — Debian/Ubuntu with s6-overlay + bashio |
| Runtime | Python 3 from `apt`, isolated venv at `/opt/venv` copied from builder stage |
| Backend framework | **FastAPI** (`>=0.100,<1.0`) + **uvicorn[standard]** (`>=0.23,<1.0`) |
| Templating | Jinja2 (`>=3.1,<4`) — autoescape on; `\|tojson` for JS embedding |
| ML stack | numpy <2, pandas <3, scikit-learn <2, scipy <2, **torch ≥2** (CPU build implied — no cu* in requirements), lightgbm 4.x, xgboost 2.x, catboost 1.x, statsforecast 1.7.x, optuna 3.5.x, pvlib 0.10.x, holidays |
| Custom-metric sandbox | **asteval** 1.x |
| HTTP client | **aiohttp** 3.x (async to HA) |
| Frontend | Server-rendered Jinja templates + **HTMX** (vendored `htmx.min.js`) + **Plotly basic** (vendored `plotly-basic.min.js`). No SPA, no build tool, no JS bundler |
| Persistence | **SQLite** via stdlib `sqlite3`, WAL mode, single shared connection guarded by `threading.RLock` |
| Process supervisor | **s6-overlay** (single longrun: `init-mlforecastlab`), `python3 -m ml_forecast_lab` exec'd from `/command/with-contenv bashio` |
| Package manager | `pip` only (no Poetry/uv lock file). Versions pinned by upper bound, no lockfile |

---

## 3. HA add-on surface

**`ml-forecast-lab/config.yaml`:**

- `slug: ml_forecast_lab`, `version: 2.29.0`, `startup: services`, `boot: auto`, `init: false`
- `arch: [aarch64, amd64, armv7]`
- `homeassistant_api: true` — supervisor exports `SUPERVISOR_TOKEN` for `http://supervisor/core/api`
- `ingress: true`, `ingress_port: 5052`, panel via `panel_icon: mdi:chart-timeline-variant-shimmer`
- Direct port also exposed: `ports: { 5052/tcp: 5052 }` — **dual exposure** (ingress AND direct port) is unusual; raised as a flag below
- `map: [addon_config:rw, homeassistant_config:ro]` — RW into `/addon_configs/.../ml_forecast_lab`, RO into `/config`
- `options: { log_level: info }`; `schema: { log_level: list(trace|debug|info|notice|warning|error|critical)? }`
- No `auth_api`, no `hassio_api`, no `host_network`, no `devices`, no `udev`. No auth header validation beyond ingress trust.

**s6 service** (`rootfs/etc/s6-overlay/s6-rc.d/init-mlforecastlab/`):

- `type` = `longrun`; `up` simply points to `run`; `run` is a bashio shell script that:
  1. `mkdir -p /data/ml_forecast_lab{,/models,/logs}`
  2. Searches `/addon_configs/ml_forecast_lab`, `/addon_configs/*_ml_forecast_lab`, `/config` for an existing dir and copies the bundled `mlfl.yaml` if absent
  3. `exec python3 -m ml_forecast_lab`
- Marker file `rootfs/etc/s6-overlay/s6-rc.d/user/contents.d/init-mlforecastlab` enables the service.

**`mlfl.yaml`** (user config, the heart of the add-on): see `ml-forecast-lab/mlfl.yaml`. Top-level keys: `timezone`, `update_every_minutes`, `experiments: [...]`. Per-experiment keys include `target_entity`, `mode (lab|production)`, `source_is_cumulative`, `reset_daily`, `interval_minutes`, `max_increment`, `days_history`, `max_age`, `future_periods`, `covariates`, `models_enabled`, `loss_fn`, `cv_strategy/folds/embargo_periods`, `metrics`, `production_metric`, `custom_metrics` (Python expressions), `production_model`, `publish_prefix`, `publish_name`, `units`, `output_units`, `country`, `log_transform`, `database`. Custom metrics are arbitrary expressions evaluated by asteval — see §11 flag.

---

## 4. External dependencies

**Home Assistant REST API** (`ml_forecast_lab/ha_interface.py`):

- `GET /api/history/period/{start_iso}?end_time=...&filter_entity_id=...&minimal_response` (history fetch; 180 s read timeout)
- `GET /api/states/{entity_id}` (current state, also used to read forecast attribute for "future" covariates)
- `POST /api/states/{entity_id}` (publishing forecast sensors — `set_state` wrapper at main.py:3571)
- `GET /api/config` (lat/lon/timezone, lazily cached as `_site_location`)
- Auth: `Authorization: Bearer ${SUPERVISOR_TOKEN}` (env var)

**Other outbound:**

- `holidays` library — local, no network
- `pvlib` — local solar geometry (no API calls)
- No third-party SaaS, no telemetry, no model download (everything trains on-box)

**Hardware/devices:** none directly. No `/dev/*` mapping, no GPIO, no USB. The add-on is pure software running on whatever HA host CPU is available (no GPU code paths — all `torch.load` uses `map_location="cpu"`).

---

## 5. Persistence

Everything writable lives under `/data/ml_forecast_lab/` (the HA per-add-on data volume):

| Path | What |
|---|---|
| `/data/ml_forecast_lab/history.db` | SQLite — per-entity tables for raw history, plus `forecast_log` and `benchmark_results` |
| `/data/ml_forecast_lab/models/<exp>/model.bin` | Pickle/torch state-dict of cached production model |
| `/data/ml_forecast_lab/models/<exp>/cache_meta.json` | Feature columns, trained_at, model_version, is_neural, window_size, addon_version |
| `/data/ml_forecast_lab/logs/mlfl.log` | RotatingFileHandler, 5 MB × 5 files |

**Addon config (rw)** `/addon_configs/<hash>_ml_forecast_lab/`:

- `mlfl.yaml` — user config (atomic-written via tmp+rename, `config.py:24`)
- `mlfl_dashboard.yaml` — generated Lovelace YAML

**HA config (ro)** `/config/` — read only; the bashio script falls back to it only for the initial mlfl.yaml lookup.

**Migrations** in `HistoryDB`: idempotent ALTER TABLE on startup (adds `upper`, `lower`, `model_version` columns to `forecast_log` if missing). No explicit schema_version table.

**No VACUUM** is ever triggered. WAL is set but no manual checkpointing.

---

## 6. Concurrency model

**Single process, single asyncio event loop** (uvicorn runs in the same loop). All long-running orchestration lives in `MLForecastLabApp.main_loop()` (`main.py:4993`). Web server is spawned as an asyncio task at `main.py:595`.

Key primitives:

- `_training_lock: asyncio.Lock` — global serialiser. Held across the whole of `_run_benchmark`, `_retrain_and_cache`, tuning and covariate-analysis triggers (`main.py:463/521/534/606`). Effectively means **one training operation across the whole add-on at a time**.
- `_retrain_queue: asyncio.Queue` — unbounded but de-duplicated by an `already_queued` guard (`main.py:5083`).
- `_forecast_running: Dict[str, bool]` — per-experiment forecast flag (`main.py:2649`).
- `_running_tasks: Dict[str, asyncio.Task]` — for stop-training cancellation.
- CPU-bound work (model `.fit`, `predict_sequence`, heavy DB scans) is offloaded with `loop.run_in_executor(None, …)` and `asyncio.to_thread(...)` (default ThreadPoolExecutor; no ProcessPoolExecutor anywhere).
- DB writes/reads from threads serialised by `threading.RLock` inside `HistoryDB` (`db.py:46`). The SQLite connection is shared with `check_same_thread=False` (`db.py:39`).
- Training cross-thread → async bridge via `TrainingEventBus` using `loop.call_soon_threadsafe()` (`training_events.py:111`).

No `multiprocessing`, no real parallel folds (the benchmark runner is a sequential loop over folds and models — see §7 of the model survey). Optuna tuning likewise runs single-process.

`AppState` (`web/app.py:222-282`) is shared mutable state (benchmark results, statuses, callbacks, training queue) **with no lock**. Safe under a single asyncio loop, but any future move to multiple workers would race instantly.

---

## 7. Frontend entry points and routing

Server-rendered Jinja + HTMX, no SPA. Plotly traces are built server-side and embedded as `|tojson` JSON inside `<script>`.

**Page routes (GET):**

- `/` → `dashboard.html` — experiment cards, status badges
- `/experiment/{name}` → `experiment.html` — benchmark results, forecast trajectories/evolution/stability, tuning panel, covariate analysis
- `/models` → `models.html` — global model config and hyperparameter overrides
- `/system` → `system.html` — hardware/runtime/config; `/settings` and `/status` redirect here
- `/log` → `logs.html`; `/debug_log` → download; `/api/log` → JSON tail

**JSON/HTMX fragments:** `/api/status`, `/api/models/params{,/reset}`, `/api/experiments/create`, `/api/settings`, `/api/experiment-settings`, `/api/ha/entities`, `/api/training/history/{name}`.

**Mutating endpoints:**

- `POST /experiment/{name}/run-benchmark | run-pipeline | retrain | stop-training | toggle-mode | select-model | promote/{model_name}`
- `POST /experiment/{name}/run-tuning | apply-tuning | run-covariate-analysis | apply-covariate-best`
- `POST /experiment/{name}/{add,remove}-covariate | {add,remove,clear}-load-subtract`
- `POST /api/experiments/create | /api/experiments/{name}/delete`
- `POST /api/models/{params,params/reset,toggle}` and `/api/experiment/{exp}/models/toggle`

**Streams:** `GET /experiment/{name}/training-stream` — Server-Sent Events; replays history unless `?no_replay=1`.

**Authentication:** none beyond HA ingress. CORS is **wide open** — `allow_origins=["*"]`, `allow_credentials=True`, all methods, all headers (`web/app.py:335-341`). No CSRF tokens. Direct port 5052 is also exposed (see §3), so the CORS posture matters.

---

## 8. Anything unusual or non-idiomatic worth flagging

1. **Dual network exposure** — both `ingress: true` and `ports: { 5052/tcp: 5052 }` in `config.yaml`. Anyone on the LAN can hit the FastAPI app directly, bypassing HA ingress auth. Combined with permissive CORS (`*` + credentials), this is the most consequential security posture issue to inspect.
2. **No auth at the FastAPI layer.** All mutation endpoints (promote model, delete experiment, write `mlfl.yaml`, trigger training) are reachable from the direct port. The implicit assumption "HA ingress will protect us" does not hold for port 5052.
3. **Custom-metric eval via asteval** (`benchmark/metrics.py:460-520`). A regex blacklists `import`, `from`, `__…__`, `open`, `exec`, `eval`, `compile`, but the interpreter is handed the full `np` module with no symbol filtering — `np.load`, `np.save`, `np.memmap`, and ndarray pickle round-trips remain in scope. The user supplies these strings in `mlfl.yaml`, but if the web UI ever surfaces a "save custom metric" endpoint without the same filter, the blast radius widens.
4. **CORS = "*" with `allow_credentials=True`** — modern browsers reject `*` + credentials, but middleware will happily echo back the calling origin if asked; verify the actual behaviour.
5. **5241-line `main.py`** and **3229-line `web/app.py`** — single-file orchestrator and single-file FastAPI. High cognitive density; broad `except Exception` blocks at most error boundaries (intentional robustness, but masks the cause of weirdness).
6. **Hardcoded URLs and paths in `dashboard.py`** — `http://homeassistant.local:5052/experiment/{exp_name}` (`dashboard.py:124`), and the sensor-name convention `sensor.{prefix}{exp_name}_{point,upper_95,lower_95,cumulative,curve}` is repeated in dashboard generation and in the actual publish path; drift between the two would generate a Lovelace dashboard that doesn't render.
7. **`asyncio.create_task` everywhere with no task tracking** in the web layer (`web/app.py:1188/1329/1613/2330/2415/3158`). If an exception fires inside one of these, it's logged but the failure does not surface to the user. There is no `set(asyncio.create_task(...))` strong-ref pattern.
8. **YAML I/O on the event loop** (`web/app.py:2813-2833`, etc.) — small files, but technically blocking inside `async def` handlers. Atomic-write helper is in `config.py` so corruption risk is low.
9. **`os.getenv("PORT", 5052)` in the stub fallback** (`__main__.py:210`) but `uvicorn.run(host="0.0.0.0", port=5052)` in the real path (`main.py:350`). The stub also re-binds `0.0.0.0` with no auth — but only fires if importing `main` fails, which would be a deployment bug.
10. **Reproducibility hole** — only CatBoost sets a seed (hardcoded `random_seed: 42`). No `np.random.seed`, no `torch.manual_seed`, no `random.seed` anywhere. Two consecutive benchmark runs on identical data will rank models differently when scores are close. Composite ranking dampens but does not eliminate this.
11. **GPU compatibility silently degraded** — every `torch.load` uses `map_location="cpu"`. A pre-existing checkpoint produced on a GPU will load on CPU, but training was never wired for GPU acceleration in the first place. Add-on users on Pi/ARM will be unsurprised; users on capable hardware get no benefit.
12. **`check_same_thread=False` SQLite** with one shared connection + RLock — works, but means **all DB activity is single-threaded**. Long forecast-accuracy scans pushed to `asyncio.to_thread` still serialise behind the lock; UI requests can stall during heavy benchmark logging.
13. **Per-entity dynamic tables** in SQLite — table name derived from `entity_id` via regex sanitisation (`db.py:49-65`). Sanitisation is the only barrier between user-controlled entity IDs and DDL — worth confirming the regex is tight.
14. **`forecast_log` schema migration via in-place ALTER TABLE** at startup with no schema-version table. Idempotent today, but next migration round will have to inspect `PRAGMA table_info` again.
15. **The "future" covariate path** in `covariates.py:182-200` returns NaN as a placeholder for forecast attributes — meaning future covariates (e.g. Predbat rates in the example config) are not actually wired in despite being documented.
16. **`MODELS_CREATED.txt`** at the add-on root looks like a scratch artefact that escaped into the repo.
17. **`asyncio.to_thread`** is used in the web layer for DB calls, but the same thread pool is shared with `run_in_executor(None, model.fit, ...)`. A long training run can starve UI DB queries on the default loop executor (default size = `min(32, os.cpu_count()+4)` since Python 3.8). On a 2-core Pi this is two slots.
18. **No HA Supervisor-token rotation handling.** The token is read once into `HAInterface.ha_key` at construction (`ha_interface.py:210`); the supervisor rotates it on add-on restart, but a long-running session that ever drops will not refresh until the process restarts.

---

Survey ends here. Pausing for your confirmation before drafting `AUDIT_PROMPT.md`.
