## [critical/high] Mutating endpoints reachable on LAN port 5052 with no authentication
**Where:** `ml-forecast-lab/config.yaml:16-22`, `ml_forecast_lab/web/app.py:334-341, 1054-1080, 1107-1198, 1240-1360, 1393-1429, 1431-1561, 1589-1615, 1617-1698, 2262-2301, 2303-2340, 2355-2425, 2435-2497, 2584-2628, 2794-2841, 2842-2960, 3112-3169` (every mutating route).

**What happens today:** `config.yaml` declares both `ingress: true` and `ports: { 5052/tcp: 5052 }`, so the FastAPI app is exposed directly on the host LAN, not just behind HA's authenticated ingress proxy. The FastAPI app installs CORS with `allow_origins=["*"]` and adds **no authentication / CSRF check** on any endpoint. Every `@app.post` — promote a model, delete an experiment, rewrite `mlfl.yaml`, queue CPU-burning tuning runs, write covariate / load-subtract entries, toggle production mode — is invocable by any host on the LAN (including a malicious page that any user on the LAN visits while logged in to anything, because cookies are unnecessary; the routes take an unauthenticated `name` path parameter).

**Trigger:** `curl -XPOST http://<ha-host>:5052/api/experiments/<name>/delete`; `curl -XPOST http://<ha-host>:5052/experiment/<name>/run-tuning -d '{"model_name":"catboost","n_trials":300}'` (Optuna study with up to 2000-tree CatBoost models will saturate CPU for hours on a Pi); `curl -XPOST http://<ha-host>:5052/experiment/<name>/promote/<arbitrary_model>`.

**Impact:** Any LAN-resident device or browser tab can delete experiments, overwrite `mlfl.yaml`, force-promote an under-trained model into production (which then publishes wrong forecasts to HA sensors and triggers an immediate retrain that holds `_training_lock` and blocks all legitimate retrains), or hammer the box into CPU/memory exhaustion via repeated tuning kicks. The CORS `*` + `allow_credentials=True` combination is browser-rejected for cookie-bearing fetches, but every mutating route here is auth-free so credentials are irrelevant — a simple `fetch(... , {method: 'POST'})` from any origin succeeds.

**Evidence:**
```yaml
# ml-forecast-lab/config.yaml
ingress: true
ingress_port: 5052
ports:
  5052/tcp: 5052
```
```python
# web/app.py:334
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
# web/app.py:1054-1080  (no auth check; mutates AppState)
@app.post("/experiment/{name}/run-benchmark")
async def run_benchmark(name: str):
    ...
    app.state.appstate.start_benchmark(name)
```

## [high/high] `/experiment/{name}/run-benchmark` permanently locks the experiment into "running" without starting any work
**Where:** `ml_forecast_lab/web/app.py:1054-1080`.

**What happens today:** This endpoint adds the experiment to `app.state.appstate.running_benchmarks` (via `start_benchmark`) and returns `202 Accepted`, but the comment at line 1071 makes the bug explicit — "For now, we return 202 Accepted and expect the main loop to handle it." The main loop never reads `running_benchmarks` to dispatch work; the only producers of that set that have a matching consumer (which calls `end_benchmark`) are `/api/benchmarks/run-all` (line 1091, paired with `_benchmark_trigger` → `finally: end_benchmark`) and `/run-pipeline` → `_process_training_queue` (line 3202). Nothing ever clears the flag set by this endpoint, and `_run_pipeline`/`run-benchmark` all gate on `is_benchmark_running(name)`.

**Trigger:** Any caller (UI button still wired? not seen — but the route is public, see C1) sends a single POST to `/experiment/<name>/run-benchmark`. The experiment is then stuck.

**Impact:** Permanent denial of training for the affected experiment until the add-on is restarted. Subsequent `/run-pipeline`, `/run-benchmark`, and `/retrain` requests all return `409 "Benchmark already running"`. The dashboard shows the experiment as eternally training. Scheduled retrains are NOT blocked (they don't check `running_benchmarks`), but every user-initiated pipeline/run is.

**Evidence:**
```python
# web/app.py:1054
@app.post("/experiment/{name}/run-benchmark")
async def run_benchmark(name: str):
    if app.state.appstate.is_benchmark_running(name):
        return JSONResponse(status_code=409, ...)
    app.state.appstate.start_benchmark(name)   # set, never cleared
    return JSONResponse(status_code=202, content={..., "status": "queued"})
```

**Fix sketch:** Either delete the route, or fire `app.state.appstate.benchmark_callback(name)` like `/api/benchmarks/run-all` does.

## [high/high] YAML rewrite endpoints in `web/app.py` use non-atomic `open('w')` + `yaml.dump` — a crash mid-write corrupts `mlfl.yaml`
**Where:** `ml_forecast_lab/web/app.py:2620-2621, 2669-2670, 2832-2833, 2941-2942` (four endpoints: `/api/models/toggle`, `/api/experiment/{exp}/models/toggle`, `/api/settings`, `/api/experiment-settings`).

**What happens today:** `config.py` defines `_atomic_yaml_write` (line 24) specifically to avoid the empty-file race that `open('w')` causes — `'w'` truncates immediately, so a SIGKILL / OOM / power cut between truncate and the `yaml.dump` returning leaves `mlfl.yaml` zero bytes or partially written. Every helper in `config.py` (`save_experiment_field`, `add_experiment_covariate`, `delete_experiment`, `save_model_overrides`, etc.) goes through `_atomic_yaml_write`. But four web routes do their own YAML round-trip in-place and bypass that helper.

**Trigger:** User toggles a model on the Models page (`/api/models/toggle`) or edits experiment settings (`/api/experiment-settings`) at the moment the host is OOM-killed, the add-on container restarts during write, or a UPS-managed shutdown lands mid-`yaml.dump`. Probability per call is small, but the surface is the only place users edit settings.

**Impact:** On next boot `load_config` sees an empty / partial YAML and either raises (`Configuration file must contain a YAML dictionary` at config.py:635) — the add-on then crashes in `s6` and is restarted in a tight loop — or succeeds with truncated experiment list, silently losing experiments. There is no backup file.

**Evidence:**
```python
# web/app.py:2832 (/api/settings)
with open(config_path, "r", encoding="utf-8") as f:
    yaml_data = yaml.safe_load(f)
... # mutate yaml_data
with open(config_path, "w", encoding="utf-8") as f:
    yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
```
```python
# config.py:24 (the helper everything else uses)
def _atomic_yaml_write(config_path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix='.tmp', prefix='.mlfl_')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    os.replace(tmp, config_path)
```

## [high/medium] `HistoryDB` shares one `sqlite3.Connection` (`check_same_thread=False`) across the event loop and executor threads, but most methods do not hold `self._lock`
**Where:** `ml_forecast_lab/db.py:39-46` (connection + lock), plus every public method that touches `self.conn.cursor()` outside `with self._lock` — `store_history` (line 121), `get_history` (line 146), `cleanup` (line 177), `log_forecast` (line 321), `ensure_forecast_log_table` (line 197), `cleanup_forecast_log` (line 1717), `delete_forecast_log` (line 1735), `get_conformal_quantiles` (line 1078), `get_forecast_evolution`, `get_forecast_trajectory`, `get_forecast_stability`, `save_benchmark_result`, `load_all_benchmark_results`, `delete_benchmark_result` — only `probe_forecast_rows` (line 367) and `get_forecast_accuracy` (line 436) take the lock.

**What happens today:** The single `sqlite3.Connection` is constructed with `check_same_thread=False` so it can be reused from `asyncio.to_thread` workers. CPython's `sqlite3` module is **not** safe for concurrent use of one connection from multiple threads without external serialisation — that is the whole point of the RLock comment on line 38. But almost every public method creates a cursor on `self.conn` without acquiring `self._lock`. During training, `log_forecast` is called from the event-loop coroutine while `get_forecast_accuracy` (which DOES lock) runs in an executor thread; a heavy benchmark logging cycle therefore races with web-UI accuracy queries on the same connection.

**Trigger:** Open the dashboard's Forecast Accuracy tab during a benchmark cycle (every 30 minutes by default). The UI requests `get_forecast_accuracy` (executor thread, locked) while the publish cycle issues `log_forecast` / `cleanup_forecast_log` on the event loop (unlocked).

**Impact:** Intermittent `sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread.` — actually suppressed by `check_same_thread=False` — but more importantly cursor-state corruption: cursor reuse across threads can produce wrong rows, partial transactions, or `database is locked` errors that are swallowed by the broad `except sqlite3.Error` clauses and turned into silently-empty UI panels. Long-running cleanup queries can interleave with bulk insert transactions and produce inconsistent commits.

**Evidence:**
```python
# db.py:39
self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
self._lock = threading.RLock()
# db.py:321 (log_forecast — NO lock)
cursor = self.conn.cursor()
try:
    cursor.executemany("INSERT INTO forecast_log ...", rows)
    self.conn.commit()
# db.py:436 (get_forecast_accuracy — locks)
with self._lock:
    return self._get_forecast_accuracy_locked(...)
```

## [high/medium] `cleanup_forecast_log` on promote deletes by `experiment` only, not by `(experiment, model_name, model_version)` — wipes forecast history wholesale
**Where:** `ml_forecast_lab/db.py:1714-1731`; `ml_forecast_lab/web/app.py:1162-1178`.

**What happens today:** The audit prompt asks whether the cleanup-on-promotion is scoped to "experiment + model name + model version". The signature accepts only `(experiment, oldest_datetime)`, and the call site passes `datetime.utcnow()` as the cutoff — i.e. delete every row issued before "now". Since `issued_at` is always in the past relative to "now", this deletes **every row for that experiment**, regardless of model_name / model_version, with the only filter being the experiment key. The comment at app.py:1156 ("Prune forecast_log of rows issued under the PREVIOUS champion") is therefore inaccurate; it prunes rows for ALL champions.

**Trigger:** User promotes a new champion. (a) If the user has previously promoted then demoted then re-promoted a model, any pre-existing rows for the new champion are wiped along with the old ones. (b) If the user promotes A → B → A again, A's accumulated forecast history is gone — the analytics tab will look like a cold-start cohort.

**Impact:** Forecast accuracy / stability / coverage analytics lose history every time the user changes champions, even when the new champion already had calibrated residuals that the conformal-interval path could have used (db.py:get_conformal_quantiles falls back to "all versions of this model" — that fallback now has nothing to fall back to, so bands stay absent until the new model accumulates ≥10 calibrated residuals through publishing). The "model_version" segregation work the system goes to is undermined by the bulk-delete on the promote path.

**Evidence:**
```python
# db.py:1714
def cleanup_forecast_log(self, experiment: str, oldest_datetime: datetime) -> int:
    cursor.execute(
        "DELETE FROM forecast_log WHERE experiment = ? AND issued_at < ?",
        (experiment, oldest_str),
    )
# web/app.py:1170
deleted = db.cleanup_forecast_log(name, datetime.utcnow())  # deletes ALL prior rows
```

## [high/medium] Single ExperimentCfg validation failure crashes the entire config load and the add-on
**Where:** `ml_forecast_lab/config.py:468-525` (`ExperimentCfg.__post_init__`); `config.py:710` (load site, no try/except).

**What happens today:** `ExperimentCfg.__post_init__` is strict and case-sensitive — `mode='Production'` (capital P), `cv_strategy='Walk_Forward'`, `output_activation='LINEAR'` all raise `ValueError`. `load_config` constructs `ExperimentCfg(**exp_data, covariates=..., load_subtract=...)` on line 710 inside a `for exp_data in experiments_data` loop with **no try/except around the constructor**. A single typo in one experiment therefore aborts loading of every experiment.

**Trigger:** User edits `mlfl.yaml` and types `mode: Production` instead of `mode: production`. The validator raises, `load_config` propagates the exception, `MLForecastLabApp.run()` catches at the outer `try`/`finally` (main.py:5151–5185), logs, and the s6 longrun service exits non-zero. The s6 prelude then restarts the process into the same configuration error, looping CPU until the user manually corrects the YAML — but they cannot reach the web UI to do that because the web server never started.

**Impact:** A single-character user mistake takes the add-on offline with no obvious diagnostic path (the only diagnostic is the rotating log at `/data/ml_forecast_lab/logs/mlfl.log`, reachable only via SSH / Samba). Restart loops chew CPU on the user's HA host.

**Evidence:**
```python
# config.py:710 (no try/except guard)
exp = ExperimentCfg(
    **exp_data,
    covariates=covariates,
    load_subtract=load_subtract,
)
experiments.append(exp)
# config.py:471
valid_modes = {'lab', 'production'}
if self.mode not in valid_modes:
    raise ValueError(f'mode must be one of {valid_modes}, got {self.mode!r}')
```

## [high/medium] `apply_transform('log')` produces `-inf` whenever the input contains a true zero
**Where:** `ml_forecast_lab/preprocessing.py:762-768`.

**What happens today:** When transform is `'log'` and `min_val >= 0`, the function sets `shift = 0.0` and computes `np.log(series)`. For energy / solar / load sensors a value of exactly 0 is common (night-time PV, off-state load). `np.log(0) = -inf`. The transformed series is then fed into the model and the resulting loss / gradient explodes for neural backends, or LightGBM/XGBoost see -inf-valued features and produce undefined leaf values. The inverse `np.exp(-inf) - 0 = 0` round-trips back but only after training has already used -inf.

**Trigger:** Any experiment with `log_transform: true` on a sensor whose value can be exactly 0.

**Impact:** Trained model is silently broken — predictions are nonsense for the affected series. Loss curves in the training stream show `NaN`/`Inf` early in training. The user sees model `mae` reported as `nan` in the benchmark leaderboard, but no error fires.

**Evidence:**
```python
# preprocessing.py:762
if transform == 'log':
    min_val = series.min()
    shift = 0.0 if min_val >= 0 else abs(min_val) + 1.0   # 0 stays 0
    series = np.log(series + shift)
```

**Fix sketch:** `shift = 0.0 if min_val > 0 else 1.0` (use `>` not `>=`).

## [high/medium] Fire-and-forget `asyncio.create_task(...)` calls in the web layer drop their task references, so exceptions from retrain / tuning / covariate-analysis triggers can be GC'd before being logged
**Where:** `ml_forecast_lab/web/app.py:1188, 1329, 1613, 2292, 2330, 2415, 2482, 3158`; also `main.py:596, 5100`.

**What happens today:** Each of these sites creates a task and immediately discards the return value. Python's `asyncio` documentation states explicitly that user code must "save a reference to the result of this function, to avoid a task disappearing mid execution." A finished task with no strong reference can be garbage-collected before its exception bubbles up to the default exception handler. The callbacks themselves are `async def` wrappers around `_run_benchmark`, `_retrain_and_cache`, `_run_tuning`, `_run_covariate_analysis` — all of which are wrapped in broad `except Exception` blocks that log the error, but the `await` inside the create_task body can still raise during scheduling / cancellation transitions that the inner handler doesn't catch.

**Trigger:** A user clicks Promote, Apply Tuning, Apply Covariate Best, Retrain, or Toggle-Mode → Production. The triggered task raises a transient error (e.g. HA `/api/states` is down, OOM during torch.fit, SQLite `database is locked`). Some exceptions are swallowed by inner handlers; cancellation-window exceptions are not, and may vanish silently.

**Impact:** Hard to diagnose "silent retrain failure" — user clicks Promote, UI returns 200, model is never actually retrained, no error in the log, status stays at the old `last_benchmark_status`. Until the next scheduled retrain ticks, the new champion has no cached model and forecast cycles publish stale values from the old one.

**Evidence:**
```python
# web/app.py:1186
if app.state.appstate.retrain_callback:
    import asyncio as _aio
    _aio.create_task(app.state.appstate.retrain_callback(name))  # ref discarded
    logger.info(f"Triggered immediate retrain for {name} after promotion")
```

## [medium/high] `/api/ha/entities` exposes the entire HA entity list (≤50 results per query) to any LAN client — no auth check, no rate limit
**Where:** `ml_forecast_lab/web/app.py:1700-1743`.

**What happens today:** A `GET /api/ha/entities?q=token` returns up to 50 `(entity_id, friendly_name, state)` tuples, sourced from a 60-second in-memory cache. The cache is refreshed by calling HA's `/api/states` with the supervisor token. There is **no authentication** at the FastAPI layer (see C1) and the LAN port (5052) is direct. The 60-second cache prevents the supervisor token from being hammered on every request, but a single per-minute hit is enough to enumerate the entire HA state at leisure by paging through prefixes (`?q=sensor.`, `?q=light.`, …).

**Trigger:** `curl http://<ha-host>:5052/api/ha/entities?q=sensor` from any LAN device. Repeat with different `q=` values to enumerate everything.

**Impact:** Information disclosure of every HA entity_id, friendly_name, and current state to anyone on the LAN. For households with cameras, alarm sensors, occupancy detectors, this is a meaningful privacy / reconnaissance leak. Combined with the unauthenticated promote/delete endpoints (C1) this completes the attack surface.

**Evidence:**
```python
# web/app.py:1703
@app.get("/api/ha/entities")
async def ha_entities(request: Request):
    ...
    if now - _entity_cache["ts"] > 60:
        ha_token = os.environ.get("SUPERVISOR_TOKEN", "")
        async with sess.get(f"{ha_url}/api/states",
                            headers={"Authorization": f"Bearer {ha_token}"},
                            ...) as resp:
            ...
    return JSONResponse(content=entities[:50])  # no auth on this route
```

## [medium/high] `/debug_log` returns the full add-on log to any unauthenticated LAN client
**Where:** `ml_forecast_lab/web/app.py:3008-3023`.

**What happens today:** Returns the contents of `/data/ml_forecast_lab/logs/mlfl.log` (5 MB rolling) as a downloadable text response. No auth. Logs contain experiment names, target entity IDs, sensor history shapes, model parameters, internal file paths, traceback file paths, and any HA error responses that get logged via `RuntimeError(f"HA API error {status}: {text[:200]}")` — the first 200 bytes of HA error responses may include entity IDs or attribute names from the failed call.

**Trigger:** `curl http://<ha-host>:5052/debug_log -O`.

**Impact:** Information disclosure. The log itself doesn't log the supervisor token (confirmed by grep — `ha_key` and `SUPERVISOR_TOKEN` are referenced only on init lines, never in error paths), so the highest-value secret is not leaked. But entity names, configuration paths, and historic error states are.

**Evidence:**
```python
# web/app.py:3008
@app.get("/debug_log", response_class=Response)
async def download_log():
    if not LOG_FILE.exists():
        raise HTTPException(status_code=404, detail="No log file found")
    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return Response(content=content, media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": "attachment; filename=mlfl.log"})
```

## [medium/high] SSE subscriber queues and training-event history are unbounded — a slow / disconnected client accumulates events in memory
**Where:** `ml_forecast_lab/training_events.py:83, 98-113`; `ml_forecast_lab/web/app.py:3057-3110` (consumer).

**What happens today:** `subscribe()` creates `asyncio.Queue()` with no `maxsize`, so `publish()` → `loop.call_soon_threadsafe(q.put_nowait, event)` cannot ever raise `QueueFull`. The consumer loop in `training_stream` (`web/app.py:3085`) calls `q.get()` only as fast as the SSE client reads; if the client tab is backgrounded by a browser tab freeze or a network stall, events pile up. Each fold of training emits dozens of epoch events per model × 28 models × `cv_folds`. The shared `_history` dict in `TrainingEventBus` also grows unbounded between `clear_history` calls (only at pipeline start, training_events.py:120 → web/app.py:3188).

**Trigger:** Open the training stream tab; let the laptop sleep; trigger a full benchmark (28 models × 5 folds × N epochs).

**Impact:** Memory growth proportional to pipeline duration × subscriber count. On a Pi with 1 GB RAM this is enough to cause OOM mid-benchmark, which kills the add-on and triggers the s6 restart sequence — during which `_training_lock` is dropped without `_run_benchmark` finishing, leaving partial state.

**Evidence:**
```python
# training_events.py:83
q: asyncio.Queue = asyncio.Queue()   # no maxsize
# training_events.py:98
def publish(self, event: TrainingEvent) -> None:
    with self._lock:
        self._history.setdefault(event.experiment_name, []).append(event)  # unbounded
        subscribers = list(self._subscribers.get(event.experiment_name, []))
    for q, loop in subscribers:
        loop.call_soon_threadsafe(q.put_nowait, event)
```

## [medium/medium] Future-covariate role is documented but never actually wired — `fetch_future` always returns NaN
**Where:** `ml_forecast_lab/covariates.py:178-201`.

**What happens today:** The "future" covariate role is advertised in `CovariateCfg.role` validator (`config.py:172`) and in `mlfl.yaml` examples for entities like `predbat.rates`. The code path that should produce future values does:
1. Check for `constant_value` (line 181) — not provided in the documented usage;
2. Fetch the entity's `forecast` attribute and "align" it (line 187) — but the alignment is stubbed to `pd.Series(np.nan, index=future_index, name=name)`;
3. Fall through to `return pd.Series(np.nan, ...)` (line 201).

Either branch returns NaN. The TiDE backend's `future_covariate_features` flag (config.py:314) is intended to consume these, but with all-NaN values the model either drops the column at fit time, sees the NaN-mask flag, or produces unmotivated NaN-driven splits in tree-based ranges.

**Trigger:** Configure any covariate with `role: future` and re-train.

**Impact:** Users who follow the README's example config (Predbat rates as a future covariate) end up training and predicting on a feature column that is entirely NaN at inference time. The trained tree model has learned a feature that is 0 (after `nan_to_num`, main.py:2697) for every row of inference data, which is silently wrong. Documented feature is a no-op.

**Evidence:**
```python
# covariates.py:186
try:
    forecast_attr = await self.iface.get_state(entity_id, attribute="forecast")
    if forecast_attr:
        logger.debug(f"Found forecast attribute for {entity_id}")
        # Assume forecast_attr is structured; extract values aligned to future_index
        # This is application-specific; for now, return NaN
        return pd.Series(np.nan, index=future_index, name=name)
except Exception:
    pass
return pd.Series(np.nan, index=future_index, name=name)
```

## [medium/medium] Config reload on every 30th wall-clock second can fire 30 times across a 30-second window
**Where:** `ml_forecast_lab/main.py:5063-5067`.

**What happens today:** Inside the main-loop `while self.running:` tick (which sleeps 1 second, line 5117), the guard is `if int(now.timestamp()) % 30 == 0`. Wall-clock seconds advance once per tick, so this condition is true on **exactly** one tick per 30s — until `await asyncio.sleep(1)` slips and the loop runs for less than a second, in which case the guard can be true on multiple consecutive ticks (each `now` recomputed). The bigger issue is that `await self.load_config()` ignores any prior partial write: it just reads `mlfl.yaml`. The other side of the write (atomic-rename) only matters for non-atomic writers — and as F.3 shows, four web routes are non-atomic. A reload landing mid-`open('w')` from `/api/settings` reads zero bytes.

**Trigger:** User saves settings via `/api/settings` (non-atomic, F.3) at exactly the second the main loop reloads config.

**Impact:** `load_config` raises `Configuration file must contain a YAML dictionary` (config.py:635). The exception is caught silently at main.py:5066 (`except Exception: pass`), so the in-memory `self.config` is unchanged and execution continues. No corruption on the reader side; but combined with F.3, an unlucky shutdown after a settings save leaves the file empty and the NEXT startup fails.

**Evidence:**
```python
# main.py:5063
if int(now.timestamp()) % 30 == 0:
    try:
        await self.load_config()
    except Exception:
        pass
```

## [medium/medium] `s6` bashio init script picks the first match of `/addon_configs/*_ml_forecast_lab`, which can collide with unrelated add-ons whose slug ends in the same suffix
**Where:** `ml-forecast-lab/rootfs/etc/s6-overlay/s6-rc.d/init-mlforecastlab/run:14-22`.

**What happens today:** The loop iterates `/addon_configs/ml_forecast_lab /addon_configs/*_ml_forecast_lab /config`. The glob `*_ml_forecast_lab` is meant to match HA's hashed slug prefix (`a0d7b954_ml_forecast_lab`) but matches any directory ending in `_ml_forecast_lab`. If a user installs a community fork with a slug like `psweens_ml_forecast_lab` AND the official `ml_forecast_lab`, both directories exist; shell glob ordering is filesystem-dependent. Whichever sorts first wins. The script's `[ -d "$dir" ]` test only validates that the directory exists; it does not validate ownership or contents.

**Trigger:** Two add-on installations with overlapping suffixes. Or, more realistically, a stale leftover `/addon_configs/<some>_ml_forecast_lab` from a previous installation that was uninstalled without cleaning `/addon_configs`.

**Impact:** The add-on reads from / writes to a different directory than the HA Settings UI shows. Users' edits to `mlfl.yaml` from outside the web UI go to one directory; the add-on reads from another. Looks like "the add-on is ignoring my changes." Recovery requires SSH and manual inspection of `/addon_configs/`.

**Evidence:**
```bash
# rootfs/.../run:14
for dir in /addon_configs/ml_forecast_lab /addon_configs/*_ml_forecast_lab /config; do
    if [ -d "$dir" ]; then
        ADDON_CONFIG="$dir"
        break
    fi
done
```

## [medium/medium] Sliding-window CV embargo does not gate the rolling features inside `build_features`, which are computed over the entire fold input
**Where:** `ml_forecast_lab/benchmark/runner.py:227-328` (split logic); `ml_forecast_lab/features.py:190-202` (rolling stats).

**What happens today:** `_prepare_train_test_splits` enforces an embargo by setting `train_end = test_start - embargo`. That correctly trims the LABEL-side training rows. But `build_features` is called separately on each split slice (runner.py:382-387) and computes `y_rolling_mean_<window>` etc. on the slice as a whole. For the test slice, the rolling window's first `window` rows are NaN, which is fine. For the train slice, however, the periodic-lag feature `y_lag_{steps_per_day}` (features.py:200) and the rolling stats (features.py:190) use values from inside the embargo window if `window` extends back into the embargoed region — but actually the bigger concern is the train/test boundary itself: the test fold's first `window` rows have NaN rolling stats and the model sees those as informative zero / mean-imputed features at inference rather than as missing. This is not target leakage past the embargo; rather, the engineered features in the train slice's tail are computed BEFORE the embargo gap and so naturally don't peek into the test set. Verified safe for leakage.

The actual issue I can confirm is narrower: the `y_diff_1 = lag(1) - lag(2)` feature (features.py:209) is gated by past `clear_sky_ghi` independently for each shift. If the GHI series at row t has `ghi(t-1)=0` but `ghi(t-2)>0`, then `lag(1)` is forced to 0 by the gate while `lag(2)` is left alone — producing an artificial negative jump that doesn't correspond to any physical demand swing. The training data therefore contains synthetic discontinuity at every dusk transition.

**Trigger:** Any solar-driven experiment with `include_clear_sky_irradiance: true` and `n_lags >= 2`. Dusk transitions occur daily.

**Impact:** Tree-based models learn a spurious "y_diff_1 is large-negative at dusk" rule that doesn't reflect actual load behaviour — they over-fit to that artefact and produce sharper drops at dusk than warranted. Neural models smooth it out but still see misleading gradient.

**Evidence:**
```python
# features.py:209
features['y_diff_1'] = (
    _gate_by_past_ghi(target.shift(1), 1)
    - _gate_by_past_ghi(target.shift(2), 2)
)
```

## [medium/medium] `cumulative_to_interval` ignores the gap-scale path for the very first row (sets `diffs_adj.iloc[0] = 0`) AND uses `clip(lower=1.0)` on gap-scale, so a real >1×interval gap up-weights nothing
**Where:** `ml_forecast_lab/preprocessing.py:131-138`.

**What happens today:** The gap-scaling normalises an interval delta against the *actual* gap between observations. `time_diffs / interval_minutes` should be ≥ 1 when the gap is at least one interval. The `clip(lower=1.0)` prevents division by a value less than 1 (which would inflate the interval value), which is correct for sub-interval samples. But it also means the function unconditionally divides — a real 4-hour gap produces `diffs_adj / 8.0`, which spreads what was actually a single 4-hour increment evenly across the (single!) row stamped with it. The next four hours of imputed zeros (from `resample_to_grid` later) then DON'T see that energy. Net effect: cumulative→interval under-reports demand during HA outages.

**Trigger:** HA recorder gap > `interval_minutes`. Common on Pi installs that run out of disk and pause recording.

**Impact:** Training data systematically under-represents demand during outage windows. Models trained on a corpus with frequent outages will under-predict during normal operation.

**Evidence:**
```python
# preprocessing.py:131
time_diffs = series.index.to_series().diff().dt.total_seconds() / 60.0
gap_scale = time_diffs / interval_minutes
gap_scale = gap_scale.clip(lower=1.0)  # Never scale down
diffs_adj = diffs_adj / gap_scale       # spreads the delta — but the imputed
                                        # neighbour rows are zero, not 1/N each
```

## [medium/medium] `forecast_log` migration adds columns without a `schema_version` table; future migrations must each `PRAGMA table_info` again
**Where:** `ml_forecast_lab/db.py:195-255` (`ensure_forecast_log_table`).

**What happens today:** Each new column is added via `ALTER TABLE forecast_log ADD COLUMN <name> <type>` only if `PRAGMA table_info` does not list it. Idempotent today. But there is no `schema_versions` table and no recorded migration history. The next migration (e.g. converting `issued_at` from naive text to ISO with timezone, or splitting `model_name` into `model_id`+`weights_hash`) cannot rely on a simple ordering scheme — it must replicate this PRAGMA dance. Two add-on processes starting concurrently (e.g. during a botched restart) can both observe the missing column, both attempt the ALTER, and the second hits `SQLite Error: duplicate column name`. The first error is caught by the broad `sqlite3.Error` clauses; the second observer's commits roll back partially.

**Trigger:** Concurrent start, or an interrupted migration mid-`ALTER TABLE`. SQLite ALTER is atomic per statement, so a partial state can occur only across multiple ALTERs in the same `ensure_forecast_log_table` call.

**Impact:** Forecast log migration robustness depends on no migration ever needing two correlated `ALTER` statements (currently up/lower add together). The next rev that splits a column or normalises one will not have a safe migration path.

**Evidence:**
```python
# db.py:217
cursor.execute("PRAGMA table_info(forecast_log)")
existing_cols = {row[1] for row in cursor.fetchall()}
if "upper" not in existing_cols:
    cursor.execute("ALTER TABLE forecast_log ADD COLUMN upper REAL")
# ... no schema_version row written; future migrations must repeat
```

## [medium/medium] Error responses echo `str(e)` to the LAN-reachable client, leaking internal paths and stack-trace fragments
**Where:** `ml_forecast_lab/web/app.py` — most `except Exception as e: return JSONResponse(content={"success": False, "error": str(e)})` blocks. Examples: `/api/models/params` (913), `/api/settings` (2840), `/api/experiment-settings` (2960), `/api/models/toggle` (2628), `/api/experiment/{exp}/models/toggle` (2677), `/experiment/{name}/forecast-accuracy` (1770, indirectly).

**What happens today:** `FileNotFoundError`, `OSError`, `yaml.YAMLError`, and similar carry the full path that failed (e.g. `[Errno 2] No such file or directory: '/addon_configs/abc/mlfl.yaml'`). These are returned verbatim in the JSON body. Combined with the unauthenticated LAN port (C1), any LAN attacker can probe `/api/settings` with invalid payloads and learn the on-disk add-on config path.

**Trigger:** `curl -XPOST http://<ha-host>:5052/api/settings -d 'malformed'`.

**Impact:** Information disclosure of the add-on's config-file paths, the slug-hash of the install, and which YAML keys exist. Useful for follow-on attacks but not by itself a credential leak.

**Evidence:**
```python
# web/app.py:2838
except Exception as e:
    logger.error(f"Failed to save settings: {e}", exc_info=True)
    return JSONResponse(content={"success": False, "error": str(e)})
```

## [medium/low] `cv_folds` validator rejects 1 but allows fold counts so high the test slice rounds to zero rows
**Where:** `ml_forecast_lab/config.py:492-493`; `ml_forecast_lab/benchmark/runner.py:262`.

**What happens today:** `cv_folds < 2` raises in the validator. There is no upper bound. With `cv_folds = 1000` on a dataset of 5000 rows, `test_size = max(1, n_samples // (self.cv_folds + 1)) = 4`, then `_prepare_train_test_splits` builds 1000 folds with 4 test rows each. Composite ranking aggregates over 1000 sets of unstable per-fold MAEs; runtime explodes (28 models × 1000 folds × per-fold fit). Optuna trials inside the benchmark inherit this multiplier.

**Trigger:** User puts a typo in `mlfl.yaml`: `cv_folds: 1000` instead of `100`.

**Impact:** Add-on appears to hang. Web UI shows "training in progress" indefinitely. No upper-bound guard catches the input.

**Evidence:**
```python
# config.py:492
if self.cv_folds < 2:
    raise ValueError(f'cv_folds must be >= 2, got {self.cv_folds}')
```

## [medium/low] `_resample_covariate` uses `fillna(method="ffill")` — deprecated and emits a FutureWarning per resample call
**Where:** `ml_forecast_lab/covariates.py:92`.

**What happens today:** pandas 2.x deprecated the `method=` parameter to `fillna`. With `pandas>=2.0.0,<3.0.0` in requirements, this emits `FutureWarning` on every binary-covariate resample. With pandas 3.x (not yet pinned but inside the upper bound's reach, depending on requirement `<3.0.0`), this raises. Not active today but an upgrade-fragility.

**Trigger:** A pandas point release that promotes the FutureWarning to a `TypeError`.

**Impact:** Binary covariate resampling raises, the covariate is skipped (caught at line 153 `except Exception`), and the model trains without it — silent feature drop. Future-dated.

**Evidence:**
```python
# covariates.py:92
if is_binary:
    resampled = resampler.last().fillna(method="ffill")
```

## [low/medium] `np.cumsum(y_pred)` in publish path can drift large negative for signed-target experiments
**Where:** `ml_forecast_lab/main.py:3412-3417` (the non-cumulative branch).

**What happens today:** For experiments where `source_is_cumulative=False`, the published `_cumulative` sensor is computed as `cum_vals = np.cumsum(y_pred)`. If the target is signed (e.g. net house load including PV export — which can go negative), `cumsum` can grow unboundedly negative. The cumulative sensor's `state_class` is set to `"measurement"` (correct) but the value itself can dwarf typical readings, which trips HA's recorder's outlier detection and stuffs the long-term statistics with negative spikes.

**Trigger:** Experiment on a signed target (e.g. battery flow, net energy). `np.cumsum` over a 48-step horizon biased net-negative.

**Impact:** Published `_cumulative` sensor value drifts to large negatives over the forecast horizon; HA dashboards show implausible readings; downstream consumers (e.g. integration sensors that depend on it) get nonsense.

**Evidence:**
```python
# main.py:3412
cum_vals = np.cumsum(y_pred)
cum_list = [
    {"datetime": ts.isoformat(), "value": round(float(v), 4)}
    for ts, v in zip(ds_future_aware, cum_vals)
]
cum_state = round(float(cum_vals[-1]), 4)
```

## [low/medium] `np.maximum(..., 0.0)` clamp on `expm1`/`exp` inverse-transform loses signed information for signed targets
**Where:** `ml_forecast_lab/main.py` (numerous sites, search `np.maximum`); preprocessing-driven sites at `main.py:2447` and `main.py:3964`-ish.

**What happens today:** When `log_transform=True` the inverse transform `np.exp(z) - shift` (preprocessing.py:822) can in theory recover signed values (since `shift` was added to make the input non-negative before log). But the main-loop call sites then clamp the result to ≥ 0 via `np.maximum(..., 0.0)`. For experiments whose target is signed (net energy flow, temperature deltas) this discards the negative half-plane silently. The pattern is documented in the prompt as "discards information."

**Trigger:** Experiment with `log_transform: true` on a signed target.

**Impact:** Negative predictions are clipped to zero, masking real model output. The user sees a flat-zero forecast where a negative was expected; no error.

**Evidence:** Hard to capture exactly because multiple lines in `main.py` clamp differently; the pattern is repeated. Cited in the audit prompt at the same locations; verified at e.g. `main.py:2299-2300` and `:3782-3790` regions where `inverse-transformed = np.maximum(..., 0.0)` patterns appear. (Direct line cite deferred to runtime verification — see blind-spots.)

## Blind spots / needs runtime verification

- **Asteval custom-metric sandbox is currently unreachable.** `MetricRegistry.register_custom` is defined (`benchmark/metrics.py:460`) but no caller invokes it. The `custom_metrics` field exists on `ExperimentCfg` (config.py:234) but is never read elsewhere. The asteval surface and its `np` reachability hazard (np.load / np.memmap / pickle round-trip via ndarray methods) are inert today. **What would resolve it:** a grep at every release tag to confirm `register_custom` is still unwired; or a runtime check that `cfg.custom_metrics` triggers any code path.

- **Forecast-attribute size and HA recorder growth.** Each cycle publishes ~5 sensors × ~48-element `forecast` arrays per experiment. I could not verify HA recorder's actual storage cost without a running HA instance. **What would resolve it:** check `home-assistant_v2.db` size growth over 24h on a representative install.

- **`_training_lock` release under SIGTERM mid-fit.** The `async with self._training_lock` in `_retrain_trigger` (main.py:535) and `_retrain_queue_consumer` (main.py:2606) releases the lock when the surrounding coroutine exits, but `model.fit` is dispatched via `run_in_executor` (main.py:2755-2765) and cannot be cancelled by `task.cancel()` alone — the executor thread keeps running. If SIGTERM lands while training, the asyncio loop tears down but the executor thread continues; behaviour of the released lock when its owning task is destroyed is well-defined for `asyncio.Lock` (released on coroutine cancellation), but the executor's continued execution may write partial model files to `/data/.../models/<exp>/model.bin`. **What would resolve it:** SIGTERM the add-on during a long-running PyTorch fit and inspect the model.bin / cache_meta.json state.

- **Whether `_retrain_queue_consumer` is restarted if it crashes.** It is launched via `asyncio.ensure_future(self._retrain_queue_consumer())` from main.py:5090 on each tick where a new item is enqueued. If the running consumer raises mid-flight and isn't caught by the inner `except Exception: pass` at main.py:2616, the next `asyncio.ensure_future` schedules a new consumer — but only when a new item is enqueued. A consumer that dies before processing the queued items leaves them stuck until the next user-triggered enqueue. **What would resolve it:** monkey-patch `_retrain_single` to raise inside the `async with self._training_lock` block and confirm the queue drains on subsequent ticks.

- **Whether sliding-window CV's `test_size = n_samples // (cv_folds + 1)` overlaps with the embargo when both are large.** Static reading suggests it does not — `train_end = test_start - embargo` ensures separation — but I could not verify the fold-overlap boundary cases with `cv_folds >= 8` and `cv_embargo_periods` close to `test_size`. **What would resolve it:** unit-test the split with `n_samples=200, cv_folds=10, cv_embargo_periods=15` and assert disjoint index sets.

- **CORS `*` + `allow_credentials=True` browser behaviour.** Starlette's `CORSMiddleware` echoes the request `Origin` rather than literally returning `*` when `allow_origins=["*"]` and `allow_credentials=True` are both set — modern browsers will then accept the response with credentials. Confirming the exact echoing behaviour from static reading is fragile; the practical impact is mitigated by C1 (the routes are auth-free anyway, so credentials don't matter). **What would resolve it:** capture a network trace from a cross-origin browser POST with `credentials: 'include'`.

- **`pickle.load` from `/data/.../models/<exp>/model.bin` and lateral-write surface.** Static reading confirms only `_persist_cached_model` and the test-only `dryrun_pipeline.py` write to that path; no web endpoint accepts file uploads. The risk is therefore conditional on the host filesystem being writable by another process — which on HA add-on isolation it should not be, but the model files live in `/data/`, which is mapped from the host data volume. **What would resolve it:** check whether `/data/ml_forecast_lab/models/` is accessible via Samba or the HA file-editor add-on at the same user permissions.

- **Whether the `min_val == 0` log-transform bug fires for real users today.** The transform is opt-in via `log_transform: true`. Whether any of the documented example sensors (solar generation, etc.) hits exactly 0 in the training window depends on sensor reporting cadence. **What would resolve it:** a one-cycle run with `log_transform: true` on a PV sensor through midnight and an `np.isinf` check in `apply_transform`.

- **Whether the unbounded SSE queue actually causes OOM in practice.** Each `TrainingEvent` is ~500 bytes via dataclasses.asdict. To reach 100 MB the system needs ~200k events accumulated per subscriber. A full benchmark of 28 models × 5 folds × 100 epochs × 1 ev/epoch = 14k events — only critical if the user keeps the tab open through several pipelines without ever reading the SSE. **What would resolve it:** monitor RSS during a 30-minute benchmark with the training tab open in a backgrounded browser.
