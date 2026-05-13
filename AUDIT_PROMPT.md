# Latent-defect audit — ml-forecast-lab

You are auditing a Home Assistant add-on at the root of this repository.
The add-on is a multi-model time-series forecasting service that trains
ML models on Home Assistant entity history, publishes forecasts back as
HA sensors, and exposes a FastAPI + HTMX + Plotly web UI on port 5052.
British English throughout.

## What you must do

Perform a **latent-defect audit** of the code as it currently stands on
this branch. You are looking for bugs, security weaknesses, correctness
gaps, hidden coupling, and resource-management problems that will
eventually bite the user. You are **not** writing new code, not
refactoring, not adding tests, not improving CI.

Read whatever you need. Use `Read`, `Grep`, `Bash` for `grep`/`rg`, and
`Agent` (subagent_type `Explore`) for breadth-first sweeps. Read large
files in full when the failure mode could hide on any line — `main.py`
(5241 ln), `web/app.py` (3229 ln), `db.py` (1823 ln), `config.py`
(1185 ln), and `models/base.py` (916 ln) all warrant it. Read every
model backend at least skim-deep enough to spot copy-paste hazards.

## Hard constraints

- **No code edits.** Do not call `Edit`, `Write`, or `NotebookEdit`. Do
  not stage or push anything. Do not modify `.claude/`, settings, hooks,
  or workflows.
- **No test/CI/lint suggestions.** Do not propose new tests, refactors,
  type hints, docstring rewrites, dependency upgrades for their own
  sake, or CI improvements. If a missing test is the only way to catch
  a bug, say so in one line under "blind spots".
- **No info-level noise.** No "consider adding logging", no "this could
  be more Pythonic", no "extract to a function", no stylistic feedback.
  Every finding must describe a behaviour that is wrong or fragile
  *today*, not an aesthetic improvement.
- **Evidence is mandatory.** Every finding gets a `path/to/file.py:LINE`
  reference and a short code snippet (≤ 8 lines, trimmed to the
  relevant slice). Findings without evidence are rejected.
- **No speculation dressed as fact.** If you cannot tell from static
  reading whether something fires, say so and demote to the blind-spots
  section rather than reporting a high-severity finding.
- **British English** in prose. American English allowed inside code
  snippets where the source already uses it.

## Severity × confidence

Tag every finding with both axes.

| Severity | Meaning |
|---|---|
| **critical** | Remote unauthenticated abuse, arbitrary code execution, data loss, full add-on takeover, persistent DB corruption, or service that will not recover without manual intervention. |
| **high** | Authenticated-but-trivial misuse, silent incorrect forecasts published to HA, deadlock or hang under realistic load, secrets leak, schema drift that breaks upgrades. |
| **medium** | Functional bug that produces wrong UI output, swallows user actions, leaks resources slowly, or breaks under specific (named) conditions. |
| **low** | Narrow edge case, defensive gap, robustness issue that only fires in unusual environments. Include only if concrete. |

| Confidence | Meaning |
|---|---|
| **high** | You can point at the code and explain the trigger end-to-end. |
| **medium** | The trigger requires an assumption you have stated; the code path is real but you have not traced every input. |
| **low** | Strong smell, not a proven path — include only if severity is high or above, otherwise demote to blind spots. |

Drop anything that is `low` × `low`.

## Output format

Produce a single markdown document, no preamble. One `##` heading per
finding, ordered critical → high → medium → low (then by confidence).
Each finding looks like:

```
## [SEV/CONF] Short title
**Where:** `relative/path.py:123` (and any other sites)
**What happens today:** one paragraph, mechanism-level.
**Trigger:** what an attacker / user / scheduler does to reach this.
**Impact:** what the user sees, what state ends up wrong.
**Evidence:**
    <≤ 8 lines of code, trimmed>
```

No "recommendations" section per finding — you are auditing, not
prescribing. If a fix is genuinely one line and obvious, you may add a
**Fix sketch:** of *at most one sentence*. Skip it otherwise.

At the end, a single `## Blind spots / needs runtime verification`
section, bulleted. Each bullet: what you could not determine
statically, why, and what would resolve it (one HA instance run, one
manual log inspection, one Optuna trial, etc.).

## What to look for — concrete to this codebase

Treat the following as the **starting set**, not the whole set. Hunt
for anything else you spot. Where I name a file or symbol, that is the
first place to look, not the only place.

### A. Network exposure & auth surface

1. `ml-forecast-lab/config.yaml` declares **both** `ingress: true` and
   `ports: { 5052/tcp: 5052 }`. Everything the FastAPI app exposes is
   therefore reachable from the LAN, not just from authenticated HA
   ingress. Walk every mutating endpoint in `ml_forecast_lab/web/app.py`
   (every `@router.post`, including `/experiment/{name}/promote/...`,
   `/api/experiments/create`, `/api/experiments/{name}/delete`,
   `/api/settings`, `/api/experiment-settings`, `/api/models/params`,
   `/experiment/{name}/run-tuning`, `/experiment/{name}/run-pipeline`,
   `/experiment/{name}/toggle-mode`) and ask: if an attacker on the LAN
   hits this directly on port 5052, what damage can they do? Report
   anything that is destructive, costly (CPU-burning tuning runs), or
   writes to `mlfl.yaml` / `history.db` / `/data/.../models/`.
2. `web/app.py` configures `CORSMiddleware(allow_origins=["*"],
   allow_credentials=True, ...)`. Inspect how the middleware actually
   echoes origin headers — `*` plus credentials is normally rejected
   by browsers, but Starlette's behaviour with `allow_origins=["*"]`
   and credentials is worth confirming against the call site.
3. The "stub" path in `ml_forecast_lab/__main__.py:stub_server` binds
   `0.0.0.0:5052` with no auth and no rate limit if the main import
   fails. Confirm it cannot be triggered by anything an attacker can
   influence (e.g. removing a file in `/data`).
4. `HAInterface.ha_key` (`ha_interface.py:~210`) reads
   `SUPERVISOR_TOKEN` once at construction. Supervisor may rotate this
   on add-on restart, but not during a running session. Is there any
   path that would 401 silently and bury the failure inside a broad
   `except`?

### B. Custom-metric sandbox (asteval) and other eval surfaces

1. `benchmark/metrics.py:460–520` evaluates user-supplied Python
   expressions from `mlfl.yaml` `custom_metrics:` via `asteval`. The
   regex blacklist forbids `import`, `from`, dunder names, `open`,
   `exec`, `eval`, `compile` — but the interpreter is handed the full
   `numpy` module as `np`. Audit what is reachable through `np` and
   ndarray methods: `np.load`, `np.memmap`, `np.save`,
   `np.ndarray.tobytes`, `np.lib`, pickle round-trips, `__class__`
   chains. Report whether arbitrary file read/write or code execution
   is reachable, and whether the web layer has any path that lets a
   LAN attacker (see A.1) write into `custom_metrics`.
2. Search for any other dynamic-code surface: `eval(`, `exec(`,
   `compile(`, `pickle.load`, `yaml.load(` (vs `safe_load`),
   `subprocess`, `os.system`, `os.popen`, shell=True. Confirm the
   tree-model save/load (`models/lightgbm_backend.py` pattern of
   pickling the entire model object plus metadata) does not load a
   pickle from a path that user input can influence.

### C. SQLite layer (`ml_forecast_lab/db.py`)

1. Per-entity dynamic tables: the table name is derived from
   `entity_id` via regex sanitisation around `db.py:49-65`. Confirm
   the regex is tight (rejects unicode tricks, leading digits,
   reserved words, length bombs) and that **every** dynamic-table
   query uses the sanitised name. Grep for f-string SQL.
2. Single `sqlite3.Connection` with `check_same_thread=False` shared
   across the event loop and the executor thread pool, guarded only by
   `threading.RLock`. Look for any code path that:
   - calls a DB method from within another DB method (re-entrant
     RLock saves you here, but only if every public method uses the
     same lock);
   - holds the lock across `await` (it should not — the RLock is
     `threading`-scoped);
   - executes long scans (`get_forecast_accuracy`,
     `get_forecast_stability`, `get_forecast_evolution`) and so
     blocks UI requests routed via `asyncio.to_thread`.
3. Schema migration: `ensure_forecast_log_table` performs `ALTER
   TABLE` adds for `upper`, `lower`, `model_version` columns. There
   is no `schema_version` table. Check for ordering hazards and
   whether a partial migration (e.g. interrupted) leaves the table
   in a state the rest of the code does not check for.
4. `forecast_log` pruning: `cleanup_forecast_log` deletes by
   `issued_at`. Confirm the time format and timezone match the format
   inserted at the call site. ISO strings sort lexically only if
   timezone-anchored consistently.
5. No `VACUUM`, no WAL checkpoint. Over months of operation the
   `-wal` file can dwarf the DB. Note the size-growth path.
6. Concurrency: WAL allows concurrent readers, but the RLock
   serialises them. Calling a heavy read while a heavy read is in
   flight will stall both. Verify whether benchmark logging during
   training contends with web UI queries.

### D. FastAPI / async correctness (`ml_forecast_lab/web/app.py`)

1. `AppState` is shared mutable state (benchmark_results,
   experiment_statuses, training queue, callbacks) **with no lock**.
   It is currently safe under a single asyncio loop, but two paths
   could break that:
   - any `loop.run_in_executor` / `asyncio.to_thread` that mutates
     `AppState` from a worker thread;
   - any future `uvicorn --workers >1`.
   Identify every mutation site and tag any that would race today
   versus those that are latent-only.
2. Fire-and-forget tasks (`asyncio.create_task(...)` at roughly
   `web/app.py:1188, 1329, 1613, 2330, 2415, 3158` and similar). No
   strong reference is held, so a finished task may be garbage-
   collected before its exception is logged. Exceptions in promote,
   covariate-analysis, tuning, and the training-queue processor can
   vanish. Confirm and report.
3. Endpoints that perform synchronous file I/O on the event loop
   (YAML rewrites in `/api/settings`, `/api/experiment-settings`,
   `add-covariate`, etc.). The files are small but writes occur
   inside `async def`. Note only if there is a plausible blocking
   length, not as style.
4. Exception messages echoed to the client (`return JSONResponse(
   {"success": False, "error": str(e)})`). Audit whether any of those
   paths can expose internal paths (`/data/...`), entity names with
   secrets, or HA token fragments.
5. `/api/ha/entities`: 60s in-memory cache (around line 1713–1720).
   Confirm there is no per-request rebuild that would let a LAN
   attacker hammer the supervisor token endpoint.
6. SSE training stream (`/experiment/{name}/training-stream`,
   `web/app.py:3057`): replay buffer, keep-alive, queue per
   subscriber. Look for unbounded queue growth if a slow client never
   disconnects, and missing cleanup on disconnect.
7. The 50+ routes share the `_get_base_path` ingress-path helper. If
   any URL is constructed without it, links break under ingress only
   — not under direct port. Spot-check rendered HTML for hard-coded
   `/experiment/...` hrefs that should be ingress-prefixed.

### E. ML correctness

1. **No global RNG seeding.** Only `catboost_backend.py` hardcodes
   `random_seed: 42`. `np.random.seed`, `torch.manual_seed`,
   `random.seed` are absent. Confirm that the "champion" selection
   in `benchmark/runner.py:781–806` (composite rank) is stable
   enough that this does not flip the production model between runs
   on identical data when scores are close. Quantify the risk by
   pointing at the ranking code.
2. **Walk-forward / sliding-window CV** (`benchmark/runner.py:206–
   328`): embargo applied only to the training side. Confirm there
   is no other path through `features.py` (lags, rolling windows)
   that re-introduces target leakage past the embargo. Pay attention
   to `create_sliding_windows` and rolling means computed over
   windows that straddle the train/test boundary.
3. **Cumulative → interval** (`preprocessing.py:cumulative_to_interval`,
   around line 80–140). The first difference is set to 0. Confirm
   downstream code never treats that 0 as a real interval (e.g. it
   does not get fed into log/MAPE/sMAPE as a true zero). Also check
   gap-aware scaling (`clip(lower=1.0)`) when the cumulative counter
   has gaps — extrapolation in the wrong direction.
4. **Log/sqrt/box-cox transforms** (`preprocessing.py:apply_transform`
   ↔ `invert_transform`). For each transform, confirm
   `invert(apply(x)) == x` for in-domain inputs and that the inverse
   does not silently coerce negatives to zero. The `np.maximum(...,
   0.0)` after `expm1` (e.g. `main.py:~2447, ~3964`) discards
   information.
5. **Load-subtract** (`preprocessing.py:apply_load_subtract`): clips
   to 0 (lossy) and raises `LoadSubtractError` only above a
   fraction-of-load threshold. Confirm the error surfaces to the UI
   rather than being swallowed inside `_run_benchmark` /
   `_retrain_and_cache`.
6. **Future covariates**: `covariates.py:182–200` returns NaN as a
   placeholder for the forecast attribute. The example `mlfl.yaml`
   documents `role: future` and `predbat.rates` as supported. Is
   future-covariate role wired anywhere downstream, or is it a
   documented feature that does not actually work?
7. **Conformal intervals**: `get_conformal_quantiles` pins to the
   current `model_version` first and falls back to all-versions pool
   if < 10 calibrated residuals. Audit the threshold and the
   fallback path — small calibration sets produce silently wide /
   narrow intervals, and there is no observability for which path
   fired.
8. **Forecast-log cohort cleanup on promotion** (around `web/app.py:
   1156–1178`): if a user promotes a new champion, prior rows are
   cleared. Confirm the deletion scope: experiment + model name +
   model version, not just experiment.
9. **Model persistence path traversal**: model save/load paths are
   constructed from `exp_cfg.name`. Confirm `exp_cfg.name` cannot
   contain `..` or `/` by the time it reaches the filesystem (look
   at `config.py:create_experiment` and the name validator).
10. **Tree-model pickle**: pickled objects loaded from
    `/data/.../models/<exp>/model.bin` execute attacker-controlled
    code if anything else can write to that path. Confirm the only
    writer is the add-on itself.

### F. Home Assistant integration (`ha_interface.py`)

1. `parse_timestamp` handles a long list of ISO variants. Look for
   any path where a None / empty / malformed timestamp from HA
   silently becomes "now" or epoch zero and feeds into training.
2. `set_state` (`POST /api/states/{entity_id}`) — confirm
   forecast arrays do not exceed HA's recorder / database size
   limits. Look for the size of `forecast` attribute arrays published
   per cycle (`future_periods: 96` × half-hour × multiple sensors per
   experiment).
3. `get_history` 180-second read timeout: under what conditions does
   this hang the main loop versus the executor? Trace the call from
   `main.py:_fetch_and_preprocess` upwards.
4. Token leakage: confirm `ha_key` is not logged or rendered in any
   error path, exception message, or template.

### G. Concurrency / lifecycle hazards in `main.py`

1. `_training_lock: asyncio.Lock` serialises all training. Confirm
   the lock is released on every exception path, including
   cancellation from `_stop_training_trigger` and uvicorn shutdown.
2. `_retrain_queue: asyncio.Queue`: unbounded but de-duplicated by
   an `already_queued` flag. If the consumer dies, items pile up
   silently. Confirm the consumer is restarted or the failure
   surfaces.
3. Signal handlers (SIGTERM/SIGINT) set `self.running = False`
   (`main.py:~5007`). Confirm the inner `await asyncio.sleep(1)` in
   `main_loop` checks the flag promptly, and that uvicorn is asked
   to shut down rather than being orphaned.
4. Config reload every 30 s by mtime comparison (`main.py:~5063`):
   confirm that a partial write to `mlfl.yaml` (between truncate
   and final write) cannot be picked up. The atomic-write helper in
   `config.py:24` mitigates this only if every writer uses it —
   verify the web layer always goes through that helper.
5. `_cached_models` dict: keyed by `exp_cfg.name`. If a user renames
   an experiment, the old entry is orphaned. Memory growth over the
   lifetime of the process. Note only if the keys can grow without
   bound.

### H. Config and validation (`config.py`)

1. Custom-metric expressions stored in `mlfl.yaml`. The web UI has
   no editor for them (verify), so the only path to add one is to
   edit the file. Confirm — if there is a hidden endpoint, that
   ties to B.1.
2. Unknown-field handling: fields are stripped with a warning
   (`config.py:~666, 688, 706–707, 725–727`). Confirm this never
   silently drops user-meant config because of a typo on a security-
   relevant field (e.g. `mode: "production"` typoed as `mode:
   "Production"` — case-sensitive comparison?).
3. Validator coverage for `ExperimentCfg.__post_init__`
   (`config.py:468–525`). Spot-check that every field with a fixed
   choice set is validated, and that numeric bounds (`cv_folds`,
   `cv_embargo_periods`, `days_history`, `interval_minutes`) reject
   nonsense values before they reach the runner.

### I. Build / s6 / packaging

1. `Dockerfile` multi-stage: builder stage installs compilers,
   runtime stage installs `libgomp1` only. Confirm every wheel in
   `requirements.txt` produces a self-contained binary (no runtime
   dlopen of `libstdc++` etc. that would fail in the slim stage).
2. `rootfs/etc/s6-overlay/s6-rc.d/init-mlforecastlab/run` searches
   `/addon_configs/ml_forecast_lab`, `/addon_configs/*_ml_forecast_lab`,
   `/config` and picks the first matching directory. Confirm the
   glob cannot match an unintended directory (e.g. another add-on's
   config) and that the copy of `mlfl.yaml` does not overwrite a
   user file. (The script gates on `[ ! -f "${ADDON_CONFIG}/mlfl.yaml" ]`
   — confirm.)
3. `s6-overlay` service type is `longrun` with `up` pointing at
   `run`. The script ends with `exec python3 -m ml_forecast_lab`. If
   the Python process exits non-zero, s6 will restart it. Confirm
   there is no scenario where the bashio prelude fails and s6 spins
   into a tight restart loop (CPU on user's HA host).
4. `init: false` is set. If `bashio` ever needs the supervisor init
   sequence it expects, this will surface as missing env vars at
   start.

### J. Logging and observability

1. `__main__.py` uses a single rotating handler at
   `/data/ml_forecast_lab/logs/mlfl.log` (5 MB × 5). With INFO
   logging across BENCH / MODEL / WEB phases on a Pi, this can roll
   inside a single training cycle. Confirm whether anything writes
   exception tracebacks at INFO instead of ERROR (chews through log
   budget fast).
2. `_parse_log_level` falls back to INFO silently on unknown input.
   Confirm the bashio `LOG_LEVEL=trace` is mapped to `DEBUG` (it
   should be) and that `OFF` does not produce a half-silenced state.
3. Any path that includes the `SUPERVISOR_TOKEN` or full HA URLs
   with credentials in a log line.

### K. Frontend templates and Plotly embedding

1. Confirm every JSON blob embedded in `<script>` uses `|tojson`
   (the survey says it does, but verify the failure modes — `|safe`
   on user-controlled data, raw `{{ }}` inside JS).
2. Plotly basic is vendored at `web/static/plotly-basic.min.js`. If
   any chart needs a non-basic trace (mapbox, 3D), it will fail
   silently in the browser. Note only if a chart construction site
   uses a trace type the basic bundle does not ship.
3. SSE EventSource on the training tab — confirm it cleans up
   subscribers on page navigation (server side, see D.6).

## Audit hygiene

- Do not duplicate findings. One issue, one heading. If the same
  defect manifests at three call sites, list them all under one
  finding under **Where:**.
- Do not pad. A 200-word finding is fine; a 1000-word finding means
  you are mixing several issues or speculating.
- If you cannot decide between two severities, pick the lower and
  explain in one sentence.
- If a section above turns up nothing, do not write "no findings" —
  silence is a clear signal you looked and found nothing.
- The blind-spots section must be specific: "I could not determine
  whether `_run_benchmark` releases `_training_lock` if uvicorn is
  killed mid-fit, because the exception path goes through three
  `except Exception` layers — running the add-on under SIGTERM during
  training would prove it" beats "concurrency is hard".

When you are done, save your findings as `AUDIT_FINDINGS.md` at the
repo root. Do not commit it.
