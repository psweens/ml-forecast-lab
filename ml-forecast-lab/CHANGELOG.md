# Changelog

## 2.27.9

### Changed

- **Night-time solar forecasts now go to zero via the model, not a
  post-hoc clamp.** Tree backends (CatBoost / LightGBM / XGBoost) do
  recursive multi-step forecasting — each step's prediction feeds
  forward as `y_lag_1` of the next step. A sunset step biased even
  ~100 W leaks into the next step's lag, producing a feature vector
  `(clear_sky_ghi=0, y_lag_1=150)` that training never saw (training
  data cleanly had `y_lag_k=0` whenever the past step was night). The
  tree then lands on a daytime leaf and predicts non-zero across the
  whole overnight horizon. Fixed at the feature contract: when
  `clear_sky_ghi` is present in the dataframe, `build_features` zeros
  `y_lag_k` and `y_diff_1` values whose corresponding shifted GHI is
  zero. The recursive inference paths
  (`_forecast_with_cached._run_recursive_forecast` and
  `_run_production_inference._run_recursive_forecast`) mirror the
  invariant by pushing `0` into the lag buffer at night steps instead
  of the raw prediction — so every future feature vector stays
  in-distribution and the tree's own learned response drives the
  published forecast down to near-zero through dusk, smoothly.

- **Removed the v2.27.8 hard clamp in `_publish_forecast_sensors`.**
  With the feature-contract fix above, the output-side clamp is no
  longer needed and was creating a visible cliff at the exact moment
  `clear_sky_ghi` crossed zero. Any residual non-zero at night after
  the next retrain reflects a genuine model error (or a
  non-solar-gated backend path) that shouldn't be hidden.

- **Neural models are unaffected by this change.** They exclude
  `y_lag_*` from their covariate set (see
  [main.py](ml-forecast-lab/ml_forecast_lab/main.py) — the sliding-
  window covariate cols subtract `y_lag_*`) and predict all horizons
  in a single forward pass, so they don't have the recursive
  drift pathology. Their inference already uses real history, which
  is cleanly zero at night. If a neural backend shows a residual
  night floor, that's the softplus output head — switchable via
  `output_activation: relu` on the experiment.

## 2.27.8

### Fixed

- **Solar forecasts no longer carry a noise floor at night.** Tree
  ensembles (CatBoost, LightGBM, XGBoost) can't output exact zero —
  any non-zero night samples in training history (inverter idle draw,
  meter noise, single-digit watts) leave a leaf bias the forest can't
  unlearn, so predictions hover around ~100-300 W even when
  `clear_sky_ghi = 0` and actual PV is exactly 0. Softplus-headed
  neural models have a ln(2)-scale floor for the same visual effect.
  Added a physics gate in `_publish_forecast_sensors` that clamps
  `y_pred` (plus conformal upper/lower bands) to exactly 0 on every
  horizon step where pvlib's clear-sky GHI is 0. Activates only when
  `include_clear_sky_irradiance: true` is set on the experiment — the
  unambiguous signal that the target is solar-driven — so non-solar
  signals are unaffected. Sited in the shared publish helper so
  `/promote`, cached forecast cycles, and the full
  `_run_production_inference` path all pick it up uniformly.

## 2.27.7

### Changed

- **Production cached models now survive restarts.** After every
  `_retrain_and_cache`, the trained model is serialised to
  `/data/ml_forecast_lab/models/<exp>/model.bin` (via the backend's
  own `save()`) alongside a `cache_meta.json` holding
  `feature_cols`, `trained_at`, `model_version`, `is_neural`, and
  `window_size`. On startup, `_restore_cached_models` loads each
  production experiment's cache before `main_loop` decides its
  timers. If the restored cache is younger than
  `retrain_every_hours`, the immediate startup retrain is skipped
  and the next retrain is scheduled at `trained_at +
  retrain_every_hours` instead. This eliminates the cascade of N
  sequential retrains every restart used to run through before the
  UI became responsive — the user experience on a clean restart
  drops from "minutes of 'no best model' placeholders while each
  production experiment retrains in turn" to "sensors publish on
  the next forecast tick, UI shows the restored champion
  immediately". Persistence failures and schema mismatches fall
  back silently to the old cold-start behaviour.

- **`_forecast_with_cached` no longer requires a cached training
  frame.** Previously the fresh-fetch-failure branch reached into
  `cache["combined"]` as a fallback; restored caches deliberately
  don't carry that frame (too big to pickle, always re-fetchable),
  so the code now `cache.get("combined")` and skips the cycle
  cleanly when both fresh-fetch and cached-fallback are unavailable.

## 2.27.6

### Fixed

- **Publish button now triggers an immediate retrain.** The
  `/experiment/{name}/promote/{model_name}` handler flipped the
  experiment to `production` mode and persisted the choice to YAML,
  but — unlike the sibling `/toggle-mode` handler — never fired
  `retrain_callback`. The experiment sat in production with no cached
  model until the next scheduled retrain tick (up to
  `retrain_every_hours` later), so sensors didn't appear and the
  Publish click looked like a no-op. Now fires the retrain callback
  at the end of `promote_model`, matching `toggle_mode:2277-2280`.

## 2.27.5

### Fixed

- **CatBoost tuning trials no longer appear to hang for tens of
  minutes.** CatBoost builds oblivious (symmetric) trees, so every
  depth level strictly doubles per-tree cost — `max_depth=16` is
  ~1000x slower per tree than `max_depth=6`, and Optuna will happily
  pick it. Tightened the CatBoost search space to match the library's
  practical range (`max_depth` 3-10, `n_estimators` 10-2000) so a
  single trial can't out-grow the 30 min study budget. Defaults are
  unchanged (depth=6, n_estimators=500, lr=0.05) so existing results
  aren't affected. The v2.27.4 iteration callback remains active and
  drives the live training UI for the benchmark and production-training
  paths — it's only the tuning path that never wired `epoch_callback`
  through `runner.run_single_model`, which is why the callback alone
  didn't surface in-trial progress.

### Changed

- **Tuning logs now emit a `Trial N starting: params=...` line per
  trial.** Previously only the completion line logged, so an in-flight
  trial was invisible in the log and indistinguishable from a hang.

## 2.27.4

### Fixed

- **CatBoost tuning no longer looks stalled.** CatBoost emitted a single
  end-of-training event, so the live training UI showed nothing at all
  during each trial even while the model was iterating. Added a CatBoost
  iteration callback (matching the LightGBM / XGBoost pattern) that
  forwards per-tree RMSE plus a synthetic patience counter to the
  training event bus.

- **Tuning now has a 30 min wall-clock budget for every backend.** The
  timeout previously only fired for neural models; tree tuning could
  sit for an hour inside a single trial when Optuna picked a tiny
  learning rate plus a large `n_estimators` (CatBoost in particular),
  with no progress signal to distinguish "training slowly" from "hung".

### Changed

- **Model tabs render alphabetically.** Both the main Models page and
  the per-experiment Models tab now order cards by display name,
  regardless of the insertion order in `MODEL_CATALOG`. Baseline and
  Classical models (Seasonal Naive, AutoARIMA / AutoETS / AutoTheta)
  interleave cleanly with the neural backends.

## 2.27.3

### Fixed

- **Accuracy publish cycle no longer freezes the event loop.** Each
  publish cycle called `get_forecast_accuracy` inline — a 3-CTE query
  over the actuals table that blocked the web UI and the HA `set_state`
  pool for the full scan duration. Now offloaded via
  `asyncio.to_thread`, with the SQLite connection opened
  `check_same_thread=False` behind an `RLock` and WAL journaling so the
  offloaded reader doesn't contend with the publish-cycle writer.

- **/forecast-accuracy skips empty-filter fallbacks up-front.** The
  endpoint's fallback ladder could run the heavy accuracy query up to
  three times when the strict `(model_name, model_version)` filter
  returned empty. Added `HistoryDB.probe_forecast_rows()` — an
  O(log N) `EXISTS` check served by the
  `(experiment, model_name, model_version)` index — which picks the
  narrowest filter with data first, so only one full accuracy query
  runs per request.

## 2.27.2

### Added

- **Covariate manifest log per preprocess cycle.** `_fetch_and_preprocess`
  now emits a single INFO block per experiment summarising every
  covariate's contribution: traffic-light status (`✓` / `⚠` / `✗`),
  configured role (`lagged` / `both` / `future` / `physics`), post-dropna
  coverage %, staleness of the most recent raw value, and Pearson
  correlation with the target on the aligned frame. A final `dropna:
  A rows → B kept (C lost; biggest culprit: <col> N NaNs)` line names
  the column whose NaNs deleted the most rows — the specific field
  that pinpoints "the whole experiment returned zero samples because
  one covariate is misconfigured". Solar physics features are included
  under `role=physics` so the block is a complete record of what the
  training frame actually contained. Mirrors the existing publish
  manifest pattern from 2.26.4: scattered per-covariate `✓ raw → aligned`
  INFO lines are kept for backwards compat, and the consolidated block
  sits alongside them. Diagnostic shape:

  ```
  Covariate manifest for optimised_solar (9 configured, +2 physics):
    ⚠ sensor.solcast_pv_forecast_forecast_today [future]  cov=100.0%  stale=6.2h  corr=+0.71  ← stale>interval×4
    ✓ sensor.openweathermap_cloud_coverage [lagged]  cov=99.8%  stale=12m  corr=-0.34
    ✗ sensor.carlton_green_south [both] — fetch failed: HTTP 500
    dropna: 8,760 rows → 8,740 kept (20 lost; biggest culprit: clear_sky_ghi 14 NaNs)
  ```

- **TiDE surfaces when its future-covariate path is inactive.** The
  existing `TiDE: future-covariate path active (N features × H horizons)`
  INFO line only fired when the caller passed `future_covariates=`; the
  silent `else` branch left users debugging "why is TiDE losing to
  Seasonal Naive on a signal with `role: future` covariates?" with no
  log evidence that the caller never wired the tensor up. An equivalent
  INACTIVE log now fires, explicitly stating that `role=future`
  covariates aren't being routed and that TiDE is degraded to a
  past-only encoder-decoder. Lets the covariate manifest (which shows
  `role=future` for Solcast/tariff sources) and the TiDE log sit
  side-by-side so the gap is immediately visible.

## 2.27.1

### Fixed

- **HA API calls no longer cascade-fail when Home Assistant is slow.** A
  single `total=30s` wall-clock budget in `HAInterface.api_call` covered
  DNS, connect, send and full response read, so a slow recorder query for
  `/api/history/period` (routine on benchmark loads with weeks of data)
  burned the whole budget and triggered `asyncio.TimeoutError`. Once that
  first call stalled the shared connector, concurrent `set_state` POSTs
  fired by `_publish_one` then timed out at the DNS-resolution phase —
  reported by users as a flurry of `mlfl_*` publish failures in the same
  second.

  Split into separate `sock_connect` (15s) and `sock_read` (30s default,
  180s for history queries) budgets so a slow body can't starve the
  connect phase, added exponential-backoff retry (3 attempts, 1s / 2s) for
  `TimeoutError` / `ClientError` / 5xx, and made both timeouts and retry
  count overridable per call. 4xx is treated as non-transient and is not
  retried.

## 2.27.0

### Added

- **Eight new model backends, taking the catalogue from 14 to 22.** All
  registered automatically and surfaced in the Models page, the
  hyperparameter editor, and the Demšar composite ranking on equal footing
  with the existing backends.

  Tier 1 (state-of-the-art lightweight):
  - **CatBoost** (`catboost`) — third tabular gradient boosting backend
    alongside LightGBM and XGBoost. Ordered boosting + oblivious symmetric
    trees often wins on noisy or covariate-rich smart-home signals.
  - **NLinear** (`nlinear`) — companion to DLinear from Zeng et al. 2023.
    Subtracts the last value, applies a single linear layer, adds it back.
    Tiny but a top-tier baseline that's strange to ship without.
  - **FITS** (`fits`) — frequency-domain low-pass complex linear, ICLR 2024
    ("Modeling Time Series with 10k Parameters"). The lightest neural
    backend in the catalogue.
  - **TimeMixer** (`timemixer`) — multiscale season/trend mixing with
    cross-scale interaction (Past-Decomposable-Mixing block), ICLR 2024.
    Currently leading several long-horizon benchmarks.

  Tier 2 (gold-standard baselines):
  - **GRU** (`gru`) — lighter recurrent baseline than LSTM. Same temporal
    attention + multi-horizon pipeline; ~25% fewer parameters per cell.
  - **TFT** (`tft`) — compact Temporal Fusion Transformer (Lim et al. 2021)
    with Variable Selection Network → LSTM encoder → interpretable
    multi-head attention. Static / known-future covariate branches are
    omitted (rare in HA sensor data).
  - **Seasonal Naive** (`seasonal_naive`) — the reference baseline every
    forecasting paper requires. ŷ[t+h] = y[t+h-period], no training. Now
    that it's a registered backend it shows up in the Demšar ranking, so
    every other model has to actually beat it to look good.
  - **statsforecast classical baselines** (`arima`, `ets`, `theta`) —
    AutoARIMA, AutoETS, AutoTheta from Nixtla's numba-JIT-compiled
    statsforecast. The classical statistical reference points the academic
    literature treats as mandatory comparisons.

  New dependencies (auto-pulled by `requirements.txt`): `catboost>=1.2`,
  `statsforecast>=1.7`. Each backend gracefully degrades to a clear
  RuntimeError if its underlying library isn't available, matching the
  existing optional-backend pattern.

## 2.26.7

### Fixed

- **Root cause of the Forecast-log button failure: HTML autoescape on
  the `EXP_UNITS` fallback emitted `var EXP_UNITS = &#34;&#34;;` for
  any experiment without a `units:` value (the default — `units: str
  = ''` in `ExperimentCfg`).** `experiment.html` had `var EXP_UNITS =
  {{ units | tojson if units else '""' }};`. The truthy branch goes
  through `|tojson`, which returns a Markup-safe string — autoescape
  leaves it alone. The else branch returned a plain Jinja literal
  `""`, which Jinja's HTML autoescape rewrote to `&#34;&#34;` on the
  way into the `<script>`. The browser's JS parser does not decode
  HTML entities inside `<script>`, so it threw `SyntaxError:
  Unexpected token '&'` at that line, aborting the IIFE before any
  of the `window.X = …` assignments below it ran — including
  `window.loadForecastLogStats`. The button's onclick then hit a
  `ReferenceError: Can't find variable: loadForecastLogStats`, which
  the v2.25.3 defensive fallback faithfully rendered. Replaced the
  conditional with `{{ (units or '') | tojson }}` so it always pipes
  through the Markup-safe filter. v2.26.6's button colocation +
  error capture turn out to be useful belt-and-braces for any future
  variant of the same class of bug, but this fix kills the actual
  trigger every existing user is hitting.

## 2.26.6

### Fixed

- **Forecast-log inspector button threw "Can't find variable:
  loadForecastLogStats" under real-world conditions.** v2.25.3 added a
  defensive try/catch around the button's onclick because the handler
  sometimes silently did nothing, but couldn't reproduce the underlying
  cause. The actual mechanism: `window.loadForecastLogStats` was assigned
  ~1300 lines into the big IIFE in `{% block scripts %}`, so any earlier
  throw during IIFE evaluation (bad Jinja-interpolated value, localStorage
  access denied, missing `Intl` features, etc.) aborted execution before
  the assignment, leaving the button's onclick referencing a symbol that
  never came into existence. The handler is now defined in its own
  `<script>` block right next to the button, with the URL rendered
  directly via Jinja rather than relying on closure vars from the main
  IIFE, so an unrelated failure elsewhere in the scripts block can't
  take the diagnostic button out.
- **Surface the root-cause error, not just the downstream symptom.** A
  top-of-block `window.addEventListener('error', …)` now stashes the
  first unhandled script error on `window.__mlflFirstError`. The
  Forecast-log button's catch appends it when the handler is still
  missing, so the next time this class of failure shows up we see the
  actual throw — not just "Can't find variable: loadForecastLogStats".

## 2.26.5

### Fixed

- **`_forecast_running` flag leaked on early returns, freezing forecasts
  for some experiments.** 2.26.2 moved the slot reservation synchronously
  into the scheduler (`self._forecast_running[name] = True` before
  `asyncio.create_task(self._forecast_single(exp_cfg))`) to close a
  same-tick double-fire race, but `_forecast_single` still had two
  early-return branches — non-production mode, and `name not in
  _cached_models` — that returned *before* the `try/finally` that clears
  the flag. In a multi-experiment setup, retrains are queued sequentially
  via `_retrain_queue`, so at the first forecast tick only the
  experiment whose retrain finished first has a cached model; the
  others hit the `no cached model` return and their `_forecast_running`
  flag stays `True` forever. From then on the scheduler's
  `not self._forecast_running.get(name, False)` guard skips those
  experiments on every tick, silently freezing their entire sensor
  family (`_forecast`, `_cumulative`, `_upper_80`, `_lower_80`,
  `_forecast_accuracy`) while other experiments keep updating — the
  "next forecast doesn't always create a new forecast for some of the
  sensors" symptom. The `try/finally` now wraps the full body so the
  flag is always cleared, regardless of which exit path the task takes.

## 2.26.4

### Added

- **Publish manifest log per cycle.** `_publish_forecast_sensors` now
  emits a single INFO line summarising every expected sensor's outcome
  — `published=N/total`, plus a `skipped` list with reason codes
  (`source_not_cumulative`, `no_conformal_bands`) and a `failed` list
  with per-entity exception context. Diagnoses the "`sensor.X
  last_updated = 3h ago` but the log says publish succeeded" symptom
  by making silent conditional skips visible: if a sensor has been
  out of the payload for hours, every cycle's manifest now shows
  exactly why. Replaces the old split "Published X/N sensors … / Failed
  to publish …" pair with one coherent report.

- **`_forecast_accuracy` sensor is always published.** Previously
  gated on `self.history_db and exp_cfg.mode == "production" and
  ltc.get("lead_minutes")`, so the HA entity didn't exist at all
  during cold start, in lab mode, or when the forecast_log query
  returned no matches yet — forcing dashboards to check for entity
  existence before binding. The sensor now lands in HA from day one
  with `state=0`, empty `lead_hours` / `mae` / `rmse` /
  `sample_count` arrays, and a new `status` attribute whose value
  names the current readiness state: `accumulating` (cold start),
  `no_history_db`, `lab_mode`, `error` (query failed), or `ready`
  once `lead_time_curve` has samples. Transitions to
  `status=ready` populate the arrays in place, so a dashboard
  binding survives the entire lifecycle without conditional logic.

### Changed

- **Add-on permissions tightened in `config.yaml`.** Removed
  `hassio_api: true`, `hassio_role: admin`, and `auth_api: true` —
  the add-on never calls the Supervisor or auth APIs and does not
  need admin role to publish sensors via the Core REST API, which is
  covered by `homeassistant_api: true`. `map` is narrowed from
  `homeassistant_config:rw`, `media:rw`, `share:rw`, `ssl:rw` to a
  single `homeassistant_config:ro` (alongside the existing
  `addon_config:rw` used for the persistent cache and logs). The
  add-on never wrote outside `/data` under the old mapping, so
  dropping write access is a no-op for behaviour but a meaningful
  narrowing of blast radius. Added `stage: stable` so HA surfaces it
  correctly in the add-on store.

- **Removed unused `models` add-on option.** The `options.models`
  list and its `schema` entry in `config.yaml` were legacy scaffolding
  from before the `mlfl.yaml` experiment-config model was introduced;
  nothing reads them. Removed both; add-on options UI now shows only
  the live `log_level` setting.

- **Docs moved to top-level `docs/`.** `CONFIG_GUIDE.md`,
  `FEATURES_GUIDE.md`, and `PREPROCESSING_GUIDE.md` have migrated
  from `ml-forecast-lab/ml_forecast_lab/` (Python package dir) to
  the repo's top-level `docs/` directory so they are a) no longer
  bundled into the Docker image, b) linkable directly on GitHub
  without going three levels deep. Deleted the stale
  `CORE_MODULES_README.md`, `CREATION_REPORT.md`, `INDEX.md`,
  `MODULES.md`, and `README_MODULES.md` files that had drifted out
  of sync with the current architecture (still referenced
  NeuralProphet, 5-backend counts, legacy `publishing.py`, pre-2.26
  sensor names).

- **`repository.yaml` url / maintainer corrected.** URL pointed at
  the archived `psweens/ha-addons` repo; now points at
  `psweens/ml-forecast-lab`. Maintainer name normalised to `Dr Paul
  W. Sweeney` to match the footer and git config.

- **README.md refreshed.** Model count corrected from 15 → 14
  (NeuralProphet removed in 2.24.0 but the count wasn't caught
  across the feature table, config example comment, architecture
  diagram, and dependencies list). Added feature sections for
  uncertainty intervals (conformal bands), forecast accuracy
  tracking with revision improvement, and load_subtract. Noted
  that neural backends' loss function (MSE / MAE / Huber) is
  configurable per experiment.

## 2.26.3

### Added

- **Phase-tag log formatter.** Every log line now carries a short `[BENCH]`,
  `[MODEL]`, `[WEB]`, `[APP]`, `[HA]`, `[DB]`, `[PREP]`, `[FEAT]`, `[COV]`,
  `[PUB]`, `[CFG]`, `[SOLAR]`, `[TRAIN]`, `[DASH]` or `[MLFL]` tag derived
  from the logger's module name, so `mlfl.log` can be filtered per
  subsystem with a simple grep. A new `_PhaseFormatter` in `__main__.py`
  injects the tag and the two handler format strings (`LOG_FORMAT` /
  `LOG_FORMAT_FILE`) left-align it in a 7-char column so the message
  column stays vertically aligned. No new dependencies — stdlib only,
  so the raw `mlfl.log` that the `/log` web endpoint streams back stays
  readable as plain text (no ANSI junk).

- **Per-fold and per-model progress markers in the benchmark runner.**
  With 14 backends × 5 folds = 70 inner iterations, the previous log
  went silent between `"Starting benchmark"` and the final leaderboard.
  Each fold now emits a start line — `[fold 3/5] lstm: train=N test=N` —
  and a completion line with the production metric plus train/infer
  timings — `[fold 3/5] lstm done: mae=0.0834 (train=11.3s, infer=0.42s,
  total=12.1s)`. The outer model loop adds a `[model 3/14] Running
  lstm` banner and a matching `[model 3/14] lstm finished in Xs`
  marker. Trial progress in the tuning path was already in shape and
  is unchanged.

- **Sensor-publish start line.** `_publish_forecast_sensors` now logs
  a single INFO line before it begins building payloads, naming the
  base entity, model, horizon (`N×Mmin (Xh)`), and the headline
  `next` value with units. Pairs with the existing success summary so
  a publish cycle has an obvious start and end in the log — previously
  you only saw the summary after the fact.

### Changed

- **Benchmark error logs now name the model.** `Feature building
  failed for fold N`, `Model training failed for fold N`, and `Model
  prediction failed for fold N` used to print only the fold index, so
  a 14-model run with one bad backend forced you to correlate timing
  to figure out which model crashed. They now include `model=<name>
  fold=<i>/<N>` so the failing backend is obvious.

- **Sensor-publish completion summary surfaces `next` and band
  width.** The `Published X/N sensors for <exp>` line now carries the
  headline `next=<val> <units>` and — when conformal bounds landed —
  `bands=±80%`, matching the start line so the pair brackets a
  complete picture of what went to HA.

### Fixed

- **Exception handlers that swallowed tracebacks.** 27 `logger.error`
  / `logger.warning` sites across `benchmark/runner.py` (4 fold/model
  failure paths), `db.py` (13 query/IO errors), `web/app.py` (16 API
  endpoint handlers), both tree model backends, `models/registry.py`,
  `covariates.py`, `ha_interface.py`, the heartbeat, the conformal
  band path, the forecast-accuracy prep, and a tuning trial-failure
  warning — all interpolated `{e}` into the message but never passed
  `exc_info=True`, so the actual stack trace was lost. All now
  propagate the traceback, turning "Failed to publish / query / load"
  warnings from one-liners into debuggable reports. The per-sensor
  publish failure uses the captured exception instance via
  `exc_info=err` so only failures with an exception print a traceback;
  `set_state returned False` cases stay concise.

## 2.26.2

### Fixed

- **Sensors within an experiment no longer drift apart in publish time.**
  `_publish_forecast_sensors` used to await each `set_state` call
  sequentially — `_forecast`, `_interval`, `_cumulative`, `_upper_{pct}`,
  `_lower_{pct}`, `_forecast_accuracy` — so HA stamped each entity with
  its own `last_updated`, spread across ~300 ms–1 s per cycle (worst
  when the accuracy sensor's DB read landed before its publish). All
  six set_state calls are now collected into a `payloads` list and
  fired together via `asyncio.gather`, so every sensor for one
  experiment lands at effectively the same HA `last_updated`.

- **Upper/lower interval bounds shared a single `try/except`.** If the
  upper publish raised, lower was silently skipped for that cycle,
  leaving the band one-sided until the next forecast. Each publish is
  now wrapped in its own `_publish_one` coroutine; a failure on one
  entity never suppresses its sibling.

- **`set_state` failures logged as successes.** `HAInterface.set_state`
  catches `RuntimeError` internally and returns `False` instead of
  raising. The publisher discarded that return value, so any failed
  POST (timeout, 5xx, stale SUPERVISOR_TOKEN) still emitted
  `"Published forecast curve to …"` at INFO. Return values are now
  checked; failures log `"Failed to publish …"` at WARNING and the
  experiment's `last_error` surfaces the count/list of missing
  sensors.

- **Three independent `datetime.now()` calls per publish cycle.** The
  forecast_log `issued_at`, the cumulative sensor's local-midnight
  boundary, and each sensor's implicit HA `last_updated` all came from
  separate clock reads. A single `issued_at` is now captured at the
  top of `_publish_forecast_sensors` and reused for `log_forecast`,
  the reset-daily partition, and as a new `issued_at` attribute on
  every sensor — so the full cycle has one consistent issuance time.

- **`last_trained` attribute only on `_forecast`.** It's now carried
  on every sensor (main, interval, cumulative, upper/lower, accuracy)
  via a shared `common_attrs` dict, alongside `model` and the new
  `issued_at`. Dashboards can correlate any sensor back to its
  training epoch without reading from the main entity.

- **Global `_forecast_running` flag allowed a same-tick double-fire
  race.** Because `asyncio.create_task` doesn't execute eagerly, two
  experiments scheduled in the same main-loop iteration could both
  pass the `not running` check before either task body set the flag;
  conversely, the one that finished first would flip the flag False
  while the other was still running. Replaced with a per-experiment
  `Dict[str, bool]` and a synchronous slot reservation in the
  scheduler before `create_task`, so each experiment's lock is
  independent.

- **Silent early-return when `ha_interface` was None or `y_pred` was
  empty.** Zero sensors published and the logs had no explanation.
  Now logs a WARNING naming which condition fired.

## 2.26.1

### Removed

- **Dead `publishing.py` module.** The legacy `publish_forecasts`
  helper and its friends (`make_entity_name`, `dict_from_series`,
  `daily_cumulative_series`, `energy_already_used_today`) were
  exported from the package but had no callers — the real publishing
  path is `_publish_forecast_sensors` in `main.py`. After the 2.26.0
  refactor consolidated `_daily_cumulative` into `_cumulative`, the
  stale module was also emitting the old entity with the pre-refactor
  attribute shape (`cumulative.timestamps/values` dict vs the new
  `forecast` list of `{datetime, value}` dicts), which would have
  diverged further over time. Deleted the module, cleaned up
  `__init__.py` imports/exports, and updated `MODULES.md` and
  `README_MODULES.md` to match. No behaviour change for users.

## 2.26.0

### Changed (breaking)

- **Publish sensors automatically, derive semantics from target config.**
  `publish_interval`, `publish_cumulative`, and `publish_daily_cumulative`
  have been removed from `ExperimentCfg`. The publishing path now always
  emits `_forecast` and `_cumulative`, and conditionally emits `_interval`
  when `source_is_cumulative` is true (otherwise it would just duplicate
  `_forecast`). The `_cumulative` sensor's behaviour is derived from the
  target's own semantics: if `source_is_cumulative` and `reset_daily` are
  both set, it resets at local midnight and is seeded with the current
  target value so the forecast meets actuals at the join point; otherwise
  it is a plain `cumsum` from zero across the horizon. A `resets_daily`
  attribute on the sensor exposes which branch was taken so downstream
  consumers (Predbat, chart cards) can adapt without parsing config.

  Migration: remove the three flags from your `mlfl.yaml` experiments —
  `source_is_cumulative` / `reset_daily` (which already exist to describe
  the target) now control the publishing behaviour too. The old
  `{prefix}{name}_daily_cumulative` entity has been consolidated into
  `{prefix}{name}_cumulative`; update any dashboards or automations that
  reference it.

## 2.25.5

### Fixed

- **Conformal-band sensors silently stopped updating** after the
  v2.24.0 upgrade introduced a `model_version` filter on
  `get_conformal_quantiles`. With the fresh version tag, only the
  handful of residuals written *since the latest retrain* matched
  the filter — far below the `min_samples` threshold the quantile
  estimator needs. `fallback_quantile` came back None, so
  `y_pred_upper` / `y_pred_lower` stayed None, `have_intervals`
  stayed False, and the `_upper_{pct}` / `_lower_{pct}` entities
  (plus the `forecast_upper` / `forecast_lower` attributes on the
  main `_forecast` sensor) stopped being written. HA continued
  showing their stale pre-upgrade values — matching the "some
  forecast sensors aren't updating" symptom while the main state
  value kept ticking over.

  Fix: after the version-filtered conformal query, check whether
  it returned a usable fallback quantile + ≥ 10 total residuals.
  If not, re-query pooled across all versions of this model so
  bands keep publishing during the post-retrain cold-start
  window. Logs one `Conformal bands: falling back to all-versions
  pool` INFO line per cycle when this kicks in, for visibility.

## 2.25.4

### Fixed

- **Results-tab model selection not surviving add-on restarts.**
  `/select-model` only updated in-memory state
  (`exp_status.selected_model`), never wrote to YAML. When the
  add-on restarted (which it does daily under the typical retrain
  cadence), `experiment_statuses` was re-initialised with
  `selected_model=None` and the next benchmark auto-promoted its
  top-ranked model. Users experienced this as "I picked XGBoost,
  reloaded the page, and it's showing LightGBM again."

  Three coordinated changes:
  - New `selected_model: Optional[str]` field on `ExperimentCfg`
    (YAML schema).
  - `/select-model` endpoint now persists to YAML via
    `save_experiment_field`, mirroring how `/promote` handles
    `production_model`. Response body includes `persisted: bool`
    so the frontend knows whether it saved.
  - Startup in main.py reads `exp_cfg.selected_model` into
    `ExperimentStatus.selected_model`, with fallback to
    `exp_cfg.production_model` for legacy configs that don't have
    the new field.
  - `/promote` also writes `selected_model` to YAML so the Results
    highlight stays in sync with the production switch across
    restarts.

### Tests

- New `TestSelectedModel` class (3 assertions) covering default
  value, explicit assignment, and YAML roundtrip.

## 2.25.3

### Fixed

- **"Load log summary" button silently doing nothing on some
  experiments.** Rendering the template through Jinja with both
  cumulative and instantaneous contexts produced structurally-
  identical JS (only `EXP_NAME` / `EXP_SOURCE_CUMULATIVE` /
  `expName` / `entityId` differed), so the root cause isn't in
  the template and is likely client-side (browser cache, ingress
  middleware, localStorage state on some setups). Rather than
  keep blind-debugging, added two defensive measures:
  - **Inline try/catch on the button's `onclick`** that writes the
    thrown error message into the `<pre>` output element, so the
    failure is visible without opening DevTools.
  - **"open in new tab" link** next to the button pointing at the
    `/forecast-log-stats` endpoint directly. Pure `<a href>` — no
    JavaScript required, works even if the button handler is
    broken by whatever the client-side state issue is.

## 2.25.2

### Fixed

- **Version-filter mismatch when `selected_model` ≠ `best_model`.**
  `ExperimentStatus.model_version` is a single field that tracks
  *whichever model was last retrained*. When a user had manually
  selected a non-champion model via the UI (e.g. picked `lightgbm`
  from a promote action, but the pipeline subsequently started
  retraining `xgboost`), the default filter the endpoint applied
  was `(model_name=lightgbm, model_version=<xgboost's timestamp>)` —
  an impossible combination that never has rows, forcing the
  fallback ladder to fire on every cycle and display stale
  pre-upgrade data. `_resolve_model_filter()` now only applies the
  version default when `default_model == best_model`; otherwise the
  name filter runs alone ("all versions of that model"), which is
  the correct semantic since no version info is tracked for
  non-champion models in the current schema. The same guard is
  mirrored in `/forecast-log-stats` so its reported
  `current_default_filter` matches what the analytics endpoints
  actually send.

### Added

- **Diagnostic fields on `/forecast-log-stats`**:
  - `selected_vs_best` showing `selected_model`, `best_model`, and
    `matches` — makes the exact mismatch condition above
    immediately visible.
  - `notes` array flagging common conditions: the
    selected-vs-champion drift above, or a default filter that
    points to a cohort with zero rows. Both are written as
    actionable sentences ("re-select `xgboost` in the UI or set
    `production_model: xgboost` in mlfl.yaml").

## 2.25.1

### Added

- **Forecast-log inspector** in the Layer 3 "Diagnostic tools"
  accordion on the Forecast Accuracy tab. A single "Load log summary"
  button fetches a new
  `GET /experiment/{name}/forecast-log-stats` endpoint and
  pretty-prints the JSON in-place, so users without shell access
  can debug "why is my chart empty?" situations directly from the
  web UI. The summary covers: total rows, the default filter the
  UI would apply, a per-`(model_name, model_version)` cohort
  breakdown with row counts and issued/target ranges, and the
  number of targets with ≥2 distinct issuances under the current
  champion+version filter (the stability query's minimum
  requirement). A hint sentence at the bottom explains how to
  interpret the three likely states.

## 2.25.0

### Added

- **Time-zone toggle on the Forecast Accuracy tab** (HA server /
  Browser / UTC). Remote users managing an HA instance in a
  different country (e.g. a California viewer of a UK HA) no longer
  have to mentally shift "the spike at Apr 16 02:00 PDT is
  actually UK mid-morning". On load the add-on reads HA's
  `time_zone` from `/api/config` and the toggle defaults to
  **HA server** so axis labels match when events physically
  happened. Chart tick labels are rendered via `Intl.DateTimeFormat`
  in the chosen TZ, with a TZ abbreviation baked into each axis
  title (e.g. "Target time (BST)"). Selection persists per
  experiment in `localStorage`.
- **HA-local day bucketing for the daily-total stability metric.**
  `SUBSTR(target_dt, 1, 10)` previously took a UTC-day prefix — on
  a BST-hosted HA instance the UK day boundary is UTC 23:00, so
  about an hour of UK-Monday demand was being filed under "Sunday"
  in the bar chart. `get_forecast_stability()` now accepts
  `day_offset_hours`, populated by the endpoint from
  `zoneinfo.ZoneInfo(tz).utcoffset()`, and the SQL shifts the
  timestamp by that offset before taking the day prefix. Correct
  everywhere except the single day containing a DST transition
  (where it's an hour off for that day only).
- **Add-on logging for the Forecast Accuracy pipeline.** Previously
  the whole Forecast-Accuracy write path was silent on success —
  there was no way to tell from the add-on logs whether
  `forecast_log` was actually accumulating rows, whether a schema
  migration had just run, or whether a UI chart was empty because
  of a version-fallback. Three new INFO-level log lines:
  - One per successful forecast cycle, e.g.
    `Logged 96 forecast_log rows for mixergy_demand
    (model=lgb, cached, v=2026-04-20T07:00:00Z, bands)`.
  - One per analytics fallback on `/forecast-accuracy`,
    `/forecast-stability`, `/forecast-trajectory`, naming the
    requested (model, version) and what the query widened to.
  - One per `ALTER TABLE` run inside `ensure_forecast_log_table`,
    listing the columns that were just added to legacy DBs.
- **`EXP_HA_TIME_ZONE` passed through to the experiment template**
  so Jinja doesn't need to assume anything about where the viewer
  is. Falls back to `null` when `/api/config` is unreachable.
- **Tests** — two new cases in
  `TestModelVersion::test_ha_local_day_bucketing_*` (shifts day
  labels correctly for BST; no-op for offset=0).

## 2.24.0

### Added

- **`model_version` column on ``forecast_log``** — an opaque,
  typically ISO-timestamp tag stamped by the pipeline each time a
  model finishes training. Cycles under the same ``model_name`` but
  different weight regimes now live in distinct cohorts, so the
  stability / accuracy / coverage / trajectory queries no longer
  silently pool v1 and v2 predictions together. Fixes the "I retrain
  every 24h under the same name and stability looks terrible"
  pattern that was the underlying cause of the overnight Apr-16
  spikes. Backwards-compatible schema migration: the column is added
  via ``ALTER TABLE ADD COLUMN`` on legacy DBs and existing rows
  carry NULL — they're treated as a separate cohort when a versioned
  cycle appears for the same experiment.
- **`?version=<tag>` / `?version=all` query params** on
  ``/forecast-accuracy``, ``/forecast-stability``, and
  ``/forecast-trajectory``. Endpoints default to the experiment's
  current training tag (``ExperimentStatus.model_version``), with
  ``?version=all`` escaping the default to show every weight regime.
  A two-step **fallback ladder** now runs when the default filter
  empties the result: first widen to "all versions of this model",
  then to "all models". Each fallback step is flagged back to the UI
  via ``model_fallback`` so the user knows what's being shown vs
  requested.
- **Model-version badge in the Layer 1 top controls** — the
  existing model badge now carries a ``since DD MMM HH:MM`` tail
  reflecting the timestamp of the current weights, with the full ISO
  tag on hover. On version fallback the badge reads ``<model> · all
  versions (fallback)`` in amber with an explanation in the tooltip.
- **`_resolve_model_filter()`** helper in ``app.py`` that factors
  the (model, version) resolution out of the three endpoints —
  single source of truth for "current champion + current weights by
  default, with escape hatches".
- **Training path stamps ``ExperimentStatus.model_version``** in
  ``_retrain_and_cache`` as soon as training completes, so the very
  next forecast is written under the new tag. The cached-model dict
  also carries ``model_version`` so ``_publish_forecast_sensors``
  can pin conformal-band residuals to the current weights and avoid
  post-retrain band inflation from stale residuals.
- **Tests** — 5 new cases in
  ``tests/test_forecast_analytics.py::TestModelVersion``:
  schema migration on a legacy table; ``log_forecast`` actually
  stamping the version; v1/v2 stability pooled vs v2-only (flipped
  median CV from 50%+ to <5%); NULL legacy rows correctly excluded
  from a version-filtered query; version filter propagating through
  accuracy / coverage / trajectory alongside stability.

### Changed

- **`/experiment/{name}/promote/{model_name}` is now idempotent.**
  Previously cleared ``forecast_log`` on every call regardless of
  whether the champion name actually changed, which was too
  aggressive — a user re-clicking promote on the already-current
  model would wipe their metrics history. The cleanup now gates on
  ``previous_model != model_name`` (matches the behaviour in
  ``_run_benchmark``). Response echoes the previous→new transition
  in the log line when a clear actually happens.
- ``clear_forecast_log_on_retrain`` (added in 2.23.0) is now
  defensive rather than load-bearing — the ``model_version`` column
  already segregates pre- and post-retrain cycles under the same
  model_name, so the cleanup is mostly for storage hygiene.

## 2.23.0

### Added

- **`stability_focus` experiment-config knob** that drives which
  run-to-run CV the Layer 1 verdict chip and headline sentence read.
  Two values:
  - ``per_moment`` (default) — median cross-cycle CV of predictions
    at the same target moment. Right when the downstream consumer
    cares about *when* demand hits: HVAC setpoints, pre-heat timing,
    battery dispatch.
  - ``daily_total`` — median cross-cycle CV of daily-total
    predictions (cumulative sensors only). Right when the downstream
    consumer integrates over the day: Predbat iBoost heating a hot-
    water tank, EV daily charging budgets, solar-export daily
    planning. For those use cases a ±50% per-moment swing can be
    fine as long as the daily total is stable, and the old
    per-moment-only chip was giving misleading "poor" verdicts.

  The Layer 3 accordion still surfaces both metrics regardless of
  focus — only the Layer 1 chip, headline, and swing-tile label
  follow this setting. Validator rejects ``daily_total`` on
  instantaneous sensors (summing them across a day isn't a physical
  quantity) and rejects unknown values.
- **`clear_forecast_log_on_retrain` experiment-config knob**
  (default ``True``) that prunes ``forecast_log`` rows issued before
  a champion promotion. Old cycles logged under the previous weights
  (even under the *same* model_name) pool into stability metrics and
  produce the "I retrained and now run-to-run looks terrible"
  artefact. Wired into two promotion paths:
  - ``/experiment/{name}/promote/{model_name}`` — the explicit UI
    promote action (app.py) now calls
    ``HistoryDB.cleanup_forecast_log()`` with a "now" cutoff,
    returning ``forecast_log_rows_cleared`` in the response.
  - The automatic champion-change path inside ``_run_benchmark``
    (main.py) does the same when the benchmark actually promotes a
    new name (no-op when the champion hasn't changed).

  Set to ``False`` to preserve full history for offline analysis.

### Tests

- New ``tests/test_config.py::TestStabilityFocus`` (4 assertions)
  and ``TestClearForecastLogOnRetrain`` (2 assertions) covering
  defaults, validation, and the cumulative-gating rule for
  ``daily_total`` focus.
- New ``tests/test_forecast_analytics.py::TestStability::
  test_cleanup_removes_pre_retrain_rows`` pins the promotion-time
  cleanup: seeded pre-retrain cycles with wildly different
  predictions produce CV > 50%; after ``cleanup_forecast_log()``
  removes them, the metric drops below 5%.

## 2.22.2

### Fixed

- **Daily-total stability metric inflated by a coverage artefact, not
  model disagreement.** `get_forecast_stability()` computed
  `SUM(predicted)` per `(issued_at, day)` and called the spread across
  cycles "daily-total CV". But each issuance covers a different
  *fraction* of any given day — an 08:00 run covers ≈32 bins of
  today, a 22:00 run covers 2 bins of today, a prior-day run covers
  all 48. Comparing a 2-bin SUM against a 48-bin SUM as "model
  disagreement" is just reading the coverage gap with extra steps,
  and on a Mixergy-style sensor where overnight carries most of the
  demand the artefact alone can produce 50–70% CV. The
  `per_cycle_day` pipeline now has a `day_max` sub-CTE and keeps only
  cycles whose `n_bins` equals the day's maximum — apples-to-apples
  across cycles. On a seeded three-cycle test (two full, one
  partial) this drops reported CV from 29.93% to 2.44% (~12×), with
  the reduction being entirely artefact.

## 2.22.1

### Fixed

- **Trajectory chart compared predictions to raw cumulative actuals**
  for cumulative sensors. `forecast_log.predicted` stores per-interval
  deltas (that's what the model is trained to emit), but the old
  trajectory query returned the actual as the raw grid-aligned value
  (e.g. 17% fill) — so the "Actual" line sat impossibly high above
  the per-interval forecast dots, turning a useful debugging view
  into a misleading one. `get_forecast_trajectory()` now takes
  `source_is_cumulative` and computes the actual as
  `value − value[t−interval]` (adjacency-guarded, matching the
  increment-mode logic in `get_forecast_accuracy`). `max_abs_error`
  used for the "biggest miss" sort is now in the same space too.
  The UI labels the series *"Actual (per-interval demand)"* on
  cumulative sensors so the units are self-describing.
- **Run-to-run disagreement pooled across every model ever logged**.
  `get_forecast_stability()` had no `model_name` filter, so tinker-era
  runs under rotated-out model names inflated the CV metric. On a
  seeded test series this flipped the reported swing from ±99% (all
  models mixed) to ±3.5% (current champion). The function and the
  `/forecast-stability` endpoint now take `model_name`, default to
  the experiment's selected/best model, and honour `?model=all`.
- **Lead-time chart silently rendered empty when the model filter had
  no data for the current champion**. After a champion swap mid-
  window, filtering the v2.22.0 queries to `selected_model` returned
  zero rows and the chart area stayed blank with no explanation.
  The `/forecast-accuracy`, `/forecast-stability`, and
  `/forecast-trajectory` endpoints now detect "filter empties the
  result but the window has data from other models" and fall back to
  unfiltered, returning a `model_fallback` flag. The UI surfaces this
  as `all (fallback)` in amber on the model badge with a hover
  tooltip explaining why. `Plotly.purge()` is also called on the
  lead-time container in the empty-state branch so the panel visibly
  clears instead of looking like it's still loading.
- **Top controls row layout**: the `Window:` dropdown was inheriting
  `.setting-input { width: 100% }` from the settings page, stretching
  full-width and forcing the `Evaluate as:` toggle onto a new line.
  `select.setting-input` inside `.acc-top-controls` is now
  `width: auto; min-width: 10rem` so both controls sit on the same
  row. Gap and margin tightened.

### Changed

- **Trajectory section renamed** from "One moment, every prediction
  of it" (awkward) to **"How predictions converge on a single
  moment"** — describes the plot's purpose directly.
- **Evaluation-mode hint is now an info-tip** matching the rest of
  the page, replacing the inline *"Recommended — avoids mistaking…"*
  sentence that was noisy and stylistically out of place.

## 2.22.0

### Added

- **Three-layer Forecast Accuracy UI** aimed at HA end-users, not
  just data scientists. The tab now opens on a **verdict card** with
  a plain-English headline ("Looking healthy" / "Something looks
  off") plus three colour chips for Accuracy / Calibration /
  Stability, each reporting good / fair / poor with a one-word
  detail. Three headline tiles show typical next-step error (absolute
  + % of typical demand), interval coverage vs the 80% target, and
  run-to-run swing. Below the verdict, Layer 2 surfaces the drivers
  — a re-worded "Does re-forecasting help?" sentence plus a
  simplified lead-time chart (MAE only by default, RMSE/bias behind
  a toggle). Layer 3 is a collapsed `<details>` accordion holding
  the existing convergence, trajectory, and stability diagnostics
  for deeper debugging.
- **Normalised headline error**: `get_forecast_accuracy()` now
  returns `typical_interval_demand` (mean |actual| over the window,
  mode-aware) so the verdict card can report *"0.4 kWh — 8% of
  typical"* without the client needing to know units or scales.
- **Forecast convergence as a fan chart**: replaces the
  12-coloured-lines overlay with a shaded band (min/max across
  recent runs), a dotted median line, a bright yellow line for the
  latest run, and white for measured. Communicates "how much the
  forecast wobbles between runs, and does the latest land on the
  actual" at a glance.
- **"Biggest miss" sort for forecast trajectory**: the target-picker
  now defaults to the largest `|predicted − actual|` in the window
  rather than the most recent timestamp. Server returns a new
  `target_meta` array with per-target `max_abs_error` and `actual`
  so the UI can sort client-side without a round trip. "Most recent"
  is still available via a dropdown.
- **Plain-English stability**: the run-to-run card now leads with
  *"Forecasts for the same future moment typically disagree by ±X%
  between runs — consistent / a little jittery / noticeably
  unstable"* rather than showing bare CV numbers. Tiles re-labelled
  ("Per-moment swing", "Runs analysed") to match.
- **Window selector** (7 / 30 / 90 days) at the top of the tab that
  cascades to every endpoint on the page, so accuracy, coverage,
  revision, and stability all share the same lookback. Replaces the
  previous hard-coded 30 days.
- **Model badge** in the top controls shows which model the metrics
  filter to — relevant now that the queries restrict to a single
  model by default (see Fixed below). Overridable via `?model=all`.
- **Cold-start aware empty states**: the empty state now
  distinguishes "waiting for HA actuals" vs "no forecasts logged
  yet" vs "nothing in this window", each with its own hint.

### Changed

- **Lead-time chart is MAE-only by default**. A "Show RMSE & bias"
  checkbox reveals the additional series for the DS view.
  Low-sample buckets (n < 10) render as open circles so a 3-sample
  bucket no longer looks as authoritative as a 300-sample one.
- **Revision card rewritten** around a single plain-English sentence
  (*"Errors dropped 18% between first and latest forecast"*) rather
  than a colour-coded `+X%` / `-X%` tile whose direction was
  ambiguous. Each MAE tile now labels its evaluation mode so the
  units aren't lost.
- **Stability charts deferred**: rendered on first accordion open so
  the initial accuracy-tab paint is fast even when there's a lot of
  stability data to plot.

### Fixed

- **`accuracyMode` defaulted to `'raw'` for cumulative sensors at
  script load** because `var EXP_SOURCE_CUMULATIVE` was hoisted but
  not initialised when the ternary read it. The UI silently reported
  raw-value MAE on a cumulative sensor (where 23:30 errors look huge
  purely because the number is huge) while showing "Per-interval
  demand" as the active toggle. Declarations moved ahead of the
  initialiser.
- **Increment mode diffed across actuals gaps**. `value - LAG(value)
  OVER (ORDER BY grid_dt)` assumed the previous row was one interval
  earlier; a 2h HA outage made the delta look like a single-interval
  demand of 2h worth, inflating MAE with data-availability
  artefacts rather than model error. Added an adjacency guard:
  delta is null unless `grid_dt − LAG(grid_dt) = interval`. Applied
  to both the actuals and forecast LAG pointers.
- **No `model_name` filter on accuracy / coverage / revision
  queries**. After a champion swap, metrics aggregated across both
  models until the retention window rolled.
  `get_forecast_accuracy()` and `get_forecast_coverage()` now accept
  an optional `model_name`; the endpoint defaults to the
  experiment's selected/best model and exposes `?model=<name>` or
  `?model=all` for overrides.
- **Unguarded `int()` on `?days=`** in the forecast-accuracy
  endpoint returned 500 on a bad value. Now guarded + clamped to
  [1, 365], matching the stability endpoint.
- **Stability CV = 0 when mean ≈ 0**: low-demand timesteps with
  non-zero prediction std were reported as "perfectly stable"
  because the CV ratio collapsed. Those rows are now skipped from
  the per-timestep series and the median-CV aggregate rather than
  polluting the chart with false zeros.

## 2.21.0

### Added

- **Signed bias (ME)** alongside MAE/RMSE in the lead-time accuracy
  curve. A third trace (dashed green, zero-reference line) reveals
  whether the model systematically over- or under-predicts — a
  dimension that absolute-error metrics collapse. Surfaced at the
  lead-time bucket level and in the revision summary
  (`first_forecast_me`, `latest_forecast_me` sit beneath each MAE).
- **Forecast trajectory plot** — a new "Forecast trajectory" section
  renders every forecast ever issued for a chosen `target_dt` as dots
  on the `issued_at` axis, with the actual as a horizontal reference
  line and the target moment as a dotted vertical marker. Answers "did
  the prediction walk smoothly toward truth, oscillate, or drift?" —
  shape information the first-vs-last revision metric collapses to two
  numbers. Backed by `HistoryDB.get_forecast_trajectory()` and a
  `/experiment/{name}/forecast-trajectory` endpoint; the dropdown is
  auto-populated with recent targets that have an actual AND >=2
  distinct issuances (single-forecast targets teach nothing so they're
  filtered out).
- **Increment-based evaluation mode** for cumulative sensors. When
  `source_is_cumulative=True`,
  `get_forecast_accuracy(..., evaluation_mode="increment")` uses LAG
  window functions to diff both forecasts and actuals before joining,
  filters midnight-reset rows (negative increments on daily-resetting
  sensors), and computes MAE/RMSE/ME on per-interval demand rather
  than raw cumulative value. Errors become comparable across the day
  instead of being dominated by the sensor's shape. The UI exposes a
  "Evaluate on: Per-interval demand / Cumulative value" toggle, which
  defaults to increment for cumulative targets and is hidden otherwise.
- **Conformal 80% prediction intervals** via online/adaptive conformal.
  `HistoryDB.get_conformal_quantiles()` computes per-lead-bucket 90th
  percentiles of absolute residuals from recent `forecast_log` rows vs
  the actuals table; `_publish_forecast_sensors` auto-applies these to
  produce upper/lower bands around each point forecast. New HA
  entities `sensor.{prefix}{exp}_upper_80` and `_lower_80` expose the
  band in their `forecast` attribute; the main forecast sensor also
  carries `forecast_upper` / `forecast_lower` arrays for in-chart
  shading. No additional model fit required — residual quantiles are
  learned from deployed forecast/actual pairs, so bands appear
  automatically once the residual buffer fills and stay absent during
  cold-start.
- **Interval coverage diagnostic**: `HistoryDB.get_forecast_coverage()`
  joins `forecast_log` rows that carry both bounds against actuals and
  reports the fraction inside `[lower, upper]`, per lead-bucket and
  overall. Surfaced as a calibration card on the accuracy tab —
  empirical coverage vs nominal 80%, colour-coded green within +/-5pp,
  amber within +/-10pp, red beyond, with a plain-English diagnosis
  ("Well calibrated", "Bands are wider than needed", "Bands are too
  tight").

### Changed

- `forecast_log` schema gained nullable `upper`, `lower` columns with
  an auto-migration path: `ensure_forecast_log_table` now runs
  `PRAGMA table_info` and appends the columns with `ALTER TABLE ADD
  COLUMN` when missing. Point-only legacy rows remain supported — the
  coverage query filters to rows with non-null bounds.
- `HistoryDB.log_forecast()` accepts optional `upper_bounds` /
  `lower_bounds` lists (same length as `predictions`); existing
  callers unchanged.
- `_publish_forecast_sensors()` signature gained optional
  `y_pred_upper`, `y_pred_lower`, `interval_level` params; when
  omitted and a residual history is available, bands are auto-computed
  inside the helper so both the full-retrain production path and the
  cached-forecast path get intervals without caller-side changes.
- `/experiment/{name}/forecast-accuracy` accepts `?mode=raw|increment`
  (defaults to `increment` for cumulative sensors) and merges the
  coverage result into its JSON payload alongside `lead_time_curve`
  and `revision_improvement`.

### Notes

- Conformal calibration is online/adaptive rather than split-conformal:
  residual exchangeability is approximate (temporal drift, model
  retrains), so coverage guarantees are empirical rather than
  finite-sample. The coverage card exists precisely to surface
  deviations from the nominal 80%, so the user can tell when bands
  drift and need recalibration.
- Cold-start: coverage card stays hidden and bands stay absent until
  `forecast_log` has accumulated enough residuals per lead bucket. The
  system silently falls back to point-only forecasts in the meantime.

## 2.20.0

### Added

- **Load subtract**: new per-experiment `load_subtract` config field for
  removing sensor contributions (EV charging, iBoost solar-divert, etc.)
  from the target load before training, so models learn the baseline
  household pattern rather than a mixed signal. Each entry is a
  `SubtractCfg` with explicit `source` (`cumulative_daily`,
  `cumulative_monotonic`, `interval`, `auto`), `on_missing` policy
  (`zero`, `drop`, `error`), optional `scale` for unit conversion
  (Wh→kWh), and `max_fraction_of_load` / `max_fraction_violation_pct`
  guards that fail fast on unit bugs or double-counted signals. Replaces
  the unwired legacy `subtract: [str]` stub, which is now deprecated and
  logs a warning when present in YAML.
- **Robustness layer** in `preprocessing.apply_load_subtract()`:
  per-sensor unit scaling, missing-data policy enforcement, negative
  clipping with clipped-row count and >5% warn threshold, fraction guard
  that raises `LoadSubtractError` with a worst-row diagnostic
  (`ratio=X.XX subtract=Y load=Z`) when subtract exceeds load on too
  many rows, leading-gap window detection for history coverage, and
  tz-awareness mismatch detection.
- **Audit logging** in the training pipeline: per-run boxed summary
  showing total load, total subtracted (with % share), clipped rows, and
  a per-sensor breakdown of sum/missing/max-fraction/violations/gap
  window. Same numbers that would drive a dry-run, emitted at training
  time.
- **Load Subtract UI card** on each experiment's config tab with entity
  picker (HA search), source / on_missing / scale / max_fraction inputs,
  and row removal — mirroring the Covariates card. Three new endpoints
  (`POST /experiment/{name}/add-load-subtract`,
  `remove-load-subtract`, `clear-load-subtract`) wrap the YAML helpers.
- **Pipeline placement chosen deliberately**: subtract runs AFTER
  cumulative→interval conversion and resample, but BEFORE outlier
  clipping. Outlier bounds are then computed on the adjusted (baseline)
  signal rather than on raw spikes from the subtracted component, so
  real baseline peaks aren't muted by EV / iBoost outliers.

### Changed

- **`searchEntities` JS** now finds its results dropdown via the input's
  `.entity-search-wrapper` parent rather than a hardcoded
  `#entity-results` ID, so multiple entity pickers (Covariates + Load
  Subtract) coexist on the experiment config page. Outside-click handler
  closes all open result divs rather than only the covariate one.

### Deprecated

- `ExperimentCfg.subtract: list[str]` — was never wired into the
  pipeline. Now emits a deprecation warning at `load_config` time
  telling users to migrate to `load_subtract`. No behaviour change for
  anyone who had it set (it did nothing before; it does nothing now).

### Tests

- **29 new tests** in `tests/test_load_subtract.py` covering
  `SubtractCfg` validation, YAML round-trip (including bare-string
  tolerance and legacy-`subtract` deprecation warning), add/remove/clear
  helpers, and every branch of the robustness checklist:
  perfect-alignment, `on_missing` policies, leading-gap detection,
  negative clipping, fraction-guard firing on unit bug, fraction-guard
  tolerating noise band, scale application, tz mismatch, and
  multi-sensor summation.

## 2.19.0

### Removed

- **NeuralProphet backend deleted.** The `neuralprophet_backend.py` module,
  its schema and catalog entries, the optional-backend registration in
  `main.py`, the sample `mlfl.yaml` entry, and the `neuralprophet` pin in
  `requirements.txt` are all gone. Docstrings in `config.py` that called
  out NeuralProphet as a special case have been simplified to just "tree
  models". Any existing `mlfl.yaml` with `"neuralprophet"` in
  `models_enabled` will silently skip the model (the registry no longer
  registers it) — remove the line to keep configs tidy.

### Changed

- **Models tab grouped into Tree Models and Neural Models**, both on the
  standalone `/models` page and each experiment's Models tab. Grouping is
  driven by `model_type == 'Tree'` on the existing catalog — no new field.
- **Model identifiers render as display names across the UI.** A new
  `model_display` Jinja filter and matching `modelDisplay()` JS helper map
  internal ids (`cnn`, `lightgbm`, `nbeats`, …) to their catalog
  `display_name` everywhere user-visible: comparison tables, fold-stability
  chart legends, holdout / residual legends, feature-importance headings,
  the tuning banner and holdout chart title, the live-training "Current
  Model" stat and event log, the model-select dropdowns on Tuning and
  Covariate Analysis, and the dashboard card's Best Model / Training
  badge. CNN is now CNN, not cnn.
- **Descriptions corrected to match implementations.**
    - LSTM: was "2-layer LSTM …" (the schema allows 1–8 layers);
      now "Multi-layer LSTM with temporal attention and multi-horizon
      output head."
    - TiDE: was "Time-series Dense Encoder with residual MLP
      encoder-decoder." (undersold the paper implementation); now "Dense
      encoder-decoder with temporal decoder and global residual skip."
- **`batch_size` and `loss_fn` excluded from the Optuna search space.**
  Both now carry `"tunable": False` in `MODEL_PARAM_SCHEMA` and the
  tuning objective loop skips any spec flagged non-tunable. Rationale:
  `batch_size` is a compute/memory knob (tuning already force-overrode
  it to 16), and `loss_fn` decides *what* is being optimised — tuning it
  makes the composite score non-comparable across trials. Both remain
  manually configurable on the Models page; `loss_fn` is still set
  per-experiment in Settings and propagated to every trial via
  `_apply_experiment_neural_params`.

## 2.18.0

### Changed

- **Daily cumulative loss is now trajectory-matching** (was: mean over
  horizons). `daily_loss_weight > 0` now penalises error in the predicted
  cumulative *curve* at every horizon step — not just a single aggregate
  constraint at the endpoint.

  **New formula** (in ``_composite_horizon_loss``):
  ```
  L_daily = mean_h  criterion( cumsum(ŷ)[h], cumsum(y)[h] )  /  H
  ```
  The cumulative error is penalised at every h=1..H, then averaged and
  normalised by H so λ stays reasonably invariant to horizon length.

  **Previous formula** (v2.16–v2.17):
  ```
  L_daily_old = criterion( mean_h(ŷ), mean_h(y) )
  ```
  The mean-of-horizons match was a single scalar per sample — a straight
  line ramp and the actual stepped daily-demand curve can have identical
  means and still look completely different visually, so the term had
  almost no effect on training.

  **Why this matters for cumulative-origin targets**
  (``sensor.x_today``-style sensors that reset at midnight): the training
  objective now directly aligns with the cumulative curve users evaluate
  against. Per-interval predictions that regress to the mean on sparse
  targets produce a constant-slope cumulative ramp that misses the actual
  curve shape — the trajectory loss penalises this drift at every
  intermediate horizon.

  **Breaking behaviour note**: experiments that already have
  ``daily_loss_weight > 0`` will train differently after upgrading.
  Models retrained on v2.18+ will produce different forecasts than
  v2.16/v2.17 did with the same setting. The default (``0.0``) preserves
  pre-composite-loss behaviour byte-identically, so experiments that
  never toggled the setting are unaffected.

  **Single-horizon outputs** (``future_periods=1`` or ``y_pred.dim()==1``)
  silently skip the daily term — cumsum of a single value equals the
  value itself, so the term would be redundant with the interval term.

### Context

Real-world testing on a Mixergy hot-water demand target
(``sensor.mixergy_demand_today``, a cumulative-reset-daily sensor) showed
the v2.16 mean-based daily loss had no measurable effect. The mean of the
forecast horizon is approximately matched by any unbiased model
regardless of curve shape, so the penalty term had no gradient to
contribute. The trajectory formulation directly penalises the
drift-accumulation failure mode.

### Audit note

Pre-release audit of the ``source_is_cumulative=True, reset_daily=True``
preprocessing path (``cumulative_to_interval``) confirmed the training
signal is clean for cumulative-reset-daily targets — primary reset
detection via ``diffs < 0`` (preprocessing.py:86) is timezone-independent
and correctly identifies real resets regardless of local-vs-UTC
midnight alignment. Two minor cosmetic issues in the publishing layer
(``daily_cumulative_series`` doesn't include a today-so-far seed when
aligning with the source sensor) will be addressed in a separate
release; they don't affect training.

## 2.17.0

### Added

- **Optimiser choice (neural)** — new per-experiment dropdown in Settings →
  Training selects between **AdamW** (default, decoupled weight decay — the
  optimiser every published time-series transformer paper uses) and
  **Adam** (classic, tied weight decay). Both share the same
  ``learning_rate``, cosine schedule, and ``weight_decay=1e-4``, so the
  choice isolates the decoupled-vs-tied weight-decay behaviour rather than
  confounding it with decay magnitude or LR.

  Default remains AdamW, preserving pre-2.17 behaviour for existing
  experiments. Applied to all 12 torch neural backends; silently ignored
  by NeuralProphet and tree models.

  Implementation: new ``ForecastModel._build_optimiser`` static helper on
  the base class replaces the previously-hardcoded ``torch.optim.AdamW(...)``
  line in each of the 12 backend ``fit()`` methods; hyperparam
  round-trips through ``get_params``/``set_params``/save/load.

### Fixed

- **Covariate Analysis and Tuning now honour Settings-level neural params.**
  Previously, the Covariate Analysis and Hyperparameter Tuning code paths
  instantiated neural models with the backend's defaults — so if you
  selected ``loss_fn='huber'``, ``daily_loss_weight>0``, or (as of 2.17)
  ``optimiser='adam'`` in Settings, those choices were silently ignored by
  anything outside the main CV loop and the production-training path. The
  main benchmark minimised one objective; Covariate Analysis and Tuning
  minimised another (typically MSE with λ=0 and AdamW), which made the
  two flows incomparable and hid the effect of Settings changes.

  New helper ``_apply_experiment_neural_params(model, exp_cfg, overrides=…)``
  in ``main.py`` propagates ``loss_fn``, ``daily_loss_weight``, and
  ``optimiser`` from the experiment config to any freshly-created neural
  model, guarded by ``hasattr`` (tree models / NeuralProphet silently skip)
  and respecting caller-supplied ``overrides`` (so Optuna-swept params and
  ``model_overrides`` still take priority). Called from:
  - Tuning baseline trial (``_run_tuning``)
  - Tuning per-trial objective
  - Holdout re-fit in Tuning (``_run_holdout``)
  - Covariate Analysis per-config-per-model fit

  The 4 pre-existing ``set_params(loss_fn=…)`` call-sites in the main CV
  and production paths are left unchanged — they already handle the same
  propagation inline and continue to work.

## 2.16.0

### Added

- **Daily cumulative loss (optional)** — new per-experiment toggle that adds a
  horizon-aggregate term to the training loss of all 12 torch neural
  backends. Useful for cumulative-origin targets (e.g.
  `sensor.mixergy_demand_today`, energy meters) where per-interval errors
  otherwise accumulate across the forecast window into a wrong daily total.

  **Loss form:** `L = L_interval(ŷ, y) + λ · L_interval(mean_H(ŷ), mean_H(y))`
  where the second term applies the existing `loss_fn` (MSE / MAE / Huber) to
  the mean-over-horizons of each sample. With `future_periods=48` and
  `interval_minutes=30` the horizon spans 24 h, so the term is exactly a
  rolling daily cumulative. Using the mean (not sum) keeps both terms on the
  same scale so λ is intuitive; single-horizon outputs silently skip the
  daily term.

  **λ:** hardcoded at 0.5 when the toggle is on (a sensible
  "daily term is half as important as interval term" default), 0.0 when off.
  Sophisticated users can still hand-edit `daily_loss_weight` as a float in
  `mlfl.yaml` — the app.py validator and `ExperimentCfg` field accept any
  value ≥ 0; the UI toggle just writes the two canonical ones.

  **Applies to:** LSTM, CNN, N-BEATS, N-HiTS, TiDE, DLinear, TSMixer,
  PatchTST, iTransformer, Crossformer, TimesNet, SparseTSF. Silently ignored
  by NeuralProphet (internal fit loop, no loss-hook) and tree models
  (LightGBM, XGBoost — sample-wise gradient API doesn't fit a cross-sample
  aggregate loss; revisit via post-hoc reconciliation if needed).

  **Implementation:** one new `ForecastModel._composite_horizon_loss`
  static helper on the base class; each of the 12 backends replaces its
  `criterion(...).mean()` train and val loss blocks with a single
  helper call; the hyperparam round-trips through `get_params`/`set_params`.
  Default 0.0 preserves byte-identical behaviour with pre-2.16 runs.

  UI toggle lives in the experiment Settings → Training section, next to the
  output-activation selector.

## 2.15.0

### Added

- **Reversible Instance Normalization (RevIN)** — per-window, per-channel
  normalisation (Kim et al. 2022,
  https://openreview.net/forum?id=cGDAkQo1C0p) added to the shared `base.py`
  and plumbed into every PyTorch neural backend except N-BEATS and N-HiTS:
  LSTM, CNN, DLinear, TiDE, TSMixer, SparseTSF, PatchTST, iTransformer,
  Crossformer, TimesNet. On by default (`use_revin=True`) because the
  reference implementations of these papers all ship with RevIN — without
  it, the codebase was systematically under-performing published benchmarks
  on non-stationary series.

  Each enabled backend now:
  - wraps its forward pass with input `normalize()` (per-sample, per-channel
    mean/std over the time axis, plus a learnable affine),
  - reverses the target-channel stats on the head output via `denormalize()`
    before the output activation,
  - stores/restores `use_revin` and `target_channel` in `get_params`,
    `set_params`, `save`, and `load`,
  - skips the pre-v2.15.0 dataset-level channel normalisation and the zscore
    target z-scoring when RevIN is active (the two schemes are mutually
    exclusive). `output_activation='zscore'` is now a no-op when RevIN is on.

  N-BEATS and N-HiTS are left untouched because their doubly-residual
  backcast-subtraction stacking handles instance-level normalisation
  architecturally; layering RevIN on top would double-normalise.

- **TiDE future-covariate path** — TiDE is now the full architecture from
  Das et al. 2023, with:
  - a feature-projection residual block for known-future covariates,
  - a per-horizon temporal decoder that fuses the decoder state with each
    horizon step's projected future features,
  - a global linear residual from the past window straight to the forecast.

  `fit()`, `predict()`, and `predict_sequence()` now accept an optional
  `future_covariates` kwarg of shape `(n_samples, n_horizons,
  n_future_covariates)`. Typical contents: calendar features (hour of day,
  day of week, holiday flag), externally-forecast weather (Solcast GHI
  p10/p50/p90, Open-Meteo temperature), or a known-future schedule. When
  not provided, TiDE degrades gracefully to a dense encoder-decoder on the
  past window alone (backwards-compatible with pre-v2.15.0 behaviour).

- **Config:** `ExperimentCfg.use_revin` (default `True`) and
  `ExperimentCfg.future_covariate_features` (default `[]`) added to expose
  the two features to experiment YAMLs.

### Changed

- **Config doc for `output_activation`** notes that `zscore` is superseded
  by RevIN when `use_revin=True`.

- **TiDE backend** is now a full paper replication rather than a simplified
  dense encoder-decoder. Older TiDE checkpoints load with
  `n_future_covariates=0` and the global-residual + temporal-decoder paths
  still function — the network shape degrades cleanly, but you'll want to
  retrain to get the temporal-decoder weights.

## 2.14.0

### Added

- **`output_activation: zscore` is now honoured by every PyTorch neural
  backend** — previously only the LSTM applied target z-score
  normalisation; the other eleven (CNN, DLinear, N-BEATS, N-HiTS, TiDE,
  TSMixer, SparseTSF, PatchTST, iTransformer, Crossformer, TimesNet)
  silently degraded to `linear`. Each backend now:
  - fits per-horizon target mean/std on training data (scalars for
    single-horizon, per-column arrays for multi-horizon),
  - trains with a linear head in z-space,
  - denormalises predictions back to physical units in
    `predict()` and `predict_sequence()`, flooring at zero, and
  - persists `y_mean`/`y_std` in `save()` / `load()` so checkpoints
    round-trip cleanly.

  Predictions from a fresh zscore-trained checkpoint are now on the
  correct physical scale for all backends. Older checkpoints saved
  under a non-zscore activation load with identity defaults and are
  unaffected.

- **Config:** documentation for `ExperimentCfg.output_activation`
  updated to reflect full-backend coverage.

### Changed

- **TimesNet: real multi-period aggregation** — the TimesBlock
  previously detected the top-k dominant FFT periods but then
  discarded everything except the median, collapsing the block to
  single-period 2D convolution and defeating the paper's core
  contribution. It now runs the shared inception block once per
  detected period and aggregates the results with softmax-weighted
  amplitudes (matching Wu et al. 2023). Periods that round to the
  same integer bucket have their amplitudes merged before softmax.

- **NeuralProphet: reproducible timestamps** — `fit()` used to call
  `pd.Timestamp.now()` for every fit, so repeated fits on the same
  `(X, y)` produced different temporal features and broke
  reproducibility. The backend now accepts `date_index` via kwargs
  (preferred — preserves real calendar effects for seasonality /
  holiday components) and falls back to a fixed `2000-01-01` anchor
  when the caller can't supply one.

### Fixed

- **N-HiTS: defensive copy of `pool_kernels`** — the inner
  `_NHiTSNet` mutated the caller's list when extending to `n_stacks`,
  which could silently corrupt a reused kernel list across multiple
  model constructions.

- **Post-hoc clipping restored for all neural backends on cumulative
  targets** — v2.13.2 added the clip inside the LSTM backend's
  `predict()` / `predict_sequence()` for the zscore path only.
  Every other neural backend (PatchTST, N-BEATS, TiDE, TSMixer,
  SparseTSF, TimesNet, iTransformer, Crossformer, DLinear, NHiTS,
  CNN) still had an unguarded linear-output edge case: with a
  manual `output_activation: linear` override, tiny negatives could
  slip through into live forecasts and the holdout chart. The clip
  is now applied in the three central neural forecast paths in
  `main.py` (`_run_production_inference`, `_forecast_with_cached`,
  and `_generate_holdout_predictions`), gated on
  `exp_cfg.source_is_cumulative=True` so signed-target experiments
  (temperature deltas, net flows) are not clipped.

  Scope note: tree-model paths are unchanged — they already cannot
  emit negatives for non-negative training targets under normal
  settings, and the existing log-transform branch's
  `np.maximum(np.expm1(...), 0.0)` still owns that case. The LSTM
  backend's internal clip from v2.13.2 is also kept, and the new
  denormalise-and-clip step in every other neural backend's
  `predict()` / `predict_sequence()` (see the zscore work above)
  provides a third layer for the zscore path.

- **Docstring drift:** CNN and LSTM module docstrings no longer
  claim `ReduceLROnPlateau` scheduling — both backends use
  `CosineAnnealingLR` (unchanged; documentation only).

## 2.13.2

### Change

- **LSTM zscore path now clips predictions at zero** —
  `output_activation: zscore` uses a linear head in z-space, and
  denormalising a slightly-negative z-prediction can emit values
  below zero for physically non-negative targets. v2.13.0 shipped
  without the clip (explicit "try without clipping first" test);
  with the unclamped output now confirmed to show a bi-modal
  flat-at-mean regime, the clip is restored in `predict()` /
  `predict_sequence()` immediately after the z-unnormalisation step.
  Matches the pre-v2.11.0 behaviour for the same (z-score + linear)
  training setup.
- **Diagnostic hypothesis under test:** the pre-today wavy-looking
  forecasts may have been clipped flat-at-zero predictions
  overlaid with occasional peaks. Clipping here won't fix the
  underlying encoder-state dominance but will let us visually
  separate "network output" from "masked-by-clip floor" in the
  evolution chart.

## 2.13.1

### Bugfix

- **`zscore` now appears in the Settings → Output activation
  dropdown** — v2.13.0 shipped the backend but the UI dropdown and
  its `?` helper tooltip still listed only the six pre-zscore
  options. The dropdown now includes *Z-score (normalised target,
  LSTM)* and the tooltip documents both the new option and the
  updated `Auto` resolution (zscore for LSTM, softplus/linear
  otherwise).
- **Settings save endpoint now accepts `zscore`** — the server-side
  validator whitelist was missing the new value, so even setting it
  via `mlfl.yaml` edit would have been stripped on any subsequent
  UI save of any other field.

## 2.13.0

### Feature

- **New `output_activation: zscore` option (LSTM target normalisation
  restored)** — when the LSTM is configured with `output_activation:
  zscore`, training targets are z-score normalised per-horizon (mean
  subtracted, divided by std computed on training data), the network
  predicts in z-space with a linear head, and predictions are
  denormalised back to physical units at inference time. This keeps
  loss-landscape curvature bounded regardless of raw target magnitude
  — making the LSTM work well out-of-the-box on targets ranging from
  small fractions to large cumulative values. No post-hoc clipping
  is applied; predictions are returned exactly as the model emits
  them (signed, potentially slightly negative near zero).
- **`output_activation: auto` now resolves to `zscore` for the LSTM
  backend** — previously `auto` meant `softplus` for cumulative
  sources and `linear` for instantaneous. Other neural backends
  (TiDE, iTransformer, PatchTST, NHiTS, N-BEATS, CNN, DLinear,
  TSMixer, SparseTSF, TimesNet, Crossformer) keep the physics-based
  `softplus` / `linear` default. The LSTM-specific change is
  motivated by empirical evidence that un-normalised targets cause
  the recurrent encoder to produce a flat vs. wavy bi-modal forecast
  pattern on high-amplitude cumulative targets (observed on Mixergy
  `demand_today`).
- **Checkpoint format: LSTM checkpoints now include `y_mean` / `y_std`
  keys** — pre-v2.13.0 checkpoints load with identity defaults
  (`y_mean=0`, `y_std=1`) so backward-compat is preserved; new
  zscore-trained models persist their per-horizon stats alongside
  the existing `channel_mean` / `channel_std` input-scaling stats
  and `sigmoid_scale` buffer.

### Notes

- Tree backends (LightGBM, XGBoost) continue to ignore
  `output_activation` entirely. Non-LSTM neural backends accept
  `zscore` but treat it as `linear` (linear head, no target
  normalisation) — a no-op fallback rather than an error.

## 2.12.1

### Bugfix

- **Forecast evolution and stability charts now display in local
  time (respects DST)** — timestamps are stored in `forecast_log` as
  UTC strings, but the charts were handing the bare strings to Plotly
  which then rendered them in UTC. For a user in BST (or any non-UTC
  TZ) this showed events an hour off the wall-clock and mis-aligned
  the midnight resets in cumulative view. Timestamps are now parsed
  as UTC (`new Date(s + 'Z')`) and passed to Plotly as `Date` objects,
  which it renders in the browser's local TZ. The x-axis label also
  shows the active TZ abbreviation (e.g. `Target time (BST)`).
- **Cumulative-reset boundary uses local date, not UTC date** —
  matches the behaviour of the underlying HA `_today` sensors, which
  reset at local midnight. Previously a BST user saw the cumulative
  curve dropping to zero at 01:00 local (UTC midnight); now it resets
  at 00:00 local as expected.

## 2.12.0

### Feature

- **Model stability panel on the Forecast Accuracy tab** — new
  section that measures *self-consistency* of the model's predictions
  across forecast issuances, distinct from accuracy (which compares
  predictions against actuals). Useful for diagnosing an unstable
  model whose forecasts swing wildly issuance-to-issuance even when
  the target isn't changing.
  - **Median step CV%** scorecard: across-cycle coefficient of
    variation of predictions for the same future timestamp.
    Colour-coded green <10%, amber <25%, red ≥25%.
  - **Median daily CV%** scorecard (cumulative-source experiments
    only): across-cycle CV of predicted daily totals.
  - **Per-timestep std chart**: x = target time, y = std of
    predictions across all cycles that forecast that target. Peaks
    highlight when the model disagrees with itself most. Hover shows
    mean prediction, CV%, and n cycles.
  - **Daily-total CV bar chart** (cumulative-source only): one bar
    per calendar day showing CV of predicted daily total across
    cycles. Colour-coded the same as the scorecards.
- **`get_forecast_stability()` on `HistoryDB`** — SQLite-native
  cross-cycle variance via the sum-of-squares identity (SQLite has no
  STDDEV): per-target-dt `sqrt(avg(p²) − avg(p)²)` clamped at zero to
  handle float rounding on constant columns. Filters to target_dts
  with ≥2 cycles.
- **`/experiment/{name}/forecast-stability` endpoint** — returns
  `per_timestep`, `daily_totals` (when `source_is_cumulative`),
  `summary`. Accepts `?days=N` (1-90, default 30).

## 2.11.6

### Feature

- **Cumulative view toggle on Forecast evolution chart** — for
  experiments with `source_is_cumulative: true`, a new "View as"
  dropdown lets the user switch between:
  - **Per-interval deltas** (default) — what the model actually
    predicts: increment per `interval_minutes` bucket.
  - **Cumulative (daily)** — reconstructed by summing the per-interval
    predictions within each calendar day, resetting at midnight when
    `reset_daily: true`. Makes it intuitive to read values as "% of
    the day's budget drawn by 6pm" rather than "% drawn in this one
    30-min bucket".
- **Y-axis label now responds to the view** — shows `<name>
  (cumulative <units>, resets daily)` in cumulative mode, `<name>
  (<units> per Nmin bucket)` for interval mode on cumulative sources,
  and `<name> (<units>)` for instantaneous sources.
- **Hover tooltip labels match the view** — says "cumulative" in
  cumulative mode, "predicted" / "measured" in interval mode.

## 2.11.5

### Improvement

- **Y-axis label now identifies the experiment and scale** — the
  previous label was just the raw `units` string (e.g. `%`), which
  doesn't tell the reader *of what*, and is actively misleading when
  the preprocessor has converted a cumulative-daily sensor into
  per-interval deltas. The label is now:
  - `<experiment_name> (<units> per <N>-min bucket)` when the target
    is cumulative (post-preprocessing scale is per-interval)
  - `<experiment_name> (<units>)` when the target is already an
    instantaneous reading
  - `<experiment_name> — <entity_id>` when no units are configured

  So a Mixergy cumulative-daily-% experiment now shows
  `mixergy_demand (% per 30-min bucket)` instead of just `%`.

## 2.11.4

### Improvement

- **Readable Forecast evolution chart** — rewrote the overlay so each
  cycle is visually distinct:
  - **Sequential HSL colour gradient** (indigo → teal → yellow-green)
    instead of same-cyan-with-opacity; adjacent issuances no longer
    blur together.
  - **Legend labels every cycle** with `DD Mon HH:MM`, grouped under
    "Forecast cycles (by issue time)", latest tagged "← latest" and
    ordered at the top. Moved the legend out to the right margin so it
    stops covering chart data.
  - **Informative y-axis** — uses the experiment's `units` when set;
    falls back to the target entity id so the user always knows which
    sensor the trace represents (was just "Value").
  - **Clearer x-axis** — "Target time (when the forecast applies)".
  - **Inline caption** under the heading explains that colour encodes
    recency and the white line is the measured value — no need to
    hover the info-tip to understand the chart.
  - **Richer hover** — bold "Forecast issued DD Mon HH:MM" header,
    explicit "predicted" / "measured" labels, actuals renamed
    "Measured (actual)" for contrast with forecast traces.

## 2.11.3

### Bugfix

- **Fix Samples=tiny in revision-improvement card** — the
  first-vs-latest comparison used `first_pred != last_pred` to filter
  out single-forecast targets, but this also excluded any target whose
  re-forecasts happened to produce numerically identical predictions,
  and the CASE-aggregation interaction dropped counts further. Replaced
  with an explicit `COUNT(*) >= 2` filter via a window function so the
  denominator now reflects actual re-forecast targets. Expect Samples
  to jump from single digits to thousands.
- **Fix lead-time chart collapsing to 3 points on fine grids** — the
  query hardcoded a 30-min lead-bucket even when `interval_minutes` was
  5 or 15, collapsing the chart to 2–3 data points and forcing MAE and
  RMSE to numerically converge within each chunky bucket. Bucket size
  now follows `interval_minutes` so a 5-min-grid experiment with a 1h
  horizon gets ~12 buckets instead of 3.

### Feature

- **Forecast evolution overlay chart** — new plot under the Accuracy
  by lead time chart. Renders each of the last N forecast cycles as a
  faded line (older = dimmer, latest = bright cyan) with the actual
  series overlaid in white. Lets you see how prediction curves
  converge toward the truth as lead time shrinks — same pattern as the
  holdout chart but for live production forecasts.
- **Cycle-count selector** — dropdown to choose 6 / 12 / 24 / 48 most
  recent cycles for the overlay.
- **`/experiment/{name}/forecast-evolution` endpoint** — returns the
  last N forecast snapshots plus grid-snapped actuals over the same
  window. Backed by `HistoryDB.get_forecast_evolution`.

## 2.11.2

### Bugfix

- **Cache-bust static assets on version bump** — `style.css` and
  `icon.png` are now requested with a `?v={{ app_version }}` query
  string, so browsers automatically fetch fresh copies after an addon
  update instead of serving stale cached files.
- **Constrain logo render at HTML parse time** — nav-bar `<img>` now
  carries `width="32" height="32"` HTML attributes so the browser
  knows the intended size before CSS loads, preventing a
  flash-of-giant-logo if `style.css` arrives late or from cache.
- **Belt-and-suspenders sizing on `.brand-logo`** — added
  `!important`, `max-width`, `max-height`, and `flex-shrink: 0` so the
  2rem constraint wins against any later rule or flex-growth.

## 2.11.1

### Branding

- **Add addon icon and logo** — `icon.png` and `logo.png` now live both at
  the repo root (for the HA add-on store listing) and inside the addon
  directory (for the supervisor tile / detail page).
- **Replace navbar emoji placeholder** — the 📈 emoji in the web UI nav
  bar is now the actual addon icon, served from `/static/icon.png`.
  Added `<link rel="icon">` so the browser tab favicon also picks it up.

## 2.11.0

### Feature

- **Per-experiment output activation** — neural models now apply a
  configurable activation (`auto` / `linear` / `softplus` / `relu` /
  `exp` / `sigmoid`) to the final Linear head, constraining predictions
  to the target's physical range **inside** the network rather than
  clipping post-hoc. `auto` resolves to softplus when
  `source_is_cumulative=true`, else linear. Settable per experiment from
  the Settings → Training dropdown.
- **Sigmoid scale auto-derived** — when `output_activation=sigmoid` is
  selected, the upper bound is derived from training data as
  `max(|y_train|) × 1.1` and persisted in the checkpoint as a registered
  buffer, so sigmoid can reach observed extrema without clipping.

### Removal

- **Drop post-hoc `np.clip(predictions, 0.0, None)`** across all 12
  neural backends — the activation layer now enforces the valid range
  directly, so the ad-hoc clamp is redundant. `expm1` drift guards
  inside `log_transform` branches are kept (they protect against
  floating-point inversion drift, not network output).
- **Drop target z-score normalization** (`self._y_mean` / `self._y_std`)
  — the activation operates on original-scale targets, so z-scoring the
  target would invalidate the activation's physical-range constraint.
  AdamW + internal LayerNorm converge fine without input/output
  symmetry. Removes ~260 lines of normalize/denormalize plumbing across
  backends.
- **Drop residual-prediction code path** — only fired for
  `n_horizons==1`, which never occurs in production (always
  multi-horizon). Dead code removed from all neural backends.

### Compatibility

- Checkpoints from v2.10.x with `y_mean`/`y_std`/`residual_prediction`
  keys will fail to load; affected experiments retrain automatically on
  startup.

## 2.9.8

### Bugfix

- **Fix flat multi-horizon forecasts in LSTM, iTransformer, and
  Crossformer** — same head bottleneck as the CNN: hidden layer narrower
  than `n_horizons` (48), forcing all horizon predictions to collapse to
  near-identical values. Hidden size is now `max(dim, n_horizons)` in
  all four backends.
- **Tighten N-BEATS tuning ranges** — worst-case combination
  (hidden_size=512, 8 stacks × 8 blocks × 8 FC layers) could allocate
  ~2.5 GB. Reduced maximums: hidden_size 512→256, stacks 8→4,
  blocks_per_stack 8→4, FC layers 8→6. Worst case now ~600 MB.

## 2.9.7

### Bugfix

- **Fix flat CNN multi-horizon forecasts** — the CNN's prediction head
  had a `n_filters // 2` hidden layer (16 neurons at default settings)
  producing 48 horizon outputs. This severe bottleneck forced the model
  to predict near-identical values for every future step, producing a
  flat line instead of a solar curve. The head hidden size is now
  `max(n_filters, n_horizons)` so each horizon can be predicted
  independently.

## 2.9.6

### Bugfix

- **Fix wrong tuning holdout predictions for neural models** — the
  holdout comparison trained CNNs/LSTMs on flat feature vectors instead
  of proper sliding-window sequences, producing garbage predictions.
  Now builds correct temporal windows for both training and test
  portions, matching the benchmark runner's pipeline.
- **Fix misleading tuned MAE in holdout chart** — the displayed MAE was
  taken from the tuning CV trial, not the holdout evaluation. The chart
  now shows the actual holdout MAE for both default and tuned models.

## 2.9.5

### Bugfix

- **Production sensors appear immediately after addon restart** —
  production experiments now retrain on startup instead of waiting for
  the next scheduled retrain (up to 24h away). After a restart or
  update, there were no cached models in memory, forecast cycles
  skipped, and sensors never got published until the retrain timer
  fired. Lab-mode experiments still defer to the normal schedule.

## 2.9.4

### Bugfix

- **Fix CNN tuning crash caused by exponential causal padding** — the
  WaveNet-style CNN uses `dilation = dilation_base^layer`. When Optuna
  suggested extreme combinations (e.g. `dilation_base=4, n_layers=10,
  kernel_size=15`), the last layer's causal padding was
  `14 × 4^9 = 3.6 million`, creating multi-GB tensors in a single
  `F.pad()` call that triggered an instant SIGKILL from the OOM killer
  with no Python traceback. Fixed by capping dilation at `seq_len` (48)
  in the CNN model — beyond that the kernel can only see one timestep
  anyway, so the cap has no accuracy cost.
- **Tightened CNN tuning search space** — reduced `n_filters` max
  256→128, `kernel_size` max 15→7, `n_layers` max 10→8,
  `dilation_base` max 4→3 to keep Optuna in architecturally sensible
  ranges for 48-step sequences.

## 2.9.3

### Bugfix

- **Fix OOM detection in Docker containers** — memory monitoring now reads
  cgroup v2/v1 limits instead of `/proc/meminfo`, which shows host RAM
  and is meaningless inside the addon's container. The pre-trial memory
  check, available-MB logging, and abort threshold now all reflect the
  container's actual memory budget. This should prevent the OOM crash
  during CNN tuning on RPi5.
- **Reduced tuning batch size** (32 → 16) to halve peak memory per trial.
- **Removed unsafe cleanup code** — `ctypes.CDLL("libc.so.6").malloc_trim`
  (glibc-specific, unsafe on Alpine/musl) and parameter tensor zeroing
  removed. Standard GC is sufficient.
- **Better memory diagnostics** — logs now show container usage/limit,
  process RSS, and per-trial memory after each trial to make memory
  leaks visible.

## 2.9.2

### Bugfix

- **Prevent OOM crashes during neural model tuning** — multiple layers
  of defence against the Linux OOM killer during Optuna tuning on
  constrained hardware (RPi5):
  - **Batch size halved** during tuning (64 → 32) to reduce peak
    activation memory per forward pass.
  - **Epochs/patience reduced** (40/8 → 30/6) to shorten each trial.
  - **Aggressive PyTorch cleanup** between trials — model parameters are
    zeroed, gradient buffers cleared, three GC generations collected,
    and `malloc_trim(0)` called to release freed pages back to the OS.
  - **Memory pressure monitor** — checks `/proc/meminfo` before each
    trial. If available RAM drops below 256 MB, tuning aborts gracefully
    with the best result so far (instead of SIGKILL with no traceback).
  - **Memory logging** — each trial now logs available system memory so
    leaks are visible in the addon log.

## 2.9.1

### Bugfix

- **Atomic YAML writes prevent config corruption on crash** — all config
  save operations now write to a temporary file and use `os.replace()` to
  atomically swap it into place. Previously, `open('w')` truncated the
  config file immediately, so an OOM SIGKILL during tuning (while a
  concurrent UI settings save was in-flight) could leave the YAML empty
  or half-written, causing all experiments to disappear on restart.
- **Periodic config reload no longer replaces config with stub on
  failure** — if the 30-second config reload encounters a parse error
  (e.g. briefly unreadable file), the existing good config is kept
  instead of falling back to a stub config with no experiments. The stub
  fallback is now only used on the very first load.

## 2.9.0

### Breaking

- **Remove ensemble functionality entirely** — the ensemble engine, all
  ensemble API routes, ensemble UI sections, and ensemble event handlers
  have been deleted from the codebase. The `ensemble/` module directory,
  `EnsembleEngine`, `EnsembleResultData`, and `EnsembleMethodResult`
  classes are gone. Pipeline steps no longer include an "ensemble" stage.
  The "Include Ensemble" checkbox has been removed from both the Training
  page and experiment detail page. This simplifies the architecture and
  removes code that was already disabled since v2.5.0.

### Bugfix

- **Trigger immediate retrain on production toggle** — when an experiment
  is switched to production mode, an immediate retrain is now triggered
  so the production model gets cached and sensors start publishing right
  away. Previously, sensors wouldn't appear until the next scheduled
  retrain cycle (potentially hours after toggling).

## 2.8.5

### Bugfix

- **Validate restored benchmark results against current model config** —
  on startup, persisted benchmark results are now filtered against each
  experiment's `models_enabled` list. Models that have been disabled since
  the last run are removed, the best model is recalculated from the
  remaining valid models, and fully stale results (where no saved models
  are still enabled) are discarded entirely. Fixes "Publish lightgbm"
  showing when only LSTM is configured.
- **Hide stale best model on dashboard during training** — the Best Model
  row is now hidden while a benchmark is actively running.

## 2.8.4

### Bugfix

- **Global training lock prevents all concurrent training** — a single
  `asyncio.Lock` now serialises benchmarks (web UI), scheduled retrains,
  and manual retrain triggers. Previously these three code paths had
  independent queues that could overlap (e.g. a scheduled retrain
  starting while a benchmark was running).

## 2.8.3

### Improvement

- **No automatic retrain on restart/update** — production experiments
  no longer force an immediate retrain when the add-on starts. The
  first retrain waits for the normal `retrain_every` schedule.
  Benchmark results are restored from SQLite, and forecasts gracefully
  skip until a cached model is available. Users can still trigger a
  manual retrain from the web UI at any time.

## 2.8.2

### Bugfix: scheduled retrains now queue sequentially

- **Production retrains no longer run in parallel** — the v2.8.0 queue
  only covered web UI "Run Pipeline" clicks. Scheduled retrains in the
  main loop used `create_task` per experiment, so all experiments due at
  the same time (especially on startup) would train simultaneously.
  Now uses an `asyncio.Queue` with a single consumer that drains one
  experiment at a time, preventing memory exhaustion on RPi.

## 2.8.1

### Bugfix

- **Fix reversed colours in covariate analysis** — percentage values in
  the drop-one table now show green for valuable covariates (removing
  hurts the model) and red for harmful ones (removing helps). Previously
  the colours were inverted, showing green next to Remove buttons.

## 2.8.0

### Feature: sequential training queue

- **Experiments now queue instead of running in parallel** — clicking
  Run Pipeline on multiple experiments queues them and runs one at a
  time, preventing memory exhaustion on constrained hardware (e.g. RPi).
- **Dashboard shows queue position** — queued experiments display an
  amber "Queued (#N)" button. Clicking it removes the experiment from
  the queue.
- **Stop Training handles queued experiments** — removes from queue
  if not yet started, or cancels the running task if in progress.

## 2.7.6

### Bugfix: Stop Training leaves UI stuck in loading state

- **Fixed wrong keyword argument in stop-training callback** — the
  `pipeline_end` event was constructed with `experiment=` instead of
  `experiment_name=`, causing a silent `TypeError`. The event was never
  stored in the training event history, so on page reload the JS
  replayed a `pipeline_start` with no matching `pipeline_end` and
  locked the Run Pipeline button in a permanent loading spinner.
- **Added server-side guard in replay logic** — the JS now checks the
  server's `is_running` flag before entering loading state. Even if the
  event history is stale, the UI won't show a stuck spinner when the
  server knows training has stopped.

## 2.7.5

### Feature: persist benchmark results across restarts

- **Benchmark results now survive add-on updates and restarts** — results
  are serialised to SQLite (`benchmark_results` table) as JSON after each
  benchmark run. On startup, stored results are restored into memory along
  with `best_model`, `selected_model`, and `last_benchmark_status`.
- Benchmark data is cleaned up when an experiment is deleted.

## 2.7.4

### Improvement

- **Graceful skip on insufficient covariate history** — when covariates
  don't have enough data, the pipeline now logs a warning and skips the
  cycle instead of failing with an error on the dashboard. The next
  scheduled cycle will retry automatically once sensors have accumulated
  enough history.

## 2.7.3

### Bugfixes

- **Fix covariate removal from Settings tab** — two conflicting
  `removeCovariate` JS function definitions caused the Covariate Analysis
  tab version (3 args) to overwrite the Settings tab version (1 arg).
  Clicking "×" on a covariate in Settings silently failed because the
  entity ID was misrouted as the experiment name. Renamed the Covariate
  Analysis version to `removeCovFromAnalysis` to eliminate the collision.
- **Guard against empty DataFrame after preprocessing** — if covariates
  have insufficient history (e.g. a freshly-created template sensor),
  the pipeline now raises a clear `ValueError` instead of crashing with
  an `IndexError` on an empty index.
- **Dashboard button colours** — Stop Training is now red (`btn-danger`),
  Publish/Publishing is now green (`btn-success`) with proper hover and
  disabled states.

## 2.7.1

### Feature: publish forecast accuracy as HA sensor

- **`sensor.mlfl_{name}_forecast_accuracy`** — publishes the lead-time
  accuracy curve as a HA sensor entity after each production forecast.
  State is the shortest-lead MAE; attributes contain `lead_hours`, `mae`,
  `rmse`, `sample_count`, and revision improvement metrics. Enables
  ApexCharts dashboard cards for accuracy visualisation.

## 2.7.0

### Feature: forecast evolution log & accuracy tracking

- **Forecast evolution log** — every production forecast is now logged to
  SQLite (`forecast_log` table) with the wall-clock issue time, each
  predicted target timestamp, lead time in minutes, model name, and whether
  it was a retrain or cached-model forecast.
- **Forecast Accuracy tab** — new tab on production experiment pages showing:
  - Lead-time vs MAE/RMSE chart (how accuracy degrades with longer horizons)
  - Revision improvement card (does re-forecasting via `forecast_every`
    actually improve accuracy?)
  - Total logged points and date range
- **Auto-slugify experiment names** — the Create Experiment modal now accepts
  human-readable names (e.g. "Optimised Solar") and auto-converts to valid
  slugs (`optimised_solar`). Removed the strict browser pattern validation.
- **Removed uppercase labels** — `.field-label` no longer forces uppercase
  text-transform across the UI.
- Forecast log is pruned alongside history cleanup and deleted when an
  experiment is removed.

## 2.6.3

### Fix: production forecast sensors never published

- **Fixed undefined `forecast_features` variable** in `_run_production_inference()`
  (line 1864). The variable was a leftover from the removed
  `create_forecast_features()` function and caused a `NameError` on every
  production forecast, silently preventing sensor publication. Now constructs
  `ds_future` from `last_ts` + interval offsets, matching the working
  `_forecast_with_cached()` implementation.

## 2.6.2

### UI: redesign Settings tab layout, move Stop Training to dashboard

- **Settings tab** — replaced flat field list with grouped card panels
  (Target, Data & Forecast, Training, Covariates). Toggles are now inline
  with labels, fields use an explicit 3-column grid, and each section has
  a bordered card background for visual separation.
- **Stop Training on dashboard** — the stop button now appears directly in
  the experiment card footer when training is running, replacing the
  Publish button. Removed from the Settings tab.
- **Auto-migrate config** — `load_config()` silently strips the deprecated
  `horizons_minutes` field from the YAML on first load, preventing the
  repeated "Ignoring unknown experiment fields" warning.

## 2.6.1

### Cleanup: remove dead `horizons_minutes` config & improve Settings layout

- **Removed `horizons_minutes`** — benchmarking evaluates the full forecast
  window (`future_periods`), not specific horizon checkpoints. The field,
  `create_forecast_features()` function, save-horizons API route, horizon
  chip UI, dashboard horizon gauges, and all related tests have been deleted.
- **Settings tab layout** — moved field grid styles from inline `<style>` to
  `style.css`, removed all-caps labels, increased spacing between sections
  and fields for better readability.

## 2.6.0

### Feature: full UI-driven experiment configuration

All experiment settings can now be managed through the web dashboard —
no more editing `mlfl.yaml` by hand.

**Settings tab** — a new first tab on every experiment page that
consolidates all per-experiment configuration in one place:

- **Target**: entity ID (read-only), cumulative source, daily reset,
  max increment
- **Data**: history days, interval minutes, log transform
- **Forecast**: future periods, per-experiment forecast and retrain
  intervals (nullable = use global default)
- **Training**: CV strategy, folds, recency half-life, production
  metric, neural loss function
- **Covariates**: full add/remove/edit management with a searchable
  Home Assistant entity picker (debounced, cached 60 s). Each covariate
  shows entity, role, aggregation, scale, and binary flag. No YAML
  editing required.
- **Stop Training**: a red button (with confirmation) that cancels a
  running retrain or tuning task. Cancellation takes effect after the
  current epoch completes for neural models.

**New Experiment creation** — a "+ New Experiment" button on the
dashboard opens a modal with name, target entity (with HA search),
cumulative source, and daily reset fields. Creates the experiment in
YAML and redirects to the new experiment's page. Also shown in the
empty-state when no experiments exist yet.

**Delete Experiment** — experiments can be removed via the API
(`POST /api/experiments/{name}/delete`), cleaning up both YAML and
in-memory state.

**System page simplified** — experiment cards on `/system` are now
read-only summaries with a "Configure →" link to the experiment's
Settings tab. Avoids two edit surfaces for the same fields.

### Backend changes

- `config.py`: added `add_experiment_covariate()`,
  `save_horizons()`, `create_experiment()`, `delete_experiment()`
- `app.py`: new routes for add-covariate, save-horizons,
  create/delete experiment, HA entity search (`/api/ha/entities`),
  stop-training; extended `experiment-settings` to accept
  `max_increment` (nullable)
- `main.py`: training tasks tracked in `_running_tasks` dict for
  cancellation support; stop callback emits `pipeline_end` SSE event
  so the Training tab transitions cleanly
- `style.css`: new component styles for horizon chips, entity search
  dropdown, covariate rows, and danger button

## 2.5.12

### Fix: neural model tuning OOM / hangs on RPi5

Optuna-based hyperparameter tuning for neural models (LSTM, CNN, etc.)
would frequently run out of memory or hang indefinitely on
memory-constrained devices like the Raspberry Pi 5 (8 GB). Three
compounding issues were identified and fixed:

**1. Memory accumulation across Optuna trials.**
Each trial instantiated a PyTorch model but never explicitly freed it.
Over 20+ trials the heap grew until the OOM killer sent SIGKILL,
crashing the container with no error in the logs. Every trial's
`objective()` now wraps execution in a `try/finally` that runs
`del model`, `torch.cuda.empty_cache()`, and `gc.collect()` — the same
pattern is applied to the baseline trial.

**2. Redundant sliding-window computation.**
`create_sliding_windows()` was called fresh on every trial for every
fold, rebuilding identical NumPy arrays each time. For a 90-day history
with `seq_len=168` this was ~3 seconds per trial × 30 trials = 90 s of
pure waste. Sliding windows are now pre-computed once before the study
begins and passed to `run_single_model()` via a new
`precomputed_sequences` parameter. The pre-computed arrays are freed in
a `finally` block after the study completes.

**3. No timeout for slow trials.**
A single bad hyperparameter combination (e.g. very large hidden size
with many layers) could train for hours without interruption. Neural
model studies now pass `timeout=1800` (30 minutes) to
`study.optimize()`, which gracefully stops the study and uses the best
result found so far. A log message is emitted when the timeout fires.

## 2.5.11

### Fix: production retrain failures are now surfaced in the UI

Three issues that made retrain/model-change failures hard to diagnose:

**1. FAILED status was never set on the production retrain path.**
`_retrain_single` caught exceptions and logged them, but never updated
`last_benchmark_status` to `"failed"`. The dashboard badge stayed at its
previous value (usually `"completed"`) even when the retrain crashed.
Both `_retrain_single` and `_forecast_single` now set `"failed"` and
store the error message on the `ExperimentStatus`.

**2. The error message was invisible in the UI.**
Added a `last_error` field to `ExperimentStatus` and an **Error** row
on the dashboard card that appears whenever the status is `"failed"`.
The next successful cycle clears it automatically. No more digging
through log files to find out what went wrong.

**3. Stale model cache when `production_model` changes.**
When you changed `production_model` in the YAML (e.g. `lstm` → `cnn`),
the old cached model stayed in memory until the next retrain replaced it.
Intermediate forecast cycles would use the stale cache (with the old
model's `feature_cols`, `seq_kwargs`, `exp_cfg`), which could produce
wrong predictions or crash with a shape mismatch. `_retrain_single` now
detects when the configured production model differs from the cached one
and invalidates the cache before starting the retrain.

## 2.5.10

### Fix: 1-hour timezone offset on the dashboard chart (DST-safe)

The forecast curve in `attributes.forecast` (and `_cumulative`,
`_daily_cumulative`, and `recent_actuals`) was being serialized with naive
ISO timestamps — e.g. `"2026-04-09T20:00:00"` with no `+00:00` suffix —
because the upstream pipeline strips timezones from the SQLite cache for
storage. JavaScript's `new Date(...)` interprets such strings as **local
time**, not UTC, so users in any timezone with an offset (BST = UTC+1,
EST = UTC−5, etc.) saw the MLFL series shifted on their charts. The
PredAI / Mixergy series came from HA's history API which always emits
tz-aware ISO strings, so they were plotted correctly — only the MLFL
series was offset.

`_publish_forecast_sensors` now localizes `ds_future` to UTC before
serializing, so all four sensors emit `"...+00:00"` strings that the
browser parses as absolute instants and renders in the local timezone.
The `recent_actuals` block in `_run_production_inference` does the same
for the historical context window.

This is fully **DST-safe** in both directions:

- The published timestamps are anchored to UTC, which has no DST
- The browser converts to local time using its own IANA tz database, so
  the chart automatically tracks BST → GMT (last Sunday of October) and
  GMT → BST (last Sunday of March) without any add-on changes
- The daily-cumulative day-bucketing already uses `zoneinfo` for the
  local-date calculation, which handles 23h/25h transition days

**Visible effects after upgrade**: from the next forecast cycle, the
MLFL series on your ApexCharts dashboard will line up exactly with the
PredAI and Mixergy actuals — no horizontal shift. Forecasts published
before upgrading will still show the offset until the next cycle runs.

## 2.5.9

### Fix: `state_class` on cumulative sensors

The `_cumulative` and `_daily_cumulative` sensors were being published
with `state_class: "total"` and `"total_increasing"` respectively. Both
were wrong: those state classes are for monotonic counters that HA's
long-term statistics engine processes as energy-meter-style totals,
which is not what these sensors are.

These sensors publish a **per-cycle snapshot** of a forecast projection:

- `_cumulative` state = predicted cumulative value at the end of the
  forecast horizon (changes each cycle as the model is re-run)
- `_daily_cumulative` state = predicted total demand for today at local
  midnight (fluctuates as the seed grows and remaining-forecast shrinks)

Neither is a monotonic counter. Both are now `state_class: "measurement"`,
which is HA's convention for values that go up and down freely.

**Visible effects after upgrade**:

- The entity history graph in HA's more-info dialog will plot the actual
  state values (a fairly flat line wobbling around the projected total),
  not an accumulating curve
- The sensors will no longer be suggested for the HA Energy dashboard
- The dashboard chart in your existing ApexCharts cards is unchanged
  (those use `attributes.forecast`, not the long-term statistics)

The existing distorted long-term statistics from before this fix will
fade as new correctly-tagged data comes in (~7 days). To clear them
immediately, go to Developer Tools → Statistics → Fix issues.

## 2.5.8

### Fix: `_daily_cumulative` state is now the end-of-today projection

The `sensor.{prefix}{name}_daily_cumulative` state was being set to the
last point in the forecast curve, which for a 48h horizon sits mid-way
through day-after-tomorrow **after** two intervening midnight resets.
That made the headline state something like 8.6% while the curve on a
chart reached ~55% at end of today, which was confusing.

The state is now set to the projected cumulative value at the **last
forecast point still within today's local date** — i.e., "what is the
total predicted demand for today by local midnight". This is directly
comparable to `sensor.<target>_today` at end of day.

The previous behaviour is preserved as two new sensor attributes:
- `end_of_today_value` — the new headline state
- `end_of_horizon_value` — the old headline state, for reference

The per-interval curve in the `forecast` attribute is unchanged — it
still resets at each local midnight throughout the whole horizon, so
dashboards that plot the full curve (like ApexCharts) continue to work.

## 2.5.7

### Fix: cumulative / daily-cumulative forecast sensors are now actually published

The `publish_interval`, `publish_cumulative` and `publish_daily_cumulative`
flags in each experiment's config were silently ignored by the
retrain-and-cache code path. Only `sensor.{prefix}{name}_forecast` (the
per-interval forecast) was being published, which made it impossible to
plot the forecast on the same scale as a daily-cumulative source sensor
without writing custom JS in your dashboard `data_generator`.

The new `_publish_forecast_sensors` helper consolidates all the publishing
logic and honours the flags. With `publish_daily_cumulative: true` and
`source_is_cumulative: true` (the typical Mixergy / energy-meter setup),
the helper now publishes:

- `sensor.{prefix}{name}_forecast` — main per-interval curve (always)
- `sensor.{prefix}{name}_interval` — same data, dedicated sensor
- `sensor.{prefix}{name}_cumulative` — running cumsum across the horizon
- `sensor.{prefix}{name}_daily_cumulative` — cumsum that **resets at local
  midnight** and is **seeded with the current value of the target sensor**
  so the forecast meets the actuals exactly at the join point

The daily-cumulative seeding uses the experiment's local timezone (from
the global `timezone` setting) and reads the live target sensor state, so
ApexCharts cards comparing the actuals against the mlfl forecast no longer
need any custom `data_generator` cumsum logic.

### Fix: stop labelling `%` forecasts as `power_factor`

`_run_production_inference` was setting `device_class: "power_factor"` on
the published forecast whenever `units == "%"`. Home Assistant's
`power_factor` device class is for AC efficiency ratios (0–1 dimensionless)
and overrides unit display, which is why the Mixergy demand forecast was
showing as a bare number with no `%` unit. The device_class is now omitted;
HA accepts `%` as a regular unit and displays it correctly.

### Refactor: deduplicated publishing path

`_run_production_inference` and `_forecast_with_cached` previously had
two near-identical inline publishing blocks. Both now route through
`_publish_forecast_sensors`, so future changes to the published attributes
or sensor naming only need to be made once.

## 2.5.6

### Removed: Hailo AI accelerator integration

The Hailo integration has been removed entirely. After investigation we
confirmed it could not possibly work on-device for custom-trained models:

- **Hailo's Data Flow Compiler (DFC) is x86-64 Linux only**. No ARM build
  exists and Hailo doesn't publish one. For custom-trained models this
  rules out any on-Pi compilation path, and Hailo's own documentation
  recommends ~32 GB RAM during quantisation — eliminating QEMU emulation
  on an 8 GB Pi 5 as a workaround.
- **The existing scaffolding was also broken in its own right**.
  `_retrain_and_cache` passed the ONNX file path to
  `HailoAcceleratedModel(model, hef_path=onnx_path)` where the class
  expected a compiled HEF file, not ONNX. The wrapper silently fell back
  to CPU inference while the validation check passed vacuously (both
  sides were CPU), so `hailo_active=True` was set in the dashboard while
  the NPU was idle. No forecasts were ever actually accelerated.
- **`compile_onnx_to_hef` in `hailo_runtime.py` was a placeholder**
  that logged instructions and returned `False` — it was never called
  from anywhere in the codebase.

Rather than ship a half-working integration that lied to the dashboard,
the entire Hailo code path is gone. Users with a Hailo hat should use it
with Frigate NVR, which works great because pre-compiled HEFs for
YOLO-family vision models are distributed with the add-on.

**What was removed**:
- `ml_forecast_lab/models/hailo_runtime.py` (entire file)
- `ml_forecast_lab/models/onnx_export.py` (entire file — only existed
  to support the `export_onnx` hooks in each backend)
- `export_onnx` and `supports_hardware_accel` methods from all 15
  model backends and from the `ForecastModel` abstract base class
- `hailo_enabled` config option from `AppConfig`
- `hailo_active` field from `ExperimentStatus`
- Hailo branch in `_retrain_and_cache`
- `is_hailo` / `hailo_accelerated` from cached model metadata and
  published sensor attributes
- Hailo checkbox from the System page + matching JS
- Hailo badge from dashboard experiment cards
- `python3-hailort` apt install + `--system-site-packages` venv tweak
  from the Dockerfile
- `/dev/hailo0` device mapping + `SYS_RAWIO` privileged cap from the
  add-on's `config.yaml`
- `onnx>=1.14.0` from `requirements.txt` (only used by `onnx_export.py`)
- Hailo section from `README.md` + `CONFIG_GUIDE.md` +
  `CREATION_REPORT.md`

**What you lose**: nothing functional. The Hailo code path never
actually ran on the NPU in practice, so removing it doesn't change
forecast correctness, training speed, or inference speed. The only
user-visible change is that the dashboard will stop lying about Hailo
being active and the `ONNX export failed: No module named 'onnxscript'`
warnings will disappear from the logs.

### Fix: NeuralProphet now works

The `neuralprophet` backend file was in the repo but the `neuralprophet`
pip package was missing from `requirements.txt`, so:

1. Docker image never installed the package
2. `from neuralprophet import NeuralProphet` raised `ImportError`
3. The `_optional_backends` loop in `main.py` silently swallowed the
   error and never registered the model
4. Enabling `neuralprophet` in `models_enabled` produced a `KeyError`
   from the registry

Fixed by adding `neuralprophet>=0.8.0,<1.0.0` to `requirements.txt`
(the first NeuralProphet release fully compatible with PyTorch 2.x and
PyTorch Lightning 2.x that the add-on already uses). Also added
`NeuralProphetModel` to the `_optional_imports` dict in
`models/__init__.py` for parity with the other backends.

Expected Docker image growth: ~50-100 MB of transitive deps
(PyTorch Lightning, matplotlib backends). Not a concern for Pi 5.

## 2.5.5

### Performance: neural-model tuning is now 10-20x faster

Neural hyperparameter tuning was catastrophically slow because of two
layered problems:

1. **`epochs` and `patience` were in the tuning search space itself.**
   Every neural `MODEL_PARAM_SCHEMA` entry listed `"epochs": {min: 10,
   max: 1000}`. Optuna would happily suggest `epochs=800` for some
   trials, so a single trial could train for up to ~10 minutes on
   RPi5 — and TPE tends to push toward high epoch counts early in the
   search because more epochs → lower validation loss (up to
   overfitting). This is a classic tuning anti-pattern: the training
   budget isn't a hyperparameter, it's a fixed resource decision.
2. **2-fold CV per trial doubled the cost** when 1 fold is enough to
   rank candidate hyperparameter sets. Optuna is robust to noisy
   objectives and a single well-sized fold is sufficient for the
   relative comparison.

Concrete numbers, Mixergy LSTM on RPi5:
* Before: ~30-60 minutes per tuning run (30 trials)
* After: ~3-6 minutes per tuning run

### Changes

* **Removed `epochs` and `patience` from all 13 neural
  `MODEL_PARAM_SCHEMA` entries** (lstm, cnn, dlinear, nbeats, nhits,
  tide, tsmixer, sparsetsf, patchtst, itransformer, crossformer,
  timesnet, neuralprophet). They're training budget decisions, not
  tuning targets.
* **Tuning now uses 1 CV fold** instead of 2 (`cv_folds = 1` in the
  runner config passed to `_run_tuning`).
* **Neural trials are capped at 40 epochs with patience 8** via a
  small `_apply_tuning_overrides(model)` helper that's called on both
  the baseline trial and every Optuna objective trial. This caps the
  per-trial budget without touching the model's own defaults — so the
  *production retrain* triggered by "Apply Tuned Params, Promote &
  Retrain" uses the full 100-epoch budget on the tuned model.
* **Log line** at tuning start shows the active budget: `Tuning
  budget: 30 trials × 1 CV fold × max 40 epochs (neural) /
  early-stopping (trees)`.
* **Tuning help tooltip** updated to explain the budget and the
  full-epoch production retrain.

Tree model tuning is unchanged (LightGBM/XGBoost already respect
their own early-stopping rounds and don't have the epochs-as-hparam
problem).

## 2.5.4

### Fix: Covariate Analysis now trains neural models correctly

Neural models in the Covariate Analysis path were being trained with
**flat features only** — no sliding windows, no dense horizons, no
residual prediction. That meant the LSTM / CNN / etc. metrics reported
in the covariate analysis grid came from a crippled version of those
models, not the production version, so the "does removing X help the
LSTM?" comparison was basically meaningless for any backend the
leaderboard considered competitive.

The inner training loop now mirrors the CV runner and holdout chart
for neural models:

* Builds sliding windows with `create_sliding_windows` using the full
  set of temporal and covariate channels
* Trains with dense `horizon_steps=[1..future_periods]` so residual
  prediction has something to optimise against
* Predicts on the test split using `horizon_steps=[1]` for one window
  per test row (full coverage)
* Reduces the multi-horizon output to h=1 for metric computation

Tree models keep the existing flat-features path unchanged. Both
families still use the same 80/20 split for consistency with each
other, so the cross-covariate comparison is fair.

### Rename: `deep_analysis` → `covariate_analysis` throughout the code

The UI has always called this feature "Covariate Analysis" but the
code used the old `deep_analysis` identifier from an earlier
iteration. Renamed all 60+ references:

* Python identifiers (`covariate_analysis_results`,
  `covariate_analysis_callback`, `_run_covariate_analysis`,
  `_covariate_analysis_trigger`)
* Pydantic classes (`CovariateAnalysisResult`,
  `CovariateAnalysisCellResult`)
* HTTP endpoints (`POST /run-covariate-analysis`,
  `GET /covariate-analysis`)
* Pipeline step name (`"covariate_analysis"` in the `run-pipeline`
  steps list)
* Template variable (`covariate_analysis`)
* HTML element IDs (`sec-covariate-analysis`,
  `covariate-analysis-model`, `covariate-analysis-btn`)
* JavaScript function (`runCovariateAnalysis`)
* Log strings, docstrings, and CSS comments

No behavioural change — this is purely a rename so the internal code
matches the user-facing name. The old endpoints no longer exist, so
any external integrations that hit them will need updating.

## 2.5.3

### Fix
- **Auto-ensemble was still running after every benchmark.** v2.5.0 hid the
  Ensemble tab from the navigation but the two `_run_ensemble()` calls
  inside `_benchmark_trigger` and `update_experiment` were still firing
  after every CV run, wasting compute and producing results no one could
  see. Both auto-trigger sites are now disabled. The ensemble code path,
  the `ensemble_callback`, and the section's HTML are all left in place
  so the feature can be re-enabled by uncommenting the nav link if needed.
- **`remove-covariate` couldn't actually find the covariate to remove.**
  Pre-existing bug: the deep-analysis "Remove" buttons send the short
  name (`current_charge`) but `remove_experiment_covariate` was matching
  against the full entity ID (`sensor.current_charge`), so the helper
  silently failed and the YAML was never edited. The helper now matches
  either form. Verified with five edge-case tests.

### New: clearer apply / publish workflow on Tuning + Covariate tabs

Both tabs now have a single prominent button that finalises the analysis
result and immediately starts a fresh retrain — no manual `mlfl.yaml`
editing, no waiting for the next scheduled retrain cycle.

**Tuning tab — "Apply Tuned Params, Promote & Retrain"** (was "Apply &
Promote"). The button now:
1. Saves tuned params to `mlfl.yaml` (existing behaviour)
2. Promotes the tuned model to production (existing behaviour)
3. **Triggers an immediate background retrain** (new) so the live
   forecast sensor picks up the new params right away
4. Switches the UI to the Training tab so the user can watch the live
   retrain progress, then reloads to refresh state
A short explanatory paragraph below the button spells out what each
click will do.

**Covariate Analysis tab — "Apply Best & Retrain"** (new green panel
above the results table). Reads the latest deep-analysis run, picks the
covariate configuration with the lowest average MAE across models, and:
* If "All covariates" wins → reports "already optimal", no changes
* If "No covariates" wins → clears the experiment's covariate list
* If "Without X" wins → removes covariate X
…then triggers an immediate background retrain. Toast shows which
covariates were dropped and the % MAE improvement vs baseline.

Backed by a new `POST /experiment/{name}/apply-covariate-best` endpoint
and a new `clear_experiment_covariates()` config helper. The existing
per-row "Remove" buttons stay as fine-grained controls.

### Plumbing
- New `retrain_callback` slot on `AppState`, registered by `main.py` at
  startup. Apply endpoints schedule retrains via this callback so they
  don't have to import `MLForecastLabApp` directly.

## 2.5.2

### Fix
- **Retrain cycle crashes with `unexpected keyword argument 'is_lab'`**.
  Two callers of `update_experiment()` (in `_retrain_single` and the
  `_run_retrain_cycle` loop) were passing `is_lab=True`, but the method's
  parameter is `is_lab_mode`. The mismatch broke every retrain cycle for
  lab-mode experiments. Fixed both call sites to use the correct keyword.

## 2.5.1

### Fix
- **Add-on crashes at startup with `ValueError: Unknown level: '5\n'`**.
  The HA add-on base image (hassio-addons/ubuntu-base, via bashio +
  s6-overlay) exports `LOG_LEVEL` as a *bashio* level, which can be a
  numeric string ("5" = NOTICE), a bashio name (`TRACE`/`NOTICE`/`FATAL`)
  that Python's `logging` module doesn't recognise, or sometimes a value
  with a stray trailing newline from `/var/run/s6/container_environment/`.
  Our `__main__.py` was passing the raw value straight to
  `root_logger.setLevel`, which crashed the entire add-on at startup.
  Replaced with a robust `_parse_log_level` helper that:
  * strips whitespace
  * maps bashio numeric levels (0–8) to Python equivalents
  * maps bashio string names (TRACE/NOTICE/FATAL/OFF) to Python equivalents
  * accepts standard Python names (DEBUG/INFO/WARNING/ERROR/CRITICAL)
  * falls back to `INFO` on any unrecognised input rather than crashing.

## 2.5.0

### New: Daily-cumulative leaderboard metrics

The Model Comparison section now has TWO tables:

- **Per-Interval Accuracy** (top, existing): MAE / RMSE / MASE / Mean Rank
  on the next-step (h=1) prediction. Drives Promote, Tuning, and live
  forecasting. Column headers relabelled to make the per-interval framing
  explicit.
- **Daily Cumulative Accuracy** (new, below): same metrics but computed on
  per-day totals — each day's predictions summed, each day's actuals summed,
  then compared. Includes a separate **Daily Rank** computed via the same
  Demšar (2006) composite as the primary rank but applied to daily metrics.

Daily Rank is informational only — it doesn't drive any production
workflow — but it lets you pick the model that matches your objective. For
use cases like daily energy or hot-water demand where the daily total is
what matters, this is now the leaderboard view to read.

The CV runner captures per-fold timestamps and groups predictions/actuals
by date inside each fold. The Demšar ranking helper has been factored out
and is called twice (interval + daily), so both rankings stay in sync with
any future ranking-logic changes.

### New: Tuning optimises a composite MAE+RMSE+MASE loss

Hyperparameter tuning previously asked Optuna to minimise MAE alone, then
picked the final winner via a post-hoc rank composite. Now Optuna's search
objective is the composite directly: each trial's score is the average of
`(mae / mae_default + rmse / rmse_default + mase / mase_default) / 3`,
where `_default` comes from one CV evaluation with the model's default
parameters run before the search starts. A composite of 1.0 matches the
default; 0.85 means a 15% average improvement across all three metrics.
Optuna's TPE/Random search now actually optimises the composite throughout,
not just at the very end.

The tuning panel UI now shows "Best composite" instead of "Best MAE", and
the help tooltip explains the composite scoring.

### Removed: Training Loss Curves panel

The collapsible "Training Loss Curves (neural models)" section in the
Results tab has been removed — the Generalisation tab already shows the
Train vs Test gap and Fold Stability lines, which are more useful for
diagnosing overfitting than per-epoch loss curves were.

### Disabled: Ensemble tab

The Ensemble tab is hidden in the navigation. The section content is left
in place behind a `{% if false %}` Jinja gate so it can be re-enabled by
uncommenting one line if needed in the future.

### Polish
- Generalisation tab now has a one-line note above the Train vs Test
  table clarifying "All values below are per-interval (h=1) errors" so
  there's no ambiguity about which metric scope you're reading.

## 2.4.0

### Critical fix: CV runner now matches the holdout chart and production paths

The CV runner was training a fundamentally different model from the one shown
on the holdout chart and used in production:

- **Sparse vs dense horizons.** CV used `horizons_minutes` from config to build
  `horizon_steps = [4, 16, 24, 48]` (4-output multi-head), while the holdout
  chart and production training both used dense `[1..future_periods]`
  (96-output dense). Same neural backend, completely different architecture.
- **Different window size.** CV used `max(48, max_horizon * 2) = 96`, the
  holdout chart used `min(48, len/3)`. Different receptive field.
- **Wrong "h=1" indexing in v2.3.1's fairness fix.** That fix used
  `horizon_steps[0]` as the ranking metric. With `[4, 16, 24, 48]` that's
  actually h=4 (2 hours ahead) — neural models were being scored on a
  fundamentally harder task than tree models, which produce h=1 (next-step)
  predictions. Hence the misleading 2.5× MASE gap on the leaderboard while
  the same models tracked actuals just fine on the holdout chart.

The CV runner now uses the **same** dense horizons and the **same** window
size as the holdout chart and production. With the per-fold test path also
switched to `horizon_steps=[1]` for inference, neural models get one window
per test row (full coverage) and the metric is computed on h=1 for both tree
and neural families. The leaderboard, the holdout chart, and the live
forecast sensor now all evaluate the same model.

### Holdout chart: full neural coverage without tail-fill

The v2.3.1 tail-fill (last window's h=2..96 outputs) was fragile because
residual prediction over long horizons collapses to "stay near the last
observed value" — that's why CNN/LSTM lines went flat for the final ~48h.
Replaced with `horizon_steps=[1]` inference, which gives one unique window
per holdout point. Each chart point is now a true 1-step-ahead prediction
from its own input window.

### Cleanup
- Dropped per-horizon sub-metrics (`mae_h2`, `rmse_h96`, `mae_havg`, etc.)
  from `fold_metrics`. They were never surfaced in the UI and were a
  consequence of the CV runner's old multi-horizon path.
- Simplified the runner's metric block — both tree and neural models now
  produce 1D `y_pred` / `y_test`, so a single `compute_all` call handles
  the full leaderboard.

## 2.3.2

### Improvements
- **Auto-assigned plot colours**: removed the hard-coded `MODEL_COLORS` /
  `ENSEMBLE_COLORS` dicts in `main.py` and the duplicated `MODEL_COLORS` map
  in `experiment.html`. All multi-trace charts now share a single 15-colour
  Plotly `colorway` defined once in the template, and traces consume colours
  in order. Adding a new model no longer requires touching a palette in two
  places, and the backend `ModelPrediction.color` field has been retired.
- **Help text refresh**: updated tooltips on the system page (forecast /
  retrain interval no longer mention a non-existent "global default", Hailo
  toggle now describes the CPU-vs-NPU validation test and graceful fallback,
  Settings section heading now covers the full set of global options) and the
  Model Comparison ranking explanation now states that all metrics are
  computed on the next-step (h=1) prediction so tree and neural models are
  compared on the same horizon.
- **README refresh**: configuration example now uses
  `forecast_every_minutes` / `retrain_every_hours` (the deprecated
  `update_every_minutes` was the v1.x name); features list now reflects all
  15 model backends, decoupled retrain/forecast cycles, hyperparameter tuning,
  ensembles, and covariate analysis; Hailo section mentions all neural
  families and the validation fallback.

### Fix
- **Config-loaded log spam**: the timer loop reloads `mlfl.yaml` every 30s
  to pick up UI edits and was logging `Configuration loaded from …` at INFO
  on every reload — drowning out training progress. Now uses an mtime check:
  the first load logs at INFO, real edits log `Configuration reloaded from
  …` at INFO, and unchanged reloads drop to DEBUG.

## 2.3.1

### Fix
- **Holdout chart now spans full time period for neural models**: with dense
  multi-horizon training (`horizon_steps=[1..96]`), `create_sliding_windows`
  produces only `len(holdout) - 95` valid windows, so the Predictions tab chart
  was missing the final ~48h of CNN/LSTM/etc. predictions while LightGBM
  covered the entire holdout. Neural models now reconstruct full coverage by
  using the LAST window's higher horizons (h=2..96) to fill the tail. Each
  holdout point gets exactly one prediction at the right horizon offset.
- **Fair leaderboard ranking between tree and neural models**: the un-suffixed
  ranking metric for neural models was the *average* across all 96 horizons,
  while tree models reported a single h=1 metric. Averaging long-horizon errors
  systematically penalised neural models — even when their h=1 predictions were
  competitive. The ranking metric now uses h=1 only for both model families,
  matching the chart and the natural tree-model evaluation point. The
  horizon-averaged variant is still recorded as `mae_havg`, `rmse_havg`, etc.
  for diagnostics.

## 2.0.3

### Improvement
- **Tuning best trial uses composite ranking**: the winning trial is now
  selected by average rank across MAE, RMSE, and MASE (same methodology as
  the Results table), not just lowest MAE. Optuna still minimises MAE to guide
  the search, but the final winner is the trial that performs consistently
  well across all three metrics.

## 2.0.2

### Fix
- **Covariate Analysis recommendations**: percentages now match the table
  (both use baseline as denominator). Wording clarified: "Consider removing X
  — dropping it reduces MAE by Y%" instead of confusing "X is harmful" language.
  Overall recommendation now says "performs better without covariates" when
  removing all covariates improves MAE.

## 2.0.1

### Fix
- Dashboard card layout: View Details button and publish toggle back in the
  card footer matching the original two-element layout.

## 2.0.0

### New: Decoupled Timers + Hailo AI Acceleration
- **Separate forecast and retrain schedules**: forecast cycle (default every 30m)
  uses a cached trained model for fast inference (<1s). Retrain cycle (default
  every 24h) trains from scratch and updates the cache. Dashboard shows both
  countdowns independently.
- **Hailo AI hat integration**: after each retrain, neural models are exported to
  ONNX and wrapped with `HailoAcceleratedModel` for NPU-accelerated inference.
  A validation test (CPU vs Hailo comparison) runs on every retrain — if it fails
  or diverges >1%, the system falls back to CPU with a logged warning.
- **Model caching**: trained models are cached in memory between retrains. Forecast
  cycles reuse the cached model, eliminating redundant retraining.
- **System page**: "Update Interval" renamed to "Forecast Interval" + new "Retrain
  Interval" field. Both configurable independently.
- **Backward compatible**: existing `update_every_minutes` in YAML is automatically
  mapped to `forecast_every_minutes`.

## 1.24.4

### Improvements
- **Dashboard publish toggle**: replaced large Publish/Stop Publishing buttons
  with a compact toggle switch matching the Models tab style.
- **Tuning holdout chart**: moved above the trials table (which is now collapsible).
  Added interval/cumulative toggle matching the Predictions tab style.

## 1.24.3

### Bug Fix
- **Neural production inference uses multi-head prediction**: replaced the
  autoregressive sliding-window loop with a single `predict_sequence()` call
  that outputs all horizons at once. This matches how the models are trained
  (direct multi-output) and avoids error accumulation between forecast steps.
  Intermediate points are linearly interpolated between horizon anchors.

## 1.24.2

### Improvements
- **Dashboard publish toggle**: replaced vague "Production" / "Lab Mode" button
  with clear "Publish" / "Stop Publishing" actions. When publishing, the card
  shows the HA sensor entity ID being published to.
- **Consistent language**: "Promote" renamed to "Publish" throughout — the
  experiment detail page header says "Publish lightgbm" and "Publishing lightgbm".
- **Mode toggle persists**: toggling publish on/off from the dashboard now saves
  to mlfl.yaml (previously only in-memory).
- **Publish guard**: the Publish button is disabled until a benchmark has been run,
  preventing publishing without a trained model.

## 1.24.1

### Improvements
- **Remove harmful covariates**: Covariate Analysis rows where dropping a covariate
  improves average MAE now show a "Remove" button. One click removes it from the
  experiment's YAML config.
- **Apply & Promote**: Tuning "Apply Best" replaced with "Apply & Promote" which
  saves tuned params AND promotes the model to production in one action.
- **Holdout comparison chart**: after tuning completes, a Plotly chart shows the
  model's default-params vs tuned-params predictions on holdout data with MAE
  improvement percentage.
- **Promotion persists**: promoting a model now saves `production_model` and `mode`
  to mlfl.yaml so the choice survives add-on restarts.

### Bug Fix
- **Pydantic validation error**: `ModelPrediction.predictions` field now accepts
  `Optional[float]` values, fixing ensemble padding errors in logs.

## 1.24.0

### New Feature: Selected Model
- **Model selection in Results**: radio buttons in the model comparison table let
  the user select any model, not just the top-ranked one. Selection persists across
  page reloads and feeds into Promote, Covariate Analysis, and Tuning.
- **Promote uses selected model**: the Promote button now promotes whichever model
  the user has selected, not just the auto-ranked best.
- **Selector defaults**: Covariate Analysis and Tuning model dropdowns default to
  the selected model instead of an arbitrary first option.
- **Per-experiment model_params**: tuning results saved per-experiment take
  precedence over global model_overrides during training.

## 1.23.1

### Improvements
- **Rename**: "Deep Analysis" renamed to "Covariate Analysis" throughout the UI.
- **Tab reorder**: Covariate Analysis now appears before Tuning in the tab bar,
  reflecting the natural workflow (analyse covariates → tune hyperparameters).
- **Per-experiment model params**: "Apply Best" in Tuning now saves hyperparameters
  per-experiment (not globally). Different experiments can have different tuned
  params for the same model. Per-experiment params take precedence over global
  model overrides.
- **Default vs tuned comparison**: the Tuning best-params summary now shows a
  side-by-side table of default → tuned values for each parameter.

## 1.23.0

### New Feature: Hyperparameter Tuning
- **Optuna-based Bayesian optimisation**: new Tuning tab in the experiment detail
  page with TPE (Tree-structured Parzen Estimator) and random search strategies.
- **Automatic search space**: parameter ranges derived from existing model schema
  with log-scale for learning rates and regularisation terms.
- **Fast 2-fold CV**: each trial uses 2-fold cross-validation for speed on
  constrained hardware (RPi5). Full benchmark validates the winner afterwards.
- **Live progress**: poll-based progress showing completed trials, best MAE so far,
  and a progress bar.
- **Trials table**: all trials sorted by MAE with params displayed, best highlighted.
- **Apply Best button**: one click saves winning hyperparameters as model overrides
  in mlfl.yaml, ready for the next pipeline run.
- **Per-model tuning**: select which model to tune from the experiment's enabled models.

## 1.22.9

### Fixes
- **Tab preserved on reload**: running ensemble or deep analysis no longer
  jumps to the Training tab on completion — the page reloads back to the
  correct tab via URL hash.
- **Deep analysis metric layout**: replaced cramped single-line layout with
  a clean three-row grid (MAE, RMSE, MASE) each with its own change %.

## 1.22.8

### Bug Fix
- Fix `NameError: name 'best_ind_metrics' is not defined` crash in ensemble
  pipeline introduced in v1.22.6.

## 1.22.7

### Deep Analysis Improvements
- **MASE metric added**: deep analysis now computes MASE alongside MAE and RMSE,
  with percentage change vs baseline for all three metrics.
- **All metrics displayed**: each cell shows MAE (primary) with RMSE and MASE on
  a second line, all with colour-coded change percentages.
- **Cross-model consensus**: per-covariate recommendations now report when all
  models (or a majority) agree a covariate is important or harmful, instead of
  using a single reference model.
- **Best-model reference**: per-covariate detail uses the best-performing model
  by baseline MAE, not a hardcoded tree model preference.

## 1.22.4

### Bug Fixes
- **Ensemble "Best" badge mismatch**: the improvement percentage text now
  consistently references the same strategy that receives the "Best" badge.
  Previously composite ranking and production metric comparison could disagree.
- **Ensemble prediction length**: ensemble traces now span the full holdout
  period by right-aligning to holdout timestamps and padding the start with
  gaps, instead of being silently truncated.

### Improvements
- **Fold stability**: replaced single-metric bar chart with three line charts
  (MAE, RMSE, MASE) showing per-fold variation for each model. Easier to spot
  instability and compare across metrics.
- **Loss curves moved**: neural model training loss curves relocated from the
  Generalisation tab into a collapsible section in the Results tab where they
  sit alongside model comparison data.
- **Feature importance note**: added guidance directing users to Deep Analysis
  for neural model feature insights.

## 1.22.3

### Improvement
- **High-contrast chart colours**: replaced the old green/blue palette with
  a high-contrast scheme (coral, amber, green, blue, purple...) so that 2-3
  model experiments are easy to read at a glance. Applied consistently to
  prediction charts, residuals, fold stability, loss curves, ensemble
  predictions, and live training fold traces.

## 1.22.2

### Fix
- **Remove early stop markers from live training charts**: the early stop
  detection and marker rendering on loss curves didn't work reliably. Early
  stop events are still logged to the training event log.

## 1.22.1

### UX Refinements
- **Per-experiment model selection**: model enable/disable toggles moved from the
  global Models page into the experiment detail page ("Models" tab). Each experiment
  now independently selects which models to train.
- **Models page simplified**: now a pure catalog with hyperparameter configuration.
  Toggle switches removed; subtitle directs users to experiment detail for selection.
- **Header decluttered**: ensemble/deep-analysis checkboxes removed from the header.
  Ensemble always included by default; deep analysis controls moved into the Deep
  Analysis tab where they have full context.

## 1.22.0

### Major UI Overhaul
- **Consolidated experiment workflow**: the dedicated Training tab has been removed.
  Pipeline controls, live training progress (loss curves, stats, event log), and
  all result sections now live on the experiment detail page ("View Details").
  Train, monitor, and evaluate an experiment without leaving the page.
- **Tab-based detail view**: the experiment detail page uses show/hide tabs instead
  of a long scrollable layout. Only the selected section is visible, reducing
  clutter. Charts render on first tab visit with automatic Plotly resize.
- **Section consolidation**: reduced from 8 sections to 6. Residuals merged into
  Predictions; Run Info moved to a compact header bar. Features and Deep Analysis
  tabs appear only when data exists.
- **Dashboard training status**: experiment cards now show the current model name
  and a progress bar when training is active. Dashboard refreshes every 10s during
  training instead of 60s.
- **Navigation simplified**: 4 tabs (Dashboard, Models, Logs, System) instead of 5.
  The old `/training` URL redirects to the Dashboard.

## 1.21.5

### Bug Fix
- **Training tab progress lost on tab switch**: the live training progress
  section now reliably restores when navigating away and back. The server
  embeds event history directly in the rendered HTML, eliminating a separate
  fetch that could fail silently through the HA ingress proxy. Also adds
  `pageshow` listener for bfcache resilience and `encodeURIComponent` for
  experiment names in fallback API calls.

## 1.21.4

### Bug Fix
- **Training tab loses live view on tab switch**: navigating away from the
  Training tab and returning now correctly restores the loss plot and live
  SSE stream. Root causes: `run-pipeline` was not calling `start_benchmark()`
  or `clear_history()`, so the UI could never detect an active run; the
  `hasEnd` check treated any historical `pipeline_end` as proof the current
  run had finished; and the Plotly chart rendered into a zero-dimension div
  before the browser had laid it out.

## 1.21.3

### Bug Fix
- **XGBoost 2.1+ compatibility**: moved `callbacks` parameter from
  `XGBRegressor.fit()` to the constructor, fixing "unexpected keyword
  argument 'callbacks'" error on XGBoost >= 2.1.

## 1.21.2

### Bug Fixes
- **Model hyperparameter auto-save**: hyperparameters in the Models tab now
  auto-save on change with a 600ms debounce, so edits persist without needing
  to click the Save button. The manual Save button remains as a fallback.
- **Training tab reconnection**: navigating away from the Training tab and
  returning now correctly restores the live training view. The backend
  pre-selects the currently-training experiment in the dropdown, and
  sessionStorage remembers the last selection for completed runs.

## 1.21.1

### Improvement
- **Neural model window size**: sliding window size is now derived from
  `max(48, 2 × max_horizon_steps)` instead of a fixed cap of 48. For
  horizons [2h, 8h, 12h, 24h] at 30-min intervals this gives 96 steps
  (48 hours of context), giving neural models twice as much lookback as
  their longest prediction horizon.

## 1.21.0

### Feature Engineering
- **Periodic lags**: `y_lag_48` and `y_lag_96` (same time yesterday and 2 days
  ago at 30-min intervals) give models direct access to daily periodicity.
- **Rate of change**: `y_diff_1` captures whether demand is accelerating or
  decelerating.
- **Interaction features**: `{covariate}_x_hour_sin` and `{covariate}_x_hour_cos`
  encode how covariates (e.g. charge level, temperature) interact with time of day.
- Production forecast features updated to compute interactions and periodic lags
  at inference time.

## 1.20.1

### Bug Fix
- **MASE in train metrics**: overfitting diagnostics were missing `y_train`
  when computing MASE, causing repeated "missing 1 required positional
  argument: 'y_train'" warnings during benchmarking.

## 1.20.0

### Hyperparameter Tuning
- **LightGBM**: stronger regularisation to reduce overfitting — num_leaves 31→20,
  min_child_samples 10→25, max_depth 6→5, reg_lambda 0.1→1.0, reg_alpha 0.1→0.5,
  learning_rate 0.05→0.03, subsample 0.8→0.7
- **XGBoost**: aligned with LightGBM changes — max_depth 6→5, learning_rate
  0.05→0.03, reg_alpha 0.1→0.5, subsample 0.8→0.7
- **DLinear**: kernel_size 25→13 (sharper daily patterns), learning_rate 2e-4→5e-4
- **SparseTSF**: dropout 0.1→0.05, learning_rate 2e-4→5e-4
- **LSTM**: hidden_size 64→32, num_layers 2→1, dropout 0.2→0.1
- **CNN**: n_filters 32→16, dropout 0.2→0.15
- **PatchTST**: d_model 32→16, n_heads 4→2, n_encoder_layers 2→1
- **N-BEATS/N-HiTS**: hidden_size 64→32

## 1.19.1

### Fix
- **Gap colouring accounts for underfitting**: a small train/test gap now shows
  orange (not green) when the model's test error is >1.5x the best model's,
  preventing false confidence from models that underfit both train and test.

## 1.19.0

### Generalisation Diagnostics
- **Train vs Test error table**: shows train MAE/RMSE alongside test metrics
  with a colour-coded gap column (green/orange/red) to highlight overfitting.
- **Loss curves**: per-epoch train and validation loss for neural models
  (DLinear, SparseTSF, etc.), surfaced from the existing `_training_history`.
- **Fold stability chart**: grouped bar chart of per-fold MAE across all models
  to visualise cross-validation consistency.

## 1.18.3

### Improvement
- **Ensemble "Best" badge**: the "Best" badge now considers the best individual
  model alongside ensemble strategies in the composite ranking. If the individual
  model beats all ensembles, it gets the badge instead.

## 1.18.2

### Bug Fix
- **Ensemble best-individual row**: the "Best Individual" row in the ensemble
  table was showing the production metric (e.g. RMSE) in the MAE column.
  Now displays correct MAE, RMSE, and MASE values in their respective columns.

## 1.18.1

### UI Fixes
- **Vibrant chart colours**: model and ensemble prediction line colours now
  use the same high-contrast palette as the loss plots (`#00d4ff`,
  `#e94560`, `#2ecc71`, `#f39c12`, `#9b59b6`, etc.)
- **Residual chart cleanup**: ensemble traces filtered out of the residual
  plot (they were only intended for the dedicated ensemble chart)

## 1.18.0

### Metrics & Ranking
- **MASE replaces MAPE**: default percentage metric is now Mean Absolute
  Scaled Error (Hyndman & Koehler 2006) — bounded, handles near-zero
  values, and answers "does this model beat naive?". MAPE remains
  available as a registered metric for custom configs.
- **Composite ranking**: model rankings and ensemble "Best" badge are now
  determined by average rank across MAE, RMSE, and MASE rather than a
  single production metric, rewarding consistent performance.

### Ensemble
- **Separate ensemble chart**: ensemble predictions are no longer mixed
  into the holdout chart. A dedicated "Ensemble Predictions on Holdout
  Data" chart with its own interval/cumulative toggle appears below the
  ensemble results table.

## 1.17.0

### UX Polish
- **Humanised experiment names**: experiment names display as title-case
  (e.g. "Hot Water Demand") throughout the UI while preserving snake_case
  in URLs and APIs. Added `humanise` Jinja filter.
- **Dedicated run-all endpoint**: `POST /api/benchmarks/run-all` replaces
  the brittle wildcard `POST /experiment/*/run-benchmark` pattern.
- **Models page hint banner**: shows guidance when no models are enabled.
- **System info N/A fallback**: memory and disk stats show "N/A" instead
  of "0GB / 0GB" when values are unavailable (e.g. running outside container).
- **Logs empty state**: helpful message shown when no log output exists.
- **Promote button state**: disabled with "In Production" label when the
  experiment is already in production mode, preventing confusing no-op clicks.
- **Toggle keyboard accessibility**: custom toggle switches now show a
  focus ring when navigated via keyboard (`focus-visible`).
- **Deep Analysis select width**: constrained to `max-width: 200px` to
  prevent stretching on wide screens.
- **Section nav offset**: sticky top increased to 70px for reliable
  clearance below the navbar.
- **CSS utility classes**: extracted 50+ common inline style patterns into
  reusable classes (`.color-success`, `.text-stat-lg`, `.flex-row`,
  `.grid-paths`, `.hint-banner`, `.btn-purple`, etc.), reducing inline
  `style=` attributes across dashboard, system, training, experiment, and
  logs templates.

## 1.16.2

### UI Fixes
- **Consistent units on metric headers**: ensemble and per-fold tables now show
  configured units (e.g., "MAE (%)") matching the model comparison table

## 1.16.1

### Bug Fixes
- **SparseTSF deep analysis crash**: fixed `RuntimeError: shape '[64, 1, 0, 48]'
  is invalid` when a "Without covariate" configuration reduced feature count below
  `period_len` (48). The model now clamps `period_len` down to `seq_len` so at
  least one complete period always fits.
- **SparseTSF negative-zero slicing**: fixed `x[:, -0:, :]` silently returning
  the full tensor instead of an empty slice, which caused the shape mismatch
  error above.

## 1.16.0

### UI Overhaul
- **Offline-ready**: HTMX and Plotly bundled locally — no more CDN dependency
- **Plotly conditional loading**: Plotly Basic (1MB) only loads on pages with charts
  (experiment detail, training), not on Dashboard, Models, Logs, or System
- **Consolidated CSS**: all inline `<style>` blocks consolidated into a single
  `style.css`; activated the previously dead 708-line stylesheet
- **Toast notifications**: all `alert()` calls replaced with themed slide-in toasts
  (`mlfl.toast()`) for success, error, and warning messages
- **Styled confirm modal**: browser `confirm()` replaced with dark-themed modal
  dialog (`mlfl.confirm()`) for destructive actions
- **Button loading spinners**: async operation buttons (Run Benchmark, Deep Analysis,
  Run Ensemble, Run Pipeline) show CSS spinner during execution
- **Experiment page navigation**: breadcrumb (Dashboard > experiment_name) and sticky
  section nav with scroll-spy (Results, Predictions, Residuals, Features, Ensemble,
  Deep Analysis, Run Info)
- **Feature importance visibility**: bar colour changed from near-invisible `#0f3460`
  to high-contrast `#00d4ff`
- **Improved empty states**: icons and helpful hint text for missing benchmark
  results, holdout predictions, and feature importance data
- **Mobile hamburger menu**: responsive navigation collapses into animated
  hamburger toggle on screens below 768px
- **Reduced motion support**: respects `prefers-reduced-motion` media query

### Cleanup
- Deleted orphaned `settings.html` and `status.html` templates (routes already
  redirect to `/system`)
- Removed dead `mlfl.drawForecastChart()` and `mlfl.drawFeatureImportance()` JS
  functions from `base.html` (never called by any template)

## 1.5.0

### Correctness Fixes
- **LSTM/CNN production inference**: autoregressive sliding window prediction
  instead of flat features. Neural models now produce proper demand curves in
  production mode.
- **Rolling stats at inference**: computed from available lag values instead
  of being set to NaN (which became 0 after nan_to_num).
- **Feature leakage**: rolling statistics now recomputed per CV fold instead
  of once on the full dataset before splitting.

### Security & Robustness
- **SQL injection**: added regex assertion after safe_table_name() sanitisation
- **eval() replaced with asteval**: safer expression evaluation for custom metrics
- **datetime.utcnow()**: replaced with datetime.now(timezone.utc) (5 instances)
- **asyncio.get_event_loop()**: replaced with get_running_loop() (4 instances)

### ML Methodology
- **Per-channel z-score standardisation**: LSTM/CNN inputs now standardised
  per channel (fitted on training data, applied to test). Persisted in save/load.
- **Outlier clipping**: default quantile raised from 0.95 to 0.995
- **Sample weight half-life**: fixed at 7 days instead of 30% of fold size
- **Future covariates**: production inference uses fetch_future() for role='future'
  covariates instead of always using last-known-value
- **holidays library**: replaces hardcoded GB/US/DE holiday dates
- **Box-Cox renamed to shifted_log**: honest naming (kept box_cox as alias)
- **Deep Analysis model selection**: dropdown to select which model to analyse

### Code Quality
- **Test suite**: 7 test modules (preprocessing, features, db, config, models,
  benchmark, metrics) with pytest fixtures
- **Multi-stage Docker build**: separates build deps from runtime, smaller image
- **.gitignore**: added with standard Python ignores
- **__pycache__ cleanup**: removed from git tracking
- **Dead code removed**: core.py, server.py, _numpy_optim.py (v1.4.1)
- **NeuralProphet made optional**: removed from requirements.txt

## 1.3.0

### New Features
- **LSTM architecture upgrade**: 2-layer LSTM with temporal attention
  (learnable weights across all timesteps), LayerNorm input normalisation,
  and MLP output head (64→32→1). Replaces naive "take last hidden state"
  approach — model now learns which timesteps matter most.
- **CNN architecture upgrade**: 4-layer WaveNet with 32 filters,
  dilations 1/2/4/8 (receptive field = 31 steps), learnable positional
  pooling (replaces global average pool), LayerNorm, and MLP head.
- **Best-model checkpointing**: both LSTM and CNN now save the best
  model state during training and restore it after early stopping.
  Previously used whatever state the model was in when patience ran out.
- **ReduceLROnPlateau**: learning rate halves when validation loss
  plateaus (patience=7), separate from early stopping (patience=15).
- **Middle-out validation split**: validation data taken from the centre
  of the training window instead of the tail, so the model trains on
  both early and recent (most valuable) data.
- **Window size 48**: restored full 24-hour daily cycle for LSTM/CNN
  sliding windows (was reduced to 12h in v1.2.3).
- **Model toggle UI**: toggle models on/off from the System Status page.
  Changes save to mlfl.yaml and take effect on the next benchmark run.

## 1.2.5

### Bug Fixes
- **Live log feed**: fixed polling logic that prevented log updates
  (compared line count instead of content, so updates were missed when
  the log window stayed the same size). Added error handling and live
  indicator now turns red on connection failure.
- **Version strings**: fixed hardcoded "0.2.0" in stub server and
  legacy server.py — now use centralised APP_VERSION from __init__.py.
- **Feature importance chart**: bar colour changed from #0f3460 (same
  as chart background, invisible) to #00d4ff (accent colour).
- **LSTM/CNN prediction**: removed redundant dummy predict call before
  the actual torch inference in benchmark runner.

## 1.2.0

### New Features
- **Time-weighted sampling**: recent data weighted higher than old data
  using exponential decay (half-life = 30% of training window). Applied
  to all models: LightGBM/XGBoost via sample_weight, PyTorch via
  weighted Huber loss.
- **SQLite cache as primary source**: checks local cache first, only
  fetches delta from HA API for new records since last cache. Reduces
  HA API load from ~11K records to just new ones. Auto-cleans records
  older than max_age.
- **Raw sequence input for LSTM/CNN**: creates sliding window sequences
  (48 steps × n_channels) from raw target + covariate time series
  instead of reshaping pre-computed features. Gives neural models
  proper temporal structure to learn from.

### Technical Details
- BenchmarkRunner generates time-decay weights and passes to all models
- LightGBM uses lgb.Dataset(weight=), XGBoost uses fit(sample_weight=)
- PyTorch models use per-sample weighted HuberLoss (reduction='none')
- create_sliding_windows() utility in features.py
- Delta fetch: loads SQLite cache → fetches only records after last
  cached timestamp → merges and deduplicates

## 1.1.0

### Major Features
- Deep Covariate Analysis: tests all models × all covariate combinations
  to find which external features help which models
- "🔬 Deep Analysis" button on experiment page triggers the analysis
- Results table shows MAE for each model under each configuration with
  colour-coded % change (green = improved, red = worse)
- Automated recommendations: "✓ current_charge is important",
  "⚠ external_temperature adds minimal value", etc.
- Progress bar shown while analysis runs in background
- Runs in thread pool — web UI stays responsive during analysis

### Tested Configurations
- All covariates (baseline)
- No covariates (control)
- Each covariate dropped one at a time

## 1.0.1

### Bug Fixes
- Version banner now reads from __init__.__version__ dynamically
  (was hardcoded as v0.3.6)
- Covariate alignment uses ffill+bfill to prevent data loss — was
  losing ~660 samples from dropna after merging covariates
- NeuralProphet/PyTorch Lightning FutureWarning spam suppressed
- All version strings in web app use centralised APP_VERSION
- Huber loss for LSTM and CNN (robust to demand spikes)

## 1.0.0

### Major Features — PyTorch Migration
- **LSTM rewritten with PyTorch** — torch.nn.LSTM replaces 500 lines of
  pure-NumPy implementation. Proper autograd, 10-50x faster matrix ops,
  correct gradient flow through all gate weights.
- **CNN rewritten with PyTorch** — WaveNet-style dilated causal convolutions
  using torch.nn.Conv1d with residual connections. Replaces manual conv
  backward pass.
- **NeuralProphet added** — Facebook's neural forecasting library combining
  trend decomposition, automatic seasonality detection, and neural AR
  components. Purpose-built for time series with covariates.
- **5 model backends** now available: LightGBM, XGBoost, LSTM, CNN, NeuralProphet

### Dependencies
- Added PyTorch (torch>=2.0.0) — first install will take ~15-20 min on Pi,
  subsequent updates use Docker cache (~30s)
- Added neuralprophet>=0.9.0

### Other
- All version strings now imported from __init__.py (no more hardcoded)
- NeuralProphet shown on System Status page with purple colour in charts
- _numpy_optim.py kept for reference but no longer used by neural backends

## 0.6.1

### Improvements
- Units displayed on all charts — Y-axis shows "Value (%)" or
  "Daily Cumulative (%)" based on experiment config
- Units shown on MAE and RMSE column headers in model comparison table
- Residual chart Y-axis shows units
- Daily cumulative chart resets at midnight (not running total)

## 0.6.0

### Major Features
- Covariate integration — external HA sensor data now included as
  features alongside temporal/lag features in model training
- Covariates configured in mlfl.yaml are fetched from HA history API,
  resampled to the target's interval grid, and merged into the feature
  matrix for all models
- Supports scaling (e.g. 0-100% → 0-1), transforms (log, sqrt),
  binary detection, and forward-fill alignment
- Production mode uses last known covariate value for future forecasts
- Benchmark header shows covariate count and names
- Feature matrix log shows breakdown: "33 features (31 temporal + 2 covariates)"

### Configured Covariates for Mixergy
- sensor.current_charge (scaled ×0.01): tank charge level as leading
  indicator of demand — should help predict usage spikes
- sensor.external_temperature: outdoor temperature for seasonal patterns

## 0.5.3

### New Features
- Interval / Cumulative toggle switch on holdout prediction chart
- Cumulative view shows total demand over time for comparing which
  model tracks overall accuracy best
- Smooth toggle animation with accent cyan knob

## 0.5.2

### Improvements
- LSTM and CNN epochs increased from 50 to 100 — gives models more
  time to converge through noisy loss landscapes
- Early stopping patience increased from 8 to 15 — prevents premature
  stopping when val_loss temporarily bounces up before improving
- Training still fast thanks to BPTT fix (~2s/epoch for LSTM on Pi)

## 0.5.0

### Major Features
- Full backpropagation through time (BPTT) for LSTM — all gate weights
  (input, forget, cell, output) now train properly instead of just the
  dense output layer. LSTM should now learn temporal patterns.
- Full backpropagation for CNN — conv kernel weights, biases, and residual
  connections all update during training. Includes backward pass through
  dilated causal convolutions.
- Gradient clipping (max_norm=5.0) prevents exploding gradients in both
  neural backends
- Feature names passed to all models for readable importance charts

### Bug Fixes
- LSTM and CNN predictions clipped to non-negative (prevents misleading
  chart visualisations from untrained weights)
- Reduced LSTM/CNN defaults for faster Pi training: LSTM ~3-4x faster,
  CNN ~2x faster

## 0.4.0

### New Features
- Settings page accessible from the nav bar:
  - System information: CPU cores, processor, memory usage, disk usage
  - Resource limits: configurable training CPU cores and process priority
  - Update interval, timezone, and Hailo toggle — editable from the UI
  - Per-experiment configuration overview with covariate details
  - Config file path and version info
  - Save button writes settings back to mlfl.yaml
- cpu_cores and nice_priority config options in mlfl.yaml

### Improvements
- Cleaner log format: short timestamps, no module name in console output
- Richer log viewer colours: cyan headers, green results, purple model progress
- Removed version number from footer (shown in Settings instead)

## 0.3.7

### Improvements
- Rich, informative log output inspired by Predbat:
  - Boxed startup banner with version
  - Section headers with ═══ separators for benchmark/production cycles
  - Data summary with mean, std, min, max, zero count after preprocessing
  - Per-model progress counter [1/4], [2/4] etc with ✓ on completion
  - Aligned results table with MAE, RMSE, MAPE, Time, Rank columns
  - ★ marker for best model in results table
  - Production forecast summary with per-horizon values
- Log viewer page colour-codes separators (cyan), results (green),
  best model (orange) in addition to errors/warnings

## 0.3.6

### Bug Fixes
- Tooltips now use fixed positioning with JavaScript so they don't get
  clipped by table headers or overflow containers

### Improvements
- Model Comparison tooltip includes guidance on Per-Fold Metrics
- System Status model cards now show type (Tree/Neural), speed,
  hardware acceleration support, and best-use-case for each backend
- Footer: "Created by Dr Paul W. Sweeney"

## 0.3.5

### New Features
- Info tooltips (?) on all section headers and metric columns explaining
  what each result means and how to interpret it
- Readable benchmark timestamp with relative time (e.g. "2h 15m ago")

### Improvements
- Footer updated to "Created by Dr Paul W. Sweeney"
- Mode toggle button text shortened to prevent wrapping

## 0.3.4

### New Features
- Progressive benchmark results — web UI updates after each model finishes
  its CV folds, so you see LightGBM results while LSTM is still training
- Model comparison table builds up in real-time during benchmark

## 0.3.3

### New Features
- Styled Logs page with colour-coded log levels and filter buttons (All/Info/Warning/Error)
- System Status page replacing raw JSON — shows health, experiments table, model backends, API docs
- Active nav indicator — current page highlighted with badge-style border
- Glow-pulse animation on mode badge when training is in progress
- Footer now shows version number and author credit

### Improvements
- Lab Mode stat number uses orange, Production Mode uses green to match badges
- Card footer buttons equal height and centred
- Pending status badge uses accent cyan instead of dull grey
- Run All Benchmarks button works through ingress
- Suppressed Uvicorn access logs

## 0.3.2

### New
- CHANGELOG.md for HA update screen

## 0.3.1

### New Features
- Production mode toggle button on each experiment card
- Live countdown timer for next update (e.g. "5h 58m" instead of "21510s")
- Best model / production model label on dashboard cards
- Toggle mode API endpoint

### Improvements
- Dashboard auto-refresh reduced to 60s
- Readable timestamp format

## 0.3.0

### New Features
- Full forecast curve generation in production mode (e.g. 96 points for 48h)
- Main forecast sensor with full curve in attributes for ApexCharts
- Per-horizon scalar sensors (e.g. sensor.mlfl_mixergy_demand_2h, _8h, _12h, _24h)
- Proper unit_of_measurement, icon, and state_class on published entities

## 0.2.4

### Bug Fixes
- Fix metric kwargs forwarding — MAE, RMSE, MAPE were failing due to unexpected y_train argument

## 0.2.3

### Bug Fixes
- Web UI no longer blocks during model training — benchmark runs in background thread

## 0.2.2

### Bug Fixes
- MASE metric now receives y_train for naive forecast baseline

## 0.2.1

### New Features
- Multi-model prediction overlay chart on holdout data
- Residual plot showing prediction errors per model
- Real feature importances from LightGBM and XGBoost
- All configured metrics computed per fold (MAE, RMSE, MAPE, SMAPE, MASE)

## 0.2.0

### New Features
- Real forecasting pipeline replacing stub methods
- Walk-forward cross-validation across LightGBM, XGBoost, LSTM, CNN
- Web UI dashboard accessible via HA sidebar (ingress support)
- Rotating log file with web viewer
- Heartbeat sensor published to HA
- Automated GitHub releases

### Bug Fixes
- HA history API parameter name fix
- Accept HTTP 201 for new entities
- Midnight cross detection for irregular timestamps
- LightGBM and XGBoost API compatibility
- Jinja2 TemplateResponse fix for newer Starlette
- CSS colour spelling fix

## 0.1.0

- Initial scaffold with stub pipeline
