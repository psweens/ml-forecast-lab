# Changelog

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
