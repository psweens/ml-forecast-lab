# IMPROVEMENTS.md — ml-forecast-lab

Proposal document. Tailored to HA power users running this add-on on a Raspberry Pi 5 (8 GB ARM64, no GPU, optional Hailo NPU, often headless, SD card or NVMe). Grounded in `SURVEY.md`. British English. No code edits in this PR.

Each proposal passes the brief's hard constraints — anything that would have failed them was dropped rather than caveated.

---

## A. Functionality gaps

Six proposals.

### A1. Stop the add-on pegging the Pi during training

- **User problem.** Turning down "Training CPU cores" or raising "Process priority" on the System page changes the YAML but not the runtime, so training saturates all four Pi 5 cores anyway and HA gets choppy mid-benchmark.
- **Proposed change.** When `AppConfig` loads (`main.py:198` `load_config`), apply `os.nice(cfg.nice_priority)` once at startup and call `torch.set_num_threads(cfg.cpu_cores or os.cpu_count())` plus set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` in the env before `torch`/`lightgbm`/`xgboost` initialise their pools. Add a single "Applied: 2 cores, nice 10" line under the form so users see it stuck.
- **Why now / why this addon.** This is a real bug masquerading as a feature — the settings are persisted to `mlfl.yaml` and displayed on `/system` (`system.html:62-100`, `config.py:571,574`), but nothing applies them (`grep -nE "set_num_threads|os\.nice|sched_setaffinity" ml_forecast_lab/` returns nothing). HA core can't fix this; it has to be the add-on.
- **Pi 5 cost.** RAM: unchanged. CPU: lower — the whole point. Disk/network/UI: nil.
- **Implementation sketch.** Touch `main.py` (apply at startup, before model imports) and the env handoff. ARM64-safe; both `os.nice` and `torch.set_num_threads` are stdlib/torch. No new dependency.
- **Effort.** S.
- **Risk.** A user who sets cores=1 will see slow training and not connect the dots. Mitigation: warn in the System form when `cores < ceil(system_count / 2)`.
- **Confidence.** High. Discoverability category: **exists in UI, broken implementation**.

### A2. Delete an experiment without using curl

- **User problem.** Experiments accumulate (test runs, sensors that turned out wrong); the only way to remove one is `POST /api/experiments/{name}/delete`. Users edit YAML by hand or live with cruft.
- **Proposed change.** Trash-can icon on each dashboard card (`dashboard.html:39-45`) and in the experiment header (`experiment.html:17-22`). Reuses the existing global confirm modal (`base.html:43-52`) and the existing delete endpoint (`web/app.py:1726`).
- **Why now / why this addon.** Endpoint exists; UI button is ~20 lines of Jinja + a fetch call. No HA-core equivalent.
- **Pi 5 cost.** Negligible.
- **Implementation sketch.** Edit `dashboard.html` and `experiment.html`; add small JS shim alongside `stopTraining` / `retrainExperiment`.
- **Effort.** S.
- **Risk.** Accidental deletion. Mitigation: confirm-modal already pattern.
- **Confidence.** High. Discoverability category: **exists in code, no UI surface**.

### A3. Compare today's benchmark against the previous run

- **User problem.** Each new benchmark replaces the previous result; there's no way to tell whether ranks have shifted, whether MAE has crept up after a covariate change, or whether the model is being chosen against a noisier window of data.
- **Proposed change.** Store the last *N* benchmark runs (default 5) per experiment instead of overwriting. Surface a "Previous runs" dropdown above the Results table (`experiment.html:518`) that swaps the metrics table contents.
- **Why now / why this addon.** `benchmark_results` is already a real SQLite table (`db.py:1856`); the `save_benchmark_result` writer uses experiment as the unique key (`db.py:1870`). Adding a timestamp PK and a `LIMIT N` retention cap is small.
- **Pi 5 cost.** RAM nil. Disk: ~50–150 KB per run (a JSON blob), capped to N runs/experiment — e.g. 5 experiments × 5 runs × 100 KB = 2.5 MB.
- **Implementation sketch.** `db.py` schema migration via `_record_version` (already in use; `db.py:67-99`). Read path in `web/app.py:987`. New `<select>` plus an htmx swap targeted at the results-table container.
- **Effort.** M.
- **Risk.** Disk drift if a user runs many benchmarks; bounded by retention cap. SQLite VACUUM is still not scheduled — that's a separate item to consider later but doesn't block this.
- **Confidence.** High. Discoverability category: **missing in the addon**.

### A4. Roll back a regressive retrain

- **User problem.** A scheduled retrain (24 h default) can produce a worse champion than the previous one — drift in the last day's data, recorder oddity, an aggressive tuning apply. There's currently no undo; the previous weights are overwritten in place.
- **Proposed change.** On each `_persist_cached_model` (`main.py:2873`), rename the existing `model.bin` to `previous.bin` before writing the new one. Add a "Roll back to previous (trained YYYY-MM-DD HH:MM)" button on the experiment header that swaps them and re-loads the cache.
- **Why now / why this addon.** The cache directory (`/data/ml_forecast_lab/models/<exp>/`) is already per-experiment; keeping one extra checkpoint is local.
- **Pi 5 cost.** Disk: doubles per-experiment cache. Tree models are ~1–10 MB each; neural backends 5–50 MB. Cap at one previous version — bounded growth.
- **Implementation sketch.** `main.py` `_persist_cached_model` and `_restore_cached_models` (`main.py:2933`); new endpoint `POST /experiment/{name}/rollback`. Touches `cache_meta.json` versioning.
- **Effort.** M.
- **Risk.** Doubles disk for a feature most users won't use; gate behind a settings toggle defaulting **on** for production-mode experiments only.
- **Confidence.** Medium. Discoverability category: **missing in the addon**.

### A5. Let HA automations react when a benchmark or retrain finishes

- **User problem.** Right now the only way to know a benchmark or retrain completed is to refresh the web UI or watch the log. Users want to fire a notification, kick a Predbat re-plan, or just know.
- **Proposed change.** Publish a small set of event-shaped sensors: `sensor.mlfl_last_benchmark_<exp>` (state = ISO timestamp, attrs = winner + composite score + duration) and `sensor.mlfl_last_retrain_<exp>`. Updated once at the end of each cycle via the existing `ha_interface.set_state` plumbing.
- **Why now / why this addon.** The HA REST publish path is already there for forecast sensors (`main.py:3608`, `ha_interface.py:373`). Two extra writes per experiment per cycle.
- **Pi 5 cost.** Network: 2 extra HA REST calls per benchmark and per retrain. Disk nil. RAM nil.
- **Implementation sketch.** A `_publish_lifecycle_sensors` helper near `_publish_forecast_sensors` (`main.py:3049`); call sites at the ends of `_run_benchmark` and `_retrain_and_cache`.
- **Effort.** S.
- **Risk.** Sensor explosion if users have many experiments; document and cap names.
- **Confidence.** High. Discoverability category: **missing in the addon**.

### A6. Stop advertising future-role covariates that don't work

- **User problem.** `mlfl.yaml` and the Settings UI both advertise `role: future` for covariates (the commented Predbat-rates example at `mlfl.yaml:148-151`; the Role dropdown at `experiment.html:294-300`). But `CovariateResolver.fetch_future` returns NaN as a placeholder (`covariates.py:159-200`), so a user who selects this gets a quietly broken pipeline.
- **Proposed change.** Either (a) hide the "Future" / "Both" options in the Settings UI dropdown until the resolver actually reads forecast attributes from the source sensor, or (b) implement it — most HA forecast sensors publish a horizon array under a known attribute key (Met.no, Solcast, Predbat). Recommend (a) first — it's a one-line fix and removes the broken promise. Track (b) as a follow-up.
- **Why now / why this addon.** Documented behaviour silently disagrees with implementation; a user setting it up the documented Predbat way gets a worse model and won't know why.
- **Pi 5 cost.** (a) Nil. (b) One extra HA REST call per covariate per cycle.
- **Implementation sketch.** (a) Edit `experiment.html:294-300`; add a docstring note in `covariates.py:159`. (b) Refactor `fetch_future` to read `state.attributes[<key>]`, schema-validate, and pad to horizon.
- **Effort.** S for (a); L for (b).
- **Risk.** (a) might surprise users who had hopes; mitigation: note in the CHANGELOG. (b) is genuinely useful work but real-money effort.
- **Confidence.** High for (a); medium for (b). Discoverability category: **exists in UI, broken implementation**.

---

## B. Frontend usability

Five proposals.

### B1. Make the experiment page load fast over HA ingress

- **User problem.** `experiment.html` is 3 736 lines (single template, all tabs rendered server-side, every Plotly chart's JSON inline). The page pulls `plotly-basic.min.js` (~700 KB minified) in `head_scripts` even on tabs that don't need it. On a 2019 laptop over HA ingress this is several seconds of jank.
- **Proposed change.** Drop the Plotly `<script>` from `head_scripts` (`experiment.html:5-7`). Inject it on first navigation to a chart-bearing tab via a `loadPlotly()` helper (idempotent, returns a Promise). Tabs hit: Predictions, Generalisation, Features, Tuning, Forecast Accuracy.
- **Why now / why this addon.** Single biggest UI-weight reduction available. Touches one template plus the tab-switcher JS.
- **Pi 5 cost.** −700 KB on the initial document load; one-off ~150–300 ms when the first chart tab is opened.
- **Implementation sketch.** Promise-cached script loader, called inside `showTab` when the tab needs Plotly. The existing `showTab` already runs per-tab init (`experiment.html:1539+`).
- **Effort.** S.
- **Risk.** Direct linking to `#sec-results` could land before Plotly is ready; await the loader before rendering.
- **Confidence.** High. Discoverability category: not applicable — pure perf.

### B2. Replace the dashboard's 10 s full-page reload with a partial swap

- **User problem.** Dashboard `setInterval(location.reload, 10_000)` while any training runs (`dashboard.html:314`). Scroll position, expanded `<details>`, and the New-experiment modal state are all reset every cycle. On a Pi 5 over ingress this is also a wasted ~80 KB document each time.
- **Proposed change.** HTMX-poll `#experiments-grid` against an existing JSON-rendering endpoint — `/api/status` is already there (`web/app.py:2592`). Refactor the experiment-card markup into a Jinja partial used by both the page render and a new fragment-returning endpoint (`/api/dashboard/cards`).
- **Why now / why this addon.** HTMX is already vendored; the data and the card markup exist. This is a refactor, not a feature.
- **Pi 5 cost.** UI weight per cycle drops from ~80 KB to ~5 KB. CPU on the server unchanged.
- **Implementation sketch.** New endpoint returning the experiments-grid fragment; `hx-get … hx-trigger="every 10s"` on the grid.
- **Effort.** M.
- **Risk.** Race with the New Experiment modal swap target; scope `hx-swap` precisely.
- **Confidence.** Medium-high. Discoverability category: not applicable — pure UX.

### B3. Tell users about the tabs they can't see yet

- **User problem.** Forecast Accuracy is hidden until production mode; Tuning, Predictions, Generalisation, Covariate Analysis are hidden until a benchmark has run; Features is hidden until tree models exist (`experiment.html:67-80`). A new user has no way to know these exist.
- **Proposed change.** Render the tabs always, with `disabled` styling and a hover-tip explaining what to do to unlock them ("Run a benchmark to populate this tab", "Promote a model to production to see live forecast accuracy"). One-liner change in the tab-strip template.
- **Why now / why this addon.** Removes a "where did the menu item go?" loop for first-time users.
- **Pi 5 cost.** Nil.
- **Implementation sketch.** `experiment.html:64-81` — flip `{% if … %}` to always render with a `disabled` class and a `title` attribute.
- **Effort.** S.
- **Risk.** Nil.
- **Confidence.** High. Discoverability category: **exists in UI, conditional visibility hides it**.

### B4. Stop autosave failures from disappearing as a 4-second toast

- **User problem.** Every Settings-tab field autosaves silently (`experiment.html` `saveSettingsField` / `saveSettingsToggle`). Success and failure both flash a 4-second toast (`base.html:88-101`) and vanish. If the YAML write fails or validation rejects a value, the user has no persistent indicator and may walk away thinking they saved a change they didn't.
- **Proposed change.** When `/api/experiment-settings` returns an error, mark the offending field with a red dot and an error tooltip that **persists** until the field saves successfully. Existing override-dot pattern on the Models page already does the visual (`models.html:46-50`); reuse it.
- **Why now / why this addon.** Silent data loss is the worst UX bug. Cheap to fix.
- **Pi 5 cost.** Nil.
- **Implementation sketch.** Save callback already has `result.success`. Add a `_markFieldError(el, message)` that pins a dot; clear on next success.
- **Effort.** S.
- **Risk.** Nil.
- **Confidence.** High. Discoverability category: **error state too transient**.

### B5. Link the auto-generated Lovelace dashboard from the UI

- **User problem.** The add-on auto-writes a Lovelace dashboard YAML (`dashboard.py:182`, `main.py:5238`) and exposes it at `/dashboard_yaml` (`web/app.py:3091`), but nothing in the UI links to it. Users discover it only via the README or docs.
- **Proposed change.** Button on the System page near the path table ("Download Lovelace dashboard") and on each experiment header ("Open in HA dashboard" — uses the generated path).
- **Why now / why this addon.** Feature exists; cost-to-surface is two `<a>` tags.
- **Pi 5 cost.** Nil.
- **Implementation sketch.** `system.html:49-56` and `experiment.html:17-22`.
- **Effort.** S.
- **Risk.** Nil.
- **Confidence.** High. Discoverability category: **exists in code, no UI surface**.

---

## C. Model improvement workflows

Four proposals.

### C1. Stop users running long benchmarks against broken data

- **User problem.** A user clicks Run Pipeline, the addon trains every enabled model on 60 days of history, the user comes back two hours later, and the rank table is meaningless because the sensor had a 14-day flatline, 40 % missing readings, or a recorder gap they didn't know about. There's no pre-flight visibility into data quality.
- **Proposed change.** A "Data sanity" panel on the experiment Settings tab with a "Check now" button. Runs the existing fetch path (`main.py:_fetch_and_preprocess`) up to but not including model training; reports: rows fetched / expected, missing-value percentage, biggest gap (e.g. "47 h on 12 Apr"), recorder freshness (carry-forward warning if any), zero-run length, target range, max increment hits. Same panel is what the user wants if a benchmark just fails with "not enough data".
- **Why now / why this addon.** The pipeline already computes most of these stats but only logs them; it doesn't surface them in the UI. Users would otherwise spend benchmarks debugging data quality.
- **Pi 5 cost.** RAM minimal (one fetch). CPU: ~2–10 seconds. Disk/network: same as one cycle's fetch.
- **Implementation sketch.** Extract the existing logging stats from `_fetch_and_preprocess` (`main.py:878`) into a return value; new endpoint `/experiment/{name}/data-report`; new Settings sub-panel.
- **Effort.** M.
- **Risk.** Misleads users on legitimately bursty sensors (rain, EV charging) — frame all stats as "in your training window".
- **Confidence.** High. Discoverability category: **stats exist in logs, not in UI**.

### C2. Give new users a sensible starting model set

- **User problem.** The Models tab presents 24 cards (`web/app.py:751-824`). The bundled example enables only LightGBM + XGBoost, but `docs/MODEL_GUIDE.md` describes proven "starter sets" by data-volume and target shape — none of which are exposed in UI.
- **Proposed change.** A "Quick presets" row above the model grid in `experiment.html` Models tab: chips like *Fast* (LightGBM + XGBoost), *Balanced* (+ DLinear + SparseTSF), *Thorough* (+ LSTM + TiDE + NHiTS). Clicking flips the toggles. Same idea on `/models` if you want a global default for the next created experiment.
- **Why now / why this addon.** Curation lives in the docs but not the UI. A new user defaulting to whatever the bundled YAML had is wasting compute on the wrong models.
- **Pi 5 cost.** Nil.
- **Implementation sketch.** A `MODEL_PRESETS` literal alongside `MODEL_CATALOG` in `web/app.py`; render as buttons in `experiment.html:441-449`; client-side toggle flip.
- **Effort.** S.
- **Risk.** Users assume the preset is optimal; mitigate with a link to MODEL_GUIDE.md on hover.
- **Confidence.** High. Discoverability category: **exists in docs, not in UI**.

### C3. Tell users whether the rank difference is real

- **User problem.** The Results tab ranks models by a Demšar composite of MAE/RMSE/MASE. Two models with composites of 0.84 and 0.87 look meaningfully different, but with five CV folds the difference is often within noise. Users promote a model that's no better than runner-up.
- **Proposed change.** Below the metrics table, render a pairwise Diebold-Mariano grid (light cells = no significant difference at α=0.05). Already implemented in `benchmark/comparison.py` and computed during benchmark runs but never surfaced. Could be a simple coloured matrix.
- **Why now / why this addon.** Code path exists; cost is the UI rendering.
- **Pi 5 cost.** Nil.
- **Implementation sketch.** Have `_update_web_benchmark` (`main.py:1402`) attach DM results to `BenchmarkResult`; render in `experiment.html` under the Results tab.
- **Effort.** S.
- **Risk.** Stats concept that not all users will read; info-tip needed.
- **Confidence.** Medium. Discoverability category: **exists in code, no UI surface**.

### C4. Tune more than one model in a single sweep

- **User problem.** Tuning is single-model only (`experiment.html:784`). To compare tuned-LightGBM against tuned-LSTM, a user has to run two separate tuning sweeps and remember which results they're looking at.
- **Proposed change.** Add a "Tune all enabled" option to the model dropdown on the Tuning tab. Loops over each enabled model serially, storing per-model `TuningResult` (the data model already supports per-model results — `AppState.tuning_results: Dict[str, TuningResult]` in `web/app.py:249`). Surface a stacked table of per-model best-composite + apply buttons.
- **Why now / why this addon.** Loop is contained; tuning already serialises behind `_training_lock` (`main.py:534`).
- **Pi 5 cost.** RAM unchanged (`TUNING_NEURAL_BATCH_SIZE = 16` cap stays). CPU: N × current tuning time per sweep. Disk: linear in trials × models.
- **Implementation sketch.** Outer loop in `_run_tuning` (`main.py:4040`); UI rendering of the stacked-result table.
- **Effort.** M.
- **Risk.** Long-running; needs Stop-after-current-model and clear progress. The queue and stop plumbing already exist.
- **Confidence.** Medium. Discoverability category: **missing in the addon**.

---

## D. Model analysis & trust

Four proposals.

### D1. Tell users when conformal bands will actually appear

- **User problem.** After promoting a model to production, `_lower_80` / `_upper_80` sensors don't publish until enough residuals accumulate (`main.py:3146-3193`). The Forecast Accuracy verdict's "Uncertainty bands" tile just shows "—" with no explanation. Users assume the bands are broken or their automations grep for sensors that never appear.
- **Proposed change.** Turn the tile into a progress indicator when the calibration count is below threshold: *"Calibrating — 12 of ~30 cycles. Bands expected from ~14:30 tomorrow."* Add the same banner on the verdict card. Cold-start fallback (`main.py:3193`) already pools across model versions; surface that as a separate "Provisional bands — calibrating against full residual history" state.
- **Why now / why this addon.** Silence is the worst trust signal; the count is already queryable via `get_conformal_quantiles`. ETA estimate uses `forecast_every_minutes` from config.
- **Pi 5 cost.** Nil — one cheap DB count per page load.
- **Implementation sketch.** Extend `/experiment/{name}/forecast-accuracy` (`web/app.py:1813`) to return residual count and threshold; render in the verdict tile (`experiment.html:1169-1175`).
- **Effort.** S.
- **Risk.** ETA can mislead if recorder stalls; phrase as "expected" and prominently show `last cycle: X`.
- **Confidence.** High. Discoverability category: **state exists in code, hidden from user**.

### D2. Make the "is my model actually any good?" answer always visible

- **User problem.** Composite ranks tell the user which model is best of those they enabled, but not whether *any* of them beats the trivial baseline (today = yesterday + last week's seasonality). Seasonal Naive may not even be in the enabled set; in either case it's not a header chip.
- **Proposed change.** Always run Seasonal Naive in every benchmark (it costs ~milliseconds — no training). Surface a single chip at the top of the Results tab and on the Forecast Accuracy verdict card: *"vs Seasonal Naive: −18 % MAE (better)"* or *"+5 % MAE (worse than naive)"*. Same chip on the dashboard card so the user sees at a glance whether the experiment is worth running at all.
- **Why now / why this addon.** Seasonal Naive is already a registered backend (`seasonal_naive_backend.py`). The single most useful trust signal for a non-ML audience.
- **Pi 5 cost.** Nil — Seasonal Naive is instant.
- **Implementation sketch.** Force-include `seasonal_naive` in every benchmark even when not in `models_enabled` (don't surface it in the rank table if not enabled, just use its MAE for the skill chip). Touch `_run_benchmark` (`main.py:1573`) and the dashboard/experiment templates.
- **Effort.** M.
- **Risk.** Users see "worse than naive" and lose faith — that's actually the desired signal.
- **Confidence.** High. Discoverability category: **missing in the addon**.

### D3. Show when (and why) a retrain changed accuracy

- **User problem.** The forecast log already tags every prediction with the `model_version` it was issued under (`main.py:2830`). The Forecast Accuracy charts respect this but don't let the user *see* the retrain events on the timeline. "Did Tuesday's retrain make things better?" is unanswerable from the current UI.
- **Proposed change.** Overlay vertical retrain markers on the lead-time error chart and the convergence fan (Forecast Accuracy tab). Hover shows: model_name, model_version, time, what changed since last retrain (covariate set diff if any). A small "retrains in this window" count appears as a chip in the top controls.
- **Why now / why this addon.** Data exists; this is a rendering job.
- **Pi 5 cost.** Nil — model_version timestamps are already columns in `forecast_log`.
- **Implementation sketch.** Extend `/experiment/{name}/forecast-accuracy` (`web/app.py:1813`) to return retrain events. Render markers in the Plotly lead-time chart (`experiment.html:1231`) and the convergence chart (`experiment.html:1375`).
- **Effort.** M.
- **Risk.** Chart clutter if many retrains; hide markers when zoomed out.
- **Confidence.** High. Discoverability category: **data exists, no UI surface**.

### D4. Surface when the recent target has drifted from the training window

- **User problem.** Bad CV scores can mean a bad model — or they can mean the last fold of data is from a regime the rest of the history doesn't cover (heatwave, EV install, behaviour change). Users have no way to tell these apart, and either retrain on more history or distrust a perfectly good model.
- **Proposed change.** A "Training window check" stat at the top of the Results tab: train-window vs test-window mean / std / 90-percentile, plus a PSI score (one number, with a tip explaining what >0.2 means). Optional small histogram. Same data the runner already splits.
- **Why now / why this addon.** The train/test split is already in `BenchmarkRunner._prepare_train_test_splits` (`benchmark/runner.py:227`); only the comparison stats are new.
- **Pi 5 cost.** Trivial; one pass over a few hundred rows.
- **Implementation sketch.** Compute drift stats in `run_benchmark` (`benchmark/runner.py:730`), attach to `BenchmarkResult`. Render in `experiment.html` Results tab header.
- **Effort.** M.
- **Risk.** PSI/KS are jargon for the target audience; lead with a plain-English verdict ("Test window differs from training window — model may underperform on similar future data") and bury the number behind an info-tip.
- **Confidence.** Medium. Discoverability category: **missing in the addon**.

---

## Prioritised top 10

Score = (Impact × Confidence) / Effort, where S=1, M=3, L=5; Impact and Confidence on 1–5 scales. Ties broken by lower risk first.

| Rank | # | Title | Impact | Confidence | Effort | Score |
|---:|---|---|:-:|:-:|:-:|:-:|
| 1 | D1 | Tell users when conformal bands will actually appear | 5 | 5 | S | 25.0 |
| 2 | A1 | Stop the add-on pegging the Pi during training | 4 | 5 | S | 20.0 |
| 3 | A5 | Let HA automations react when a benchmark or retrain finishes | 4 | 5 | S | 20.0 |
| 4 | B1 | Make the experiment page load fast over HA ingress | 4 | 5 | S | 20.0 |
| 5 | C2 | Give new users a sensible starting model set | 4 | 5 | S | 20.0 |
| 6 | A2 | Delete an experiment without using curl | 3 | 5 | S | 15.0 |
| 7 | A6 | Stop advertising future-role covariates that don't work (option a) | 3 | 5 | S | 15.0 |
| 8 | B3 | Tell users about the tabs they can't see yet | 3 | 5 | S | 15.0 |
| 9 | B4 | Stop autosave failures from disappearing as a 4-second toast | 3 | 5 | S | 15.0 |
| 10 | B5 | Link the auto-generated Lovelace dashboard from the UI | 3 | 5 | S | 15.0 |

Below the cut, in priority order, in case you want to slice differently: **C3** Diebold-Mariano surface (9.0), **D2** Naive-skill chip (8.33), **A3** Last-N benchmark history (6.67), **C1** Pre-flight data report (6.67), **D3** Retrain-event timeline (5.33), **B2** HTMX partial dashboard refresh (4.0), **D4** Training-window drift (3.0), **C4** Tune-all (3.0), **A4** Roll back retrain (3.0).

Every item in the top 10 is an **S**. That's not an accident — once the hard constraints rule out big-batch ML work, the highest-leverage changes are the small UX fixes that close gaps between what the code already does and what the user sees.

---

## What I'd build first and why

**A1 — Stop the add-on pegging the Pi during training.**

It's the change with the highest *outcome-to-risk ratio* of anything in the list:

- It directly fulfils one of your hard constraints ("Sustained CPU load must be bounded"). At present, no code path applies the very settings the UI offers — clicking "Training CPU cores → 2" on a Pi 5 changes the YAML and does nothing else.
- It's an honest bug-fix, not a feature: `cpu_cores` and `nice_priority` are persisted, displayed, and explained in the System page (`system.html:62-100`), but `grep` for `set_num_threads`, `os.nice`, `OMP_NUM_THREADS`, or `sched_setaffinity` across the codebase returns zero matches.
- The fix is a few lines in `main.py` at startup plus an env-variable handoff before `torch` / `lightgbm` / `xgboost` import. ARM64 safe. No new dependency. No Pi 5 cost — it's a *reduction* in cost.
- The risk surface is tiny: a user who picks 1 core and notices slow training; mitigate with a guidance message in the form when `cores < ceil(system_count / 2)`.
- It's a self-contained ~1-day change that I can ship as a PR with screenshots of `top` or `htop` before and after on a Pi 5.

If you want a second pick for the same week, **D1** (conformal-band countdown) is the highest-impact trust improvement at S effort and would land naturally alongside.
