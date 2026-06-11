# ml-forecast-lab — Codebase Audit Report

**Date:** 2026-06-10 · **HEAD:** `e4654b2` (v2.40.14) · **Audit scope:** correctness, UI contract, Pi-5 performance envelope.
All paths below are relative to `ml-forecast-lab/` (the add-on directory) unless prefixed with `/`.

## Environment & baseline

- Audit environment: Linux 6.18.5 x86_64 container, Python 3.11.15, torch 2.12.0, numpy 1.26.4. **Not a Pi-5** — timings are relative indicators; structural findings transfer, absolute numbers don't (expect roughly 2–4× slower on 4×A76 with SD/NVMe I/O).
- `pip install -r requirements.txt -r tests/requirements-dev.txt`: clean.
- Baseline `pytest tests/ --ignore=tests/synthetic`: **397 passed, 10 failed, 5m22s**. All 10 failures are in `tests/integration/test_pv_forecast_pipeline.py` (see F1). `pytest tests/smoke/`: **74 passed, 5.0s**. All later "tests pass" claims reference this baseline.

## Executive summary

1. **The repo's own integration suite fails 10/10 at HEAD with the exact user-reported "flat-zero forecast" signature, on the production-default config — and CI never runs it.** The failure reproduces at every version back to v2.40.9; an old test config (`daily_loss_weight=0.25`) masked it, production (`0.0`) never had the mask. The commit that exposed it (`b8ad21a`) merged with the failures acknowledged as "follow-up". (F1)
2. **Published conformal bands don't target their nominal coverage**: a math error uses the (1−α/2) quantile of |residual| where (1−α) is required — nominal-80% bands deliver ~90% coverage and are ~1.5× too wide. Verified by simulation. (F2)
3. **The analytics SQL joins are O(N_forecasts × N_actuals)** — a CTE re-scanned per row. Measured: conformal quantiles 78 s, `/forecast-accuracy` 240 s on one experiment's 30-day log. The conformal query runs **synchronously on the event loop every forecast publish**, and the accuracy query exceeds the frontend's 60 s timeout, so the tab can never load at realistic volume. The same codebase already contains the fix (an indexed temp table, 431 ms on identical data) — applied to only one of ~9 query sites. (F3, F4)
4. forecast_log has **no age-based retention** (only champion-change pruning), so the above degrade without bound. (F5)
5. The "global training lock" covers benchmark + retrain but **not tuning or covariate analysis** — two heavyweight training jobs can run concurrently on the Pi. (F7)
6. UI contract is in good shape: all 30 GET routes return 200 with correct types, every rendered URL honours the ingress prefix, HTMX targets resolve, JS field names match endpoint JSON. The gaps are perf-induced (timeouts), not wiring.

---

## Findings table

| ID | Sev | Phase | Location | Tag | Description |
|----|-----|-------|----------|-----|-------------|
| F1 | P0 | 1/2.6 | `tests/integration/test_pv_forecast_pipeline.py`; `.github/workflows/tests.yml` | [EXECUTED] | Production-default neural path produces flat-zero forecasts in repo's own integration scenarios; suite fails 10/10 at HEAD and is absent from CI |
| F2 | P1 | 1.4 | `db.py:1737-1738`, `main.py:5029-5030` | [EXECUTED] | Conformal band uses (1−α/2) quantile of \|residual\| → 80% bands actually ~90% coverage, ~1.5× too wide |
| F3 | P1 | 1.1/3.2 | `db.py:1667-1688`, `main.py:4981` | [EXECUTED] | O(N×M) conformal query (78 s measured) runs synchronously on the event loop on every forecast publish; scales with `max_age` (no date cutoff on actuals CTE) |
| F4 | P1 | 3.2/2 | `db.py:519-963,1313,2146`; `web/app.py:2990,3082,3203,2837` | [EXECUTED] | Same O(N×M) join in accuracy (240 s > 60 s JS timeout), trajectory (10 s, on event loop), evolution, log-stats (raw cursor, no lock); fix already exists at `db.py:1847-1934` |
| F5 | P2 | 1.2/3 | `main.py:2587-2592`, `web/app.py:1542` | [EXECUTED] | No age-based forecast_log retention — 16.5 MB / 69k rows per experiment-month measured; F3/F4 degrade unboundedly |
| F6 | P1 | 1.3 | `preprocessing.py:322-333` vs `main.py:5989,3904`, `runner.py:618-638` | [STATIC] | `apply_log_transform` derives shift=\|min\|+1 for signed targets but every inversion hard-codes `expm1` (shift=1) → wrong published values for signed target + `log_transform` |
| F7 | P2 | 1.2 | `main.py:530-532,6035,6659` vs `879,1007,4063` | [STATIC] | `_training_lock` claims "all code paths" but tuning and covariate analysis never acquire it — concurrent heavy training possible |
| F8 | P2 | 1.3 | `benchmark/runner.py:485-490` vs `main.py:4247-4261` | [STATIC] | Benchmark CV trains past-only windows; production trains extended (past+future) windows — leaderboard ranks a different input architecture than the one deployed |
| F9 | P2 | 1.2 | `db.py:41-63` (`check_same_thread=False`, single conn, RLock) | [EXECUTED] | Single shared connection + module lock defeats WAL read concurrency; the in-code comment claiming "WAL lets offloaded readers proceed" is wrong — F4's 240 s query blocks every DB user, and sync DB calls from the event loop (`main.py:5083,1703`) then freeze the loop behind the lock |
| F10 | P2 | 1.5 | `main.py:1020-1047` (stop), `web/app.py:2157-2196` (delete) | [STATIC] | Stop-training cancels the asyncio task but the executor thread keeps training to completion (no cooperative cancel flag in any backend); experiment delete doesn't cancel either — minutes-to-hours of orphaned CPU burn on Pi |
| F11 | P2 | 1.6 | `config.py:293,325,377,437`; `mlfl.yaml:117` | [STATIC] | `output_units`, `custom_metrics`, `stability_focus`, `future_covariate_features` are parsed (and `output_units` shipped in the example YAML) but read by nothing — silent misconfiguration |
| F12 | P3 | 1.6 | `__main__.py:197-200` | [STATIC] | Any ImportError of the main app silently degrades to a stub HTTP server that reports `healthy` |
| F13 | P3 | 3.1/3.5 | `main.py:718-758`; measured | [EXECUTED] | Eager registration imports the full ML stack at boot: ~4.4 s import, ~800 MB RSS floor before any data (x86); all trained benchmark models stay referenced until the benchmark ends (`main.py:2747-2794,2923`) |
| F14 | P3 | 1.3 | `benchmark/runner.py:556-559` | [STATIC] | Embargo rows are excluded from the neural test-window bridge → first `window_size` test windows per fold span a time discontinuity |
| F15 | P3 | 1.2 | `main.py:4453` vs `4079-4106` | [STATIC] | Post-retrain forecast doesn't set `_forecast_running` — can run concurrently with a scheduled forecast for the same experiment (duplicate publish/log rows) |
| F16 | P3 | 1.5 | `main.py:4516` vs `4570-4572` | [STATIC] | `model.bin` written in place (meta JSON is atomic) — crash mid-save can pair old meta with torn/new binary; restore path catches load failures so recovery is a forced retrain |
| F17 | P3 | 2.2 | `web/app.py:1177-1186,1369,1395` | [EXECUTED] | Dead handler context: `models_enabled` (read from the *first* experiment only) never referenced by `models.html`; `forecast_data`/`training_summary` passed to `experiment.html` but never referenced |
| F18 | P3 | 1.6 | `config.py:549` vs `benchmark/runner.py:422` | [STATIC] | `recency_half_life_days` default drift: dataclass 0.0 (disabled) vs runner fallback 7.0 (latent — only fires if the key is absent from the cfg dict) |
| F19 | P3 | 1.3 | `features.py:629-636`, `build_features` | [STATIC] | All time-of-day features are computed on naive **UTC** timestamps — human-schedule patterns shift 1 h in feature space at each DST transition (GB/EU targets); holiday flags keyed to UTC dates |

---

## Per-finding detail (ordered by severity)

### F1 — P0 — Production-default neural path emits flat-zero forecasts; the suite pinning it is merged failing and not a CI gate `[EXECUTED]`

**Evidence.**
- Baseline run: 10/10 tests in `tests/integration/test_pv_forecast_pipeline.py` fail at HEAD with `"forecast is identically zero … This is the user's primary failure signature"` (e.g. `test_pv_forecast_save_load_predict_roundtrip`, `test_pv_forecast_no_covariates_no_log_transform`, all 6 `anchor_robust` variants). The file's docstring states it runs "the exact code path the addon uses in production" (`create_sliding_windows` → `fit` → `build_inference_window` → `predict_sequence` → `expm1`), mirroring `_retrain_and_cache`/`_forecast_with_cached`.
- Causal isolation (all executed in this environment, same torch 2.12):
  - HEAD test file + HEAD code → **fail**.
  - HEAD test file + every mainline merge back to `7997c14` (v2.40.9) → **fail** — so not introduced by PR #76.
  - *Old* test file (which set `daily_loss_weight=0.25`) + old code `c0220ce` → **pass**. The 0.25 cumulative-loss term was masking the collapse; commit `b8ad21a` changed the mock to the production default `0.0` explicitly "to match what actual experiments saw", and its own message says "Follow-up commit will address any tests that still fail after this change" — no such commit exists.
- Release gate: `.github/workflows/tests.yml` defines only `smoke` (`tests/smoke/`) and `unit` (`tests/unit/`) jobs. `tests/integration/` is in **no** CI job, so PR #76 merged green.
- The shipped image installs `torch>=2.0.0` unpinned (`requirements.txt:8`, `Dockerfile:27-30`) — a fresh build resolves the same torch line this audit used.

**Repro.** `pytest tests/integration/test_pv_forecast_pipeline.py -x` at HEAD.

**Fix direction.** Treat the integration failures as the bug report they were designed to be: root-cause the NLinear extended-window collapse under `daily_loss_weight=0` (the masking effect of the removed cumulative term suggests the interval-only objective lets the head collapse to the softplus/log-space zero attractor); add `tests/integration/` as a CI job; pin or upper-bound torch.

**Confidence.** High that the production-default code path fails the repo's own production-mirror scenarios at HEAD. Medium that field deployments reproduce it (synthetic PV data, though built to match a real user report at real scale; affected scope is the neural extended-window path — tree backends unaffected).

### F2 — P1 — Conformal bands over-cover: wrong quantile level for absolute residuals `[EXECUTED]`

**Evidence.** `db.py:1737-1738`: `alpha = 1−level; q = 1−alpha/2`, applied to **absolute** residuals; `main.py:5029-5030` builds the band as `pred ± q`. For a symmetric band from |residual|, coverage = P(|r| ≤ q̂) = the quantile level itself: using 1−α/2 yields 1−α/2 coverage. The docstring (`db.py:1577-1579`) and `web/app.py:2758` repeat the error ("For an 80%-band … that is the 90th percentile") — that rule applies to *signed*-residual two-sided bands, not absolute ones.

**Repro.** `/tmp/conformal_check.py` (simulation mirroring the code): nominal 80% → realized 90.0% coverage, band 1.55× wider than the correct 80th-percentile band; nominal 90% → realized 95%. The repo's own `scripts/conformal_coverage_check.py` would report this as a persistent "+10 pp / WIDE (bands too conservative)" verdict without identifying the cause.

**Fix direction.** Use the `level` quantile of |residual| (optionally with the finite-sample (n+1) correction), or switch to signed-residual two-sided quantiles. Recalibration is automatic on next publish.

**Confidence.** High.

### F3 — P1 — O(N×M) conformal query on the event loop every forecast publish `[EXECUTED]`

**Evidence.** `main.py:4981/4994`: `_publish_forecast_sensors` calls `self.history_db.get_conformal_quantiles(...)` directly (no `to_thread`/executor) on every production publish — i.e. every ~30 min per experiment, plus after each retrain. The query (`db.py:1667-1688`) joins `forecast_log` to an actuals-grid CTE; `EXPLAIN QUERY PLAN` shows `CO-ROUTINE actuals_grid` + `SCAN ag` per outer row — the grid (with per-row `strftime`) is re-scanned for each of the N forecast rows. The CTE has **no date cutoff on the actuals table**, so M grows to the full `max_age` window (365-day default ⇒ ~17.5k grid rows at 30-min).
Measured on a synthetic 30-day single-experiment DB (69k forecast_log rows, 1.4k actuals — *smaller* than a mature install): **77.9 s** on x86. While it runs, the event loop is frozen: web UI, HTMX polls, HA ingress, scheduler ticks all stall.

**Repro.** `/tmp/synth_db.py` + `/tmp/synth_db2.py` (EXPLAIN output included above).

**Fix direction.** The codebase already contains the fix: `get_forecast_coverage` v2.39.3 materializes the grid into an indexed temp table (`db.py:1847-1934`) — **431 ms on the identical data, a ~180× difference**. Apply the same pattern (plus a date cutoff on the actuals CTE) and move the call off the event loop.

**Confidence.** High.

### F4 — P1 — Same O(N×M) join in the web analytics; accuracy can never load; three endpoints block the event loop `[EXECUTED]`

**Evidence.**
- `/experiment/{name}/forecast-accuracy` (offloaded via `to_thread`, `web/app.py:2670`): **240 s** end-to-end on the synthetic DB. The frontend aborts at 60 s (`experiment.html:3192`, `ACCURACY_FETCH_TIMEOUT_MS = 60000`) — at realistic volume the Forecast Accuracy tab times out forever while the server keeps grinding with `db._lock` held (see F9 for the convoy effect).
- `/forecast-trajectory` (`web/app.py:2990`): **9.8 s**, called **directly in the async handler** — event loop frozen; the fallback ladder can run the query up to 3×.
- `/forecast-evolution` (`web/app.py:3082`) and `/forecast-stability` (`web/app.py:3203`): same unoffloaded pattern (fast today only because their SQL shape differs; evolution 2 ms, stability 214 ms measured).
- `/forecast-log-stats` (`web/app.py:2837`): uses `db.conn.cursor()` **directly in the handler** — bypasses both the documented lock discipline (`db.py:22-30`) and thread offload.

**Fix direction.** Temp-table join as in F3; offload all five endpoints uniformly; remove the raw-cursor access.

**Confidence.** High.

### F5 — P2 — forecast_log grows without bound `[EXECUTED]`

**Evidence.** `cleanup_forecast_log` is invoked only on champion change after a benchmark (`main.py:2587-2592`) and on promote (`web/app.py:1542`) — both no-ops for a stable production experiment. Measured growth: 69k rows / 16.5 MB per experiment-month at the default 30-min cadence × 48 horizons. F3/F4 latency scales with it.

**Fix direction.** Periodic age-based prune (e.g. `issued_at < now − max(30, conformal max_age_days)` per experiment) in the retrain cycle.

**Confidence.** High.

### F6 — P1 (conditional) — log-transform inversion ignores the derived shift for signed targets `[STATIC]`

**Evidence.** `apply_log_transform` (`preprocessing.py:322-333`) derives `shift = |min|+1` when the series has negative values and stores it in `series.attrs`; `invert_log_transform` exists to read it back but is **never called** — all six inversion sites hard-code `np.expm1` (shift=1): `main.py:5989` (cached publish), `main.py:3904,3931` (production inference), `main.py:3331-3333` (holdout display), `runner.py:618-638` (benchmark metrics). For a signed target (e.g. grid import/export power) with `log_transform: true`, every published value is `exp(x)−1` instead of `exp(x)−shift` — systematically offset and distorted, then clipped at 0.

**Repro (not executed).** Could be demonstrated with a 10-line script feeding a signed series through `_fetch_and_preprocess`'s transform then the publish inverse; not executed because the affected combination (signed target + log_transform) is config-dependent and the arithmetic mismatch is unambiguous from the code.

**Fix direction.** Persist the shift in the model cache meta and use `invert_log_transform`; or refuse `log_transform` for signed targets at config validation.

**Confidence.** High that the code is inconsistent; medium for field impact (depends on users enabling log_transform on signed targets — nothing prevents it).

### F7 — P2 — Tuning and covariate analysis train outside the "global" training lock `[STATIC]`

**Evidence.** `main.py:530-532` documents `_training_lock` as "ensures only one training operation … across all code paths". Holders: benchmark trigger (`main.py:879`), retrain trigger (`main.py:1007`), retrain queue (`main.py:4063`). Non-holders that do full training workloads: `_run_tuning` (`main.py:6035`, Optuna n_trials × CV fits) and `_run_covariate_analysis` (`main.py:6659`, models × covariate-combination fits), reachable from `/run-tuning`, `/run-tuning-all`, `/run-covariate-analysis` (`web/app.py:3314,3385,3262`). A scheduled retrain or benchmark can start mid-tuning: two training stacks concurrently on 4 cores / 8 GB.

**Fix direction.** Acquire the lock in both, or funnel them through the retrain queue.

**Confidence.** High.

### F8 — P2 — Leaderboard evaluates a different neural input pipeline than production deploys `[STATIC]`

**Evidence.** Production training extends windows with future-known features (`main.py:4247-4261`, `future_features_df=...`, rationale comment: "brings the neural path to parity"). Benchmark CV (`benchmark/runner.py:485-490`) calls `create_sliding_windows` **without** `future_features_df` — `runner.py` never imports `compute_known_future_features`. So model selection (Promote, auto-select) ranks neural backends trained on past-only windows, then deploys them retrained on extended windows; the holdout chart (`main.py:3184-3289`) *does* use extended windows, so the chart and the leaderboard also disagree with each other.

**Fix direction.** Pass the same `future_features_df` construction into the CV fold training (per-fold index, no leakage — the features are deterministic or in-sample observed).

**Confidence.** High (structural); impact on actual rankings unmeasured.

### F9 — P2 — Single shared SQLite connection + RLock defeats WAL; sync calls from the loop convoy behind slow readers `[EXECUTED]`

**Evidence.** `db.py:41-63`: one connection (`check_same_thread=False`), every method behind a `threading.RLock`. The comment claims "WAL lets the offloaded readers proceed while the publish-cycle writer owns the connection" — WAL read concurrency requires **separate connections**; with one connection the lock fully serializes. Mechanism measured in F4: a 240 s accuracy read holds the lock; the forecast cycle then calls `log_forecast` / `get_history` synchronously on the event loop (`main.py:5083,1703,1761,1767`) and blocks the **entire event loop** until the lock frees.

**Fix direction.** Connection-per-thread (or a small read pool) + keep the writer single; or route every DB call through one worker thread and make the loop-side API async.

**Confidence.** High.

### F10 — P2 — "Stop training" and experiment delete don't stop the executor thread `[STATIC]`

**Evidence.** Training runs via `run_in_executor` (`main.py:2918,4325`). `_stop_training_trigger` (`main.py:1020-1047`) cancels the asyncio task — the coroutine exits at the await, but the thread running `model.fit`/`run_single_model` is not interruptible and no backend checks a cancel flag (grep over `models/` finds none). The UI reports "cancelled" while up to 4 cores stay saturated until the current fit (or the whole `run_single_model` fold loop) finishes; epoch events keep streaming. `delete_experiment_route` (`web/app.py:2157-2196`) doesn't even attempt cancellation; the orphaned benchmark later writes results for the deleted experiment (self-healed on next restart by the stale-benchmark filter, `main.py:845-852`).

**Fix direction.** Thread a `threading.Event` into `fit` via the existing `epoch_callback` plumbing and check it per epoch/fold.

**Confidence.** High.

### F11 — P2 — Config keys parsed and advertised but never read `[STATIC]`

**Evidence.** Defined on `ExperimentCfg` and accepted from YAML, consumed nowhere (grep over the package excluding `config.py`): `output_units` (`config.py:325`; shipped in the example `mlfl.yaml:117` and the smoke fixture — a user setting `units: W, output_units: kW` gets no conversion and no warning), `custom_metrics` (`config.py:293`), `stability_focus` (`config.py:437`), `future_covariate_features` (`config.py:377`). Contrast: the deprecated `subtract` field *does* get a loud warning (`config.py:935-940`), so the mechanism exists.

**Fix direction.** Either wire them or remove from the dataclass + example YAML and warn like `subtract`.

**Confidence.** High.

### F12 — P3 — ImportError silently degrades the add-on to a healthy-looking stub `[STATIC]`

`__main__.py:197-200`: any `ImportError` raised importing `ml_forecast_lab.main` (including transitive failures of torch/lightgbm in a broken image) logs once, then serves a stub `/health` returning `healthy`. No forecasts, no UI, sensors go stale, watchdogs see green. Fix direction: crash (s6 restarts and surfaces the error in the supervisor log) or report unhealthy.

### F13 — P3 — Startup/memory floor; benchmark keeps all trained models resident `[EXECUTED]`

Measured: `import torch` 1.7 s, full backend set ~4.4 s and **~800 MB max RSS** before any data (x86; Pi-5 boot from SD will be substantially slower — one-time cost, then `retrain IMMEDIATELY` for production experiments compounds first-boot load, `main.py:7179`). During a benchmark, every trained model object stays referenced in `models` until the run ends (`main.py:2747-2794`, results at `2923`) — with 20+ backends enabled, the sum of all trained models is resident at once on top of the torch floor. With default `days_history=30` the sliding-window tensors are small (~5 MB); at 365 days they reach ~40–200 MB per neural fold (shape `(17329, 96, n_ch)` float32 observed). Fix direction: drop each model after its holdout predictions are captured.

### F14 — P3 — Embargo punches a hole in neural test windows `[STATIC]`

`runner.py:556-559`: the bridge prepends `train_idx[-n_bridge:]` to `test_idx`; with `cv_embargo_periods=2` (default) the two embargo rows are missing, so `create_sliding_windows` builds test windows spanning a 2-step discontinuity (values and temporal channels jump). Affects the first `window_size` test windows per fold for neural models — a small, systematic distortion of CV metrics. Fix direction: bridge with `test_start − n_bridge` rows (context may overlap the embargo — embargo only needs to keep them out of *training*), or mask the seam windows.

### F15 — P3 — Duplicate concurrent forecast for the same experiment after retrain `[STATIC]`

`_retrain_and_cache` tail-calls `_forecast_with_cached` (`main.py:4453`) without setting `_forecast_running`; a simultaneously scheduled `_forecast_single` (`main.py:4079-4106`) can interleave → two publishes and duplicate `forecast_log` rows in the same minute, possibly under different `model_version` tags mid-swap. Self-limiting but pollutes per-issuance analytics. Fix direction: reserve the flag in the retrain tail call too.

### F16 — P3 — Non-atomic `model.bin` write `[STATIC]`

`main.py:4516` writes `model.bin` in place while `cache_meta.json` gets tmp+rename (`main.py:4570-4572`). Crash between them can leave old-meta + new/torn binary; `_restore_cached_models` catches load failures (forced retrain — recoverable), but a same-architecture stale-meta pairing would load silently with wrong `feature_cols`. The channel-parity guard (`main.py:5703-5714`) catches most neural cases; tree caches have no equivalent guard. Fix direction: write `model.bin.tmp` + rename, meta last.

### F17 — P3 — Dead handler context / dead config read `[EXECUTED]`

Template-contract diff (live render with a recording `Undefined` + `jinja2.meta` over the include graph): no undefined variables rendered on any page in the no-data state; drift is one-directional — `models.html` never references `models_enabled` (the handler reads it from **the first experiment only**, `web/app.py:1177-1186` — dead and misleading code), `experiment.html` never references `forecast_data` or `training_summary`. Remaining "missing" names are Jinja `{% set %}` locals (meta false positives, verified by inspection).

### F18 — P3 — `recency_half_life_days` default drift `[STATIC]`

`config.py:549` defaults 0.0 (weighting disabled); `runner.py:422` falls back to 7.0 when the key is missing. Currently latent (`dataclasses.asdict` always supplies the key), but any caller building a partial cfg dict gets weighting silently enabled. Align the fallback to 0.0.

### F19 — P3 — Time features in UTC, not site-local `[STATIC]`

The whole pipeline is naive-UTC (deliberate, `main.py:4902-4913`), including `hour_of_day`/`hour_sin` features (`features.py:629-636`) and holiday lookups by UTC date. Correctness of timestamps is preserved end-to-end (publish re-localizes to UTC, JS converts — verified), but human-schedule patterns shift by 1 h in feature space at each DST transition and the model must relearn; holidays start/end offset by the UTC gap. A modeling-quality wrinkle, not a data bug.

---

## Phase 2 — UI contract verification (summary of executed checks)

- **Boot & probe** `[EXECUTED]` (`/tmp/probe_app.py`, app booted exactly like the smoke suite): all 30 GET routes 200 with correct content-type (expected 404s for no-data endpoints: `/forecast`, `/tuning`, `/covariate-analysis`, `/debug_log` with no log file; expected 503s for DB-less analytics). 15 POST probes behave (400/409/202/200 as designed). `/settings`, `/status`, `/system` all render the same `system.html` — intentional aliases.
- **Handler ↔ template contract** `[EXECUTED]`: clean except F17. Caveat: the no-data state skips data-rich branches; a populated-state render could still hide undefined names (see could-not-verify).
- **HTMX wiring** `[EXECUTED]`: only one HTMX surface (dashboard grid swap, `dashboard.html:37`/`_dashboard_grid.html:7`) — URL prefixed, method matches, target `#experiments-grid` present in both full page and fragment; fragment returns a fragment.
- **Ingress safety** `[EXECUTED]` (`/tmp/ingress_check.py`): with `X-Ingress-Path` set, every `href/src/action/hx-*` URL in all rendered pages carries the prefix; all dynamic JS requests (40+ `fetch`, 2 `EventSource`) go through `BASE`/`BASE_PATH` from `base.html:11`; client-side redirects prefix `result.redirect` (`dashboard.html:264`). No violations found.
- **JS/Plotly data contract** `[EXECUTED]` on the populated synthetic DB: `accuracy` (`lead_time_curve`, `coverage.overall`, `calibration`, `nominal_interval_level`), `evolution` (`cycles[].{issued_at,predictions,targets}`, `actuals.{targets,values}`), `stability` (`summary.median_*`), `log-stats` keys all match the fields `experiment.html` reads. Timestamps published with explicit `+00:00` (`main.py:4910-4913`) as the JS expects. Band ordering: upper/lower computed as pred±q, lower clipped ≥0 only for cumulative targets (`main.py:5031-5032`) — a signed target can legitimately have lower<0, JS handles it.
- **Payload size** `[EXECUTED]`: analytics responses 1–24 KB (gzip 0–7 KB) — server-side aggregation already bounds Plotly payloads. **Not a finding.**

## Release-gate gaps (prioritised)

The smoke suite (74 tests, 11 files — boot, page renders, create/delete, promote/toggle-mode, model-params CRUD, model toggles, covariate-remove disambiguation, settings save, analytics empty-state, HA-entities offline, production-model resolution) is a solid *empty-state* gate. Not covered, in priority order:

1. **`tests/integration/` is not a CI job** — the only tests that exercise train→predict→publish are exactly the ones failing (F1). Add the job; mark slow tests accordingly.
2. **Analytics endpoints against a populated `forecast_log`** — a seeded-DB smoke fixture (like this audit's synthetic DB) would have caught F3/F4 timings and guards the JSON contract where data-dependent branches live.
3. **Conformal band content** — a unit test asserting realized coverage of the constructed band on synthetic residuals would have caught F2 in one assert.
4. **Ingress-prefix regression test** — one smoke test rendering pages with `X-Ingress-Path` and asserting no unprefixed URLs (this audit's check is directly portable).
5. **Scheduler/lock semantics** — no test covers `main_loop` timers, the retrain queue, `_training_lock` coverage (F7), or stop-training actually stopping (F10).
6. **Cache persist/restore round-trip** — `_persist_cached_model`/`_restore_cached_models`/rollback with real files (F16, schema_version skew) is untested.
7. **SQLite delta-fetch / carry-forward logic** in `_fetch_and_preprocess` (cache hit + HA delta merge + recorder-gap synthesis) — pure-logic, easily unit-tested, currently only exercised in production.

## Could not verify (and what would resolve it)

- **Field reproduction of F1** on real HA data/hardware — needs one affected user's `debug_save_training_dumps` bundle or a Pi-5 run of the integration suite against the shipped image.
- **Pi-5 absolute timings** — all measurements are x86; structure (query plans, O(N×M), lock convoy) transfers, constants don't.
- **Populated-state template rendering** (benchmark results, tuning results, covariate analysis in context) — would need a fabricated full `BenchmarkResult` fixture; the recording-Undefined harness in `/tmp/probe_app.py` is reusable for this.
- **SSE `training-stream` under ingress** — skipped (infinite stream) in the probe; static reading shows prefixed `EventSource` URLs, but proxy buffering behaviour is environment-specific.
- **A full 20+-backend benchmark's peak RSS** — too heavy for this environment; F13's per-component numbers are measured, the sum is extrapolated.
- **Whether any real config combines signed targets with `log_transform`** (F6 trigger) — nothing in code or docs prevents it.
