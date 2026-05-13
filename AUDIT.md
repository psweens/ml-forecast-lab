# ML Forecast Lab — Improvement Audit

**Scope.** Functionality, front-end usability, and tools that help users *understand and improve* their models. Constrained by the deployment reality: a Home Assistant add-on running on a Raspberry Pi 5 (4–8 GB RAM, 4 cores, aarch64), serving one human via LAN.

**Method.** Static read of the codebase at `claude/follow-audit-prompt-1cRLE`. No runtime profiling. File:line citations let you jump straight to each call-site.

---

## TL;DR

The add-on already has the bones of a serious benchmarking tool: 22 backends, walk-forward CV with embargo, Demšar composite ranking, online conformal bands, hyperparameter tuning, covariate-impact analysis, and an SSE-driven live training UI. The biggest gaps are not modelling — they're **explanation, comparison, and onboarding**:

1. The UI tells you *which* model won but not *why* one beat another, *where* either fails, or *whether* the difference is statistically meaningful.
2. There's no first-run wizard or data-quality preview, so new users hit walls (recorder retention, lag/history mismatches, NaN-heavy sensors) without diagnostic signal.
3. Neural models are interpretability black boxes — feature importance is tree-only.
4. The Pi-5 footprint is mostly fine, but a 1.05 MB Plotly bundle is loaded eagerly and there's no hardware-aware preset for picking model sets.

Concrete, prioritised backlog at the bottom.

---

## 1. Front-end usability

### 1.1 No first-run onboarding
A brand-new user must hand-write `mlfl.yaml` before the dashboard shows anything useful. `main.py:253–270` writes a stub config pointing at a fake `sensor.test_value`, which means the first dashboard view is empty or broken.

**Add a "Create your first experiment" wizard** (web modal, three steps):
1. Pick target entity (browse HA entities via existing `ha_interface` client).
2. Show last 7 days of values + computed data-quality flags (gaps, resets, stalls, NaN ratio, recorder-retention vs requested `days_history`).
3. Choose preset: *Lightweight (Pi-friendly)* / *Balanced* / *Accurate* — each ships a model set, `n_lags`, and CV folds tuned to the data shape.

There's no preset bundle today (`config.py:50–525` is all schema, no recipes). Adding three named bundles closes the biggest onboarding gap.

### 1.2 Metrics are unexplained
The Results tab (`web/templates/experiment.html:517–656`) renders MAE / RMSE / MASE / composite rank columns with no tooltips. A Home Assistant user evaluating LightGBM vs PatchTST has no in-app way to find out what MASE means or why composite rank is the *primary* sort.

**Add inline `<abbr>` tooltips on every metric header** with a one-sentence definition and the unit ("MAE — Mean Absolute Error, in target units"). The same pattern across Generalisation, Tuning, and Forecast Accuracy tabs costs maybe 200 lines of HTML and removes 90% of the "what am I looking at" friction.

### 1.3 Empty states are sparse
`README.md` already mentions the v2.28.4 work on Forecast Accuracy empty-state messaging, but the rest of the UI still has bare empty states. The Predictions, Generalisation, Features, and Covariate Analysis tabs render blank shells until a benchmark completes — no "Run a benchmark to populate this" pointer with a one-click action.

**Standardise an `<empty-state>` partial** with: short headline, one-line explanation, primary action button. Reuse the existing lead-time-card pattern.

### 1.4 No side-by-side model comparison
The Models tab ranks models in a table but you can't pick two and compare them: hyperparameters, residual patterns, per-fold metrics, time-of-day errors. The pieces all exist (`ModelResult.fold_metrics`, `fold_predictions`, `feature_importances`) — just not joined in a view.

**Add a "Compare" toggle** on the Results table that lets the user select 2–3 models and renders a diff view: hyperparameter table side-by-side, overlaid residual histograms, paired Diebold-Mariano p-value (the test is already implemented at `benchmark/comparison.py:60–148` but not surfaced in the UI). This is the single highest-value analysis feature you don't currently have.

### 1.5 No export / reproducibility hooks
There's no "Export benchmark results" button. Users who want to compare runs across sensor histories, share with a friend, or paste into a spreadsheet have to scrape the HTML.

**One-click CSV/JSON export** of the Results table, fold metrics, and the active config. Also: a "Copy config to clipboard" button on the Settings tab so users can back up `mlfl.yaml` from the UI rather than SSH'ing to `/addon_configs/`.

### 1.6 Mobile is functional but unloved
The base template has a hamburger and a couple of `max-width: 600px` rules (`templates/base.html:20–22`, `templates/experiment.html:458–505`), but Results tables have 7+ columns that horizontally overflow on phones and Plotly charts default to 350–500 px min-height. The HA companion app is the natural mobile entry point — make sure the dashboard, model picker, and Forecast Accuracy verdict card all render cleanly at 360 px width.

---

## 2. Helping users understand and improve their models

### 2.1 Interpretability is tree-only
Feature importance comes from `feature_importances_` on the LightGBM/XGBoost/CatBoost backends (`main.py:1910–1930`, `web/app.py:202–206`). Neural backends ship no attributions at all.

Two low-cost wins:
1. **Permutation importance** as a model-agnostic fallback for *any* backend. Implementable in 50 lines: shuffle each feature in the holdout fold, measure MAE delta, rank. Cheap enough to run once per CV split. Works for LSTM, PatchTST, N-BEATS — everything.
2. **Surface the `training_history`** that's already captured for neural models (`runner.py:668–676`). Plot train-vs-val loss per epoch on the Models tab so users can see overfitting / divergence visually rather than inferring it from the Generalisation gap table.

### 2.2 Residuals are computed but not stored or diagnosed
Residuals are reconstructed ad-hoc as `fold_predictions − fold_actuals` (per the runner) and never persisted as a structured artifact. There's no:
- ACF/PACF on residuals (would expose missing seasonality / lag specification)
- Q-Q plot (catches heavy-tailed errors that wreck MAE-based ranking)
- Residual time-series with trend overlay (catches drift the metric averages hide)
- Heteroskedasticity check by time-of-day or by predicted-value bucket

**Add a "Diagnostics" sub-tab under Results** with ACF/PACF (statsmodels has a one-liner), Q-Q against normal, and residual-vs-fitted scatter. These are 50 LOC each in matplotlib/Plotly and they're the standard things a forecaster looks at to decide *what to try next*.

### 2.3 Conformal coverage is published but never validated
Online conformal intervals are well-implemented (`db.py:1020+`, per-lead-bucket quantiles of `|residual|`, fallback for short buckets, pinned to `model_version`). The published bands are 80% — but **the UI never reports actual achieved coverage**. If your bands cover 65% of arrivals instead of 80%, you have a calibration problem and no way to see it.

**Add a calibration tile to the Forecast Accuracy verdict card**: empirical coverage of `_lower_80`/`_upper_80` bands over the last 7/30 days, with a green/amber/red status and a reliability diagram (predicted vs observed coverage at multiple quantile levels). The data is all in `forecast_log`.

### 2.4 Per-horizon error is partially exposed
Forecast Accuracy already has a lead-time error curve (`templates/experiment.html:1238` area), but ranking still uses h=1 only (`runner.py:590–603`). For a 48-step forecast, "best at h=1" can be a different model from "best at h=24."

**Optional ranking modes** on the Models tab: rank by composite at h=1 (current), at mean horizon, at peak-of-day horizon, at user-selected lead. Same Demšar machinery, different input metric.

### 2.5 Statistical tests exist but are hidden
`benchmark/comparison.py:60–148` implements Diebold-Mariano. It is not wired into the web API or the Models tab. As a result, the rank table presents differences that may be statistical noise as if they were meaningful.

**Show a DM p-value column** next to the rank on the Results table (model-vs-leader), and grey out the rank badge when `p > 0.05`. This is a one-day change with outsized epistemic payoff for the user.

### 2.6 No data-quality preview before training
`preprocessing.py` already detects resets, spikes, gaps, and computes `max_increment` heuristics — but the results aren't surfaced. A user with a flaky sensor has no warning until benchmark fails or produces garbage.

**A "Data" tab** on each experiment showing: raw sensor count, NaN ratio, gap histogram, daily-reset detection result, outlier count after clipping, recorder retention vs requested `days_history`. Block "Run Pipeline" with a warning (not a hard block) if any are red. This single tab will prevent a class of "the model is bad" tickets that are actually "the data is bad."

### 2.7 Optuna trial visualisation is shallow
Trials are listed in a table (`web/app.py:168–200`) but there's no parallel-coordinates plot, no per-hyperparameter learning curve (composite-score-vs-trial), no importance-of-hyperparameter inference. Optuna ships these as one-liners (`optuna.visualization.plot_parallel_coordinate`, `plot_param_importances`).

**Add the two Optuna built-in plots** to the Tuning tab. They turn the trial table into something the user can reason about.

---

## 3. Functionality & workflow

### 3.1 Experiments are serial — make the cost legible
Global `_training_lock` (`main.py:158–162`) serialises all training across experiments. This is correct on a Pi-5 — you don't want two PatchTST runs fighting for cores — but the UI never tells the user "your second experiment is waiting because experiment X is benchmarking." The Dashboard already shows next-forecast countdowns; add a **training queue widget** that lists what's running, what's queued, and an ETA derived from past `ModelResult.train_time` values.

### 3.2 No model preset bundles
Closely related to onboarding. Users today either accept the example config or read `docs/MODEL_GUIDE.md` and translate by hand. Three checked-in presets in `config.py` (lightweight / balanced / accurate) plus a UI dropdown removes 80% of "which models should I enable" friction. Map roughly to:
- *Lightweight:* `seasonal_naive`, `lightgbm`, `dlinear` — sub-minute training on Pi-5.
- *Balanced:* + `xgboost`, `nlinear`, `tsmixer`.
- *Accurate:* + `patchtst`, `nbeats`, `tide` — accept ~10–15 min training.

### 3.3 No automatic retrain trigger on accuracy drift
The accuracy tab measures drift; nothing acts on it. Retrain runs on the configured schedule regardless of whether the model is still good. A **drift-triggered retrain** (e.g., if 7-day MAE exceeds CV MAE by 50%, schedule a retrain on the next cycle) closes the production-monitoring loop. The hooks are all there in `db.py` analytics queries.

### 3.4 No experiment cloning
Users running 4 similar experiments (different rooms, different appliances) re-type the YAML. A **"Duplicate experiment"** action in the Dashboard that opens the New Experiment modal pre-populated from the source is ~30 LOC.

### 3.5 HA history fetch is single-shot, not incremental
`ha_interface.py` calls `/api/history/period` for the full `days_history` range every time it's invoked. For a daily retrain on 90 days of history at 30-min interval, that's the same 4000+ rows pulled every day. The SQLite history table already exists (`db.py:67–88` per-entity tables with `ds` unique constraint); the loader could just fetch *new* data since the last stored `ds` and merge. Cuts HA load and retrain wall-time. Low priority but cleanly scoped.

### 3.6 Logs page lacks search
`/log` polls and renders coloured output (`templates/logs.html`). The tags are great (`[BENCH]`, `[MODEL]`, etc.) but a server-side `?filter=BENCH` query param + a text-input would make debugging dramatically easier when something goes wrong inside an 8-fold × 22-model run.

---

## 4. Pi-5 footprint

### 4.1 Plotly is loaded eagerly and is 1.05 MB
`static/plotly-basic.min.js` is the largest single asset and is included on every page via the base template. On a Pi-5 serving over `homeassistant.local` it's not slow, but it's wasted bytes for users on the Dashboard who never open a chart. **Lazy-load Plotly on the first chart-bearing tab** (`Models`, `Predictions`, `Generalisation`, `Tuning`, `Features`, `Forecast Accuracy`) with a 1-line dynamic import.

Alternatively: `uPlot` is ~40 KB and renders faster on ARM. The migration cost is non-trivial (different API), but if you're rebuilding the comparison view from §1.4 anyway, it's worth pricing.

### 4.2 No hardware-aware model gating
Nothing in `config.py` or `models/registry.py` knows it's running on a Pi vs a desktop. A user can `models_enabled: [tft, crossformer, timesnet]` and get a benchmark that takes hours. The preset bundles in §3.2 are the soft fix; a **hardware-aware warning** ("you've enabled 4 transformer backends; estimated training on aarch64 is 47 minutes — continue?") is the harder one. `psutil` + a small lookup table of measured per-backend train times by arch is enough.

### 4.3 Polling cadence is generous
Dashboard polls every 10s during training (acceptable), `/log` every 3s (heavy on a Pi if the page is left open). **Pause polling when the tab is hidden** (`document.visibilityState !== 'visible'`). Single-line JS change, meaningful CPU reduction for the typical "left the tab open all afternoon" user.

### 4.4 No memory ceiling on parallel data fetches
`covariates.py` fetches each covariate sensor via `ha_interface`. With 5+ covariates on a high-frequency sensor over 90 days the in-memory DataFrame can spike. The tuning path is cgroup-aware (`MEM_FLOOR_MB = 128` in the tuning function); the fetch path isn't. Worth adding a budgeted batch loop for the covariate fetcher before users start chaining solar + weather + electricity-price feeds.

---

## Prioritised backlog

### P0 — High impact, low cost (one-day-each changes)
1. Metric tooltips on every header (§1.2).
2. Permutation importance for all backends (§2.1).
3. Conformal coverage tile on Forecast Accuracy (§2.3).
4. Diebold-Mariano p-value column on Results table (§2.5).
5. Three named model presets in `config.py` + Dashboard dropdown (§3.2).
6. Lazy-load Plotly (§4.1) and pause polling on hidden tabs (§4.3).

### P1 — High impact, medium cost (one-week changes)
7. First-run wizard with data-quality preview (§1.1, §2.6).
8. Side-by-side model comparison view (§1.4).
9. Residual diagnostics sub-tab — ACF/PACF, Q-Q, residual scatter (§2.2).
10. Optuna parallel-coords and param-importance plots (§2.7).
11. Training queue widget on Dashboard (§3.1).
12. Drift-triggered retrain (§3.3).

### P2 — Nice to have / longer-tail
13. Per-horizon ranking mode (§2.4).
14. Surface neural `training_history` plots in the UI (§2.1, second bullet).
15. CSV/JSON export and clipboard-copy config (§1.5).
16. Mobile pass at 360 px (§1.6).
17. Incremental HA history fetch (§3.5).
18. Logs filter (§3.6).
19. Hardware-aware warning before enabling heavy backends (§4.2).
20. Experiment cloning (§3.4).

---

## Non-goals (call out so they don't drift in by accident)

- **Don't add SHAP.** It's slow on Pi-5 and permutation importance covers the same intuition for far less compute.
- **Don't add a SPA framework.** The HTMX + Jinja2 setup is correct for this scale; React/Vue would balloon the bundle and the build complexity for a single-user dashboard.
- **Don't add multi-tenant / multi-user features.** This is a single-household add-on; auth, RBAC, and audit trails are scope creep.
- **Don't ship a TensorBoard integration.** Surfacing `training_history` directly in the existing Plotly pane is the right level of polish.

---

## Where each finding is grounded in code

| Section | Key file(s) |
|---|---|
| 1.1 | `main.py:253–270`, `config.py:50–525` |
| 1.2 | `web/templates/experiment.html:517–656` |
| 1.3 | `web/templates/experiment.html:1238` (existing empty-state pattern) |
| 1.4 | `benchmark/comparison.py:60–148`, `web/app.py:168–206` |
| 1.5 | `web/templates/experiment.html` (no export buttons) |
| 2.1 | `main.py:1910–1930`, `runner.py:668–676` |
| 2.2 | `benchmark/runner.py:48–50, 663–665` (predictions stored, residuals not) |
| 2.3 | `db.py:1020–1172`, `main.py:3109–3127` (publish path) |
| 2.4 | `runner.py:590–603, 829–916` |
| 2.5 | `benchmark/comparison.py:60–148` |
| 2.6 | `preprocessing.py:26–246`, `ha_interface.py:192–250` |
| 2.7 | `web/app.py:168–200` |
| 3.1 | `main.py:149–162, 4993+` |
| 3.2 | `config.py:50–525`, `docs/MODEL_GUIDE.md` |
| 3.3 | `db.py:340+` (forecast accuracy queries) |
| 3.5 | `ha_interface.py:192–250`, `db.py:67–134` |
| 4.1 | `web/static/plotly-basic.min.js`, `web/templates/base.html` |
| 4.3 | `web/templates/logs.html:127`, dashboard poll loop |
| 4.4 | `covariates.py:99–156`, `main.py` tuning context (`MEM_FLOOR_MB`) |
