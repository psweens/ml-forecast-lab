# Changelog

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
