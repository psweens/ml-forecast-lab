# Changelog

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
