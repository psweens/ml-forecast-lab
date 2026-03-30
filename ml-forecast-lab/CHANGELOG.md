# Changelog

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
