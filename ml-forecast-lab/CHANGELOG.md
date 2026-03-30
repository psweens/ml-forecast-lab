# Changelog

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
