# METHOD_SURVEY — ML Forecast Lab

Repo root: `ml-forecast-lab/`. Code lives in `ml-forecast-lab/ml_forecast_lab/`. Paths below are relative to that package directory unless otherwise stated. All citations are file:line. This is descriptive, not judgemental — the audit document follows separately.

---

## 1. Settings — every ML-relevant configurable option

All experiment-level ML behaviour is configured in `ExperimentCfg` (`config.py:200-653`). App-level orchestration in `AppConfig` (`config.py:657-700`). Values are loaded from YAML (canonical path `/addon_configs/ml_forecast_lab/mlfl.yaml`) by `load_config` (`config.py:703-905`).

### Data window & cadence

| Field | Default | Range / values | Consumed at |
|---|---|---|---|
| `days_history` | 14 | ≥1 | `main.py:1349` — sets `start = now - timedelta(days=...)` for HA history fetch |
| `interval_minutes` | 30 | ≥1 | `main.py:1350` (`freq=f"{...}min"`), `features.py:142,210`, `runner.py:611-612` (season for MASE) |
| `max_age` | 365 | days | SQLite cache retention (`db.py` / `main.py:1369`) |
| `future_periods` | 48 | ≥1 | `main.py:464` (dense horizons), `main.py:4669` (inference horizon) |
| `forecast_every_minutes` | 30 | ≥1 (AppConfig) | `main.py:3957-3968` forecast cycle |
| `retrain_every_hours` | 24.0 | ≥0.1 (AppConfig) | retrain cycle |

### Source semantics & cleaning

| Field | Default | Consumed at |
|---|---|---|
| `source_is_cumulative` | False | `main.py:1511-1519` (drives `cumulative_to_interval` and resample method `'sum'` vs `'mean'`) |
| `reset_daily` | False | passed to `cumulative_to_interval` (`main.py:1513`) |
| `max_increment` | None | spike cap; estimated from 95th percentile of positive diffs when None (`preprocessing.py:104-112`) |
| `target_is_nonnegative` | False | drives `output_activation='auto'` resolution in `_apply_output_activation` |
| `gap_handling` | `'interpolate'` | `{ffill, interpolate, mask}`; `preprocessing.py:198-221` |
| `gap_max_minutes` | 90 | cap for linear interpolation |
| `outlier_method` | `'quantile'` | `{quantile, mad, off}`; `preprocessing.py:224-289` |
| `outlier_quantile` | 0.999 | upper-tail trim quantile |
| `outlier_lower` | `'auto'` | `{auto, zero, symmetric, off}` |
| `load_subtract` | `[]` | `apply_load_subtract` invoked at `main.py:1546-1548` |

### Target transform

| Field | Default | Consumed at |
|---|---|---|
| `log_transform` | False | `main.py:1570-1571` — `apply_log_transform` invoked on `series` BEFORE the dataframe is constructed |
| `output_activation` | `'auto'` | `{auto, linear, softplus, relu, exp, sigmoid, zscore}` — `base.py:34-92` `_build_activation`, applied per backend |
| `use_revin` | True | Reversible Instance Normalization (Kim et al. 2022) — `base.py:139-278` `_RevIN`; consumed by all PyTorch backends except N-BEATS / N-HiTS |
| `quantiles` | `[]` | `(0,1)` floats; native quantile head; currently only DLinear (`config.py:520-529`, `dlinear_backend.py:151-339`) |
| `conformal_coverage` | 0.8 | nominal coverage of post-hoc conformal band (`main.py:4096`) |

### Training-loop options

| Field | Default | Consumed at |
|---|---|---|
| `loss_fn` | `'huber'` | `{mse, mae, huber, tweedie}`; passed via `m.set_params(loss_fn=…)` at `main.py:2280-2287` |
| `optimiser` | `'adamw'` | `{adam, adamw}`; `base.py:741-794` `_build_optimiser`; passed to neural backends at `main.py:2291-2293` |
| `daily_loss_weight` | 0.0 | λ for cumulative-trajectory loss term; `base.py:796-892` `_composite_horizon_loss` |
| `recency_half_life_days` | 0.0 | exponential sample-weight half-life; `runner.py:411-419` |

### Cross-validation

| Field | Default | Consumed at |
|---|---|---|
| `cv_strategy` | `'walk_forward'` | `{walk_forward, sliding_window}`; `runner.py:258-322` `_prepare_train_test_splits` |
| `cv_folds` | 5 | bounded [2, 20] (`config.py:611-621`) |
| `cv_embargo_periods` | 2 | gap excluded before each test slice; `runner.py:256, 267-272` |

### Models & per-model overrides

| Field | Default | Consumed at |
|---|---|---|
| `models_enabled` | `['lightgbm','xgboost','lstm','cnn']` | `main.py:2255-2310` instantiates each from registry |
| `production_model` | None (auto-select) | `main.py:2092-2096` falls back to rank-1 winner |
| `selected_model` | None | UI display only |
| `production_metric` | `'seasonal_mase'` | drives `BenchmarkRunner` ranking and Optuna objective (`main.py:5466-5497`, `runner.py:218`) |
| `metrics` | `['mae','rmse','mase','seasonal_mase']` | passed to `MetricRegistry.compute_all` (`runner.py:614-617`) |
| `model_params` | `{}` | per-experiment hyperparameter overrides; merged with global `model_overrides` at `main.py:2273-2302` |
| `custom_metrics` | None | Python expressions over `y_true`, `y_pred` |

### Covariate / physics

| Field | Default | Consumed at |
|---|---|---|
| `covariates` | `[]` | `CovariateCfg`; fetched at `main.py:1595-1617` |
| `country` | None | holiday flag (`features.py:29-62, 242-244`) |
| `future_covariate_features` | `[]` | TiDE temporal decoder routing (`tide_backend.py`) |
| `include_sun_elevation` | False | adds deterministic solar elevation column (`main.py:1658-1666`) |
| `include_clear_sky_irradiance` | False | adds pvlib Ineichen GHI; also gates lag features in features.py |

### Per-covariate (`CovariateCfg` — config.py:146-196)

| Field | Default | Values |
|---|---|---|
| `role` | `'lagged'` | `{lagged, future, both, concurrent}` — never enforced at inference scoring (see §4) |
| `scale` | None | optional multiplier |
| `transform` | None | `{log, sqrt, box_cox, None}` |
| `aggregation` | `'mean'` | `{mean, sum, max, min, last}` for resampling |
| `is_binary` | False | drives ffill vs mean in `covariates.py:90-97` |
| `future_attribute` | `'forecast'` | HA attribute to pull future values from |
| `future_value_key` | None | inner key when attribute is a list-of-dicts |

---

## 2. Data path — raw HA history → fit

End-to-end pipeline lives in `main.py::_fetch_and_preprocess` (`main.py:1331-1700+`) and is consumed by both `_run_benchmark` (`main.py:2123+`) and `_run_production_inference` (called per cycle).

```
HA history (REST)
  → SQLite cache merge (db.py / main.py:1369-1427)
  → cumulative_to_interval         (main.py:1511-1516; preprocessing.py:26-159)
  → resample_to_grid               (main.py:1520-1524; preprocessing.py:162-221)
  → apply_load_subtract            (main.py:1546-1548; preprocessing.py:401-727)
  → clip_outliers                  (main.py:1561-1567; preprocessing.py:224-289)
  → apply_log_transform (optional) (main.py:1570-1571; preprocessing.py:292-318)
  → covariate fetch + reindex(ffill→bfill)   (main.py:1595-1617)
  → solar physics columns           (main.py:1658-1666; solar_physics.py)
  → pd.DataFrame `df` with 'y' and covariate columns

  → build_features(df, 'y', country=…)       (main.py:2162-2167; features.py:65-251)
        adds hour_sin/cos, dow_sin/cos, is_weekend, month, day_of_month,
        is_holiday, y_lag_1..12 (n_lags=12 hardcoded),
        y_rolling_{mean,std,max}_{6,24,72},
        periodic lags y_lag_{steps_per_day}, y_lag_{2*steps_per_day},
        y_diff_1, covariate × hour_sin / hour_cos interactions
        Lag/rolling features are all .shift(1) before window → past-only.

  → `combined = features_df.copy(); combined['target']=df['y']; for cov: combined[cov]=df[cov]`
  → combined.dropna()              (main.py:2178)

  → fold_indices = runner._prepare_train_test_splits(combined)   (main.py:2323)
  → for each fold:
        feature_builder(df_train, …) and feature_builder(df_test, …)
                                       (runner.py:382-387; defined main.py:2200-2221)
            re-computes rolling stats and periodic-day lags on the fold's slice
            (so rolling means do not see across fold boundaries),
            but lags y_lag_1..12 carry their globally-computed values.
        recency_half_life_days → sample_weights vector             (runner.py:411-419)
        if neural: create_sliding_windows(…, horizon_steps=range(1,future_periods+1))
                                       (runner.py:449-490; features.py:555-712)
        model.fit(X_train, y_train, feature_names, sample_weight, …)
        model.predict / model.predict_sequence on test fold
        log-inverse + metric compute                (runner.py:606-617)
```

### Missing-data handling

- `cumulative_to_interval` drops rows that span >1.5× expected interval to NaN (`preprocessing.py:147-155`).
- `resample_to_grid` chooses one of three policies per `gap_handling`:
  - `interpolate` (default): linear, capped at `gap_max_minutes`, then bfill leading NaNs.
  - `ffill`: legacy, propagate last value across any gap.
  - `mask`: leave NaN and let downstream dropna remove.
- Covariates are reindexed onto target's index with `ffill().bfill()` (`main.py:1614-1616`).
- After feature build, `combined.dropna()` removes warmup rows where lags or rolling stats are still NaN (`main.py:2178`).

### Recency-weighting

`runner.py:411-419` — exponential time-decay sample weights with half-life from `recency_half_life_days`. Default 0.0 (uniform). Older config (`recency_half_life_days=7`) gave older rows ~25 % weight; the audited default was changed to uniform.

---

## 3. Splitting — temporal? rolling? where do scalers live?

**Splits are strictly temporal.** `runner._prepare_train_test_splits` (`runner.py:227-328`):

- `walk_forward` (default): expanding train window, fixed test slice of size `n // (cv_folds+1)`, with the final fold's test ending at `n_samples`. Embargo (`cv_embargo_periods`, default 2) excluded immediately before each test slice.
- `sliding_window`: fixed train and test sizes; stride chosen so final fold's test ends at `n_samples`.
- Both honour `cv_embargo_periods` via `train_end = test_start - embargo`.

There is no train/val/test 3-way split inside the benchmark. Each model's `fit()` carves its own **tail validation** for early stopping. E.g. LSTM (`lstm_backend.py:297`), CNN (`cnn_backend.py:296`), DLinear (`dlinear_backend.py:296`), NLinear, all use `_tail_val_split(n_total, val_split, gap=n_horizons)` (`base.py:699-721`) — last 20 % of training rows as val, with a purge gap equal to the forecast horizon. LightGBM uses an unpurged 80/20 tail split (`lightgbm_backend.py:168`); XGBoost (`xgboost_backend.py:164`) and CatBoost (`catboost_backend.py:113`) likewise.

### Scalers / encoders

- **No global `StandardScaler` / `MinMaxScaler` is fit.** Per-fold or per-model standardisation comes from:
  - RevIN (`base.py:139-278`) — per-window per-channel instance normalisation, fit on-the-fly inside each forward pass. Reversible: `denormalize` runs at the head.
  - `output_activation='zscore'` (target-only): training-data target mean/std stored on the model; reversed at inference (per backend, e.g. `lstm_backend`).
  - Tree models receive raw features (no scaling).
- Holiday lookup is computed on the fly (`features.py:29-62`).
- Log-transform is fit globally (`apply_log_transform`, `main.py:1570-1571`) BEFORE the dataframe is split. The "fit" here is data-dependent only via `min(series)` for the additive shift (default 1.0 for non-negative series). Inverted with `np.expm1` at metric / inference time.

---

## 4. Regularisation — every mechanism

### Across pipeline

- `cv_embargo_periods` — temporal purge gap before each test fold (`runner.py:256-272`).
- `_tail_val_split(…, gap=n_horizons)` — per-backend purge between train and internal val (`base.py:699-721`).
- `outlier_method` clip — robust to extreme spikes (`preprocessing.py:224-289`).
- `loss_fn='huber'` default — bounded influence on tail residuals (`config.py:455-462`).
- Composite cumulative-trajectory loss when `daily_loss_weight>0` (`base.py:796-892`).
- Conformal post-hoc residual quantiles for prediction intervals (`main.py:4076-4173`, `db.py:1370-1382`).

### Tree backends (LightGBM / XGBoost / CatBoost)

- L1 `reg_alpha`, L2 `reg_lambda` (LightGBM/XGBoost defaults 0.5 / 1.0; CatBoost `l2_leaf_reg=3.0`).
- `subsample` (row bagging) ~0.7-0.8, `colsample_bytree`/`colsample_bylevel` ~0.8.
- `min_child_samples` / `min_data_in_leaf` ~10-25.
- Early stopping `patience=50` rounds on internal tail-val (`lightgbm_backend.py:168, 251`; `xgboost_backend.py:164, 232`; `catboost_backend.py:113, 203`).
- Sample weights propagated.

### Neural backends (PyTorch)

- Dropout (typ. 0.1-0.2 — LSTM 0.1, CNN 0.15, DLinear/PatchTST/TSMixer/TiDE 0.2, GRU 0.1, NHiTS/NBeats 0). Hardcoded defaults per backend; not tuned by default.
- `weight_decay=1e-4` baked into `_build_optimiser` (`base.py:741-794`). AdamW gives decoupled decay; Adam gives tied decay. Not tunable from the schema.
- Gradient clipping `max_norm=5.0` everywhere (LSTM:360, GRU:289, CNN:357, DLinear:363, etc.).
- Early stopping on the internal tail-val split, patience=20 (`base.py:_tail_val_split` plus per-backend `patience` defaults; e.g. LSTM:169).
- CosineAnnealingLR scheduler with `eta_min=1e-6` for all neural backends.
- RevIN — per-window per-channel normalisation (`base.py:139-278`) — N-BEATS/N-HiTS opt out architecturally.
- L1 / explicit prior / data augmentation: not used.
- Ensembling: no cross-model ensembling. Composite ranking is done at evaluation, not at prediction.

### Classical / baseline

- `seasonal_naive` — no learnable parameters; `seasonal_period=48` hardcoded default (`seasonal_naive_backend.py:46`), settable via constructor.
- `ARIMA` (auto-ARIMA) — search grid capped (`max_p=2, max_q=2, max_P=1, max_Q=1, nmodels=10`, `statsforecast_backend.py:340-346`) and training history capped to `4*seasonal_period` (line 128).
- `ETS` — locked to additive seasonality (`model='ZZA'`, line 368).
- `Theta` — no regularisation; decompositional.

---

## 5. Covariate analysis

**Entry point.** `_run_covariate_analysis` (`main.py:5693+`); UI endpoints `POST /experiment/{name}/run-covariate-analysis` (`web/app.py:2673`) and `POST /experiment/{name}/apply-covariate-best` (`web/app.py:1393`).

**Method.** Model-based **leave-one-covariate-out (LOCO) retraining**. Three configurations are constructed (`main.py:5766-5775`):
1. "All covariates" (baseline).
2. "No covariates".
3. One "Without {covariate}" per covariate.

For each configuration × each enabled model, the model is **retrained from scratch** on a single 80 / 20 chronological split (`main.py:5816`), then MAE / RMSE / MASE are computed on the 20 % tail. % change vs the "All covariates" baseline is computed and reported. No correlation, mutual information, permutation importance over a fitted model, SHAP, or VIF is used.

**Data on which scoring is done.** `df_full = _fetch_and_preprocess(exp_cfg)` (`main.py:5751`) — the full series. `features_base = build_features(df_full, …)` is computed once on the full series (`main.py:5757-5761`); per-configuration `combined = features_base.copy()` then prepends or removes specific covariate columns and `combined.dropna()` is applied; split is `split = int(len(combined) * 0.8)` (`main.py:5816`).

The 80/20 split is chronological (`combined.iloc[:split]` vs `combined.iloc[split:]`) — not random.

**Lag / autocorrelation handling.** None. No lagged-correlation, CCF, autocorrelation control, or partial correlation is computed anywhere in the analysis (the only correlation reported anywhere is a per-covariate Pearson against target in the diagnostic log at `main.py:1829-1845`, used to print a "noise" flag only).

**Role differentiation.** `CovariateCfg.role` is set per covariate (`config.py:152`) and recorded in the dataframe metadata (`main.py:1628`), but is **not enforced** during the LOCO scoring — `covariate_cols = [c for c in df_full.columns if c != "y"]` (`main.py:5754`) treats every covariate uniformly. A `role='future'` covariate is scored using its historical values from `_fetch_and_preprocess`, not a simulated "value known at forecast time".

**Multicollinearity.** Not detected, surfaced, or controlled. No VIF, condition number, or pairwise correlation matrix.

**Recommendations engine.** Heuristic thresholds on MAE % change (`main.py:5986-6090`):
- "Covariates help {model}" if removing-all increases MAE > 5 %.
- "Better without covariates" if removing-all decreases MAE > 3 %.
- Per-covariate: "Keep X" if dropping X increases error > 3 % across all models; "Remove X" if dropping X decreases error > 2 % across all models; majority voting otherwise.
- No recommendation to ADD a covariate the user has not configured.

**Best-set selection.** Greedy single-covariate removal — picks the configuration with the lowest average MAE across all tested models (`main.py:6041-6090`). No multi-covariate subset search.

**UI surface.** `experiment.html:1422-1433` — table of MAE / RMSE / MASE plus % change vs baseline; recommendations panel with icon + variant ("good" / "warning" / "bad" / "info").

---

## 6. Hyperparameter tuning

**Entry point.** `_run_tuning(experiment, model, n_trials, strategy, param_schema)` (`main.py:5070+`). UI: `POST /experiment/{name}/tune` and `apply-tuning` (`web/app.py:2849-2911`).

**Search method.** Optuna study, `direction="minimize"`. Sampler: `TPESampler(seed=42)` when `strategy="tpe"` (default), else `RandomSampler(seed=42)` (`main.py:5534-5539`). **No pruner.** Trials run to completion or until memory pressure aborts them.

**Tuning budget / stopping rule.**
- `n_trials=30` default (user-changeable via UI).
- `study_timeout = 30 * 60` seconds (30-min wall-clock) (`main.py:5540`). `study.optimize(objective, n_trials=n_trials, timeout=study_timeout)`.
- Memory-floor abort (`main.py:5414-5420`): if cgroup-aware available RAM drops below 256 MB, `trial.study.stop()` and the trial returns `inf`.
- Neural trials capped at `epochs=30`, `patience=6`, `batch_size=16` during tuning (`main.py:5242-5256`). Production retrain after Apply uses the backend's full default budget.

**Search space.** Per-model schema in `web/app.py:543-749` (the `TUNING_SCHEMA` dict). Each parameter declares `{"type": int|float|select, "default", "min", "max"}`; `learning_rate`, `reg_alpha`, `reg_lambda` are flagged log-uniform; categorical for `loss_fn`. `batch_size` and `loss_fn` are marked `"tunable": False` for neural models so the search doesn't burn trials on them. Twenty backends are tunable; `seasonal_naive`, `arima`, `ets`, `theta` are not.

**Objective.** The user-selected `production_metric` is what Optuna minimises (`main.py:5466-5497`):
```python
primary = result.metrics.get(exp_cfg.production_metric,
                              result.metrics.get("mae", float("inf")))
…
return composite                 # = primary if finite, else inf
```
MAE / RMSE / MASE are always recorded per trial; the rank-based final-winner selection is computed across MAE / RMSE / MASE composite rank (`main.py:5573`) after Optuna terminates.

**Validation protocol used to score candidates.**
- `exp_cfg_dict["cv_folds"] = 1` forced inside `_run_tuning` (`main.py:5170`). The same `BenchmarkRunner._prepare_train_test_splits` produces a SINGLE walk-forward (or sliding-window) fold per trial.
- Neural sliding windows are pre-computed once per fold and reused across trials (`main.py:5189-5230`).
- The **single tuning fold's score is what the leaderboard updates with `tuning_state.best_score`** during the run (`main.py:5510-5513`).

**Honest holdout re-estimation.** After the search ends, `main.py:5590-5688` runs a separate 80/20 chronological split:
```python
split_80 = int(len(combined) * 0.8)
X_tr_h, X_te_h = X_all[:split_80], X_all[split_80:]
…
preds_default = _run_holdout(default_overrides)
preds_tuned   = _run_holdout(best_params)
…
tuning_state.tuned_mae = round(tuned_mae, 6)
tuning_state.best_score = round(tuned_mae, 6)        # overwritten with holdout MAE
```
The 20% tail of this holdout overlaps in time with the test slice of the single-fold CV that was used to drive Optuna (walk_forward 1-fold puts the test slice in the tail half of the series).

**Apply-tuning.** `web/app.py:2849-2911`:
1. Persist `best_params` into `experiment.model_params[model_name]` via `save_experiment_model_params` (`config.py:943-982`).
2. Switch experiment to `mode="production"`.
3. Spawn `retrain_callback` — which calls `_retrain_and_cache` and uses the experiment's full CV schedule and the backend's full default epoch budget.

---

## 7. Inference

### Trigger and cache

`_run_production_inference` runs every `forecast_every_minutes`; `_retrain_and_cache` runs every `retrain_every_hours`. Between retrains, `_forecast_with_cached` (`main.py:4618+`) reuses the cached model weights and rebuilds features from freshly-fetched data each cycle.

### Multi-step strategy

**Neural backends.** Direct (one-shot) multi-output head:
```python
multi_pred = model.predict_sequence(last_window)        # main.py:4787
y_pred = multi_pred[:future_periods].astype(np.float32)
```
Training builds dense horizons `horizon_steps = list(range(1, future_periods+1))` (`runner.py:463-464`, `main.py:5204`) — each sample's target is a vector of length `future_periods`. `predict_sequence` returns the full vector.

**Tree backends.** **Recursive.** `main.py:5003-5037` loops over each horizon step, building one feature row per step via `_build_feature_row`. After each step, the prediction is appended to `lag_buffer` and used as `y_lag_1` for the next step (`main.py:4952-4957`). Error accumulates with horizon depth. A **physics gate** (`main.py:5015-5033`) overrides the lag with 0 when `clear_sky_ghi <= 0` at that timestamp, mirroring the training-time gate in `features.py:188-195`.

### Inverse transforms

- `log_transform`: `np.expm1` then `np.maximum(0, …)` at `main.py:5042-5045`.
- Per-window RevIN `denormalize` runs inside each backend's forward pass.
- `output_activation='zscore'`: denormalisation happens inside the backend's predict path using cached `(mean, std)`.
- `output_units`: stored only in HA sensor's `unit_of_measurement` attribute (`main.py:4039, 4273-4274`); no runtime unit conversion in the inference path — the user is expected to align `units` and `output_units` upstream.

### Uncertainty / prediction intervals

**Adaptive split conformal on residuals from the production forecast log** (`main.py:4076-4173`, `db.py:1202-1392`).

- Residuals are `|forecast - actual|` collected from `forecast_log` for the last 14 days (`db.py:1209` default) bucketed by **lead time** in minutes.
- Coverage level `level = exp_cfg.conformal_coverage` (default 0.8) → quantile `q = 1 - α/2 = 0.9` (`db.py:1370-1382`).
- `y_pred_upper = y_pred + q_vec`, `y_pred_lower = y_pred - q_vec` (`main.py:4158-4163`). Symmetric ± band.
- Cold-start fallback (`main.py:4122-4141`): if the current model_version has < 10 residuals at a given lead, pool across all model_versions.
- Achieved (empirical) coverage is computed by lead bucket and surfaced on the Results tab (`web/app.py:2135-2146`, `db.py:1395-1550`).

**Native quantile head** (alternative to conformal). When `exp_cfg.quantiles` is non-empty, DLinear (`dlinear_backend.py:151-339`) replaces its point loss with a pinball loss across the requested quantiles and outputs a `(batch, H, Q)` tensor. Other backends ignore the flag and continue with conformal wrapping (`main.py:2297-2300`).

### Publish path

`_publish_forecast_sensors` (`main.py:3979+`). HA sensor state = `str(next_val)` (first forecast point). Attributes carry:
```python
main_attrs = {
    "forecast": forecast_list,       # [{datetime, value}, …]
    "forecast_upper": upper_list,    # if intervals available
    "forecast_lower": lower_list,
    "interval_level": interval_level,
    "model": model_name,
    "last_trained": last_trained_iso,
    "issued_at": issued_at_iso,
    …
}
```

### Stationarity / assumption checking

No explicit stationarity test or visualisation. A **drift diagnostic** (`main.py:2331-2349+`) computes mean / std shift and a PSI on deciles between earliest-fold train and last-fold test windows and exposes the verdict to the UI.

---

## 8. Models available — assumptions per model

Registered in `main.py:391-422`. Family from `ForecastModel.model_family` (`base.py:434-449`).

### Tree (`family='tree'`)
- **LightGBM** (`lightgbm_backend.py`) — additive boosted trees, no temporal structure assumed; relies on lag/rolling features to encode time. Recursive multi-step at inference. Tunable, sample-weight-aware.
- **XGBoost** (`xgboost_backend.py`) — as above; `tree_method='hist'` default.
- **CatBoost** (`catboost_backend.py`) — oblivious trees; `bootstrap_type='Bernoulli'` for subsample support.

### Neural (`family='neural'`) — all PyTorch, all `use_revin=True` by default
- **LSTM, GRU, CNN (WaveNet-style causal)** — local recurrent / convolutional. Past-only mask used in extended-window mode (`base.py:280-304`).
- **DLinear / NLinear** (Zeng et al. 2023) — assume additive trend+seasonal decomposition. NLinear uses last-value anchor.
- **N-BEATS, N-HiTS** (Oreshkin 2020, Challu 2023) — doubly-residual backcast/forecast stacking; opt out of RevIN. NHiTS multi-rate pooling.
- **TiDE** (Das et al. 2023) — encoder/decoder MLP with temporal decoder for known-future covariates (`future_covariate_features`).
- **TSMixer / TimeMixer** (Chen 2023) — alternating time/feature mixing; assume decomposability.
- **SparseTSF** (Lin 2024) — sparse channel routing.
- **PatchTST** (Nie 2023) — channel-independent patch transformer.
- **iTransformer** (Liu 2024) — variate-attention transformer.
- **Crossformer** (Zhang & Yan 2023) — cross-variate transformer.
- **TimesNet** (Wu 2023) — 2D-period-folded vision backbone.
- **TFT** (Lim & Arik 2021) — variable-selection + multi-head attention + LSTM encoder/decoder.
- **FITS** (frequency-domain low-pass) — assumes periodic structure; non-causal complex-linear head.

### Classical / Baseline
- **seasonal_naive** (`family='baseline'`) — period m=48 default; assumes stationary seasonality.
- **ARIMA / ETS / Theta** (`statsforecast_backend.py`, `family='classical'`) — auto-ARIMA capped at `max_p=2, max_q=2`; ETS locked to additive `model='ZZA'`; Theta decompositional. Falls back to seasonal-naive on degenerate (constant/zero) series (`statsforecast_backend.py:205`).

All implicitly assume **temporal exchangeability of windows** during training. None test stationarity. Neural models implicitly assume sliding windows are i.i.d. samples from the data-generating distribution.

---

## 9. Evaluation — metrics, baselines, space

### Metrics (`benchmark/metrics.py`)
- **MAE** (l.19-44), **RMSE** (l.47-72): standard, NaN-safe.
- **MAPE** (l.75-103): skips rows where `y_true == 0` (`np.abs(y_true) > 0` mask). Returned as percent.
- **SMAPE** (l.106-141): `2|y - ŷ| / (|y| + |ŷ|) × 100`. Handles 0/0.
- **MASE** (l.144-188): denominator = `mean(|y_train[t] - y_train[t-1]|)` (1-step naive).
- **seasonal_MASE** (l.191-229): denominator = `mean(|y_train[season:] - y_train[:-season]|)` (m-step naive). Falls back to 1-step naive when `len(y_train) <= season`. Season passed by the runner as `1440 // interval_minutes` (`runner.py:611-612`) — i.e. 48 at 30-min sampling (daily cycle).
- **r_squared** (l.232-269), **pinball_loss** (l.272-319), **coverage** (l.322-361). All registered (`metrics.py:377-388`).

### Space of evaluation
- `runner.py:606-617`: inverts `log_transform` via `np.expm1` for `yt_metric, yp_metric, yt_train_metric` BEFORE computing metrics. So metrics are in the sensor's transform-original space.
- `output_units` is not applied here — metrics are in the post-`log` original target space, not in any UI-display unit.

### Baselines
- **Seasonal Naive is always force-included** in the benchmark roster (`main.py:2255-2262`) and its MAE captured as `naive_baseline_mae` for the leaderboard "vs Seasonal Naive" skill chip (`main.py:2054-2073`). Hidden from the rank table when the user didn't explicitly enable it (l.2062-2063).
- **No plain 1-step naive baseline** is included as a separate model — but the MASE denominator IS the 1-step naive forecast error on training data (`metrics.py:180`).
- **Composite ranking** (`runner.py:852-939`): Demšar (2006) per-metric ranks averaged across folds. Tie-breaking by integer rank. Used to populate `result.rankings` and `result.best_model`.
- **Diebold-Mariano paired tests + Model Confidence Set** (`benchmark/comparison.py:60-240`) — Newey-West HAC variance, Bonferroni correction. Output stored on `WebBenchmarkResult.pairwise_dm`.

### Per-fold variance
- `BenchmarkResult.to_dataframe` (`runner.py:132-166`) reports per-metric `mean` and `std` across folds. UI shows mean ± std.
- Train metrics also computed per fold (`runner.py:619-637`) to surface train-vs-test gap.
- Daily-cumulative metrics computed in parallel (`runner.py:639-679`) for use cases where the daily total matters more than per-interval precision.

---

## 10. Hardcoded vs user-controlled — at every stage

| Stage | Hardcoded | User-controlled |
|---|---|---|
| HA fetch window | — | `days_history`, `interval_minutes` |
| Cumulative conversion | spike cap = 95th percentile of positive diffs when unset (`preprocessing.py:107`) | `source_is_cumulative`, `reset_daily`, `max_increment` |
| Resample method | `'sum'` if cumulative else `'mean'` (`main.py:1519`) | `gap_handling`, `gap_max_minutes` |
| Outlier clip | MAD scale `k=3.5` (`preprocessing.py:269`) | `outlier_method`, `outlier_quantile`, `outlier_lower` |
| Log transform shift | `1.0` for non-negative, `|min|+1` otherwise (`preprocessing.py:307-312`) | `log_transform` (bool only) |
| Lags | `n_lags = 12` hardcoded everywhere (`main.py:2941, 4669`, `features.py:69`) | None |
| Rolling windows | `[3·s, 12·s, 36·s]` where `s = max(1, 60//interval_minutes)` (`features.py:142-147`) | None directly |
| Periodic lags | `1×, 2× steps_per_day` (`features.py:210-216`) | None |
| Interaction features | every covariate × `hour_sin / hour_cos` (`features.py:236-239`) | Implicit — user adds covariate, gets interaction |
| Sample weight half-life | — | `recency_half_life_days` (default 0 = uniform) |
| CV strategy | walk-forward default; sliding-window alternative | `cv_strategy`, `cv_folds`, `cv_embargo_periods` |
| Per-backend val-split | tail 20 % with `gap=n_horizons` purge | None |
| Early stopping | tree patience = 50, neural patience = 20 | None (capped to `patience=6` during tuning) |
| Optimiser weight decay | `1e-4` hardcoded (`base.py:744-746`) | None — schema excludes from tuning |
| Gradient clip | `max_norm=5.0` everywhere | None |
| Dropout | hardcoded default per backend (0.0-0.2) | tunable in schema for most neural backends |
| Tuning sampler | `seed=42` | `strategy` ('tpe'/'random'), `n_trials` |
| Tuning fold count | `cv_folds=1` forced (`main.py:5170`) | None |
| Tuning neural budget | `epochs=30, patience=6, batch_size=16` | None |
| Tuning wall-clock cap | 30 min | None |
| Post-tune holdout split | `int(len(combined)*0.8)` | None |
| Covariate analysis split | `int(len*0.8)` chronological | None |
| Recommendation thresholds | 5 % / 3 % / 2 % | None |
| Conformal residual window | 14 days, min 10 per lead bucket | `conformal_coverage` (level) |
| Conformal bound shape | symmetric ± `q_vec` (`main.py:4158-4163`) | None — choose `quantiles=[…]` to switch to native pinball head (DLinear only) |
| MASE season | `1440 // interval_minutes` (`runner.py:611-612`) | `interval_minutes` |
| Seasonal-naive period | `seasonal_period=48` default (`seasonal_naive_backend.py:46`) | constructor override only |
| ARIMA grid caps | `max_p=2, max_q=2, max_P=1, max_Q=1, nmodels=10` | None |
| ETS family | `'ZZA'` (additive) | None |
| `physics gate` for lag features | active whenever `clear_sky_ghi` column present | `include_clear_sky_irradiance` |

---

## Quick-reference file map

| Module | Purpose |
|---|---|
| `config.py` | `ExperimentCfg`, `CovariateCfg`, `SubtractCfg`, `AppConfig`; YAML load/save |
| `preprocessing.py` | cumulative→interval, resample, outlier, log/sqrt, load-subtract |
| `features.py` | `build_features`, `prepare_train_test`, `create_sliding_windows`, `build_inference_window`, `compute_known_future_features` |
| `covariates.py` | `CovariateResolver.fetch_history` / `fetch_future` |
| `solar_physics.py` | pvlib Ineichen GHI + elevation |
| `benchmark/runner.py` | `BenchmarkRunner` — CV splits, per-fold metrics, composite ranking |
| `benchmark/metrics.py` | `MetricRegistry` with mae/rmse/mape/smape/mase/seasonal_mase/r²/pinball/coverage |
| `benchmark/comparison.py` | Diebold-Mariano, Model Confidence Set |
| `models/base.py` | `ForecastModel` ABC, `_RevIN`, `_build_optimiser`, `_composite_horizon_loss`, `_tail_val_split` |
| `models/registry.py` | Registry / factory |
| `models/*_backend.py` | 21 concrete backends |
| `main.py` | Orchestrator — `_fetch_and_preprocess` (data path), `_run_benchmark`, `_run_covariate_analysis`, `_run_tuning`, `_retrain_and_cache`, `_forecast_with_cached`, `_publish_forecast_sensors`, conformal residual application |
| `web/app.py` | FastAPI surface, `TUNING_SCHEMA` (per-model param spaces), `apply-tuning` / `apply-covariate-best` endpoints |
| `db.py` | SQLite history cache, `forecast_log`, conformal quantile computation, coverage analytics |

---

This survey is descriptive only. The methodological judgement (ERROR / SIMPLIFICATION / UNJUSTIFIED) is in METHOD_AUDIT.md, produced after your sign-off.
