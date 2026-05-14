# SURVEY.md — ML methodology recon (Settings-tab preprocessing & loss audit)

Scope: the data-preprocessing and loss-function options exposed under the **Settings** tab of the ml-forecast-lab Home Assistant add-on, plus the surrounding evaluation pipeline that consumes them. British English. File paths are absolute from the repo root. Defaults are quoted from the source on `claude/audit-ha-forecasting-ml-FzGKT` at HEAD.

This file deliberately **judges nothing** — that is Phase 2 (`ML_AUDIT.md`). Findings, severity, proposals come next. The four observations marked **⚠ EVIDENCE** at the bottom are placed here only so the audit can cite the exact code excerpts without re-reading.

---

## 1. The Settings tab — every preprocessing option

All controls live in the Settings tab of the per-experiment page (`experiment.html`, sections "Target", "Data & forecast", "Training", "Solar physics", "Covariates", "Load subtract"). Form values POST to `/api/experiment-settings` (and the two dedicated covariate / load-subtract routes), validate against allowed values, and persist to `mlfl.yaml` via `atomic_yaml_write`. Mapping: each UI field corresponds to a field on `ExperimentCfg` (`ml_forecast_lab/config.py`), `CovariateCfg`, or `SubtractCfg`.

### 1.1 Target semantics

| Field (UI name) | Type | Default | Allowed | Maps to | Consumed at |
|---|---|---|---|---|---|
| `source_is_cumulative` | checkbox | `false` | bool | `ExperimentCfg.source_is_cumulative` | `main.py:1335-1345` → `preprocessing.cumulative_to_interval` |
| `reset_daily` | checkbox | `false` | bool | `ExperimentCfg.reset_daily` | same |
| `max_increment` | number (nullable) | empty → auto | `> 0` or empty | `ExperimentCfg.max_increment` | same; when empty, `preprocessing.py:103-112` uses the 95th-percentile of observed positive diffs |

Form rendering: `ml_forecast_lab/web/templates/experiment.html:147-168`. Handler: `ml_forecast_lab/web/app.py:3209-3221`.

### 1.2 Data window and forecast horizon

| Field | Type | Default | Allowed | Maps to |
|---|---|---|---|---|
| `days_history` | int | `14` | `≥ 1` | `ExperimentCfg.days_history` |
| `interval_minutes` | int | `30` | `≥ 1` | `ExperimentCfg.interval_minutes` |
| `future_periods` | int | `48` (= 24 h at 30 min) | `≥ 1` | `ExperimentCfg.future_periods` |
| `forecast_every_minutes` | int (nullable) | empty → global | `≥ 1` or empty | `ExperimentCfg.forecast_every_minutes` |
| `retrain_every_hours` | float (nullable) | empty → global | `≥ 0.1` or empty | `ExperimentCfg.retrain_every_hours` |

Form rendering: `experiment.html:179-207`. Handler: `app.py:3206-3213`.

### 1.3 Transforms

| Field | Type | Default | Allowed | Maps to | Consumed at |
|---|---|---|---|---|---|
| `log_transform` | checkbox | `false` | bool | `ExperimentCfg.log_transform` | `main.py:1385-1386` → `apply_log_transform(series)` (shift = `1.0`, hardcoded in `preprocessing.py:268-295`) |

Inversion: `main.py:2939-2941` and `:4603-4604` (production forecast publish) — `np.expm1(y_pred)` then `np.maximum(0)`. **Benchmark CV metrics are not inverted** — see §10 and ⚠ EVIDENCE B.

### 1.4 Cross-validation and recency weighting

| Field | Type | Default | Allowed | Maps to | Consumed at |
|---|---|---|---|---|---|
| `cv_strategy` | dropdown | `walk_forward` | `walk_forward`, `sliding_window` | `ExperimentCfg.cv_strategy` | `benchmark/runner.py:227-328` |
| `cv_folds` | int | `5` | `2`–`20` | `ExperimentCfg.cv_folds` | same |
| `recency_half_life_days` | float | `7` | `0`–`365` (0 disables) | `ExperimentCfg.recency_half_life_days` | `benchmark/runner.py:411-418` — exponential sample weights `exp(decay * arange(n))` |

`cv_embargo_periods` (default `2`, `config.py:242`) is **YAML-only**, no UI control.

### 1.5 Loss & training-objective options (the core of this audit)

| Field | Type | Default | Allowed | Maps to | Applies to |
|---|---|---|---|---|---|
| `production_metric` | dropdown | `rmse` | `mae`, `rmse`, `mase` | `ExperimentCfg.production_metric` | model-selection only |
| `loss_fn` | dropdown | `mse` | `mse`, `mae`, `huber` | `ExperimentCfg.loss_fn` | all torch neural backends; tree backends ignore |
| `optimiser` | dropdown | `adamw` | `adamw`, `adam` | `ExperimentCfg.optimiser` | torch backends only |
| `output_activation` | dropdown | `auto` | `auto`, `linear`, `softplus`, `relu`, `exp`, `sigmoid`, `zscore` | `ExperimentCfg.output_activation` | torch backends; resolved per-model in `_apply_output_activation` |
| `daily_loss_weight` | checkbox (boolean) → maps `true` → `0.5`, `false` → `0.0` | `0.0` | float `≥ 0` | `ExperimentCfg.daily_loss_weight` | torch backends; tree backends silently ignore |

Form rendering: `experiment.html:246-286`. Handler: `app.py:3214-3226`. UI deliberately exposes `daily_loss_weight` as a checkbox; the underlying field is a float, so YAML users can set any non-negative value.

Note: **there is no UI control for the tree-model loss/objective** (LightGBM, XGBoost, CatBoost). Their objective is hardcoded — see §6.

### 1.6 Solar physics features

| Field | Type | Default | Maps to | Consumed at |
|---|---|---|---|---|
| `include_sun_elevation` | checkbox | `false` | `ExperimentCfg.include_sun_elevation` | `solar_physics.py` via `main.py` |
| `include_clear_sky_irradiance` | checkbox | `false` | `ExperimentCfg.include_clear_sky_irradiance` | same |

When `include_clear_sky_irradiance=true`, a `clear_sky_ghi` column is materialised in the dataframe — `features.build_features` then **physics-gates the lag features** so a night row sees lag = 0 (`features.py:178-186, 195-202, 215-219`). The gate is enabled by the column's presence, not by a separate flag.

### 1.7 Covariates (per-row controls)

Rendered in `experiment.html:318-373`; added/removed via `/experiment/{name}/add-covariate` and `/remove-covariate`. Each row is a `CovariateCfg`:

| Field | Type | Default | Allowed | UI? |
|---|---|---|---|---|
| `entity` | text (autocomplete) | — | HA entity_id | yes |
| `role` | dropdown | `lagged` | UI exposes only `lagged`; config also allows `future`, `both`, `concurrent` (NaN-only stubs) | yes |
| `aggregation` | dropdown | `mean` | `mean`, `sum`, `max`, `min`, `last` | yes |
| `scale` | number (nullable) | `None` | float or empty | yes |
| `is_binary` | checkbox | `false` | bool | yes |
| `scaling` | — | `None` | `standard`, `minmax`, or `None` | **YAML only** (`config.py:152-200`) |
| `transform` | — | `None` | `log`, `sqrt`, `shifted_log`, or `None` | **YAML only** |

### 1.8 Load subtract (per-row controls)

`experiment.html:383-449`. Each row is a `SubtractCfg`:

| Field | Type | Default | Allowed | UI? |
|---|---|---|---|---|
| `entity_id` | text | — | HA entity_id | yes |
| `source` | dropdown | `auto` | `auto`, `cumulative_daily`, `cumulative_monotonic`, `interval` | yes |
| `on_missing` | dropdown | `zero` | `zero`, `drop`, `error` | yes |
| `scale` | number (nullable) | `None` | float | yes |
| `max_fraction_of_load` | number | `1.0` | `≥ 0` | yes |
| `max_fraction_violation_pct` | — | `5.0` | float | **YAML only** |

Consumed by `preprocessing.apply_load_subtract` (`preprocessing.py:378-704`) before clipping.

### 1.9 What is in the codebase but NOT exposed in the Settings tab

- `cv_embargo_periods` (`config.py:242`, default `2`)
- `metrics` — list of standard metrics to compute (default `["mae", "rmse", "mase"]`)
- `custom_metrics` — dict of Python expressions evaluated via `asteval` (`benchmark/metrics.py:323+`)
- `country` — holiday calendar code (`features.py:228-231`); UI has no field
- `use_revin` (`config.py:307-326`, default `True`) — Reversible Instance Normalisation for torch backends
- `n_lags` (default `12`) and `lag_windows` (default `[6, 24, 72]`) for the feature builder
- The outlier-clipping quantile (`0.995`, hardcoded in `preprocessing.py:218`)
- The log-transform shift (`1.0`, hardcoded in `preprocessing.py:268-295`)
- Per-covariate `scaling` and `transform` (above)
- `future_covariate_features`, `stability_focus`, `clear_forecast_log_on_retrain`, `publish_prefix`, `output_units`, `units`, `country`, `production_model`

---

## 2. Loss functions — mathematical definitions as implemented

### 2.1 Tree backends — hardcoded objectives

| Backend | File | Objective | User can change via UI? |
|---|---|---|---|
| LightGBM | `models/lightgbm_backend.py:191-203` | `"objective": "regression"` (MSE), `"metric": "rmse"` | No |
| XGBoost | `models/xgboost_backend.py:211-224` | `XGBRegressor(...)` defaults (`reg:squarederror`, i.e. MSE) | No |
| CatBoost | `models/catboost_backend.py:116-126` | `"loss_function": "RMSE"` (= sqrt-of-MSE training loss) | No |

There is no UI dropdown for tree-model objectives. `loss_fn` in the experiment Settings is silently ignored by all three.

### 2.2 Neural backends — three-option dropdown

Every torch backend pulls `loss_fn` from its constructor and maps it identically (`models/lstm_backend.py:311-314` and equivalents in cnn, dlinear, nbeats, nhits, tide, tsmixer, sparsetsf, patchtst, itransformer, crossformer, timesnet, gru, nlinear, fits, timemixer, tft):

```python
_loss_map = {
    'mse':   nn.MSELoss,
    'mae':   nn.L1Loss,
    'l1':    nn.L1Loss,
    'huber': nn.SmoothL1Loss,
}
criterion = _loss_map.get(self.loss_fn, nn.MSELoss)(reduction='none')
```

Implemented losses, per torch's documented definitions with `reduction='none'`:

- **MSE** (`nn.MSELoss`): per-sample `(y − ŷ)²`; default reduction would average across the batch+horizon dims. Reduction is performed explicitly downstream (see §2.3).
- **MAE** (`nn.L1Loss`): per-sample `|y − ŷ|`.
- **Huber** (`nn.SmoothL1Loss`): per-sample, with the default `beta = 1.0`:
  `0.5·(y − ŷ)² / β` when `|y − ŷ| < β`, else `|y − ŷ| − 0.5·β`.
  Note: torch's `SmoothL1Loss` and `HuberLoss` differ by a factor of `β` at the quadratic branch; `SmoothL1Loss` is the half-MSE variant.

There is no quantile/pinball training loss, no asymmetric loss, no log-cosh, no Tweedie, no negative-binomial — nothing for over/under-prediction asymmetry or for near-zero-with-occasional-spike series.

### 2.3 Composite horizon loss (`daily_loss_weight`)

`models/base.py:747-842` adds an optional cumulative-trajectory term to the per-step loss:

```
L = L_interval + λ · L_daily
L_interval = criterion(y_pred, y_true)                       # per-interval MSE/MAE/Huber
L_daily    = mean_h[ criterion(cumsum(ŷ)[h], cumsum(y)[h]) ] / H
```

where `λ = self.daily_loss_weight` (default `0.0`), and the cumulative term is only added when `λ > 0` AND `y_pred.dim() == 2 AND y_pred.size(1) > 1`. Sample-weighted means (recency decay) are honoured throughout. Tree backends never enter this code path.

The name is a misnomer: "daily" refers only to the user-facing intent (energy-per-day shape); the term is actually "errors of the cumulative-sum trajectory across the full horizon", scaled by `1/H`.

### 2.4 RevIN (Reversible Instance Normalisation)

`models/base.py:139-254`. **Per-window, per-channel** affine normalisation with stats computed on each input window's first dim (time), stored as `self._mean`, `self._stdev`, then reversed on the prediction tensor. `affine_weight`/`affine_bias` are learnable. Skipped by N-BEATS and N-HiTS (which subtract a backcast instead). Default-on for every other torch backend.

This is **not** a fitted scaler in the sklearn sense — there is no train/test fit step; the network sees raw values and self-normalises on every batch.

### 2.5 Optimiser

`models/base.py:692-744`: `adamw` (default, decoupled L2) and `adam` (tied L2). `weight_decay = 1e-4`. Learning rate is per-backend (typical `1e-3` to `2e-4`) scheduled by `CosineAnnealingLR(T_max=epochs, eta_min=1e-6)`. Gradient clipping `max_norm=5.0` is applied in each backend's `fit`.

### 2.6 Optuna tuning objective

`main.py:4943-5088`. The tuning study minimises a **composite score**, not the training loss:

```
composite = mean([mae / anchor.mae, rmse / anchor.rmse, mase / anchor.mase])
```

Anchor is a baseline model run; lower is better; `direction="minimize"`; sampler TPE (seeded 42); timeout 30 minutes. So a user who picks `loss_fn=mae` is still tuned against a baseline-relative average of three metrics, none of which is `mae` alone.

---

## 3. Evaluation metrics

### 3.1 Registry (`benchmark/metrics.py`)

| Metric | File:line | Definition as coded |
|---|---|---|
| `mae` | `:19-44` | `mean(|y − ŷ|)` over `~isnan` mask; returns `nan` if all NaN |
| `rmse` | `:47-72` | `sqrt(mean((y − ŷ)²))` |
| `mape` | `:75-103` | mean of `|y − ŷ|/|y|` over `y ≠ 0 AND ~isnan`, returned as percent; rows where `y == 0` are **silently skipped** |
| `smape` | `:106-141` | `mean( 2·|y − ŷ| / (|y| + |ŷ|) )` × 100; samples with `|y| + |ŷ| == 0` are skipped |
| `mase` | `:144-188` | `mean(|y − ŷ|) / naive_scale` where `naive_scale = nanmean(|diff(y_train)|)` (1-step naive on the **training fold**); returns `nan` if `naive_scale == 0` or `len(y_train) < 2` |
| `r_squared` | `:191-228` | `1 − SS_res / SS_tot`; returns `nan` if `SS_tot == 0` |
| `pinball_loss` | `:231-278` | `mean( where(e≥0, q·e, (q−1)·e) )` with `q` default `0.5`; **evaluation only** |
| `coverage` | `:281-320` | `mean( (lower ≤ y) AND (y ≤ upper) )`; used for the 80 % conformal band |

### 3.2 Which metrics the Settings tab actually shows the user

Results tab (`experiment.html:672-681`) shows `MAE`, `RMSE`, `MASE`, `Mean Rank`. There is a "Daily" pair of the same three under the daily-totals sub-table. MAPE, sMAPE, R², pinball are computed only when registered as a custom metric or queried via the API; they are not on the default Results tab.

### 3.3 Where metrics are computed

`benchmark/runner.py:596-603`:

```python
metrics_to_compute = list(set(self.metrics + [self.production_metric]))
yt_metric = y_test.ravel() if y_test.ndim == 2 and y_test.shape[1] == 1 else y_test
yp_metric = y_pred.ravel() if y_pred.ndim == 2 and y_pred.shape[1] == 1 else y_pred
yt_train_metric = y_train if y_train.ndim == 1 else y_train[:, 0]
fold_metrics = self.metric_registry.compute_all(
    metrics_to_compute, yt_metric, yp_metric, y_train=yt_train_metric,
)
```

The metric registry receives `y_test` and `y_pred` exactly as they leave the model. **No log-inversion** is performed here — see ⚠ EVIDENCE B.

### 3.4 Demšar composite rank

`benchmark/runner.py:784-829`. Within each CV fold, every model is ranked across MAE / RMSE / MASE; the per-fold composite rank is averaged across metrics; the final `mean_rank` is the mean over folds. Lower = better. This drives the leaderboard.

### 3.5 Seasonal-naive baseline

`main.py:2056-2068` force-includes `seasonal_naive` in every benchmark run so the UI can render a "vs Seasonal Naive" skill chip even when the user didn't enable it. Implementation: `models/seasonal_naive_backend.py` (predicts the value from one seasonal period ago).

### 3.6 Conformal bands (80 % nominal coverage)

`web/app.py:2105` calls `db.get_conformal_quantiles(...)` with a hardcoded coverage of `0.8`, a 14-day residual lookback, and a 10-sample minimum. Bands are built **post-hoc** from logged forecast/actual residuals; the model itself trains only on a point loss.

---

## 4. The full data path: raw HA → metric

```
HA recorder (irregular timestamps)
   │  ha_interface.get_history()      [ha_interface.py:309-342, ~14 days at default]
   │
   ▼
optional SQLite cache (off by default)
   │  HistoryDB.store_history / get_history     [db.py:142-217]
   │
   ▼
cumulative_to_interval(series, interval_minutes, reset_daily, max_increment)
   │  preprocessing.py:26-159          [skipped if source_is_cumulative=false]
   │  - detects resets (negative diff; midnight crosses if reset_daily)
   │  - caps spikes > max_increment (default = 95th-pctile pos-diff)
   │  - drops multi-interval gap rows to NaN
   │
   ▼
resample_to_grid(series, freq=f"{interval_minutes}min",
                 method="sum" if source_is_cumulative else "mean")
   │  preprocessing.py:162-213, called at main.py:1343-1345
   │  - always applies .ffill() then .bfill() after aggregation
   │
   ▼
apply_load_subtract(series, [...])   if exp_cfg.load_subtract non-empty
   │  preprocessing.py:378-704        [main.py:1354-1379]
   │  - per-sensor scale, on_missing policy, max_fraction guard
   │  - result.clip(lower=0)
   │
   ▼
clip_outliers(series, quantile=0.995, positive_only=source_is_cumulative)
   │  preprocessing.py:216-265        [main.py:1382]    [HARDCODED quantile]
   │
   ▼
apply_log_transform(series, shift=1.0)   if exp_cfg.log_transform
   │  preprocessing.py:268-295        [main.py:1385-1386]    [HARDCODED shift]
   │  ★ from here onward, `series` is in log space ★
   │
   ▼
DataFrame{"y": series}; covariates fetched, resampled, ffill/bfill, aligned
   │  main.py:1389-1461  →  covariates.py
   │
   ▼
build_features(df, target_col="y", n_lags=12, lag_windows=[6,24,72], country=...)
   │  features.py:65-237 (the offline feature library)
   │  AND
   │  feature_builder closure at main.py:2009-2027 (the per-fold rebuild used in CV)
   │  - both call target.rolling(w).mean()/.std()/.max()  ← see ⚠ EVIDENCE A
   │
   ▼
combined = features_df; combined["target"] = df["y"]; combined.dropna()
   │  main.py:1985-1993
   │
   ▼
runner._prepare_train_test_splits(combined)
   │  benchmark/runner.py:227-328
   │  - walk_forward (expanding) or sliding_window (fixed) folds
   │  - cv_embargo_periods=2 rows dropped before each test window
   │
   ▼
per-fold: feature_builder(df_train); feature_builder(df_test)        [runner.py:381-396]
   │  X_train, X_test are float32 arrays; y_train, y_test are df['target'].values
   │
   ▼
model.fit(X_train, y_train, sample_weights, ...);  y_pred = model.predict(X_test)
   │  benchmark/runner.py:494-577
   │  - tree models: raw fit on X_train (no scaling)
   │  - neural models: trained on sliding-window tensors built by features.create_sliding_windows
   │                   with RevIN normalising inside the forward pass
   │
   ▼
metric_registry.compute_all(metrics, y_test, y_pred, y_train=y_train)
   │  benchmark/runner.py:596-603
   │  - whatever space y_test / y_pred are in is the space metrics get computed in
   │  - NO log-inversion here    ← see ⚠ EVIDENCE B
   │
   ▼
Demšar mean_rank across folds → leaderboard → production_model selection by production_metric
   │  benchmark/runner.py:784-829, main.py post-benchmark
   │
   ▼ (production forecast cycle, separate code path)
_forecast_with_cached → predict → expm1 inversion (main.py:2939-2941, :4603-4604)
   │  ★ only the production publish path inverts log_transform ★
   │
   ▼
conformal band built from logged residuals (web/app.py:2105, db.get_conformal_quantiles)
publish_state(...) → HA forecast sensors
```

---

## 5. Train / validation / test split mechanism

There is no separate validation split — the benchmark uses **k-fold time-series CV** with no inner validation set. Early stopping (when present, e.g. LightGBM `early_stopping_rounds=50`) is configured against the train fold itself or a slice of it carved out inside the backend; there is no global `val_idx`.

**Split strategy (chosen by `cv_strategy`):**

- `walk_forward` — expanding train window, fixed test size. Fold *f* (0-indexed):
  - `test_size = max(1, n // (n_folds + 1))`
  - `test_start = n − test_size · (n_folds − f)`
  - `train_end = max(0, test_start − cv_embargo_periods)`
  - `train_idx = arange(0, train_end)`; `test_idx = arange(test_start, test_start + test_size)`
- `sliding_window` — fixed train and test sizes, sliding stride sized so the last fold's test ends at `n`.

Both are strictly temporal — `train_idx < test_idx` for every fold; no shuffling. (`benchmark/runner.py:227-328`.)

**Embargo:** `cv_embargo_periods = 2` rows are excluded between train and test specifically to prevent rolling/lag features that span the boundary from leaking. The comment at `:251-253` says this is for "rolling features whose forecast horizon overlaps the test inputs". The embargo is **not the same** as the rolling-feature leak inside the train fold — see ⚠ EVIDENCE A.

**Where scaling/normalisation is fit:**

- Tree backends: nothing is fit; trees consume raw values.
- Neural backends with `use_revin=True` (default): **per-window** stats are computed inside the network's forward pass; no train-vs-test fit step exists. Each input window auto-normalises against itself.
- Neural backends with `output_activation="zscore"` (LSTM auto-default; other backends treat as `linear`): per-horizon mean/std are computed from the **training fold's targets** in the backend's `fit`, then used to denormalise at inference. This is fit on train only.
- Per-covariate `scaling` (`standard`, `minmax`) is YAML-only and applied **on the full series before splitting**, per `covariates.py`. That is a leak point under YAML usage — see ⚠ EVIDENCE C.

---

## 6. Missing data, irregular sampling, outliers — what the user controls

| Concern | Behaviour | User control |
|---|---|---|
| Irregular HA timestamps | Always resampled to a fixed `interval_minutes` grid with `mean` (or `sum` when cumulative). `ffill` then `bfill` applied unconditionally after aggregation. | `interval_minutes` is in UI; the aggregation method and the post-aggregation fill are **hardcoded** (`preprocessing.py:175-178`). |
| Multi-interval gaps in cumulative series | `cumulative_to_interval` writes `NaN` for rows whose preceding gap is > 1.5 × interval, so resample treats them as missing rather than synthesising a single inflated bucket. | Not directly controllable, but `max_increment` shapes spike detection. |
| Gaps after resampling | Forward-fill, then back-fill leading NaNs. | No control over fill method (no drop / no linear interpolation / no NaN mask). |
| Negative-diff / counter-reset spikes | If `reset_daily=true`, midnight-cross + small/negative diff is treated as a reset and the cumulative value is used as the interval. Otherwise, any negative diff is treated as a reset. Spikes `> max_increment` and `not multi_interval_gap` are clipped. | `source_is_cumulative`, `reset_daily`, `max_increment` cover the common cases. |
| Outliers post-resample | Symmetric quantile clip at **q=0.995** (or `[0, q]` for cumulative). Hardcoded. | None — the quantile is not in the UI or the YAML schema. |
| Sensor spikes for non-cumulative targets | Same hardcoded 0.5 % top / 0.5 % bottom clip applies. No robust scaler, no MAD/IQR clip, no Hampel filter. | None. |
| Load-subtract gaps | `on_missing` in `{zero, drop, error}` per subtract row. `max_fraction_of_load` fails fast at 5 % violation (YAML-only override of the 5 %). | `on_missing`, `scale`, `max_fraction_of_load` in UI. |
| Train/test boundary leakage from ffill | `cv_embargo_periods=2` rows are dropped between train and test, partly mitigating rolling-feature spillover. Forward-fill at the boundary is not separately guarded — if a value is ffilled across a multi-row gap that straddles the boundary, that value is shared by train and test. | Indirect, via `cv_embargo_periods` (YAML only). |

There is no per-sensor or per-experiment imputation policy (drop/ffill/interpolate/mask); ffill is the only behaviour and it is global.

---

## 7. Feature engineering — what is built, and from what

Source of truth at benchmark time is the closure `feature_builder` at `main.py:2009-2027`, plus the temporal/lag features carried over from `features.build_features` at preprocessing time (`features.py:65-237`).

Features generated (per fold, on the fold's local target):

| Family | How it is computed | Look-ahead-safe? |
|---|---|---|
| Calendar | `hour_of_day`, `day_of_week`, `is_weekend`, `month`, `day_of_month`, plus `hour_sin/cos`, `dow_sin/cos`. Deterministic from `DatetimeIndex`. | Yes |
| Holiday | `is_holiday`, when `ExperimentCfg.country` is set; via the `holidays` library. | Yes (calendar-deterministic) |
| Per-step lags | `y_lag_k = target.shift(k)` for `k ∈ 1..n_lags` (default `n_lags=12`). Physics-gated to 0 at night when `clear_sky_ghi` is present. | Yes — shift(k) at row t reads target[t-k] only. |
| Periodic lags | `y_lag_{steps_per_day}`, `y_lag_{2·steps_per_day}` — "same time yesterday/two days ago". Also physics-gated. | Yes |
| Rolling statistics | `y_rolling_mean_{w}`, `y_rolling_std_{w}`, `y_rolling_max_{w}` for `w ∈ [6, 24, 72]` (hardcoded). Re-computed per fold from the fold-local target. | **No — see ⚠ EVIDENCE A.** |
| Rate of change | `y_diff_1 = target.shift(1) − target.shift(2)`. Gated by `clear_sky_ghi.shift(1)` if solar. | Yes |
| Covariate × time-of-day interactions | `{cov}_x_hour_sin`, `{cov}_x_hour_cos` for every covariate column, using the **current-row** covariate value. | Yes for the `lagged` role (the covariate column is already a shifted/aligned input); ambiguous for `concurrent`/`future` roles, which are not currently functional. |

User exposure: covariate list and physics flags are in the UI; everything else (`n_lags`, `lag_windows`, country, future-covariate names, RevIN) is YAML-only. There is no UI control for which lag horizons, no toggle for rolling statistics, no exogenous-sensor-as-future-known feature beyond the TiDE `future_covariate_features` YAML stub.

For neural backends: `features.create_sliding_windows` (`features.py:444-549`) builds `(n, window_size, n_channels)` tensors from raw values (target + covariates + temporal sin/cos channels). Lag and rolling **feature columns are not consumed by neural backends** — those models look at the window directly. The leakage path in ⚠ EVIDENCE A therefore affects tree backends only.

---

## 8. Scaling, normalisation, and inverse transforms

| Path | Scaler / normaliser | Fit on | Where it is applied |
|---|---|---|---|
| Tree backends | none | n/a | raw `X_train, y_train` |
| Neural backends (default, `use_revin=True`) | RevIN — per-window, per-channel affine normalisation | self-fits on each input window | inside `model.forward`; `denormalize` in the same forward |
| Neural backends with `output_activation="zscore"` and `use_revin=False` | per-horizon `(μ, σ)` of `y_train` | train fold targets only | model output head; reversed at inference |
| Covariate `scaling: standard/minmax` | sklearn-style fit | **full covariate series** before any split | `covariates.py` — fit is global, not per-fold |
| Log transform | analytical `log(y + 1)` → `exp(y) − 1` | n/a | applied to the target series before the DataFrame is built (`main.py:1385-1386`); inverted only on the **production publish** path (`main.py:2939-2941`, `:4603-4604`) — **not** before benchmark metrics. |

The model itself, the validation metric, and the user-visible Results-tab numbers are computed in whatever space the data is in when `metric_registry.compute_all` is called. See ⚠ EVIDENCE B.

---

## 9. Hardcoded vs user-configurable

| Item | Status | Default | Configurable via |
|---|---|---|---|
| Sampling frequency | configurable | 30 min | UI |
| History length | configurable | 14 d | UI |
| Forecast horizon | configurable | 48 × 30 min = 24 h | UI |
| Aggregation method (sum vs mean) | hardcoded | implied by `source_is_cumulative` | none |
| Gap-fill method | hardcoded | `ffill` then `bfill` | none |
| Outlier-clip quantile | hardcoded | `0.995` | none |
| Outlier-clip method | hardcoded | symmetric quantile | none |
| Log-transform shift | hardcoded | `1.0` | none |
| `n_lags` | hardcoded | `12` | YAML only (not in UI; would need a schema field) |
| `lag_windows` | hardcoded | `[6, 24, 72]` | YAML only |
| `cv_embargo_periods` | configurable | `2` | YAML only |
| RevIN on/off | configurable | `True` | YAML only |
| Conformal coverage | hardcoded | `0.80` | none |
| Conformal lookback | hardcoded | 14 d | none |
| Tree-model objective | hardcoded | per backend (MSE / RMSE) | none |
| Neural `loss_fn` | configurable | `mse` | UI |
| `daily_loss_weight` | configurable | `0.0` | UI (checkbox → `0.5`) |
| `production_metric` | configurable | `rmse` | UI |
| `cv_strategy`, `cv_folds`, `recency_half_life_days` | configurable | `walk_forward`, `5`, `7` | UI |
| `optimiser`, `output_activation` | configurable | `adamw`, `auto` | UI |
| `production_metric` choices | hardcoded list | `{mae, rmse, mase}` | none (MAPE/sMAPE/R²/pinball not selectable) |
| Tuning objective | hardcoded | composite of MAE/RMSE/MASE | none |

---

## 10. Four pieces of evidence that the audit will lean on

### ⚠ EVIDENCE A — Rolling features include the prediction target at training time (tree-backend leakage + train/inference skew)

`ml_forecast_lab/main.py:2009-2027` (the per-fold feature builder used by `BenchmarkRunner`):

```python
def feature_builder(df_sub, config, purpose="train"):
    df_out = df_sub.copy()
    target = df_out["target"]
    for window in rolling_windows:                # [6, 24, 72]
        df_out[f"y_rolling_mean_{window}"]  = target.rolling(window=window).mean()
        df_out[f"y_rolling_std_{window}"]   = target.rolling(window=window).std()
        df_out[f"y_rolling_max_{window}"]   = target.rolling(window=window).max()
    for d in [1, 2]:
        lag_steps = steps_per_day * d
        if lag_steps <= len(target):
            df_out[f"y_lag_{lag_steps}"] = target.shift(lag_steps)
    df_out["y_diff_1"] = target.shift(1) - target.shift(2)
    cols = [c for c in df_out.columns if c != "target"]
    X = df_out[cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    return X
```

The same pattern is in the offline library at `ml_forecast_lab/features.py:190-193`:

```python
for window in lag_windows:
    features[f'y_rolling_mean_{window}'] = target.rolling(window=window).mean()
    features[f'y_rolling_std_{window}']  = target.rolling(window=window).std()
    features[f'y_rolling_max_{window}']  = target.rolling(window=window).max()
```

`pandas.Series.rolling(window=w).mean()` is right-closed by default — at row *t* the window spans `[t−w+1, t]` **inclusive of row t**. So at training-fold row *t* the feature `y_rolling_mean_6[t]` is `mean(target[t−5], ..., target[t])`, which **contains the value the model is being asked to predict**. The lag feature row `y_lag_1[t] = target[t−1]` and `y_diff_1[t] = target[t−1] − target[t−2]` are properly shifted; only the rolling family is not.

The mismatch becomes a hard train/inference skew at the production recursive forecast (`main.py:2874-2884`):

```python
for w in [6, 24, 72]:
    window = buf[-w:] if len(buf) >= w else buf
    if window:
        row[f'y_rolling_mean_{w}'] = float(np.mean(window))
        row[f'y_rolling_std_{w}']  = float(np.std(window))
        row[f'y_rolling_max_{w}']  = float(np.max(window))
```

At inference, `buf` is populated with values up to and including step *k−1* before predicting step *k*. So inference computes rolling stats over the `w` values **strictly before** the prediction step, while training computed them over `w` values **up to and including** the prediction step. The feature distribution at training is not the feature distribution at inference.

Consequences (to be judged in `ML_AUDIT.md`):
- Tree-backend CV metrics overstate accuracy because the model can recover `target[t]` from `(y_rolling_mean_6[t] · 6 − Σ y_lag_{1..5}[t])` exactly.
- Production accuracy differs from CV accuracy by an amount that depends on how much weight the tree puts on the rolling features (which is usually substantial for power/load forecasting).
- Neural backends are unaffected — they consume the sliding window directly, not these feature columns.

Embargo (`cv_embargo_periods=2`) does not prevent this leak: it protects the test set from rolling spillover *across the fold boundary*, not the train rows from referencing their own target.

### ⚠ EVIDENCE B — Benchmark metrics are computed in log space when `log_transform=true`

`ml_forecast_lab/main.py:1383-1389`:

```python
# --- Optional log transform ---
if exp_cfg.log_transform:
    series = apply_log_transform(series)

# --- Build DataFrame ---
result = pd.DataFrame({"y": series}, index=series.index)
```

Downstream, `combined["target"] = df["y"]` is in log space (`main.py:1985-1986`), and the benchmark runner reads `y_test = df_test["target"].values` (`benchmark/runner.py:400-401`). At the metric call (`benchmark/runner.py:596-603`):

```python
metrics_to_compute = list(set(self.metrics + [self.production_metric]))
yt_metric = y_test.ravel() if y_test.ndim == 2 and y_test.shape[1] == 1 else y_test
yp_metric = y_pred.ravel() if y_pred.ndim == 2 and y_pred.shape[1] == 1 else y_pred
fold_metrics = self.metric_registry.compute_all(
    metrics_to_compute, yt_metric, yp_metric, y_train=yt_train_metric,
)
```

A `grep -n "log\|expm1\|invert" benchmark/runner.py` returns no expm1 / `invert_log_transform` call. The log inversion exists in the runtime production path only (`main.py:2939-2941`, `:4603-4604`, `:2492-2496` for the holdout chart).

Consequence (to be judged in `ML_AUDIT.md`):
- "RMSE: 0.42" on the Results tab is in `log(y+1)` units when `log_transform=true`, not in the sensor's units. Users have no indication of this.
- `production_metric=rmse` selects the model with lowest log-space RMSE, which corresponds to a geometric/relative error penalty — different model than the one that minimises absolute-error in original units.
- The chart on the Results tab (which un-logs for display) and the leaderboard numbers (which do not) are not in the same units.

### ⚠ EVIDENCE C — Covariate `scaling` is fit on the full series, before splitting

`covariates.py` applies the per-covariate `scaling` (`standard` or `minmax`) to the **full fetched series** before it is concatenated with the target and split into folds. There is no per-fold refit. The field is YAML-only; users who set `scaling: standard` on a covariate get test-set statistics leaking into their training features by construction.

The leak is statistically small for most covariates but is a textbook split-leakage example that contradicts the per-fold-rolling fix already present in the rest of the pipeline.

### ⚠ EVIDENCE D — No probabilistic training loss; only point losses; quantile/pinball is evaluation-only; conformal coverage is hardcoded 80 %

`models/base.py:747-842` (composite horizon loss) and the `_loss_map` in every neural backend offer only `mse`, `mae`, and `huber`. `benchmark/metrics.pinball_loss` exists but no training path uses it; `coverage` is computed against bands built post-hoc by `db.get_conformal_quantiles(..., 0.8, ...)` (`web/app.py:2105`). There is no UI control for coverage level, no quantile dropdown, no asymmetric over/under-prediction loss, no Tweedie/NB for spiky-near-zero series.

This is not a bug — it's a capability gap and it shapes what the user can express. The "will my battery be empty by 6pm?" use case is, today, served by a 50 % point forecast plus a fixed-coverage interval; users cannot ask for a 90 % upper band, cannot penalise under-prediction more than over-prediction, and cannot train a model with calibrated quantile output.

---

## 11. End of Phase 1

Per the brief, this file is recon only. The judgements — what to add, what to change, what to remove, what defaults to flip — are in Phase 2 (`ML_AUDIT.md`), produced after you confirm this survey is accurate.
