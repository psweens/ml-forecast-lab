# ML_AUDIT.md — methodology findings (Phase 2)

Scope: judgement of the preprocessing and loss-function options exposed under the Settings tab of the ml-forecast-lab Home Assistant add-on. Built on `SURVEY.md` (Phase 1, this branch). British English. Proposals respect the Pi 5 envelope (8 GB RAM, ARM64, no CUDA).

This is a methodology audit, not a code-correctness audit; correctness bugs are flagged where they materially affect forecast quality. Each finding states *what to add / change / remove*, the *sensible default* for a non-expert HA power user, and a *severity × confidence* judgement. No edits in this PR.

---

## Top 5 biggest wins (impact × confidence / effort)

These are the items that, in our judgement, will move forecast quality the most for the most users. Ordered.

1. **Stop leaking the prediction target into tree-backend rolling features** (Finding `C-1`). One-line fix; restores CV honesty for LightGBM / XGBoost / CatBoost — the three most-used backends on this add-on. *Critical / high confidence / S.*
2. **Invert log-transform before computing benchmark metrics** (Finding `C-2`). Four-line fix in `benchmark/runner.py`. Today the leaderboard shows log-space numbers when `log_transform=true`; users select a different model under log-space ranking than under original-units ranking. *Critical / high confidence / S.*
3. **Switch the displayed MASE to a seasonal MASE (sMASE) using `lag = steps_per_day`** (Finding `H-1`). The current 1-step MASE on a daily-seasonal series is essentially "MAE divided by yesterday's noise"; sMASE divides by the seasonal-naive MAE, which is the comparison HA users actually care about (and matches the "vs Seasonal Naive" skill chip the UI already shows). *High / high confidence / S.*
4. **Expose outlier-handling as a real choice instead of a hardcoded `q=0.995` symmetric clip** (Finding `H-2`). The clip method most domain users want is "robust ceiling on the top tail, no floor clip" or "MAD-based clip". Today everyone gets the same fixed bound, with no UI or YAML escape hatch. *High / high confidence / S.*
5. **Add a probabilistic-forecast pathway: quantile (pinball) training loss + a UI control for conformal coverage level** (Finding `H-3`). The "will my battery be empty by 6pm?" use case is the single most-requested HA forecasting question; today the answer is a fixed-0.8 conformal band wrapped around a point forecast. *High / medium-high confidence / M.*

The remaining findings (D-1 through M-7, plus L-1/L-2) materially improve the addon but are smaller wins per unit of effort.

---

## Critical

> Silent leakage, metrics computed in the wrong space, or a default that systematically degrades the model the user is about to deploy.

### C-1 — Rolling-feature target leakage at training (and train/inference distribution skew)

- **Area:** Feature engineering / splitting & leakage
- **Evidence:** `ml_forecast_lab/main.py:2009-2027`, `ml_forecast_lab/features.py:190-193`.

  ```python
  # main.py:2014  (per-fold feature builder used by BenchmarkRunner)
  for window in rolling_windows:
      df_out[f"y_rolling_mean_{window}"] = target.rolling(window=window).mean()
      df_out[f"y_rolling_std_{window}"]  = target.rolling(window=window).std()
      df_out[f"y_rolling_max_{window}"]  = target.rolling(window=window).max()
  ```

  Pandas `rolling(w).mean()` is right-closed by default; at row *t* the window spans `[t-w+1, t]` **inclusive of t**, so the training feature row at *t* contains `target[t]`. Lags one row above are properly shifted (`target.shift(lag)`) but rolling is not. The production recursive forecast uses `buf[-w:]` (strictly past) at `main.py:2874-2884`, so inference computes the rolling statistic over a window that *does not include* the prediction step.
- **Why it harms quality:** Tree backends (LightGBM, XGBoost, CatBoost) consume these feature columns directly. They can recover `target[t]` analytically from `(y_rolling_mean_6[t] · 6) − Σ y_lag_{1..5}[t]`, so they learn to rely on the rolling feature for the answer. Two consequences: (i) CV MAE / RMSE / MASE for tree models are biased downwards, so the Demšar composite leaderboard systematically over-ranks tree backends; (ii) at production time the model sees a *different* feature distribution (rolling excludes step *t*), so deployed accuracy is worse than the benchmark advertised — a textbook train/serve skew. Neural backends are unaffected because they consume the sliding window directly via `features.create_sliding_windows`, not these column features.
- **Proposed change:** Shift the target by one step before rolling, in **both** `main.py:2014-2016` and `features.py:191-193`:
  ```python
  shifted = target.shift(1)
  for w in rolling_windows:
      df_out[f"y_rolling_mean_{w}"] = shifted.rolling(window=w).mean()
      df_out[f"y_rolling_std_{w}"]  = shifted.rolling(window=w).std()
      df_out[f"y_rolling_max_{w}"]  = shifted.rolling(window=w).max()
  ```
  This makes the training-time feature match the inference-time `buf[-w:]` semantics — both become "rolling statistic of the *w* values immediately preceding the prediction step". Add a `tests/unit/test_features_no_leak.py` that asserts `feature_row(t)` is independent of `target[t]` for every column.
- **Expected effect:** Tree-backend CV metrics will *worsen* on the leaderboard (the previously-bogus advantage vanishes), but production accuracy will *not* worsen — it stays where it is because production never had the leak. Net result: the leaderboard tells the truth, the auto-promoted production model is chosen on real merit, the conformal coverage holds the level the user asked for. The shape of the relative ranking between tree and neural backends may change materially.
- **Pi 5 cost:** None. Same number of features.
- **Effort:** S.
- **Severity:** critical.
- **Confidence:** high — verified by direct reading of both code paths.

### C-2 — Benchmark metrics computed in log space when `log_transform=true`

- **Area:** Scaling / evaluation
- **Evidence:** `ml_forecast_lab/main.py:1385-1386` applies `apply_log_transform(series)` *before* the dataframe is built; `ml_forecast_lab/benchmark/runner.py:596-603` calls `metric_registry.compute_all(metrics, y_test, y_pred, ...)` with no `np.expm1` / `invert_log_transform` step anywhere upstream:

  ```python
  # main.py:1385-1386
  if exp_cfg.log_transform:
      series = apply_log_transform(series)         # ★ from here onward, log space

  # benchmark/runner.py:596-603
  fold_metrics = self.metric_registry.compute_all(
      metrics_to_compute, yt_metric, yp_metric, y_train=yt_train_metric,
  )
  ```

  The production publish path *does* invert (`main.py:2939-2941`, `:4603-4604`), so the holdout chart and the published HA forecast sensor are in original units while the leaderboard is in log units.
- **Why it harms quality:** Three failures at once:
  1. The user-visible number is wrong. "RMSE: 0.42" on the Results tab looks like 0.42 kWh; it is actually a log-of-`(y+1)` RMSE. There is no way for the user to know.
  2. `production_metric=rmse` selects a different model under log-space ranking than under original-units ranking. Log-space RMSE penalises proportional error; original-units RMSE penalises absolute error. On a power series with a few large peaks, the log-space ranking favours the model that gets the small values right and lets peaks blow out; the user wanted the opposite.
  3. The "vs Seasonal Naive" skill chip and the Demšar composite rank inherit the same bug, so the chip is misleading in the same direction.
- **Proposed change:** Inverse-transform `y_test` and `y_pred` to the original units immediately before the metric registry call. The cleanest place is in `BenchmarkRunner.run_single_model` just above line 596:
  ```python
  if self.experiment_cfg.get('log_transform'):
      yt_metric = np.expm1(yt_metric)
      yp_metric = np.expm1(yp_metric)
      yt_train_metric = np.expm1(yt_train_metric)
  ```
  (or thread `invert_log_transform` through). Apply the same inversion before computing the daily-cumulative metrics block. The model continues to *train* on log-space targets — that is fine and stays the same — but it is *scored* in the units the user understands.
- **Expected effect:** Leaderboard numbers become interpretable. Auto-selected production model under `production_metric=rmse` may change. The "vs Seasonal Naive" chip becomes meaningful for log-trained models. User trust improves materially.
- **Pi 5 cost:** None.
- **Effort:** S.
- **Severity:** critical.
- **Confidence:** high — `grep -n "log\|expm1\|invert" benchmark/runner.py` returns no inversion call.

### C-3 — Per-covariate `scaling` is fit on the full series, before splitting

- **Area:** Splitting & leakage (covariates)
- **Evidence:** `CovariateCfg.scaling ∈ {standard, minmax, None}` (YAML only, no UI control — `config.py:152-200`). The covariate is fetched, resampled, and scaled in `covariates.py` against statistics computed on the **full fetched series** *before* it is concatenated with the target and split into folds (`main.py:1399-1461`).
- **Why it harms quality:** Test-window statistics leak into the training feature distribution by construction. This is the same class of bug as the rolling-feature leak (C-1) but smaller in magnitude and gated behind a YAML field most users don't know exists. The pipeline goes to considerable trouble to recompute rolling stats per fold (see the comment at `main.py:2004`); covariate scaling silently violates that contract.
- **Proposed change:** Either (a) move covariate scaling into the per-fold feature builder so its statistics fit on the training fold only; (b) remove the YAML field and rely on RevIN's per-window normalisation, which is what neural backends already use; or (c) document that `scaling` is leaky and replace it with a non-leaky `transform: log / sqrt / shifted_log` (which is already half-implemented in `preprocessing.apply_transform`). Recommended: option (b) — drop the field, let RevIN handle scale variance for neural backends, and let tree backends consume raw covariates (they are scale-invariant).
- **Expected effect:** Closes one leakage path; simplifies the schema by one field. For users who rely on the field today (likely few — YAML-only), behaviour changes slightly but neural backends will still self-normalise.
- **Pi 5 cost:** None.
- **Effort:** S.
- **Severity:** critical (by class — leakage) — but lower priority than C-1 / C-2 because adoption of this YAML field is presumably low.
- **Confidence:** medium-high (assumes the user-base does not heavily rely on the YAML field; would need a quick search of community configs to confirm).

---

## High

> Missing capability or a default that materially caps achievable accuracy for the typical HA target sensor.

### H-1 — MASE uses 1-step naive scale; should be seasonal MASE for HA sensors

- **Area:** Evaluation / metrics
- **Evidence:** `ml_forecast_lab/benchmark/metrics.py:176-188`:
  ```python
  naive_errors = np.abs(np.diff(y_train))
  naive_scale = np.nanmean(naive_errors)
  ...
  return float(np.mean(abs_errors) / naive_scale)
  ```
  This is MASE against the *1-step* naive forecast (`ŷ_t = y_{t-1}`).
- **Why it harms quality / misleads:** Most HA targets (power, occupancy, temperature, humidity, energy-per-day, water flow) have a dominant **daily** seasonality. The honest naive baseline for a daily-seasonal series is "same time yesterday", not "value one slot ago". A 1-step naive on a daily-seasonal series has a very small scale (consecutive 30-min values are highly correlated), so MASE values are systematically biased *upward* (away from 1.0) and the metric loses its "MASE < 1 ⇔ beats the baseline you actually care about" interpretation. Worse, this is **inconsistent** with the rest of the addon: `main.py:2056-2068` force-includes a `seasonal_naive` model as the baseline for the UI's skill chip — so the Results tab shows a chip computed against seasonal naive while the MASE column is computed against 1-step naive. Two different baselines, no way for the user to know.
- **Proposed change:** Add a `seasonal_mase(y_true, y_pred, y_train, season)` metric that uses `naive_scale = nanmean(|y_train[season:] − y_train[:-season]|)` with `season = steps_per_day` from `interval_minutes`. Make `seasonal_mase` the default on the Results tab and in the production-metric dropdown; keep the 1-step `mase` available for users who explicitly want it. Update the Demšar composite to use the seasonal version. Fall back to 1-step when `len(y_train) < 2 · season`.
- **Expected effect:** MASE numbers become interpretable on the user's terms. Demšar ranking may change order for some experiments. The "vs Seasonal Naive" skill chip and the MASE column finally tell the same story.
- **Pi 5 cost:** None.
- **Effort:** S.
- **Severity:** high.
- **Confidence:** high.

### H-2 — Outlier handling is a hardcoded 0.995 symmetric clip with no UI or YAML override

- **Area:** Outliers & robustness
- **Evidence:** `ml_forecast_lab/main.py:1382`:
  ```python
  series = clip_outliers(series, positive_only=exp_cfg.source_is_cumulative)
  ```
  `clip_outliers` (`preprocessing.py:216-265`) defaults `quantile=0.995`. There is no `clip_quantile`, `clip_method`, or `clip_off` field anywhere in `ExperimentCfg`. The user cannot turn the clip off, cannot change the quantile, cannot pick a robust (MAD/IQR) method, and cannot pick "upper-only" clipping for a non-cumulative target.
- **Why it harms quality / misleads:**
  - For near-zero series with occasional legitimate spikes (boiler kW when the kettle goes on, peak EV charge draw, peak rainfall in a storm), a fixed 0.995 cap throws away exactly the values the user most wants the model to predict.
  - For non-cumulative two-sided sensors (temperature deltas, wind direction encoded as a signed feature), the symmetric `[0.005, 0.995]` clip is wrong on the lower tail.
  - For sensors where MSE training is destabilised by spikes, the clip is in the wrong place anyway — it should be paired with a robust loss (Huber/MAE/Tweedie), and the user has no control over either.
  - The quantile is **not** in YAML — even power users have no escape hatch.
- **Proposed change:** Introduce a small "Robustness" subsection under the Settings tab with three controls:
  - `outlier_handling`: dropdown `{off, quantile, mad}` — default `quantile` (matches today's behaviour).
  - `outlier_quantile`: float, default `0.999` (less aggressive than today; HA sensor noise rarely needs a 0.5 % top trim).
  - `outlier_lower`: dropdown `{auto, zero, symmetric, off}` — `auto` reproduces today's `positive_only=source_is_cumulative` logic.
  Also expose `outlier_quantile` in YAML for power users.
  Internally, swap the call to a robust path when `mad` is selected: `clip(median ± k·MAD)` with `k = 3.5` (Iglewicz-Hoaglin). Cheap; ARM64-safe.
- **Expected effect:** Users with spiky sensors (rainfall, occupancy, intermittent loads) can stop the clip from amputating their training data. Users with truly noisy sensors gain a robust alternative to the quantile. Default behaviour is unchanged unless the user opts in.
- **Pi 5 cost:** Negligible.
- **Effort:** S–M (UI form + handler + 20 LOC in `preprocessing.py`).
- **Severity:** high.
- **Confidence:** high.

### H-3 — No probabilistic training loss; conformal bands are hardcoded post-hoc at 0.80

- **Area:** Loss functions
- **Evidence:** `_loss_map` in every neural backend (e.g. `models/lstm_backend.py:311-314`) offers only `mse / mae / huber`. `benchmark/metrics.pinball_loss` (`:231-278`) exists but is referenced only as an evaluation metric — no training path uses it. Conformal-band coverage is hardcoded to `0.8` at `web/app.py:2105`, with a 14-day residual lookback and a 10-sample minimum; there is no UI or YAML control.
- **Why it harms quality:**
  - The single most-asked HA forecasting question is probabilistic — "will my battery be empty by 18:00?", "will the freezer go above −15 °C tonight?", "is there a 90 % chance the immersion will run for at least an hour?". A point forecast plus a fixed-coverage interval cannot answer those at any other coverage level.
  - Post-hoc conformal calibration assumes residual exchangeability between calibration and test windows. HA sensors regularly violate this (heatwaves, school holidays, EV fleet changes, daylight-saving step). The result is over- or under-coverage that the user cannot diagnose.
  - For asymmetric-cost decisions (e.g. an over-prediction of solar generation costs you a battery cycle; an under-prediction costs you nothing) a *quantile* objective is the right answer; today neither the loss nor the metric supports it.
- **Proposed change:** Two parts:
  1. **Quantile training path.** In `models/base.py:_loss_map`, add a `pinball` option that consumes a `quantiles: list[float]` config field (default `[0.1, 0.5, 0.9]`) and a multi-quantile output head. Implement for **one** representative neural backend first — TiDE or DLinear are the smallest and cover the lab-vs-production split well. Pinball is a one-line criterion (`torch.maximum(q*e, (q-1)*e).mean(axis over horizon × quantiles)`). The output head adds `len(quantiles)` × `horizon` parameters per channel — negligible.
  2. **Conformal-coverage UI control.** Add a Settings-tab slider `conformal_coverage` ∈ {0.5, 0.8, 0.9, 0.95}, default `0.8` (preserves today's behaviour). Pass through to `db.get_conformal_quantiles(...)`. Add a small "calibration health" line to the verdict tile that shows the *achieved* empirical coverage on the residual log alongside the target.
- **Expected effect:** Users gain a real probabilistic-forecast option for one neural backend, with calibration the model itself learned (not a wrapper). HA automations can ask coverage-specific questions. The UI surfaces miscalibration when conformal assumptions are violated.
- **Pi 5 cost:** Quantile head is `O(K · H)` extra parameters per channel — negligible at typical `K=3, H=48`. Inference is unchanged. Training time grows marginally because the loss now has to sweep `K` quantiles.
- **Effort:** M.
- **Severity:** high.
- **Confidence:** medium-high.

### H-4 — Tree-backend objective is hardcoded MSE/RMSE; no Tweedie for zero-inflated spiky series

- **Area:** Loss functions
- **Evidence:** `models/lightgbm_backend.py:191-203` hardcodes `"objective": "regression"` (MSE); `models/xgboost_backend.py:211-224` uses `XGBRegressor` defaults (`reg:squarederror`); `models/catboost_backend.py:116-126` uses `"loss_function": "RMSE"`. The Settings-tab `loss_fn` dropdown does *not* affect any of them.
- **Why it harms quality:** A large fraction of HA targets are *zero-inflated, right-skewed, near-zero series with occasional spikes*: rainfall (mm/h), heating-circulator power, EV charge rate, occupancy count, dishwasher draw. MSE on such a series spends its capacity matching the spike values at the expense of the (much more numerous) near-zero points; the trained model produces persistently above-zero baseline predictions, breaking automations that gate on "is this near zero?". The Tweedie distribution (compound Poisson-Gamma) is the standard fit for this shape, and **both LightGBM and XGBoost support it natively** (`objective="tweedie"` with `tweedie_variance_power ∈ (1, 2)`). CatBoost supports it via `loss_function="Tweedie:variance_power=1.5"`. No extra Python dependencies, no ARM64 issues.
- **Proposed change:** Make the existing `loss_fn` dropdown apply to tree backends as well, with a per-backend mapping:
  - `mse` → LightGBM `regression`, XGBoost `reg:squarederror`, CatBoost `RMSE` (today's behaviour).
  - `mae` → LightGBM `regression_l1`, XGBoost `reg:absoluteerror`, CatBoost `MAE`.
  - `huber` → LightGBM `huber`, XGBoost `reg:pseudohubererror`, CatBoost `Huber:delta=1.0`.
  - `tweedie` *(new option)* → LightGBM `tweedie` (`tweedie_variance_power=1.5`), XGBoost `reg:tweedie` (`tweedie_variance_power=1.5`), CatBoost `Tweedie:variance_power=1.5`. Skip silently on neural backends.
  - `quantile_0.5` *(new option, paired with H-3)* → LightGBM `quantile alpha=0.5`, etc. — gives users the median objective and a built-in symmetric robustness story.
- **Expected effect:** Materially better fits for zero-inflated spiky targets, which today's MSE silently mistreats. Removes the "loss_fn is silently ignored by trees" footgun documented in the UI tooltip.
- **Pi 5 cost:** None.
- **Effort:** S–M.
- **Severity:** high.
- **Confidence:** high — Tweedie/MAE/Huber are first-party objectives in all three tree backends.

### H-5 — Optuna tuning objective ignores the user's `production_metric`

- **Area:** Loss / tuning
- **Evidence:** `main.py:4943-4955`:
  ```python
  def _composite_score(mae, rmse, mase):
      ...
      return float(np.mean([
          mae / anchor["mae"],
          rmse / anchor["rmse"],
          mase / anchor["mase"],
      ]))
  ```
  Optuna minimises `_composite_score` regardless of what the user picked under `production_metric`.
- **Why it harms quality / misleads:** A user who sets `production_metric=mae` because they care about absolute kWh error gets hyperparameters tuned against a one-third-RMSE, one-third-MASE blend. RMSE-dominant tuning will pick smaller, smoother models that under-fit peak demand — exactly the wrong choice if the user is forecasting peak loads. The tuning result and the production-selection result then disagree, with no surfaced reason.
- **Proposed change:** Default the Optuna objective to the user's `production_metric`. Optionally retain the composite as one of three radio choices on the Tuning tab (`production_metric` | `composite` | `seasonal_naive_skill`), default `production_metric`. The `_composite_score` function survives behind option two for users who want it.
- **Expected effect:** Tuned hyperparameters become consistent with the model-selection criterion; user-trust improves; the "winner" rarely contradicts the tuning leaderboard.
- **Pi 5 cost:** None.
- **Effort:** S.
- **Severity:** high.
- **Confidence:** high.

### H-6 — `loss_fn` default is `mse` for a domain dominated by spiky, near-zero series

- **Area:** Loss / defaults
- **Evidence:** `ExperimentCfg.loss_fn` default `"mse"` (`config.py`); rendered as the default-selected option in `experiment.html:254-258`.
- **Why it harms quality:** MSE is the wrong default for the typical HA target. The brief lists exactly the failure modes: "frequent gaps/dropouts, sensor spikes, many series near zero (power, rainfall, occupancy)". MSE is dominated by the squared spike and pulls the predicted baseline above zero across the rest of the day; MAE and Huber are both better defaults for this shape.
- **Proposed change:** Change the default to `huber` (smooth-L1, beta=1.0 — torch's `SmoothL1Loss`). It is quadratic near zero (so gradients still flow on small errors) and linear in the tails (so spikes do not dominate). For the same data, the median trained model under Huber predicts a near-zero baseline correctly while still chasing the peaks. Keep `mse` and `mae` available.
- **Expected effect:** Across the typical HA target mix, Huber-trained neural backends should beat the same backend on MSE on MAE and on calibration. The change is invisible to advanced users (the dropdown still exists) and helpful to defaults users (most of them).
- **Pi 5 cost:** None.
- **Effort:** S.
- **Severity:** high.
- **Confidence:** medium-high — needs an empirical sweep (see V-1) but the prior is strong.

### H-7 — Gap handling is hardcoded ffill+bfill; no mask, no indicator, no interpolation choice

- **Area:** Missing data
- **Evidence:** `preprocessing.py:175-178`:
  ```python
  resampled = resampled.ffill()
  resampled = resampled.bfill()
  ```
  Applied unconditionally after every resample, plus an extra ffill+bfill on every covariate at `main.py:1429-1431`.
- **Why it harms quality:**
  - A multi-hour ffill across a recorder outage injects a synthetic flat segment that the model treats as a strong "stay flat" signal. Tree models in particular over-fit those flat ribbons.
  - There is no boolean "this row was imputed" indicator the model could use to down-weight imputed regions.
  - At the train/test fold boundary, ffill that straddles the boundary causes the *same imputed value* to appear in both train and test (an embargo of 2 rows does not help if the ffill is, say, 12 rows long).
  - The user cannot opt into "drop rows in gaps longer than X minutes" or "linearly interpolate gaps shorter than Y, drop the rest".
- **Proposed change:** Replace the unconditional fill with a small policy:
  - `gap_handling`: `{ffill, interpolate, mask}` — default `interpolate` for short gaps (≤ 3 intervals) and `mask` (drop the row) for longer ones.
  - `gap_max_minutes` to switch between "fill" and "mask".
  - Add a single boolean feature column `was_imputed` (off by default, on when `gap_handling=interpolate`) so the model can learn to discount those rows.
  - Make the embargo size dynamic: at minimum `cv_embargo_periods = max(2, gap_max_intervals + 1)`.
- **Expected effect:** Removes a quiet source of training noise on every HA setup that has had a recorder restart, a HA OS upgrade, or a network outage during the history window — i.e. most of them.
- **Pi 5 cost:** None.
- **Effort:** M.
- **Severity:** high.
- **Confidence:** high.

---

## Medium

> Suboptimal default, missing guidance, robustness gap. The model still trains; the experience is worse than it needs to be.

### M-1 — No UI guidance — every dropdown is a blind pick

- **Area:** Defaults & guidance
- **Evidence:** `experiment.html:254-286` renders the `loss_fn`, `optimiser`, `output_activation`, `daily_loss_weight`, and `production_metric` controls as plain `<select>` elements with no tooltip, no recommended setting, no "when should I pick this?" hint. The schema docstrings in `config.py:282-326` are excellent, but never reach the user.
- **Why it harms quality:** The brief is explicit that "users are HA power users, not ML practitioners — they pick options from a dropdown with limited understanding of the consequences". The `output_activation` dropdown alone has seven options (`auto, linear, softplus, relu, exp, sigmoid, zscore`); `relu` produces dead units on many-zero targets, `exp` blows up on negative residuals, `zscore` does nothing for non-LSTM backends. There is currently no information path from the well-documented config docstrings to the dropdown.
- **Proposed change:** Inline help. For every dropdown, add an info-tip (the codebase already has an `info-tip JS` per `base.html:141`) that shows a one-sentence description and a "recommended for: <sensor type>" hint. Source content directly from the existing `config.py` docstrings — no new content needed. Specifically:
  - `loss_fn`: "Huber for typical HA sensors (power, occupancy). MAE for clean signals. MSE only when peaks matter more than baseline."
  - `production_metric`: "RMSE penalises peak errors; MAE penalises everyday errors; MASE compares against the seasonal-naive baseline."
  - `output_activation`: collapse to `{auto, linear, softplus, sigmoid}` — remove `relu`, `exp`, `zscore` from the dropdown (still settable in YAML). Today's `auto` already does the right thing for 95 % of cases.
- **Expected effect:** Users stop picking pathological options by accident. Specifically, `output_activation=exp` and `=relu` cause silent training failures on near-zero targets today; pruning them from the dropdown removes those failure modes for non-experts.
- **Pi 5 cost:** None.
- **Effort:** S–M.
- **Severity:** medium.
- **Confidence:** high.

### M-2 — `daily_loss_weight` as a boolean checkbox misrepresents a continuous parameter

- **Area:** Loss / UX
- **Evidence:** `experiment.html:281-286` renders `daily_loss_weight` as a checkbox that posts `true → 0.5, false → 0.0` (`app.py:3220`). The underlying field is a `float ≥ 0` (`config.py:442-461`); the docstring says "typical useful range: 0.1–1.0".
- **Why it harms quality / misleads:** A user who finds the cumulative loss helps slightly but `0.5` is too strong has nowhere to go. A user who wants to crank it up for an experiment with very strong daily-shape priors cannot. The hidden YAML escape hatch is power-user territory.
- **Proposed change:** Replace the checkbox with a 0-1.5 slider, default `0.0`, with marks at `0`, `0.25`, `0.5`, `1.0`. Keep the checkbox label "Daily cumulative loss" as a section header.
- **Expected effect:** Users can actually tune the trade-off the codebase already supports.
- **Pi 5 cost:** None.
- **Effort:** S.
- **Severity:** medium.
- **Confidence:** high.

### M-3 — No exogenous "known-future" covariate in the UI; weather/solar forecasts go unused

- **Area:** Feature engineering
- **Evidence:** `ExperimentCfg.future_covariate_features` is YAML-only and feeds only the TiDE backend (`config.py`); the UI exposes `role` as a dropdown but only `lagged` is wired (`experiment.html:340-347`).
- **Why it harms quality:** HA users commonly have a weather-forecast integration (Met Office, AccuWeather, OpenWeatherMap) or a solar-forecast integration (Forecast.Solar, Solcast). These are *known-future* covariates of exactly the kind that materially help a 24-h power or temperature forecast. Today there is no path to expose them to the model.
- **Proposed change:** Expose two things in the UI:
  1. A `role = future` option in the covariate row (already in the config; needs the UI wired and the resolver implemented past the NaN-only stub).
  2. A "fetch from forecast attribute" toggle on each covariate, defaulting off. When on, the covariate is read from the entity's `forecast` attribute (HA convention for weather/solar integrations) rather than its history.
  Restrict `future` covariates to backends that can consume them — TiDE today, plus N-HiTS / TFT if the backends accept them.
- **Expected effect:** A whole class of HA setups (PV + battery, heat pump + thermostat) gets a forecast input the model currently cannot see.
- **Pi 5 cost:** Marginal — one extra HTTP fetch per covariate per cycle.
- **Effort:** M.
- **Severity:** medium.
- **Confidence:** medium.

### M-4 — Conformal coverage hardcoded at 0.80; no per-experiment override; no calibration diagnostic

- **Area:** Probabilistic evaluation
- **Evidence:** `web/app.py:2105` calls `db.get_conformal_quantiles(name, actuals_table, 0.8, model_name, ..., 14, 10, model_version)`. The 0.8 is a magic number.
- **Why it harms quality:** Already discussed in H-3; tracked here separately because the partial fix (a UI slider, *without* the quantile-loss training path) is independently useful for advanced users and is a one-day change.
- **Proposed change:** Add `conformal_coverage` to `ExperimentCfg` (default `0.8`), add the slider to the Settings tab, thread to `get_conformal_quantiles`. Surface the empirical achieved coverage on the verdict tile.
- **Severity:** medium.
- **Confidence:** high.
- **Effort:** S.
- **Pi 5 cost:** None.

### M-5 — Lag selection and rolling-window sizes are hardcoded (`n_lags=12`, `lag_windows=[6, 24, 72]`)

- **Area:** Feature engineering
- **Evidence:** `features.py:69, 137`. The values are reasonable for the default `interval_minutes=30` (12 × 30 min = 6 h of lags, 72 × 30 min = 36 h of rolling), but tied to that interval. A user who picks `interval_minutes=5` gets 1 h of lags and a 6 h max rolling window — far short of the daily horizon they need.
- **Why it harms quality:** Lag horizons should scale with the interval, not be a constant. The model has no chance of learning the daily seasonality from lag features if the lags don't reach back a day.
- **Proposed change:** Either derive `lag_windows` from `interval_minutes` (e.g. windows at 1 h / 6 h / 24 h converted to lag counts), or expose `lag_horizons_hours: [1, 6, 24]` and `n_lag_steps: 12` as YAML defaults (no need for UI controls — power-user territory). Keep `target.shift(steps_per_day)` and `shift(2 · steps_per_day)` as already wired.
- **Expected effect:** Models on 5-min and 1-min intervals get reasonable feature support; current 30-min users are unaffected.
- **Pi 5 cost:** Larger feature matrices on small intervals — `O(n × n_lags)` extra; manageable.
- **Effort:** S.
- **Severity:** medium.
- **Confidence:** high.

### M-6 — Log-transform shift is hardcoded `1.0`; no choice and no guidance

- **Area:** Transforms
- **Evidence:** `preprocessing.apply_log_transform(series, shift=1.0)` (`preprocessing.py:268-295`), called from `main.py:1386` with no override. A more sophisticated `apply_transform` exists for covariates (`preprocessing.py:753-816`) that handles `min_val == 0` and negative `min_val` cases, but it is **not** used for the target.
- **Why it harms quality:** For a sensor measured in kWh (typical daily energy: 5–30 kWh), `log(y + 1)` compresses meaningfully; the shift is irrelevant. For a sensor measured in Wh (5,000–30,000), `log(y + 1)` is dominated by `log(y)` and the shift is harmless. For a sensor in MWh (0.005–0.030), `log(y + 1) ≈ log(1)` and the model learns nothing. The fixed shift is appropriate for one of these three but not all.
- **Proposed change:** Route `log_transform` through `apply_transform(transform="shifted_log")`, which already computes the shift sensibly (`max(1, |min_val|+1)`), and store the shift in `series.attrs['transform_shift']` so the inverse on the production path can read it back deterministically. Keep the field a checkbox in the UI; the smarter shift is internal.
- **Expected effect:** `log_transform=true` becomes safe across unit scales.
- **Pi 5 cost:** None.
- **Effort:** S.
- **Severity:** medium.
- **Confidence:** high.

### M-7 — `cv_embargo_periods` is YAML-only and tied to a magic number `2`

- **Area:** Splitting & leakage
- **Evidence:** `config.py:242`, used at `benchmark/runner.py:240, 272`. The comment at `:251-253` says "embargo ... prevents target leakage through lag / rolling features whose forecast horizon overlaps the test inputs", but `2` is smaller than every rolling window today (`[6, 24, 72]`). On a 30-min grid, a 72-step rolling window contaminates a test row's feature from up to 36 h into the train fold; an embargo of 2 (1 h) is nowhere near enough.
- **Why it harms quality:** Under-sized embargo lets rolling features fitted on the last `max(rolling_windows)` train rows leak into the first test row's features. Combined with C-1, the effect is double leakage on the fold boundary.
- **Proposed change:** Default `cv_embargo_periods = max(rolling_windows + [steps_per_day])` — i.e. at least one full day plus the longest rolling window. Expose in the UI as a small read-only chip ("Embargo: 72 steps = 36 h") so users see what the protection actually is.
- **Expected effect:** Removes the fold-boundary leakage path; CV metrics become genuinely out-of-sample.
- **Pi 5 cost:** None.
- **Effort:** S.
- **Severity:** medium (becomes low if C-1 is fixed *and* embargo is widened; high if either is fixed in isolation).
- **Confidence:** high.

### M-8 — Recency half-life default `7` days is unjustified

- **Area:** Defaults
- **Evidence:** `ExperimentCfg.recency_half_life_days: float = 7.0`. Sample weights `exp(decay · arange(n))` are computed at `benchmark/runner.py:411-418`.
- **Why it harms quality / misleads:** With the default `days_history=14`, a half-life of 7 days means the most recent training row has roughly 4× the weight of the oldest. That is a strong recency bias for a series with weekly seasonality (the weekend pattern from 8 days ago is down-weighted to ~30 % of "yesterday"). The right half-life depends on the seasonality strength: for strong weekly seasonality (most domestic loads) you want the half-life ≥ `days_history`, i.e. essentially uniform. For a regime-changing sensor (recently moved house, just installed a heat pump) a short half-life helps.
- **Proposed change:** Default `recency_half_life_days = max(14, days_history)` — effectively flat unless the user shortens it. Keep the slider in the UI. Add a one-line tip: "Lower values weight recent data more strongly. Set to 0 for uniform weighting (recommended for stable setups)."
- **Expected effect:** Removes a quiet bias against the (often most informative) older rows in the history window.
- **Pi 5 cost:** None.
- **Effort:** S.
- **Severity:** medium.
- **Confidence:** medium — the right default depends on whether the typical user is regime-stable; would benefit from V-3.

---

## Low

### L-1 — Holiday feature requires a `country` YAML field with no UI

- **Area:** Feature engineering
- **Evidence:** `features.py:228-231`. UI has no country input. The Demšar leaderboard gain from holidays is small in steady-state domestic forecasting, but non-trivial in commercial/office settings.
- **Proposed change:** Add a `country` dropdown in the Settings tab (ISO-3166 alpha-2, default empty), wired straight to `ExperimentCfg.country`. The `holidays` library is already a dependency.
- **Severity:** low. **Effort:** S. **Confidence:** high.

### L-2 — `output_activation` exposes seven options; three are footguns

- **Area:** Loss / output head
- **Evidence:** `experiment.html:269-277`; `relu` (dead units on many-zero targets), `exp` (numerical blow-up on negative residuals before activation), `zscore` (treated as `linear` by non-LSTM backends — silent no-op).
- **Proposed change:** Remove `relu`, `exp`, `zscore` from the UI dropdown (keep them YAML-settable for backwards compatibility). Folded into M-1.
- **Severity:** low (because most users leave it on `auto`).

---

## Already good — kept here so the audit does not over-claim

- **CV is strictly temporal** (`benchmark/runner.py:227-328`); no shuffle anywhere; both walk-forward (expanding) and sliding-window choices.
- **A seasonal-naive baseline is force-run on every benchmark** (`main.py:2056-2068`) so the UI can render a skill chip. This is unusually good for an "add-on for non-experts" and should not be lost in refactors.
- **Cumulative-to-interval handling** in `preprocessing.cumulative_to_interval` is genuinely careful — handles daily resets, midnight crossings, spike capping, and multi-interval gap detection with `multi_interval_gap = gap_scale > 1.5` so a 4-hour outage is dropped to NaN rather than synthesised into a single inflated bucket. Few addon-grade implementations get this right.
- **Load-subtract robustness checklist** (`preprocessing.apply_load_subtract`) is best-in-class: per-sensor `on_missing` policy, scale-aware unit-bug guard with a `max_fraction_violation_pct` canary, explicit leading/trailing gap reporting, audit dict returned to caller.
- **RevIN as the default per-window normaliser** (`models/base.py:139-254`) is the correct choice for non-stationary HA series and matches the published transformer literature.
- **The composite Demšar rank** is a reasonable cross-metric aggregator; the upgrade (H-1) is in the *metrics* being ranked, not the ranking.
- **AdamW as the default optimiser** with cosine annealing and gradient clipping `max_norm=5.0` is the right baseline for transformer-family backends on this hardware.

---

## Needs empirical verification

These are claims we cannot fully prove from a static read; each needs a backtest sweep on a real dataset (the addon ships a `tests/dryrun_pipeline.py` harness that is the natural place for it). We have stated which direction we expect the result to go.

- **V-1 — Magnitude of the rolling-feature leak.** Run the existing benchmark suite with the C-1 fix and compare tree-backend CV metrics before/after on three representative HA targets: (a) cumulative daily energy (smooth, seasonal), (b) instantaneous power (spiky, near-zero), (c) outdoor temperature (smooth, slow-varying). We expect tree-backend MAE / RMSE / MASE to *worsen materially* on (a) and (b) and only marginally on (c). If the change is < 5 % even on (b), the leak is a smaller deal than we are flagging.
- **V-2 — Does C-2 (log-space metric inversion) change the auto-selected production model?** Re-run the leaderboard with and without the inversion on a target where `log_transform=true` is sensible (cumulative daily energy). We expect the auto-selected model to change in roughly half of such experiments; if not, C-2 is still a UX/trust fix but lower priority.
- **V-3 — Right default for `recency_half_life_days`.** Sweep `{0, 7, 14, 28}` on a stable household-power target and a regime-changing one (e.g. after a heat pump install). We expect `0` (uniform) to win on stable, `~7` on regime-changing.
- **V-4 — Huber vs MSE as the default `loss_fn`.** Cross-sweep `{mse, mae, huber}` on the three V-1 targets, ranked by `seasonal_mase` (post-H-1). We expect Huber to win or tie on (a) and (b), MAE to win on extremely spiky targets (rainfall), and MSE to be rarely the right answer.
- **V-5 — Quantile-loss neural backend vs post-hoc conformal bands.** Once H-3 ships, compare empirical coverage and pinball loss for the same target between (i) point-MSE + 0.8 conformal and (ii) quantile loss with `[0.1, 0.5, 0.9]`. We expect the quantile model to win on calibration during regime shifts (DST, school-holiday transitions), tie elsewhere.
- **V-6 — Tweedie for tree backends on zero-inflated sensors.** Compare LightGBM Tweedie vs MSE on rainfall and occupancy targets, scored on `seasonal_mase` and on "frac-of-time predicted ≈ 0 when actual = 0". We expect a clear Tweedie win on the latter.
- **V-7 — Embargo size effect after fixing C-1.** Once C-1 is fixed, the embargo size primarily protects against ffill straddling the boundary (H-7) and against rolling spillover that would otherwise be small. Re-sweep `cv_embargo_periods ∈ {2, 24, 72, 144}` post-fix to see whether the bigger embargo actually matters.

---

## Summary table

| ID | Title | Severity | Confidence | Effort | Pi 5 cost |
|---|---|---|---|---|---|
| C-1 | Rolling-feature target leakage at training | critical | high | S | none |
| C-2 | Benchmark metrics computed in log space | critical | high | S | none |
| C-3 | Covariate scaling fit on full series | critical (class) | medium-high | S | none |
| H-1 | MASE uses 1-step not seasonal naive | high | high | S | none |
| H-2 | Outlier handling is fixed q=0.995 | high | high | S–M | none |
| H-3 | No quantile loss; conformal hardcoded 0.8 | high | medium-high | M | negligible |
| H-4 | Tree-backend objectives hardcoded MSE | high | high | S–M | none |
| H-5 | Optuna objective ignores `production_metric` | high | high | S | none |
| H-6 | `loss_fn=mse` is the wrong default for HA | high | medium-high | S | none |
| H-7 | Gap handling hardcoded ffill+bfill | high | high | M | none |
| M-1 | No dropdown guidance; footgun activations exposed | medium | high | S–M | none |
| M-2 | `daily_loss_weight` is checkbox, not slider | medium | high | S | none |
| M-3 | No known-future covariate in UI | medium | medium | M | marginal |
| M-4 | Conformal coverage hardcoded 0.8 | medium | high | S | none |
| M-5 | Lag/rolling sizes don't scale with interval | medium | high | S | small |
| M-6 | Log-shift hardcoded 1.0 | medium | high | S | none |
| M-7 | `cv_embargo_periods=2` smaller than rolling windows | medium | high | S | none |
| M-8 | `recency_half_life_days=7` unjustified default | medium | medium | S | none |
| L-1 | No UI for `country` (holiday flag) | low | high | S | none |
| L-2 | `output_activation` has 3 footgun options | low | high | S | none |
