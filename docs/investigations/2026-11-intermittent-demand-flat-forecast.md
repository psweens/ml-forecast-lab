# Flat, under-level forecasts on intermittent-demand targets (2026-11)

## Symptom

A production experiment on a half-hourly demand target built from a
daily-resetting cumulative counter published a point forecast with almost no
daily shape and a daily total about 20 % below the measured one. The target
is intermittent: a small constant background in most slots, with events in
about a fifth of slots carrying about four fifths of the demand (median 0.2,
mean ~1.3, 99th percentile ~15 per slot). The selected backend was LightGBM.
The 80 % band was symmetric about the point forecast and clipped at zero.

## Reproduction

No stored data was available, so the pathology was reproduced on a synthetic
counter calibrated to those statistics
(`ml-forecast-lab/tests/synthetic/intermittent_demand.py`): local-time
weekday/weekend event profile, reset at local midnight, change-only history
rows, a 40-day window that spans a daylight-saving change.
`tests/synthetic/run_intermittent_demand.py` drives `_retrain_and_cache` and
`_compute_cached_forecast` on it.

With every setting at its default, on a series whose expected slot profile
runs from 0.2 overnight to ~5 at the morning peak (expected total for the
first forecast day 63.3), issued at 19:30 on a 96-step horizon:

| Tree objective | Forecast range | Day-1 total | Amplitude vs expected profile |
|---|---|---|---|
| Huber (default) | 0.70 – 1.36 | 41.6 (−34 %) | 0.12 |
| Huber, `learning_rate: 0.001` | 0.93 – 1.19 | 50.7 (−20 %) | 0.06 |
| L2 | 0.79 – 3.57 | 58.6 (−7 %) | 0.35 |
| Tweedie (p = 1.5) | 0.47 – 1.82 | 48.2 (−24 %) | 0.28 |

Amplitude is the standard deviation of the forecast over the horizon divided
by that of the generator's expected profile; 1.0 is a full-depth daily shape.
The training target in every run summed to within 0.7 % of ground truth.

Across four seeds and 14/30/40-day windows the default Huber forecast stayed
inside roughly 0.4 – 1.9 in eleven of twelve runs, and its day-1 total was
15 – 43 % under the window's mean daily total in all twelve.

## Findings

### 1. The default tree objective estimates a quantity near the median (primary)

`LightGBMModel` defaults to `loss_fn='huber'`, mapped to LightGBM's `huber`
objective with its default `alpha = 0.9` — a threshold in target units. On
this target almost every event is further than 0.9 from any plausible
prediction, so its gradient is clipped to ±0.9 and the fit behaves as an
absolute-error fit: it converges toward the conditional median, which is the
background level in every slot whose event probability is under one half —
all but the sharpest peak hours. Boosting starts from the mean (the L2
initial score) and each round moves the level down. The same mapping exists
in the XGBoost and CatBoost backends.

Early stopping compounds it. The validation metric is fixed at RMSE while
the objective is Huber, so as the level drifts down validation RMSE rises and
training stops after a few dozen rounds at `learning_rate = 0.03`
(best iteration 10 – 66 in ten of twelve runs): the model is caught
part-way between the mean and the median, with the shape shrunk by the
learning rate. Lowering the learning rate makes this worse, not better: at
0.001 the forecast range on the synthetic series is 0.93 – 1.19 with a daily
total 20 % low — the closest match to the reported output.

Changing the objective alone does not meet a 5 % level tolerance. Thirty
rounds at 0.03 leave any objective about 60 % of the way to its fit, so L2
shows a third of the true amplitude; and Tweedie boosts in log space, where
a shrunken fit has a lower arithmetic mean than the data (−24 % here). The
stopping rule — RMSE on the last fifth of a noisy window, with an absolute
`min_delta` of 0.001 — ends training before the profile is learned.

### 2. Every ranking metric prefers a constant below the mean

On the validation tail of the synthetic series:

| Forecast | MAE | RMSE |
|---|---|---|
| constant 0 | 1.18 | 2.78 |
| constant median (0.2) | 1.07 | 2.70 |
| constant mean | 1.70 | 2.53 |
| LightGBM, Huber | 1.37 | 2.46 |
| LightGBM, L2 | 1.56 | 2.41 |

The composite rank averages MAE, RMSE and MASE; `production_metric` defaults
to `seasonal_mase`. Two of the three rank metrics, and the default
selection and tuning objective, are MAE scaled by a constant, so they order
models by how close they sit to the median. The correctly-levelled L2 model
loses to the under-levelled Huber model, and both lose to a constant. Tuning
searches on the same metric: forty trials over the Tuning tab's LightGBM
ranges minimising one-step MAE produced a model whose recursive daily total
was 21.5 against 61.

### 3. The leaderboard scores a different forecast from the one published

`BenchmarkRunner.run_single_model` scores tree backends with
`model.predict(X_test)`: one step ahead, with measured lags. Production
publishes a `future_periods`-step recursive forecast in which every lag and
rolling statistic beyond the first step is the model's own output. On this
target the calendar features carry most of the gain (56 – 70 %), so recursion
is not what flattens the forecast — the objective is — but the leaderboard,
the tuning objective and the accuracy that follows are measured on a
forecast nobody receives. Lag-dominated models will rank well and publish a
decaying forecast.

### 4. Production retraining ignores the experiment's loss for tree backends

`_retrain_and_cache` and `_run_production_inference` apply
`exp_cfg.loss_fn` only when `model.is_neural`. The benchmark loop and
`_apply_experiment_neural_params` (tuning, holdout) apply it to tree
backends too. An experiment set to `loss_fn: tweedie` is therefore
benchmarked and tuned with a Tweedie tree and retrained for production with
a Huber one. Only a `model_params` / `model_overrides` entry reaches the
production tree.

### 5. Calendar features are computed in UTC

History is normalised to tz-naive UTC and `build_features` reads
`index.hour` / `index.dayofweek` from it; the recursive forecast row does
the same. For any zone with daylight saving, a training window that spans a
change holds two profiles one hour apart under the same `hour_of_day`, and
day-of-week / weekend flip at UTC midnight rather than local midnight. On
the synthetic series the weekday morning peak (local 07 – 09 h: 4.9, 4.7)
reads as 3.1, 5.4, 1.8 across UTC hours 06 – 08. This costs peak sharpness
for a few weeks after each change; it is not the cause of the flat output.
There is also no sub-hour feature, so the two half-hours of an hour are
indistinguishable to a tree except through lags.

### 6. The interval is symmetric by construction

`_conformal_bands` publishes `ŷ ± q`, with `q` the level-quantile of
absolute residuals per lead bucket, and clips the lower edge at zero for
cumulative sources. On a right-skewed target the band below the forecast is
mostly impossible values and the band above stops well short of the events.

### 7. Target construction is close to unbiased, with one exception

On the synthetic counter the training target summed to within 1 % of ground
truth through local-midnight resets in both UTC offsets, a three-hour outage
in a peak, an outage across the reset, and repeated short outages (an outage
moves demand into the slot where readings resume; it does not lose it).
Resampling uses `sum` for cumulative sources and the 0.999 outlier clip
removed 0.03 %.

The exception is the default increment cap (`_rate_based_cap`). When each
event arrives as a single jump between two readings, the cap removed 4.3 %
of demand; with events spread over several readings, 1 – 1.5 %. It does not
account for a 20 % shortfall on the synthetic series, but the loss depends
on a sensor's update cadence and should be pinned by a test.

## Confirmation on a measured series

A reference series from a real counter of this kind (change-only readings
about every two minutes for ten days, hourly totals for forty) was then run
through the same code.

- **Target construction is exact on it.** The reconstructed half-hourly
  target matched the counter's own totals on every day (0.0 % overall). The
  default increment cap clipped no reading: at that cadence it allows about
  7 units per minute and the largest single increment is 4.6, although 24 %
  of demand arrives in increments of 3 or more. The 0.999 outlier clip
  removed 0.3 %. The loss seen on the synthetic counter comes from readings
  seconds apart, where a cap that scales with elapsed time has almost no
  allowance; flooring the elapsed time would stop that but lets through the
  glitch recovery `test_a_daily_counters_glitch_is_still_capped` pins, so
  the cap is unchanged.
- **The flat forecast reproduces.** With ten days of readings and default
  settings the production forecast ranged 0.66 – 1.11 with a day-1 total of
  40.8 against a measured mean of 57. With `learning_rate: 0.001` it ranged
  0.86 – 1.13, mean 0.995, day-1 total 47.6 — the reported output.
- **Scored as published**, from 74 issue times over the last 12.6 days
  (`benchmark.horizon`):

  | Forecast | MAE | RMSE | Level | CRPS |
  |---|---|---|---|---|
  | seasonal profile, median | 1.14 | 2.91 | −40 % | 0.940 |
  | seasonal profile, mean | 1.40 | 2.86 | +9 % | 0.940 |
  | constant mean | 1.51 | 2.97 | +8 % | 0.957 |
  | LightGBM Huber, `learning_rate` 0.001 | 1.38 | 2.94 | −6 % | 0.962 |
  | LightGBM Huber (default) | 1.29 | 2.86 | −11 % | 1.009 |
  | LightGBM L2 | 1.56 | 2.92 | +19 % | 1.041 |
  | LightGBM Tweedie | 1.39 | 2.90 | −0 % | 1.090 |
  | zero | 1.25 | 3.22 | −100 % | 1.246 |
  | same slot previous day | 1.63 | 3.89 | +3 % | 1.441 |

  No LightGBM variant beats the seasonal-mean profile; the default Huber,
  L2 and Tweedie models are worse on a paired day-block bootstrap. The
  models' distributions here are their point forecast plus earlier signed
  errors pooled by lead, which is blunter than a per-slot empirical
  distribution; that, not the point forecast, is most of the CRPS gap.

Two properties of this class of target limit what any evaluation can say:

- **Slot-level scores barely separate a daily profile from a constant.**
  CRPS 0.940 against 0.957, RMSE 2.86 against 2.97. Most of the
  distribution sits at the background level in every slot, so shape moves
  the score by a few percent. The remaining-in-day total separates
  forecasts far more sharply: at a midnight issue the seasonal-mean profile
  has an RMSE of 19 and the MAE-preferred median profile 35.
- **Level cannot be verified to 5 % on short windows.** Daily totals have
  a coefficient of variation of 0.35, so the standard error of a mean
  daily total is 11 % over ten days and 5.5 % over forty. A seasonal mean
  fitted on twelve days of the synthetic series was 18 % off the following
  four weeks by sampling alone.

## Direction

- Tree objective for non-negative skewed targets must estimate the mean
  (Tweedie, or a hurdle model), applied identically in benchmark, tuning and
  production; early stopping should monitor the objective's own metric, and
  the fitted level needs checking against the training mean, since an
  early-stopped log-link model is biased low.
- Selection and tuning need a proper scoring rule (CRPS / pinball) and
  explicit baselines (zero, mean, seasonal profile, same slot yesterday), and
  must score the multi-step forecast that is published.
- Calendar features in the configured timezone, with a slot-of-day feature.
- Asymmetric (quantile) intervals for non-negative targets.
