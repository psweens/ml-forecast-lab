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

### Interval construction

How a point forecast's past signed errors are grouped before their
quantiles are added to a new forecast decides most of its CRPS on this
series (same 74 issue times; the seasonal-mean profile with its own
per-slot quantiles scores 0.940):

| Point forecast | by lead | by forecast size | by local time block |
|---|---|---|---|
| LightGBM Huber | 1.009 | 0.941 | 1.041 |
| LightGBM L2 | 1.041 | 0.975 | 1.074 |
| LightGBM Tweedie | 1.090 | 0.989 | 1.105 |
| seasonal-mean profile, point only | 1.156 | 1.002 | 1.112 |

Errors depend on how much was forecast far more than on how far ahead:
when little is forecast they are small and one-sided, when a lot is
forecast they are large and right-skewed. Grouping by forecast size into
equal-count bins (`level_conditioned_offsets`) captures that with a few
dozen errors per bin (the table used four bins; eight, the value adopted,
improved every forecaster on every series by a further 0.005 – 0.027); grouping by three-hour local block and
weekday/weekend spreads the same errors over sixteen groups and is worse
than grouping by lead. The published asymmetric band
(`interval_method`) and the evaluation's default distribution both use
the forecast-size grouping. Realised coverage of the nominal 80 % band on
this window was 0.76 – 0.77.

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

## What changed

- **Selection** (v2.58.0): the benchmark scores each backend's published
  forecast and, for non-negative targets with `production_metric` unset
  or for `production_metric: crps`, selects and tunes on CRPS with a
  period-total guardrail (see *Review follow-up*). On ten days of
  the reference series with LightGBM, XGBoost, the seasonal-naive backend
  and one linear sequence backend, the one-step composite rank put XGBoost
  first; its published forecast was 40 % under the measured level and
  failed the guardrail, and LightGBM (−12 %) was selected. Both seasonal
  profile references scored better than every backend. Design and rules:
  `docs/RANKING_NOTES.md`.
- **Production loss**: tree backends are retrained with the experiment's
  `loss_fn`.
- **Intervals**: asymmetric by default for non-negative targets.

- **Calendar features** are read in the configured timezone through one
  process-wide setting (`features.set_calendar_timezone`, set at the start
  of every pipeline) rather than a parameter: nine places build a calendar
  value — the feature frame, future-known features, two sliding-window
  builders, the neural inference window and three recursive forecast rows
  — and a path that missed a parameter would train on one calendar and
  forecast on another. Sub-hourly frames gain `slot_of_day`. With the zone
  unset or UTC every earlier column is unchanged. The model-cache schema
  moved to 4, so every experiment retrains once.
- **`seasonal_profile` backend**: the mean by local slot-of-day and
  weekday/weekend. In the app's benchmark on ten days of the reference
  series it was selected (CRPS 0.839, level +2 %, against LightGBM 0.916
  and −13 %, XGBoost 0.856 and −43 %); its production forecast ranged
  0.03 – 8.7 with a day-1 total of 56.1 against a measured mean of 57.
  On four series — the reference and three synthetic — no LightGBM
  objective, with or without a level correction, beat this profile on
  CRPS; their forecasts carried a quarter to a third of the true
  amplitude against the profile's half.

### Tree objectives, measured on the published forecast

LightGBM on the reference series and three synthetic series, 96-step
recursive forecasts, level relative to the measured total (the
seasonal-mean profile's own level on each window is shown, since a short
test window differs from its training window by sampling alone):

| | reference | synthetic 0 | synthetic 1 | synthetic 2 |
|---|---|---|---|---|
| seasonal-mean profile: CRPS / level | 0.940 / +9 % | 0.997 / −7 % | 1.044 / −2 % | 0.963 / −8 % |
| Huber, stop on RMSE | 0.941 / −11 % | 1.121 / −29 % | 1.216 / −37 % | 1.060 / −38 % |
| Huber, stop on Huber | 0.933 / −14 % | 1.080 / −62 % | 1.151 / −57 % | |
| L2 | 0.975 / +19 % | 1.111 / −20 % | 1.224 / −4 % | 1.095 / −12 % |
| Tweedie, stop on deviance | 0.971 / +0 % | 1.084 / −35 % | 1.154 / −17 % | 1.069 / −24 % |
| Tweedie, level-corrected | 0.990 / +21 % | 1.092 / −20 % | 1.162 / −3 % | 1.078 / −14 % |

- **Stopping Huber on its own metric is worse.** Training runs four to six
  times longer and the level falls to 57 – 62 % under: the Huber metric
  keeps improving as the fit slides to the median. RMSE stopping is what
  limits the damage, so Huber keeps it.
- **Tweedie's shortfall is the log link.** Its in-sample fit averaged
  17 % below the training mean (correction factors 1.17 – 1.21), where an
  L2 fit matches it (0.97 – 1.00). Early stopping leaves the log-space
  term shrunk, and a shrunken exponent has a lower arithmetic mean. With
  the correction (`log_link_level_scale`) Tweedie's level tracks L2's on
  every series. Its uncorrected level on the reference series was right
  by coincidence of that shortfall and a test window lower than its
  training window.
- **Stopping Tweedie on its deviance** is a small consistent gain
  (0.971 vs 0.989, 1.084 vs 1.091, 1.154 vs 1.173).
- **The correction costs about 0.01 CRPS** on every series and takes the
  remaining-in-day RMSE down by 9 – 21 %. A distribution built from
  signed errors absorbs a level error; a total does not.
- **No default objective is changed.** Tweedie is better than Huber on
  level and totals everywhere and on CRPS on the synthetic series, and
  worse on CRPS on the reference series. That is not enough to move every
  non-negative experiment off Huber, and with selection made on the
  published forecast and a profile backend on the leaderboard, an
  under-levelled Huber tree is no longer what gets published.

### After the calendar change

With calendar features in local time and `slot_of_day` present, the same
comparison (CRPS / level; eight error bins):

| | reference | synthetic 0 | synthetic 1 | synthetic 2 |
|---|---|---|---|---|
| seasonal-mean profile (own quantiles) | 0.940 / +9 % | 0.997 / −7 % | 1.044 / −2 % | 0.963 / −8 % |
| `seasonal_profile` backend | 0.970 / +4 % | 1.035 / −5 % | 1.053 / +0 % | |
| LightGBM Huber | 0.943 / −11 % | 1.067 / −31 % | 1.092 / −45 % | (1.026 / −36 %) |
| LightGBM L2 | (0.976 / +29 %) | (1.070 / −18 %) | (1.103 / −5 %) | (1.014 / −13 %) |
| LightGBM Tweedie | 0.986 / +24 % | 1.058 / −23 % | 1.039 / −7 % | (0.986 / −12 %) |

Bracketed cells were measured with four bins. Three things stand out.

- **The profile backend and the profile reference make the same point
  forecast and score differently.** The reference carries each slot's own
  empirical quantiles; the backend, like every backend, gets its
  distribution from pooled past errors. That costs it 0.01 – 0.04 CRPS, so
  no backend beats the seasonal reference on CRPS on any series, although
  the profile backend matches or beats it on RMSE and level. Closing that
  gap needs backends that return their own quantiles, which the
  `ForecastModel` interface does not provide.
- **Default Huber is worse on level after the change, not better.** With
  a sharper calendar signal it trains longer and slides further towards
  the median: on the production path of the synthetic series its day-1
  total went from 34 % to 57 % under, and it fails the period-total
  guardrail on two of three synthetic series. On the reference series it
  has the best backend CRPS with a level 11 % under.
- **Tweedie beats L2 and Huber on CRPS on all three synthetic series and
  is last on the reference series**, on 12.6 days where none of those
  differences is separable.

### Intraday state features

Tested as a direct multi-horizon LightGBM (one row per issue time and
lead; target slot, weekend flag and lead as features) with and without
state known at issue time: intervals since the last event of 1 or more,
whether the day's first event has happened, events so far today, and
demand so far today.

| | reference CRPS | synthetic CRPS |
|---|---|---|
| calendar and lead | 1.008 | 1.039 |
| plus state | 0.982 | 1.038 |
| paired difference | −0.026 [−0.047, −0.005] | −0.000 [−0.008, +0.008] |

The state features help on the measured series and not on the synthetic
one, whose events are independent by construction — consistent with a
late event replacing an early one. But the gain appears at every lead,
including 42 – 48 hours, where issue-time state should carry little; RMSE
is not separable; the remaining-in-day total is worse (15.1 against
14.2); and both direct models score below the seasonal profile. On one
series and under a fortnight that does not justify a second training path
for tabular backends. Not adopted.

## Review follow-up

Changes made after review of the first cut, all in v2.58.0.

- **A configured metric is honoured.** The first cut selected every
  non-negative experiment on CRPS whenever `production_metric` was `mae`,
  `mase` or `seasonal_mase`, including where the user had chosen it.
  `production_metric` is now unset by default, which is automatic (CRPS
  on a non-negative target, `seasonal_mase` otherwise); `ExperimentCfg`
  resolves it to `seasonal_mase` for every consumer and records
  `production_metric_explicit`. A configured median-seeking metric on a
  non-negative target is used as set and named in the benchmark log.
- **Paired comparison with the production model, not a promotion gate.**
  The review first added a gate that kept the previous CRPS winner as
  "Best" unless a challenger won a paired day-block comparison by at least
  1 %. Promotion is the user's decision — **Promote** pins
  `production_model`, and lab mode only reports — so the gate's one
  effect in normal use was a "Best" label that disagreed with the table.
  It was replaced by `horizon.compare_with_production`: the Results tab
  states how the table's winner compares with the promoted model over the
  issue times both were scored on, and whether the difference is
  consistent. Measured on the forty days of hourly reference totals
  (29.4 days scored; LightGBM default Huber, LightGBM Tweedie,
  `seasonal_profile`):

  | Pair | paired CRPS difference [95 %] | verdict |
  |---|---|---|
  | `seasonal_profile` vs LightGBM Huber | −0.105 [−0.204, −0.001], −5.8 % | consistent |
  | `seasonal_profile` vs LightGBM Tweedie | −0.005 [−0.092, +0.082], −0.3 % | within variation |
  | LightGBM Tweedie vs LightGBM Huber | −0.099 [−0.148, −0.053], −5.5 % | consistent |

  On the ten-day half-hourly reference the same harness scored 4.1 days
  and no pair separated, so on a fortnight of history the note will
  usually say a difference is within the variation.
- **Tweedie on forty days.** On the hourly reference series above,
  Tweedie (level-corrected) and the profile are inseparable on CRPS
  (1.699 against 1.679) and Tweedie has the lowest rest-of-day RMSE of any
  forecast (12.8 against 13.2 for the profile and 17.1 for Huber); Huber
  is 22 % under the measured level and 5.5 % worse on CRPS than Tweedie.
  That is the longest real window measured, and it reverses the
  ten-day result where Tweedie was last. It bears on the default-objective
  decision, which this release leaves unchanged.
- **The level correction is weighted like the fit.** With recency
  weighting a tree fit tracks the weighted mean, so
  `log_link_level_scale` now compares weighted means; unweighted, it
  would pull a model whose recent level rose back towards older, lower
  data. The correction itself is an in-sample ratio, so a level the
  features explain is carried through.
- **Residuals grouped by time of day were not adopted.** Grouping a point
  forecast's past errors by local time block is worse than grouping by
  lead or by forecast size on every forecaster (see *Interval
  construction*). The remaining CRPS gap between `seasonal_profile` and
  the per-slot reference needs backends that return their own quantiles.

## Direction (as first written)

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
