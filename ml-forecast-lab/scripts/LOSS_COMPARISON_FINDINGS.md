# Loss-function comparison — findings

Empirical study behind the "should we minimise per-interval or daily
cumulative?" question, run via `scripts/loss_comparison.py` on
synthetic Mixergy-like demand (zero-inflated ~91%, right-skewed,
morning/evening draw peaks, 30-min grid, predict one full day → sum =
daily total). All numbers from the **actual** neural loss path
(`ForecastModel._composite_horizon_loss` via the NLinear backend),
3-fold walk-forward CV, 50 epochs.

These are synthetic numbers — the *mechanisms* transfer; the exact
percentages do not. Re-run on a real export with `--csv`.

## The one identity that always holds

    daily_error = Σ_t (ŷ_t − y_t)            (sum of signed per-interval errors)
                = H·(mean per-interval bias) + (cancelling noise, ~√H)

Confirmed exactly in every run: per-interval bias × 48 = daily bias to
the decimal. **The daily total is dominated by per-interval _bias_,
not per-interval _noise_** — noise cancels over the day, bias
accumulates linearly.

## Result 1 — output activation dominates on a zero-inflated target

Same data (91% zeros), vary only the neural output activation:

| activation | Huber α=0 daily | MSE α=0 daily | winner |
|---|---|---|---|
| **softplus** (floors at 0.69) | +0.16 bias · 75% | +0.25 bias · 100% | Huber |
| **linear** (true zero) | −0.15 bias · 67% | −0.02 bias · **54%** | **MSE** |

`softplus(0) = 0.69` — the model physically cannot predict zero, so on
~44 zero-intervals/day it leaks a positive bias that swamps the loss
effect and *inverts the ranking*. **On heavily zero-inflated demand,
the activation floor is doing more damage than the loss function.**
`Auto` picks softplus for cumulative sources — worth overriding to
linear/relu (which allow a true zero) on sparse demand targets.

## Result 2 — with the confound removed, MSE beats Huber on the daily total

Linear activation, the mean-vs-median property shows cleanly:

| loss | per-interval bias | daily MAE |
|---|---|---|
| Huber α=0 | −0.145 (under-predicts; median-seeking) | 67% |
| **MSE α=0** | −0.021 (near-unbiased; mean-seeking) | **54%** |

Huber/MAE/MASE minimise the conditional *median*, which on right-skew
sits below the mean → systematic per-interval under-prediction →
daily-total shortfall. MSE minimises the conditional *mean* →
unbiased → best daily total. This is the real reason "switch to MSE"
helps — via the bias mechanism, not magic.

## Result 3 — the cumulative loss blend (α) HURT, as a cliff not a curve

MSE, linear activation, sweep α:

| α | daily MAE | per-interval bias |
|---|---|---|
| **0.0** | **53%** | −0.02 |
| 0.1 | 90% | −0.21 |
| 0.3 | 91% | −0.21 |
| 0.5 | 96% | −0.21 |
| 0.8 | 89% | −0.20 |
| 1.0 | 91% | −0.21 |

No α>0 beat pure per-interval. And it's a **discontinuity**: the
instant any cumulative weight engages, per-interval bias jumps
−0.02→−0.21 and stays flat across the whole α range. That's not a
gentle objective tradeoff — the EMA-normalised cumulative-trajectory
term flips the model into a worse regime as soon as it's switched on.

This contradicts the original "modest α≈0.3 defends the total"
intuition **for this target shape**. Likely cause: the cumulative
trajectory loss (running sum over the horizon) is poorly conditioned
when 91% of steps are zero — its gradient is dominated by the few
draw events and the EMA normaliser is estimated off a near-degenerate
distribution. Flagged for a closer look at
`ForecastModel._cumulative_trajectory_loss` before trusting the
loss-balance slider on sparse demand sensors.

## Result 4 — log_transform's retransformation bias is the daily-total killer

This is the one that explains the real-world "wins per-interval, loses
the daily total to Seasonal Naive" leaderboard. Three runs, MSE +
linear, identical except log handling (`--log-transform`, `--smearing`):

| config | per-interval MAE | per-interval bias | daily bias | daily MAE | vs naive 7.88 |
|---|---|---|---|---|---|
| no log | 0.518 | −0.018 | −0.85 | 6.89 | **beats** |
| log, uncorrected | **0.393** | −0.122 | **−5.84** | 8.06 | **loses** |
| log + smearing | 0.475 | +0.001 | +0.05 | 6.91 | **beats** |

Mechanism — classic retransformation (Jensen) bias. You train on
`log(y+1)`; the model predicts the centre of the log-space
distribution; `invert_log_transform` does the uncorrected
`exp(ẑ) − 1`. Because `exp` is convex, `exp(E[log y]) < E[y]` — the
back-transform under-predicts the mean by ~`exp(σ²/2)`. That bias is
small per interval (the log-space fit is actually the *best* of the
three — log helps the model learn the shape) but it **accumulates**
into a −5.84 daily bias = **48 % systematic under-prediction of the
daily total** on a mean of 12.1. No loss change touches it: the bias
is born in the back-transform, downstream of the loss.

Duan's smearing estimator fixes it: multiply the inverse by
`smear = mean_i exp(z_i − ẑ_i)` over the training-set log-space
residuals (≥ 1 by Jensen, exactly the factor the convex transform
shrinks by). Daily bias → +0.05, and the model beats naive again
while keeping most of log's per-interval benefit.

`invert_log_transform` (preprocessing.py) currently does the
uncorrected form. A production smearing correction needs the smear
factor computed at train time (log-space residuals), stored with the
model, and applied at inference. Helps **every** log-transformed
cumulative experiment, not just demand.

## Practical takeaways for a daily-total demand forecaster

0. **If log_transform is ON, it's probably your biggest daily-total
   bias source** (Result 4) — a ~48 % systematic under-shoot from the
   uncorrected back-transform. Either turn it off, or (better) add a
   smearing correction so you keep the per-interval benefit.
1. **Loss = MSE**, not Huber — unbiased per-interval → unbiased total.
2. **Output activation = linear (or relu)**, not softplus — let the
   model predict true zeros; the softplus floor is the biggest single
   bias source on sparse targets.
3. **loss_balance = 0.0 (pure per-interval)** on this evidence — the
   cumulative blend as currently implemented hurt at every α. Keep the
   *selection metric* on the daily total, but train per-interval.
4. The daily total is then defended by getting the per-interval
   **bias** to ~0, which MSE + linear does — not by optimising the
   cumulative directly.

## Caveats

- Synthetic data; one model family (NLinear); 50 epochs. The α cliff
  in particular deserves confirmation on real data and a second
  backend before we act on it in the product.
- Run the identical comparison on a real export:
  `python scripts/loss_comparison.py --csv mixergy.csv --value-col demand`
