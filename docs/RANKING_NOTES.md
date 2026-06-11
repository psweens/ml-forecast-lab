# Ranking Notes — what the leaderboard does and does not claim

## TL;DR

The leaderboard reports the **composite mean rank** of each backend across CV
folds, averaged across MAE / RMSE / MASE. It uses the *averaging step* from
Demšar (2006), but **not** the Friedman + Nemenyi tests that the full Demšar
procedure prescribes — those tests assume the per-dataset measurements are
independent, and our N is "CV folds of one series" rather than "independent
datasets". Bootstrap CIs over fold resamples are reported alongside each rank
so you can spot when the leaderboard is genuinely "tied" rather than crowning
a single winner that's within noise of #2.

If you want one rule of thumb: **trust the rank only when the bootstrap CIs
of #1 and the next-best model don't overlap.** When they do overlap, the
leaderboard's "winner" is roughly a coin flip with the runner-up.

## What's computed, step by step

For each CV fold:

1. For each ranking metric (`mae`, `rmse`, `mase`, `seasonal_mase` —
   anything not in the higher-is-better set):
   1. Sort models by that metric on this fold; assign integer ranks 1..K.
2. Average a model's ranks across metrics → this model's *composite fold rank*.

Then, across folds:

3. Take the mean of each model's composite fold ranks → `mean_rank` (lower
   is better, 1.0 = first place on every fold and every metric).
4. Sort by `mean_rank` → integer leaderboard rank.

Bootstrap CIs are computed by:

5. Resampling fold IDs with replacement (B = 1000 resamples by default),
   recomputing step 3 on each resample, taking the 2.5 / 97.5 percentile of
   the resulting mean-rank distribution per model.

A model is only ranked if it completed **every** fold. Models that errored
on one or more folds appear under "Did not complete" rather than being
silently ranked last — last-place phantoms would otherwise inflate the
apparent gap between the survivors.

## Why this is *not* the Demšar procedure

Demšar (2006) — "Statistical Comparisons of Classifiers over Multiple Data
Sets" — describes:

1. A **Friedman test** to reject the null "all algorithms are equivalent" across
   K algorithms and N datasets.
2. **Post-hoc tests** (Nemenyi for all-vs-all, Bonferroni-Holm for one-vs-rest)
   with critical-difference (CD) diagrams that show which pairwise gaps are
   significant.

The Friedman statistic's null distribution assumes the N rank vectors are
**statistically independent**. In Demšar's setting, N = number of datasets;
in ours, N = number of CV folds of *one* time series. Walk-forward and
sliding-window CV folds share most of their training data, identical feature
pipelines, and identical generative process — they're emphatically not
independent.

A second issue is scale: at typical settings (K = 28 backends, N = 5 folds),
the Nemenyi critical difference is

$$\mathrm{CD} = q_{0.05} \cdot \sqrt{\frac{K(K+1)}{6N}} \approx 3.50 \cdot \sqrt{\frac{28 \cdot 29}{30}} \approx 18.2$$

— wider than the entire rank scale. Even at K = 5, N = 5 the CD is ~2.7 rank
units. So even if the independence assumption *did* hold, the test would
report "indistinguishable" for almost every pair. Reporting a CD bar at this
sample size would be reassurance theatre.

## Why we still report a rank

A rank is a sensible aggregate UI ordering — it normalises across metrics of
incompatible scale and gives a one-glance leaderboard. It just isn't a
*test*, so it shouldn't be presented as one. Bootstrap CIs are the honest
way to express how much the order is shaped by fold-to-fold noise. With
N = 5 folds the CIs are typically wide; that's the signal, not a bug.

## When the Demšar test *would* apply

If the leaderboard were aggregated across **independent experiments**
(different `target_entity` values — e.g. one rank vector per sensor across
PV, hot water, heating, EV, occupancy), the per-experiment rank vectors
would be much closer to Demšar's "independent datasets" assumption. A
cross-experiment Friedman + Nemenyi would then be defensible. That's a
future enhancement, not what the current per-experiment leaderboard does.

## Practical reading guide for the UI

| Display | Meaning |
|---|---|
| `mean_rank = 1.4 [1.0–2.8]` | Mean rank 1.4 across folds; 95% bootstrap CI says the rank could plausibly be anywhere in 1.0–2.8 given fold noise. |
| Badge **#1** | Lowest mean rank — and the leader's CI does not overlap any other model's. Treat as the winner. |
| Badge **T#1** | This model's CI overlaps the leader's. Treat as tied — the data doesn't support picking between them. |
| **Did not complete: …** | These models failed at least one fold and are excluded from the rank to keep the comparison like-for-like. |

## References

- Demšar, J. (2006). [Statistical Comparisons of Classifiers over Multiple
  Data Sets](https://www.jmlr.org/papers/v7/demsar06a.html). *Journal of
  Machine Learning Research*, 7, 1–30.
- Benavoli, A., Corani, G., Demšar, J., & Zaffalon, M. (2017). [Time for a
  Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian
  Analysis](https://www.jmlr.org/papers/v18/16-305.html). *JMLR*, 18, 1–36.
  (Argues for Bayesian alternatives — equally inapplicable here without
  independent datasets, but useful background on why CD-bar plots are
  often misleading.)
