# Changelog

## 2.40.12

**Early stopping: four related improvements.** All neural backends
plus LightGBM / XGBoost / CatBoost now share a uniform, smarter early-
stopping policy, and a single per-experiment **Patience** Setting
overrides the backend default.

### 1. `min_delta` margin on the improvement check

The strict `val_loss < best_val_loss` comparison reset patience on any
micro-improvement — a 0.0001 win on a noisy 2.0 loss reset the
20-epoch patience counter, occasionally letting training run hours
past where it should have stopped. The new check is

    val_loss < best_val_loss * (1 - min_delta)

with `min_delta = 1e-3` (0.1 % relative improvement) as the default.
Backwards-compatible: `min_delta=0` recovers the pre-fix path.

### 2. EMA-smoothed val_loss for the stop decision

Raw val_loss is jittery epoch-to-epoch — one unlucky batch can reset
the best, one lucky batch can extend training pointlessly. The
*stop decision* now compares an EMA of val_loss (α=0.3, ~3-4 epoch
effective window) against an EMA-based best, while the *best-model
checkpoint* keeps tracking raw val_loss so the saved weights are
still the truly best ones seen. `ema_alpha=1.0` recovers the
no-smoothing legacy path.

Both refinements ship as a shared helper on the base class —
`ForecastModel._step_early_stop` — and every neural backend
(17 of them: LSTM, GRU, CNN, NHITS, N-BEATS, TiDE, TSMixer,
TimeMixer, TimesNet, NLinear, DLinear, FITS, PatchTST, iTransformer,
Crossformer, TFT, SparseTSF) was mechanically refactored to use it.
Behaviour change is uniform across the model registry.

### 3. LightGBM ignored `self.patience` (hardcoded `50`)

Real bug: `lightgbm_backend.py:225` hardcoded `patience_limit = 50`
regardless of the constructor `patience` param — which didn't even
exist on the LightGBM backend until now. Pulled `patience` into the
constructor, fed it to both the per-round progress callback and the
library's `lgb.early_stopping(stopping_rounds=…)` call. Same fix
applied to XGBoost and CatBoost (also hardcoded `50`). All three tree
backends now respect the param uniformly and honour the per-
experiment Patience Setting.

### 4. Per-experiment **Patience** Setting

New entry in the Training section of Settings — an integer that
overrides every backend's default uniformly across the experiment, so
"neural runs cut off at 20, LightGBM at 50" stops being an apples-to-
oranges artefact. Leave empty for backend defaults (20 neural, 50
tree); set to an integer (1–500) to pin uniformly. Plumbed via a new
`_apply_patience` helper in `main.py`, called from every training-
setup site (benchmark CV, holdout, production retrain, tuning) so it
applies wherever a model is built — same surface as
`_apply_loss_balance` (v2.40.2).

Regression: 12 new tests pin the helper semantics — `min_delta`
threshold, EMA-vs-checkpoint independence, legacy-behaviour recovery
when both refinements are disabled, per-experiment Setting respecting
per-model overrides, and the cross-backend Setting unification.

## 2.40.11

**UI polish: disambiguate the Accuracy chip + MAE tile.** The verdict
chip used to render `Accuracy · Good · 4% of typical`, which most
users read as "the model is 4% accurate" — a non-sensical
interpretation since the 4 % is the *normalised error* (MAE ÷ typical
interval demand), not an accuracy score. Now reads:

    Accuracy · Good · off by 2.30 % (4% of typical)

The `off by …` phrasing makes it unambiguous that the headline number
is an *error magnitude in the sensor's own units*, and the
parenthesised ratio is the dimensionless quantity that drives the
Good/Fair/Poor rating. The MAE tile subtitle gains the same
clarification (`= 4% of typical demand`) and the info-tip now spells
out the rating thresholds (Good < 15 %, Fair < 30 %).

No code-path / metric semantic change — purely labelling.

## 2.40.10

**Three Forecast-Accuracy / dashboard polish fixes following v2.40.9.**

### 1. Dashboard card "Production model" shows the wrong model

PR #66 fixed the wrong-model display on the experiment page but
missed the same bug in `_dashboard_card.html` (a separate template).
Same root cause: the card rendered `exp.best_model` (latest
leaderboard winner) with no fallback through `exp.production_model`
(the YAML-pinned value the inference path actually uses), so after a
Promote+rerun cycle the dashboard label disagreed with the deployed
model. Fix: pass `production_model_by_exp` from YAML into the
dashboard context and use the full `selected_model or production_model
or best_model` chain — mirroring the experiment-page fix.

### 2. End-of-day total headline ±X% blew up when avg actual was near zero

The v2.40.9 plain-English headline rendered `mae / actual × 100`
unconditionally. For sensors where the avg actual is small (zero-
inflated demand, or the TZ bug in #3 below pinning the actual near
zero), the ratio could read "±3655.4% of the actual daily total" —
mathematically correct, narratively useless. The headline now falls
back to absolute-units framing (`"Forecasts are off by ±X kWh on
average — predicted Y vs actual Z — that's ±W% of a typical daily
total"`) when the per-cycle %-of-actual exceeds 100 % or the actual
is < 5 % of the typical scale.

### 3. Daily-cumulative SQL bucketed by UTC midnight instead of HA-local midnight (root cause of the ±3655% above)

For a deployment in BST (UTC+1), the v2.40.9 daily_cumulative query
defined "same day" using UTC date. The "last same-day target" then
landed at 23:30 UTC = 00:30 local — *right after* the local-midnight
reset of `sensor.<x>_today` — and the End-of-day card read the
post-reset value of ~0 every cycle. Plumbed `day_offset_hours`
through the accuracy endpoint and inlined the offset into the
`target_day` / `issued_day` SQL expressions (mirroring the stability
function at `db.py:2169-2180`). Bucketing now follows local midnight
on TZ-shifted deployments; UTC deployments unchanged.

Regression: a new
`test_daily_cumulative_day_offset_hours_shifts_bucket` seeds an
ATypical-BST scenario (counter at 30 through 22:30 UTC, reset to 0 at
23:00 UTC) and asserts that with `day_offset_hours=1.0` the End-of-
day "actual" reads 30 (pre-reset), not 0 (post-reset).

### Known follow-up (not in this release)

The "× Show all cohorts" button still times out for the
daily_cumulative mode on populated multi-cohort DBs. The 60 s fetch
budget is being eaten by the heavier window-function SQL serializing
through `HistoryDB._lock` against the other 3 accuracy-tab queries.
Two real fixes possible: per-request read connections (so WAL can
serve multiple readers in parallel) or query optimization (combining
the 3 daily_cumulative sub-queries, narrowing the actuals_grid scan).
Tracked separately — neither is a one-line change.

## 2.40.9

**Feature: a real "Daily cumulative" accuracy view for daily-reset
cumulative sensors.** The v2.40.7 audit removed the old "Cumulative
value" toggle because the comparison was meaningless — per-interval
delta predictions vs raw cumulative actuals, different spaces, MAE ≈
the cumulative level. That removal lost a view some users genuinely
wanted: "how close did my forecast come to the actual end-of-day
total?". This release brings it back, done properly.

For each forecast row, the predicted cumulative at ``target_dt`` is

    predicted_cumulative = seed
                          + Σ (per-interval predictions within
                               target_dt's local day, in chronological
                               order up to and including target_dt)

where ``seed = actual_cumulative_at(issued_at)`` when target_dt is in
the same local day as issued_at, and ``0`` otherwise (so the midnight
reset on the underlying counter is respected — the prior day's
accumulation does not carry over). Compared against the raw cumulative
actual at target_dt — for a daily-reset sensor that reading IS the
demand-so-far on that day, so both sides live in the same space.

Implementation:

- New ``evaluation_mode="daily_cumulative"`` on
  ``HistoryDB.get_forecast_accuracy``. Dispatches to
  ``_get_forecast_accuracy_daily_cumulative_locked`` which uses a
  window function to cumsum predictions per (issued_at, target_day),
  joins the seed from ``actuals_grid`` at the floored issuance time,
  and joins the actual at target_dt.
- ``/experiment/{name}/forecast-accuracy`` accepts ``?mode=daily_cumulative``;
  defensively coerces to ``raw`` if the sensor is not cumulative.
- The Forecast Accuracy header now exposes a real two-button toggle
  for cumulative sensors: **Per-interval demand** | **Daily
  cumulative**.
- New **End-of-day total** card (daily_cumulative mode only) shows
  average predicted vs actual daily totals, signed bias, the average
  error in real units, and a plain-English headline ("Forecasts land
  within ±X% of the actual daily total on average").
- ``typical_interval_demand`` in daily_cumulative mode is the mean of
  daily maximums (typical end-of-day total) so the verdict-card's
  ``nmae`` normalises against a meaningful scale rather than the
  per-interval ~0.5 kWh used by increment mode.
- Y-axis label on the lead-time chart adapts: "Error (kWh per bin)"
  for increment, "Error (kWh, running daily total)" for daily
  cumulative.

What to expect that ISN'T "the model got better":

- Daily-cumulative MAE numbers will be larger than per-interval MAE
  in the same units. Cumulative integrates per-interval errors, so a
  0.05 kWh/bin error over 16 hours lands ~1.6 kWh off at end of day.
- The lead-time chart slopes upward with lead time because errors
  accumulate. A flat curve would actually mean per-bin error shrinks
  with lead — rare.
- Forecasts issued earlier in the day have larger end-of-day errors
  than forecasts issued late afternoon (more forecasted-rest-of-day,
  less observed-so-far). Physical, not model badness.

Per-cohort decomposition and revision_improvement in daily-cumulative
space are deferred to a follow-up. Day-bucketing uses UTC midnight for
now — accurate for most deployments, ≤1h off at day boundaries for
TZ-shifted ones (passing ``day_offset_hours`` through the endpoint
mirrors the stability function and is the natural follow-up).

Regression: three new tests in ``TestForecastAccuracyDailyCumulativeMode``
lock in (a) perfect same-day forecast → MAE=0, (b) +1/bin
over-prediction → cumulative error grows linearly with lead, and
(c) cross-midnight forecast correctly resets the seed at the day
boundary.

## 2.40.8

**Bugfix: raised the Forecast Accuracy fetch timeout 20 s → 60 s.**
The 20 s ceiling introduced in 2.40.7 was the right safety net (it
converted indefinite "Computing…" into a recoverable error) but too
tight in practice: the tab fires four concurrent fetches that
serialize through ``HistoryDB._lock``, so on a populated multi-cohort
DB the last queued fetch could land 30–50 s after the first one
started and trigger the abort even though every query was healthy.
60 s gives the queue room to drain without disguising a true hang.

A proper concurrency fix (per-request DB connections so reads don't
serialize) is a larger change tracked separately; raising the timeout
unblocks the tab today.

## 2.40.7

**Forecast Accuracy tab — eight bugs found by audit and fixed.** The
recurring failure mode in this tab is mixing *per-interval delta* space
with *cumulative* space for cumulative-source sensors (the same class
of bug as the v2.40.5 halved-demand fix). The lead audit finding is a
silent wrong-numbers-by-default bug on the very view a hot-water /
energy demand user lands on first; the rest range from a misleading
button label to a runaway loading shimmer.

### 1. CRITICAL — lead-time accuracy double-differenced the forecast

For cumulative sensors, `forecast_log.predicted` is logged from
`y_pred` (`main.py:5068`), which is the model's **per-interval delta**
output (the HA cumulative sensor is built downstream by cumsumming
`y_pred` at `main.py:5194-5208` and is never logged). Increment mode
— the default at `app.py:2524` — applied a *second* LAG diff to
`predicted` while the actuals took a *first* LAG diff of the raw
cumulative. The forecast ended up as a second-difference compared
against a first-difference actual: different spaces.

Concretely: a *perfect* model on cumulative actuals 10 → 12 → 15 → 16
logs predictions [2, 3, 1]. Increment mode computed `predicted −
LAG(predicted)` = [NULL, 1, −2], then the `fv.value >= 0` filter
dropped the −2, and the surviving row scored MAE = 2 (predicted=1 vs
actual=3). MAE collapsed to roughly the typical demand for any model.

The fix mirrors what the trajectory function (`db.py:1040-1053, 1084`)
already does correctly: diff the actual, pass `predicted` through
unchanged. Blast radius covers the lead-time MAE/RMSE/bias curve, the
per-cohort decomposition, the `nmae` ratio fed to the verdict's
accuracy chip, and the revision-improvement tile — all of which reused
the same `forecast_vals` CTE. Regression locked in by a unit test that
seeds a perfect cumulative-source prediction and asserts MAE = 0.

### 2. Raw mode for cumulative sensors removed

Raw mode compared per-interval delta predictions against `AVG(raw
cumulative)` actuals — MAE ≈ the cumulative level rather than model
error. The "Cumulative value" toggle button is removed from the
header; the remaining badge documents what the chart is evaluating.
The backend defensively coerces `?mode=raw` → `?mode=increment` for
cumulative sensors so a bookmarked URL still works.

### 3. "× Show all cohorts" served champion only

The clear-filter button stripped every cohort param from the URL, but
the backend's `_resolve_model_filter` (`app.py:768-783`) defaults to
the champion when no param is present. So the button labelled "× Show
all cohorts" actually narrowed the view to the champion's latest
weights — the opposite of its name. Fix: the button now emits
`?model=all&version=all`, the documented "every cohort" signal.

### 4. "Computing…" shimmer could hang indefinitely

`accuracyFetch()` retried once on failure but had no timeout — a
stalled request (e.g. SQLite RLock contention during a publish-cycle
write) left the `chart-loading` CSS class on the chart forever,
rendering the "Computing…" shimmer indefinitely (style.css:1057).
Fix: wrap each fetch with a 20-second `AbortController` timeout so a
stall surfaces as a recoverable error through the existing `.catch`
path. The cohort-button fix above may itself shrink the query enough
to avoid the stall; this is belt-and-braces.

### 5/6. Verdict stability chip + headline switched to daily-total for cumulative sensors

The Layer 1 verdict chip and the plain-English headline sentence both
read `median_step_cv_pct` by default — a coefficient of variation of
per-interval predictions across cycles. For a zero-inflated demand
sensor where most intervals have mean ≈ 0.02 with std ≈ 0.05, CV
legitimately runs into the hundreds of percent. The guard at
`db.py:2128` only caught `|mean| < 1e-9`; the realistic small-but-
nonzero-mean regime passed through, and the headline rendered "±X% —
noticeably unstable" with confidence. The daily-total CV is computed
correctly (cohort-aware, local-midnight bucketed, full-coverage
gated) and is the trustworthy stability number for these sensors.

For cumulative experiments the chip and the leading clause of the
headline sentence now prefer the daily CV; the per-interval number is
demoted to a secondary clause with an honest caveat about why it's
noisy. Falls back to per-interval if daily isn't available yet.

### 7. Bias sign convention spelled out in the UI

Bias is `predicted − actual` throughout (`db.py:680, 837`), so `+`
means over-prediction and `−` under-prediction. The chart legend now
says `Bias (+ over / − under)`; hovertext says `… (+ over-predicts, −
under-predicts)`; the revision tile annotates the value with `(over)`
or `(under)`. No sign was ever wrong; the convention was just never
documented in the UI.

### 8. Removed redundant `@_locked` on `get_retrain_events`

`db.py:2382-2383` had the decorator applied twice. Harmless because
the underlying `RLock` is re-entrant, but redundant. Cosmetic
cleanup.

## 2.40.6

**Bugfix: experiment page showed the wrong "production model" name after a
new benchmark.** The model labelled as "production" in the UI — including
the green *Publishing X* button on the experiment header, the highlighted
row on the Per-Interval Accuracy leaderboard, and the default-selected
model in the Tuning / Covariate Analysis dropdowns — could disagree with
the model that inference actually deployed.

Two fallback chains had quietly diverged:

- **UI** rendered `selected_model or best_model` (`experiment.html:48,
  822, 1164, 1375, 5739`).
- **Inference / production retrain** used `production_model or
  best_model_name` (`main.py:3475–3482`, `4005`, `4140`).

The two are NOT equivalent. After a *Promote* writes `production_model: Y`
to `mlfl.yaml` and a later benchmark crowns a different winner `Z`, the
in-memory `best_model` rolls forward to `Z`, `selected_model` stays
`None`, and the UI displays `Z` while the deployed forecast keeps running
`Y`. Users saw "the production model always seems to be set to the best
model but not necessarily the one which is actually used in production".

The fix injects the YAML's `production_model` into the experiment page
template context and inserts it as the middle term in every fallback:
`selected_model or production_model or best_model`. The five template
sites now resolve to the same model as the inference path. An explicit
`selected_model` (set by clicking *Select* on the Results tab) still
takes precedence, so user picks aren't overridden.

Regression covered by a new smoke test
(`tests/smoke/test_production_model_resolution.py`): three cases hit the
HTTP route, parse the rendered HTML's `prodModel` JS variable and the
*Publishing X* button label, and assert they match what inference would
deploy in each fallback case.

## 2.40.5

**Bugfix: cumulative targets undercounted (~halved) because demand across
recorder gaps was discarded.** This was the real cause behind "the actual is
plotted at half" and every model under-forecasting a daily-reset demand
sensor.

HA's recorder stores only state *changes* (`minimal_response`), so a
daily-reset counter like `sensor.mixergy_demand_today` has **no rows during
quiet periods** (overnight, between draw-offs). The draw-off that *ends* a
quiet period then spans more than 1.5 sample intervals — and
`cumulative_to_interval` was **dropping that increment to NaN** (the
sum-resample then counts NaN as 0), silently discarding real demand. For hot
water the biggest draw (the morning shower after the overnight reset) is
exactly such a post-gap increment, so the daily total came out ~half. Because
the *training target* was halved, every model — tree and neural — under-
predicted by the same amount, and the holdout "Actual" plotted at half while
the raw sensor showed the true total.

Dropping a gap is right for an *interval* sensor (a gap = missing data) but
wrong for a *cumulative* one (a gap where the value rose = real accumulated
demand). The conversion now **keeps the gap increment**, attributed to the
row where the change was recorded — i.e. when the draw actually happened —
so the daily total is preserved exactly. (The log line changes from
"dropping to NaN…" to "keeping the accumulated delta…".)

After updating, **retrain**: the holdout "Daily Cumulative" Actual should rise
to its true level and the forecasts should follow. Only then is the
loss-balance slider a meaningful fine-tune rather than fighting halved data.

Tests: new `test_quiet_period_gap_demand_is_preserved` in
`test_preprocessing.py` (a sparse change-only day with a 7-hour quiet gap must
re-sum to the full daily total, not half).

## 2.40.4

Fixes neural models (LSTM, CNN, TiDE) appearing to stop short of the tree
models (LightGBM, CatBoost) on the holdout chart — most visible on the Daily
Cumulative view with a large `future_periods`.

Neural backends predict the holdout via sliding windows, so the last
`max_horizon - 1` (= `future_periods - 1`) points have no `h=1` window and
were left blank — e.g. with `future_periods=96` at 10-min that's ~16 h of
the neural lines missing from the right of every day, while tree models
(which `predict()` per point) cover the whole holdout. Not a model-quality
difference — a charting artifact of how multi-horizon neural models are
scored on the holdout.

Those tail points *were* predicted: the last formed window's `h=2..H` outputs
land exactly on them (at the shortest horizon available for each). The new
`_holdout_display_from_windows()` helper fills the tail from there, so neural
lines now span the full holdout — essential for the Daily Cumulative view
whose per-day sum needs every point. Display-only; leaderboard metrics
(from the CV folds) are unchanged.

Tests: 5 new cases in `test_holdout_display.py` (tail fill from last window,
no-tail when lengths match, 1-D fallback, single-horizon NaN tail, partial
fill guard).

## 2.40.3

Fixes the live Training progress reading past 100% (e.g. **"9/5 models
complete — 180%"**).

The progress counters (the live Training tab JS, and the two server-side
readers behind the dashboard card / page restore) incremented
`completed_models` on every `model_end` event but **never reset it on
`pipeline_start`**. So when one open SSE stream saw more than one run — a
re-run, a scheduled benchmark, or replayed history on reconnect — the count
accumulated across runs (a finished 5-model run + 4 of the next ⇒ 9/5).

Now the completion counter **resets at each `pipeline_start`** (only the
latest run's completions count) and is **clamped to the declared total**, so
it can never read past the total or exceed 100% even if a stray `model_end`
slips through. The two duplicated server-side readers were consolidated into
a single tested `training_events.summarise_history()` helper to stop them
drifting apart again.

Tests: 5 new cases in `test_training_progress.py` (single run, stale-prior-run
no-inflate, clamp-to-total, current-model/epoch tracking, empty history).

## 2.40.2

**Bugfix: the loss-balance slider was a no-op on the paths that produce
results.** Setting it (e.g. to 0.8) didn't change end-of-day cumulative
accuracy because the primary training paths — the **benchmark CV loop**
(which feeds the Daily Cumulative Accuracy table) and the **production
retrain** (the live forecast model) — set `daily_loss_weight` by hand and
never set `loss_balance`. Only the secondary analysis paths
(tuning / covariate analysis) went through `_apply_experiment_neural_params`
where the value was wired in. So the slider persisted to YAML but the models
that matter never received it.

Root cause was duplication: four separate hand-rolled neural-param setup
sites. Fixed by centralising into a single `_apply_loss_balance(model,
exp_cfg, overrides)` helper called by all four (benchmark, production
retrain, holdout refit, holdout-neural) plus `_apply_experiment_neural_params`.
The slider now actually drives training everywhere.

**To see the effect:** re-run the pipeline (benchmark) and/or let the
production model retrain — the change only takes effect on the next training
run, not on cached-model forecast cycles. Also confirm the production model
is a **neural** backend (LightGBM / XGBoost ignore the loss entirely), and
that the horizon spans the day you care about (`future_periods × interval`).

Tests: 3 new cases pinning `_apply_loss_balance` (wires neural models, no-op
for tree models, respects overrides) — the gap a unit test would have caught.

## 2.40.1

Consolidates the v2.40.0 loss controls to a **single always-on slider**.

The standalone "Daily cumulative loss" weight box and the "Enable balance
slider" checkbox are gone — the per-interval ⟷ cumulative slider is now the
one neural loss control, always active, defaulting to **per-interval (α=0)**.
At α=0 the loss is byte-for-byte the legacy interval-only loss (no EMA
rescale), so the default changes nothing.

Existing settings are migrated, not dropped: `config.effective_loss_balance`
(now the single source of truth for both the slider and the trainer) maps any
`daily_loss_weight` λ to α=λ/(1+λ), **including the non-negative / cumulative
auto-default of λ=0.5** (the PF9 behaviour that keeps PV-style forecasts from
flat-collapsing) → α≈0.33. So a PV/tank experiment that relied on the implicit
cumulative nudge keeps it; a signed target with no weight defaults to pure
per-interval. `daily_loss_weight` remains readable from YAML (and the API) for
back-compat but is no longer surfaced in the UI; an explicit slider value
always wins over it.

Tests: `test_loss_balance.py` updated for the α=0 raw-interval short-circuit
and a new `effective_loss_balance` resolution test (explicit wins / λ
migration / PF9 non-negative default / per-interval default).

## 2.40.0

New feature: a **per-interval ⟷ cumulative loss-balance slider** for neural
training (Settings → Data & Forecast).

Until now the cumulative-trajectory loss was an *additive* term:
`L = L_interval + λ·L_daily` (`daily_loss_weight`), where the per-interval
term always carried weight 1.0 and λ piled cumulative pressure on top — with
no way to ask for a purely cumulative objective. The new slider reframes this
as a **convex blend**:

    L = (1 − α) · L_interval  +  α · L_cumulative

α = 0 is pure per-interval (each step's value), α = 1 is pure
cumulative-trajectory (matches the running daily total at every horizon step,
which also pins the end-of-day total). Use the cumulative end when only the
daily figure matters — e.g. a Mixergy hot-water tank's daily demand.

**Magnitude-aware:** each term is divided by a detached exponential moving
average of itself before blending, so the slider position is the true balance
of *gradient influence*, not a nominal weight. Without this the cumulative
term (typically 10-100× the per-interval term in raw magnitude, and
loss-function-dependent) would dominate even at α ≈ 0.3. The EMA updates only
on training batches (validation runs under `no_grad`), so val-loss stays
comparable for early stopping.

**Backward compatible & opt-in:** `loss_balance` defaults to `None` →
the exact pre-2.40 additive `daily_loss_weight` path, byte-for-byte. Moving
the slider sets `loss_balance` (0–1) and supersedes the weight; toggling it
off restores the legacy path. Neural backends only; tree models ignore it.
`_composite_horizon_loss` became an instance method (transparent to the 18
backend call sites, which already invoked it via `self.`).

Tests: 13 new cases in `test_loss_balance.py` (legacy additive equivalence,
first-call normalisation, α-extreme behaviour, EMA train/val update rule,
single-horizon collapse, clamping, differentiability, and an end-to-end
NLinear fit through the blend path). Full unit suite passes.

## 2.39.5

Flags unstable models on the leaderboard so an outlier-robust mean rank
can't make a blow-up-prone model look like a solid pick.

The composite mean rank is deliberately outlier-robust — a single
catastrophic CV fold costs a model only one last-place finish, so a
model that is strong on most folds but blows up on one can out-rank a
consistently-mediocre model. The mean rank tells you "usually beats the
others"; it does NOT tell you "safe to deploy." A real example: N-HiTS
showing `MASE 9201 ± 18402` (one fold blew up) ranked *above* a DLinear
at `MASE 1.15 ± 0.33` (tight, reliable), purely because N-HiTS won most
individual folds.

The Results leaderboard now shows a `⚠ unstable` badge next to any model
whose per-fold spread on the production metric reveals this, with a
tooltip explaining which regime tripped it:

- **catastrophic fold** — the worst fold is >= 10x the median fold
  ("one fold blew up; mean rank hides this"), or
- **high dispersion** — std >= mean across folds (coefficient of
  variation >= 1.0; "unstable fold-to-fold").

Leaderboard-only: the ranking math is unchanged, and the flag does not
alter Promote / Tuning / live-forecast selection — it surfaces the
signal so you can choose worst-case stability over "typically wins" when
it matters. Assessed on the production metric (falling back to mase /
mae); models with fewer than 2 finite folds or all-zero error are never
flagged. The `unstable` / `instability_reason` fields default to
`False` / `None`, so benchmark results persisted by older versions
deserialize unchanged.

Tests: 7 new cases in `test_model_instability.py` (blow-up fold,
consistent model, high-dispersion-without-blowup, single-fold guard,
all-zero guard, primary-metric fallback, NaN/empty-fold handling). Full
unit suite (281 tests) passes.

## 2.39.4

Incremental covariate-history caching — cuts the per-forecast-cycle HA
load for covariate-heavy experiments.

The target series was already cached incrementally
(`main._fetch_and_preprocess`): each cycle reads the SQLite cache, finds
the latest stored timestamp, and fetches only the delta from HA. Covariates
were not — every forecast cycle re-fetched the **full `days_history`
window** for **every** covariate (`CovariateResolver.fetch_history` called
HA directly with no cache). On a 30-min cycle with a 30-day history that's
the same ~30 days of each covariate re-pulled from the recorder every 30
minutes when only one new interval actually arrived.

`CovariateResolver` now takes an optional `history_db` and caches raw
`(ds, value)` observations per `(entity, attribute_key)`, fetching only the
delta since the last cached observation — the same pattern the target uses.
Keyed by `attribute_key` so two covariates on one weather entity reading
different attributes (`temperature` vs `cloud_coverage`) cache independently;
namespaced under `cov_` so it never collides with a target table. Any cache
error degrades to the original full-window fetch, so caching can't break a
forecast cycle. When no `history_db` is injected (unit tests, embedded use)
behaviour is unchanged.

**Measured effect** (synthetic benchmark, real `CovariateResolver` +
`HistoryDB`): rows fetched from HA per cycle drop **~46×** (e.g. 43,205 → 934
for 5 weather covariates over 30 days at a 5-min recorder cadence). That is
the dominant per-cycle cost on a Pi-class host — the recorder query + JSON
serialization + network transfer all scale with row count. Client-side
parse/resample CPU is roughly unchanged (the merged window is still
resampled in full each cycle for correctness), so the end-to-end win is
concentrated where the HA round-trip is expensive: covariate-heavy
experiments on slow hosts. Experiments with no covariates are unaffected
(the target was already cached).

Tests: 5 new cases in `test_covariate_cache.py` (delta-only second-cycle
fetch, resample equivalence vs full-window fetch, per-attribute cache keys,
graceful degradation on cache error, unchanged behaviour without a
`history_db`). Full unit suite (274 tests) passes.

## 2.39.3

Code-review pass over everything since 2.38.0. 15 correctness fixes and
3 performance fixes — most are silent quality regressions a user would
have no way to notice from the UI.

**Critical (silent data loss / quality regressions):**

- **SeasonalNaive cache predating v2.38.6 silently re-introduces the
  zero-prediction bug on HA restart.** Pickles written by versions
  <2.38.6 don't carry the ``past_window_size`` field; ``load()`` was
  defaulting to ``None`` and falling back to the legacy
  ``past_len = len(target_series)`` path that indexes into the
  zero-padded future block when ``role: future`` covariates are
  configured. ``load()`` now rejects stale pickles so the orchestrator
  forces a re-fit.

- **Conformal residual buffer wiped experiment-wide on champion change.**
  ``cleanup_forecast_log`` was being called without ``exclude_model_name``,
  so promoting a new champion deleted ALL residuals for the experiment
  (including those already belonging to the incoming champion).
  Published ``_upper_/_lower_`` bands disappeared for ~10 forecast cycles
  while the new buffer rebuilt. Pass the new champion's name through so
  its rows survive.

- **``remove_experiment_covariate`` stripped every same-entity row at
  once.** v2.38.2 enabled the same entity to be configured multiple
  times with distinct ``future_value_key`` values (cloud_coverage +
  temperature), but the remove path still matched on entity_id alone —
  clicking × on one row removed the sibling rows too. Now mirrors the
  ``_same_covariate`` (entity, role, future_attribute, future_value_key)
  tuple matching from the add path; refuses to remove without
  disambiguators when multiple same-entity rows exist.

**High (misleading users in important diagnostic surface):**

- **Verdict-card nominal hardcoded to 0.8.** Users with
  ``conformal_coverage: 0.95`` saw "target 80%" verdict text and a
  worst-bucket selected against the wrong nominal. The whole v2.39.0
  regime-aware coverage feature is now correct for non-default levels.

- **Bootstrap CI is now a paired bootstrap.** Each model drew its own
  fold-id resamples — independent draws across models inflated marginal
  CIs and the "T#1 (tied within fold noise)" chip over-reported ties.
  One shared ``(bootstrap_iters, n_folds)`` index matrix per
  ``_compute_composite_ranks`` call now applies the same fold ids to
  every model so the across-model fold structure is preserved.

- **Empty-column guard's drop-vs-zero-fill decision made on wrong role
  when the same entity had multiple roles.** ``cov_roles_by_name`` was
  a dict overwriting on duplicate key. Now a set per column; drop only
  when every role for that column is lagged, otherwise zero-fill so the
  future-side channel still gets values at inference.

- **Daily-rank "did not complete" mis-classified models whose fold legitimately
  spanned <2 distinct dates.** The completeness check treated empty ``{}``
  (real fold failure) and "metric not computable on this fold"
  identically — silently dropping models that fitted and predicted fine.
  ``run_single_model`` now emits a ``{'__skipped__': True}`` sentinel for
  the latter case and ``_compute_composite_ranks`` distinguishes them,
  ranking the model normally on the folds where the metric is
  computable.

**Medium (correctness / UX gaps):**

- **Daily-only DNCs now surfaced in BenchmarkResult.did_not_complete.**
  ``_dnc_daily`` was destructured with an underscore prefix and
  discarded; users saw ``—`` in the Daily Rank column with no entry in
  the "Did not complete" section explaining why.

- **``classify_covariate_state`` returns ``partial`` for non-weather
  categorical state with a parseable future_attribute.** The docstring
  promised partial for this case but the implementation only returned
  partial when the attribute failed to parse — green chip on rows where
  the lagged channel is all-NaN.

- **Validator no longer false-flags weather entities whose attributes
  are string numerics.** OpenWeatherMap / met.no return
  ``temperature: '16.5'``; the production path tolerates strings via
  ``state_to_float`` and the validator now does too.

- **Non-weather entity with ``future_value_key`` now routes through the
  attribute-history path.** Pre-fix the path was weather-only, so two
  configs of the same non-weather entity with distinct value_keys both
  fell back to ``.state`` and the model trained on two columns of
  identical data.

- **``_cov_column_name`` silent column collision.** When two same-entity
  configs both lack a distinguishing ``future_value_key`` and share the
  default ``future_attribute``, both returned the bare base — second
  ``result[cov_name] = ...`` silently overwrote the first. Now appends
  a positional ``__N`` suffix and warns so the user knows to set
  ``future_value_key`` explicitly.

- **Deep-covariate-analysis labels now match the results dict keys.**
  Label list used ``entity.split('.')[-1]`` while the results dict
  used ``_cov_column_name`` (suffixed) — duplicate blank rows in the
  analysis tab for any multi-key same-entity covariate.

- **Holdout chart no longer silently truncates the trailing
  ``max_horizon-1`` points.** The v2.38.5 alignment fix was correct
  but the chart ended ~24 h short of the holdout window with no
  annotation. Predictions are now NaN-padded to span the full window
  so the right-edge gap is visible.

**Low (CLI-only / cosmetic):**

- **``worst_bucket`` in ``db.get_forecast_coverage`` now selected by
  max|deviation| from a caller-supplied nominal.** Pre-fix it picked
  ``min(coverage)``, so over-covered buckets were never flagged. The
  CLI ``scripts/conformal_coverage_check.py`` reads this field
  directly and now respects ``--nominal`` for the selection. The web
  UI was already overriding correctly.

**Training-speed concerns:**

- **HA history fetch with ``include_attributes=True`` now passes
  ``significant_changes_only``.** Without it, weather entities return
  every recorder tick with their full attribute dict (forecast arrays
  + supported_features etc.) — ~5× larger payloads per covariate per
  fold for the v2.38.4 attribute-history path. The flag only suppresses
  ticks where the state didn't change; the attribute we actually parse
  is unaffected. Lengthens benchmark/retrain cycles measurably less.

- **Coverage tab now builds the ``actuals_grid`` aggregation once per
  request, not three times.** Previously each of the per-lead, overall,
  and breakdown queries rebuilt the same WITH-clause CTE — three
  aggregations per Forecast Accuracy tab load while holding the SQLite
  write lock, which could contend with the training pipeline's
  residual/forecast writes. Materialised once into a TEMP table now.

- **Shared aiohttp session for the validate-covariate endpoint.** The
  v2.38.7 chip fires one ``/api/covariates/validate`` per row on page
  load; pre-fix each handler invocation opened a fresh
  ``ClientSession`` (full TLS handshake), 20 in parallel for a
  20-covariate experiment. Lazy module-level session bound to the
  running loop, reused across requests.

**Visualisation follow-ups (verifying the rankings/results render
correctly):**

- **All-skipped / inf-mean models no longer get fabricated integer
  ranks.** A model that passed the completeness check but produced no
  ranked folds (every fold ``__skipped__`` or all-inf) was still
  assigned a 1/2/3 rank in dict-insertion order. Now demoted to DNC for
  that metric_source so the leaderboard doesn't show meaningless ranks.

- **Daily DNCs surfaced separately from interval DNCs.** A model ranked
  in the per-interval leaderboard but excluded from the daily ranking
  (fold span <2 distinct dates) was being added to the same
  ``did_not_complete`` list — so it appeared BOTH with a rank in the
  main table AND under "Did not complete". Now there's a separate
  ``did_not_complete_daily`` field rendered under the Daily Cumulative
  Accuracy table ("No daily rank: …") with its own explanation.

- **Holdout residual chart no longer paints a spurious spike at the
  NaN-padded tail.** The residual map guarded only the actual side;
  with the v2.39.3 prediction NaN-padding, ``actual - null`` coerces to
  ``actual`` in JS. Now guards both sides → the residual line gaps
  where predictions stop.

- **Cumulative holdout view gaps the prediction line at the tail**
  instead of flat-lining as if the model predicted 0 there
  (``cumsum`` now propagates ``null`` rather than treating it as 0).

Tests: 16 new test cases covering the new contracts (SeasonalNaive
cache rejection + extended-window pw=0 honoured, validator
string-numerics + partial-on-empty-lagged, paired bootstrap rank-sum
invariant, daily DNC sentinel handling, all-skipped → DNC, n_folds=0 →
empty, daily-DNC list separation, ``remove_experiment_covariate``
disambiguation, ``worst_bucket`` selection by max|deviation| at two
different nominal levels). Full unit suite (269 tests) passes.

## 2.39.2

Fixes three false positives / misleading messages in the covariate
validator surfaced after v2.39.1 made the messages visible:

**Lagged-only rows no longer warn about "future attribute 'forecast'".**
`CovariateCfg.future_attribute` defaults to the string `'forecast'`,
so the Jinja template was emitting `data-future-attribute="forecast"`
on every row regardless of role. The validator dutifully tried to
parse a non-existent `forecast` attribute on plain numeric sensors
and flagged them as `partial`. The template now only emits the
future-side data attributes when `role` is `future` / `both`.

**Weather-service forecasts no longer false-flag as partial.** HA
2023.9+ moved hourly/daily/twice_daily forecasts out of state
attributes and into the `weather.get_forecasts` service call. The
resolver already short-circuits to that service (covariates.py:230),
but the validator was still parsing the missing state attribute and
returning `partial`. `classify_covariate_state` now recognises
weather-service future types and treats them as expected-to-work
rather than probing a path that's no longer there.

**Partial message describes the actual lagged source.** When a
categorical-state weather entity uses the attribute-history path for
lagged history, the message used to read `Lagged side ok (last=None)`
— misleading, because `None` isn't the real lagged value. It now
reads `Lagged side ok (via attribute 'uv_index' (current=4.2))` so
the message reflects what the resolver actually does.

Five new tests in `test_validate_covariate_endpoint.py` cover the
weather-service path, the lagged-side wording, and the
legacy-attribute compatibility (Solcast `forecast`, custom
integrations) that the service short-circuit must not break.

## 2.39.1

Surfaces covariate validation messages in the UI so warnings are
actually readable — especially on mobile.

The data-availability chip (✓ / ⚠ / ✗) shipped in v2.38.7 already
explained *why* a row was flagged, but only via the chip's native
`title=` tooltip — invisible on touch devices and easy to miss on
desktop. The Settings → Covariates section now paints the message
inline under each non-ok row (yellow italic for `partial`, red for
`broken`) and adds a small `N errors · M warnings` tally next to
the **Covariates** heading so problems are discoverable at a
glance without scrolling the list.

No backend changes — the `message` string was already returned by
`/api/covariates/validate`. Frontend-only fix in
`web/templates/experiment.html` and `web/static/style.css`.

## 2.39.0

Honest uncertainty on the leaderboard, regime-aware conformal
coverage diagnostics, and like-for-like ranking when some backends
fail.

**Leaderboard now shows rank uncertainty.** Every model's mean rank
ships with a 95% bootstrap confidence interval over fold resamples
(e.g. `mean_rank 1.4 [1.0–2.8]`). When a non-leader's CI overlaps
the rank-1 model's, the leaderboard renders its badge as **T#1**
("tied within fold noise") rather than a discrete number — so you
don't promote a "winner" that's actually within noise of #2. With
small fold counts the CIs are typically wide; that's the point. A
new `docs/RANKING_NOTES.md` explains exactly what the rank does
and does not claim and where the Demšar (2006) framework genuinely
applies vs where it doesn't.

**"Did not complete" handling.** Models that errored on at least
one CV fold are now listed under a separate "Did not complete"
section instead of being assigned last-place on the failed folds.
Previously, a backend that OOM'd on one fold gave every surviving
model a free "win" against it, inflating the leader's apparent
dominance. The comparison is now strictly like-for-like across
the ranked pool.

**Conformal coverage: regime breakdowns.** The Forecast Accuracy
tab now surfaces *where* the published 80% bands are mis-covering,
not just the headline number. The verdict chip now adds e.g.
*"target 80% · under by 4pp · worst: 62% on weekday evenings"*
when a specific hour-of-day or weekday/weekend bucket is materially
off (≥5pp). Hour-of-day uses your HA-configured local time.

**Offline coverage diagnostic.** New
`scripts/conformal_coverage_check.py` reads a `history.db` dump
and prints the full per-hour, per-weekday/weekend, per-lead
breakdown — useful for ad-hoc analysis without a running app.

**Documentation truthfulness fixes.** Code comments and docstrings
in `main.py` and `db.py` previously described the conformal band
as "Adaptive (online) conformal". It is not — the implementation
is split conformal with a rolling residual buffer. Updated to
describe what's actually happening. A new "How the conformal bands
are calibrated" section in `DOCS.md` walks through the buffer
semantics across retrains (same-champion bump vs champion change,
cold-start fallback, when the bands lie).

Tests: 4 new test cases in `tests/unit/test_benchmark.py` cover
the bootstrap CI computation, did-not-complete exclusion, and the
all-fail edge case. Full unit suite (224 tests) passes.

User impact: the leaderboard now tells you when its #1 pick is
actually contested rather than presenting a single winner with
false confidence, the coverage diagnostic tells you which slice
of the week your bands are wrong in, and the conformal docs no
longer over-promise about the band methodology.

## 2.38.7

Adds an auto-validating data-availability chip per covariate row in
the experiment page. Tells you ahead of training whether the
entity is reachable, the lagged side has numeric history, and the
future attribute (if configured) actually parses.

Three states surfaced to the UI:

* **✓ ok** (green) — entity reachable; state numeric (or weather
  attribute-history path works); future attribute parses if
  configured.
* **⚠ partial** (yellow) — lagged side ok but future attribute
  didn't parse (wrong key name or unparseable shape). Chart will
  still get historical data but the future block will be NaN
  through the horizon.
* **✗ broken** (red) — entity missing, state non-numeric without a
  fallback, or future attribute unreachable.

Implementation:

* New ``classify_covariate_state(entity_id, state_obj, future_attribute,
  future_value_key)`` pure function in ``web/app.py`` — the decision
  matrix, no IO.
* New ``GET /api/covariates/validate?entity_id=...&future_attribute=...&future_value_key=...``
  endpoint — thin transport wrapper that does one HA
  ``/api/states/{entity_id}`` call (no history fetch, no service
  probe) and forwards to the classifier.
* Experiment template gains a ``.cov-validate-chip`` span per row,
  initially "…". On ``DOMContentLoaded`` the page kicks off a
  validation per row with an 80 ms stagger (so a 20-covariate
  experiment isn't 20 simultaneous HA calls). The newly added row
  in ``addCovariate``'s success handler also fires a validation.
* Session-scoped JS cache keyed by ``entity|attr|key`` with a 5 min
  TTL — re-opening the experiment page doesn't re-hit HA for
  unchanged rows.
* Data attributes on each row (``data-entity``, ``data-future-attribute``,
  ``data-future-value-key``) carry the config the validator needs;
  the Jinja template and the dynamic-row JS were updated in sync so
  both paths populate them.

The classifier folds in the v2.38.4 attribute-history path: a
weather entity with categorical state but ``future_value_key``
pointing at a numeric attribute (the
``weather.met_office_balsham`` + ``temperature`` pattern) is
classified ok, since the resolver's lagged-side fetch routes
through the attribute path.

Tests: 8 new tests in
``tests/unit/test_validate_covariate_endpoint.py`` pin the
decision matrix end-to-end through the classifier. The pure-function
shape avoids any FastAPI / httpx test-client dependency.

224/225 unit tests pass (one unrelated XGBoost-not-installed failure
pre-existing).

User impact: silent failures you previously only discovered by
reading training logs ("``met_office_balsham__results: 0 raw → 0
aligned``" / "``Failed holdout predictions for nlinear``") now
surface as a yellow / red chip the moment the row is added.

## 2.38.6

Fixes SeasonalNaive returning **zero for every holdout step**
whenever any ``role: future`` covariate was configured — the
"flat blue baseline" symptom users reported on the cumulative
holdout chart.

Root cause: ``create_sliding_windows`` extends each window with
``max(horizon_steps)`` future positions when ``future_features_df``
is supplied (the v2.37+ horizon-anchored covariate path). The
target channel's slot in those future positions is always a
zero placeholder — only the configured future-covariate
channels get populated by name match. SeasonalNaive's
``_per_window_predict`` used ``seq_len = len(target_series)``,
so every lookback ``idx = seq_len + offset`` landed in the
zero-padded future tail of the window. The recursion at
``offset >= 0`` then propagated 0 through the rest of the
horizon. Result: SeasonalNaive predicted 0 everywhere, while
every other model produced normal daily PV bell curves.

This is the same architectural blind spot as v2.38.5 (extended-
window awareness missing on a model), but on a different
backend. NLinear / TiDE took a shape-multiply crash; SeasonalNaive
silently returned zeros.

Fix:

* ``SeasonalNaiveModel.fit`` captures ``past_window_size`` from
  kwargs when ``extended_window=True`` (same kwargs the
  benchmark holdout-neural path already sets at L2937-2938
  of main.py).
* ``_per_window_predict`` confines all index arithmetic to
  ``[0, past_window_size)`` so lookbacks always land in real
  past data.
* ``save``/``load`` persist the new field so a model loaded
  from disk keeps its extended-window awareness.

Tests: 3 new tests in
``tests/unit/test_seasonal_naive_extended_window.py``:

- Legacy past-only mode unchanged (back-compat).
- Extended-window mode produces non-zero predictions — the
  exact "flat blue line" reproducer.
- Recursion across the period boundary keeps propagating real
  values (not zeros).

216/216 unit tests pass (excluding the pre-existing
XGBoost-not-installed failure).

User impact: SeasonalNaive now produces a sensible daily PV
bell curve on the holdout chart instead of flat zero. The
Demšar ranking will reflect the real baseline performance
again — previously it was being compared against a degenerate
"always 0" baseline which any model trivially beat.

Note (separate, not fixed here): the existing
``offset = h + 1 - period`` formula appears off-by-one — for
h=0 it indexes ``past_series[1]`` instead of ``past_series[0]``.
This shifts the predicted seasonal cycle by one position
(15 min at 30-min sampling). Pre-existing behaviour, present
since the model was added. Worth a follow-up PR but not in
scope for the "flat zero" fix.

## 2.38.5

Fixes a holdout-prediction shape crash that surfaces whenever a
neural model with ``future_features_df`` (i.e. any
``role: future`` covariate) is benchmarked.

User-visible symptom:
``WARNING Failed holdout predictions for nlinear: mat1 and mat2
shapes cannot be multiplied (894x2253 and 6528x96)``.

Root cause: ``main.py`` called ``create_sliding_windows`` twice
in the holdout neural path — once for the fit, once for the
predict. The fit-side call used the full ``horizon_steps`` list
(say ``[1..96]``); the predict-side call used
``horizon_steps=[1]`` to "save tail rows". But
``create_sliding_windows`` extends each window by
``max(horizon_steps)`` future positions when
``future_features_df`` is supplied — so the fit-side window was
48+96=144 steps and the predict-side was 48+1=49 steps. Linear-
head backends (NLinear's ``nn.Linear(flat, n_horizons)``, TiDE's
projection) size their weights at fit time off the fit-window
flat input; presenting them a narrower predict-window input
fails the matmul. The exact numbers in the crash:

* fit-flat = 48*46 + 96*45 = **6528** (PF7: future block drops target)
* predict-flat = 48*46 + 1*45 = **2253**
* trained Linear is (6528, 96); predict input is (894, 2253) — crash.

Fix: pass the same ``horizon_steps`` to both calls. Trade-off is
losing the last ``max_horizon - 1`` rows from the holdout tail
(no window can be formed for them — a genuine constraint of
sliding-window prediction, not something we could paper over).
That's strictly preferable to crashing the entire holdout metric.

Also fixed: the length-matching code at line 3027 used ``[-n:]``
which was correct only when n_samples == len(holdout_part) (the
old ``horizon_steps=[1]`` case). With the new fix it must use
``[:n]`` — predictions align with the *first* n holdout
points, not the last. Without this fix the chart would show
correctly-shaped but mis-aligned holdout actuals vs predictions.

Tests: 2 new tests in ``tests/unit/test_holdout_neural_shapes.py``
pin the invariant — fit-side and predict-side
``create_sliding_windows`` outputs must agree on ``shape[1]``
when ``future_features_df`` is set. The negative test reproduces
the 6528 / 2253 mismatch exactly. 235/236 unit tests pass (one
unrelated XGBoost-not-installed failure).

Affects: NLinear, TiDE, any linear-head neural backend benched
on a config with ``role: future`` covariates. Tree models
(XGBoost, RF, LightGBM) and TFT are unaffected (they don't use
this sliding-window code path or the linear-head sizing
contract).

## 2.38.4

Closes the v2.38.3 workaround properly: ``weather.*`` entities used
as future covariates now produce **real historical numeric data**
from HA's recorder, not zero-filled padding. The model can finally
learn the past relationship between Met Office's reported
cloud_coverage / temperature / etc. and the target.

HA's recorder stores each state-change's *full state object*,
including the ``.attributes`` dict. For a weather entity, every
historical record carries ``temperature``, ``cloud_coverage``,
``humidity`` etc. in attributes alongside the categorical ``.state``
field. v2.38.3 only knew how to read ``.state``, so weather entities
produced 0% coverage and the past block was zero-filled. v2.38.4
adds an **attribute-history path** that reads from
``record["attributes"][value_key]`` instead.

* **``HAInterface.get_history``** gains ``include_attributes`` flag.
  Default ``False`` (preserves the v2.37 ``minimal_response`` payload
  optimization). ``True`` drops the flag from the HA query so the
  response carries the full attribute dicts.
* **``normalise_history``** gains ``attribute_key`` argument. When
  set, extracts ``record["attributes"][attribute_key]`` instead of
  ``record["state"]``. Missing attribute keys → NaN (handled by
  the existing resample / ffill / interpolate downstream).
* **``CovariateResolver.fetch_history``** auto-detects the attribute
  path: when the entity is in the ``weather.*`` domain AND
  ``future_value_key`` is set on the cov_cfg, it routes through the
  attribute-history path. Logs ``Fetching covariate history:
  weather.met_office_balsham (attribute=cloud_coverage)`` so it's
  visible.
* **Plumbed through ``_fetch_and_preprocess``**: cov_dict now
  forwards ``future_value_key`` to the resolver. No change to YAML
  schema or UI — the existing value-key picker already populates
  this field.

The v2.38.3 empty-column guard remains as a safety net for any
other zero-coverage cause (a sensor offline, a freshly-added
covariate with no history yet).

Tests: 7 new tests in
``tests/unit/test_attribute_history.py`` pin the contract:
- normalise_history state-path unchanged when no attribute_key
- normalise_history reads from attributes when set
- normalise_history handles missing attribute (NaN, not crash)
- get_history include_attributes flag controls minimal_response
- fetch_history routes weather entity + value_key through
  attribute path
- fetch_history keeps state path for regular numeric sensors
- fetch_history keeps state path for weather without value_key
  (the v2.38.3 empty-column guard still handles that case)

73 tests pass total.

Practical user impact: re-add
``weather.met_office_balsham`` as ``role: future`` with
``future_value_key: cloud_coverage``. The log line should now read
``met_office_balsham__cloud_coverage: 4344 raw → 4344 aligned``
(matching the openweathermap line) instead of the v2.38.3
``→ 0 aligned`` zero-fill case. The model sees real past
cloud_coverage AND real future forecasts — the full TFT/TiDE
pattern.

## 2.38.3

Fixes a v2.38.2 regression that killed experiments when a user
added a ``weather.*`` entity (Met Office DataHub, OpenWeatherMap,
etc.) as a ``role: future`` covariate. ``weather.*`` entities have
a **categorical string state** (``partlycloudy`` / ``sunny`` /
``rainy``), so ``fetch_history`` returned 0 numeric values; the
resulting column was 100% NaN; ``result.dropna()`` then deleted
every row, leaving 0 training samples and skipping the cycle with
``⚠ No samples remaining after preprocessing``.

* **Empty-column guard** in ``_fetch_and_preprocess``: after the
  covariate-fetch loop and before the dropna, detect columns that
  are 100% NaN. For ``role: future`` / ``both`` covariates, fill
  the past with zeros (the future block at inference will still
  receive real values via the forecast attribute / service API).
  For ``role: lagged``, drop the column entirely. Logs a clear
  warning naming the covariate, its role, and the reason so users
  can spot it without diving into the manifest.
* **UI — covariate row metadata**: future-role covariates in the
  Add-Covariate list now show ``attr: hourly`` and
  ``key: cloud_coverage`` chips so users can tell at a glance which
  forecast attribute and value key each covariate is pulling from.
  Especially useful for the v2.38.2 multi-metric pattern where the
  same entity appears multiple times with different keys — the rows
  used to look identical. Both server-rendered rows and JS-appended
  ones (after clicking Add) get the new chips.

Practical user impact: covariates that fail to fetch (or are
configured against entities without numeric state) no longer kill
the cycle. The experiment proceeds with the surviving covariates
and a warning naming the problem. For `weather.*` entities used
as future covariates, the model still gets the forecast signal at
inference even though past values are zero-filled — this is the
common pattern in time-series forecasting when a future-known
covariate has no observable past (e.g. a calendar / event flag
that only exists going forward).

## 2.38.2

Allows the same entity to be configured as multiple covariates with
distinct future-value sources — the natural pattern for weather
entities exposing several useful metrics per ``hourly`` /
``daily`` / ``twice_daily`` service forecast (e.g. ``cloud_coverage``
AND ``temperature`` AND ``humidity`` from one
``weather.met_office_balsham`` entity).

* **Dedup relaxation** (``add_experiment_covariate``): the v2.38.1
  guard rejected any second covariate sharing an entity_id. Now
  duplicates are only flagged when the full
  (entity, role, future_attribute, future_value_key) tuple matches,
  so the same entity can carry distinct future-role covariates for
  different metrics.
* **Column-name disambiguation** (``_cov_column_name``): when the
  same entity appears in multiple covariate configs, each column
  in ``combined`` gets a ``__<value_key>`` suffix (e.g.
  ``met_office_balsham__cloud_coverage``,
  ``met_office_balsham__temperature``). Single-occurrence entities
  keep the bare base name so existing cached models survive the
  upgrade unchanged. The new helper is the single source of truth
  used by every site (train, benchmark holdout, legacy production,
  tree-path inference, neural-path inference).
* **Genuine duplicates still rejected** — clicking Add twice with
  identical attribute + value_key still returns
  "Covariate already exists".

5 new regression tests in ``tests/unit/test_future_covariate_wiring.py``
pin the new helpers (single-entity uses bare name, multi-entity gets
suffix, ``_same_covariate`` allows differing value_keys but blocks
true duplicates, ``_cov_column_name`` without ``all_covs`` is
back-compat).

Practical user impact: in the Add-Covariate UI you can now configure
``weather.met_office_balsham`` once for ``cloud_coverage``, again
for ``temperature``, again for ``humidity`` — three separate
covariates, each a distinct channel in the model.

## 2.38.1

UI follow-up to v2.38.0: the Add-Covariate "Forecast attribute"
dropdown now auto-detects ``weather.get_forecasts`` service-API
forecasts in addition to state-attribute forecasts. v2.38.0 added
backend support but the dropdown only inspected state attributes,
so HA 2023.9+ weather entities (Met Office DataHub, OpenWeatherMap,
AccuWeather, modern met.no) showed empty in the UI — users had to
edit YAML or guess that they should type ``hourly`` manually.

* **Endpoint** ``/api/ha/forecast-attrs``: when entity domain is
  ``weather.*``, reads the entity's ``supported_features`` bitmask
  (1 = daily, 2 = hourly, 4 = twice_daily — HA's
  ``WeatherEntityFeature`` flags). For each supported type, calls
  ``weather.get_forecasts?return_response`` once during the inspect
  to learn the entity's actual numeric forecast keys
  (``temperature``, ``cloud_coverage``, ``humidity``, ``uv_index``,
  …). Returns those alongside the existing attribute-based options
  with a new ``format: "weather-service"`` marker so the frontend
  can label them appropriately.
* **Frontend**: option labels now read ``hourly forecast (weather
  service API)`` / ``detailedForecast (list-of-dict)`` / similar so
  users can tell which mechanism each option uses. Value-key
  dropdown auto-populates from the probe's numeric-key list. No
  schema change to ``addCovariate`` POST body — the existing
  ``future_attribute`` field carries the type name (``hourly`` /
  ``daily`` / ``twice_daily``) and the v2.38.0 resolver routes
  through the service API based on entity domain.

Probe-failure handling: if the service call fails or returns no
forecast (e.g. the entity is unavailable at config time), the
option is still surfaced with an empty key list — users can pick
"Auto" for the value key and let the resolver's runtime fallback
take over.

Practical impact for your Met Office / OpenWeatherMap setup:

1. Open Add-Covariate → search for ``weather.met_office_balsham``.
2. Select Role: **Future**.
3. **Forecast attribute** dropdown now shows:
   - ``hourly forecast (weather service API)``
   - ``daily forecast (weather service API)``
   - ``twice_daily forecast (weather service API)``
4. Pick ``hourly`` → **Value key** dropdown populates with the
   entity's actual keys (``cloud_coverage``, ``temperature``,
   ``humidity``, etc.).
5. Click Add. The covariate is saved with
   ``future_attribute: hourly`` and the right
   ``future_value_key`` — no YAML editing needed.

## 2.38.0

Minor-version cap on the v2.37 future-covariate feature arc. The
six v2.37.x patch releases between 2.37.2 and 2.37.7 each shipped
new functionality (debug bundles, idle_value field, end-to-end
future-covariate wiring, future_aux_head for the 3 broken
backends, dynamic UI dropdown, weather-service API support) — that
cumulative scope warrants a minor bump per semver, and v2.38.0 is
a natural milestone now that the future-covariate feature is
complete and validated end-to-end (train → benchmark → predict →
publish, across all 17 backends, with three covariate source
shapes: state-attribute, service-API, and HA-recorded numeric
sensor).

This release adds support for the HA 2023.9+
``weather.get_forecasts`` service API so modern weather integrations
work as future covariates.

**Background**: HA 2023.9 deprecated the ``forecast`` attribute on
``weather.*`` entities. Forecasts moved to a separate
``weather.get_forecasts`` service call that returns the array via
``service_response`` instead of state attributes. Integrations that
shipped or migrated after the switch — **Met Office DataHub**,
**OpenWeatherMap**, **AccuWeather**, modern **met.no** —
no longer expose ``attributes.forecast``, so the v2.37.5+ future-
covariate plumbing returned NaN for them. The Add-Covariate UI
correctly showed no forecast-attribute options (because there
weren't any).

**Fix**: when the covariate resolver sees an entity in the
``weather.*`` domain with ``future_attribute`` set to one of
``hourly`` / ``daily`` / ``twice_daily``, it now calls
``POST /api/services/weather/get_forecasts?return_response`` with
the requested type, parses the per-entity ``service_response``,
and feeds the same downstream alignment path as the attribute
route. Legacy attribute-exposing integrations (Solcast's
``detailedForecast``, Forecast.Solar, older met.no, custom
integrations) are unchanged — the resolver falls through to the
existing attribute fetch for any other ``future_attribute`` value.

**UI** (no change in this release): the existing v2.37.7 dropdown
still reads from state attributes. Modern weather entities will
show no forecast options there. Set ``future_attribute`` manually
to one of ``hourly`` / ``daily`` / ``twice_daily`` in the YAML
(via the Forecast attribute field — type it in if the dropdown
doesn't surface it). A follow-up release will wire
``supported_features`` bitmask detection into the dropdown so
modern weather integrations also get one-click setup.

**Note on `weather.*` entities and `role: lagged`**: a weather
entity's *state* is a categorical string
(``partlycloudy`` / ``sunny`` / ``rainy``), not numeric. Using a
weather entity with ``role: lagged`` returns non-numeric values
that the resolver drops. For **historical** numeric temperature /
cloud / wind, use the per-metric ``sensor.*`` entities the
integration exposes alongside the ``weather.*`` entity (e.g.
``sensor.met_office_balsham_temperature``). Documented in
DOCS.md.

**Practical user impact**: Met Office DataHub users can now use
``weather.met_office_<location>`` as a `role: future` covariate
for their PV experiment:

```yaml
covariates:
  - entity: weather.met_office_balsham
    role: future
    future_attribute: hourly
    future_value_key: cloud_coverage
```

The same recipe works for OpenWeatherMap, AccuWeather, and any
other HA 2023.9+ weather integration that supports the service
API (``supported_features`` bitmask non-zero).

## 2.37.7

Closes the remaining gaps from v2.37.5/v2.37.6: the three backends
that previously dropped user future covariates now consume them via
an auxiliary head, and the UI gains dynamic dropdowns for the
``future_attribute`` / ``future_value_key`` fields so users no
longer have to edit YAML to use a Solcast / Forecast.Solar
covariate.

### Backend fixes — N-BEATS, N-HiTS, iTransformer

Each of these backends explicitly sliced ``x[:, :past_window_size, :]``
in their forward pass (v2.37 PF4 / PF6), dropping the future block.
v2.37.5+ wrote user future-covariate values into those future
positions; all three ignored them silently. The v2.37.6 config-load
warning surfaced this, but didn't fix it.

v2.37.7 adds an **auxiliary future-feature head** to each backend.
The head is a small MLP that:

* Reads the future block as a flat ``(batch, future_window_size * n_channels)`` tensor.
* Projects it to a per-horizon adjustment that's **added** to the
  basis / encoder output.
* Has its **final layer zero-initialised**, so at training step 0
  the model is behaviourally identical to v2.37.6 — no surprise
  regression for users upgrading with existing checkpoints (the
  optimiser has to actively learn to use the future signal).

Same pattern across all three backends — small, focused
contribution paths that preserve each architecture's identity
(N-BEATS / N-HiTS keep past-only basis decomposition; iTransformer
keeps the PF6 past-only channel embedding).

**Result**: every neural backend in the registry now consumes user
future covariates. The v2.37.6 config-load warning has been
removed.

### UI — dynamic future_attribute dropdown

The Add-Covariate form previously omitted ``future_attribute`` and
``future_value_key`` — the tooltip directed users to YAML. For
Solcast / Forecast.Solar entities (which use ``detailedForecast``
not the default ``forecast``), this meant the UI silently created
broken covariates.

v2.37.7 adds two new fields to the form, shown only when role is
**Future** or **Both**:

* **Forecast attribute** — a dropdown populated from the entity's
  actual HA attributes. Each candidate attribute is inspected on
  selection: list-of-dict format (Solcast, Met.no weather) or
  flat date-keyed dict format (Forecast.Solar). Out-of-range or
  non-numeric attributes are filtered out so users only see
  attributes that look like a real forecast array.
* **Value key** — for list-of-dict attributes, populated from the
  first entry's numeric keys (e.g. ``pv_estimate``, ``temperature``,
  ``pv_estimate90``). Hidden for date-dict attributes which don't
  need it.

Both default to "Auto" so the form still works the same way for
common cases (Met.no's ``weather.*`` entities → ``Auto / Auto``
just works). Power users picking Solcast get ``detailedForecast``
+ ``pv_estimate`` as a one-click selection. The matching backend
endpoint ``/api/ha/forecast-attrs`` inspects the entity's live
state — no YAML hardcoding.

### Tests

* 9 new regression tests in
  ``tests/unit/test_future_aux_head_backends.py`` pin the
  three-backend contract: zero-init at step 0 produces output
  identical to past-only (no upgrade regressions), permuting the
  future block AFTER bumping aux-head weights changes the output
  (wiring works), and the head is ``None`` in legacy non-extended
  mode (no extra parameters for past-only users).
* Existing ``test_future_covariate_wiring.py`` updated — the
  v2.37.6 warning-fires test is replaced with a
  warning-must-not-fire test pinning the new
  every-backend-supports-future-covariates contract.

All tests pass.

### Practical impact

* Re-run benchmarks with future covariates enabled — N-BEATS,
  N-HiTS, and iTransformer are now in the running (likely still
  behind TiDE / NLinear / DLinear for PV-style problems, but no
  longer information-starved).
* Adding a Solcast covariate via the UI now Just Works without YAML
  editing — the dropdown shows ``detailedForecast`` and
  ``pv_estimate`` as the obvious picks.
* If you previously enabled N-BEATS / N-HiTS / iTransformer with a
  future covariate, the model needs a fresh retrain after upgrading
  — old cached weights pre-date the aux head.

## 2.37.6

Completes the future-covariate wiring rolled out in v2.37.5 and
flags the three backends that can't consume it.

**v2.37.5 left two gaps:**

1. The benchmark / holdout path (``_generate_holdout_predictions``)
   trained candidates on a **past-window-only** architecture even
   when production used extended-window. Comparing TiDE vs LightGBM
   in the benchmark was therefore unfair to TiDE on TWO axes —
   missing future covariates AND missing the future-position
   extension architecture entirely. The "best model" picked from
   the benchmark could differ from what'd actually win in production.
2. The legacy non-cached production-inference path
   (``_run_production_inference``) — used on first-ever run before
   any cached model exists — had the same gap.

**This release wires both:**

* **Benchmark holdout training**: now builds the same extended-window
  seq_X as production (window_size past + future_periods future)
  with temporal + solar physics + user future-covariate values at
  horizon positions. Logs ``Holdout future covariates (horizon-
  aware): [...]`` when active.
* **Benchmark holdout inference**: rebuilds the matching extended
  window from ``combined_holdout`` so ``predict_sequence`` reads
  the same channel layout it was trained against.
* **Legacy production inference**: training + inference both extended,
  user future covariates fetched from HA's forecast attribute at
  inference time exactly like the cached path.
* **New shared helper** ``_collect_train_future_covariates``: single
  source of truth for "which user covariates land at horizon
  positions during training". Used by all 4 training sites
  (cached production, benchmark holdout, legacy production
  inference, and any future caller). Pure function, easy to test.

**Backend audit + warning for 3 broken backends:**

A predict_sequence audit of all 17 neural backends revealed that
three — **N-BEATS**, **N-HiTS**, and **iTransformer** — explicitly
slice their input to ``x[:, :past_window_size, :]`` in their
forward pass (v2.37 PF4 / PF6 commits). They cannot consume future
covariates regardless of how perfectly we wire them in.

* **Config-load warning**: ``ExperimentCfg.__post_init__`` now warns
  when ``models_enabled`` includes any of these three AND the
  experiment has at least one ``role: future`` covariate. The
  warning names the bad models, names the covariate entities, and
  confirms which models WILL use them. Surfaces in the addon log
  immediately at startup so users know not to expect a Solcast
  benefit on these specific backends.
* **The 14 working backends** (CNN, Crossformer, DLinear, FITS,
  GRU, LSTM, NLinear, PatchTST, SparseTSF, TFT, TiDE, TimeMixer,
  TimesNet, TSMixer) all properly consume future positions and
  benefit from the v2.37.5 + v2.37.6 wiring.
* **Fix for the 3 broken backends** requires per-backend
  forward-pass changes (removing the slice + handling the
  variable input length). Tracked as future work; not in scope
  for this release.

**Practical impact:**
* Re-run any benchmark on an experiment with a future-role
  covariate — TiDE / NLinear / DLinear should now score
  comparably to (or better than) LightGBM, which previously had
  an information advantage. The "best model" pick from the
  benchmark is now apples-to-apples with production training.
* If your YAML enables N-BEATS / N-HiTS / iTransformer alongside
  a Solcast future covariate, expect a warning at startup and
  unchanged behaviour from those three (they'll see only past lags).

Tests: 5 new regression tests in
``tests/unit/test_future_covariate_wiring.py`` pin the helper
contract and the warning behaviour (with-future-cov + broken
backend → warning; without future cov OR only compatible backends
→ no warning). Total 58 tests pass.

## 2.37.5

Closes the **future-covariate asymmetry** between tree and neural
backends. Tree models (LightGBM / XGBoost / CatBoost) routed user-
configured ``role: future`` covariates into every recursive forecast
step via ``future_cov_values`` — so a Solcast forecast value at
horizon h directly informed the prediction at h. Neural extended-
window backends (NLinear, DLinear, TSMixer, TiDE, etc.) had the
same plumbing point (``compute_known_future_features`` accepts a
``future_covariate_values`` parameter) but **the caller never
passed it in**, so user future covariates only reached the model
as past-window lags. This systematically biased benchmarks: any
head-to-head comparison with a strong future-known covariate
(Solcast, met.no weather, anything from a forecast service)
favoured tree models on information grounds, not architecture.

* **Training side** (``_retrain_and_cache``): builds a
  ``future_cov_for_neural`` dict from ``exp_cfg.covariates`` with
  ``role`` in ``{future, both}``, mapping each to its in-sample
  historical observations from ``combined``. Passes through
  ``compute_known_future_features.future_covariate_values``. The
  "future" positions of each training window now carry the actual
  past observations of those covariates at those timestamps —
  giving the neural head per-horizon ground-truth signal during
  training.
* **Inference side** (``_forecast_with_cached``): mirrors the
  tree-path pattern — for each cached
  ``seq_kwargs.future_covariate_names`` entry, calls
  ``covariate_resolver.fetch_future`` against HA's forecast
  attribute, reindexes to the inference ``future_index``, and
  passes the dict through to ``compute_known_future_features``.
  The 96 horizon positions of the inference window now read the
  forecast values directly. Falls back to channel-zero if the
  forecast attribute is missing / all-NaN; channel-parity guard
  surfaces a warning if a future covariate was cached but removed
  from YAML since.
* **Cache persistence**: ``cache_meta.json`` gains a
  ``future_covariate_names`` field listing just the user-future
  channels (subset of ``future_feature_cols``). Deterministic
  columns (temporal, solar physics) are recomputed at inference
  from the future_index alone; this list tells the inference path
  which channels need a HA fetch. Survives addon restart.
* **Channel parity preserved**: nothing changes about channel
  ordering — user future covariates appear in the same channel
  slots they always did (raw_cov_cols path). The wiring only
  affects WHAT VALUES populate the future positions of those
  slots. The channel-parity guard at inference still catches any
  drift between cached and live channel sets.

Practical user impact:
* If you're running a benchmark with Solcast / weather covariates
  configured as ``role: future``, **rerun it after v2.37.5** —
  the neural-vs-tree gap will narrow (potentially flip) because
  the neural models are no longer information-starved.
* If you've been on tree models for PV forecasting because neural
  models seemed worse, give NLinear / DLinear / TiDE a fresh try
  with a Solcast covariate — TiDE in particular is designed for
  this exact pattern (long horizon + future-known covariates) and
  should now show its strength.

Tests: 5 new regression tests in
``tests/unit/test_future_covariate_wiring.py`` pin the contract —
training-side future covariate placement, inference-side
placement from a synthetic forecast, the no-fcv legacy path
(future positions stay zero), reindex + ffill handling of sparse
covariate observations, multi-covariate channel isolation. All
56 tests pass (13 integration + 17 config + 7 debug + 14 idle +
5 future-cov).

## 2.37.4

Generalises the v2.37.3 solar night-fill to a per-experiment opt-in
field that covers the wider class of intermittent sensors. HA's
delta-storage recorder drops every row from training when ANY sensor
sits at a constant value (or goes ``unavailable``) for >90 min —
that's solar at night (fixed in v2.37.3) but also EV chargers
between sessions, solar pumps in winter, idle batteries, holiday
absences. v2.37.4 lets users declare the idle value once per
experiment so the addon can fill those gaps with the physically
correct value instead of dropping them.

* **New per-experiment field — ``idle_value: float | None``**
  (default ``None``). When set and ``target_is_nonnegative=True``:
    - For solar / irradiance targets (``clear_sky_ghi`` or
      ``sun_elevation`` present in result): fills NaN slots where
      the sun is below the horizon with ``idle_value`` instead of
      the default 0.0. Lets users with a measurable inverter
      standby override the night-fill default.
    - For non-solar non-negative targets (no physics features):
      fills ALL remaining NaN slots with ``idle_value``. Covers
      EV chargers, solar pumps, idle batteries.
* **Default behaviour unchanged**: ``idle_value=None`` preserves
  the v2.37.3 solar night-fill exactly (NaN → 0 where physics says
  night). Users on v2.37.3 see no change without opting in.
* **Gate is intentional**: signed targets (net grid flow,
  temperature delta) still drop NaN regardless of ``idle_value``,
  because "what should -5 W mean when the sensor is offline" has
  no universal answer.
* **Settings UI field** — Target group → "Idle value", number
  input next to Max increment. Empty string clears the override
  back to default. Info-tip explains the trade-off (silently
  masks real outages on non-solar paths) so users opt in
  knowingly.
* **Renamed helper ``_apply_solar_night_fill`` →
  ``_apply_idle_value_fill``** to reflect the broader scope. The
  old name remains as a module-level alias so v2.37.3-pinned
  imports / tests / docs still resolve.
* **6 new regression tests** (``tests/unit/test_solar_night_fill.py``
  ``test_idle_value_*``) pin the contract: EV-style fill-all-NaN
  path, default-None preserves drop behaviour for non-solar,
  solar override of the 0.0 default, signed-target gate, negative
  idle values allowed, alias back-compat. 14 tests total (8 from
  v2.37.3 + 6 new).
* **Logged line generalised**: replaces the v2.37.3
  ``Solar night-time fill: N → 0`` line with
  ``Idle fill: N (sun below horizon → 0 / all idle NaN → X)``
  so the source of the fill is visible.

Known scope limit: this addresses the **target series** gap-fill.
Covariates with the same intermittent pattern (e.g. an EV charge
state used as a `role: lagged` covariate) still get the original
drop-on-NaN behaviour at their fetch stage. Covariate-side fill
is a separate concern tracked separately.

## 2.37.3

Fixes the v2.37 neural-PV "daytime-only training set" regression
diagnosed via the v2.37.2 debug bundle. User's ``optimised_solar``
forecast was predicting 0.3-0.7 kW at 23:00 with the daily peak
phase-shifted to 18:00 — symptoms of a model that had never seen
night-time data during training.

* **Root cause**: HA's recorder is delta-storage based. When a PV
  sensor sits at 0 W from sunset to sunrise it records one
  transition and then nothing, or reports ``unavailable`` (parsed as
  NaN) while the inverter sleeps. The default
  ``gap_handling='interpolate'`` only fills gaps up to
  ``gap_max_minutes`` (90), so the 10-14h night gap stays NaN and
  the downstream ``result.dropna()`` deletes every night-time row.
  The user's debug bundle confirmed: 2085 of 2088 training rows had
  ``sun_elevation >= 0`` — hours 21-03 were completely absent from
  the training index. The model never learned "PV = 0 when sun is
  below the horizon", so at inference it predicted non-zero across
  the night and a phase-shifted bell curve.
* **Fix — solar night-time zero-fill**: new
  ``_apply_solar_night_fill`` helper runs after solar physics
  features are computed and before the dropna step. For experiments
  with ``target_is_nonnegative=True`` AND ``clear_sky_ghi`` (or
  ``sun_elevation``) in the result columns, fills NaN ``y`` slots
  with 0 wherever the physics says the sun is below the horizon.
  Where ``clear_sky_ghi`` is present it's used as the gate (matches
  the existing physics-gate in ``features.py`` line 192); otherwise
  falls back to ``sun_elevation < -0.833°`` (standard astronomical
  horizon, accounts for atmospheric refraction).
* **log_transform=True works without inverse**: log(1+0) = 0, so
  writing 0.0 is correct whether the target series is in raw or
  log-transformed space — no inverse needed.
* **Daytime NaN is preserved**: a sensor outage during daylight
  (clear_sky_ghi > 0 but y is NaN) is NOT filled — it stays NaN and
  drops out at the dropna step. Silent fill of genuine sensor
  failures would mask real data quality problems.
* **Gated on ``target_is_nonnegative=True``**: signed targets (net
  grid flow, temperature delta) keep the original drop-on-NaN
  behaviour. Only solar / irradiance-style experiments are touched.
* **Log line surfaces the fill**: each retrain now logs ``Solar
  night-time fill: N NaN rows (sun below horizon) → 0`` so users
  can confirm the fix took effect — typical doubling of training
  rows from ~2088 to ~4272 on a 30-min PV experiment.
* **8 regression tests** (``tests/unit/test_solar_night_fill.py``)
  pin the contract: gate short-circuits, ghi/sun_elev fallback,
  daytime NaN preserved, idempotence, full 24-hour coverage after
  fill+dropna, log_transform compatibility.

Known scope limit: the wider issue — that ANY sensor going
"unavailable" for >90 min gets dropped from training (EV charger
when idle, battery flow when sleeping, etc.) — is separate and
will be addressed in a follow-up. Solar is the only case where the
correct fill value is deterministically known.

## 2.37.2

Diagnostic surface for retrain regressions. Synthetic integration tests
can pin a code path's correctness but cannot reproduce a regression
that only shows up against a specific user's data — the 50% dropna
loss, the time-of-day peak drift, the magnitude over-prediction. This
release adds an opt-in per-experiment toggle that dumps the exact
production training inputs and outputs to disk so a maintainer can
inspect them offline.

* **New per-experiment setting — ``debug_save_training_dumps``**: when
  enabled, every retrain writes a bundle to
  ``<config_dir>/debug/<experiment>/<UTC-timestamp>/`` containing:
    - ``meta.json`` — hyperparameters, target stats, channel order,
      data range, ``seq_kwargs`` (PF1–PF10 flags), addon version,
      forecast range after the paired inference call;
    - ``training.parquet`` — full ``combined`` dataframe (target +
      features) that fed ``create_sliding_windows`` / ``model.fit``;
    - ``sliding_window.npz`` — neural-path ``seq_X`` / ``seq_y`` /
      ``channel_names`` (omitted for tree-only experiments);
    - ``forecast.parquet`` — the immediate post-retrain forecast with
      raw model output AND post-log-inverse physical values.
* **Bounded disk usage**: rotation keeps the 5 most recent bundles
  per experiment; older timestamp directories are auto-deleted at the
  start of each new dump. ~0.5–2 MB per bundle on a 30-min PV target.
* **Bundle location ``<config_dir>/debug/``**: sits next to
  ``mlfl.yaml`` so HA's File Editor / Samba / SSH add-ons can browse
  the dumps without an extra path mapping. To share a bundle with a
  maintainer: zip the timestamp directory and attach.
* **Default OFF**: no overhead for ordinary users. Toggle lives in
  Settings → Diagnostics → "Save training dumps". Turn on, trigger
  one retrain, share the bundle, turn off.
* **7 regression tests** (``tests/unit/test_debug_dump.py``) pin the
  bundle contract: file set, meta-JSON keys, forecast→training dir
  pairing, no-op when no training dump pending, rotation, parquet
  engine fallback to CSV.
* **Logged-line discoverability**: each successful dump logs
  ``Debug dump: training → <path>`` and ``Debug dump: forecast →
  <path>`` so the addon log surfaces where the bundles landed.

No production code path changes — the dumper is a pure observer that
runs after ``model.fit`` succeeds and after the cached forecast
finalises ``y_pred``. Errors are swallowed and logged so a failing
dump can never break the retrain.

## 2.37.1

Hotfix for the v2.37.0 PF8 regression that caused the auto-resolver
to pick ``relu`` for non-negative targets, producing a flat-zero
NLinear forecast in production via the classic "dying ReLU"
collapse on the PF2 anchor add-back. Reverts that one branch to
``softplus`` (the pre-v2.37 default) for all auto non-negative
cases — both ``source_is_cumulative=True`` and the new v2.37
``target_is_nonnegative=True`` flag.

* **PF8 revert — auto resolves to softplus for non-negative
  targets**: ``_resolve_output_activation`` returns ``'softplus'``
  whenever ``source_is_cumulative=True`` OR
  ``target_is_nonnegative=True``, restoring the pre-v2.37 default
  for cumulative and extending it to the new instantaneous
  non-negative flag. The original PF8 ReLU pick was synthetic-
  validated against small-magnitude cumulative kWh intervals on
  the theory that softplus's +log(2)≈0.69 physical-space floor
  would dominate predictions; in production, the dying-ReLU
  collapse on extended-window NLinear's anchor add-back proved a
  far larger regression. Softplus has non-zero gradient everywhere
  and its physical floor (~1 unit) is negligible for any non-trivial
  target. The existing
  ``test_cumulative_source_picks_softplus`` (added 2026-05-16, the
  day before v2.37) had been pinning the correct behaviour all
  along — PF8 silently broke it and the next PR's CI surfaced it.
* **Settings UI — Non-negative target toggle**: new toggle next to
  "Daily reset" exposes the ``target_is_nonnegative`` flag, so
  users with non-cumulative non-negative targets (PV power,
  irradiance, instantaneous demand) can opt into PF8/PF9 without
  hand-editing YAML.
* **PF1-PF10 retrain diagnostic log line**: ``_retrain_and_cache``
  now logs every neural retrain's resolved ``past_window_size``,
  ``extended_window``, ``output_activation``, ``daily_loss_weight``,
  ``use_revin``, ``learning_rate``, ``optimiser``, ``log_transform``,
  ``source_is_cumulative``, ``target_is_nonnegative``. First place
  to look when investigating a post-v2.37 forecast that still
  misbehaves — confirms exactly which PF1-PF10 path was entered.
* **Persistent log archive — bumped retention**: rotating
  ``mlfl.log`` is now 10 MB × 5 backups (was 5 × 5), plus a new
  ``mlfl-daily.log`` with UTC daily rotation kept for 14 days.
  Total disk footprint bounded. Suppress the daily archive with
  ``MLFL_DAILY_LOG_KEEP=0``.

No retrain is forced by this hotfix; users who already toggled
``Non-negative target`` ON in v2.37.0 should retrain after
upgrading to pick up the softplus activation. Users on the legacy
``output_activation: linear`` path (the v2.36.x default for
non-cumulative targets) are unaffected.

## 2.37.0

Implements every fix identified by the neural-PV investigation
(``docs/investigations/2026-05-neural-pv.md``). The v2.36.0 extended-
window mode reproduced the user's bizarre PV forecasts on synthetic
data with ground truth, and the same synthetic harness was used to
verify the fixes.

* **PF1 — RevIN past-only stats**: ``_RevIN.normalize`` accepts a new
  ``past_window_size`` kwarg. When provided, per-window mean/std are
  computed over the past slice only, undoing the 50% mean bias that
  the future-position zero-target padding induces. Applied across all
  12 backends that use RevIN. Backwards-compatible (None = legacy
  whole-window behaviour).
* **PF2 — NLinear past-end anchor**: NLinear's "subtract the last
  value, re-add it" trick now anchors on ``x[:, past_window_size - 1,
  target_channel]`` (the last past observation) instead of the literal
  last row (a future zero in v2.36 extended mode).
* **PF2-variant — TFT past-end query**: TFT's "last step as query"
  trick at ``tft_backend.py:179`` was degenerate for the same reason
  as PF2; now uses the past-end position.
* **PF3 — past-only attention mask**: LSTM, GRU temporal attention;
  TFT multi-head attention; PatchTST, Crossformer transformer
  encoders. Future-position scores are pushed to ``-inf`` before
  softmax so the attention can't read absolute time from
  zero-target slots. Restored LSTM peak hour from 1 AM to noon.
* **PF4 — N-BEATS / N-HiTS past-only backcast**: the doubly-residual
  stack now operates on the past slice only. Without this the
  backcast learned trivial zeros for the future block and the
  forecast residual collapsed.
* **PF5 — CNN past-only pool mask**: CNN's learnable pool weights are
  masked at future positions before softmax so the pooled context
  excludes zero-target slots.
* **PF6 — iTransformer past-only channel embed**: the per-channel
  ``Linear(seq_len, d_model)`` embedder now consumes only the past
  slice so the target channel's token isn't biased low.
* **PF7 — head-input mask**: NLinear, DLinear, TSMixer drop the
  future-position target-channel slots from their flat head input
  (always zero by construction; including them only adds
  head-input variance imbalance). Other linear-head backends are
  already restored to flatness ≈ 1.0 by PF1 alone.
* **PF8 — ReLU default for non-negative targets**: new
  ``ExperimentCfg.target_is_nonnegative`` flag mirrors
  ``source_is_cumulative``; when either is True and
  ``output_activation='auto'``, the resolver picks ReLU instead of
  linear. (Pre-v2.37 the auto path picked softplus for cumulative
  targets — this was reverted to ReLU because softplus's
  +log(2)≈0.69 floor in physical space catastrophically biased
  low-magnitude cumulative interval targets. Verified on synthetic
  cumulative-with-daily-reset data: softplus produced 900%
  daily-total error vs ReLU's 30%.) LSTM keeps its zscore default
  (its specialised path). Set ``target_is_nonnegative`` for PV
  power, irradiance, household demand intervals.
* **PF9 — daily_loss_weight default for non-negative targets**: new
  ``_resolve_daily_loss_weight()`` defaults the cumulative-trajectory
  loss weight to 0.5 (from 0.0) for non-negative neural targets when
  the user hasn't explicitly set it. Penalises systematic bias more
  aggressively than per-interval MSE alone.

**Cache invalidation**: ``schema_version`` bumped from 1 to 2. Old
v2.36-era caches are silently ignored on startup and a fresh
benchmark + retrain is scheduled — necessary because the old models
were trained against the biased RevIN / degenerate anchors that
PF1-PF9 fix, so loading them would just re-publish the broken
forecasts.

Verification (``make_realistic_pv(0)``, extended_window=True default,
40 training epochs):

| backend       | v2.36 flat / peak | v2.37 flat / peak |
| ---           |             ---:  |             ---:  |
| nlinear       | 0.82 / 12         | **1.33 / 12** ✓   |
| dlinear       | 0.66 / 12         | **1.06 / 11** ✓   |
| sparsetsf     | 0.71 / 12         | **1.08 / 11** ✓   |
| fits          | 0.50 / 11         | **0.93 / 12** ✓   |
| tsmixer       | 0.78 / 12         | **1.19 / 12** ✓   |
| timemixer     | 0.69 / 12         | **0.99 / 12** ✓   |
| tide          | 0.70 / 12         | **1.69 / 11** ✓   |
| lstm          | 0.05 / 1 (!)      | **0.81 / 11** ✓   |
| gru           | 0.05 / 8          | **0.99 / 12** ✓   |
| cnn           | 0.03 / 10         | **0.98 / 12** ✓   |
| patchtst      | 0.13 / 13         | **0.79 / 12** ✓   |
| itransformer  | 0.58 / 12         | **0.74 / 10** ✓   |
| crossformer   | 0.25 / 11         | **0.95 / 12** ✓   |
| timesnet      | 0.57 / 12         | **1.20 / 13** ✓   |
| tft           | 0.74 / 12         | **0.88 / 13** ✓   |
| nbeats        | 0.11 / 10         | **1.45 / 12** (needs 100 epochs) |
| nhits         | 0.08 / 3          | 0.10 / 7 (still broken — architectural follow-up) |

True peak hour is 12 (noon, UTC) on this dataset; ideal flatness is
1.0. Every backend except N-HiTS is recognisably correct after PF1-PF9.

Cumulative-with-daily-reset verification: a second synthetic dataset
``make_cumulative_daily_reset(0)`` (interval-form household demand,
peaks at 19:00 evening, resets at midnight) was added. 16/17 backends
produce correct peak hour (19) with daily_total_mape ≤ 38%; the only
exception is SparseTSF which has peak=15 (its period-aware
architecture struggles with the morning+evening dual-peak pattern).
N-BEATS and N-HiTS — the two architecturally-different backends — now
ALSO produce correct shapes on cumulative interval data
(flat 0.74-0.78 with peak=19 ✓), which they couldn't do on PV.

Tests: ``tests/unit/test_neural_pv_regression.py`` adds 12 tests
covering each PF and the cumulative-with-daily-reset dataset
invariants. All 12 PASS.

## 2.36.0

A user-reported PV forecast on NLinear produced a spurious peak at
~3 AM — i.e. in the middle of the night, when the sun is physically
below the horizon. Confirmed across multiple neural backends and
across solar-features-on / solar-features-off retrains: NLinear and
SparseTSF produced smeared shapes with night-time peaks, LSTM and
CNN collapsed to a flat constant ≈ training mean. Same root cause:
a multi-horizon neural head with a single linear projection from a
pooled / flattened past-window encoder cannot disambiguate "horizon
h" from "absolute hour at h", because h corresponds to different
absolute hours across windows that end at different times. The
weights for h are forced into a phase-smeared compromise (linear
models) or the model gives up and predicts the unconditional mean
(LSTM/CNN). The tree-model production path doesn't have this
problem — its recursive inference loop already feeds each future
horizon row with its own pvlib-computed `sun_elevation` /
`clear_sky_ghi` and `hour_sin/cos`. This release closes the
neural-path asymmetry.

### Added

- **`features.compute_known_future_features`** — helper that returns
  a DataFrame of features that are deterministically known for any
  future timestamp: temporal (hour_sin/cos, dow_sin/cos, is_weekend),
  holiday indicator (when a country is configured), and solar
  physics (`sun_elevation`, `clear_sky_ghi` via pvlib) when site
  lat/lon is available and the experiment opted into the solar
  covariate channels. Forecast-style covariates whose future
  values are knowable (Solcast etc.) can be threaded through too.

- **Extended-window mode for `create_sliding_windows` and
  `build_inference_window`.** Both accept an optional
  `future_features_df`. When provided, each window is appended
  with `n_horizons` future positions; channels matching
  `future_features_df` columns are populated at those future
  positions (target is left as zero — no leakage), all other
  channels at future positions are zero. The resulting per-sample
  tensor has shape `(window_size + n_horizons, n_channels)`. The
  model's multi-horizon head can now read each horizon's own
  time-anchored signal directly from its corresponding future
  position, instead of having to phase-disambiguate from the
  past window alone.

- **`extended_window` flag + `past_window_size` + `future_feature_cols`
  in the cached `seq_kwargs` and persisted `cache_meta.json`.** Lets
  the inference path (and the post-restart restore path) reproduce
  the same past/future split the trainer used. Caches written before
  this release lack the flag and take the legacy past-only path —
  they keep working unchanged until they're retrained on the normal
  schedule.

### Changed

- **`_retrain_and_cache` neural path** now computes future-known
  features for `combined.index` (temporal always; `sun_elevation` /
  `clear_sky_ghi` only when the experiment has the matching solar
  covariates enabled and lat/lon is configured) and passes them
  through to `create_sliding_windows`. Per-channel parity with the
  inference path is preserved because both sides go through
  `compute_known_future_features` with the same arguments and
  consume the same channel-name list.

- **`_forecast_with_cached` neural path** now detects the
  `extended_window` flag on the cached `seq_kwargs`, computes
  future-known features for the inference horizon timestamps
  (`ds_future`), and passes them to `build_inference_window` to
  produce the same extended-shape tensor the model was trained on.
  Old caches without the flag take the original past-only path,
  so the upgrade is non-breaking.

### Trade-offs and known effects

- **NLinear's last-value anchor trick degrades when the cache is
  retrained on extended windows.** The trick subtracts
  `x[:, -1:, target_channel]` from every step and adds it back to
  the output, anchoring the prediction on the most recent
  observation. With future-position target slots set to zero (no
  leakage), `last_val` is zero and the anchor becomes a no-op —
  effectively NLinear collapses to a plain linear head over the
  larger input. This is the deliberate trade. The
  per-horizon-time-anchored signal that future positions now carry
  is a much stronger inductive bias than the level-anchor for
  strongly-periodic targets, but if your previous NLinear ranking
  came specifically from the anchor trick it may no longer be the
  bench winner under the new scheme. Re-benchmark to confirm.

- **Holdout / benchmark training paths are not touched in this
  release** — they remain on the past-only window scheme. So the
  bench-winner ranking that picks the production model is computed
  under the old scheme, while production training and inference run
  under the new scheme. A follow-up will plumb extended windows
  through holdout for ranking consistency.

## 2.35.3

The user-reported "odd predictions" turned out to be a real bug in the
shared neural inference plumbing — not a model-side issue. Confirmed
when switching from NLinear to SparseTSF produced the same misaligned
shape: two architectures, same symptom, so the cause lives in the
code both paths share.

### Fixed

- **Off-by-one in the neural inference window construction.** The
  production forecast path (`_forecast_with_cached`) and the legacy
  inference path (`_run_production_inference`) both built the
  inference window by calling
  `create_sliding_windows(tail_df, ..., horizon_steps=[1])` on the
  last `window_size + 1` rows of the combined dataframe. The intent
  was "give me the most recent window"; the actual behaviour was
  "give me a window ending at `combined.iloc[-2]`, because the final
  row is reserved as the (unused-at-inference) h=1 y-label". The
  model was therefore fed a window whose last timestep was one
  interval *before* `last_ts`, while `_publish_forecast_sensors`
  timestamped its predictions starting at `last_ts + 1 interval`.
  Net effect: every published forecast value was the model's
  prediction for the slot one interval *earlier* than the
  timestamp it was published under. Visible as a time-of-day skew
  in the dashboard — most pronounced on dense 96-horizon backends
  (NLinear, SparseTSF, DLinear) where a user can eyeball where the
  peak sits relative to the labelled axis.

  Fix: new helper `features.build_inference_window` constructs a
  single `(1, window_size, n_channels)` tensor directly from
  `df.iloc[-window_size:]`, so the window's last timestep IS
  `last_ts`. Channel ordering matches `create_sliding_windows`
  exactly, so the v2.35.2 channel-parity guard still compares
  apples to apples between the cached training names and the
  inference-rebuilt names. Both inference paths are switched over.

### Added

- **Unit tests pinning the inference-window contract.**
  `TestInferenceWindowAlignment` covers four invariants:
  - the window ends at `df.index[-1]` (catches the original bug),
  - channel ordering matches `create_sliding_windows` exactly
    (keeps the parity guard's cached names directly comparable),
  - the window's last temporal-feature row encodes the hour at
    `df.index[-1]` (the model's anchor must be the actual
    most-recent timestamp, not one step earlier — the failure
    mode of the original bug),
  - too-few-rows raises a clear `ValueError`.

## 2.35.2

A correctness fix triggered by user-reported "odd predictions" from
NLinear: forecasts with a peak at the wrong hour of day even though
NLinear ranked best in the benchmark.

### Fixed

- **Channel-parity guard between training and inference for neural
  models.** The production retrain path
  (`_retrain_and_cache`) stored the trained `sequence_data` tensor in
  the in-memory cache but did **not** store the per-channel meaning
  (`channel_names` returned by `create_sliding_windows`). At forecast
  time, `_forecast_with_cached` re-derived the channel ordering from
  a freshly-fetched dataframe with no verification that it matched
  what the model was trained on. The benchmark-holdout path stored
  `channel_names` correctly — only the production cache was missing
  it.

  Consequence: if anything ever shifted the column order between
  train and inference — a transient empty covariate fetch leaving
  a hole at one tick, a covariate added or removed in Settings
  since the last retrain, a future change to `build_features`
  output order — the model would silently consume mis-labelled
  channels (e.g. `sun_elevation` arriving in the slot the network
  learned for `clear_sky_ghi`) and the published forecast would
  look wrong in oddly time-specific ways. There was no error
  raised because every channel still had the right *shape*, just
  the wrong *meaning*. NLinear is the most exposed backend to this
  because a single linear layer has no capacity to compensate for
  swapped channels, but every neural backend was affected.

  Fix: store `channel_names` in `seq_kwargs` at production retrain
  (matching what the benchmark-holdout already did), persist it in
  `cache_meta.json` so the guard survives restarts, restore it into
  the in-memory cache on startup, and verify in
  `_forecast_with_cached` that the inference-time ordering matches
  the cached training-time ordering. On mismatch the forecast cycle
  logs a clear `ERROR` and skips publishing for one tick rather
  than emit a confidently-wrong sensor value — the next retrain
  rebuilds the cache with the current ordering.

  Existing on-disk caches without `channel_names` keep working
  (the guard treats absent metadata as "no check" rather than
  failing) and are upgraded transparently on the next retrain.

### Changed

- **Output-activation tooltip clarified for log-transformed
  targets.** The Experiment Settings tooltip for *Output activation
  (neural)* now explicitly suggests picking **Softplus** when Log
  transform is on and you want the activation to own the
  non-negativity contract instead of relying on the post-hoc
  `max(0, expm1(.))` clamp. The `auto` default remains unchanged
  — `softplus` for cumulative sources, `linear` otherwise — but
  the trade-off is documented so users debugging spurious
  predictions on log-transformed PV / energy targets have a
  documented next thing to try.

### Added

- **Unit tests pinning the output-activation resolution rules and
  the channel-ordering filter.** `TestResolveOutputActivation`
  fixes the auto-rule contract (LSTM→zscore, cumulative→softplus,
  log_transform-alone stays linear, explicit choice wins) so the
  default can't drift undetected. `TestNLinear` adds a
  softplus-floor sanity check. `TestChannelParityFilter` exercises
  the production filter directly: same-df-columns → same channel
  order, and re-ordered covariates → divergent channel order
  (which is exactly the scenario the new runtime guard catches).

## 2.35.0

A UX-focused minor release. Two themes: making the Covariate
Analysis tab safer to act on when removing more than one covariate,
and making the Experiment Settings fully self-documenting so a new
user never has to guess what a field does.

### Added

- **Covariate Analysis — bulk-remove and stale-result banner.**
  User report: *"if I have several covariates and I remove more than
  one, what happens when I click remove on those?"* Three changes:
  - Each row with a Remove button now also has a **bulk** checkbox.
    Tick several and confirm them together via a single red action
    bar that appears above the table — one dialog covers the whole
    batch, sequential POSTs with per-entity success/failure
    reporting.
  - After **any** removal (per-row or bulk), an amber banner above
    the table flags the displayed percentages as stale (they were
    computed against the previous covariate set). The banner has a
    one-click **Re-run Covariate Analysis** button that fires the
    full analysis with the current model selection — no need to
    scroll back up to the controls.
  - The post-remove toast was reworded from "re-run pipeline to see
    effect" → "re-run Covariate Analysis to update the table". The
    old wording was easy to misread as "re-run the benchmark
    pipeline" (a different button entirely).

  Section tooltip rewritten to describe the new flow.

- **Help icons (`?`) on every editable field in the Settings tab.**
  Audit pass added concise tooltips to 14 previously-undocumented
  controls:
  - Source / Forecast: **Cumulative source**, **Daily reset**
  - Data & Forecast: **History (days)**, **Interval (minutes)**,
    **Log transform**
  - Robustness: **Gap max (minutes)**, **Outlier quantile**
  - Training: **CV strategy**, **CV folds**, **Production metric**
  - Covariate add form: **Entity**, **Aggregation**, **Scale**,
    **Binary**
  - Load Subtract add form: **Entity**

  Every tooltip names the default, when to change it, and the
  trade-off — so a new user never has to guess from the field name
  alone.

### Changed

- **Data sanity report tooltip clarifies its scope.** Made the
  current "target only, covariates skipped" behaviour explicit so
  users know the report doesn't audit their covariate entities. A
  follow-up task is tracked to extend the report to cover covariates
  too. *(Superseded in 2.35.1 — covariates are now checked.)*

## 2.35.1

Follow-ups to 2.35.0 surfaced when reviewing the same screens.

### Added

- **Data sanity report now covers covariates too.** Previously the
  pre-flight sanity check only fetched and analysed the target
  entity — covariates were silently skipped, so a broken covariate
  (gap, dead sensor, wrong units) only surfaced an hour into a
  benchmark. The per-entity analysis is factored into a helper that
  runs once for the target and once per configured covariate; the
  report now includes a `covariates` array with per-entity verdict,
  coverage %, gap stats, recorder freshness, NaN rate, and any
  warnings. A covariate alert escalates the experiment-level verdict
  to "warning" (covariates are non-fatal — training still runs — but
  worth knowing). Frontend renders each covariate as a compact card
  with a verdict-coloured left border below the target's detail
  rows. The data-report tooltip is updated to reflect the new scope.

### Changed

- **Output activation tooltip rewritten + dropdown expanded.** The
  old tip lumped <code>zscore</code> with deprecated <code>relu</code>/<code>exp</code>
  as "YAML-only footguns", which was wrong: <code>zscore</code> is a
  current, valid option (the Auto choice for LSTM and the
  per-horizon normalisation strategy honoured by every PyTorch
  neural backend). The tip now describes <code>zscore</code> properly,
  notes the RevIN interaction (zscore becomes a no-op when
  <code>use_revin=True</code>, the default on most backends), and is
  honest that <i>relu</i> and <i>exp</i> remain YAML-only footguns.
  <code>zscore</code> is also now a regular dropdown option, so users
  can pick it without editing YAML.

## 2.34.7

### Fixed

- **"Show RMSE & bias" toggle on the lead-time chart didn't visibly
  update.** User report. The onchange handler called the full
  `loadForecastAccuracy()` flow, which re-rendered every tile on the
  Forecast Accuracy tab (model badge, retrain chips, lead-time
  chart, revision improvement, etc.). The cached-fetch path should
  have made this instant, but the toggle was reported as a no-op.

  Refactored the lead-time chart render into a stand-alone
  `_renderLeadTimeChart(data)` helper. The toggle now calls a new
  `onLeadTimeDetailToggle()` wrapper that re-renders directly off
  `_lastAccuracyData` without re-fetching or re-rendering anything
  else — single Plotly call, no chained side-effects. Falls back to
  a full load if the initial fetch hasn't completed yet.

## 2.34.6

Public-release readiness pass driven by the release-gate review.

### Removed

- **Auto-generated Lovelace dashboard YAML** (`dashboard.py`, the
  `/dashboard_yaml` download route, the System-page **Download
  Lovelace dashboard** button, and the per-experiment **Lovelace
  YAML** button). The generated YAML referenced sensors that never
  existed (`_point`, `_curve`), used an attribute shape that didn't
  match the published sensors (parallel `timestamps` / `values`
  arrays vs the actual `[{datetime, value}, …]` list), and embedded a
  `http://homeassistant.local:5052/...` link that has been dead
  since v2.30.0 retired direct port exposure. Fix scope outweighed
  the feature's value; removed in full. The `[DASH]` log-phase tag
  is gone with the module.
- **`AUDIT_PROMPT.md`** at the repo root — internal dev artefact that
  outlived the cleanup of four similar files in v2.33.2.

### Changed

- **Bundled `mlfl.yaml` example moved from author-specific entities
  to a generic household-load template** (`sensor.power_consumption_w`,
  one optional outside-temperature covariate, `seasonal_naive` added
  to the starter models). Existing users are unaffected — the s6
  init script (`rootfs/.../init-mlforecastlab/run`) only copies the
  bundled default when `${ADDON_CONFIG}/mlfl.yaml` is absent.
- **Documentation, code comments, UI tooltips, and tests
  genericised.** Mixergy / iBoost / Predbat illustrative references
  are now described by the underlying device category (solar-divert
  dump, tank-heating automation, switchable load, etc.). No behaviour
  change.
- **README support section** spells out that this is the first
  public release of a previously private project and that the
  add-on is maintained on a best-effort basis.

### Added

- **`SECURITY.md`** at the repo root routes vulnerability reports
  through GitHub's private vulnerability reporting with a best-effort
  acknowledgement window.

## 2.34.5

### Fixed

- **`UnboundLocalError: local variable 'df' referenced before
  assignment`** when starting a benchmark with an empty SQLite
  actuals cache (or when every cached row fell outside
  `days_history`). `_fetch_and_preprocess` only bound `df` inside
  the cache-hit branch but referenced it unconditionally on the
  delta-fetch path. Regression from the v2.33.1 database-flag
  removal cleanup. Now initialises `df` to an empty DataFrame
  before the cache lookup so the full-fetch fallback works.

### Changed

- **Forecast Accuracy "?" tooltips refreshed across the tab.** Stale
  copy fixed and new inline help added so the recent v2.34.x UX
  changes are discoverable without reading the changelog:
  - **Trajectory chart tooltip** no longer describes the horizontal
    "Actual" line that v2.34.1 replaced — now mentions the diamond
    marker and dashed Y-reference.
  - **Convergence chart tooltip** points users to the "View as"
    toggle and explains that the white actual is drawn in the same
    space as the predictions (deltas or cumulative).
  - **Lead-time chart tooltip** mentions the per-cohort overlay and
    what RMSE / bias add when toggled on.
  - **Run-to-run disagreement tooltip** explains the multi-cohort
    overlay and how chip-strip clicks narrow / expand the view.
  - **Retrain-history chip strip tooltip** documents the click
    interactions added in v2.34.3 (single click to focus, click
    again to deselect, Cmd/Ctrl-click overlay, Shift-click range).

### Added

- Inline "?" tooltips on the **Window**, **Show last**, **View as**,
  and **Sort by** controls on the Forecast Accuracy tab — concise
  descriptions of what each selector does and when to pick each
  option, so users don't have to hover the chart heading to find out.
- Hover titles on the **Run Pipeline** and **Publish** action buttons
  spelling out what each one actually does (benchmark every enabled
  model with walk-forward CV vs. promote winner + switch to
  production + start publishing sensors).
- "?" tooltips on the system **Training CPU cores**, **Process
  priority**, and **Timezone** fields. Explains what "All available"
  costs the Pi, what Unix nice values do, and what the timezone is
  used for (daily-reset midnight boundary).

## 2.34.4

### Fixed

- **Convergence chart X-axis labels bunched on the left** when the
  actuals window was much shorter than the prediction window (every
  freshly-retrained cohort, until enough past targets accumulate).
  `_tzTicks` used `dates[0]` and `dates[last]` for the tick range,
  but the convergence chart concatenates `fanX + actuals.targets`
  unsorted — so `dates[last]` was the last actual (early in the
  window) rather than the last fanX target (end of horizon). All 8
  ticks ended up inside a few-hour slice. Now uses min/max across
  the array, which also nan-guards each timestamp.

- **Trajectory chart had the same bunching** for a different reason:
  forecasts are clustered in the issued-at history, but the chart
  extends the X-axis to include `target_dt` (where the Actual
  marker and target-time line sit). `_tzTicks` only saw the
  forecasts and emitted ticks for that narrow range. Now passes
  `xs + [targetTs]` so the ticks span the full visible X range.

### Changed

- **Lead-time chart legend moved out of the plot area.** With
  20+ cohort traces (e.g. after 3 weeks of daily retrains widened
  to all versions), the previous top-left-inside legend covered the
  data. Now anchored to the right outside the plot, matching the
  convergence chart's pattern.

## 2.34.3

### Changed

- **Retrain-history chip strip — easier escape from a focused
  cohort back to the multi-cohort overlay.** Three small fixes for
  the same reported friction ("I selected one chip and can't get
  back to all"):
  - Clicking the currently-focused chip now toggles it off (back to
    "show all"), matching how every chip-style filter elsewhere
    behaves.
  - The "× Clear filter" pill is renamed "× Show all cohorts" so the
    escape hatch reads as what it does.
  - The summary line under the strip now spells out the escape
    ("click the chip again (or 'Show all') to restore overlay")
    when a single cohort is focused.

## 2.34.2

### Fixed

- **Forecast convergence (fan chart) — measured actuals were plotted
  in the wrong space for cumulative sources.** `get_forecast_evolution`
  returned raw values from the actuals table; `forecast_log` stores
  predictions in delta space when `source_is_cumulative=True`. With
  both rendered on the same Y-axis, the "Measured (actual)" line
  climbed the day's accumulation (0 → 30 % in the user's report) while
  predictions hugged zero. Backend now diffs actuals via SQL LAG with
  an adjacency guard, and clamps negative deltas to zero so a midnight
  reset doesn't produce a huge negative spike. The cumulative-view
  path also benefits: it was previously double-cumulating (raw
  actuals → `toCumulative` on the client) and now correctly
  integrates the deltas.

  Three new tests cover (a) cumulative source returns deltas,
  (b) non-cumulative source passes through, (c) reset clamps to 0.

### Changed

- **"Actual at target" marker on the trajectory chart was too
  prominent.** v2.34.1 introduced a size-14 white star; tone it down
  to a size-9 diamond on user feedback — still anchored to
  `(target_dt, value)`, just less attention-grabbing.

## 2.34.1

### Fixed

- **Forecast Accuracy stayed empty for the first ~horizon-worth of
  time after every retrain.** `probe_forecast_rows` — the cheap
  EXISTS check the widening ladder uses to decide whether the strict
  `(model, version)` cohort has any data — only checked
  `issued_at >= cutoff`. Immediately after a retrain, the new
  cohort has a full horizon's worth of predictions logged (e.g. 96
  rows at 30-min × 48), but every one targets the future, so the
  INNER JOIN against actuals comes back empty. The probe said
  "rows exist" → ladder didn't widen → user saw an empty lead-time
  curve with `empty_reason: "no_overlap"` even though 24 older
  versions of the same model would have rendered fine. Probe now
  also requires `target_dt <= now`, so a future-only cohort
  correctly falls back to the model-only (or all-models) filter.

  Pinned with a new probe test (`test_future_only_targets_do_not_register`).

### Changed

- **"How predictions converge on a single moment" — Actual is now
  a marker anchored to the target moment, not a horizontal line.**
  The previous solid white line spanned the X-axis (the issued-at
  range, often several days), which read as "the actual demand was
  0.8 across all those days". The actual is one observation at one
  moment; a marker placed at (target_dt, value) plus a dim
  dashed Y-reference (showlegend off) makes that geometry explicit.
  Label changed to "Actual at target (per-interval demand)" for
  the same reason. The green dotted target-time vertical line is
  unchanged.

### Removed

- Bare lead-in paragraph above the Forecast Accuracy heading
  ("Is your forecast working well? …") — the heading and the
  diagnostic cards explain themselves.

## 2.34.0

Multi-cohort Forecast Accuracy view. The user asked "how does the
view handle a tune or a new model used for the experiment? both
should appear on the graphs such as the run-to-run disagreement,
and swing values updated to not use predictions of other types of
model". Two parts, in that order:

### Fixed

- **Run-to-run stability, coverage, and conformal calibration no
  longer pool predictions across cohorts.** Each query
  (`get_forecast_stability` × 2 sub-queries, `get_forecast_coverage`
  × 2, `get_conformal_quantiles`) now pre-aggregates per
  `(model_name, model_version)` cohort and picks ONE winner per
  output row (target_dt / lead_bucket / day) via `ROW_NUMBER()` —
  most rows wins, ties broken by most recent `model_version`.
  Previously, when the caller's filter widened (e.g. `?model=catboost`
  with no version pinned, or `?model=all`), a single target_dt could
  pool predictions from old + new weight regimes; a retrain mid-
  window made the std balloon as "noise" that was actually just
  "different weights making different predictions".

  The endpoint-level version-widening fallback on `/forecast-stability`
  is removed in the same pass — the SQL self-protects now, so
  widening was redundant. The failure mode it tried to mask
  (cohort warming up with <2 cycles per target) is now an explicit
  `empty_reason: "cohort_warming_up"` on the response.

  Three existing tests were documenting the bug as if it were the
  spec ("Mixed: huge CV from pooling two regimes" etc.) — updated
  to assert the new invariant: mixed-filter results equal a single
  cohort's value, never the pooled mean. Two new tests added to
  pin "larger cohort wins" as the v2.34.0 contract.

### Added

- **Multi-cohort overlay on the lead-time + run-to-run stability
  charts.** When the response carries ≥2 cohorts (i.e. the user has
  not pinned a single cohort via the chip strip and the window
  contains a retrain or model switch), each chart renders ONE line
  per cohort instead of the single dominant-cohort line. Visible at
  a glance: how today's retrain's error curve compares to yesterday's,
  or how the new lightgbm champion's stability tracks against the
  old catboost one.

  - Lead-time chart: per-cohort MAE traces, each grouped via
    `legendgroup` so clicking a legend entry toggles the line.
    RMSE / Bias stay single-cohort (pooled) to keep the chart
    readable when "Show detail" is on — N × 3 traces is too many.
  - Stability std chart: per-cohort std-vs-target_dt lines, time-
    tiled (cohorts naturally occupy non-overlapping windows so
    legibility stays high).
  - Convergence (fan) chart, trajectory chart, and stability daily
    bars stay single-cohort: stacked translucent fans become
    illegible quickly, trajectory is already per-target.

- **Per-cohort colour mapping.** New `_cohortColor(model_name,
  model_version)` hashes the compound key into `PLOT_COLORWAY` —
  stable across page reloads. Used by the lead-time chart, the
  stability std chart, AND the retrain-history chip strip, so a
  cohort wears the same colour on its chip dot and its lines on
  every chart. The chip → trace correspondence is now visual, not
  just textual.

- **Multi-select retrain-history chip strip.** Three click modes:
  - Plain click → focus on this cohort only (legacy single-select)
  - Cmd/Ctrl-click → toggle cohort in the overlay set
  - Shift-click → range select from last-clicked chip to this one

  Selection lives in `_activeFilter.versions` as a Set keyed by
  `"model_name::model_version"` and serialises to URL as
  `?versions=key1,key2`. Single-select (`?model=&version=`) and
  multi-select (`?versions=`) are mutually exclusive in the URL.

  The backend stays oblivious to multi-select: when `versions=` is
  the active mode, no model filter is sent and the backend returns
  the full `cohorts` array; the frontend filters client-side via
  `_filterCohortsBySelection()`. Means re-toggling chips doesn't
  pay the SQL round-trip — re-renders directly from the cached
  payload. Range selects feel instant even on a Pi 5.

- **`empty_reason: "cohort_warming_up"`** on `/forecast-stability`
  when the current cohort has fewer than the minimum cycles
  (replaces the silently-removed widening fallback). Verdict chip
  shows "need ≥2 cycles per moment" — same copy as before, now
  the canonical path.

### Backend

- `get_forecast_accuracy` response carries a new `cohorts` array
  (one entry per `(model_name, model_version)`, each with its own
  `lead_time_curve` block). Skipped when caller pinned a single
  cohort — only one cohort to chart, redundant work.

- `get_forecast_stability` response carries the same `cohorts`
  array shape, each entry with its own `per_timestep` block.

- Pooled `lead_time_curve` and `per_timestep` scalars stay in
  the response unchanged — the verdict-card chip + calibration
  tile + older frontends keep working.

- The `forecast_vals` CTE in `get_forecast_accuracy` now carries
  `model_version` so the per-cohort lead-time GROUP BY can
  reference it without rejoining `forecast_log`.

212 tests pass.

## 2.33.2

Docs cleanup. Two parts:

- **`DOCS.md` per-experiment config table** still listed the
  `database` field that v2.33.1 removed from `ExperimentCfg`.
  Dropped the row and amended the neighbouring `max_age`
  description to call out that the SQLite actuals cache is
  always-on now (`max_age` is the only knob that controls
  retention).

- **Removed four stale internal audit documents** from the repo
  root: `SURVEY.md`, `ML_AUDIT.md`, `IMPROVEMENTS.md`,
  `DOC_SURVEY.md`. These were working-document outputs of earlier
  audit rounds (the ML preprocessing/loss audit merged in v2.31.0
  via PRs #9–#11, plus the docs-restructure audit of the same
  cycle). The proposals they contained have all landed; the docs
  themselves are now stale snapshots that confuse new readers of
  the repo. Git history preserves them — anyone wanting the
  original audit text can `git log --diff-filter=D --name-only`
  to find the deletion commit. Total ~1,615 lines removed.

No code changes.

## 2.33.1

Removes the `database` per-experiment flag and improves the dashboard
card's destructive-action affordance. Both noticed while debugging a
real "Forecast Accuracy view is empty" report against v2.33.0 — the
underlying cause was `database: false` silently breaking every query
that joins forecasts against the per-target actuals cache. Rather
than build self-diagnosis surfaces around the foot-gun, we removed
the foot-gun.

### Removed

- **`database: bool` per-experiment config field.** Actuals caching
  is now unconditional. The flag was a foot-gun: every Forecast
  Accuracy query (`get_forecast_accuracy`, `get_forecast_stability`,
  `get_forecast_coverage`, `get_conformal_quantiles`) reads from
  the cached per-entity actuals table, and `store_history` only
  populated that table when the flag was on. Disabling it broke
  the entire tab with no signal. The cost of always-on caching
  is negligible (~72 KB / experiment for a 30-day window, bounded
  by `cleanup(table_name, oldest)`); no reason to keep the choice.

  - `ExperimentCfg.database` removed from the dataclass.
  - Both gates removed from `_fetch_and_preprocess` (read path at
    `main.py:1007`, write path at `main.py:1254`).
  - **Auto-migration**: yamls carrying `database: true` or
    `database: false` have the field silently stripped on first
    load by `load_config`, alongside the existing `horizons_minutes`
    cleanup. A single INFO line records the migration.
  - Removed from the `/api/experiment-settings` editable whitelist.
  - "Cache actuals" toggle removed from the Settings → Target
    section.

### Changed

- **Dashboard card delete button moved to the top-right corner
  of each experiment card.** Previously a faint `×` sat in the
  footer-actions row between "View Details" and "Retrain" /
  "Publishing" — it read as a separator rather than a destructive
  control. New `.btn-card-delete` is absolutely positioned in the
  card corner, always-visible with a faint red tint, and brightens
  fully on hover. The mode badge in the card header reserves
  matching right-margin so the corner X never collides with it on
  narrow viewports.

## 2.33.0

Audit-driven UX, accessibility, and correctness pass on the
Forecast Accuracy and Covariate Analysis views — produced from a
static audit of both surfaces (see `VIEW_SURVEY.md` and
`VIEW_AUDIT.md`). 30 distinct fixes; companion to the v2.31.0 wave
which addressed adjacent concerns (lazy-load Plotly, retrain
history chips, calibration progress tile).

### Fixed

- **Forecast convergence (fan) chart silently mixed predictions
  from rotated-out models.** `db.get_forecast_evolution` and its
  `/forecast-evolution` endpoint had no `model_name` /
  `model_version` parameters — every other accuracy endpoint
  applies the default-then-widen ladder via
  `_resolve_model_filter`, but this one ran unfiltered. After any
  retrain (which stamps a new `model_version`) or champion
  rotation, the "Latest run" yellow line and the fan band could
  span two different model weight regimes on the same axes, and
  the chart looked perfectly fine. Threaded the filter through
  both the issuance query and the per-cycle rows query; the
  endpoint now uses the same probe-then-widen ladder and surfaces
  a `model_fallback` payload when it widens.

- **Forecast Accuracy → Trajectory could spin in an unbounded
  re-fetch loop.** When the server's `data.target_dt` didn't
  match the `want` the client had just picked (e.g. the target
  was purged from `forecast_log` between the dropdown render and
  the row fetch, or a string-comparison mismatch), the loader
  called itself again with no attempt cap. Added a one-shot
  `window._trajRetries` guard — after one retry the loader bails,
  purges the chart, and shows "Couldn't load the requested
  target — pick another from the dropdown" rather than hammering
  the Pi.

- **Trajectory error path left a stale chart visible behind the
  empty state text.** The empty-data branch already called
  `Plotly.purge('accuracy-trajectory-chart')`; the `.catch` did
  not. A previously-rendered trajectory could therefore stay on
  screen while the page told the user "no targets yet". Added the
  same purge to the error path.

- **Apply Best & Retrain scored covariate configs against the
  mean MAE across every tested model — not the model actually
  publishing forecasts.** When the user ran the analysis with the
  default "All models" dropdown selection, the winning row could
  be one that helped a weak model but hurt the production model,
  and the button silently applied it. Scoring now reads from
  `experiment.selected_model || experiment.best_model` first; falls
  back to the cross-model mean only when the production model's
  cells are all NaN (e.g. it was disabled when the analysis ran).
  Response carries a `score_source` field describing which rule
  applied.

- **Calibration ETA on the verdict tile was off by up to 48×.**
  The new "Calibrating: 3 of 10 residuals · ~12 h to bands" tile
  (v2.31.0) divided remaining residuals by `forecast_every_minutes`
  assuming one residual per cycle. Every cycle actually produces
  `future_periods` residuals as actuals arrive. Now divides by
  `future_periods` so the displayed ETA is meaningful rather
  than wildly conservative.

### Changed

- **Forecast Accuracy now shows a loading skeleton on every chart
  card.** Tab-open fires four concurrent fetches against backend
  SQL that, against a mature `forecast_log` on a Pi 5, plausibly
  take 1–3 s combined. Previously the page rendered empty
  `<div>`s for the duration — empty looked identical to broken
  and to "still computing". A `.chart-loading` CSS class with a
  cyan shimmer + "Computing…" caption is toggled on each chart
  host via `_setCardLoading()`; the class is cleared on settle
  (success or empty). Honours `prefers-reduced-motion`.

- **All Forecast Accuracy fetches now share a JSON helper with
  one-shot retry, memoisation, and a user-facing failure toast.**
  `accuracyFetch()` retries once with 2 s backoff before giving
  up (covers most ingress hiccups in the 4-fetch storm on tab
  open), caches the last 8 responses per URL so flipping between
  30 and 90 days doesn't re-run the full SQL pipeline twice, and
  on persistent failure emits a non-blocking `mlfl.toast`. The
  cache is invalidated on `pageshow` (bfcache restore), on
  `pipeline_end` SSE events, and on `popstate` after a chip
  filter changes the URL.

- **Plotly bundle now ships compressed and cacheable.** Added
  `GZipMiddleware(minimum_size=1024)` to the FastAPI app — cuts
  the 1.05 MB Plotly Basic bundle to roughly 330 kB on the wire.
  `StaticFiles` is now subclassed to set
  `Cache-Control: public, max-age=31536000, immutable` on
  `*.min.js` / `*.min.css` (versions are pinned by filename at
  build time, safe to long-cache) and a 5-minute cache on
  everything else so addon updates ship without hard refreshes.
  Stacks with the lazy-loader landed in v2.31.0.

- **Apply Best & Retrain now requires confirmation.** The action
  rewrites `mlfl.yaml` and triggers an immediate retrain — a
  single misclick previously did both with no recovery path. The
  click now opens an `mlfl.confirm` (matching the pattern already
  used by the inline Remove button on the same tab) that names
  the production model the config will be optimised for and
  warns that the change overwrites existing covariate settings.

- **Covariate analysis cell colour convention reversed to follow
  the value direction.** The macro previously coloured cells
  green when `change_pct > 2` (because "this row says the
  covariate is useful") and red when `change_pct < -2` — a
  positive number meaning "MAE got worse" was therefore green,
  which the reader cannot rely on grasping. Cells now follow the
  value: positive (worse) → red, negative (better) → green. A
  legend strip above the table spells out the new convention,
  the section info-tip lists the colour mapping explicitly, and
  "MAE" is framed as "typical forecast error" in user-facing
  strings (column headers keep the technical names with hover
  tooltips expanding them).

- **Covariate analysis recommendations rewritten.** Per-model
  "performs better without covariates" recs now suppress
  contradictory per-covariate "Keep X" recs for the same model
  (they used to fire in the same panel and confuse the reader).
  Per-covariate threshold symmetric on ±2% to match the inline
  Remove button. Every text string replaces unexpanded "MAE"
  with "typical forecast error" so HA-user readers don't need
  ML training to parse the verdicts. A single-fold caveat is
  appended to every populated recommendations block. When the
  rules collectively fire nothing, the panel now emits an
  informational rec ("No covariate had a strong effect… your
  configuration looks well-balanced") rather than showing blank
  — empty recs no longer look like an analysis failure.

- **Stability "Daily-total disagreement by day" chart encoding
  now works for colourblind users.** Previously the bars carried
  the green/amber/red CV-threshold mapping via hue alone, with
  no legend. Added two dotted threshold lines at CV = 10%
  (green) and CV = 25% (red) with right-anchored annotations
  ("Stable", "Unstable") — encoding is now readable by position
  alone; colour is a redundant channel.

- **Covariate Analysis tab finally has a cold-start empty state.**
  When the benchmark has run but the covariate analysis has not,
  the tab previously rendered just the controls — a model
  picker and a button with no surrounding context. Added an
  empty-state card under the controls explaining what the
  analysis does, how many model runs it will perform for the
  experiment's current configuration, and what to expect for
  duration.

- **Lowest cross-model mean MAE row is now highlighted as best,
  not the baseline.** The `.best-model` row highlight is
  computed server-side in Jinja and applied to the actually-best
  row; the baseline row gets a separate "baseline" pill in the
  first column so the comparison anchor is still visible.

- **Forecast convergence card and trajectory card now surface
  their empty / error state inline.** Previously the convergence
  card vanished silently (`wrap.style.display = 'none'`) on any
  error or empty payload — users had no way to tell whether the
  diagnostic had been removed or had failed. Both cards now stay
  visible with a one-line message ("Convergence chart needs at
  least two forecast cycles in this window." / "Could not load
  convergence data — <reason>") inside the chart host.

- **Forecast Accuracy empty-state distinguishes the three real
  conditions behind it.** The backend `/forecast-accuracy`
  response now carries an `empty_reason` field — `ok` /
  `warming_up` / `no_actuals` / `no_overlap` — derived from row
  counts and table existence, so the UI can show the right hint
  without substring-matching on error strings. Each branch shows
  a distinct title + hint pair.

- **Retrain history chip strip now identifies the active filter,
  filters softly, caps overflow, and honours timezone.** Four
  related improvements to the v2.31.0 strip: (1) the chip
  matching the currently-rendered cohort gets a cyan outline +
  "active" badge so the user can see what's filtering the view;
  (2) chip click uses `history.pushState` + an in-page reload
  rather than navigating away, preserving page state and any
  warm caches; (3) experiments with >12 retrains in the window
  get a "Show all (N)" expand toggle so the strip doesn't dominate
  the page; (4) chip timestamps now use `Intl.DateTimeFormat`
  with the active accuracyTz setting instead of the hand-rolled
  UTC slice. A "× Clear filter" pill appears alongside the chips
  when a filter is active.

- **Calibration progress now scales `min_samples` with the
  coverage level.** Previously hardcoded at 10. Higher coverage
  levels need more residuals for stable quantile estimation —
  now uses `max(10, ceil(10 / (1 - level)))` so a 0.95-coverage
  experiment waits for 200 residuals before publishing bands,
  not 10.

- **Run Covariate Analysis button matches the rest of the page
  palette.** Switched from `btn-purple` (used nowhere else in
  the Forecast Accuracy / Covariate Analysis tabs) to the
  standard `btn-primary` (cyan), with a small `⚗` glyph to
  retain the "heavyweight analysis" cue.

- **Covariate analysis polling dropped from 5 s to 2 s.** The
  per-row JSON is small and the reduced lag visibly improves
  the Run button → completion handoff.

- **Trajectory "miss X" annotation includes units now.**
  "miss 0.85" is hard to interpret without knowing the sensor's
  typical range; the dropdown now appends `EXP_UNITS` and, when
  available, a "% of typical" suffix derived from the cached
  accuracy payload's `typical_interval_demand`.

- **Bias trace on the lead-time chart switched from green to a
  neutral yellow.** Green encodes "good" elsewhere in the UI
  (verdict chip, success status, "keep" recommendation), and
  bias is a signed metric with no good direction — green for
  bias was misleading. `#facc15` carries no semantic baggage.

- **Inline Remove-covariate button threshold aligned with the
  recommendation threshold.** Both now fire at ±2% (was ±1% for
  the button, ±2% for the recommendation), so a row's button can
  no longer disagree with whether the recommendations endorse
  dropping that covariate.

- **Recommendations payload now ships
  `variant: good|warning|bad|info` instead of a hex colour.**
  Template maps the variant to a CSS variable client-side, so
  palette changes don't require touching the backend. Legacy
  `rec.color` is kept as a fallback for responses produced by
  older add-on versions still in memory.

- **Forecast-log inspector and "View raw JSON" buttons are now
  collapsed behind a `<details>` summary.** Both are developer
  diagnostics; surfacing them as primary buttons on every page
  load confused first-time users. Available with one click for
  anyone debugging "why is my chart empty?".

- **"⚠ HA TZ unknown" inline note next to the Times-in selector
  when Home Assistant didn't report a timezone.** The dropdown
  silently falls back to browser-local in that case, which can
  bucket midnight resets wrong if the browser and HA server are
  in different timezones. The note explains the fallback.

- **Plotly load failures now surface a toast and reset the
  deferred-call queue.** When the lazy-load `<script>` tag's
  `onerror` fired, the Promise rejected silently — only
  `console.error` was emitted, and the pending-call queue
  continued to accumulate stub invocations indefinitely. Adds
  a `mlfl.toast('Charts unavailable — check network connection.')`
  on failure, resets `_pendingPlotlyCalls`, and caps the queue
  size at 64 to prevent unbounded growth under retry storms.

- **Dead `colorway: PLOT_COLORWAY` removed from the lead-time
  and trajectory layouts.** Every trace on those charts pins its
  own colour, so the colourway setting was unused. Left in place
  on the Predictions tab's holdout / residual charts where it
  legitimately drives multi-model trace colours.

## 2.31.0

Product / UX wave from the IMPROVEMENTS.md proposal. 19 changes
grouped into functionality gaps, frontend usability, model-improvement
workflows, and model-analysis trust signals. All grounded in existing
code paths; no new third-party dependencies; every item passes the
Pi-5 hard constraints (~4 GB RAM headroom, no CUDA, ARM64-buildable,
bounded CPU, light UI over HA ingress, SD-card-friendly disk writes).

### Functionality

- **A1** — `cpu_cores` and `nice_priority` on the System page are
  actually applied at startup now. Sets `OMP_NUM_THREADS` /
  `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
  `NUMEXPR_NUM_THREADS` / `VECLIB_MAXIMUM_THREADS` plus
  `torch.set_num_threads` and `torch.set_num_interop_threads`, and
  applies the nice value via `os.setpriority`. The System form now
  renders a "Currently applied: N threads, nice = X" confirmation
  line. Previously these settings were persisted to `mlfl.yaml` and
  ignored — training saturated every core regardless of the user's
  choice.
- **A2** — Delete-experiment button on dashboard cards and the
  experiment header. Reuses the existing confirm modal and the
  `/api/experiments/{name}/delete` endpoint that previously had no
  UI affordance.
- **A3** — Last 5 benchmark runs are kept on disk in a new
  `benchmark_history` table (append-only, capped per experiment).
  Results-tab gains a "Compare with previous run" strip that picks
  any prior run from a dropdown and renders a one-line diff: winner
  change + MAE delta percent.
- **A4** — One-click retrain rollback. Every `_persist_cached_model`
  now archives the existing champion into `<model_dir>/previous/`
  before overwriting; a new `Roll back` button on production
  experiment headers swaps current ↔ previous and re-hydrates the
  live cache. Single-generation cap keeps SD-card writes bounded.
  Endpoints: `POST /experiment/{name}/rollback`,
  `GET /experiment/{name}/rollback-available`.
- **A5** — HA lifecycle sensors fire on benchmark / retrain
  completion. Two new companion entities per experiment:
  `sensor.{prefix}{name}_last_benchmark` and
  `sensor.{prefix}{name}_last_retrain`. State is an ISO timestamp
  (HA picks up `device_class=timestamp` automatically); attributes
  carry outcome, duration, winner / model_version, and a truncated
  error string when the cycle failed. Unblocks HA automations
  reacting to ML-Forecast-Lab events without scraping logs.
- **A6** — Future / Both options removed from the covariate Role
  dropdown. `CovariateResolver.fetch_future` still returns NaN so
  selecting them silently broke the pipeline; kept as Lagged-only
  with an info-tip explaining the gap. Implementation is tracked
  separately.

### Frontend usability

- **B1** — Plotly is now lazy-loaded on the experiment page. The
  ~700 KB script no longer blocks the initial HTML render — a new
  `mlflLoadPlotly()` injects it on first activation of any
  chart-bearing tab. A transparent `Plotly.*` stub queues calls
  that arrive before the bundle lands and replays them in order
  once it does, so the many fetch-handler chart-render call sites
  need no changes.
- **B2** — Dashboard refresh swapped from `setInterval(location.reload)`
  to an HTMX poll against a new `/api/dashboard/grid` fragment.
  Scroll position, expanded `<details>`, and the New-experiment
  modal state now survive each refresh. Per-cycle payload drops
  from ~80 KB (full page) to ~5 KB (just the grid).
- **B3** — Forecast Accuracy, Tuning, Predictions, Generalisation,
  Covariate Analysis and Results tabs are now always rendered with
  a `disabled` style + hover-tip when their unlock conditions
  aren't met, so first-time users discover them.
- **B4** — Settings-tab autosave errors now persist with a red
  border + tooltip until the next successful save (the existing
  4-second toast was too transient — users walked away thinking
  they'd saved changes they hadn't). Error toasts also linger for
  10 s with a dismiss button.
- **B5** — Download buttons for the auto-generated Lovelace
  dashboard YAML, surfaced on the System page and every experiment
  header. The file was being written under `/addon_configs` since
  v2.x but only discoverable via the docs.

### Model improvement workflows

- **C1** — Pre-flight "Data sanity check" panel on the Settings
  tab, backed by a new `POST /experiment/{name}/data-report`
  endpoint. Fetches the raw target history the same way the
  benchmark would and reports rows fetched vs expected, biggest
  gap, recorder freshness, missing-value rate, zero-run length,
  and (for cumulative sensors) max-increment hits. Catches "your
  sensor has a 14-day flatline / 40% missing values" BEFORE you
  spend an hour on a benchmark.
- **C2** — Quick-preset chips above the per-experiment Models tab
  (Fast / Balanced / Thorough). Flips toggles in batches against
  the existing `/api/experiment/{exp}/models/toggle` endpoint.
  Combinations match the starter sets in `docs/MODEL_GUIDE.md`
  that previously only lived in the docs.
- **C3** — Pairwise model comparison matrix on the Results tab.
  Paired-t test on per-fold MAE differences with a normal-approx
  two-tailed p-value. Honest framing — with the default 5 folds
  the test is weak; the info-tip is explicit. Star marks pairs
  where the difference is unlikely to be inside fold noise at
  α=0.05.
- **C4** — "Tune all enabled" sweep across every enabled model.
  Loops the existing `_run_tuning` sequentially; each model's
  final result is captured in a new `tune_all_results[experiment]`
  list. A Sweep-results table renders one row per model with
  best composite, default vs tuned MAE, delta percent, and a
  per-row Apply button. Endpoints:
  `POST /experiment/{name}/run-tuning-all`,
  `GET /experiment/{name}/tuning-all`.

### Model analysis & trust

- **D1** — Conformal-band calibration countdown on the Forecast
  Accuracy verdict. Cold-start production experiments left the
  Uncertainty Bands tile at "—" with no explanation; now it renders
  *"Calibrating · 4 of 10 residuals · ~3 h to bands"* with a percent
  chip, sourced from the same `db.get_conformal_quantiles` count
  that drives the live publish path.
- **D2** — Always-on "vs Seasonal Naive" skill chip on Results.
  Seasonal Naive is force-included in every benchmark (no training
  cost; excluded from auto-promote when force-added) so the chip
  is present even when the user hasn't enabled the baseline. Green
  pill when the chosen model beats naive; red when it doesn't — a
  useful signal that the learned models aren't adding value on this
  target.
- **D3** — Retrain history chip strip on the Forecast Accuracy tab.
  Distinct `(model_name, model_version)` pairs in the window, ordered
  by first_seen. Click a chip to filter the verdict + charts to just
  that one cohort via the existing `?version=…` URL parameter.
  Backed by a new `db.get_retrain_events`.
- **D4** — Training-window vs test-window drift verdict on Results.
  Comparing target distribution stats over the earliest fold's
  training rows vs the latest fold's test rows, with a PSI score
  classed as stable (< 0.10), moderate shift (0.10–0.20), or
  significant shift (> 0.20). Helps users distinguish "the model is
  bad" from "the test window is from a regime the training rows
  don't cover".

### Internal

- `BenchmarkResult` Pydantic model gains `pairwise_dm`,
  `naive_baseline_mae`, `naive_baseline_was_enabled`, and `drift`
  fields. All optional; persisted through the JSON round-trip
  alongside the rest of the benchmark result.
- New shared `_build_dashboard_context` builder used by both the
  dashboard page and the HTMX grid fragment, eliminating drift
  between the two.
- New `_dashboard_card.html` and `_dashboard_grid.html` Jinja
  partials.
- `HistoryDB.BENCHMARK_HISTORY_RETAIN_PER_EXP = 5` constant
  governs the per-experiment retention cap for the new benchmark
  history table.

## 2.30.0

### Security

- **Removed the direct port 5052 exposure from `config.yaml`.** The web
  UI is now reached exclusively through Home Assistant's authenticated
  ingress proxy. Previously the FastAPI app was listening on the host
  LAN with `CORSMiddleware(allow_origins=["*"], allow_credentials=True)`
  and zero authentication on every mutating endpoint — any LAN device
  could delete experiments, promote arbitrary models, overwrite
  `mlfl.yaml`, or hammer the box into CPU exhaustion with repeated
  tuning kicks. Users who were relying on the direct port for an
  external dashboard need to proxy through HA ingress instead.
- CORS middleware removed entirely (ingress is same-origin).
- Error responses no longer echo raw exception strings to the client;
  filesystem paths are redacted via `_safe_error()` so internal
  layout doesn't leak into JSON bodies.

### Fixed

- `POST /experiment/{name}/run-benchmark` no longer jams the experiment
  permanently: the route now invokes `benchmark_callback` instead of
  setting a status flag with no consumer to clear it.
- All four web routes that rewrite `mlfl.yaml`
  (`/api/settings`, `/api/experiment-settings`, `/api/models/toggle`,
  `/api/experiment/{exp}/models/toggle`) now route through
  `atomic_yaml_write` — a crash mid-write no longer corrupts the file.
- `HistoryDB` now serialises every public method through the existing
  RLock via a `@_locked` decorator. Before this, most methods touched
  the shared `sqlite3.Connection` without holding the lock, racing
  benchmark-cycle writes against UI accuracy queries.
- `cleanup_forecast_log` on promote now preserves the incoming
  champion's forecast history (`exclude_model_name=` filter) so a
  demote → re-promote cycle doesn't wipe its calibrated residuals.
- One malformed experiment in `mlfl.yaml` no longer crashes
  `load_config`; the offending entry is skipped with an `ERROR` log
  line and the remaining experiments load normally.
- `apply_transform('log')` no longer produces `-inf` on inputs that
  contain exact zeros (night-time PV, off-state load). Zero-min inputs
  now use a `+1` shift; previously a `min_val == 0` branch left the
  shift at zero and `np.log(0)` propagated `-inf` into the model.
- `cumulative_to_interval` no longer under-reports demand during HA
  recorder gaps. Rows whose preceding time-gap spans more than ~1.5
  intervals are now dropped to `NaN` so the resampler treats the gap
  as missing rather than as a single under-scaled bucket.
- `y_diff_1` feature no longer encodes a synthetic dusk discontinuity.
  The previous per-shift GHI gate clamped `lag_1` to 0 at the first
  night sample while leaving `lag_2` at its daytime value, training
  trees to expect a fictitious daily drop of size ~daytime-peak.

### Changed

- Subscriber queues in `TrainingEventBus` are now bounded
  (`SUBSCRIBER_QUEUE_MAX = 8192`) with oldest-drop semantics. The
  per-experiment event history is capped at the same size. A
  backgrounded SSE consumer can no longer drive the add-on into OOM
  during a long benchmark.
- Fire-and-forget asyncio tasks in both `MLForecastLabApp` and the
  FastAPI app are now scheduled through a `spawn()` helper that
  retains a strong reference and surfaces unhandled exceptions via
  the logger. Previously a GC'd task could swallow its own failure.
- `cv_folds` validator now rejects values above 20 to stop typo
  configurations (e.g. `cv_folds: 1000`) from turning the benchmark
  into a multi-hour hang with no UI feedback.
- The `future` covariate role is now explicitly flagged as a
  stub at config-load time. `fetch_future` returns `NaN` rather than
  pretending to consult the HA `forecast` attribute.
- Cumulative forecast sensor is no longer published when predictions
  are signed; `np.cumsum` would drift unboundedly and pollute HA's
  long-term statistics. Other sensors still publish.

### Schema

- `HistoryDB` now records applied migrations in a new
  `schema_versions` table. Existing installs are migrated to schema
  v1 on first boot under 2.30.0.

### Infrastructure

- Bashio init script and all Python config-path helpers now anchor
  the slug-hash glob to HA's actual 8-hex-character prefix
  (`[0-9a-f]{8}_ml_forecast_lab`) so an unrelated add-on or fork with
  a suffix collision can't hijack the lookup.
- `_resample_covariate` no longer calls the deprecated
  `Series.fillna(method="ffill")` API — uses `.ffill()` directly.

## 2.29.0

### Fixed

- **AutoARIMA/AutoETS/AutoTheta no longer stall the benchmark
  pipeline.** Each of the three statsforecast backends refits its
  auto-search on every inference window, and the benchmark runs
  hundreds of windows per fold; before this release each refit was
  given the full 1024-sample `train_history` plus the window, and
  AutoTheta's decomposition is roughly O(n²) — concrete timings on
  half-hourly daily data: AutoARIMA >90 s/call, AutoTheta 26 s/call,
  AutoETS 0.7 s/call. A single CV fold would not finish. Three
  compounding fixes in `models/statsforecast_backend.py`:

  - Per-call history is now capped at 4 × `seasonal_period`
    (~192 samples for half-hourly daily data) via a new
    `_cap_history()` helper. The auto-search converges on the same
    orders from the shorter tail, so forecast quality is unchanged.
  - `AutoARIMA`'s default search grid (max_p=5, max_q=5, max_P=2,
    max_Q=2, nmodels=94, approximation=False) is impractically wide
    for per-window refits. Tightened to (max_p=2, max_q=2, max_P=1,
    max_Q=1, nmodels=10, approximation=True), which the classical
    forecasting literature treats as covering essentially all
    reasonable seasonal-ARIMA models. Cuts per-window cost from
    ~2.2 s to ~0.6 s.
  - The numba JIT inside statsforecast compiled lazily on first
    `forecast()` call — AutoTheta took ~25 s the very first call,
    then 0.05 s thereafter. That cold-start cost surfaced inside
    `predict_sequence` and made the first benchmark window look
    stuck. Moved the cost to `fit()` via a new `_warmup_jit()` call
    on a tiny synthetic series so the compiled paths are cached
    before inference starts.

  After the fix the four classical/baseline backends are well under
  a 1 s/window budget end-to-end (validated by a new section in
  `tests/dryrun_pipeline.py`): arima 0.59 s, ets 0.18 s,
  seasonal_naive 0.00 s, theta 0.03 s.

- **Sliding-window CV bug surfaced by data-flow audit.** Walk-forward
  CV could produce zero-length test folds on shorter series because
  the window builder dropped the last `window_size` rows of every
  fold (no future targets to predict), then the split used
  pre-window indices. Folds whose post-window length was smaller than
  the configured test size silently became empty and were excluded
  from the Demšar ranking without any warning. Fix mirrors the
  window builder's drop logic at split time so test folds always
  contain the requested number of samples.

- **CV embargo gap was being silently ignored when the dataset was
  too small.** The embargo (configurable gap between train and test
  to prevent peek-ahead leakage) was applied unconditionally, but
  when the series wasn't long enough to satisfy
  `train_size + embargo + test_size`, the code fell back to a
  zero-embargo split without logging anything — a quiet correctness
  regression for users running on short HA histories. Now logs a
  clear warning and falls back to the smallest viable embargo.

- **`align_series` could drop the last row of the longer series.**
  When two series had different end timestamps, `align_series`
  truncated to the shorter, but it used `< min_end` instead of
  `<= min_end`, dropping the final aligned sample. Fixed off-by-one
  and added regression tests.

- **`apply_load_subtract` trailing-gap handling.** When the subtracted
  series ended before the main series, the trailing gap was filled
  with NaN and then quietly forward-filled from the last valid value,
  producing a flat extrapolation. Now the trailing gap is explicitly
  zeroed (treating "no measurement" as "no contribution") with a
  warning, matching the existing leading-gap behaviour.

### Added

- **New `model_family` taxonomy.** Every registered backend now
  reports one of `'tree'`, `'neural'`, `'classical'`, or `'baseline'`
  via a new property on `ForecastModel`. The base default derives
  from `is_neural`, so all 17 neural backends and 3 tree backends
  auto-classify without any per-backend code; the four non-NN
  backends override explicitly:

      baseline   (1):  seasonal_naive
      classical  (3):  arima, ets, theta
      neural    (17):  cnn, crossformer, dlinear, fits, gru,
                       itransformer, lstm, nbeats, nhits, nlinear,
                       patchtst, sparsetsf, tft, tide, timemixer,
                       timesnet, tsmixer
      tree       (3):  catboost, lightgbm, xgboost

  Two new registry helpers expose the grouping for downstream code:
  `registry.list_by_family()` (dict[family → sorted names]) and
  `registry.family_of(name)`. The web `MODEL_CATALOG` already
  mirrors this taxonomy, so no UI changes are needed in this patch
  — future code can query the registry directly instead of
  maintaining a parallel hardcoded mapping.

  `is_neural` is intentionally unchanged. It semantically means
  "this model needs sliding-window input" (which classical/baseline
  backends also do, because they pull the target series out of a
  window channel), and renaming it would have rippled through
  benchmark/runner.py, main.py production training, and Covariate
  Analysis. `model_family` is the source of truth for grouping/UI
  purposes only.

- **`tests/dryrun_pipeline.py` — synthetic-data walkthrough of every
  backend.** Standalone debug script that exercises every model
  backend, the covariate-analysis enumeration, and the
  hyperparameter-tuning sample round-trip without paying the
  model-training cost. Intended as a fast local debug aid when
  changing the model registry, the parameter schema, or any shared
  helper. Runs in under 30 s and exits 1 on any failure.

  The dryrun now also: (a) prints the family grouping so a
  contributor can spot a backend that's accidentally landed in the
  wrong category, and (b) times `predict_sequence` on the
  classical/baseline backends, failing the run if any backend
  exceeds 1 s/window — caught the AutoTheta-on-cold-JIT regression
  during development of this release.

## 2.28.4

### Added

- **Debug buttons under the lead-time chart on Forecast Accuracy.**
  Two buttons — *View raw JSON* (dumps the response body into an
  inline `<pre>` block on the page) and *Download as file* (saves it
  as `<experiment_name>-forecast-accuracy.json`). Both fetch the same
  `/forecast-accuracy?days=N` endpoint the chart itself uses, via
  the live page session — so they work without any DevTools or
  ingress-token URL surgery. Useful when the chart shows empty and
  you need to know *why* the lead-time aggregation came back without
  data (no logged forecasts, version filter mismatched, actuals
  missing, etc.).

### Changed

- **Forecast Accuracy empty-state moved into the lead-time card.**
  Previously the empty-state lived at the very bottom of the tab,
  after the diagnostics accordion. When `loadForecastAccuracy()`
  returned no data, `_renderAccuracyEmpty` would hide the verdict and
  revision cards but the diagnostics accordion populated independently
  via separate fetches — so users saw mixed state: empty top,
  populated middle, stranded "No forecast accuracy data" panel at the
  bottom. The empty-state now sits inside the same card as the
  lead-time chart, so when there's no data it visually replaces the
  chart in-place rather than appearing as an orphaned panel below
  the rest of the tab.

## 2.28.3

### Fixed

- **Daily Rank column on the Results tab's Daily Cumulative Accuracy
  table never populated.** Three call-sites in `main.py` push web
  state via `_update_web_benchmark`; one of them (the final
  post-holdout "completed" overwrite at line 2058) was missing the
  `daily_rankings=` kwarg. The earlier completed call did pass it,
  so the rank column briefly showed correct values mid-run before the
  final overwrite landed with `daily_rank=None` for every model and
  the column collapsed to `—`. Threaded `daily_rankings` through.

- **Forecast Accuracy → Diagnostic Tools accordion was collapsed by
  default, and "How predictions converge on a single moment" +
  "Forecast convergence" charts only fired their fetches inside the
  accordion's `toggle` handler.** Users who never opened the
  accordion never saw those charts populate. Two changes: the
  `<details>` now opens by default (matches the user expectation
  that the diagnostic surface is part of the page, not hidden behind
  a click), and `_bindDiagAccordion` now triggers `_loadDiagOnce()`
  immediately if the accordion starts open — `toggle` events don't
  fire on initial open state, so the deferred-load path never ran.

### Changed

- **Section headers across the Forecast Accuracy tab no longer
  render as ALL CAPS.** Source HTML has them in sentence case
  ("How error grows with forecast horizon", "Forecast convergence",
  "How predictions converge on a single moment", etc.) but the CSS
  rule `.settings-group-title` was applying `text-transform: uppercase`
  + `letter-spacing: 0.04em`. That made sense when the only uses
  were short eyebrow labels in the Settings tab ("Target",
  "Training", "Covariates"), but reads as shouting on full-sentence
  titles. Removed both rules; bumped font-size 0.85→0.95rem and
  weight 700→600 to keep the visual hierarchy. Affects every section
  header that uses the class — Settings tab labels also revert to
  sentence case, which is consistent with the rest of the UI.

- **Three Layer 1 tooltips on Forecast Accuracy expanded with more
  context**: "Typical next-step error" now explains why near-term
  error is the cleanest comparison metric, "Uncertainty bands" calls
  out what well-calibrated coverage means for downstream automations,
  and "Run-to-run swing" explains why prediction-spread between runs
  matters when consuming the forecasts.

## 2.28.2

### Fixed

- **"Tune" on AutoETS / AutoARIMA / AutoTheta / Seasonal Naive
  appeared stuck at the start.** The classical backends had
  `seasonal_period` listed as a tunable hyperparameter with range
  1–1440, so Optuna would pick a value per trial and run a full CV
  benchmark — but each inference window inside that benchmark spawns
  a fresh AutoETS / AutoARIMA / AutoTheta which does its own internal
  auto-search, so trials took many minutes apiece while producing
  results that aren't really hyperparameter search anyway. Two
  changes: (1) `seasonal_period` is now marked `tunable: False` on
  these four backends — it's a data-cadence property (48 for
  half-hourly daily, 168 for hourly weekly, etc.), set once based on
  your sampling rate, not searchable; (2) the `/run-tuning` endpoint
  now returns a clean `400` if a model has no tunable params, with
  the message *"X has no tunable hyperparameters — the auto-model
  selects them internally. Set fixed parameters (e.g. seasonal_period)
  on the Models page instead."* No more silent spinning.

  Locked in with 9 new smoke tests (`tests/smoke/test_tuning_guard.py`):
  the four auto-models must 400, four representative ML models
  (LightGBM, XGBoost, LSTM, CNN) must still 202, unknown model still
  400. Smoke suite is now 70 tests.

## 2.28.1

### Fixed

- **AutoETS log spam: every inference window emitted a `Parameters out
  of range` WARNING and fell back to seasonal-naive.** Two compounding
  causes. (1) `ETSModel._make_model` used `model='ZZZ'`, letting AutoETS
  pick *multiplicative* seasonality — mathematically ill-defined on
  series with zeros or near-zero values, which is most HA sensors
  (overnight energy demand, solar at night, intermittent appliances).
  Switched to `'ZZA'`: auto error type, auto trend type, additive
  seasonality only — keeps the auto-search useful while staying
  numerically well-defined on zero-bearing data. (2) The fallback
  warning fired per failing window, flooding logs during benchmark and
  tuning runs with hundreds of identical lines. `_forecast_single` now
  increments a per-batch counter silently; `predict_sequence` and
  `predict` log a single summary at the end of each batch
  (`ets fell back on 47/50 window(s); first error: ...`).

## 2.28.0

### Added

- **Smoke-test harness for the FastAPI web app.** New `tests/smoke/`
  suite (61 tests, 1.9s wall time) boots the real `create_app()` factory
  against a tmp `mlfl.yaml` and walks the eight golden user flows: top-
  level page renders, experiment CRUD, model-param round-trip, global
  and per-experiment settings persistence, promote / mode-toggle (incl.
  4xx empty-states), the analytics route family's empty-state contracts,
  and the HA entity picker's graceful-degrade-when-HA-unreachable path.
  No model training, no live HA instance, no addon paths required.
  Designed as a cheap release gate that catches "page returns 500" or
  "YAML write contract changed" regressions before they ship.

- **`tests.yml` GitHub Actions workflow.** Two-job split — `smoke`
  (required, <30s, lightweight deps) gates every PR and main push;
  `unit` (full deps incl. torch) runs alongside. Complements the
  existing `validate.yml` syntax / version-consistency checks.

- **`tests/unit/` reorg + shared fixtures.** Existing 9 test files
  moved into `tests/unit/` to make room for `tests/smoke/`. Synthetic
  pandas fixtures stay at the `tests/` root so both layers share them.
  `tests/requirements-dev.txt` pins the dev-only deps.

- **`docs/MODEL_GUIDE.md`** — practical "which of the 24 backends should
  I enable?" with starter sets keyed to data volume, target shape, and
  Pi compute budget. Linked from the README.

### Changed

- **Standardised UI text capitalisation across all templates.** Form
  labels, dropdown options, and checkboxes are now consistently sentence
  case ("Process priority", "Target entity", "Best model", "Last run",
  "Next forecast", "Next retrain", "Production model", "Training CPU
  cores", "Very low", "Include covariate analysis"). Page titles,
  section headers, tab labels, and multi-word command buttons stay
  Title Case ("Save Settings", "Reset to Defaults", "Load Log Summary",
  "Create New Experiment"). Table column headers stay Title Case as a
  distinct convention. Style rule documented in the v2.28.0 commit.

- **Onboarding hints in the empty-state surfaces a first-time user
  hits.** Dashboard empty state now explains what an experiment is and
  the lab → production progression. Create-experiment modal now has
  info-tip help on `Cumulative source` and `Daily reset` (the two
  toggles new users with no timeseries background find most opaque)
  plus a footer line clarifying that covariates / models / training
  settings are configured per-experiment after creation. Three empty-
  state blocks on the experiment detail page (no benchmark, no holdout,
  no forecast accuracy) now explain what'll happen when the user clicks
  the visible button.

- **Deduped config-path discovery in `web/app.py`.** Four inline
  copy-pasted blocks of `/addon_configs/ml_forecast_lab/mlfl.yaml`
  fallback logic (~30 lines) folded into the existing
  `_find_config_path()` helper, which all 21 other callsites already
  used. The `create_app(config_path=...)` parameter — previously
  declared in the signature but ignored — now actually overrides the
  discovery, used by tests to inject tmp configs without filesystem
  trickery. Behaviour unchanged in production.

- **README sweep:** stale "22 backends" → 24 (table already showed 24,
  prose lagged); positioning vs EMHASS / Solar Forecast ML / predbat;
  "benchmark once, run forever" framing; six-step Quick start;
  Troubleshooting; Development.

### Fixed

- **Pre-existing unit-test drift surfaced by the new smoke harness.**
  `test_config.py::test_defaults` updated to expect the actual
  `production_metric` default (`rmse`, has been since the initial
  upload). `test_db.py` rewritten against the real public API
  (`store_history` / `get_history` / `cleanup` with datetime — old
  tests called methods that never existed). `test_models.py` LSTM/CNN
  z-score test split into `use_revin=False` (asserts channel stats
  are stored) and `use_revin=True` (asserts they're NOT, because RevIN
  handles per-window normalisation). All 124 unit + 61 smoke = 185
  tests now green; the unit job in `tests.yml` is no longer
  best-effort.

- **Dead code removed in `web/app.py`.** `/api/models` endpoint that
  returned 4 hardcoded models (LightGBM, XGBoost, LSTM, CNN) — out of
  sync with the 24-backend `MODEL_CATALOG` rendered on `/models`, no
  callers anywhere. Plus the `ModelInfo` Pydantic class that backed it,
  and two orphaned `import glob as _glob` left over from the
  config-path dedup.

## 2.27.12

### Fixed

- **Forecast horizon anchored hours in the past when the source sensor's
  recorder history goes quiet.** HA's recorder dedups identical state
  writes, so a target sensor whose value doesn't change for hours
  (canonical case: a `cumulative` + `reset_daily` daily-energy counter
  during a holiday with zero usage — the entity is alive and reporting
  but the value isn't varying, so the recorder writes nothing) leaves
  no new history rows. `_fetch_and_preprocess` would then end at a
  `last_ts` hours behind wall-clock; the downstream `ds_future =
  last_ts + i*interval` placed the forecast attribute's first
  timestamp in the past, the `_cumulative` sensor's `today_seed` came
  from an outdated state, and the `_forecast`/`_interval` sensor
  states landed on values identical cycle-to-cycle (so HA's
  `last_changed` never advanced and any card displaying "X minutes
  ago" looked stuck).

  After the recorder fetch, `_fetch_and_preprocess` now checks whether
  the most-recent recorded timestamp is more than `interval_minutes ×
  2` behind `now`. If it is, it `get_state`s the live entity value via
  `/api/states/{entity}` and synthesises samples at the configured
  cadence from `last_ts + interval` up to `now`, all carrying that
  live value. For cumulative sources the synthetic carry-forward
  produces zero per-interval increments (correct: the source didn't
  tick), so `cumulative_to_interval` and the rest of the pipeline see
  a frame consistent with current wall-clock — the forecast horizon
  starts at `now + interval`, the cumulative seed reflects the actual
  current target value, and the model's lag features include the
  recent flat period.

  Synthetics are NOT persisted to the SQLite cache; they're
  regenerated each cycle from the live state, so a real source
  resumption immediately supersedes them with no cache contamination.
  When the carry-forward kicks in, a single WARNING line names the
  entity, the gap size, the live value carried, and the synthetic
  tick count — so cycles where the source really did go quiet are
  visibly distinct from cycles where it ticked normally.

## 2.27.11

### Fixed

- **Hotfix for v2.27.10: don't poison future covariates with an
  all-NaN series.** `CovariateResolver.fetch_future` at
  [covariates.py:157](ml-forecast-lab/ml_forecast_lab/covariates.py:157)
  is currently a stub — for every entity type except
  `constant_value`, it returns `pd.Series(np.nan, ...)` and logs
  `No future covariate data for ...`. v2.27.10 accepted those NaN
  series into `future_cov_values`; `ffill().bfill()` on all-NaN
  leaves NaN, and the downstream `np.nan_to_num` then slammed the
  covariate to `0` for every horizon step. For solar experiments
  that meant Solcast was being fed as 0-during-the-day to a tree
  that trained on Solcast's actual time-varying readings — the
  tree collapsed its forecast curve toward 0 across tomorrow's
  daytime too, visibly worse than pre-v2.27.10. Now only keep the
  fetched future series when it contains any non-NaN value;
  otherwise fall back to the pre-v2.27.10 last-observed
  carry-forward for that covariate. Applied in both
  `_forecast_with_cached` and `_run_production_inference` (same
  latent bug, different exposure).

  Once `fetch_future` grows a real forecast-attribute parser
  (Solcast's `detailed_forecast`, weather providers' hourly
  payloads, etc.), the cached cycle will automatically start
  using the time-varying values without further changes here.

## 2.27.10

### Fixed

- **Cached forecast cycle now fetches future covariate values.** The
  tree-model branch of `_forecast_with_cached` (the path that drives
  the live publish cycle) was pinning every `role: future` / `role:
  both` covariate — Solcast forecasts, tariff schedules, weather-
  forecast sensors — to its last observed value for all 48 horizon
  steps. Training, however, had each row's *time-current* Solcast
  reading as the covariate value, so the model learned a mapping
  from a time-varying signal but was being fed a single constant at
  inference. The mismatch starved the tree of its most-informative
  daytime covariate and was the dominant driver of peak under-
  prediction on solar targets. Now mirrors
  `_run_production_inference`:
  `covariate_resolver.fetch_future(cov, future_index)` runs once per
  role=future covariate, the returned series is reindexed onto the
  forecast grid, and `_build_feature_row` uses each step's value at
  the right horizon index (falling back to carry-forward only when
  the future fetch fails or the covariate is lagged-only). Logged
  per cycle so you can confirm which future covariates actually
  resolved: `Fetched future covariate series for N/M future-role
  covariates: [names]`.

## 2.27.9

### Changed

- **Night-time solar forecasts now go to zero via the model, not a
  post-hoc clamp.** Tree backends (CatBoost / LightGBM / XGBoost) do
  recursive multi-step forecasting — each step's prediction feeds
  forward as `y_lag_1` of the next step. A sunset step biased even
  ~100 W leaks into the next step's lag, producing a feature vector
  `(clear_sky_ghi=0, y_lag_1=150)` that training never saw (training
  data cleanly had `y_lag_k=0` whenever the past step was night). The
  tree then lands on a daytime leaf and predicts non-zero across the
  whole overnight horizon. Fixed at the feature contract: when
  `clear_sky_ghi` is present in the dataframe, `build_features` zeros
  `y_lag_k` and `y_diff_1` values whose corresponding shifted GHI is
  zero. The recursive inference paths
  (`_forecast_with_cached._run_recursive_forecast` and
  `_run_production_inference._run_recursive_forecast`) mirror the
  invariant by pushing `0` into the lag buffer at night steps instead
  of the raw prediction — so every future feature vector stays
  in-distribution and the tree's own learned response drives the
  published forecast down to near-zero through dusk, smoothly.

- **Removed the v2.27.8 hard clamp in `_publish_forecast_sensors`.**
  With the feature-contract fix above, the output-side clamp is no
  longer needed and was creating a visible cliff at the exact moment
  `clear_sky_ghi` crossed zero. Any residual non-zero at night after
  the next retrain reflects a genuine model error (or a
  non-solar-gated backend path) that shouldn't be hidden.

- **Neural models are unaffected by this change.** They exclude
  `y_lag_*` from their covariate set (see
  [main.py](ml-forecast-lab/ml_forecast_lab/main.py) — the sliding-
  window covariate cols subtract `y_lag_*`) and predict all horizons
  in a single forward pass, so they don't have the recursive
  drift pathology. Their inference already uses real history, which
  is cleanly zero at night. If a neural backend shows a residual
  night floor, that's the softplus output head — switchable via
  `output_activation: relu` on the experiment.

## 2.27.8

### Fixed

- **Solar forecasts no longer carry a noise floor at night.** Tree
  ensembles (CatBoost, LightGBM, XGBoost) can't output exact zero —
  any non-zero night samples in training history (inverter idle draw,
  meter noise, single-digit watts) leave a leaf bias the forest can't
  unlearn, so predictions hover around ~100-300 W even when
  `clear_sky_ghi = 0` and actual PV is exactly 0. Softplus-headed
  neural models have a ln(2)-scale floor for the same visual effect.
  Added a physics gate in `_publish_forecast_sensors` that clamps
  `y_pred` (plus conformal upper/lower bands) to exactly 0 on every
  horizon step where pvlib's clear-sky GHI is 0. Activates only when
  `include_clear_sky_irradiance: true` is set on the experiment — the
  unambiguous signal that the target is solar-driven — so non-solar
  signals are unaffected. Sited in the shared publish helper so
  `/promote`, cached forecast cycles, and the full
  `_run_production_inference` path all pick it up uniformly.

## 2.27.7

### Changed

- **Production cached models now survive restarts.** After every
  `_retrain_and_cache`, the trained model is serialised to
  `/data/ml_forecast_lab/models/<exp>/model.bin` (via the backend's
  own `save()`) alongside a `cache_meta.json` holding
  `feature_cols`, `trained_at`, `model_version`, `is_neural`, and
  `window_size`. On startup, `_restore_cached_models` loads each
  production experiment's cache before `main_loop` decides its
  timers. If the restored cache is younger than
  `retrain_every_hours`, the immediate startup retrain is skipped
  and the next retrain is scheduled at `trained_at +
  retrain_every_hours` instead. This eliminates the cascade of N
  sequential retrains every restart used to run through before the
  UI became responsive — the user experience on a clean restart
  drops from "minutes of 'no best model' placeholders while each
  production experiment retrains in turn" to "sensors publish on
  the next forecast tick, UI shows the restored champion
  immediately". Persistence failures and schema mismatches fall
  back silently to the old cold-start behaviour.

- **`_forecast_with_cached` no longer requires a cached training
  frame.** Previously the fresh-fetch-failure branch reached into
  `cache["combined"]` as a fallback; restored caches deliberately
  don't carry that frame (too big to pickle, always re-fetchable),
  so the code now `cache.get("combined")` and skips the cycle
  cleanly when both fresh-fetch and cached-fallback are unavailable.

## 2.27.6

### Fixed

- **Publish button now triggers an immediate retrain.** The
  `/experiment/{name}/promote/{model_name}` handler flipped the
  experiment to `production` mode and persisted the choice to YAML,
  but — unlike the sibling `/toggle-mode` handler — never fired
  `retrain_callback`. The experiment sat in production with no cached
  model until the next scheduled retrain tick (up to
  `retrain_every_hours` later), so sensors didn't appear and the
  Publish click looked like a no-op. Now fires the retrain callback
  at the end of `promote_model`, matching `toggle_mode:2277-2280`.

## 2.27.5

### Fixed

- **CatBoost tuning trials no longer appear to hang for tens of
  minutes.** CatBoost builds oblivious (symmetric) trees, so every
  depth level strictly doubles per-tree cost — `max_depth=16` is
  ~1000x slower per tree than `max_depth=6`, and Optuna will happily
  pick it. Tightened the CatBoost search space to match the library's
  practical range (`max_depth` 3-10, `n_estimators` 10-2000) so a
  single trial can't out-grow the 30 min study budget. Defaults are
  unchanged (depth=6, n_estimators=500, lr=0.05) so existing results
  aren't affected. The v2.27.4 iteration callback remains active and
  drives the live training UI for the benchmark and production-training
  paths — it's only the tuning path that never wired `epoch_callback`
  through `runner.run_single_model`, which is why the callback alone
  didn't surface in-trial progress.

### Changed

- **Tuning logs now emit a `Trial N starting: params=...` line per
  trial.** Previously only the completion line logged, so an in-flight
  trial was invisible in the log and indistinguishable from a hang.

## 2.27.4

### Fixed

- **CatBoost tuning no longer looks stalled.** CatBoost emitted a single
  end-of-training event, so the live training UI showed nothing at all
  during each trial even while the model was iterating. Added a CatBoost
  iteration callback (matching the LightGBM / XGBoost pattern) that
  forwards per-tree RMSE plus a synthetic patience counter to the
  training event bus.

- **Tuning now has a 30 min wall-clock budget for every backend.** The
  timeout previously only fired for neural models; tree tuning could
  sit for an hour inside a single trial when Optuna picked a tiny
  learning rate plus a large `n_estimators` (CatBoost in particular),
  with no progress signal to distinguish "training slowly" from "hung".

### Changed

- **Model tabs render alphabetically.** Both the main Models page and
  the per-experiment Models tab now order cards by display name,
  regardless of the insertion order in `MODEL_CATALOG`. Baseline and
  Classical models (Seasonal Naive, AutoARIMA / AutoETS / AutoTheta)
  interleave cleanly with the neural backends.

## 2.27.3

### Fixed

- **Accuracy publish cycle no longer freezes the event loop.** Each
  publish cycle called `get_forecast_accuracy` inline — a 3-CTE query
  over the actuals table that blocked the web UI and the HA `set_state`
  pool for the full scan duration. Now offloaded via
  `asyncio.to_thread`, with the SQLite connection opened
  `check_same_thread=False` behind an `RLock` and WAL journaling so the
  offloaded reader doesn't contend with the publish-cycle writer.

- **/forecast-accuracy skips empty-filter fallbacks up-front.** The
  endpoint's fallback ladder could run the heavy accuracy query up to
  three times when the strict `(model_name, model_version)` filter
  returned empty. Added `HistoryDB.probe_forecast_rows()` — an
  O(log N) `EXISTS` check served by the
  `(experiment, model_name, model_version)` index — which picks the
  narrowest filter with data first, so only one full accuracy query
  runs per request.

## 2.27.2

### Added

- **Covariate manifest log per preprocess cycle.** `_fetch_and_preprocess`
  now emits a single INFO block per experiment summarising every
  covariate's contribution: traffic-light status (`✓` / `⚠` / `✗`),
  configured role (`lagged` / `both` / `future` / `physics`), post-dropna
  coverage %, staleness of the most recent raw value, and Pearson
  correlation with the target on the aligned frame. A final `dropna:
  A rows → B kept (C lost; biggest culprit: <col> N NaNs)` line names
  the column whose NaNs deleted the most rows — the specific field
  that pinpoints "the whole experiment returned zero samples because
  one covariate is misconfigured". Solar physics features are included
  under `role=physics` so the block is a complete record of what the
  training frame actually contained. Mirrors the existing publish
  manifest pattern from 2.26.4: scattered per-covariate `✓ raw → aligned`
  INFO lines are kept for backwards compat, and the consolidated block
  sits alongside them. Diagnostic shape:

  ```
  Covariate manifest for optimised_solar (9 configured, +2 physics):
    ⚠ sensor.solcast_pv_forecast_forecast_today [future]  cov=100.0%  stale=6.2h  corr=+0.71  ← stale>interval×4
    ✓ sensor.openweathermap_cloud_coverage [lagged]  cov=99.8%  stale=12m  corr=-0.34
    ✗ sensor.carlton_green_south [both] — fetch failed: HTTP 500
    dropna: 8,760 rows → 8,740 kept (20 lost; biggest culprit: clear_sky_ghi 14 NaNs)
  ```

- **TiDE surfaces when its future-covariate path is inactive.** The
  existing `TiDE: future-covariate path active (N features × H horizons)`
  INFO line only fired when the caller passed `future_covariates=`; the
  silent `else` branch left users debugging "why is TiDE losing to
  Seasonal Naive on a signal with `role: future` covariates?" with no
  log evidence that the caller never wired the tensor up. An equivalent
  INACTIVE log now fires, explicitly stating that `role=future`
  covariates aren't being routed and that TiDE is degraded to a
  past-only encoder-decoder. Lets the covariate manifest (which shows
  `role=future` for Solcast/tariff sources) and the TiDE log sit
  side-by-side so the gap is immediately visible.

## 2.27.1

### Fixed

- **HA API calls no longer cascade-fail when Home Assistant is slow.** A
  single `total=30s` wall-clock budget in `HAInterface.api_call` covered
  DNS, connect, send and full response read, so a slow recorder query for
  `/api/history/period` (routine on benchmark loads with weeks of data)
  burned the whole budget and triggered `asyncio.TimeoutError`. Once that
  first call stalled the shared connector, concurrent `set_state` POSTs
  fired by `_publish_one` then timed out at the DNS-resolution phase —
  reported by users as a flurry of `mlfl_*` publish failures in the same
  second.

  Split into separate `sock_connect` (15s) and `sock_read` (30s default,
  180s for history queries) budgets so a slow body can't starve the
  connect phase, added exponential-backoff retry (3 attempts, 1s / 2s) for
  `TimeoutError` / `ClientError` / 5xx, and made both timeouts and retry
  count overridable per call. 4xx is treated as non-transient and is not
  retried.

## 2.27.0

### Added

- **Eight new model backends, taking the catalogue from 14 to 22.** All
  registered automatically and surfaced in the Models page, the
  hyperparameter editor, and the Demšar composite ranking on equal footing
  with the existing backends.

  Tier 1 (state-of-the-art lightweight):
  - **CatBoost** (`catboost`) — third tabular gradient boosting backend
    alongside LightGBM and XGBoost. Ordered boosting + oblivious symmetric
    trees often wins on noisy or covariate-rich smart-home signals.
  - **NLinear** (`nlinear`) — companion to DLinear from Zeng et al. 2023.
    Subtracts the last value, applies a single linear layer, adds it back.
    Tiny but a top-tier baseline that's strange to ship without.
  - **FITS** (`fits`) — frequency-domain low-pass complex linear, ICLR 2024
    ("Modeling Time Series with 10k Parameters"). The lightest neural
    backend in the catalogue.
  - **TimeMixer** (`timemixer`) — multiscale season/trend mixing with
    cross-scale interaction (Past-Decomposable-Mixing block), ICLR 2024.
    Currently leading several long-horizon benchmarks.

  Tier 2 (gold-standard baselines):
  - **GRU** (`gru`) — lighter recurrent baseline than LSTM. Same temporal
    attention + multi-horizon pipeline; ~25% fewer parameters per cell.
  - **TFT** (`tft`) — compact Temporal Fusion Transformer (Lim et al. 2021)
    with Variable Selection Network → LSTM encoder → interpretable
    multi-head attention. Static / known-future covariate branches are
    omitted (rare in HA sensor data).
  - **Seasonal Naive** (`seasonal_naive`) — the reference baseline every
    forecasting paper requires. ŷ[t+h] = y[t+h-period], no training. Now
    that it's a registered backend it shows up in the Demšar ranking, so
    every other model has to actually beat it to look good.
  - **statsforecast classical baselines** (`arima`, `ets`, `theta`) —
    AutoARIMA, AutoETS, AutoTheta from Nixtla's numba-JIT-compiled
    statsforecast. The classical statistical reference points the academic
    literature treats as mandatory comparisons.

  New dependencies (auto-pulled by `requirements.txt`): `catboost>=1.2`,
  `statsforecast>=1.7`. Each backend gracefully degrades to a clear
  RuntimeError if its underlying library isn't available, matching the
  existing optional-backend pattern.

## 2.26.7

### Fixed

- **Root cause of the Forecast-log button failure: HTML autoescape on
  the `EXP_UNITS` fallback emitted `var EXP_UNITS = &#34;&#34;;` for
  any experiment without a `units:` value (the default — `units: str
  = ''` in `ExperimentCfg`).** `experiment.html` had `var EXP_UNITS =
  {{ units | tojson if units else '""' }};`. The truthy branch goes
  through `|tojson`, which returns a Markup-safe string — autoescape
  leaves it alone. The else branch returned a plain Jinja literal
  `""`, which Jinja's HTML autoescape rewrote to `&#34;&#34;` on the
  way into the `<script>`. The browser's JS parser does not decode
  HTML entities inside `<script>`, so it threw `SyntaxError:
  Unexpected token '&'` at that line, aborting the IIFE before any
  of the `window.X = …` assignments below it ran — including
  `window.loadForecastLogStats`. The button's onclick then hit a
  `ReferenceError: Can't find variable: loadForecastLogStats`, which
  the v2.25.3 defensive fallback faithfully rendered. Replaced the
  conditional with `{{ (units or '') | tojson }}` so it always pipes
  through the Markup-safe filter. v2.26.6's button colocation +
  error capture turn out to be useful belt-and-braces for any future
  variant of the same class of bug, but this fix kills the actual
  trigger every existing user is hitting.

## 2.26.6

### Fixed

- **Forecast-log inspector button threw "Can't find variable:
  loadForecastLogStats" under real-world conditions.** v2.25.3 added a
  defensive try/catch around the button's onclick because the handler
  sometimes silently did nothing, but couldn't reproduce the underlying
  cause. The actual mechanism: `window.loadForecastLogStats` was assigned
  ~1300 lines into the big IIFE in `{% block scripts %}`, so any earlier
  throw during IIFE evaluation (bad Jinja-interpolated value, localStorage
  access denied, missing `Intl` features, etc.) aborted execution before
  the assignment, leaving the button's onclick referencing a symbol that
  never came into existence. The handler is now defined in its own
  `<script>` block right next to the button, with the URL rendered
  directly via Jinja rather than relying on closure vars from the main
  IIFE, so an unrelated failure elsewhere in the scripts block can't
  take the diagnostic button out.
- **Surface the root-cause error, not just the downstream symptom.** A
  top-of-block `window.addEventListener('error', …)` now stashes the
  first unhandled script error on `window.__mlflFirstError`. The
  Forecast-log button's catch appends it when the handler is still
  missing, so the next time this class of failure shows up we see the
  actual throw — not just "Can't find variable: loadForecastLogStats".

## 2.26.5

### Fixed

- **`_forecast_running` flag leaked on early returns, freezing forecasts
  for some experiments.** 2.26.2 moved the slot reservation synchronously
  into the scheduler (`self._forecast_running[name] = True` before
  `asyncio.create_task(self._forecast_single(exp_cfg))`) to close a
  same-tick double-fire race, but `_forecast_single` still had two
  early-return branches — non-production mode, and `name not in
  _cached_models` — that returned *before* the `try/finally` that clears
  the flag. In a multi-experiment setup, retrains are queued sequentially
  via `_retrain_queue`, so at the first forecast tick only the
  experiment whose retrain finished first has a cached model; the
  others hit the `no cached model` return and their `_forecast_running`
  flag stays `True` forever. From then on the scheduler's
  `not self._forecast_running.get(name, False)` guard skips those
  experiments on every tick, silently freezing their entire sensor
  family (`_forecast`, `_cumulative`, `_upper_80`, `_lower_80`,
  `_forecast_accuracy`) while other experiments keep updating — the
  "next forecast doesn't always create a new forecast for some of the
  sensors" symptom. The `try/finally` now wraps the full body so the
  flag is always cleared, regardless of which exit path the task takes.

## 2.26.4

### Added

- **Publish manifest log per cycle.** `_publish_forecast_sensors` now
  emits a single INFO line summarising every expected sensor's outcome
  — `published=N/total`, plus a `skipped` list with reason codes
  (`source_not_cumulative`, `no_conformal_bands`) and a `failed` list
  with per-entity exception context. Diagnoses the "`sensor.X
  last_updated = 3h ago` but the log says publish succeeded" symptom
  by making silent conditional skips visible: if a sensor has been
  out of the payload for hours, every cycle's manifest now shows
  exactly why. Replaces the old split "Published X/N sensors … / Failed
  to publish …" pair with one coherent report.

- **`_forecast_accuracy` sensor is always published.** Previously
  gated on `self.history_db and exp_cfg.mode == "production" and
  ltc.get("lead_minutes")`, so the HA entity didn't exist at all
  during cold start, in lab mode, or when the forecast_log query
  returned no matches yet — forcing dashboards to check for entity
  existence before binding. The sensor now lands in HA from day one
  with `state=0`, empty `lead_hours` / `mae` / `rmse` /
  `sample_count` arrays, and a new `status` attribute whose value
  names the current readiness state: `accumulating` (cold start),
  `no_history_db`, `lab_mode`, `error` (query failed), or `ready`
  once `lead_time_curve` has samples. Transitions to
  `status=ready` populate the arrays in place, so a dashboard
  binding survives the entire lifecycle without conditional logic.

### Changed

- **Add-on permissions tightened in `config.yaml`.** Removed
  `hassio_api: true`, `hassio_role: admin`, and `auth_api: true` —
  the add-on never calls the Supervisor or auth APIs and does not
  need admin role to publish sensors via the Core REST API, which is
  covered by `homeassistant_api: true`. `map` is narrowed from
  `homeassistant_config:rw`, `media:rw`, `share:rw`, `ssl:rw` to a
  single `homeassistant_config:ro` (alongside the existing
  `addon_config:rw` used for the persistent cache and logs). The
  add-on never wrote outside `/data` under the old mapping, so
  dropping write access is a no-op for behaviour but a meaningful
  narrowing of blast radius. Added `stage: stable` so HA surfaces it
  correctly in the add-on store.

- **Removed unused `models` add-on option.** The `options.models`
  list and its `schema` entry in `config.yaml` were legacy scaffolding
  from before the `mlfl.yaml` experiment-config model was introduced;
  nothing reads them. Removed both; add-on options UI now shows only
  the live `log_level` setting.

- **Docs moved to top-level `docs/`.** `CONFIG_GUIDE.md`,
  `FEATURES_GUIDE.md`, and `PREPROCESSING_GUIDE.md` have migrated
  from `ml-forecast-lab/ml_forecast_lab/` (Python package dir) to
  the repo's top-level `docs/` directory so they are a) no longer
  bundled into the Docker image, b) linkable directly on GitHub
  without going three levels deep. Deleted the stale
  `CORE_MODULES_README.md`, `CREATION_REPORT.md`, `INDEX.md`,
  `MODULES.md`, and `README_MODULES.md` files that had drifted out
  of sync with the current architecture (still referenced
  NeuralProphet, 5-backend counts, legacy `publishing.py`, pre-2.26
  sensor names).

- **`repository.yaml` url / maintainer corrected.** URL pointed at
  the archived `psweens/ha-addons` repo; now points at
  `psweens/ml-forecast-lab`. Maintainer name normalised to `Dr Paul
  W. Sweeney` to match the footer and git config.

- **README.md refreshed.** Model count corrected from 15 → 14
  (NeuralProphet removed in 2.24.0 but the count wasn't caught
  across the feature table, config example comment, architecture
  diagram, and dependencies list). Added feature sections for
  uncertainty intervals (conformal bands), forecast accuracy
  tracking with revision improvement, and load_subtract. Noted
  that neural backends' loss function (MSE / MAE / Huber) is
  configurable per experiment.

## 2.26.3

### Added

- **Phase-tag log formatter.** Every log line now carries a short `[BENCH]`,
  `[MODEL]`, `[WEB]`, `[APP]`, `[HA]`, `[DB]`, `[PREP]`, `[FEAT]`, `[COV]`,
  `[PUB]`, `[CFG]`, `[SOLAR]`, `[TRAIN]`, `[DASH]` or `[MLFL]` tag derived
  from the logger's module name, so `mlfl.log` can be filtered per
  subsystem with a simple grep. A new `_PhaseFormatter` in `__main__.py`
  injects the tag and the two handler format strings (`LOG_FORMAT` /
  `LOG_FORMAT_FILE`) left-align it in a 7-char column so the message
  column stays vertically aligned. No new dependencies — stdlib only,
  so the raw `mlfl.log` that the `/log` web endpoint streams back stays
  readable as plain text (no ANSI junk).

- **Per-fold and per-model progress markers in the benchmark runner.**
  With 14 backends × 5 folds = 70 inner iterations, the previous log
  went silent between `"Starting benchmark"` and the final leaderboard.
  Each fold now emits a start line — `[fold 3/5] lstm: train=N test=N` —
  and a completion line with the production metric plus train/infer
  timings — `[fold 3/5] lstm done: mae=0.0834 (train=11.3s, infer=0.42s,
  total=12.1s)`. The outer model loop adds a `[model 3/14] Running
  lstm` banner and a matching `[model 3/14] lstm finished in Xs`
  marker. Trial progress in the tuning path was already in shape and
  is unchanged.

- **Sensor-publish start line.** `_publish_forecast_sensors` now logs
  a single INFO line before it begins building payloads, naming the
  base entity, model, horizon (`N×Mmin (Xh)`), and the headline
  `next` value with units. Pairs with the existing success summary so
  a publish cycle has an obvious start and end in the log — previously
  you only saw the summary after the fact.

### Changed

- **Benchmark error logs now name the model.** `Feature building
  failed for fold N`, `Model training failed for fold N`, and `Model
  prediction failed for fold N` used to print only the fold index, so
  a 14-model run with one bad backend forced you to correlate timing
  to figure out which model crashed. They now include `model=<name>
  fold=<i>/<N>` so the failing backend is obvious.

- **Sensor-publish completion summary surfaces `next` and band
  width.** The `Published X/N sensors for <exp>` line now carries the
  headline `next=<val> <units>` and — when conformal bounds landed —
  `bands=±80%`, matching the start line so the pair brackets a
  complete picture of what went to HA.

### Fixed

- **Exception handlers that swallowed tracebacks.** 27 `logger.error`
  / `logger.warning` sites across `benchmark/runner.py` (4 fold/model
  failure paths), `db.py` (13 query/IO errors), `web/app.py` (16 API
  endpoint handlers), both tree model backends, `models/registry.py`,
  `covariates.py`, `ha_interface.py`, the heartbeat, the conformal
  band path, the forecast-accuracy prep, and a tuning trial-failure
  warning — all interpolated `{e}` into the message but never passed
  `exc_info=True`, so the actual stack trace was lost. All now
  propagate the traceback, turning "Failed to publish / query / load"
  warnings from one-liners into debuggable reports. The per-sensor
  publish failure uses the captured exception instance via
  `exc_info=err` so only failures with an exception print a traceback;
  `set_state returned False` cases stay concise.

## 2.26.2

### Fixed

- **Sensors within an experiment no longer drift apart in publish time.**
  `_publish_forecast_sensors` used to await each `set_state` call
  sequentially — `_forecast`, `_interval`, `_cumulative`, `_upper_{pct}`,
  `_lower_{pct}`, `_forecast_accuracy` — so HA stamped each entity with
  its own `last_updated`, spread across ~300 ms–1 s per cycle (worst
  when the accuracy sensor's DB read landed before its publish). All
  six set_state calls are now collected into a `payloads` list and
  fired together via `asyncio.gather`, so every sensor for one
  experiment lands at effectively the same HA `last_updated`.

- **Upper/lower interval bounds shared a single `try/except`.** If the
  upper publish raised, lower was silently skipped for that cycle,
  leaving the band one-sided until the next forecast. Each publish is
  now wrapped in its own `_publish_one` coroutine; a failure on one
  entity never suppresses its sibling.

- **`set_state` failures logged as successes.** `HAInterface.set_state`
  catches `RuntimeError` internally and returns `False` instead of
  raising. The publisher discarded that return value, so any failed
  POST (timeout, 5xx, stale SUPERVISOR_TOKEN) still emitted
  `"Published forecast curve to …"` at INFO. Return values are now
  checked; failures log `"Failed to publish …"` at WARNING and the
  experiment's `last_error` surfaces the count/list of missing
  sensors.

- **Three independent `datetime.now()` calls per publish cycle.** The
  forecast_log `issued_at`, the cumulative sensor's local-midnight
  boundary, and each sensor's implicit HA `last_updated` all came from
  separate clock reads. A single `issued_at` is now captured at the
  top of `_publish_forecast_sensors` and reused for `log_forecast`,
  the reset-daily partition, and as a new `issued_at` attribute on
  every sensor — so the full cycle has one consistent issuance time.

- **`last_trained` attribute only on `_forecast`.** It's now carried
  on every sensor (main, interval, cumulative, upper/lower, accuracy)
  via a shared `common_attrs` dict, alongside `model` and the new
  `issued_at`. Dashboards can correlate any sensor back to its
  training epoch without reading from the main entity.

- **Global `_forecast_running` flag allowed a same-tick double-fire
  race.** Because `asyncio.create_task` doesn't execute eagerly, two
  experiments scheduled in the same main-loop iteration could both
  pass the `not running` check before either task body set the flag;
  conversely, the one that finished first would flip the flag False
  while the other was still running. Replaced with a per-experiment
  `Dict[str, bool]` and a synchronous slot reservation in the
  scheduler before `create_task`, so each experiment's lock is
  independent.

- **Silent early-return when `ha_interface` was None or `y_pred` was
  empty.** Zero sensors published and the logs had no explanation.
  Now logs a WARNING naming which condition fired.

## 2.26.1

### Removed

- **Dead `publishing.py` module.** The legacy `publish_forecasts`
  helper and its friends (`make_entity_name`, `dict_from_series`,
  `daily_cumulative_series`, `energy_already_used_today`) were
  exported from the package but had no callers — the real publishing
  path is `_publish_forecast_sensors` in `main.py`. After the 2.26.0
  refactor consolidated `_daily_cumulative` into `_cumulative`, the
  stale module was also emitting the old entity with the pre-refactor
  attribute shape (`cumulative.timestamps/values` dict vs the new
  `forecast` list of `{datetime, value}` dicts), which would have
  diverged further over time. Deleted the module, cleaned up
  `__init__.py` imports/exports, and updated `MODULES.md` and
  `README_MODULES.md` to match. No behaviour change for users.

## 2.26.0

### Changed (breaking)

- **Publish sensors automatically, derive semantics from target config.**
  `publish_interval`, `publish_cumulative`, and `publish_daily_cumulative`
  have been removed from `ExperimentCfg`. The publishing path now always
  emits `_forecast` and `_cumulative`, and conditionally emits `_interval`
  when `source_is_cumulative` is true (otherwise it would just duplicate
  `_forecast`). The `_cumulative` sensor's behaviour is derived from the
  target's own semantics: if `source_is_cumulative` and `reset_daily` are
  both set, it resets at local midnight and is seeded with the current
  target value so the forecast meets actuals at the join point; otherwise
  it is a plain `cumsum` from zero across the horizon. A `resets_daily`
  attribute on the sensor exposes which branch was taken so downstream
  consumers (Predbat, chart cards) can adapt without parsing config.

  Migration: remove the three flags from your `mlfl.yaml` experiments —
  `source_is_cumulative` / `reset_daily` (which already exist to describe
  the target) now control the publishing behaviour too. The old
  `{prefix}{name}_daily_cumulative` entity has been consolidated into
  `{prefix}{name}_cumulative`; update any dashboards or automations that
  reference it.

## 2.25.5

### Fixed

- **Conformal-band sensors silently stopped updating** after the
  v2.24.0 upgrade introduced a `model_version` filter on
  `get_conformal_quantiles`. With the fresh version tag, only the
  handful of residuals written *since the latest retrain* matched
  the filter — far below the `min_samples` threshold the quantile
  estimator needs. `fallback_quantile` came back None, so
  `y_pred_upper` / `y_pred_lower` stayed None, `have_intervals`
  stayed False, and the `_upper_{pct}` / `_lower_{pct}` entities
  (plus the `forecast_upper` / `forecast_lower` attributes on the
  main `_forecast` sensor) stopped being written. HA continued
  showing their stale pre-upgrade values — matching the "some
  forecast sensors aren't updating" symptom while the main state
  value kept ticking over.

  Fix: after the version-filtered conformal query, check whether
  it returned a usable fallback quantile + ≥ 10 total residuals.
  If not, re-query pooled across all versions of this model so
  bands keep publishing during the post-retrain cold-start
  window. Logs one `Conformal bands: falling back to all-versions
  pool` INFO line per cycle when this kicks in, for visibility.

## 2.25.4

### Fixed

- **Results-tab model selection not surviving add-on restarts.**
  `/select-model` only updated in-memory state
  (`exp_status.selected_model`), never wrote to YAML. When the
  add-on restarted (which it does daily under the typical retrain
  cadence), `experiment_statuses` was re-initialised with
  `selected_model=None` and the next benchmark auto-promoted its
  top-ranked model. Users experienced this as "I picked XGBoost,
  reloaded the page, and it's showing LightGBM again."

  Three coordinated changes:
  - New `selected_model: Optional[str]` field on `ExperimentCfg`
    (YAML schema).
  - `/select-model` endpoint now persists to YAML via
    `save_experiment_field`, mirroring how `/promote` handles
    `production_model`. Response body includes `persisted: bool`
    so the frontend knows whether it saved.
  - Startup in main.py reads `exp_cfg.selected_model` into
    `ExperimentStatus.selected_model`, with fallback to
    `exp_cfg.production_model` for legacy configs that don't have
    the new field.
  - `/promote` also writes `selected_model` to YAML so the Results
    highlight stays in sync with the production switch across
    restarts.

### Tests

- New `TestSelectedModel` class (3 assertions) covering default
  value, explicit assignment, and YAML roundtrip.

## 2.25.3

### Fixed

- **"Load log summary" button silently doing nothing on some
  experiments.** Rendering the template through Jinja with both
  cumulative and instantaneous contexts produced structurally-
  identical JS (only `EXP_NAME` / `EXP_SOURCE_CUMULATIVE` /
  `expName` / `entityId` differed), so the root cause isn't in
  the template and is likely client-side (browser cache, ingress
  middleware, localStorage state on some setups). Rather than
  keep blind-debugging, added two defensive measures:
  - **Inline try/catch on the button's `onclick`** that writes the
    thrown error message into the `<pre>` output element, so the
    failure is visible without opening DevTools.
  - **"open in new tab" link** next to the button pointing at the
    `/forecast-log-stats` endpoint directly. Pure `<a href>` — no
    JavaScript required, works even if the button handler is
    broken by whatever the client-side state issue is.

## 2.25.2

### Fixed

- **Version-filter mismatch when `selected_model` ≠ `best_model`.**
  `ExperimentStatus.model_version` is a single field that tracks
  *whichever model was last retrained*. When a user had manually
  selected a non-champion model via the UI (e.g. picked `lightgbm`
  from a promote action, but the pipeline subsequently started
  retraining `xgboost`), the default filter the endpoint applied
  was `(model_name=lightgbm, model_version=<xgboost's timestamp>)` —
  an impossible combination that never has rows, forcing the
  fallback ladder to fire on every cycle and display stale
  pre-upgrade data. `_resolve_model_filter()` now only applies the
  version default when `default_model == best_model`; otherwise the
  name filter runs alone ("all versions of that model"), which is
  the correct semantic since no version info is tracked for
  non-champion models in the current schema. The same guard is
  mirrored in `/forecast-log-stats` so its reported
  `current_default_filter` matches what the analytics endpoints
  actually send.

### Added

- **Diagnostic fields on `/forecast-log-stats`**:
  - `selected_vs_best` showing `selected_model`, `best_model`, and
    `matches` — makes the exact mismatch condition above
    immediately visible.
  - `notes` array flagging common conditions: the
    selected-vs-champion drift above, or a default filter that
    points to a cohort with zero rows. Both are written as
    actionable sentences ("re-select `xgboost` in the UI or set
    `production_model: xgboost` in mlfl.yaml").

## 2.25.1

### Added

- **Forecast-log inspector** in the Layer 3 "Diagnostic tools"
  accordion on the Forecast Accuracy tab. A single "Load log summary"
  button fetches a new
  `GET /experiment/{name}/forecast-log-stats` endpoint and
  pretty-prints the JSON in-place, so users without shell access
  can debug "why is my chart empty?" situations directly from the
  web UI. The summary covers: total rows, the default filter the
  UI would apply, a per-`(model_name, model_version)` cohort
  breakdown with row counts and issued/target ranges, and the
  number of targets with ≥2 distinct issuances under the current
  champion+version filter (the stability query's minimum
  requirement). A hint sentence at the bottom explains how to
  interpret the three likely states.

## 2.25.0

### Added

- **Time-zone toggle on the Forecast Accuracy tab** (HA server /
  Browser / UTC). Remote users managing an HA instance in a
  different country (e.g. a California viewer of a UK HA) no longer
  have to mentally shift "the spike at Apr 16 02:00 PDT is
  actually UK mid-morning". On load the add-on reads HA's
  `time_zone` from `/api/config` and the toggle defaults to
  **HA server** so axis labels match when events physically
  happened. Chart tick labels are rendered via `Intl.DateTimeFormat`
  in the chosen TZ, with a TZ abbreviation baked into each axis
  title (e.g. "Target time (BST)"). Selection persists per
  experiment in `localStorage`.
- **HA-local day bucketing for the daily-total stability metric.**
  `SUBSTR(target_dt, 1, 10)` previously took a UTC-day prefix — on
  a BST-hosted HA instance the UK day boundary is UTC 23:00, so
  about an hour of UK-Monday demand was being filed under "Sunday"
  in the bar chart. `get_forecast_stability()` now accepts
  `day_offset_hours`, populated by the endpoint from
  `zoneinfo.ZoneInfo(tz).utcoffset()`, and the SQL shifts the
  timestamp by that offset before taking the day prefix. Correct
  everywhere except the single day containing a DST transition
  (where it's an hour off for that day only).
- **Add-on logging for the Forecast Accuracy pipeline.** Previously
  the whole Forecast-Accuracy write path was silent on success —
  there was no way to tell from the add-on logs whether
  `forecast_log` was actually accumulating rows, whether a schema
  migration had just run, or whether a UI chart was empty because
  of a version-fallback. Three new INFO-level log lines:
  - One per successful forecast cycle, e.g.
    `Logged 96 forecast_log rows for mixergy_demand
    (model=lgb, cached, v=2026-04-20T07:00:00Z, bands)`.
  - One per analytics fallback on `/forecast-accuracy`,
    `/forecast-stability`, `/forecast-trajectory`, naming the
    requested (model, version) and what the query widened to.
  - One per `ALTER TABLE` run inside `ensure_forecast_log_table`,
    listing the columns that were just added to legacy DBs.
- **`EXP_HA_TIME_ZONE` passed through to the experiment template**
  so Jinja doesn't need to assume anything about where the viewer
  is. Falls back to `null` when `/api/config` is unreachable.
- **Tests** — two new cases in
  `TestModelVersion::test_ha_local_day_bucketing_*` (shifts day
  labels correctly for BST; no-op for offset=0).

## 2.24.0

### Added

- **`model_version` column on ``forecast_log``** — an opaque,
  typically ISO-timestamp tag stamped by the pipeline each time a
  model finishes training. Cycles under the same ``model_name`` but
  different weight regimes now live in distinct cohorts, so the
  stability / accuracy / coverage / trajectory queries no longer
  silently pool v1 and v2 predictions together. Fixes the "I retrain
  every 24h under the same name and stability looks terrible"
  pattern that was the underlying cause of the overnight Apr-16
  spikes. Backwards-compatible schema migration: the column is added
  via ``ALTER TABLE ADD COLUMN`` on legacy DBs and existing rows
  carry NULL — they're treated as a separate cohort when a versioned
  cycle appears for the same experiment.
- **`?version=<tag>` / `?version=all` query params** on
  ``/forecast-accuracy``, ``/forecast-stability``, and
  ``/forecast-trajectory``. Endpoints default to the experiment's
  current training tag (``ExperimentStatus.model_version``), with
  ``?version=all`` escaping the default to show every weight regime.
  A two-step **fallback ladder** now runs when the default filter
  empties the result: first widen to "all versions of this model",
  then to "all models". Each fallback step is flagged back to the UI
  via ``model_fallback`` so the user knows what's being shown vs
  requested.
- **Model-version badge in the Layer 1 top controls** — the
  existing model badge now carries a ``since DD MMM HH:MM`` tail
  reflecting the timestamp of the current weights, with the full ISO
  tag on hover. On version fallback the badge reads ``<model> · all
  versions (fallback)`` in amber with an explanation in the tooltip.
- **`_resolve_model_filter()`** helper in ``app.py`` that factors
  the (model, version) resolution out of the three endpoints —
  single source of truth for "current champion + current weights by
  default, with escape hatches".
- **Training path stamps ``ExperimentStatus.model_version``** in
  ``_retrain_and_cache`` as soon as training completes, so the very
  next forecast is written under the new tag. The cached-model dict
  also carries ``model_version`` so ``_publish_forecast_sensors``
  can pin conformal-band residuals to the current weights and avoid
  post-retrain band inflation from stale residuals.
- **Tests** — 5 new cases in
  ``tests/test_forecast_analytics.py::TestModelVersion``:
  schema migration on a legacy table; ``log_forecast`` actually
  stamping the version; v1/v2 stability pooled vs v2-only (flipped
  median CV from 50%+ to <5%); NULL legacy rows correctly excluded
  from a version-filtered query; version filter propagating through
  accuracy / coverage / trajectory alongside stability.

### Changed

- **`/experiment/{name}/promote/{model_name}` is now idempotent.**
  Previously cleared ``forecast_log`` on every call regardless of
  whether the champion name actually changed, which was too
  aggressive — a user re-clicking promote on the already-current
  model would wipe their metrics history. The cleanup now gates on
  ``previous_model != model_name`` (matches the behaviour in
  ``_run_benchmark``). Response echoes the previous→new transition
  in the log line when a clear actually happens.
- ``clear_forecast_log_on_retrain`` (added in 2.23.0) is now
  defensive rather than load-bearing — the ``model_version`` column
  already segregates pre- and post-retrain cycles under the same
  model_name, so the cleanup is mostly for storage hygiene.

## 2.23.0

### Added

- **`stability_focus` experiment-config knob** that drives which
  run-to-run CV the Layer 1 verdict chip and headline sentence read.
  Two values:
  - ``per_moment`` (default) — median cross-cycle CV of predictions
    at the same target moment. Right when the downstream consumer
    cares about *when* demand hits: HVAC setpoints, pre-heat timing,
    battery dispatch.
  - ``daily_total`` — median cross-cycle CV of daily-total
    predictions (cumulative sensors only). Right when the downstream
    consumer integrates over the day: Predbat iBoost heating a hot-
    water tank, EV daily charging budgets, solar-export daily
    planning. For those use cases a ±50% per-moment swing can be
    fine as long as the daily total is stable, and the old
    per-moment-only chip was giving misleading "poor" verdicts.

  The Layer 3 accordion still surfaces both metrics regardless of
  focus — only the Layer 1 chip, headline, and swing-tile label
  follow this setting. Validator rejects ``daily_total`` on
  instantaneous sensors (summing them across a day isn't a physical
  quantity) and rejects unknown values.
- **`clear_forecast_log_on_retrain` experiment-config knob**
  (default ``True``) that prunes ``forecast_log`` rows issued before
  a champion promotion. Old cycles logged under the previous weights
  (even under the *same* model_name) pool into stability metrics and
  produce the "I retrained and now run-to-run looks terrible"
  artefact. Wired into two promotion paths:
  - ``/experiment/{name}/promote/{model_name}`` — the explicit UI
    promote action (app.py) now calls
    ``HistoryDB.cleanup_forecast_log()`` with a "now" cutoff,
    returning ``forecast_log_rows_cleared`` in the response.
  - The automatic champion-change path inside ``_run_benchmark``
    (main.py) does the same when the benchmark actually promotes a
    new name (no-op when the champion hasn't changed).

  Set to ``False`` to preserve full history for offline analysis.

### Tests

- New ``tests/test_config.py::TestStabilityFocus`` (4 assertions)
  and ``TestClearForecastLogOnRetrain`` (2 assertions) covering
  defaults, validation, and the cumulative-gating rule for
  ``daily_total`` focus.
- New ``tests/test_forecast_analytics.py::TestStability::
  test_cleanup_removes_pre_retrain_rows`` pins the promotion-time
  cleanup: seeded pre-retrain cycles with wildly different
  predictions produce CV > 50%; after ``cleanup_forecast_log()``
  removes them, the metric drops below 5%.

## 2.22.2

### Fixed

- **Daily-total stability metric inflated by a coverage artefact, not
  model disagreement.** `get_forecast_stability()` computed
  `SUM(predicted)` per `(issued_at, day)` and called the spread across
  cycles "daily-total CV". But each issuance covers a different
  *fraction* of any given day — an 08:00 run covers ≈32 bins of
  today, a 22:00 run covers 2 bins of today, a prior-day run covers
  all 48. Comparing a 2-bin SUM against a 48-bin SUM as "model
  disagreement" is just reading the coverage gap with extra steps,
  and on a Mixergy-style sensor where overnight carries most of the
  demand the artefact alone can produce 50–70% CV. The
  `per_cycle_day` pipeline now has a `day_max` sub-CTE and keeps only
  cycles whose `n_bins` equals the day's maximum — apples-to-apples
  across cycles. On a seeded three-cycle test (two full, one
  partial) this drops reported CV from 29.93% to 2.44% (~12×), with
  the reduction being entirely artefact.

## 2.22.1

### Fixed

- **Trajectory chart compared predictions to raw cumulative actuals**
  for cumulative sensors. `forecast_log.predicted` stores per-interval
  deltas (that's what the model is trained to emit), but the old
  trajectory query returned the actual as the raw grid-aligned value
  (e.g. 17% fill) — so the "Actual" line sat impossibly high above
  the per-interval forecast dots, turning a useful debugging view
  into a misleading one. `get_forecast_trajectory()` now takes
  `source_is_cumulative` and computes the actual as
  `value − value[t−interval]` (adjacency-guarded, matching the
  increment-mode logic in `get_forecast_accuracy`). `max_abs_error`
  used for the "biggest miss" sort is now in the same space too.
  The UI labels the series *"Actual (per-interval demand)"* on
  cumulative sensors so the units are self-describing.
- **Run-to-run disagreement pooled across every model ever logged**.
  `get_forecast_stability()` had no `model_name` filter, so tinker-era
  runs under rotated-out model names inflated the CV metric. On a
  seeded test series this flipped the reported swing from ±99% (all
  models mixed) to ±3.5% (current champion). The function and the
  `/forecast-stability` endpoint now take `model_name`, default to
  the experiment's selected/best model, and honour `?model=all`.
- **Lead-time chart silently rendered empty when the model filter had
  no data for the current champion**. After a champion swap mid-
  window, filtering the v2.22.0 queries to `selected_model` returned
  zero rows and the chart area stayed blank with no explanation.
  The `/forecast-accuracy`, `/forecast-stability`, and
  `/forecast-trajectory` endpoints now detect "filter empties the
  result but the window has data from other models" and fall back to
  unfiltered, returning a `model_fallback` flag. The UI surfaces this
  as `all (fallback)` in amber on the model badge with a hover
  tooltip explaining why. `Plotly.purge()` is also called on the
  lead-time container in the empty-state branch so the panel visibly
  clears instead of looking like it's still loading.
- **Top controls row layout**: the `Window:` dropdown was inheriting
  `.setting-input { width: 100% }` from the settings page, stretching
  full-width and forcing the `Evaluate as:` toggle onto a new line.
  `select.setting-input` inside `.acc-top-controls` is now
  `width: auto; min-width: 10rem` so both controls sit on the same
  row. Gap and margin tightened.

### Changed

- **Trajectory section renamed** from "One moment, every prediction
  of it" (awkward) to **"How predictions converge on a single
  moment"** — describes the plot's purpose directly.
- **Evaluation-mode hint is now an info-tip** matching the rest of
  the page, replacing the inline *"Recommended — avoids mistaking…"*
  sentence that was noisy and stylistically out of place.

## 2.22.0

### Added

- **Three-layer Forecast Accuracy UI** aimed at HA end-users, not
  just data scientists. The tab now opens on a **verdict card** with
  a plain-English headline ("Looking healthy" / "Something looks
  off") plus three colour chips for Accuracy / Calibration /
  Stability, each reporting good / fair / poor with a one-word
  detail. Three headline tiles show typical next-step error (absolute
  + % of typical demand), interval coverage vs the 80% target, and
  run-to-run swing. Below the verdict, Layer 2 surfaces the drivers
  — a re-worded "Does re-forecasting help?" sentence plus a
  simplified lead-time chart (MAE only by default, RMSE/bias behind
  a toggle). Layer 3 is a collapsed `<details>` accordion holding
  the existing convergence, trajectory, and stability diagnostics
  for deeper debugging.
- **Normalised headline error**: `get_forecast_accuracy()` now
  returns `typical_interval_demand` (mean |actual| over the window,
  mode-aware) so the verdict card can report *"0.4 kWh — 8% of
  typical"* without the client needing to know units or scales.
- **Forecast convergence as a fan chart**: replaces the
  12-coloured-lines overlay with a shaded band (min/max across
  recent runs), a dotted median line, a bright yellow line for the
  latest run, and white for measured. Communicates "how much the
  forecast wobbles between runs, and does the latest land on the
  actual" at a glance.
- **"Biggest miss" sort for forecast trajectory**: the target-picker
  now defaults to the largest `|predicted − actual|` in the window
  rather than the most recent timestamp. Server returns a new
  `target_meta` array with per-target `max_abs_error` and `actual`
  so the UI can sort client-side without a round trip. "Most recent"
  is still available via a dropdown.
- **Plain-English stability**: the run-to-run card now leads with
  *"Forecasts for the same future moment typically disagree by ±X%
  between runs — consistent / a little jittery / noticeably
  unstable"* rather than showing bare CV numbers. Tiles re-labelled
  ("Per-moment swing", "Runs analysed") to match.
- **Window selector** (7 / 30 / 90 days) at the top of the tab that
  cascades to every endpoint on the page, so accuracy, coverage,
  revision, and stability all share the same lookback. Replaces the
  previous hard-coded 30 days.
- **Model badge** in the top controls shows which model the metrics
  filter to — relevant now that the queries restrict to a single
  model by default (see Fixed below). Overridable via `?model=all`.
- **Cold-start aware empty states**: the empty state now
  distinguishes "waiting for HA actuals" vs "no forecasts logged
  yet" vs "nothing in this window", each with its own hint.

### Changed

- **Lead-time chart is MAE-only by default**. A "Show RMSE & bias"
  checkbox reveals the additional series for the DS view.
  Low-sample buckets (n < 10) render as open circles so a 3-sample
  bucket no longer looks as authoritative as a 300-sample one.
- **Revision card rewritten** around a single plain-English sentence
  (*"Errors dropped 18% between first and latest forecast"*) rather
  than a colour-coded `+X%` / `-X%` tile whose direction was
  ambiguous. Each MAE tile now labels its evaluation mode so the
  units aren't lost.
- **Stability charts deferred**: rendered on first accordion open so
  the initial accuracy-tab paint is fast even when there's a lot of
  stability data to plot.

### Fixed

- **`accuracyMode` defaulted to `'raw'` for cumulative sensors at
  script load** because `var EXP_SOURCE_CUMULATIVE` was hoisted but
  not initialised when the ternary read it. The UI silently reported
  raw-value MAE on a cumulative sensor (where 23:30 errors look huge
  purely because the number is huge) while showing "Per-interval
  demand" as the active toggle. Declarations moved ahead of the
  initialiser.
- **Increment mode diffed across actuals gaps**. `value - LAG(value)
  OVER (ORDER BY grid_dt)` assumed the previous row was one interval
  earlier; a 2h HA outage made the delta look like a single-interval
  demand of 2h worth, inflating MAE with data-availability
  artefacts rather than model error. Added an adjacency guard:
  delta is null unless `grid_dt − LAG(grid_dt) = interval`. Applied
  to both the actuals and forecast LAG pointers.
- **No `model_name` filter on accuracy / coverage / revision
  queries**. After a champion swap, metrics aggregated across both
  models until the retention window rolled.
  `get_forecast_accuracy()` and `get_forecast_coverage()` now accept
  an optional `model_name`; the endpoint defaults to the
  experiment's selected/best model and exposes `?model=<name>` or
  `?model=all` for overrides.
- **Unguarded `int()` on `?days=`** in the forecast-accuracy
  endpoint returned 500 on a bad value. Now guarded + clamped to
  [1, 365], matching the stability endpoint.
- **Stability CV = 0 when mean ≈ 0**: low-demand timesteps with
  non-zero prediction std were reported as "perfectly stable"
  because the CV ratio collapsed. Those rows are now skipped from
  the per-timestep series and the median-CV aggregate rather than
  polluting the chart with false zeros.

## 2.21.0

### Added

- **Signed bias (ME)** alongside MAE/RMSE in the lead-time accuracy
  curve. A third trace (dashed green, zero-reference line) reveals
  whether the model systematically over- or under-predicts — a
  dimension that absolute-error metrics collapse. Surfaced at the
  lead-time bucket level and in the revision summary
  (`first_forecast_me`, `latest_forecast_me` sit beneath each MAE).
- **Forecast trajectory plot** — a new "Forecast trajectory" section
  renders every forecast ever issued for a chosen `target_dt` as dots
  on the `issued_at` axis, with the actual as a horizontal reference
  line and the target moment as a dotted vertical marker. Answers "did
  the prediction walk smoothly toward truth, oscillate, or drift?" —
  shape information the first-vs-last revision metric collapses to two
  numbers. Backed by `HistoryDB.get_forecast_trajectory()` and a
  `/experiment/{name}/forecast-trajectory` endpoint; the dropdown is
  auto-populated with recent targets that have an actual AND >=2
  distinct issuances (single-forecast targets teach nothing so they're
  filtered out).
- **Increment-based evaluation mode** for cumulative sensors. When
  `source_is_cumulative=True`,
  `get_forecast_accuracy(..., evaluation_mode="increment")` uses LAG
  window functions to diff both forecasts and actuals before joining,
  filters midnight-reset rows (negative increments on daily-resetting
  sensors), and computes MAE/RMSE/ME on per-interval demand rather
  than raw cumulative value. Errors become comparable across the day
  instead of being dominated by the sensor's shape. The UI exposes a
  "Evaluate on: Per-interval demand / Cumulative value" toggle, which
  defaults to increment for cumulative targets and is hidden otherwise.
- **Conformal 80% prediction intervals** via online/adaptive conformal.
  `HistoryDB.get_conformal_quantiles()` computes per-lead-bucket 90th
  percentiles of absolute residuals from recent `forecast_log` rows vs
  the actuals table; `_publish_forecast_sensors` auto-applies these to
  produce upper/lower bands around each point forecast. New HA
  entities `sensor.{prefix}{exp}_upper_80` and `_lower_80` expose the
  band in their `forecast` attribute; the main forecast sensor also
  carries `forecast_upper` / `forecast_lower` arrays for in-chart
  shading. No additional model fit required — residual quantiles are
  learned from deployed forecast/actual pairs, so bands appear
  automatically once the residual buffer fills and stay absent during
  cold-start.
- **Interval coverage diagnostic**: `HistoryDB.get_forecast_coverage()`
  joins `forecast_log` rows that carry both bounds against actuals and
  reports the fraction inside `[lower, upper]`, per lead-bucket and
  overall. Surfaced as a calibration card on the accuracy tab —
  empirical coverage vs nominal 80%, colour-coded green within +/-5pp,
  amber within +/-10pp, red beyond, with a plain-English diagnosis
  ("Well calibrated", "Bands are wider than needed", "Bands are too
  tight").

### Changed

- `forecast_log` schema gained nullable `upper`, `lower` columns with
  an auto-migration path: `ensure_forecast_log_table` now runs
  `PRAGMA table_info` and appends the columns with `ALTER TABLE ADD
  COLUMN` when missing. Point-only legacy rows remain supported — the
  coverage query filters to rows with non-null bounds.
- `HistoryDB.log_forecast()` accepts optional `upper_bounds` /
  `lower_bounds` lists (same length as `predictions`); existing
  callers unchanged.
- `_publish_forecast_sensors()` signature gained optional
  `y_pred_upper`, `y_pred_lower`, `interval_level` params; when
  omitted and a residual history is available, bands are auto-computed
  inside the helper so both the full-retrain production path and the
  cached-forecast path get intervals without caller-side changes.
- `/experiment/{name}/forecast-accuracy` accepts `?mode=raw|increment`
  (defaults to `increment` for cumulative sensors) and merges the
  coverage result into its JSON payload alongside `lead_time_curve`
  and `revision_improvement`.

### Notes

- Conformal calibration is online/adaptive rather than split-conformal:
  residual exchangeability is approximate (temporal drift, model
  retrains), so coverage guarantees are empirical rather than
  finite-sample. The coverage card exists precisely to surface
  deviations from the nominal 80%, so the user can tell when bands
  drift and need recalibration.
- Cold-start: coverage card stays hidden and bands stay absent until
  `forecast_log` has accumulated enough residuals per lead bucket. The
  system silently falls back to point-only forecasts in the meantime.

## 2.20.0

### Added

- **Load subtract**: new per-experiment `load_subtract` config field for
  removing sensor contributions (EV charging, iBoost solar-divert, etc.)
  from the target load before training, so models learn the baseline
  household pattern rather than a mixed signal. Each entry is a
  `SubtractCfg` with explicit `source` (`cumulative_daily`,
  `cumulative_monotonic`, `interval`, `auto`), `on_missing` policy
  (`zero`, `drop`, `error`), optional `scale` for unit conversion
  (Wh→kWh), and `max_fraction_of_load` / `max_fraction_violation_pct`
  guards that fail fast on unit bugs or double-counted signals. Replaces
  the unwired legacy `subtract: [str]` stub, which is now deprecated and
  logs a warning when present in YAML.
- **Robustness layer** in `preprocessing.apply_load_subtract()`:
  per-sensor unit scaling, missing-data policy enforcement, negative
  clipping with clipped-row count and >5% warn threshold, fraction guard
  that raises `LoadSubtractError` with a worst-row diagnostic
  (`ratio=X.XX subtract=Y load=Z`) when subtract exceeds load on too
  many rows, leading-gap window detection for history coverage, and
  tz-awareness mismatch detection.
- **Audit logging** in the training pipeline: per-run boxed summary
  showing total load, total subtracted (with % share), clipped rows, and
  a per-sensor breakdown of sum/missing/max-fraction/violations/gap
  window. Same numbers that would drive a dry-run, emitted at training
  time.
- **Load Subtract UI card** on each experiment's config tab with entity
  picker (HA search), source / on_missing / scale / max_fraction inputs,
  and row removal — mirroring the Covariates card. Three new endpoints
  (`POST /experiment/{name}/add-load-subtract`,
  `remove-load-subtract`, `clear-load-subtract`) wrap the YAML helpers.
- **Pipeline placement chosen deliberately**: subtract runs AFTER
  cumulative→interval conversion and resample, but BEFORE outlier
  clipping. Outlier bounds are then computed on the adjusted (baseline)
  signal rather than on raw spikes from the subtracted component, so
  real baseline peaks aren't muted by EV / iBoost outliers.

### Changed

- **`searchEntities` JS** now finds its results dropdown via the input's
  `.entity-search-wrapper` parent rather than a hardcoded
  `#entity-results` ID, so multiple entity pickers (Covariates + Load
  Subtract) coexist on the experiment config page. Outside-click handler
  closes all open result divs rather than only the covariate one.

### Deprecated

- `ExperimentCfg.subtract: list[str]` — was never wired into the
  pipeline. Now emits a deprecation warning at `load_config` time
  telling users to migrate to `load_subtract`. No behaviour change for
  anyone who had it set (it did nothing before; it does nothing now).

### Tests

- **29 new tests** in `tests/test_load_subtract.py` covering
  `SubtractCfg` validation, YAML round-trip (including bare-string
  tolerance and legacy-`subtract` deprecation warning), add/remove/clear
  helpers, and every branch of the robustness checklist:
  perfect-alignment, `on_missing` policies, leading-gap detection,
  negative clipping, fraction-guard firing on unit bug, fraction-guard
  tolerating noise band, scale application, tz mismatch, and
  multi-sensor summation.

## 2.19.0

### Removed

- **NeuralProphet backend deleted.** The `neuralprophet_backend.py` module,
  its schema and catalog entries, the optional-backend registration in
  `main.py`, the sample `mlfl.yaml` entry, and the `neuralprophet` pin in
  `requirements.txt` are all gone. Docstrings in `config.py` that called
  out NeuralProphet as a special case have been simplified to just "tree
  models". Any existing `mlfl.yaml` with `"neuralprophet"` in
  `models_enabled` will silently skip the model (the registry no longer
  registers it) — remove the line to keep configs tidy.

### Changed

- **Models tab grouped into Tree Models and Neural Models**, both on the
  standalone `/models` page and each experiment's Models tab. Grouping is
  driven by `model_type == 'Tree'` on the existing catalog — no new field.
- **Model identifiers render as display names across the UI.** A new
  `model_display` Jinja filter and matching `modelDisplay()` JS helper map
  internal ids (`cnn`, `lightgbm`, `nbeats`, …) to their catalog
  `display_name` everywhere user-visible: comparison tables, fold-stability
  chart legends, holdout / residual legends, feature-importance headings,
  the tuning banner and holdout chart title, the live-training "Current
  Model" stat and event log, the model-select dropdowns on Tuning and
  Covariate Analysis, and the dashboard card's Best Model / Training
  badge. CNN is now CNN, not cnn.
- **Descriptions corrected to match implementations.**
    - LSTM: was "2-layer LSTM …" (the schema allows 1–8 layers);
      now "Multi-layer LSTM with temporal attention and multi-horizon
      output head."
    - TiDE: was "Time-series Dense Encoder with residual MLP
      encoder-decoder." (undersold the paper implementation); now "Dense
      encoder-decoder with temporal decoder and global residual skip."
- **`batch_size` and `loss_fn` excluded from the Optuna search space.**
  Both now carry `"tunable": False` in `MODEL_PARAM_SCHEMA` and the
  tuning objective loop skips any spec flagged non-tunable. Rationale:
  `batch_size` is a compute/memory knob (tuning already force-overrode
  it to 16), and `loss_fn` decides *what* is being optimised — tuning it
  makes the composite score non-comparable across trials. Both remain
  manually configurable on the Models page; `loss_fn` is still set
  per-experiment in Settings and propagated to every trial via
  `_apply_experiment_neural_params`.

## 2.18.0

### Changed

- **Daily cumulative loss is now trajectory-matching** (was: mean over
  horizons). `daily_loss_weight > 0` now penalises error in the predicted
  cumulative *curve* at every horizon step — not just a single aggregate
  constraint at the endpoint.

  **New formula** (in ``_composite_horizon_loss``):
  ```
  L_daily = mean_h  criterion( cumsum(ŷ)[h], cumsum(y)[h] )  /  H
  ```
  The cumulative error is penalised at every h=1..H, then averaged and
  normalised by H so λ stays reasonably invariant to horizon length.

  **Previous formula** (v2.16–v2.17):
  ```
  L_daily_old = criterion( mean_h(ŷ), mean_h(y) )
  ```
  The mean-of-horizons match was a single scalar per sample — a straight
  line ramp and the actual stepped daily-demand curve can have identical
  means and still look completely different visually, so the term had
  almost no effect on training.

  **Why this matters for cumulative-origin targets**
  (``sensor.x_today``-style sensors that reset at midnight): the training
  objective now directly aligns with the cumulative curve users evaluate
  against. Per-interval predictions that regress to the mean on sparse
  targets produce a constant-slope cumulative ramp that misses the actual
  curve shape — the trajectory loss penalises this drift at every
  intermediate horizon.

  **Breaking behaviour note**: experiments that already have
  ``daily_loss_weight > 0`` will train differently after upgrading.
  Models retrained on v2.18+ will produce different forecasts than
  v2.16/v2.17 did with the same setting. The default (``0.0``) preserves
  pre-composite-loss behaviour byte-identically, so experiments that
  never toggled the setting are unaffected.

  **Single-horizon outputs** (``future_periods=1`` or ``y_pred.dim()==1``)
  silently skip the daily term — cumsum of a single value equals the
  value itself, so the term would be redundant with the interval term.

### Context

Real-world testing on a Mixergy hot-water demand target
(``sensor.mixergy_demand_today``, a cumulative-reset-daily sensor) showed
the v2.16 mean-based daily loss had no measurable effect. The mean of the
forecast horizon is approximately matched by any unbiased model
regardless of curve shape, so the penalty term had no gradient to
contribute. The trajectory formulation directly penalises the
drift-accumulation failure mode.

### Audit note

Pre-release audit of the ``source_is_cumulative=True, reset_daily=True``
preprocessing path (``cumulative_to_interval``) confirmed the training
signal is clean for cumulative-reset-daily targets — primary reset
detection via ``diffs < 0`` (preprocessing.py:86) is timezone-independent
and correctly identifies real resets regardless of local-vs-UTC
midnight alignment. Two minor cosmetic issues in the publishing layer
(``daily_cumulative_series`` doesn't include a today-so-far seed when
aligning with the source sensor) will be addressed in a separate
release; they don't affect training.

## 2.17.0

### Added

- **Optimiser choice (neural)** — new per-experiment dropdown in Settings →
  Training selects between **AdamW** (default, decoupled weight decay — the
  optimiser every published time-series transformer paper uses) and
  **Adam** (classic, tied weight decay). Both share the same
  ``learning_rate``, cosine schedule, and ``weight_decay=1e-4``, so the
  choice isolates the decoupled-vs-tied weight-decay behaviour rather than
  confounding it with decay magnitude or LR.

  Default remains AdamW, preserving pre-2.17 behaviour for existing
  experiments. Applied to all 12 torch neural backends; silently ignored
  by NeuralProphet and tree models.

  Implementation: new ``ForecastModel._build_optimiser`` static helper on
  the base class replaces the previously-hardcoded ``torch.optim.AdamW(...)``
  line in each of the 12 backend ``fit()`` methods; hyperparam
  round-trips through ``get_params``/``set_params``/save/load.

### Fixed

- **Covariate Analysis and Tuning now honour Settings-level neural params.**
  Previously, the Covariate Analysis and Hyperparameter Tuning code paths
  instantiated neural models with the backend's defaults — so if you
  selected ``loss_fn='huber'``, ``daily_loss_weight>0``, or (as of 2.17)
  ``optimiser='adam'`` in Settings, those choices were silently ignored by
  anything outside the main CV loop and the production-training path. The
  main benchmark minimised one objective; Covariate Analysis and Tuning
  minimised another (typically MSE with λ=0 and AdamW), which made the
  two flows incomparable and hid the effect of Settings changes.

  New helper ``_apply_experiment_neural_params(model, exp_cfg, overrides=…)``
  in ``main.py`` propagates ``loss_fn``, ``daily_loss_weight``, and
  ``optimiser`` from the experiment config to any freshly-created neural
  model, guarded by ``hasattr`` (tree models / NeuralProphet silently skip)
  and respecting caller-supplied ``overrides`` (so Optuna-swept params and
  ``model_overrides`` still take priority). Called from:
  - Tuning baseline trial (``_run_tuning``)
  - Tuning per-trial objective
  - Holdout re-fit in Tuning (``_run_holdout``)
  - Covariate Analysis per-config-per-model fit

  The 4 pre-existing ``set_params(loss_fn=…)`` call-sites in the main CV
  and production paths are left unchanged — they already handle the same
  propagation inline and continue to work.

## 2.16.0

### Added

- **Daily cumulative loss (optional)** — new per-experiment toggle that adds a
  horizon-aggregate term to the training loss of all 12 torch neural
  backends. Useful for cumulative-origin targets (e.g.
  `sensor.mixergy_demand_today`, energy meters) where per-interval errors
  otherwise accumulate across the forecast window into a wrong daily total.

  **Loss form:** `L = L_interval(ŷ, y) + λ · L_interval(mean_H(ŷ), mean_H(y))`
  where the second term applies the existing `loss_fn` (MSE / MAE / Huber) to
  the mean-over-horizons of each sample. With `future_periods=48` and
  `interval_minutes=30` the horizon spans 24 h, so the term is exactly a
  rolling daily cumulative. Using the mean (not sum) keeps both terms on the
  same scale so λ is intuitive; single-horizon outputs silently skip the
  daily term.

  **λ:** hardcoded at 0.5 when the toggle is on (a sensible
  "daily term is half as important as interval term" default), 0.0 when off.
  Sophisticated users can still hand-edit `daily_loss_weight` as a float in
  `mlfl.yaml` — the app.py validator and `ExperimentCfg` field accept any
  value ≥ 0; the UI toggle just writes the two canonical ones.

  **Applies to:** LSTM, CNN, N-BEATS, N-HiTS, TiDE, DLinear, TSMixer,
  PatchTST, iTransformer, Crossformer, TimesNet, SparseTSF. Silently ignored
  by NeuralProphet (internal fit loop, no loss-hook) and tree models
  (LightGBM, XGBoost — sample-wise gradient API doesn't fit a cross-sample
  aggregate loss; revisit via post-hoc reconciliation if needed).

  **Implementation:** one new `ForecastModel._composite_horizon_loss`
  static helper on the base class; each of the 12 backends replaces its
  `criterion(...).mean()` train and val loss blocks with a single
  helper call; the hyperparam round-trips through `get_params`/`set_params`.
  Default 0.0 preserves byte-identical behaviour with pre-2.16 runs.

  UI toggle lives in the experiment Settings → Training section, next to the
  output-activation selector.

## 2.15.0

### Added

- **Reversible Instance Normalization (RevIN)** — per-window, per-channel
  normalisation (Kim et al. 2022,
  https://openreview.net/forum?id=cGDAkQo1C0p) added to the shared `base.py`
  and plumbed into every PyTorch neural backend except N-BEATS and N-HiTS:
  LSTM, CNN, DLinear, TiDE, TSMixer, SparseTSF, PatchTST, iTransformer,
  Crossformer, TimesNet. On by default (`use_revin=True`) because the
  reference implementations of these papers all ship with RevIN — without
  it, the codebase was systematically under-performing published benchmarks
  on non-stationary series.

  Each enabled backend now:
  - wraps its forward pass with input `normalize()` (per-sample, per-channel
    mean/std over the time axis, plus a learnable affine),
  - reverses the target-channel stats on the head output via `denormalize()`
    before the output activation,
  - stores/restores `use_revin` and `target_channel` in `get_params`,
    `set_params`, `save`, and `load`,
  - skips the pre-v2.15.0 dataset-level channel normalisation and the zscore
    target z-scoring when RevIN is active (the two schemes are mutually
    exclusive). `output_activation='zscore'` is now a no-op when RevIN is on.

  N-BEATS and N-HiTS are left untouched because their doubly-residual
  backcast-subtraction stacking handles instance-level normalisation
  architecturally; layering RevIN on top would double-normalise.

- **TiDE future-covariate path** — TiDE is now the full architecture from
  Das et al. 2023, with:
  - a feature-projection residual block for known-future covariates,
  - a per-horizon temporal decoder that fuses the decoder state with each
    horizon step's projected future features,
  - a global linear residual from the past window straight to the forecast.

  `fit()`, `predict()`, and `predict_sequence()` now accept an optional
  `future_covariates` kwarg of shape `(n_samples, n_horizons,
  n_future_covariates)`. Typical contents: calendar features (hour of day,
  day of week, holiday flag), externally-forecast weather (Solcast GHI
  p10/p50/p90, Open-Meteo temperature), or a known-future schedule. When
  not provided, TiDE degrades gracefully to a dense encoder-decoder on the
  past window alone (backwards-compatible with pre-v2.15.0 behaviour).

- **Config:** `ExperimentCfg.use_revin` (default `True`) and
  `ExperimentCfg.future_covariate_features` (default `[]`) added to expose
  the two features to experiment YAMLs.

### Changed

- **Config doc for `output_activation`** notes that `zscore` is superseded
  by RevIN when `use_revin=True`.

- **TiDE backend** is now a full paper replication rather than a simplified
  dense encoder-decoder. Older TiDE checkpoints load with
  `n_future_covariates=0` and the global-residual + temporal-decoder paths
  still function — the network shape degrades cleanly, but you'll want to
  retrain to get the temporal-decoder weights.

## 2.14.0

### Added

- **`output_activation: zscore` is now honoured by every PyTorch neural
  backend** — previously only the LSTM applied target z-score
  normalisation; the other eleven (CNN, DLinear, N-BEATS, N-HiTS, TiDE,
  TSMixer, SparseTSF, PatchTST, iTransformer, Crossformer, TimesNet)
  silently degraded to `linear`. Each backend now:
  - fits per-horizon target mean/std on training data (scalars for
    single-horizon, per-column arrays for multi-horizon),
  - trains with a linear head in z-space,
  - denormalises predictions back to physical units in
    `predict()` and `predict_sequence()`, flooring at zero, and
  - persists `y_mean`/`y_std` in `save()` / `load()` so checkpoints
    round-trip cleanly.

  Predictions from a fresh zscore-trained checkpoint are now on the
  correct physical scale for all backends. Older checkpoints saved
  under a non-zscore activation load with identity defaults and are
  unaffected.

- **Config:** documentation for `ExperimentCfg.output_activation`
  updated to reflect full-backend coverage.

### Changed

- **TimesNet: real multi-period aggregation** — the TimesBlock
  previously detected the top-k dominant FFT periods but then
  discarded everything except the median, collapsing the block to
  single-period 2D convolution and defeating the paper's core
  contribution. It now runs the shared inception block once per
  detected period and aggregates the results with softmax-weighted
  amplitudes (matching Wu et al. 2023). Periods that round to the
  same integer bucket have their amplitudes merged before softmax.

- **NeuralProphet: reproducible timestamps** — `fit()` used to call
  `pd.Timestamp.now()` for every fit, so repeated fits on the same
  `(X, y)` produced different temporal features and broke
  reproducibility. The backend now accepts `date_index` via kwargs
  (preferred — preserves real calendar effects for seasonality /
  holiday components) and falls back to a fixed `2000-01-01` anchor
  when the caller can't supply one.

### Fixed

- **N-HiTS: defensive copy of `pool_kernels`** — the inner
  `_NHiTSNet` mutated the caller's list when extending to `n_stacks`,
  which could silently corrupt a reused kernel list across multiple
  model constructions.

- **Post-hoc clipping restored for all neural backends on cumulative
  targets** — v2.13.2 added the clip inside the LSTM backend's
  `predict()` / `predict_sequence()` for the zscore path only.
  Every other neural backend (PatchTST, N-BEATS, TiDE, TSMixer,
  SparseTSF, TimesNet, iTransformer, Crossformer, DLinear, NHiTS,
  CNN) still had an unguarded linear-output edge case: with a
  manual `output_activation: linear` override, tiny negatives could
  slip through into live forecasts and the holdout chart. The clip
  is now applied in the three central neural forecast paths in
  `main.py` (`_run_production_inference`, `_forecast_with_cached`,
  and `_generate_holdout_predictions`), gated on
  `exp_cfg.source_is_cumulative=True` so signed-target experiments
  (temperature deltas, net flows) are not clipped.

  Scope note: tree-model paths are unchanged — they already cannot
  emit negatives for non-negative training targets under normal
  settings, and the existing log-transform branch's
  `np.maximum(np.expm1(...), 0.0)` still owns that case. The LSTM
  backend's internal clip from v2.13.2 is also kept, and the new
  denormalise-and-clip step in every other neural backend's
  `predict()` / `predict_sequence()` (see the zscore work above)
  provides a third layer for the zscore path.

- **Docstring drift:** CNN and LSTM module docstrings no longer
  claim `ReduceLROnPlateau` scheduling — both backends use
  `CosineAnnealingLR` (unchanged; documentation only).

## 2.13.2

### Change

- **LSTM zscore path now clips predictions at zero** —
  `output_activation: zscore` uses a linear head in z-space, and
  denormalising a slightly-negative z-prediction can emit values
  below zero for physically non-negative targets. v2.13.0 shipped
  without the clip (explicit "try without clipping first" test);
  with the unclamped output now confirmed to show a bi-modal
  flat-at-mean regime, the clip is restored in `predict()` /
  `predict_sequence()` immediately after the z-unnormalisation step.
  Matches the pre-v2.11.0 behaviour for the same (z-score + linear)
  training setup.
- **Diagnostic hypothesis under test:** the pre-today wavy-looking
  forecasts may have been clipped flat-at-zero predictions
  overlaid with occasional peaks. Clipping here won't fix the
  underlying encoder-state dominance but will let us visually
  separate "network output" from "masked-by-clip floor" in the
  evolution chart.

## 2.13.1

### Bugfix

- **`zscore` now appears in the Settings → Output activation
  dropdown** — v2.13.0 shipped the backend but the UI dropdown and
  its `?` helper tooltip still listed only the six pre-zscore
  options. The dropdown now includes *Z-score (normalised target,
  LSTM)* and the tooltip documents both the new option and the
  updated `Auto` resolution (zscore for LSTM, softplus/linear
  otherwise).
- **Settings save endpoint now accepts `zscore`** — the server-side
  validator whitelist was missing the new value, so even setting it
  via `mlfl.yaml` edit would have been stripped on any subsequent
  UI save of any other field.

## 2.13.0

### Feature

- **New `output_activation: zscore` option (LSTM target normalisation
  restored)** — when the LSTM is configured with `output_activation:
  zscore`, training targets are z-score normalised per-horizon (mean
  subtracted, divided by std computed on training data), the network
  predicts in z-space with a linear head, and predictions are
  denormalised back to physical units at inference time. This keeps
  loss-landscape curvature bounded regardless of raw target magnitude
  — making the LSTM work well out-of-the-box on targets ranging from
  small fractions to large cumulative values. No post-hoc clipping
  is applied; predictions are returned exactly as the model emits
  them (signed, potentially slightly negative near zero).
- **`output_activation: auto` now resolves to `zscore` for the LSTM
  backend** — previously `auto` meant `softplus` for cumulative
  sources and `linear` for instantaneous. Other neural backends
  (TiDE, iTransformer, PatchTST, NHiTS, N-BEATS, CNN, DLinear,
  TSMixer, SparseTSF, TimesNet, Crossformer) keep the physics-based
  `softplus` / `linear` default. The LSTM-specific change is
  motivated by empirical evidence that un-normalised targets cause
  the recurrent encoder to produce a flat vs. wavy bi-modal forecast
  pattern on high-amplitude cumulative targets (observed on Mixergy
  `demand_today`).
- **Checkpoint format: LSTM checkpoints now include `y_mean` / `y_std`
  keys** — pre-v2.13.0 checkpoints load with identity defaults
  (`y_mean=0`, `y_std=1`) so backward-compat is preserved; new
  zscore-trained models persist their per-horizon stats alongside
  the existing `channel_mean` / `channel_std` input-scaling stats
  and `sigmoid_scale` buffer.

### Notes

- Tree backends (LightGBM, XGBoost) continue to ignore
  `output_activation` entirely. Non-LSTM neural backends accept
  `zscore` but treat it as `linear` (linear head, no target
  normalisation) — a no-op fallback rather than an error.

## 2.12.1

### Bugfix

- **Forecast evolution and stability charts now display in local
  time (respects DST)** — timestamps are stored in `forecast_log` as
  UTC strings, but the charts were handing the bare strings to Plotly
  which then rendered them in UTC. For a user in BST (or any non-UTC
  TZ) this showed events an hour off the wall-clock and mis-aligned
  the midnight resets in cumulative view. Timestamps are now parsed
  as UTC (`new Date(s + 'Z')`) and passed to Plotly as `Date` objects,
  which it renders in the browser's local TZ. The x-axis label also
  shows the active TZ abbreviation (e.g. `Target time (BST)`).
- **Cumulative-reset boundary uses local date, not UTC date** —
  matches the behaviour of the underlying HA `_today` sensors, which
  reset at local midnight. Previously a BST user saw the cumulative
  curve dropping to zero at 01:00 local (UTC midnight); now it resets
  at 00:00 local as expected.

## 2.12.0

### Feature

- **Model stability panel on the Forecast Accuracy tab** — new
  section that measures *self-consistency* of the model's predictions
  across forecast issuances, distinct from accuracy (which compares
  predictions against actuals). Useful for diagnosing an unstable
  model whose forecasts swing wildly issuance-to-issuance even when
  the target isn't changing.
  - **Median step CV%** scorecard: across-cycle coefficient of
    variation of predictions for the same future timestamp.
    Colour-coded green <10%, amber <25%, red ≥25%.
  - **Median daily CV%** scorecard (cumulative-source experiments
    only): across-cycle CV of predicted daily totals.
  - **Per-timestep std chart**: x = target time, y = std of
    predictions across all cycles that forecast that target. Peaks
    highlight when the model disagrees with itself most. Hover shows
    mean prediction, CV%, and n cycles.
  - **Daily-total CV bar chart** (cumulative-source only): one bar
    per calendar day showing CV of predicted daily total across
    cycles. Colour-coded the same as the scorecards.
- **`get_forecast_stability()` on `HistoryDB`** — SQLite-native
  cross-cycle variance via the sum-of-squares identity (SQLite has no
  STDDEV): per-target-dt `sqrt(avg(p²) − avg(p)²)` clamped at zero to
  handle float rounding on constant columns. Filters to target_dts
  with ≥2 cycles.
- **`/experiment/{name}/forecast-stability` endpoint** — returns
  `per_timestep`, `daily_totals` (when `source_is_cumulative`),
  `summary`. Accepts `?days=N` (1-90, default 30).

## 2.11.6

### Feature

- **Cumulative view toggle on Forecast evolution chart** — for
  experiments with `source_is_cumulative: true`, a new "View as"
  dropdown lets the user switch between:
  - **Per-interval deltas** (default) — what the model actually
    predicts: increment per `interval_minutes` bucket.
  - **Cumulative (daily)** — reconstructed by summing the per-interval
    predictions within each calendar day, resetting at midnight when
    `reset_daily: true`. Makes it intuitive to read values as "% of
    the day's budget drawn by 6pm" rather than "% drawn in this one
    30-min bucket".
- **Y-axis label now responds to the view** — shows `<name>
  (cumulative <units>, resets daily)` in cumulative mode, `<name>
  (<units> per Nmin bucket)` for interval mode on cumulative sources,
  and `<name> (<units>)` for instantaneous sources.
- **Hover tooltip labels match the view** — says "cumulative" in
  cumulative mode, "predicted" / "measured" in interval mode.

## 2.11.5

### Improvement

- **Y-axis label now identifies the experiment and scale** — the
  previous label was just the raw `units` string (e.g. `%`), which
  doesn't tell the reader *of what*, and is actively misleading when
  the preprocessor has converted a cumulative-daily sensor into
  per-interval deltas. The label is now:
  - `<experiment_name> (<units> per <N>-min bucket)` when the target
    is cumulative (post-preprocessing scale is per-interval)
  - `<experiment_name> (<units>)` when the target is already an
    instantaneous reading
  - `<experiment_name> — <entity_id>` when no units are configured

  So a Mixergy cumulative-daily-% experiment now shows
  `mixergy_demand (% per 30-min bucket)` instead of just `%`.

## 2.11.4

### Improvement

- **Readable Forecast evolution chart** — rewrote the overlay so each
  cycle is visually distinct:
  - **Sequential HSL colour gradient** (indigo → teal → yellow-green)
    instead of same-cyan-with-opacity; adjacent issuances no longer
    blur together.
  - **Legend labels every cycle** with `DD Mon HH:MM`, grouped under
    "Forecast cycles (by issue time)", latest tagged "← latest" and
    ordered at the top. Moved the legend out to the right margin so it
    stops covering chart data.
  - **Informative y-axis** — uses the experiment's `units` when set;
    falls back to the target entity id so the user always knows which
    sensor the trace represents (was just "Value").
  - **Clearer x-axis** — "Target time (when the forecast applies)".
  - **Inline caption** under the heading explains that colour encodes
    recency and the white line is the measured value — no need to
    hover the info-tip to understand the chart.
  - **Richer hover** — bold "Forecast issued DD Mon HH:MM" header,
    explicit "predicted" / "measured" labels, actuals renamed
    "Measured (actual)" for contrast with forecast traces.

## 2.11.3

### Bugfix

- **Fix Samples=tiny in revision-improvement card** — the
  first-vs-latest comparison used `first_pred != last_pred` to filter
  out single-forecast targets, but this also excluded any target whose
  re-forecasts happened to produce numerically identical predictions,
  and the CASE-aggregation interaction dropped counts further. Replaced
  with an explicit `COUNT(*) >= 2` filter via a window function so the
  denominator now reflects actual re-forecast targets. Expect Samples
  to jump from single digits to thousands.
- **Fix lead-time chart collapsing to 3 points on fine grids** — the
  query hardcoded a 30-min lead-bucket even when `interval_minutes` was
  5 or 15, collapsing the chart to 2–3 data points and forcing MAE and
  RMSE to numerically converge within each chunky bucket. Bucket size
  now follows `interval_minutes` so a 5-min-grid experiment with a 1h
  horizon gets ~12 buckets instead of 3.

### Feature

- **Forecast evolution overlay chart** — new plot under the Accuracy
  by lead time chart. Renders each of the last N forecast cycles as a
  faded line (older = dimmer, latest = bright cyan) with the actual
  series overlaid in white. Lets you see how prediction curves
  converge toward the truth as lead time shrinks — same pattern as the
  holdout chart but for live production forecasts.
- **Cycle-count selector** — dropdown to choose 6 / 12 / 24 / 48 most
  recent cycles for the overlay.
- **`/experiment/{name}/forecast-evolution` endpoint** — returns the
  last N forecast snapshots plus grid-snapped actuals over the same
  window. Backed by `HistoryDB.get_forecast_evolution`.

## 2.11.2

### Bugfix

- **Cache-bust static assets on version bump** — `style.css` and
  `icon.png` are now requested with a `?v={{ app_version }}` query
  string, so browsers automatically fetch fresh copies after an addon
  update instead of serving stale cached files.
- **Constrain logo render at HTML parse time** — nav-bar `<img>` now
  carries `width="32" height="32"` HTML attributes so the browser
  knows the intended size before CSS loads, preventing a
  flash-of-giant-logo if `style.css` arrives late or from cache.
- **Belt-and-suspenders sizing on `.brand-logo`** — added
  `!important`, `max-width`, `max-height`, and `flex-shrink: 0` so the
  2rem constraint wins against any later rule or flex-growth.

## 2.11.1

### Branding

- **Add addon icon and logo** — `icon.png` and `logo.png` now live both at
  the repo root (for the HA add-on store listing) and inside the addon
  directory (for the supervisor tile / detail page).
- **Replace navbar emoji placeholder** — the 📈 emoji in the web UI nav
  bar is now the actual addon icon, served from `/static/icon.png`.
  Added `<link rel="icon">` so the browser tab favicon also picks it up.

## 2.11.0

### Feature

- **Per-experiment output activation** — neural models now apply a
  configurable activation (`auto` / `linear` / `softplus` / `relu` /
  `exp` / `sigmoid`) to the final Linear head, constraining predictions
  to the target's physical range **inside** the network rather than
  clipping post-hoc. `auto` resolves to softplus when
  `source_is_cumulative=true`, else linear. Settable per experiment from
  the Settings → Training dropdown.
- **Sigmoid scale auto-derived** — when `output_activation=sigmoid` is
  selected, the upper bound is derived from training data as
  `max(|y_train|) × 1.1` and persisted in the checkpoint as a registered
  buffer, so sigmoid can reach observed extrema without clipping.

### Removal

- **Drop post-hoc `np.clip(predictions, 0.0, None)`** across all 12
  neural backends — the activation layer now enforces the valid range
  directly, so the ad-hoc clamp is redundant. `expm1` drift guards
  inside `log_transform` branches are kept (they protect against
  floating-point inversion drift, not network output).
- **Drop target z-score normalization** (`self._y_mean` / `self._y_std`)
  — the activation operates on original-scale targets, so z-scoring the
  target would invalidate the activation's physical-range constraint.
  AdamW + internal LayerNorm converge fine without input/output
  symmetry. Removes ~260 lines of normalize/denormalize plumbing across
  backends.
- **Drop residual-prediction code path** — only fired for
  `n_horizons==1`, which never occurs in production (always
  multi-horizon). Dead code removed from all neural backends.

### Compatibility

- Checkpoints from v2.10.x with `y_mean`/`y_std`/`residual_prediction`
  keys will fail to load; affected experiments retrain automatically on
  startup.

## 2.9.8

### Bugfix

- **Fix flat multi-horizon forecasts in LSTM, iTransformer, and
  Crossformer** — same head bottleneck as the CNN: hidden layer narrower
  than `n_horizons` (48), forcing all horizon predictions to collapse to
  near-identical values. Hidden size is now `max(dim, n_horizons)` in
  all four backends.
- **Tighten N-BEATS tuning ranges** — worst-case combination
  (hidden_size=512, 8 stacks × 8 blocks × 8 FC layers) could allocate
  ~2.5 GB. Reduced maximums: hidden_size 512→256, stacks 8→4,
  blocks_per_stack 8→4, FC layers 8→6. Worst case now ~600 MB.

## 2.9.7

### Bugfix

- **Fix flat CNN multi-horizon forecasts** — the CNN's prediction head
  had a `n_filters // 2` hidden layer (16 neurons at default settings)
  producing 48 horizon outputs. This severe bottleneck forced the model
  to predict near-identical values for every future step, producing a
  flat line instead of a solar curve. The head hidden size is now
  `max(n_filters, n_horizons)` so each horizon can be predicted
  independently.

## 2.9.6

### Bugfix

- **Fix wrong tuning holdout predictions for neural models** — the
  holdout comparison trained CNNs/LSTMs on flat feature vectors instead
  of proper sliding-window sequences, producing garbage predictions.
  Now builds correct temporal windows for both training and test
  portions, matching the benchmark runner's pipeline.
- **Fix misleading tuned MAE in holdout chart** — the displayed MAE was
  taken from the tuning CV trial, not the holdout evaluation. The chart
  now shows the actual holdout MAE for both default and tuned models.

## 2.9.5

### Bugfix

- **Production sensors appear immediately after addon restart** —
  production experiments now retrain on startup instead of waiting for
  the next scheduled retrain (up to 24h away). After a restart or
  update, there were no cached models in memory, forecast cycles
  skipped, and sensors never got published until the retrain timer
  fired. Lab-mode experiments still defer to the normal schedule.

## 2.9.4

### Bugfix

- **Fix CNN tuning crash caused by exponential causal padding** — the
  WaveNet-style CNN uses `dilation = dilation_base^layer`. When Optuna
  suggested extreme combinations (e.g. `dilation_base=4, n_layers=10,
  kernel_size=15`), the last layer's causal padding was
  `14 × 4^9 = 3.6 million`, creating multi-GB tensors in a single
  `F.pad()` call that triggered an instant SIGKILL from the OOM killer
  with no Python traceback. Fixed by capping dilation at `seq_len` (48)
  in the CNN model — beyond that the kernel can only see one timestep
  anyway, so the cap has no accuracy cost.
- **Tightened CNN tuning search space** — reduced `n_filters` max
  256→128, `kernel_size` max 15→7, `n_layers` max 10→8,
  `dilation_base` max 4→3 to keep Optuna in architecturally sensible
  ranges for 48-step sequences.

## 2.9.3

### Bugfix

- **Fix OOM detection in Docker containers** — memory monitoring now reads
  cgroup v2/v1 limits instead of `/proc/meminfo`, which shows host RAM
  and is meaningless inside the addon's container. The pre-trial memory
  check, available-MB logging, and abort threshold now all reflect the
  container's actual memory budget. This should prevent the OOM crash
  during CNN tuning on RPi5.
- **Reduced tuning batch size** (32 → 16) to halve peak memory per trial.
- **Removed unsafe cleanup code** — `ctypes.CDLL("libc.so.6").malloc_trim`
  (glibc-specific, unsafe on Alpine/musl) and parameter tensor zeroing
  removed. Standard GC is sufficient.
- **Better memory diagnostics** — logs now show container usage/limit,
  process RSS, and per-trial memory after each trial to make memory
  leaks visible.

## 2.9.2

### Bugfix

- **Prevent OOM crashes during neural model tuning** — multiple layers
  of defence against the Linux OOM killer during Optuna tuning on
  constrained hardware (RPi5):
  - **Batch size halved** during tuning (64 → 32) to reduce peak
    activation memory per forward pass.
  - **Epochs/patience reduced** (40/8 → 30/6) to shorten each trial.
  - **Aggressive PyTorch cleanup** between trials — model parameters are
    zeroed, gradient buffers cleared, three GC generations collected,
    and `malloc_trim(0)` called to release freed pages back to the OS.
  - **Memory pressure monitor** — checks `/proc/meminfo` before each
    trial. If available RAM drops below 256 MB, tuning aborts gracefully
    with the best result so far (instead of SIGKILL with no traceback).
  - **Memory logging** — each trial now logs available system memory so
    leaks are visible in the addon log.

## 2.9.1

### Bugfix

- **Atomic YAML writes prevent config corruption on crash** — all config
  save operations now write to a temporary file and use `os.replace()` to
  atomically swap it into place. Previously, `open('w')` truncated the
  config file immediately, so an OOM SIGKILL during tuning (while a
  concurrent UI settings save was in-flight) could leave the YAML empty
  or half-written, causing all experiments to disappear on restart.
- **Periodic config reload no longer replaces config with stub on
  failure** — if the 30-second config reload encounters a parse error
  (e.g. briefly unreadable file), the existing good config is kept
  instead of falling back to a stub config with no experiments. The stub
  fallback is now only used on the very first load.

## 2.9.0

### Breaking

- **Remove ensemble functionality entirely** — the ensemble engine, all
  ensemble API routes, ensemble UI sections, and ensemble event handlers
  have been deleted from the codebase. The `ensemble/` module directory,
  `EnsembleEngine`, `EnsembleResultData`, and `EnsembleMethodResult`
  classes are gone. Pipeline steps no longer include an "ensemble" stage.
  The "Include Ensemble" checkbox has been removed from both the Training
  page and experiment detail page. This simplifies the architecture and
  removes code that was already disabled since v2.5.0.

### Bugfix

- **Trigger immediate retrain on production toggle** — when an experiment
  is switched to production mode, an immediate retrain is now triggered
  so the production model gets cached and sensors start publishing right
  away. Previously, sensors wouldn't appear until the next scheduled
  retrain cycle (potentially hours after toggling).

## 2.8.5

### Bugfix

- **Validate restored benchmark results against current model config** —
  on startup, persisted benchmark results are now filtered against each
  experiment's `models_enabled` list. Models that have been disabled since
  the last run are removed, the best model is recalculated from the
  remaining valid models, and fully stale results (where no saved models
  are still enabled) are discarded entirely. Fixes "Publish lightgbm"
  showing when only LSTM is configured.
- **Hide stale best model on dashboard during training** — the Best Model
  row is now hidden while a benchmark is actively running.

## 2.8.4

### Bugfix

- **Global training lock prevents all concurrent training** — a single
  `asyncio.Lock` now serialises benchmarks (web UI), scheduled retrains,
  and manual retrain triggers. Previously these three code paths had
  independent queues that could overlap (e.g. a scheduled retrain
  starting while a benchmark was running).

## 2.8.3

### Improvement

- **No automatic retrain on restart/update** — production experiments
  no longer force an immediate retrain when the add-on starts. The
  first retrain waits for the normal `retrain_every` schedule.
  Benchmark results are restored from SQLite, and forecasts gracefully
  skip until a cached model is available. Users can still trigger a
  manual retrain from the web UI at any time.

## 2.8.2

### Bugfix: scheduled retrains now queue sequentially

- **Production retrains no longer run in parallel** — the v2.8.0 queue
  only covered web UI "Run Pipeline" clicks. Scheduled retrains in the
  main loop used `create_task` per experiment, so all experiments due at
  the same time (especially on startup) would train simultaneously.
  Now uses an `asyncio.Queue` with a single consumer that drains one
  experiment at a time, preventing memory exhaustion on RPi.

## 2.8.1

### Bugfix

- **Fix reversed colours in covariate analysis** — percentage values in
  the drop-one table now show green for valuable covariates (removing
  hurts the model) and red for harmful ones (removing helps). Previously
  the colours were inverted, showing green next to Remove buttons.

## 2.8.0

### Feature: sequential training queue

- **Experiments now queue instead of running in parallel** — clicking
  Run Pipeline on multiple experiments queues them and runs one at a
  time, preventing memory exhaustion on constrained hardware (e.g. RPi).
- **Dashboard shows queue position** — queued experiments display an
  amber "Queued (#N)" button. Clicking it removes the experiment from
  the queue.
- **Stop Training handles queued experiments** — removes from queue
  if not yet started, or cancels the running task if in progress.

## 2.7.6

### Bugfix: Stop Training leaves UI stuck in loading state

- **Fixed wrong keyword argument in stop-training callback** — the
  `pipeline_end` event was constructed with `experiment=` instead of
  `experiment_name=`, causing a silent `TypeError`. The event was never
  stored in the training event history, so on page reload the JS
  replayed a `pipeline_start` with no matching `pipeline_end` and
  locked the Run Pipeline button in a permanent loading spinner.
- **Added server-side guard in replay logic** — the JS now checks the
  server's `is_running` flag before entering loading state. Even if the
  event history is stale, the UI won't show a stuck spinner when the
  server knows training has stopped.

## 2.7.5

### Feature: persist benchmark results across restarts

- **Benchmark results now survive add-on updates and restarts** — results
  are serialised to SQLite (`benchmark_results` table) as JSON after each
  benchmark run. On startup, stored results are restored into memory along
  with `best_model`, `selected_model`, and `last_benchmark_status`.
- Benchmark data is cleaned up when an experiment is deleted.

## 2.7.4

### Improvement

- **Graceful skip on insufficient covariate history** — when covariates
  don't have enough data, the pipeline now logs a warning and skips the
  cycle instead of failing with an error on the dashboard. The next
  scheduled cycle will retry automatically once sensors have accumulated
  enough history.

## 2.7.3

### Bugfixes

- **Fix covariate removal from Settings tab** — two conflicting
  `removeCovariate` JS function definitions caused the Covariate Analysis
  tab version (3 args) to overwrite the Settings tab version (1 arg).
  Clicking "×" on a covariate in Settings silently failed because the
  entity ID was misrouted as the experiment name. Renamed the Covariate
  Analysis version to `removeCovFromAnalysis` to eliminate the collision.
- **Guard against empty DataFrame after preprocessing** — if covariates
  have insufficient history (e.g. a freshly-created template sensor),
  the pipeline now raises a clear `ValueError` instead of crashing with
  an `IndexError` on an empty index.
- **Dashboard button colours** — Stop Training is now red (`btn-danger`),
  Publish/Publishing is now green (`btn-success`) with proper hover and
  disabled states.

## 2.7.1

### Feature: publish forecast accuracy as HA sensor

- **`sensor.mlfl_{name}_forecast_accuracy`** — publishes the lead-time
  accuracy curve as a HA sensor entity after each production forecast.
  State is the shortest-lead MAE; attributes contain `lead_hours`, `mae`,
  `rmse`, `sample_count`, and revision improvement metrics. Enables
  ApexCharts dashboard cards for accuracy visualisation.

## 2.7.0

### Feature: forecast evolution log & accuracy tracking

- **Forecast evolution log** — every production forecast is now logged to
  SQLite (`forecast_log` table) with the wall-clock issue time, each
  predicted target timestamp, lead time in minutes, model name, and whether
  it was a retrain or cached-model forecast.
- **Forecast Accuracy tab** — new tab on production experiment pages showing:
  - Lead-time vs MAE/RMSE chart (how accuracy degrades with longer horizons)
  - Revision improvement card (does re-forecasting via `forecast_every`
    actually improve accuracy?)
  - Total logged points and date range
- **Auto-slugify experiment names** — the Create Experiment modal now accepts
  human-readable names (e.g. "Optimised Solar") and auto-converts to valid
  slugs (`optimised_solar`). Removed the strict browser pattern validation.
- **Removed uppercase labels** — `.field-label` no longer forces uppercase
  text-transform across the UI.
- Forecast log is pruned alongside history cleanup and deleted when an
  experiment is removed.

## 2.6.3

### Fix: production forecast sensors never published

- **Fixed undefined `forecast_features` variable** in `_run_production_inference()`
  (line 1864). The variable was a leftover from the removed
  `create_forecast_features()` function and caused a `NameError` on every
  production forecast, silently preventing sensor publication. Now constructs
  `ds_future` from `last_ts` + interval offsets, matching the working
  `_forecast_with_cached()` implementation.

## 2.6.2

### UI: redesign Settings tab layout, move Stop Training to dashboard

- **Settings tab** — replaced flat field list with grouped card panels
  (Target, Data & Forecast, Training, Covariates). Toggles are now inline
  with labels, fields use an explicit 3-column grid, and each section has
  a bordered card background for visual separation.
- **Stop Training on dashboard** — the stop button now appears directly in
  the experiment card footer when training is running, replacing the
  Publish button. Removed from the Settings tab.
- **Auto-migrate config** — `load_config()` silently strips the deprecated
  `horizons_minutes` field from the YAML on first load, preventing the
  repeated "Ignoring unknown experiment fields" warning.

## 2.6.1

### Cleanup: remove dead `horizons_minutes` config & improve Settings layout

- **Removed `horizons_minutes`** — benchmarking evaluates the full forecast
  window (`future_periods`), not specific horizon checkpoints. The field,
  `create_forecast_features()` function, save-horizons API route, horizon
  chip UI, dashboard horizon gauges, and all related tests have been deleted.
- **Settings tab layout** — moved field grid styles from inline `<style>` to
  `style.css`, removed all-caps labels, increased spacing between sections
  and fields for better readability.

## 2.6.0

### Feature: full UI-driven experiment configuration

All experiment settings can now be managed through the web dashboard —
no more editing `mlfl.yaml` by hand.

**Settings tab** — a new first tab on every experiment page that
consolidates all per-experiment configuration in one place:

- **Target**: entity ID (read-only), cumulative source, daily reset,
  max increment
- **Data**: history days, interval minutes, log transform
- **Forecast**: future periods, per-experiment forecast and retrain
  intervals (nullable = use global default)
- **Training**: CV strategy, folds, recency half-life, production
  metric, neural loss function
- **Covariates**: full add/remove/edit management with a searchable
  Home Assistant entity picker (debounced, cached 60 s). Each covariate
  shows entity, role, aggregation, scale, and binary flag. No YAML
  editing required.
- **Stop Training**: a red button (with confirmation) that cancels a
  running retrain or tuning task. Cancellation takes effect after the
  current epoch completes for neural models.

**New Experiment creation** — a "+ New Experiment" button on the
dashboard opens a modal with name, target entity (with HA search),
cumulative source, and daily reset fields. Creates the experiment in
YAML and redirects to the new experiment's page. Also shown in the
empty-state when no experiments exist yet.

**Delete Experiment** — experiments can be removed via the API
(`POST /api/experiments/{name}/delete`), cleaning up both YAML and
in-memory state.

**System page simplified** — experiment cards on `/system` are now
read-only summaries with a "Configure →" link to the experiment's
Settings tab. Avoids two edit surfaces for the same fields.

### Backend changes

- `config.py`: added `add_experiment_covariate()`,
  `save_horizons()`, `create_experiment()`, `delete_experiment()`
- `app.py`: new routes for add-covariate, save-horizons,
  create/delete experiment, HA entity search (`/api/ha/entities`),
  stop-training; extended `experiment-settings` to accept
  `max_increment` (nullable)
- `main.py`: training tasks tracked in `_running_tasks` dict for
  cancellation support; stop callback emits `pipeline_end` SSE event
  so the Training tab transitions cleanly
- `style.css`: new component styles for horizon chips, entity search
  dropdown, covariate rows, and danger button

## 2.5.12

### Fix: neural model tuning OOM / hangs on RPi5

Optuna-based hyperparameter tuning for neural models (LSTM, CNN, etc.)
would frequently run out of memory or hang indefinitely on
memory-constrained devices like the Raspberry Pi 5 (8 GB). Three
compounding issues were identified and fixed:

**1. Memory accumulation across Optuna trials.**
Each trial instantiated a PyTorch model but never explicitly freed it.
Over 20+ trials the heap grew until the OOM killer sent SIGKILL,
crashing the container with no error in the logs. Every trial's
`objective()` now wraps execution in a `try/finally` that runs
`del model`, `torch.cuda.empty_cache()`, and `gc.collect()` — the same
pattern is applied to the baseline trial.

**2. Redundant sliding-window computation.**
`create_sliding_windows()` was called fresh on every trial for every
fold, rebuilding identical NumPy arrays each time. For a 90-day history
with `seq_len=168` this was ~3 seconds per trial × 30 trials = 90 s of
pure waste. Sliding windows are now pre-computed once before the study
begins and passed to `run_single_model()` via a new
`precomputed_sequences` parameter. The pre-computed arrays are freed in
a `finally` block after the study completes.

**3. No timeout for slow trials.**
A single bad hyperparameter combination (e.g. very large hidden size
with many layers) could train for hours without interruption. Neural
model studies now pass `timeout=1800` (30 minutes) to
`study.optimize()`, which gracefully stops the study and uses the best
result found so far. A log message is emitted when the timeout fires.

## 2.5.11

### Fix: production retrain failures are now surfaced in the UI

Three issues that made retrain/model-change failures hard to diagnose:

**1. FAILED status was never set on the production retrain path.**
`_retrain_single` caught exceptions and logged them, but never updated
`last_benchmark_status` to `"failed"`. The dashboard badge stayed at its
previous value (usually `"completed"`) even when the retrain crashed.
Both `_retrain_single` and `_forecast_single` now set `"failed"` and
store the error message on the `ExperimentStatus`.

**2. The error message was invisible in the UI.**
Added a `last_error` field to `ExperimentStatus` and an **Error** row
on the dashboard card that appears whenever the status is `"failed"`.
The next successful cycle clears it automatically. No more digging
through log files to find out what went wrong.

**3. Stale model cache when `production_model` changes.**
When you changed `production_model` in the YAML (e.g. `lstm` → `cnn`),
the old cached model stayed in memory until the next retrain replaced it.
Intermediate forecast cycles would use the stale cache (with the old
model's `feature_cols`, `seq_kwargs`, `exp_cfg`), which could produce
wrong predictions or crash with a shape mismatch. `_retrain_single` now
detects when the configured production model differs from the cached one
and invalidates the cache before starting the retrain.

## 2.5.10

### Fix: 1-hour timezone offset on the dashboard chart (DST-safe)

The forecast curve in `attributes.forecast` (and `_cumulative`,
`_daily_cumulative`, and `recent_actuals`) was being serialized with naive
ISO timestamps — e.g. `"2026-04-09T20:00:00"` with no `+00:00` suffix —
because the upstream pipeline strips timezones from the SQLite cache for
storage. JavaScript's `new Date(...)` interprets such strings as **local
time**, not UTC, so users in any timezone with an offset (BST = UTC+1,
EST = UTC−5, etc.) saw the MLFL series shifted on their charts. The
PredAI / Mixergy series came from HA's history API which always emits
tz-aware ISO strings, so they were plotted correctly — only the MLFL
series was offset.

`_publish_forecast_sensors` now localizes `ds_future` to UTC before
serializing, so all four sensors emit `"...+00:00"` strings that the
browser parses as absolute instants and renders in the local timezone.
The `recent_actuals` block in `_run_production_inference` does the same
for the historical context window.

This is fully **DST-safe** in both directions:

- The published timestamps are anchored to UTC, which has no DST
- The browser converts to local time using its own IANA tz database, so
  the chart automatically tracks BST → GMT (last Sunday of October) and
  GMT → BST (last Sunday of March) without any add-on changes
- The daily-cumulative day-bucketing already uses `zoneinfo` for the
  local-date calculation, which handles 23h/25h transition days

**Visible effects after upgrade**: from the next forecast cycle, the
MLFL series on your ApexCharts dashboard will line up exactly with the
PredAI and Mixergy actuals — no horizontal shift. Forecasts published
before upgrading will still show the offset until the next cycle runs.

## 2.5.9

### Fix: `state_class` on cumulative sensors

The `_cumulative` and `_daily_cumulative` sensors were being published
with `state_class: "total"` and `"total_increasing"` respectively. Both
were wrong: those state classes are for monotonic counters that HA's
long-term statistics engine processes as energy-meter-style totals,
which is not what these sensors are.

These sensors publish a **per-cycle snapshot** of a forecast projection:

- `_cumulative` state = predicted cumulative value at the end of the
  forecast horizon (changes each cycle as the model is re-run)
- `_daily_cumulative` state = predicted total demand for today at local
  midnight (fluctuates as the seed grows and remaining-forecast shrinks)

Neither is a monotonic counter. Both are now `state_class: "measurement"`,
which is HA's convention for values that go up and down freely.

**Visible effects after upgrade**:

- The entity history graph in HA's more-info dialog will plot the actual
  state values (a fairly flat line wobbling around the projected total),
  not an accumulating curve
- The sensors will no longer be suggested for the HA Energy dashboard
- The dashboard chart in your existing ApexCharts cards is unchanged
  (those use `attributes.forecast`, not the long-term statistics)

The existing distorted long-term statistics from before this fix will
fade as new correctly-tagged data comes in (~7 days). To clear them
immediately, go to Developer Tools → Statistics → Fix issues.

## 2.5.8

### Fix: `_daily_cumulative` state is now the end-of-today projection

The `sensor.{prefix}{name}_daily_cumulative` state was being set to the
last point in the forecast curve, which for a 48h horizon sits mid-way
through day-after-tomorrow **after** two intervening midnight resets.
That made the headline state something like 8.6% while the curve on a
chart reached ~55% at end of today, which was confusing.

The state is now set to the projected cumulative value at the **last
forecast point still within today's local date** — i.e., "what is the
total predicted demand for today by local midnight". This is directly
comparable to `sensor.<target>_today` at end of day.

The previous behaviour is preserved as two new sensor attributes:
- `end_of_today_value` — the new headline state
- `end_of_horizon_value` — the old headline state, for reference

The per-interval curve in the `forecast` attribute is unchanged — it
still resets at each local midnight throughout the whole horizon, so
dashboards that plot the full curve (like ApexCharts) continue to work.

## 2.5.7

### Fix: cumulative / daily-cumulative forecast sensors are now actually published

The `publish_interval`, `publish_cumulative` and `publish_daily_cumulative`
flags in each experiment's config were silently ignored by the
retrain-and-cache code path. Only `sensor.{prefix}{name}_forecast` (the
per-interval forecast) was being published, which made it impossible to
plot the forecast on the same scale as a daily-cumulative source sensor
without writing custom JS in your dashboard `data_generator`.

The new `_publish_forecast_sensors` helper consolidates all the publishing
logic and honours the flags. With `publish_daily_cumulative: true` and
`source_is_cumulative: true` (the typical Mixergy / energy-meter setup),
the helper now publishes:

- `sensor.{prefix}{name}_forecast` — main per-interval curve (always)
- `sensor.{prefix}{name}_interval` — same data, dedicated sensor
- `sensor.{prefix}{name}_cumulative` — running cumsum across the horizon
- `sensor.{prefix}{name}_daily_cumulative` — cumsum that **resets at local
  midnight** and is **seeded with the current value of the target sensor**
  so the forecast meets the actuals exactly at the join point

The daily-cumulative seeding uses the experiment's local timezone (from
the global `timezone` setting) and reads the live target sensor state, so
ApexCharts cards comparing the actuals against the mlfl forecast no longer
need any custom `data_generator` cumsum logic.

### Fix: stop labelling `%` forecasts as `power_factor`

`_run_production_inference` was setting `device_class: "power_factor"` on
the published forecast whenever `units == "%"`. Home Assistant's
`power_factor` device class is for AC efficiency ratios (0–1 dimensionless)
and overrides unit display, which is why the Mixergy demand forecast was
showing as a bare number with no `%` unit. The device_class is now omitted;
HA accepts `%` as a regular unit and displays it correctly.

### Refactor: deduplicated publishing path

`_run_production_inference` and `_forecast_with_cached` previously had
two near-identical inline publishing blocks. Both now route through
`_publish_forecast_sensors`, so future changes to the published attributes
or sensor naming only need to be made once.

## 2.5.6

### Removed: Hailo AI accelerator integration

The Hailo integration has been removed entirely. After investigation we
confirmed it could not possibly work on-device for custom-trained models:

- **Hailo's Data Flow Compiler (DFC) is x86-64 Linux only**. No ARM build
  exists and Hailo doesn't publish one. For custom-trained models this
  rules out any on-Pi compilation path, and Hailo's own documentation
  recommends ~32 GB RAM during quantisation — eliminating QEMU emulation
  on an 8 GB Pi 5 as a workaround.
- **The existing scaffolding was also broken in its own right**.
  `_retrain_and_cache` passed the ONNX file path to
  `HailoAcceleratedModel(model, hef_path=onnx_path)` where the class
  expected a compiled HEF file, not ONNX. The wrapper silently fell back
  to CPU inference while the validation check passed vacuously (both
  sides were CPU), so `hailo_active=True` was set in the dashboard while
  the NPU was idle. No forecasts were ever actually accelerated.
- **`compile_onnx_to_hef` in `hailo_runtime.py` was a placeholder**
  that logged instructions and returned `False` — it was never called
  from anywhere in the codebase.

Rather than ship a half-working integration that lied to the dashboard,
the entire Hailo code path is gone. Users with a Hailo hat should use it
with Frigate NVR, which works great because pre-compiled HEFs for
YOLO-family vision models are distributed with the add-on.

**What was removed**:
- `ml_forecast_lab/models/hailo_runtime.py` (entire file)
- `ml_forecast_lab/models/onnx_export.py` (entire file — only existed
  to support the `export_onnx` hooks in each backend)
- `export_onnx` and `supports_hardware_accel` methods from all 15
  model backends and from the `ForecastModel` abstract base class
- `hailo_enabled` config option from `AppConfig`
- `hailo_active` field from `ExperimentStatus`
- Hailo branch in `_retrain_and_cache`
- `is_hailo` / `hailo_accelerated` from cached model metadata and
  published sensor attributes
- Hailo checkbox from the System page + matching JS
- Hailo badge from dashboard experiment cards
- `python3-hailort` apt install + `--system-site-packages` venv tweak
  from the Dockerfile
- `/dev/hailo0` device mapping + `SYS_RAWIO` privileged cap from the
  add-on's `config.yaml`
- `onnx>=1.14.0` from `requirements.txt` (only used by `onnx_export.py`)
- Hailo section from `README.md` + `CONFIG_GUIDE.md` +
  `CREATION_REPORT.md`

**What you lose**: nothing functional. The Hailo code path never
actually ran on the NPU in practice, so removing it doesn't change
forecast correctness, training speed, or inference speed. The only
user-visible change is that the dashboard will stop lying about Hailo
being active and the `ONNX export failed: No module named 'onnxscript'`
warnings will disappear from the logs.

### Fix: NeuralProphet now works

The `neuralprophet` backend file was in the repo but the `neuralprophet`
pip package was missing from `requirements.txt`, so:

1. Docker image never installed the package
2. `from neuralprophet import NeuralProphet` raised `ImportError`
3. The `_optional_backends` loop in `main.py` silently swallowed the
   error and never registered the model
4. Enabling `neuralprophet` in `models_enabled` produced a `KeyError`
   from the registry

Fixed by adding `neuralprophet>=0.8.0,<1.0.0` to `requirements.txt`
(the first NeuralProphet release fully compatible with PyTorch 2.x and
PyTorch Lightning 2.x that the add-on already uses). Also added
`NeuralProphetModel` to the `_optional_imports` dict in
`models/__init__.py` for parity with the other backends.

Expected Docker image growth: ~50-100 MB of transitive deps
(PyTorch Lightning, matplotlib backends). Not a concern for Pi 5.

## 2.5.5

### Performance: neural-model tuning is now 10-20x faster

Neural hyperparameter tuning was catastrophically slow because of two
layered problems:

1. **`epochs` and `patience` were in the tuning search space itself.**
   Every neural `MODEL_PARAM_SCHEMA` entry listed `"epochs": {min: 10,
   max: 1000}`. Optuna would happily suggest `epochs=800` for some
   trials, so a single trial could train for up to ~10 minutes on
   RPi5 — and TPE tends to push toward high epoch counts early in the
   search because more epochs → lower validation loss (up to
   overfitting). This is a classic tuning anti-pattern: the training
   budget isn't a hyperparameter, it's a fixed resource decision.
2. **2-fold CV per trial doubled the cost** when 1 fold is enough to
   rank candidate hyperparameter sets. Optuna is robust to noisy
   objectives and a single well-sized fold is sufficient for the
   relative comparison.

Concrete numbers, Mixergy LSTM on RPi5:
* Before: ~30-60 minutes per tuning run (30 trials)
* After: ~3-6 minutes per tuning run

### Changes

* **Removed `epochs` and `patience` from all 13 neural
  `MODEL_PARAM_SCHEMA` entries** (lstm, cnn, dlinear, nbeats, nhits,
  tide, tsmixer, sparsetsf, patchtst, itransformer, crossformer,
  timesnet, neuralprophet). They're training budget decisions, not
  tuning targets.
* **Tuning now uses 1 CV fold** instead of 2 (`cv_folds = 1` in the
  runner config passed to `_run_tuning`).
* **Neural trials are capped at 40 epochs with patience 8** via a
  small `_apply_tuning_overrides(model)` helper that's called on both
  the baseline trial and every Optuna objective trial. This caps the
  per-trial budget without touching the model's own defaults — so the
  *production retrain* triggered by "Apply Tuned Params, Promote &
  Retrain" uses the full 100-epoch budget on the tuned model.
* **Log line** at tuning start shows the active budget: `Tuning
  budget: 30 trials × 1 CV fold × max 40 epochs (neural) /
  early-stopping (trees)`.
* **Tuning help tooltip** updated to explain the budget and the
  full-epoch production retrain.

Tree model tuning is unchanged (LightGBM/XGBoost already respect
their own early-stopping rounds and don't have the epochs-as-hparam
problem).

## 2.5.4

### Fix: Covariate Analysis now trains neural models correctly

Neural models in the Covariate Analysis path were being trained with
**flat features only** — no sliding windows, no dense horizons, no
residual prediction. That meant the LSTM / CNN / etc. metrics reported
in the covariate analysis grid came from a crippled version of those
models, not the production version, so the "does removing X help the
LSTM?" comparison was basically meaningless for any backend the
leaderboard considered competitive.

The inner training loop now mirrors the CV runner and holdout chart
for neural models:

* Builds sliding windows with `create_sliding_windows` using the full
  set of temporal and covariate channels
* Trains with dense `horizon_steps=[1..future_periods]` so residual
  prediction has something to optimise against
* Predicts on the test split using `horizon_steps=[1]` for one window
  per test row (full coverage)
* Reduces the multi-horizon output to h=1 for metric computation

Tree models keep the existing flat-features path unchanged. Both
families still use the same 80/20 split for consistency with each
other, so the cross-covariate comparison is fair.

### Rename: `deep_analysis` → `covariate_analysis` throughout the code

The UI has always called this feature "Covariate Analysis" but the
code used the old `deep_analysis` identifier from an earlier
iteration. Renamed all 60+ references:

* Python identifiers (`covariate_analysis_results`,
  `covariate_analysis_callback`, `_run_covariate_analysis`,
  `_covariate_analysis_trigger`)
* Pydantic classes (`CovariateAnalysisResult`,
  `CovariateAnalysisCellResult`)
* HTTP endpoints (`POST /run-covariate-analysis`,
  `GET /covariate-analysis`)
* Pipeline step name (`"covariate_analysis"` in the `run-pipeline`
  steps list)
* Template variable (`covariate_analysis`)
* HTML element IDs (`sec-covariate-analysis`,
  `covariate-analysis-model`, `covariate-analysis-btn`)
* JavaScript function (`runCovariateAnalysis`)
* Log strings, docstrings, and CSS comments

No behavioural change — this is purely a rename so the internal code
matches the user-facing name. The old endpoints no longer exist, so
any external integrations that hit them will need updating.

## 2.5.3

### Fix
- **Auto-ensemble was still running after every benchmark.** v2.5.0 hid the
  Ensemble tab from the navigation but the two `_run_ensemble()` calls
  inside `_benchmark_trigger` and `update_experiment` were still firing
  after every CV run, wasting compute and producing results no one could
  see. Both auto-trigger sites are now disabled. The ensemble code path,
  the `ensemble_callback`, and the section's HTML are all left in place
  so the feature can be re-enabled by uncommenting the nav link if needed.
- **`remove-covariate` couldn't actually find the covariate to remove.**
  Pre-existing bug: the deep-analysis "Remove" buttons send the short
  name (`current_charge`) but `remove_experiment_covariate` was matching
  against the full entity ID (`sensor.current_charge`), so the helper
  silently failed and the YAML was never edited. The helper now matches
  either form. Verified with five edge-case tests.

### New: clearer apply / publish workflow on Tuning + Covariate tabs

Both tabs now have a single prominent button that finalises the analysis
result and immediately starts a fresh retrain — no manual `mlfl.yaml`
editing, no waiting for the next scheduled retrain cycle.

**Tuning tab — "Apply Tuned Params, Promote & Retrain"** (was "Apply &
Promote"). The button now:
1. Saves tuned params to `mlfl.yaml` (existing behaviour)
2. Promotes the tuned model to production (existing behaviour)
3. **Triggers an immediate background retrain** (new) so the live
   forecast sensor picks up the new params right away
4. Switches the UI to the Training tab so the user can watch the live
   retrain progress, then reloads to refresh state
A short explanatory paragraph below the button spells out what each
click will do.

**Covariate Analysis tab — "Apply Best & Retrain"** (new green panel
above the results table). Reads the latest deep-analysis run, picks the
covariate configuration with the lowest average MAE across models, and:
* If "All covariates" wins → reports "already optimal", no changes
* If "No covariates" wins → clears the experiment's covariate list
* If "Without X" wins → removes covariate X
…then triggers an immediate background retrain. Toast shows which
covariates were dropped and the % MAE improvement vs baseline.

Backed by a new `POST /experiment/{name}/apply-covariate-best` endpoint
and a new `clear_experiment_covariates()` config helper. The existing
per-row "Remove" buttons stay as fine-grained controls.

### Plumbing
- New `retrain_callback` slot on `AppState`, registered by `main.py` at
  startup. Apply endpoints schedule retrains via this callback so they
  don't have to import `MLForecastLabApp` directly.

## 2.5.2

### Fix
- **Retrain cycle crashes with `unexpected keyword argument 'is_lab'`**.
  Two callers of `update_experiment()` (in `_retrain_single` and the
  `_run_retrain_cycle` loop) were passing `is_lab=True`, but the method's
  parameter is `is_lab_mode`. The mismatch broke every retrain cycle for
  lab-mode experiments. Fixed both call sites to use the correct keyword.

## 2.5.1

### Fix
- **Add-on crashes at startup with `ValueError: Unknown level: '5\n'`**.
  The HA add-on base image (hassio-addons/ubuntu-base, via bashio +
  s6-overlay) exports `LOG_LEVEL` as a *bashio* level, which can be a
  numeric string ("5" = NOTICE), a bashio name (`TRACE`/`NOTICE`/`FATAL`)
  that Python's `logging` module doesn't recognise, or sometimes a value
  with a stray trailing newline from `/var/run/s6/container_environment/`.
  Our `__main__.py` was passing the raw value straight to
  `root_logger.setLevel`, which crashed the entire add-on at startup.
  Replaced with a robust `_parse_log_level` helper that:
  * strips whitespace
  * maps bashio numeric levels (0–8) to Python equivalents
  * maps bashio string names (TRACE/NOTICE/FATAL/OFF) to Python equivalents
  * accepts standard Python names (DEBUG/INFO/WARNING/ERROR/CRITICAL)
  * falls back to `INFO` on any unrecognised input rather than crashing.

## 2.5.0

### New: Daily-cumulative leaderboard metrics

The Model Comparison section now has TWO tables:

- **Per-Interval Accuracy** (top, existing): MAE / RMSE / MASE / Mean Rank
  on the next-step (h=1) prediction. Drives Promote, Tuning, and live
  forecasting. Column headers relabelled to make the per-interval framing
  explicit.
- **Daily Cumulative Accuracy** (new, below): same metrics but computed on
  per-day totals — each day's predictions summed, each day's actuals summed,
  then compared. Includes a separate **Daily Rank** computed via the same
  Demšar (2006) composite as the primary rank but applied to daily metrics.

Daily Rank is informational only — it doesn't drive any production
workflow — but it lets you pick the model that matches your objective. For
use cases like daily energy or hot-water demand where the daily total is
what matters, this is now the leaderboard view to read.

The CV runner captures per-fold timestamps and groups predictions/actuals
by date inside each fold. The Demšar ranking helper has been factored out
and is called twice (interval + daily), so both rankings stay in sync with
any future ranking-logic changes.

### New: Tuning optimises a composite MAE+RMSE+MASE loss

Hyperparameter tuning previously asked Optuna to minimise MAE alone, then
picked the final winner via a post-hoc rank composite. Now Optuna's search
objective is the composite directly: each trial's score is the average of
`(mae / mae_default + rmse / rmse_default + mase / mase_default) / 3`,
where `_default` comes from one CV evaluation with the model's default
parameters run before the search starts. A composite of 1.0 matches the
default; 0.85 means a 15% average improvement across all three metrics.
Optuna's TPE/Random search now actually optimises the composite throughout,
not just at the very end.

The tuning panel UI now shows "Best composite" instead of "Best MAE", and
the help tooltip explains the composite scoring.

### Removed: Training Loss Curves panel

The collapsible "Training Loss Curves (neural models)" section in the
Results tab has been removed — the Generalisation tab already shows the
Train vs Test gap and Fold Stability lines, which are more useful for
diagnosing overfitting than per-epoch loss curves were.

### Disabled: Ensemble tab

The Ensemble tab is hidden in the navigation. The section content is left
in place behind a `{% if false %}` Jinja gate so it can be re-enabled by
uncommenting one line if needed in the future.

### Polish
- Generalisation tab now has a one-line note above the Train vs Test
  table clarifying "All values below are per-interval (h=1) errors" so
  there's no ambiguity about which metric scope you're reading.

## 2.4.0

### Critical fix: CV runner now matches the holdout chart and production paths

The CV runner was training a fundamentally different model from the one shown
on the holdout chart and used in production:

- **Sparse vs dense horizons.** CV used `horizons_minutes` from config to build
  `horizon_steps = [4, 16, 24, 48]` (4-output multi-head), while the holdout
  chart and production training both used dense `[1..future_periods]`
  (96-output dense). Same neural backend, completely different architecture.
- **Different window size.** CV used `max(48, max_horizon * 2) = 96`, the
  holdout chart used `min(48, len/3)`. Different receptive field.
- **Wrong "h=1" indexing in v2.3.1's fairness fix.** That fix used
  `horizon_steps[0]` as the ranking metric. With `[4, 16, 24, 48]` that's
  actually h=4 (2 hours ahead) — neural models were being scored on a
  fundamentally harder task than tree models, which produce h=1 (next-step)
  predictions. Hence the misleading 2.5× MASE gap on the leaderboard while
  the same models tracked actuals just fine on the holdout chart.

The CV runner now uses the **same** dense horizons and the **same** window
size as the holdout chart and production. With the per-fold test path also
switched to `horizon_steps=[1]` for inference, neural models get one window
per test row (full coverage) and the metric is computed on h=1 for both tree
and neural families. The leaderboard, the holdout chart, and the live
forecast sensor now all evaluate the same model.

### Holdout chart: full neural coverage without tail-fill

The v2.3.1 tail-fill (last window's h=2..96 outputs) was fragile because
residual prediction over long horizons collapses to "stay near the last
observed value" — that's why CNN/LSTM lines went flat for the final ~48h.
Replaced with `horizon_steps=[1]` inference, which gives one unique window
per holdout point. Each chart point is now a true 1-step-ahead prediction
from its own input window.

### Cleanup
- Dropped per-horizon sub-metrics (`mae_h2`, `rmse_h96`, `mae_havg`, etc.)
  from `fold_metrics`. They were never surfaced in the UI and were a
  consequence of the CV runner's old multi-horizon path.
- Simplified the runner's metric block — both tree and neural models now
  produce 1D `y_pred` / `y_test`, so a single `compute_all` call handles
  the full leaderboard.

## 2.3.2

### Improvements
- **Auto-assigned plot colours**: removed the hard-coded `MODEL_COLORS` /
  `ENSEMBLE_COLORS` dicts in `main.py` and the duplicated `MODEL_COLORS` map
  in `experiment.html`. All multi-trace charts now share a single 15-colour
  Plotly `colorway` defined once in the template, and traces consume colours
  in order. Adding a new model no longer requires touching a palette in two
  places, and the backend `ModelPrediction.color` field has been retired.
- **Help text refresh**: updated tooltips on the system page (forecast /
  retrain interval no longer mention a non-existent "global default", Hailo
  toggle now describes the CPU-vs-NPU validation test and graceful fallback,
  Settings section heading now covers the full set of global options) and the
  Model Comparison ranking explanation now states that all metrics are
  computed on the next-step (h=1) prediction so tree and neural models are
  compared on the same horizon.
- **README refresh**: configuration example now uses
  `forecast_every_minutes` / `retrain_every_hours` (the deprecated
  `update_every_minutes` was the v1.x name); features list now reflects all
  15 model backends, decoupled retrain/forecast cycles, hyperparameter tuning,
  ensembles, and covariate analysis; Hailo section mentions all neural
  families and the validation fallback.

### Fix
- **Config-loaded log spam**: the timer loop reloads `mlfl.yaml` every 30s
  to pick up UI edits and was logging `Configuration loaded from …` at INFO
  on every reload — drowning out training progress. Now uses an mtime check:
  the first load logs at INFO, real edits log `Configuration reloaded from
  …` at INFO, and unchanged reloads drop to DEBUG.

## 2.3.1

### Fix
- **Holdout chart now spans full time period for neural models**: with dense
  multi-horizon training (`horizon_steps=[1..96]`), `create_sliding_windows`
  produces only `len(holdout) - 95` valid windows, so the Predictions tab chart
  was missing the final ~48h of CNN/LSTM/etc. predictions while LightGBM
  covered the entire holdout. Neural models now reconstruct full coverage by
  using the LAST window's higher horizons (h=2..96) to fill the tail. Each
  holdout point gets exactly one prediction at the right horizon offset.
- **Fair leaderboard ranking between tree and neural models**: the un-suffixed
  ranking metric for neural models was the *average* across all 96 horizons,
  while tree models reported a single h=1 metric. Averaging long-horizon errors
  systematically penalised neural models — even when their h=1 predictions were
  competitive. The ranking metric now uses h=1 only for both model families,
  matching the chart and the natural tree-model evaluation point. The
  horizon-averaged variant is still recorded as `mae_havg`, `rmse_havg`, etc.
  for diagnostics.

## 2.0.3

### Improvement
- **Tuning best trial uses composite ranking**: the winning trial is now
  selected by average rank across MAE, RMSE, and MASE (same methodology as
  the Results table), not just lowest MAE. Optuna still minimises MAE to guide
  the search, but the final winner is the trial that performs consistently
  well across all three metrics.

## 2.0.2

### Fix
- **Covariate Analysis recommendations**: percentages now match the table
  (both use baseline as denominator). Wording clarified: "Consider removing X
  — dropping it reduces MAE by Y%" instead of confusing "X is harmful" language.
  Overall recommendation now says "performs better without covariates" when
  removing all covariates improves MAE.

## 2.0.1

### Fix
- Dashboard card layout: View Details button and publish toggle back in the
  card footer matching the original two-element layout.

## 2.0.0

### New: Decoupled Timers + Hailo AI Acceleration
- **Separate forecast and retrain schedules**: forecast cycle (default every 30m)
  uses a cached trained model for fast inference (<1s). Retrain cycle (default
  every 24h) trains from scratch and updates the cache. Dashboard shows both
  countdowns independently.
- **Hailo AI hat integration**: after each retrain, neural models are exported to
  ONNX and wrapped with `HailoAcceleratedModel` for NPU-accelerated inference.
  A validation test (CPU vs Hailo comparison) runs on every retrain — if it fails
  or diverges >1%, the system falls back to CPU with a logged warning.
- **Model caching**: trained models are cached in memory between retrains. Forecast
  cycles reuse the cached model, eliminating redundant retraining.
- **System page**: "Update Interval" renamed to "Forecast Interval" + new "Retrain
  Interval" field. Both configurable independently.
- **Backward compatible**: existing `update_every_minutes` in YAML is automatically
  mapped to `forecast_every_minutes`.

## 1.24.4

### Improvements
- **Dashboard publish toggle**: replaced large Publish/Stop Publishing buttons
  with a compact toggle switch matching the Models tab style.
- **Tuning holdout chart**: moved above the trials table (which is now collapsible).
  Added interval/cumulative toggle matching the Predictions tab style.

## 1.24.3

### Bug Fix
- **Neural production inference uses multi-head prediction**: replaced the
  autoregressive sliding-window loop with a single `predict_sequence()` call
  that outputs all horizons at once. This matches how the models are trained
  (direct multi-output) and avoids error accumulation between forecast steps.
  Intermediate points are linearly interpolated between horizon anchors.

## 1.24.2

### Improvements
- **Dashboard publish toggle**: replaced vague "Production" / "Lab Mode" button
  with clear "Publish" / "Stop Publishing" actions. When publishing, the card
  shows the HA sensor entity ID being published to.
- **Consistent language**: "Promote" renamed to "Publish" throughout — the
  experiment detail page header says "Publish lightgbm" and "Publishing lightgbm".
- **Mode toggle persists**: toggling publish on/off from the dashboard now saves
  to mlfl.yaml (previously only in-memory).
- **Publish guard**: the Publish button is disabled until a benchmark has been run,
  preventing publishing without a trained model.

## 1.24.1

### Improvements
- **Remove harmful covariates**: Covariate Analysis rows where dropping a covariate
  improves average MAE now show a "Remove" button. One click removes it from the
  experiment's YAML config.
- **Apply & Promote**: Tuning "Apply Best" replaced with "Apply & Promote" which
  saves tuned params AND promotes the model to production in one action.
- **Holdout comparison chart**: after tuning completes, a Plotly chart shows the
  model's default-params vs tuned-params predictions on holdout data with MAE
  improvement percentage.
- **Promotion persists**: promoting a model now saves `production_model` and `mode`
  to mlfl.yaml so the choice survives add-on restarts.

### Bug Fix
- **Pydantic validation error**: `ModelPrediction.predictions` field now accepts
  `Optional[float]` values, fixing ensemble padding errors in logs.

## 1.24.0

### New Feature: Selected Model
- **Model selection in Results**: radio buttons in the model comparison table let
  the user select any model, not just the top-ranked one. Selection persists across
  page reloads and feeds into Promote, Covariate Analysis, and Tuning.
- **Promote uses selected model**: the Promote button now promotes whichever model
  the user has selected, not just the auto-ranked best.
- **Selector defaults**: Covariate Analysis and Tuning model dropdowns default to
  the selected model instead of an arbitrary first option.
- **Per-experiment model_params**: tuning results saved per-experiment take
  precedence over global model_overrides during training.

## 1.23.1

### Improvements
- **Rename**: "Deep Analysis" renamed to "Covariate Analysis" throughout the UI.
- **Tab reorder**: Covariate Analysis now appears before Tuning in the tab bar,
  reflecting the natural workflow (analyse covariates → tune hyperparameters).
- **Per-experiment model params**: "Apply Best" in Tuning now saves hyperparameters
  per-experiment (not globally). Different experiments can have different tuned
  params for the same model. Per-experiment params take precedence over global
  model overrides.
- **Default vs tuned comparison**: the Tuning best-params summary now shows a
  side-by-side table of default → tuned values for each parameter.

## 1.23.0

### New Feature: Hyperparameter Tuning
- **Optuna-based Bayesian optimisation**: new Tuning tab in the experiment detail
  page with TPE (Tree-structured Parzen Estimator) and random search strategies.
- **Automatic search space**: parameter ranges derived from existing model schema
  with log-scale for learning rates and regularisation terms.
- **Fast 2-fold CV**: each trial uses 2-fold cross-validation for speed on
  constrained hardware (RPi5). Full benchmark validates the winner afterwards.
- **Live progress**: poll-based progress showing completed trials, best MAE so far,
  and a progress bar.
- **Trials table**: all trials sorted by MAE with params displayed, best highlighted.
- **Apply Best button**: one click saves winning hyperparameters as model overrides
  in mlfl.yaml, ready for the next pipeline run.
- **Per-model tuning**: select which model to tune from the experiment's enabled models.

## 1.22.9

### Fixes
- **Tab preserved on reload**: running ensemble or deep analysis no longer
  jumps to the Training tab on completion — the page reloads back to the
  correct tab via URL hash.
- **Deep analysis metric layout**: replaced cramped single-line layout with
  a clean three-row grid (MAE, RMSE, MASE) each with its own change %.

## 1.22.8

### Bug Fix
- Fix `NameError: name 'best_ind_metrics' is not defined` crash in ensemble
  pipeline introduced in v1.22.6.

## 1.22.7

### Deep Analysis Improvements
- **MASE metric added**: deep analysis now computes MASE alongside MAE and RMSE,
  with percentage change vs baseline for all three metrics.
- **All metrics displayed**: each cell shows MAE (primary) with RMSE and MASE on
  a second line, all with colour-coded change percentages.
- **Cross-model consensus**: per-covariate recommendations now report when all
  models (or a majority) agree a covariate is important or harmful, instead of
  using a single reference model.
- **Best-model reference**: per-covariate detail uses the best-performing model
  by baseline MAE, not a hardcoded tree model preference.

## 1.22.4

### Bug Fixes
- **Ensemble "Best" badge mismatch**: the improvement percentage text now
  consistently references the same strategy that receives the "Best" badge.
  Previously composite ranking and production metric comparison could disagree.
- **Ensemble prediction length**: ensemble traces now span the full holdout
  period by right-aligning to holdout timestamps and padding the start with
  gaps, instead of being silently truncated.

### Improvements
- **Fold stability**: replaced single-metric bar chart with three line charts
  (MAE, RMSE, MASE) showing per-fold variation for each model. Easier to spot
  instability and compare across metrics.
- **Loss curves moved**: neural model training loss curves relocated from the
  Generalisation tab into a collapsible section in the Results tab where they
  sit alongside model comparison data.
- **Feature importance note**: added guidance directing users to Deep Analysis
  for neural model feature insights.

## 1.22.3

### Improvement
- **High-contrast chart colours**: replaced the old green/blue palette with
  a high-contrast scheme (coral, amber, green, blue, purple...) so that 2-3
  model experiments are easy to read at a glance. Applied consistently to
  prediction charts, residuals, fold stability, loss curves, ensemble
  predictions, and live training fold traces.

## 1.22.2

### Fix
- **Remove early stop markers from live training charts**: the early stop
  detection and marker rendering on loss curves didn't work reliably. Early
  stop events are still logged to the training event log.

## 1.22.1

### UX Refinements
- **Per-experiment model selection**: model enable/disable toggles moved from the
  global Models page into the experiment detail page ("Models" tab). Each experiment
  now independently selects which models to train.
- **Models page simplified**: now a pure catalog with hyperparameter configuration.
  Toggle switches removed; subtitle directs users to experiment detail for selection.
- **Header decluttered**: ensemble/deep-analysis checkboxes removed from the header.
  Ensemble always included by default; deep analysis controls moved into the Deep
  Analysis tab where they have full context.

## 1.22.0

### Major UI Overhaul
- **Consolidated experiment workflow**: the dedicated Training tab has been removed.
  Pipeline controls, live training progress (loss curves, stats, event log), and
  all result sections now live on the experiment detail page ("View Details").
  Train, monitor, and evaluate an experiment without leaving the page.
- **Tab-based detail view**: the experiment detail page uses show/hide tabs instead
  of a long scrollable layout. Only the selected section is visible, reducing
  clutter. Charts render on first tab visit with automatic Plotly resize.
- **Section consolidation**: reduced from 8 sections to 6. Residuals merged into
  Predictions; Run Info moved to a compact header bar. Features and Deep Analysis
  tabs appear only when data exists.
- **Dashboard training status**: experiment cards now show the current model name
  and a progress bar when training is active. Dashboard refreshes every 10s during
  training instead of 60s.
- **Navigation simplified**: 4 tabs (Dashboard, Models, Logs, System) instead of 5.
  The old `/training` URL redirects to the Dashboard.

## 1.21.5

### Bug Fix
- **Training tab progress lost on tab switch**: the live training progress
  section now reliably restores when navigating away and back. The server
  embeds event history directly in the rendered HTML, eliminating a separate
  fetch that could fail silently through the HA ingress proxy. Also adds
  `pageshow` listener for bfcache resilience and `encodeURIComponent` for
  experiment names in fallback API calls.

## 1.21.4

### Bug Fix
- **Training tab loses live view on tab switch**: navigating away from the
  Training tab and returning now correctly restores the loss plot and live
  SSE stream. Root causes: `run-pipeline` was not calling `start_benchmark()`
  or `clear_history()`, so the UI could never detect an active run; the
  `hasEnd` check treated any historical `pipeline_end` as proof the current
  run had finished; and the Plotly chart rendered into a zero-dimension div
  before the browser had laid it out.

## 1.21.3

### Bug Fix
- **XGBoost 2.1+ compatibility**: moved `callbacks` parameter from
  `XGBRegressor.fit()` to the constructor, fixing "unexpected keyword
  argument 'callbacks'" error on XGBoost >= 2.1.

## 1.21.2

### Bug Fixes
- **Model hyperparameter auto-save**: hyperparameters in the Models tab now
  auto-save on change with a 600ms debounce, so edits persist without needing
  to click the Save button. The manual Save button remains as a fallback.
- **Training tab reconnection**: navigating away from the Training tab and
  returning now correctly restores the live training view. The backend
  pre-selects the currently-training experiment in the dropdown, and
  sessionStorage remembers the last selection for completed runs.

## 1.21.1

### Improvement
- **Neural model window size**: sliding window size is now derived from
  `max(48, 2 × max_horizon_steps)` instead of a fixed cap of 48. For
  horizons [2h, 8h, 12h, 24h] at 30-min intervals this gives 96 steps
  (48 hours of context), giving neural models twice as much lookback as
  their longest prediction horizon.

## 1.21.0

### Feature Engineering
- **Periodic lags**: `y_lag_48` and `y_lag_96` (same time yesterday and 2 days
  ago at 30-min intervals) give models direct access to daily periodicity.
- **Rate of change**: `y_diff_1` captures whether demand is accelerating or
  decelerating.
- **Interaction features**: `{covariate}_x_hour_sin` and `{covariate}_x_hour_cos`
  encode how covariates (e.g. charge level, temperature) interact with time of day.
- Production forecast features updated to compute interactions and periodic lags
  at inference time.

## 1.20.1

### Bug Fix
- **MASE in train metrics**: overfitting diagnostics were missing `y_train`
  when computing MASE, causing repeated "missing 1 required positional
  argument: 'y_train'" warnings during benchmarking.

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
