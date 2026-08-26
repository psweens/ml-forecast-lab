# Missingness — masking, indicators, and pipeline ordering (v2.51.0)

**Status:** Shipped in v2.51.0. This is the design rationale, defect
analysis, and verification record behind that release's CHANGELOG entry —
the depth that deliberately does not live in user-facing release notes.

Implements the design "Missingness: masking, indicators, and pipeline
ordering" (2026-08-23). The pinning tests are at
`ml-forecast-lab/tests/unit/test_missingness_masking.py` (113 tests) and
`tests/unit/test_backend_count.py` is unrelated but landed alongside.

Reproduce the guarantees locally:

```bash
cd ml-forecast-lab/
pytest tests/unit/test_missingness_masking.py -v   # ordering, label sanctity,
                                                   # causal imputation, pinning,
                                                   # window-frame semantics
pytest tests/unit/test_db.py -k NullValues -v      # accuracy NULL-pair guard
```

---

## TL;DR

| Defect (all live before v2.51.0) | Measured impact | Fix |
| --- | --- | --- |
| Rows dropped before features built — `target.shift(48)` meant "48 surviving rows back" | 10-day window, one 6 h gap: **38 of 413 rows** had a wrong `y_lag_48` | Features build on the complete grid; row selection happens once, downstream (`_supervised_frame` → `resolve_missingness`) |
| Sequence windows built over the punctured supervised frame | 20-day frame, one 6 h gap: **47 of 757 training windows** time-warped | Windows build over a dedicated window frame (complete grid, causally-imputed y, per-row label mask); windows with unmeasured horizon labels are dropped |
| Masked gaps reached models as `np.nan_to_num(..., 0.0)` — physically meaningful zeros | The v2.27.10 regression class | Role-based resolution (below); the `nan_to_num` calls remain as backstops that WARN with column names via `_nan_to_num_guarded` |
| Covariate alignment ended `.ffill().bfill()` — unbounded | 10-day covariate on a 2-year window: constant for **98.6 %** of training rows | Hold bounded by the entity's own cadence (p90 observation gap × 2, the same reach the manifest measures) |
| Accuracy queries had no NULL guard in raw mode | One NULL actual aborted the whole Forecast Accuracy prep (`round(None, 4)`) every cycle | NULL pairs excluded on both sides of the join, all four query sites, with Python backstops |

## The role rule

Everything reduces to one distinction:

* A missing **label** means the row cannot be a supervised sample. It is
  excluded from training, scoring, and conformal calibration, and **never
  imputed**. Imputing a label teaches the model something false and then
  scores it against that fabrication — and imputed values are smooth, so
  they are unusually easy to predict: every backend is flattered and the
  Demšar composite starts rewarding whichever model best reproduces the
  imputation scheme. Conformal bands would calibrate on fabricated
  residuals for the same reason.
* A missing **feature** — a covariate cell, or a lag that reaches into a
  target gap — is masked, causally imputed, and flagged, so its row
  survives. Imputing a feature adds noise the model can learn to
  discount, which is what the flag is for.

Windows inherit the rule in their own shape: window *contents* are
features (imputable, flagged via a per-row `y_missing` channel); the
horizon values a window is fitted or scored against are labels (windows
whose labels land in a gap are dropped, never filled).

## Design decisions and their reasons

**Causal imputation.** Expanding median over strictly prior observations
— leak-free by construction and fold-independent, because the CV harness
splits *after* preprocessing. Binary covariates hold the last prior
observation instead: the median of an even split is 0.5, a value that
appears in no observed row of an on/off channel. Leading gaps take the
first observed value — a single boundary scalar, a small real leak
accepted by the design (§4) and flagged across the whole region. Note
the honest consequence: a covariate whose history covers only the tail
of the window still trains on that one constant across the leading ~99 %,
bit-for-bit what the old back-fill produced; what v2.51.0 adds there is
the flag, not a different value. The structural fix for that case is the
covariate cache accumulating history (Spec A), not a cleverer fill.

**One aggregate `y_missing`, not per-column flags.** A single target gap
makes every lag, rolling statistic, and diff column gappy at once;
per-column companions would be ~24 near-constant columns whose *set*
changes every cycle as the history window slides — and an unstable column
list breaks a cached model outright. In the window frame the flag is
per-row ("the y at THIS row is invented") because a sequence model
consumes raw y, not lags; the two semantics are deliberate and tested.

**Indicator provenance is the cache, never the column name.** Which
columns have gaps is a property of the window, and the window slides
between a retrain and the next forecast. The trained indicator set is
stored with the model (`missing_indicators`) and pinned at inference.
Re-deriving it from the `_missing` suffix was tried and rejected by
adversarial review: an entity legitimately named
`binary_sensor.pump_missing` would have had its live channel zeroed on
every forecast.

**Inference flags answer the training question.** Training flags a cell
when no observation fell within the entity's cadence of that grid point.
A forecast row carries the flag value of the last measured row (0 when a
real future value arrives). Forcing 1 whenever no future source exists —
the first implementation — answers a different question, and measurably
moved forecasts: +1.7 mean bias (74 % of target sd) in the LightGBM
repro, because every horizon step sat in a regime the model saw on ~3 %
of training rows.

**Bounded covariate hold.** The reach is `max(interval, 2 × p90
observation gap)` — the same measurement the coverage manifest uses, read
off the same aligned series, so the diagnostic and the training frame
cannot disagree. p90 rather than median so a cache spanning a
`scan_interval` change is judged by its slower cadence; a quantile rather
than the max so outages (the thing being measured) don't set the reach.

**Warm-up is anchored at the first measured label.** Counting from grid
row 0 left the first supervised row after a leading target outage with
unfilled lag features, which the leading-gap rule then filled from the
first observed value — for `y_lag_k` that is a *later label*: the row
would be handed its own answer. Warm-up also shortens under the
clear-sky gate (`_gate_by_past_ghi` writes 0.0 over would-be warm-up NaN
in every lag column), which the arithmetic accounts for — without that,
every gap-free PV experiment would silently lose 24 rows.

## Verification

* **Gap-free bit-identity** (the release's compatibility promise):
  the real `_fetch_and_preprocess` of v2.50.0 (`f442852`) and v2.51.0 ran
  over identical stubbed recorder data; feature matrices, row counts, and
  sliding windows compared byte-equal at 5/15/30/60-minute intervals,
  with and without the clear-sky physics path, with and without
  covariates.
* **Gap scenarios:** every lag verified as a true time offset; every
  window span verified as `window_size × interval` of wall clock; every
  `seq_y` label verified equal to a measured y value.
* **Adversarial review:** a multi-agent review of the diff produced 10
  findings; 8 were fixed pre-merge (including the two above rejected
  designs), 2 accepted and documented: the leading-constant behaviour
  (spec-mandated, see above) and per-horizon loss masking (below).

## Deliberately deferred

* **Per-horizon loss masking.** Windows near a gap that have *some*
  measured horizon labels are currently dropped whole. Masking only the
  unmeasured horizons out of the loss would keep them, at the cost of a
  change to every neural backend's fit path. Dropping fabricates nothing;
  it merely trains on slightly fewer samples.
* **`<name>_age` staleness feature.** The flag is binary, not a clock: a
  tree backend cannot distinguish "stale by an hour" from "stale by a
  week" on a single row. If gappy-experiment benchmarks show models
  floundering on the binary flag, an age-in-minutes column (capped,
  pinned like the flags) is the designed next step.

## Cache migration

`schema_version` bumped 2 → 3 (v2.37 precedent): a model trained on
punctured windows would otherwise be served complete-grid inputs — a
distribution it never saw. Old caches are discarded on the first start
after upgrading and every experiment retrains once.
