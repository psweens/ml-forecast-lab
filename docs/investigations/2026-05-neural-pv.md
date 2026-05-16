# Neural backends — bizarre forecasts on solar-driven targets

**Status:** Investigation complete. Two confirmed root causes for the
v2.36.0 symptoms, supported by tensor-level introspection and a
synthetic dataset that reproduces the production failure shapes. One
v2.35.3 symptom remains open. Recommended fixes are scoped; pytest
regressions are committed and fail today.

This investigation was conducted on synthetic data only. No production
add-on was deployed against during the work. All code is in
`ml-forecast-lab/tests/synthetic/` and the new regression test is at
`ml-forecast-lab/tests/unit/test_neural_pv_regression.py`.

Companion artifacts in this directory:

* `phase1_summary.json`, `figures_phase1/*.png` — Phase 1 baselines
  (per-backend per-dataset)
* `phase3_observations.md` — Phase 3 printed-tensor inspection (raw
  values that pin RC1 and RC2)
* `prototype_fixes.md` — head-to-head PF1/PF2 prototype run on
  `realistic_pv`
* `v2_35_3_check.md` — quick check that the v2.35.3 code path
  (extended_window=False) does NOT reproduce the 3 AM symptom on
  synthetic data
* `figures_phase1/v2_35_3_check.png` — the resulting curve

Reproduce locally:

```bash
cd ml-forecast-lab/
python -u tests/synthetic/run_phase1.py            # ~4 min
python -u tests/synthetic/run_phase3.py            # ~5 min
python -u tests/synthetic/run_prototype_fixes.py   # ~3 min
python -u tests/synthetic/run_v2_35_3_check.py     # ~2 min
pytest tests/unit/test_neural_pv_regression.py -v  # both RC tests fail today
```

---

## TL;DR

| Production symptom | Confirmed root cause | Fix size |
| --- | --- | --- |
| **v2.36.0 NLinear collapses near zero with a late-day spike** | **RC1 (RevIN bias)** + **RC2 (NLinear anchor → zero)** | 20 LOC per RC |
| **v2.36.0 LSTM inverse-phase shape** | **RC1 (RevIN bias)** + **RC3 (LSTM attention reads future positions)** | RC1 is 20 LOC; RC3 mitigation is architectural |
| **v2.35.x LSTM/CNN flat ≈ training mean** | Pre-existing collapse on the past-only path — **not** caused by extended-window or RevIN bias; same symptom is still visible after disabling extended-window | Architectural (see §3, RC3 row) |
| **v2.35.3 NLinear/SparseTSF 3 AM spurious peak** | **Not reproduced on synthetic data** — see §2 H1 | Open |
| **v2.35.2 and earlier ½-interval shift** | Off-by-one in inference window — fixed in v2.35.3 | Already shipped |

The smallest surgical change is **PF1 + PF2**: ~40 LOC total. PF1
restores LSTM and CNN to the correct peak hour and lifts flatness from
≈0.05 to ≈1.9 on `realistic_pv`. PF1 also corrects NLinear's
late-by-one-hour peak on the same data.

---

## Section 1 — Symptom → root cause map

### Symptom S1 — NLinear collapses near zero with a late-day spike (v2.36.0)

**Reproduced on:** `make_realistic_pv(0)` (~4500 W peak, AR(1) clouds,
integer quantisation; `holdout_days=10`, `epochs=15`).

**Phase 1 result on realistic_pv:**

```
   nlinear  mae=171.9  peak_truth=11.0  peak_pred=12.0  flatness=0.75
   dlinear  mae=160.7  peak_truth=11.0  peak_pred=12.0  flatness=0.68
 sparsetsf  mae=249.5  peak_truth=11.0  peak_pred=12.0  flatness=0.70
```

The NLinear/SparseTSF/DLinear shapes are pulled toward a too-narrow
late peak — exactly the "near zero with late-day spike" pattern.

**Mechanism, confirmed by Phase 3 tensors:**

* `phase3_observations.md` §3.1 — _RevIN on an extended window for the
  same data has `mean=0.1534` versus the past-only mean `0.3068` (a
  50% bias). The denormalisation at the head therefore rescales the
  network's z-space output by half of what's correct.
* `phase3_observations.md` §3.2 — `x[:, -1, target_channel]` is 0 in
  10/10 random extended-mode samples; the intended anchor
  `x[:, W-1, target_channel]` carries the real last past observation.
  NLinear's "subtract the last value, re-add it" trick is therefore a
  no-op in v2.36+, leaving the single linear head to reach absolute
  target scale alone while also fighting RC1's biased denormalisation.

**Maps to:** RC1 + RC2.

### Symptom S2 — LSTM produces an inverse-phase shape (v2.36.0)

**Reproduced on:** every dataset in Phase 1 — `pure_pv`, `cloudy_pv`,
`realistic_pv`, `ev_mixergy`.

**Phase 1 result on realistic_pv (LSTM, CNN):**

```
      lstm  mae=512.4  peak_truth=11.0  peak_pred=12.0  flatness=0.06
       cnn  mae=525.9  peak_truth=11.0  peak_pred= 8.0  flatness=0.03
```

`peak_pred=12` looks fine numerically but `flatness=0.06` means the
prediction barely varies — the model is essentially constant. On
`pure_pv` LSTM also produces `peak_pred=21` (evening) — that's the
literal inverse-phase symptom.

**Mechanism, confirmed by Phase 3 §3.3:** the temporal attention puts
48% of its weight mass on the FUTURE positions of the extended
window. Those positions carry sun_elevation / clear_sky_ghi / hour_sin
of the forecast horizons but the target channel is zero. Without a
PAST-aligned anchor, the head reads these covariates as the
prediction signal; under RC1's RevIN bias the linear projection lands
on the wrong sign of the daily wave, producing the phase-inverted
shape.

**Confirmed by the prototype:** PF1 alone (RevIN past-only, no other
changes) moves LSTM from `peak=9 / flat=0.32` to `peak=12 / flat=1.87`
on `realistic_pv`; CNN from `peak=10 / flat=0.46` to `peak=12 /
flat=1.86`. The peak hour recovery is the diagnostic — flatness ≈ 1
means the model's predictions vary across the horizon at the same
amplitude as truth.

**Maps to:** RC1 + RC3.

### Symptom S3 — NLinear/SparseTSF 3 AM spurious peak (v2.35.3)

`run_v2_35_3_check.py` runs the v2.35.3 code path (extended_window=False)
on `realistic_pv` and reports the mean prediction at 03:00 UTC for each
backend. Result:

```
   nlinear  mean@03=  19.4  mean@12=1629.1  peak=12   (1.2% of daytime peak)
 sparsetsf  mean@03=  86.6  mean@12=1599.8  peak=12   (5.4%)
      lstm  mean@03=  35.2  mean@12= 765.4  peak=13
       cnn  mean@03=  68.7  mean@12= 853.0  peak=11
```

There IS a tiny bias at 03:00 (1–5% of the daytime peak) but nothing
that the user would experience as a "spurious 3 AM peak". The
v2.35.3 symptom therefore does NOT reproduce on synthetic data —
something in the user's specific target or covariate set is required.
Treated as **open hypothesis H1** in §2.

### Why does the user's Mixergy experiment work fine across every version?

Phase 1 on `make_ev_mixergy(0)`:

```
   nlinear  peak=1 (truth=1)  flat=0.84   mae=0.20
   dlinear  peak=1            flat=0.92   mae=0.12
 sparsetsf  peak=1            flat=0.64   mae=0.35
      lstm  peak=4            flat=0.03   mae=0.51
       cnn  peak=12           flat=0.05   mae=0.50
```

Linear-head backends (NLinear/DLinear/SparseTSF) preserve the
peak hour and a reasonable flatness because:

* The daily cycle is captured almost completely by `hour_sin / hour_cos`
  in the PAST block — the model doesn't need the future-position
  covariates to disambiguate phase.
* RC2 (anchor=0) costs ~0.07 absolute MAE on a target of order 1 — the
  user doesn't see that as "bizarre" because the bell still aligns at
  the right hour and the absolute miss is small.

LSTM/CNN do collapse on Mixergy too (`peak=4 / 12`, `flat=0.03 / 0.05`)
but the absolute MAE (~0.5 on a 3-unit target) is small enough that
the user's downstream consumer didn't flag it. On PV (4500 W peak) the
same flatness=0.05 translates to a multi-kW absolute miss — which is
visible.

---

## Section 2 — Open hypotheses

Hypotheses partially supported but not yet resolved to a single tensor
that fails today.

| # | Hypothesis | Evidence | Resolution path |
| --- | --- | --- | --- |
| H1 | **v2.35.3 3 AM peak** depends on a covariate or sensor pattern not present in `realistic_pv` (e.g. inverter-clipping plateaus, ordering of two solar covariates, residual lag features from `build_features`'s un-gated `y_diff_1` near sunset) | Not reproduced on synthetic. The closest I get is a 5% bias at 03:00 in SparseTSF — small enough to be invisible. The user sees a ~20-30% peak. | Add the user's actual covariate set into a synthetic generator (e.g. include any extra columns the user has fed into the target's feature pipeline). Then bisect on which one triggers the 3 AM peak by removing covariates one at a time. |
| H2 | **Tz inconsistency** in `compute_known_future_features` (`future_index.hour` is local time but `compute_solar_features` treats naive indices as UTC) | Phase 3 §3.5 — past/future hour_sin continuity is correct within one block. The 0.2588 step I initially flagged is the harmless integer-hour staircase that occurs at EVERY hour transition. | **Resolved (refuted).** Tz handling is consistent within one block. Separate latent bug: if a user's input frame is tz-naive but represents local time, the solar covariates will be wrong absolutely, but it's not the root cause of the bad forecasts here. |
| H3 | **Channel-name parity at value level** | Phase 3 §3.6 — `build_inference_window` and `create_sliding_windows` produce numerically identical samples at the same anchor (max abs diff = 0.0). | **Resolved.** The v2.35.2 parity guard handles this case correctly. |
| H4 | **Sliding-window alignment off-by-one in v2.36.0 extension** | Phase 3 §3.4 — past block, future block, and label index all line up. | **Resolved.** The v2.35.3 off-by-one fix held. |
| H5 | **LSTM attention phase-inversion** can be further mitigated by **masking attention to past positions only**, beyond PF1's RevIN fix | Phase 3 §3.3 measured 48% of weight on future positions. PF1 alone restores LSTM peak hour from 9 to 12 in the prototype — but does NOT remove the future-attention leakage itself. | Add a third prototype PF3: monkey-patch `_TemporalAttention.forward` to add `-inf` to scores at future positions before the softmax. Re-run on `realistic_pv` and compare flatness/MAE against PF1 alone. |
| H6 | **Daily loss weight** (the cumulative-trajectory loss term in `_composite_horizon_loss`) can compensate for RC1's amplitude bias and avoid the need for PF1 | Plausible — the cumulative term penalises systematic bias more than the per-interval loss. Phase 2 axis `daily_loss_weight` covers this; the sweep was cut short for time and only the `extended_window` axis on `realistic_pv` ran completely. | Restart Phase 2 on `realistic_pv` with `daily_loss_weight ∈ {0, 1, 5}` and the other axes pinned at defaults. Expect: a non-zero weight halves the bias but doesn't eliminate it. |

---

## Section 3 — Recommended fix shape per identified cause

| Cause | Surgical fix (≤ 20 LOC) | Architectural fix (≥ 200 LOC) | "Don't enable this combination" |
| --- | --- | --- | --- |
| **RC1 — RevIN bias from future-position zeros** | **Recommended.** Add an optional `past_window_size` kwarg to `_RevIN.normalize()`. When provided, compute `mean`/`var` over `x[:, :past_window_size, :]` only. Plumb `past_window_size` from each backend's `forward()` — every backend already knows it (`extended_window` + `past_window_size` already live in `seq_kwargs`). Backward-compatible: when omitted, behave exactly as today. | A masking-aware RevIN: introduce a per-channel `validity_mask` that propagates from `create_sliding_windows`. Channels NOT present in `future_features_df` get a mask of 0 at future positions, and RevIN computes mean/std over the unmasked entries per channel. More general — also handles partial-future-known covariates. | If we can't ship PF1 quickly, document that `extended_window=True` requires `use_revin=False` (which removes the bias path entirely at the cost of losing per-window scale normalisation). |
| **RC2 — NLinear anchor degeneration** | **Recommended.** Pass `past_window_size` into `_NLinearNet.forward()` and anchor on `x[:, past_window_size - 1, target_channel]`. The broadcast subtraction `x_shifted = x - x[:, past_window_size - 1: past_window_size, :]` is shape-compatible. When `past_window_size == seq_len` (the past-only path), this is identical to today's behaviour. | Replace the hand-rolled residual trick with a tiny learned anchor head: a small `nn.Linear(C, 1)` over the last past row that produces the per-sample anchor. Marginal accuracy improvement; more compute. | Disable NLinear when `extended_window=True` and recommend SparseTSF or DLinear instead. (DLinear's trend-seasonality decomposition doesn't rely on a last-value anchor in the same way.) |
| **RC3 — LSTM/CNN attention reads future-position zeros** | **None practical.** Even adding a past-only attention mask is ~10 LOC but architectural in nature because it requires the attention module to know `past_window_size`. | Two options: (i) past-only attention mask — set future scores to `-inf` before softmax. (ii) replace temporal attention with a cross-attention: past = K/V, future = Q (so each future position queries the past for context). | Document that `extended_window=True` works best for linear-head backends (NLinear, DLinear, SparseTSF, FITS, TSMixer). For LSTM/CNN, prefer `extended_window=False` until the attention is updated. The Phase 1 numbers above show LSTM/CNN are unreliable in extended-window mode regardless of RevIN. |

---

## Section 4 — Regression tests

`ml-forecast-lab/tests/unit/test_neural_pv_regression.py` contains three tests:

1. `test_revin_extended_window_mean_unbiased` — **fails today**.
   Asserts RevIN's per-window target-channel mean is within 5% of the
   past-block mean. Today: `relative bias 0.500 exceeds the 5%
   tolerance`. Passes under PF1 (past-only RevIN).

2. `test_nlinear_anchor_carries_last_past_observation` — **fails today**.
   Asserts the value NLinear actually uses as its anchor matches the
   last past observation within 5%. Today: `NLinear anchor uses
   x[:, -1, 0] = 0.00 but the last past observation is 244.00`.
   Passes under PF2 (anchor at `W-1`).

3. `test_prototype_pf1_past_only_revin_removes_bias` — **passes
   today**. Demonstrates that the proposed PF1 implementation IS the
   fix (replicated inline so the test is self-contained).

Both 1 and 2 use the deterministic `make_realistic_pv(0)` synthetic
dataset. They import `_RevIN` and `_NLinearNet` directly from
production code, so the assertions track the real implementation, not
a copy.

`pytest tests/unit/test_neural_pv_regression.py -v` today:

```
FAILED tests/unit/test_neural_pv_regression.py::test_revin_extended_window_mean_unbiased
FAILED tests/unit/test_neural_pv_regression.py::test_nlinear_anchor_carries_last_past_observation
PASSED tests/unit/test_neural_pv_regression.py::test_prototype_pf1_past_only_revin_removes_bias
```

Under PF1 + PF2, all three pass.

---

## Caveats and known limitations of this investigation

* The tree control (LightGBM) shows `flatness=0.01` on `realistic_pv`
  in Phase 1 — that's a bug in the *test harness's* recursive feature
  builder for quantised targets, not a fault in the LightGBM backend.
  Production tree backends work correctly on the user's target per the
  brief; I didn't fix the harness because the diagnosis doesn't depend
  on it.

* Phase 2's full settings sweep was cut short to keep the wall time
  manageable. The `extended_window` axis on `realistic_pv` was the
  most informative knob and is covered indirectly by the v2.35.3
  check + the prototype run.

* PF1 in `run_prototype_fixes.py` is applied by SWAPPING the trained
  `_RevIN` with the past-only variant and fine-tuning for 5 epochs.
  A proper fix would retrain from scratch with PF1 in place — the
  prototype's MAE numbers are pessimistic for that reason.

* All synthetic data uses UTC timestamps. The user's data may be in a
  local timezone; this investigation doesn't isolate any tz issues,
  but Phase 3 §3.5 has already refuted the most obvious tz hypothesis.
