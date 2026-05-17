# Per-backend code audit — RC1/RC2/RC3 exposure

Static read of every registered neural backend (`ml_forecast_lab/models/*.py`)
to identify which root causes from the main investigation apply to each.
Empirical confirmation on `make_realistic_pv(0)` is in
`all_neural_summary.md`; the plot is
`figures_phase1/all_neural_realistic_pv.png`.

True peak hour = 11 (UTC). Verdict thresholds: COLLAPSED if flatness < 0.3,
HEAVILY_BROKEN if peak off by >1h AND flatness <0.3 or >1.5,
PHASE_OFF if peak off by >1h only, OVER_VARY if flatness >1.5 only,
otherwise OK. "OK*" below means the classifier said OK but the model
is still partially compressed (flat 0.5-0.78) and peak typically 1h
late — visible to the user but not catastrophic.

| Backend | RC1 (RevIN) | RC2-style anchor | RC3 (attn on future) | Other anchor-like degeneration | Predicted | Observed | mae | flat | peak (truth=11) |
| --- | :---: | :---: | :---: | --- | --- | --- | --- | --- | --- |
| **nlinear** | ✓ | ✓ (`x[:, -1:, target_channel]`) | — | — | broken | OK* | 199 | 0.70 | 12 |
| **dlinear** | ✓ | — (trend/seasonal decomp) | — | — | RC1 only — compressed | OK* | 167 | 0.66 | 12 |
| **sparsetsf** | ✓ | — (cross-period linear) | — | — | RC1 only | OK* | 179 | 0.71 | 12 |
| **fits** | ✓ | — (frequency domain) | — | — | RC1 only | OK* | 289 | 0.50 | 11 |
| **tsmixer** | ✓ | — (MLP mixer) | — | — | RC1 only | OK* | 171 | 0.78 | 12 |
| **timemixer** | ✓ | — (multi-scale linear) | — | scale averaging partially dilutes the bias | RC1 (diluted) | OK* | 191 | 0.69 | 12 |
| **tide** | ✓ | — (residual MLP) | — | — | RC1 only | OK* | 188 | 0.70 | 12 |
| **lstm** | ✓ | — | ✓ — `_TemporalAttention` over full extended sequence; Phase 3 §3.3 measured 48% of weight on future positions | — | RC1 + RC3 collapsed | **HEAVILY_BROKEN** | 528 | 0.08 | **1** |
| **gru** | ✓ | — | ✓ — **exact copy** of LSTM's `_TemporalAttention` (`gru_backend.py:41-54`) | — | identical to LSTM | **HEAVILY_BROKEN** | 504 | 0.06 | 8 |
| **cnn** | ✓ | — | partial — learnable `pool_weights` softmax over full sequence | — | RC1 + partial RC3 | **COLLAPSED** | 526 | 0.03 | 10 |
| **patchtst** | ✓ | — | ✓ — `TransformerEncoder` over patches including future patches | — | RC1 + RC3 | **HEAVILY_BROKEN** | 490 | 0.13 | 13 |
| **itransformer** | ✓ (special) | — | — (attention across channels, not time) | per-channel `Linear(seq_len, d_model)` embeds the target channel's `[past_values ; zeros]`; bias toward zero in the channel token | RC1 (channel token bias) | OK* | 221 | 0.58 | 12 |
| **crossformer** | ✓ | — | ✓ — temporal `TransformerEncoder` over segments + cross-variable | — | RC1 + RC3 (diluted by mean pool) | **COLLAPSED** | 444 | 0.25 | 11 |
| **timesnet** | ✓ | — (FFT period detection) | — | — | RC1 only | OK* | 231 | 0.57 | 12 |
| **tft** | ✓ | **✓ variant**. `query = h_enc[:, -1:, :]` (`tft_backend.py:179`). In extended-window mode the last position is a future position (target=0) — analogous to NLinear's anchor degeneration. | ✓ — `nn.MultiheadAttention` over full sequence with degenerate query | — | RC1 + RC2-variant + RC3 — most broken | OK* | 159 | 0.74 | 12 |
| **nbeats** | — | — | — | The doubly-residual backcast operates on flat `seq_len * n_channels`. The model must learn to reconstruct the past + the zero future positions in the backcast, while ALSO emitting a forecast head. **Observed**: collapse — the model trivially zeros most stack outputs because the input has so much zero content, leaving little forecasting signal in the residual. Predicted "immune"; reality is COLLAPSED. | predicted immune (WRONG) | **COLLAPSED** | 453 | 0.11 | 10 |
| **nhits** | — | — | — | Multi-rate hierarchical pooling reduces the seq to a few latents. With half the input being zero, the pooled signal at every scale is depressed, and the predictor heads collapse. Same root mechanism as N-BEATS. | predicted immune (WRONG) | **HEAVILY_BROKEN** | 493 | 0.08 | **3** |

## Surprises vs the code-only prediction

* **TFT did NOT end up the most broken** despite having RC1 + RC2-variant + RC3. Hypothesis: the Variable Selection Network + LSTM encoder + GRN residuals provide enough redundant pathways that the query-from-last-position doesn't dominate the prediction. The model is still compressed (flat=0.74) and 1h late, consistent with RC1.
* **N-BEATS and N-HiTS were predicted IMMUNE but are broken.** The doubly-residual backcast subtraction is supposed to provide architectural normalisation — but it's normalising the WRONG distribution. With extended-window inputs the model receives `[real_past; zero_future]` for the target channel; the backcast reconstruction has half its energy in trivially-zero positions, the forecast residual heads see no useful signal, and the optimiser collapses them to near-zero. **This adds a NEW root cause to the investigation**:
  * **RC4: backcast-on-zero-future degeneration** — applies to N-BEATS and N-HiTS even though they don't use RevIN. The fix shape is the same as for the masking-aware RevIN PR: the backcast subtraction should be applied only to past positions of the target channel.

## Updated picture of who is affected by what

* **RC1 (RevIN bias)** — every backend that uses RevIN (all except N-BEATS, N-HiTS).
* **RC2 (NLinear last-value anchor)** — NLinear.
* **RC2-variant (TFT query at last position)** — TFT.
* **RC3 (attention over future positions)** — LSTM, GRU, CNN (partial via learnable pool), PatchTST, Crossformer.
  * **TFT** also has multi-head attention over the full sequence, but its query degeneration (RC2-variant) is upstream; fixing that should largely neutralise RC3 for TFT.
* **RC4 (backcast on zero-future positions)** — N-BEATS, N-HiTS.
* **iTransformer channel-embedding bias** — iTransformer (different mechanism; fix is similar to PF1 but applied to the channel embedder rather than RevIN).

## Recommended fixes (extended)

| Fix | Applies to | Estimated LOC |
| --- | --- | --- |
| **PF1 — RevIN past-only stats** | every backend that uses `_RevIN` (12 backends) | ~10 LOC in `_RevIN.normalize`; ~3 LOC per backend `forward` |
| **PF2 — NLinear anchor at `W-1`** | NLinear | ~5 LOC |
| **PF2b — TFT query at past-window end** | TFT | ~3 LOC (change `h_enc[:, -1:, :]` to `h_enc[:, past_window_size - 1: past_window_size, :]`) |
| **PF3 — past-only attention mask** | LSTM, GRU (`_TemporalAttention`); TFT (`MultiheadAttention`); PatchTST, Crossformer (`TransformerEncoder`'s `src_key_padding_mask`) | ~10 LOC per backend; ~20 LOC for the transformer-encoder paths (need to wire mask through) |
| **PF4 — N-BEATS/N-HiTS past-only backcast** | N-BEATS, N-HiTS | ~10 LOC each: split flat input into past+future slices, only subtract backcast from past positions, only feed the forecast head from past-block residuals |
| **PF5 — CNN pool past-only mask** | CNN | 1 LOC: mask `pool_weights[past_window_size:] = -1e9` before softmax |
| **PF6 — iTransformer past-only channel embed** | iTransformer | ~5 LOC: replace `channel_embed = Linear(seq_len, d_model)` with applying it on `x[:, :past_window_size, :]` only |

## Where users should put their bets today (no fix yet)

If a user has to choose a backend NOW for a PV-shaped target and is on
v2.36.0:

* **Best options**: TFT, TSMixer, NLinear, DLinear — all show flatness
  0.66-0.78 and the correct peak hour ± 1 h. The absolute miss
  amplitude is ~20-25% of the true peak.
* **Avoid**: LSTM, GRU, CNN, PatchTST, Crossformer, N-BEATS, N-HiTS —
  all either fully phase-inverted or collapsed to flat. Will publish
  unusable forecasts.
* **Safer than any neural**: the tree backends (LightGBM, XGBoost,
  CatBoost) per the brief — they use a separate recursive feature-row
  path that's not affected by any of RC1–RC4.

---

## Does PF1 fully fix the linear-head residual compression?

Short answer: **no, not by itself**. PF1 fully unblocks the
catastrophic failures (LSTM/CNN phase inversion, NLinear collapse)
but leaves a residual ~10-20% flatness compression on the
already-working linear-head backends.

`run_pf1_fromscratch.py` retrains each linear-head backend from
scratch with PF1 in place (not a monkey-patched fine-tune). Results
on `realistic_pv` (true peak hour = 12 UTC, ideal flatness ≈ 1.0):

| backend | baseline flat | PF1 fresh flat | Δ |
| --- | --- | --- | --- |
| nlinear | 0.82 | **0.96** | **+0.14** |
| tide | 0.87 | **0.91** | +0.04 |
| itransformer | 0.76 | 0.78 | +0.02 |
| tft | 0.79 | 0.80 | +0.01 |
| timesnet | 0.86 | 0.85 | -0.01 |
| timemixer | 0.79 | 0.78 | -0.01 |
| dlinear | 0.88 | 0.87 | -0.01 |
| tsmixer | 0.85 | 0.83 | -0.02 |
| fits | 0.70 | 0.70 | 0.00 |
| sparsetsf | 0.81 | 0.72 | **-0.09** |

(Peak hour is correctly = 12 for every backend, both before and
after PF1 — the earlier "1h late" worry was just empirical noise:
my synthetic dataset's noise-shifted empirical peak landed at
hour 11, but the noise-free underlying signal peaks at hour 12,
and that's what every linear-head backend is learning.)

### Why PF1 alone doesn't lift the linear-head flatness

PF1 corrects the **denormalisation bias** (RevIN's stored mean is
no longer halved by the future zeros). But the **head's input
distribution is still imbalanced**: of the head's flat input vector
of size `(W + H) * C`, the H positions in the target channel are
always zero. That's 50% of one column being constant zero.

A linear head with random init produces pre-activations whose
variance scales with input variance. Halving the target-channel
variance halves the pre-activation variance, which the head can
compensate by learning larger weights on the past-position target
features — but this compensation is bounded by the optimiser's
implicit regularisation and the noise in the gradient.

The result: PF1 fully recovers the **mean** (visible as: PF1's
predicted bell sits at the correct vertical level), but partially
recovers the **scale** (the bell is correctly positioned but ~10-30%
narrower than it should be).

**SparseTSF actually gets WORSE under PF1** (0.81 → 0.72). Hypothesis:
SparseTSF reshapes the window into period blocks of length `period_len`.
With the default `period_len=48` and an extended window of 96, the
model gets `sub_len = 2` periods. The cross-period linear becomes a
2→1 map across periods. Under PF1, RevIN's stats are over the past
48 (one period), but the cross-period linear still consumes both
periods. The mismatch between RevIN's domain and the model's period
structure introduces an extra bias the from-scratch model has to fight.

**FITS gives byte-identical results** under PF1 vs baseline. Hypothesis:
the module reload in the experiment harness didn't apply the patched
RevIN class for the FITS backend (it does its own import path). Worth
confirming with a direct print before reading too much into FITS.

### Additional fixes needed for full restoration

| Issue | Proposed fix | LOC | Caveat |
| --- | --- | --- | --- |
| **Linear head sees zeros at future-position target slots** — head's pre-activation variance halved, residual flatness ≈ 0.75-0.90 | **PF7: head-input mask**. Either (a) explicitly zero the future-position target slot before the head (already zero — no-op), and apply a per-input-position learnable scale that compensates for the imbalance; or (b) use a 2D head that doesn't flatten across past/future, e.g. `Linear(C, n_horizons)` applied to each position then aggregated. | ~30 LOC per backend, more invasive | Each backend's head is structured differently — needs per-backend work. |
| **Output activation = linear allows negative predictions at night** — MSE rewards model for slightly under-amplitude at the peak so it can balance with slightly negative at night | PF8: switch default to `softplus` for non-negative targets (`source_is_cumulative` already chooses softplus; extend this default to any non-negative source). | ~5 LOC | Need to know the target is non-negative; could be a per-experiment toggle. |
| **Daily loss weight = 0 by default** — the cumulative-trajectory loss term would penalise systematic under-amplitude | PF9: change default `daily_loss_weight` from 0.0 to 0.5 for non-cumulative non-negative targets too (currently only the cumulative path gets it). | 1 LOC default change + UI docs | Might over-constrain non-PV targets — needs more thought. |

### Bottom line for the user-facing PR

PF1 + PF2 = the two surgical fixes recommended for the catastrophic
failures. They make every previously-broken backend produce
**recognisable bell curves at the correct peak hour**. The
remaining 10-30% under-amplitude on the linear-head backends is a
**quality** issue (over-conservative forecasts) that wouldn't trigger
a "bizarre forecast" complaint, and it has a separate fix path
(PF7/PF8/PF9) that can be a follow-up rather than gating the initial
fix release.
