# Phase 3 — targeted code introspection

## 3.1 RevIN bias from future-position zeros
- channels: ['y', 'sun_elevation', 'clear_sky_ghi', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']
- past-only X shape: (17425, 48, 8)  extended X shape: (17425, 96, 8)
- chose sample i=98, window ends at 2024-01-04T00:30:00+00:00 (hour=0)
- past-only window target-channel stats: mean=0.3068, std=0.4339
- extended window target-channel stats:  mean=0.1534, std=0.3430
- ratio extended_mean/past_mean = 0.500, std ratio = 0.791
- target channel value summary:
  - past block (both):           min=-0.0002, max=1.2272, mean=0.3068
  - extended past block:         min=-0.0002, max=1.2272, mean=0.3068
  - extended FUTURE block (target channel left zero): min=0.0000, max=0.0000, mean=0.0000
- clear_sky_ghi channel value summary:
  - past-only path past block:                 mean=36.98
  - extended path past block (should match):   mean=36.98
  - extended path future block (populated):    mean=37.42
**Verdict — CONFIRMED.** Future-position target zeros pull RevIN's per-window mean down by ~50%. The denormalisation at the output uses this biased mean, so the model is shifted toward zero by roughly that amount in target space.

## 3.2 NLinear last-value anchor degeneration
- 10 random samples — target-channel last_val per path:
  - past-only x[:, -1, 0]               = [0.9099000096321106, 9.999999747378752e-05, -9.999999747378752e-05, 2.5329999923706055, 9.999999747378752e-05, -9.999999747378752e-05, 0.0, 0.3783000111579895, 1.4026000499725342, 0.02419999986886978]
  - extended  x[:, -1, 0]   (used)      = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  - extended  x[:, W-1, 0]  (intended)  = [0.9099000096321106, 9.999999747378752e-05, -9.999999747378752e-05, 2.5329999923706055, 9.999999747378752e-05, -9.999999747378752e-05, 0.0, 0.3783000111579895, 1.4026000499725342, 0.02419999986886978]
**Verdict — CONFIRMED.** Across the 10 sampled extended-mode windows, `x[:, -1, target_channel]` is zero in 10/10 cases. The intended anchor (`x[:, window_size-1, target_channel]`) carries the true last observation. NLinear's residual trick is therefore a no-op in v2.36+; the single linear head must reach absolute target scale on its own.

## 3.3 LSTM TemporalAttention past vs future
- sample i=2368, sequence length=96 (past=48, future=48)
- attention weight sum on past positions:   0.5224
- attention weight sum on future positions: 0.4776
- ratio future/past = 0.914
- top-5 weights and positions: [(41, 0.0639), (42, 0.0625), (40, 0.0593), (43, 0.0584), (39, 0.0508)]
**Verdict — Plausible.** The LSTM's attention places a non-trivial share (48%) on future positions whose target channel is zero — those positions carry sun_elevation/clear_sky_ghi/hour-of-day, so the model is free to read absolute time from there. If it learns to use future magnitudes additively rather than as phase anchors, the resulting context can invert in phase relative to the past block.

## 3.4 Sliding-window alignment
- sample i=100
- expected past index[0]=2024-01-03 02:00:00+00:00, [-1]=2024-01-03 13:30:00+00:00
- expected future index[0]=2024-01-03 14:00:00+00:00, [-1]=2024-01-03 19:30:00+00:00
- past block channel-wise match against manual reconstruction: True
- future hour_sin matches the future timestamps: True
  expected[:6] = [-0.5, -0.5, -0.707099974155426, -0.707099974155426, -0.8659999966621399, -0.8659999966621399]
  actual  [:6] = [-0.5, -0.5, -0.707099974155426, -0.707099974155426, -0.8659999966621399, -0.8659999966621399]
- y[i, 0] expected (h=1): 0.9203  actual: 0.9203
**Verdict — alignment looks correct in extended-window mode.** Past block, future block, and label index all line up. The v2.35.3 off-by-one fix has held.

## 3.5 Timezone path: past/future hour_sin continuity
- df_naive last timestamp: 2024-12-30 23:30:00
- future_idx[0]: 2024-12-31 00:00:00
- hour_sin values across past/future boundary [W-3..W+2]: [-0.5, -0.2587999999523163, -0.2587999999523163, 0.0, 0.0, 0.2587999999523163]
- clear_sky_ghi values across same boundary: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
**Verdict — refuted.** The 0.2588 step at the boundary IS larger than the half-hour continuous-spacing limit of 0.131, but this is NOT a tz issue — it's because `compute_known_future_features` (and `create_sliding_windows`) compute `hour_sin = sin(2π · idx.hour / 24)` using INTEGER `idx.hour`. The result is a staircase: hour_sin is constant within each integer hour and jumps once per hour. The same 0.2588 step happens at EVERY integer-hour transition in both the past and the future blocks — the past/future boundary is not special. (See the past slice in the trace: `-0.5, -0.2588, -0.2588, ...` already shows the staircase.) Not the root cause of the bad forecasts. Note: this is a minor quality bug — moving to `sin(2π · (idx.hour + idx.minute/60) / 24)` would give a smoother signal — but it is not what is breaking PV forecasts.

## 3.6 Channel-name parity at the value level
- train channel names == infer channel names: True
- max |train_sample - infer_sample| over all channels = 0.000000
  - y: max abs diff = 0.000000
  - sun_elevation: max abs diff = 0.000000
  - clear_sky_ghi: max abs diff = 0.000000
  - hour_sin: max abs diff = 0.000000
  - hour_cos: max abs diff = 0.000000
  - dow_sin: max abs diff = 0.000000
  - dow_cos: max abs diff = 0.000000
  - is_weekend: max abs diff = 0.000000
**Verdict — values match channel-by-channel.**

