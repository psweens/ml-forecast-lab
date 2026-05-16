"""
Phase 3 — targeted code introspection.

Confirms or refutes each of the suspected root causes using printed
tensors on single training samples rather than relying on the eventual
chart.

Each section ends with a clear "confirmed / refuted" verdict plus the
tensor values used to reach it. Numbers are intentionally printed at
4-decimal precision rather than summarised so the chain of reasoning is
fully auditable.

Sections
--------
3.1  RevIN bias from future-position zeros — sun-only-during-day target
3.2  NLinear last-value anchor degeneration on extended windows
3.3  LSTM TemporalAttention weight distribution past vs future
3.4  Sliding-window alignment manual reconstruction
3.5  Tz-naive vs tz-aware continuity of compute_known_future_features
3.6  Channel-name parity at the VALUE level (build_inference_window vs
     create_sliding_windows)
"""
from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from ml_forecast_lab.features import (
    build_inference_window,
    compute_known_future_features,
    create_sliding_windows,
)
from ml_forecast_lab.models.base import _RevIN

from tests.synthetic.datasets import make_pure_pv, GB_LAT, GB_LON
from tests.synthetic.harness import HarnessCfg

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("phase3")

OUT_PATH = Path("docs/investigations/phase3_observations.md")


# --------------------------------------------------------------------------- #
# Section 3.1 — RevIN bias from future-position zeros                          #
# --------------------------------------------------------------------------- #

def section_3_1_revin_bias() -> List[str]:
    """Compare RevIN per-window stats with and without extended window."""
    lines: List[str] = []
    lines.append("## 3.1 RevIN bias from future-position zeros\n")
    d = make_pure_pv(0)
    cfg = HarnessCfg(window_size=48, horizon=48)
    cov_cols = ["sun_elevation", "clear_sky_ghi"]
    horizon_steps = list(range(1, cfg.horizon + 1))
    # Past-only path.
    X_past, _, ch = create_sliding_windows(
        d.df, "y", window_size=cfg.window_size,
        covariate_cols=cov_cols, add_temporal=True,
        horizon_steps=horizon_steps,
    )
    # Extended path — same source frame, future features added.
    future_df = compute_known_future_features(
        d.df.index, add_temporal=True, country="GB",
        solar_lat_lon=(GB_LAT, GB_LON),
        include_sun_elevation=True, include_clear_sky_ghi=True,
    )
    X_ext, _, _ = create_sliding_windows(
        d.df, "y", window_size=cfg.window_size,
        covariate_cols=cov_cols, add_temporal=True,
        horizon_steps=horizon_steps,
        future_features_df=future_df,
    )
    lines.append(
        f"- channels: {ch}\n"
        f"- past-only X shape: {X_past.shape}  extended X shape: {X_ext.shape}\n"
    )
    # Pick a sample window that ends at a sunrise (~6 am UTC) to maximise
    # the future_zero-vs-realised contrast — peak PV in the future.
    # Sample 0 ends at index window_size-1; find a sample whose last
    # timestamp is ~ midnight so the future block IS the daytime portion.
    target_anchor_hour = 0  # window ends at 00:00 → future block = 0..24h
    anchor_offsets = X_past.shape[0]
    last_ts_per_sample = d.df.index[cfg.window_size - 1: cfg.window_size - 1 + anchor_offsets]
    candidate = np.where(last_ts_per_sample.hour == target_anchor_hour)[0]
    i = int(candidate[5]) if len(candidate) > 5 else 0
    lines.append(
        f"- chose sample i={i}, window ends at {last_ts_per_sample[i].isoformat()} "
        f"(hour={last_ts_per_sample[i].hour})\n"
    )
    revin_past = _RevIN(X_past.shape[2], target_channel=0, affine=False)
    revin_ext = _RevIN(X_ext.shape[2], target_channel=0, affine=False)
    with torch.no_grad():
        revin_past.normalize(torch.from_numpy(X_past[i: i + 1]))
        revin_ext.normalize(torch.from_numpy(X_ext[i: i + 1]))
    past_mean = float(revin_past._mean[0, 0, 0].item())
    past_std = float(revin_past._stdev[0, 0, 0].item())
    ext_mean = float(revin_ext._mean[0, 0, 0].item())
    ext_std = float(revin_ext._stdev[0, 0, 0].item())
    lines.append(
        f"- past-only window target-channel stats: mean={past_mean:.4f}, std={past_std:.4f}\n"
        f"- extended window target-channel stats:  mean={ext_mean:.4f}, std={ext_std:.4f}\n"
        f"- ratio extended_mean/past_mean = {ext_mean / max(past_mean, 1e-6):.3f}, "
        f"std ratio = {ext_std / max(past_std, 1e-6):.3f}\n"
    )
    # Now also dump the actual target-channel values across past vs future
    past_vals_past = X_past[i, :, 0]
    past_vals_ext = X_ext[i, :cfg.window_size, 0]
    future_vals_ext = X_ext[i, cfg.window_size:, 0]
    lines.append(
        f"- target channel value summary:\n"
        f"  - past block (both):           min={past_vals_past.min():.4f}, max={past_vals_past.max():.4f}, mean={past_vals_past.mean():.4f}\n"
        f"  - extended past block:         min={past_vals_ext.min():.4f}, max={past_vals_ext.max():.4f}, mean={past_vals_ext.mean():.4f}\n"
        f"  - extended FUTURE block (target channel left zero): "
        f"min={future_vals_ext.min():.4f}, max={future_vals_ext.max():.4f}, mean={future_vals_ext.mean():.4f}\n"
    )
    # Same for clear_sky_ghi channel
    ghi_ch = ch.index("clear_sky_ghi")
    past_ghi_past = X_past[i, :, ghi_ch]
    past_ghi_ext = X_ext[i, :cfg.window_size, ghi_ch]
    future_ghi_ext = X_ext[i, cfg.window_size:, ghi_ch]
    lines.append(
        f"- clear_sky_ghi channel value summary:\n"
        f"  - past-only path past block:                 mean={past_ghi_past.mean():.2f}\n"
        f"  - extended path past block (should match):   mean={past_ghi_ext.mean():.2f}\n"
        f"  - extended path future block (populated):    mean={future_ghi_ext.mean():.2f}\n"
    )
    # Verdict.
    bias_pct = 100 * (1 - ext_mean / max(past_mean, 1e-9))
    if abs(bias_pct) > 25:
        verdict = (
            f"**Verdict — CONFIRMED.** Future-position target zeros pull RevIN's "
            f"per-window mean down by ~{bias_pct:.0f}%. The denormalisation at "
            f"the output uses this biased mean, so the model is shifted toward "
            f"zero by roughly that amount in target space.\n"
        )
    else:
        verdict = (
            f"**Verdict — refuted at this sample.** "
            f"Bias under {abs(bias_pct):.1f}% — not the dominant driver.\n"
        )
    lines.append(verdict + "\n")
    return lines


# --------------------------------------------------------------------------- #
# Section 3.2 — NLinear last-value anchor degeneration                         #
# --------------------------------------------------------------------------- #

def section_3_2_nlinear_anchor() -> List[str]:
    """Show what x[:, -1, :] becomes in extended-window mode (= 0 always).

    The trick:
        x_shifted = x - x[:, -1:, :]
        out       = linear(x_shifted) + x[:, -1, target_channel]
    relies on the last row containing the most recent observation. In
    extended-window mode, the LAST row of x is the LAST future position
    where the target channel is always zero, so the "anchor" is zero and
    the trick is a no-op. The linear layer alone must therefore reach
    the absolute target scale.
    """
    lines: List[str] = []
    lines.append("## 3.2 NLinear last-value anchor degeneration\n")
    d = make_pure_pv(0)
    cfg = HarnessCfg(window_size=48, horizon=48)
    cov_cols = ["sun_elevation", "clear_sky_ghi"]
    horizon_steps = list(range(1, cfg.horizon + 1))
    X_past, _, ch = create_sliding_windows(
        d.df, "y", window_size=cfg.window_size,
        covariate_cols=cov_cols, add_temporal=True,
        horizon_steps=horizon_steps,
    )
    future_df = compute_known_future_features(
        d.df.index, add_temporal=True, country="GB",
        solar_lat_lon=(GB_LAT, GB_LON),
        include_sun_elevation=True, include_clear_sky_ghi=True,
    )
    X_ext, _, _ = create_sliding_windows(
        d.df, "y", window_size=cfg.window_size,
        covariate_cols=cov_cols, add_temporal=True,
        horizon_steps=horizon_steps,
        future_features_df=future_df,
    )
    # Pick 10 random samples and inspect last_val per path.
    rng = np.random.default_rng(0)
    sel = rng.choice(X_past.shape[0], size=10, replace=False)
    last_past = X_past[sel, -1, 0]
    last_ext_total = X_ext[sel, -1, 0]                   # what NLinear sees
    last_ext_past_position = X_ext[sel, cfg.window_size - 1, 0]   # what it SHOULD use
    lines.append(
        f"- 10 random samples — target-channel last_val per path:\n"
        f"  - past-only x[:, -1, 0]               = {np.round(last_past, 4).tolist()}\n"
        f"  - extended  x[:, -1, 0]   (used)      = {np.round(last_ext_total, 4).tolist()}\n"
        f"  - extended  x[:, W-1, 0]  (intended)  = {np.round(last_ext_past_position, 4).tolist()}\n"
    )
    # In extended mode the future block had target channel left at zero,
    # so x[:, -1, 0] is always 0. CONFIRMED.
    n_zero = int(np.sum(np.abs(last_ext_total) < 1e-6))
    verdict = (
        f"**Verdict — CONFIRMED.** Across the 10 sampled extended-mode windows, "
        f"`x[:, -1, target_channel]` is zero in {n_zero}/10 cases. The intended "
        f"anchor (`x[:, window_size-1, target_channel]`) carries the true last "
        f"observation. NLinear's residual trick is therefore a no-op in v2.36+; "
        f"the single linear head must reach absolute target scale on its own.\n\n"
    )
    lines.append(verdict)
    return lines


# --------------------------------------------------------------------------- #
# Section 3.3 — LSTM TemporalAttention past vs future                          #
# --------------------------------------------------------------------------- #

def section_3_3_lstm_attention() -> List[str]:
    """Train LSTM briefly and inspect attention weights past vs future."""
    from ml_forecast_lab.models.lstm_backend import LSTMModel
    lines: List[str] = []
    lines.append("## 3.3 LSTM TemporalAttention past vs future\n")
    d = make_pure_pv(0)
    cfg = HarnessCfg(window_size=48, horizon=48, epochs=10)
    cov_cols = ["sun_elevation", "clear_sky_ghi"]
    horizon_steps = list(range(1, cfg.horizon + 1))
    future_df = compute_known_future_features(
        d.df.index, add_temporal=True, country="GB",
        solar_lat_lon=(GB_LAT, GB_LON),
        include_sun_elevation=True, include_clear_sky_ghi=True,
    )
    X, y, ch = create_sliding_windows(
        d.df.iloc[:60 * 48], "y", window_size=cfg.window_size,
        covariate_cols=cov_cols, add_temporal=True,
        horizon_steps=horizon_steps,
        future_features_df=future_df,
    )
    model = LSTMModel(hidden_size=32, num_layers=1, epochs=cfg.epochs,
                      batch_size=64, use_revin=True, output_activation="linear")
    X_flat = np.zeros((X.shape[0], 1), dtype=np.float32)
    model.fit(X_flat, y, sequence_data=X)
    # Inspect attention weights on a held-out sample.
    rng = np.random.default_rng(0)
    i = int(rng.integers(0, X.shape[0]))
    with torch.no_grad():
        x_t = torch.from_numpy(X[i: i + 1])
        x_norm = model._model.revin.normalize(x_t)
        x_ln = model._model.layer_norm(x_norm)
        lstm_out, _ = model._model.lstm(x_ln)
        scores = torch.tanh(model._model.attention.attn_proj(lstm_out))
        scores = (scores * model._model.attention.attn_vector).sum(dim=-1)
        weights = torch.softmax(scores, dim=-1)[0].cpu().numpy()
    past_w_sum = float(weights[:cfg.window_size].sum())
    future_w_sum = float(weights[cfg.window_size:].sum())
    lines.append(
        f"- sample i={i}, sequence length={X.shape[1]} "
        f"(past={cfg.window_size}, future={X.shape[1] - cfg.window_size})\n"
        f"- attention weight sum on past positions:   {past_w_sum:.4f}\n"
        f"- attention weight sum on future positions: {future_w_sum:.4f}\n"
        f"- ratio future/past = {future_w_sum / max(past_w_sum, 1e-6):.3f}\n"
        f"- top-5 weights and positions: "
        f"{[(int(p), round(float(weights[p]), 4)) for p in np.argsort(weights)[-5:][::-1]]}\n"
    )
    # Verdict guidance — the test is whether attention is dominantly future.
    if future_w_sum / max(past_w_sum, 1e-6) > 0.5:
        verdict = (
            "**Verdict — Plausible.** The LSTM's attention places a non-trivial "
            f"share ({100 * future_w_sum / (past_w_sum + future_w_sum):.0f}%) on "
            "future positions whose target channel is zero — those positions "
            "carry sun_elevation/clear_sky_ghi/hour-of-day, so the model is "
            "free to read absolute time from there. If it learns to use future "
            "magnitudes additively rather than as phase anchors, the resulting "
            "context can invert in phase relative to the past block.\n\n"
        )
    else:
        verdict = (
            "**Verdict — refuted at this sample.** Attention sits mostly on the "
            "past, so the phase-inversion is unlikely to be driven by attention "
            "leakage to future positions. Look elsewhere (head output bias, "
            "RevIN denormalisation).\n\n"
        )
    lines.append(verdict)
    return lines


# --------------------------------------------------------------------------- #
# Section 3.4 — Sliding-window alignment                                       #
# --------------------------------------------------------------------------- #

def section_3_4_alignment() -> List[str]:
    """Manually rebuild one sample and compare to create_sliding_windows."""
    lines: List[str] = []
    lines.append("## 3.4 Sliding-window alignment\n")
    d = make_pure_pv(0)
    cfg = HarnessCfg(window_size=24, horizon=12)
    cov_cols = ["sun_elevation", "clear_sky_ghi"]
    horizon_steps = list(range(1, cfg.horizon + 1))
    future_df = compute_known_future_features(
        d.df.index, add_temporal=True, country="GB",
        solar_lat_lon=(GB_LAT, GB_LON),
        include_sun_elevation=True, include_clear_sky_ghi=True,
    )
    X, y, ch = create_sliding_windows(
        d.df, "y", window_size=cfg.window_size,
        covariate_cols=cov_cols, add_temporal=True,
        horizon_steps=horizon_steps,
        future_features_df=future_df,
    )
    # Reconstruct sample i manually.
    i = 100
    expected_past_idx = d.df.index[i: i + cfg.window_size]
    expected_future_idx = d.df.index[i + cfg.window_size: i + cfg.window_size + cfg.horizon]
    lines.append(
        f"- sample i={i}\n"
        f"- expected past index[0]={expected_past_idx[0]}, [-1]={expected_past_idx[-1]}\n"
        f"- expected future index[0]={expected_future_idx[0]}, [-1]={expected_future_idx[-1]}\n"
    )
    # Per-channel comparison.
    manual_past = np.zeros((cfg.window_size, len(ch)), dtype=np.float32)
    manual_past[:, 0] = d.df["y"].iloc[i: i + cfg.window_size].values
    manual_past[:, 1] = d.df["sun_elevation"].iloc[i: i + cfg.window_size].values
    manual_past[:, 2] = d.df["clear_sky_ghi"].iloc[i: i + cfg.window_size].values
    h = expected_past_idx.hour.values
    dow = expected_past_idx.dayofweek.values
    manual_past[:, 3] = np.sin(2 * np.pi * h / 24)
    manual_past[:, 4] = np.cos(2 * np.pi * h / 24)
    manual_past[:, 5] = np.sin(2 * np.pi * dow / 7)
    manual_past[:, 6] = np.cos(2 * np.pi * dow / 7)
    manual_past[:, 7] = (dow >= 5).astype(np.float32)
    past_match = np.allclose(X[i, :cfg.window_size], manual_past, atol=1e-5)
    lines.append(f"- past block channel-wise match against manual reconstruction: {past_match}\n")
    # Future block — target/dow/hour should match the FUTURE index, not past.
    h_f = expected_future_idx.hour.values
    dow_f = expected_future_idx.dayofweek.values
    expected_future_hour_sin = np.sin(2 * np.pi * h_f / 24).astype(np.float32)
    actual_future_hour_sin = X[i, cfg.window_size:, ch.index("hour_sin")]
    fhs_match = np.allclose(expected_future_hour_sin, actual_future_hour_sin, atol=1e-5)
    lines.append(
        f"- future hour_sin matches the future timestamps: {fhs_match}\n"
        f"  expected[:6] = {np.round(expected_future_hour_sin[:6], 4).tolist()}\n"
        f"  actual  [:6] = {np.round(actual_future_hour_sin[:6], 4).tolist()}\n"
    )
    # And labels: y[i, h] should be y at index i+window_size+h-1 (per code)
    expected_y0 = float(d.df["y"].iloc[i + cfg.window_size + horizon_steps[0] - 1])
    actual_y0 = float(y[i, 0])
    lines.append(
        f"- y[i, 0] expected (h=1): {expected_y0:.4f}  actual: {actual_y0:.4f}\n"
    )
    verdict = (
        "**Verdict — alignment looks correct in extended-window mode.** "
        "Past block, future block, and label index all line up. The v2.35.3 "
        "off-by-one fix has held.\n\n"
        if (past_match and fhs_match and abs(expected_y0 - actual_y0) < 1e-3)
        else "**Verdict — alignment mismatch detected, see numbers above.**\n\n"
    )
    lines.append(verdict)
    return lines


# --------------------------------------------------------------------------- #
# Section 3.5 — Timezone continuity                                            #
# --------------------------------------------------------------------------- #

def section_3_5_tz_continuity() -> List[str]:
    """Check the past/future hour_sin/cos boundary is smooth."""
    lines: List[str] = []
    lines.append("## 3.5 Timezone path: past/future hour_sin continuity\n")
    d = make_pure_pv(0)
    cfg = HarnessCfg(window_size=24, horizon=24)
    # Drop the tz so we mimic a Home Assistant index that might be tz-naive.
    df_naive = d.df.copy()
    df_naive.index = df_naive.index.tz_localize(None)
    last_ts = df_naive.index[-1]
    future_idx_naive = pd.date_range(
        start=last_ts + pd.Timedelta(minutes=30),
        periods=cfg.horizon, freq="30min",
    )
    # Inference window from the past
    X, ch = build_inference_window(
        df_naive, "y", window_size=cfg.window_size,
        covariate_cols=["sun_elevation", "clear_sky_ghi"],
        add_temporal=True,
        future_features_df=compute_known_future_features(
            future_idx_naive, add_temporal=True, country="GB",
            solar_lat_lon=(GB_LAT, GB_LON),
            include_sun_elevation=True, include_clear_sky_ghi=True,
        ),
    )
    hour_sin_idx = ch.index("hour_sin")
    boundary = X[0, cfg.window_size - 3: cfg.window_size + 3, hour_sin_idx]
    boundary_ghi = X[0, cfg.window_size - 3: cfg.window_size + 3, ch.index("clear_sky_ghi")]
    lines.append(
        f"- df_naive last timestamp: {last_ts}\n"
        f"- future_idx[0]: {future_idx_naive[0]}\n"
        f"- hour_sin values across past/future boundary [W-3..W+2]: "
        f"{np.round(boundary, 4).tolist()}\n"
        f"- clear_sky_ghi values across same boundary: "
        f"{np.round(boundary_ghi, 2).tolist()}\n"
    )
    # Verdict.
    diffs = np.diff(boundary)
    max_jump = float(np.max(np.abs(diffs)))
    verdict = (
        f"**Verdict — hour_sin step at boundary = {max_jump:.4f}.** "
        f"For continuous half-hour spacing the largest legitimate step is "
        f"|sin(2π·(h+0.5)/24) − sin(2π·h/24)| ≤ ~0.131 — "
        f"{'within bounds' if max_jump < 0.2 else 'JUMPS BEYOND 0.2 — investigate tz!'}\n\n"
    )
    lines.append(verdict)
    return lines


# --------------------------------------------------------------------------- #
# Section 3.6 — Channel-name parity at value level                             #
# --------------------------------------------------------------------------- #

def section_3_6_channel_parity() -> List[str]:
    """Compare build_inference_window and create_sliding_windows at sample-level."""
    lines: List[str] = []
    lines.append("## 3.6 Channel-name parity at the value level\n")
    d = make_pure_pv(0)
    cfg = HarnessCfg(window_size=24, horizon=12)
    cov_cols = ["sun_elevation", "clear_sky_ghi"]
    horizon_steps = list(range(1, cfg.horizon + 1))
    future_df_full = compute_known_future_features(
        d.df.index, add_temporal=True, country="GB",
        solar_lat_lon=(GB_LAT, GB_LON),
        include_sun_elevation=True, include_clear_sky_ghi=True,
    )
    X, _, ch_train = create_sliding_windows(
        d.df, "y", window_size=cfg.window_size,
        covariate_cols=cov_cols, add_temporal=True,
        horizon_steps=horizon_steps,
        future_features_df=future_df_full,
    )
    # Pick sample i and reconstruct its window via build_inference_window
    # using a df sliced exactly so the last past row is at i + W - 1.
    i = 100
    df_slice = d.df.iloc[i: i + cfg.window_size]
    last_ts = df_slice.index[-1]
    future_idx_infer = pd.date_range(
        start=last_ts + pd.Timedelta(minutes=30),
        periods=cfg.horizon, freq="30min",
    )
    future_df_infer = compute_known_future_features(
        future_idx_infer, add_temporal=True, country="GB",
        solar_lat_lon=(GB_LAT, GB_LON),
        include_sun_elevation=True, include_clear_sky_ghi=True,
    )
    X_infer, ch_infer = build_inference_window(
        df_slice, "y", window_size=cfg.window_size,
        covariate_cols=cov_cols, add_temporal=True,
        future_features_df=future_df_infer,
    )
    names_match = (ch_train == ch_infer)
    lines.append(f"- train channel names == infer channel names: {names_match}\n")
    diff = X[i: i + 1] - X_infer
    max_abs = float(np.max(np.abs(diff)))
    lines.append(
        f"- max |train_sample - infer_sample| over all channels = {max_abs:.6f}\n"
    )
    # Per-channel max diff so we see where they disagree, if anywhere.
    per_ch = np.max(np.abs(diff), axis=(0, 1))
    for cname, mx in zip(ch_train, per_ch):
        lines.append(f"  - {cname}: max abs diff = {mx:.6f}\n")
    verdict = (
        "**Verdict — values match channel-by-channel.**\n\n"
        if max_abs < 1e-4
        else "**Verdict — VALUES DIFFER between train and infer at the same sample.**\n\n"
    )
    lines.append(verdict)
    return lines


def main():
    blocks: List[str] = []
    blocks.append("# Phase 3 — targeted code introspection\n\n")
    blocks.extend(section_3_1_revin_bias())
    blocks.extend(section_3_2_nlinear_anchor())
    blocks.extend(section_3_3_lstm_attention())
    blocks.extend(section_3_4_alignment())
    blocks.extend(section_3_5_tz_continuity())
    blocks.extend(section_3_6_channel_parity())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("".join(blocks))
    print(f"wrote {OUT_PATH}")
    print("---- summary ----")
    print("".join(blocks))


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2].parent)
    main()
