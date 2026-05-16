"""
Prototype fixes for the two confirmed root causes — and a head-to-head
comparison against current (broken) behaviour on realistic_pv.

Prototypes
----------
PF1 — RevIN-past-only: per-window mean/std computed over the past
      positions only (positions [:past_window_size]). Denormalisation
      uses the same past-only stats. Implemented as a tiny subclass that
      monkey-patches in via the backend's existing ``_RevIN`` instance.

PF2 — NLinear anchor at past window end: anchor on
      ``x[:, past_window_size - 1, target_channel]`` instead of
      ``x[:, -1, target_channel]``. Implemented by subclassing _NLinearNet
      and overriding ``forward``.

These prototypes are constructed in this file only; **no production
code is modified**.

The script runs four conditions per backend on realistic_pv:

  - baseline  (current code — RevIN over whole window, anchor at -1)
  - PF1 only  (RevIN-past-only, NLinear anchor unchanged)
  - PF2 only  (RevIN unchanged, NLinear anchor at past-end)
  - PF1 + PF2

and prints overall MAE + flatness + peak_hour for each.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.synthetic.datasets import make_realistic_pv
from tests.synthetic.harness import HarnessCfg

from ml_forecast_lab.features import (
    build_inference_window, compute_known_future_features, create_sliding_windows,
)
from ml_forecast_lab.models.base import _RevIN

logging.basicConfig(level=logging.WARNING)
PROTOTYPE_LOG = Path("docs/investigations/prototype_fixes.md")
GB_LAT, GB_LON = 52.0, -1.0


# --------------------------------------------------------------------------- #
# Prototype 1 — RevIN past-only                                                #
# --------------------------------------------------------------------------- #

class _RevINPastOnly(_RevIN):
    """RevIN that computes per-window stats over PAST positions only.

    The future block is left zero-padded for the target channel in
    extended-window training (v2.36+). Including those zeros in the
    per-window mean halves it — and the denormalisation at the head
    then rescales predictions to that halved level. Past-only stats
    avoid that bias.
    """

    def __init__(self, n_channels: int, past_window_size: int, **kwargs: Any) -> None:
        super().__init__(n_channels, **kwargs)
        self.past_window_size = int(past_window_size)

    def normalize(self, x: "torch.Tensor") -> "torch.Tensor":
        past = x[:, : self.past_window_size, :]
        mean = past.mean(dim=1, keepdim=True).detach()
        var = past.var(dim=1, keepdim=True, unbiased=False).detach()
        stdev = torch.sqrt(var + self.eps)
        self._mean = mean
        self._stdev = stdev
        x_norm = (x - mean) / stdev
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        return x_norm


# --------------------------------------------------------------------------- #
# Prototype 2 — NLinear anchored at past_window_size - 1                       #
# --------------------------------------------------------------------------- #

def _patch_nlinear(model, past_window_size: int) -> None:
    """Replace forward() of an _NLinearNet to anchor at the past-window end."""
    net = model._model
    if net is None:
        raise RuntimeError("Model has no _model")
    pw = int(past_window_size)
    orig_forward_target_channel = net.target_channel
    activation = net.activation
    use_revin = net.use_revin
    linear = net.linear

    def forward(x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[no-redef]
        if net.revin is not None:
            x = net.revin.normalize(x)
        # Anchor on the LAST PAST position rather than the literal last row.
        last_val = x[:, pw - 1: pw, orig_forward_target_channel]      # (B, 1)
        anchor_full = x[:, pw - 1: pw, :]                              # (B, 1, C)
        x_shifted = x - anchor_full
        flat = x_shifted.reshape(x_shifted.size(0), -1)
        out = linear(flat)
        out = out + last_val
        if net.revin is not None:
            out = net.revin.denormalize(out)
        out = activation(out)
        if net.n_horizons == 1:
            return out.squeeze(-1)
        return out

    net.forward = forward  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #

def _build_train_data(df: pd.DataFrame, cfg: HarnessCfg, cov_cols: List[str]):
    horizon_steps = list(range(1, cfg.horizon + 1))
    future_df = compute_known_future_features(
        df.index, add_temporal=True, country="GB",
        solar_lat_lon=(GB_LAT, GB_LON),
        include_sun_elevation=True, include_clear_sky_ghi=True,
    )
    X, y, ch = create_sliding_windows(
        df, "y", window_size=cfg.window_size,
        covariate_cols=cov_cols, add_temporal=True,
        horizon_steps=horizon_steps,
        future_features_df=future_df,
    )
    return X, y, ch


def _train_and_eval(
    backend_name: str,
    backend_cls,
    cfg: HarnessCfg,
    df_train: pd.DataFrame,
    df_full: pd.DataFrame,
    cov_cols: List[str],
    holdout_start_idx: int,
    n_eval: int,
    apply_pf1: bool,
    apply_pf2: bool,
) -> Tuple[float, float, int, str]:
    """Returns (mae, flatness, peak_hour_pred, note)."""
    np.random.seed(0)
    torch.manual_seed(0)
    X, y, ch = _build_train_data(df_train, cfg, cov_cols)
    model = backend_cls(epochs=cfg.epochs, batch_size=64, use_revin=cfg.use_revin,
                        output_activation="linear", daily_loss_weight=0.0,
                        optimiser="adamw", patience=8)
    X_flat = np.zeros((X.shape[0], 1), dtype=np.float32)
    model.fit(X_flat, y, sequence_data=X)
    # Apply PF1 by swapping the RevIN module in the trained model with a
    # past-only one and copying the affine params. We then re-run a
    # MINI fine-tune for 5 epochs so the model adapts to the new stats.
    if apply_pf1 and model._model.revin is not None:
        old = model._model.revin
        new = _RevINPastOnly(
            old.n_channels, cfg.window_size,
            target_channel=old.target_channel, eps=old.eps, affine=old.affine,
        )
        if old.affine:
            with torch.no_grad():
                new.affine_weight.copy_(old.affine_weight)
                new.affine_bias.copy_(old.affine_bias)
        model._model.revin = new
        # Mini re-fine-tune so the linear head adapts to the new stats.
        from torch.optim import AdamW
        opt = AdamW(model._model.parameters(), lr=5e-4, weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        for _epoch in range(5):
            perm = torch.randperm(len(X_t))
            for s in range(0, len(X_t), 64):
                idx = perm[s: s + 64]
                opt.zero_grad()
                pred = model._model(X_t[idx])
                loss_fn(pred, y_t[idx]).backward()
                opt.step()
    if apply_pf2 and backend_name == "nlinear":
        _patch_nlinear(model, cfg.window_size)
        # Mini re-fit because the head bias has to absorb the new anchor.
        from torch.optim import AdamW
        opt = AdamW(model._model.parameters(), lr=5e-4, weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        for _epoch in range(5):
            perm = torch.randperm(len(X_t))
            for s in range(0, len(X_t), 64):
                idx = perm[s: s + 64]
                opt.zero_grad()
                pred = model._model(X_t[idx])
                loss_fn(pred, y_t[idx]).backward()
                opt.step()
    # Evaluate.
    preds = []; truths = []; idxs = []
    H = cfg.horizon
    for k in range(n_eval):
        a = holdout_start_idx + k
        if a + H > len(df_full) or a < cfg.window_size:
            continue
        history = df_full.iloc[a - cfg.window_size: a]
        future_idx = pd.date_range(
            start=history.index[-1] + pd.Timedelta(minutes=cfg.interval_minutes),
            periods=H, freq=f"{cfg.interval_minutes}min",
        )
        future_df = compute_known_future_features(
            future_idx, add_temporal=True, country="GB",
            solar_lat_lon=(GB_LAT, GB_LON),
            include_sun_elevation=True, include_clear_sky_ghi=True,
        )
        Xinf, _ = build_inference_window(
            history, "y", window_size=cfg.window_size,
            covariate_cols=cov_cols, add_temporal=True,
            future_features_df=future_df,
        )
        pred = model.predict_sequence(Xinf).reshape(-1)[:H]
        preds.append(pred)
        truths.append(df_full["y"].iloc[a: a + H].values)
        idxs.append(df_full.index[a: a + H])
    if not preds:
        return float("nan"), float("nan"), -1, "no eval windows"
    pa = np.vstack(preds); ta = np.vstack(truths)
    mae = float(np.mean(np.abs(pa - ta)))
    flat = float(pa.std(axis=1).mean() / max(1e-6, ta.std(axis=1).mean()))
    pairs = pd.DataFrame([
        {"hour": idxs[w][h].hour, "pred": pa[w, h]}
        for w in range(len(idxs)) for h in range(pa.shape[1])
    ])
    peak = int(pairs.groupby("hour")["pred"].mean().idxmax())
    return mae, flat, peak, ""


def main():
    from ml_forecast_lab.models.nlinear_backend import NLinearModel
    from ml_forecast_lab.models.lstm_backend import LSTMModel
    from ml_forecast_lab.models.cnn_backend import CNNModel
    from ml_forecast_lab.models.sparsetsf_backend import SparseTSFModel
    BACKENDS = [
        ("nlinear", NLinearModel),
        ("sparsetsf", SparseTSFModel),
        ("lstm", LSTMModel),
        ("cnn", CNNModel),
    ]
    cfg = HarnessCfg(epochs=20, window_size=48, horizon=48)
    ds = make_realistic_pv(0)
    cov_cols = ["sun_elevation", "clear_sky_ghi"]
    steps_per_day = 1440 // cfg.interval_minutes
    holdout_start = len(ds.df) - cfg.holdout_days * steps_per_day
    df_train = ds.df.iloc[holdout_start - 60 * steps_per_day: holdout_start]
    lines: List[str] = []
    lines.append("# Prototype fix comparison\n\n")
    lines.append("Dataset: `realistic_pv` (~4500 W peak, AR(1) clouds, integer quantisation).\n\n")
    lines.append("|backend|condition|mae|flatness|peak_hour|\n|---|---|---:|---:|---:|\n")
    print("Backend  | condition       | mae | flat | peak")
    print("-" * 60)
    for name, cls in BACKENDS:
        for cond_name, pf1, pf2 in [
            ("baseline (broken)", False, False),
            ("PF1 (RevIN past-only)", True, False),
            ("PF2 (NLinear anchor)", False, True),
            ("PF1 + PF2", True, True),
        ]:
            if cond_name.startswith("PF2") and name != "nlinear":
                continue  # PF2 only affects NLinear
            t0 = time.time()
            mae, flat, peak, note = _train_and_eval(
                name, cls, cfg, df_train, ds.df, cov_cols,
                holdout_start, n_eval=10,
                apply_pf1=pf1, apply_pf2=pf2,
            )
            line = f"{name:>9}  | {cond_name:<22} | {mae:7.1f} | {flat:.2f} | {peak:>2} ({time.time() - t0:.1f}s)"
            print(line)
            lines.append(f"|{name}|{cond_name}|{mae:.1f}|{flat:.2f}|{peak}|\n")
    PROTOTYPE_LOG.parent.mkdir(parents=True, exist_ok=True)
    PROTOTYPE_LOG.write_text("".join(lines))
    print(f"\nwrote {PROTOTYPE_LOG}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2].parent)
    main()
