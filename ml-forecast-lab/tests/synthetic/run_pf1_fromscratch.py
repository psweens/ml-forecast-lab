"""
From-scratch retrain with PF1 in place — does this clear the residual
compression on linear-head backends?

The original `run_prototype_fixes.py` applies PF1 by monkey-patching
the trained RevIN module and fine-tuning 5 epochs. That's enough to
show the LSTM/CNN phase recovery, but not enough for the linear-head
backends' head to fully re-learn the correct scale.

Here we monkey-patch the global `_RevIN` class so it computes
past-only stats during training (not just at inference), then train
each linear-head backend FROM SCRATCH for the full 20 epochs.

Also reports peak_hour_pred against an "ideal" peak that uses the
truth's hour-of-noise-free-signal, not the noise-shifted empirical
peak — to separate "the model learned to peak late" from "the
empirical truth happened to peak at hour 11".
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.synthetic.datasets import make_realistic_pv, _pv_shape
from tests.synthetic.harness import (
    HarnessCfg, _train_neural_backend, _rolling_evaluate_neural,
)

import ml_forecast_lab.models.base as base_mod
from ml_forecast_lab.models.base import _RevIN as _OrigRevIN

logging.basicConfig(level=logging.WARNING)


class _RevINPastOnlyGlobal(_OrigRevIN):
    """A RevIN variant that reads past_window_size from a class attribute.

    We set this attribute BEFORE creating the model so every RevIN
    instance the backend builds picks it up automatically. After the
    run we restore the original class. Hack, but keeps changes
    confined to this script.
    """
    PAST_WINDOW_SIZE: int = 48  # set by main()

    def normalize(self, x):
        past = x[:, : self.PAST_WINDOW_SIZE, :]
        mean = past.mean(dim=1, keepdim=True).detach()
        var = past.var(dim=1, keepdim=True, unbiased=False).detach()
        stdev = torch.sqrt(var + self.eps)
        self._mean = mean
        self._stdev = stdev
        x_norm = (x - mean) / stdev
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        return x_norm


def _all_linear_head_backends():
    catalog = [
        ("nlinear", "nlinear_backend", "NLinearModel"),
        ("dlinear", "dlinear_backend", "DLinearModel"),
        ("sparsetsf", "sparsetsf_backend", "SparseTSFModel"),
        ("fits", "fits_backend", "FITSModel"),
        ("tsmixer", "tsmixer_backend", "TSMixerModel"),
        ("timemixer", "timemixer_backend", "TimeMixerModel"),
        ("tide", "tide_backend", "TiDEModel"),
        ("itransformer", "itransformer_backend", "iTransformerModel"),
        ("timesnet", "timesnet_backend", "TimesNetModel"),
        ("tft", "tft_backend", "TFTModel"),
    ]
    out = []
    for name, mod, cls in catalog:
        try:
            m = __import__(f"ml_forecast_lab.models.{mod}", fromlist=[cls])
            out.append((name, getattr(m, cls)))
        except Exception as e:
            print(f"  SKIP {name}: {e}")
    return out


def _diag(r, noise_free_peak_hour: int) -> Dict[str, Any]:
    if r.pred.size == 0:
        return {"mae": float("nan"), "peak_pred": -1, "flatness": float("nan"),
                "noise_free_peak": noise_free_peak_hour}
    H = r.pred.shape[1]
    rows = [{"hour": r.truth_idx[w][h].hour,
             "pred": r.pred[w, h], "truth": r.truth[w, h]}
            for w in range(len(r.truth_idx)) for h in range(H)]
    df = pd.DataFrame(rows)
    by_hour = df.groupby("hour").mean()
    peak_truth_emp = int(by_hour["truth"].idxmax())
    peak_pred = int(by_hour["pred"].idxmax())
    flatness = float(r.pred.std(axis=1).mean() / max(1e-6, r.truth.std(axis=1).mean()))
    return {
        "mae": r.mae,
        "peak_truth_empirical": peak_truth_emp,
        "peak_pred": peak_pred,
        "noise_free_peak": noise_free_peak_hour,
        "flatness": flatness,
    }


def main():
    ds = make_realistic_pv(0)
    cfg = HarnessCfg(epochs=25, holdout_days=10, window_size=48, horizon=48)
    cov_cols = ["sun_elevation", "clear_sky_ghi"]
    steps_per_day = 1440 // cfg.interval_minutes
    holdout_start = len(ds.df) - cfg.holdout_days * steps_per_day
    df_train = ds.df.iloc[holdout_start - 60 * steps_per_day: holdout_start]

    # Compute the noise-free peak hour from the underlying _pv_shape.
    # (The realistic_pv is _pv_shape * cloud + noise + dropouts; the
    # "true" peak before noise is at the maximum of _pv_shape.)
    noise_free = _pv_shape(ds.df.index, scale=4500.0)
    nf = pd.Series(noise_free, index=ds.df.index)
    nf_by_hour = nf.groupby(nf.index.hour).mean()
    nf_peak = int(nf_by_hour.idxmax())
    print(f"Noise-free signal peaks at hour {nf_peak} (UTC)")

    backends = _all_linear_head_backends()
    rows = []
    for name, cls in backends:
        # Baseline.
        print(f"\n== {name} ==", flush=True)
        # Restore stock RevIN
        base_mod._RevIN = _OrigRevIN
        # Force the model import to pick up the right RevIN
        import importlib
        mod = importlib.import_module(cls.__module__)
        importlib.reload(mod)
        cls_fresh = getattr(mod, cls.__name__)

        np.random.seed(0); torch.manual_seed(0)
        t0 = time.time()
        try:
            model, _, _ = _train_neural_backend(cls_fresh, df_train, cfg, cov_cols)
            r = _rolling_evaluate_neural(
                model, ds.df, cfg, cov_cols, cfg.window_size,
                holdout_start_idx=holdout_start, n_eval_windows=10,
                backend_name=name,
            )
            base = _diag(r, nf_peak)
            base["fit_s"] = time.time() - t0
            print(f"  baseline  mae={base['mae']:.1f}  peak={base['peak_pred']} "
                  f"(emp={base['peak_truth_empirical']}, signal={nf_peak})  "
                  f"flat={base['flatness']:.2f}  ({base['fit_s']:.1f}s)", flush=True)
        except Exception as e:
            print(f"  baseline FAILED: {e!r}")
            base = {"mae": float("nan"), "flatness": float("nan"),
                    "peak_pred": -1, "fit_s": 0.0,
                    "peak_truth_empirical": -1, "noise_free_peak": nf_peak}

        # PF1 from scratch.
        _RevINPastOnlyGlobal.PAST_WINDOW_SIZE = cfg.window_size
        base_mod._RevIN = _RevINPastOnlyGlobal
        importlib.reload(mod)
        cls_pf1 = getattr(mod, cls.__name__)
        np.random.seed(0); torch.manual_seed(0)
        t0 = time.time()
        try:
            model, _, _ = _train_neural_backend(cls_pf1, df_train, cfg, cov_cols)
            r = _rolling_evaluate_neural(
                model, ds.df, cfg, cov_cols, cfg.window_size,
                holdout_start_idx=holdout_start, n_eval_windows=10,
                backend_name=name,
            )
            pf1 = _diag(r, nf_peak)
            pf1["fit_s"] = time.time() - t0
            print(f"  PF1 fresh mae={pf1['mae']:.1f}  peak={pf1['peak_pred']}  "
                  f"flat={pf1['flatness']:.2f}  ({pf1['fit_s']:.1f}s)", flush=True)
        except Exception as e:
            print(f"  PF1 FAILED: {e!r}")
            pf1 = {"mae": float("nan"), "flatness": float("nan"),
                   "peak_pred": -1, "fit_s": 0.0,
                   "peak_truth_empirical": -1, "noise_free_peak": nf_peak}

        rows.append({
            "backend": name,
            "baseline_mae": base["mae"], "baseline_flat": base["flatness"],
            "baseline_peak": base["peak_pred"],
            "pf1_mae": pf1["mae"], "pf1_flat": pf1["flatness"],
            "pf1_peak": pf1["peak_pred"],
            "noise_free_peak": nf_peak,
        })
        # Restore for next iteration
        base_mod._RevIN = _OrigRevIN

    # Write a markdown summary.
    out_md = Path("docs/investigations/pf1_fromscratch_linearheads.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# PF1 from-scratch retrain — linear-head backends\n\n"]
    lines.append(f"Dataset: `realistic_pv`. Noise-free signal peaks at hour {nf_peak} (UTC); "
                 "empirical truth peak hour varies per holdout due to cloud noise.\n\n")
    lines.append("|backend|baseline MAE|baseline flat|baseline peak|"
                 "PF1 MAE|PF1 flat|PF1 peak|peak fixed?|flatness lift|\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
    for row in rows:
        if not np.isfinite(row["pf1_flat"]):
            lines.append(f"|{row['backend']}|—|—|—|FAILED|—|—|—|—|\n")
            continue
        peak_fixed = "yes" if row["pf1_peak"] == nf_peak else f"no (off {abs(row['pf1_peak']-nf_peak)}h)"
        if np.isfinite(row["baseline_flat"]) and row["baseline_flat"] > 0:
            flat_lift = f"{row['pf1_flat']/max(1e-6, row['baseline_flat']):.2f}x"
        else:
            flat_lift = "—"
        lines.append(
            f"|{row['backend']}|"
            f"{row['baseline_mae']:.1f}|{row['baseline_flat']:.2f}|{row['baseline_peak']}|"
            f"{row['pf1_mae']:.1f}|{row['pf1_flat']:.2f}|{row['pf1_peak']}|"
            f"{peak_fixed}|{flat_lift}|\n"
        )
    out_md.write_text("".join(lines))
    print(f"\nwrote {out_md}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2].parent)
    main()
