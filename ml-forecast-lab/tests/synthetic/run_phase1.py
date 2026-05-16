"""
Phase 1 — synthetic ground-truth baseline.

For each of pure_pv / cloudy_pv / ev_mixergy we train every neural
backend and a tree control, then record:

  - overall MAE
  - per-horizon MAE
  - per-hour-of-day MAE (does the model predict ~0 at 03:00?)
  - mean predicted bell shape across the holdout windows
  - peak hour vs ground-truth peak hour (catches phase inversion)
  - "flat-as-mean" detector — is std(pred over horizon) << std(truth)?

Outputs go to docs/investigations/figures_phase1/ as PNGs and one JSON
summary; phase 1 observations are appended to docs/investigations/2026-05-neural-pv.md.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Allow `python tests/synthetic/run_phase1.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.synthetic.datasets import (
    make_cloudy_pv, make_ev_mixergy, make_pure_pv, make_realistic_pv,
    SyntheticData,
)
from tests.synthetic.harness import HarnessCfg, run_neural_and_tree

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("phase1")

OUT_DIR = Path("docs/investigations/figures_phase1")
DOC_PATH = Path("docs/investigations/2026-05-neural-pv.md")
SUMMARY_PATH = Path("docs/investigations/phase1_summary.json")


def _save_plot(name: str, fig) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  wrote {path}")


def _diagnose_backend(name: str, r) -> Dict[str, float]:
    """Build the per-backend diagnostic numbers used in the writeup."""
    if r.pred.size == 0:
        return {
            "mae": float("nan"),
            "peak_hour_truth": float("nan"),
            "peak_hour_pred": float("nan"),
            "phase_offset_h": float("nan"),
            "pred_std_over_horizon": float("nan"),
            "truth_std_over_horizon": float("nan"),
            "flatness_ratio": float("nan"),
            "night_mae": float("nan"),
            "day_mae": float("nan"),
            "note": r.note,
        }
    # Mean curve across all forecast windows, indexed by hour.
    rows = []
    for w in range(len(r.truth_idx)):
        for h in range(r.pred.shape[1]):
            rows.append({
                "hour": r.truth_idx[w][h].hour,
                "pred": r.pred[w, h],
                "truth": r.truth[w, h],
            })
    df = pd.DataFrame(rows)
    by_hour = df.groupby("hour").mean()
    peak_truth = float(by_hour["truth"].idxmax())
    peak_pred = float(by_hour["pred"].idxmax())
    pred_std = float(r.pred.std(axis=1).mean())
    truth_std = float(r.truth.std(axis=1).mean())
    flatness = pred_std / max(1e-6, truth_std)
    night_mask = df["hour"].isin([0, 1, 2, 3, 4, 22, 23])
    day_mask = df["hour"].isin([10, 11, 12, 13, 14])
    night_mae = float((df.loc[night_mask, "pred"] - df.loc[night_mask, "truth"]).abs().mean())
    day_mae = float((df.loc[day_mask, "pred"] - df.loc[day_mask, "truth"]).abs().mean())
    return {
        "mae": r.mae,
        "peak_hour_truth": peak_truth,
        "peak_hour_pred": peak_pred,
        "phase_offset_h": (peak_pred - peak_truth) % 24,
        "pred_std_over_horizon": pred_std,
        "truth_std_over_horizon": truth_std,
        "flatness_ratio": flatness,
        "night_mae": night_mae,
        "day_mae": day_mae,
        "note": r.note,
    }


def _plot_dataset(dataset: SyntheticData, results: Dict[str, object], cfg: HarnessCfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Plot 1: mean predicted curve across the holdout, per backend.
    fig, ax = plt.subplots(figsize=(10, 6))
    # Mean truth — average y_truth at each horizon step across windows
    sample = next(iter(results.values()))
    if sample.pred.size == 0:
        plt.close(fig); return
    H = sample.pred.shape[1]
    mean_truth_by_hour = pd.DataFrame([
        {"hour": sample.truth_idx[w][h].hour, "truth": sample.truth[w, h]}
        for w in range(len(sample.truth_idx)) for h in range(H)
    ]).groupby("hour")["truth"].mean()
    ax.plot(mean_truth_by_hour.index, mean_truth_by_hour.values, "k--",
            linewidth=2.5, label="truth (mean by hour)")
    for name, r in results.items():
        if r.pred.size == 0:
            continue
        df = pd.DataFrame([
            {"hour": r.truth_idx[w][h].hour, "pred": r.pred[w, h]}
            for w in range(len(r.truth_idx)) for h in range(r.pred.shape[1])
        ])
        s = df.groupby("hour")["pred"].mean()
        ax.plot(s.index, s.values, marker="o", linewidth=1.4, label=name)
    ax.set_xlabel("hour of day (UTC, at forecast target time)")
    ax.set_ylabel("y")
    ax.set_title(f"{dataset.name} — mean forecast curve by hour, extended_window={cfg.extended_window}, RevIN={cfg.use_revin}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    _save_plot(f"mean_by_hour__{dataset.name}", fig)
    plt.close(fig)
    # Plot 2: example single window — first eval window predictions vs truth
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(H), sample.truth[0], "k--", linewidth=2.5, label="truth")
    for name, r in results.items():
        if r.pred.size == 0:
            continue
        ax.plot(np.arange(H), r.pred[0], marker="o", linewidth=1.3, label=name)
    ax.set_xlabel("horizon step (30-min)")
    ax.set_ylabel("y")
    starts = sample.truth_idx[0][0].strftime("%Y-%m-%d %H:%M")
    ax.set_title(f"{dataset.name} — first holdout window starting {starts}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    _save_plot(f"window0__{dataset.name}", fig)
    plt.close(fig)


def main():
    cfg = HarnessCfg(
        epochs=15,
        holdout_days=10,
        extended_window=True,         # v2.36+ path — DEFAULT
        use_solar_covariates=True,
        use_revin=True,
        output_activation="linear",
        daily_loss_weight=0.0,
        optimiser="adamw",
    )
    backends = ["nlinear", "dlinear", "sparsetsf", "lstm", "cnn", "lightgbm"]
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for ds in (make_pure_pv(0), make_cloudy_pv(0), make_realistic_pv(0), make_ev_mixergy(0)):
        print(f"\n=== Dataset: {ds.name} ===  n_rows={len(ds.df)} cols={list(ds.df.columns)}")
        t0 = time.time()
        results = run_neural_and_tree(
            ds.df, cfg, backends=backends,
            train_subset_days=45, n_eval_windows=8,
        )
        print(f"  trained in {time.time() - t0:.1f}s")
        diag = {}
        for n in backends:
            r = results.get(n)
            if r is None:
                continue
            diag[n] = _diagnose_backend(n, r)
            d = diag[n]
            print(
                f"  {n:>10}  mae={d['mae']:.4f}  peak_truth={d['peak_hour_truth']:>4.1f}  "
                f"peak_pred={d['peak_hour_pred']:>4.1f}  flatness={d['flatness_ratio']:.2f}  "
                f"night_mae={d['night_mae']:.4f}  day_mae={d['day_mae']:.4f}  {d['note'][:40]}"
            )
        summary[ds.name] = diag
        _plot_dataset(ds, results, cfg)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2].parent)  # repo root
    main()
