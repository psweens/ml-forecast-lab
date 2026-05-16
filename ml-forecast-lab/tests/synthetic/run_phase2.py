"""
Phase 2 — settings sweep.

Phase 1 showed the production failure does not reproduce on clean
synthetic data — so this sweep has two goals:

  A. on `cloudy_pv` (which is the closest analogue of the user's PV
     target), sweep the same axes the brief lists and record whether
     any combination DOES break the neural backends.
  B. on `ev_mixergy`, where LSTM/CNN already collapse, sweep the same
     axes and find which knob flips them back to a correct shape — that
     tells us what about the production target is causing it to collapse.

The most important comparison is `extended_window=on vs off` with
everything else held constant — that isolates the v2.36 effect.

Outputs go to docs/investigations/phase2_summary.json and produce
one PNG per (dataset × axis) showing per-axis MAE bars.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.synthetic.datasets import (
    make_cloudy_pv, make_ev_mixergy, make_pure_pv,
)
from tests.synthetic.harness import HarnessCfg, run_neural_and_tree

logging.basicConfig(level=logging.WARNING)
OUT_DIR = Path("docs/investigations/figures_phase2")
SUMMARY_PATH = Path("docs/investigations/phase2_summary.json")


AXES: List[Tuple[str, List[Any]]] = [
    # (cfg_field, values)  — defaults first. Most important first.
    ("extended_window", [True, False]),       # ⭐ v2.35.3 vs v2.36.0
    ("use_revin", [True, False]),             # ⭐ RC1 mitigation
    ("output_activation", ["linear", "softplus"]),
    ("daily_loss_weight", [0.0, 1.0]),
    ("use_solar_covariates", [True, False]),
    ("window_size", [48, 24, 96]),
    ("horizon", [48, 24, 96]),
    ("optimiser", ["adamw", "adam"]),
]

BACKENDS = ["nlinear", "sparsetsf", "lstm", "cnn"]


def _per_run(name: str, dataset, cfg: HarnessCfg) -> Dict[str, Dict[str, float]]:
    t0 = time.time()
    res = run_neural_and_tree(dataset.df, cfg, backends=BACKENDS,
                              train_subset_days=60, n_eval_windows=8)
    out: Dict[str, Dict[str, float]] = {}
    for b in BACKENDS:
        r = res.get(b)
        if r is None or r.pred.size == 0:
            out[b] = {"mae": float("nan"), "flatness": float("nan"),
                      "night_mae": float("nan"), "day_mae": float("nan")}
            continue
        # quick stats
        rows = []
        for w in range(len(r.truth_idx)):
            for h in range(r.pred.shape[1]):
                rows.append({"hour": r.truth_idx[w][h].hour,
                             "pred": r.pred[w, h],
                             "truth": r.truth[w, h]})
        df = pd.DataFrame(rows)
        night = df["hour"].isin([0, 1, 2, 3, 4, 22, 23])
        day = df["hour"].isin([10, 11, 12, 13, 14])
        pred_std = float(r.pred.std(axis=1).mean())
        truth_std = float(r.truth.std(axis=1).mean())
        out[b] = {
            "mae": r.mae,
            "flatness": pred_std / max(1e-9, truth_std),
            "night_mae": float((df.loc[night, "pred"] - df.loc[night, "truth"]).abs().mean()),
            "day_mae": float((df.loc[day, "pred"] - df.loc[day, "truth"]).abs().mean()),
        }
    print(f"  [{name}] elapsed {time.time() - t0:.1f}s")
    return out


def _plot_axis(dataset_name: str, axis_name: str,
               values: List[Any], result_per_value: List[Dict[str, Dict[str, float]]]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # MAE bar chart per backend across values
    width = 0.18
    x = np.arange(len(values))
    for i, b in enumerate(BACKENDS):
        maes = [r[b]["mae"] for r in result_per_value]
        axes[0].bar(x + i * width, maes, width, label=b)
    axes[0].set_xticks(x + 1.5 * width)
    axes[0].set_xticklabels([str(v) for v in values])
    axes[0].set_xlabel(axis_name)
    axes[0].set_ylabel("MAE")
    axes[0].set_title(f"{dataset_name} — MAE by {axis_name}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    for i, b in enumerate(BACKENDS):
        flats = [r[b]["flatness"] for r in result_per_value]
        axes[1].bar(x + i * width, flats, width, label=b)
    axes[1].axhline(1.0, color="black", linestyle="--", alpha=0.5, label="ideal=1")
    axes[1].set_xticks(x + 1.5 * width)
    axes[1].set_xticklabels([str(v) for v in values])
    axes[1].set_xlabel(axis_name)
    axes[1].set_ylabel("flatness (pred_std / truth_std)")
    axes[1].set_title(f"{dataset_name} — flatness by {axis_name}")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{dataset_name}__{axis_name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    from tests.synthetic.datasets import make_realistic_pv
    datasets = [
        make_realistic_pv(0),  # closest to production target
        make_ev_mixergy(0),    # LSTM/CNN already collapse in phase 1
    ]
    summary: Dict[str, Dict[str, List[Dict[str, Dict[str, float]]]]] = {}
    base_cfg = HarnessCfg(epochs=18, holdout_days=7)
    for ds in datasets:
        print(f"\n=== Dataset: {ds.name} ===")
        summary[ds.name] = {}
        for axis_name, values in AXES:
            print(f" axis {axis_name} values={values}")
            results_for_axis: List[Dict[str, Dict[str, float]]] = []
            for v in values:
                cfg = copy.deepcopy(base_cfg)
                # When sweeping window or horizon, ensure consistency:
                # set both to v if axis is window_size, else leave horizon
                # = base unless axis IS horizon.
                setattr(cfg, axis_name, v)
                # When extended_window=False the future-features path is
                # skipped (mirrors v2.35.x). Note v2.35.x also had no
                # daily_loss_weight, but we keep it independent here.
                run_name = f"{axis_name}={v}"
                res = _per_run(run_name, ds, cfg)
                results_for_axis.append(res)
            summary[ds.name][axis_name] = [
                {"value": str(v), "results": r}
                for v, r in zip(values, results_for_axis)
            ]
            _plot_axis(ds.name, axis_name, values, results_for_axis)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {SUMMARY_PATH}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2].parent)
    main()
