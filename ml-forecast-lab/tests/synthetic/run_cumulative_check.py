"""
Verify that PF1-PF9 work on cumulative-with-daily-reset targets.

The production code path for these targets is:
    raw_cumulative_sensor → cumulative_to_interval → train on intervals
                                                  → predict intervals
                                                  → re-cumsum for display

We exercise the TRAINING path (intervals) here because that's the shape
the model actually sees. The test:

1. Generates ``make_cumulative_daily_reset(0)`` — a cumulative+reset
   dataset whose ``y_interval`` column is the interval form.
2. Trains each registered neural backend on the interval form with
   ``target_is_nonnegative=True`` set on the harness config (which
   triggers PF8 softplus + PF9 daily_loss_weight=0.5).
3. Re-cumsums the predictions per holdout window and compares against
   the truth's cumulative trajectory to confirm the daily reset is
   preserved.
4. Records per-backend verdict (OK / COLLAPSED / PHASE_OFF / HEAVILY_BROKEN).

Outputs:
    docs/investigations/cumulative_summary.md
    docs/investigations/cumulative_summary.json
    docs/investigations/figures_phase1/cumulative_realistic.png
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.synthetic.datasets import make_cumulative_daily_reset
from tests.synthetic.harness import HarnessCfg
from tests.synthetic.run_phase1b_all_backends import (
    _all_backends, _classify, _diag,
)
from tests.synthetic.harness import (
    _train_neural_backend, _rolling_evaluate_neural,
)

logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

OUT_PNG = Path("docs/investigations/figures_phase1/cumulative_realistic.png")
OUT_MD = Path("docs/investigations/cumulative_summary.md")
OUT_JSON = Path("docs/investigations/cumulative_summary.json")


def main():
    ds = make_cumulative_daily_reset(0)
    # Train on the INTERVAL form — that's what production does after
    # cumulative_to_interval. The 'y' column in the harness must be the
    # interval form; the cumulative is used for display only.
    df_interval = ds.df[["y_interval"]].rename(columns={"y_interval": "y"})
    cfg = HarnessCfg(
        epochs=40, holdout_days=10, window_size=48, horizon=48,
        # No solar covariates — this is a household-demand target,
        # nothing solar about it.
        use_solar_covariates=False,
        # PF8 (softplus) + PF9 (daily_loss_weight=0.5) — the production
        # path resolves these automatically when target_is_nonnegative=True
        # or source_is_cumulative=True.
        target_is_nonnegative=True,
    )
    cov_cols: List[str] = []
    steps_per_day = 1440 // cfg.interval_minutes
    holdout_start = len(df_interval) - cfg.holdout_days * steps_per_day
    df_train = df_interval.iloc[holdout_start - 60 * steps_per_day: holdout_start]
    backends = _all_backends()
    summary = []
    for name, cls in backends:
        print(f"== {name} ==", flush=True)
        try:
            t0 = time.time()
            model, ch_names, past_ws = _train_neural_backend(
                cls, df_train, cfg, cov_cols,
            )
            res = _rolling_evaluate_neural(
                model, df_interval, cfg, cov_cols, past_ws,
                holdout_start_idx=holdout_start, n_eval_windows=10,
                backend_name=name,
            )
            d = _diag(name, res)
            d["fit_seconds"] = float(time.time() - t0)
            d["_result"] = res
            summary.append(d)
            print(f"  {d['verdict']:>15}  mae={d['mae']:.4f}  "
                  f"peak={d['peak_pred']:>2} (truth={d['peak_truth']})  "
                  f"flatness={d['flatness']:.2f}  "
                  f"daily_total_mape={_daily_total_mape(res):.1%}  "
                  f"({d['fit_seconds']:.1f}s)", flush=True)
        except Exception as e:
            print(f"  FAILED: {e!r}")
            summary.append({"name": name, "verdict": "FAILED",
                            "mae": float("nan"), "flatness": float("nan"),
                            "note": repr(e)[:120]})
    _write_md(summary)
    _write_json(summary)
    _plot(summary)


def _daily_total_mape(r) -> float:
    """How well does the model's interval forecast cumulate to match truth?

    Returns mean absolute percent error of the predicted day-total
    versus the actual day-total, averaged across holdout windows.
    Smaller is better; for a cumulative-target user this is the most
    important metric — the addon re-cumsums interval predictions to
    publish a cumulative forecast.
    """
    if r.pred.size == 0:
        return float("nan")
    pred_totals = r.pred.sum(axis=1)
    truth_totals = r.truth.sum(axis=1)
    return float(np.mean(np.abs(pred_totals - truth_totals)
                          / np.maximum(np.abs(truth_totals), 1e-6)))


def _write_md(summary):
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# All neural backends on cumulative-with-daily-reset target\n\n"]
    lines.append("Dataset: ``make_cumulative_daily_reset(0)`` — interval form ")
    lines.append("(per-30min household demand), trained with ")
    lines.append("``target_is_nonnegative`` implied for the harness. ")
    lines.append("True peak interval hour: 19 (evening). Reset at midnight. ")
    lines.append("Ideal flatness ≈ 1.0.\n\n")
    lines.append("``daily_total_mape`` = how far the model's predicted ")
    lines.append("day-total (sum of intervals over 24h) is from truth. ")
    lines.append("This is the cumulative-target user's key metric because ")
    lines.append("the addon re-cumsums interval predictions for display.\n\n")
    lines.append("|backend|verdict|peak_truth|peak_pred|flatness|MAE|"
                 "daily_total_mape|\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|\n")
    for d in summary:
        if "_result" not in d:
            lines.append(f"|{d['name']}|{d['verdict']}|—|—|—|—|—|\n")
            continue
        r = d["_result"]
        dt_mape = _daily_total_mape(r)
        try:
            pt = int(d.get("peak_truth", -1))
            pp = int(d.get("peak_pred", -1))
        except Exception:
            pt = pp = "n/a"
        lines.append(
            f"|{d['name']}|{d['verdict']}|{pt}|{pp}|"
            f"{d.get('flatness', float('nan')):.2f}|"
            f"{d.get('mae', float('nan')):.4f}|"
            f"{dt_mape:.1%}|\n"
        )
    OUT_MD.write_text("".join(lines))
    print(f"\nwrote {OUT_MD}")


def _write_json(summary):
    j_clean = []
    for d in summary:
        x = {k: v for k, v in d.items() if k != "_result"}
        for k, v in list(x.items()):
            if isinstance(v, np.floating):
                x[k] = float(v)
            elif isinstance(v, np.integer):
                x[k] = int(v)
        if "_result" in d and d["_result"].pred.size > 0:
            x["daily_total_mape"] = _daily_total_mape(d["_result"])
        j_clean.append(x)
    OUT_JSON.write_text(json.dumps(j_clean, indent=2))
    print(f"wrote {OUT_JSON}")


def _plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    successes = [d for d in summary if "_result" in d and d["_result"].pred.size > 0]
    if not successes:
        return
    sample = successes[0]["_result"]
    H = sample.pred.shape[1]
    truth_rows = [
        {"hour": sample.truth_idx[w][h].hour, "truth": sample.truth[w, h]}
        for w in range(len(sample.truth_idx)) for h in range(H)
    ]
    truth_curve = pd.DataFrame(truth_rows).groupby("hour")["truth"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Left: interval predictions by hour-of-day
    ax = axes[0]
    ax.plot(truth_curve.index, truth_curve.values, "k--",
            linewidth=2.5, label="truth")
    for d in successes:
        r = d["_result"]
        rows = [{"hour": r.truth_idx[w][h].hour, "pred": r.pred[w, h]}
                for w in range(len(r.truth_idx)) for h in range(H)]
        s = pd.DataFrame(rows).groupby("hour")["pred"].mean()
        ax.plot(s.index, s.values, marker=".", linewidth=1.0,
                label=f"{d['name']} ({d['verdict']})")
    ax.set_xlabel("hour of day (UTC)")
    ax.set_ylabel("y_interval")
    ax.set_title("Interval prediction by hour-of-day — household demand")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    # Right: re-cumsum-ed curve for a single holdout window
    ax = axes[1]
    truth_cum = sample.truth[0].cumsum()
    ax.plot(np.arange(H), truth_cum, "k--", linewidth=2.5, label="truth")
    for d in successes:
        r = d["_result"]
        cum = r.pred[0].cumsum()
        ax.plot(np.arange(H), cum, linewidth=1.0,
                label=f"{d['name']} ({d['verdict']})")
    ax.set_xlabel("horizon step (30-min)")
    ax.set_ylabel("cumulative y over 24 h")
    starts = sample.truth_idx[0][0].strftime("%Y-%m-%d %H:%M")
    ax.set_title(f"Re-cumsum-ed forecast — window starting {starts}")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2].parent)
    main()
