"""
Phase 1B — all neural backends checked for RC1 / RC2 / RC3 patterns.

Runs the harness on `make_realistic_pv(0)` for every multi-horizon
neural backend with default Settings + extended_window=True. Records:

  - overall MAE
  - per-hour-of-day MAE  → tells us about phase
  - mean predicted curve peak hour vs truth peak hour  → phase inversion
  - flatness (pred_std_over_horizon / truth_std_over_horizon)
        ~ 1.0  → correct amplitude
        << 1.0 → collapsed to mean
        >> 1.0 → over-varies (NLinear-style)

Then prints a per-backend table classifying each into:

  OK              shape and amplitude correct (peak hour within ±1, flatness 0.5..1.5)
  PHASE_OFF       peak hour off by >1 h
  COLLAPSED       flatness < 0.3 (predictions essentially constant)
  OVER_VARY       flatness > 1.5 (predictions overshoot)
  HEAVILY_BROKEN  peak off AND flatness off
  IMPORT_ERR      backend not available in this environment

Output: docs/investigations/figures_phase1/all_neural_realistic_pv.png
        docs/investigations/all_neural_summary.md
        docs/investigations/all_neural_summary.json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.synthetic.datasets import make_realistic_pv
from tests.synthetic.harness import HarnessCfg, _train_neural_backend, _rolling_evaluate_neural

logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

OUT_PNG = Path("docs/investigations/figures_phase1/all_neural_realistic_pv.png")
OUT_MD = Path("docs/investigations/all_neural_summary.md")
OUT_JSON = Path("docs/investigations/all_neural_summary.json")


def _classify(peak_truth: int, peak_pred: int, flatness: float) -> str:
    if not np.isfinite(flatness):
        return "FAILED"
    peak_diff = min(abs(peak_pred - peak_truth), 24 - abs(peak_pred - peak_truth))
    phase_off = peak_diff > 1
    collapsed = flatness < 0.3
    over_vary = flatness > 1.5
    if phase_off and (collapsed or over_vary):
        return "HEAVILY_BROKEN"
    if phase_off:
        return "PHASE_OFF"
    if collapsed:
        return "COLLAPSED"
    if over_vary:
        return "OVER_VARY"
    return "OK"


def _diag(name: str, r) -> Dict[str, Any]:
    if r.pred.size == 0:
        return {"name": name, "verdict": "FAILED", "mae": float("nan"),
                "peak_truth": float("nan"), "peak_pred": float("nan"),
                "flatness": float("nan"), "night_mae": float("nan"),
                "day_mae": float("nan"), "note": r.note}
    rows = []
    for w in range(len(r.truth_idx)):
        for h in range(r.pred.shape[1]):
            rows.append({"hour": r.truth_idx[w][h].hour,
                         "pred": r.pred[w, h], "truth": r.truth[w, h]})
    df = pd.DataFrame(rows)
    by_hour = df.groupby("hour").mean()
    peak_truth = int(by_hour["truth"].idxmax())
    peak_pred = int(by_hour["pred"].idxmax())
    flatness = float(r.pred.std(axis=1).mean() / max(1e-6, r.truth.std(axis=1).mean()))
    night = df["hour"].isin([0, 1, 2, 3, 4, 22, 23])
    day = df["hour"].isin([10, 11, 12, 13, 14])
    night_mae = float((df.loc[night, "pred"] - df.loc[night, "truth"]).abs().mean())
    day_mae = float((df.loc[day, "pred"] - df.loc[day, "truth"]).abs().mean())
    return {
        "name": name, "mae": r.mae, "peak_truth": peak_truth, "peak_pred": peak_pred,
        "flatness": flatness, "night_mae": night_mae, "day_mae": day_mae,
        "verdict": _classify(peak_truth, peak_pred, flatness),
        "note": "",
    }


def _all_backends() -> List[Tuple[str, type]]:
    out = []
    catalog = [
        ("nlinear", "nlinear_backend", "NLinearModel"),
        ("dlinear", "dlinear_backend", "DLinearModel"),
        ("sparsetsf", "sparsetsf_backend", "SparseTSFModel"),
        ("fits", "fits_backend", "FITSModel"),
        ("tsmixer", "tsmixer_backend", "TSMixerModel"),
        ("timemixer", "timemixer_backend", "TimeMixerModel"),
        ("tide", "tide_backend", "TiDEModel"),
        ("lstm", "lstm_backend", "LSTMModel"),
        ("gru", "gru_backend", "GRUModel"),
        ("cnn", "cnn_backend", "CNNModel"),
        ("patchtst", "patchtst_backend", "PatchTSTModel"),
        ("itransformer", "itransformer_backend", "iTransformerModel"),
        ("crossformer", "crossformer_backend", "CrossformerModel"),
        ("timesnet", "timesnet_backend", "TimesNetModel"),
        ("tft", "tft_backend", "TFTModel"),
        ("nbeats", "nbeats_backend", "NBeatsModel"),
        ("nhits", "nhits_backend", "NHiTSModel"),
        ("timexer", "timexer_backend", "TimeXerModel"),
        ("moderntcn", "moderntcn_backend", "ModernTCNModel"),
        ("segrnn", "segrnn_backend", "SegRNNModel"),
        ("xpatch", "xpatch_backend", "XPatchModel"),
        # chronos_bolt / ttm are deliberately absent: zero-shot foundation
        # backends take no gradient steps, so the epochs-based harness has
        # nothing to exercise, and a benchmark run here would trigger a
        # Hugging Face weight download.
    ]
    for name, mod, cls_name in catalog:
        try:
            m = __import__(f"ml_forecast_lab.models.{mod}", fromlist=[cls_name])
            out.append((name, getattr(m, cls_name)))
        except Exception as e:
            print(f"  SKIP {name}: {e}")
    return out


def main():
    ds = make_realistic_pv(0)
    cfg = HarnessCfg(epochs=15, holdout_days=10, window_size=48, horizon=48,
                     extended_window=True, use_revin=True)
    cov_cols = ["sun_elevation", "clear_sky_ghi"]
    steps_per_day = 1440 // cfg.interval_minutes
    holdout_start = len(ds.df) - cfg.holdout_days * steps_per_day
    df_train = ds.df.iloc[holdout_start - 45 * steps_per_day: holdout_start]
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
                model, ds.df, cfg, cov_cols, past_ws,
                holdout_start_idx=holdout_start, n_eval_windows=8,
                backend_name=name,
            )
            d = _diag(name, res)
            d["fit_seconds"] = float(time.time() - t0)
            d["_result"] = res  # keep for plotting
            summary.append(d)
            print(f"  {d['verdict']:>15}  mae={d['mae']:8.2f}  peak={d['peak_pred']:>2} (truth={d['peak_truth']})  "
                  f"flatness={d['flatness']:.2f}  ({d['fit_seconds']:.1f}s)", flush=True)
        except Exception as e:
            print(f"  FAILED: {e!r}", flush=True)
            summary.append({"name": name, "verdict": "FAILED",
                            "mae": float("nan"), "flatness": float("nan"),
                            "note": repr(e)[:120]})
    # Build markdown
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# All neural backends on `realistic_pv` (v2.36.0 path)\n\n"]
    lines.append("Dataset: synthetic Watt-scale PV. extended_window=True, use_revin=True (defaults).\n\n")
    lines.append("Verdict: OK / PHASE_OFF (peak hour off >1h) / COLLAPSED (flat<0.3) / "
                 "OVER_VARY (flat>1.5) / HEAVILY_BROKEN (both) / FAILED.\n\n")
    lines.append("|backend|verdict|peak_truth|peak_pred|flatness|MAE|night MAE|day MAE|\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
    for d in summary:
        row = "|{name}|{verdict}|{pt}|{pp}|{f}|{m}|{nm}|{dm}|\n".format(
            name=d["name"], verdict=d["verdict"],
            pt=int(d.get("peak_truth", -1)) if np.isfinite(d.get("peak_truth", float("nan"))) else "n/a",
            pp=int(d.get("peak_pred", -1)) if np.isfinite(d.get("peak_pred", float("nan"))) else "n/a",
            f=f"{d.get('flatness', float('nan')):.2f}",
            m=f"{d.get('mae', float('nan')):.1f}",
            nm=f"{d.get('night_mae', float('nan')):.1f}" if "night_mae" in d else "—",
            dm=f"{d.get('day_mae', float('nan')):.1f}" if "day_mae" in d else "—",
        )
        lines.append(row)
    OUT_MD.write_text("".join(lines))
    print(f"\nwrote {OUT_MD}")
    # JSON (drop the result object first)
    j_clean = []
    for d in summary:
        x = {k: v for k, v in d.items() if k != "_result"}
        for k, v in list(x.items()):
            if isinstance(v, np.floating):
                x[k] = float(v)
            elif isinstance(v, np.integer):
                x[k] = int(v)
        j_clean.append(x)
    OUT_JSON.write_text(json.dumps(j_clean, indent=2))
    print(f"wrote {OUT_JSON}")
    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    successes = [d for d in summary if "_result" in d and d["_result"].pred.size > 0]
    if successes:
        fig, ax = plt.subplots(figsize=(14, 7))
        H = successes[0]["_result"].pred.shape[1]
        sample = successes[0]["_result"]
        truth_rows = []
        for w in range(len(sample.truth_idx)):
            for h in range(H):
                truth_rows.append({"hour": sample.truth_idx[w][h].hour,
                                    "truth": sample.truth[w, h]})
        truth_curve = pd.DataFrame(truth_rows).groupby("hour")["truth"].mean()
        ax.plot(truth_curve.index, truth_curve.values, "k--",
                linewidth=2.5, label="truth")
        for d in successes:
            r = d["_result"]
            rows = []
            for w in range(len(r.truth_idx)):
                for h in range(H):
                    rows.append({"hour": r.truth_idx[w][h].hour,
                                 "pred": r.pred[w, h]})
            s = pd.DataFrame(rows).groupby("hour")["pred"].mean()
            ax.plot(s.index, s.values, marker="o", linewidth=1.2,
                    label=f"{d['name']} ({d['verdict']})")
        ax.set_xlabel("hour of day (UTC)")
        ax.set_ylabel("y (W)")
        ax.set_title("All neural backends on realistic_pv — extended_window=True")
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2].parent)
    main()
