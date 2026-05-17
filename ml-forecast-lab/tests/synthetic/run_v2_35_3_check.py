"""
Quick test of the v2.35.3 code path (extended_window=False) — does the
3 AM spurious peak symptom appear on synthetic PV?

Per the brief, v2.35.3 had:
  - bell-shape forecast but with a spurious peak at ~3 AM (sun physically
    below horizon)

If this test shows mean-by-hour with a non-trivial value at 03:00 UTC,
the symptom reproduces. If 03:00 is essentially zero, the v2.35.3
symptom is data-dependent (different from the user's PV target).
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.synthetic.datasets import make_realistic_pv
from tests.synthetic.harness import HarnessCfg, run_neural_and_tree

logging.basicConfig(level=logging.WARNING)

OUT_PNG = Path("docs/investigations/figures_phase1/v2_35_3_check.png")
OUT_MD = Path("docs/investigations/v2_35_3_check.md")


def main():
    ds = make_realistic_pv(0)
    cfg = HarnessCfg(
        extended_window=False,    # v2.35.3 path
        use_revin=True,
        epochs=20,
        holdout_days=10,
        window_size=48,
        horizon=48,
    )
    backends = ["nlinear", "sparsetsf", "lstm", "cnn"]
    results = run_neural_and_tree(ds.df, cfg, backends=backends,
                                   train_subset_days=60, n_eval_windows=12)
    lines = []
    lines.append("# v2.35.3 code path check (extended_window=False)\n\n")
    lines.append("Dataset: `realistic_pv` (~4500 W peak).\n\n")
    lines.append("|backend|mean@03|mean@12|mean@18|peak_hour|note|\n|---|---:|---:|---:|---:|---|\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    sample = next(iter(results.values()))
    # Mean truth by hour
    H = sample.pred.shape[1]
    rows = []
    for w in range(len(sample.truth_idx)):
        for h in range(H):
            rows.append({"hour": sample.truth_idx[w][h].hour,
                         "truth": sample.truth[w, h]})
    truth_by_hour = pd.DataFrame(rows).groupby("hour")["truth"].mean()
    ax.plot(truth_by_hour.index, truth_by_hour.values, "k--",
            linewidth=2.5, label="truth")
    for name, r in results.items():
        if r.pred.size == 0:
            lines.append(f"|{name}|FAILED|FAILED|FAILED|n/a|{r.note[:40]}|\n")
            continue
        rows = []
        for w in range(len(r.truth_idx)):
            for h in range(H):
                rows.append({"hour": r.truth_idx[w][h].hour,
                             "pred": r.pred[w, h]})
        s = pd.DataFrame(rows).groupby("hour")["pred"].mean()
        peak = int(s.idxmax())
        mean3 = float(s.get(3, np.nan))
        mean12 = float(s.get(12, np.nan))
        mean18 = float(s.get(18, np.nan))
        if mean3 > 100.0:
            note = "**3 AM PEAK reproduced**"
        elif mean3 > 20.0:
            note = "small 3 AM bias"
        else:
            note = "ok at 03:00"
        lines.append(f"|{name}|{mean3:.1f}|{mean12:.1f}|{mean18:.1f}|{peak}|{note}|\n")
        ax.plot(s.index, s.values, marker="o", linewidth=1.4, label=name)
        print(f"{name:>10}  mean@03={mean3:7.1f}  mean@12={mean12:7.1f}  peak={peak}")
    ax.set_xlabel("hour of day (UTC)")
    ax.set_ylabel("y")
    ax.set_title("v2.35.3 code path (extended_window=False) — mean forecast by hour")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    OUT_MD.write_text("".join(lines))
    print(f"wrote {OUT_MD} and {OUT_PNG}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2].parent)
    main()
