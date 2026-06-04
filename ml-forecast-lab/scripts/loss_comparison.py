#!/usr/bin/env python3
"""Loss-function side-by-side for daily-total forecasting.

Demonstrates, on the ACTUAL neural loss code path
(``ForecastModel._composite_horizon_loss`` via the NLinear backend),
how the choice of per-interval loss and the per-interval⟷cumulative
``loss_balance`` (α) affect:

    * per-interval accuracy (MAE)
    * per-interval *bias*       (mean signed error — the thing that
                                 accumulates into the daily total)
    * daily-total accuracy (MAE on Σ over a day)
    * daily-total *bias*        (Σ of the per-interval bias)

Thesis (see the loss discussion): the daily-total error is the SUM of
the signed per-interval errors, so

    daily_error ≈ H · (per-interval bias)  +  √H · (per-interval noise)

Median-seeking losses (Huber / MAE) under-predict each interval on a
right-skewed target; those small biases SUM into a large daily-total
shortfall. MSE seeks the conditional mean → unbiased per interval →
unbiased daily total. A modest cumulative blend (α≈0.3-0.5) then
defends the total against correlated drift without the under-
constrained null space of pure-cumulative (α=1).

Usage
-----
    # Synthetic Mixergy-like demand (zero-inflated, right-skewed,
    # morning+evening peaks, 30-min grid):
    python scripts/loss_comparison.py

    # Your own export — a CSV with a datetime column and a
    # per-interval demand column (NOT cumulative; diff it first if
    # your sensor is a _today counter):
    python scripts/loss_comparison.py --csv mixergy.csv \
        --time-col datetime --value-col demand --interval-min 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the package importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def synth_demand(
    days: int = 90, interval_min: int = 30, seed: int = 7,
    gamma_shape: float = 1.6,
) -> np.ndarray:
    """Mixergy-like per-interval hot-water demand.

    Zero-inflated (long quiet periods), right-skewed (occasional big
    draws), with a morning and an evening peak in draw *probability*.
    Returns a 1-D array of per-interval demand on a regular grid.
    """
    rng = np.random.default_rng(seed)
    per_day = 24 * 60 // interval_min
    n = days * per_day
    # Hour-of-day draw intensity: quiet overnight, morning + evening peaks.
    hours = (np.arange(n) % per_day) * (interval_min / 60.0)
    morning = np.exp(-0.5 * ((hours - 7.0) / 1.3) ** 2)
    evening = np.exp(-0.5 * ((hours - 19.0) / 1.8) ** 2)
    intensity = 0.02 + 0.9 * morning + 0.7 * evening  # P(draw) per bin
    draws = rng.random(n) < (intensity / intensity.max() * 0.35)
    # Draw magnitude: Gamma → right-skewed (median << mean). Lower
    # shape = heavier skew (shape→1 is exponential, mean/median≈1.44).
    mag = rng.gamma(shape=gamma_shape, scale=2.2, size=n)
    demand = np.where(draws, mag, 0.0).astype(np.float32)
    # A little measurement noise on the non-zero draws.
    demand[draws] += rng.normal(0, 0.15, draws.sum()).astype(np.float32)
    return np.clip(demand, 0.0, None)


def build_windows(series: np.ndarray, window: int, horizon: int):
    """Sliding windows: X (N, window, 1) past demand → y (N, horizon)
    next-day per-interval demand. With horizon = one day, each y-row is
    a full day, so its sum is the daily total."""
    n = len(series)
    N = n - window - horizon + 1
    if N <= 0:
        raise ValueError("series too short for window+horizon")
    X = np.empty((N, window, 1), dtype=np.float32)
    y = np.empty((N, horizon), dtype=np.float32)
    for i in range(N):
        X[i, :, 0] = series[i:i + window]
        y[i] = series[i + window:i + window + horizon]
    return X, y


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-interval and daily-total accuracy + bias.

    y_true / y_pred: (N, H) where H is one day's worth of intervals,
    so axis-1 sum is the daily total.
    """
    err = y_pred - y_true                       # signed per-interval error
    interval_mae = float(np.mean(np.abs(err)))
    interval_bias = float(np.mean(err))         # mean signed → the accumulator
    daily_true = y_true.sum(axis=1)
    daily_pred = y_pred.sum(axis=1)
    daily_err = daily_pred - daily_true
    daily_mae = float(np.mean(np.abs(daily_err)))
    daily_bias = float(np.mean(daily_err))
    daily_pct = (daily_mae / max(np.mean(daily_true), 1e-9)) * 100.0
    return {
        "interval_mae": interval_mae,
        "interval_bias": interval_bias,
        "daily_mae": daily_mae,
        "daily_bias": daily_bias,
        "daily_pct": daily_pct,
        "mean_daily_actual": float(np.mean(daily_true)),
    }


def run_config(loss_fn: str, alpha: float, folds, window, horizon,
               epochs: int, activation: str = "softplus") -> dict:
    """Train NLinear under (loss_fn, α) on each walk-forward fold and
    average the held-out metrics."""
    from ml_forecast_lab.models.nlinear_backend import NLinearModel

    accum: list = []
    for (Xtr, ytr, Xte, yte) in folds:
        model = NLinearModel(
            epochs=epochs, patience=12, learning_rate=1e-3,
            loss_fn=loss_fn, output_activation=activation,
            use_revin=False,
        )
        # α=None → legacy interval-only; a float engages the convex
        # blend in _composite_horizon_loss. This is the exact attribute
        # _apply_loss_balance sets in production.
        model.loss_balance = None if alpha == 0.0 else float(alpha)
        model._loss_ema = None
        # X_flat is unused for sequence models but required positionally.
        x_flat = np.zeros((len(ytr), window), dtype=np.float32)
        model.fit(x_flat, ytr, sequence_data=Xtr)
        pred = model.predict_sequence(Xte)
        accum.append(evaluate(yte, pred))

    # Average across folds.
    keys = accum[0].keys()
    return {k: float(np.mean([a[k] for a in accum])) for k in keys}


def walk_forward(X, y, n_folds=3, min_train_frac=0.5):
    """Expanding-window walk-forward folds."""
    N = len(X)
    first = int(N * min_train_frac)
    fold_size = (N - first) // n_folds
    folds = []
    for k in range(n_folds):
        tr_end = first + k * fold_size
        te_end = tr_end + fold_size if k < n_folds - 1 else N
        folds.append((X[:tr_end], y[:tr_end], X[tr_end:te_end], y[tr_end:te_end]))
    return folds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=str, default=None,
                    help="CSV of per-interval demand (not cumulative).")
    ap.add_argument("--time-col", type=str, default="datetime")
    ap.add_argument("--value-col", type=str, default="demand")
    ap.add_argument("--interval-min", type=int, default=30)
    ap.add_argument("--days", type=int, default=90,
                    help="Synthetic-data length (ignored with --csv).")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--activation", type=str, default="softplus",
                    help="Neural output activation (softplus floors at "
                         "0.69 → confounds zero-inflated bias; 'linear' "
                         "or 'relu' allow true zero).")
    ap.add_argument("--gamma-shape", type=float, default=1.6,
                    help="Synthetic draw skew; lower = heavier right tail.")
    ap.add_argument("--losses", type=str, default="huber,mse",
                    help="Comma-separated loss functions to compare.")
    ap.add_argument("--alphas", type=str, default="0.0,0.5",
                    help="Comma-separated loss_balance α values to sweep.")
    args = ap.parse_args()

    per_day = 24 * 60 // args.interval_min
    horizon = per_day            # predict one full day → sum = daily total
    window = per_day * 3         # 3 days of history

    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        series = df[args.value_col].to_numpy(dtype=np.float32)
        source = f"{args.csv} ({len(series)} rows)"
    else:
        series = synth_demand(days=args.days, interval_min=args.interval_min,
                              seed=args.seed, gamma_shape=args.gamma_shape)
        source = (f"synthetic Mixergy-like demand, {args.days}d @ "
                  f"{args.interval_min}min, gamma_shape={args.gamma_shape}")

    # Distribution sanity — the whole argument rests on right-skew.
    nz = series[series > 1e-6]
    skew_note = ""
    if len(nz):
        med, mean = float(np.median(nz)), float(np.mean(nz))
        skew_note = (f"non-zero draws: median={med:.2f}, mean={mean:.2f} "
                     f"(mean/median={mean / max(med, 1e-9):.2f}× → "
                     f"{'right-skewed' if mean > med * 1.1 else 'symmetric'}); "
                     f"zero-inflation={100 * (1 - len(nz) / len(series)):.0f}%")

    X, y = build_windows(series, window, horizon)
    folds = walk_forward(X, y, n_folds=args.folds)

    loss_list = [s.strip() for s in args.losses.split(",") if s.strip()]
    alpha_list = [float(s) for s in args.alphas.split(",") if s.strip()]
    configs = [(lf, a) for lf in loss_list for a in alpha_list]
    print(f"\nData: {source}")
    if skew_note:
        print(f"      {skew_note}")
    print(f"Windows: N={len(X)}, window={window}, horizon={horizon} "
          f"(=1 day), folds={args.folds}, epochs={args.epochs}, "
          f"activation={args.activation}")
    print(f"Mean actual daily total ≈ {float(np.mean(y.sum(axis=1))):.1f}\n")

    header = (f"{'loss':>6} {'α':>4} | {'int MAE':>8} {'int bias':>9} | "
              f"{'daily MAE':>10} {'daily bias':>11} {'daily %':>8}")
    print(header)
    print("-" * len(header))
    results = {}
    for loss_fn, alpha in configs:
        r = run_config(loss_fn, alpha, folds, window, horizon, args.epochs,
                       activation=args.activation)
        results[(loss_fn, alpha)] = r
        print(f"{loss_fn:>6} {alpha:>4.1f} | "
              f"{r['interval_mae']:>8.3f} {r['interval_bias']:>+9.3f} | "
              f"{r['daily_mae']:>10.2f} {r['daily_bias']:>+11.2f} "
              f"{r['daily_pct']:>7.1f}%")

    # Headline takeaways, computed not asserted. Robust to whatever
    # --losses / --alphas grid was actually run.
    print("\nReadout:")
    # Per-interval bias drives the daily total; surface the lowest-
    # |bias| config and the best daily-MAE config (often the same).
    least_bias = min(results.items(), key=lambda kv: abs(kv[1]["interval_bias"]))
    print(f"  • Lowest per-interval |bias|: {least_bias[0][0]} "
          f"α={least_bias[0][1]} → bias {least_bias[1]['interval_bias']:+.3f} "
          f"× {horizon} = daily bias {least_bias[1]['daily_bias']:+.1f}")
    best = min(results.items(), key=lambda kv: kv[1]["daily_mae"])
    print(f"  • Best daily MAE: {best[0][0]} α={best[0][1]} "
          f"→ {best[1]['daily_mae']:.2f} ({best[1]['daily_pct']:.0f}%)")
    # If α was swept, report whether any cumulative weight beat α=0.
    alphas_run = sorted({a for (_, a) in results})
    if 0.0 in alphas_run and len(alphas_run) > 1:
        for lf in sorted({lf for (lf, _) in results}):
            base = results.get((lf, 0.0))
            if base is None:
                continue
            better = [a for a in alphas_run if a > 0.0
                      and results[(lf, a)]["daily_mae"] < base["daily_mae"]]
            verdict = (f"α={better} beat α=0" if better
                       else "no α>0 beat pure per-interval (α=0)")
            print(f"  • {lf}: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
