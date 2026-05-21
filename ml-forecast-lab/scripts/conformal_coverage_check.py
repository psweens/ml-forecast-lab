#!/usr/bin/env python3
"""Offline conformal-coverage diagnostic.

Reads an ML Forecast Lab history.db dump and reports empirical coverage
of the published 80% bands across the breakdowns that are most likely
to reveal exchangeability violations: overall, per-lead, per-hour-of-day
(in local TZ), and weekday-vs-weekend.

The split-CP scheme described in
``ml_forecast_lab.db.get_conformal_quantiles`` pools residuals across
all of those dimensions, so if the realised coverage diverges
materially (e.g. >5pp from nominal) on any specific bucket — say,
weekday evenings — that's the signal that exchangeability is failing
on that bucket and a more adaptive method (ACI, weighted CP) would
help.

Usage::

    python scripts/conformal_coverage_check.py \\
        --db /data/ml_forecast_lab/history.db \\
        --experiment pv_forecast \\
        --target-table sensor_pv_power \\
        --days 30 \\
        --tz Europe/London

This is a read-only diagnostic — it does not modify the DB.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path,
                        help="Path to history.db")
    parser.add_argument("--experiment", required=True,
                        help="Experiment name in forecast_log")
    parser.add_argument("--target-table", required=True,
                        help="Sanitised target sensor table name (e.g. sensor_pv_power)")
    parser.add_argument("--days", type=int, default=30,
                        help="Lookback window in days (default 30)")
    parser.add_argument("--interval-minutes", type=int, default=30,
                        help="Forecast interval in minutes (default 30)")
    parser.add_argument("--model-name", default=None,
                        help="Restrict to one model (default: dominant cohort)")
    parser.add_argument("--model-version", default=None,
                        help="Restrict to one model version")
    parser.add_argument("--tz", default=None,
                        help="Local time zone (e.g. Europe/London) for hour-of-day breakdown. "
                             "Default: UTC.")
    parser.add_argument("--nominal", type=float, default=0.8,
                        help="Nominal coverage level the bands were calibrated at (default 0.8)")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 2

    # Import lazily so the script can be invoked without the full
    # package's runtime dependencies (HA client etc.) being importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ml_forecast_lab.db import HistoryDB

    db = HistoryDB(args.db)
    cov = db.get_forecast_coverage(
        experiment=args.experiment,
        actuals_table=args.target_table,
        interval_minutes=args.interval_minutes,
        max_age_days=args.days,
        model_name=args.model_name,
        model_version=args.model_version,
        tz=args.tz,
    )

    nominal = args.nominal
    overall = cov.get("overall") or {}
    print("=" * 60)
    print(f"Conformal coverage diagnostic — {args.experiment}")
    print(f"  Lookback: {args.days} days")
    print(f"  Nominal level: {nominal*100:.0f}%")
    print(f"  TZ for hour-of-day: {cov.get('tz', 'UTC')}")
    print("=" * 60)
    print()

    if not overall:
        print("No interval forecasts in this window. Nothing to diagnose.")
        print()
        print("Common causes:")
        print(" - The experiment hasn't published _upper_/_lower_ sensors yet")
        print("   (cold-start: needs ~10 residuals before bands appear)")
        print(" - Wrong --target-table — must match HistoryDB.safe_table_name(<target_entity>)")
        print(" - --days too small — try increasing.")
        return 1

    overall_cov = overall["coverage"] * 100
    overall_n = overall["n"]
    delta_pp = overall_cov - nominal * 100
    verdict = "ON TARGET" if abs(delta_pp) < 5 else (
        "WIDE (bands too conservative)" if delta_pp > 0 else "TIGHT (bands miscoverage)"
    )
    print(f"Overall: {overall_cov:.1f}% (n={overall_n}) — {verdict} "
          f"[{'+' if delta_pp >= 0 else ''}{delta_pp:.1f} pp vs nominal]")
    print()

    def _print_breakdown(title: str, container: dict, key: str, fmt) -> None:
        rows = list(zip(container.get(key, []), container.get("coverage", []), container.get("n", [])))
        if not rows:
            return
        print(title)
        print(f"  {'bucket':<20} {'coverage':>10} {'n':>8} {'Δ vs nominal':>15}")
        for v, c, n in rows:
            dpp = c * 100 - nominal * 100
            flag = " *" if abs(dpp) >= 5 and n >= 20 else ""
            print(f"  {fmt(v):<20} {c*100:>9.1f}% {n:>8d} {'+' if dpp >= 0 else ''}{dpp:>10.1f} pp{flag}")
        print()

    _print_breakdown(
        "By hour of day (local TZ):",
        cov.get("by_hour_of_day", {}), "hour", lambda h: f"hour {int(h):02d}",
    )
    _print_breakdown(
        "By weekday vs weekend:",
        cov.get("by_weekday_weekend", {}), "bucket", str,
    )
    _print_breakdown(
        "By lead time:",
        cov.get("by_lead", {}), "lead_minutes", lambda v: f"+{int(v)} min",
    )

    worst = cov.get("worst_bucket")
    if worst:
        print(f"Worst bucket: {worst['kind']}={worst['label']}: "
              f"{worst['coverage']*100:.1f}% (n={worst['n']})")
        print()

    print("Notes:")
    print(" * = bucket deviates from nominal by ≥5 pp with n≥20 — material miscalibration.")
    print()
    print("Interpretation: if specific hour-of-day or weekday/weekend buckets are")
    print("flagged while the overall coverage looks fine, the split-CP residual buffer")
    print("is pooling regimes that aren't exchangeable. The pragmatic fixes are (in order):")
    print("  1. Increase the residual-buffer window (max_age_days) so each regime has")
    print("     enough samples to influence the quantile.")
    print("  2. Switch to adaptive conformal (ACI / Gibbs-Candès 2021) which updates")
    print("     the effective level online and degrades less in non-exchangeable settings.")
    print("  3. Split the calibration set by regime (e.g. separate weekday and weekend")
    print("     quantiles) — possible but adds buffer-warmup time per regime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
