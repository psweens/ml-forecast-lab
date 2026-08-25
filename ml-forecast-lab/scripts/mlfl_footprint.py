#!/usr/bin/env python3
"""Report the on-disk and in-memory footprint of ML Forecast Lab.

Read-only. Safe to run against a live add-on — the database is opened in
read-only mode and nothing outside stdout is written.

Usage (inside the add-on container)::

    python3 mlfl_footprint.py

Usage (from the Home Assistant host, e.g. the SSH add-on)::

    python3 mlfl_footprint.py \
        --db /addon_configs/<slug>_ml_forecast_lab/mlfl.db \
        --data /addon_configs/<slug>_ml_forecast_lab/../ml_forecast_lab

Anything the paths do not resolve to is reported as "not found" and
skipped, so partial installs still produce a useful report.
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

# Tables that are not per-entity history caches. Everything else in the
# schema is a `sensor_*` / `binary_sensor_*` style cache table created by
# HistoryDB.ensure_table.
CORE_TABLES = {
    "schema_versions",
    "forecast_log",
    "external_forecast_log",
    "benchmark_results",
    "benchmark_history",
}


def mb(n):
    return f"{n / 1e6:,.1f} MB"


def rule(title):
    print(f"\n{title}")
    print("-" * len(title))


# ----------------------------------------------------------------------
# SQLite
# ----------------------------------------------------------------------

def report_db(path):
    rule(f"DATABASE  {path}")
    if not os.path.exists(path):
        print("  not found — pass --db with the correct location")
        return
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    page = conn.execute("PRAGMA page_size").fetchone()[0]
    total = conn.execute("PRAGMA page_count").fetchone()[0] * page
    free = conn.execute("PRAGMA freelist_count").fetchone()[0] * page

    # dbstat is compiled into most builds but not all; without it we can
    # still report row counts, just not per-table bytes.
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.dbs USING dbstat")
        sizes = dict(conn.execute("SELECT name, SUM(pgsize) FROM temp.dbs GROUP BY name"))
    except sqlite3.Error:
        sizes = {}

    wal = os.path.getsize(path + "-wal") if os.path.exists(path + "-wal") else 0
    print(f"  file            {mb(total)}")
    if wal:
        print(f"  -wal            {mb(wal)}")
    pct = (free / total * 100) if total else 0
    print(f"  reclaimable     {mb(free)}  ({pct:.0f}% — freed by DELETE, "
          f"returned to the filesystem only by VACUUM)")
    if not sizes:
        print("  note            dbstat unavailable; per-table bytes omitted")

    rows = []
    for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ):
        try:
            n = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        except sqlite3.Error:
            continue
        idx = [i for (i,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
            (name,),
        )]
        tbl_b = sizes.get(name, 0)
        idx_b = sum(sizes.get(i, 0) for i in idx)
        rows.append((tbl_b + idx_b, tbl_b, idx_b, n, name))

    rows.sort(reverse=True)
    print(f"\n  {'total':>11} {'table':>11} {'indexes':>11} {'rows':>12}  name")
    for tot_b, tbl_b, idx_b, n, name in rows:
        print(f"  {mb(tot_b):>11} {mb(tbl_b):>11} {mb(idx_b):>11} {n:>12,}  {name}")

    cache = [r for r in rows if r[4] not in CORE_TABLES]
    if cache:
        cache_bytes = sum(r[0] for r in cache)
        cache_idx = sum(r[2] for r in cache)
        print(f"\n  {len(cache)} per-entity cache table(s), {mb(cache_bytes)} total")
        print(f"  of which {mb(cache_idx)} is index — roughly half of that is the "
              f"redundant idx_*_ds\n  duplicating the UNIQUE(ds) autoindex")

    conn.close()


# ----------------------------------------------------------------------
# Saved models
# ----------------------------------------------------------------------

def report_models(models_dir):
    rule(f"SAVED MODELS  {models_dir}")
    root = Path(models_dir)
    if not root.is_dir():
        print("  not found — pass --data with the correct location")
        return

    grand = 0
    stale = []
    for exp in sorted(p for p in root.iterdir() if p.is_dir()):
        parts = []
        exp_total = 0
        for label, d in (("current", exp), ("previous", exp / "previous")):
            if not d.is_dir():
                continue
            size = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
            if size:
                parts.append(f"{label} {mb(size)}")
                exp_total += size
        # A leftover *.tmp is the XGBoost sidecar naming bug: the sidecar
        # is written next to model.bin.tmp and never renamed with it.
        stale += [f for f in exp.rglob("*.tmp*") if f.is_file()]
        grand += exp_total
        print(f"  {exp.name:<28} {mb(exp_total):>10}   {', '.join(parts)}")

    print(f"\n  total {mb(grand)} across {len(list(root.iterdir()))} experiment(s)")
    if stale:
        print(f"\n  {len(stale)} stale .tmp file(s) — harmless in size, but "
              f"model.bin.tmp.metadata.json\n  means XGBoost metadata is not "
              f"being found on restore:")
        for f in stale[:10]:
            print(f"    {f}  ({f.stat().st_size:,} B)")


def report_dir(title, path, pattern="*"):
    rule(f"{title}  {path}")
    root = Path(path)
    if not root.is_dir():
        print("  not found")
        return
    files = [f for f in root.rglob(pattern) if f.is_file()]
    total = sum(f.stat().st_size for f in files)
    print(f"  {len(files)} file(s), {mb(total)}")
    for f in sorted(files, key=lambda f: -f.stat().st_size)[:8]:
        print(f"    {mb(f.stat().st_size):>10}  {f.relative_to(root)}")


# ----------------------------------------------------------------------
# Process memory
# ----------------------------------------------------------------------

def report_rss():
    rule("PROCESS MEMORY")
    found = False
    self_pid = str(os.getpid())
    for pid in sorted(p for p in os.listdir("/proc") if p.isdigit()):
        if pid == self_pid:
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            # The add-on runs `python3 -m ml_forecast_lab` (see
            # rootfs/.../init-mlforecastlab/run). Match that shape rather
            # than the bare package name, so this script and any editor
            # or shell sitting in the source tree are not reported.
            if "-m ml_forecast_lab" not in cmd and "ml_forecast_lab.__main__" not in cmd:
                continue
            status = Path(f"/proc/{pid}/status").read_text()
            vals = {}
            for line in status.splitlines():
                for key in ("VmRSS:", "VmHWM:"):
                    if line.startswith(key):
                        vals[key] = int(line.split()[1]) * 1024
            found = True
            print(f"  pid {pid}")
            print(f"    current RSS   {mb(vals.get('VmRSS:', 0))}")
            print(f"    peak RSS      {mb(vals.get('VmHWM:', 0))}  "
                  f"(high-water mark — glibc does not return this to the OS)")
            print(f"    cmd           {cmd.strip()[:90]}")
        except (OSError, ValueError):
            continue
    if not found:
        print("  no ml_forecast_lab process visible from this container")
        print("  from the HA host instead:")
        print("    docker stats --no-stream $(docker ps --filter name=ml_forecast_lab -q)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="/config/mlfl.db")
    ap.add_argument("--data", default="/data/ml_forecast_lab")
    ap.add_argument("--config", default="/config")
    args = ap.parse_args()

    print("ML Forecast Lab — footprint report")
    report_db(args.db)
    report_models(os.path.join(args.data, "models"))
    report_dir("LOGS", os.path.join(args.data, "logs"))
    report_dir("DEBUG DUMPS", os.path.join(args.config, "debug"))
    report_rss()
    print()


if __name__ == "__main__":
    sys.exit(main())
