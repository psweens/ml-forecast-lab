#!/usr/bin/env python3
"""Report the on-disk and in-memory footprint of ML Forecast Lab.

Read-only. Safe to run against a live add-on — the database is opened in
read-only mode and nothing outside stdout is written, apart from SQLite's
own -shm index: a database with a -wal beside it (the live add-on's, or a
copy that kept its -wal) is read through that index, which SQLite creates
or updates. A WAL database without a -wal is opened immutable and gains
no -wal/-shm files.

The defaults are the in-container layout: the database at
``/data/ml_forecast_lab/history.db``, models and logs under
``/data/ml_forecast_lab``, debug dumps under ``/config/debug``. The
script is not part of the image and ``/data`` is private to the add-on,
so run it inside the add-on container. From a host shell with Docker
access, stream it in over stdin (stdlib only, no copy needed)::

    docker exec -i $(docker ps -q --filter name=ml_forecast_lab) \\
        python3 - < mlfl_footprint.py

Against a copied-out ``/data/ml_forecast_lab`` tree, point ``--data``
(and ``--db`` if the database sits elsewhere) at it instead.

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


def sized(paths):
    """(path, size) for each regular file in ``paths`` that still exists.

    The live add-on renames model.bin.tmp, rotates logs and prunes debug
    dumps while this runs, so a file (or directory) can vanish between
    listing and stat; it is skipped rather than aborting the report.
    """
    out = []
    try:
        for f in paths:
            try:
                if f.is_file():
                    out.append((f, f.stat().st_size))
            except OSError:
                continue
    except OSError:
        pass
    return out


# ----------------------------------------------------------------------
# SQLite
# ----------------------------------------------------------------------

def open_ro(path):
    """Read-only connection; creates no files beside a database without a -wal.

    The add-on keeps a connection open, so its WAL database always has a
    -wal file while it runs. A WAL database without one has no connection
    and no writer: it is opened immutable, which reads the main file alone
    and needs no -shm, so a copy in an unwritable directory still opens.
    With a -wal present the WAL pages must be read, so SQLite opens (and if
    need be creates) the -shm beside it.
    """
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    if not os.path.exists(path + "-wal"):
        with open(path, "rb") as f:
            header = f.read(20)
        # Header bytes 18/19 are the file-format write/read versions; 2 = WAL.
        if header.startswith(b"SQLite format 3\0") and header[18] == 2:
            uri += "&immutable=1"
    return sqlite3.connect(uri, uri=True)


def report_db(path):
    rule(f"DATABASE  {path}")
    if not os.path.exists(path):
        print("  not found — pass --data (or --db) with the correct location")
        return
    conn = None
    try:
        conn = open_ro(path)
        report_db_tables(conn, path)
    except (OSError, sqlite3.Error) as e:
        print(f"  could not read the database: {e}")
    finally:
        if conn is not None:
            conn.close()


def report_db_tables(conn, path):
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
    # On-disk size; `total` (page_count) also counts pages still in the WAL.
    print(f"  file            {mb(os.path.getsize(path))}")
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
        if sizes:
            print(f"  {mb(tot_b):>11} {mb(tbl_b):>11} {mb(idx_b):>11} {n:>12,}  {name}")
        else:
            print(f"  {'-':>11} {'-':>11} {'-':>11} {n:>12,}  {name}")

    cache = [r for r in rows if r[4] not in CORE_TABLES]
    if cache and not sizes:
        print(f"\n  {len(cache)} per-entity cache table(s)")
    elif cache:
        cache_bytes = sum(r[0] for r in cache)
        cache_idx = sum(r[2] for r in cache)
        print(f"\n  {len(cache)} per-entity cache table(s), {mb(cache_bytes)} total")
        print(f"  of which {mb(cache_idx)} is index — roughly half of that is the "
              f"redundant idx_*_ds\n  duplicating the UNIQUE(ds) autoindex")


# ----------------------------------------------------------------------
# Saved models
# ----------------------------------------------------------------------

def report_models(models_dir):
    rule(f"SAVED MODELS  {models_dir}")
    root = Path(models_dir)
    if not root.is_dir():
        print("  not found — pass --data with the correct location")
        return
    if not os.access(root, os.R_OK | os.X_OK):
        print("  not readable — permission denied")
        return

    grand = 0
    stale = []
    for exp in sorted(p for p in root.iterdir() if p.is_dir()):
        parts = []
        exp_total = 0
        for label, d in (("current", exp), ("previous", exp / "previous")):
            if not d.is_dir():
                continue
            # *.tmp files are not part of either generation; they are
            # counted once, below.
            size = sum(s for f, s in sized(d.iterdir()) if ".tmp" not in f.name)
            if size:
                parts.append(f"{label} {mb(size)}")
                exp_total += size
        # Persists write *.tmp then rename, so a leftover *.tmp is either an
        # interrupted (or in-flight) save or a model.bin.tmp.metadata.json
        # stranded by v2.52.2 and earlier (XGBoost metadata now rides in the
        # booster).
        exp_stale = sized(exp.rglob("*.tmp*"))
        tmp_size = sum(s for _, s in exp_stale)
        if tmp_size:
            parts.append(f"tmp {mb(tmp_size)}")
            exp_total += tmp_size
        stale += exp_stale
        grand += exp_total
        print(f"  {exp.name:<28} {mb(exp_total):>10}   {', '.join(parts)}")

    n_exp = sum(1 for p in root.iterdir() if p.is_dir())
    print(f"\n  total {mb(grand)} across {n_exp} experiment(s)")
    if stale:
        print(f"\n  {len(stale)} leftover .tmp file(s) — not read on restore; an "
              f"interrupted or in-flight save,\n  or model.bin.tmp.metadata.json "
              f"stranded by v2.52.2 or earlier:")
        for f, size in stale[:10]:
            print(f"    {f}  ({size:,} B)")


def report_dir(title, path, pattern="*"):
    rule(f"{title}  {path}")
    root = Path(path)
    if not root.is_dir():
        print("  not found")
        return
    # rglob skips unreadable directories silently; say so rather than "0 files".
    if not os.access(root, os.R_OK | os.X_OK):
        print("  not readable — permission denied")
        return
    files = sized(root.rglob(pattern))
    total = sum(size for _, size in files)
    print(f"  {len(files)} file(s), {mb(total)}")
    for f, size in sorted(files, key=lambda fs: -fs[1])[:8]:
        print(f"    {mb(size):>10}  {f.relative_to(root)}")


# ----------------------------------------------------------------------
# Process memory
# ----------------------------------------------------------------------

def report_rss():
    rule("PROCESS MEMORY")
    found = False
    self_pid = str(os.getpid())
    try:
        pids = sorted(p for p in os.listdir("/proc") if p.isdigit())
    except OSError:  # no procfs (e.g. reading a copied-out tree on macOS)
        pids = []
    for pid in pids:
        if pid == self_pid:
            continue
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
            cmd = " ".join(argv)
            # The add-on runs `python3 -m ml_forecast_lab` (see
            # rootfs/.../init-mlforecastlab/run). Match those argv tokens
            # rather than a substring, so this script, editors, and shells
            # whose `-c` string merely mentions the command are not reported.
            app_mods = ("ml_forecast_lab", "ml_forecast_lab.__main__")
            is_app = any(a == "-m" and b in app_mods
                         for a, b in zip(argv, argv[1:]))
            if not is_app:
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
                  f"(highest RSS since start; never decreases, even after frees)")
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
    ap.add_argument("--data", default="/data/ml_forecast_lab",
                    help="add-on data directory (models/, logs/, history.db)")
    ap.add_argument("--db", default=None,
                    help="SQLite database (default: <data>/history.db)")
    ap.add_argument("--config", default="/config",
                    help="add-on config directory (debug dumps under debug/)")
    args = ap.parse_args()

    print("ML Forecast Lab — footprint report")
    sections = [
        lambda: report_db(args.db or os.path.join(args.data, "history.db")),
        lambda: report_models(os.path.join(args.data, "models")),
        lambda: report_dir("LOGS", os.path.join(args.data, "logs")),
        lambda: report_dir("DEBUG DUMPS", os.path.join(args.config, "debug")),
        report_rss,
    ]
    # One unreadable path (e.g. permissions on a copied-out tree) must not
    # cost the sections after it.
    for section in sections:
        try:
            section()
        except OSError as e:
            print(f"  could not read: {e}")
    print()


if __name__ == "__main__":
    sys.exit(main())
