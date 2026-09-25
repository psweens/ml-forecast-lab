"""
Replay bundles: capture an experiment's pipeline inputs, re-run them offline.

A bundle records every response the training-frame pipeline received at its
two I/O boundaries — the Home Assistant interface (``get_history``,
``get_state``, ``get_config``) and the SQLite history cache
(``get_history`` reads) — together with the full experiment config and the
wall-clock instant the capture ran at. It also stores what the pipeline
produced from them: the resampled grid, the supervised frame, the window
frame, the missingness report, and (for a neural production model) a
fingerprint of the sliding windows.

Replay rebuilds an app around the recorded responses and runs the same
production methods (``_fetch_and_preprocess`` → ``build_features`` →
``_supervised_frame`` → ``_build_training_windows``) at the captured
instant, then compares each stage against the recorded output. Replay is
strict: a request the capture never saw raises ``UnrecordedCall`` rather
than returning empty data, because an empty response is indistinguishable
from a real data gap downstream.

Bundle layout (a zip, or the same files in a directory)::

    manifest.json       format version, add-on version, captured_at,
                        experiment + app config, production model
    calls.json          recorded HA / cache responses, in call order
    expected/grid.npz   _fetch_and_preprocess output
    expected/frame.npz  _supervised_frame output (the training matrix)
    expected/window_frame.npz   imputed complete grid windows are cut from
    expected/report.json        missingness report + window fingerprint

Only numpy and the standard library are used for storage: the add-on image
has no parquet engine.

CLI::

    python -m ml_forecast_lab.replay <bundle> [--until grid|frame|windows]

Exit status: 0 every stage matches, 1 a stage differs, 2 the replay could
not run (unrecorded call, unreadable bundle).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import io
import json
import logging
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
STAGES = ("grid", "frame", "windows")

# HA config keys the pipeline reads (``_get_site_location``). Everything else
# in /api/config (location name, installed components, …) stays out of the
# bundle.
_HA_CONFIG_KEYS = ("latitude", "longitude")


class UnrecordedCall(RuntimeError):
    """Replay asked for a response the capture never recorded."""


# ---------------------------------------------------------------------------
# Call keys
# ---------------------------------------------------------------------------

def _iso(t: Any) -> str:
    return pd.Timestamp(t).isoformat()


def _key(method: str, *parts: Any) -> str:
    return json.dumps([method, *parts], default=str)


def _history_key(entity_id, start, end, include_attributes=False) -> str:
    return _key("get_history", entity_id, _iso(start), _iso(end),
                bool(include_attributes))


def _state_key(entity_id, default=None, attribute=None) -> str:
    return _key("get_state", entity_id, attribute, repr(default))


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, (datetime, pd.Timestamp)):
        return _iso(obj)
    if dataclasses.is_dataclass(obj):
        return _jsonable(dataclasses.asdict(obj))
    return repr(obj)


# ---------------------------------------------------------------------------
# Recording side
# ---------------------------------------------------------------------------

class _CallLog:
    def __init__(self):
        self.calls: list[dict] = []

    def add(self, target: str, key: str, response: Any = None,
            error: Optional[str] = None) -> None:
        entry = {"target": target, "key": key}
        if error is not None:
            entry["error"] = error
        else:
            entry["response"] = response
        self.calls.append(entry)


class RecordingHA:
    """Wraps a live ``HAInterface``; records each response it returns.

    Methods outside the recorded set raise ``AttributeError`` so new I/O on
    the fetch path fails loudly instead of escaping the bundle.
    """

    def __init__(self, inner, log: _CallLog):
        self._inner = inner
        self._log = log

    async def get_history(self, entity_id, start, end, include_attributes=False):
        key = _history_key(entity_id, start, end, include_attributes)
        try:
            resp = await self._inner.get_history(
                entity_id, start, end, include_attributes=include_attributes,
            )
        except Exception as e:
            self._log.add("ha", key, error=f"{type(e).__name__}: {e}")
            raise
        resp = _jsonable(resp)
        self._log.add("ha", key, resp)
        return resp

    async def get_state(self, entity_id, default=None, attribute=None):
        key = _state_key(entity_id, default, attribute)
        try:
            resp = await self._inner.get_state(
                entity_id, default=default, attribute=attribute,
            )
        except Exception as e:
            self._log.add("ha", key, error=f"{type(e).__name__}: {e}")
            raise
        resp = _jsonable(resp)
        self._log.add("ha", key, resp)
        return resp

    async def get_config(self):
        key = _key("get_config")
        try:
            full = await self._inner.get_config()
        except Exception as e:
            self._log.add("ha", key, error=f"{type(e).__name__}: {e}")
            raise
        # The captured run sees the same trimmed dict replay will serve.
        resp = {k: _jsonable(full.get(k)) for k in _HA_CONFIG_KEYS if k in full}
        self._log.add("ha", key, resp)
        return resp

    def __getattr__(self, name):
        raise AttributeError(f"replay capture does not record HA method {name!r}")


def _history_frame_to_rows(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"rows": []}
    y = [None if (isinstance(v, float) and np.isnan(v)) else _jsonable(v)
         for v in df["y"]]
    return {
        "rows": [[_iso(ds), v] for ds, v in zip(df["ds"], y)],
        "y_dtype": str(df["y"].dtype),
    }


def _rows_to_history_frame(payload: dict) -> pd.DataFrame:
    # Mirrors HistoryDB.get_history's construction, then pins the recorded
    # value dtype (a NULL-bearing column must come back float64, not object).
    rows = payload.get("rows") or []
    if not rows:
        return pd.DataFrame(columns=["ds", "y"])
    df = pd.DataFrame(rows, columns=["ds", "value"])
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.rename(columns={"value": "y"})
    if payload.get("y_dtype"):
        df["y"] = df["y"].astype(payload["y_dtype"])
    return df


class RecordingHistoryDB:
    """Wraps the live ``HistoryDB``: records reads, suppresses writes.

    Capture must not change the user's cache, and replay ignores writes, so
    both sides see the same sequence of reads.
    """

    def __init__(self, inner, log: _CallLog):
        self._inner = inner
        self._log = log

    def safe_table_name(self, entity_id: str) -> str:
        name = self._inner.safe_table_name(entity_id)
        self._log.add("db", _key("safe_table_name", entity_id), name)
        return name

    def get_history(self, table_name: str) -> pd.DataFrame:
        df = self._inner.get_history(table_name)
        self._log.add("db", _key("get_history", table_name),
                      _history_frame_to_rows(df))
        return df

    def store_history(self, table_name, df) -> int:
        return 0

    def cleanup(self, table_name, oldest_datetime) -> int:
        return 0

    def __getattr__(self, name):
        raise AttributeError(f"replay capture does not record HistoryDB method {name!r}")


# ---------------------------------------------------------------------------
# Replay side
# ---------------------------------------------------------------------------

class _CallQueue:
    def __init__(self, calls: list[dict]):
        self._q: dict[tuple, list[dict]] = {}
        for c in calls:
            self._q.setdefault((c["target"], c["key"]), []).append(c)

    def pop(self, target: str, key: str) -> Any:
        q = self._q.get((target, key))
        if not q:
            raise UnrecordedCall(f"{target} request not in bundle: {key}")
        entry = q.pop(0)
        if "error" in entry:
            raise RuntimeError(f"recorded failure: {entry['error']}")
        return entry["response"]

    def unused(self) -> list[str]:
        return [k for (t, k), q in self._q.items() if q]


class ReplayHA:
    def __init__(self, queue: _CallQueue):
        self._queue = queue

    async def get_history(self, entity_id, start, end, include_attributes=False):
        return self._queue.pop(
            "ha", _history_key(entity_id, start, end, include_attributes))

    async def get_state(self, entity_id, default=None, attribute=None):
        return self._queue.pop("ha", _state_key(entity_id, default, attribute))

    async def get_config(self):
        return self._queue.pop("ha", _key("get_config"))

    def __getattr__(self, name):
        raise UnrecordedCall(f"HA method {name!r} is not part of a replay bundle")


class ReplayHistoryDB:
    def __init__(self, queue: _CallQueue):
        self._queue = queue

    def safe_table_name(self, entity_id: str) -> str:
        return self._queue.pop("db", _key("safe_table_name", entity_id))

    def get_history(self, table_name: str) -> pd.DataFrame:
        return _rows_to_history_frame(
            self._queue.pop("db", _key("get_history", table_name)))

    def store_history(self, table_name, df) -> int:
        return 0

    def cleanup(self, table_name, oldest_datetime) -> int:
        return 0

    def __getattr__(self, name):
        raise UnrecordedCall(f"HistoryDB method {name!r} is not part of a replay bundle")


# ---------------------------------------------------------------------------
# Frame storage (npz, no pickle)
# ---------------------------------------------------------------------------

def _frame_to_npz(df: pd.DataFrame) -> bytes:
    idx = pd.DatetimeIndex(df.index)
    tz = str(idx.tz) if idx.tz is not None else ""
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    values = np.empty((len(df), len(df.columns)), dtype=np.float64)
    for j, c in enumerate(df.columns):
        values[:, j] = pd.to_numeric(df[c], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan)
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        index=idx.asi8,
        tz=np.asarray(tz),
        columns=np.asarray([str(c) for c in df.columns], dtype=str),
        dtypes=np.asarray([str(t) for t in df.dtypes], dtype=str),
        values=values,
    )
    return buf.getvalue()


def _npz_to_parts(raw: bytes) -> dict:
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _frame_parts(df: pd.DataFrame) -> dict:
    return _npz_to_parts(_frame_to_npz(df))


def _compare_frames(name: str, expected: dict, actual: dict) -> list[str]:
    """Return human-readable differences; empty when bit-identical."""
    diffs: list[str] = []
    e_cols, a_cols = list(expected["columns"]), list(actual["columns"])
    if e_cols != a_cols:
        missing = [c for c in e_cols if c not in a_cols]
        extra = [c for c in a_cols if c not in e_cols]
        diffs.append(f"{name}: columns differ (missing={missing}, extra={extra}"
                     + (", order changed" if not missing and not extra else "") + ")")
    if str(expected["tz"]) != str(actual["tz"]):
        diffs.append(f"{name}: index timezone {expected['tz']!s} → {actual['tz']!s}")
    e_idx, a_idx = expected["index"], actual["index"]
    if len(e_idx) != len(a_idx) or not np.array_equal(e_idx, a_idx):
        diffs.append(
            f"{name}: index differs ({len(e_idx)} → {len(a_idx)} rows"
            + _first_index_difference(e_idx, a_idx) + ")")
        return diffs
    e_types = dict(zip(e_cols, expected["dtypes"]))
    a_types = dict(zip(a_cols, actual["dtypes"]))
    a_pos = {c: j for j, c in enumerate(a_cols)}
    for j, c in enumerate(e_cols):
        if c not in a_pos:
            continue
        if e_types[c] != a_types[c]:
            diffs.append(f"{name}.{c}: dtype {e_types[c]} → {a_types[c]}")
        ev = expected["values"][:, j]
        av = actual["values"][:, a_pos[c]]
        if np.array_equal(ev, av, equal_nan=True):
            continue
        nan_flip = int(np.sum(np.isnan(ev) != np.isnan(av)))
        both = ~np.isnan(ev) & ~np.isnan(av)
        max_abs = float(np.max(np.abs(ev[both] - av[both]))) if both.any() else 0.0
        bad = np.flatnonzero(~((ev == av) | (np.isnan(ev) & np.isnan(av))))
        first = pd.Timestamp(int(e_idx[bad[0]])).isoformat() if len(bad) else "?"
        diffs.append(
            f"{name}.{c}: {len(bad)} cell(s) differ from {first} "
            f"(max |Δ|={max_abs:.6g}, NaN mismatches={nan_flip})")
    return diffs


def _first_index_difference(e: np.ndarray, a: np.ndarray) -> str:
    n = min(len(e), len(a))
    neq = np.flatnonzero(e[:n] != a[:n])
    if len(neq):
        i = int(neq[0])
        return (f", first at position {i}: {pd.Timestamp(int(e[i])).isoformat()}"
                f" vs {pd.Timestamp(int(a[i])).isoformat()}")
    if len(e) and len(a):
        return (f", spans {pd.Timestamp(int(e[0])).isoformat()}…{pd.Timestamp(int(e[-1])).isoformat()}"
                f" vs {pd.Timestamp(int(a[0])).isoformat()}…{pd.Timestamp(int(a[-1])).isoformat()}")
    return ""


# ---------------------------------------------------------------------------
# Window fingerprint
# ---------------------------------------------------------------------------

def _sha(arr: Optional[np.ndarray]) -> Optional[str]:
    if arr is None:
        return None
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _window_fingerprint(windows) -> Optional[dict]:
    """Hashes plus per-channel stats of the neural training windows.

    The tensor itself is not stored: at a 5-minute grid with a long horizon
    it runs to hundreds of MB. The hash detects any change; the per-channel
    stats say which channel moved.
    """
    if windows is None:
        return None
    seq_X, seq_y, channel_names, seq_kwargs = windows
    X = np.asarray(seq_X, dtype=np.float64)
    channels = {}
    for c, name in enumerate(channel_names):
        col = X[:, :, c]
        channels[str(name)] = {
            "sha256": _sha(np.ascontiguousarray(seq_X[:, :, c])),
            "mean": float(np.nanmean(col)) if col.size else None,
            "std": float(np.nanstd(col)) if col.size else None,
            "nan": int(np.isnan(col).sum()),
        }
    meta = {k: _jsonable(v) for k, v in seq_kwargs.items()
            if k not in ("sequence_data", "window_step_index")}
    meta["window_step_index_sha256"] = _sha(seq_kwargs.get("window_step_index"))
    return {
        "seq_X_shape": list(seq_X.shape),
        "seq_y_shape": list(np.asarray(seq_y).shape),
        "seq_X_sha256": _sha(seq_X),
        "seq_y_sha256": _sha(seq_y),
        "channel_names": [str(c) for c in channel_names],
        "channels": channels,
        "seq_kwargs": meta,
    }


def _compare_windows(expected: Optional[dict], actual: Optional[dict]) -> list[str]:
    if expected is None and actual is None:
        return []
    if expected is None or actual is None:
        return [f"windows: built={expected is not None} → {actual is not None}"]
    diffs = []
    for k in ("seq_X_shape", "seq_y_shape"):
        if expected[k] != actual[k]:
            diffs.append(f"windows.{k}: {expected[k]} → {actual[k]}")
    if expected["channel_names"] != actual["channel_names"]:
        diffs.append(f"windows.channel_names: {expected['channel_names']} → "
                     f"{actual['channel_names']}")
    for name, e in expected["channels"].items():
        a = actual["channels"].get(name)
        if a is not None and e["sha256"] != a["sha256"]:
            diffs.append(
                f"windows[{name}]: mean {e['mean']:.6g} → {a['mean']:.6g}, "
                f"std {e['std']:.6g} → {a['std']:.6g}, NaN {e['nan']} → {a['nan']}")
    if expected["seq_y_sha256"] != actual["seq_y_sha256"]:
        diffs.append("windows.seq_y: labels differ")
    for k in sorted(set(expected["seq_kwargs"]) | set(actual["seq_kwargs"])):
        if expected["seq_kwargs"].get(k) != actual["seq_kwargs"].get(k):
            diffs.append(f"windows.seq_kwargs.{k}: {expected['seq_kwargs'].get(k)!r}"
                         f" → {actual['seq_kwargs'].get(k)!r}")
    if not diffs and expected["seq_X_sha256"] != actual["seq_X_sha256"]:
        diffs.append("windows.seq_X: tensor hash differs")
    return diffs


def _report_summary(report: dict) -> dict:
    return {k: _jsonable(v) for k, v in report.items()
            if k not in ("window_frame", "window_label_mask")}


# ---------------------------------------------------------------------------
# Pipeline (shared by capture and replay)
# ---------------------------------------------------------------------------

async def _run_pipeline(app, exp_cfg, now: datetime, build_windows: bool,
                        until: str = "windows") -> dict:
    out: dict = {}
    if until == "grid":
        out["grid"] = await app._fetch_and_preprocess(exp_cfg, now=now)
        return out
    prepared = await app._prepare_training_frame(exp_cfg, now=now)
    if prepared is None:
        out["grid"] = None
        return out
    out["grid"], out["frame"], out["report"] = prepared
    combined, report = out["frame"], out["report"]
    if until == "frame":
        return out
    out["windows"] = (
        await app._build_training_windows(exp_cfg, combined, report)
        if build_windows else None
    )
    return out


def _shadow_app(config, ha, db):
    from ml_forecast_lab.covariates import CovariateResolver
    from ml_forecast_lab.main import MLForecastLabApp

    app = MLForecastLabApp()
    app.config = config
    app.ha_interface = ha
    app.history_db = db
    app.covariate_resolver = CovariateResolver(
        ha, history_db=db, retention_provider=app._retention_days_for_table,
    )
    return app


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

async def capture_bundle(live_app, exp_cfg, now: Optional[datetime] = None) -> bytes:
    """Run the training-frame pipeline for ``exp_cfg`` against the live HA
    and cache, recording every response. Returns the bundle as zip bytes.

    Runs on a separate app instance so the capture neither races the live
    forecast/retrain ticks nor writes to the user's cache.
    """
    from ml_forecast_lab import __version__

    now = now or datetime.now(timezone.utc)
    log = _CallLog()
    ha = RecordingHA(live_app.ha_interface, log)
    db = RecordingHistoryDB(live_app.history_db, log) if live_app.history_db else None
    shadow = _shadow_app(live_app.config, ha, db)

    prod_model = live_app._production_model_name(exp_cfg)
    build_windows = False
    registry = getattr(live_app, "model_registry", None)
    if registry is not None:
        try:
            build_windows = bool(registry.create(prod_model).is_neural)
        except Exception as e:  # unknown / unavailable backend
            logger.debug(f"replay capture: cannot instantiate {prod_model}: {e}")

    result = await _run_pipeline(shadow, exp_cfg, now, build_windows)

    app_cfg = {
        f.name: _jsonable(getattr(live_app.config, f.name))
        for f in dataclasses.fields(live_app.config) if f.name != "experiments"
    }
    manifest = {
        "format_version": FORMAT_VERSION,
        "addon_version": __version__,
        "captured_at": _iso(now),
        "experiment": exp_cfg.name,
        "experiment_config": _jsonable(exp_cfg),
        "app_config": app_cfg,
        "production_model": prod_model,
        "windows_built": build_windows,
        "privacy": "Contains this experiment's sensor history and the HA site "
                   "latitude/longitude.",
    }
    expected = {}
    if result.get("grid") is not None:
        expected["expected/grid.npz"] = _frame_to_npz(result["grid"])
    if "frame" in result:
        expected["expected/frame.npz"] = _frame_to_npz(result["frame"])
        expected["expected/window_frame.npz"] = _frame_to_npz(
            result["report"]["window_frame"])
    report = {
        "missingness": _report_summary(result["report"]) if "report" in result else None,
        "windows": _window_fingerprint(result.get("windows")),
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("calls.json", json.dumps(log.calls))
        zf.writestr("expected/report.json", json.dumps(report, indent=2))
        for name, raw in expected.items():
            zf.writestr(name, raw)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

class _Bundle:
    def __init__(self, path: Path):
        path = Path(path)
        self._files: dict[str, bytes] = {}
        if path.is_dir():
            for p in path.rglob("*"):
                if p.is_file():
                    self._files[p.relative_to(path).as_posix()] = p.read_bytes()
        else:
            with zipfile.ZipFile(path) as zf:
                for n in zf.namelist():
                    self._files[n] = zf.read(n)

    def json(self, name: str) -> Any:
        return json.loads(self._files[name])

    def get(self, name: str) -> Optional[bytes]:
        return self._files.get(name)


def _load_config(manifest: dict):
    """Rebuild AppConfig + ExperimentCfg through the normal YAML loader, so
    replay applies the same parsing/validation the add-on does."""
    import yaml
    from ml_forecast_lab.config import load_config

    data = dict(manifest["app_config"])
    data["experiments"] = [manifest["experiment_config"]]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "mlfl.yaml"
        p.write_text(yaml.safe_dump(data, sort_keys=False))
        cfg = load_config(p)
    return cfg, cfg.experiments[0]


@dataclasses.dataclass
class ReplayResult:
    manifest: dict
    stages: dict  # stage -> list of differences ([] = identical)
    unused_calls: list

    @property
    def matches(self) -> bool:
        return all(not d for d in self.stages.values())

    def render(self) -> str:
        m = self.manifest
        lines = [
            f"Replay of '{m['experiment']}' captured {m['captured_at']} "
            f"on v{m['addon_version']} (production model: {m['production_model']})",
        ]
        from ml_forecast_lab import __version__
        lines.append(f"Current tree: v{__version__}")
        for stage, diffs in self.stages.items():
            lines.append(f"  {stage:<8} {'identical' if not diffs else 'DIFFERS'}")
            lines.extend(f"      {d}" for d in diffs)
        if self.unused_calls:
            lines.append(f"  note: {len(self.unused_calls)} recorded response(s) "
                         f"were not requested on replay:")
            lines.extend(f"      {k}" for k in self.unused_calls[:10])
        return "\n".join(lines)


async def replay_bundle(path, until: str = "windows") -> ReplayResult:
    if until not in STAGES:
        raise ValueError(f"until must be one of {STAGES}")
    bundle = _Bundle(path)
    manifest = bundle.json("manifest.json")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported bundle format {manifest.get('format_version')}")
    config, exp_cfg = _load_config(manifest)
    queue = _CallQueue(bundle.json("calls.json"))
    has_db = any(c["target"] == "db" for c in bundle.json("calls.json"))
    app = _shadow_app(config, ReplayHA(queue),
                      ReplayHistoryDB(queue) if has_db else None)
    now = pd.Timestamp(manifest["captured_at"]).to_pydatetime()
    result = await _run_pipeline(
        app, exp_cfg, now, bool(manifest.get("windows_built")), until=until,
    )

    stages: dict[str, list[str]] = {}
    exp_grid = bundle.get("expected/grid.npz")
    stages["grid"] = _compare_frames(
        "grid", _npz_to_parts(exp_grid), _frame_parts(result["grid"]),
    ) if exp_grid is not None and result.get("grid") is not None else (
        [] if exp_grid is None and result.get("grid") is None
        else ["grid: produced on one side only"])
    report = bundle.json("expected/report.json")
    if until in ("frame", "windows") and "frame" in result:
        diffs = _compare_frames("frame", _npz_to_parts(bundle.get("expected/frame.npz")),
                                _frame_parts(result["frame"]))
        diffs += _compare_frames(
            "window_frame", _npz_to_parts(bundle.get("expected/window_frame.npz")),
            _frame_parts(result["report"]["window_frame"]))
        exp_miss = report.get("missingness") or {}
        act_miss = _report_summary(result["report"])
        for k in sorted(set(exp_miss) | set(act_miss)):
            if exp_miss.get(k) != act_miss.get(k):
                diffs.append(f"missingness.{k}: {exp_miss.get(k)!r} → {act_miss.get(k)!r}")
        stages["frame"] = diffs
    if until == "windows" and "frame" in result:
        stages["windows"] = _compare_windows(
            report.get("windows"), _window_fingerprint(result.get("windows")))
    return ReplayResult(manifest=manifest, stages=stages, unused_calls=queue.unused())


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ml_forecast_lab.replay",
        description="Re-run a captured experiment's pipeline and compare each stage.",
    )
    ap.add_argument("bundle", type=Path, help="replay bundle (.zip or directory)")
    ap.add_argument("--until", choices=STAGES, default="windows",
                    help="last stage to run (default: windows)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show the pipeline's own log output")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        result = asyncio.run(replay_bundle(args.bundle, until=args.until))
    except (UnrecordedCall, ValueError, KeyError, zipfile.BadZipFile) as e:
        print(f"replay failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(result.render())
    return 0 if result.matches else 1


if __name__ == "__main__":
    sys.exit(main())
