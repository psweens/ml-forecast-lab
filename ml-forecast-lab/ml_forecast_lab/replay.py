"""
Replay bundles: capture an experiment's pipeline inputs, re-run them offline.

A bundle records every response the pipeline received at its two I/O
boundaries — the Home Assistant interface (``get_history``, ``get_state``,
``get_config`` and the ``weather.get_forecasts`` service) and the SQLite
history database (history-cache and conformal-quantile reads) — together with
the full experiment config and the wall-clock instant the capture ran at. It
also stores what the pipeline produced from them, in two stages:

* **Training frame** — the resampled grid, the supervised frame, the window
  frame, the missingness report and (for a neural production model) a
  fingerprint of the sliding windows.
* **Forecast** — the production model itself (saved from memory with the
  backend's own ``save``), the forecast it produced through
  ``_compute_cached_forecast`` (model inputs, raw output, published point
  forecast) and its conformal band. Capture also checks that the saved model
  reproduces the live one (``model_roundtrip``) and whether the copy on disk
  matches memory (``disk_consistent``).

Replay rebuilds an app around the recorded responses and runs the same
production methods (``_fetch_and_preprocess`` → ``_prepare_training_frame`` →
``_build_training_windows``; ``_compute_cached_forecast`` →
``_conformal_bands``) at the captured instant, then compares each stage
against the recorded output. Replay is strict: a request the capture never
saw raises ``UnrecordedCall`` rather than returning empty data, and because
the pipeline swallows some I/O errors by design, every miss is also logged —
on either side — and fails the run, so a swallowed miss cannot pass for a
real data gap.

Loading the bundled model runs the backend's loader, which for several
backends is ``pickle``: replaying a bundle someone else made can execute code.
The forecast stage therefore only runs with ``--trust-model``.

Bundle layout (a zip, or the same files in a directory)::

    manifest.json               format version, add-on version, captured_at,
                                configs, host fingerprint, stage summaries
    calls.json                  training-frame stage calls, in call order
    forecast/calls.json         forecast stage calls
    responses/<sha256>.json     large responses, stored once
    expected/grid.npz           _fetch_and_preprocess output
    expected/frame.npz          _supervised_frame output (the training matrix)
    expected/window_frame.npz   imputed complete grid windows are cut from
    expected/report.json        missingness report + window fingerprint
    forecast/model/…            the saved model (model.bin + any sidecar)
    forecast/meta.json          the cache_meta.json a restart would read
    forecast/expected.npz       forecast arrays (point, raw, inputs, band)
    forecast/report.json        forecast status, band, round-trip + disk checks

Only numpy and the standard library are used for storage: the add-on image
has no parquet engine.

CLI::

    python -m ml_forecast_lab.replay <bundle> [--until grid|frame|windows|forecast]
                                              [--trust-model]

Exit status: 0 every stage matches, 1 a stage differs, 2 the replay could
not run (unrecorded call, unreadable bundle).
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import dataclasses
import hashlib
import inspect
import io
import json
import logging
import os
import platform
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FORMAT_VERSION = 2
READABLE_FORMATS = (1, 2)
STAGES = ("grid", "frame", "windows", "forecast")

# HA config keys the pipeline reads (``_get_site_location``). Everything else
# in /api/config (location name, installed components, …) stays out of the
# bundle.
_HA_CONFIG_KEYS = ("latitude", "longitude")

# The only HA ``api_call`` endpoints the pipeline may use during capture.
# api_call is also the write path, so anything else fails the capture.
_API_CALL_ALLOWLIST = {
    ("POST", "/api/services/weather/get_forecasts?return_response"),
}

# Responses larger than this are stored once under responses/<sha256>.json:
# the forecast stage re-fetches the same history the frame stage did, and
# DEFLATE cannot deduplicate across zip entries.
_INLINE_RESPONSE_BYTES = 512

# Backend families for the cross-host comparison rule (see _tolerance).
_EXACT_FAMILY = {"lightgbm", "xgboost", "catboost", "seasonal_naive", "daily_profile"}
_INFORMATIONAL_FAMILY = {"arima", "ets", "theta"}
_NEURAL_REL_TOL = 1e-4

# Distributions whose versions can move a forecast (holidays → is_holiday,
# pvlib → solar features, the model libraries → predictions).
_FINGERPRINT_DISTS = (
    "numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost",
    "statsforecast", "torch", "pvlib", "holidays", "chronos-forecasting",
    "granite-tsfm",
)


class UnrecordedCall(RuntimeError):
    """Replay asked for a response the capture never recorded."""


# ---------------------------------------------------------------------------
# Call keys and response codecs
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


def _api_key(method, endpoint, params=None, json_data=None) -> str:
    return _key("api_call", str(method).upper(), endpoint,
                json.dumps(params, sort_keys=True, default=str),
                json.dumps(json_data, sort_keys=True, default=str))


def _conformal_key(*args, **kwargs) -> str:
    from ml_forecast_lab.db import HistoryDB
    bound = inspect.signature(HistoryDB.get_conformal_quantiles).bind(
        None, *args, **kwargs)
    bound.apply_defaults()
    params = {k: v for k, v in bound.arguments.items() if k != "self"}
    return _key("get_conformal_quantiles",
                json.dumps(params, sort_keys=True, default=str))


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


def _history_frame_to_rows(df: pd.DataFrame) -> dict:
    # Timestamps travel as integer epoch nanoseconds: an ISO string drops
    # ".000000" on whole-second rows, and a table mixing both shapes cannot
    # be parsed back with one inferred format.
    if df is None or df.empty:
        return {"rows": []}
    ds = pd.to_datetime(df["ds"])
    tz = str(ds.dt.tz) if ds.dt.tz is not None else None
    if tz:
        ds = ds.dt.tz_convert("UTC").dt.tz_localize(None)
    y = [None if (isinstance(v, float) and np.isnan(v)) else _jsonable(v)
         for v in df["y"]]
    return {
        "ds_encoding": "ns",
        "ds_tz": tz,
        "rows": [[int(t), v] for t, v in zip(ds.astype("int64"), y)],
        "y_dtype": str(df["y"].dtype),
    }


def _rows_to_history_frame(payload: dict) -> pd.DataFrame:
    # Mirrors HistoryDB.get_history's construction, then pins the recorded
    # value dtype (a NULL-bearing column must come back float64, not object).
    rows = payload.get("rows") or []
    if not rows:
        return pd.DataFrame(columns=["ds", "y"])
    df = pd.DataFrame(rows, columns=["ds", "value"])
    if payload.get("ds_encoding") == "ns":
        df["ds"] = pd.to_datetime(df["ds"].astype("int64"), unit="ns")
        if payload.get("ds_tz"):
            df["ds"] = df["ds"].dt.tz_localize("UTC").dt.tz_convert(payload["ds_tz"])
    else:  # format-1 bundles: ISO strings, with or without a fraction
        df["ds"] = pd.to_datetime(df["ds"], format="ISO8601")
    df = df.rename(columns={"value": "y"})
    if payload.get("y_dtype"):
        df["y"] = df["y"].astype(payload["y_dtype"])
    return df


_INT_KEYED = ("quantiles", "sample_counts")


def _encode_conformal(resp: dict) -> dict:
    # JSON object keys are strings; the band lookup is quantiles.get(int(b)),
    # so the lead-bucket keys travel as [int, value] pairs instead.
    out = {}
    for k, v in (resp or {}).items():
        if k in _INT_KEYED and isinstance(v, dict):
            out[k] = [[int(b), _jsonable(x)] for b, x in v.items()]
        else:
            out[k] = _jsonable(v)
    return out


def _decode_conformal(payload: dict) -> dict:
    out = {}
    for k, v in (payload or {}).items():
        if k in _INT_KEYED and isinstance(v, list):
            out[k] = {int(b): x for b, x in v}
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Recording side
# ---------------------------------------------------------------------------

class _CallLog:
    def __init__(self):
        self.calls: list[dict] = []
        self.misses: list[str] = []

    def add(self, target: str, key: str, response: Any = None,
            error: Optional[BaseException] = None) -> None:
        entry = {"target": target, "key": key}
        if error is not None:
            entry["error_type"] = type(error).__name__
            entry["error"] = str(error)
        else:
            entry["response"] = response
        self.calls.append(entry)

    def miss(self, what: str) -> None:
        self.misses.append(what)


class RecordingHA:
    """Wraps a live ``HAInterface``; records each response it returns.

    The capture pipeline is handed the recorded (JSON-normalised) value, the
    same one replay will serve. Methods outside the recorded set are logged
    as misses and raise, so new I/O on the fetch path fails the capture
    instead of escaping the bundle — even where the pipeline swallows it.
    """

    def __init__(self, inner, log: _CallLog):
        self._inner = inner
        self._log = log

    async def _record(self, key, call):
        try:
            resp = await call()
        except Exception as e:
            self._log.add("ha", key, error=e)
            raise
        resp = _jsonable(resp)
        self._log.add("ha", key, resp)
        return resp

    async def get_history(self, entity_id, start, end, include_attributes=False):
        return await self._record(
            _history_key(entity_id, start, end, include_attributes),
            lambda: self._inner.get_history(
                entity_id, start, end, include_attributes=include_attributes))

    async def get_state(self, entity_id, default=None, attribute=None):
        return await self._record(
            _state_key(entity_id, default, attribute),
            lambda: self._inner.get_state(
                entity_id, default=default, attribute=attribute))

    async def get_config(self):
        key = _key("get_config")
        try:
            full = await self._inner.get_config()
        except Exception as e:
            self._log.add("ha", key, error=e)
            raise
        # The captured run sees the same trimmed dict replay will serve.
        resp = {k: _jsonable(full.get(k)) for k in _HA_CONFIG_KEYS if k in full}
        self._log.add("ha", key, resp)
        return resp

    async def api_call(self, method, endpoint, params=None, json_data=None, **kwargs):
        if (str(method).upper(), endpoint) not in _API_CALL_ALLOWLIST:
            self._log.miss(f"HA.api_call {method} {endpoint}")
            raise PermissionError(
                f"replay capture does not record api_call {method} {endpoint}")
        return await self._record(
            _api_key(method, endpoint, params, json_data),
            lambda: self._inner.api_call(
                method, endpoint, params=params, json_data=json_data, **kwargs))

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        self._log.miss(f"HA.{name}")
        raise AttributeError(f"replay capture does not record HA method {name!r}")


class RecordingHistoryDB:
    """Wraps the live ``HistoryDB``: records reads, suppresses writes.

    Capture must not change the user's database, and replay ignores writes,
    so both sides see the same sequence of reads. Reads are handed back
    decoded from their recorded form, exactly as replay will serve them.
    """

    def __init__(self, inner, log: _CallLog, floor: Optional[datetime] = None):
        self._inner = inner
        self._log = log
        # Rows older than this are dropped before recording: the tables are
        # keyed by entity, so they can hold another experiment's longer
        # history, and every consumer filters to ds >= start anyway.
        self._floor = None
        if floor is not None:
            ts = pd.Timestamp(floor)
            ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts
            self._floor = ts.tz_localize(None)  # the cache stores naive UTC

    def safe_table_name(self, entity_id: str) -> str:
        # Pure function of the entity id: passed through, not recorded.
        return self._inner.safe_table_name(entity_id)

    def get_history(self, table_name: str) -> pd.DataFrame:
        key = _key("get_history", table_name)
        try:
            df = self._inner.get_history(table_name)
        except Exception as e:
            self._log.add("db", key, error=e)
            raise
        if self._floor is not None and not df.empty:
            df = df[pd.to_datetime(df["ds"]) >= self._floor].reset_index(drop=True)
        payload = _history_frame_to_rows(df)
        self._log.add("db", key, payload)
        return _rows_to_history_frame(payload)

    def get_conformal_quantiles(self, *args, **kwargs) -> dict:
        key = _conformal_key(*args, **kwargs)
        try:
            resp = self._inner.get_conformal_quantiles(*args, **kwargs)
        except Exception as e:
            self._log.add("db", key, error=e)
            raise
        payload = _encode_conformal(resp)
        self._log.add("db", key, payload)
        return _decode_conformal(payload)

    def store_history(self, table_name, df) -> int:
        return 0

    def cleanup(self, table_name, oldest_datetime) -> int:
        return 0

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        self._log.miss(f"HistoryDB.{name}")
        raise AttributeError(f"replay capture does not record HistoryDB method {name!r}")


# ---------------------------------------------------------------------------
# Replay side
# ---------------------------------------------------------------------------

def _rebuild_error(entry: dict) -> BaseException:
    """The recorded exception, with the same class name and ``str()``.

    The pipeline records failures as ``f"{type(e).__name__}: {e}"``
    (``fetch_error``, ``bands_error``), so a replayed failure must render
    identically. Builtin types keep their class, so ``except`` clauses still
    match; others become a same-named RuntimeError subclass.
    """
    import builtins

    name = entry.get("error_type")
    msg = entry.get("error", "")
    if name is None:  # format 1 stored "Type: message"
        name, _, msg = str(msg).partition(": ")
    name = name or "RecordedError"
    base = getattr(builtins, name, None)
    if not (isinstance(base, type) and issubclass(base, Exception)):
        base = RuntimeError
    try:
        return type(name, (base,), {"__str__": lambda self: msg})(msg)
    except Exception:
        return type(name, (RuntimeError,), {"__str__": lambda self: msg})(msg)


class _CallQueue:
    def __init__(self, calls: list[dict]):
        self._q: dict[tuple, list[dict]] = {}
        self.misses: list[str] = []
        for c in calls:
            self._q.setdefault((c["target"], c["key"]), []).append(c)

    def pop(self, target: str, key: str) -> Any:
        q = self._q.get((target, key))
        if not q:
            self.misses.append(f"{target} {key}")
            raise UnrecordedCall(f"{target} request not in bundle: {key}")
        entry = q.pop(0)
        if "error" in entry:
            raise _rebuild_error(entry)
        return entry["response"]

    def miss(self, what: str) -> None:
        self.misses.append(what)

    def unused(self) -> list[str]:
        # Format-1 bundles recorded safe_table_name, which is now computed.
        return [k for (t, k), q in self._q.items()
                if q and not k.startswith('["safe_table_name"')]


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

    async def api_call(self, method, endpoint, params=None, json_data=None, **kwargs):
        return self._queue.pop("ha", _api_key(method, endpoint, params, json_data))

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        self._queue.miss(f"HA.{name}")
        raise UnrecordedCall(f"HA method {name!r} is not part of a replay bundle")


class ReplayHistoryDB:
    def __init__(self, queue: _CallQueue):
        self._queue = queue

    def safe_table_name(self, entity_id: str) -> str:
        from ml_forecast_lab.db import HistoryDB
        return HistoryDB.safe_table_name(self, entity_id)

    def get_history(self, table_name: str) -> pd.DataFrame:
        return _rows_to_history_frame(
            self._queue.pop("db", _key("get_history", table_name)))

    def get_conformal_quantiles(self, *args, **kwargs) -> dict:
        return _decode_conformal(
            self._queue.pop("db", _conformal_key(*args, **kwargs)))

    def store_history(self, table_name, df) -> int:
        return 0

    def cleanup(self, table_name, oldest_datetime) -> int:
        return 0

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        self._queue.miss(f"HistoryDB.{name}")
        raise UnrecordedCall(f"HistoryDB method {name!r} is not part of a replay bundle")


# ---------------------------------------------------------------------------
# Array storage (npz, no pickle)
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


def _arrays_to_npz(arrays: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: np.asarray(v) for k, v in arrays.items()
                                if v is not None})
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


def _compare_array(name: str, expected, actual, tol: float = 0.0) -> list[str]:
    """Exact comparison when ``tol`` is 0, else max |Δ| ≤ tol."""
    if expected is None and actual is None:
        return []
    if expected is None or actual is None:
        return [f"{name}: present={expected is not None} → {actual is not None}"]
    e = np.asarray(expected)
    a = np.asarray(actual)
    if e.shape != a.shape:
        return [f"{name}: shape {list(e.shape)} → {list(a.shape)}"]
    if e.dtype.kind in "fc" or a.dtype.kind in "fc":
        e64 = e.astype(np.float64)
        a64 = a.astype(np.float64)
        if np.array_equal(e64, a64, equal_nan=True):
            return []
        nan_flip = int(np.sum(np.isnan(e64) != np.isnan(a64)))
        both = ~np.isnan(e64) & ~np.isnan(a64)
        diff = np.abs(e64 - a64)
        max_abs = float(np.max(diff[both])) if both.any() else 0.0
        if tol > 0 and nan_flip == 0 and max_abs <= tol:
            return []
        flat = np.flatnonzero(
            (~(e64 == a64) & ~(np.isnan(e64) & np.isnan(a64))).ravel())
        return [f"{name}: {len(flat)} value(s) differ, first at index "
                f"{np.unravel_index(int(flat[0]), e.shape) if len(flat) else '?'} "
                f"(max |Δ|={max_abs:.6g}"
                + (f", tolerance {tol:.3g}" if tol > 0 else "")
                + f", NaN mismatches={nan_flip})"]
    if np.array_equal(e, a):
        return []
    return [f"{name}: values differ"]


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
# Host fingerprint
# ---------------------------------------------------------------------------

def _same_host(a: Optional[dict], b: Optional[dict]) -> bool:
    """Same machine, Python and library versions; the torch thread count is
    compared only when both sides imported torch."""
    if not a or not b:
        return False
    if any(a.get(k) != b.get(k) for k in ("machine", "python", "versions")):
        return False
    ca, cb = a.get("cpu") or {}, b.get("cpu") or {}
    for k in ("model", "numpy_features"):
        if ca.get(k) != cb.get(k):
            return False
    for x, y in ((a.get("torch_threads"), b.get("torch_threads")),
                 (ca.get("torch_capability"), cb.get("torch_capability"))):
        if x is not None and y is not None and x != y:
            return False
    return True


def _cpu_identity() -> dict:
    """What decides which SIMD kernels numpy and torch dispatch to: the same
    image on a different CPU can differ in the last bits."""
    info: dict = {"model": None, "numpy_features": None, "torch_capability": None}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k.strip() in ("model name", "CPU part", "Model"):
                    info["model"] = v.strip()
                    break
    except OSError:
        info["model"] = platform.processor() or None
    for mod in ("numpy.core._multiarray_umath", "numpy._core._multiarray_umath"):
        try:
            feats = __import__(mod, fromlist=["__cpu_features__"]).__cpu_features__
            info["numpy_features"] = sorted(k for k, on in feats.items() if on)
            break
        except Exception:
            continue
    if "torch" in sys.modules:
        try:
            info["torch_capability"] = str(
                sys.modules["torch"].backends.cpu.get_cpu_capability())
        except Exception:
            pass
    return info


def _host_fingerprint() -> dict:
    from importlib import metadata

    versions = {}
    for dist in _FINGERPRINT_DISTS:
        try:
            versions[dist] = metadata.version(dist)
        except Exception:
            versions[dist] = None
    threads = None
    if "torch" in sys.modules:
        try:
            threads = int(sys.modules["torch"].get_num_threads())
        except Exception:
            threads = None
    return {
        "machine": platform.machine(),
        "cpu": _cpu_identity(),
        "python": platform.python_version(),
        "versions": versions,
        "torch_threads": threads,
    }


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
    out["windows"] = (
        await app._build_training_windows(exp_cfg, combined, report)
        if build_windows else None
    )
    return out


async def _run_forecast(app, entry: dict, now: datetime) -> dict:
    """``_compute_cached_forecast`` then, as publishing would, the band."""
    fc = await app._compute_cached_forecast(entry, now=now)
    out = {"fc": fc, "bands": None, "bands_error": None}
    exp_cfg = fc.exp_cfg
    # The same gate _publish_forecast_sensors applies before computing bands.
    if fc.status == "ok" and app.history_db and exp_cfg.mode == "production":
        try:
            out["bands"] = await app._conformal_bands(
                exp_cfg, fc.y_pred, fc.ds_future, fc.model_name,
                entry.get("model_version"),
            )
        except Exception as e:
            out["bands_error"] = f"{type(e).__name__}: {e}"
    return out


def _forecast_arrays(run: dict) -> dict:
    fc, bands = run["fc"], run["bands"]
    diag = fc.diag or {}
    arrays = {
        "y_pred": fc.y_pred,
        "y_pred_raw": fc.y_pred_raw,
        "model_output": diag.get("model_output"),
        "ds_future": (pd.DatetimeIndex(fc.ds_future).asi8
                      if fc.ds_future is not None else None),
        "X_rows": diag.get("X_rows"),
        "window": diag.get("window"),
        "steps_tick": diag.get("steps_tick"),
    }
    if bands is not None:
        arrays.update(q_vec=bands.q_vec, upper=bands.upper, lower=bands.lower)
    return arrays


def _forecast_summary(run: dict) -> dict:
    fc, bands = run["fc"], run["bands"]
    diag = fc.diag or {}
    return {
        "status": fc.status,
        "model_name": fc.model_name,
        "model_version": fc.model_version,
        "used_fresh_frame": fc.used_fresh_frame,
        "fetch_error": fc.fetch_error,
        "path": diag.get("path"),
        "future_covariates": diag.get("future_covariates"),
        "channel_names": diag.get("channel_names"),
        "bands": None if bands is None else {
            "level": bands.level,
            "total_samples": bands.total_samples,
            "pooled_versions": bands.pooled_versions,
        },
        "bands_error": run["bands_error"],
    }


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

def _roundtrip_check(twin, loaded_output, diag) -> dict:
    """Does the live model (``twin``: a deep copy of it) reproduce what the
    model reloaded from the bundled bytes output?

    A copy is used so the live model's state is never touched from this
    thread (RevIN writes per-forward statistics onto the module).
    """
    if twin is None:
        return {"status": "not_checked", "reason": "the live model could not be copied"}
    if loaded_output is None:
        return {"status": "not_checked", "reason": "no model output"}
    try:
        if diag.get("path") == "tree":
            rows = np.asarray(diag["X_rows"], dtype=np.float32)
            outs = []
            for i in range(len(rows)):
                y = twin.predict(rows[i:i + 1])
                outs.append(float(y.ravel()[0] if hasattr(y, "ravel") else y[0]))
            live_out = np.asarray(outs, dtype=np.float32)
        elif diag.get("path") == "neural":
            from ml_forecast_lab.models.base import predict_sequence_with_context
            live_out = np.asarray(predict_sequence_with_context(
                twin, diag["window"], diag["steps_tick"])).ravel()
        else:
            return {"status": "not_checked", "reason": "unknown inference path"}
    except Exception as e:
        return {"status": "not_checked", "reason": f"live predict failed: {e}"}
    diffs = _compare_array("model_output", live_out, loaded_output)
    return {"status": "differs" if diffs else "equal", "diffs": diffs}


def _disk_check(live_app, name: str, meta: dict) -> dict:
    """Compare the copy a restart would load with the model in memory."""
    model_dir = live_app._cached_model_dir(name)
    meta_file, model_bin = model_dir / "cache_meta.json", model_dir / "model.bin"
    if not meta_file.exists() or not model_bin.exists():
        return {"consistent": False, "reason": "no persisted model"}
    try:
        disk = json.loads(meta_file.read_text())
        disk_sha = hashlib.sha256(model_bin.read_bytes()).hexdigest()
    except Exception as e:
        return {"consistent": False, "reason": f"unreadable: {e}"}
    fields = ("schema_version", "model_name", "model_version", "trained_at",
              "feature_cols", "missing_indicators", "window_size", "channel_names")
    mismatched = [f for f in fields if disk.get(f) != meta.get(f)]
    return {"consistent": not mismatched, "mismatched_fields": mismatched,
            "model_bin_sha256": disk_sha}


def _history_floor(now: datetime, exp_cfg) -> datetime:
    # One day of slack below the window start the stage will request.
    return now - timedelta(days=int(exp_cfg.days_history) + 1)


async def _capture_forecast(live_app, name: str, now: datetime, files: dict) -> dict:
    """Capture the forecast stage; returns the manifest summary for it."""
    from ml_forecast_lab.main import (
        _cache_entry_from_meta, _cache_meta, build_model_registry,
    )

    cache = live_app._cached_models.get(name)
    if not cache:
        return {"status": "absent", "reason": "no cached production model"}
    meta = _cache_meta(cache)
    summary: dict = {"status": "captured", "model_meta": meta,
                     "experiment_config": _jsonable(cache["exp_cfg"]),
                     "has_db": live_app.history_db is not None}
    files["forecast/meta.json"] = json.dumps(meta, indent=2).encode()
    # Independent of the bundled model, so it runs even when saving or
    # reloading fails — the case where disk most likely holds another model.
    summary["disk"] = await asyncio.to_thread(_disk_check, live_app, name, meta)

    # Everything below works on a private copy: save() and predict() on the
    # live object from this thread could race a live tick (XGBoost's
    # save_model sets and clears Booster attributes).
    try:
        twin = await asyncio.to_thread(copy.deepcopy, cache["model"])
    except Exception as e:
        summary["copy_error"] = f"{type(e).__name__}: {e}"
        twin = None
    registry = getattr(live_app, "model_registry", None) or build_model_registry()
    with tempfile.TemporaryDirectory() as tmp:
        model_bin = Path(tmp) / "model.bin"
        try:
            await asyncio.to_thread((twin or cache["model"]).save, str(model_bin))
        except Exception as e:
            summary.update(status="model_save_failed", error=f"{type(e).__name__}: {e}")
            return summary
        for p in sorted(Path(tmp).rglob("*")):
            if p.is_file():
                files[f"forecast/model/{p.relative_to(tmp).as_posix()}"] = p.read_bytes()
        summary["disk"]["model_bin_sha_equal"] = (
            summary["disk"].pop("model_bin_sha256", None)
            == hashlib.sha256(model_bin.read_bytes()).hexdigest())
        try:
            model = registry.create(meta["model_name"])
            await asyncio.to_thread(model.load, str(model_bin))
        except Exception as e:
            summary.update(status="model_load_failed", error=f"{type(e).__name__}: {e}")
            return summary

    entry = _cache_entry_from_meta(meta, model, cache["exp_cfg"])
    log = _CallLog()
    ha = RecordingHA(live_app.ha_interface, log)
    db = (RecordingHistoryDB(live_app.history_db, log,
                             floor=_history_floor(now, cache["exp_cfg"]))
          if live_app.history_db else None)
    shadow = _shadow_app(live_app.config, ha, db)
    shadow.model_registry = registry
    shadow._cached_models[name] = entry
    try:
        run = await _run_forecast(shadow, entry, now)
    except Exception as e:
        summary.update(status="forecast_failed", error=f"{type(e).__name__}: {e}")
        files["forecast/calls.json"] = log.calls
        summary["misses"] = log.misses
        return summary

    files["forecast/calls.json"] = log.calls
    files["forecast/expected.npz"] = _arrays_to_npz(_forecast_arrays(run))
    report = _forecast_summary(run)
    report["model_roundtrip"] = await asyncio.to_thread(
        _roundtrip_check, twin, (run["fc"].diag or {}).get("model_output"),
        run["fc"].diag or {})
    files["forecast/report.json"] = json.dumps(report, indent=2).encode()
    summary.update(forecast=report, misses=log.misses)
    return summary


def _store_calls(calls: list[dict], blobs: dict) -> list[dict]:
    out = []
    for c in calls:
        entry = {k: v for k, v in c.items() if k != "response"}
        if "response" in c:
            raw = json.dumps(c["response"]).encode()
            if len(raw) > _INLINE_RESPONSE_BYTES:
                sha = hashlib.sha256(raw).hexdigest()
                blobs[f"responses/{sha}.json"] = raw
                entry["ref"] = sha
            else:
                entry["response"] = c["response"]
        out.append(entry)
    return out


async def capture_bundle(live_app, exp_cfg, now: Optional[datetime] = None) -> bytes:
    """Capture both stages for ``exp_cfg`` against the live HA and database,
    recording every response. Returns the bundle as zip bytes.

    Each stage runs on its own app instance with its own call log, so the
    capture neither races the live forecast/retrain ticks nor writes to the
    user's database, and one stage's cache state cannot leak into the other.
    The frame stage is serialised before the forecast stage starts, so the
    two stages' working sets never coexist.
    """
    from ml_forecast_lab import __version__

    now = now or datetime.now(timezone.utc)
    log = _CallLog()
    ha = RecordingHA(live_app.ha_interface, log)
    db = (RecordingHistoryDB(live_app.history_db, log,
                             floor=_history_floor(now, exp_cfg))
          if live_app.history_db else None)
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
    blobs: dict = {}
    frame_calls = _store_calls(log.calls, blobs)
    frame_misses = list(log.misses)
    del result, shadow, log, ha, db

    files: dict = {}
    try:
        forecast = await _capture_forecast(live_app, exp_cfg.name, now, files)
    except Exception as e:
        # The training-frame stage stands on its own; a forecast-stage
        # failure is recorded, never allowed to lose it.
        logger.warning(f"replay capture: forecast stage failed: {e}", exc_info=True)
        files = {}
        forecast = {"status": "capture_error", "error": f"{type(e).__name__}: {e}"}
    forecast_calls = (_store_calls(files.pop("forecast/calls.json"), blobs)
                      if "forecast/calls.json" in files else None)

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
        "has_db": live_app.history_db is not None,
        "host": _host_fingerprint(),
        "capture_misses": frame_misses,
        "forecast": {k: v for k, v in forecast.items()
                     if k not in ("forecast", "model_meta")},
        "privacy": "Contains this experiment's sensor and covariate history "
                   "for its history window, weather/solar forecasts it read, "
                   "the trained model (which for some backends holds recent "
                   "sensor values), and the HA site latitude/longitude.",
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("calls.json", json.dumps(frame_calls))
        if forecast_calls is not None:
            zf.writestr("forecast/calls.json", json.dumps(forecast_calls))
        zf.writestr("expected/report.json", json.dumps(report, indent=2))
        for name, raw in {**expected, **files, **blobs}.items():
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

    def names(self, prefix: str) -> list[str]:
        return sorted(n for n in self._files if n.startswith(prefix))

    def calls(self, name: str) -> list[dict]:
        out = []
        for c in self.json(name):
            if "ref" in c:
                c = dict(c)
                c["response"] = json.loads(self._files[f"responses/{c.pop('ref')}.json"])
            out.append(c)
        return out


def _load_config(app_config: dict, experiment_config: dict):
    """Rebuild AppConfig + ExperimentCfg through the normal YAML loader, so
    replay applies the same parsing/validation the add-on does."""
    import yaml
    from ml_forecast_lab.config import load_config

    data = dict(app_config)
    data["experiments"] = [experiment_config]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "mlfl.yaml"
        p.write_text(yaml.safe_dump(data, sort_keys=False))
        cfg = load_config(p)
    return cfg, cfg.experiments[0]


def _comparison_policy(model_name: str, same_host: bool) -> str:
    """``exact``, ``relative`` or ``informational`` for forecast values.

    Same host fingerprint: everything must be bit-identical. Across hosts,
    tree and profile backends are order-fixed sums and comparisons and stay
    exact; torch and foundation models may drift in the last bits; the
    statsforecast backends refit with a discrete order search at predict
    time, so a cross-host difference there is not evidence of a bug.
    """
    if same_host or model_name in _EXACT_FAMILY:
        return "exact"
    if model_name in _INFORMATIONAL_FAMILY:
        return "informational"
    return "relative"


def _array_tolerance(policy: str, expected: dict, key: str) -> float:
    """Absolute tolerance for one compared array under ``policy``.

    Scaled by that array's own magnitude and by the raw model output's, so
    a forecast clamped to zero (night-time PV) or in log space still gets a
    tolerance on the scale the model actually computed at.
    """
    if policy != "relative":
        return 0.0
    scale = 0.0
    for k in (key, "model_output"):
        a = expected.get(k)
        if a is not None and np.asarray(a).size:
            m = np.nanmax(np.abs(np.asarray(a, dtype=np.float64)))
            if np.isfinite(m):
                scale = max(scale, float(m))
    return _NEURAL_REL_TOL * max(scale, 1e-12)


@dataclasses.dataclass
class ReplayResult:
    manifest: dict
    stages: dict  # stage -> list of differences ([] = identical)
    unused_calls: list
    misses: list = dataclasses.field(default_factory=list)
    notes: list = dataclasses.field(default_factory=list)

    @property
    def matches(self) -> bool:
        return not self.misses and all(not d for d in self.stages.values())

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
        for n in self.notes:
            lines.append(f"  note: {n}")
        if self.misses:
            lines.append(f"  FAILED: {len(self.misses)} request(s) were not in the bundle:")
            lines.extend(f"      {k}" for k in self.misses[:20])
        if self.unused_calls:
            lines.append(f"  note: {len(self.unused_calls)} recorded response(s) "
                         f"were not requested on replay:")
            lines.extend(f"      {k}" for k in self.unused_calls[:10])
        return "\n".join(lines)


async def _replay_forecast(bundle: _Bundle, manifest: dict, trust_model: bool,
                           result: ReplayResult) -> None:
    from ml_forecast_lab.main import (
        CACHE_SCHEMA_VERSION, _cache_entry_from_meta, build_model_registry,
    )

    fsum = manifest.get("forecast") or {}
    if bundle.get("forecast/report.json") is None:
        reason = fsum.get("error") or fsum.get("reason") or fsum.get("status") \
            or "bundle predates forecast capture"
        result.notes.append(f"forecast stage not captured ({reason})")
        return
    if not trust_model:
        result.notes.append(
            "forecast stage skipped: loading the bundled model can execute code "
            "(several backends save with pickle) — re-run with --trust-model if "
            "you trust where this bundle came from")
        return
    meta = bundle.json("forecast/meta.json")
    if meta.get("schema_version") != CACHE_SCHEMA_VERSION:
        result.notes.append(
            f"forecast stage not comparable: bundle model cache schema "
            f"v{meta.get('schema_version')}, this tree expects v{CACHE_SCHEMA_VERSION}")
        return

    expected = bundle.json("forecast/report.json")
    exp_arrays = _npz_to_parts(bundle.get("forecast/expected.npz"))
    config, fc_exp = _load_config(manifest["app_config"], fsum["experiment_config"])

    registry = build_model_registry()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        for name in bundle.names("forecast/model/"):
            dest = _safe_member_path(root, name[len("forecast/model/"):])
            if dest is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(bundle.get(name))
        model = registry.create(meta["model_name"])
        model.load(str(root / "model.bin"))
    entry = _cache_entry_from_meta(meta, model, fc_exp)

    queue = _CallQueue(bundle.calls("forecast/calls.json"))
    has_db = fsum.get("has_db")
    if has_db is None:
        has_db = any(c["target"] == "db" for c in bundle.json("forecast/calls.json"))
    app = _shadow_app(config, ReplayHA(queue), ReplayHistoryDB(queue) if has_db else None)
    app.model_registry = registry
    app._cached_models[fc_exp.name] = entry
    now = pd.Timestamp(manifest["captured_at"]).to_pydatetime()
    try:
        run = await _run_forecast(app, entry, now)
    finally:
        result.misses.extend(queue.misses)
        result.unused_calls.extend(queue.unused())

    actual = _forecast_summary(run)
    act_arrays = _forecast_arrays(run)
    same_host = _same_host(manifest.get("host"), _host_fingerprint())
    policy = _comparison_policy(meta["model_name"], same_host)
    if not same_host:
        result.notes.append(
            "forecast captured on a different host, CPU or library versions: "
            + {"exact": "values compared exactly (order-fixed backend)",
               "informational": "value differences are informational "
                                "(statsforecast refits at predict time)",
               "relative": f"values compared within {_NEURAL_REL_TOL:g} of "
                           f"their scale"}[policy])

    diffs: list[str] = []
    for k in ("status", "used_fresh_frame", "fetch_error", "path",
              "future_covariates", "channel_names", "bands", "bands_error"):
        if expected.get(k) != actual.get(k):
            diffs.append(f"forecast.{k}: {expected.get(k)!r} → {actual.get(k)!r}")
    diffs += _compare_array("forecast.ds_future", exp_arrays.get("ds_future"),
                            act_arrays.get("ds_future"))
    diffs += _compare_array("forecast.q_vec", exp_arrays.get("q_vec"),
                            act_arrays.get("q_vec"))
    # Model inputs: exact whatever the host, so a mismatch pins the cause on
    # input construction (pvlib/holidays/libm or a code change), not the model.
    diffs += _input_diffs(exp_arrays, act_arrays, meta, expected.get("channel_names"))
    value_diffs: list[str] = []
    for k in ("model_output", "y_pred_raw", "y_pred", "upper", "lower"):
        value_diffs += _compare_array(f"forecast.{k}", exp_arrays.get(k),
                                      act_arrays.get(k),
                                      _array_tolerance(policy, exp_arrays, k))
    if policy == "informational":
        result.notes.extend(value_diffs)
    else:
        diffs += value_diffs
    result.stages["forecast"] = diffs
    rt = (expected.get("model_roundtrip") or {})
    if rt.get("status") == "differs":
        result.notes.append(
            "at capture, the saved model did not reproduce the live one "
            f"({'; '.join(rt.get('diffs') or [])}) — a save/load fidelity problem")
    elif rt.get("status") == "not_checked":
        result.notes.append(f"capture could not check the model round trip: {rt.get('reason')}")
    disk = fsum.get("disk") or {}
    if disk and not disk.get("consistent", True):
        result.notes.append(
            "at capture, the model on disk did not match the one in memory "
            f"({disk.get('reason') or ', '.join(disk.get('mismatched_fields') or [])}) "
            "— a restart would have served a different model")


def _safe_member_path(root: Path, rel: str) -> Optional[Path]:
    """``root / rel`` if it stays inside ``root``; None for directory entries.

    Bundle entry names come from whoever made the bundle: an absolute name
    or one with ``..`` would otherwise write outside the temp directory.
    """
    from pathlib import PurePosixPath

    if not rel or rel.endswith("/"):
        return None
    parts = PurePosixPath(rel)
    if parts.is_absolute() or ".." in parts.parts or "\\" in rel:
        raise ValueError(f"unsafe bundle entry {rel!r}")
    dest = (root / parts).resolve()
    if not dest.is_relative_to(root):
        raise ValueError(f"unsafe bundle entry {rel!r}")
    return dest


def _input_diffs(exp_arrays: dict, act_arrays: dict, meta: dict,
                 channel_names) -> list[str]:
    diffs: list[str] = []
    e_rows, a_rows = exp_arrays.get("X_rows"), act_arrays.get("X_rows")
    if e_rows is not None and a_rows is not None and e_rows.shape == a_rows.shape:
        bad = ~((e_rows == a_rows) | (np.isnan(e_rows) & np.isnan(a_rows)))
        if bad.any():
            cols = list(meta.get("feature_cols") or [])
            step = int(np.flatnonzero(bad.any(axis=1))[0])
            names = [cols[j] if j < len(cols) else str(j)
                     for j in np.flatnonzero(bad[step])]
            diffs.append(f"forecast.X_rows: first differs at step {step + 1}, "
                         f"feature(s) {names[:8]}")
    else:
        diffs += _compare_array("forecast.X_rows", e_rows, a_rows)
    e_win, a_win = exp_arrays.get("window"), act_arrays.get("window")
    if e_win is not None and a_win is not None and e_win.shape == a_win.shape:
        chans = list(channel_names or [])
        for c in range(e_win.shape[-1]):
            ec, ac = e_win[..., c], a_win[..., c]
            if not np.array_equal(ec, ac, equal_nan=True):
                label = chans[c] if c < len(chans) else str(c)
                diffs.append(f"forecast.window[{label}]: channel differs "
                             f"(max |Δ|={float(np.nanmax(np.abs(ec - ac))):.6g})")
    else:
        diffs += _compare_array("forecast.window", e_win, a_win)
    diffs += _compare_array("forecast.steps_tick", exp_arrays.get("steps_tick"),
                            act_arrays.get("steps_tick"))
    return diffs


async def replay_bundle(path, until: str = "forecast",
                        trust_model: bool = False) -> ReplayResult:
    if until not in STAGES:
        raise ValueError(f"until must be one of {STAGES}")
    bundle = _Bundle(path)
    manifest = bundle.json("manifest.json")
    if manifest.get("format_version") not in READABLE_FORMATS:
        raise ValueError(f"unsupported bundle format {manifest.get('format_version')}")
    config, exp_cfg = _load_config(manifest["app_config"], manifest["experiment_config"])
    queue = _CallQueue(bundle.calls("calls.json"))
    has_db = manifest.get("has_db")
    if has_db is None:  # format 1
        has_db = any(c["target"] == "db" for c in bundle.json("calls.json"))
    app = _shadow_app(config, ReplayHA(queue),
                      ReplayHistoryDB(queue) if has_db else None)
    now = pd.Timestamp(manifest["captured_at"]).to_pydatetime()
    result = ReplayResult(manifest=manifest, stages={}, unused_calls=[])
    for miss in manifest.get("capture_misses") or []:
        result.notes.append(f"capture could not record {miss}")

    frame_until = "windows" if until == "forecast" else until
    try:
        run = await _run_pipeline(
            app, exp_cfg, now, bool(manifest.get("windows_built")), until=frame_until,
        )
    finally:
        result.misses.extend(queue.misses)
        result.unused_calls.extend(queue.unused())

    exp_grid = bundle.get("expected/grid.npz")
    result.stages["grid"] = _compare_frames(
        "grid", _npz_to_parts(exp_grid), _frame_parts(run["grid"]),
    ) if exp_grid is not None and run.get("grid") is not None else (
        [] if exp_grid is None and run.get("grid") is None
        else ["grid: produced on one side only"])
    report = bundle.json("expected/report.json")
    if frame_until in ("frame", "windows") and "frame" in run:
        diffs = _compare_frames("frame", _npz_to_parts(bundle.get("expected/frame.npz")),
                                _frame_parts(run["frame"]))
        diffs += _compare_frames(
            "window_frame", _npz_to_parts(bundle.get("expected/window_frame.npz")),
            _frame_parts(run["report"]["window_frame"]))
        exp_miss = report.get("missingness") or {}
        act_miss = _report_summary(run["report"])
        for k in sorted(set(exp_miss) | set(act_miss)):
            if exp_miss.get(k) != act_miss.get(k):
                diffs.append(f"missingness.{k}: {exp_miss.get(k)!r} → {act_miss.get(k)!r}")
        result.stages["frame"] = diffs
    if frame_until == "windows" and "frame" in run:
        result.stages["windows"] = _compare_windows(
            report.get("windows"), _window_fingerprint(run.get("windows")))
    if until == "forecast":
        await _with_recorded_threads(
            manifest, _replay_forecast(bundle, manifest, trust_model, result))
    return result


async def _with_recorded_threads(manifest: dict, coro):
    """Run ``coro`` with torch at the capture's intra-op thread count, which
    changes float reduction order; restored afterwards."""
    threads = (manifest.get("host") or {}).get("torch_threads")
    torch = None
    previous = None
    if threads:
        try:
            import torch
            previous = torch.get_num_threads()
            torch.set_num_threads(int(threads))
        except Exception:
            torch = None
    try:
        return await coro
    finally:
        if torch is not None and previous:
            torch.set_num_threads(previous)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ml_forecast_lab.replay",
        description="Re-run a captured experiment's pipeline and compare each stage.",
    )
    ap.add_argument("bundle", type=Path, help="replay bundle (.zip or directory)")
    ap.add_argument("--until", choices=STAGES, default="forecast",
                    help="last stage to run (default: forecast)")
    ap.add_argument("--trust-model", action="store_true",
                    help="load the bundled model and replay the forecast stage. "
                         "The backend loader may unpickle it, which runs code "
                         "from whoever made the bundle: for a bundle from a "
                         "public issue, do this in a throwaway container or VM")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show the pipeline's own log output")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Foundation-model weights must come from the local cache, never a
    # download that may have moved on since the capture.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        result = asyncio.run(replay_bundle(
            args.bundle, until=args.until, trust_model=args.trust_model))
    except (UnrecordedCall, ValueError, KeyError, zipfile.BadZipFile) as e:
        print(f"replay failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(result.render())
    if result.misses:
        return 2
    return 0 if result.matches else 1


if __name__ == "__main__":
    sys.exit(main())
