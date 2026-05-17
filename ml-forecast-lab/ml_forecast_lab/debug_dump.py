"""
Per-retrain training/forecast bundle dumper for diagnosis.

Writes a bundle to ``<config_dir>/debug/<experiment>/<UTC-iso>/`` whenever an
experiment with ``debug_save_training_dumps: true`` retrains. The bundle
captures the exact production training surface so a maintainer can examine
the inputs and outputs offline — synthetic tests passing while production
fails is the regression signature this is built for.

Files in each timestamp directory:
    meta.json              hyperparams, target stats, channel order, data
                           range, dropna report, PF1-PF10 flags
    training.parquet       the full ``combined`` dataframe (target + features)
                           that fed ``create_sliding_windows`` / ``model.fit``
    sliding_window.npz     seq_X, seq_y, channel_names from the neural path
                           (omitted for tree-only experiments)
    forecast.parquet       the immediate post-retrain forecast: datetime,
                           y_pred_raw (model output), y_pred_physical (after
                           log_transform inverse if active)

Rotation keeps the most recent ``KEEP_LAST_N`` bundles per experiment;
older timestamp directories are deleted at the start of each new dump so
the disk footprint stays bounded.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KEEP_LAST_N = 5


def _sanitize_for_json(obj: Any) -> Any:
    """Best-effort JSON-serialisation for ad-hoc nested dicts."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if is_dataclass(obj):
        return _sanitize_for_json(asdict(obj))
    return repr(obj)


def _slug(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


class DebugDumper:
    """Owns the on-disk debug bundle layout. One instance per ForecastService.

    Stateless across calls except for ``_pending_dirs`` which links a
    training dump to its immediate post-retrain forecast dump (the
    forecast cycle runs in a separate method and needs to find the same
    timestamp directory to append to).
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self._pending_dirs: dict[str, Path] = {}

    @classmethod
    def from_config_path(cls, config_path: Optional[Path]) -> Optional["DebugDumper"]:
        if config_path is None:
            return None
        return cls(Path(config_path).parent / "debug")

    def _experiment_dir(self, exp_name: str) -> Path:
        return self.root / _slug(exp_name)

    def _rotate(self, exp_name: str) -> None:
        exp_dir = self._experiment_dir(exp_name)
        if not exp_dir.exists():
            return
        timestamp_dirs = sorted(
            (p for p in exp_dir.iterdir() if p.is_dir()),
            key=lambda p: p.name,
        )
        excess = len(timestamp_dirs) - KEEP_LAST_N + 1
        for old in timestamp_dirs[:max(0, excess)]:
            try:
                shutil.rmtree(old)
            except Exception as e:
                logger.debug(f"debug_dump: could not rotate {old}: {e}")

    def dump_training(
        self,
        *,
        exp_name: str,
        model_name: str,
        exp_cfg: Any,
        combined: pd.DataFrame,
        feature_cols: list[str],
        target_stats: dict,
        seq_X: Optional[np.ndarray] = None,
        seq_y: Optional[np.ndarray] = None,
        channel_names: Optional[list[str]] = None,
        seq_kwargs: Optional[dict] = None,
        model_params: Optional[dict] = None,
        rows_before_dropna: Optional[int] = None,
        rows_after_dropna: Optional[int] = None,
    ) -> Optional[Path]:
        """Write the training-side files. Returns the timestamp dir on
        success, or None on failure (errors are swallowed and logged)."""
        try:
            self._rotate(exp_name)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_dir = self._experiment_dir(exp_name) / ts
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                combined.to_parquet(out_dir / "training.parquet")
            except Exception as e:
                # Parquet engine missing or column dtype unsupported — fall
                # back to CSV so the bundle is still useful for analysis.
                logger.debug(f"debug_dump: parquet failed, using CSV: {e}")
                combined.to_csv(out_dir / "training.csv")

            if seq_X is not None and seq_y is not None:
                np.savez_compressed(
                    out_dir / "sliding_window.npz",
                    seq_X=seq_X,
                    seq_y=seq_y,
                    channel_names=np.asarray(channel_names or [], dtype=object),
                )

            meta = {
                "experiment": exp_name,
                "model_name": model_name,
                "timestamp_utc": ts,
                "addon_version": _addon_version(),
                "target_stats": _sanitize_for_json(target_stats),
                "data_range": {
                    "start": combined.index.min().isoformat() if len(combined) else None,
                    "end": combined.index.max().isoformat() if len(combined) else None,
                    "interval_minutes": getattr(exp_cfg, "interval_minutes", None),
                    "rows": int(len(combined)),
                    "rows_before_dropna": rows_before_dropna,
                    "rows_after_dropna": rows_after_dropna,
                },
                "feature_cols": list(feature_cols),
                "channel_names": list(channel_names) if channel_names else None,
                "seq_shapes": {
                    "seq_X": list(seq_X.shape) if seq_X is not None else None,
                    "seq_y": list(seq_y.shape) if seq_y is not None else None,
                },
                "seq_kwargs": _sanitize_for_json({
                    k: v for k, v in (seq_kwargs or {}).items()
                    if k not in ("sequence_data",)  # tensor; in sliding_window.npz
                }),
                "model_params": _sanitize_for_json(model_params or {}),
                "experiment_config": _sanitize_for_json({
                    k: getattr(exp_cfg, k, None) for k in (
                        "name", "target_entity", "interval_minutes",
                        "future_periods", "days_history", "log_transform",
                        "source_is_cumulative", "target_is_nonnegative",
                        "reset_daily", "loss_fn", "daily_loss_weight",
                        "optimiser", "country", "recency_half_life_days",
                        "production_model",
                    )
                }),
            }
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

            self._pending_dirs[exp_name] = out_dir
            logger.info(f"  Debug dump: training → {out_dir}")
            return out_dir
        except Exception as e:
            logger.warning(f"debug_dump: training write failed for {exp_name}: {e}")
            return None

    def dump_forecast(
        self,
        *,
        exp_name: str,
        y_pred_raw: Optional[np.ndarray],
        y_pred_physical: np.ndarray,
        ds_future: pd.DatetimeIndex,
        model_version: Optional[str] = None,
        log_transform_applied: bool = False,
    ) -> Optional[Path]:
        """Append a forecast.parquet to the most recent training dump for
        this experiment (paired with dump_training via _pending_dirs).
        If no training dump is pending (e.g. forecast without retrain),
        no-op — by design we only capture the immediate post-retrain
        forecast to keep disk usage bounded."""
        out_dir = self._pending_dirs.pop(exp_name, None)
        if out_dir is None or not out_dir.exists():
            return None
        try:
            n = len(y_pred_physical)
            df = pd.DataFrame({
                "datetime": [t.isoformat() for t in ds_future[:n]],
                "y_pred_physical": np.asarray(y_pred_physical, dtype=np.float64),
            })
            if y_pred_raw is not None and len(y_pred_raw) == n:
                df["y_pred_raw"] = np.asarray(y_pred_raw, dtype=np.float64)
            try:
                df.to_parquet(out_dir / "forecast.parquet")
            except Exception:
                df.to_csv(out_dir / "forecast.csv")

            try:
                meta_path = out_dir / "meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    meta["forecast"] = {
                        "model_version": model_version,
                        "log_transform_applied": log_transform_applied,
                        "n_points": int(n),
                        "range_physical": [
                            float(np.min(y_pred_physical)),
                            float(np.max(y_pred_physical)),
                        ],
                    }
                    meta_path.write_text(json.dumps(meta, indent=2))
            except Exception as e:
                logger.debug(f"debug_dump: meta update failed: {e}")

            logger.info(f"  Debug dump: forecast → {out_dir}")
            return out_dir
        except Exception as e:
            logger.warning(f"debug_dump: forecast write failed for {exp_name}: {e}")
            return None


def _addon_version() -> str:
    try:
        from ml_forecast_lab import __version__
        return __version__
    except Exception:
        return "unknown"
