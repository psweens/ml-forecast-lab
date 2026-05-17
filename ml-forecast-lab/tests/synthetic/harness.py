"""
Mini training harness that mirrors the production neural path:

    1.  ``create_sliding_windows`` with ``horizon_steps = [1..H]`` and an
        optional ``future_features_df`` (extended-window mode, v2.36+).
    2.  ``backend.fit(X_flat, y_seq, sequence_data=X_seq, ...)``.
    3.  ``backend.predict_sequence(inference_window)`` on a window
        rebuilt with ``build_inference_window`` whose last past row is
        the row immediately preceding the forecast — same as
        ``_forecast_with_cached``.

A held-out tail is reserved before any windows are constructed, so the
training set never sees the evaluation period.

Tree backends (LightGBM) are exercised via the same flat feature path
the production add-on uses (``build_features``), so they can be compared
on identical inputs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml_forecast_lab.features import (
    build_features,
    build_inference_window,
    compute_known_future_features,
    create_sliding_windows,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public configuration                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class HarnessCfg:
    target_col: str = "y"
    window_size: int = 48              # 24 h at 30-min steps
    horizon: int = 48                  # 24 h forecast
    extended_window: bool = True       # v2.36+ path
    use_solar_covariates: bool = True
    add_temporal_in_future: bool = True
    use_revin: bool = True
    output_activation: str = "linear"
    daily_loss_weight: float = 0.0
    optimiser: str = "adamw"
    epochs: int = 30                   # short — synthetic data converges fast
    batch_size: int = 64
    interval_minutes: int = 30
    seed: int = 0
    lat_lon: Tuple[float, float] = (52.0, -1.0)
    country: Optional[str] = "GB"
    # holdout: how many *days* at the tail to reserve for evaluation
    holdout_days: int = 14
    # When True, the harness applies PF8 (softplus default) and PF9
    # (daily_loss_weight = 0.5) — mirrors what main.py's
    # _apply_output_activation / _resolve_daily_loss_weight do in
    # production for non-negative targets.
    target_is_nonnegative: bool = False


@dataclass
class BackendResult:
    name: str
    pred: np.ndarray            # (n_windows, H)
    truth: np.ndarray           # (n_windows, H)
    truth_idx: List[pd.DatetimeIndex]   # one per window
    mae: float
    mae_per_horizon: np.ndarray  # (H,)
    mae_per_hour: pd.Series      # MAE indexed by hour-of-day 0..23
    note: str = ""


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #

def _make_future_features(idx: pd.DatetimeIndex,
                          cfg: HarnessCfg,
                          include_solar: bool) -> pd.DataFrame:
    return compute_known_future_features(
        idx,
        add_temporal=cfg.add_temporal_in_future,
        country=cfg.country,
        solar_lat_lon=cfg.lat_lon if include_solar else None,
        include_sun_elevation=include_solar,
        include_clear_sky_ghi=include_solar,
    )


def _train_neural_backend(
    cls,
    df_train: pd.DataFrame,
    cfg: HarnessCfg,
    cov_cols: List[str],
    *,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, List[str], int]:
    """Returns (model, channel_names, past_window_size)."""
    horizon_steps = list(range(1, cfg.horizon + 1))
    future_df = (
        _make_future_features(df_train.index, cfg, include_solar=cfg.use_solar_covariates)
        if cfg.extended_window else None
    )
    X, y, channel_names = create_sliding_windows(
        df_train, cfg.target_col,
        window_size=cfg.window_size,
        covariate_cols=cov_cols if cov_cols else None,
        add_temporal=True,
        horizon_steps=horizon_steps,
        future_features_df=future_df,
    )
    # Build the backend with the experiment's knobs. PF8/PF9: when
    # target_is_nonnegative is set and the user hasn't overridden, swap
    # in relu (or zscore for LSTM) and a default daily_loss_weight —
    # mirrors main.py's _resolve_output_activation / _resolve_daily_loss_weight.
    model_name = getattr(cls, "__name__", "").lower().replace("model", "")
    if cfg.target_is_nonnegative:
        if cfg.output_activation == "linear":
            # LSTM gets zscore (its own normalisation path); other
            # backends get relu (clamp at 0, no positive bias).
            out_act = "zscore" if model_name == "lstm" else "relu"
        else:
            out_act = cfg.output_activation
        dlw = cfg.daily_loss_weight if cfg.daily_loss_weight > 0 else 0.5
    else:
        out_act = cfg.output_activation
        dlw = cfg.daily_loss_weight
    kwargs: Dict[str, Any] = dict(
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        use_revin=cfg.use_revin,
        output_activation=out_act,
        daily_loss_weight=dlw,
        optimiser=cfg.optimiser,
        patience=max(8, cfg.epochs // 4),
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    model = cls(**{k: v for k, v in kwargs.items() if k in cls.__init__.__code__.co_varnames})
    # We pass sequence_data so the backend's flat-feature reshape is bypassed.
    # X_flat is a stub of correct n_samples but irrelevant content — fit()
    # only uses it for shape inference when sequence_data is absent.
    # past_window_size mirrors the production training path's seq_kwargs
    # (main.py:_retrain_and_cache) so backends can apply the v2.37 PF1+
    # fixes during training.
    X_flat = np.zeros((X.shape[0], 1), dtype=np.float32)
    fit_kwargs: Dict[str, Any] = {"sequence_data": X}
    if cfg.extended_window:
        fit_kwargs["past_window_size"] = cfg.window_size
    model.fit(X_flat, y, **fit_kwargs)
    return model, channel_names, cfg.window_size


def _infer_neural_backend(
    model,
    df_history: pd.DataFrame,
    cfg: HarnessCfg,
    cov_cols: List[str],
    past_window_size: int,
) -> np.ndarray:
    """Inference on a single tail window — mirrors _forecast_with_cached."""
    last_ts = df_history.index[-1]
    future_idx = pd.date_range(
        start=last_ts + pd.Timedelta(minutes=cfg.interval_minutes),
        periods=cfg.horizon,
        freq=f"{cfg.interval_minutes}min",
    )
    future_df = (
        _make_future_features(future_idx, cfg, include_solar=cfg.use_solar_covariates)
        if cfg.extended_window else None
    )
    X, _ = build_inference_window(
        df_history, cfg.target_col,
        window_size=past_window_size,
        covariate_cols=cov_cols if cov_cols else None,
        add_temporal=True,
        future_features_df=future_df,
    )
    pred = model.predict_sequence(X)
    pred = np.asarray(pred).reshape(1, -1)
    return pred  # shape (1, H)


def _rolling_evaluate_neural(
    model,
    df_full: pd.DataFrame,
    cfg: HarnessCfg,
    cov_cols: List[str],
    past_window_size: int,
    holdout_start_idx: int,
    n_eval_windows: int,
    backend_name: str,
) -> BackendResult:
    """Run the model on a stride-1 rolling window over the holdout tail."""
    preds = []
    truths = []
    idxs = []
    H = cfg.horizon
    step = 1  # one prediction per timestep
    # We need at least past_window_size rows of "history" before the
    # forecast origin. We pull from df_full, treating slot i as the most
    # recent past row.
    for k in range(n_eval_windows):
        anchor = holdout_start_idx + k * step
        if anchor + H > len(df_full):
            break
        if anchor < past_window_size:
            continue
        history = df_full.iloc[anchor - past_window_size: anchor]
        truth_slice = df_full[cfg.target_col].iloc[anchor: anchor + H].values
        truth_idx = df_full.index[anchor: anchor + H]
        pred = _infer_neural_backend(
            model, history, cfg, cov_cols, past_window_size
        ).reshape(-1)
        preds.append(pred[:H])
        truths.append(truth_slice)
        idxs.append(truth_idx)
    pred_arr = np.vstack(preds)            # (n_windows, H)
    truth_arr = np.vstack(truths)
    mae = float(np.mean(np.abs(pred_arr - truth_arr)))
    mae_per_horizon = np.mean(np.abs(pred_arr - truth_arr), axis=0)
    # MAE per hour-of-day at the *target* (truth) timestamps.
    pairs = []
    for w in range(len(idxs)):
        for h in range(H):
            pairs.append((idxs[w][h].hour, abs(pred_arr[w, h] - truth_arr[w, h])))
    df_hour = pd.DataFrame(pairs, columns=["hour", "ae"])
    mae_per_hour = df_hour.groupby("hour")["ae"].mean()
    return BackendResult(
        name=backend_name,
        pred=pred_arr,
        truth=truth_arr,
        truth_idx=idxs,
        mae=mae,
        mae_per_horizon=mae_per_horizon,
        mae_per_hour=mae_per_hour,
    )


def _train_eval_tree(
    df_full: pd.DataFrame,
    cfg: HarnessCfg,
    cov_cols: List[str],
    holdout_start_idx: int,
    n_eval_windows: int,
) -> BackendResult:
    """LightGBM positive control via build_features + recursive multi-step.

    Mirrors the production tree path closely enough for this investigation:
    train on a single-step target with build_features, then forecast each
    horizon step recursively, refreshing future-position temporal/solar
    features at each step.
    """
    from ml_forecast_lab.models.lightgbm_backend import LightGBMModel
    target = df_full[cfg.target_col]
    feats = build_features(
        df_full[[cfg.target_col] + cov_cols].rename(columns={cfg.target_col: "y"}),
        target_col="y",
        interval_minutes=cfg.interval_minutes,
        country=cfg.country,
    )
    full = feats.copy()
    full["__y"] = target.values
    full = full.dropna()
    train_mask = np.array([t < df_full.index[holdout_start_idx] for t in full.index])
    Xtr = full.drop(columns=["__y"]).values.astype(np.float32)
    ytr = full["__y"].values.astype(np.float32)
    feat_cols = list(full.drop(columns=["__y"]).columns)
    model = LightGBMModel(num_leaves=31, learning_rate=0.05, n_estimators=200)
    model.fit(Xtr[train_mask], ytr[train_mask])

    # Recursive multi-step forecast at each anchor in the holdout.
    H = cfg.horizon
    preds = []
    truths = []
    idxs = []
    for k in range(n_eval_windows):
        anchor = holdout_start_idx + k
        if anchor + H > len(df_full):
            break
        # Build a single-row feature vector at each forecast step by
        # rolling the lag buffer forward with the prediction.
        lag_buf = list(target.iloc[:anchor].values[-(48 * 2 + 1):])  # 2 days
        pred_h = np.zeros(H, dtype=np.float32)
        for h in range(H):
            t = df_full.index[anchor + h]
            row = {}
            row["hour_of_day"] = t.hour
            row["day_of_week"] = t.dayofweek
            row["is_weekend"] = int(t.dayofweek >= 5)
            row["month"] = t.month
            row["day_of_month"] = t.day
            row["hour_sin"] = float(np.sin(2 * np.pi * t.hour / 24))
            row["hour_cos"] = float(np.cos(2 * np.pi * t.hour / 24))
            row["dow_sin"] = float(np.sin(2 * np.pi * t.dayofweek / 7))
            row["dow_cos"] = float(np.cos(2 * np.pi * t.dayofweek / 7))
            for c in cov_cols:
                row[c] = float(df_full[c].iloc[anchor + h])
            # Lags from the buffer + previous predictions.
            ghi_t = float(df_full["clear_sky_ghi"].iloc[anchor + h]) if "clear_sky_ghi" in df_full.columns else 1.0
            for lag in range(1, 13):
                if lag <= h:
                    val = float(pred_h[h - lag])
                else:
                    val = lag_buf[-(lag - h)]
                if "clear_sky_ghi" in df_full.columns:
                    # Gate by past ghi to match build_features.
                    past_t = df_full.index[anchor + h - lag] if (anchor + h - lag) >= 0 else None
                    past_ghi = float(df_full["clear_sky_ghi"].iloc[anchor + h - lag]) if past_t is not None else 0.0
                    if past_ghi <= 0:
                        val = 0.0
                row[f"y_lag_{lag}"] = val
            # Periodic lags + rolling stats — approximate using buf+preds.
            steps_per_day = 1440 // cfg.interval_minutes
            for d in (1, 2):
                lag = steps_per_day * d
                if lag <= h:
                    val = float(pred_h[h - lag])
                else:
                    idx_back = -(lag - h)
                    val = lag_buf[idx_back] if abs(idx_back) <= len(lag_buf) else 0.0
                if "clear_sky_ghi" in df_full.columns:
                    past_ghi = float(df_full["clear_sky_ghi"].iloc[anchor + h - lag]) if (anchor + h - lag) >= 0 else 0.0
                    if past_ghi <= 0:
                        val = 0.0
                row[f"y_lag_{lag}"] = val
            # Rolling stats — use the recent buffer including predictions
            recent = lag_buf + list(pred_h[:h])
            for window in (6, 24, 72):
                if len(recent) >= window:
                    w = np.asarray(recent[-window:])
                    row[f"y_rolling_mean_{window}"] = float(w.mean())
                    row[f"y_rolling_std_{window}"] = float(w.std())
                    row[f"y_rolling_max_{window}"] = float(w.max())
                else:
                    row[f"y_rolling_mean_{window}"] = 0.0
                    row[f"y_rolling_std_{window}"] = 0.0
                    row[f"y_rolling_max_{window}"] = 0.0
            # diff
            if h == 0:
                row["y_diff_1"] = float(lag_buf[-1] - lag_buf[-2])
            elif h == 1:
                row["y_diff_1"] = float(pred_h[0] - lag_buf[-1])
            else:
                row["y_diff_1"] = float(pred_h[h - 1] - pred_h[h - 2])
            # Interaction features.
            for c in cov_cols:
                row[f"{c}_x_hour_sin"] = row[c] * row["hour_sin"]
                row[f"{c}_x_hour_cos"] = row[c] * row["hour_cos"]
            # Holiday — set to 0 (synthetic data isn't a holiday)
            row.setdefault("is_holiday", 0)
            xrow = np.zeros((1, len(feat_cols)), dtype=np.float32)
            for ci, name in enumerate(feat_cols):
                xrow[0, ci] = float(row.get(name, 0.0))
            yh = float(model.predict(xrow).reshape(-1)[0])
            pred_h[h] = max(0.0, yh)
        preds.append(pred_h)
        truths.append(df_full[cfg.target_col].iloc[anchor: anchor + H].values)
        idxs.append(df_full.index[anchor: anchor + H])
    pred_arr = np.vstack(preds)
    truth_arr = np.vstack(truths)
    mae = float(np.mean(np.abs(pred_arr - truth_arr)))
    mae_per_horizon = np.mean(np.abs(pred_arr - truth_arr), axis=0)
    pairs = []
    for w in range(len(idxs)):
        for h in range(H):
            pairs.append((idxs[w][h].hour, abs(pred_arr[w, h] - truth_arr[w, h])))
    df_hour = pd.DataFrame(pairs, columns=["hour", "ae"])
    mae_per_hour = df_hour.groupby("hour")["ae"].mean()
    return BackendResult(
        name="lightgbm",
        pred=pred_arr,
        truth=truth_arr,
        truth_idx=idxs,
        mae=mae,
        mae_per_horizon=mae_per_horizon,
        mae_per_hour=mae_per_hour,
    )


# --------------------------------------------------------------------------- #
# Public driver                                                               #
# --------------------------------------------------------------------------- #

def neural_backend_classes() -> Dict[str, Any]:
    """Return {name: class} for the multi-horizon neural backends we test."""
    from ml_forecast_lab.models.nlinear_backend import NLinearModel
    from ml_forecast_lab.models.dlinear_backend import DLinearModel
    from ml_forecast_lab.models.sparsetsf_backend import SparseTSFModel
    from ml_forecast_lab.models.lstm_backend import LSTMModel
    from ml_forecast_lab.models.cnn_backend import CNNModel
    return {
        "nlinear": NLinearModel,
        "dlinear": DLinearModel,
        "sparsetsf": SparseTSFModel,
        "lstm": LSTMModel,
        "cnn": CNNModel,
    }


def run_neural_and_tree(
    df: pd.DataFrame,
    cfg: HarnessCfg,
    backends: Optional[List[str]] = None,
    train_subset_days: int = 60,
    n_eval_windows: int = 24,
) -> Dict[str, BackendResult]:
    """Train every selected neural backend + the tree control on `df`.

    Returns a dict {backend_name: BackendResult}.
    """
    np.random.seed(cfg.seed)
    try:
        import torch
        torch.manual_seed(cfg.seed)
    except Exception:
        pass

    cov_cols = [c for c in df.columns if c != cfg.target_col]
    if not cfg.use_solar_covariates:
        cov_cols = [c for c in cov_cols if c not in ("sun_elevation", "clear_sky_ghi")]

    steps_per_day = 1440 // cfg.interval_minutes
    holdout_start_idx = len(df) - cfg.holdout_days * steps_per_day
    # Train on a recent contiguous slice that ends right before the holdout.
    train_n_rows = min(train_subset_days * steps_per_day, holdout_start_idx)
    df_train = df.iloc[holdout_start_idx - train_n_rows: holdout_start_idx]

    results: Dict[str, BackendResult] = {}
    classes = neural_backend_classes()
    if backends is None:
        backends = list(classes.keys())
    for name in backends:
        if name == "lightgbm":
            continue
        cls = classes[name]
        try:
            model, ch_names, past_ws = _train_neural_backend(
                cls, df_train, cfg, cov_cols,
            )
            res = _rolling_evaluate_neural(
                model, df, cfg, cov_cols, past_ws,
                holdout_start_idx=holdout_start_idx,
                n_eval_windows=n_eval_windows,
                backend_name=name,
            )
            results[name] = res
        except Exception as e:  # do not abort the sweep on one failure
            logger.exception("backend %s failed", name)
            results[name] = BackendResult(
                name=name, pred=np.zeros((0, cfg.horizon)),
                truth=np.zeros((0, cfg.horizon)),
                truth_idx=[], mae=float("nan"),
                mae_per_horizon=np.zeros(cfg.horizon),
                mae_per_hour=pd.Series(dtype=float),
                note=f"FAILED: {e!r}",
            )

    if "lightgbm" in (backends or []):
        try:
            results["lightgbm"] = _train_eval_tree(
                df, cfg, cov_cols, holdout_start_idx, n_eval_windows,
            )
        except Exception as e:
            logger.exception("lightgbm failed")
            results["lightgbm"] = BackendResult(
                name="lightgbm", pred=np.zeros((0, cfg.horizon)),
                truth=np.zeros((0, cfg.horizon)),
                truth_idx=[], mae=float("nan"),
                mae_per_horizon=np.zeros(cfg.horizon),
                mae_per_hour=pd.Series(dtype=float),
                note=f"FAILED: {e!r}",
            )
    return results
