"""
Main application entry point and event loop for ML Forecast Lab.

Orchestrates the loading of configuration, initialisation of components,
and management of the main forecast/benchmark loop with FastAPI web server.
"""

import asyncio
import dataclasses
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import uvicorn

from ml_forecast_lab.training_events import TrainingEvent, TrainingEventBus

logger = logging.getLogger(__name__)

# Age-based forecast_log retention (audit F5). Must stay above the UI's
# largest analytics window (90 days) so pruning never truncates a view
# the user can select. Pruned per-experiment on the retrain cadence.
FORECAST_LOG_RETENTION_DAYS = 120


def _resolve_output_activation(exp_cfg, model_name: str = '') -> str:
    """
    Resolve ``ExperimentCfg.output_activation`` to a concrete activation.

    The ``'auto'`` alias picks a sensible default based on the model backend
    and target's physical nature:

    - LSTM                                            → ``'zscore'`` (target
      z-score normalisation with linear head, denormalised at inference;
      conditions gradients across widely-varying target scales and is the
      best general default for recurrent backends)
    - Other neural                                    → ``'linear'``
      (unbounded output; non-negativity for cumulative / non-negative
      targets is enforced by the publish-time clamp instead — see the
      ``target_is_nonnegative`` clip in ``_forecast_with_cached`` /
      ``_run_production_inference``)

    History: v2.37 PF8 originally picked ``'relu'`` for non-negative
    targets, which produced the user-reported flat-zero forecast via
    dying-ReLU; v2.37.1 switched to ``'softplus'`` on the theory that
    its non-zero gradient everywhere immunises against the collapse.
    Empirically it does not: with the (since-removed) cumulative-loss
    term inactive — which is every real deployment, since
    ``daily_loss_weight`` defaulted to 0 — the zero-valued half of a
    PV/demand target keeps pushing the pre-activation down until
    float32 softplus saturates to exactly 0 with a vanishing gradient,
    and the daytime signal can no longer recover. The integration suite
    (``tests/integration/test_pv_forecast_pipeline.py``) pins this:
    softplus collapses flat, a linear head trains cleanly. v2.41.0
    therefore resolves 'auto' to 'linear' and moves the non-negativity
    guarantee to the publish boundary, where a clamp costs nothing and
    cannot interfere with optimisation. Users can still pin
    ``output_activation: softplus`` explicitly.
    """
    act = getattr(exp_cfg, 'output_activation', 'auto')
    if act == 'auto':
        # LSTM's encoder state is highly sensitive to target scale; z-scoring
        # the target keeps loss-landscape curvature bounded regardless of the
        # raw target magnitude. Makes the LSTM work well out-of-the-box on
        # targets ranging from small fractions to large cumulative values.
        if model_name == 'lstm':
            return 'zscore'
        return 'linear'
    return act


def _apply_output_activation(model, exp_cfg) -> None:
    """
    Apply the experiment's ``output_activation`` to a freshly-created model.

    Silently no-ops for tree backends (lightgbm/xgboost) that don't expose
    the parameter — ``set_params`` raises ``ValueError`` for unknown kwargs
    and we treat that as "not a neural model, skip".
    """
    if not getattr(model, 'is_neural', False):
        return
    try:
        model.set_params(output_activation=_resolve_output_activation(
            exp_cfg, getattr(model, 'name', '')
        ))
    except (ValueError, TypeError):
        # Backend doesn't support output_activation (older checkpoint being
        # loaded pre-v2.11.0, or a neural backend not yet migrated).
        pass


def _apply_idle_value_fill(result: 'pd.DataFrame', exp_cfg) -> int:
    """
    Fill NaN slots in ``result['y']`` with an idle value before the
    final ``dropna``. Returns the number of rows filled.

    Two paths, both gated on ``target_is_nonnegative=True``:

    1. **Solar / irradiance** (``clear_sky_ghi`` or ``sun_elevation``
       in result columns): use solar physics to identify night-time
       slots and fill ONLY those with ``exp_cfg.idle_value`` (default
       0.0). Daytime NaN is preserved — a clear_sky_ghi > 0 row with
       a NaN target is a real sensor outage worth surfacing via the
       dropna step rather than silently masking.

    2. **Non-solar non-negative** (no physics features present, but
       ``idle_value`` explicitly set): fill ALL remaining NaN slots
       with ``idle_value``. Covers EV chargers / batteries / pumps
       that report ``unavailable`` when idle. The user asserts
       "this sensor is at <idle_value> when it's not reporting"; if
       a daytime outage shows up it'll be masked the same way. This
       path is opt-in only — when ``idle_value`` is None (default)
       the original drop-on-NaN behaviour is preserved.

    Background: HA's recorder is delta-storage based — when a sensor
    sits at a constant value (or goes ``unavailable``) for hours, it
    records one transition and then nothing. The default
    ``gap_handling='interpolate'`` only fills gaps up to
    ``gap_max_minutes`` (90), so 10-14h overnight gaps stay NaN and
    ``result.dropna()`` deletes every idle row. The model then trains
    on an active-only window and has no signal for the idle state —
    at inference it produces non-zero predictions during the idle
    period and the daily shape is distorted.

    ``log_transform=True`` is fine for ``idle_value=0`` (log(1+0)=0).
    For non-zero ``idle_value`` with log_transform, the caller is
    expected to have already log-transformed the target series; the
    written value is taken as-is (no auto log-transform here — the
    user knows their data better than we do).
    """
    if not getattr(exp_cfg, "target_is_nonnegative", False):
        return 0
    if "y" not in result.columns:
        return 0
    if result["y"].isna().sum() == 0:
        return 0

    idle_value = getattr(exp_cfg, "idle_value", None)
    has_physics = (
        "clear_sky_ghi" in result.columns
        or "sun_elevation" in result.columns
    )

    if has_physics:
        # Solar path: use physics to gate the fill so daytime outages
        # remain visible to dropna. idle_value overrides the default 0.0
        # for users with a measurable inverter standby.
        fill_value = idle_value if idle_value is not None else 0.0
        if "clear_sky_ghi" in result.columns:
            night_mask = result["clear_sky_ghi"].fillna(0) <= 0
        else:
            # -0.833° is the standard astronomical horizon (accounts
            # for atmospheric refraction). Anything below is night.
            night_mask = result["sun_elevation"].fillna(-90) < -0.833
        fill_mask = result["y"].isna() & night_mask
    else:
        # Non-solar path: ALL NaN → idle_value, but only when the
        # user has explicitly declared what idle means. No physics
        # to distinguish "natural idle" from "real outage" — that's
        # the trade-off the idle_value field surfaces.
        if idle_value is None:
            return 0
        fill_value = idle_value
        fill_mask = result["y"].isna()

    n_filled = int(fill_mask.sum())
    if n_filled > 0:
        result.loc[fill_mask, "y"] = fill_value
    return n_filled


# Backwards-compatibility shim — v2.37.3 introduced
# _apply_solar_night_fill as the public helper; v2.37.4 generalised
# it. Keep the old name as an alias so external imports / pinned
# tests don't break.
_apply_solar_night_fill = _apply_idle_value_fill


def _cov_column_name(cov_cfg, all_covs: Optional[list] = None) -> str:
    """Canonical column name for a covariate's series in the
    ``combined`` dataframe.

    Default: the last component of the entity_id (e.g.
    ``sensor.solcast_pv_forecast_forecast_today`` → ``solcast_pv_forecast_forecast_today``).

    When the same entity is configured multiple times in the same
    experiment (v2.38.2+ — supports e.g. consuming both
    ``cloud_coverage`` and ``temperature`` from
    ``weather.met_office_balsham``), the column name gets a
    ``__<value_key>`` suffix so the per-covariate columns don't
    collide in the dataframe. ``all_covs`` is the full list of
    covariate configs for the experiment; if omitted, suffix logic
    falls back to "single instance" semantics (backwards-compatible
    with existing experiments).

    v2.39.3: when two same-entity configs both lack a distinguishing
    ``future_value_key`` AND share the default ``future_attribute``,
    the previous code returned the bare base for both — silent
    DataFrame column collision (second assignment overwrote first).
    The fallback now appends a positional ``__N`` suffix so each
    row resolves to a distinct column and emits a warning so the
    user can disambiguate explicitly.
    """
    base = cov_cfg.entity.split('.')[-1]
    if all_covs is None:
        return base
    # Count how many configs share this entity. One occurrence → bare
    # base name (preserves cache-meta channel parity for existing
    # experiments). More than one → disambiguate with value_key.
    same_entity = [c for c in all_covs if c.entity == cov_cfg.entity]
    if len(same_entity) <= 1:
        return base
    value_key = getattr(cov_cfg, 'future_value_key', None)
    if value_key:
        return f"{base}__{value_key}"
    # Fallback: no value_key set. Use future_attribute if it's
    # distinctive (e.g. ``hourly`` / ``daily``); otherwise we'd
    # collide on bare base, so disambiguate by position within the
    # same-entity group and warn so the user knows to set
    # future_value_key explicitly.
    attr = getattr(cov_cfg, 'future_attribute', 'forecast') or 'forecast'
    if attr != 'forecast':
        return f"{base}__{attr}"
    try:
        position = same_entity.index(cov_cfg)
    except ValueError:
        position = 0
    if position == 0:
        return base
    logger.warning(
        "Covariate %r is configured multiple times with no distinguishing "
        "future_value_key / future_attribute — assigning positional column "
        "name %r to avoid a silent DataFrame collision. Set "
        "future_value_key on each row to disambiguate explicitly.",
        cov_cfg.entity, f"{base}__{position + 1}",
    )
    return f"{base}__{position + 1}"


def _assess_model_instability(
    fold_metrics_list: list,
    primary_metrics: list,
) -> tuple:
    """Flag a model whose per-fold error spread hides a blow-up fold.

    The composite mean rank is outlier-robust: a single catastrophic
    fold costs only one last-place finish, so a model that is strong on
    most folds but catastrophic on one can out-rank a consistently
    mediocre model (see docs/RANKING_NOTES.md). This inspects the
    per-fold values of the first available metric in ``primary_metrics``
    (production metric first, then mase / mae fallbacks) and flags two
    regimes:

      - catastrophic fold: the worst fold is >= 10x the median fold —
        the "great on 4 folds, blew up on 1" pattern the mean rank hides;
      - high dispersion: std >= mean (coefficient of variation >= 1.0),
        i.e. fold-to-fold spread is at least as large as the mean.

    Returns ``(unstable: bool, reason: Optional[str])``. ``reason`` is
    ``None`` when stable or when there are fewer than 2 finite folds to
    judge from (a flag off a single fold would be meaningless).
    """
    for metric in primary_metrics:
        if not metric:
            continue
        vals = []
        for fm in fold_metrics_list:
            if not fm:
                continue
            v = fm.get(metric)
            if v is None:
                continue
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                continue
            vals.append(float(v))
        if len(vals) < 2:
            continue  # not enough data for THIS metric — try the next
        arr = np.asarray(vals, dtype=float)
        mean = float(arr.mean())
        if mean <= 0:
            return False, None  # degenerate / all-zero error — nothing to flag
        std = float(arr.std())
        median = float(np.median(arr))
        worst = float(arr.max())
        blowup = worst / median if median > 0 else float("inf")
        if blowup >= 10.0:
            return True, (
                f"worst fold ({worst:.3g}) is {blowup:.0f}x its median fold "
                f"({median:.3g}) on {metric} — one fold blew up; mean rank "
                f"hides this"
            )
        cv = std / mean
        if cv >= 1.0:
            return True, (
                f"{metric} swings ±{cv * 100:.0f}% across folds "
                f"(mean {mean:.3g} +/- {std:.3g}) — unstable fold-to-fold"
            )
        return False, None  # assessed on the primary metric; it's stable
    return False, None


def _collect_train_future_covariates(
    combined: 'pd.DataFrame', exp_cfg
) -> Dict[str, 'pd.Series']:
    """Build the ``future_covariate_values`` dict for training-side
    callers of ``compute_known_future_features``. Pulls in-sample
    historical values from ``combined`` for every covariate declared
    with ``role`` in ``{future, both}``. Returns an empty dict when
    no such covariates are configured.

    Centralised so the production cache path, benchmark holdout
    path, and legacy production-inference path all agree on which
    columns reach the neural head's future positions. Without this,
    benchmark scores were measured on a different feature surface
    from production training and the "best model" pick was unfair
    to backends that benefit from horizon-anchored covariate signal.
    """
    out: Dict[str, 'pd.Series'] = {}
    covs = getattr(exp_cfg, 'covariates', None)
    if not covs:
        return out
    for cov_cfg in covs:
        if getattr(cov_cfg, 'role', None) not in ('future', 'both'):
            continue
        cov_name = _cov_column_name(cov_cfg, all_covs=covs)
        if cov_name in combined.columns:
            out[cov_name] = combined[cov_name]
    return out


def _holdout_display_from_windows(y_p: 'np.ndarray', target_len: int) -> 'np.ndarray':
    """Build a full-length per-point holdout display series from a windowed
    multi-horizon prediction array.

    Neural backends predict the holdout via sliding windows: ``y_p`` is
    ``(n_windows, H)``, and the ``h=1`` column gives a 1-step prediction for
    the first ``n_windows`` holdout points. The trailing ``H-1`` points have
    no ``h=1`` window (the window would need future rows past the end of the
    slice), so a naive ``y_p[:, 0]`` stops ``max_horizon-1`` points short —
    very visible when ``future_periods`` is large (e.g. 96 ⇒ ~16 h gap),
    where tree models that ``predict()`` per point cover the whole holdout.

    Those tail points *were* predicted: the LAST formed window's
    ``h=2..H`` outputs land exactly on them (and at the shortest horizon
    available for each). Filling from there lets neural models span the full
    holdout — essential for the daily-cumulative view whose per-day sum needs
    every point. Display-only (does not affect leaderboard metrics, which
    come from the CV folds).
    """
    out = np.full(target_len, np.nan, dtype=np.float32)
    y_p = np.asarray(y_p)
    if y_p.ndim == 1:
        n = min(y_p.shape[0], target_len)
        out[:n] = y_p[:n]
        return out
    n = min(y_p.shape[0], target_len)
    out[:n] = y_p[:n, 0]
    tail_len = target_len - n
    if tail_len > 0 and y_p.shape[0] >= 1 and y_p.shape[1] > 1:
        last = y_p[-1]  # the last window's full h=1..H forecast
        avail = min(tail_len, last.shape[0] - 1)
        if avail > 0:
            out[n:n + avail] = last[1:1 + avail]
    return out


def _apply_patience(model, exp_cfg, overrides=None) -> None:
    """v2.40.12: apply the per-experiment ``patience`` Setting to a model.

    Honours every backend that exposes ``self.patience`` (every neural
    backend, plus LightGBM / XGBoost / CatBoost after the v2.40.12 fix).
    Skipped when the user already pinned ``patience`` via per-model
    ``overrides`` / ``model_params``, and skipped when the experiment
    setting is ``None`` (which means "use the backend's constructor
    default" — 20 for neural, 50 for tree).
    """
    if overrides and 'patience' in overrides:
        return
    p = getattr(exp_cfg, 'patience', None)
    if p is None:
        return
    if not hasattr(model, 'patience'):
        return
    model.patience = int(p)


def _apply_experiment_neural_params(model, exp_cfg, overrides=None) -> None:
    """
    Propagate experiment-level neural training settings to a model.

    Currently propagates ``loss_fn`` and ``optimiser`` from ``exp_cfg``
    so every code path that instantiates a
    neural model (benchmark CV, production training, Tuning trials,
    holdout refits, Covariate Analysis) honours the user's Settings
    selection — not just the main CV loop. Without this helper, secondary
    paths would silently train with the backend's default ``loss_fn='mse'``
    / ``optimiser='adamw'`` regardless of what the user picked in
    Settings.

    Silently no-ops for tree backends (``hasattr`` guard).
    Skips any param already present in ``overrides`` so user-provided
    ``model_overrides`` and Optuna-swept params take priority.

    Parameters
    ----------
    model : ForecastModel
        Freshly-created model instance.
    exp_cfg : ExperimentCfg
        Experiment configuration to read settings from.
    overrides : dict, optional
        Dict of param names already applied (e.g. Optuna trial params or
        ``model_overrides``). Entries here are NOT overwritten, preserving
        the caller's intent.
    """
    if not getattr(model, 'is_neural', False):
        return
    overrides = overrides or {}
    for attr in ('loss_fn', 'optimiser'):
        if attr in overrides:
            continue
        value = getattr(exp_cfg, attr, None)
        if value is None:
            continue
        if not hasattr(model, attr):
            continue
        try:
            model.set_params(**{attr: value})
        except (ValueError, TypeError):
            # Backend doesn't support this kwarg (old checkpoint, or not yet
            # migrated) — silently skip rather than break the whole run.
            pass

    _apply_patience(model, exp_cfg, overrides)


class MLForecastLabApp:
    """
    Main application controller for ML Forecast Lab.

    Manages the lifecycle of the forecasting engine, including configuration
    loading, component initialisation, web server management, and the main
    update loop that drives forecasting and benchmarking.
    """

    def __init__(self):
        """Initialise the application."""
        self.config = None
        self.ha_interface = None
        self.history_db = None
        self.covariate_resolver = None
        self.model_registry = None
        self.web_app = None
        self.server = None
        self.running = False
        self._update_running = False
        # Per-experiment forecast lock. Using a global flag meant two
        # experiments scheduled in the same main_loop tick could both pass
        # the `not running` check (create_task doesn't execute eagerly),
        # and later the one that finishes first would flip the flag False
        # while the other was still running — breaking the guarantee.
        self._forecast_running: Dict[str, bool] = {}
        self.last_update = None
        self.benchmarks_to_run = set()
        # Cached trained models for fast forecast cycles
        # Key: experiment name → dict with model, feature_cols, combined, etc.
        self._cached_models = {}
        # Track running asyncio tasks for stop-training support
        self._running_tasks: Dict[str, asyncio.Task] = {}
        # Cooperative cancel flags for executor-thread training (audit
        # F10): task.cancel() can't interrupt a thread, so stop-training
        # also sets this event, which the epoch callbacks check.
        self._cancel_events: Dict[str, threading.Event] = {}
        # Sequential retrain queue — prevents parallel training
        self._retrain_queue: asyncio.Queue = asyncio.Queue()
        self._retrain_consumer_running = False
        # Global training lock — ensures only one training operation
        # (benchmark, retrain, tuning, or covariate analysis) runs at a
        # time across all code paths. v2.41.0 closed the gap where
        # tuning and covariate analysis trained outside it (audit F7).
        self._training_lock: asyncio.Lock = asyncio.Lock()
        # Track config file mtime so we only log on real changes (the timer
        # loop reloads config every 30s; we don't want a log line each time).
        self._last_config_path: Optional[Path] = None
        self._last_config_mtime: Optional[float] = None
        # Debug bundle dumper (created lazily on first retrain that has
        # debug_save_training_dumps=True). Rooted at <config_dir>/debug/
        # so the bundles sit next to mlfl.yaml and are visible via HA's
        # Samba / File Editor add-ons without an extra path mapping.
        self._debug_dumper = None
        # Cached site location (lat, lon) from HA's /api/config — used for
        # deterministic solar physics covariates. Fetched lazily on first use.
        self._site_location: Optional[tuple[float, float]] = None
        # Per-target-entity cache of the source sensor's HA
        # ``unit_of_measurement``, used to auto-inherit the forecast
        # unit when the experiment leaves ``units`` blank. Fetched lazily
        # on first publish for an entity and reused for the process
        # lifetime (a source sensor's unit effectively never changes;
        # a restart re-resolves it). Keyed by target_entity so multiple
        # experiments on the same sensor share one lookup.
        self._source_unit_cache: Dict[str, str] = {}
        # Last *successfully* resolved non-empty source unit per target
        # entity. Unlike ``_source_unit_cache`` this is NEVER invalidated on
        # retrain — it's the fallback that stops a transient HA fetch failure
        # (or the source sensor being momentarily ``unavailable``) during the
        # post-retrain re-resolve from republishing an empty unit and making
        # HA flag a unit_of_measurement change.
        self._source_unit_last_good: Dict[str, str] = {}
        # Resolved runtime-resource limits actually applied to this process
        # (CPU thread cap, nice value). Surfaced on the System page so the
        # user can verify their settings took effect.
        self._applied_cpu_threads: Optional[int] = None
        self._applied_nice: Optional[int] = None
        # Strong refs for fire-and-forget tasks scheduled from main_loop and
        # start_web_server. Same rationale as AppState._background_tasks:
        # asyncio holds only weak refs to running tasks, so without this set
        # a coroutine scheduled via create_task can be GC'd before its
        # exception is logged.
        self._background_tasks: set = set()

    def _spawn(self, coro) -> asyncio.Task:
        """Schedule *coro* and retain a strong reference until completion.

        Exceptions raised inside *coro* surface via the logger rather than
        vanishing into a GC'd task. Callers should not await the returned
        task unless they need cancellation semantics.
        """
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)

        def _on_done(t: asyncio.Task) -> None:
            self._background_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.error("Background task failed: %s", exc, exc_info=exc)

        task.add_done_callback(_on_done)
        return task

    async def load_config(self, config_path: Optional[Path] = None):
        """
        Load configuration from YAML file.

        Searches multiple paths in order:
        1. Explicit config_path (if provided)
        2. /addon_configs/ml_forecast_lab/mlfl.yaml
        3. /config/mlfl.yaml
        4. Bundled mlfl.yaml (for development)
        5. Falls back to stub config
        """
        import glob

        search_paths = []
        if config_path is not None:
            search_paths.append(Path(config_path))
        search_paths.extend([
            Path("/addon_configs/ml_forecast_lab/mlfl.yaml"),
            Path("/config/mlfl.yaml"),
            Path(__file__).parent.parent / "mlfl.yaml",
        ])

        # Also check HA's hashed slug paths (e.g. /addon_configs/47b4bbf0_ml_forecast_lab/).
        # The glob is anchored to HA's 8-hex-character prefix so a community
        # fork with a slug like `psweens_ml_forecast_lab` can't hijack the
        # lookup and route reads/writes to the wrong directory.
        for match in glob.glob(
            "/addon_configs/[0-9a-f][0-9a-f][0-9a-f][0-9a-f]"
            "[0-9a-f][0-9a-f][0-9a-f][0-9a-f]_ml_forecast_lab/mlfl.yaml"
        ):
            search_paths.insert(0, Path(match))

        found_path = None
        for p in search_paths:
            if p.exists():
                found_path = p
                break

        if found_path is None:
            logger.warning(
                f"Configuration file not found in any of: "
                f"{[str(p) for p in search_paths]}"
            )
            logger.info("Creating stub configuration...")
            self.config = self._create_stub_config()
        else:
            try:
                from ml_forecast_lab.config import load_config

                new_config = load_config(found_path)
                # Only log at INFO when the config actually changes (first load
                # or mtime change). The timer loop calls load_config() every
                # 30 seconds to pick up UI edits — logging every reload would
                # spam the log file. Steady-state reloads stay at DEBUG.
                try:
                    current_mtime = found_path.stat().st_mtime
                except OSError:
                    current_mtime = None

                first_load = self._last_config_path is None
                changed = (
                    self._last_config_path != found_path
                    or self._last_config_mtime != current_mtime
                )
                if first_load:
                    logger.info(f"Configuration loaded from {found_path}")
                elif changed:
                    logger.info(f"Configuration reloaded from {found_path}")
                else:
                    logger.debug(f"Configuration reloaded (no changes) from {found_path}")

                self._last_config_path = found_path
                self._last_config_mtime = current_mtime
                self.config = new_config
            except Exception as e:
                if self.config is not None:
                    # Periodic reload failed — keep the existing good config
                    # instead of replacing with stub (which would make all
                    # experiments disappear). This can happen if the YAML is
                    # briefly unreadable during a concurrent write.
                    logger.warning(
                        f"Failed to reload configuration (keeping current): {e}"
                    )
                else:
                    # First load — no config to fall back to
                    logger.error(f"Failed to load configuration: {e}", exc_info=True)
                    self.config = self._create_stub_config()

    def _create_stub_config(self):
        """Create a minimal stub configuration for testing."""
        from ml_forecast_lab.config import AppConfig, ExperimentCfg

        return AppConfig(
            update_every_minutes=5,
            timezone="UTC",
            experiments=[
                ExperimentCfg(
                    name="test_experiment",
                    target_entity="sensor.test_value",
                    days_history=7,
                    interval_minutes=30,
                    models_enabled=["lightgbm"],
                    cv_folds=3,
                )
            ],
        )

    async def initialise_components(self):
        """
        Initialise all application components.

        Includes HAInterface, HistoryDB, CovariateResolver, and ModelRegistry.
        """
        logger.info("Initialising application components...")

        try:
            # Initialise HAInterface
            from ml_forecast_lab.ha_interface import HAInterface

            self.ha_interface = HAInterface()
            logger.info("HAInterface initialised")

            # Initialise HistoryDB
            from ml_forecast_lab.db import HistoryDB

            db_path = Path("/data/ml_forecast_lab/history.db")
            self.history_db = HistoryDB(db_path)
            self.history_db.ensure_forecast_log_table()
            self.history_db.ensure_external_forecast_log_table()
            self.history_db.ensure_benchmark_table()
            logger.info(f"HistoryDB initialised at {db_path}")

            # Initialise CovariateResolver
            from ml_forecast_lab.covariates import CovariateResolver

            self.covariate_resolver = CovariateResolver(
                self.ha_interface, history_db=self.history_db,
            )
            logger.info("CovariateResolver initialised (covariate history cached)")

            # Initialise ModelRegistry with all available backends
            from ml_forecast_lab.models.registry import ModelRegistry
            from ml_forecast_lab.models.lightgbm_backend import LightGBMModel
            from ml_forecast_lab.models.xgboost_backend import XGBoostModel
            from ml_forecast_lab.models.lstm_backend import LSTMModel
            from ml_forecast_lab.models.cnn_backend import CNNModel

            self.model_registry = ModelRegistry()
            self.model_registry.register("lightgbm", LightGBMModel)
            self.model_registry.register("xgboost", XGBoostModel)
            self.model_registry.register("lstm", LSTMModel)
            self.model_registry.register("cnn", CNNModel)

            # Register optional backends
            _optional_backends = [
                ("catboost", "catboost_backend", "CatBoostModel"),
                ("gru", "gru_backend", "GRUModel"),
                ("dlinear", "dlinear_backend", "DLinearModel"),
                ("nlinear", "nlinear_backend", "NLinearModel"),
                ("fits", "fits_backend", "FITSModel"),
                ("nbeats", "nbeats_backend", "NBeatsModel"),
                ("nhits", "nhits_backend", "NHiTSModel"),
                ("tide", "tide_backend", "TiDEModel"),
                ("tsmixer", "tsmixer_backend", "TSMixerModel"),
                ("timemixer", "timemixer_backend", "TimeMixerModel"),
                ("sparsetsf", "sparsetsf_backend", "SparseTSFModel"),
                ("patchtst", "patchtst_backend", "PatchTSTModel"),
                ("itransformer", "itransformer_backend", "iTransformerModel"),
                ("crossformer", "crossformer_backend", "CrossformerModel"),
                ("timesnet", "timesnet_backend", "TimesNetModel"),
                ("tft", "tft_backend", "TFTModel"),
                ("timexer", "timexer_backend", "TimeXerModel"),
                ("moderntcn", "moderntcn_backend", "ModernTCNModel"),
                ("seasonal_naive", "seasonal_naive_backend", "SeasonalNaiveModel"),
                ("arima", "statsforecast_backend", "ARIMAModel"),
                ("ets", "statsforecast_backend", "ETSModel"),
                ("theta", "statsforecast_backend", "ThetaModel"),
                # Zero-shot foundation models. Optional heavy deps
                # (chronos-forecasting / granite-tsfm) — skipped cleanly
                # here when the packages aren't installed, e.g. on armv7
                # where the transformers stack has no wheels.
                ("chronos_bolt", "chronos_bolt_backend", "ChronosBoltModel"),
                ("ttm", "ttm_backend", "TTMModel"),
            ]
            for _name, _module, _cls_name in _optional_backends:
                try:
                    _mod = __import__(f"ml_forecast_lab.models.{_module}", fromlist=[_cls_name])
                    self.model_registry.register(_name, getattr(_mod, _cls_name))
                except Exception as e:
                    logger.debug(f"{_name} not available: {e}")

            logger.info(f"ModelRegistry initialised with {len(self.model_registry.list_available())} backends")

        except Exception as e:
            logger.error(f"Failed to initialise components: {e}", exc_info=True)
            logger.info("Continuing with partial initialisation...")

    async def start_web_server(self, host: str = "0.0.0.0", port: int = 5052):
        """
        Start the FastAPI web server in background.

        Parameters
        ----------
        host : str
            Host to listen on (default: 0.0.0.0)
        port : int
            Port to listen on (default: 5052)
        """
        logger.info(f"Starting web server on {host}:{port}...")

        try:
            from ml_forecast_lab.web.app import create_app

            self.web_app = create_app()

            # Initialise experiment statuses in web app state
            for exp_cfg in self.config.experiments:
                from ml_forecast_lab.web.app import ExperimentStatus

                _pub_name = exp_cfg.publish_name or exp_cfg.name
                # Restore the user's UI selection from YAML so
                # `/select-model` clicks survive add-on restarts. Before
                # this field existed, selected_model was re-initialised to
                # None on every boot and the next benchmark auto-picked
                # its top-ranked model — which users experienced as "the
                # Results tab forgets which model I picked". Fall back
                # to production_model when selected_model is unset so
                # legacy configs still get a sensible default.
                _selected = (
                    getattr(exp_cfg, "selected_model", None)
                    or getattr(exp_cfg, "production_model", None)
                )
                status = ExperimentStatus(
                    name=exp_cfg.name,
                    target_entity=exp_cfg.target_entity,
                    mode=exp_cfg.mode,
                    selected_model=_selected,
                    last_benchmark_status="pending",
                    next_forecast_in_seconds=self.config.forecast_every_minutes * 60,
                    next_retrain_in_seconds=int(self.config.retrain_every_hours * 3600),
                    next_update_in_seconds=self.config.forecast_every_minutes * 60,
                    publish_entity=f"sensor.{exp_cfg.publish_prefix}{_pub_name}_forecast",
                )
                self.web_app.state.appstate.experiment_statuses[exp_cfg.name] = (
                    status
                )

            # Restore benchmark results from SQLite
            if self.history_db:
                from ml_forecast_lab.web.app import BenchmarkResult as WebBenchmarkResult
                saved = self.history_db.load_all_benchmark_results()
                # Build lookup of current model config per experiment
                _exp_models = {
                    c.name: set(c.models_enabled)
                    for c in self.config.experiments
                }
                for exp_name, json_str in saved.items():
                    try:
                        br = WebBenchmarkResult.model_validate_json(json_str)
                        current_models = _exp_models.get(exp_name, set())

                        # Filter out models that are no longer enabled
                        valid_models = [
                            m for m in br.models
                            if m.name in current_models
                        ]
                        if valid_models and len(valid_models) < len(br.models):
                            br.models = valid_models
                            # Recalculate best from remaining models
                            br.best_model_name = min(
                                valid_models, key=lambda m: m.rank
                            ).name
                            logger.info(
                                f"  Filtered stale models from saved benchmark "
                                f"for {exp_name} — kept {[m.name for m in valid_models]}"
                            )
                        elif not valid_models and br.models:
                            # None of the saved models are still enabled — discard
                            logger.info(
                                f"  Discarding stale benchmark for {exp_name} — "
                                f"none of the saved models are still enabled"
                            )
                            self.history_db.delete_benchmark_result(exp_name)
                            continue

                        self.web_app.state.appstate.benchmark_results[exp_name] = br
                        # Restore best_model / selected_model in status
                        st = self.web_app.state.appstate.experiment_statuses.get(exp_name)
                        if st and br.best_model_name:
                            st.best_model = br.best_model_name
                            if not st.selected_model:
                                st.selected_model = br.best_model_name
                            st.last_benchmark_status = br.status
                            st.last_benchmark_timestamp = br.timestamp
                        logger.info(f"  Restored benchmark results for {exp_name}")
                    except Exception as e:
                        logger.warning(f"  Failed to restore benchmark for {exp_name}: {e}")

            config = uvicorn.Config(
                app=self.web_app,
                host=host,
                port=port,
                log_level="warning",
            )
            self.server = uvicorn.Server(config)

            # Register benchmark callback (triggered from Training tab)
            async def _benchmark_trigger(experiment_name: str):
                # Acquire global training lock so benchmarks and scheduled
                # retrains never overlap.
                async with self._training_lock:
                    # Reload config so UI-edited overrides are picked up
                    await self.load_config()
                    exp_cfg = None
                    for cfg in self.config.experiments:
                        if cfg.name == experiment_name:
                            exp_cfg = cfg
                            break
                    if exp_cfg:
                        try:
                            self.web_app.state.appstate.start_benchmark(experiment_name)
                            await self._run_benchmark(exp_cfg)
                        except Exception as e:
                            logger.error(f"Benchmark failed: {e}", exc_info=True)
                        finally:
                            self.web_app.state.appstate.end_benchmark(experiment_name)
                            self._running_tasks.pop(experiment_name, None)

            self.web_app.state.appstate.benchmark_callback = _benchmark_trigger

            # Register covariate analysis callback
            async def _covariate_analysis_trigger(experiment_name: str, selected_model: str = "all"):
                exp_cfg = None
                for cfg in self.config.experiments:
                    if cfg.name == experiment_name:
                        exp_cfg = cfg
                        break
                if exp_cfg:
                    # Covariate analysis trains models × covariate
                    # combinations — a full training workload. Take the
                    # global training lock so it can't overlap a
                    # scheduled retrain or benchmark (audit F7).
                    async with self._training_lock:
                        try:
                            await self._run_covariate_analysis(exp_cfg, selected_model=selected_model)
                        except Exception as e:
                            logger.error(f"Covariate analysis failed: {e}", exc_info=True)

            self.web_app.state.appstate.covariate_analysis_callback = _covariate_analysis_trigger

            # Register tuning callback
            async def _tuning_trigger(experiment_name: str, model_name: str,
                                      n_trials: int = 30, strategy: str = "tpe",
                                      param_schema: dict = None):
                try:
                    # Tuning runs n_trials × CV fits — a full training
                    # workload. Serialise behind the global training
                    # lock (audit F7: the lock's "all code paths" claim
                    # previously excluded tuning + covariate analysis).
                    async with self._training_lock:
                        await self._run_tuning(
                            experiment_name, model_name, n_trials, strategy, param_schema
                        )
                except Exception as e:
                    logger.error(f"Tuning failed: {e}", exc_info=True)
                    tr = self.web_app.state.appstate.tuning_results.get(experiment_name)
                    if tr:
                        tr.status = "failed"
                        tr.error_message = str(e)

            self.web_app.state.appstate.tuning_callback = _tuning_trigger

            # 'Tune all enabled' sweep — loops _run_tuning over every
            # model in models_enabled. Each iteration overwrites the
            # single-model tuning_results slot (so the live progress
            # UI continues to work) and copies the final result into
            # tune_all_results[experiment] for the summary table.
            async def _tune_all_trigger(experiment_name: str,
                                        n_trials: int = 30,
                                        strategy: str = "tpe"):
                await self.load_config()
                exp_cfg = next(
                    (c for c in self.config.experiments if c.name == experiment_name),
                    None,
                )
                if exp_cfg is None:
                    logger.warning("tune-all: '%s' not in config", experiment_name)
                    return
                models = list(exp_cfg.models_enabled or [])
                if not models:
                    logger.warning("tune-all: no models enabled for '%s'", experiment_name)
                    return
                # Reset the sweep slot at the start of the run.
                self.web_app.state.appstate.tune_all_results[experiment_name] = []
                logger.info(
                    "tune-all: %s — running %d model(s): %s",
                    experiment_name, len(models), models,
                )
                for m_name in models:
                    try:
                        # Per-model lock acquisition (not one hold for
                        # the whole sweep) so a scheduled retrain can
                        # interleave between models rather than waiting
                        # hours for the full tune-all to finish.
                        async with self._training_lock:
                            await self._run_tuning(
                                experiment_name, m_name, n_trials, strategy, None,
                            )
                    except Exception as e:
                        logger.error("tune-all: %s/%s failed: %s",
                                     experiment_name, m_name, e, exc_info=True)
                        continue
                    tr = self.web_app.state.appstate.tuning_results.get(experiment_name)
                    if tr is not None:
                        try:
                            self.web_app.state.appstate.tune_all_results[
                                experiment_name
                            ].append(tr.model_copy(deep=True))
                        except Exception:
                            # model_copy might not exist on older Pydantic
                            # versions; fall back to a shallow snapshot.
                            self.web_app.state.appstate.tune_all_results[
                                experiment_name
                            ].append(tr)
                logger.info(
                    "tune-all: %s complete — %d result(s) captured",
                    experiment_name,
                    len(self.web_app.state.appstate.tune_all_results.get(
                        experiment_name, [],
                    )),
                )

            self.web_app.state.appstate.tune_all_callback = _tune_all_trigger

            # Register retrain callback. All user-initiated retrains
            # (dashboard button, apply-tuning, apply-covariate-best,
            # toggle-mode→production) want to retrain the chosen/production
            # model only, never to kick off a full benchmark — so we go
            # straight to _retrain_and_cache rather than _retrain_single
            # (which also has a lab-mode branch that runs a benchmark).
            async def _retrain_trigger(experiment_name: str):
                # Re-read config so any YAML edits applied just before this
                # call are picked up before the retrain runs.
                await self.load_config()
                exp_cfg = next(
                    (c for c in self.config.experiments if c.name == experiment_name),
                    None,
                )
                if exp_cfg is None:
                    logger.warning(
                        f"Retrain trigger: experiment '{experiment_name}' not found in config"
                    )
                    return
                async with self._training_lock:
                    try:
                        await self._retrain_and_cache(exp_cfg)
                        await self.publish_heartbeat()
                    except Exception as e:
                        logger.error(
                            f"Retrain trigger failed for {experiment_name}: {e}",
                            exc_info=True,
                        )

            self.web_app.state.appstate.retrain_callback = _retrain_trigger

            # Register stop-training callback
            async def _stop_training_trigger(experiment_name: str) -> bool:
                # Set the cooperative cancel flag FIRST: cancelling the
                # asyncio task only abandons the coroutine — the
                # executor thread running model.fit keeps the CPU
                # saturated until it checks this flag (audit F10).
                ev = self._cancel_events.get(experiment_name)
                if ev is not None:
                    ev.set()
                task = self._running_tasks.get(experiment_name)
                # Also check pipeline tasks launched from the web UI
                if not task or task.done():
                    pt = getattr(self.web_app.state.appstate, '_pipeline_tasks', {})
                    task = pt.get(experiment_name)
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass
                    self.web_app.state.appstate.end_benchmark(experiment_name)
                    # Emit pipeline_end event so the Training tab SSE stream closes
                    try:
                        from ml_forecast_lab.training_events import TrainingEventBus, TrainingEvent
                        TrainingEventBus.get_instance().publish(TrainingEvent(
                            event_type="pipeline_end",
                            experiment_name=experiment_name,
                            message="Training cancelled by user",
                        ))
                    except Exception:
                        pass
                    logger.info(f"Cancelled training task for {experiment_name}")
                    return True
                return False

            self.web_app.state.appstate.stop_training_callback = _stop_training_trigger
            self.web_app.state.appstate.history_db = self.history_db

            # Rollback callback for the "Roll back to previous" button on
            # the experiment header. Synchronous swap so the UI's response
            # reflects the new champion immediately.
            def _rollback_trigger(experiment_name: str) -> tuple[bool, Optional[str]]:
                return self._rollback_cached_model(experiment_name)
            self.web_app.state.appstate.rollback_callback = _rollback_trigger

            # Pre-flight data sanity report callback for the Settings tab.
            async def _data_report_trigger(experiment_name: str) -> dict:
                exp_cfg = next(
                    (e for e in self.config.experiments if e.name == experiment_name),
                    None,
                )
                if exp_cfg is None:
                    return {"verdict": "alert", "warnings": ["Experiment not found"], "ok": False}
                return await self.compute_data_report(exp_cfg)
            self.web_app.state.appstate.data_report_callback = _data_report_trigger

            # Cached-model directory accessor — lets the web layer check
            # whether a "previous" version exists for the rollback button
            # without duplicating the slugify logic.
            self.web_app.state.appstate.cached_model_dir = self._cached_model_dir
            # Expose the runtime-resource values actually applied so the
            # System page can show "Applied: N threads, nice X" and the
            # user can verify their cpu_cores / nice_priority took effect.
            self.web_app.state.appstate.applied_cpu_threads = self._applied_cpu_threads
            self.web_app.state.appstate.applied_nice = self._applied_nice

            # Pull HA's configured time zone so the Forecast Accuracy tab
            # can render charts in the TZ where events physically happen,
            # not the viewer's browser TZ. Needed when the user is
            # remote (e.g. on holiday, or living in a different country
            # from the HA instance). Best-effort — if the fetch fails
            # the frontend falls back to browser local.
            try:
                ha_cfg = await self.ha_interface.get_config()
                tz_name = ha_cfg.get("time_zone") if isinstance(ha_cfg, dict) else None
                if tz_name:
                    self.web_app.state.appstate.ha_time_zone = tz_name
                    logger.info(f"HA time zone: {tz_name}")
            except Exception as e:
                logger.warning(f"Could not read HA time_zone from /api/config: {e}")

            # Run in a background task — held via _spawn so a failure here
            # is logged rather than silently swallowed by GC.
            self._spawn(self.server.serve())
            logger.info(f"Web server started successfully on {host}:{port}")

        except Exception as e:
            logger.error(f"Failed to start web server: {e}", exc_info=True)

    async def update_experiment(self, experiment_name: str, is_lab_mode: bool):
        """
        Run update for a single experiment.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment to update
        is_lab_mode : bool
            If True, run full benchmark. If False, run production inference only.
        """
        logger.info(f"Updating experiment: {experiment_name} (mode={'lab' if is_lab_mode else 'production'})")

        try:
            # Find experiment config
            exp_cfg = None
            for cfg in self.config.experiments:
                if cfg.name == experiment_name:
                    exp_cfg = cfg
                    break

            if not exp_cfg:
                logger.error(f"Experiment config not found: {experiment_name}")
                return

            _started_at = datetime.now(timezone.utc)
            if is_lab_mode:
                await self._run_benchmark(exp_cfg)
            else:
                await self._run_production_inference(exp_cfg)

            # Update web app state
            if self.web_app:
                status = self.web_app.state.appstate.experiment_statuses.get(
                    experiment_name
                )
                if status:
                    status.last_benchmark_timestamp = datetime.now(timezone.utc).isoformat()
                    status.last_benchmark_status = "completed"
                    status.last_error = None  # Clear any previous error
                    self.web_app.state.appstate.end_benchmark(experiment_name)

            # Publish lifecycle sensor for HA automations
            if is_lab_mode:
                _duration = (datetime.now(timezone.utc) - _started_at).total_seconds()
                _attrs = {"duration_seconds": round(_duration, 1)}
                br = (self.web_app.state.appstate.benchmark_results.get(experiment_name)
                      if self.web_app else None)
                if br and getattr(br, "best_model_name", None):
                    _attrs["winner"] = br.best_model_name
                await self._publish_lifecycle_sensor(
                    exp_cfg, "benchmark", "completed", _attrs,
                )

        except Exception as e:
            logger.error(
                f"Error updating experiment {experiment_name}: {e}",
                exc_info=True,
            )
            if self.web_app:
                status = self.web_app.state.appstate.experiment_statuses.get(
                    experiment_name
                )
                if status:
                    status.last_benchmark_status = "failed"
                    status.last_error = str(e)
                    self.web_app.state.appstate.end_benchmark(experiment_name)
            try:
                await self._publish_lifecycle_sensor(
                    exp_cfg, "benchmark", "failed", {"error": str(e)[:200]},
                )
            except Exception:
                pass

    async def _get_site_location(self) -> Optional[tuple[float, float]]:
        """
        Return cached (latitude, longitude) from HA's /api/config.

        Fetched once on first call and cached for the lifetime of the app.
        Returns None if the HA interface is unavailable or the config doesn't
        expose coordinates.
        """
        if self._site_location is not None:
            return self._site_location
        if self.ha_interface is None:
            return None
        try:
            ha_cfg = await self.ha_interface.get_config()
        except Exception as e:
            logger.warning(f"Could not fetch HA config for site location: {e}")
            return None
        lat = ha_cfg.get("latitude")
        lon = ha_cfg.get("longitude")
        if lat is None or lon is None:
            logger.warning("HA config does not expose latitude/longitude")
            return None
        try:
            self._site_location = (float(lat), float(lon))
            logger.info(
                f"Site location: lat={self._site_location[0]:.4f}, "
                f"lon={self._site_location[1]:.4f}"
            )
        except (TypeError, ValueError):
            return None
        return self._site_location

    async def _resolve_units(self, exp_cfg) -> str:
        """Resolve the unit to publish on this experiment's HA sensors.

        Precedence:
          1. ``exp_cfg.units`` if the user set it explicitly (YAML wins).
          2. Otherwise auto-inherit the ``unit_of_measurement`` of the
             target sensor from HA — the forecast is a prediction of that
             same sensor, so its unit is the right default (W stays W,
             kWh stays kWh, °C stays °C). This is what makes units "just
             work" for experiments created through the web UI, which has
             no units field.
          3. The last successfully-resolved unit if a fresh fetch comes back
             empty or fails — so a transient HA hiccup (or the source sensor
             being momentarily ``unavailable``) on the post-retrain
             re-resolve doesn't republish an empty unit and make HA flag a
             unit_of_measurement change.
          4. Empty string only if the source genuinely has no unit and we've
             never resolved one — identical to the prior cold-start
             behaviour, never raises.

        The inherited value is cached per target entity so this costs at
        most one extra HA call per entity per process, not one per
        publish cycle.
        """
        if exp_cfg.units:
            return exp_cfg.units

        entity = exp_cfg.target_entity
        cached = self._source_unit_cache.get(entity)
        if cached is not None:
            return cached

        unit = None
        if self.ha_interface:
            try:
                fetched = await self.ha_interface.get_state(
                    entity, attribute="unit_of_measurement", default=None,
                )
                if isinstance(fetched, str) and fetched.strip():
                    unit = fetched.strip()
            except Exception as e:
                logger.debug(
                    f"  Could not auto-resolve unit from {entity}: {e}"
                )

        if unit:
            # Successful resolve — remember it as the authoritative fallback.
            prev_good = self._source_unit_last_good.get(entity)
            if unit != prev_good:
                logger.info(
                    f"  Auto-inherited unit '{unit}' from {entity} "
                    f"for {exp_cfg.name} (no units set in config)"
                )
            self._source_unit_cache[entity] = unit
            self._source_unit_last_good[entity] = unit
            return unit

        # Fetch failed or the source reported no unit. Prefer the last known
        # good unit over republishing "": a transient miss on the
        # post-retrain re-resolve must not clobber a good value and make HA
        # flag a unit_of_measurement change.
        last_good = self._source_unit_last_good.get(entity)
        if last_good:
            logger.debug(
                f"  Unit fetch for {entity} returned empty; keeping "
                f"last-known unit '{last_good}' for {exp_cfg.name}"
            )
            self._source_unit_cache[entity] = last_good
            return last_good

        # Never resolved a unit for this entity — cold-start empty, as before.
        self._source_unit_cache[entity] = ""
        return ""

    async def _prepare_load_subtract_inputs(
        self,
        exp_cfg,
        start: datetime,
        now: datetime,
        freq: str,
    ) -> list:
        """Fetch and preprocess ``load_subtract`` sensors for one experiment.

        Returns a list of ``(cfg_dict, series)`` pairs suitable for passing
        directly to ``apply_load_subtract``. Each series is on the same
        grid (``freq``) as the target load.

        Fetch path mirrors the target-load path in ``_fetch_and_preprocess``
        (raw history via HA REST, tz-naive normalisation), but runs the
        cumulative→interval step based on each entry's explicit
        ``source`` field. ``source='auto'`` defers to the target's
        ``source_is_cumulative``/``reset_daily`` — correct only when the
        subtract sensor has the same semantics as the parent.

        A fetch failure for one sensor does NOT abort the experiment; the
        sensor is skipped and logged. The fail-fast guards live in
        ``apply_load_subtract`` itself, where they can see the aligned
        signal and produce a ratio-based diagnostic.
        """
        import dataclasses as _dc

        from ml_forecast_lab.ha_interface import normalise_history
        from ml_forecast_lab.preprocessing import (
            cumulative_to_interval,
            resample_to_grid,
        )

        inputs: list = []
        for sub_cfg in exp_cfg.load_subtract:
            entity_id = sub_cfg.entity_id
            try:
                raw_records = await self.ha_interface.get_history(
                    entity_id, start, now,
                )
                sub_df = normalise_history(raw_records)
                if sub_df.empty:
                    logger.warning(
                        f"    ⊘ load_subtract[{entity_id}]: no history returned "
                        f"from HA — skipping"
                    )
                    continue

                # Normalise tz-naive to match target-load conventions. Done up
                # front so apply_load_subtract sees matching index tz-awareness.
                if (
                    hasattr(sub_df["ds"].dtype, "tz")
                    and sub_df["ds"].dt.tz is not None
                ):
                    sub_df["ds"] = sub_df["ds"].dt.tz_localize(None)

                sub_df = sub_df.set_index("ds").sort_index()
                sub_series = sub_df["value"]

                # Apply cumulative semantics explicitly. source='auto' inherits
                # from the parent target — convenience, not correctness: only
                # safe when subtract sensor matches the parent's cumulative type.
                source = sub_cfg.source
                if source == "auto":
                    if exp_cfg.source_is_cumulative:
                        source = (
                            "cumulative_daily" if exp_cfg.reset_daily
                            else "cumulative_monotonic"
                        )
                    else:
                        source = "interval"

                if source == "cumulative_daily":
                    sub_series = cumulative_to_interval(
                        sub_series,
                        interval_minutes=exp_cfg.interval_minutes,
                        reset_daily=True,
                    )
                elif source == "cumulative_monotonic":
                    sub_series = cumulative_to_interval(
                        sub_series,
                        interval_minutes=exp_cfg.interval_minutes,
                        reset_daily=False,
                    )
                # source == "interval": no transformation needed

                # Resample to target grid. Use 'sum' for converted-from-cumulative
                # signals (interval energy should aggregate), 'mean' otherwise.
                sub_method = (
                    "sum" if source.startswith("cumulative") else "mean"
                )
                sub_series = resample_to_grid(
                    sub_series, freq=freq, method=sub_method,
                )

                # Pass the full SubtractCfg (as dict) so apply_load_subtract
                # sees all robustness-relevant fields (scale, on_missing,
                # max_fraction_*). asdict is cheap and avoids coupling the
                # preprocessing module to the dataclass.
                inputs.append((_dc.asdict(sub_cfg), sub_series))

                logger.info(
                    f"    ↓ load_subtract[{entity_id}]: "
                    f"{len(raw_records)} raw → {len(sub_series)} aligned "
                    f"(source={source}, on_missing={sub_cfg.on_missing})"
                )
            except Exception as e:
                # Non-fatal per-sensor: log and skip. apply_load_subtract will
                # still run on whichever sensors succeeded.
                logger.warning(
                    f"    ✗ load_subtract[{entity_id}] fetch failed: {e}"
                )

        return inputs

    def _log_load_subtract_audit(self, exp_cfg, audit: dict) -> None:
        """Emit a human-readable summary of load_subtract audit stats.

        Format matches the boxed / ✓ marker style used elsewhere in the
        runner. Keeps the log readable when several sensors are subtracted.
        """
        load_total = audit.get("load_total_kwh", 0.0)
        sub_total = audit.get("subtract_total_kwh", 0.0)
        n_clipped = audit.get("n_clipped_rows", 0)
        clipped_pct = audit.get("clipped_pct", 0.0)
        share_pct = (100.0 * sub_total / load_total) if load_total > 0 else 0.0

        logger.info(
            f"  ✓ load_subtract applied: "
            f"load={load_total:.2f} kWh, "
            f"subtracted={sub_total:.2f} kWh ({share_pct:.1f}% of load), "
            f"clipped={n_clipped} rows ({clipped_pct:.2f}%)"
        )
        for sensor in audit.get("per_sensor", []):
            gap = ""
            if sensor.get("gap_start"):
                gap = (
                    f", gap={sensor['gap_start'][:10]}→"
                    f"{sensor['gap_end'][:10]}"
                )
            dropped = ""
            if sensor.get("rows_dropped", 0) > 0:
                dropped = f", dropped={sensor['rows_dropped']}"
            logger.info(
                f"      · {sensor['entity_id']}: "
                f"sum={sensor['sum_kwh']:.2f} kWh, "
                f"missing={sensor['rows_missing']}"
                f"{dropped}"
                f", max_frac={sensor['max_fraction']:.2f}, "
                f"violations={sensor['violation_rows']} "
                f"({sensor['violation_pct']:.2f}%)"
                f"{gap}"
            )

    async def _analyse_entity_history(
        self,
        entity_id: str,
        start,
        now,
        interval_minutes: int,
        max_increment=None,
        source_is_cumulative: bool = False,
        big_gap_multiplier: int = 4,
    ) -> dict:
        """Per-entity history fetch + summary.

        Returns the same per-entity stats the target analysis used to
        compute inline; factored out so the data-report endpoint can
        run identical checks against the target and each covariate
        ("does this sensor look healthy enough to feed the model?").

        Cache + delta-fetch strategy mirrors `_fetch_and_preprocess`.
        Threshold defaults match the historical target checks:
          - recorder stale if > 4× interval
          - coverage warning < 70 %
          - big gap = > big_gap_multiplier × interval
          - NaN warning > 5 %

        Returns dict — never raises; on HA failure / empty data the
        caller decides what verdict to attach.
        """
        from ml_forecast_lab.ha_interface import normalise_history

        table_name = (
            self.history_db.safe_table_name(entity_id)
            if self.history_db else None
        )

        df = pd.DataFrame(columns=["ds", "value"])
        cache_rows = 0
        if self.history_db and table_name:
            try:
                cached = self.history_db.get_history(table_name)
                if not cached.empty:
                    cached = cached.rename(columns={"y": "value"})
                    start_naive = start.replace(tzinfo=None)
                    cached = cached[cached["ds"] >= start_naive]
                    cache_rows = int(len(cached))
                    if cache_rows > 0:
                        df = cached
            except Exception as e:
                logger.debug("data-report cache read failed for %s: %s", entity_id, e)

        fetch_start = df["ds"].max() if len(df) > 0 else start
        try:
            if hasattr(fetch_start, "tzinfo") and fetch_start.tzinfo is None:
                fetch_start = fetch_start.tz_localize("UTC")
        except (AttributeError, TypeError):
            pass

        ha_rows = 0
        fetch_error = None
        try:
            raw = await self.ha_interface.get_history(entity_id, fetch_start, now)
            new_df = normalise_history(raw)
            if not new_df.empty:
                if hasattr(new_df["ds"].dtype, "tz") and new_df["ds"].dt.tz is not None:
                    new_df["ds"] = new_df["ds"].dt.tz_localize(None)
                ha_rows = int(len(new_df))
                if len(df) > 0:
                    df = pd.concat([df, new_df], ignore_index=True)
                    df = df.drop_duplicates(subset=["ds"], keep="last")
                else:
                    df = new_df
        except Exception as e:
            fetch_error = str(e)
            logger.warning("data-report HA fetch failed for %s: %s", entity_id, e)

        if df.empty:
            return {
                "entity_id": entity_id,
                "verdict": "no-data",
                "rows_total": 0,
                "rows_cache": cache_rows,
                "rows_fetched": ha_rows,
                "warnings": (
                    ["HA fetch error: " + fetch_error] if fetch_error
                    else ["No history rows found for this entity"]
                ),
            }

        df = df.sort_values("ds").reset_index(drop=True)
        df["ds"] = pd.to_datetime(df["ds"])

        last_ts = pd.Timestamp(df["ds"].max())
        if last_ts.tzinfo is not None:
            last_ts = last_ts.tz_convert(None)
        now_naive = pd.Timestamp(now.replace(tzinfo=None))
        recorder_age_min = float((now_naive - last_ts).total_seconds()) / 60.0

        expected_step = pd.Timedelta(minutes=interval_minutes)
        expected_rows = int(
            max(1, (now_naive - df["ds"].min()).total_seconds() / expected_step.total_seconds())
        )
        coverage_pct = round(min(100.0, 100.0 * len(df) / expected_rows), 1) if expected_rows else None

        gaps_min = (df["ds"].diff().dt.total_seconds() / 60.0).dropna()
        biggest_gap_min = float(gaps_min.max()) if len(gaps_min) else 0.0
        big_gap_threshold_min = interval_minutes * big_gap_multiplier
        big_gap_count = int((gaps_min > big_gap_threshold_min).sum())

        series = pd.to_numeric(df["value"], errors="coerce")
        non_null = series.dropna()
        nan_pct = round(100.0 * (1.0 - len(non_null) / max(1, len(series))), 1)

        if len(non_null):
            v_min = float(non_null.min())
            v_max = float(non_null.max())
            v_median = float(non_null.median())
            v_std = float(non_null.std())
        else:
            v_min = v_max = v_median = v_std = None

        zero_run = 0
        max_zero_run = 0
        for v in non_null.values:
            if v == 0:
                zero_run += 1
                if zero_run > max_zero_run:
                    max_zero_run = zero_run
            else:
                zero_run = 0

        max_inc_hits = None
        if source_is_cumulative and max_increment:
            diffs = series.diff().dropna()
            max_inc_hits = int((diffs > max_increment).sum())

        warnings = []
        verdict = "ok"
        if recorder_age_min > interval_minutes * 4:
            warnings.append(
                f"Recorder is {recorder_age_min:.0f} min behind wall-clock "
                f"(expected ≤ {interval_minutes * 2} min)"
            )
            verdict = "warning"
        if coverage_pct is not None and coverage_pct < 70:
            warnings.append(
                f"Only {coverage_pct:.0f}% of the expected rows are present "
                f"({len(df)} / ~{expected_rows})"
            )
            verdict = "warning"
        if big_gap_count > 0:
            warnings.append(
                f"{big_gap_count} gap(s) of >{big_gap_threshold_min:.0f} min — "
                f"biggest is {biggest_gap_min / 60:.1f} h"
            )
            if biggest_gap_min > interval_minutes * 24:
                verdict = "alert"
        if nan_pct > 5:
            warnings.append(f"{nan_pct:.1f}% of rows have non-numeric values")
            verdict = "warning" if verdict == "ok" else verdict
        if max_zero_run > interval_minutes / 5 and max_zero_run > 10:
            warnings.append(
                f"Longest zero-value run is {max_zero_run} consecutive samples "
                f"(may be legitimate for solar / off-state sensors)"
            )

        return {
            "entity_id": entity_id,
            "verdict": verdict,
            "rows_total": int(len(df)),
            "rows_cache": cache_rows,
            "rows_fetched": ha_rows,
            "rows_expected": expected_rows,
            "coverage_pct": coverage_pct,
            "recorder_age_minutes": round(recorder_age_min, 1),
            "first_ts": df["ds"].min().isoformat(),
            "last_ts": last_ts.isoformat(),
            "biggest_gap_minutes": round(biggest_gap_min, 1),
            "big_gap_count": big_gap_count,
            "big_gap_threshold_minutes": big_gap_threshold_min,
            "nan_pct": nan_pct,
            "value_min": v_min,
            "value_max": v_max,
            "value_median": v_median,
            "value_std": v_std,
            "max_zero_run_samples": int(max_zero_run),
            "max_increment_hits": max_inc_hits,
            "max_increment_config": max_increment,
            "warnings": warnings,
        }

    async def compute_data_report(self, exp_cfg) -> dict:
        """Pre-flight data sanity report for an experiment.

        Fetches the raw target history the same way the benchmark would
        (SQLite cache + HA delta fetch) and summarises it without
        running the full pipeline — so the user can spot recorder
        gaps, flatlines, max-increment hits, and missing-value rates
        BEFORE spending an hour on a benchmark.

        v2.35.0: also runs the same checks over every configured
        covariate entity so a broken covariate (gaps, wrong units,
        dead sensor) doesn't waste a benchmark either. The covariate
        verdicts are returned in a separate ``covariates`` list and
        rolled into the top-level verdict (a covariate alert escalates
        the experiment's verdict to "warning" — covariate failures are
        non-fatal because the benchmark can train without them, but
        worth flagging).
        """
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=exp_cfg.days_history)

        # Target — same checks as before, just running through the helper.
        target = await self._analyse_entity_history(
            exp_cfg.target_entity, start, now,
            interval_minutes=exp_cfg.interval_minutes,
            max_increment=exp_cfg.max_increment,
            source_is_cumulative=exp_cfg.source_is_cumulative,
        )

        # Covariates — same helper, no max-increment / cumulative
        # semantics (those only apply to the target). Empty list when
        # the experiment has no covariates configured.
        covariate_reports = []
        for cov in (exp_cfg.covariates or []):
            try:
                entity_id = cov.entity
            except AttributeError:
                entity_id = cov.get("entity") if isinstance(cov, dict) else None
            if not entity_id:
                continue
            cov_report = await self._analyse_entity_history(
                entity_id, start, now,
                interval_minutes=exp_cfg.interval_minutes,
            )
            cov_report["role"] = getattr(cov, "role", None) or (
                cov.get("role") if isinstance(cov, dict) else None
            )
            covariate_reports.append(cov_report)

        # Roll covariate verdicts into the experiment-level verdict.
        # Target verdict still drives the headline; covariate issues
        # bump us to at least "warning" because the benchmark can
        # technically train without them, but a sensible user wants
        # to know.
        target_verdict = target.get("verdict", "no-data")
        rolled = target_verdict
        for cov in covariate_reports:
            cv = cov.get("verdict", "ok")
            if cv in ("alert", "no-data") and rolled == "ok":
                rolled = "warning"
            elif cv == "warning" and rolled == "ok":
                rolled = "warning"

        # Preserve the v2.34.x response shape: target fields are
        # promoted to the top level so the existing frontend renderer
        # keeps working unchanged. Per-entity fields live under
        # ``covariates`` and a top-level ``target`` mirror.
        if target_verdict == "no-data":
            return {
                "experiment": exp_cfg.name,
                "target_entity": exp_cfg.target_entity,
                "verdict": rolled if rolled != "ok" else "no-data",
                "ok": False,
                "rows_total": 0,
                "rows_cache": target.get("rows_cache", 0),
                "rows_fetched": target.get("rows_fetched", 0),
                "warnings": target.get("warnings", []),
                "covariates": covariate_reports,
                "target": target,
                "checked_at": now.isoformat(),
            }

        return {
            "experiment": exp_cfg.name,
            "target_entity": exp_cfg.target_entity,
            "verdict": rolled,
            "ok": rolled != "alert",
            "rows_total": target["rows_total"],
            "rows_cache": target["rows_cache"],
            "rows_fetched": target["rows_fetched"],
            "rows_expected": target["rows_expected"],
            "coverage_pct": target["coverage_pct"],
            "recorder_age_minutes": target["recorder_age_minutes"],
            "first_ts": target["first_ts"],
            "last_ts": target["last_ts"],
            "biggest_gap_minutes": target["biggest_gap_minutes"],
            "big_gap_count": target["big_gap_count"],
            "big_gap_threshold_minutes": target["big_gap_threshold_minutes"],
            "nan_pct": target["nan_pct"],
            "target_min": target["value_min"],
            "target_max": target["value_max"],
            "target_median": target["value_median"],
            "target_std": target["value_std"],
            "max_zero_run_samples": target["max_zero_run_samples"],
            "max_increment_hits": target["max_increment_hits"],
            "max_increment_config": target["max_increment_config"],
            "interval_minutes": exp_cfg.interval_minutes,
            "warnings": target["warnings"],
            "covariates": covariate_reports,
            "target": target,
            "checked_at": now.isoformat(),
        }

    async def _fetch_and_preprocess(self, exp_cfg) -> Optional[pd.DataFrame]:
        """
        Fetch history and preprocess for an experiment.

        Returns DataFrame with DatetimeIndex and 'y' column containing the
        preprocessed target values, ready for feature engineering.
        """
        from ml_forecast_lab.ha_interface import normalise_history
        from ml_forecast_lab.preprocessing import (
            cumulative_to_interval,
            resample_to_grid,
            clip_outliers,
            apply_log_transform,
            apply_load_subtract,
            LoadSubtractError,
        )

        now = datetime.now(timezone.utc)
        start = now - timedelta(days=exp_cfg.days_history)
        freq = f"{exp_cfg.interval_minutes}min"
        table_name = self.history_db.safe_table_name(exp_cfg.target_entity) if self.history_db else None

        # Initialise empty so the cache miss + full-fetch path is well-
        # defined. Without this `df` is only bound inside the cache-hit
        # branch below, and the `if len(df) > 0` check further down
        # raises UnboundLocalError when the cache is empty or all
        # cached rows fall outside `days_history` (e.g. on a first
        # benchmark, or when the user widens history). Regression from
        # the v2.33.1 database-flag removal cleanup.
        df = pd.DataFrame(columns=["ds", "value"])

        # --- Try SQLite cache first ---
        # v2.33.1: actuals caching is unconditional. The previous
        # `exp_cfg.database` gate was removed — the field was a footgun
        # (off → entire Forecast Accuracy view silently broken) and the
        # cost is negligible (~72 KB / experiment for a 30-day window,
        # bounded by the cleanup call below).
        if self.history_db and table_name:
            # Offloaded: the DB lock may be held by a long analytics
            # read in a worker thread; waiting for it inline would
            # block the whole event loop (audit F9).
            cached_df = await asyncio.to_thread(
                self.history_db.get_history, table_name,
            )
            if not cached_df.empty:
                # Rename 'y' back to 'value' for consistency
                cached_df = cached_df.rename(columns={"y": "value"})
                # Ensure tz-naive for comparison (SQLite stores naive, start is tz-aware)
                start_naive = start.replace(tzinfo=None)
                cached_df = cached_df[cached_df["ds"] >= start_naive]
                if len(cached_df) > 0:
                    df = cached_df
                    logger.info(
                        f"  Loaded {len(df)} cached records for {exp_cfg.target_entity}"
                    )

        # --- Fetch delta from HA API ---
        if len(df) > 0:
            # Only fetch records newer than our latest cached record
            last_cached = df["ds"].max()
            # Ensure tz-aware for HA API
            if hasattr(last_cached, 'tzinfo') and last_cached.tzinfo is None:
                last_cached = last_cached.tz_localize("UTC")
            fetch_start = last_cached
        else:
            fetch_start = start

        raw_records = await self.ha_interface.get_history(
            exp_cfg.target_entity, fetch_start, now
        )
        new_df = normalise_history(raw_records)

        # Normalise all timestamps to tz-naive UTC for consistency
        if not new_df.empty and hasattr(new_df["ds"].dtype, "tz") and new_df["ds"].dt.tz is not None:
            new_df["ds"] = new_df["ds"].dt.tz_localize(None)

        if not new_df.empty:
            if len(df) > 0:
                # Merge: append new records, deduplicate by timestamp
                df = pd.concat([df, new_df], ignore_index=True)
                df = df.drop_duplicates(subset=["ds"], keep="last").sort_values("ds").reset_index(drop=True)
                logger.info(
                    f"  Fetched {len(new_df)} new records from HA API "
                    f"(total: {len(df)})"
                )
            else:
                df = new_df
                logger.info(
                    f"  Fetched {len(df)} records for {exp_cfg.target_entity} "
                    f"({exp_cfg.days_history} days, {start.strftime('%d %b')} to {now.strftime('%d %b %H:%M')})"
                )

        if df.empty:
            raise ValueError(
                f"No history data for {exp_cfg.target_entity}"
            )

        # --- Store in SQLite cache ---
        # v2.33.1: unconditional — the `exp_cfg.database` gate was
        # removed (see the matching comment on the cache-read above).
        if self.history_db and table_name:
            inserted = await asyncio.to_thread(
                self.history_db.store_history, table_name, df,
            )
            if inserted > 0:
                logger.info(f"  Cached {inserted} new records in SQLite")

            # Cleanup old records beyond max_age
            oldest = now - timedelta(days=exp_cfg.max_age)
            await asyncio.to_thread(
                self.history_db.cleanup, table_name, oldest,
            )

        # --- Carry-forward when recorder has gone quiet -------------------
        # HA's recorder dedups identical state writes, so a sensor whose
        # value doesn't change for hours leaves no new history rows even
        # though the entity is alive and reporting. Without this guard,
        # `last_ts = df["ds"].max()` falls hours behind wall-clock and the
        # downstream forecast horizon (`ds_future = last_ts + i*interval`)
        # publishes timestamps in the past — the holiday/empty-tank case
        # for a `cumulative` + `reset_daily` daily counter is the canonical
        # trigger. Synthesise samples from `last_ts + interval` up to
        # `now`, all carrying the live state, so `last_ts ≈ now` and the
        # rest of the pipeline (lag features, cumulative seed, forecast
        # anchoring) sees a frame consistent with current wall-clock.
        # Synthetics are NOT persisted — they're regenerated from the
        # live state each cycle so a real source resumption immediately
        # supersedes them.
        recorder_last_ts = pd.Timestamp(df["ds"].max())
        if recorder_last_ts.tzinfo is not None:
            recorder_last_ts = recorder_last_ts.tz_convert(None)
        now_naive = pd.Timestamp(
            now.replace(tzinfo=None) if now.tzinfo is not None else now
        )
        state_age = now_naive - recorder_last_ts
        stale_threshold = pd.Timedelta(minutes=exp_cfg.interval_minutes * 2)
        if state_age > stale_threshold:
            try:
                live_raw = await self.ha_interface.get_state(
                    exp_cfg.target_entity, default=None,
                )
            except Exception as e:
                live_raw = None
                logger.debug(
                    f"  Carry-forward: get_state failed for "
                    f"{exp_cfg.target_entity}: {e}"
                )
            live_val: Optional[float] = None
            if live_raw not in (None, "", "unknown", "unavailable"):
                try:
                    live_val = float(live_raw)
                except (TypeError, ValueError):
                    live_val = None

            if live_val is not None:
                synth_start = recorder_last_ts + pd.Timedelta(
                    minutes=exp_cfg.interval_minutes
                )
                synth_index = pd.date_range(
                    start=synth_start,
                    end=now_naive,
                    freq=f"{exp_cfg.interval_minutes}min",
                )
                if len(synth_index) > 0:
                    synth_df = pd.DataFrame({
                        "ds": synth_index,
                        "value": float(live_val),
                    })
                    df = pd.concat([df, synth_df], ignore_index=True)
                    age_secs = state_age.total_seconds()
                    age_str = (
                        f"{age_secs / 60:.0f}m"
                        if age_secs < 7200
                        else f"{age_secs / 3600:.1f}h"
                    )
                    logger.warning(
                        f"  ⚠ {exp_cfg.target_entity} recorder gap "
                        f"{age_str} (last={recorder_last_ts}, now="
                        f"{now_naive.strftime('%H:%M')}) — carried forward "
                        f"live value {live_val} for {len(synth_index)} "
                        f"synthetic ticks (not persisted to cache)"
                    )

        # --- Set DatetimeIndex ---
        df = df.set_index("ds").sort_index()
        series = df["value"]

        # --- Cumulative to interval ---
        if exp_cfg.source_is_cumulative:
            series = cumulative_to_interval(
                series,
                interval_minutes=exp_cfg.interval_minutes,
                reset_daily=exp_cfg.reset_daily,
                max_increment=exp_cfg.max_increment,
            )

        # --- Resample to regular grid ---
        resample_method = "sum" if exp_cfg.source_is_cumulative else "mean"
        series = resample_to_grid(
            series, freq=freq, method=resample_method,
            gap_handling=exp_cfg.gap_handling,
            gap_max_minutes=exp_cfg.gap_max_minutes,
        )

        # --- Load subtract (optional) ---
        #
        # Runs BEFORE clip_outliers so that outlier bounds are computed on the
        # adjusted (baseline) signal rather than on the raw signal which may
        # contain subtractable spikes (EV charging, solar-divert dumps). If
        # clipping ran first, those spikes would elevate the 99.5-percentile
        # bound and mute real baseline peaks after subtraction.
        if getattr(exp_cfg, "load_subtract", None):
            try:
                subtract_inputs = await self._prepare_load_subtract_inputs(
                    exp_cfg, start, now, freq,
                )
            except Exception as e:
                logger.error(
                    f"  ✗ load_subtract fetch failed for {exp_cfg.name}: {e}"
                )
                subtract_inputs = []

            if subtract_inputs:
                try:
                    series, sub_audit = apply_load_subtract(
                        series, subtract_inputs,
                    )
                    self._log_load_subtract_audit(exp_cfg, sub_audit)
                except LoadSubtractError as e:
                    # Fail-fast guard fired — do NOT silently proceed with a
                    # broken subtract. Surface the error to the caller so the
                    # experiment run aborts and the user sees the diagnostic.
                    logger.error(
                        f"  ✗ load_subtract robustness check failed for "
                        f"{exp_cfg.name}: {e}"
                    )
                    raise

        # --- Clip outliers ---
        series = clip_outliers(
            series,
            quantile=exp_cfg.outlier_quantile,
            positive_only=exp_cfg.source_is_cumulative,
            method=exp_cfg.outlier_method,
            lower_bound=exp_cfg.outlier_lower,
        )

        # --- Optional log transform ---
        if exp_cfg.log_transform:
            # Pin the transform to log1p (shift=1). apply_log_transform
            # derives shift=|min|+1 for signed series, but every
            # inversion site in the pipeline is a hard-coded
            # np.expm1 (shift=1) — a signed target would be inverted
            # with the wrong shift and publish systematically-offset
            # values (audit F6). log_transform is meant for
            # non-negative magnitudes; clip and warn rather than
            # silently publish garbage.
            _n_neg = int((series.dropna() < 0).sum())
            if _n_neg > 0:
                logger.warning(
                    f"  {exp_cfg.name}: log_transform=true but the "
                    f"target has {_n_neg} negative value(s). "
                    f"log_transform assumes a non-negative magnitude "
                    f"(energy, power, demand); negatives are clipped "
                    f"to 0 before the transform. For signed targets "
                    f"(net grid flow, temperature deltas) disable "
                    f"log_transform."
                )
                series = series.clip(lower=0.0)
            series = apply_log_transform(series, shift=1.0)

        # --- Build DataFrame ---
        result = pd.DataFrame({"y": series}, index=series.index)

        # --- Fetch covariates ---
        # `cov_stats` accumulates one row per covariate so the end-of-fetch
        # manifest log can report role, coverage, staleness, and target
        # correlation together instead of scattered across per-covariate
        # INFO lines. Keeps the `[SOLAR]` physics features and fetch
        # failures in the same manifest so "why did my forecast skip this
        # cycle?" is answerable from one block.
        cov_stats: list[dict] = []
        if exp_cfg.covariates and self.covariate_resolver:
            logger.info(f"  Fetching {len(exp_cfg.covariates)} covariate(s)...")
            for cov_cfg in exp_cfg.covariates:
                try:
                    # Per-entity-per-key column naming: same entity
                    # configured multiple times (e.g. cloud_coverage +
                    # temperature from one weather entity) gets disambiguated
                    # via the value_key suffix in _cov_column_name.
                    canonical_name = _cov_column_name(
                        cov_cfg, all_covs=exp_cfg.covariates,
                    )
                    # Build dict for CovariateResolver (expects entity_id, not entity)
                    # ``future_value_key`` plumbed through so the
                    # attribute-history path (v2.38.4) can fire for
                    # weather.* entities: their .state is categorical
                    # but the per-metric numeric values live in
                    # .attributes, and value_key tells the resolver
                    # which attribute to extract.
                    cov_dict = {
                        "entity_id": cov_cfg.entity,
                        "name": canonical_name,
                        "binary": cov_cfg.is_binary,
                        "future_value_key": getattr(
                            cov_cfg, "future_value_key", None,
                        ),
                    }

                    cov_series = await self.covariate_resolver.fetch_history(
                        cov_dict, start, now, freq
                    )

                    if cov_series.empty:
                        logger.warning(f"    No data for covariate {cov_cfg.entity}, skipping")
                        continue

                    # Apply scaling factor if configured
                    if cov_cfg.scale is not None:
                        cov_series = cov_series * cov_cfg.scale

                    # Apply transform if configured
                    if cov_cfg.transform is not None:
                        from ml_forecast_lab.preprocessing import apply_transform
                        cov_series = apply_transform(cov_series, cov_cfg.transform)

                    # Align to target index and merge
                    cov_name = cov_dict["name"]
                    cov_aligned = cov_series.reindex(result.index, method="ffill")
                    # Back-fill any leading NaNs and forward-fill trailing ones
                    cov_aligned = cov_aligned.ffill().bfill()
                    result[cov_name] = cov_aligned

                    valid_count = result[cov_name].notna().sum()
                    logger.info(
                        f"    ✓ {cov_cfg.entity} → '{cov_name}': "
                        f"{len(cov_series)} raw → {valid_count} aligned"
                        f"{f', scaled ×{cov_cfg.scale}' if cov_cfg.scale else ''}"
                    )
                    cov_stats.append({
                        "entity": cov_cfg.entity,
                        "name": cov_name,
                        "role": cov_cfg.role,
                        "raw_count": len(cov_series),
                        "aligned_count": int(valid_count),
                        "last_ts": (
                            cov_series.index[-1]
                            if len(cov_series) else None
                        ),
                        "ok": True,
                    })

                except Exception as e:
                    logger.warning(f"    ✗ Failed to fetch {cov_cfg.entity}: {e}")
                    cov_stats.append({
                        "entity": cov_cfg.entity,
                        "name": _cov_column_name(cov_cfg, all_covs=exp_cfg.covariates),
                        "role": cov_cfg.role,
                        "ok": False,
                        "error": str(e),
                    })

        # --- Compute deterministic solar physics features ---
        if (
            getattr(exp_cfg, "include_sun_elevation", False)
            or getattr(exp_cfg, "include_clear_sky_irradiance", False)
        ):
            loc = await self._get_site_location()
            if loc is not None:
                lat, lon = loc
                try:
                    from ml_forecast_lab.solar_physics import compute_solar_features
                    solar_df = compute_solar_features(
                        result.index,
                        latitude=lat,
                        longitude=lon,
                        include_elevation=exp_cfg.include_sun_elevation,
                        include_clear_sky=exp_cfg.include_clear_sky_irradiance,
                    )
                    for col in solar_df.columns:
                        result[col] = solar_df[col].values
                        cov_stats.append({
                            "entity": col,
                            "name": col,
                            "role": "physics",
                            "raw_count": len(solar_df),
                            "aligned_count": int(result[col].notna().sum()),
                            "last_ts": None,
                            "ok": True,
                        })
                    if len(solar_df.columns) > 0:
                        logger.info(
                            f"  ✓ Added solar physics features: {list(solar_df.columns)}"
                        )
                except Exception as e:
                    logger.warning(f"  ✗ Solar physics computation failed: {e}")
            else:
                logger.warning(
                    "  Solar physics requested but site location unavailable"
                )

        # --- Idle-value fill (solar night / declared idle) --------------
        # See _apply_idle_value_fill docstring for the full rationale.
        # Short version: HA's delta-storage recorder + the default
        # 90-min ``gap_handling='interpolate'`` cap drops every
        # idle / overnight row from training data for any sensor that
        # goes constant-or-unavailable for >90 min. The fill restores
        # those rows with the physically correct value (0 for solar
        # night by default; user-declared idle_value for non-solar
        # non-negative targets like EV chargers).
        y_nan_before = int(result["y"].isna().sum()) if "y" in result.columns else 0
        n_filled = _apply_idle_value_fill(result, exp_cfg)
        if n_filled > 0:
            remaining_nan = y_nan_before - n_filled
            idle_value = getattr(exp_cfg, "idle_value", None)
            has_physics = (
                "clear_sky_ghi" in result.columns
                or "sun_elevation" in result.columns
            )
            if has_physics:
                fv = idle_value if idle_value is not None else 0.0
                label = f"sun below horizon → {fv}"
            else:
                label = f"all idle NaN → {idle_value}"
            logger.info(
                f"  Idle fill: {n_filled} NaN rows "
                f"({label}) for {exp_cfg.name} "
                f"[{remaining_nan} non-idle NaN remain for dropna]"
            )

        # --- Empty-column guard (v2.38.3) --------------------------------
        # A covariate whose entire column is NaN (every aligned value
        # parsed as non-numeric) would, on the next line, drag every
        # single row into the dropna and leave 0 samples for training
        # ("⚠ No samples remaining after preprocessing — Skipping this
        # cycle"). Most common cause: a `weather.*` entity configured
        # as `role: future` whose state is a categorical string
        # ("partlycloudy") — fetch_history returns no numeric values.
        # Drop these columns before dropna so one bad covariate can't
        # kill the experiment; the future-block at inference will still
        # receive forecast values via the service API path so the model
        # still gets the signal where it matters most.
        if exp_cfg.covariates and self.covariate_resolver:
            # Track every role assigned to each canonical column name —
            # two covariates of the same entity may share a column name
            # (one role=lagged, one role=future). The drop-vs-zero-fill
            # decision must consider all of them: drop only when every
            # role for that column would discard it, otherwise zero-fill
            # so the future-side channel still gets values at inference.
            cov_roles_by_name: Dict[str, set] = {}
            for c in exp_cfg.covariates:
                col_name = _cov_column_name(c, all_covs=exp_cfg.covariates)
                cov_roles_by_name.setdefault(col_name, set()).add(
                    getattr(c, 'role', 'lagged')
                )
            empty_cols = [
                c for c in result.columns
                if c != 'y' and result[c].notna().sum() == 0
            ]
            for col in empty_cols:
                roles = cov_roles_by_name.get(col, {'lagged'})
                has_future = bool(roles & {'future', 'both'})
                role_label = '|'.join(sorted(roles))
                if has_future:
                    # Future-role columns will receive real values at
                    # inference via the forecast/service path. Fill the
                    # past with zeros so dropna doesn't eat every row
                    # — the model just sees "no past signal" for this
                    # channel, which is the truth.
                    result[col] = 0.0
                    logger.warning(
                        "  Covariate '%s' (role=%s) had 0%% "
                        "historical coverage — filling past with 0 so "
                        "the experiment isn't killed. Future values "
                        "still come from the forecast attribute / "
                        "service API at inference. For richer past "
                        "signal on weather.* entities, add the per-"
                        "metric sensor.* sibling entity as a separate "
                        "role:lagged covariate.", col, role_label,
                    )
                else:
                    # Lagged-only column with 0% coverage adds nothing.
                    # Drop it entirely so the dropna doesn't eat rows.
                    result = result.drop(columns=[col])
                    logger.warning(
                        "  Covariate '%s' (role=%s) had 0%% "
                        "historical coverage — dropping the column "
                        "entirely. Likely cause: entity reports a "
                        "non-numeric state, or the recorder has no "
                        "matching rows in the requested window.",
                        col, role_label,
                    )

        # --- Covariate manifest -------------------------------------------
        # Summarise every covariate's contribution BEFORE dropna so the
        # user can see, in one log block per retrain/forecast cycle:
        #   role, coverage %, staleness, target correlation, and the
        #   dropna culprit (which column's NaNs deleted the most rows).
        # The "future path routed to <model>" confirmation lives in the
        # backends themselves — this block only covers the covariate
        # assembly that happens here in `_fetch_and_preprocess`.
        rows_before_dropna = len(result)
        pre_drop_nan_counts = {
            col: int(result[col].isna().sum())
            for col in result.columns if col != "y"
        }

        result = result.dropna()
        rows_after_dropna = len(result)

        self._log_covariate_manifest(
            exp_cfg=exp_cfg,
            cov_stats=cov_stats,
            result=result,
            now=now,
            rows_before_dropna=rows_before_dropna,
            rows_after_dropna=rows_after_dropna,
            pre_drop_nan_counts=pre_drop_nan_counts,
        )

        if len(result) == 0:
            logger.warning(
                f"  ⚠ No samples remaining after preprocessing for "
                f"{exp_cfg.name} — one or more covariates may have "
                f"insufficient history (need ≥{exp_cfg.days_history} day(s)). "
                f"Skipping this cycle."
            )
            return None

        # Rich data summary
        y = result["y"]
        logger.info(
            f"  Preprocessed: {len(result)} samples at {freq} intervals"
        )
        logger.info(
            f"  Data range: {result.index[0].strftime('%d %b %H:%M')} → "
            f"{result.index[-1].strftime('%d %b %H:%M')}"
        )
        logger.info(
            f"  Target stats: mean={y.mean():.3f}, std={y.std():.3f}, "
            f"min={y.min():.3f}, max={y.max():.3f}, zeros={int((y == 0).sum())}/{len(y)}"
        )
        if exp_cfg.source_is_cumulative:
            logger.info(
                f"  Cumulative→interval conversion: reset_daily={exp_cfg.reset_daily}, "
                f"max_increment={exp_cfg.max_increment}"
            )

        return result

    def _log_covariate_manifest(
        self,
        exp_cfg,
        cov_stats: list[dict],
        result: pd.DataFrame,
        now: datetime,
        rows_before_dropna: int,
        rows_after_dropna: int,
        pre_drop_nan_counts: dict,
    ) -> None:
        """Emit a single log block summarising every covariate's state.

        One row per covariate: traffic-light status, role, coverage %,
        staleness, and Pearson correlation with the target. A final
        `dropna` line names the biggest NaN contributor — the column
        responsible for deleting the most rows in
        `result.dropna()` — which is the single most useful field when
        an experiment returns zero samples from preprocessing.

        Staleness threshold is `interval_minutes × 4` (four missed ticks),
        matching the "sensor stopped updating" heuristic used elsewhere.
        Correlation magnitude cutoffs: |r|<0.05 noise, |r|<0.10 weak.
        """
        if not cov_stats and rows_before_dropna == rows_after_dropna:
            return

        stale_threshold = pd.Timedelta(minutes=exp_cfg.interval_minutes * 4)
        now_naive = (
            now.replace(tzinfo=None) if now.tzinfo is not None else now
        )

        configured = len(exp_cfg.covariates or [])
        header = (
            f"Covariate manifest for {exp_cfg.name} "
            f"({configured} configured"
            f"{f', +{len(cov_stats) - configured} physics' if len(cov_stats) > configured else ''}):"
        )
        lines = [header]

        for cs in cov_stats:
            if not cs.get("ok", False):
                lines.append(
                    f"  ✗ {cs['entity']} [{cs['role']}] — fetch failed: "
                    f"{cs.get('error', '?')}"
                )
                continue

            name = cs["name"]
            role = cs["role"]
            aligned = cs.get("aligned_count", 0)

            # Coverage: fraction of post-dropna rows this column was
            # non-NaN on. For a column that was part of the reason rows
            # got dropped, this reads 100% (survivors are by definition
            # non-NaN); the dropna-culprit line below captures the
            # pre-dropna perspective.
            if name in result.columns and rows_after_dropna > 0:
                post_valid = int(result[name].notna().sum())
                coverage_pct = 100.0 * post_valid / rows_after_dropna
            else:
                coverage_pct = (
                    100.0 * aligned / rows_before_dropna
                    if rows_before_dropna else 0.0
                )

            # Staleness: age of the most recent raw value.
            stale_str = ""
            flags = []
            last_ts = cs.get("last_ts")
            if last_ts is not None:
                last_ts_naive = (
                    last_ts.tz_convert(None)
                    if getattr(last_ts, "tzinfo", None) is not None
                    else last_ts
                )
                age = now_naive - pd.Timestamp(last_ts_naive)
                if age.total_seconds() > 0:
                    age_min = age.total_seconds() / 60
                    if age_min < 120:
                        stale_str = f"stale={int(age_min)}m"
                    else:
                        stale_str = f"stale={age_min/60:.1f}h"
                    if age > stale_threshold:
                        flags.append("stale>interval×4")

            # Correlation with target on the post-dropna frame.
            corr_str = ""
            if name in result.columns and rows_after_dropna > 1:
                col = result[name]
                if col.std() > 1e-12 and result["y"].std() > 1e-12:
                    r = float(col.corr(result["y"]))
                    corr_str = f"corr={r:+.2f}"
                    if abs(r) < 0.05:
                        flags.append("|corr|<0.05 noise")
                else:
                    corr_str = "corr=const"

            parts = [f"cov={coverage_pct:.1f}%"]
            if stale_str:
                parts.append(stale_str)
            if corr_str:
                parts.append(corr_str)
            flag_str = f"  ← {', '.join(flags)}" if flags else ""
            marker = "⚠" if flags else "✓"
            lines.append(
                f"  {marker} {cs['entity']} [{role}]  "
                f"{'  '.join(parts)}{flag_str}"
            )

        # Dropna culprit — the column whose pre-dropna NaN count was the
        # largest contributor to rows lost. When rows_lost is small this
        # is noise; when it's the entire dataset it pinpoints the column
        # that's killing the cycle.
        rows_lost = rows_before_dropna - rows_after_dropna
        if pre_drop_nan_counts:
            culprit_col, culprit_nans = max(
                pre_drop_nan_counts.items(), key=lambda kv: kv[1]
            )
        else:
            culprit_col, culprit_nans = None, 0

        dropna_line = (
            f"  dropna: {rows_before_dropna} rows → {rows_after_dropna} kept"
        )
        if rows_lost > 0:
            dropna_line += f" ({rows_lost} lost"
            if culprit_col and culprit_nans > 0:
                dropna_line += (
                    f"; biggest culprit: {culprit_col} {culprit_nans} NaNs"
                )
            dropna_line += ")"
        lines.append(dropna_line)

        logger.info("\n".join(lines))

    def _update_web_benchmark(
        self, exp_cfg, model_results, rankings, best_model_name,
        status="running", daily_rankings=None,
        naive_was_enabled: Optional[bool] = None,
        drift: Optional[dict] = None,
        did_not_complete: Optional[List[str]] = None,
        did_not_complete_daily: Optional[List[str]] = None,
    ):
        """
        Update web app state with current benchmark progress.

        Called after each model completes so the UI updates progressively.

        Parameters
        ----------
        daily_rankings : dict[str, int], optional
            Per-model daily-cumulative integer rank (composite mean rank
            over daily metrics, Demšar-style averaging step only — the
            full Demšar (2006) test does not apply to CV folds, see
            ``docs/RANKING_NOTES.md``). When omitted (e.g. progressive
            intermediate updates) the daily rank columns simply show "—"
            until the final update arrives.
        did_not_complete : list[str], optional
            Model names that failed at least one fold during the
            benchmark. Surfaced separately in the UI so they don't
            appear at the bottom of the leaderboard with a fake
            last-place rank.
        """
        from ml_forecast_lab.web.app import (
            BenchmarkResult as WebBenchmarkResult,
            ModelResult as WebModelResult,
            MetricValue,
        )

        daily_rankings = daily_rankings or {}

        web_models = []
        for model_name, runner_model_result in sorted(
            model_results.items(),
            key=lambda x: rankings.get(x[0], 999),
        ):
            rank = rankings.get(model_name, 0)
            fold_metrics_list = runner_model_result.fold_metrics
            daily_fold_list = getattr(runner_model_result, 'daily_fold_metrics', [])

            metric_means = {}
            metric_stds = {}
            for metric_name in exp_cfg.metrics:
                values = [
                    fm.get(metric_name, np.nan)
                    for fm in fold_metrics_list
                    if fm
                ]
                metric_means[metric_name] = float(np.nanmean(values)) if values else 0.0
                metric_stds[metric_name] = float(np.nanstd(values)) if len(values) > 1 else 0.0

            # Daily-cumulative metric means/stds (parallel computation)
            daily_means: Dict[str, float] = {}
            daily_stds: Dict[str, float] = {}
            for metric_name in exp_cfg.metrics:
                vals = [
                    fm.get(metric_name, np.nan)
                    for fm in daily_fold_list
                    if fm
                ]
                vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
                if vals:
                    daily_means[metric_name] = float(np.nanmean(vals))
                    daily_stds[metric_name] = float(np.nanstd(vals)) if len(vals) > 1 else 0.0

            # Train metric means/stds for overfitting table
            train_metric_means = {}
            train_metric_stds = {}
            if runner_model_result.fold_train_metrics:
                for mn in ["mae", "rmse"]:
                    vals = [
                        fm.get(mn, np.nan)
                        for fm in runner_model_result.fold_train_metrics if fm
                    ]
                    if vals:
                        train_metric_means[mn] = float(np.nanmean(vals))
                        train_metric_stds[mn] = float(np.nanstd(vals)) if len(vals) > 1 else 0.0

            daily_mean_rank = runner_model_result.metrics.get("mean_rank_daily")
            if daily_mean_rank is not None and (
                isinstance(daily_mean_rank, float) and np.isinf(daily_mean_rank)
            ):
                daily_mean_rank = None

            mean_rank_low = runner_model_result.metrics.get("mean_rank_low")
            mean_rank_high = runner_model_result.metrics.get("mean_rank_high")
            daily_mean_rank_low = runner_model_result.metrics.get("mean_rank_daily_low")
            daily_mean_rank_high = runner_model_result.metrics.get("mean_rank_daily_high")

            # Stability check — surface a model whose mean-rank standing
            # is propped up by being good on most folds while hiding a
            # catastrophic fold (the mean rank is outlier-robust and
            # won't show it). Assessed on the production metric.
            unstable, instability_reason = _assess_model_instability(
                [fm for fm in fold_metrics_list if fm],
                [exp_cfg.production_metric, "mase", "mae"],
            )

            web_models.append(WebModelResult(
                name=model_name,
                mae=MetricValue(
                    mean=metric_means.get("mae", 0.0),
                    std=metric_stds.get("mae", 0.0),
                ),
                rmse=MetricValue(
                    mean=metric_means.get("rmse", 0.0),
                    std=metric_stds.get("rmse", 0.0),
                ),
                mase=MetricValue(
                    mean=metric_means.get("mase", 0.0),
                    std=metric_stds.get("mase", 0.0),
                ),
                train_time_seconds=runner_model_result.mean_train_time,
                rank=rank,
                mean_rank=runner_model_result.metrics.get("mean_rank", 0.0),
                mean_rank_low=mean_rank_low,
                mean_rank_high=mean_rank_high,
                is_production=(model_name == best_model_name),
                fold_results=[fm for fm in fold_metrics_list if fm],
                train_mae=MetricValue(
                    mean=train_metric_means["mae"], std=train_metric_stds["mae"],
                ) if "mae" in train_metric_means else None,
                train_rmse=MetricValue(
                    mean=train_metric_means["rmse"], std=train_metric_stds["rmse"],
                ) if "rmse" in train_metric_means else None,
                training_history=runner_model_result.training_history,
                daily_mae=MetricValue(
                    mean=daily_means["mae"], std=daily_stds["mae"],
                ) if "mae" in daily_means else None,
                daily_rmse=MetricValue(
                    mean=daily_means["rmse"], std=daily_stds["rmse"],
                ) if "rmse" in daily_means else None,
                daily_mase=MetricValue(
                    mean=daily_means["mase"], std=daily_stds["mase"],
                ) if "mase" in daily_means else None,
                daily_rank=daily_rankings.get(model_name),
                daily_mean_rank=daily_mean_rank,
                daily_mean_rank_low=daily_mean_rank_low,
                daily_mean_rank_high=daily_mean_rank_high,
                unstable=unstable,
                instability_reason=instability_reason,
            ))

        # Pairwise comparison: paired difference of per-fold MAE values
        # per model pair. With small fold counts the formal DM test is
        # weak, so we report a paired t-statistic + p-value alongside the
        # mean MAE difference — the UI frames this as "is the difference
        # inside fold noise?" rather than a hypothesis test.
        pairwise_dm: Optional[List[Dict[str, Any]]] = None
        try:
            from itertools import combinations as _combinations

            fold_mae_by_model: Dict[str, list] = {}
            for _mn, _mr in model_results.items():
                _vals = [
                    fm.get("mae", float("nan"))
                    for fm in (_mr.fold_metrics or [])
                ]
                if _vals:
                    fold_mae_by_model[_mn] = _vals

            if len(fold_mae_by_model) >= 2:
                rows = []
                for ma, mb in _combinations(sorted(fold_mae_by_model.keys()), 2):
                    a = np.asarray(fold_mae_by_model[ma], dtype=float)
                    b = np.asarray(fold_mae_by_model[mb], dtype=float)
                    n = int(min(len(a), len(b)))
                    if n < 2:
                        continue
                    d = a[:n] - b[:n]
                    d_clean = d[~np.isnan(d)]
                    n = int(len(d_clean))
                    if n < 2:
                        continue
                    mean_diff = float(np.mean(d_clean))
                    std_diff = float(np.std(d_clean, ddof=1))
                    if std_diff > 0:
                        t_stat = mean_diff / (std_diff / np.sqrt(n))
                        # Two-tailed normal-approx p-value (n typically 5;
                        # full t-distribution not justified given the
                        # already-approximate input)
                        from math import erf as _erf, sqrt as _sqrt
                        p_value = 2.0 * (1.0 - 0.5 * (1.0 + _erf(abs(t_stat) / _sqrt(2.0))))
                    else:
                        t_stat = 0.0
                        p_value = 1.0
                    rows.append({
                        "model_a": ma,
                        "model_b": mb,
                        "mean_diff": round(mean_diff, 6),
                        "t_stat": round(t_stat, 3),
                        "p_value": round(p_value, 4),
                        "n_folds": n,
                        "significant": bool(p_value < 0.05),
                    })
                pairwise_dm = rows or None
        except Exception as _e:
            logger.debug("Pairwise comparison build failed: %s", _e)

        # Capture the Seasonal Naive baseline MAE for the "vs Seasonal
        # Naive" skill chip. Seasonal Naive is force-run by the runner as
        # the reference baseline even when the user didn't enable it: it
        # stays visible in the results tables (it's the yardstick the other
        # models are measured against) and is part of the rank pool, so the
        # leaderboard ranks stay contiguous. It is only excluded from the
        # auto-promote decision below (so the user's mlfl.yaml is never
        # silently switched to a baseline they didn't choose) — never from
        # the displayed results.
        naive_baseline_mae: Optional[float] = None
        for _wm in web_models:
            if _wm.name == "seasonal_naive":
                naive_baseline_mae = float(_wm.mae.mean) if _wm.mae else None
                break

        web_result = WebBenchmarkResult(
            experiment_name=exp_cfg.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
            models=web_models,
            best_model_name=best_model_name,
            pairwise_dm=pairwise_dm,
            naive_baseline_mae=naive_baseline_mae,
            naive_baseline_was_enabled=naive_was_enabled,
            drift=drift,
            did_not_complete=list(did_not_complete or []),
            did_not_complete_daily=list(did_not_complete_daily or []),
        )

        self.web_app.state.appstate.benchmark_results[exp_cfg.name] = web_result

        # Persist to SQLite so results survive restarts
        if self.history_db:
            try:
                self.history_db.save_benchmark_result(
                    exp_cfg.name, web_result.model_dump_json()
                )
            except Exception as e:
                logger.warning(f"Failed to persist benchmark result: {e}")

        # Update experiment status with best model; default selected_model
        # to rank-1 if the user hasn't manually chosen one yet.
        exp_status = self.web_app.state.appstate.experiment_statuses.get(exp_cfg.name)
        champion_changed = False
        if exp_status and best_model_name:
            champion_changed = (exp_status.best_model != best_model_name)
            exp_status.best_model = best_model_name
            if not exp_status.selected_model:
                exp_status.selected_model = best_model_name

        # Clear forecast_log rows issued under the previous champion when
        # the benchmark actually promotes a new one, so the stability
        # metric doesn't pool residuals across two model-weight regimes.
        # Gated on ExperimentCfg.clear_forecast_log_on_retrain (default
        # True); no-op when the champion hasn't changed.
        #
        # Pass exclude_model_name=best_model_name so the INCOMING
        # champion's residuals (and any from this run already logged
        # under its name) survive — without it the conformal residual
        # buffer is wiped experiment-wide and the published _upper_/
        # _lower_ bands disappear for ~10 forecast cycles while the
        # new champion accumulates enough residuals.
        if (
            champion_changed
            and self.history_db
            and getattr(exp_cfg, "clear_forecast_log_on_retrain", True)
        ):
            try:
                deleted = self.history_db.cleanup_forecast_log(
                    exp_cfg.name, datetime.utcnow(),
                    exclude_model_name=best_model_name,
                )
                if deleted:
                    logger.info(
                        f"Champion change for {exp_cfg.name} "
                        f"({best_model_name}): cleared {deleted} "
                        f"pre-promotion forecast_log rows "
                        f"(retained rows belonging to new champion)"
                    )
            except Exception as e:
                logger.warning(
                    f"forecast_log cleanup after champion change failed: {e}"
                )

    async def _run_benchmark(self, exp_cfg):
        """
        Run full benchmark across all enabled models using cross-validation.
        """
        from ml_forecast_lab.features import build_features
        from ml_forecast_lab.benchmark.runner import BenchmarkRunner
        from ml_forecast_lab.benchmark.metrics import get_metric_registry
        from ml_forecast_lab.models.base import TrainingCancelled
        from ml_forecast_lab.web.app import (
            BenchmarkResult as WebBenchmarkResult,
            ModelResult as WebModelResult,
            MetricValue,
        )

        logger.info(f"")
        logger.info(f"{'=' * 60}")
        logger.info(f"  BENCHMARK: {exp_cfg.name}")
        logger.info(f"  Target: {exp_cfg.target_entity}")
        logger.info(f"  Models: {', '.join(exp_cfg.models_enabled)}")
        logger.info(f"  Covariates: {len(exp_cfg.covariates)}" + (
            f" ({', '.join(c.entity.split('.')[-1] for c in exp_cfg.covariates)})" if exp_cfg.covariates else ""
        ))
        logger.info(f"  CV: {exp_cfg.cv_strategy}, {exp_cfg.cv_folds} folds, metric={exp_cfg.production_metric}")
        logger.info(f"{'=' * 60}")

        if self.web_app:
            self.web_app.state.appstate.start_benchmark(exp_cfg.name)

        # 1. Fetch and preprocess data
        df = await self._fetch_and_preprocess(exp_cfg)
        if df is None:
            return

        if len(df) < exp_cfg.cv_folds * 10:
            raise ValueError(
                f"Insufficient data for benchmark: {len(df)} samples "
                f"(need at least {exp_cfg.cv_folds * 10})"
            )

        # 2. Build temporal + lag features from target
        features_df = build_features(
            df,
            target_col="y",
            interval_minutes=exp_cfg.interval_minutes,
            country=exp_cfg.country,
        )

        # 3. Combine features + covariates + target, drop NaN from lag warmup
        combined = features_df.copy()
        combined["target"] = df["y"]

        # Add covariate columns from df (they were merged in _fetch_and_preprocess)
        covariate_cols = [c for c in df.columns if c != "y"]
        for col in covariate_cols:
            combined[col] = df[col]

        combined = combined.dropna()

        feature_cols = [c for c in combined.columns if c != "target"]
        n_cov = len(covariate_cols)
        logger.info(
            f"  Feature matrix: {len(combined)} samples, "
            f"{len(feature_cols)} features ({len(feature_cols) - n_cov} temporal + {n_cov} covariates)"
        )

        # 4. Create feature_builder callback for BenchmarkRunner.
        # Re-compute rolling stats per fold to prevent feature leakage.
        # Windows scale with interval_minutes so the daily seasonality is
        # captured at any sampling rate (legacy 30-min default → [6, 24, 72]).
        _steps_per_hour = max(1, 60 // max(exp_cfg.interval_minutes, 1))
        rolling_windows = [
            max(2, 3 * _steps_per_hour),
            max(3, 12 * _steps_per_hour),
            max(4, 36 * _steps_per_hour),
        ]

        steps_per_day = max(1, 1440 // exp_cfg.interval_minutes)

        def feature_builder(df_sub, config, purpose="train"):
            df_out = df_sub.copy()
            target = df_out["target"]
            # Shift before rolling so the feature at row t uses target[t-w..t-1]
            # only. Without the shift pandas rolling includes target[t] (the
            # value being predicted): target leakage for tree backends and
            # a train/inference skew vs the recursive forecast which uses
            # buf[-w:].
            shifted_target = target.shift(1)
            for window in rolling_windows:
                df_out[f"y_rolling_mean_{window}"] = shifted_target.rolling(window=window).mean()
                df_out[f"y_rolling_std_{window}"] = shifted_target.rolling(window=window).std()
                df_out[f"y_rolling_max_{window}"] = shifted_target.rolling(window=window).max()
            for d in [1, 2]:
                lag_steps = steps_per_day * d
                if lag_steps <= len(target):
                    df_out[f"y_lag_{lag_steps}"] = target.shift(lag_steps)
            df_out["y_diff_1"] = target.shift(1) - target.shift(2)
            cols = [c for c in df_out.columns if c != "target"]
            X = df_out[cols].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0)
            return X

        # 5. Refresh models_enabled from config (picks up UI toggle changes)
        try:
            import yaml as _yaml
            import glob as _glob
            _cfg_path = None
            for _p in [Path("/addon_configs/ml_forecast_lab/mlfl.yaml"), Path("/config/mlfl.yaml"),
                        Path(__file__).parent.parent / "mlfl.yaml"]:
                if _p.exists():
                    _cfg_path = _p
                    break
            for _m in _glob.glob(
                "/addon_configs/[0-9a-f][0-9a-f][0-9a-f][0-9a-f]"
                "[0-9a-f][0-9a-f][0-9a-f][0-9a-f]_ml_forecast_lab/mlfl.yaml"
            ):
                _cfg_path = Path(_m)
                break
            if _cfg_path and _cfg_path.exists():
                with open(_cfg_path, "r", encoding="utf-8") as _f:
                    _yaml_data = _yaml.safe_load(_f)
                for _exp in _yaml_data.get("experiments", []):
                    if _exp.get("name") == exp_cfg.name:
                        exp_cfg.models_enabled = _exp.get("models_enabled", exp_cfg.models_enabled)
                        logger.debug(f"Refreshed models_enabled for {exp_cfg.name}: {exp_cfg.models_enabled}")
                        break
        except Exception as _e:
            logger.debug(f"Config refresh skipped: {_e}")

        # Always include a Seasonal Naive baseline so the UI can render a
        # "vs Seasonal Naive" skill chip even if the user hasn't enabled
        # it. If it was force-added we'll hide it from the rank table
        # below (otherwise it appears as a normal model). Seasonal Naive
        # has no training cost so the overhead is negligible.
        _naive_was_enabled = "seasonal_naive" in (exp_cfg.models_enabled or [])
        _models_to_run = list(exp_cfg.models_enabled or [])
        if (
            not _naive_was_enabled
            and self.model_registry is not None
            and "seasonal_naive" in self.model_registry.list_available()
        ):
            _models_to_run.append("seasonal_naive")

        # Instantiate models — pass loss_fn to neural models
        models = {}
        for model_name in _models_to_run:
            try:
                m = self.model_registry.create(model_name)
                # Apply hyperparameter overrides:
                # 1. Global model_overrides (from Models page)
                # 2. Per-experiment model_params (from Tuning / config)
                # Per-experiment takes precedence over global.
                overrides = dict(self.config.model_overrides.get(model_name, {}))
                exp_params = getattr(exp_cfg, 'model_params', {}).get(model_name, {})
                if exp_params:
                    overrides.update(exp_params)
                # Apply experiment-level loss_fn first, then overrides. Neural
                # backends accept mse/mae/huber; tree backends additionally
                # accept tweedie and map huber/mae to their native objectives.
                if hasattr(m, 'loss_fn') and 'loss_fn' not in overrides:
                    user_loss = exp_cfg.loss_fn
                    if m.is_neural and user_loss == 'tweedie':
                        # Tweedie has no native torch loss; fall back to huber
                        # for neural backends so the choice still degrades
                        # gracefully on a mixed-backend benchmark.
                        user_loss = 'huber'
                    m.set_params(loss_fn=user_loss)
                _apply_patience(m, exp_cfg, overrides)
                if (m.is_neural and hasattr(m, 'optimiser')
                        and 'optimiser' not in overrides):
                    m.set_params(optimiser=exp_cfg.optimiser)
                # Quantile-aware backends (currently DLinear) pick up the
                # experiment-level quantiles list when one is configured.
                # Backends without the attribute silently ignore it.
                if (m.is_neural and hasattr(m, 'quantiles')
                        and 'quantiles' not in overrides
                        and exp_cfg.quantiles):
                    m.set_params(quantiles=list(exp_cfg.quantiles))
                if overrides:
                    m.set_params(**overrides)
                    logger.info(f"Applied {len(overrides)} override(s) for {model_name}"
                                + (" (inc. experiment-level)" if exp_params else ""))
                # Output activation (applied last so experiment-level setting
                # wins over any stray override — activation is a model-
                # architecture choice tied to the target, not a hyperparam).
                if 'output_activation' not in overrides:
                    _apply_output_activation(m, exp_cfg)
                models[model_name] = m
            except Exception as e:
                logger.warning(
                    f"Skipping model {model_name}: {e}"
                )

        if not models:
            raise ValueError("No models could be created for benchmark")

        # 6. Run benchmark model-by-model, updating web UI after each
        exp_cfg_dict = dataclasses.asdict(exp_cfg)
        metric_registry = get_metric_registry()
        runner = BenchmarkRunner(exp_cfg_dict, feature_builder, metric_registry)
        fold_indices = runner._prepare_train_test_splits(combined)

        # Training-window vs test-window drift on the target column.
        # Surfaces "your recent data has shifted from what the model
        # trained on" as a UI verdict so users can distinguish "the
        # model is bad" from "the test window happens to be a regime
        # the model has never seen". Uses the earliest fold's training
        # rows vs the latest fold's test rows for the largest gap.
        drift_stats: Optional[dict] = None
        try:
            if fold_indices and "target" in combined.columns:
                train_idx_first, _ = fold_indices[0]
                _, test_idx_last = fold_indices[-1]
                train_y = combined["target"].iloc[train_idx_first].dropna().values
                test_y = combined["target"].iloc[test_idx_last].dropna().values
                if len(train_y) >= 20 and len(test_y) >= 5:
                    train_mean = float(np.mean(train_y))
                    test_mean = float(np.mean(test_y))
                    train_std = float(np.std(train_y))
                    test_std = float(np.std(test_y))
                    # PSI on deciles of the training window. Bins with
                    # zero population on either side get a small floor
                    # to keep the log finite.
                    edges = np.quantile(train_y, np.linspace(0, 1, 11))
                    edges = np.unique(edges)
                    if len(edges) >= 3:
                        train_hist, _ = np.histogram(train_y, bins=edges)
                        test_hist, _ = np.histogram(test_y, bins=edges)
                        train_pct = train_hist / max(1, train_hist.sum())
                        test_pct = test_hist / max(1, test_hist.sum())
                        floor = 1e-4
                        train_pct = np.where(train_pct < floor, floor, train_pct)
                        test_pct = np.where(test_pct < floor, floor, test_pct)
                        psi = float(np.sum((test_pct - train_pct) * np.log(test_pct / train_pct)))
                    else:
                        psi = 0.0
                    if psi < 0.1:
                        verdict, severity = "stable", "ok"
                    elif psi < 0.2:
                        verdict, severity = "moderate shift", "warning"
                    else:
                        verdict, severity = "significant shift", "alert"
                    drift_stats = {
                        "psi": round(psi, 4),
                        "verdict": verdict,
                        "severity": severity,
                        "train": {
                            "n": int(len(train_y)),
                            "mean": round(train_mean, 4),
                            "std": round(train_std, 4),
                            "p10": round(float(np.quantile(train_y, 0.10)), 4),
                            "p90": round(float(np.quantile(train_y, 0.90)), 4),
                        },
                        "test": {
                            "n": int(len(test_y)),
                            "mean": round(test_mean, 4),
                            "std": round(test_std, 4),
                            "p10": round(float(np.quantile(test_y, 0.10)), 4),
                            "p90": round(float(np.quantile(test_y, 0.90)), 4),
                        },
                    }
        except Exception as _e:
            logger.debug("Drift computation skipped: %s", _e)

        completed_models = {}
        rankings = {}

        # Fresh cooperative-cancel flag for this run (audit F10).
        cancel_ev = threading.Event()
        self._cancel_events[exp_cfg.name] = cancel_ev

        # v2.41.0 (audit F8): build the SAME future-known feature frame
        # production training uses (_retrain_and_cache) and pass it into
        # every CV fold, so the leaderboard ranks the extended-window
        # architecture that actually ships. Deterministic columns
        # (temporal, solar) are exact; user future-covariates use
        # in-sample observed values — identical to the production
        # training side, and equally optimistic for every backend, so
        # relative ranking is preserved. (Inference-side production uses
        # HA forecasts for those channels; CV cannot, short of storing
        # historical forecast snapshots.)
        bench_future_features_df = None
        try:
            from ml_forecast_lab.features import compute_known_future_features
            _engineered_bench = {
                'hour_of_day', 'day_of_week', 'is_weekend', 'month',
                'day_of_month', 'hour_sin', 'hour_cos', 'dow_sin',
                'dow_cos', 'is_holiday',
            }
            _engineered_bench.update(
                c for c in combined.columns if c.startswith('y_lag_')
            )
            _raw_cov_bench = [
                c for c in combined.columns
                if c not in _engineered_bench and c != 'target'
            ]
            _loc = await self._get_site_location()
            _future_cov_bench = _collect_train_future_covariates(
                combined, exp_cfg,
            )
            bench_future_features_df = compute_known_future_features(
                combined.index,
                add_temporal=True,
                country=getattr(exp_cfg, 'country', None),
                solar_lat_lon=_loc if _loc is not None else None,
                include_sun_elevation='sun_elevation' in _raw_cov_bench,
                include_clear_sky_ghi='clear_sky_ghi' in _raw_cov_bench,
                future_covariate_values=_future_cov_bench or None,
            )
        except Exception as _e:
            logger.warning(
                f"  Benchmark future-feature build failed — CV falls "
                f"back to past-only windows: {_e}"
            )

        # Set up live training event bus
        event_bus = TrainingEventBus.get_instance()
        event_bus.clear_history(exp_cfg.name)
        event_bus.publish(TrainingEvent(
            event_type="pipeline_start",
            experiment_name=exp_cfg.name,
            message=f"Starting benchmark with {len(models)} model(s)",
        ))

        for model_idx, (model_name, model) in enumerate(models.items(), 1):
            if cancel_ev.is_set():
                logger.info(f"  Benchmark for {exp_cfg.name} cancelled — "
                            f"stopping before {model_name}")
                break
            logger.info(f"")
            logger.info(f"  [{model_idx}/{len(models)}] Benchmarking: {model_name}")

            event_bus.publish(TrainingEvent(
                event_type="model_start",
                experiment_name=exp_cfg.name,
                model_name=model_name,
                message=f"Model {model_idx}/{len(models)}",
            ))

            # Create epoch callback for live streaming
            def _make_epoch_cb(exp_name, m_name):
                _start = time.time()
                def _cb(**data):
                    event_bus.publish(TrainingEvent(
                        event_type="epoch",
                        experiment_name=exp_name,
                        model_name=m_name,
                        fold=data.get("fold", 0),
                        total_folds=data.get("total_folds", 0),
                        epoch=data.get("epoch", 0),
                        total_epochs=data.get("total_epochs", 0),
                        train_loss=data.get("train_loss", 0.0),
                        val_loss=data.get("val_loss", 0.0),
                        learning_rate=data.get("lr", 0.0),
                        patience_counter=data.get("patience_counter", 0),
                        patience_limit=data.get("patience_limit", 0),
                        best_val_loss=data.get("best_val_loss", 0.0),
                        elapsed_seconds=time.time() - _start,
                    ))
                return _cb
            epoch_cb = _make_epoch_cb(exp_cfg.name, model_name)

            loop = asyncio.get_running_loop()
            try:
                model_result = await loop.run_in_executor(
                    None, lambda: runner.run_single_model(
                        combined, model, fold_indices, epoch_callback=epoch_cb,
                        cancel_event=cancel_ev,
                        future_features_df=bench_future_features_df,
                    )
                )
            except TrainingCancelled:
                logger.info(
                    f"  Benchmark for {exp_cfg.name} cancelled during "
                    f"{model_name} — stopping"
                )
                break
            completed_models[model_name] = model_result
            # Release the trained model — its metrics/predictions are
            # captured in model_result, and the holdout chart re-fits
            # fresh instances. Keeping every trained backend referenced
            # until the run ends made peak RSS the SUM of all trained
            # models on a 20+-model benchmark (audit F13).
            models[model_name] = None

            event_bus.publish(TrainingEvent(
                event_type="model_end",
                experiment_name=exp_cfg.name,
                model_name=model_name,
            ))

            # Log model result summary
            mae_val = model_result.metrics.get("mae", np.nan)
            rmse_val = model_result.metrics.get("rmse", np.nan)
            logger.info(
                f"  ✓ {model_name}: MAE={mae_val:.4f}, RMSE={rmse_val:.4f}, "
                f"time={model_result.mean_train_time:.1f}s/fold"
            )

            # Rank completed models so far
            metric_values = {
                n: mr.metrics.get(runner.production_metric, np.inf)
                for n, mr in completed_models.items()
            }
            sorted_models = sorted(metric_values.items(), key=lambda x: x[1])
            rankings = {n: rank + 1 for rank, (n, _) in enumerate(sorted_models)}

            # Update web UI progressively
            if self.web_app:
                self._update_web_benchmark(
                    exp_cfg, completed_models, rankings,
                    sorted_models[0][0] if sorted_models else None,
                    status="running",
                    naive_was_enabled=_naive_was_enabled,
                    drift=drift_stats,
                )

        # Final composite mean-rank via the runner's shared helper
        # (Demšar-style averaging across folds; see
        # docs/RANKING_NOTES.md for the caveat — the full Demšar test
        # is not applicable to CV folds of one series). Computed twice:
        # once on per-interval (h=1) metrics for the primary
        # leaderboard, once on daily-cumulative metrics for the
        # secondary leaderboard. Bootstrap CIs over fold resamples are
        # carried into model_result.metrics so the UI can surface ties
        # honestly. The interval rank still drives Promote / Tuning /
        # sensor publishing; daily_rankings is informational only.
        (
            interval_mean_ranks, rankings,
            interval_rank_cis, dnc_interval,
        ) = runner._compute_composite_ranks(
            completed_models, metric_source='fold_metrics',
        )
        (
            daily_mean_ranks, daily_rankings,
            daily_rank_cis, dnc_daily,
        ) = runner._compute_composite_ranks(
            completed_models, metric_source='daily_fold_metrics',
        )
        # Daily-only DNCs (models ranked in the per-interval table but
        # excluded from the daily ranking — e.g. fold span <2 distinct
        # dates) are surfaced under the Daily table specifically, NOT
        # the main 'Did not complete' section, so a per-interval-ranked
        # model isn't confusingly listed as not having completed.
        # Pre-v2.39.3 _dnc_daily was discarded entirely.
        dnc_daily_only = sorted(set(dnc_daily) - set(dnc_interval))
        for name in completed_models:
            completed_models[name].metrics['mean_rank'] = (
                interval_mean_ranks.get(name, float('inf'))
            )
            completed_models[name].metrics['mean_rank_daily'] = (
                daily_mean_ranks.get(name, float('inf'))
            )
            ci = interval_rank_cis.get(name)
            if ci is not None:
                completed_models[name].metrics['mean_rank_low'] = ci[0]
                completed_models[name].metrics['mean_rank_high'] = ci[1]
            ci_d = daily_rank_cis.get(name)
            if ci_d is not None:
                completed_models[name].metrics['mean_rank_daily_low'] = ci_d[0]
                completed_models[name].metrics['mean_rank_daily_high'] = ci_d[1]
        mean_ranks = interval_mean_ranks  # for downstream logging

        # If Seasonal Naive was force-included for the skill chip (not
        # user-enabled), exclude it from the auto-promote decision — we
        # don't want the user's mlfl.yaml silently switched to a baseline
        # they never asked to deploy.
        _rank_pool = interval_mean_ranks
        if not _naive_was_enabled:
            _rank_pool = {
                k: v for k, v in interval_mean_ranks.items() if k != "seasonal_naive"
            }
        sorted_by_mean_rank = sorted(_rank_pool.items(), key=lambda x: x[1])
        best_model_name = sorted_by_mean_rank[0][0] if sorted_by_mean_rank else None
        best_metric_value = completed_models[best_model_name].metrics.get(
            runner.production_metric, np.nan
        ) if best_model_name else np.nan

        # Log final rankings — composite mean rank (Demšar-style
        # averaging across folds; not a Friedman/Nemenyi test). CI
        # printed in [low, high] form so a reader can spot ties.
        logger.info("")
        logger.info("  Final rankings (composite mean rank, 95%% bootstrap CI):")
        for name, _ in sorted_by_mean_rank:
            mr = completed_models[name]
            daily_str = (
                f", daily=#{daily_rankings[name]}"
                if daily_rankings.get(name) else ""
            )
            ci_low = mr.metrics.get('mean_rank_low')
            ci_high = mr.metrics.get('mean_rank_high')
            ci_str = (
                f" [{ci_low:.2f}-{ci_high:.2f}]"
                if ci_low is not None and ci_high is not None
                else ""
            )
            logger.info(
                f"    #{rankings[name]} {name}: "
                f"{runner.production_metric}={mr.metrics.get(runner.production_metric, np.nan):.4f}, "
                f"mean_rank={mean_ranks[name]:.2f}{ci_str}{daily_str}"
            )
        if dnc_interval or dnc_daily_only:
            parts = []
            if dnc_interval:
                parts.append(f"interval: {', '.join(sorted(dnc_interval))}")
            if dnc_daily_only:
                parts.append(f"daily-only: {', '.join(dnc_daily_only)}")
            logger.info(
                "  Did not complete (excluded from corresponding "
                "rankings): " + "; ".join(parts)
            )

        # Update web UI with final mean-rank-based rankings
        if self.web_app:
            self._update_web_benchmark(
                exp_cfg, completed_models, rankings,
                best_model_name,
                status="completed",
                daily_rankings=daily_rankings,
                naive_was_enabled=_naive_was_enabled,
                drift=drift_stats,
                did_not_complete=dnc_interval,
                did_not_complete_daily=dnc_daily_only,
            )

        # Build a BenchmarkResult-compatible object for downstream use
        from ml_forecast_lab.benchmark.runner import BenchmarkResult as RunnerBenchmarkResult
        bench_result = RunnerBenchmarkResult(
            experiment_name=exp_cfg.name,
            model_results=completed_models,
            rankings=rankings,
            best_model=best_model_name or "",
            best_metric_value=best_metric_value,
            metric_used=runner.production_metric,
            cv_strategy=runner.cv_strategy,
            n_folds=runner.cv_folds,
            did_not_complete=dnc_interval,
            did_not_complete_daily=dnc_daily_only,
        )

        # 7. Generate holdout predictions from each model for visualisation
        #    Use last 20% of data as holdout, train on first 80%
        from ml_forecast_lab.web.app import (
            ModelPrediction,
            LabForecastData,
            FeatureImportanceData,
        )

        holdout_frac = 0.2
        split_idx = int(len(combined) * (1 - holdout_frac))
        train_part = combined.iloc[:split_idx]
        holdout_part = combined.iloc[split_idx:]

        X_train_hold = train_part[feature_cols].values.astype(np.float32)
        X_train_hold = np.nan_to_num(X_train_hold, nan=0.0)
        y_train_hold = train_part["target"].values.astype(np.float32)

        X_holdout = holdout_part[feature_cols].values.astype(np.float32)
        X_holdout = np.nan_to_num(X_holdout, nan=0.0)
        y_holdout = holdout_part["target"].values
        holdout_timestamps = [
            ts.isoformat() for ts in holdout_part.index
        ]

        # Trace colours are auto-assigned by the frontend (Plotly colorway).
        # No per-model mapping kept here so adding a new model doesn't require
        # touching this file or staying in sync with a JS palette.

        # Pre-fetch site location once on the event loop so the
        # thread-pool _generate_holdout_predictions can pass it through
        # to compute_known_future_features without needing async
        # access. Same for site_location used downstream by every
        # neural-extended-window build inside the closure.
        _holdout_site_location = await self._get_site_location()

        def _generate_holdout_predictions():
            """Run holdout predictions in thread pool to avoid blocking."""
            _model_predictions = []
            _feature_importance_list = []

            # Compute horizon steps from config for neural multi-output
            future_periods_bench = getattr(exp_cfg, 'future_periods', 48)
            horizon_steps = list(range(1, future_periods_bench + 1))

            for m_name in exp_cfg.models_enabled:
                try:
                    m = self.model_registry.create(m_name)
                    overrides = self.config.model_overrides.get(m_name, {})
                    if m.is_neural and hasattr(m, 'loss_fn') and 'loss_fn' not in overrides:
                        m.set_params(loss_fn=exp_cfg.loss_fn)
                    _apply_patience(m, exp_cfg, overrides)
                    if (m.is_neural and hasattr(m, 'optimiser')
                            and 'optimiser' not in overrides):
                        m.set_params(optimiser=exp_cfg.optimiser)
                    if overrides:
                        m.set_params(**overrides)
                    if 'output_activation' not in overrides:
                        _apply_output_activation(m, exp_cfg)

                    # Neural models need sliding window data
                    hold_seq_kwargs = {}
                    _y_train_h = y_train_hold
                    _X_train_h = X_train_hold
                    _y_holdout = y_holdout
                    _holdout_ts = holdout_timestamps

                    neural_ok = False
                    hold_future_features_df = None
                    if m.is_neural:
                        try:
                            from ml_forecast_lab.features import (
                                create_sliding_windows, compute_known_future_features,
                            )
                            target_col = 'target'
                            engineered = {
                                'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
                                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_holiday',
                            }
                            engineered.update(c for c in train_part.columns if c.startswith('y_lag_'))
                            cov_cols = [c for c in train_part.columns if c not in engineered and c != target_col]

                            window_size = min(48, len(train_part) // 3)
                            if window_size >= 12:
                                # Match the production-cache path: build
                                # an extended window with future-known
                                # features at horizon positions so the
                                # benchmark trains the same architecture
                                # production uses. Without this, the
                                # holdout-side model was past-window only
                                # while production was extended-window —
                                # the benchmark winner could be a model
                                # that did well past-only but loses to a
                                # rival that benefits from horizon-anchored
                                # signal (TiDE in particular).
                                solar_lat_lon = _holdout_site_location
                                include_sun_elevation = 'sun_elevation' in cov_cols
                                include_clear_sky_ghi = 'clear_sky_ghi' in cov_cols
                                future_cov_train = _collect_train_future_covariates(
                                    train_part, exp_cfg,
                                )
                                hold_future_features_df = compute_known_future_features(
                                    train_part.index,
                                    add_temporal=True,
                                    country=getattr(exp_cfg, 'country', None),
                                    solar_lat_lon=solar_lat_lon,
                                    include_sun_elevation=include_sun_elevation,
                                    include_clear_sky_ghi=include_clear_sky_ghi,
                                    future_covariate_values=future_cov_train or None,
                                )
                                seq_X, seq_y, channel_names = create_sliding_windows(
                                    train_part, target_col, window_size=window_size,
                                    covariate_cols=cov_cols if cov_cols else None,
                                    add_temporal=True,
                                    horizon_steps=horizon_steps,
                                    future_features_df=hold_future_features_df,
                                )
                                hold_seq_kwargs['sequence_data'] = seq_X
                                hold_seq_kwargs['channel_names'] = channel_names
                                # Mark extended_window in seq_kwargs so the
                                # backend's fit/predict know to handle the
                                # past+future split. past_window_size is
                                # the original window_size (before the
                                # future-block extension).
                                hold_seq_kwargs['extended_window'] = True
                                hold_seq_kwargs['past_window_size'] = window_size
                                hold_seq_kwargs['future_feature_cols'] = list(
                                    hold_future_features_df.columns
                                )
                                _y_train_h = seq_y
                                _X_train_h = X_train_hold[-len(seq_y):]
                                neural_ok = True
                                if future_cov_train:
                                    logger.info(
                                        f"  Holdout future covariates "
                                        f"(horizon-aware): "
                                        f"{list(future_cov_train)}"
                                    )
                        except Exception as e:
                            logger.warning(f'Holdout sliding windows failed for {m_name}: {e}', exc_info=True)

                    if m.is_neural and neural_ok:
                        logger.info(
                            f"  Holdout {m_name}: sliding windows "
                            f"{hold_seq_kwargs['sequence_data'].shape[1]} steps × "
                            f"{hold_seq_kwargs['sequence_data'].shape[2]} channels, "
                            f"horizons={horizon_steps}"
                        )
                    elif m.is_neural:
                        logger.warning(f"  Holdout {m_name}: falling back to flat features (no sliding windows)")

                    m.fit(_X_train_h, _y_train_h, feature_names=feature_cols, **hold_seq_kwargs)

                    if m.is_neural and neural_ok:
                        # Bridge fold boundary: prepend train tail for holdout context
                        from ml_forecast_lab.features import (
                            create_sliding_windows, compute_known_future_features,
                        )
                        combined_holdout = pd.concat([
                            train_part.iloc[-window_size:],
                            holdout_part,
                        ])
                        # Mirror the training-side extended-window build so
                        # the inference seq_X matches the trained
                        # architecture (and future covariate values land
                        # in the same channel slots the model learned to
                        # read at horizon positions).
                        solar_lat_lon = _holdout_site_location
                        include_sun_elevation = 'sun_elevation' in cov_cols
                        include_clear_sky_ghi = 'clear_sky_ghi' in cov_cols
                        future_cov_ho = _collect_train_future_covariates(
                            combined_holdout, exp_cfg,
                        )
                        ho_future_features_df = compute_known_future_features(
                            combined_holdout.index,
                            add_temporal=True,
                            country=getattr(exp_cfg, 'country', None),
                            solar_lat_lon=solar_lat_lon,
                            include_sun_elevation=include_sun_elevation,
                            include_clear_sky_ghi=include_clear_sky_ghi,
                            future_covariate_values=future_cov_ho or None,
                        )
                        # Must use the SAME horizon_steps as the
                        # fit-side call above. ``create_sliding_windows``
                        # extends each window by ``max(horizon_steps)``
                        # future positions when ``future_features_df``
                        # is provided, so a shorter horizon list at
                        # predict-time would produce a window with
                        # fewer future positions — and the trained
                        # neural head (built around the fit-time flat
                        # size) can't accept it. v2.38.5: was [1] here,
                        # which broke NLinear / TiDE / any backend
                        # whose Linear head is sized off the fit-time
                        # window length. The trade-off is losing
                        # ``max_horizon - 1`` rows from the holdout
                        # tail (no window can be formed for them),
                        # acceptable given the alternative is a
                        # ``mat1 and mat2 shapes cannot be multiplied``
                        # crash on the entire holdout slice. We still
                        # take the h=1 column from the dense output
                        # for the display series.
                        seq_X_ho, _, _ = create_sliding_windows(
                            combined_holdout, target_col, window_size=window_size,
                            covariate_cols=cov_cols if cov_cols else None,
                            add_temporal=True,
                            horizon_steps=horizon_steps,
                            future_features_df=ho_future_features_df,
                        )
                        y_p = m.predict_sequence(seq_X_ho)
                        _y_holdout_display = holdout_part[target_col].values.astype(np.float32)
                        # Assemble a full-length display series: h=1 column for
                        # the points that have a window, plus the last window's
                        # h=2..H outputs for the trailing max_horizon-1 points
                        # that don't (so neural lines span the whole holdout
                        # instead of stopping ~future_periods points short —
                        # the LSTM/CNN "not as far along as LightGBM" artifact).
                        # Display-only; leaderboard metrics come from CV folds.
                        y_p_display = _holdout_display_from_windows(
                            y_p, len(_y_holdout_display),
                        )
                        _holdout_ts = holdout_timestamps
                    else:
                        y_p = m.predict(X_holdout)
                        # For chart display: use first horizon (shortest-term)
                        if y_p.ndim == 2:
                            y_p_display = y_p[:, 0]
                            _y_holdout_display = _y_holdout[:, 0] if _y_holdout.ndim == 2 else _y_holdout
                        else:
                            y_p_display = y_p
                            _y_holdout_display = _y_holdout

                        if y_p_display.ndim > 1:
                            y_p_display = y_p_display.ravel()

                    # Post-hoc clip for cumulative (non-negative) targets.
                    # Covers any neural backend whose output head can emit
                    # tiny negatives (LSTM + zscore, linear activation, etc.);
                    # for tree models this is a no-op in practice but safe.
                    # Gate on source_is_cumulative so signed targets are
                    # untouched. Applied before the log-transform invert so
                    # the branch below still owns its own expm1 clamp.
                    if getattr(exp_cfg, 'source_is_cumulative', False):
                        y_p_display = np.maximum(y_p_display, 0.0).astype(np.float32)

                    # Invert log-transform for display — both actuals and
                    # predictions live in log(y+1) space while training.
                    if exp_cfg.log_transform:
                        _y_holdout_display = np.maximum(
                            np.expm1(_y_holdout_display), 0.0,
                        )
                        y_p_display = np.maximum(np.expm1(y_p_display), 0.0)

                    _model_predictions.append(ModelPrediction(
                        model_name=m_name,
                        timestamps=_holdout_ts,
                        actuals=[float(v) if not np.isnan(v) else None for v in _y_holdout_display],
                        # NaN-pad the trailing positions the holdout-neural
                        # path can't form windows for (v2.39.3); maps to
                        # JSON null so the chart renders a visible gap at
                        # the right edge rather than silently truncating.
                        predictions=[float(v) if not np.isnan(v) else None for v in y_p_display],
                    ))

                    if hasattr(m, 'training_metadata') and m.training_metadata:
                        importances = m.training_metadata.get("feature_importances", {})
                        if importances:
                            sorted_feats = sorted(
                                importances.items(), key=lambda x: x[1], reverse=True
                            )[:20]
                            _feature_importance_list.append(FeatureImportanceData(
                                model_name=m_name,
                                features=[
                                    {"name": name, "importance": float(imp)}
                                    for name, imp in sorted_feats
                                ],
                            ))

                    logger.info(f"Generated holdout predictions for {m_name}")

                except Exception as e:
                    logger.warning(f"Failed holdout predictions for {m_name}: {e}")

            return _model_predictions, _feature_importance_list

        model_predictions, feature_importance_list = await asyncio.get_running_loop().run_in_executor(
            None, _generate_holdout_predictions
        )

        # Store lab forecast data
        if model_predictions and self.web_app:
            lab_forecast = LabForecastData(
                experiment_name=exp_cfg.name,
                holdout_start=holdout_timestamps[0] if holdout_timestamps else "",
                holdout_end=holdout_timestamps[-1] if holdout_timestamps else "",
                model_predictions=model_predictions,
            )
            self.web_app.state.appstate.lab_forecast_data[exp_cfg.name] = lab_forecast
            logger.info(
                f"Stored lab predictions: {len(model_predictions)} models, "
                f"{len(holdout_timestamps)} holdout points"
            )

        if feature_importance_list and self.web_app:
            self.web_app.state.appstate.feature_importances[exp_cfg.name] = feature_importance_list

        # 8. Final web state update (mark as completed). Pass daily_rankings
        # explicitly — without this the final overwrite wipes the daily-rank
        # column in the UI (the earlier "completed" call at the top of this
        # function did include it; this one was missing the kwarg, so the
        # last-write-wins overwrite landed with daily_rank=None for every
        # model).
        if self.web_app:
            self._update_web_benchmark(
                exp_cfg, completed_models, rankings,
                best_model_name,
                status="completed",
                daily_rankings=daily_rankings,
                naive_was_enabled=_naive_was_enabled,
                drift=drift_stats,
            )

        # Final results summary table
        logger.info(f"")
        logger.info(f"  {'─' * 56}")
        logger.info(f"  {'Model':<12} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'Time':>8} {'Rank':>6}")
        logger.info(f"  {'─' * 56}")
        for m_name in sorted(completed_models.keys(), key=lambda x: rankings.get(x, 99)):
            mr = completed_models[m_name]
            rank = rankings.get(m_name, 0)
            marker = " ★" if m_name == best_model_name else ""
            logger.info(
                f"  {m_name:<12} "
                f"{mr.metrics.get('mae', np.nan):>8.4f} "
                f"{mr.metrics.get('rmse', np.nan):>8.4f} "
                f"{mr.metrics.get('mase', np.nan):>8.3f} "
                f"{mr.mean_train_time:>7.1f}s "
                f"{'#' + str(rank):>5}{marker}"
            )
        logger.info(f"  {'─' * 56}")
        logger.info(f"  Best model: {best_model_name} ({runner.production_metric}={best_metric_value:.4f})")
        logger.info(f"{'=' * 60}")
        logger.info(f"")

        event_bus.publish(TrainingEvent(
            event_type="pipeline_end",
            experiment_name=exp_cfg.name,
            message=f"Benchmark complete — best model: {best_model_name}",
        ))

    @staticmethod
    def _build_window_channels(df_slice, cov_cols, target_col='target'):
        """Build (window_size, n_channels) array matching create_sliding_windows channel layout."""
        work = df_slice[[target_col]].copy()
        for c in cov_cols:
            if c in df_slice.columns:
                work[c] = df_slice[c]
        idx = df_slice.index
        if isinstance(idx, pd.DatetimeIndex):
            hour_rad = 2 * np.pi * idx.hour / 24
            dow_rad = 2 * np.pi * idx.dayofweek / 7
            work['hour_sin'] = np.sin(hour_rad)
            work['hour_cos'] = np.cos(hour_rad)
            work['dow_sin'] = np.sin(dow_rad)
            work['dow_cos'] = np.cos(dow_rad)
            work['is_weekend'] = (idx.dayofweek >= 5).astype(np.float32)
        return work.values.astype(np.float32)

    async def _run_production_inference(self, exp_cfg):
        """
        Run production mode: train best model on full data, generate a full
        forecast curve, and publish results as HA sensor entities.
        """
        from ml_forecast_lab.features import build_features

        logger.info(f"")
        logger.info(f"{'=' * 60}")
        logger.info(f"  PRODUCTION: {exp_cfg.name}")
        logger.info(f"  Target: {exp_cfg.target_entity}")
        logger.info(f"{'=' * 60}")

        # 1. Fetch and preprocess
        df = await self._fetch_and_preprocess(exp_cfg)
        if df is None:
            return

        # 2. Build features + covariates
        features_df = build_features(
            df,
            target_col="y",
            interval_minutes=exp_cfg.interval_minutes,
            country=exp_cfg.country,
        )
        combined = features_df.copy()
        combined["target"] = df["y"]

        # Add covariate columns
        covariate_cols = [c for c in df.columns if c != "y"]
        for col in covariate_cols:
            combined[col] = df[col]

        combined = combined.dropna()

        feature_cols = [c for c in combined.columns if c != "target"]

        X = combined[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)
        y = combined["target"].values.astype(np.float32)

        # 3. Determine production model
        prod_model_name = exp_cfg.production_model
        if not prod_model_name:
            if self.web_app:
                bench = self.web_app.state.appstate.benchmark_results.get(
                    exp_cfg.name
                )
                if bench and bench.best_model_name:
                    prod_model_name = bench.best_model_name

        if not prod_model_name:
            prod_model_name = exp_cfg.models_enabled[0]
            logger.info(
                f"No production model set, defaulting to {prod_model_name}"
            )

        # 4. Train model on full data
        model = self.model_registry.create(prod_model_name)
        overrides = self.config.model_overrides.get(prod_model_name, {})
        if model.is_neural and hasattr(model, 'loss_fn') and 'loss_fn' not in overrides:
            model.set_params(loss_fn=exp_cfg.loss_fn)
        _apply_patience(model, exp_cfg, overrides)
        if (model.is_neural and hasattr(model, 'optimiser')
                and 'optimiser' not in overrides):
            model.set_params(optimiser=exp_cfg.optimiser)
        if overrides:
            model.set_params(**overrides)
            logger.info(f"Applied {len(overrides)} override(s) for {prod_model_name}")
        if 'output_activation' not in overrides:
            _apply_output_activation(model, exp_cfg)
        is_neural = model.is_neural
        logger.info(f"Training {prod_model_name} on {len(X)} samples...")
        train_start = time.time()

        # Neural models need sliding window training with multi-horizon targets.
        # Use dense horizons (1..future_periods) so the multi-head output
        # covers every forecast step directly — no interpolation needed.
        seq_kwargs = {}
        raw_cov_cols_prod = []
        window_size_prod = None
        future_periods_pre = getattr(exp_cfg, 'future_periods', 48)
        horizon_steps_prod = list(range(1, future_periods_pre + 1))
        prod_future_features_df = None
        if is_neural:
            from ml_forecast_lab.features import (
                create_sliding_windows, compute_known_future_features,
            )
            engineered = {
                'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_holiday',
            }
            engineered.update(c for c in combined.columns if c.startswith('y_lag_'))
            raw_cov_cols_prod = [c for c in combined.columns if c not in engineered and c != 'target']
            window_size_prod = min(48, len(combined) // 3)
            if window_size_prod >= 12:
                # Match _retrain_and_cache: extended-window training with
                # future-known features at horizon positions, including
                # user-configured future covariates. Required so the
                # legacy non-cached production inference path produces
                # the same model architecture as the cached path.
                loc = await self._get_site_location()
                solar_lat_lon = loc if loc is not None else None
                include_sun_elevation = 'sun_elevation' in raw_cov_cols_prod
                include_clear_sky_ghi = 'clear_sky_ghi' in raw_cov_cols_prod
                future_cov_prod = _collect_train_future_covariates(
                    combined, exp_cfg,
                )
                prod_future_features_df = compute_known_future_features(
                    combined.index,
                    add_temporal=True,
                    country=getattr(exp_cfg, 'country', None),
                    solar_lat_lon=solar_lat_lon,
                    include_sun_elevation=include_sun_elevation,
                    include_clear_sky_ghi=include_clear_sky_ghi,
                    future_covariate_values=future_cov_prod or None,
                )
                seq_X, seq_y, channel_names = create_sliding_windows(
                    combined, 'target', window_size=window_size_prod,
                    covariate_cols=raw_cov_cols_prod if raw_cov_cols_prod else None,
                    add_temporal=True,
                    horizon_steps=horizon_steps_prod,
                    future_features_df=prod_future_features_df,
                )
                seq_kwargs['sequence_data'] = seq_X
                seq_kwargs['channel_names'] = channel_names
                seq_kwargs['extended_window'] = True
                seq_kwargs['past_window_size'] = window_size_prod
                seq_kwargs['future_feature_cols'] = list(prod_future_features_df.columns)
                if future_cov_prod:
                    seq_kwargs['future_covariate_names'] = list(future_cov_prod)
                y_train_seq = seq_y
                X_train_seq = X[-len(seq_y):]
                logger.info(
                    f"  Sliding windows: {seq_X.shape[1]} steps × {seq_X.shape[2]} channels, "
                    f"horizons={horizon_steps_prod}"
                )

                def _train_neural():
                    model.fit(X_train_seq, y_train_seq, **seq_kwargs)
                await asyncio.get_running_loop().run_in_executor(None, _train_neural)
            else:
                is_neural = False
                await asyncio.get_running_loop().run_in_executor(None, model.fit, X, y)
        else:
            await asyncio.get_running_loop().run_in_executor(None, model.fit, X, y)

        train_time = time.time() - train_start
        logger.info(f"Training completed in {train_time:.1f}s")

        # 5. Generate full forecast curve at regular intervals
        n_lags = 12
        last_ts = combined.index[-1]
        lag_values = y[-n_lags:]
        future_periods = getattr(exp_cfg, 'future_periods', 48)

        if is_neural and 'sequence_data' in seq_kwargs:
            # ----- Dense multi-head prediction for neural models -----
            # Model was trained with horizons [1, 2, ..., future_periods] so
            # predict_sequence returns all steps directly — no interpolation.
            # Uses build_inference_window (not create_sliding_windows) so
            # the window's last timestep IS combined.iloc[-1] = last_ts,
            # not last_ts - 1 interval. See features.build_inference_window
            # docstring for the rationale.
            from ml_forecast_lab.features import (
                build_inference_window, compute_known_future_features,
            )

            # window_size here is the EFFECTIVE seq_X width (past + future
            # if extended). past_window_size is what build_inference_window
            # needs for the past block; the future block is rebuilt from
            # future_features_df. Falls back to whole tensor width for
            # legacy non-extended caches.
            extended_window_prod = seq_kwargs.get('extended_window', False)
            past_window_size_prod = seq_kwargs.get(
                'past_window_size',
                seq_kwargs['sequence_data'].shape[1],
            )
            inference_future_features_df = None
            if extended_window_prod:
                # Real-future inference: fetch the HA forecast attribute
                # for each user future covariate (matches the cached
                # production path in _forecast_with_cached).
                future_index_prod = pd.date_range(
                    start=last_ts + pd.Timedelta(minutes=exp_cfg.interval_minutes),
                    periods=future_periods,
                    freq=f'{exp_cfg.interval_minutes}min',
                )
                future_feature_cols_prod = set(
                    seq_kwargs.get('future_feature_cols') or []
                )
                loc_prod = await self._get_site_location()
                solar_lat_lon_prod = loc_prod if loc_prod is not None else None
                future_cov_inf: dict[str, pd.Series] = {}
                neural_future_cov_names_prod = list(
                    seq_kwargs.get('future_covariate_names') or []
                )
                if (
                    neural_future_cov_names_prod
                    and exp_cfg.covariates
                    and self.covariate_resolver
                ):
                    # Reverse-index covariates by their canonical
                    # (per-experiment, value_key-suffixed) column name
                    # so we can find the right config when the same
                    # entity appears multiple times.
                    cov_by_name = {
                        _cov_column_name(c, all_covs=exp_cfg.covariates): c
                        for c in exp_cfg.covariates
                    }
                    for cov_name in neural_future_cov_names_prod:
                        cov_cfg = cov_by_name.get(cov_name)
                        if cov_cfg is None:
                            continue
                        try:
                            cov_dict = {
                                "entity_id": cov_cfg.entity,
                                "name": cov_name,
                                "future_attribute": getattr(cov_cfg, "future_attribute", "forecast"),
                                "future_value_key": getattr(cov_cfg, "future_value_key", None),
                            }
                            future_series = await self.covariate_resolver.fetch_future(
                                cov_dict, future_index_prod,
                            )
                            if future_series is None or future_series.empty:
                                continue
                            if cov_cfg.scale is not None:
                                future_series = future_series * cov_cfg.scale
                            aligned = future_series.reindex(future_index_prod).ffill().bfill()
                            if aligned.notna().any():
                                future_cov_inf[cov_name] = aligned
                        except Exception as e:
                            logger.debug(
                                f"  Future cov fetch failed for {cov_name}: {e}"
                            )
                inference_future_features_df = compute_known_future_features(
                    future_index_prod,
                    add_temporal=True,
                    country=getattr(exp_cfg, 'country', None),
                    solar_lat_lon=solar_lat_lon_prod,
                    include_sun_elevation='sun_elevation' in future_feature_cols_prod,
                    include_clear_sky_ghi='clear_sky_ghi' in future_feature_cols_prod,
                    future_covariate_values=future_cov_inf or None,
                )

            last_window, _ = build_inference_window(
                combined, 'target', window_size=past_window_size_prod,
                covariate_cols=raw_cov_cols_prod if raw_cov_cols_prod else None,
                add_temporal=True,
                future_features_df=inference_future_features_df,
            )

            def _predict_multihead():
                return model.predict_sequence(last_window)

            multi_pred = await asyncio.get_running_loop().run_in_executor(
                None, _predict_multihead
            )
            multi_pred = multi_pred.ravel()

            if len(multi_pred) >= future_periods:
                y_pred = multi_pred[:future_periods].astype(np.float32)
                logger.info(f"  Dense multi-head: {len(y_pred)} direct predictions")
            elif len(multi_pred) == 1:
                y_pred = np.full(future_periods, float(multi_pred[0]), dtype=np.float32)
            else:
                # Legacy fallback — interpolate between sparse horizons
                horizon_x = np.linspace(1, future_periods, len(multi_pred), dtype=np.float32)
                all_x = np.arange(1, future_periods + 1, dtype=np.float32)
                y_pred = np.interp(all_x, horizon_x, multi_pred.astype(np.float32)).astype(np.float32)
                logger.info(f"  Sparse multi-head: {len(multi_pred)} → {len(y_pred)} interpolated")

            # Post-hoc clip for cumulative (non-negative) targets. The
            # activation (softplus/relu/sigmoid) should already handle this,
            # but linear-head paths (LSTM + zscore, or any neural backend
            # with an unconstrained output head) can emit tiny negatives
            # that Plotly renders as visually-misleading dips. Gate on
            # source_is_cumulative so signed targets (temperature deltas,
            # net flows) stay untouched.
            if getattr(exp_cfg, 'source_is_cumulative', False):
                y_pred = np.maximum(y_pred, 0.0).astype(np.float32)
        else:
            # ----- Tree models: RECURSIVE multi-step forecast -----
            # Build a fresh feature row at each step using the rolling
            # lag buffer (which grows with each new prediction), then
            # predict, append, and repeat.
            raw_cov_cols = [c for c in covariate_cols if c != 'target']
            steps_per_day = max(1, 1440 // exp_cfg.interval_minutes)
            _steps_per_hour = max(1, 60 // max(exp_cfg.interval_minutes, 1))
            rolling_windows = [
                max(2, 3 * _steps_per_hour),
                max(3, 12 * _steps_per_hour),
                max(4, 36 * _steps_per_hour),
            ]

            # Try to fetch future covariate values where available
            future_index = pd.date_range(
                start=last_ts + pd.Timedelta(minutes=exp_cfg.interval_minutes),
                periods=future_periods,
                freq=f'{exp_cfg.interval_minutes}min',
            )
            future_cov_values = {}
            if exp_cfg.covariates and self.covariate_resolver:
                for cov_cfg in exp_cfg.covariates:
                    cov_name = _cov_column_name(
                        cov_cfg, all_covs=exp_cfg.covariates,
                    )
                    if cov_name in covariate_cols and cov_cfg.role in ('future', 'both'):
                        try:
                            cov_dict = {
                                "entity_id": cov_cfg.entity,
                                "name": cov_name,
                                "future_attribute": getattr(cov_cfg, "future_attribute", "forecast"),
                                "future_value_key": getattr(cov_cfg, "future_value_key", None),
                            }
                            future_series = await self.covariate_resolver.fetch_future(
                                cov_dict, future_index,
                            )
                            if future_series is not None and not future_series.empty:
                                if cov_cfg.scale is not None:
                                    future_series = future_series * cov_cfg.scale
                                aligned = future_series.reindex(
                                    future_index
                                ).ffill().bfill()
                                # Skip all-NaN future series (fetch_future
                                # stub fallback); let carry-forward handle
                                # those covariates instead of zero-ing
                                # them via nan_to_num.
                                if aligned.notna().any():
                                    future_cov_values[cov_name] = aligned
                        except Exception as e:
                            logger.debug(f"Future fetch failed for {cov_name}: {e}")

            last_cov_vals = {
                c: float(combined[c].iloc[-1]) if c in combined.columns else 0.0
                for c in raw_cov_cols
            }

            # Deterministic future solar values for the physics-gated
            # lag buffer (mirrors _forecast_with_cached). When the user
            # has `include_clear_sky_irradiance: true` on the
            # experiment, future clear_sky_ghi is known a priori from
            # pvlib — we use it below to zero the lag buffer at night
            # steps, keeping the recursive feature vectors in the same
            # distribution the tree saw during training.
            prod_future_solar = None
            if getattr(exp_cfg, "include_clear_sky_irradiance", False):
                try:
                    loc = await self._get_site_location()
                    if loc is not None:
                        lat, lon = loc
                        from ml_forecast_lab.solar_physics import compute_solar_features
                        prod_future_solar = compute_solar_features(
                            future_index, latitude=lat, longitude=lon,
                            include_elevation=False,
                            include_clear_sky=True,
                        )
                except Exception as e:
                    logger.debug(
                        f"  _run_production_inference future solar compute failed: {e}"
                    )

            # Lag buffer: chronological, grows with each prediction
            lag_buffer = list(y[-max(n_lags, steps_per_day * 2 + 1):])

            def _build_feature_row(ts, buf, step_idx):
                row = {}
                # Temporal
                row['hour_of_day'] = ts.hour
                row['day_of_week'] = ts.dayofweek
                row['is_weekend'] = 1.0 if ts.dayofweek >= 5 else 0.0
                row['month'] = ts.month
                row['day_of_month'] = ts.day
                hr_rad = 2 * np.pi * ts.hour / 24
                dw_rad = 2 * np.pi * ts.dayofweek / 7
                row['hour_sin'] = float(np.sin(hr_rad))
                row['hour_cos'] = float(np.cos(hr_rad))
                row['dow_sin'] = float(np.sin(dw_rad))
                row['dow_cos'] = float(np.cos(dw_rad))
                # Lag features
                for lag in range(1, n_lags + 1):
                    row[f'y_lag_{lag}'] = float(buf[-lag]) if lag <= len(buf) else 0.0
                # Periodic lags
                for d in [1, 2]:
                    lag_steps = steps_per_day * d
                    row[f'y_lag_{lag_steps}'] = float(buf[-lag_steps]) if lag_steps <= len(buf) else 0.0
                for w in rolling_windows:
                    window = buf[-w:] if len(buf) >= w else buf
                    if window:
                        row[f'y_rolling_mean_{w}'] = float(np.mean(window))
                        row[f'y_rolling_std_{w}'] = float(np.std(window))
                        row[f'y_rolling_max_{w}'] = float(np.max(window))
                    else:
                        row[f'y_rolling_mean_{w}'] = 0.0
                        row[f'y_rolling_std_{w}'] = 0.0
                        row[f'y_rolling_max_{w}'] = 0.0
                row['y_diff_1'] = float(buf[-1] - buf[-2]) if len(buf) >= 2 else 0.0
                # Covariates (use future values if available, else last-known)
                for c in raw_cov_cols:
                    if c in future_cov_values:
                        try:
                            row[c] = float(future_cov_values[c].iloc[step_idx])
                        except Exception:
                            row[c] = last_cov_vals.get(c, 0.0)
                    else:
                        row[c] = last_cov_vals.get(c, 0.0)
                # Interaction features
                for c in raw_cov_cols:
                    row[f'{c}_x_hour_sin'] = row[c] * row['hour_sin']
                    row[f'{c}_x_hour_cos'] = row[c] * row['hour_cos']
                return row

            def _run_recursive_forecast():
                preds = []
                for step in range(future_periods):
                    ts = last_ts + pd.Timedelta(minutes=exp_cfg.interval_minutes * (step + 1))
                    row_dict = _build_feature_row(ts, lag_buffer, step)
                    row_vals = [row_dict.get(c, 0.0) for c in feature_cols]
                    X_row = np.array([row_vals], dtype=np.float32)
                    X_row = np.nan_to_num(X_row, nan=0.0)
                    pred = model.predict(X_row)
                    val = float(pred.ravel()[0] if hasattr(pred, 'ravel') else pred[0])
                    preds.append(val)
                    # Physics-gated lag buffer — same invariant as
                    # build_features and _forecast_with_cached: night
                    # steps push 0 into the buffer so downstream lag
                    # features stay in-distribution even when the
                    # model's raw prediction at a previous night step
                    # was slightly positive.
                    ghi_now = None
                    if prod_future_solar is not None and ts in prod_future_solar.index:
                        ghi_now = float(prod_future_solar.loc[ts, 'clear_sky_ghi'])
                    if ghi_now is not None and ghi_now <= 0:
                        lag_buffer.append(0.0)
                    else:
                        lag_buffer.append(val)
                return np.array(preds, dtype=np.float32)

            y_pred = await asyncio.get_running_loop().run_in_executor(
                None, _run_recursive_forecast,
            )

        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        # Invert log-transform if active. Training ran on log(y+1) space,
        # so predictions come back there too; lag buffers / recursion state
        # stay in log space (the model expects it) — only the final array
        # is converted back to original units before publishing.
        if exp_cfg.log_transform:
            y_pred = np.expm1(y_pred).astype(np.float32)
            y_pred = np.maximum(y_pred, 0.0)

        # Publish-boundary non-negativity clamp — same rationale as in
        # _forecast_with_cached (v2.41.0 linear-head change).
        if (
            getattr(exp_cfg, 'target_is_nonnegative', False)
            or getattr(exp_cfg, 'source_is_cumulative', False)
        ):
            y_pred = np.maximum(y_pred, 0.0).astype(np.float32)

        logger.info(
            f"Forecast curve: {len(y_pred)} points over "
            f"{future_periods * exp_cfg.interval_minutes / 60:.0f}h, "
            f"range [{y_pred.min():.3f}, {y_pred.max():.3f}]"
        )

        ds_future = pd.DatetimeIndex([
            last_ts + pd.Timedelta(minutes=exp_cfg.interval_minutes * (i + 1))
            for i in range(len(y_pred))
        ])

        # 7. Publish forecast sensors via shared helper
        # Recent actuals for context (last 24h). The combined index is naive
        # UTC (the SQLite cache strips tz); localize before isoformat so the
        # serialized strings carry "+00:00" and JS parses them correctly.
        recent_n = min(int(24 * 60 / exp_cfg.interval_minutes), len(combined))
        recent_idx = combined.index[-recent_n:]
        if recent_idx.tz is None:
            recent_idx = recent_idx.tz_localize("UTC")
        # Recent actuals live in log(y+1) space when log_transform is on
        # (combined["target"] was fed through apply_log_transform upstream).
        # Invert for display so the dashboard shows original units.
        recent_actuals_y = combined["target"].values[-recent_n:]
        if exp_cfg.log_transform:
            recent_actuals_y = np.maximum(np.expm1(recent_actuals_y), 0.0)
        recent_actuals = [
            {"datetime": ts.isoformat(), "value": round(float(val), 4)}
            for ts, val in zip(recent_idx, recent_actuals_y)
        ]

        await self._publish_forecast_sensors(
            exp_cfg=exp_cfg,
            y_pred=y_pred,
            ds_future=ds_future,
            model_name=prod_model_name,
            last_trained_iso=datetime.now(timezone.utc).isoformat(),
            extra_main_attrs={
                "train_time_seconds": round(train_time, 1),
                "recent_actuals": recent_actuals,
            },
        )

        # Update web app status
        if self.web_app:
            status = self.web_app.state.appstate.experiment_statuses.get(
                exp_cfg.name
            )
            if status:
                status.best_model = prod_model_name
                status.mode = "production"

        logger.info(f"")
        logger.info(f"  {'─' * 50}")
        logger.info(f"  Production inference complete")
        logger.info(f"  Model: {prod_model_name}, trained in {train_time:.1f}s")
        logger.info(f"  Forecast: {len(y_pred)} points, {future_periods * exp_cfg.interval_minutes / 60:.0f}h ahead")
        logger.info(f"  Next interval: {round(float(y_pred[0]), 4)} {exp_cfg.units or ''}")
        logger.info(f"  {'─' * 50}")
        logger.info(f"{'=' * 60}")
        logger.info(f"")

    async def _run_update_cycle(self, next_update: datetime):
        """
        Run a full update cycle in the background.

        This runs as an asyncio task so the web server remains responsive
        during model training.
        """
        self._update_running = True
        try:
            logger.info("=== Starting scheduled update cycle ===")

            # Reload config
            await self.load_config()

            # Run updates for each experiment
            for exp_cfg in self.config.experiments:
                is_lab = exp_cfg.mode == "lab"
                await self.update_experiment(exp_cfg.name, is_lab)

            # Publish heartbeat
            await self.publish_heartbeat()

            logger.info(
                f"Update cycle completed. Next update at {next_update.isoformat()}"
            )

            # Update web app state with next update time
            if self.web_app:
                now = datetime.now(timezone.utc)
                for exp_cfg in self.config.experiments:
                    status = self.web_app.state.appstate.experiment_statuses.get(
                        exp_cfg.name
                    )
                    if status:
                        status.next_update_in_seconds = int(
                            (next_update - now).total_seconds()
                        )

        except Exception as e:
            logger.error(f"Error in update cycle: {e}", exc_info=True)
        finally:
            self._update_running = False

    async def _retrain_single(self, exp_cfg):
        """Retrain a single experiment (per-experiment timer)."""
        self._update_running = True
        try:
            # Invalidate stale cache when the production model changes.
            # Without this, intermediate forecast cycles between config
            # reload and the next retrain would use the old model's cache
            # (with its old exp_cfg, feature_cols, seq_kwargs), which can
            # silently produce wrong predictions or crash.
            cached = self._cached_models.get(exp_cfg.name)
            if cached and exp_cfg.mode == "production":
                cached_model = cached.get("model_name", "")
                wanted_model = exp_cfg.production_model or ""
                if wanted_model and cached_model != wanted_model:
                    logger.info(
                        f"  Production model changed ({cached_model} → {wanted_model})"
                        f" — clearing stale cache for {exp_cfg.name}"
                    )
                    del self._cached_models[exp_cfg.name]

            if exp_cfg.mode == "lab":
                await self.update_experiment(exp_cfg.name, is_lab_mode=True)
            else:
                await self._retrain_and_cache(exp_cfg)
            await self.publish_heartbeat()

            # Age-based forecast_log retention (audit F5). Before
            # v2.41.0 the log was only pruned on champion change, so a
            # stable production experiment grew it without bound
            # (~69k rows / 16.5 MB per experiment-month at 30-min
            # cadence × 48 horizons) and every analytics query slowed
            # with it. 120 days keeps the UI's largest window (90 days)
            # fully served with margin. Runs on the retrain cadence
            # (~daily) so the delete is small and cheap.
            if self.history_db:
                try:
                    pruned = await asyncio.to_thread(
                        self.history_db.cleanup_forecast_log,
                        exp_cfg.name,
                        datetime.now(timezone.utc).replace(tzinfo=None)
                        - timedelta(days=FORECAST_LOG_RETENTION_DAYS),
                    )
                    if pruned:
                        logger.info(
                            f"  forecast_log retention: pruned {pruned} "
                            f"rows older than "
                            f"{FORECAST_LOG_RETENTION_DAYS}d for "
                            f"{exp_cfg.name}"
                        )
                except Exception as e:
                    logger.warning(
                        f"  forecast_log retention prune failed for "
                        f"{exp_cfg.name}: {e}"
                    )
                # Age-based retention for the external forecast log
                # (attribute-mode third-party trajectories), on its OWN
                # configurable window (default 60d, System tab) — separate
                # from forecast_log's 120d because external data is
                # higher-volume. State-mode external caches are pruned at
                # capture time alongside actuals (to the experiment's
                # max_age); this bounds the trajectory log.
                ext_retention = int(getattr(
                    self.config, "external_forecast_retention_days", 60,
                ))
                try:
                    ext_pruned = await asyncio.to_thread(
                        self.history_db.cleanup_external_forecast_log,
                        exp_cfg.name,
                        datetime.now(timezone.utc).replace(tzinfo=None)
                        - timedelta(days=ext_retention),
                    )
                    if ext_pruned:
                        logger.info(
                            f"  external_forecast_log retention: pruned "
                            f"{ext_pruned} rows older than "
                            f"{ext_retention}d for {exp_cfg.name}"
                        )
                except Exception as e:
                    logger.warning(
                        f"  external_forecast_log retention prune failed for "
                        f"{exp_cfg.name}: {e}"
                    )
        except Exception as e:
            logger.error(f"Retrain failed for {exp_cfg.name}: {e}", exc_info=True)
            # Surface the error in the web UI so the user doesn't have to
            # dig through log files to find out what went wrong.
            if self.web_app:
                status = self.web_app.state.appstate.experiment_statuses.get(
                    exp_cfg.name
                )
                if status:
                    status.last_benchmark_status = "failed"
                    status.last_error = str(e)
        finally:
            self._update_running = False

    async def _retrain_queue_consumer(self):
        """Drain the retrain queue one experiment at a time."""
        if self._retrain_consumer_running:
            return
        self._retrain_consumer_running = True
        try:
            while not self._retrain_queue.empty():
                exp_cfg = await self._retrain_queue.get()
                # Skip if experiment was deleted or already running
                if exp_cfg.name in self._running_tasks:
                    self._retrain_queue.task_done()
                    continue
                # Acquire global lock so retrains and benchmarks never overlap
                async with self._training_lock:
                    task = asyncio.create_task(self._retrain_single(exp_cfg))
                    self._running_tasks[exp_cfg.name] = task
                    task.add_done_callback(
                        lambda t, n=exp_cfg.name: self._running_tasks.pop(n, None)
                    )
                    try:
                        await task
                    except asyncio.CancelledError:
                        logger.info(f"Retrain for {exp_cfg.name} was cancelled")
                    except Exception:
                        pass
                self._retrain_queue.task_done()
        finally:
            self._retrain_consumer_running = False

    async def _forecast_single(self, exp_cfg):
        """Run forecast for a single experiment (per-experiment timer).

        The scheduler reserves `_forecast_running[name] = True` synchronously
        before spawning this task, so the `finally` MUST run on every exit
        path — including the early returns below. Otherwise the flag stays
        True forever and the scheduler's `not running` guard skips this
        experiment on every subsequent tick, silently freezing all of its
        sensors while other experiments keep updating.
        """
        try:
            if exp_cfg.mode != "production":
                return
            if exp_cfg.name not in self._cached_models:
                logger.debug(f"  No cached model for {exp_cfg.name} — waiting for retrain")
                return
            await self._forecast_with_cached(exp_cfg.name)
        except Exception as e:
            logger.error(f"Forecast failed for {exp_cfg.name}: {e}", exc_info=True)
            if self.web_app:
                status = self.web_app.state.appstate.experiment_statuses.get(
                    exp_cfg.name
                )
                if status:
                    status.last_benchmark_status = "failed"
                    status.last_error = f"Forecast: {e}"
        finally:
            self._forecast_running[exp_cfg.name] = False

    async def _run_retrain_cycle(self):
        """[Legacy] Bulk retrain for all experiments (kept for backward compat)."""
        self._update_running = True
        try:
            logger.info("=== Starting retrain cycle ===")
            await self.load_config()

            for exp_cfg in self.config.experiments:
                if exp_cfg.mode == "lab":
                    await self.update_experiment(exp_cfg.name, is_lab_mode=True)
                else:
                    try:
                        await self._retrain_and_cache(exp_cfg)
                    except Exception as e:
                        logger.error(f"Retrain failed for {exp_cfg.name}: {e}", exc_info=True)

            await self.publish_heartbeat()
            logger.info("=== Retrain cycle completed ===")
        except Exception as e:
            logger.error(f"Error in retrain cycle: {e}", exc_info=True)
        finally:
            self._update_running = False

    async def _retrain_and_cache(self, exp_cfg):
        """Train a production model and cache it for fast forecast cycles."""
        from ml_forecast_lab.features import build_features
        from ml_forecast_lab.models.base import TrainingCancelled

        logger.info(f"  Retraining {exp_cfg.name}...")

        # Re-resolve the auto-inherited unit on each retrain (~daily) so a
        # value cached as empty during a cold-start miss, or a source
        # sensor whose unit was changed in HA, gets picked up. The
        # immediate post-retrain forecast then publishes with it.
        self._source_unit_cache.pop(exp_cfg.target_entity, None)

        # Cooperative cancel flag (audit F10) — checked by the epoch
        # callback below so stop-training can halt the executor thread
        # at the next epoch boundary, not just abandon it.
        _cancel_ev = threading.Event()
        self._cancel_events[exp_cfg.name] = _cancel_ev

        def _cancel_cb(**_kw):
            if _cancel_ev.is_set():
                raise TrainingCancelled(
                    f"{exp_cfg.name}: retrain cancelled via stop-training"
                )

        # Fetch and prepare data
        df = await self._fetch_and_preprocess(exp_cfg)
        if df is None:
            return
        features_df = build_features(
            df, target_col="y",
            interval_minutes=exp_cfg.interval_minutes,
            country=exp_cfg.country,
        )
        combined = features_df.copy()
        combined["target"] = df["y"]
        for col in [c for c in df.columns if c != "y"]:
            combined[col] = df[col]
        combined = combined.dropna()

        feature_cols = [c for c in combined.columns if c != "target"]
        X = combined[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)
        y = combined["target"].values.astype(np.float32)

        # Determine production model
        prod_model_name = exp_cfg.production_model
        if not prod_model_name and self.web_app:
            bench = self.web_app.state.appstate.benchmark_results.get(exp_cfg.name)
            if bench and bench.best_model_name:
                prod_model_name = bench.best_model_name
        if not prod_model_name:
            prod_model_name = exp_cfg.models_enabled[0] if exp_cfg.models_enabled else "lightgbm"

        # Create and configure model
        model = self.model_registry.create(prod_model_name)
        overrides = dict(self.config.model_overrides.get(prod_model_name, {}))
        exp_params = getattr(exp_cfg, 'model_params', {}).get(prod_model_name, {})
        if exp_params:
            overrides.update(exp_params)
        if model.is_neural and hasattr(model, 'loss_fn') and 'loss_fn' not in overrides:
            model.set_params(loss_fn=exp_cfg.loss_fn)
        _apply_patience(model, exp_cfg, overrides)
        if (model.is_neural and hasattr(model, 'optimiser')
                and 'optimiser' not in overrides):
            model.set_params(optimiser=exp_cfg.optimiser)
        if overrides:
            model.set_params(**overrides)
        if 'output_activation' not in overrides:
            _apply_output_activation(model, exp_cfg)

        # Train
        is_neural = model.is_neural
        seq_kwargs = {}
        if is_neural:
            from ml_forecast_lab.features import (
                create_sliding_windows, compute_known_future_features,
            )
            engineered = {
                'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_holiday',
            }
            engineered.update(c for c in combined.columns if c.startswith('y_lag_'))
            raw_cov_cols = [c for c in combined.columns if c not in engineered and c != 'target']
            window_size = min(48, len(combined) // 3)
            # Train with dense horizons (1, 2, ..., future_periods) so the
            # multi-head output covers every forecast step directly — no
            # interpolation or autoregression needed at inference time.
            future_periods = getattr(exp_cfg, 'future_periods', 48)
            horizon_steps = list(range(1, future_periods + 1))
            if window_size >= 12:
                # Extend each training window with future-known features at
                # horizon positions. Without this, a multi-horizon neural
                # head has only the past window to project from, and a
                # single linear layer (NLinear / SparseTSF) cannot
                # disambiguate "horizon h" from "absolute hour at h" because
                # h corresponds to different absolute hours across windows
                # ending at different times — the weights are forced into a
                # phase-smeared compromise. LSTM/CNN hit the same wall via
                # their pooled-context → linear head and tend to collapse
                # to the unconditional mean. Tree models avoid the issue
                # because their recursive inference path already passes
                # future temporal/solar features per horizon row; this
                # change brings the neural path to parity.
                loc = await self._get_site_location()
                solar_lat_lon = loc if loc is not None else None
                include_sun_elevation = 'sun_elevation' in raw_cov_cols
                include_clear_sky_ghi = 'clear_sky_ghi' in raw_cov_cols

                # User-configured future-role covariates (e.g. Solcast PV
                # forecast, met.no weather). The tree (recursive) inference
                # path always saw these at horizon positions via
                # ``future_cov_values``; the neural extended-window path
                # historically did not, so neural backends were
                # information-starved relative to tree backends in
                # benchmarks. At training time the "future" positions are
                # actually past timestamps we have ground-truth observations
                # for — use the in-sample historical values from ``combined``.
                # The matching inference-side call in _forecast_with_cached
                # fetches the HA forecast attribute for real-future
                # timestamps.
                future_cov_for_neural = _collect_train_future_covariates(
                    combined, exp_cfg
                )
                neural_future_cov_names = list(future_cov_for_neural)
                if neural_future_cov_names:
                    logger.info(
                        f"  Neural future covariates (horizon-aware): "
                        f"{neural_future_cov_names}"
                    )

                future_features_df = compute_known_future_features(
                    combined.index,
                    add_temporal=True,
                    country=getattr(exp_cfg, 'country', None),
                    solar_lat_lon=solar_lat_lon,
                    include_sun_elevation=include_sun_elevation,
                    include_clear_sky_ghi=include_clear_sky_ghi,
                    future_covariate_values=future_cov_for_neural or None,
                )
                seq_X, seq_y, channel_names = create_sliding_windows(
                    combined, 'target', window_size=window_size,
                    covariate_cols=raw_cov_cols if raw_cov_cols else None,
                    add_temporal=True, horizon_steps=horizon_steps,
                    future_features_df=future_features_df,
                )
                seq_kwargs['sequence_data'] = seq_X
                # Cache the per-channel meaning so the forecast cycle can
                # verify it's feeding the model channels in the SAME order
                # they were trained on. Without this, a covariate fetch
                # that silently re-orders (e.g. a transient empty cov_series
                # at one tick, or a future build_features rearrangement)
                # would make NLinear/DLinear/etc. predict from mis-labelled
                # channels and produce nonsense (e.g. spurious early-morning
                # peaks) with no error raised. Backend fit() methods accept
                # **kwargs and silently ignore unknown keys, so passing
                # channel_names through is harmless during training; it's
                # only consumed by _forecast_with_cached. Matches what the
                # benchmark-holdout path has done for two minor releases.
                seq_kwargs['channel_names'] = channel_names
                # Mark this cache as carrying an extended (past + future)
                # window so _forecast_with_cached knows to rebuild the
                # inference tensor the same way. Old caches that pre-date
                # this flag take the legacy path (past window only). The
                # split index lets inference know where the past window
                # ends — it's the size we asked create_sliding_windows to
                # use, before the future-position extension.
                seq_kwargs['extended_window'] = True
                seq_kwargs['past_window_size'] = window_size
                seq_kwargs['future_feature_cols'] = list(future_features_df.columns)
                # Sub-list — just the columns that came from user
                # covariates with role in (future, both). The
                # deterministic columns (temporal, solar physics) can
                # be recomputed at inference from the future_index
                # alone; these need a HA history / forecast fetch.
                if neural_future_cov_names:
                    seq_kwargs['future_covariate_names'] = list(
                        neural_future_cov_names
                    )
                logger.info(
                    f"  Extended training windows: "
                    f"{window_size} past + {len(horizon_steps)} future "
                    f"= {seq_X.shape[1]} steps × {seq_X.shape[2]} channels, "
                    f"future cols={list(future_features_df.columns)}"
                )
                # v2.37 PF1-PF9 diagnostic — surfaces the exact knobs the
                # neural backend will receive in fit(). When a user reports
                # "the LSTM forecast is still flat after the v2.37 upgrade"
                # this log line is the first place to look: it confirms
                # whether the past_window_size / extended_window /
                # output_activation values are what
                # the PF1-PF9 fixes expect.
                logger.info(
                    f"  PF1-PF10 diagnostics for {prod_model_name}: "
                    f"past_window_size={seq_kwargs.get('past_window_size')}, "
                    f"extended_window={seq_kwargs.get('extended_window')}, "
                    f"output_activation={getattr(model, 'output_activation', '<n/a>')}, "
                    f"use_revin={getattr(model, 'use_revin', '<n/a>')}, "
                    f"learning_rate={getattr(model, 'learning_rate', getattr(model, 'lr', '<n/a>'))}, "
                    f"optimiser={getattr(exp_cfg, 'optimiser', '<n/a>')}, "
                    f"log_transform={getattr(exp_cfg, 'log_transform', False)}, "
                    f"source_is_cumulative={getattr(exp_cfg, 'source_is_cumulative', False)}, "
                    f"target_is_nonnegative={getattr(exp_cfg, 'target_is_nonnegative', False)}"
                )
                y_train_seq = seq_y
                X_train_seq = X[-len(seq_y):]

                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, lambda: model.fit(
                        X_train_seq, y_train_seq,
                        epoch_callback=_cancel_cb, **seq_kwargs,
                    )
                )
            else:
                is_neural = False
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda: model.fit(X, y, epoch_callback=_cancel_cb),
                )
        else:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, lambda: model.fit(X, y, epoch_callback=_cancel_cb),
            )

        # Debug bundle: capture training inputs for offline analysis. Gated
        # per-experiment so disk usage stays bounded and ordinary users
        # don't accumulate noise. The matched dump_forecast call is in
        # _forecast_with_cached.
        if getattr(exp_cfg, "debug_save_training_dumps", False):
            try:
                from ml_forecast_lab.debug_dump import DebugDumper
                if self._debug_dumper is None and self._last_config_path is not None:
                    self._debug_dumper = DebugDumper.from_config_path(self._last_config_path)
                if self._debug_dumper is not None:
                    target_stats = {
                        "mean": float(np.mean(y)) if len(y) else None,
                        "std": float(np.std(y)) if len(y) else None,
                        "min": float(np.min(y)) if len(y) else None,
                        "max": float(np.max(y)) if len(y) else None,
                        "zeros": int(np.sum(y == 0)) if len(y) else 0,
                        "n_samples": int(len(y)),
                    }
                    model_params = {
                        k: getattr(model, k, None) for k in (
                            "learning_rate", "lr", "output_activation",
                            "daily_loss_weight", "use_revin", "loss_fn",
                            "optimiser", "epochs", "batch_size", "hidden_size",
                            "num_layers", "dropout",
                        )
                        if hasattr(model, k)
                    }
                    self._debug_dumper.dump_training(
                        exp_name=exp_cfg.name,
                        model_name=prod_model_name,
                        exp_cfg=exp_cfg,
                        combined=combined,
                        feature_cols=feature_cols,
                        target_stats=target_stats,
                        seq_X=seq_kwargs.get("sequence_data") if is_neural else None,
                        seq_y=y_train_seq if is_neural else None,
                        channel_names=seq_kwargs.get("channel_names") if is_neural else None,
                        seq_kwargs=seq_kwargs if is_neural else None,
                        model_params=model_params,
                    )
            except Exception as e:
                logger.debug(f"debug_dump training hook failed: {e}")

        logger.info(f"  {prod_model_name} trained on {len(X)} samples")
        # Surface the feature set the model was actually trained with, so a
        # user who enabled a covariate or solar toggle and then retrained can
        # verify that the new column actually made it into the trained model.
        solar_in_features = [
            c for c in feature_cols if c in ("sun_elevation", "clear_sky_ghi")
        ]
        cov_in_features = [
            c for c in feature_cols
            if c not in ("target",)
            and not c.startswith("y_")
            and not c.endswith("_x_hour_sin")
            and not c.endswith("_x_hour_cos")
            and c not in (
                "hour_of_day", "day_of_week", "is_weekend", "month",
                "day_of_month", "hour_sin", "hour_cos", "dow_sin",
                "dow_cos", "is_holiday",
            )
        ]
        logger.info(
            f"  {exp_cfg.name} feature set: {len(feature_cols)} cols, "
            f"{len(cov_in_features)} raw covariates {cov_in_features}, "
            f"solar={solar_in_features or 'none'}"
        )

        # Cache the trained model. trained_at doubles as the model_version
        # tag on subsequent forecast_log writes — analytics queries filter
        # on it by default so predictions from this retrain don't pool
        # with previous-weights predictions under the same model_name.
        trained_at = datetime.now(timezone.utc)
        model_version = trained_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        self._cached_models[exp_cfg.name] = {
            "model": model,
            "model_name": prod_model_name,
            "feature_cols": feature_cols,
            "combined": combined,
            "exp_cfg": exp_cfg,
            "trained_at": trained_at,
            "model_version": model_version,
            "is_neural": is_neural,
            "seq_kwargs": seq_kwargs,
        }

        # Update web status — advance model_version BEFORE the post-retrain
        # forecast so its log_forecast call carries the new tag.
        if self.web_app:
            status = self.web_app.state.appstate.experiment_statuses.get(exp_cfg.name)
            if status:
                status.best_model = prod_model_name
                status.model_version = model_version
                status.last_benchmark_timestamp = trained_at.isoformat()
                status.last_benchmark_status = "completed"
                status.last_error = None  # Clear any previous error

        # Persist to disk so a restart can skip the immediate retrain for
        # this experiment and start publishing from the same weights.
        self._persist_cached_model(exp_cfg.name)

        # Publish HA lifecycle sensor so automations can fire on retrain.
        # Reuses the trained_at timestamp computed above so this sensor's
        # state, model_version, and the forecast_log model_version tag
        # all match.
        try:
            await self._publish_lifecycle_sensor(
                exp_cfg, "retrain", "completed",
                {
                    "model": prod_model_name,
                    "model_version": model_version,
                },
            )
        except Exception:
            pass

        # Also run a forecast immediately after retrain. Reserve the
        # per-experiment forecast slot first — without it a
        # concurrently-scheduled _forecast_single could publish in
        # parallel with this call, double-writing sensors and
        # forecast_log rows for the same minute (audit F15). If a
        # scheduled forecast is already mid-flight, skip ours — the
        # running one re-reads the cache entry we just stored.
        if not self._forecast_running.get(exp_cfg.name, False):
            self._forecast_running[exp_cfg.name] = True
            try:
                await self._forecast_with_cached(exp_cfg.name)
            finally:
                self._forecast_running[exp_cfg.name] = False
        else:
            logger.info(
                f"  Skipping post-retrain forecast for {exp_cfg.name} — "
                f"a scheduled forecast is already running"
            )

    @staticmethod
    def _cached_model_dir(exp_name: str) -> Path:
        """Return the per-experiment cache directory, slugifying the name
        so odd characters in experiment IDs can't escape the base path."""
        import re
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", exp_name)
        return Path("/data/ml_forecast_lab/models") / safe

    def _persist_cached_model(self, exp_name: str) -> None:
        """Serialise the in-memory cached production model to disk.

        Called at the tail of _retrain_and_cache. Writes two files to
        /data/ml_forecast_lab/models/<exp>/:
            model.bin       — backend-defined binary (pickle / torch.save)
            cache_meta.json — feature_cols, trained_at, model_version,
                              is_neural, window_size, addon_version

        Before overwriting, the previous champion (if any) is moved into
        a ``previous/`` sub-directory so a regressive retrain can be
        rolled back via the API / Roll back button. Only one generation
        of previous weights is kept to bound disk usage on an SD card.

        Swallows and logs all errors: a failed persist must never break
        the retrain path. The live _cached_models dict is the source of
        truth at runtime; disk is only read on startup.
        """
        import json
        import shutil
        cache = self._cached_models.get(exp_name)
        if not cache:
            return
        try:
            from ml_forecast_lab import __version__
            model_dir = self._cached_model_dir(exp_name)
            model_dir.mkdir(parents=True, exist_ok=True)

            model_bin = model_dir / "model.bin"
            meta_file = model_dir / "cache_meta.json"

            # Archive the current champion into previous/ before we
            # overwrite it. Single-generation cap keeps SD-card writes
            # bounded — older snapshots are silently dropped.
            try:
                if model_bin.exists() and meta_file.exists():
                    prev_dir = model_dir / "previous"
                    prev_dir.mkdir(parents=True, exist_ok=True)
                    for fname in ("model.bin", "cache_meta.json"):
                        src = model_dir / fname
                        dst = prev_dir / fname
                        if src.exists():
                            if dst.exists():
                                dst.unlink()
                            shutil.copy2(src, dst)
                    logger.debug(
                        f"  Archived previous {exp_name} cache → {prev_dir}"
                    )
            except Exception as _e:
                logger.debug(
                    f"  Could not archive previous {exp_name} cache: {_e}"
                )

            # Write-then-rename so a crash mid-save can't leave a torn
            # model.bin paired with the previous (matching-schema) meta
            # — the restore path would load it silently (audit F16).
            # The meta JSON below already uses the same pattern.
            model_bin_tmp = model_dir / "model.bin.tmp"
            cache["model"].save(str(model_bin_tmp))
            model_bin_tmp.replace(model_bin)

            # window_size is only meaningful for neural backends that
            # trained through the sliding-window path — derive it from
            # the live seq_kwargs.sequence_data shape so we don't store
            # redundant state. channel_names is persisted so the
            # forecast-cycle parity guard survives a restart (without
            # it, the first post-restart forecast would skip the guard).
            # extended_window + past_window_size + future_feature_cols
            # let the post-restart inference path reproduce the same
            # past/future split and recompute future-known features
            # without consulting the live training tensor.
            is_neural = cache.get("is_neural", False)
            seq_kwargs = cache.get("seq_kwargs", {})
            window_size = None
            channel_names = None
            extended_window = bool(seq_kwargs.get("extended_window", False))
            past_window_size = seq_kwargs.get("past_window_size")
            future_feature_cols = seq_kwargs.get("future_feature_cols")
            if is_neural and "sequence_data" in seq_kwargs:
                window_size = int(seq_kwargs["sequence_data"].shape[1])
            if is_neural and seq_kwargs.get("channel_names") is not None:
                channel_names = list(seq_kwargs["channel_names"])

            meta = {
                # schema_version bumped to 2 in v2.37 to force a re-train
                # after the neural-PV root-cause fixes (PF1-PF9). Old
                # caches written under schema_version=1 are silently
                # ignored on load and a fresh training cycle is scheduled
                # — see docs/investigations/2026-05-neural-pv.md.
                "schema_version": 2,
                "addon_version": __version__,
                "model_name": cache["model_name"],
                "feature_cols": list(cache["feature_cols"]),
                "trained_at": cache["trained_at"].isoformat(),
                "model_version": cache["model_version"],
                "is_neural": is_neural,
                "window_size": window_size,
                "channel_names": channel_names,
                "extended_window": extended_window,
                "past_window_size": (
                    int(past_window_size) if past_window_size is not None else None
                ),
                "future_feature_cols": (
                    list(future_feature_cols) if future_feature_cols is not None else None
                ),
                # Subset of future_feature_cols that came from user
                # covariates and require a HA history / forecast fetch
                # at inference. Deterministic columns (temporal /
                # solar physics) are recomputed from the future_index.
                "future_covariate_names": (
                    list(seq_kwargs.get("future_covariate_names") or [])
                ),
            }
            tmp = meta_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(meta, indent=2))
            tmp.replace(meta_file)
            logger.info(
                f"  Persisted cached {cache['model_name']} for {exp_name} "
                f"→ {model_dir}"
            )
        except Exception as e:
            logger.warning(
                f"  Failed to persist cached model for {exp_name}: {e}",
                exc_info=True,
            )

    def _restore_cached_models(self) -> None:
        """On startup, reload any production models a previous run left
        on disk so the first forecast cycle after restart can use them
        immediately instead of waiting for a fresh retrain to finish.

        Silently ignores missing / corrupt / schema-mismatched entries;
        main_loop's cache-freshness check will schedule a cold-start
        retrain for any experiment that didn't restore.
        """
        import json
        if not self.config or not self.config.experiments:
            return

        base_dir = Path("/data/ml_forecast_lab/models")
        if not base_dir.exists():
            return

        restored = 0
        for exp_cfg in self.config.experiments:
            if exp_cfg.mode != "production":
                continue
            model_dir = self._cached_model_dir(exp_cfg.name)
            meta_file = model_dir / "cache_meta.json"
            model_bin = model_dir / "model.bin"
            if not meta_file.exists() or not model_bin.exists():
                continue
            try:
                meta = json.loads(meta_file.read_text())
                # v2.37: cache schema bumped to 2 after the PF1-PF9 neural
                # fixes (see docs/investigations/2026-05-neural-pv.md). Old
                # schema=1 caches were trained against the biased RevIN /
                # degenerate anchors / collapsed backcasts, so loading them
                # would just re-publish the broken forecasts that PF1-PF9
                # were designed to fix. Force re-train by ignoring them.
                if meta.get("schema_version") != 2:
                    logger.info(
                        f"  Cached model for {exp_cfg.name} has schema "
                        f"v{meta.get('schema_version')}, ignoring (v2.37 "
                        f"PF1-PF9 fixes require schema_version=2 — a fresh "
                        f"benchmark + retrain will be scheduled)"
                    )
                    continue

                model_name = meta["model_name"]
                model = self.model_registry.create(model_name)
                model.load(str(model_bin))

                trained_at = datetime.fromisoformat(meta["trained_at"])
                is_neural = meta.get("is_neural", False)
                window_size = meta.get("window_size")

                # Rebuild a minimal seq_kwargs so _forecast_with_cached
                # can still read window_size from sequence_data.shape[1]
                # without us having to persist the full training array.
                # channel_names from the persisted meta drives the
                # post-restart parity guard against silent column-order
                # drift between train and inference.
                # extended_window / past_window_size / future_feature_cols
                # carry the new (post-v2.35.x) split-window information
                # forward so a freshly-restarted addon publishes the same
                # forecasts as the live process did before the restart.
                # Old metas missing these keys fall back to the legacy
                # past-only path — those caches will be retrained on the
                # next schedule tick anyway.
                seq_kwargs: Dict = {}
                if is_neural and window_size:
                    seq_kwargs["sequence_data"] = np.zeros(
                        (1, window_size, 1), dtype=np.float32
                    )
                    cached_ch = meta.get("channel_names")
                    if cached_ch:
                        seq_kwargs["channel_names"] = list(cached_ch)
                    if meta.get("extended_window"):
                        seq_kwargs["extended_window"] = True
                        past_ws = meta.get("past_window_size")
                        if past_ws is not None:
                            seq_kwargs["past_window_size"] = int(past_ws)
                        ffc = meta.get("future_feature_cols")
                        if ffc is not None:
                            seq_kwargs["future_feature_cols"] = list(ffc)
                        fcn = meta.get("future_covariate_names")
                        if fcn:
                            seq_kwargs["future_covariate_names"] = list(fcn)

                self._cached_models[exp_cfg.name] = {
                    "model": model,
                    "model_name": model_name,
                    "feature_cols": meta["feature_cols"],
                    "combined": None,  # re-fetched on first forecast
                    "exp_cfg": exp_cfg,
                    "trained_at": trained_at,
                    "model_version": meta["model_version"],
                    "is_neural": is_neural,
                    "seq_kwargs": seq_kwargs,
                }
                # Also mirror into web status so the UI reflects the
                # restored champion immediately, not "unknown" until the
                # next retrain writes one.
                if self.web_app:
                    status = self.web_app.state.appstate.experiment_statuses.get(
                        exp_cfg.name
                    )
                    if status:
                        status.best_model = model_name
                        status.model_version = meta["model_version"]
                        status.last_benchmark_timestamp = meta["trained_at"]
                age_m = (datetime.now(timezone.utc) - trained_at).total_seconds() / 60
                logger.info(
                    f"  Restored cached {model_name} for {exp_cfg.name} "
                    f"(trained {age_m:.0f}m ago, addon "
                    f"v{meta.get('addon_version', '?')})"
                )
                restored += 1
            except Exception as e:
                logger.warning(
                    f"  Failed to restore cached model for {exp_cfg.name}: {e}",
                    exc_info=True,
                )

        if restored:
            logger.info(
                f"Restored {restored} cached production model(s) from disk — "
                f"skipping immediate retrain for these experiments"
            )

    def _rollback_cached_model(self, exp_name: str) -> tuple[bool, Optional[str]]:
        """Roll a regressive retrain back to the previous champion.

        Looks for ``previous/`` in the per-experiment cache dir. If
        present, swaps current ↔ previous on disk and re-loads the
        in-memory cache. Returns ``(success, message)``.
        """
        import json
        import shutil
        if not self.model_registry:
            return False, "Model registry not initialised"
        if not self.config:
            return False, "Config not loaded"
        exp_cfg = next(
            (e for e in self.config.experiments if e.name == exp_name), None,
        )
        if exp_cfg is None:
            return False, "Experiment not found"

        model_dir = self._cached_model_dir(exp_name)
        prev_dir = model_dir / "previous"
        prev_bin = prev_dir / "model.bin"
        prev_meta = prev_dir / "cache_meta.json"
        if not prev_bin.exists() or not prev_meta.exists():
            return False, "No previous version on disk to roll back to"

        try:
            # Swap: current → previous_tmp, previous → current, previous_tmp → previous.
            # Keeps the rollback symmetric so it can be performed again
            # (toggling between two generations).
            tmp_dir = model_dir / "_swap_tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            for fname in ("model.bin", "cache_meta.json"):
                src = model_dir / fname
                if src.exists():
                    shutil.move(str(src), str(tmp_dir / fname))
            for fname in ("model.bin", "cache_meta.json"):
                src = prev_dir / fname
                if src.exists():
                    shutil.move(str(src), str(model_dir / fname))
            for fname in ("model.bin", "cache_meta.json"):
                src = tmp_dir / fname
                if src.exists():
                    shutil.move(str(src), str(prev_dir / fname))
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as e:
            return False, f"Swap failed: {e}"

        # Re-load into the live cache.
        try:
            meta = json.loads((model_dir / "cache_meta.json").read_text())
            model_name = meta["model_name"]
            model = self.model_registry.create(model_name)
            model.load(str(model_dir / "model.bin"))
            trained_at = datetime.fromisoformat(meta["trained_at"])
            is_neural = meta.get("is_neural", False)
            window_size = meta.get("window_size")
            seq_kwargs: Dict = {}
            if is_neural and window_size:
                seq_kwargs["sequence_data"] = np.zeros(
                    (1, window_size, 1), dtype=np.float32
                )
                cached_ch = meta.get("channel_names")
                if cached_ch:
                    seq_kwargs["channel_names"] = list(cached_ch)
                if meta.get("extended_window"):
                    seq_kwargs["extended_window"] = True
                    past_ws = meta.get("past_window_size")
                    if past_ws is not None:
                        seq_kwargs["past_window_size"] = int(past_ws)
                    ffc = meta.get("future_feature_cols")
                    if ffc is not None:
                        seq_kwargs["future_feature_cols"] = list(ffc)
                    fcn = meta.get("future_covariate_names")
                    if fcn:
                        seq_kwargs["future_covariate_names"] = list(fcn)
            self._cached_models[exp_name] = {
                "model": model,
                "model_name": model_name,
                "feature_cols": meta["feature_cols"],
                "combined": None,
                "exp_cfg": exp_cfg,
                "trained_at": trained_at,
                "model_version": meta["model_version"],
                "is_neural": is_neural,
                "seq_kwargs": seq_kwargs,
            }
            if self.web_app:
                status = self.web_app.state.appstate.experiment_statuses.get(exp_name)
                if status:
                    status.best_model = model_name
                    status.model_version = meta["model_version"]
                    status.last_benchmark_timestamp = meta["trained_at"]
                    status.last_error = None
            logger.info(
                "Rolled back %s to previous champion %s (trained %s)",
                exp_name, model_name, meta["trained_at"],
            )
            return True, f"Rolled back to {model_name} trained {meta['trained_at']}"
        except Exception as e:
            return False, f"Rollback restore failed: {e}"

    async def _run_forecast_cycle(self):
        """
        Forecast cycle: use cached models for fast inference + publish sensors.
        If no cached model exists, skip (retrain cycle will create one).
        """
        try:
            await self.load_config()
            for exp_cfg in self.config.experiments:
                if exp_cfg.mode != "production":
                    continue
                if exp_cfg.name not in self._cached_models:
                    logger.debug(f"  No cached model for {exp_cfg.name} — waiting for retrain")
                    continue
                self._forecast_running[exp_cfg.name] = True
                try:
                    await self._forecast_with_cached(exp_cfg.name)
                except Exception as e:
                    logger.error(f"Forecast failed for {exp_cfg.name}: {e}", exc_info=True)
                finally:
                    self._forecast_running[exp_cfg.name] = False
        except Exception as e:
            logger.error(f"Error in forecast cycle: {e}", exc_info=True)

    async def _capture_external_forecast(
        self, exp_cfg, issued_at: datetime, ds_future: pd.DatetimeIndex,
    ) -> None:
        """Snapshot each configured third-party forecast for head-to-head
        scoring on the experiment's "External Comparison" tab.

        For every entry in ``exp_cfg.external_forecasts``:

        - ``state`` mode: read the external entity's current state and append
          a single grid point to its cached-history table (``store_history``),
          then prune that table to ``max_age`` days. Accumulated over cycles
          this builds the external series the same way actuals are cached —
          independent of HA recorder retention.
        - ``attribute`` mode: resolve the trajectory onto this cycle's forecast
          grid (reusing the covariate ``fetch_future`` resolver, which handles
          list-of-dict / date-dict attributes and the ``weather.get_forecasts``
          service shape) and log it to ``external_forecast_log`` under the
          entity as ``source`` so a per-lead-time comparison is possible.
          Note: ``fetch_future`` interpolates / edge-fills onto the grid, so
          targets beyond the external source's own horizon are forward-filled;
          the comparison surfaces per-lead sample counts so that fill is
          visible.
        """
        externals = getattr(exp_cfg, "external_forecasts", None) or []
        if not externals or not self.ha_interface or not self.history_db:
            return

        issued_naive = (
            issued_at.replace(tzinfo=None)
            if issued_at.tzinfo is not None else issued_at
        )

        for spec in externals:
            entity = getattr(spec, "entity_id", None)
            if not entity:
                continue
            mode = getattr(spec, "mode", "state") or "state"
            try:
                if mode == "attribute":
                    attr_name = getattr(spec, "attribute", "forecast") or "forecast"
                    value_key = getattr(spec, "value_key", None)
                    from ml_forecast_lab.covariates import CovariateResolver

                    resolver = getattr(self, "covariate_resolver", None)
                    if resolver is None:
                        resolver = CovariateResolver(
                            self.ha_interface, history_db=self.history_db,
                        )
                    cov_cfg = {
                        "entity_id": entity,
                        "future_attribute": attr_name,
                        "future_value_key": value_key,
                    }
                    series = await resolver.fetch_future(cov_cfg, ds_future)
                    if series is None or len(series) == 0:
                        continue
                    series = series.dropna()
                    if series.empty:
                        continue
                    targets = [
                        pd.Timestamp(t).to_pydatetime() for t in series.index
                    ]
                    targets = [
                        t.replace(tzinfo=None) if t.tzinfo is not None else t
                        for t in targets
                    ]
                    values = [float(v) for v in series.values]
                    n = await asyncio.to_thread(
                        self.history_db.log_external_forecast,
                        experiment=exp_cfg.name,
                        source=spec.source_key,
                        issued_at=issued_naive,
                        targets=targets,
                        values=values,
                    )
                    if n:
                        logger.info(
                            f"  Logged {n} external_forecast_log rows for "
                            f"{exp_cfg.name} ({entity}.{attr_name})"
                        )
                else:  # state mode
                    from ml_forecast_lab.ha_interface import state_to_float

                    raw = await self.ha_interface.get_state(entity, default=None)
                    val = state_to_float(raw)
                    if val is None:
                        logger.debug(
                            f"  External forecast (state) for {exp_cfg.name}: "
                            f"{entity} state {raw!r} not numeric"
                        )
                        continue
                    table = self.history_db.safe_table_name(entity)
                    df = pd.DataFrame({"ds": [issued_naive], "value": [float(val)]})
                    await asyncio.to_thread(self.history_db.store_history, table, df)
                    # Retention: keep the external cache bounded like actuals.
                    oldest = datetime.now(timezone.utc).replace(tzinfo=None) \
                        - timedelta(days=exp_cfg.max_age)
                    await asyncio.to_thread(
                        self.history_db.cleanup, table, oldest,
                    )
            except Exception as e:
                logger.warning(
                    f"  External forecast capture failed for "
                    f"{exp_cfg.name} / {entity}: {e}"
                )

    async def _publish_forecast_sensors(
        self,
        exp_cfg,
        y_pred: np.ndarray,
        ds_future: pd.DatetimeIndex,
        model_name: str,
        last_trained_iso: str,
        extra_main_attrs: Optional[dict] = None,
        y_pred_upper: Optional[np.ndarray] = None,
        y_pred_lower: Optional[np.ndarray] = None,
        interval_level: Optional[float] = None,
    ) -> None:
        """Publish forecast sensors to Home Assistant.

        Always publishes:

        - ``_forecast``: per-interval forecast curve (main sensor)
        - ``_cumulative``: integrated forecast. Semantics derived from the
          experiment config — if ``source_is_cumulative`` and ``reset_daily``
          are both set, resets at local midnight and is seeded with the
          current target value so the curve stays continuous with actuals.
          Otherwise a running ``cumsum`` from zero across the horizon.

        Conditionally publishes:

        - ``_interval``: per-interval increments (only when
          ``source_is_cumulative`` — otherwise duplicates ``_forecast``).
        - ``_upper_{pct}`` / ``_lower_{pct}``: conformal interval bounds when
          ``y_pred_upper`` / ``y_pred_lower`` are provided (pct = level*100).
        """
        if not self.ha_interface:
            logger.warning(
                f"  Skipping sensor publish for {exp_cfg.name}: no HA interface"
            )
            return
        if len(y_pred) == 0:
            logger.warning(
                f"  Skipping sensor publish for {exp_cfg.name}: "
                f"forecast array is empty"
            )
            return

        # The post-hoc "clear_sky_ghi = 0 → y_pred = 0" hard clamp from
        # v2.27.8 used to live here. It was removed in v2.27.9 after
        # the train-/inference-side fix in features.build_features and
        # _run_recursive_forecast: the model now sees physics-gated lag
        # features during both training and recursive inference, so its
        # own learned response produces near-zero at night — no
        # post-processing needed. Leaving the clamp in would mask real
        # model errors at dusk/dawn and create a visible cliff.

        # One timestamp shared across log_forecast, every sensor's attrs,
        # and the cumulative sensor's "today" partition — so within a single
        # publish cycle every downstream consumer sees the same issuance
        # instant. Without this the forecast_log's issued_at, the cumulative
        # reset-daily boundary, and each sensor's HA last_updated all drift
        # apart by the time the sequential awaits complete.
        issued_at = datetime.now(timezone.utc)
        issued_at_iso = issued_at.isoformat()

        units = await self._resolve_units(exp_cfg)
        publish_name = exp_cfg.publish_name or exp_cfg.name
        prefix = exp_cfg.publish_prefix
        base_entity = f"sensor.{prefix}{publish_name}"
        future_periods = len(y_pred)
        extra_main_attrs = extra_main_attrs or {}

        # Ensure ds_future is tz-aware UTC. The upstream pipeline strips
        # timezones to keep the SQLite cache and pandas operations naive,
        # but `isoformat()` on a naive Timestamp produces a string with no
        # timezone marker (e.g. "2026-04-09T20:00:00"). JavaScript's
        # `new Date(...)` then interprets such strings as LOCAL time, which
        # introduces a UTC-offset shift in dashboard charts (e.g. ~1h in BST,
        # 5h in EST). Localizing to UTC here makes `isoformat()` emit
        # "...+00:00" so JS converts it back to local time correctly.
        if ds_future.tz is None:
            ds_future_aware = ds_future.tz_localize("UTC")
        else:
            ds_future_aware = ds_future.tz_convert("UTC")

        # Per-interval forecast list (used by main + interval sensors)
        forecast_list = [
            {"datetime": ts.isoformat(), "value": round(float(val), 4)}
            for ts, val in zip(ds_future_aware, y_pred)
        ]
        next_val = round(float(y_pred[0]), 4)

        unit_str = f" {units}" if units else ""
        horizon_min = future_periods * exp_cfg.interval_minutes
        horizon_hours = horizon_min / 60.0
        logger.info(
            f"Publishing forecast for {exp_cfg.name}: "
            f"base={base_entity}, model={model_name}, "
            f"horizon={future_periods}×{exp_cfg.interval_minutes}min "
            f"({horizon_hours:.1f}h), next={next_val}{unit_str}"
        )

        # Auto-compute conformal 80% interval bands when bounds aren't
        # provided explicitly and we have a residual history.
        #
        # NOTE on method: this is split conformal prediction with a
        # rolling residual buffer (capped at `max_age_days=14`), NOT
        # ACI / Gibbs-Candès (2021) or any other adaptive variant —
        # there is no α_t update step, no γ-step, no exponential
        # reweighting. Quantiles come from deployed forecast / actual
        # pairs in forecast_log so no extra model fit is required. The
        # cost is that residuals aren't strictly exchangeable (trend,
        # weekly seasonality, regime shifts) so finite-sample coverage
        # is approximate; the realised-coverage diagnostic on the
        # Forecast Accuracy tab is the user-visible signal that the
        # bands need recalibrating. Cold-start returns no quantiles →
        # bands stay absent until the residual buffer fills, which
        # surfaces naturally as "point-only forecast" in the UI.
        # See DOCS.md "Conformal band calibration" for how the buffer
        # behaves across retrains.
        if (
            y_pred_upper is None
            and y_pred_lower is None
            and self.history_db
            and exp_cfg.mode == "production"
        ):
            try:
                actuals_table = self.history_db.safe_table_name(
                    exp_cfg.target_entity
                )
                target_level = (
                    interval_level
                    if interval_level is not None
                    else float(getattr(exp_cfg, 'conformal_coverage', 0.8))
                )
                # Pin residual quantiles to the current model_version
                # when we have enough calibrated residuals for it;
                # otherwise pool across all weight regimes of this
                # model so bands still get published during the
                # cold-start period right after a retrain.
                #
                # Without this fallback, v2.24.0 introduced a
                # regression where the conformal query filtered so
                # strictly to the fresh (hours-old) model_version that
                # it returned zero usable quantiles. `have_intervals`
                # then stayed False, so the _upper_{pct} / _lower_{pct}
                # sensors stopped being written at all — HA kept
                # showing their stale pre-retrain values and the user
                # saw "some forecast sensors aren't updating".
                cached = self._cached_models.get(exp_cfg.name) or {}
                current_version = cached.get("model_version")
                # Offloaded: this query joins forecast_log against the
                # actuals grid and scales with both; running it inline
                # froze the event loop (web UI + scheduler) for the
                # duration of every publish cycle (audit F3).
                cq = await asyncio.to_thread(
                    self.history_db.get_conformal_quantiles,
                    exp_cfg.name,
                    actuals_table,
                    level=target_level,
                    model_name=model_name,
                    model_version=current_version,
                    interval_minutes=exp_cfg.interval_minutes,
                )
                if (
                    current_version
                    and (cq.get("fallback_quantile") is None
                         or cq.get("total_samples", 0) < 10)
                ):
                    cq_all = await asyncio.to_thread(
                        self.history_db.get_conformal_quantiles,
                        exp_cfg.name,
                        actuals_table,
                        level=target_level,
                        model_name=model_name,
                        model_version=None,
                        interval_minutes=exp_cfg.interval_minutes,
                    )
                    if cq_all.get("fallback_quantile") is not None:
                        logger.info(
                            f"  Conformal bands: falling back to all-versions "
                            f"pool for {exp_cfg.name} (current version has "
                            f"{cq.get('total_samples', 0)} residuals, need >=10)"
                        )
                        cq = cq_all
                quantiles = cq.get("quantiles") or {}
                fallback = cq.get("fallback_quantile")
                if fallback is not None:
                    bucket_min = max(1, int(exp_cfg.interval_minutes))
                    # Lead-minutes w.r.t. the effective issuance time
                    # (one interval before ds_future[0]); matches the
                    # convention used by log_forecast's lead computation
                    # to within a retrieval-cycle of clock skew.
                    issued_ref = ds_future[0] - pd.Timedelta(
                        minutes=exp_cfg.interval_minutes
                    )
                    lead_min_arr = np.array([
                        int((ts - issued_ref).total_seconds() / 60)
                        for ts in ds_future
                    ], dtype=int)
                    lead_buckets = (lead_min_arr // bucket_min) * bucket_min
                    q_vec = np.array([
                        quantiles.get(int(b), fallback)
                        for b in lead_buckets
                    ], dtype=float)
                    y_pred_upper = (y_pred + q_vec).astype(np.float32)
                    y_pred_lower = (y_pred - q_vec).astype(np.float32)
                    if getattr(exp_cfg, "source_is_cumulative", False):
                        y_pred_lower = np.maximum(y_pred_lower, 0.0)
                    interval_level = target_level
                    logger.info(
                        f"  Conformal {int(target_level*100)}% band: "
                        f"n={cq.get('total_samples', 0)} residuals, "
                        f"median width={float(np.median(q_vec*2)):.3f}"
                    )
            except Exception as e:
                logger.warning(f"  Conformal band computation failed: {e}", exc_info=True)

        # Optional conformal interval lists. Both arrays must be present
        # and sized to y_pred to be considered valid.
        have_intervals = (
            y_pred_upper is not None
            and y_pred_lower is not None
            and len(y_pred_upper) == len(y_pred)
            and len(y_pred_lower) == len(y_pred)
        )
        upper_list: list = []
        lower_list: list = []
        if have_intervals:
            upper_list = [
                {"datetime": ts.isoformat(), "value": round(float(val), 4)}
                for ts, val in zip(ds_future_aware, y_pred_upper)
            ]
            lower_list = [
                {"datetime": ts.isoformat(), "value": round(float(val), 4)}
                for ts, val in zip(ds_future_aware, y_pred_lower)
            ]

        if forecast_list:
            vals = [p["value"] for p in forecast_list]
            logger.debug(
                f"  Forecast attribute: {len(forecast_list)} points, "
                f"dt[0]={forecast_list[0]['datetime']}, "
                f"dt[-1]={forecast_list[-1]['datetime']}, "
                f"val range=[{min(vals):.4f}, {max(vals):.4f}]"
            )

        # --- Log forecast evolution (non-blocking) ---------------------------
        if self.history_db and exp_cfg.mode == "production":
            try:
                forecast_type = (
                    "retrain" if extra_main_attrs.get("train_time_seconds")
                    else "cached"
                )
                # Tag each row with the training timestamp of the cached
                # model so analytics queries segregate pre- and post-
                # retrain cycles under the same model_name.
                cached = self._cached_models.get(exp_cfg.name) or {}
                model_version = cached.get("model_version")
                n_logged = await asyncio.to_thread(
                    self.history_db.log_forecast,
                    experiment=exp_cfg.name,
                    issued_at=issued_at,
                    targets=ds_future_aware.tolist(),
                    predictions=y_pred.tolist(),
                    model_name=model_name,
                    forecast_type=forecast_type,
                    upper_bounds=(
                        y_pred_upper.tolist() if have_intervals else None
                    ),
                    lower_bounds=(
                        y_pred_lower.tolist() if have_intervals else None
                    ),
                    model_version=model_version,
                )
                # One INFO line per cycle so the operator can see
                # forecast_log actually accumulating — without this,
                # "why is my Forecast Accuracy chart empty?" had no
                # observable signal in the add-on logs. Keep it concise;
                # runs 48x/day/experiment at 30-min cadence.
                if n_logged:
                    extras = [f"model={model_name}", forecast_type]
                    if model_version:
                        extras.append(f"v={model_version}")
                    if have_intervals:
                        extras.append("bands")
                    logger.info(
                        f"  Logged {n_logged} forecast_log rows "
                        f"for {exp_cfg.name} ({', '.join(extras)})"
                    )
            except Exception as e:
                logger.warning(f"Failed to log forecast evolution: {e}")

        # --- Capture the external (third-party) forecast ---------------------
        # Logged at the SAME issuance instant as the app's own forecast so the
        # External Comparison tab scores like-for-like cycles. Production-only
        # and best-effort — a flaky external sensor never blocks publishing.
        if (
            self.history_db
            and exp_cfg.mode == "production"
            and getattr(exp_cfg, "external_forecasts", None)
        ):
            try:
                await self._capture_external_forecast(
                    exp_cfg, issued_at, ds_future_aware,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to capture external forecast for "
                    f"{exp_cfg.name}: {e}"
                )

        # Common attrs applied to every sensor in this cycle. Sharing
        # last_trained + issued_at lets the dashboard correlate bounds,
        # cumulative, and accuracy back to the same forecast issuance
        # even though HA assigns each entity its own last_updated.
        common_attrs = {
            "model": model_name,
            "last_trained": last_trained_iso,
            "issued_at": issued_at_iso,
        }

        # Build (entity_id, state, attrs) payloads synchronously; the
        # actual HTTP POSTs are fired together at the end so every sensor
        # lands at effectively the same HA last_updated.
        #
        # `skipped_sensors` captures conditionally-omitted entities with a
        # reason code so the end-of-cycle manifest log explains why an
        # entity's HA `last_updated` is stale ("no _upper_80 in the
        # payload this cycle → bands unavailable"), rather than leaving
        # the operator to guess between "failed" and "never attempted".
        payloads: list[tuple[str, str, dict]] = []
        skipped_sensors: list[tuple[str, str]] = []

        # --- 1. Main forecast sensor (always published) ----------------------
        main_attrs = {
            "friendly_name": f"{publish_name} Forecast",
            "unit_of_measurement": units,
            "icon": "mdi:chart-timeline-variant-shimmer",
            "state_class": "measurement",
            "forecast_periods": future_periods,
            "interval_minutes": exp_cfg.interval_minutes,
            "forecast": forecast_list,
            **common_attrs,
        }
        if have_intervals:
            main_attrs["forecast_upper"] = upper_list
            main_attrs["forecast_lower"] = lower_list
            if interval_level is not None:
                main_attrs["interval_level"] = interval_level
        main_attrs.update(extra_main_attrs)
        payloads.append((f"{base_entity}_forecast", str(next_val), main_attrs))

        # --- 2. Interval sensor (only when source is cumulative) ------------
        # When the source is already per-interval, _interval duplicates the
        # main _forecast sensor. Only meaningful when the forecast values
        # are interval deltas reconstructed from a cumulative source.
        if exp_cfg.source_is_cumulative:
            interval_attrs = {
                "friendly_name": f"{publish_name} Interval Forecast",
                "unit_of_measurement": units,
                "icon": "mdi:chart-bar",
                "state_class": "measurement",
                "interval_minutes": exp_cfg.interval_minutes,
                "forecast": forecast_list,
                **common_attrs,
            }
            payloads.append(
                (f"{base_entity}_interval", str(next_val), interval_attrs)
            )
        else:
            skipped_sensors.append(("_interval", "source_not_cumulative"))

        # --- 3. Cumulative sensor (always published) ------------------------
        # Behaviour is derived from source semantics rather than a separate
        # flag:
        #   source_is_cumulative + reset_daily → resets at local midnight
        #     and is seeded with the current target value so the forecast
        #     meets actuals at the join point, mirroring today-style HA
        #     energy sensors.
        #   otherwise → plain cumsum across the horizon anchored at zero
        #     (end-of-horizon projection, not a monotonic counter).
        if exp_cfg.source_is_cumulative and exp_cfg.reset_daily:
            try:
                from zoneinfo import ZoneInfo
                tz_name = (self.config.timezone if self.config else None) or "UTC"
                local_tz = ZoneInfo(tz_name)
            except Exception:
                local_tz = timezone.utc

            today_seed = 0.0
            if exp_cfg.target_entity:
                try:
                    raw = await self.ha_interface.get_state(
                        exp_cfg.target_entity, default=None,
                    )
                    if raw not in (None, "", "unknown", "unavailable"):
                        today_seed = float(raw)
                except Exception:
                    today_seed = 0.0

            now_local_date = issued_at.astimezone(local_tz).date()
            running_by_day: dict = {}
            cum_list = []
            # Headline state = the projected end-of-today total (last forecast
            # point still within today's local date). Keeps the sensor's state
            # directly comparable to sensor.<target>_today at midnight rather
            # than landing mid-way through day-after-tomorrow.
            end_of_today_value = today_seed
            for ts, val in zip(ds_future_aware, y_pred):
                local_ts = ts.tz_convert(local_tz)
                day_key = local_ts.date()
                if day_key not in running_by_day:
                    running_by_day[day_key] = (
                        today_seed if day_key == now_local_date else 0.0
                    )
                running_by_day[day_key] += float(val)
                cum_value = round(running_by_day[day_key], 4)
                if day_key == now_local_date:
                    end_of_today_value = cum_value
                cum_list.append({
                    "datetime": ts.isoformat(),
                    "value": cum_value,
                })

            cum_state = round(end_of_today_value, 4)
            cum_attrs = {
                "friendly_name": f"{publish_name} Cumulative Forecast",
                "unit_of_measurement": units,
                "icon": "mdi:chart-timeline-variant",
                # measurement (not total_increasing) — the state is a
                # per-cycle projection of today's end total, which fluctuates
                # as the seed grows and the remaining-forecast shrinks. It is
                # NOT a monotonic counter and should not be processed by HA's
                # long-term statistics engine as one.
                "state_class": "measurement",
                "forecast": cum_list,
                "resets_daily": True,
                "seeded_with": round(today_seed, 4),
                "end_of_today_value": cum_state,
                "end_of_horizon_value": (
                    round(cum_list[-1]["value"], 4) if cum_list else cum_state
                ),
                **common_attrs,
            }
        elif np.any(np.asarray(y_pred) < 0):
            # Signed predictions (e.g. net battery flow with PV export):
            # cumsum can drift unboundedly negative or positive and the
            # published "_cumulative" sensor stops matching the user's
            # mental model of the target. Skip cumulative entirely and
            # record the skip so the system page makes the omission visible.
            logger.debug(
                "Skipping cumulative sensor for %s: signed predictions "
                "(min=%.3f) would produce unbounded drift in HA.",
                publish_name, float(np.min(y_pred)),
            )
            skipped_sensors.append(("_cumulative", "signed_predictions"))
            cum_attrs = None
        else:
            cum_vals = np.cumsum(y_pred)
            cum_list = [
                {"datetime": ts.isoformat(), "value": round(float(v), 4)}
                for ts, v in zip(ds_future_aware, cum_vals)
            ]
            cum_state = round(float(cum_vals[-1]), 4)
            cum_attrs = {
                "friendly_name": f"{publish_name} Cumulative Forecast",
                "unit_of_measurement": units,
                "icon": "mdi:chart-line-stacked",
                # measurement (not total) — the state is a per-cycle snapshot
                # of the predicted end-of-horizon cumulative, not a monotonic
                # counter. Using total_* would make HA accumulate it in
                # long-term statistics and suggest it for the Energy dashboard.
                "state_class": "measurement",
                "forecast": cum_list,
                "resets_daily": False,
                **common_attrs,
            }
        if cum_attrs is not None:
            payloads.append(
                (f"{base_entity}_cumulative", str(cum_state), cum_attrs)
            )

        # --- 4. Conformal interval sensors ----------------------------------
        # Separate upper/lower entities let the user graph the band in HA
        # directly (ApexCharts area-between-series, etc) without having
        # to unpack the `forecast_upper` / `forecast_lower` attrs on the
        # main sensor. Name the level in the entity slug so dashboards
        # don't silently rescale if we ever switch to e.g. 95%.
        if have_intervals:
            lvl_pct = (
                int(round(interval_level * 100))
                if interval_level is not None else 80
            )
            upper_state = round(float(y_pred_upper[0]), 4)
            lower_state = round(float(y_pred_lower[0]), 4)
            upper_attrs = {
                "friendly_name": f"{publish_name} Forecast Upper {lvl_pct}%",
                "unit_of_measurement": units,
                "icon": "mdi:arrow-expand-up",
                "state_class": "measurement",
                "interval_level": interval_level,
                "forecast": upper_list,
                **common_attrs,
            }
            lower_attrs = {
                "friendly_name": f"{publish_name} Forecast Lower {lvl_pct}%",
                "unit_of_measurement": units,
                "icon": "mdi:arrow-expand-down",
                "state_class": "measurement",
                "interval_level": interval_level,
                "forecast": lower_list,
                **common_attrs,
            }
            payloads.append(
                (f"{base_entity}_upper_{lvl_pct}",
                 str(upper_state), upper_attrs)
            )
            payloads.append(
                (f"{base_entity}_lower_{lvl_pct}",
                 str(lower_state), lower_attrs)
            )
        else:
            # Band sensors share the cold-start reason: either no
            # conformal quantiles are calibrated yet, or the residual
            # query errored earlier in this cycle. The level defaults
            # to 80 so the skip name matches what will eventually land.
            skipped_sensors.append(("_upper_80", "no_conformal_bands"))
            skipped_sensors.append(("_lower_80", "no_conformal_bands"))

        # --- 5. Forecast accuracy sensor (always published) -----------------
        # Publishes from day one so the HA entity exists in the registry
        # immediately and dashboards can bind to it without conditional
        # glue. On cold start (no forecast_log rows yet, lab mode, DB
        # unavailable, or query failure) the state is 0 with empty
        # arrays and a `status` attribute naming the reason; state
        # transitions to "ready" once `lead_time_curve` has samples.
        acc_state: Union[int, float] = 0
        acc_attrs = {
            "friendly_name": f"{publish_name} Forecast Accuracy",
            "unit_of_measurement": units,
            "icon": "mdi:chart-scatter-plot",
            "state_class": "measurement",
            "lead_hours": [],
            "mae": [],
            "rmse": [],
            "sample_count": [],
            "total_logged": 0,
            "status": "accumulating",
            **common_attrs,
        }
        if not self.history_db:
            acc_attrs["status"] = "no_history_db"
        elif exp_cfg.mode != "production":
            acc_attrs["status"] = "lab_mode"
        else:
            try:
                actuals_table = self.history_db.safe_table_name(
                    exp_cfg.target_entity
                )
                # Offload the scan-heavy query — three CTE passes over
                # the actuals table per call, multiplied by N experiments
                # per publish cycle. Running inline would freeze the
                # event loop for the full cycle duration, starving the
                # web UI and the HA set_state HTTPX pool.
                accuracy = await asyncio.to_thread(
                    self.history_db.get_forecast_accuracy,
                    exp_cfg.name, actuals_table, 30,
                    exp_cfg.interval_minutes,
                )
                ltc = accuracy.get("lead_time_curve", {})
                rev = accuracy.get("revision_improvement", {})
                acc_attrs["total_logged"] = accuracy.get("total_logged", 0)
                if ltc.get("lead_minutes"):
                    acc_attrs["lead_hours"] = [
                        round(m / 60, 2) for m in ltc["lead_minutes"]
                    ]
                    acc_attrs["mae"] = ltc["mae"]
                    acc_attrs["rmse"] = ltc["rmse"]
                    acc_attrs["sample_count"] = ltc["sample_count"]
                    acc_attrs["status"] = "ready"
                    acc_state = round(ltc["mae"][0], 4) if ltc["mae"] else 0
                if rev:
                    acc_attrs["revision_first_mae"] = rev.get(
                        "first_forecast_mae"
                    )
                    acc_attrs["revision_latest_mae"] = rev.get(
                        "latest_forecast_mae"
                    )
                    acc_attrs["revision_improvement_pct"] = rev.get(
                        "improvement_pct"
                    )
            except Exception as e:
                acc_attrs["status"] = "error"
                logger.warning(
                    f"  Forecast accuracy prep failed for {exp_cfg.name}: {e}",
                    exc_info=True,
                )
        payloads.append((
            f"{base_entity}_forecast_accuracy",
            str(acc_state),
            acc_attrs,
        ))

        # --- Parallel publish ------------------------------------------------
        # Fire every set_state concurrently so all sensors for this
        # experiment land at effectively the same HA last_updated. Sequential
        # awaits otherwise spread them across ~300ms–1s — enough for a
        # dashboard to render "updated X seconds ago" inconsistently across
        # sensors that ought to share a cycle.
        #
        # Each publish is wrapped so one failure doesn't cancel the rest
        # (gather with return_exceptions=False would abort the sibling
        # tasks on the first raise). We also check set_state's bool return
        # value — it silently swallows HA errors and returns False instead
        # of raising, so the outer try/except alone would log success when
        # the POST actually failed.
        async def _publish_one(entity_id: str, state: str, attrs: dict):
            try:
                ok = await self.ha_interface.set_state(entity_id, state, attrs)
                return entity_id, bool(ok), None
            except Exception as err:
                return entity_id, False, err

        results = await asyncio.gather(*[
            _publish_one(eid, s, a) for eid, s, a in payloads
        ])

        succeeded = [eid for eid, ok, _ in results if ok]
        failed = [(eid, err) for eid, ok, err in results if not ok]

        # Publish manifest — one line per cycle summarising every
        # expected sensor's outcome (published, skipped with reason,
        # failed with reason). Makes "sensor.X last_updated = 3h ago"
        # directly explainable from the log: either the entity was
        # skipped each cycle (reason visible), or it failed (reason
        # visible), or it published and HA dropped it internally.
        total_expected = len(payloads) + len(skipped_sensors)
        published_items = [
            f"{eid.split('.')[-1]}={s}"
            for eid, s, _ in payloads
            if eid in succeeded
        ]
        failed_items = [
            f"{eid.split('.')[-1]}({repr(err) if err else 'set_state=False'})"
            for eid, err in failed
        ]
        skipped_items = [f"{suffix}({reason})" for suffix, reason in skipped_sensors]

        manifest_parts = [
            f"published={len(succeeded)}/{total_expected}"
        ]
        if skipped_items:
            manifest_parts.append(f"skipped={len(skipped_items)}")
        if failed_items:
            manifest_parts.append(f"failed={len(failed_items)}")

        logger.info(
            f"  Publish manifest for {exp_cfg.name} "
            f"({', '.join(manifest_parts)}): "
            f"[{', '.join(published_items)}]"
        )
        if skipped_items:
            logger.info(f"    skipped: {', '.join(skipped_items)}")
        for eid, err in failed:
            reason = repr(err) if err else "set_state returned False"
            logger.warning(
                f"    failed: {eid.split('.')[-1]} — {reason}",
                exc_info=err if err else False,
            )

        if failed and self.web_app:
            status = self.web_app.state.appstate.experiment_statuses.get(
                exp_cfg.name
            )
            if status:
                status.last_error = (
                    f"Publish: {len(failed)}/{total_expected} sensors failed "
                    f"({', '.join(e.split('.')[-1] for e, _ in failed)})"
                )

    async def _forecast_with_cached(self, experiment_name: str):
        """Run inference with a cached model and publish sensors.

        Fetches fresh recent data on each call so that lag features and
        timestamps are current, even though the model itself is cached
        from the last retrain cycle.
        """
        from ml_forecast_lab.features import build_features

        cache = self._cached_models.get(experiment_name)
        if not cache:
            return

        model = cache["model"]
        exp_cfg = cache["exp_cfg"]
        feature_cols = cache["feature_cols"]
        is_neural = cache.get("is_neural", False)
        seq_kwargs = cache.get("seq_kwargs", {})
        prod_model_name = cache.get("model_name", "unknown")

        # Fetch FRESH data so lag features and last_ts are current
        try:
            df_fresh = await self._fetch_and_preprocess(exp_cfg)
            if df_fresh is None:
                logger.warning(f"  Skipping forecast cycle for {exp_cfg.name} — insufficient data")
                return
            features_fresh = build_features(
                df_fresh, target_col="y",
                interval_minutes=exp_cfg.interval_minutes,
                country=exp_cfg.country,
            )
            combined = features_fresh.copy()
            combined["target"] = df_fresh["y"]
            for col in [c for c in df_fresh.columns if c != "y"]:
                combined[col] = df_fresh[col]
            combined = combined.dropna()
            logger.debug(f"  Fresh data: {len(combined)} samples, last={combined.index[-1]}")
        except Exception as e:
            logger.warning(f"  Fresh data fetch failed, using cached data: {e}")
            combined = cache.get("combined")
            if combined is None:
                # _restore_cached_models deliberately doesn't persist the
                # training frame — so after a restart, fresh data is the
                # only source. If that also failed, skip this cycle and
                # wait for the next forecast tick rather than crash.
                logger.warning(
                    f"  No cached data frame for {exp_cfg.name} after a "
                    f"restore — skipping this forecast cycle"
                )
                return

        n_lags = 12
        last_ts = combined.index[-1]
        y = combined["target"].values.astype(np.float32)
        # Provide enough lag values for periodic lags (same-time-yesterday etc.)
        steps_per_day = max(1, 1440 // exp_cfg.interval_minutes)
        n_lag_values = max(n_lags, steps_per_day * 2 + 1)
        lag_values = y[-n_lag_values:]
        future_periods = getattr(exp_cfg, 'future_periods', 48)

        # Prediction: neural uses dense multi-head output, tree uses recursive
        if is_neural and 'sequence_data' in seq_kwargs:
            from ml_forecast_lab.features import (
                build_inference_window, compute_known_future_features,
            )
            cached_seq_len = seq_kwargs['sequence_data'].shape[1]
            # Caches written before extended-window support set neither flag
            # nor split index; they're past-only, so seq_len IS the past
            # window size. New caches stamp `extended_window: True` plus an
            # explicit `past_window_size` so we don't have to back-derive
            # it from future_periods (which could disagree with the cached
            # tensor if exp_cfg.future_periods has been changed since).
            extended_window = seq_kwargs.get('extended_window', False)
            if extended_window:
                window_size = int(
                    seq_kwargs.get('past_window_size', cached_seq_len - future_periods)
                )
            else:
                window_size = cached_seq_len

            engineered = {
                'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_holiday',
            }
            engineered.update(c for c in combined.columns if c.startswith('y_lag_'))
            raw_cov_cols = [c for c in combined.columns if c not in engineered and c != 'target']

            # Build the single inference window whose last timestep IS
            # combined.index[-1] = last_ts. The previous implementation
            # called create_sliding_windows with horizon_steps=[1] on a
            # (window_size + 1)-row tail, which reserves the final row
            # as an unused y-label and so produces a window ending at
            # combined.iloc[-2]. That half-hour misalignment shifted
            # every published prediction one interval later than the
            # model intended, surfacing as visible time-of-day skew in
            # the forecast — most pronounced on dense 96-horizon
            # backends like NLinear / SparseTSF where the user can
            # eyeball where the peak sits relative to the labelled
            # timestamps. build_inference_window has the same
            # channel-construction logic as create_sliding_windows so
            # ch_names_now is directly comparable to the cached
            # training channel_names below.
            future_features_df = None
            if extended_window:
                # Mirror what _retrain_and_cache did: compute the
                # deterministically-known feature values for every horizon
                # timestamp and append them as future positions on the
                # inference window. The channel-parity guard below catches
                # any mismatch between the cached and current channel sets
                # (covariate added/removed/reordered since last retrain),
                # so the future positions are guaranteed to land on the
                # same channel slots the model trained against.
                future_index = pd.date_range(
                    start=last_ts + pd.Timedelta(minutes=exp_cfg.interval_minutes),
                    periods=future_periods,
                    freq=f'{exp_cfg.interval_minutes}min',
                )
                future_feature_cols = set(seq_kwargs.get('future_feature_cols') or [])
                loc = await self._get_site_location()
                solar_lat_lon = loc if loc is not None else None

                # Fetch user-configured future covariates (Solcast,
                # met.no, etc.) from HA's forecast attribute and align
                # them to the inference horizon. Closes the asymmetry
                # where tree backends saw these at each recursive step
                # via ``future_cov_values`` but the neural extended-
                # window path historically only saw past-window lags.
                # See _retrain_and_cache for the matching training-side
                # call that uses in-sample observed values for the
                # same channels.
                neural_future_cov_names = list(
                    seq_kwargs.get('future_covariate_names') or []
                )
                future_cov_for_inference: dict[str, pd.Series] = {}
                if neural_future_cov_names and exp_cfg.covariates and self.covariate_resolver:
                    cov_by_name = {
                        _cov_column_name(c, all_covs=exp_cfg.covariates): c
                        for c in exp_cfg.covariates
                    }
                    for cov_name in neural_future_cov_names:
                        cov_cfg = cov_by_name.get(cov_name)
                        if cov_cfg is None:
                            # Configured at train time, removed in YAML
                            # since — drop with a debug log; channel
                            # parity guard further down will surface
                            # the mismatch as a warning.
                            logger.debug(
                                f"  Future cov {cov_name} cached at "
                                f"train but not in current YAML — "
                                f"channel will be zero at inference"
                            )
                            continue
                        try:
                            cov_dict = {
                                "entity_id": cov_cfg.entity,
                                "name": cov_name,
                                "future_attribute": getattr(cov_cfg, "future_attribute", "forecast"),
                                "future_value_key": getattr(cov_cfg, "future_value_key", None),
                            }
                            future_series = await self.covariate_resolver.fetch_future(
                                cov_dict, future_index,
                            )
                            if future_series is None or future_series.empty:
                                continue
                            if cov_cfg.scale is not None:
                                future_series = future_series * cov_cfg.scale
                            aligned = future_series.reindex(future_index).ffill().bfill()
                            # All-NaN aligned series → leave the
                            # channel at zero (build_inference_window's
                            # NaN-safe path handles this) rather than
                            # injecting a sentinel.
                            if aligned.notna().any():
                                future_cov_for_inference[cov_name] = aligned
                        except Exception as e:
                            logger.debug(
                                f"  Future cov fetch failed for "
                                f"{cov_name}: {e}"
                            )
                if future_cov_for_inference:
                    logger.info(
                        f"  Future covariates wired to neural horizon "
                        f"positions: {list(future_cov_for_inference)}"
                    )

                future_features_df = compute_known_future_features(
                    future_index,
                    add_temporal=True,
                    country=getattr(exp_cfg, 'country', None),
                    solar_lat_lon=solar_lat_lon,
                    include_sun_elevation='sun_elevation' in future_feature_cols,
                    include_clear_sky_ghi='clear_sky_ghi' in future_feature_cols,
                    future_covariate_values=future_cov_for_inference or None,
                )
            seq_X_prod, ch_names_now = build_inference_window(
                combined, 'target', window_size=window_size,
                covariate_cols=raw_cov_cols if raw_cov_cols else None,
                add_temporal=True,
                future_features_df=future_features_df,
            )
            # Channel-parity guard.
            #
            # The model was trained with a specific (target + covariates +
            # temporal) channel ordering. At inference we rebuild the
            # window from a freshly-fetched dataframe; if anything has
            # shifted the column order — a transient empty covariate fetch
            # leaving a hole that subsequent ticks filled, a covariate
            # added/removed in Settings since the last retrain, a
            # build_features change in a future version — the model would
            # silently predict from mis-labelled channels and the
            # published forecast would look wrong in oddly time-specific
            # ways (a spike at the wrong hour, a constant offset, peaks
            # at midnight) with NO error logged.
            #
            # The cached channel_names from the retrain that produced this
            # model is the ground truth. If they disagree, log a clear
            # warning and skip publishing — better a stale sensor for one
            # tick than a confidently-published wrong forecast. The retrain
            # cycle will rebuild the cache.
            cached_ch_names = seq_kwargs.get('channel_names')
            if cached_ch_names is not None and list(ch_names_now) != list(cached_ch_names):
                logger.error(
                    f"  Channel-name mismatch for {exp_cfg.name}: cached "
                    f"({len(cached_ch_names)} channels) vs current "
                    f"({len(ch_names_now)} channels). "
                    f"Cached={list(cached_ch_names)}, "
                    f"Current={list(ch_names_now)}. "
                    f"Skipping forecast publish — wait for next retrain "
                    f"cycle to rebuild the cached channel order."
                )
                return
            # build_inference_window already returns shape (1, window, ch).
            last_window = seq_X_prod

            loop = asyncio.get_running_loop()
            multi_pred = await loop.run_in_executor(None, lambda: model.predict_sequence(last_window))
            multi_pred = multi_pred.ravel()

            # Neural models trained with dense horizons output all
            # future_periods predictions directly.
            if len(multi_pred) >= future_periods:
                y_pred = multi_pred[:future_periods].astype(np.float32)
            elif len(multi_pred) == 1:
                y_pred = np.full(future_periods, float(multi_pred[0]), dtype=np.float32)
            else:
                # Legacy cached model with fewer outputs — pad by repeating last
                y_pred = np.concatenate([
                    multi_pred.astype(np.float32),
                    np.full(future_periods - len(multi_pred),
                            float(multi_pred[-1]), dtype=np.float32),
                ])
            # Post-hoc clip for cumulative (non-negative) targets — see
            # _run_production_inference for rationale.
            if getattr(exp_cfg, 'source_is_cumulative', False):
                y_pred = np.maximum(y_pred, 0.0).astype(np.float32)
        else:
            # Tree models: RECURSIVE multi-step forecast.
            # At each step, we build a single feature row using the most recent
            # lag values (which include previous predictions), predict, then
            # roll the lag buffer forward with the new prediction.
            engineered = {
                'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_holiday', 'y_diff_1',
            }
            raw_cov_cols = [
                c for c in combined.columns
                if c != 'target'
                and c not in engineered
                and not c.startswith('y_')
                and not c.endswith('_x_hour_sin')
                and not c.endswith('_x_hour_cos')
            ]

            steps_per_day = max(1, 1440 // exp_cfg.interval_minutes)
            _steps_per_hour = max(1, 60 // max(exp_cfg.interval_minutes, 1))
            rolling_windows = [
                max(2, 3 * _steps_per_hour),
                max(3, 12 * _steps_per_hour),
                max(4, 36 * _steps_per_hour),
            ]

            # Rolling lag buffer — starts with the last observed values,
            # grows as we append predictions.
            lag_buffer = list(y[-max(n_lags, steps_per_day * 2 + 1):])

            # Last known covariate values — used as the fallback when a
            # covariate has no role=future entry (lagged-only signals)
            # or its future fetch fails. The full tree feature row
            # prefers future_cov_values[c] below when available.
            last_cov_vals = {
                c: float(combined[c].iloc[-1]) if c in combined.columns else 0.0
                for c in raw_cov_cols
            }

            # Future covariate values for role='future' / role='both'
            # entries — ported from _run_production_inference so the
            # cached forecast cycle stops pinning Solcast-style forecast
            # sensors to their last observed reading for all 48 horizon
            # steps. Without this, training saw a time-varying Solcast
            # column but inference saw one constant, which regresses
            # peak predictions toward the mean. fetch_future knows how
            # to handle each covariate type (Solcast forecast series,
            # tariff schedules, weather-forecast sensors); its result
            # is reindexed/ffill-bfill'd onto the forecast grid.
            future_index = pd.date_range(
                start=last_ts + pd.Timedelta(minutes=exp_cfg.interval_minutes),
                periods=future_periods,
                freq=f'{exp_cfg.interval_minutes}min',
            )
            future_cov_values: Dict = {}
            if exp_cfg.covariates and self.covariate_resolver:
                for cov_cfg in exp_cfg.covariates:
                    cov_name = _cov_column_name(
                        cov_cfg, all_covs=exp_cfg.covariates,
                    )
                    if cov_name in raw_cov_cols and cov_cfg.role in ('future', 'both'):
                        try:
                            cov_dict = {
                                "entity_id": cov_cfg.entity,
                                "name": cov_name,
                                "future_attribute": getattr(cov_cfg, "future_attribute", "forecast"),
                                "future_value_key": getattr(cov_cfg, "future_value_key", None),
                            }
                            future_series = await self.covariate_resolver.fetch_future(
                                cov_dict, future_index,
                            )
                            if future_series is not None and not future_series.empty:
                                if cov_cfg.scale is not None:
                                    future_series = future_series * cov_cfg.scale
                                aligned = future_series.reindex(
                                    future_index
                                ).ffill().bfill()
                                # fetch_future returns an all-NaN series
                                # for entity types it can't resolve
                                # (which is most of them until the
                                # forecast-attribute parser lands).
                                # Accepting NaN here would land the
                                # covariate at 0 after nan_to_num and
                                # starve the tree of its training-time
                                # signal (see v2.27.10 regression).
                                # Only keep the fetched series when it
                                # actually contains real values;
                                # otherwise fall back to last-observed
                                # carry-forward below.
                                if aligned.notna().any():
                                    future_cov_values[cov_name] = aligned
                        except Exception as e:
                            logger.debug(
                                f"  Future fetch failed for {cov_name}: {e}"
                            )
            if future_cov_values:
                logger.info(
                    f"  Fetched future covariate series for "
                    f"{len(future_cov_values)}/"
                    f"{sum(1 for cc in (exp_cfg.covariates or []) if cc.role in ('future', 'both'))} "
                    f"future-role covariates: "
                    f"{list(future_cov_values.keys())}"
                )

            # Pre-compute deterministic solar features for future timestamps.
            # Unlike other covariates, sun elevation and clear-sky irradiance
            # are exactly known for any future time given (lat, lon), so we
            # should NOT carry forward the last value — that would leave the
            # recursive forecast stuck in a single point of the day.
            future_solar = None
            solar_cols = [
                c for c in ("sun_elevation", "clear_sky_ghi")
                if c in raw_cov_cols
            ]
            if solar_cols:
                loc = await self._get_site_location()
                if loc is not None:
                    lat, lon = loc
                    try:
                        from ml_forecast_lab.solar_physics import compute_solar_features
                        future_solar = compute_solar_features(
                            future_index,
                            latitude=lat,
                            longitude=lon,
                            include_elevation="sun_elevation" in solar_cols,
                            include_clear_sky="clear_sky_ghi" in solar_cols,
                        )
                    except Exception as e:
                        logger.debug(f"Future solar pre-compute failed: {e}")

            def _build_feature_row(ts: pd.Timestamp, buf: list, step_idx: int) -> dict:
                """Construct a single feature row matching the training schema."""
                row = {}
                # Temporal
                row['hour_of_day'] = ts.hour
                row['day_of_week'] = ts.dayofweek
                row['is_weekend'] = 1.0 if ts.dayofweek >= 5 else 0.0
                row['month'] = ts.month
                row['day_of_month'] = ts.day
                hr_rad = 2 * np.pi * ts.hour / 24
                dw_rad = 2 * np.pi * ts.dayofweek / 7
                row['hour_sin'] = float(np.sin(hr_rad))
                row['hour_cos'] = float(np.cos(hr_rad))
                row['dow_sin'] = float(np.sin(dw_rad))
                row['dow_cos'] = float(np.cos(dw_rad))
                # Lag features (buf is chronological: oldest..newest,
                # y_lag_i = i-th most recent past value)
                for lag in range(1, n_lags + 1):
                    row[f'y_lag_{lag}'] = float(buf[-lag]) if lag <= len(buf) else 0.0
                # Periodic lags
                for d in [1, 2]:
                    lag_steps = steps_per_day * d
                    row[f'y_lag_{lag_steps}'] = float(buf[-lag_steps]) if lag_steps <= len(buf) else 0.0
                for w in rolling_windows:
                    window = buf[-w:] if len(buf) >= w else buf
                    if window:
                        row[f'y_rolling_mean_{w}'] = float(np.mean(window))
                        row[f'y_rolling_std_{w}'] = float(np.std(window))
                        row[f'y_rolling_max_{w}'] = float(np.max(window))
                    else:
                        row[f'y_rolling_mean_{w}'] = 0.0
                        row[f'y_rolling_std_{w}'] = 0.0
                        row[f'y_rolling_max_{w}'] = 0.0
                # Rate of change
                row['y_diff_1'] = float(buf[-1] - buf[-2]) if len(buf) >= 2 else 0.0
                # Covariates: preference order per value:
                #   1. pvlib-computed future_solar for sun_elevation /
                #      clear_sky_ghi (deterministic, always correct).
                #   2. future_cov_values for user-configured
                #      role=future / role=both entries (Solcast, tariff,
                #      weather forecasts). This matches what the tree
                #      saw at training time, where each row had the
                #      current Solcast reading — not a single carried
                #      value across all horizon steps.
                #   3. last_cov_vals carry-forward fallback for
                #      lagged-only covariates with no future source.
                for c, v in last_cov_vals.items():
                    if (
                        future_solar is not None
                        and c in future_solar.columns
                        and ts in future_solar.index
                    ):
                        row[c] = float(future_solar.loc[ts, c])
                    elif c in future_cov_values:
                        try:
                            row[c] = float(future_cov_values[c].iloc[step_idx])
                        except Exception:
                            row[c] = v
                    else:
                        row[c] = v
                # Interaction features — use row[c] so solar / future
                # covariate interactions reflect the true future value,
                # not the carried one.
                for c in raw_cov_cols:
                    row[f'{c}_x_hour_sin'] = row.get(c, 0.0) * row['hour_sin']
                    row[f'{c}_x_hour_cos'] = row.get(c, 0.0) * row['hour_cos']
                return row

            def _run_recursive_forecast():
                preds = []
                for step in range(future_periods):
                    ts = last_ts + pd.Timedelta(minutes=exp_cfg.interval_minutes * (step + 1))
                    row_dict = _build_feature_row(ts, lag_buffer, step)
                    # Align to training feature order
                    row_vals = [row_dict.get(c, 0.0) for c in feature_cols]
                    X_row = np.array([row_vals], dtype=np.float32)
                    X_row = np.nan_to_num(X_row, nan=0.0)
                    y = model.predict(X_row)
                    val = float(y.ravel()[0] if hasattr(y, 'ravel') else y[0])
                    preds.append(val)
                    # Physics-gated lag buffer: mirror the train-side
                    # gate in build_features so every future feature
                    # vector stays in-distribution. When the current
                    # step is at night (clear_sky_ghi = 0), push 0
                    # into the buffer instead of the raw prediction;
                    # training's y_lag_k at rows following a night
                    # step was 0 by construction, so feeding 0 forward
                    # keeps the recursive model inputs matching what
                    # the tree actually saw at fit time. Leaves preds[]
                    # untouched so the model's own learned response —
                    # on in-distribution inputs — drives what gets
                    # published.
                    ghi_now = None
                    if future_solar is not None and ts in future_solar.index:
                        ghi_now = float(future_solar.loc[ts, 'clear_sky_ghi'])
                    if ghi_now is not None and ghi_now <= 0:
                        lag_buffer.append(0.0)
                    else:
                        lag_buffer.append(val)
                return np.array(preds, dtype=np.float32)

            loop = asyncio.get_running_loop()
            y_pred = await loop.run_in_executor(None, _run_recursive_forecast)

        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        # Snapshot raw model output before any unit conversion so the
        # debug bundle can show what the head actually emitted (in
        # log space when log_transform is on, in physical space when
        # it's off). Cheap copy of ≤96 float32s; only the next branch
        # mutates y_pred in place.
        y_pred_raw_snapshot = (
            np.asarray(y_pred, dtype=np.float32).copy()
            if getattr(exp_cfg, "debug_save_training_dumps", False)
            else None
        )

        # Invert log-transform if active (see _run_production_inference).
        if exp_cfg.log_transform:
            y_pred = np.expm1(y_pred).astype(np.float32)
            y_pred = np.maximum(y_pred, 0.0)

        # Publish-boundary non-negativity clamp (v2.41.0). With 'auto'
        # output_activation now resolving to a linear head, physically
        # non-negative targets can emit small negative dips; clamping
        # here keeps the published sensors physical without putting a
        # saturating activation back into the optimisation path.
        if (
            getattr(exp_cfg, 'target_is_nonnegative', False)
            or getattr(exp_cfg, 'source_is_cumulative', False)
        ):
            y_pred = np.maximum(y_pred, 0.0).astype(np.float32)

        logger.info(
            f"  Forecast {exp_cfg.name}: {len(y_pred)} points, "
            f"range [{y_pred.min():.3f}, {y_pred.max():.3f}]"
        )

        # Build the future timestamp index and publish via shared helper
        ds_future = pd.DatetimeIndex([
            last_ts + pd.Timedelta(minutes=exp_cfg.interval_minutes * (i + 1))
            for i in range(len(y_pred))
        ])
        last_trained = cache.get("trained_at", datetime.now(timezone.utc))
        if not isinstance(last_trained, datetime):
            last_trained = datetime.now(timezone.utc)

        # Debug bundle: pair this forecast with the most recent training
        # dump for the same experiment. Only fires when dump_training
        # left a pending dir behind (i.e. this is the immediate
        # post-retrain forecast). Later 30-min cycles no-op so the
        # bundle stays small.
        if (
            getattr(exp_cfg, "debug_save_training_dumps", False)
            and self._debug_dumper is not None
        ):
            try:
                self._debug_dumper.dump_forecast(
                    exp_name=exp_cfg.name,
                    y_pred_raw=y_pred_raw_snapshot,
                    y_pred_physical=y_pred,
                    ds_future=ds_future,
                    model_version=cache.get("model_version"),
                    log_transform_applied=bool(exp_cfg.log_transform),
                )
            except Exception as e:
                logger.debug(f"debug_dump forecast hook failed: {e}")

        await self._publish_forecast_sensors(
            exp_cfg=exp_cfg,
            y_pred=y_pred,
            ds_future=ds_future,
            model_name=prod_model_name,
            last_trained_iso=last_trained.isoformat(),
        )

    async def _run_tuning(self, experiment_name: str, model_name: str,
                          n_trials: int = 30, strategy: str = "tpe",
                          param_schema: dict = None):
        """
        Run Optuna-based hyperparameter tuning for a single model.

        Uses TPE (Tree-structured Parzen Estimator) or random search with
        2-fold cross-validation for fast evaluation on constrained hardware.
        """
        import optuna
        import time as _time
        from ml_forecast_lab.features import build_features
        from ml_forecast_lab.benchmark.runner import BenchmarkRunner
        from ml_forecast_lab.benchmark.metrics import get_metric_registry
        from ml_forecast_lab.web.app import TuningTrialResult, TuningResult

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Find experiment config
        await self.load_config()
        exp_cfg = None
        for cfg in self.config.experiments:
            if cfg.name == experiment_name:
                exp_cfg = cfg
                break
        if not exp_cfg:
            raise ValueError(f"Experiment {experiment_name} not found")

        if not param_schema:
            param_schema = {}

        logger.info(f"")
        logger.info(f"{'=' * 60}")
        logger.info(f"  TUNING: {exp_cfg.name} / {model_name}")
        logger.info(f"  Strategy: {strategy}, Trials: {n_trials}")
        logger.info(f"  Params: {len(param_schema)} tuneable parameters")
        logger.info(f"{'=' * 60}")

        # Initialise result in AppState
        tuning_state = TuningResult(
            experiment_name=experiment_name,
            model_name=model_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="running",
            search_strategy=strategy,
            n_trials=n_trials,
        )
        if self.web_app:
            self.web_app.state.appstate.tuning_results[experiment_name] = tuning_state

        # Prepare data (same as _run_benchmark)
        df = await self._fetch_and_preprocess(exp_cfg)
        if df is None:
            return

        features_df = build_features(
            df, target_col="y",
            interval_minutes=exp_cfg.interval_minutes,
            country=exp_cfg.country,
        )

        combined = features_df.copy()
        combined["target"] = df["y"]
        for col in [c for c in df.columns if c != "y"]:
            combined[col] = df[col]
        combined = combined.dropna()

        # Feature builder (same as _run_benchmark). Windows scale with
        # interval_minutes so daily seasonality is captured at any rate.
        _steps_per_hour = max(1, 60 // max(exp_cfg.interval_minutes, 1))
        rolling_windows = [
            max(2, 3 * _steps_per_hour),
            max(3, 12 * _steps_per_hour),
            max(4, 36 * _steps_per_hour),
        ]
        steps_per_day = max(1, 1440 // exp_cfg.interval_minutes)

        def feature_builder(df_sub, config, purpose="train"):
            df_out = df_sub.copy()
            target = df_out["target"]
            shifted_target = target.shift(1)
            for window in rolling_windows:
                df_out[f"y_rolling_mean_{window}"] = shifted_target.rolling(window=window).mean()
                df_out[f"y_rolling_std_{window}"] = shifted_target.rolling(window=window).std()
                df_out[f"y_rolling_max_{window}"] = shifted_target.rolling(window=window).max()
            for d in [1, 2]:
                lag_steps = steps_per_day * d
                if lag_steps <= len(target):
                    df_out[f"y_lag_{lag_steps}"] = target.shift(lag_steps)
            df_out["y_diff_1"] = target.shift(1) - target.shift(2)
            cols = [c for c in df_out.columns if c != "target"]
            X = df_out[cols].values.astype(np.float32)
            return np.nan_to_num(X, nan=0.0)

        # Create runner with 1-fold CV for tuning speed. Walk-forward or
        # sliding-window is overkill when we're just looking for the
        # relative ranking between hyperparameter sets — Optuna is robust
        # to noisy objectives, and one well-sized train/test fold is
        # enough to pick winners. (Production retraining after "Apply"
        # uses the full CV schedule again so nothing is lost.)
        exp_cfg_dict = dataclasses.asdict(exp_cfg)
        exp_cfg_dict["cv_folds"] = 1
        metric_registry = get_metric_registry()
        runner = BenchmarkRunner(exp_cfg_dict, feature_builder, metric_registry)
        fold_indices = runner._prepare_train_test_splits(combined)

        # Pre-compute sliding windows once for neural models. The fold
        # split is fixed across all trials (1-fold CV), so creating
        # sliding windows inside run_single_model on every trial wastes
        # both CPU and memory — on an RPi5, the redundant allocations
        # compound across 30 trials and can trigger the OOM killer.
        precomputed_sequences = None
        _is_neural_model = False
        try:
            _probe = self.model_registry.create(model_name)
            _is_neural_model = getattr(_probe, 'is_neural', False)
            del _probe
        except Exception:
            pass

        if _is_neural_model and fold_indices:
            try:
                from ml_forecast_lab.features import create_sliding_windows
                target_col = 'target'
                engineered = {
                    'hour_of_day', 'day_of_week', 'is_weekend', 'month',
                    'day_of_month', 'hour_sin', 'hour_cos', 'dow_sin',
                    'dow_cos', 'is_holiday',
                }
                engineered.update(c for c in combined.columns if c.startswith('y_lag_'))
                _neural_cov_cols = [
                    c for c in combined.columns
                    if c not in engineered and c != target_col
                ]
                future_periods_val = getattr(exp_cfg, 'future_periods', 48)
                _horizon_steps = list(range(1, future_periods_val + 1))

                precomputed_sequences = {}
                for fi, (train_idx, _test_idx) in enumerate(fold_indices):
                    df_train_raw = combined.iloc[train_idx]
                    _ws = min(48, len(df_train_raw) // 3)
                    if _ws >= 12:
                        _sX, _sY, _cnames = create_sliding_windows(
                            df_train_raw, target_col, window_size=_ws,
                            covariate_cols=_neural_cov_cols if _neural_cov_cols else None,
                            add_temporal=True, horizon_steps=_horizon_steps,
                        )
                        precomputed_sequences[fi] = {
                            'seq_X': _sX, 'seq_y': _sY,
                            'channel_names': _cnames,
                            'window_size': _ws,
                            'neural_cov_cols': _neural_cov_cols,
                            'horizon_steps': _horizon_steps,
                        }
                        logger.info(
                            f"  Pre-computed sliding windows: "
                            f"{_sX.shape[0]} samples × {_sX.shape[1]} steps "
                            f"× {_sX.shape[2]} channels"
                        )
            except Exception as e:
                logger.debug(f"  Sliding window pre-computation failed: {e}")
                precomputed_sequences = None

        # Parameters that benefit from log-scale search
        LOG_PARAMS = {"learning_rate", "reg_alpha", "reg_lambda"}

        # Tuning-time overrides for neural backends. The schema no longer
        # exposes `epochs` / `patience` so Optuna cannot burn trials
        # tuning the training budget itself — we set both here to a
        # small-but-sufficient value that trains each trial fast while
        # still letting early stopping converge on the winning config.
        # Production retraining (after "Apply") uses the model's own
        # defaults so the final deployed model gets the full epoch budget.
        TUNING_NEURAL_EPOCHS = 30
        TUNING_NEURAL_PATIENCE = 6
        TUNING_NEURAL_BATCH_SIZE = 16  # Halved from default 64 to reduce peak memory

        def _apply_tuning_overrides(m):
            """Cap neural training budget and shrink batch size during tuning."""
            try:
                if getattr(m, 'is_neural', False):
                    m.set_params(
                        epochs=TUNING_NEURAL_EPOCHS,
                        patience=TUNING_NEURAL_PATIENCE,
                        batch_size=TUNING_NEURAL_BATCH_SIZE,
                    )
            except Exception:
                pass  # Backends without epochs/patience params just skip
            # Tuning inherits the experiment's output_activation — the target's
            # physical range doesn't change with hyperparameter search, so the
            # activation must stay consistent to get comparable metrics.
            _apply_output_activation(m, exp_cfg)
            return m

        # Memory monitoring — abort tuning before OOM SIGKILL.
        # The addon runs inside a Docker container with cgroup memory
        # limits.  /proc/meminfo shows the HOST's free RAM, which is
        # misleading — the OOM killer enforces the cgroup ceiling.
        def _get_rss_mb():
            """Return current process RSS in MB via /proc/self/status."""
            try:
                with open('/proc/self/status', 'r') as f:
                    for line in f:
                        if line.startswith('VmRSS:'):
                            return int(line.split()[1]) / 1024  # KB → MB
            except Exception:
                pass
            try:
                import resource, platform
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                return rss / 1024 if platform.system() == 'Linux' else rss / (1024 * 1024)
            except Exception:
                return 0

        def _get_cgroup_memory():
            """Return (usage_mb, limit_mb) from cgroup v2 or v1."""
            usage, limit = None, None
            # cgroup v2
            try:
                with open('/sys/fs/cgroup/memory.current', 'r') as f:
                    usage = int(f.read().strip()) / (1024 * 1024)
                with open('/sys/fs/cgroup/memory.max', 'r') as f:
                    val = f.read().strip()
                    limit = float('inf') if val == 'max' else int(val) / (1024 * 1024)
                return usage, limit
            except Exception:
                pass
            # cgroup v1
            try:
                with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
                    usage = int(f.read().strip()) / (1024 * 1024)
                with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                    val = int(f.read().strip())
                    # Kernel uses a huge value for "no limit"
                    limit = float('inf') if val > 2**60 else val / (1024 * 1024)
                return usage, limit
            except Exception:
                pass
            return None, None

        def _get_available_mb():
            """Return available memory in MB (cgroup-aware)."""
            usage, limit = _get_cgroup_memory()
            if usage is not None and limit is not None and limit != float('inf'):
                return limit - usage
            # Fallback: host-level /proc/meminfo
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemAvailable:'):
                            return int(line.split()[1]) / 1024  # KB → MB
            except Exception:
                pass
            return float('inf')  # Can't read → don't gate on it

        # Log detailed memory picture
        cg_usage, cg_limit = _get_cgroup_memory()
        rss = _get_rss_mb()
        logger.info(
            f"  Tuning budget: {n_trials} trials × 1 CV fold × "
            f"max {TUNING_NEURAL_EPOCHS} epochs (neural) / early-stopping (trees)"
        )
        if cg_limit is not None and cg_limit != float('inf'):
            logger.info(
                f"  Container memory: {cg_usage:.0f} / {cg_limit:.0f} MB "
                f"({cg_limit - cg_usage:.0f} MB free) | Process RSS: {rss:.0f} MB"
            )
        else:
            logger.info(
                f"  Memory: {_get_available_mb():.0f} MB available (host) | "
                f"Process RSS: {rss:.0f} MB | No cgroup limit detected"
            )

        # --- Composite objective baseline ---
        # Run one CV evaluation with the model's DEFAULT parameters first.
        # The (mae, rmse, mase) of that run becomes the anchor used to
        # normalise every subsequent trial's metrics. This lets Optuna
        # optimise on a single composite scalar that weights all three
        # metrics equally and is interpretable: composite = 1.0 means the
        # trial matches default performance, < 1.0 means it improves.
        baseline_params = {
            pname: spec.get("default")
            for pname, spec in param_schema.items()
            if spec.get("default") is not None
        }
        baseline_model = None
        try:
            baseline_model = self.model_registry.create(model_name, **baseline_params)
            _apply_tuning_overrides(baseline_model)
            _apply_experiment_neural_params(
                baseline_model, exp_cfg, overrides=baseline_params
            )
            baseline_result = runner.run_single_model(
                combined, baseline_model, fold_indices,
                precomputed_sequences=precomputed_sequences,
            )
            baseline_mae = max(baseline_result.metrics.get("mae", 1.0), 1e-6)
            baseline_rmse = max(baseline_result.metrics.get("rmse", 1.0), 1e-6)
            baseline_mase = max(baseline_result.metrics.get("mase", 1.0), 1e-6)
            logger.info(
                f"  Baseline (default params): "
                f"MAE={baseline_mae:.4f}, RMSE={baseline_rmse:.4f}, MASE={baseline_mase:.3f}"
            )
        except Exception as e:
            logger.warning(
                f"  Baseline trial failed ({e}); falling back to first-trial anchors"
            )
            baseline_mae = baseline_rmse = baseline_mase = None
        finally:
            import gc as _gc
            del baseline_model
            _gc.collect(0); _gc.collect(1); _gc.collect(2)
            logger.info(
                f"  Memory after baseline: {_get_available_mb():.0f} MB free "
                f"| RSS: {_get_rss_mb():.0f} MB"
            )

        # Mutable anchors so the first valid trial can become the baseline if
        # the explicit baseline trial above failed.
        anchor = {"mae": baseline_mae, "rmse": baseline_rmse, "mase": baseline_mase}

        def _composite_score(mae, rmse, mase):
            """Average of (metric / baseline) across MAE, RMSE, MASE.

            Lower is better; 1.0 = matches baseline; 0.5 = half the average error.
            Returns +inf for failed trials.
            """
            if not all(np.isfinite([mae, rmse, mase])):
                return float("inf")
            return float(np.mean([
                mae / anchor["mae"],
                rmse / anchor["rmse"],
                mase / anchor["mase"],
            ]))

        # Memory threshold: abort tuning if available RAM drops below this.
        # Leaves headroom for the OS, HA core, and other addon processes.
        MEM_FLOOR_MB = 256

        def objective(trial):
            import gc as _gc
            t_start = _time.time()

            # --- Memory pressure check BEFORE starting a new trial ---
            avail = _get_available_mb()
            if avail < MEM_FLOOR_MB:
                logger.warning(
                    f"  Trial {trial.number}: only {avail:.0f} MB free "
                    f"(floor={MEM_FLOOR_MB} MB) — aborting tuning early"
                )
                trial.study.stop()
                return float("inf")

            params = {}
            for pname, spec in param_schema.items():
                if spec.get("tunable", True) is False:
                    continue
                ptype = spec.get("type", "float")
                if ptype == "int":
                    params[pname] = trial.suggest_int(pname, spec["min"], spec["max"])
                elif ptype == "float":
                    log = pname in LOG_PARAMS and spec.get("min", 0) > 0
                    params[pname] = trial.suggest_float(
                        pname, spec["min"], spec["max"], log=log
                    )
                elif ptype == "select":
                    params[pname] = trial.suggest_categorical(pname, spec["options"])
                elif ptype == "bool":
                    params[pname] = trial.suggest_categorical(pname, [True, False])

            # Emit a trial-start log line so the user can see an
            # in-flight trial rather than waiting in silence for the
            # completion line. The previous-trial composite is already
            # logged on completion, so between that and this line the
            # transition from one trial to the next is always visible.
            logger.info(
                f"  [{tuning_state.completed_trials}/{n_trials}] "
                f"Trial {trial.number} starting: params={params}"
            )

            model = None
            result = None
            try:
                model = self.model_registry.create(model_name, **params)
                _apply_tuning_overrides(model)
                _apply_experiment_neural_params(model, exp_cfg, overrides=params)
                result = runner.run_single_model(
                    combined, model, fold_indices,
                    precomputed_sequences=precomputed_sequences,
                )
                mae = result.metrics.get("mae", float("inf"))
                rmse = result.metrics.get("rmse", float("inf"))
                mase = result.metrics.get("mase", float("inf"))
                # Tuning objective tracks the user's selected production
                # metric. Falls back to MAE if production_metric is missing
                # from the result (e.g. seasonal_mase before the runner
                # registered it).
                primary = result.metrics.get(
                    exp_cfg.production_metric,
                    result.metrics.get("mae", float("inf")),
                )
                status = "completed"
            except Exception as e:
                logger.warning(f"  Trial {trial.number} failed: {e}", exc_info=True)
                mae = float("inf")
                rmse = float("inf")
                mase = float("inf")
                primary = float("inf")
                status = "failed"
            finally:
                del model
                del result
                _gc.collect(0); _gc.collect(1); _gc.collect(2)

            # Lazily seed the anchor if the explicit baseline run failed
            if anchor["mae"] is None and status == "completed":
                anchor["mae"] = max(mae, 1e-6)
                anchor["rmse"] = max(rmse, 1e-6)
                anchor["mase"] = max(mase, 1e-6)
                logger.info(
                    f"  Anchoring composite to trial {trial.number}: "
                    f"MAE={mae:.4f}, RMSE={rmse:.4f}, MASE={mase:.3f}"
                )

            # Optuna minimises the user's chosen production_metric directly so
            # tuning, model selection, and the leaderboard agree on what
            # "better" means. The composite is still computed for the trial
            # log but no longer drives the search direction.
            composite = primary if np.isfinite(primary) else float("inf")

            duration = _time.time() - t_start

            trial_result = TuningTrialResult(
                trial_id=trial.number, params=params,
                mae=round(mae, 6), rmse=round(rmse, 6), mase=round(mase, 4),
                duration_seconds=round(duration, 1), status=status,
            )
            tuning_state.trials.append(trial_result)
            tuning_state.completed_trials = len(tuning_state.trials)

            # Update best so far based on composite score
            if composite < (tuning_state.best_score or float("inf")):
                tuning_state.best_score = round(composite, 6)
                tuning_state.best_params = params
                tuning_state.best_trial_id = trial.number

            _avail_after = _get_available_mb()
            _rss_after = _get_rss_mb()
            logger.info(
                f"  [{tuning_state.completed_trials}/{n_trials}] "
                f"Trial {trial.number}: composite={composite:.4f} "
                f"(MAE={mae:.4f}, RMSE={rmse:.4f}, MASE={mase:.3f}) "
                f"{'best ' if trial.number == tuning_state.best_trial_id else ''}"
                f"{duration:.1f}s | avail={_avail_after:.0f} MB, RSS={_rss_after:.0f} MB"
            )

            return composite

        # Create and run Optuna study.
        # Cap total wall-clock time to 30 minutes for every backend.
        # Optuna stops cleanly after the current trial finishes, so the
        # best-so-far result is always available. Applies to trees as
        # well: CatBoost with a small learning rate + large n_estimators
        # can sit for many minutes inside a single trial before early
        # stopping fires, which is indistinguishable from a stall.
        if strategy == "tpe":
            sampler = optuna.samplers.TPESampler(seed=42)
        else:
            sampler = optuna.samplers.RandomSampler(seed=42)

        study = optuna.create_study(direction="minimize", sampler=sampler)
        study_timeout = 30 * 60

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: study.optimize(
                objective, n_trials=n_trials, timeout=study_timeout,
            ),
        )

        # Update completed count in case the timeout or memory pressure stopped early
        actual_trials = len(tuning_state.trials)
        if actual_trials < n_trials:
            reason = "memory pressure" if _get_available_mb() < MEM_FLOOR_MB * 2 else f"timeout={study_timeout}s"
            logger.info(
                f"  Tuning stopped after {actual_trials}/{n_trials} trials "
                f"({reason})"
            )
            tuning_state.completed_trials = actual_trials
            tuning_state.n_trials = actual_trials  # Update so UI shows correct count

        # Finalise: select best trial by composite ranking across MAE, RMSE, MASE
        # (Optuna minimised MAE to guide the search, but the winner is picked
        # by average rank across all three metrics for robustness.)
        valid_trials = [t for t in tuning_state.trials if t.status == "completed" and t.mae < 1e6]
        if valid_trials:
            for metric_key in ("mae", "rmse", "mase"):
                sorted_by = sorted(valid_trials, key=lambda t: getattr(t, metric_key))
                for rank, t in enumerate(sorted_by):
                    if not hasattr(t, '_ranks'):
                        t._ranks = []
                    t._ranks.append(rank + 1)

            best_trial = min(valid_trials, key=lambda t: np.mean(t._ranks))
            tuning_state.best_trial_id = best_trial.trial_id
            tuning_state.best_params = best_trial.params
            tuning_state.best_score = best_trial.mae

            # Clean up temp attribute
            for t in valid_trials:
                if hasattr(t, '_ranks'):
                    del t._ranks

            logger.info(f"")
            logger.info(f"  Tuning Complete — Best trial #{best_trial.trial_id} "
                        f"(MAE={best_trial.mae:.4f}, RMSE={best_trial.rmse:.4f}, MASE={best_trial.mase:.3f})")
            logger.info(f"  Best params: {best_trial.params}")
        else:
            logger.warning(f"  Tuning Complete — no valid trials")

        # Run holdout comparison: default params vs tuned params
        try:
            logger.info(f"  Running holdout comparison...")
            feature_cols = [c for c in combined.columns if c != "target"]
            split_80 = int(len(combined) * 0.8)
            X_all = combined[feature_cols].values.astype(np.float32)
            X_all = np.nan_to_num(X_all, nan=0.0)
            y_all = combined["target"].values.astype(np.float32)
            X_tr_h, X_te_h = X_all[:split_80], X_all[split_80:]
            y_tr_h, y_te_h = y_all[:split_80], y_all[split_80:]
            holdout_ts = [t.isoformat() for t in combined.index[split_80:]]

            # For neural models, build proper sliding-window inputs.
            # The flat-feature path (m.fit(X, y) + m.predict(X)) produces
            # garbage for CNNs/LSTMs because _reshape_to_sequences() can't
            # reconstruct the temporal structure the model needs.
            _ho_seq_train = None
            _ho_seq_test = None
            _ho_window_size = None
            if _is_neural_model:
                try:
                    from ml_forecast_lab.features import create_sliding_windows
                    pc_fold = (precomputed_sequences or {}).get(0, {})
                    _ho_window_size = pc_fold.get('window_size', min(48, split_80 // 3))
                    _ho_cov_cols = pc_fold.get('neural_cov_cols', [])
                    _ho_horizon_steps = pc_fold.get('horizon_steps', list(range(1, int(getattr(exp_cfg, 'future_periods', 48)) + 1)))
                    if _ho_window_size >= 12:
                        df_train_ho = combined.iloc[:split_80]
                        sX, sY, ch = create_sliding_windows(
                            df_train_ho, 'target', window_size=_ho_window_size,
                            covariate_cols=_ho_cov_cols if _ho_cov_cols else None,
                            add_temporal=True, horizon_steps=_ho_horizon_steps,
                        )
                        _ho_seq_train = {'seq_X': sX, 'seq_y': sY, 'channel_names': ch}
                        # Test windows: bridge across train/test boundary for context
                        n_bridge = min(_ho_window_size, split_80)
                        df_test_ho = combined.iloc[split_80 - n_bridge:]
                        tX, _, _ = create_sliding_windows(
                            df_test_ho, 'target', window_size=_ho_window_size,
                            covariate_cols=_ho_cov_cols if _ho_cov_cols else None,
                            add_temporal=True, horizon_steps=[1],
                        )
                        _ho_seq_test = tX
                        logger.info(
                            f"  Holdout sliding windows: train={sX.shape}, test={tX.shape}"
                        )
                except Exception as e:
                    logger.debug(f"  Holdout sliding window build failed: {e}")
                    _ho_seq_train = None

            def _run_holdout(params):
                import gc as _gc
                m = self.model_registry.create(model_name, **params)
                _apply_output_activation(m, exp_cfg)
                _apply_experiment_neural_params(m, exp_cfg, overrides=params)
                if _ho_seq_train is not None:
                    # Neural path: proper sliding windows
                    m.fit(
                        X_tr_h[-len(_ho_seq_train['seq_y']):],
                        _ho_seq_train['seq_y'],
                        sequence_data=_ho_seq_train['seq_X'],
                        channel_names=_ho_seq_train['channel_names'],
                    )
                    p = m.predict_sequence(_ho_seq_test)
                    if p.ndim == 2:
                        p = p[:, 0]  # h=1 predictions
                    # Align to test actuals
                    n_test = len(y_te_h)
                    p = p[-n_test:] if len(p) >= n_test else np.concatenate([np.full(n_test - len(p), np.nan), p])
                else:
                    # Tree path: flat features
                    m.fit(X_tr_h, y_tr_h)
                    p = m.predict(X_te_h)
                    p = p.ravel() if p.ndim > 1 else p
                del m
                _gc.collect()
                return p.tolist()

            # Default params holdout
            default_overrides = dict(self.config.model_overrides.get(model_name, {}))
            preds_default = await loop.run_in_executor(None, lambda: _run_holdout(default_overrides))
            default_mae = float(np.nanmean(np.abs(y_te_h - np.array(preds_default))))

            # Tuned params holdout
            best_params = tuning_state.best_params or dict(study.best_params)
            preds_tuned = await loop.run_in_executor(None, lambda: _run_holdout(best_params))
            tuned_mae = float(np.nanmean(np.abs(y_te_h - np.array(preds_tuned))))

            tuning_state.holdout_timestamps = holdout_ts
            tuning_state.holdout_actuals = [float(v) for v in y_te_h]
            tuning_state.holdout_default = [float(v) for v in preds_default]
            tuning_state.holdout_tuned = [float(v) for v in preds_tuned]
            tuning_state.default_mae = round(default_mae, 6)
            tuning_state.tuned_mae = round(tuned_mae, 6)
            tuning_state.best_score = round(tuned_mae, 6)

            logger.info(f"  Holdout: default MAE={default_mae:.4f}, tuned MAE={tuned_mae:.4f}")
        except Exception as e:
            logger.warning(f"  Holdout comparison failed: {e}", exc_info=True)

        tuning_state.status = "completed"
        logger.info(f"{'=' * 60}")

    async def _run_covariate_analysis(self, exp_cfg, selected_model: str = "all"):
        """
        Run deep covariate analysis: selected model(s) × all covariate combinations.

        Tests each model with:
        1. All covariates (baseline)
        2. No covariates (control)
        3. Each covariate dropped one at a time

        Generates recommendations based on MAE impact.

        Parameters
        ----------
        selected_model : str
            Model name to analyse, or 'all' for all enabled models.
        """
        from ml_forecast_lab.features import build_features
        from ml_forecast_lab.benchmark.metrics import get_metric_registry
        from ml_forecast_lab.web.app import (
            CovariateAnalysisResult,
            CovariateAnalysisCellResult,
        )

        # Determine which models to run
        if selected_model == "all":
            models_to_run = exp_cfg.models_enabled
        else:
            models_to_run = [selected_model] if selected_model in exp_cfg.models_enabled else exp_cfg.models_enabled

        logger.info("")
        logger.info(f"{'=' * 60}")
        logger.info(f"  DEEP ANALYSIS: {exp_cfg.name}")
        logger.info(f"  Models: {', '.join(models_to_run)}")
        logger.info(f"  Covariates: {len(exp_cfg.covariates)}")
        logger.info(f"{'=' * 60}")

        # Update status
        if self.web_app:
            # Build labels from the SAME canonical column names that the
            # configs loop below (line ~6530) uses for the results dict
            # keys — otherwise the Jinja `results.get(label, {})` lookup
            # misses for any same-entity-multi-config covariate
            # (template renders duplicate blank rows).
            cov_names = [
                _cov_column_name(c, all_covs=exp_cfg.covariates)
                for c in exp_cfg.covariates
            ]
            covariate_labels = ["All covariates", "No covariates"] + [
                f"Without {name}" for name in cov_names
            ]
            total_runs = len(models_to_run) * len(covariate_labels)

            self.web_app.state.appstate.covariate_analysis_results[exp_cfg.name] = CovariateAnalysisResult(
                experiment_name=exp_cfg.name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                status="running",
                baseline_label="All covariates",
                covariate_labels=covariate_labels,
                model_names=models_to_run,
                results={},
                recommendations=[],
                total_runs=total_runs,
                completed_runs=0,
            )

        # Fetch and preprocess with all covariates
        df_full = await self._fetch_and_preprocess(exp_cfg)
        if df_full is None:
            return
        covariate_cols = [c for c in df_full.columns if c != "y"]

        # Build base features
        features_base = build_features(
            df_full, target_col="y",
            interval_minutes=exp_cfg.interval_minutes,
            country=exp_cfg.country,
        )

        # Define covariate configurations to test
        configs = []

        # 1. All covariates
        configs.append(("All covariates", covariate_cols[:]))

        # 2. No covariates
        configs.append(("No covariates", []))

        # 3. Drop one at a time
        for cov_col in covariate_cols:
            remaining = [c for c in covariate_cols if c != cov_col]
            configs.append((f"Without {cov_col}", remaining))

        results = {}
        completed = 0

        # Engineered feature columns to exclude when picking covariates for
        # the neural sequence builder (these go in as separate channels via
        # add_temporal=True or as derived features inside the network).
        _engineered = {
            'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_holiday',
        }
        future_periods = int(getattr(exp_cfg, 'future_periods', 48))
        dense_horizons = list(range(1, future_periods + 1))

        for config_label, cov_cols_to_use in configs:
            results[config_label] = {}

            for model_name in models_to_run:
                try:
                    # Build combined feature matrix
                    combined = features_base.copy()
                    combined["target"] = df_full["y"]

                    # Add only the specified covariates
                    for col in cov_cols_to_use:
                        combined[col] = df_full[col]

                    combined = combined.dropna()

                    if len(combined) < 100:
                        # Not enough data for a meaningful split
                        results[config_label][model_name] = CovariateAnalysisCellResult(
                            mae=float('nan'), rmse=float('nan'), mase=float('nan'),
                        )
                        completed += 1
                        continue

                    feature_cols = [c for c in combined.columns if c != "target"]

                    # Split point used by both branches (80/20)
                    split = int(len(combined) * 0.8)

                    # Train and evaluate
                    model = self.model_registry.create(model_name)
                    overrides = self.config.model_overrides.get(model_name, {})
                    if overrides:
                        model.set_params(**overrides)
                    if 'output_activation' not in overrides:
                        _apply_output_activation(model, exp_cfg)
                    # Honour Settings-level neural params (loss_fn,
                    # optimiser) so covariate analysis
                    # minimises the SAME objective as the main benchmark.
                    _apply_experiment_neural_params(model, exp_cfg, overrides=overrides)

                    def _train_and_eval():
                        if model.is_neural:
                            # Use the SAME sliding-window + dense-horizon
                            # pipeline as the CV runner, holdout chart, and
                            # production training. Without this, neural
                            # models in covariate analysis are crippled (flat
                            # features only, no residual prediction) and
                            # the covariate comparison becomes meaningless.
                            from ml_forecast_lab.features import create_sliding_windows

                            target_col = 'target'
                            engineered = set(_engineered)
                            engineered.update(
                                c for c in combined.columns if c.startswith('y_lag_')
                            )
                            seq_cov_cols = [
                                c for c in combined.columns
                                if c not in engineered and c != target_col
                            ]

                            df_train_part = combined.iloc[:split]
                            df_test_part = combined.iloc[split:]

                            window_size = min(48, len(df_train_part) // 3)
                            if window_size < 12:
                                return float('nan'), float('nan'), float('nan')

                            # Train: dense horizons so the model learns to
                            # predict h=1..future_periods (matches production)
                            seq_X_tr, seq_y_tr, channel_names = create_sliding_windows(
                                df_train_part, target_col,
                                window_size=window_size,
                                covariate_cols=seq_cov_cols if seq_cov_cols else None,
                                add_temporal=True,
                                horizon_steps=dense_horizons,
                            )
                            if len(seq_y_tr) == 0:
                                return float('nan'), float('nan'), float('nan')

                            # Flat X is required by fit() signature but is
                            # ignored when sequence_data is provided.
                            X_tr_flat = df_train_part[feature_cols].values.astype(np.float32)
                            X_tr_flat = np.nan_to_num(X_tr_flat, nan=0.0)
                            X_tr_flat = X_tr_flat[-len(seq_y_tr):]

                            model.fit(
                                X_tr_flat, seq_y_tr,
                                feature_names=feature_cols,
                                sequence_data=seq_X_tr,
                                channel_names=channel_names,
                            )

                            # Test: bridge train tail + test, use h=1 for
                            # full coverage with one window per test row
                            bridge = pd.concat([
                                df_train_part.iloc[-window_size:],
                                df_test_part,
                            ])
                            seq_X_te, _, _ = create_sliding_windows(
                                bridge, target_col,
                                window_size=window_size,
                                covariate_cols=seq_cov_cols if seq_cov_cols else None,
                                add_temporal=True,
                                horizon_steps=[1],
                            )
                            y_pred_full = model.predict_sequence(seq_X_te)
                            if y_pred_full.ndim == 2:
                                y_pred_flat = y_pred_full[:, 0].astype(np.float32)
                            else:
                                y_pred_flat = y_pred_full.astype(np.float32)

                            y_te = df_test_part[target_col].values.astype(np.float32)
                            y_tr_for_naive = df_train_part[target_col].values.astype(np.float32)
                        else:
                            # Tree models keep the existing flat-features path
                            X = combined[feature_cols].values.astype(np.float32)
                            X = np.nan_to_num(X, nan=0.0)
                            y_all = combined["target"].values.astype(np.float32)
                            X_tr, X_te = X[:split], X[split:]
                            y_tr_for_naive, y_te = y_all[:split], y_all[split:]

                            model.fit(X_tr, y_tr_for_naive)
                            y_pred = model.predict(X_te)
                            y_pred_flat = y_pred.ravel() if y_pred.ndim > 1 else y_pred

                        # Defensive length alignment (e.g. neural bridge can
                        # leave a row off the end if max_horizon clips it)
                        if len(y_pred_flat) != len(y_te):
                            n = min(len(y_pred_flat), len(y_te))
                            y_pred_flat = y_pred_flat[-n:]
                            y_te = y_te[-n:]

                        if len(y_te) == 0:
                            return float('nan'), float('nan'), float('nan')

                        mae_val = float(np.mean(np.abs(y_te - y_pred_flat)))
                        rmse_val = float(np.sqrt(np.mean((y_te - y_pred_flat) ** 2)))
                        # MASE: scale by naive 1-step forecast error from train
                        naive_err = float(np.mean(np.abs(np.diff(y_tr_for_naive))))
                        mase_val = (
                            float(np.mean(np.abs(y_te - y_pred_flat)) / naive_err)
                            if naive_err > 0 else float('nan')
                        )
                        return mae_val, rmse_val, mase_val

                    mae_val, rmse_val, mase_val = await asyncio.get_running_loop().run_in_executor(
                        None, _train_and_eval
                    )

                    results[config_label][model_name] = CovariateAnalysisCellResult(
                        mae=round(mae_val, 4),
                        rmse=round(rmse_val, 4),
                        mase=round(mase_val, 3) if np.isfinite(mase_val) else float('nan'),
                    )

                    completed += 1
                    logger.info(
                        f"  [{completed}/{len(configs) * len(exp_cfg.models_enabled)}] "
                        f"{config_label} × {model_name}: MAE={mae_val:.4f}"
                    )

                    # Update progress
                    if self.web_app:
                        da = self.web_app.state.appstate.covariate_analysis_results[exp_cfg.name]
                        da.completed_runs = completed
                        da.results = results

                except Exception as e:
                    logger.warning(f"  Covariate analysis failed for {config_label} × {model_name}: {e}")
                    results[config_label][model_name] = CovariateAnalysisCellResult(mae=np.nan, rmse=np.nan, mase=float('nan'))
                    completed += 1

        # Compute % change vs baseline for all three metrics
        _nan_cell = CovariateAnalysisCellResult(mae=np.nan, rmse=np.nan, mase=float('nan'))
        baseline = results.get("All covariates", {})
        for config_label in results:
            for model_name in results[config_label]:
                cell = results[config_label][model_name]
                base = baseline.get(model_name, _nan_cell)
                if base.mae > 0 and np.isfinite(cell.mae) and np.isfinite(base.mae):
                    cell.change_pct = round((cell.mae - base.mae) / base.mae * 100, 1)
                if base.rmse > 0 and np.isfinite(cell.rmse) and np.isfinite(base.rmse):
                    cell.rmse_change_pct = round((cell.rmse - base.rmse) / base.rmse * 100, 1)
                if base.mase > 0 and np.isfinite(cell.mase) and np.isfinite(base.mase):
                    cell.mase_change_pct = round((cell.mase - base.mase) / base.mase * 100, 1)

        # Generate recommendations. All percentages use baseline (All
        # covariates) as denominator, matching the change_pct shown in
        # the table. Two passes:
        #   1. Per-model overall — does the model want covariates at all?
        #   2. Per-covariate consensus — across models that DO want
        #      covariates, is there agreement on a specific one to keep
        #      or drop?
        # The `variant` field is a semantic name (good / warning / bad /
        # info); the colour mapping lives in the template so the palette
        # can evolve without touching the backend (LOW-5).
        recommendations = []
        no_cov = results.get("No covariates", {})

        # Pass 1: per-model overall covariate value.
        # Track which models would do better with no covariates so we
        # can suppress contradictory per-covariate "Keep X" recs for
        # those models in pass 2 (HIGH-5: avoid contradictory recs).
        models_better_without_covs: set = set()
        for model_name in models_to_run:
            base_mae = baseline.get(model_name, _nan_cell).mae
            no_cov_mae = no_cov.get(model_name, _nan_cell).mae
            if not np.isnan(base_mae) and not np.isnan(no_cov_mae) and base_mae > 0:
                # Positive = removing covariates increases error (covariates help)
                # Negative = removing covariates decreases error (covariates hurt)
                change_pct = (no_cov_mae - base_mae) / base_mae * 100
                if change_pct > 5:
                    recommendations.append({
                        "icon": "✓",
                        "text": (
                            f"Covariates help {model_name} — without them, typical "
                            f"forecast error is {change_pct:.1f}% larger."
                        ),
                        "variant": "good",
                    })
                elif change_pct < -3:
                    models_better_without_covs.add(model_name)
                    recommendations.append({
                        "icon": "⚠",
                        "text": (
                            f"{model_name} forecasts better without covariates — "
                            f"removing them all reduces typical error by {abs(change_pct):.1f}%."
                        ),
                        "variant": "warning",
                    })

        # Pass 2: per-covariate cross-model consensus.
        # Exclude models from pass 1 that already prefer no covariates,
        # so the "Keep X" voice doesn't contradict their own overall
        # recommendation. If excluding leaves us with <2 models, skip
        # the rec rather than emit a single-model claim.
        for cov_col in covariate_cols:
            label = f"Without {cov_col}"
            dropped = results.get(label, {})

            impacts = {}
            for m in models_to_run:
                if m in models_better_without_covs:
                    continue
                d_mae = dropped.get(m, _nan_cell).mae
                b_mae = baseline.get(m, _nan_cell).mae
                if np.isfinite(d_mae) and np.isfinite(b_mae) and b_mae > 0:
                    impacts[m] = (d_mae - b_mae) / b_mae * 100

            if len(impacts) < 2:
                continue

            avg_impact = np.mean(list(impacts.values()))
            # MED-7: symmetric ±2% threshold matching the row-level
            # Remove button gate in the template.
            n_helps = sum(1 for v in impacts.values() if v > 2)
            n_hurts = sum(1 for v in impacts.values() if v < -2)
            n_models = len(impacts)
            suppressed = len(models_better_without_covs)
            qual = (
                f" (excluding {suppressed} model(s) that prefer no covariates)"
                if suppressed else ""
            )

            if n_helps == n_models and avg_impact > 3:
                recommendations.append({
                    "icon": "✓",
                    "text": (
                        f"Keep {cov_col} — dropping it grows typical error by "
                        f"{avg_impact:.1f}% across all {n_models} tested models{qual}."
                    ),
                    "variant": "good",
                })
            elif n_hurts == n_models and avg_impact < -2:
                recommendations.append({
                    "icon": "✗",
                    "text": (
                        f"Consider removing {cov_col} — dropping it reduces typical "
                        f"error by {abs(avg_impact):.1f}% across all {n_models} "
                        f"tested models{qual}."
                    ),
                    "variant": "bad",
                })
            elif n_helps > n_models / 2 and avg_impact > 3:
                recommendations.append({
                    "icon": "✓",
                    "text": (
                        f"Keep {cov_col} — {n_helps}/{n_models} models forecast "
                        f"worse without it (+{avg_impact:.1f}% typical error){qual}."
                    ),
                    "variant": "good",
                })
            elif n_hurts > n_models / 2 and avg_impact < -2:
                recommendations.append({
                    "icon": "✗",
                    "text": (
                        f"Consider removing {cov_col} — {n_hurts}/{n_models} models "
                        f"forecast better without it ({avg_impact:.1f}% typical error){qual}."
                    ),
                    "variant": "bad",
                })

        # MED-5: never leave the panel silent on a completed run. If no
        # rule fired, tell the user the analysis ran but found no strong
        # signal — that's a useful conclusion in its own right and
        # reassures them the run wasn't a failure.
        if not recommendations and results:
            recommendations.append({
                "icon": "ℹ",
                "text": (
                    "No covariate had a strong effect (>3% change in typical error) "
                    "on any tested model. Your current configuration looks well-balanced."
                ),
                "variant": "info",
            })

        # Single-fold caveat. Always emitted alongside any populated
        # rec block so the user reads the rankings with appropriate
        # confidence (HIGH-5).
        if recommendations:
            recommendations.append({
                "icon": "ⓘ",
                "text": (
                    "Rankings are based on one 80/20 split — rerun the analysis "
                    "to confirm borderline results before applying."
                ),
                "variant": "info",
            })

        # Store final results
        if self.web_app:
            da = self.web_app.state.appstate.covariate_analysis_results[exp_cfg.name]
            da.status = "completed"
            da.results = results
            da.recommendations = recommendations
            da.completed_runs = completed

        logger.info("")
        logger.info(f"  Covariate Analysis Complete")
        logger.info(f"  {'─' * 50}")
        for rec in recommendations:
            logger.info(f"  {rec['icon']} {rec['text']}")
        logger.info(f"  {'─' * 50}")
        logger.info(f"{'=' * 60}")

    async def publish_heartbeat(self):
        """
        Publish heartbeat sensor to Home Assistant.

        Creates/updates a sensor.mlfl_last_run entity with the current timestamp.
        """
        try:
            if not self.ha_interface:
                return

            timestamp = datetime.now(timezone.utc).isoformat()
            logger.debug(f"Publishing heartbeat: {timestamp}")

            await self.ha_interface.set_state(
                "sensor.mlfl_last_run",
                timestamp,
                attributes={
                    "friendly_name": "ML Forecast Lab Last Run",
                    "icon": "mdi:clock-check",
                    "experiments": len(self.config.experiments) if self.config else 0,
                },
            )

        except Exception as e:
            logger.error(f"Failed to publish heartbeat: {e}", exc_info=True)

    async def main_loop(self):
        """
        Main application event loop.

        Runs continuously, performing updates at configured intervals for each experiment.
        """
        logger.info("Starting main event loop...")
        self.running = True

        # Register signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating graceful shutdown...")
            self.running = False

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Per-experiment timers: each experiment has its own forecast/retrain
        # schedule, falling back to global config if not set.
        self._next_forecast_per_exp = {}
        self._next_retrain_per_exp = {}

        now = datetime.now(timezone.utc)
        for exp_cfg in self.config.experiments:
            fc_mins = exp_cfg.forecast_every_minutes or self.config.forecast_every_minutes
            rt_hrs = exp_cfg.retrain_every_hours or self.config.retrain_every_hours
            if exp_cfg.mode == "production":
                # Production experiments retrain immediately on startup so
                # cached models exist and forecast sensors get published
                # right away — unless _restore_cached_models already
                # populated a fresh cache from disk, in which case defer
                # to when the schedule would have fired next. Avoids
                # thrashing through N sequential retrains on every restart.
                cached = self._cached_models.get(exp_cfg.name)
                cached_ts = cached.get("trained_at") if cached else None
                if cached_ts is not None and (now - cached_ts).total_seconds() < rt_hrs * 3600:
                    age_m = (now - cached_ts).total_seconds() / 60
                    self._next_retrain_per_exp[exp_cfg.name] = cached_ts + timedelta(seconds=rt_hrs * 3600)
                    logger.info(
                        f"Timers for {exp_cfg.name}: forecast every {fc_mins}m, "
                        f"retrain in {((self._next_retrain_per_exp[exp_cfg.name] - now).total_seconds() / 3600):.1f}h "
                        f"(restored cache is {age_m:.0f}m old)"
                    )
                else:
                    self._next_retrain_per_exp[exp_cfg.name] = now
                    stale_reason = (
                        f"cache {((now - cached_ts).total_seconds() / 3600):.1f}h old, stale"
                        if cached_ts is not None
                        else "no cached model"
                    )
                    logger.info(
                        f"Timers for {exp_cfg.name}: forecast every {fc_mins}m, "
                        f"retrain IMMEDIATELY ({stale_reason})"
                    )
            else:
                self._next_retrain_per_exp[exp_cfg.name] = now + timedelta(
                    seconds=rt_hrs * 3600
                )
                logger.info(
                    f"Timers for {exp_cfg.name}: forecast every {fc_mins}m, "
                    f"retrain in {rt_hrs}h"
                )
            self._next_forecast_per_exp[exp_cfg.name] = now + timedelta(seconds=fc_mins * 60)

        while self.running:
            try:
                now = datetime.now(timezone.utc)

                # Reload config so schedule changes from UI take effect
                # (only reload occasionally to avoid disk thrashing)
                if int(now.timestamp()) % 30 == 0:
                    try:
                        await self.load_config()
                    except Exception:
                        pass

                # Check each experiment's schedule
                for exp_cfg in self.config.experiments:
                    fc_mins = exp_cfg.forecast_every_minutes or self.config.forecast_every_minutes
                    rt_hrs = exp_cfg.retrain_every_hours or self.config.retrain_every_hours

                    # Initialise new experiments that didn't exist before
                    if exp_cfg.name not in self._next_forecast_per_exp:
                        self._next_retrain_per_exp[exp_cfg.name] = now
                        self._next_forecast_per_exp[exp_cfg.name] = now + timedelta(seconds=fc_mins * 60)

                    # Retrain this experiment (queued sequentially)
                    if (now >= self._next_retrain_per_exp[exp_cfg.name]
                            and exp_cfg.name not in self._running_tasks):
                        # Check not already queued
                        already_queued = any(
                            q.name == exp_cfg.name
                            for q in self._retrain_queue._queue
                        )
                        if not already_queued:
                            self._next_retrain_per_exp[exp_cfg.name] = now + timedelta(seconds=rt_hrs * 3600)
                            await self._retrain_queue.put(exp_cfg)
                            self._spawn(self._retrain_queue_consumer())

                    # Forecast this experiment
                    if (now >= self._next_forecast_per_exp[exp_cfg.name]
                            and not self._forecast_running.get(exp_cfg.name, False)):
                        self._next_forecast_per_exp[exp_cfg.name] = now + timedelta(seconds=fc_mins * 60)
                        # Reserve the slot synchronously so a second scheduler
                        # tick can't double-fire this experiment before the
                        # task body runs and sets the flag itself.
                        self._forecast_running[exp_cfg.name] = True
                        self._spawn(self._forecast_single(exp_cfg))

                # Update UI countdowns
                if self.web_app:
                    for exp_cfg in self.config.experiments:
                        status = self.web_app.state.appstate.experiment_statuses.get(
                            exp_cfg.name
                        )
                        if status and exp_cfg.name in self._next_forecast_per_exp:
                            status.next_forecast_in_seconds = max(
                                0, int((self._next_forecast_per_exp[exp_cfg.name] - now).total_seconds())
                            )
                            status.next_retrain_in_seconds = max(
                                0, int((self._next_retrain_per_exp[exp_cfg.name] - now).total_seconds())
                            )
                            status.next_update_in_seconds = status.next_forecast_in_seconds

                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(5)

        logger.info("Main event loop terminated")

    async def shutdown(self):
        """Gracefully shutdown the application."""
        logger.info("Shutting down ML Forecast Lab...")

        self.running = False

        if self.server:
            logger.info("Shutting down web server...")
            self.server.should_exit = True
            await asyncio.sleep(1)

        if self.history_db:
            logger.info("Closing database connection...")
            try:
                self.history_db.close()
            except Exception as e:
                logger.warning(f"Error closing database: {e}")

        logger.info("Shutdown complete")

    async def run(self):
        """
        Main entry point for the application.

        Initialises all components and starts the main event loop with web server.
        """
        try:
            from ml_forecast_lab import __version__
            logger.info("")
            logger.info("╔══════════════════════════════════════════════╗")
            logger.info(f"║  ML Forecast Lab v{__version__:<27}║")
            logger.info("║  Multi-model ML forecasting for HA          ║")
            logger.info("╚══════════════════════════════════════════════╝")
            logger.info("")

            # Setup directories
            self._setup_directories()

            # Load configuration
            await self.load_config()

            # Apply runtime resource limits (CPU threads, nice value) from
            # AppConfig BEFORE component init pulls in the heavy ML stacks
            # (torch, lightgbm, xgboost) so their thread pools spawn with
            # the right ceiling. Previously these settings were stored in
            # mlfl.yaml and shown on the System page but never applied.
            self._apply_runtime_resources()

            # Initialise components
            await self.initialise_components()

            # Restore any production models persisted by a previous run
            # before main_loop decides whether to queue immediate retrains.
            self._restore_cached_models()

            # Start web server
            await self.start_web_server()

            # Run main event loop
            await self.main_loop()

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            await self.shutdown()

    async def _publish_lifecycle_sensor(
        self,
        exp_cfg,
        event: str,
        outcome: str,
        attrs: Optional[dict] = None,
    ) -> None:
        """Publish a `last_<event>` companion sensor for HA automations.

        ``event`` is one of ``"benchmark"`` / ``"retrain"``. ``outcome``
        is ``"completed"`` / ``"failed"`` / ``"started"`` — captured as the
        ``outcome`` attribute, while the sensor state is the ISO timestamp
        of the event (typed as device_class=timestamp so HA picks it up
        without further config).

        Entity ID follows the same prefix / name convention as the
        forecast sensors: ``sensor.{publish_prefix}{publish_name}_last_<event>``.
        Best-effort: failures are logged but don't propagate, so a stale HA
        REST token won't break the user-visible benchmark or retrain cycle.
        """
        if not self.ha_interface:
            return
        try:
            publish_name = exp_cfg.publish_name or exp_cfg.name
            prefix = exp_cfg.publish_prefix or "mlfl_"
            entity_id = f"sensor.{prefix}{publish_name}_last_{event}"
            now_iso = datetime.now(timezone.utc).isoformat()
            payload_attrs = {
                "outcome": outcome,
                "experiment": exp_cfg.name,
                "event": event,
                "device_class": "timestamp",
                "friendly_name": f"{exp_cfg.name} last {event}",
                "icon": "mdi:flask" if event == "benchmark" else "mdi:autorenew",
            }
            if attrs:
                payload_attrs.update(attrs)
            await self.ha_interface.set_state(entity_id, now_iso, payload_attrs)
        except Exception as e:
            # Lifecycle publish is best-effort. A failure here must not
            # abort the calling benchmark / retrain code path.
            logger.debug(
                "Lifecycle sensor publish for %s/%s failed: %s",
                getattr(exp_cfg, "name", "?"), event, e,
            )

    def _apply_runtime_resources(self) -> None:
        """Apply ``cpu_cores`` and ``nice_priority`` from AppConfig.

        Previously these knobs were persisted to mlfl.yaml and surfaced on
        the System page but nothing in the codebase read them — training
        saturated every core regardless of the user's choice. We now cap
        the BLAS / OMP / torch thread pools and set the process nice value
        once at startup, before any model backend imports. Per-thread
        env-vars also affect numpy's downstream BLAS calls in workers that
        spawn after this point.
        """
        if not self.config:
            return

        cpu_cores = int(getattr(self.config, "cpu_cores", 0) or 0)
        system_cores = os.cpu_count() or 1
        threads = cpu_cores if 1 <= cpu_cores <= system_cores else system_cores

        # Set the env vars BLAS libraries pick up at import time. Workers
        # forked after this point inherit them; libraries already imported
        # (numpy is loaded at the top of this module) won't change their
        # current pool size from the env var alone, which is why we also
        # call torch.set_num_threads below.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[var] = str(threads)

        try:
            import torch
            torch.set_num_threads(threads)
            try:
                torch.set_num_interop_threads(threads)
            except RuntimeError:
                # set_num_interop_threads can only be called before any
                # parallel work begins; subsequent calls raise. Safe to
                # ignore — the value persists from the first call.
                pass
        except ImportError:
            pass

        self._applied_cpu_threads = threads

        nice_priority = int(getattr(self.config, "nice_priority", 10) or 0)
        try:
            # PRIO_PROCESS=0 sets an absolute nice value (preferred over
            # os.nice which is relative). Linux kernel silently clamps to
            # [-20, 19]; raising priority (negative values) requires
            # CAP_SYS_NICE which an unprivileged add-on container does not
            # have, so values below 0 will fail with PermissionError —
            # treat as "stay at current priority" rather than crash.
            os.setpriority(os.PRIO_PROCESS, 0, nice_priority)
            self._applied_nice = nice_priority
        except (PermissionError, OSError) as e:
            logger.warning(
                "Could not set nice priority to %d (need CAP_SYS_NICE for "
                "values below 0): %s. Process will run at default priority.",
                nice_priority, e,
            )
            try:
                self._applied_nice = os.getpriority(os.PRIO_PROCESS, 0)
            except OSError:
                self._applied_nice = None

        warn_floor = max(1, (system_cores + 1) // 2)
        if threads < warn_floor:
            logger.warning(
                "Training thread cap of %d is below half the available cores "
                "(%d) — benchmarks will be noticeably slower. Raise "
                "'Training CPU cores' on the System page if this is not "
                "intentional.",
                threads, system_cores,
            )

        logger.info(
            "Runtime resources applied: %d CPU thread(s), nice=%s "
            "(system has %d core(s))",
            threads,
            self._applied_nice if self._applied_nice is not None else "unchanged",
            system_cores,
        )

    @staticmethod
    def _setup_directories():
        """Create necessary directories for application operation."""
        directories = [
            Path("/data/ml_forecast_lab"),
            Path("/data/ml_forecast_lab/models"),
            Path("/data/ml_forecast_lab/logs"),
            Path("/config/ml_forecast_lab"),
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ready: {directory}")


async def main():
    """
    Async entry point for ML Forecast Lab.

    Creates and runs the main application instance.
    """
    app = MLForecastLabApp()
    await app.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
