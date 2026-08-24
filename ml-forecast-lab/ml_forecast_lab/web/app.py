"""
FastAPI web application for ML Forecast Lab.

Provides dashboard and API endpoints for monitoring and managing forecasting
experiments, model benchmarking, and production deployment.
"""

import asyncio
import json
import logging
import math
import os
import platform
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_PATH_REDACT_RE = re.compile(r"(?:[A-Za-z]:)?(?:/[\w.\-+]+)+")


def _safe_error(exc: BaseException) -> str:
    """User-facing error string with filesystem paths redacted.

    The full traceback is logged server-side via ``exc_info=True``; callers
    pass this through to JSON bodies so internal paths don't leak into
    client-visible diagnostics. Only the first line of ``str(exc)`` is kept.
    """
    msg = str(exc).split("\n", 1)[0]
    return f"{type(exc).__name__}: {_PATH_REDACT_RE.sub('<path>', msg)}"


def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN / ±Infinity) with None.

    A non-finite float is not representable in spec-compliant JSON. Starlette
    renders with ``json.dumps(allow_nan=False)``, so one raises ``ValueError``
    during render (older Starlette emitted a bare ``NaN`` token instead — still
    invalid JSON). Either way a strict client parser rejects the result: WebKit
    (Safari and the iOS Home Assistant app's WKWebView) throws ``SyntaxError:
    The string did not match the expected pattern.`` and the whole payload is
    lost. Any endpoint returning computed floats (metrics, ratios, per-bin means
    over possibly-empty groups) can produce a NaN, so we scrub the structure at
    the response boundary rather than chasing every arithmetic site. ``None``
    becomes JSON ``null``, which the frontends already handle (the charts use
    ``connectgaps:false``). Used by ``SafeJSONResponse``.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj

from ml_forecast_lab import __version__ as APP_VERSION

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse as _StarletteJSONResponse
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.requests import Request

logger = logging.getLogger(__name__)


class SafeJSONResponse(_StarletteJSONResponse):
    """``JSONResponse`` that replaces NaN / ±Infinity with ``null``.

    Endpoints that return computed floats (metrics, ratios, per-bin means
    over possibly-empty groups) can produce a non-finite value. Starlette
    renders with ``json.dumps(allow_nan=False)``, so such a value raises
    ``ValueError`` *during render* — after the endpoint's own try/except has
    already returned — surfacing as an unhandled 500 with a non-JSON body.
    (Older Starlette instead emitted a bare ``NaN`` token, i.e. invalid
    JSON.) Either way a strict client parser chokes: WebKit — used by Safari
    **and the iOS Home Assistant companion app's WKWebView** — throws
    ``SyntaxError: The string did not match the expected pattern.`` and the
    whole tab fails to load.

    Sanitising in ``render`` (rather than at each arithmetic site) guarantees
    every payload is spec-valid JSON. ``null`` is what the frontends already
    expect for gaps (the charts use ``connectgaps:false``).
    """

    def render(self, content: Any) -> bytes:
        return super().render(_json_safe(content))


# Bind the module-wide ``JSONResponse`` name to the NaN-safe subclass so EVERY
# endpoint in this file (current and future) renders through it — a single
# chokepoint, rather than remembering to opt in per-route. Non-finite floats
# can appear in metric/results/report payloads from many endpoints (benchmark
# results, forecast accuracy, the comparison tab, the data report), and any one
# of them would otherwise 500 a strict WebKit client (Safari / the iOS HA app).
# ``_StarletteJSONResponse`` remains available for anything that genuinely needs
# the unmodified base.
JSONResponse = SafeJSONResponse



# Shared aiohttp session for short HA probes from request handlers
# (forecast-attrs, validate-covariate, ha-entities). The v2.38.7 chip
# fires one /api/covariates/validate per row on page load — without a
# shared session, a 20-row experiment opened 20 separate TLS handshakes
# in ~1.6s, competing with the training loop's HA traffic. Lazy-init on
# the running event loop so test code doesn't pay the cost just for
# importing the module.
_shared_ha_session: Optional['_aiohttp.ClientSession'] = None
_shared_ha_session_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_shared_ha_session():
    """Return a process-wide aiohttp ClientSession bound to the running
    loop. Recreated if the loop changed (e.g. test fixtures spawning
    fresh loops) or if the cached session has been closed."""
    import aiohttp as _aiohttp
    global _shared_ha_session, _shared_ha_session_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop: only happens off the request path; fall back to a
        # one-shot session the caller will own and close.
        return _aiohttp.ClientSession()
    if (_shared_ha_session is None
            or _shared_ha_session.closed
            or _shared_ha_session_loop is not loop):
        _shared_ha_session = _aiohttp.ClientSession()
        _shared_ha_session_loop = loop
    return _shared_ha_session


# Data models
class MetricValue(BaseModel):
    """Single metric value with mean and std across folds."""

    mean: float
    std: float


class ModelResult(BaseModel):
    """Results for a single model in a benchmark."""

    name: str
    mae: MetricValue
    rmse: MetricValue
    mase: MetricValue
    train_time_seconds: float
    rank: int
    mean_rank: float = 0.0
    # 95% bootstrap CI on mean_rank computed by resampling CV folds
    # with replacement (B=1000). Used by the UI to surface ties — if
    # this model's CI overlaps the leader's CI we render its rank as
    # "tied with #1" rather than a discrete number. See
    # docs/RANKING_NOTES.md for what the CI does and does not claim.
    # Null when the benchmark had <2 valid folds (a CI off one fold
    # is meaningless).
    mean_rank_low: Optional[float] = None
    mean_rank_high: Optional[float] = None
    is_production: bool = False
    fold_results: Optional[List[Dict[str, float]]] = None
    train_mae: Optional[MetricValue] = None
    train_rmse: Optional[MetricValue] = None
    training_history: Optional[Dict[str, List[float]]] = None
    # Daily-cumulative metrics. Same MAE/RMSE/MASE methodology, but
    # computed on per-day totals (each day's predictions summed, each
    # day's actuals summed, then compared). Better for use cases where
    # daily totals matter more than 30-minute precision (e.g. hot-water
    # or energy demand). The Daily Rank uses the same composite
    # mean-rank averaging step (Demšar-style aggregation only — the
    # full Demšar (2006) test does not apply, see
    # docs/RANKING_NOTES.md) and is informational; it does NOT drive
    # Promote/Tuning workflows.
    daily_mae: Optional[MetricValue] = None
    daily_rmse: Optional[MetricValue] = None
    daily_mase: Optional[MetricValue] = None
    daily_rank: Optional[int] = None
    daily_mean_rank: Optional[float] = None
    daily_mean_rank_low: Optional[float] = None
    daily_mean_rank_high: Optional[float] = None
    # Stability flag: True when the model's per-fold error metric spread
    # is large enough that its mean-rank "typically wins" story hides a
    # blow-up fold. The composite mean rank is outlier-robust (a single
    # catastrophic fold only costs one last-place finish), so a model
    # that is great on most folds but catastrophic on one can out-rank a
    # consistently-mediocre model. This flag surfaces that so an
    # unstable model can't masquerade as a solid mid-pack pick. See
    # docs/RANKING_NOTES.md.
    unstable: bool = False
    instability_reason: Optional[str] = None


class BenchmarkResult(BaseModel):
    """Complete benchmark run results for an experiment."""

    experiment_name: str
    timestamp: str
    status: str  # 'running', 'completed', 'failed'
    models: List[ModelResult]
    best_model_name: Optional[str] = None
    error_message: Optional[str] = None
    # Pairwise model comparison — paired t-test on fold MAE differences.
    # Each row: {model_a, model_b, mean_diff, t_stat, p_value, n_folds,
    # significant}. With few folds (typically 5) the test is weak; we use
    # it as a "are the differences inside fold noise?" indicator rather
    # than a formal hypothesis test — the UI info-tip says so.
    pairwise_dm: Optional[List[Dict[str, Any]]] = None
    # Seasonal-naive reference baseline. Always populated when the
    # baseline ran successfully — the UI shows a "vs Seasonal Naive"
    # skill chip so users can see whether the best learned model is
    # actually beating "today equals yesterday + last week's seasonality".
    naive_baseline_mae: Optional[float] = None
    # Whether the baseline was user-enabled (appears in the rank table)
    # or force-included by the runner for the skill chip only (hidden
    # from the table but used to compute skill).
    naive_baseline_was_enabled: Optional[bool] = None
    # Training-window vs test-window drift statistics. Comparing the
    # target distribution in the earliest fold's train window against
    # the latest fold's test window. PSI < 0.1 is "stable",
    # 0.1–0.2 "moderate", >0.2 "shifted" — used as a UI verdict to
    # explain why CV scores might disagree with live behaviour.
    drift: Optional[Dict[str, Any]] = None
    # Names of models that failed at least one fold during the
    # benchmark. Surfaced separately in the UI under a "did not
    # complete" section so they don't appear at the bottom of the
    # leaderboard with a fabricated last-place rank that inflates
    # other models' apparent dominance.
    did_not_complete: List[str] = []
    # Names of models excluded from the DAILY-cumulative ranking
    # specifically — typically because the test span on some fold
    # covered <2 distinct dates so daily totals weren't computable.
    # These models ARE ranked in the per-interval leaderboard; this
    # list is rendered under the Daily Cumulative Accuracy table only,
    # so a per-interval-ranked model isn't confusingly shown as
    # "did not complete".
    did_not_complete_daily: List[str] = []


class ExperimentStatus(BaseModel):
    """Status of an experiment."""

    name: str
    target_entity: str
    mode: str  # 'lab' or 'production'
    best_model: Optional[str] = None
    selected_model: Optional[str] = None  # User's chosen model (defaults to best)
    # ISO-timestamp tag set each time the selected model finishes
    # training. Stamped on every subsequent log_forecast so analytics
    # queries can segregate pre- and post-retrain cycles of the same
    # model_name — fixes the "retrain under same name silently
    # contaminates stability" artefact.
    model_version: Optional[str] = None
    last_benchmark_timestamp: Optional[str] = None
    last_benchmark_status: str = "pending"
    last_error: Optional[str] = None  # Human-readable error from last failed cycle
    next_forecast_in_seconds: Optional[int] = None
    next_retrain_in_seconds: Optional[int] = None
    next_update_in_seconds: Optional[int] = None  # Legacy alias for forecast
    publish_entity: Optional[str] = None  # e.g. "sensor.mlfl_solar_forecast"


class ForecastPoint(BaseModel):
    """Single forecast point."""

    timestamp: str
    actual: Optional[float] = None
    predicted_mean: Optional[float] = None
    predicted_lower: Optional[float] = None
    predicted_upper: Optional[float] = None


class ForecastData(BaseModel):
    """Forecast data for charting."""

    experiment_name: str
    horizon_minutes: int
    points: List[ForecastPoint]
    model_name: Optional[str] = None


class ModelPrediction(BaseModel):
    """Predictions from a single model on holdout data.

    Note: trace colours are no longer set here. The frontend (Plotly)
    auto-assigns colours from a colorway in trace order.
    """

    model_name: str
    timestamps: List[str]
    actuals: List[Optional[float]]
    predictions: List[Optional[float]]


class LabForecastData(BaseModel):
    """Multi-model prediction data for lab mode visualisation."""

    experiment_name: str
    holdout_start: str
    holdout_end: str
    model_predictions: List[ModelPrediction]


class CovariateAnalysisCellResult(BaseModel):
    """Result for one model × one covariate configuration."""
    mae: float
    rmse: float
    mase: float = float('nan')
    change_pct: Optional[float] = None      # MAE % change vs baseline
    rmse_change_pct: Optional[float] = None  # RMSE % change vs baseline
    mase_change_pct: Optional[float] = None  # MASE % change vs baseline


class CovariateAnalysisResult(BaseModel):
    """Full covariate analysis results."""
    experiment_name: str
    timestamp: str
    status: str  # 'running', 'completed', 'failed'
    baseline_label: str  # "All covariates"
    covariate_labels: List[str]  # ["No covariates", "Without charge", ...]
    model_names: List[str]
    # results[covariate_label][model_name] = CovariateAnalysisCellResult
    results: Dict[str, Dict[str, CovariateAnalysisCellResult]]
    recommendations: List[Dict[str, str]]  # [{"icon": "✓", "text": "...", "color": "green"}, ...]
    total_runs: int = 0
    completed_runs: int = 0


class TuningTrialResult(BaseModel):
    """Result for one hyperparameter tuning trial."""
    trial_id: int
    params: Dict[str, Any]
    mae: float
    rmse: float
    mase: float
    duration_seconds: float = 0.0
    status: str = "completed"  # completed, pruned, failed


class TuningResult(BaseModel):
    """Hyperparameter tuning results for one model."""
    experiment_name: str
    model_name: str
    timestamp: str
    status: str  # running, completed, failed
    search_strategy: str = "tpe"
    n_trials: int = 30
    completed_trials: int = 0
    trials: List[TuningTrialResult] = []
    best_trial_id: Optional[int] = None
    best_params: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    error_message: Optional[str] = None
    # Holdout comparison: default vs tuned predictions
    holdout_timestamps: Optional[List[str]] = None
    holdout_actuals: Optional[List[Optional[float]]] = None
    holdout_default: Optional[List[Optional[float]]] = None  # predictions with default params
    holdout_tuned: Optional[List[Optional[float]]] = None    # predictions with best tuned params
    default_mae: Optional[float] = None
    tuned_mae: Optional[float] = None


class FeatureImportanceData(BaseModel):
    """Feature importance from a trained model."""

    model_name: str
    features: List[Dict[str, Any]]  # [{"name": "hour_of_day", "importance": 0.25}, ...]


class HealthStatus(BaseModel):
    """System health status."""

    status: str
    service: str
    version: str
    timestamp: str
    experiments_total: int
    experiments_lab: int
    experiments_production: int


# In-memory state management
class AppState:
    """Simple in-memory state for benchmark results and experiment status."""

    def __init__(self):
        """Initialise state."""
        self.benchmark_results: Dict[str, BenchmarkResult] = {}
        self.experiment_statuses: Dict[str, ExperimentStatus] = {}
        self.forecast_data: Dict[str, ForecastData] = {}
        self.lab_forecast_data: Dict[str, LabForecastData] = {}
        self.feature_importances: Dict[str, List[FeatureImportanceData]] = {}
        self.covariate_analysis_results: Dict[str, CovariateAnalysisResult] = {}
        self.covariate_analysis_callback = None  # Set by main app for triggering
        self.benchmark_callback = None  # Set by main app for triggering
        self.tuning_results: Dict[str, TuningResult] = {}
        self.tuning_callback = None  # Set by main app for triggering
        # Sweep mode: when the user triggers "Tune all enabled" on the
        # Tuning tab the final per-model results accumulate here so the
        # UI can render a stacked table after the sweep completes.
        # tune_all_results[experiment] = List[TuningResult] (one per
        # model tuned during the sweep).
        self.tune_all_results: Dict[str, List[TuningResult]] = {}
        # Set by main app — runs _run_tuning sequentially over models.
        self.tune_all_callback = None
        # Trigger an immediate retrain for one experiment. Used by the
        # apply-tuning and apply-covariate-best endpoints so the user
        # doesn't have to wait for the next scheduled retrain cycle.
        self.retrain_callback = None  # Set by main app
        self.stop_training_callback = None  # Set by main app
        self.running_benchmarks: set = set()
        self.training_queue: List[Dict] = []  # Queue of pending pipeline requests
        self._queue_processing: bool = False
        self._pipeline_tasks: Dict[str, Any] = {}
        self.history_db = None  # Set by main app for forecast accuracy queries
        self.last_update: Optional[datetime] = None
        self.next_update_seconds: Optional[int] = None
        # HA's configured time zone (IANA name, e.g. "Europe/London").
        # Populated on startup by the main app via HAInterface.get_config().
        # None until set — frontend falls back to browser TZ in that case.
        # Matters for Californian users managing a UK HA: charts render in
        # HA-local time so axis labels match when events physically happened.
        self.ha_time_zone: Optional[str] = None
        # Resolved runtime-resource caps actually applied to this process —
        # surfaced on the System page so the user can verify their
        # cpu_cores / nice_priority settings took effect.
        self.applied_cpu_threads: Optional[int] = None
        self.applied_nice: Optional[int] = None
        # Rollback support (wired by main.py once components are
        # initialised). Used by the per-experiment 'Roll back' button.
        self.rollback_callback = None
        self.cached_model_dir = None
        # Drop a trained model (in-memory cache + on-disk weights) for one
        # experiment. Wired by main.py; used when an experiment's target
        # sensor is replaced so the stale model can't forecast the new signal.
        self.reset_model_callback = None
        # Pre-flight data sanity check — see /experiment/{name}/data-report.
        self.data_report_callback = None
        # Strong references to fire-and-forget tasks. asyncio holds only a
        # weak reference to running tasks; without this set a coroutine
        # scheduled via create_task can be garbage-collected before its
        # exception is logged. discard() runs in the done callback.
        self._background_tasks: set = set()

    def spawn(self, coro):
        """Schedule *coro* on the running loop and keep a strong reference.

        Returns the underlying ``asyncio.Task``. The done callback both
        releases the reference and logs any unhandled exception so silent
        failures (cancelled tasks, OSError on the HA API, etc.) don't vanish.
        """
        import asyncio as _aio
        task = _aio.create_task(coro)
        self._background_tasks.add(task)

        def _on_done(t: _aio.Task) -> None:
            self._background_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.error(
                    "Background task failed: %s", exc, exc_info=exc,
                )

        task.add_done_callback(_on_done)
        return task

    def start_benchmark(self, experiment_name: str):
        """Mark benchmark as running."""
        self.running_benchmarks.add(experiment_name)

    def end_benchmark(self, experiment_name: str):
        """Mark benchmark as completed."""
        self.running_benchmarks.discard(experiment_name)

    def is_benchmark_running(self, experiment_name: str) -> bool:
        """Check if benchmark is running."""
        return experiment_name in self.running_benchmarks

    def get_queue_position(self, experiment_name: str) -> int:
        """Get 1-based position in queue, or 0 if not queued."""
        for i, item in enumerate(self.training_queue):
            if item["name"] == experiment_name:
                return i + 1
        return 0

    def remove_from_queue(self, experiment_name: str) -> bool:
        """Remove experiment from queue. Returns True if it was queued."""
        for i, item in enumerate(self.training_queue):
            if item["name"] == experiment_name:
                self.training_queue.pop(i)
                return True
        return False


def classify_covariate_state(
    entity_id: str,
    state_obj: dict,
    future_attribute: Optional[str] = None,
    future_value_key: Optional[str] = None,
) -> dict:
    """Classify a covariate row's data availability from a single
    HA ``/api/states`` response (v2.38.7). Pure function — no IO —
    so it's straightforward to unit-test without the FastAPI machinery.

    Decision matrix:

    * State numeric, no ``future_attribute`` → **ok** (last value).
    * State numeric, ``future_attribute`` parses → **ok**.
    * State numeric, ``future_attribute`` given but doesn't parse → **partial**.
    * State categorical (``weather.partlycloudy`` etc.), ``future_attribute``
      parses → **partial** (lagged side won't carry numeric history
      unless the resolver routes through the v2.38.4 attribute path
      for ``weather.*`` entities — UI flags it so the user knows the
      past block depends on that path working).
    * State categorical, ``weather.*`` + ``future_value_key`` and the
      named attribute exists numerically on the state object → **ok**
      (v2.38.4 attribute-history path will handle the lagged side).
    * State categorical, ``future_attribute`` missing or broken → **broken**.
    * ``weather.*`` entity with ``future_attribute`` in {hourly, daily,
      twice_daily} → the resolver fetches via the
      ``weather.get_forecasts`` SERVICE (HA 2023.9+, see
      covariates.py:230), NOT via state attributes. The state-only
      fetch this validator does can't probe that path, so we treat
      it as "expected to work" rather than false-flagging it as
      partial / broken on a missing state attribute.

    Return shape::

        {
            ok: bool,
            status: "ok" | "partial" | "broken",
            state_value: float | null,
            last_changed: str | null,
            message: str,
            attribute_preview: <numeric>|null,
        }
    """
    from ml_forecast_lab.covariates import _parse_forecast_attribute
    from ml_forecast_lab.ha_interface import state_to_float

    raw_state = state_obj.get("state")
    last_changed = state_obj.get("last_changed")
    attrs = state_obj.get("attributes") or {}
    state_value = state_to_float(raw_state)
    state_is_numeric = state_value is not None

    # HA 2023.9+ moved weather forecasts out of state attributes and
    # into the weather.get_forecasts service call. The resolver short-
    # circuits to that service when future_attribute is one of the
    # service types, so we can't (and shouldn't) try to parse it from
    # the state we just fetched — there's nothing there to parse.
    WEATHER_SERVICE_TYPES = {"hourly", "daily", "twice_daily"}
    is_weather_service_future = (
        isinstance(entity_id, str)
        and entity_id.startswith("weather.")
        and future_attribute in WEATHER_SERVICE_TYPES
    )

    attribute_preview = None
    attribute_parsed_ok = None
    if future_attribute and not is_weather_service_future:
        attr_raw = attrs.get(future_attribute)
        if attr_raw is None:
            attribute_parsed_ok = False
        else:
            parsed = _parse_forecast_attribute(attr_raw, value_key=future_value_key)
            if parsed is not None and not parsed.empty:
                attribute_parsed_ok = True
                attribute_preview = float(parsed.iloc[0])
            else:
                attribute_parsed_ok = False

    # The attribute-history path (covariates.py:137-144, v2.38.4+ and
    # generalised in v2.39.3) routes through ``state_to_float`` which
    # already parses numeric strings (OpenWeatherMap / met.no often
    # store temperature as ``'16.5'``). Don't reject those here.
    weather_attr_path_ok = (
        isinstance(entity_id, str)
        and entity_id.startswith("weather.")
        and future_value_key
        and state_to_float(attrs.get(future_value_key)) is not None
    )

    if (not state_is_numeric
            and not weather_attr_path_ok
            and attribute_parsed_ok is not True
            and not is_weather_service_future):
        return {
            "ok": False, "status": "broken",
            "state_value": None, "last_changed": last_changed,
            "message": (
                f"State '{raw_state}' is not numeric and no usable "
                "future_attribute / weather attribute-history path."
            ),
            "attribute_preview": None,
        }

    if future_attribute and attribute_parsed_ok is False:
        # Describe the actual lagged-side source so the message isn't
        # misleading. ``last=None`` for a categorical weather state was
        # confusing when the resolver pulls lagged history from the
        # weather-attr-history path instead.
        if state_is_numeric:
            lagged_desc = f"last={state_value}"
        elif weather_attr_path_ok:
            lagged_desc = (
                f"via attribute '{future_value_key}'"
                f" (current={attrs[future_value_key]})"
            )
        else:
            lagged_desc = "no lagged source"
        return {
            "ok": True, "status": "partial",
            "state_value": state_value,
            "last_changed": last_changed,
            "message": (
                f"Lagged side ok ({lagged_desc}) but future "
                f"attribute '{future_attribute}' didn't parse "
                "— check key name or value_key."
            ),
            "attribute_preview": None,
        }

    # Categorical state + (parsing future_attribute OR weather service
    # future) but no lagged-side numeric source = partial: the future
    # block works but the resolver will receive an all-NaN lagged
    # column, which the v2.38.3 empty-column guard either drops or
    # zero-fills. Without this branch the validator returned ``ok``
    # for the case its own docstring promises is partial.
    if (not state_is_numeric
            and not weather_attr_path_ok
            and (attribute_parsed_ok is True or is_weather_service_future)):
        future_desc = (
            f"weather.get_forecasts({future_attribute})"
            if is_weather_service_future
            else f"attribute '{future_attribute}'"
        )
        return {
            "ok": True, "status": "partial",
            "state_value": None,
            "last_changed": last_changed,
            "message": (
                f"Future side ok ({future_desc}) but state "
                f"'{raw_state}' is non-numeric and no future_value_key "
                "is set — lagged history will be empty (model loses "
                "the past signal for this channel)."
            ),
            "attribute_preview": attribute_preview,
        }

    bits = []
    if state_is_numeric:
        bits.append(f"last={state_value}")
    elif weather_attr_path_ok:
        bits.append(f"{future_value_key}={attrs[future_value_key]}")
    if attribute_parsed_ok:
        bits.append(f"future attr parses (first={attribute_preview})")
    elif is_weather_service_future:
        bits.append(
            f"future via weather.get_forecasts({future_attribute})"
            " — not directly probed"
        )
    return {
        "ok": True, "status": "ok",
        "state_value": state_value,
        "last_changed": last_changed,
        "message": ", ".join(bits) or "Entity reachable",
        "attribute_preview": attribute_preview,
    }


def create_app(config_path: Optional[Path] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Parameters
    ----------
    config_path : Optional[Path]
        Path to configuration file (not currently used by web app directly)

    Returns
    -------
    FastAPI
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="ML Forecast Lab",
        description="Multi-model ML forecasting and benchmarking system",
        version=APP_VERSION,
    )

    # Initialize state
    state = AppState()
    app.state.appstate = state

    # Setup template and static paths
    template_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"

    # Create directories if they don't exist
    template_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)

    templates = Jinja2Templates(directory=str(template_dir))

    # Template globals — available in every rendered template without
    # threading through each TemplateResponse context dict. Used by
    # base.html to cache-bust static assets on version bumps.
    #
    # When the process booted from a developer branch overlay, the version
    # shown across the whole UI is annotated (e.g. "2.42.0 (dev: foo@1a2b3c4)")
    # and a banner is rendered site-wide. Both are resolved once at startup
    # because the overlay only changes via an add-on restart, so startup is
    # exactly the right time to read it.
    from ml_forecast_lab import dev_branch
    templates.env.globals["app_version"] = dev_branch.version_label(APP_VERSION)
    templates.env.globals["dev_overlay"] = (
        dev_branch.active_status() if dev_branch.is_overlay_running() else None
    )

    # Custom Jinja filters
    def _humanise_name(value: str) -> str:
        """Convert snake_case experiment names to Title Case for display."""
        return value.replace("_", " ").title()

    templates.env.filters["humanise"] = _humanise_name

    # Mount static files with a long Cache-Control on the third-party
    # JS bundles. The Plotly bundle is ~1 MB and is version-locked at
    # build time (the filename is the cache-busting handle), so there's
    # no reason to revalidate it on every page navigation. Templates
    # and our own CSS/JS get a short cache so addon updates take effect
    # quickly without a hard refresh.
    class CachedStaticFiles(StaticFiles):
        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)
            if path.endswith((".min.js", ".min.css")):
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            else:
                response.headers["Cache-Control"] = "public, max-age=300"
            return response

    if static_dir.exists():
        app.mount(
            "/static",
            CachedStaticFiles(directory=str(static_dir)),
            name="static",
        )

    # GZip the JSON analytics responses (forecast-accuracy, stability,
    # evolution, trajectory) and the Plotly bundle. Cuts the cold-load
    # cost of the Forecast Accuracy and Covariate Analysis tabs by ~70%.
    # minimum_size=1024 avoids the overhead on the small flag responses.
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # The add-on is reached only via Home Assistant ingress (same-origin),
    # so cross-origin requests have no legitimate use case. No CORSMiddleware
    # is installed: any third-party origin attempting a fetch will be blocked
    # by the browser's default same-origin policy.

    # ========== Ingress support ==========

    def _get_base_path(request: Request) -> str:
        """Get the ingress base path from HA proxy headers, or empty string."""
        return request.headers.get("X-Ingress-Path", "")

    def _find_config_path() -> Optional[Path]:
        """Locate the mlfl.yaml config file.

        Honours an explicit override passed to ``create_app(config_path=...)``
        so tests can inject a temp config without touching real add-on paths.
        The slug-hashed glob is anchored to HA's actual 8-hex-character prefix
        so a community fork with a slug like ``psweens_ml_forecast_lab`` can
        not accidentally hijack the lookup.
        """
        if config_path is not None and Path(config_path).exists():
            return Path(config_path)
        import glob as _glob
        for p in [
            Path("/addon_configs/ml_forecast_lab/mlfl.yaml"),
            Path("/config/mlfl.yaml"),
        ]:
            if p.exists():
                return p
        for match in _glob.glob(
            "/addon_configs/[0-9a-f][0-9a-f][0-9a-f][0-9a-f]"
            "[0-9a-f][0-9a-f][0-9a-f][0-9a-f]_ml_forecast_lab/mlfl.yaml"
        ):
            return Path(match)
        return None

    def _resolve_model_filter(experiment_name: str, request: Request):
        """Resolve the (model_name, model_version) filter for analytics queries.

        Contract:
          - Default (no query params) → filter to the experiment's current
            selected/best model AND its latest training tag. This keeps
            metrics pinned to the active champion's current weights so
            pre-retrain or rotated-out data doesn't pool in.
          - ``?model=all`` → no model filter AND no version filter (show
            everything).
          - ``?model=<name>`` → filter to that name, no version filter
            (ask for the whole history of that model across retrains).
          - ``?version=all`` → suppress just the version filter (keep
            the name filter but include all weight regimes).
          - ``?version=<tag>`` → filter to that specific version.

        Returns
        -------
        (model_name: Optional[str], model_version: Optional[str])
        """
        exp_status = app.state.appstate.experiment_statuses.get(experiment_name)
        default_model = (
            getattr(exp_status, "selected_model", None)
            or getattr(exp_status, "best_model", None)
            if exp_status else None
        )
        # `status.model_version` is a single field — it tracks whichever
        # model was *last retrained*, not the user's UI selection. If
        # the user has selected a non-champion model (e.g. they picked
        # lightgbm but the pipeline is training xgboost), applying
        # status.model_version as the version filter yields an
        # impossible combo: (lightgbm, xgboost's training timestamp)
        # never has rows. Only apply the version default when
        # selected_model matches the model whose version we're
        # tracking — i.e. the current champion. Otherwise fall back to
        # "all versions of that model", which is the right semantic
        # given we don't currently track per-model versions.
        best_model = getattr(exp_status, "best_model", None) if exp_status else None
        default_version = None
        if exp_status and default_model and default_model == best_model:
            default_version = getattr(exp_status, "model_version", None)

        model_param = request.query_params.get("model")
        version_param = request.query_params.get("version")

        if model_param == "all":
            model_name = None
            model_version = None  # 'all' overrides both dimensions
        elif model_param:
            model_name = model_param
            # When the caller asks for a specific model but doesn't pin a
            # version, give them the whole history of that model.
            model_version = None if version_param in (None, "all") else version_param
        else:
            model_name = default_model
            if version_param == "all":
                model_version = None
            elif version_param:
                model_version = version_param
            else:
                model_version = default_version
        return model_name, model_version

    # ---- Model parameter schema (type, default, display label) ----

    MODEL_PARAM_SCHEMA: Dict[str, Dict[str, dict]] = {
        "lightgbm": {
            "n_estimators": {"type": "int", "default": 500, "label": "Number of trees", "min": 10, "max": 5000},
            "max_depth": {"type": "int", "default": 6, "label": "Max tree depth", "min": 1, "max": 20},
            "learning_rate": {"type": "float", "default": 0.05, "label": "Learning rate", "min": 0.001, "max": 1.0, "step": 0.001},
            "num_leaves": {"type": "int", "default": 31, "label": "Max leaves", "min": 2, "max": 256},
            "min_child_samples": {"type": "int", "default": 10, "label": "Min samples per leaf", "min": 1, "max": 100},
            "subsample": {"type": "float", "default": 0.8, "label": "Row subsample ratio", "min": 0.1, "max": 1.0, "step": 0.05},
            "colsample_bytree": {"type": "float", "default": 0.8, "label": "Column subsample ratio", "min": 0.1, "max": 1.0, "step": 0.05},
            "reg_alpha": {"type": "float", "default": 0.1, "label": "L1 regularisation", "min": 0.0, "max": 10.0, "step": 0.01},
            "reg_lambda": {"type": "float", "default": 0.1, "label": "L2 regularisation", "min": 0.0, "max": 10.0, "step": 0.01},
        },
        "xgboost": {
            "n_estimators": {"type": "int", "default": 500, "label": "Number of trees", "min": 10, "max": 5000},
            "max_depth": {"type": "int", "default": 6, "label": "Max tree depth", "min": 1, "max": 20},
            "learning_rate": {"type": "float", "default": 0.05, "label": "Learning rate", "min": 0.001, "max": 1.0, "step": 0.001},
            "subsample": {"type": "float", "default": 0.8, "label": "Row subsample ratio", "min": 0.1, "max": 1.0, "step": 0.05},
            "colsample_bytree": {"type": "float", "default": 0.8, "label": "Column subsample ratio", "min": 0.1, "max": 1.0, "step": 0.05},
            "reg_alpha": {"type": "float", "default": 0.1, "label": "L1 regularisation", "min": 0.0, "max": 10.0, "step": 0.01},
            "reg_lambda": {"type": "float", "default": 1.0, "label": "L2 regularisation", "min": 0.0, "max": 10.0, "step": 0.01},
        },
        "lstm": {
            "hidden_size": {"type": "int", "default": 64, "label": "Hidden size", "min": 8, "max": 512},
            "num_layers": {"type": "int", "default": 2, "label": "LSTM layers", "min": 1, "max": 8},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        "cnn": {
            "n_filters": {"type": "int", "default": 32, "label": "Filters per layer", "min": 8, "max": 128},
            "kernel_size": {"type": "int", "default": 3, "label": "Kernel size", "min": 2, "max": 7},
            "n_layers": {"type": "int", "default": 4, "label": "Conv layers", "min": 1, "max": 8},
            "dilation_base": {"type": "int", "default": 2, "label": "Dilation base", "min": 1, "max": 3},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        "dlinear": {
            "kernel_size": {"type": "int", "default": 25, "label": "Decomposition kernel", "min": 3, "max": 101},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        "nbeats": {
            "hidden_size": {"type": "int", "default": 64, "label": "Hidden size", "min": 8, "max": 256},
            "n_stacks": {"type": "int", "default": 2, "label": "Stacks", "min": 1, "max": 4},
            "blocks_per_stack": {"type": "int", "default": 2, "label": "Blocks per stack", "min": 1, "max": 4},
            "n_fc_layers": {"type": "int", "default": 4, "label": "FC layers per block", "min": 1, "max": 6},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        "nhits": {
            "hidden_size": {"type": "int", "default": 64, "label": "Hidden size", "min": 8, "max": 512},
            "n_stacks": {"type": "int", "default": 3, "label": "Stacks", "min": 1, "max": 8},
            "blocks_per_stack": {"type": "int", "default": 1, "label": "Blocks per stack", "min": 1, "max": 8},
            "n_fc_layers": {"type": "int", "default": 4, "label": "FC layers per block", "min": 1, "max": 8},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        "tide": {
            "hidden_size": {"type": "int", "default": 64, "label": "Hidden size", "min": 8, "max": 512},
            "encoder_layers": {"type": "int", "default": 2, "label": "Encoder layers", "min": 1, "max": 8},
            "decoder_layers": {"type": "int", "default": 2, "label": "Decoder layers", "min": 1, "max": 8},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        "tsmixer": {
            "n_mixer_layers": {"type": "int", "default": 4, "label": "Mixer layers", "min": 1, "max": 12},
            "hidden": {"type": "int", "default": 64, "label": "Hidden size", "min": 8, "max": 512},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        "sparsetsf": {
            "period_len": {"type": "int", "default": 48, "label": "Period length", "min": 2, "max": 336},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        "patchtst": {
            "patch_len": {"type": "int", "default": 8, "label": "Patch length", "min": 2, "max": 48},
            "stride": {"type": "int", "default": 4, "label": "Stride", "min": 1, "max": 24},
            "d_model": {"type": "int", "default": 32, "label": "Model dimension", "min": 8, "max": 256},
            "n_heads": {"type": "int", "default": 4, "label": "Attention heads", "min": 1, "max": 16},
            "n_encoder_layers": {"type": "int", "default": 2, "label": "Encoder layers", "min": 1, "max": 8},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        "itransformer": {
            "d_model": {"type": "int", "default": 32, "label": "Model dimension", "min": 8, "max": 256},
            "n_heads": {"type": "int", "default": 4, "label": "Attention heads", "min": 1, "max": 16},
            "n_encoder_layers": {"type": "int", "default": 2, "label": "Encoder layers", "min": 1, "max": 8},
            "dim_feedforward": {"type": "int", "default": 64, "label": "Feedforward dimension", "min": 16, "max": 512},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        "crossformer": {
            "seg_len": {"type": "int", "default": 6, "label": "Segment length", "min": 2, "max": 48},
            "d_model": {"type": "int", "default": 32, "label": "Model dimension", "min": 8, "max": 256},
            "n_heads": {"type": "int", "default": 4, "label": "Attention heads", "min": 1, "max": 16},
            "n_layers": {"type": "int", "default": 2, "label": "Encoder layers", "min": 1, "max": 8},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        "timesnet": {
            "d_model": {"type": "int", "default": 16, "label": "Model dimension", "min": 8, "max": 256},
            "n_layers": {"type": "int", "default": 2, "label": "TimesBlock layers", "min": 1, "max": 8},
            "top_k": {"type": "int", "default": 3, "label": "Top-K periods", "min": 1, "max": 10},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        "catboost": {
            # CatBoost builds oblivious (symmetric) trees, so every
            # depth level strictly doubles per-tree cost regardless of
            # data. max_depth=16 runs ~1000x slower per tree than
            # depth=6 and caused tuning trials to appear stalled —
            # capped to CatBoost's practical range (docs recommend
            # 6-10). n_estimators capped at 2000 so a pathological
            # lr=0.001 trial that never triggers early-stopping still
            # finishes within the study budget.
            "n_estimators": {"type": "int", "default": 500, "label": "Number of trees", "min": 10, "max": 2000},
            "max_depth": {"type": "int", "default": 6, "label": "Max tree depth", "min": 3, "max": 10},
            "learning_rate": {"type": "float", "default": 0.05, "label": "Learning rate", "min": 0.001, "max": 1.0, "step": 0.001},
            "l2_leaf_reg": {"type": "float", "default": 3.0, "label": "L2 leaf regularisation", "min": 0.0, "max": 30.0, "step": 0.1},
            "subsample": {"type": "float", "default": 0.8, "label": "Row subsample ratio", "min": 0.1, "max": 1.0, "step": 0.05},
            "colsample_bylevel": {"type": "float", "default": 0.8, "label": "Column subsample ratio", "min": 0.1, "max": 1.0, "step": 0.05},
            "min_data_in_leaf": {"type": "int", "default": 10, "label": "Min samples per leaf", "min": 1, "max": 100},
        },
        "gru": {
            "hidden_size": {"type": "int", "default": 32, "label": "Hidden size", "min": 8, "max": 512},
            "num_layers": {"type": "int", "default": 1, "label": "GRU layers", "min": 1, "max": 8},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        "nlinear": {
            "learning_rate": {"type": "float", "default": 5e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        "fits": {
            "cutoff_ratio": {"type": "float", "default": 0.25, "label": "Frequency cutoff ratio", "min": 0.05, "max": 0.95, "step": 0.05},
            "learning_rate": {"type": "float", "default": 5e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        "timemixer": {
            "n_scales": {"type": "int", "default": 3, "label": "Multiscale levels", "min": 1, "max": 6},
            "downsample": {"type": "int", "default": 2, "label": "Downsample factor", "min": 2, "max": 4},
            "hidden_mult": {"type": "int", "default": 2, "label": "Hidden multiplier", "min": 1, "max": 8},
            "decomp_kernel": {"type": "int", "default": 25, "label": "Decomposition kernel", "min": 3, "max": 101},
            "n_pdm_blocks": {"type": "int", "default": 1, "label": "PDM blocks", "min": 1, "max": 4},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 5e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        "tft": {
            "hidden_size": {"type": "int", "default": 32, "label": "Hidden size", "min": 8, "max": 256},
            "n_heads": {"type": "int", "default": 4, "label": "Attention heads", "min": 1, "max": 16},
            "n_lstm_layers": {"type": "int", "default": 1, "label": "LSTM encoder layers", "min": 1, "max": 4},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 5e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        "timexer": {
            "patch_len": {"type": "int", "default": 8, "label": "Patch length", "min": 2, "max": 48},
            "d_model": {"type": "int", "default": 32, "label": "Model dimension", "min": 8, "max": 256},
            "n_heads": {"type": "int", "default": 4, "label": "Attention heads", "min": 1, "max": 16},
            "n_encoder_layers": {"type": "int", "default": 1, "label": "Encoder layers", "min": 1, "max": 8},
            "dim_feedforward": {"type": "int", "default": 64, "label": "Feedforward dimension", "min": 16, "max": 512},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        "moderntcn": {
            "patch_len": {"type": "int", "default": 4, "label": "Patch length", "min": 1, "max": 24},
            "patch_stride": {"type": "int", "default": 2, "label": "Patch stride", "min": 1, "max": 24},
            "d_model": {"type": "int", "default": 16, "label": "Model dimension", "min": 4, "max": 128},
            "large_kernel": {"type": "int", "default": 13, "label": "Large DW kernel", "min": 5, "max": 51},
            "ffn_ratio": {"type": "int", "default": 2, "label": "FFN expansion ratio", "min": 1, "max": 8},
            "n_blocks": {"type": "int", "default": 1, "label": "Backbone blocks", "min": 1, "max": 4},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512, "tunable": False},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"], "tunable": False},
        },
        # `seasonal_period` is marked non-tunable on the classical backends:
        # it's a data-cadence property (set once based on sampling rate, e.g.
        # 48 for half-hourly daily, 168 for hourly weekly), not a hyperparameter
        # to search over. The auto-models (ARIMA/ETS/Theta) auto-select all
        # remaining hyperparameters internally — there is genuinely nothing for
        # Optuna to tune on these backends. Tuning is blocked at the API layer
        # for any model whose schema has zero tunable params, so users get a
        # clear error rather than an Optuna study spinning on no-op trials.
        # The same applies to the zero-shot foundation backends
        # (chronos_bolt / ttm): the pretrained weights are frozen and the
        # remaining knobs are checkpoint-selection properties, not search
        # dimensions.
        "seasonal_naive": {
            "seasonal_period": {"type": "int", "default": 48, "label": "Seasonal period (steps)", "min": 1, "max": 1440, "tunable": False},
        },
        "arima": {
            "seasonal_period": {"type": "int", "default": 48, "label": "Seasonal period (steps)", "min": 1, "max": 1440, "tunable": False},
            "train_history": {"type": "int", "default": 1024, "label": "Max train history", "min": 64, "max": 8192, "tunable": False},
        },
        "ets": {
            "seasonal_period": {"type": "int", "default": 48, "label": "Seasonal period (steps)", "min": 1, "max": 1440, "tunable": False},
            "train_history": {"type": "int", "default": 1024, "label": "Max train history", "min": 64, "max": 8192, "tunable": False},
        },
        "theta": {
            "seasonal_period": {"type": "int", "default": 48, "label": "Seasonal period (steps)", "min": 1, "max": 1440, "tunable": False},
            "train_history": {"type": "int", "default": 1024, "label": "Max train history", "min": 64, "max": 8192, "tunable": False},
        },
        "chronos_bolt": {
            "model_name": {"type": "select", "default": "amazon/chronos-bolt-tiny", "label": "Pretrained checkpoint",
                           "options": ["amazon/chronos-bolt-tiny", "amazon/chronos-bolt-mini", "amazon/chronos-bolt-small"], "tunable": False},
            "context_length": {"type": "int", "default": 512, "label": "Max context length", "min": 32, "max": 2048, "tunable": False},
            "train_history": {"type": "int", "default": 512, "label": "Cached history tail", "min": 64, "max": 8192, "tunable": False},
        },
        "ttm": {
            "model_path": {"type": "select", "default": "ibm-granite/granite-timeseries-ttm-r2", "label": "Pretrained checkpoint",
                           "options": ["ibm-granite/granite-timeseries-ttm-r2", "ibm-granite/granite-timeseries-ttm-r1"], "tunable": False},
            "context_length": {"type": "int", "default": 512, "label": "Context length", "min": 52, "max": 1536, "tunable": False},
            "train_history": {"type": "int", "default": 1024, "label": "Cached history tail", "min": 64, "max": 8192, "tunable": False},
        },
    }

    # ========== HTML Routes ==========

    def _build_dashboard_context(request: Request) -> dict:
        """Shared context for the full dashboard page and the HTMX fragment.

        Both code paths need the same view-state — extracting the build
        avoids drift between the two and keeps the page render + 10-second
        refresh in lock-step.
        """
        experiments = list(app.state.appstate.experiment_statuses.values())

        from ml_forecast_lab.training_events import TrainingEventBus, summarise_history
        training_summaries: Dict[str, Dict] = {}
        event_bus = TrainingEventBus.get_instance()
        for exp_name in app.state.appstate.running_benchmarks:
            history = event_bus.get_history(exp_name)
            if history:
                training_summaries[exp_name] = summarise_history(history)

        # v2.40.10: dashboard cards need ``production_model`` (the YAML
        # pinned value) to render the correct deployed-model label —
        # without it, the card falls back to ``best_model`` (the latest
        # leaderboard winner) and disagrees with what the inference
        # path actually runs. Same bug class as PR #66 for the
        # experiment page; this is the dashboard-template fix.
        production_model_by_exp: Dict[str, Optional[str]] = {}
        try:
            from ml_forecast_lab.config import load_config as _lc
            cfg_path = _find_config_path()
            if cfg_path and cfg_path.exists():
                cfg = _lc(cfg_path)
                for exp in cfg.experiments:
                    production_model_by_exp[exp.name] = exp.production_model
        except Exception as e:
            logger.debug(
                f"Could not load production_model map for dashboard: {e}"
            )

        return {
            "request": request,
            "base_path": _get_base_path(request),
            "active_page": "dashboard",
            "version": APP_VERSION,
            "experiments": experiments,
            "total_experiments": len(experiments),
            "lab_experiments": sum(1 for e in experiments if e.mode == "lab"),
            "production_experiments": sum(
                1 for e in experiments if e.mode == "production"
            ),
            "training_summaries": training_summaries,
            "running_experiments": app.state.appstate.running_benchmarks,
            "queued_experiments": {
                item["name"]: i + 1
                for i, item in enumerate(app.state.appstate.training_queue)
            },
            "production_model_by_exp": production_model_by_exp,
        }

    @app.get("/", response_class=Response)
    async def dashboard(request: Request):
        """Full dashboard page."""
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context=_build_dashboard_context(request),
        )

    @app.get("/api/dashboard/grid", response_class=Response)
    async def dashboard_grid_fragment(request: Request):
        """HTMX partial: just the experiments-grid <section>.

        Drives the dashboard's auto-refresh without a full page reload —
        keeps scroll position, expanded details and the New-experiment
        modal state intact. Polling cadence (10 s while a benchmark is
        running, 60 s otherwise) is encoded into the fragment's
        hx-trigger so it adapts after each swap.
        """
        return templates.TemplateResponse(
            request=request,
            name="_dashboard_grid.html",
            context=_build_dashboard_context(request),
        )

    # Model catalog (shared between Models page and experiment detail).
    # Sorted by display_name (case-insensitive) so cards render
    # alphabetically in both the /models tab and the per-experiment
    # Models tab, regardless of insertion order here.
    MODEL_CATALOG = sorted([
        {"name": "cnn", "display_name": "CNN", "model_type": "PyTorch",
         "description": "WaveNet-style dilated causal convolutions with residual connections.",
         "speed": "🔶 Moderate", "best_for": "Periodic/seasonal signals"},
        {"name": "crossformer", "display_name": "Crossformer", "model_type": "PyTorch",
         "description": "Segment embedding with temporal + cross-variable attention.",
         "speed": "🔶 Moderate", "best_for": "Joint temporal + cross-variate modelling"},
        {"name": "dlinear", "display_name": "DLinear", "model_type": "PyTorch",
         "description": "Decomposition-Linear: separate linear layers for trend and seasonal.",
         "speed": "⚡ Very Fast", "best_for": "Simple baseline — surprisingly competitive"},
        {"name": "itransformer", "display_name": "iTransformer", "model_type": "PyTorch",
         "description": "Inverted Transformer: attention across variables.",
         "speed": "🔶 Moderate", "best_for": "Cross-variate correlations"},
        {"name": "lightgbm", "display_name": "LightGBM", "model_type": "Tree",
         "description": "Gradient boosting framework optimised for speed and memory efficiency.",
         "speed": "⚡ Very Fast", "best_for": "Default choice — fast and accurate"},
        {"name": "lstm", "display_name": "LSTM", "model_type": "PyTorch",
         "description": "Multi-layer LSTM with temporal attention and multi-horizon output head.",
         "speed": "🔶 Moderate", "best_for": "Complex temporal patterns"},
        {"name": "nbeats", "display_name": "N-BEATS", "model_type": "PyTorch",
         "description": "Neural Basis Expansion with doubly-residual stacking.",
         "speed": "🔶 Moderate", "best_for": "Pure time-series without covariates"},
        {"name": "nhits", "display_name": "N-HiTS", "model_type": "PyTorch",
         "description": "Hierarchical interpolation with multi-rate temporal downsampling.",
         "speed": "🔶 Moderate", "best_for": "Multi-scale temporal patterns"},
        {"name": "patchtst", "display_name": "PatchTST", "model_type": "PyTorch",
         "description": "Channel-independent Patch Transformer with encoder.",
         "speed": "🔶 Moderate", "best_for": "Long-range dependencies"},
        {"name": "sparsetsf", "display_name": "SparseTSF", "model_type": "PyTorch",
         "description": "Period-based sparse cross-period linear model.",
         "speed": "⚡ Very Fast", "best_for": "Strong daily/weekly periodicity"},
        {"name": "tide", "display_name": "TiDE", "model_type": "PyTorch",
         "description": "Dense encoder-decoder with temporal decoder and global residual skip.",
         "speed": "🔶 Moderate", "best_for": "Efficient long-horizon forecasting"},
        {"name": "timesnet", "display_name": "TimesNet", "model_type": "PyTorch",
         "description": "FFT period detection with 2D inception convolutions.",
         "speed": "🐢 Slower", "best_for": "Multi-periodic signals"},
        {"name": "tsmixer", "display_name": "TSMixer", "model_type": "PyTorch",
         "description": "Alternating time-mixing and feature-mixing MLP layers.",
         "speed": "🔶 Moderate", "best_for": "Multivariate cross-channel patterns"},
        {"name": "xgboost", "display_name": "XGBoost", "model_type": "Tree",
         "description": "Extreme gradient boosting with L1/L2 regularisation.",
         "speed": "⚡ Fast", "best_for": "When LightGBM overfits"},
        {"name": "catboost", "display_name": "CatBoost", "model_type": "Tree",
         "description": "Ordered boosting with oblivious symmetric trees.",
         "speed": "⚡ Fast", "best_for": "Noisy or covariate-rich tabular data"},
        {"name": "gru", "display_name": "GRU", "model_type": "PyTorch",
         "description": "Gated Recurrent Unit with temporal attention — lighter LSTM.",
         "speed": "🔶 Moderate", "best_for": "Same as LSTM but with fewer parameters"},
        {"name": "nlinear", "display_name": "NLinear", "model_type": "PyTorch",
         "description": "Single linear layer with last-value subtraction (Zeng et al. 2023).",
         "speed": "⚡ Very Fast", "best_for": "Tiny baseline alongside DLinear"},
        {"name": "fits", "display_name": "FITS", "model_type": "PyTorch",
         "description": "Frequency-domain low-pass complex linear (~10k parameters).",
         "speed": "⚡ Very Fast", "best_for": "Lightest neural backend; periodic signals"},
        {"name": "timemixer", "display_name": "TimeMixer", "model_type": "PyTorch",
         "description": "Multiscale season/trend mixing with cross-scale interaction.",
         "speed": "🔶 Moderate", "best_for": "Mixed short-term + long-term patterns"},
        {"name": "tft", "display_name": "TFT", "model_type": "PyTorch",
         "description": "Temporal Fusion Transformer with variable selection.",
         "speed": "🐢 Slower", "best_for": "Interpretable forecasts with covariates"},
        {"name": "timexer", "display_name": "TimeXer", "model_type": "PyTorch",
         "description": "Endogenous patch tokens with cross-attention to exogenous variate tokens.",
         "speed": "🔶 Moderate", "best_for": "Covariate-driven targets (solar, heating)"},
        {"name": "moderntcn", "display_name": "ModernTCN", "model_type": "PyTorch",
         "description": "Modernised pure-convolution backbone with large-kernel depthwise temporal mixing.",
         "speed": "⚡ Fast", "best_for": "Transformer-class accuracy at convolution cost"},
        {"name": "chronos_bolt", "display_name": "Chronos-Bolt", "model_type": "Foundation",
         "description": "Amazon's pretrained zero-shot forecaster — no training on your data.",
         "speed": "⚡ Fast", "best_for": "Cold start with little history; strong zero-shot accuracy"},
        {"name": "ttm", "display_name": "Granite TTM", "model_type": "Foundation",
         "description": "IBM's Tiny Time Mixer — pretrained zero-shot forecaster in the 1-5M param range.",
         "speed": "⚡ Fast", "best_for": "Cold start on tight hardware budgets"},
        {"name": "seasonal_naive", "display_name": "Seasonal Naive", "model_type": "Baseline",
         "description": "Reference baseline: ŷ[t+h] = y[t+h-period]. No training.",
         "speed": "⚡ Instant", "best_for": "Sanity-check reference for all other models"},
        {"name": "arima", "display_name": "AutoARIMA", "model_type": "Classical",
         "description": "Auto seasonal-ARIMA via AIC search (statsforecast).",
         "speed": "🔶 Moderate", "best_for": "Classical statistical baseline"},
        {"name": "ets", "display_name": "AutoETS", "model_type": "Classical",
         "description": "Hyndman exponential smoothing state-space (statsforecast).",
         "speed": "⚡ Fast", "best_for": "Robust seasonal baseline"},
        {"name": "theta", "display_name": "AutoTheta", "model_type": "Classical",
         "description": "Theta-method decomposition (M3/M4 winner family).",
         "speed": "⚡ Fast", "best_for": "Strong seasonal univariate baseline"},
    ], key=lambda m: m["display_name"].lower())

    _MODEL_DISPLAY_NAMES = {m["name"]: m["display_name"] for m in MODEL_CATALOG}

    def _model_display(name):
        """Map a model identifier (e.g. 'cnn') to its display name ('CNN')."""
        if not name:
            return name
        return _MODEL_DISPLAY_NAMES.get(name, name)

    templates.env.filters["model_display"] = _model_display

    @app.get("/models", response_class=Response)
    async def models_page(request: Request):
        """Models configuration page with per-model hyperparameter editing."""
        import yaml

        models_list = MODEL_CATALOG

        # Load model overrides from config. (A models_enabled read —
        # taken from the FIRST experiment only — used to live here too,
        # but the template never referenced it; per-experiment toggles
        # live on the experiment page. Removed in v2.41.0, audit F17.)
        model_overrides = {}
        config_path = _find_config_path()
        if config_path:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f) or {}
                model_overrides = yaml_data.get("model_overrides", {})
            except Exception:
                pass

        return templates.TemplateResponse(
            request=request,
            name="models.html",
            context={
                "request": request,
                "base_path": _get_base_path(request),
                "active_page": "models",
                "version": APP_VERSION,
                "models": models_list,
                "model_overrides": model_overrides,
                "param_schema": MODEL_PARAM_SCHEMA,
            },
        )

    @app.get("/api/models/params")
    async def get_all_model_params():
        """Return parameter schema, defaults, and current overrides for all models."""
        import yaml
        model_overrides = {}
        config_path = _find_config_path()
        if config_path:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f) or {}
                model_overrides = yaml_data.get("model_overrides", {})
            except Exception:
                pass

        result = {}
        for model_name, schema in MODEL_PARAM_SCHEMA.items():
            defaults = {k: v["default"] for k, v in schema.items()}
            overrides = model_overrides.get(model_name, {})
            current = {**defaults, **overrides}
            result[model_name] = {
                "defaults": defaults,
                "overrides": overrides,
                "current": current,
                "schema": schema,
            }
        return JSONResponse(content=result)

    @app.post("/api/models/params")
    async def save_model_params(request: Request):
        """
        Save hyperparameter overrides for a model.
        Body: {"model_name": "lstm", "params": {"hidden_size": 128, "dropout": 0.3}}
        """
        import yaml
        from ml_forecast_lab.config import save_model_overrides

        try:
            data = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        model_name = data.get("model_name")
        params = data.get("params", {})

        if not model_name or model_name not in MODEL_PARAM_SCHEMA:
            return JSONResponse(content={"success": False, "error": f"Unknown model: {model_name}"})

        schema = MODEL_PARAM_SCHEMA[model_name]

        # Validate and cast param values
        validated = {}
        for k, v in params.items():
            if k not in schema:
                return JSONResponse(content={"success": False, "error": f"Unknown param '{k}' for {model_name}"})
            spec = schema[k]
            try:
                if spec["type"] == "int":
                    v = int(v)
                elif spec["type"] == "float":
                    v = float(v)
                elif spec["type"] == "bool":
                    v = bool(v)
                elif spec["type"] == "select":
                    v = str(v)
                    if "options" in spec and v not in spec["options"]:
                        return JSONResponse(content={"success": False, "error": f"Invalid value '{v}' for {k}"})
            except (ValueError, TypeError) as e:
                return JSONResponse(content={"success": False, "error": f"Invalid type for {k}: {e}"})
            validated[k] = v

        # Only store values that differ from defaults
        overrides = {k: v for k, v in validated.items() if v != schema[k]["default"]}

        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        try:
            save_model_overrides(config_path, model_name, overrides)
            defaults = {k: v["default"] for k, v in schema.items()}
            current = {**defaults, **overrides}
            logger.info(f"Saved {len(overrides)} override(s) for {model_name}")
            return JSONResponse(content={"success": True, "overrides": overrides, "current": current})
        except Exception as e:
            logger.error(f"Failed to save model params: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

    @app.post("/api/models/params/reset")
    async def reset_model_params(request: Request):
        """Reset a model's params to defaults by removing its overrides."""
        from ml_forecast_lab.config import save_model_overrides

        try:
            data = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        model_name = data.get("model_name")
        if not model_name or model_name not in MODEL_PARAM_SCHEMA:
            return JSONResponse(content={"success": False, "error": f"Unknown model: {model_name}"})

        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        try:
            save_model_overrides(config_path, model_name, None)
            defaults = {k: v["default"] for k, v in MODEL_PARAM_SCHEMA[model_name].items()}
            logger.info(f"Reset {model_name} to defaults")
            return JSONResponse(content={"success": True, "defaults": defaults})
        except Exception as e:
            logger.error(f"Failed to reset model params: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

    @app.get("/experiment/{name}", response_class=Response)
    async def experiment_detail(request: Request, name: str):
        """
        Experiment detail page with model comparison, forecast charts, and metrics.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        exp_status = app.state.appstate.experiment_statuses[name]
        benchmark_result = app.state.appstate.benchmark_results.get(name)
        lab_forecast = app.state.appstate.lab_forecast_data.get(name)
        feature_imps = app.state.appstate.feature_importances.get(name, [])
        covariate_analysis = app.state.appstate.covariate_analysis_results.get(name)
        is_running = app.state.appstate.is_benchmark_running(name)

        # Embed training event history so the page can restore live
        # progress without a separate fetch (same pattern as training_page).
        from ml_forecast_lab.training_events import TrainingEventBus
        embedded_history: Dict[str, list] = {}
        event_bus = TrainingEventBus.get_instance()
        exp_history = event_bus.get_history(name)
        if exp_history:
            embedded_history[name] = [ev.to_dict() for ev in exp_history]

        # Get units, per-experiment models_enabled, and full config from config
        units = ""
        exp_models_enabled: list = []
        exp_config = None
        try:
            from ml_forecast_lab.config import load_config as _lc
            cfg_path = _find_config_path()
            if cfg_path and cfg_path.exists():
                cfg = _lc(cfg_path)
                for exp in cfg.experiments:
                    if exp.name == name:
                        units = exp.units or ""
                        exp_models_enabled = list(exp.models_enabled)
                        exp_config = exp
        except Exception:
            pass

        return templates.TemplateResponse(
            request=request,
            name="experiment.html",
            context={
                "request": request,
                "base_path": _get_base_path(request),
                "active_page": "dashboard",
                "version": APP_VERSION,
                "experiment": exp_status,
                "benchmark_result": benchmark_result,
                "lab_forecast": lab_forecast,
                "feature_importances": feature_imps,
                "is_running": is_running,
                "models": benchmark_result.models if benchmark_result else [],
                "best_model": benchmark_result.best_model_name
                if benchmark_result
                else None,
                # The model the inference path actually runs (mlfl.yaml
                # ``production_model``). Templates fall back through this
                # before ``best_model`` so the UI matches what's deployed
                # — see main.py:3475 for the inference resolution order.
                "production_model": exp_config.production_model
                if exp_config
                else None,
                "covariate_analysis": covariate_analysis,
                "units": units,
                "models_json": [m.model_dump() for m in (benchmark_result.models if benchmark_result else [])],
                "embedded_history": embedded_history,
                "model_catalog": MODEL_CATALOG,
                "exp_models_enabled": exp_models_enabled,
                "exp_config": exp_config,
                "ha_time_zone": app.state.appstate.ha_time_zone,
                "tuning_result": app.state.appstate.tuning_results.get(name),
                "param_defaults": {m: {p: s["default"] for p, s in schema.items()}
                                   for m, schema in MODEL_PARAM_SCHEMA.items()},
            },
        )

    # ========== API Routes ==========

    @app.post("/experiment/{name}/run-benchmark")
    async def run_benchmark(name: str):
        """
        Trigger a benchmark run for an experiment (async, returns 202 Accepted).
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        if app.state.appstate.is_benchmark_running(name):
            return JSONResponse(
                status_code=409,
                content={"error": "Benchmark already running for this experiment"},
            )

        app.state.appstate.start_benchmark(name)

        # Without this dispatch the experiment is marked "running" but no
        # work is queued — start_benchmark() only flips a flag. The matching
        # end_benchmark() lives inside benchmark_callback's finally block,
        # so if we don't call the callback the flag is never cleared and
        # every subsequent run-benchmark / run-pipeline / retrain rejects
        # with 409 until the add-on restarts.
        if app.state.appstate.benchmark_callback:
            try:
                app.state.appstate.benchmark_callback(name)
            except Exception:
                # Clear the flag if dispatch itself raises so the experiment
                # isn't permanently jammed; the callback's own error path
                # handles clear-on-completion failures.
                app.state.appstate.end_benchmark(name)
                raise
        else:
            # No callback wired (test / stub harness) — clear the flag so
            # the response below isn't a lie.
            app.state.appstate.end_benchmark(name)

        return JSONResponse(
            status_code=202,
            content={
                "message": "Benchmark run accepted",
                "experiment": name,
                "status": "queued",
            },
        )

    @app.post("/api/benchmarks/run-all")
    async def run_all_benchmarks():
        """Trigger benchmark runs for all experiments."""
        queued = []
        skipped = []
        for name, status in app.state.appstate.experiment_statuses.items():
            if app.state.appstate.is_benchmark_running(name):
                skipped.append(name)
            else:
                app.state.appstate.start_benchmark(name)
                if app.state.appstate.benchmark_callback:
                    try:
                        app.state.appstate.benchmark_callback(name)
                    except Exception:
                        pass
                queued.append(name)
        return JSONResponse(
            status_code=202,
            content={
                "message": f"Queued {len(queued)} benchmark(s)",
                "queued": queued,
                "skipped": skipped,
            },
        )

    @app.post("/experiment/{name}/promote/{model_name}")
    async def promote_model(name: str, model_name: str):
        """
        Promote a model to production.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        exp_status = app.state.appstate.experiment_statuses[name]
        benchmark_result = app.state.appstate.benchmark_results.get(name)

        if not benchmark_result:
            raise HTTPException(
                status_code=400, detail="No benchmark results available"
            )

        # Verify model exists in results
        if not any(m.name == model_name for m in benchmark_result.models):
            raise HTTPException(status_code=404, detail="Model not found in results")

        # Track whether the champion name actually changed. Re-promoting
        # the same name (idempotent click) shouldn't wipe forecast_log
        # since there's no new model to disambiguate from.
        previous_model = exp_status.best_model
        champion_changed = (previous_model != model_name)

        # Update in-memory status
        exp_status.best_model = model_name
        exp_status.selected_model = model_name
        exp_status.mode = "production"

        if benchmark_result:
            for model in benchmark_result.models:
                model.is_production = model.name == model_name

        # Persist to YAML so promotion survives restarts. Also write
        # selected_model so the Results-tab UI highlights match after
        # a restart (promote implicitly aligns the selection with the
        # new champion).
        from ml_forecast_lab.config import load_config, save_experiment_field
        config_path = _find_config_path()
        if config_path:
            try:
                save_experiment_field(config_path, name, "production_model", model_name)
                save_experiment_field(config_path, name, "selected_model", model_name)
                save_experiment_field(config_path, name, "mode", "production")
            except Exception as e:
                logger.warning(f"Failed to persist promotion to YAML: {e}")

        # Prune forecast_log of rows issued under the PREVIOUS champion.
        # Only fires when the name actually changes — same-name idempotent
        # promotes don't touch history. The new champion's own history (if
        # any, from a prior demote → re-promote cycle) is preserved via the
        # exclude_model_name filter so conformal calibration and analytics
        # don't reset on every champion switch. Controlled by
        # ExperimentCfg.clear_forecast_log_on_retrain.
        deleted = 0
        try:
            db = app.state.appstate.history_db
            if db and config_path and champion_changed:
                cfg = load_config(config_path)
                exp_cfg = next((e for e in cfg.experiments if e.name == name), None)
                if exp_cfg and getattr(exp_cfg, "clear_forecast_log_on_retrain", True):
                    from datetime import datetime
                    deleted = db.cleanup_forecast_log(
                        name,
                        datetime.utcnow(),
                        exclude_model_name=model_name,
                    )
                    if deleted:
                        logger.info(
                            f"Promotion {previous_model!r} → {model_name!r} "
                            f"for {name}: cleared {deleted} pre-promotion "
                            f"forecast_log rows (preserving {model_name!r} history)"
                        )
        except Exception as e:
            logger.warning(f"Forecast-log cleanup on promote failed: {e}")

        # Fire the retrain callback so the promoted model is trained, cached,
        # and its sensors start publishing on the same cycle — mirroring what
        # /toggle-mode does when it flips lab → production. Without this the
        # experiment sat in production with no cached model until the next
        # scheduled retrain tick (up to retrain_every_hours later), which
        # looked like "the Publish button didn't do anything".
        if app.state.appstate.retrain_callback:
            import asyncio as _aio
            app.state.appstate.spawn(app.state.appstate.retrain_callback(name))
            logger.info(f"Triggered immediate retrain for {name} after promotion")

        return JSONResponse(
            content={
                "message": f"Model {model_name} promoted to production",
                "experiment": name,
                "model": model_name,
                "forecast_log_rows_cleared": deleted,
            }
        )

    @app.post("/experiment/{name}/select-model")
    async def select_model(name: str, request: Request):
        """Set the user's selected model for this experiment."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        model_name = body.get("model_name")
        if not model_name:
            return JSONResponse(content={"success": False, "error": "model_name required"})

        # Validate model exists in benchmark results
        benchmark_result = app.state.appstate.benchmark_results.get(name)
        if benchmark_result and not any(m.name == model_name for m in benchmark_result.models):
            return JSONResponse(content={"success": False, "error": "Model not in results"})

        app.state.appstate.experiment_statuses[name].selected_model = model_name
        # Persist to YAML so the selection survives add-on restarts.
        # Without this, `status.selected_model` resets to None at startup
        # and the next benchmark auto-promotes its top-ranked model —
        # which users experience as "I chose X but the page forgets".
        persisted = False
        try:
            from ml_forecast_lab.config import save_experiment_field
            config_path = _find_config_path()
            if config_path:
                save_experiment_field(config_path, name, "selected_model", model_name)
                persisted = True
        except Exception as e:
            logger.warning(f"Failed to persist selected_model to YAML: {e}")
        return JSONResponse(content={
            "success": True,
            "selected_model": model_name,
            "persisted": persisted,
        })

    @app.post("/experiment/{name}/apply-covariate-best")
    async def apply_covariate_best(name: str):
        """
        Apply the winning covariate configuration from the latest deep
        analysis run, then trigger a background retrain.

        Determines which covariate set has the lowest average MAE across
        all tested models. The winner is one of:
            * "All covariates" — no changes needed
            * "No covariates"  — all covariates removed
            * "Without X"      — single covariate X removed

        Returns a JSON payload describing the action taken.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        result = app.state.appstate.covariate_analysis_results.get(name)
        if not result or result.status != "completed" or not result.results:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "No completed covariate analysis results"},
            )

        # Score every covariate label against the PRODUCTION model. The
        # previous behaviour (mean MAE across every tested model) could
        # pick a config that helps a weak model but hurts the model
        # actually publishing forecasts. When the production model
        # wasn't in the run (e.g. user picked "All models" but the
        # champion was disabled when the analysis ran), fall back to
        # the cross-model mean so the action still resolves.
        exp_status_for_score = app.state.appstate.experiment_statuses.get(name)
        production_model = (
            getattr(exp_status_for_score, "selected_model", None)
            or getattr(exp_status_for_score, "best_model", None)
            if exp_status_for_score else None
        )

        def _is_nan(v):
            return isinstance(v, float) and v != v

        score_source = "mean"  # finalised below
        label_scores: Dict[str, float] = {}
        if production_model and any(
            production_model in result.results.get(lbl, {})
            for lbl in result.covariate_labels
        ):
            score_source = "production_model"
            for label in result.covariate_labels:
                cells = result.results.get(label, {})
                cell = cells.get(production_model)
                if cell is not None and cell.mae is not None and not _is_nan(cell.mae):
                    label_scores[label] = cell.mae
            # If the production model has NaN for every label, fall back
            # to the cross-model mean so the action still resolves.
            if not label_scores:
                score_source = "mean"

        if score_source == "mean":
            for label in result.covariate_labels:
                cells = result.results.get(label, {})
                maes = [
                    cell.mae for cell in cells.values()
                    if cell is not None and cell.mae is not None and not _is_nan(cell.mae)
                ]
                if maes:
                    label_scores[label] = sum(maes) / len(maes)

        if not label_scores:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "No valid metrics in covariate analysis results"},
            )

        best_label = min(label_scores, key=label_scores.get)
        best_score = label_scores[best_label]
        baseline_score = label_scores.get(result.baseline_label, best_score)

        from ml_forecast_lab.config import (
            remove_experiment_covariate,
            clear_experiment_covariates,
        )
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(
                content={"success": False, "error": "Config file not found"},
            )

        action = "no_change"
        removed_covariates: List[str] = []
        try:
            if best_label == result.baseline_label:
                # All covariates already optimal
                action = "no_change"
            elif best_label == "No covariates":
                n_removed = clear_experiment_covariates(config_path, name)
                action = "cleared"
                removed_covariates = [f"{n_removed} covariate(s)"]
            elif best_label.startswith("Without "):
                short_name = best_label[len("Without "):]
                if remove_experiment_covariate(config_path, name, short_name):
                    action = "removed"
                    removed_covariates = [short_name]
                else:
                    return JSONResponse(content={
                        "success": False,
                        "error": f"Could not find covariate '{short_name}' to remove",
                    })
            else:
                return JSONResponse(content={
                    "success": False,
                    "error": f"Unsupported best label: {best_label}",
                })

            # Trigger a background retrain so the user sees the new
            # configuration take effect on the live forecast sensor without
            # waiting for the next scheduled retrain cycle. Skipped when
            # action == "no_change" — nothing changed, no retrain needed.
            retrain_scheduled = False
            if action != "no_change" and app.state.appstate.retrain_callback:
                import asyncio as _aio
                app.state.appstate.spawn(app.state.appstate.retrain_callback(name))
                retrain_scheduled = True
                logger.info(
                    f"Scheduled immediate retrain for {name} after apply-covariate-best"
                )

            improvement_pct = None
            if baseline_score and baseline_score > 0:
                improvement_pct = (baseline_score - best_score) / baseline_score * 100

            logger.info(
                f"apply-covariate-best ({name}): winner='{best_label}' "
                f"action={action} removed={removed_covariates} "
                f"improvement={improvement_pct:+.1f}% retraining={retrain_scheduled}"
                if improvement_pct is not None else
                f"apply-covariate-best ({name}): winner='{best_label}' "
                f"action={action} retraining={retrain_scheduled}"
            )

            return JSONResponse(content={
                "success": True,
                "action": action,
                "best_label": best_label,
                "best_mae": round(best_score, 6),
                "baseline_mae": round(baseline_score, 6),
                "improvement_pct": round(improvement_pct, 2) if improvement_pct is not None else None,
                "removed": removed_covariates,
                "retraining": retrain_scheduled,
                "score_source": score_source,
                "production_model": production_model if score_source == "production_model" else None,
            })
        except Exception as e:
            logger.error(f"Failed to apply covariate-best: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

    @app.post("/experiment/{name}/remove-covariate")
    async def remove_covariate(name: str, request: Request):
        """Remove a covariate from experiment config."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        entity_id = body.get("entity_id")
        if not entity_id:
            return JSONResponse(content={"success": False, "error": "entity_id required"})
        # v2.40.14: accept disambiguators so multi-row entities can be
        # removed individually. ``remove_experiment_covariate`` refuses
        # without them when the entity is configured > 1 time.
        role = body.get("role") or None
        future_attribute = body.get("future_attribute") or None
        future_value_key = body.get("future_value_key") or None

        from ml_forecast_lab.config import remove_experiment_covariate
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        try:
            removed = remove_experiment_covariate(
                config_path, name, entity_id,
                role=role,
                future_attribute=future_attribute,
                future_value_key=future_value_key,
            )
            if removed:
                logger.info(f"Removed covariate {entity_id} from {name}")
                return JSONResponse(content={"success": True, "entity_id": entity_id})
            else:
                return JSONResponse(content={"success": False, "error": "Covariate not found"})
        except Exception as e:
            logger.error(f"Failed to remove covariate: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

    @app.post("/experiment/{name}/add-covariate")
    async def add_covariate(name: str, request: Request):
        """Add a covariate to an experiment's config."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        entity = body.get("entity")
        if not entity:
            return JSONResponse(content={"success": False, "error": "entity is required"})

        cov_dict = {"entity": entity}
        for opt_field in (
            "role", "aggregation", "scale", "is_binary",
            "future_attribute", "future_value_key",
        ):
            if opt_field in body and body[opt_field] is not None:
                cov_dict[opt_field] = body[opt_field]

        from ml_forecast_lab.config import add_experiment_covariate
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        try:
            added = add_experiment_covariate(config_path, name, cov_dict)
            if added:
                logger.info(f"Added covariate {entity} to {name}")
                return JSONResponse(content={"success": True, "entity": entity})
            else:
                return JSONResponse(content={"success": False, "error": "Covariate already exists or experiment not found"})
        except ValueError as e:
            return JSONResponse(content={"success": False, "error": _safe_error(e)})
        except Exception as e:
            logger.error(f"Failed to add covariate: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

    @app.post("/experiment/{name}/add-load-subtract")
    async def add_load_subtract(name: str, request: Request):
        """Add a load-subtract sensor to an experiment's config.

        Body: {entity_id (required), source?, on_missing?, scale?,
               max_fraction_of_load?, max_fraction_violation_pct?}.

        Validation is delegated to ``SubtractCfg.__post_init__`` inside
        ``add_experiment_load_subtract`` — a 200 with ``success=false`` is
        returned on duplicate/invalid rather than raising, so the UI can
        surface the error in a toast.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        entity_id = body.get("entity_id")
        if not entity_id:
            return JSONResponse(
                content={"success": False, "error": "entity_id is required"}
            )

        sub_dict = {"entity_id": entity_id}
        for opt_field in (
            "source", "on_missing", "scale",
            "max_fraction_of_load", "max_fraction_violation_pct",
        ):
            if opt_field in body and body[opt_field] is not None:
                sub_dict[opt_field] = body[opt_field]

        from ml_forecast_lab.config import add_experiment_load_subtract
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(
                content={"success": False, "error": "Config file not found"}
            )

        try:
            added = add_experiment_load_subtract(config_path, name, sub_dict)
            if added:
                logger.info(f"Added load_subtract {entity_id} to {name}")
                return JSONResponse(
                    content={"success": True, "entity_id": entity_id}
                )
            return JSONResponse(content={
                "success": False,
                "error": "load_subtract already exists or experiment not found",
            })
        except ValueError as e:
            # SubtractCfg validation failure — message is user-actionable.
            return JSONResponse(content={"success": False, "error": _safe_error(e)})
        except Exception as e:
            logger.error(f"Failed to add load_subtract: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

    @app.post("/experiment/{name}/remove-load-subtract")
    async def remove_load_subtract(name: str, request: Request):
        """Remove a single load-subtract entry from an experiment's config.

        Body: {entity_id}. Accepts either the full ID (``sensor.ev_today``)
        or the short suffix (``ev_today``) — matches ``remove_experiment_
        load_subtract``'s suffix-matching behaviour.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        entity_id = body.get("entity_id")
        if not entity_id:
            return JSONResponse(
                content={"success": False, "error": "entity_id required"}
            )

        from ml_forecast_lab.config import remove_experiment_load_subtract
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(
                content={"success": False, "error": "Config file not found"}
            )

        try:
            removed = remove_experiment_load_subtract(
                config_path, name, entity_id,
            )
            if removed:
                logger.info(f"Removed load_subtract {entity_id} from {name}")
                return JSONResponse(
                    content={"success": True, "entity_id": entity_id}
                )
            return JSONResponse(content={
                "success": False, "error": "load_subtract entry not found",
            })
        except Exception as e:
            logger.error(f"Failed to remove load_subtract: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

    @app.post("/experiment/{name}/clear-load-subtract")
    async def clear_load_subtract(name: str):
        """Remove all load-subtract entries from an experiment's config.

        Returns the count of entries that were removed so the UI can show
        a meaningful toast."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        from ml_forecast_lab.config import clear_experiment_load_subtract
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(
                content={"success": False, "error": "Config file not found"}
            )

        try:
            n_removed = clear_experiment_load_subtract(config_path, name)
            logger.info(
                f"Cleared {n_removed} load_subtract entrie(s) from {name}"
            )
            return JSONResponse(
                content={"success": True, "removed": n_removed}
            )
        except Exception as e:
            logger.error(f"Failed to clear load_subtract: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

    @app.post("/experiment/{name}/stop-training")
    async def stop_training(name: str):
        """Stop a running training/tuning task, or remove from queue."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        # Check if experiment is queued (not yet started)
        if app.state.appstate.remove_from_queue(name):
            logger.info(f"Removed {name} from training queue")
            return JSONResponse(content={"success": True, "was_queued": True})

        cb = app.state.appstate.stop_training_callback
        if not cb:
            return JSONResponse(content={"success": False, "error": "Stop not available"})

        try:
            stopped = await cb(name)
            if stopped:
                logger.info(f"Stopped training for {name}")
                return JSONResponse(content={"success": True})
            else:
                return JSONResponse(content={"success": False, "error": "No running task for this experiment"})
        except Exception as e:
            logger.error(f"Failed to stop training for {name}: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

    @app.post("/experiment/{name}/retrain")
    async def retrain_experiment(name: str):
        """
        Trigger an immediate retrain of an experiment's production model.

        Used from the dashboard after changing settings (e.g. solar
        toggles, covariates) so the cached model picks up the new
        feature schema without waiting for the next scheduled retrain.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        if app.state.appstate.is_benchmark_running(name):
            return JSONResponse(content={
                "success": False, "error": "Training already running"
            })

        cb = app.state.appstate.retrain_callback
        if not cb:
            return JSONResponse(content={
                "success": False, "error": "Retrain not available"
            })

        app.state.appstate.spawn(cb(name))
        logger.info(f"User-triggered retrain for {name}")
        return JSONResponse(content={"success": True})

    @app.post("/experiment/{name}/rollback")
    async def rollback_experiment(name: str):
        """Swap the cached production model back to the previous champion.

        The previous-generation weights are archived inside
        ``previous/`` by ``_persist_cached_model`` on every successful
        retrain, so this swap is symmetric: calling it twice toggles
        between two generations.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")
        if app.state.appstate.is_benchmark_running(name):
            return JSONResponse(content={
                "success": False, "error": "Training is in progress; wait for it to finish before rolling back",
            })
        cb = app.state.appstate.rollback_callback
        if not cb:
            return JSONResponse(content={
                "success": False, "error": "Rollback unavailable",
            })
        try:
            ok, msg = cb(name)
        except Exception as e:
            logger.error("Rollback failed for %s: %s", name, e, exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})
        if not ok:
            return JSONResponse(content={"success": False, "error": msg or "Rollback failed"})
        return JSONResponse(content={"success": True, "message": msg})

    @app.post("/experiment/{name}/data-report")
    async def data_report(name: str):
        """Run the pre-flight data sanity report for an experiment.

        Fetches the raw target history (cache + HA delta) and computes
        coverage, gap, freshness and value-distribution stats so users
        can spot data issues before they spend an hour on a benchmark.
        Synchronous from the caller's perspective; takes a few seconds.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")
        cb = app.state.appstate.data_report_callback
        if not cb:
            return JSONResponse(content={"verdict": "alert", "warnings": ["Data report unavailable"], "ok": False})
        try:
            report = await cb(name)
            return JSONResponse(content=report)
        except Exception as e:
            logger.error("data-report failed for %s: %s", name, e, exc_info=True)
            return JSONResponse(content={"verdict": "alert", "warnings": [_safe_error(e)], "ok": False}, status_code=500)

    @app.get("/experiment/{name}/rollback-available")
    async def rollback_available(name: str):
        """Check whether a `previous/` snapshot exists on disk."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")
        getter = app.state.appstate.cached_model_dir
        if not getter:
            return JSONResponse(content={"available": False})
        try:
            model_dir = getter(name)
            prev = Path(model_dir) / "previous"
            available = (prev / "model.bin").exists() and (prev / "cache_meta.json").exists()
            payload: Dict[str, Any] = {"available": bool(available)}
            if available:
                try:
                    meta = json.loads((prev / "cache_meta.json").read_text())
                    payload["previous_model"] = meta.get("model_name")
                    payload["previous_trained_at"] = meta.get("trained_at")
                except Exception:
                    pass
            return JSONResponse(content=payload)
        except Exception as e:
            return JSONResponse(content={"available": False, "error": _safe_error(e)})

    @app.post("/api/experiments/create")
    async def create_experiment_route(request: Request):
        """Create a new experiment."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        name = body.get("name", "")
        target_entity = body.get("target_entity", "")

        if not name or not target_entity:
            return JSONResponse(content={"success": False, "error": "name and target_entity are required"})

        exp_dict = {"name": name, "target_entity": target_entity}
        for opt in ("source_is_cumulative", "reset_daily", "target_is_nonnegative"):
            if opt in body:
                exp_dict[opt] = bool(body[opt])

        from ml_forecast_lab.config import create_experiment
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        try:
            create_experiment(config_path, exp_dict)
        except ValueError as e:
            return JSONResponse(content={"success": False, "error": _safe_error(e)})
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

        # Register in-memory so it appears immediately
        from ml_forecast_lab.web.app import ExperimentStatus
        app.state.appstate.experiment_statuses[name] = ExperimentStatus(
            name=name,
            target_entity=target_entity,
            mode="lab",
        )

        logger.info(f"Created experiment '{name}' targeting {target_entity}")
        return JSONResponse(content={"success": True, "redirect": f"/experiment/{name}"})

    @app.post("/api/experiments/{name}/delete")
    async def delete_experiment_route(name: str):
        """Delete an experiment."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        from ml_forecast_lab.config import delete_experiment
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        try:
            removed = delete_experiment(config_path, name)
            if not removed:
                return JSONResponse(content={"success": False, "error": "Experiment not found in config"})
        except Exception as e:
            logger.error(f"Failed to delete experiment: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

        # Remove from in-memory state
        app.state.appstate.experiment_statuses.pop(name, None)
        app.state.appstate.benchmark_results.pop(name, None)
        app.state.appstate.forecast_data.pop(name, None)
        app.state.appstate.lab_forecast_data.pop(name, None)
        app.state.appstate.feature_importances.pop(name, None)
        app.state.appstate.covariate_analysis_results.pop(name, None)
        app.state.appstate.tuning_results.pop(name, None)

        # Clean up persistent data
        db = app.state.appstate.history_db
        if db:
            try:
                db.delete_forecast_log(name)
                db.delete_external_forecast_log(name)
                db.delete_benchmark_result(name)
            except Exception:
                pass

        logger.info(f"Deleted experiment '{name}'")
        return JSONResponse(content={"success": True, "redirect": "/"})

    @app.post("/experiment/{name}/replace-target")
    async def replace_target_route(name: str, request: Request):
        """Replace an experiment's target sensor (``target_entity``).

        Body: ``{"target_entity": "sensor.new_signal"}``.

        Changing the target invalidates everything derived from the old
        sensor: the trained/cached model, logged forecasts, benchmark scores
        and the in-memory analysis caches were all computed against a
        different series. We rewrite the YAML, then clear that stale state so
        the UI and the next training cycle start clean rather than showing
        metrics that silently belong to the previous sensor. The experiment
        keeps its name, covariates and all other settings.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        new_target = (body.get("target_entity") or "").strip()
        if not new_target:
            return JSONResponse(
                content={"success": False, "error": "target_entity is required"}
            )

        from ml_forecast_lab.config import replace_experiment_target
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(
                content={"success": False, "error": "Config file not found"}
            )

        try:
            previous = replace_experiment_target(config_path, name, new_target)
        except ValueError as e:
            return JSONResponse(content={"success": False, "error": _safe_error(e)})
        except Exception as e:
            logger.error(f"Failed to replace target: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

        if previous is None:
            return JSONResponse(content={
                "success": False,
                "error": f"Experiment '{name}' not found in config",
            })

        exp_status = app.state.appstate.experiment_statuses[name]

        # No-op: same sensor. Don't wipe history for a non-change.
        if previous == new_target:
            return JSONResponse(content={
                "success": True, "changed": False,
                "target_entity": new_target,
            })

        # Update the in-memory status and drop the now-stale training markers
        # (they describe a model trained on the old sensor).
        exp_status.target_entity = new_target
        exp_status.best_model = None
        exp_status.model_version = None
        exp_status.last_benchmark_status = "pending"
        exp_status.last_benchmark_timestamp = None
        exp_status.last_error = None

        # Clear in-memory analysis caches keyed by experiment name — every one
        # was computed against the previous target. Mirrors the delete route.
        st = app.state.appstate
        st.benchmark_results.pop(name, None)
        st.forecast_data.pop(name, None)
        st.lab_forecast_data.pop(name, None)
        st.feature_importances.pop(name, None)
        st.covariate_analysis_results.pop(name, None)
        st.tuning_results.pop(name, None)
        st.tune_all_results.pop(name, None)

        # Clear persisted, target-derived data. forecast_log /
        # external_forecast_log / benchmark_results are keyed by experiment
        # name; the old per-target actuals cache table is left in place
        # (harmless and age-pruned) since other experiments may share it.
        db = st.history_db
        if db:
            try:
                db.delete_forecast_log(name)
                db.delete_external_forecast_log(name)
                db.delete_benchmark_result(name)
            except Exception as e:
                logger.warning(f"Failed to clear logs on target replace: {e}")

        # Drop the trained model (in-memory + on-disk). Without this a forecast
        # cycle could publish predictions for the new sensor using a model
        # trained on the old one, since the cache is keyed by experiment name.
        reset_cb = getattr(st, "reset_model_callback", None)
        if reset_cb:
            try:
                reset_cb(name)
            except Exception as e:
                logger.warning(f"Failed to reset cached model on target replace: {e}")

        logger.info(
            f"Replaced target for '{name}': {previous} -> {new_target}"
        )

        # In production, kick off an immediate retrain against the new sensor
        # so forecasts resume without waiting for the next scheduled cycle
        # (mirrors the production-toggle behaviour).
        if exp_status.mode == "production" and st.retrain_callback:
            st.spawn(st.retrain_callback(name))
            logger.info(f"Triggered retrain for {name} after target replace")

        return JSONResponse(content={
            "success": True, "changed": True,
            "previous_target": previous,
            "target_entity": new_target,
        })

    async def _probe_weather_forecast_keys(
        ha_url: str, ha_token: str, entity_id: str, forecast_type: str,
        val_keys_priority: set,
    ) -> Optional[List[str]]:
        """One-shot call to ``weather.get_forecasts?return_response`` for
        ``entity_id`` with the requested ``type`` (hourly/daily/
        twice_daily) so the UI can offer the entity's actual numeric
        forecast keys as ``future_value_key`` options.

        Returns the ordered list of numeric keys (common ones from
        VAL_KEYS first, then any integration-specific extras), or
        ``None`` on service failure / empty response (caller surfaces
        the forecast type anyway with an empty key list — Auto still
        works at resolve time).
        """
        import aiohttp as _aiohttp
        try:
            async with _aiohttp.ClientSession() as sess:
                headers = {"Authorization": f"Bearer {ha_token}"}
                async with sess.post(
                    f"{ha_url}/api/services/weather/get_forecasts?return_response",
                    headers=headers,
                    json={"entity_id": entity_id, "type": forecast_type},
                    timeout=_aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        return None
                    body = await resp.json()
        except Exception as e:
            logger.debug(
                f"weather.get_forecasts probe failed for "
                f"{entity_id} type={forecast_type}: {e}"
            )
            return None
        service_resp = (body or {}).get("service_response") or {}
        entity_block = service_resp.get(entity_id) or {}
        forecast_list = entity_block.get("forecast") or []
        if not forecast_list or not isinstance(forecast_list[0], dict):
            return None
        first = forecast_list[0]
        # Exclude datetime / condition / categorical fields; keep only
        # numerics so the resolver has something to interpolate.
        numeric_keys = [
            k for k, v in first.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        common = [k for k in numeric_keys if k in val_keys_priority]
        other = [k for k in numeric_keys if k not in val_keys_priority]
        return common + other

    # ---- HA entity search (cached) ----
    _entity_cache: Dict[str, Any] = {"data": [], "ts": 0.0}

    @app.get("/api/ha/entities")
    async def ha_entities(request: Request):
        """Search HA entities for the covariate / target entity picker."""
        import time
        import aiohttp as _aiohttp

        q = (request.query_params.get("q") or "").lower().strip()
        now = time.time()

        # Refresh cache every 60 seconds
        if now - _entity_cache["ts"] > 60:
            ha_url = os.environ.get("HA_URL", "http://supervisor/core")
            ha_token = os.environ.get("SUPERVISOR_TOKEN", "")
            try:
                async with _aiohttp.ClientSession() as sess:
                    headers = {"Authorization": f"Bearer {ha_token}"}
                    async with sess.get(f"{ha_url}/api/states", headers=headers, timeout=_aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            states = await resp.json()
                            _entity_cache["data"] = [
                                {
                                    "entity_id": s["entity_id"],
                                    "friendly_name": s.get("attributes", {}).get("friendly_name", ""),
                                    "state": str(s.get("state", "")),
                                }
                                for s in states
                                if isinstance(s, dict) and "entity_id" in s
                            ]
                            _entity_cache["ts"] = now
            except Exception as e:
                logger.debug(f"Entity cache refresh failed: {e}")
                # Use stale cache or empty list

        entities = _entity_cache["data"]
        if q:
            entities = [
                e for e in entities
                if q in e["entity_id"].lower() or q in e["friendly_name"].lower()
            ]

        return JSONResponse(content=entities[:50])

    @app.get("/api/ha/forecast-attrs")
    async def ha_forecast_attrs(request: Request):
        """Inspect an HA entity's attributes and return those that look
        like a forecast array — used by the covariate UI to populate
        the ``future_attribute`` dropdown when role is future / both.

        An attribute "looks like a forecast" if:
        - It's a ``list[dict]`` where each dict has at least one
          recognisable datetime key (``datetime``, ``period_start``,
          ``time``, ``dt``, ``start``) and at least one numeric value
          key (``value``, ``pv_estimate``, ``temperature``, ...);
        - OR it's a flat ``dict[str, float]`` where the keys parse as
          datetimes (Forecast.Solar's ``detailedForecast`` schema).

        Returns ``{"forecast_attributes": [{"name": ..., "format":
        "list-of-dict"|"date-dict", "sample_keys": [...]}, ...]}``.
        The frontend picks the attribute name into the form and offers
        the sample_keys as choices for ``future_value_key`` (None =
        auto-detect via the resolver's common-key fallback).
        """
        import aiohttp as _aiohttp

        entity_id = (request.query_params.get("entity") or "").strip()
        if not entity_id:
            return JSONResponse(content={"forecast_attributes": []})

        ha_url = os.environ.get("HA_URL", "http://supervisor/core")
        ha_token = os.environ.get("SUPERVISOR_TOKEN", "")
        try:
            async with _aiohttp.ClientSession() as sess:
                headers = {"Authorization": f"Bearer {ha_token}"}
                async with sess.get(
                    f"{ha_url}/api/states/{entity_id}",
                    headers=headers,
                    timeout=_aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        return JSONResponse(content={
                            "forecast_attributes": [],
                            "error": f"HA returned {resp.status}",
                        })
                    state_obj = await resp.json()
        except Exception as e:
            logger.debug(f"forecast-attrs fetch failed for {entity_id}: {e}")
            return JSONResponse(content={
                "forecast_attributes": [],
                "error": str(e),
            })

        # Keep this list aligned with covariates.py:VAL_KEYS / DT_KEYS so
        # the UI surfaces the same set the backend resolver knows how
        # to auto-detect. Out-of-band keys still work via the explicit
        # future_value_key field — the dropdown just doesn't suggest
        # them.
        DT_KEYS = {
            'datetime', 'period_start', 'period_end',
            'time', 'dt', 'start',
        }
        VAL_KEYS = {
            'value', 'pv_estimate', 'state',
            'temperature', 'cloud_coverage', 'cloud_cover',
            'wind_speed', 'precipitation', 'humidity',
        }

        results: list[dict] = []
        for name, attr in (state_obj.get("attributes") or {}).items():
            # list-of-dict format (Solcast, Met.no weather, etc.)
            if isinstance(attr, list) and attr and isinstance(attr[0], dict):
                first = attr[0]
                has_dt = any(k in DT_KEYS for k in first.keys())
                if not has_dt:
                    continue
                # Collect numeric keys from the first entry as candidate
                # value_keys — preserve insertion order so the most
                # likely match (per VAL_KEYS priority) bubbles up.
                numeric_keys = [
                    k for k, v in first.items()
                    if k not in DT_KEYS and isinstance(v, (int, float))
                ]
                # Prefer common value keys at the top of the dropdown,
                # then any other numeric fields the entity exposes
                # (e.g. ``pv_estimate90`` on Solcast).
                common = [k for k in numeric_keys if k in VAL_KEYS]
                other = [k for k in numeric_keys if k not in VAL_KEYS]
                sample_keys = common + other
                if not sample_keys:
                    continue
                results.append({
                    "name": name,
                    "format": "list-of-dict",
                    "sample_keys": sample_keys,
                })
            # flat date-keyed dict format (Forecast.Solar detailedForecast)
            elif isinstance(attr, dict) and len(attr) >= 2:
                # Heuristic: sample first 5 keys, check if they look
                # like ISO datetimes. Avoids inspecting the whole dict
                # for large arrays.
                sample_count = 0
                dt_count = 0
                for k in list(attr.keys())[:5]:
                    sample_count += 1
                    if not isinstance(k, str):
                        continue
                    # Cheap ISO-ish check — full date parsing would
                    # need datetime.fromisoformat which is overly
                    # strict for some sources.
                    if (len(k) >= 10 and k[4] == '-' and k[7] == '-'):
                        dt_count += 1
                if sample_count >= 2 and dt_count == sample_count:
                    results.append({
                        "name": name,
                        "format": "date-dict",
                        "sample_keys": [],  # no value_key needed for this format
                    })

        # HA 2023.9+ weather entities (Met Office DataHub, OpenWeatherMap,
        # AccuWeather, modern met.no) expose forecasts via the
        # weather.get_forecasts service call rather than state
        # attributes — so the attribute-scan above misses them entirely.
        # Detect them via the supported_features bitmask and call the
        # service once per supported forecast type to learn what
        # numeric keys are available (each integration exposes a
        # different subset: Met Office gives cloud_coverage, met.no
        # adds uv_index, etc.).
        if entity_id.startswith("weather."):
            FORECAST_DAILY = 1
            FORECAST_HOURLY = 2
            FORECAST_TWICE_DAILY = 4
            supported = (
                state_obj.get("attributes", {}).get("supported_features", 0)
            )
            if isinstance(supported, (int, float)) and supported > 0:
                supported = int(supported)
                forecast_types = []
                if supported & FORECAST_HOURLY:
                    forecast_types.append("hourly")
                if supported & FORECAST_DAILY:
                    forecast_types.append("daily")
                if supported & FORECAST_TWICE_DAILY:
                    forecast_types.append("twice_daily")
                for ftype in forecast_types:
                    sample_keys = await _probe_weather_forecast_keys(
                        ha_url, ha_token, entity_id, ftype, VAL_KEYS,
                    )
                    if sample_keys is None:
                        # Service call failed — surface the option
                        # anyway so the user can still pick it; the
                        # value_key dropdown will be empty (Auto only).
                        sample_keys = []
                    results.append({
                        "name": ftype,
                        "format": "weather-service",
                        "sample_keys": sample_keys,
                    })

        return JSONResponse(content={"forecast_attributes": results})

    @app.get("/api/covariates/validate")
    async def validate_covariate(request: Request):  # noqa: D401 (handler)
        """Lite-flavour data-availability check for one covariate row
        (v2.38.7). Single HA ``/api/states/{entity_id}`` call — no
        history fetch, no service probe — so the UI can validate on
        page load and after add without burning the user's HA quota.

        See ``classify_covariate_state`` for the decision matrix and
        return-shape contract. This handler is a thin transport
        wrapper: fetch HA, classify, jsonify.
        """
        import aiohttp as _aiohttp

        entity_id = (request.query_params.get("entity_id") or "").strip()
        future_attribute = (request.query_params.get("future_attribute") or "").strip()
        future_value_key = (request.query_params.get("future_value_key") or "").strip() or None

        if not entity_id:
            return JSONResponse(content={
                "ok": False, "status": "broken",
                "state_value": None, "last_changed": None,
                "message": "entity_id is required",
                "attribute_preview": None,
            })

        ha_url = os.environ.get("HA_URL", "http://supervisor/core")
        ha_token = os.environ.get("SUPERVISOR_TOKEN", "")
        try:
            sess = _get_shared_ha_session()
            headers = {"Authorization": f"Bearer {ha_token}"}
            async with sess.get(
                f"{ha_url}/api/states/{entity_id}",
                headers=headers,
                timeout=_aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 404:
                    return JSONResponse(content={
                        "ok": False, "status": "broken",
                        "state_value": None, "last_changed": None,
                        "message": "Entity not found in HA",
                        "attribute_preview": None,
                    })
                if resp.status != 200:
                    return JSONResponse(content={
                        "ok": False, "status": "broken",
                        "state_value": None, "last_changed": None,
                        "message": f"HA returned {resp.status}",
                        "attribute_preview": None,
                    })
                state_obj = await resp.json()
        except Exception as e:
            logger.debug(f"validate-covariate fetch failed for {entity_id}: {e}")
            return JSONResponse(content={
                "ok": False, "status": "broken",
                "state_value": None, "last_changed": None,
                "message": f"HA fetch error: {e}",
                "attribute_preview": None,
            })

        return JSONResponse(content=classify_covariate_state(
            entity_id=entity_id,
            state_obj=state_obj,
            future_attribute=future_attribute or None,
            future_value_key=future_value_key,
        ))

    # ---- Forecast accuracy (evolution log) ----

    @app.get("/experiment/{name}/forecast-accuracy")
    async def forecast_accuracy(name: str, request: Request):
        """Return forecast accuracy data grouped by lead time."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        db = app.state.appstate.history_db
        if not db:
            return JSONResponse(content={"error": "Database not available"}, status_code=503)

        # Find the experiment config to get the target entity table name
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"error": "Config not found"}, status_code=503)

        from ml_forecast_lab.config import load_config
        try:
            cfg = load_config(config_path)
            exp_cfg = next((e for e in cfg.experiments if e.name == name), None)
            if not exp_cfg:
                return JSONResponse(content={"error": "Experiment not in config"}, status_code=404)
            actuals_table = db.safe_table_name(exp_cfg.target_entity)
        except Exception as e:
            return JSONResponse(content={"error": _safe_error(e)}, status_code=500)

        try:
            days = int(request.query_params.get("days", "30"))
        except (TypeError, ValueError):
            days = 30
        days = max(1, min(365, days))
        # Default to increment-based evaluation for cumulative sensors —
        # raw-value MAE/RMSE on a daily-resetting cumulative sensor mostly
        # reflects the sensor's shape through the day rather than model
        # skill. UI can override via ?mode=raw.
        mode_param = request.query_params.get("mode")
        if mode_param in ("raw", "increment", "daily_cumulative"):
            evaluation_mode = mode_param
        else:
            evaluation_mode = "increment" if exp_cfg.source_is_cumulative else "raw"
        # v2.40.7: defensively coerce raw → increment for cumulative
        # sensors. Raw mode compared per-interval predictions (the only
        # thing forecast_log stores) against raw cumulative actuals — a
        # space mismatch that produced MAE ≈ the cumulative level
        # rather than model error. The UI no longer offers the raw
        # toggle for these experiments, but a bookmarked or cached
        # ?mode=raw URL would still hit this branch.
        if evaluation_mode == "raw" and exp_cfg.source_is_cumulative:
            evaluation_mode = "increment"
        # v2.40.9: daily_cumulative mode is only meaningful for
        # cumulative-source sensors (it reads the seed from actuals
        # at issued_at and cumsums per-interval predictions to compare
        # against the cumulative actual). For non-cumulative sensors,
        # fall back to the default per-interval mode rather than
        # produce nonsense.
        if (evaluation_mode == "daily_cumulative"
                and not exp_cfg.source_is_cumulative):
            evaluation_mode = "raw"
        # Default filter: current champion + its latest training tag. UI
        # can escape via ?model=all or ?version=all. See
        # _resolve_model_filter for the full contract.
        model_name, model_version = _resolve_model_filter(name, request)
        model_param = request.query_params.get("model")
        version_param = request.query_params.get("version")

        # Pick the narrowest filter that has forecast_log rows in the
        # window *before* running the expensive accuracy query. Each
        # full call scans the actuals table and does three CTE passes;
        # the old ladder could run it three times when the strict filter
        # came up empty. probe_forecast_rows uses the
        # (experiment, model_name, model_version) index so each probe
        # is sub-millisecond.
        requested_model = model_name
        requested_version = model_version
        model_fallback: Optional[Dict[str, Any]] = None

        if model_name or model_version:
            has_strict = await asyncio.to_thread(
                db.probe_forecast_rows,
                name, model_name, model_version, days,
            )
            if not has_strict and model_version and not version_param:
                # Drop version, keep model name.
                has_model = await asyncio.to_thread(
                    db.probe_forecast_rows,
                    name, model_name, None, days,
                )
                if has_model:
                    logger.info(
                        f"/forecast-accuracy fallback for {name}: "
                        f"no cycles for {model_name!r} v={model_version!r}; "
                        f"widening to all versions of this model."
                    )
                    model_fallback = {
                        "requested_model": requested_model,
                        "requested_version": requested_version,
                        "used_model": model_name,
                        "used_version": None,
                        "reason": "No forecasts logged for this version yet; showing all versions of this model.",
                    }
                    model_version = None
                elif model_name and not model_param:
                    logger.info(
                        f"/forecast-accuracy fallback for {name}: "
                        f"no cycles for {model_name!r}; widening to all models."
                    )
                    model_fallback = {
                        "requested_model": requested_model,
                        "requested_version": requested_version,
                        "used_model": None,
                        "used_version": None,
                        "reason": "No forecasts logged for this model in the selected window.",
                    }
                    model_name = None
                    model_version = None
            elif not has_strict and model_name and not model_param:
                logger.info(
                    f"/forecast-accuracy fallback for {name}: "
                    f"no cycles for {model_name!r}; widening to all models."
                )
                model_fallback = {
                    "requested_model": requested_model,
                    "requested_version": requested_version,
                    "used_model": None,
                    "used_version": None,
                    "reason": "No forecasts logged for this model in the selected window.",
                }
                model_name = None
                model_version = None

        # v2.40.10: compute UTC→HA-local hour offset so the
        # daily_cumulative bucketing aligns with the physical
        # ``_today`` sensor reset at HA local midnight (mirrors the
        # stability endpoint at app.py:3120-3134). Without this, BST
        # / other-TZ deployments had the "last same-day target" land
        # right after local midnight reset, making the avg actual
        # day-total read near zero. Only daily_cumulative uses it; the
        # other modes ignore the param.
        day_offset_hours: Optional[float] = None
        if evaluation_mode == "daily_cumulative":
            tz_name = app.state.appstate.ha_time_zone
            if tz_name:
                try:
                    from zoneinfo import ZoneInfo
                    from datetime import datetime as _dt, timezone as _tz
                    now_utc = _dt.now(_tz.utc)
                    _offset = now_utc.astimezone(ZoneInfo(tz_name)).utcoffset()
                    if _offset is not None:
                        day_offset_hours = _offset.total_seconds() / 3600.0
                except Exception as e:
                    logger.debug(
                        f"Could not compute day offset for {tz_name}: {e}"
                    )

        result = await asyncio.to_thread(
            db.get_forecast_accuracy,
            name, actuals_table, days,
            exp_cfg.interval_minutes,
            evaluation_mode,
            model_name,
            model_version,
            day_offset_hours,
        )
        result["model_name"] = model_name
        result["model_version"] = model_version
        if model_fallback:
            result["model_fallback"] = model_fallback
        # Merge empirical interval coverage. Always on raw values (that's
        # what the published entities are); independent of evaluation_mode.
        # Pass HA's configured time zone so the hour-of-day breakdown
        # surfaces in the user's local "evening peak" / "Sunday morning"
        # terms rather than UTC.
        try:
            ha_tz = getattr(
                request.app.state.appstate, "ha_time_zone", None,
            ) if request is not None else None
        except Exception:
            ha_tz = None
        try:
            # Read the user's configured nominal level — defaults to
            # 0.8 if absent. The verdict-card chip, worst-bucket
            # selection, and reported deviations all need to compare
            # against the level the bands were calibrated at; hard-
            # coding 0.8 here mis-labels every experiment running on
            # a different ``conformal_coverage`` (e.g. 0.9 / 0.95).
            nominal = float(getattr(exp_cfg, 'conformal_coverage', 0.8))
            coverage = await asyncio.to_thread(
                db.get_forecast_coverage,
                name, actuals_table,
                exp_cfg.interval_minutes,
                days,
                model_name,
                model_version,
                ha_tz,
                nominal,
            )
            buckets = []
            for kind, container, key, label_fmt in [
                ("hour_of_day", coverage.get("by_hour_of_day", {}), "hour", lambda v: f"hour {int(v):02d}"),
                ("weekday_weekend", coverage.get("by_weekday_weekend", {}), "bucket", lambda v: str(v)),
                ("lead", coverage.get("by_lead", {}), "lead_minutes", lambda v: f"+{int(v)}min"),
            ]:
                for v, cov, n in zip(container.get(key, []), container.get("coverage", []), container.get("n", [])):
                    if n >= 20:
                        buckets.append({
                            "kind": kind, "label": label_fmt(v),
                            "coverage": cov, "n": n,
                            "deviation": round(cov - nominal, 4),
                        })
            if buckets:
                # Worst = biggest absolute deviation from the nominal
                # level. Picking the largest |deviation| matches the
                # verdict-card UX: "your bands are off by the most
                # here".
                coverage["worst_bucket"] = max(
                    buckets, key=lambda b: abs(b["deviation"]),
                )
            result["coverage"] = coverage
            result["nominal_interval_level"] = nominal
        except Exception as e:
            result["coverage"] = {"error": _safe_error(e)}

        # Retrain history — distinct (model_name, model_version) pairs
        # in the window, ordered by first_seen. Used by the Forecast
        # Accuracy tab to render markers on the diagnostic charts so
        # the user can see "did the retrain on Tuesday make things
        # better or worse?".
        try:
            result["retrain_events"] = await asyncio.to_thread(
                db.get_retrain_events, name, days, model_name,
            )
        except Exception as e:
            result["retrain_events"] = []
            logger.debug("get_retrain_events failed for %s: %s", name, e)

        # Calibration progress. Surfaces "we have N of the M residuals
        # needed before _upper_80 / _lower_80 sensors start publishing"
        # so a freshly-promoted experiment doesn't leave the user
        # wondering why the bands tile says "—" with no explanation.
        try:
            level = float(getattr(exp_cfg, 'conformal_coverage', 0.8))
            # NEW-D1.2: scale `min_samples` with the requested coverage
            # level. The conformal quantile at level=0.8 is the 80th
            # percentile of absolute residuals (v2.41.0 — see
            # get_conformal_quantiles); at level=0.95 it's the 95th.
            # Higher percentiles need more samples for stable
            # estimates. Rule-of-thumb: max(10, ceil(10 / (1 - level)))
            # — at level=0.8 that's 50, at level=0.95 it's 200. The
            # floor of 10 keeps backwards compatibility for the
            # default 0.8 case.
            import math as _math
            n_need = max(10, int(_math.ceil(10.0 / max(1e-6, 1.0 - level))))
            cq = await asyncio.to_thread(
                db.get_conformal_quantiles,
                name, actuals_table,
                level,
                model_name,
                exp_cfg.interval_minutes,
                14,
                n_need,
                model_version,
            )
            n_have = int(cq.get("total_samples") or 0)
            forecast_every = exp_cfg.forecast_every_minutes or 30
            ready = bool(cq.get("fallback_quantile") is not None) or n_have >= n_need
            # NEW-D1.1: each forecast cycle produces `future_periods`
            # residuals as actuals arrive (one per horizon step), not 1.
            # Previously we computed ETA as cycles_remaining * cycle_period
            # which over-estimates by ~future_periods. Divide by an
            # estimate of residuals-per-cycle so the displayed ETA is
            # actually meaningful. Conservative floor of 1 in case
            # future_periods isn't set.
            residuals_per_cycle = max(1, int(getattr(exp_cfg, 'future_periods', 1)))
            samples_remaining = max(0, n_need - n_have)
            cycles_remaining_eff = max(0.0, samples_remaining / float(residuals_per_cycle))
            result["calibration"] = {
                "ready": ready,
                "total_samples": n_have,
                "min_samples": n_need,
                "forecast_every_minutes": forecast_every,
                "eta_minutes": int(round(cycles_remaining_eff * forecast_every)) if not ready else 0,
                "level": cq.get("level", 0.8),
            }
        except Exception as e:
            result["calibration"] = {"error": _safe_error(e)}
        return JSONResponse(content=result)

    @app.get("/experiment/{name}/external-forecast-comparison")
    async def external_forecast_comparison(name: str, request: Request):
        """Head-to-head: this add-on's forecast vs a third-party (external)
        forecast sensor, both scored against the actuals.

        Reads only local SQLite (the cached actuals, this add-on's
        ``forecast_log``, and — for ``attribute`` mode — the captured
        ``external_forecast_log``); the external data is ingested by the
        production forecast cycle, not fetched live here.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        db = app.state.appstate.history_db
        if not db:
            return JSONResponse(content={"error": "Database not available"}, status_code=503)

        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"error": "Config not found"}, status_code=503)

        from ml_forecast_lab.config import load_config
        try:
            cfg = load_config(config_path)
            exp_cfg = next((e for e in cfg.experiments if e.name == name), None)
            if not exp_cfg:
                return JSONResponse(content={"error": "Experiment not in config"}, status_code=404)
        except Exception as e:
            return JSONResponse(content={"error": _safe_error(e)}, status_code=500)

        externals_cfg = getattr(exp_cfg, "external_forecasts", None) or []
        if not externals_cfg:
            # Tab is configured off — tell the frontend so it can show the
            # "configure an external sensor in Settings" empty state.
            return JSONResponse(content={"configured": False})

        try:
            days = int(request.query_params.get("days", "30"))
        except (TypeError, ValueError):
            days = 30
        days = max(1, min(365, days))

        evaluation_mode = "increment" if exp_cfg.source_is_cumulative else "raw"

        # Per-sensor HA units for unit-aware conversion (cached). Target unit
        # prefers the experiment's explicit ``units``, else the source
        # sensor's HA unit_of_measurement.
        try:
            units_map = await _entity_units(
                [exp_cfg.target_entity] + [e.entity_id for e in externals_cfg]
            )
        except Exception:
            units_map = {}
        target_unit = exp_cfg.units or units_map.get(exp_cfg.target_entity)

        # Build the per-external spec list the DB layer consumes. State-mode
        # entries resolve to their cached-history table; attribute-mode ones
        # read from external_forecast_log (table=None). is_cumulative is
        # passed through as-is: None lets the DB auto-detect a cumulative
        # shape; True/False is an explicit override.
        def _default_ext_label(e):
            base = e.entity_id.split(".")[-1]
            if e.mode == "attribute":
                extra = e.value_key or (
                    e.attribute if e.attribute and e.attribute != "forecast" else None
                )
                if extra:
                    return base + " · " + extra
            return base

        try:
            actuals_table = db.safe_table_name(exp_cfg.target_entity)
            specs = []
            for e in externals_cfg:
                specs.append({
                    "entity": e.entity_id,
                    # Composite identity so the same entity can supply several
                    # distinct forecasts (different attribute / value key).
                    "source": e.source_key,
                    "table": db.safe_table_name(e.entity_id) if e.mode == "state" else None,
                    "mode": e.mode,
                    "scale": e.scale,
                    "is_cumulative": e.is_cumulative,
                    "label": e.label or _default_ext_label(e),
                    "unit": units_map.get(e.entity_id),
                })
        except Exception as e:
            return JSONResponse(content={"error": _safe_error(e)}, status_code=500)

        analysis = request.query_params.get("analysis")
        analysis_mode = analysis if analysis in ("per_interval", "cumulative") else "per_interval"

        # Optional model pin (default: compare against everything the
        # add-on actually published, latest-per-target — that's the deployed
        # forecast the user is comparing). ?model=/?version= narrow it.
        model_param = request.query_params.get("model")
        version_param = request.query_params.get("version")
        model_name = model_param if model_param and model_param != "all" else None
        model_version = version_param if version_param and version_param != "all" else None

        try:
            result = await asyncio.to_thread(
                db.get_external_forecast_comparison,
                name,
                actuals_table,
                specs,
                days,
                exp_cfg.interval_minutes,
                evaluation_mode,
                model_name,
                model_version,
                analysis_mode,
                target_unit,
            )
        except Exception as e:
            logger.error(
                "external-forecast-comparison failed for %s: %s",
                name, e, exc_info=True,
            )
            return JSONResponse(content={"error": _safe_error(e)}, status_code=500)

        result["units"] = exp_cfg.units or ""
        result["days"] = days
        # JSONResponse here is the module's NaN-safe subclass — the metric
        # floats below can be non-finite and would otherwise fail a strict
        # WebKit parser (Safari / the iOS HA app).
        return JSONResponse(content=result)

    _unit_cache: Dict[str, tuple] = {}  # entity -> (fetched_at, unit_or_None)

    async def _entity_units(entities: list) -> Dict[str, Optional[str]]:
        """Resolve each entity's HA unit_of_measurement, cached 5 min, for
        unit-aware comparison conversion. One HA session for all misses;
        failures resolve to None (→ the comparison falls back to raw + the
        scale-mismatch guard)."""
        import time as _t
        out: Dict[str, Optional[str]] = {}
        missing = []
        for e in entities:
            if not e:
                continue
            c = _unit_cache.get(e)
            if c and (_t.time() - c[0]) < 300:
                out[e] = c[1]
            else:
                missing.append(e)
        if missing:
            from ml_forecast_lab.ha_interface import HAInterface
            iface = HAInterface()
            try:
                for e in missing:
                    try:
                        u = await iface.get_state(
                            e, attribute="unit_of_measurement", default=None,
                        )
                        u = u.strip() if isinstance(u, str) and u.strip() else None
                    except Exception:
                        u = None
                    _unit_cache[e] = (_t.time(), u)
                    out[e] = u
            finally:
                await iface.close()
        return out

    async def _backfill_external_state(entity: str, days: int = 90) -> int:
        """Cache a state-mode external sensor's existing HA recorder history
        so the comparison populates immediately rather than only accruing
        from the next production cycle. Best-effort; returns rows stored.

        Bounded by HA's recorder retention (it returns only what it kept);
        the per-cycle capture extends the series forward beyond that.
        """
        db = app.state.appstate.history_db
        if not db:
            return 0
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        from ml_forecast_lab.ha_interface import HAInterface, normalise_history
        end = _dt.now(_tz.utc)
        start = end - _td(days=days)
        iface = HAInterface()
        try:
            raw = await iface.get_history(entity, start, end)
        finally:
            await iface.close()
        df = normalise_history(raw)
        if df is None or df.empty:
            return 0
        table = db.safe_table_name(entity)
        return await asyncio.to_thread(db.store_history, table, df)

    async def _probe_external_attribute(entity, attribute, value_key):
        """Best-effort live check that an attribute-mode external resolves.
        Returns a warning string if the chosen attribute is missing/empty or
        the value_key doesn't appear in the forecast entries — so a typo is
        caught at add time instead of silently logging zero rows for days.
        Returns None when it looks fine OR when HA is unreachable (we never
        block the add on a transient fetch failure)."""
        import aiohttp as _aiohttp
        attr = (attribute or "forecast").strip()
        ha_url = os.environ.get("HA_URL", "http://supervisor/core")
        ha_token = os.environ.get("SUPERVISOR_TOKEN", "")
        try:
            async with _aiohttp.ClientSession() as sess:
                async with sess.get(
                    f"{ha_url}/api/states/{entity}",
                    headers={"Authorization": f"Bearer {ha_token}"},
                    timeout=_aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        return None  # can't confirm — don't block the add
                    state_obj = await resp.json()
        except Exception:
            return None
        attrs = (state_obj or {}).get("attributes", {}) or {}
        if attr not in attrs:
            avail = [a for a in attrs if isinstance(attrs[a], (list, dict))]
            return (
                f"Attribute '{attr}' not found on {entity} right now. "
                + (f"Forecast-shaped attributes available: {', '.join(avail[:6])}. "
                   if avail else "")
                + "The comparison will stay empty until it appears."
            )
        val = attrs.get(attr)
        if not val:
            return f"Attribute '{attr}' on {entity} is currently empty."
        if value_key and isinstance(val, list) and val and isinstance(val[0], dict):
            if value_key not in val[0]:
                keys = [k for k in val[0].keys()]
                return (
                    f"Value key '{value_key}' not in '{attr}' entries "
                    f"(keys: {', '.join(str(k) for k in keys[:8])}). "
                    "Check the value key."
                )
        return None

    @app.post("/experiment/{name}/add-external-forecast")
    async def add_external_forecast(name: str, request: Request):
        """Add a third-party forecast to an experiment's external_forecasts.

        Body: {entity_id (required), mode?, attribute?, value_key?, scale?,
               is_cumulative?, label?}. Capped at MAX_EXTERNAL_FORECASTS.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        entity = (body.get("entity_id") or body.get("entity") or "").strip()
        if not entity:
            return JSONResponse(content={"success": False, "error": "entity_id is required"})

        ext = {"entity_id": entity}
        for fld in ("mode", "attribute", "value_key", "label"):
            if body.get(fld) not in (None, ""):
                ext[fld] = str(body[fld]).strip()
        if body.get("scale") not in (None, ""):
            try:
                ext["scale"] = float(body["scale"])
            except (TypeError, ValueError):
                return JSONResponse(content={"success": False, "error": "scale must be a number"})
        if "is_cumulative" in body and body["is_cumulative"] not in (None, ""):
            v = str(body["is_cumulative"]).lower()
            if v in ("true", "1", "yes", "on"):
                ext["is_cumulative"] = True
            elif v in ("false", "0", "no", "off"):
                ext["is_cumulative"] = False

        from ml_forecast_lab.config import (
            add_experiment_external_forecast, MAX_EXTERNAL_FORECASTS,
        )
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"success": False, "error": "Config file not found"})
        try:
            added = add_experiment_external_forecast(config_path, name, ext)
            if added:
                logger.info(f"Added external forecast {entity} to {name}")
                backfilled = 0
                warning = None
                # State-mode sensors have real recorder history — backfill it
                # now so the comparison line + head-to-head show immediately
                # instead of only accruing from the next production cycle.
                # (Attribute/trajectory sensors can't be backfilled: HA
                # doesn't retain past forecast attributes.)
                if ext.get("mode", "state") == "state":
                    try:
                        backfilled = await _backfill_external_state(entity)
                    except Exception as e:
                        logger.debug(f"External backfill for {entity} failed: {e}")
                else:
                    # Attribute mode: confirm the chosen attribute/value_key
                    # resolves live (non-fatal — a typo shouldn't only surface
                    # after days of empty logging).
                    try:
                        warning = await _probe_external_attribute(
                            entity, ext.get("attribute"), ext.get("value_key"),
                        )
                    except Exception:
                        warning = None
                return JSONResponse(content={
                    "success": True, "entity_id": entity,
                    "backfilled": backfilled, "warning": warning,
                })
            return JSONResponse(content={
                "success": False,
                "error": (
                    f"Already configured, experiment not found, or at the "
                    f"max of {MAX_EXTERNAL_FORECASTS} external forecasts."
                ),
            })
        except ValueError as e:
            return JSONResponse(content={"success": False, "error": _safe_error(e)})
        except Exception as e:
            logger.error(f"Failed to add external forecast: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

    @app.post("/experiment/{name}/remove-external-forecast")
    async def remove_external_forecast(name: str, request: Request):
        """Remove a third-party forecast from an experiment. The full identity
        (entity_id + mode + attribute + value_key) is used so the right one is
        removed when an entity is configured more than once."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})
        entity = (body.get("entity_id") or body.get("entity") or "").strip()
        if not entity:
            return JSONResponse(content={"success": False, "error": "entity_id is required"})
        mode = (body.get("mode") or None)
        attribute = body.get("attribute") if body.get("attribute") not in (None, "") else None
        value_key = body.get("value_key") if body.get("value_key") not in (None, "") else None

        from ml_forecast_lab.config import (
            remove_experiment_external_forecast, external_source_key,
        )
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"success": False, "error": "Config file not found"})
        try:
            removed = remove_experiment_external_forecast(
                config_path, name, entity, mode=mode,
                attribute=attribute, value_key=value_key,
            )
            # Drop the captured trajectory rows for this exact source too
            # (best effort) so a re-added forecast starts clean.
            db = app.state.appstate.history_db
            if removed and db:
                try:
                    db.delete_external_forecast_source(
                        name, external_source_key(entity, mode or "state", attribute, value_key),
                    )
                except Exception:
                    pass
            return JSONResponse(content={"success": bool(removed)})
        except Exception as e:
            logger.error(f"Failed to remove external forecast: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

    @app.get("/experiment/{name}/forecast-log-stats")
    async def forecast_log_stats(name: str, request: Request):
        """
        Diagnostic endpoint — summarises what's actually in forecast_log
        for this experiment. Used to debug "why is my Forecast Accuracy
        tab empty?" without needing shell access to the add-on.

        Returns a by-(model_name, model_version) breakdown plus the
        filter the UI would apply by default, so the user can tell at
        a glance whether:
          - log_forecast is writing at all,
          - the new model_version tag is being stamped,
          - the current champion+version cohort actually has enough
            cycles for stability (≥2 per target).
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")
        db = app.state.appstate.history_db
        if not db:
            return JSONResponse(content={"error": "Database not available"}, status_code=503)

        exp_status = app.state.appstate.experiment_statuses.get(name)
        default_model = (
            getattr(exp_status, "selected_model", None)
            or getattr(exp_status, "best_model", None)
            if exp_status else None
        )
        # Keep in sync with _resolve_model_filter: the version default
        # only applies when selected_model matches best_model.
        best_model = getattr(exp_status, "best_model", None) if exp_status else None
        default_version = None
        if exp_status and default_model and default_model == best_model:
            default_version = getattr(exp_status, "model_version", None)

        try:
            # v2.41.0 (audit F4): queries moved into
            # HistoryDB.get_forecast_log_stats — this handler used to
            # run them on a raw cursor, on the event loop, without the
            # DB lock.
            stats = await asyncio.to_thread(
                db.get_forecast_log_stats, name,
                default_model, default_version,
            )
            cohorts = stats["cohorts"]
            totals = stats["totals"]
            targets_with_multi_issuances = stats[
                "targets_with_multi_issuances"
            ]
        except Exception as e:
            logger.error(f"forecast-log-stats failed for {name}: {e}", exc_info=True)
            return JSONResponse(content={"error": _safe_error(e)}, status_code=500)

        # Surface common diagnostic conditions so future debugging
        # doesn't require re-deriving them from the raw cohorts.
        notes: list = []
        if default_model and best_model and default_model != best_model:
            notes.append(
                f"selected_model={default_model!r} differs from "
                f"best_model={best_model!r}. The version filter would "
                "be spurious (status.model_version tracks the last "
                "retrained model) so it's been suppressed automatically. "
                "To see metrics for the current champion, re-select "
                f"{best_model!r} in the UI or set "
                f"production_model: {best_model} in mlfl.yaml."
            )
        if default_model and default_version:
            matched = any(
                c["model_name"] == default_model
                and c["model_version"] == default_version
                for c in cohorts
            )
            if not matched:
                notes.append(
                    "current_default_filter points to a "
                    "(model_name, model_version) combination with zero "
                    "rows. The endpoint will fall back to all versions "
                    "of this model (and then to all models) until the "
                    "cohort accumulates rows."
                )

        return JSONResponse(content={
            "experiment": name,
            "current_default_filter": {
                "model_name": default_model,
                "model_version": default_version,
            },
            "selected_vs_best": {
                "selected_model": getattr(exp_status, "selected_model", None) if exp_status else None,
                "best_model": best_model,
                "matches": default_model == best_model,
            },
            "totals": totals,
            "targets_with_multi_issuances_under_default_filter": targets_with_multi_issuances,
            "cohorts": cohorts,
            "notes": notes,
            "hint": (
                "If 'cohorts' has a row matching current_default_filter with rows>=96 "
                "and targets_with_multi_issuances>=1, the fallback should stop firing "
                "within a cycle or two. If rows=0 under the current filter but other "
                "cohorts have data, check 'notes' and 'selected_vs_best' — the usual "
                "cause is a selected_model that isn't the current champion."
            ),
        })

    @app.get("/experiment/{name}/forecast-trajectory")
    async def forecast_trajectory(name: str, request: Request):
        """Return every forecast ever issued for one target_dt + the actual."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        db = app.state.appstate.history_db
        if not db:
            return JSONResponse(content={"error": "Database not available"}, status_code=503)

        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"error": "Config not found"}, status_code=503)

        from ml_forecast_lab.config import load_config
        try:
            cfg = load_config(config_path)
            exp_cfg = next((e for e in cfg.experiments if e.name == name), None)
            if not exp_cfg:
                return JSONResponse(content={"error": "Experiment not in config"}, status_code=404)
            actuals_table = db.safe_table_name(exp_cfg.target_entity)
        except Exception as e:
            return JSONResponse(content={"error": _safe_error(e)}, status_code=500)

        target_dt = request.query_params.get("target_dt") or None
        model_name, model_version = _resolve_model_filter(name, request)
        model_param = request.query_params.get("model")
        version_param = request.query_params.get("version")

        def _fetch(mn, mv):
            return db.get_forecast_trajectory(
                name, actuals_table,
                target_dt=target_dt,
                interval_minutes=exp_cfg.interval_minutes,
                source_is_cumulative=bool(exp_cfg.source_is_cumulative),
                model_name=mn, model_version=mv,
            )

        result = await asyncio.to_thread(_fetch, model_name, model_version)
        # Same fallback ladder as /forecast-accuracy:
        def _empty(res):
            return not res.get("available_targets")
        if _empty(result) and model_version and not version_param:
            relaxed = await asyncio.to_thread(_fetch, model_name, None)
            if not _empty(relaxed):
                logger.info(
                    f"/forecast-trajectory fallback for {name}: no targets "
                    f"for {model_name!r} v={model_version!r}; widening to all versions."
                )
                result = relaxed
                result["model_fallback"] = {
                    "requested_model": model_name,
                    "requested_version": model_version,
                    "used_model": model_name,
                    "used_version": None,
                    "reason": "No forecasts logged for this version yet; showing all versions of this model.",
                }
                model_version = None
        if _empty(result) and model_name and not model_param:
            relaxed = await asyncio.to_thread(_fetch, None, None)
            if not _empty(relaxed):
                logger.info(
                    f"/forecast-trajectory fallback for {name}: no targets "
                    f"for {model_name!r}; widening to all models."
                )
                result = relaxed
                result["model_fallback"] = {
                    "requested_model": model_name,
                    "requested_version": model_version,
                    "used_model": None,
                    "used_version": None,
                    "reason": "No forecasts logged for this model in the selected window.",
                }
                model_name = None
                model_version = None
        result["model_name"] = model_name
        result["model_version"] = model_version
        return JSONResponse(content=result)

    @app.get("/experiment/{name}/forecast-evolution")
    async def forecast_evolution(name: str, request: Request):
        """Return the last N forecast snapshots + actuals for overlay plot."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        db = app.state.appstate.history_db
        if not db:
            return JSONResponse(content={"error": "Database not available"}, status_code=503)

        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"error": "Config not found"}, status_code=503)

        from ml_forecast_lab.config import load_config
        try:
            cfg = load_config(config_path)
            exp_cfg = next((e for e in cfg.experiments if e.name == name), None)
            if not exp_cfg:
                return JSONResponse(content={"error": "Experiment not in config"}, status_code=404)
            actuals_table = db.safe_table_name(exp_cfg.target_entity)
        except Exception as e:
            return JSONResponse(content={"error": _safe_error(e)}, status_code=500)

        # Clamp n_cycles to [2, 48] — 2 is the minimum for a "change over
        # time" visual, 48 keeps the chart legible and the query cheap.
        try:
            n_cycles = int(request.query_params.get("cycles", "12"))
        except (TypeError, ValueError):
            n_cycles = 12
        n_cycles = max(2, min(48, n_cycles))

        # Apply the same default-then-widen ladder as /forecast-accuracy
        # so the fan chart never mixes rotated-out predictions with the
        # current champion. Without this, a retrain mid-window mixes
        # two weight regimes on the same axes and the "Latest run" line
        # is incoherent against the band.
        model_name, model_version = _resolve_model_filter(name, request)
        model_param = request.query_params.get("model")
        version_param = request.query_params.get("version")
        model_fallback: Optional[Dict[str, Any]] = None

        def _fetch(mn, mv):
            return db.get_forecast_evolution(
                name, actuals_table,
                n_cycles=n_cycles,
                interval_minutes=exp_cfg.interval_minutes,
                model_name=mn, model_version=mv,
                source_is_cumulative=bool(exp_cfg.source_is_cumulative),
            )

        result = await asyncio.to_thread(_fetch, model_name, model_version)

        # Empty if either there are no cycles at all, or every cycle's
        # targets are still in the future so the actuals query returns
        # nothing. The latter happens for ~one horizon-worth of time
        # after every retrain — without it, the chart shows a single
        # future-only cycle with no actual curve. Same family of bug
        # as the v2.34.1 probe fix on /forecast-accuracy.
        def _empty(res):
            if not res.get("cycles"):
                return True
            actuals = res.get("actuals") or {}
            return not actuals.get("values")

        if _empty(result) and model_version and not version_param:
            relaxed = await asyncio.to_thread(_fetch, model_name, None)
            if not _empty(relaxed):
                logger.info(
                    f"/forecast-evolution fallback for {name}: no cycles "
                    f"for {model_name!r} v={model_version!r}; widening to all versions."
                )
                result = relaxed
                model_fallback = {
                    "requested_model": model_name,
                    "requested_version": model_version,
                    "used_model": model_name,
                    "used_version": None,
                    "reason": "No forecasts logged for this version yet; showing all versions of this model.",
                }
                model_version = None
        if _empty(result) and model_name and not model_param:
            relaxed = await asyncio.to_thread(_fetch, None, None)
            if not _empty(relaxed):
                logger.info(
                    f"/forecast-evolution fallback for {name}: no cycles "
                    f"for {model_name!r}; widening to all models."
                )
                result = relaxed
                model_fallback = {
                    "requested_model": model_name,
                    "requested_version": model_version,
                    "used_model": None,
                    "used_version": None,
                    "reason": "No forecasts logged for this model in the selected window.",
                }
                model_name = None
                model_version = None

        result["model_name"] = model_name
        result["model_version"] = model_version
        if model_fallback:
            result["model_fallback"] = model_fallback
        return JSONResponse(content=result)

    @app.get("/experiment/{name}/forecast-stability")
    async def forecast_stability(name: str, request: Request):
        """
        Return cross-cycle self-consistency metrics for the model.

        Unlike forecast-accuracy (prediction vs actual), this measures
        how much predictions for the same future target swing from one
        issuance to the next — i.e. model stability, independent of
        whether predictions are correct.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        db = app.state.appstate.history_db
        if not db:
            return JSONResponse(content={"error": "Database not available"}, status_code=503)

        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"error": "Config not found"}, status_code=503)

        from ml_forecast_lab.config import load_config
        try:
            cfg = load_config(config_path)
            exp_cfg = next((e for e in cfg.experiments if e.name == name), None)
            if not exp_cfg:
                return JSONResponse(content={"error": "Experiment not in config"}, status_code=404)
        except Exception as e:
            return JSONResponse(content={"error": _safe_error(e)}, status_code=500)

        try:
            days = int(request.query_params.get("days", "30"))
        except (TypeError, ValueError):
            days = 30
        days = max(1, min(90, days))

        model_name, model_version = _resolve_model_filter(name, request)
        model_param = request.query_params.get("model")
        version_param = request.query_params.get("version")

        # Compute the UTC→HA-local hour offset so daily-total bucketing
        # aligns with the physical "_today" sensor reset (at HA local
        # midnight). Best-effort — falls back to UTC-day buckets when
        # zoneinfo or the HA TZ is unavailable. Approximated from
        # utcnow; off by 1h for ~24h per DST transition, no-op otherwise.
        day_offset_hours: Optional[float] = None
        tz_name = app.state.appstate.ha_time_zone
        if tz_name:
            try:
                from zoneinfo import ZoneInfo
                from datetime import datetime as _dt, timezone as _tz
                now_utc = _dt.now(_tz.utc)
                offset = now_utc.astimezone(ZoneInfo(tz_name)).utcoffset()
                if offset is not None:
                    day_offset_hours = offset.total_seconds() / 3600.0
            except Exception as e:
                logger.debug(f"Could not compute day offset for {tz_name}: {e}")

        def _fetch(mn, mv):
            return db.get_forecast_stability(
                name,
                max_age_days=days,
                source_is_cumulative=bool(exp_cfg.source_is_cumulative),
                model_name=mn, model_version=mv,
                day_offset_hours=day_offset_hours,
            )

        result = await asyncio.to_thread(_fetch, model_name, model_version)
        # v2.34.0: the version-widening + model-widening fallbacks
        # were removed. Stability is a per-cohort metric — pooling
        # cross-version cycles produces a number that doesn't reflect
        # any actual model's run-to-run swing. The new SQL also
        # self-protects (partitions per cohort, picks one winner per
        # target_dt), so the widening was redundant with the SQL fix
        # AND misleading on cohorts with <2 cycles. Cold-start cases
        # now return empty with `empty_reason: cohort_warming_up`
        # and the verdict-card chip handles the empty payload
        # gracefully (it already showed "need ≥2 cycles per moment"
        # in that branch — now it's the canonical path).
        if not result.get("per_timestep", {}).get("target_dt"):
            result["empty_reason"] = "cohort_warming_up"
        result["model_name"] = model_name
        result["model_version"] = model_version
        return JSONResponse(content=result)

    @app.post("/experiment/{name}/toggle-mode")
    async def toggle_mode(name: str):
        """
        Toggle experiment between lab and production mode.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        exp_status = app.state.appstate.experiment_statuses[name]
        old_mode = exp_status.mode
        new_mode = "production" if old_mode == "lab" else "lab"
        exp_status.mode = new_mode

        # Persist to YAML
        from ml_forecast_lab.config import save_experiment_field
        config_path = _find_config_path()
        if config_path:
            try:
                save_experiment_field(config_path, name, "mode", new_mode)
                if new_mode == "production" and exp_status.selected_model:
                    save_experiment_field(config_path, name, "production_model",
                                         exp_status.selected_model)
            except Exception as e:
                logger.warning(f"Failed to persist mode toggle: {e}")

        # When switching to production, trigger an immediate retrain so the
        # model gets cached and forecasts/sensors start publishing without
        # waiting for the next scheduled retrain cycle.
        if new_mode == "production" and app.state.appstate.retrain_callback:
            import asyncio as _aio
            app.state.appstate.spawn(app.state.appstate.retrain_callback(name))
            logger.info(f"Triggered immediate retrain for {name} after production toggle")

        return JSONResponse(
            content={
                "message": f"Switched {name} to {new_mode} mode",
                "experiment": name,
                "mode": new_mode,
            }
        )

    @app.post("/experiment/{name}/run-covariate-analysis")
    async def run_covariate_analysis(name: str, request: Request):
        """
        Trigger a covariate analysis (async, returns 202 Accepted).
        Body (optional): {"model": "lightgbm"} or {"model": "all"}
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        existing = app.state.appstate.covariate_analysis_results.get(name)
        if existing and existing.status == "running":
            return JSONResponse(
                status_code=409,
                content={"error": "Covariate analysis already running"},
            )

        # Parse optional model selection
        selected_model = "all"
        try:
            body = await request.json()
            selected_model = body.get("model", "all")
        except Exception:
            pass

        # Trigger via callback if available
        import asyncio
        if app.state.appstate.covariate_analysis_callback:
            app.state.appstate.spawn(app.state.appstate.covariate_analysis_callback(name, selected_model))

        return JSONResponse(
            status_code=202,
            content={
                "message": "Covariate analysis started",
                "experiment": name,
                "model": selected_model,
                "status": "running",
            },
        )

    @app.get("/experiment/{name}/covariate-analysis")
    async def get_covariate_analysis(name: str):
        """Get covariate analysis results as JSON."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        result = app.state.appstate.covariate_analysis_results.get(name)
        if not result:
            raise HTTPException(status_code=404, detail="No covariate analysis results")
        return result.model_dump()

    # ========== Tuning endpoints ==========

    @app.post("/experiment/{name}/run-tuning")
    async def run_tuning(name: str, request: Request):
        """Start hyperparameter tuning. Body: {model_name, n_trials?, strategy?}"""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        # Check if tuning is already running
        existing = app.state.appstate.tuning_results.get(name)
        if existing and existing.status == "running":
            return JSONResponse(
                status_code=409,
                content={"error": "Tuning already running for this experiment"},
            )

        try:
            body = await request.json()
        except Exception:
            body = {}

        model_name = body.get("model_name")
        if not model_name:
            return JSONResponse(
                status_code=400,
                content={"error": "model_name is required"},
            )

        n_trials = body.get("n_trials", 30)
        strategy = body.get("strategy", "tpe")

        # Get the parameter schema for this model
        schema = MODEL_PARAM_SCHEMA.get(model_name, {})
        if not schema:
            return JSONResponse(
                status_code=400,
                content={"error": f"No parameter schema for model: {model_name}"},
            )

        # Reject tuning on auto-models that have no searchable hyperparameters.
        # Without this guard the request would silently kick off an Optuna
        # study where every trial uses the same defaults — wasting compute
        # and looking like a hang to the user (the per-window AutoETS fits
        # are slow enough to mask the no-op).
        tunable = {k: v for k, v in schema.items() if v.get("tunable", True)}
        if not tunable:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        f"{model_name} has no tunable hyperparameters — "
                        f"the auto-model selects them internally. "
                        f"Set fixed parameters (e.g. seasonal_period) on "
                        f"the Models page instead."
                    )
                },
            )

        if not app.state.appstate.tuning_callback:
            raise HTTPException(status_code=501, detail="Tuning callback not registered")

        app.state.appstate.spawn(
            app.state.appstate.tuning_callback(
                name, model_name, n_trials, strategy, schema
            )
        )

        return JSONResponse(
            status_code=202,
            content={"message": "Tuning started", "model": model_name,
                      "n_trials": n_trials, "strategy": strategy},
        )

    @app.post("/experiment/{name}/run-tuning-all")
    async def run_tuning_all(name: str, request: Request):
        """Tune every enabled model sequentially.

        Body: {n_trials?, strategy?}. Each model's final TuningResult
        is accumulated in tune_all_results[experiment]; the existing
        single-model live-progress widgets continue to work because
        each iteration overwrites the standard tuning_results slot.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")
        existing = app.state.appstate.tuning_results.get(name)
        if existing and existing.status == "running":
            return JSONResponse(
                status_code=409,
                content={"error": "Tuning already running for this experiment"},
            )
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not app.state.appstate.tune_all_callback:
            raise HTTPException(status_code=501, detail="Tune-all callback not registered")
        app.state.appstate.spawn(
            app.state.appstate.tune_all_callback(
                name,
                int(body.get("n_trials", 30) or 30),
                body.get("strategy", "tpe"),
            )
        )
        return JSONResponse(
            status_code=202,
            content={"message": "Sweep started"},
        )

    @app.get("/experiment/{name}/tuning-all")
    async def get_tuning_all(name: str):
        """Return the per-model results of the most recent sweep."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")
        rows = app.state.appstate.tune_all_results.get(name, [])
        return JSONResponse(content={
            "experiment": name,
            "results": [r.model_dump() for r in rows],
        })

    @app.get("/experiment/{name}/tuning")
    async def get_tuning(name: str):
        """Get current tuning state (for polling)."""
        result = app.state.appstate.tuning_results.get(name)
        if not result:
            return JSONResponse(status_code=404, content={"error": "No tuning results"})
        return JSONResponse(content=result.model_dump())

    @app.post("/experiment/{name}/apply-tuning")
    async def apply_tuning(name: str):
        """Apply best tuning params and promote the model to production."""
        result = app.state.appstate.tuning_results.get(name)
        if not result or not result.best_params:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "No tuning results to apply"},
            )

        from ml_forecast_lab.config import (
            save_experiment_model_params,
            save_experiment_field,
        )
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(
                content={"success": False, "error": "Config file not found"},
            )

        try:
            # 1. Save tuned params per-experiment
            save_experiment_model_params(
                config_path, name, result.model_name, result.best_params
            )

            # 2. Promote the tuned model to production
            exp_status = app.state.appstate.experiment_statuses.get(name)
            if exp_status:
                exp_status.best_model = result.model_name
                exp_status.selected_model = result.model_name
                exp_status.mode = "production"

            benchmark = app.state.appstate.benchmark_results.get(name)
            if benchmark:
                for m in benchmark.models:
                    m.is_production = m.name == result.model_name

            save_experiment_field(config_path, name, "production_model", result.model_name)
            save_experiment_field(config_path, name, "mode", "production")

            # 3. Trigger an immediate retrain so the user doesn't have to
            # wait for the next scheduled retrain cycle to see the tuned
            # params take effect on the live forecast sensor.
            retrain_scheduled = False
            if app.state.appstate.retrain_callback:
                import asyncio as _aio
                app.state.appstate.spawn(app.state.appstate.retrain_callback(name))
                retrain_scheduled = True
                logger.info(f"Scheduled immediate retrain for {name} after apply-tuning")

            logger.info(
                f"Applied tuned params and promoted {result.model_name} "
                f"for experiment {name}"
            )
            return JSONResponse(content={
                "success": True, "model": result.model_name,
                "params": result.best_params, "promoted": True,
                "retraining": retrain_scheduled,
            })
        except Exception as e:
            logger.error(f"Failed to apply tuning: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

    @app.get("/experiment/{name}/results")
    async def get_results(name: str):
        """
        Get latest benchmark results as JSON.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        result = app.state.appstate.benchmark_results.get(name)
        if not result:
            raise HTTPException(status_code=404, detail="No benchmark results yet")

        return result.model_dump()

    @app.get("/experiment/{name}/benchmark-history")
    async def benchmark_history(name: str, request: Request):
        """Return up to N previous benchmark runs for this experiment.

        Used by the Results-tab "Previous runs" dropdown. Each run is
        a JSON-serialised BenchmarkResult; the latest sits at index 0.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")
        db = app.state.appstate.history_db
        if not db:
            return JSONResponse(content={"runs": []})
        try:
            limit = int(request.query_params.get("limit", "5"))
        except (TypeError, ValueError):
            limit = 5
        try:
            rows = await asyncio.to_thread(
                db.load_benchmark_history, name, max(1, min(50, limit)),
            )
        except Exception as e:
            return JSONResponse(content={"runs": [], "error": _safe_error(e)}, status_code=500)
        # Parse each saved blob enough to expose the headline fields the
        # dropdown needs (timestamp, winner, naive_baseline) without
        # forcing the client to JSON.parse twice.
        out = []
        for r in rows:
            entry: Dict[str, Any] = {"ran_at": r.get("ran_at")}
            try:
                blob = json.loads(r.get("data") or "{}")
                entry["timestamp"] = blob.get("timestamp")
                entry["best_model_name"] = blob.get("best_model_name")
                entry["naive_baseline_mae"] = blob.get("naive_baseline_mae")
                # The full models list is heavy; surface just MAE of the
                # winner so a sparkline of "best MAE over time" is cheap
                # to build client-side.
                if blob.get("best_model_name"):
                    for m in blob.get("models", []):
                        if m.get("name") == blob.get("best_model_name"):
                            entry["best_mae"] = (
                                m.get("mae", {}) or {}
                            ).get("mean")
                            break
                entry["data"] = r.get("data")
            except Exception:
                entry["data"] = r.get("data")
            out.append(entry)
        return JSONResponse(content={"runs": out})

    @app.get("/experiment/{name}/forecast")
    async def get_forecast(name: str):
        """
        Get latest forecast data for charting.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        forecast = app.state.appstate.forecast_data.get(name)
        if not forecast:
            raise HTTPException(status_code=404, detail="No forecast data yet")

        return forecast.model_dump()

    @app.get("/api/status")
    async def health_check() -> HealthStatus:
        """
        Health check and overall status.
        """
        experiments = app.state.appstate.experiment_statuses.values()
        lab_count = sum(1 for e in experiments if e.mode == "lab")
        prod_count = sum(1 for e in experiments if e.mode == "production")

        return HealthStatus(
            status="healthy",
            service="ml-forecast-lab",
            version=APP_VERSION,
            timestamp=datetime.now(timezone.utc).isoformat(),
            experiments_total=len(list(experiments)),
            experiments_lab=lab_count,
            experiments_production=prod_count,
        )

    # ========== Log Routes ==========

    LOG_FILE = Path("/data/ml_forecast_lab/logs/mlfl.log")

    @app.get("/log", response_class=Response)
    async def view_log(request: Request, lines: int = 500):
        """
        View recent log output in styled template.
        """
        log_text = ""
        for log_path in [LOG_FILE.with_suffix(".log.1"), LOG_FILE]:
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        log_text += f.read()
                except Exception as e:
                    log_text += f"\n[Error reading {log_path}: {e}]\n"

        all_lines = log_text.splitlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return templates.TemplateResponse(
            request=request,
            name="logs.html",
            context={
                "request": request,
                "base_path": _get_base_path(request),
                "active_page": "logs",
                "version": APP_VERSION,
                "log_content": "\n".join(tail),
            },
        )

    @app.get("/status")
    async def status_page(request: Request):
        """Redirect old status page to /system."""
        return RedirectResponse(url=f"{_get_base_path(request)}/system", status_code=301)

    @app.post("/api/models/toggle")
    async def toggle_model(request: Request):
        """
        Toggle a model on/off in the config. Updates models_enabled in mlfl.yaml.
        Body: {"model_name": "lstm", "enabled": true}
        """
        import yaml
        from ml_forecast_lab.config import atomic_yaml_write

        try:
            data = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        model_name = data.get("model_name")
        enabled = data.get("enabled", True)

        if not model_name:
            return JSONResponse(content={"success": False, "error": "model_name required"})

        config_path = _find_config_path()
        if not config_path or not config_path.exists():
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            # Update models_enabled in all experiments
            for exp in yaml_data.get("experiments", []):
                models = exp.get("models_enabled", [])
                if enabled and model_name not in models:
                    models.append(model_name)
                elif not enabled and model_name in models:
                    models.remove(model_name)
                exp["models_enabled"] = models

            atomic_yaml_write(config_path, yaml_data)

            logger.info(f"Model {model_name} {'enabled' if enabled else 'disabled'}")
            return JSONResponse(content={"success": True})

        except Exception as e:
            logger.error(f"Failed to toggle model: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

    @app.post("/api/experiment/{exp_name}/models/toggle")
    async def toggle_experiment_model(exp_name: str, request: Request):
        """
        Toggle a model on/off for a specific experiment.
        Body: {"model_name": "lstm", "enabled": true}
        """
        import yaml
        from ml_forecast_lab.config import atomic_yaml_write

        try:
            data = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        model_name = data.get("model_name")
        enabled = data.get("enabled", True)

        if not model_name:
            return JSONResponse(content={"success": False, "error": "model_name required"})

        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            # Update models_enabled only for the target experiment
            for exp in yaml_data.get("experiments", []):
                if exp.get("name") != exp_name:
                    continue
                models = exp.get("models_enabled", [])
                if enabled and model_name not in models:
                    models.append(model_name)
                elif not enabled and model_name in models:
                    models.remove(model_name)
                exp["models_enabled"] = models
                break

            atomic_yaml_write(config_path, yaml_data)

            logger.info(f"Model {model_name} {'enabled' if enabled else 'disabled'} for {exp_name}")
            return JSONResponse(content={"success": True})

        except Exception as e:
            logger.error(f"Failed to toggle model for {exp_name}: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

    @app.get("/settings")
    async def settings_page(request: Request):
        """Redirect old settings page to /system."""
        return RedirectResponse(url=f"{_get_base_path(request)}/system", status_code=301)

    @app.get("/system", response_class=Response)
    async def system_page(request: Request):
        """
        Unified system page: health, hardware, settings, experiments.
        Replaces the former separate /status and /settings pages.
        """
        import yaml

        experiment_statuses = list(app.state.appstate.experiment_statuses.values())
        lab_count = sum(1 for e in experiment_statuses if e.mode == "lab")
        prod_count = sum(1 for e in experiment_statuses if e.mode == "production")

        health = {
            "status": "healthy",
            "version": APP_VERSION,
            "experiments_total": len(experiment_statuses),
            "experiments_lab": lab_count,
            "experiments_production": prod_count,
        }

        # Hardware info
        cpu_count = os.cpu_count() or 4
        try:
            cpu_model = platform.processor() or platform.machine()
        except Exception:
            cpu_model = platform.machine()

        try:
            import psutil
            mem = psutil.virtual_memory()
            memory_total_gb = round(mem.total / (1024**3), 1)
            memory_used_gb = round(mem.used / (1024**3), 1)
            memory_percent = mem.percent
        except ImportError:
            try:
                with open("/proc/meminfo") as f:
                    meminfo = {line.split(":")[0]: int(line.split()[1]) for line in f if len(line.split()) >= 2}
                memory_total_gb = round(meminfo.get("MemTotal", 0) / (1024**2), 1)
                memory_used_gb = round((meminfo.get("MemTotal", 0) - meminfo.get("MemAvailable", 0)) / (1024**2), 1)
                memory_percent = round(memory_used_gb / max(memory_total_gb, 0.1) * 100, 1)
            except Exception:
                memory_total_gb = memory_used_gb = memory_percent = 0
        except Exception:
            memory_total_gb = memory_used_gb = memory_percent = 0

        try:
            disk = shutil.disk_usage("/data")
            disk_total_gb = round(disk.total / (1024**3), 1)
            disk_used_gb = round(disk.used / (1024**3), 1)
            disk_percent = round(disk.used / disk.total * 100, 1)
        except Exception:
            disk_total_gb = disk_used_gb = disk_percent = 0

        system_info = {
            "cpu_cores": cpu_count,
            "cpu_model": cpu_model,
            "memory_total_gb": memory_total_gb,
            "memory_used_gb": memory_used_gb,
            "memory_percent": memory_percent,
            "disk_total_gb": disk_total_gb,
            "disk_used_gb": disk_used_gb,
            "disk_percent": disk_percent,
            "applied_cpu_threads": app.state.appstate.applied_cpu_threads,
            "applied_nice": app.state.appstate.applied_nice,
        }

        # Config data
        config_data = {
            "forecast_every_minutes": 30,
            "retrain_every_hours": 24.0,
            "update_every_minutes": 30,
            "timezone": "UTC",
            "cpu_cores": 0,
            "nice_priority": 10,
        }
        config_path_str = "unknown"
        experiment_configs = []
        cp = _find_config_path()
        if cp:
            config_path_str = str(cp)
        try:
            from ml_forecast_lab.config import load_config as _load_config
            cfg = _load_config(config_path_str)
            config_data = {
                "forecast_every_minutes": cfg.forecast_every_minutes,
                "retrain_every_hours": cfg.retrain_every_hours,
                "update_every_minutes": cfg.forecast_every_minutes,
                "timezone": cfg.timezone,
                "cpu_cores": cfg.cpu_cores,
                "nice_priority": cfg.nice_priority,
                "external_forecast_retention_days": cfg.external_forecast_retention_days,
            }
            experiment_configs = cfg.experiments
        except Exception:
            pass

        # Developer-mode branch overlay state (maintainer aid; default off).
        # The card and its endpoints are inert unless developer_mode is on,
        # so normal users never see this.
        from ml_forecast_lab import dev_branch
        developer_mode = dev_branch.developer_mode_enabled()
        dev_installed = dev_branch.active_status() if developer_mode else None

        return templates.TemplateResponse(
            request=request,
            name="system.html",
            context={
                "request": request,
                "base_path": _get_base_path(request),
                "active_page": "system",
                "version": APP_VERSION,
                "health": health,
                "system": system_info,
                "config": config_data,
                "config_path": config_path_str,
                "experiment_statuses": experiment_statuses,
                "experiment_configs": experiment_configs,
                "developer_mode": developer_mode,
                "dev_installed": dev_installed,
                "dev_running": dev_branch.is_overlay_running(),
                "dev_compatible": dev_branch.overlay_is_compatible(),
                "dev_repo": f"{dev_branch.REPO_OWNER}/{dev_branch.REPO_NAME}",
            },
        )

    @app.post("/api/settings")
    async def save_settings(request: Request):
        """
        Save settings back to mlfl.yaml.
        """
        import yaml

        try:
            data = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        config_path = _find_config_path()

        if not config_path or not config_path.exists():
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        from ml_forecast_lab.config import atomic_yaml_write

        try:
            # Read existing YAML
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            # Update fields
            if "forecast_every_minutes" in data:
                yaml_data["forecast_every_minutes"] = int(data["forecast_every_minutes"])
                yaml_data["update_every_minutes"] = int(data["forecast_every_minutes"])
            elif "update_every_minutes" in data:
                yaml_data["update_every_minutes"] = int(data["update_every_minutes"])
            if "retrain_every_hours" in data:
                yaml_data["retrain_every_hours"] = float(data["retrain_every_hours"])
            if "timezone" in data:
                yaml_data["timezone"] = str(data["timezone"])
            if "cpu_cores" in data:
                yaml_data["cpu_cores"] = int(data["cpu_cores"])
            if "nice_priority" in data:
                yaml_data["nice_priority"] = int(data["nice_priority"])
            if "external_forecast_retention_days" in data:
                _r = int(data["external_forecast_retention_days"])
                if _r < 1:
                    return JSONResponse(content={
                        "success": False,
                        "error": "external_forecast_retention_days must be >= 1",
                    })
                yaml_data["external_forecast_retention_days"] = _r

            atomic_yaml_write(config_path, yaml_data)

            logger.info(f"Settings saved to {config_path}")
            return JSONResponse(content={"success": True})

        except Exception as e:
            logger.error(f"Failed to save settings: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

    # ===== Developer-mode branch overlay (maintainer aid; default off) =====
    #
    # These endpoints let a developer run an arbitrary branch of the app's
    # own repository in place of the bundled release. They are gated behind
    # the `developer_mode` add-on option: when it is off, every endpoint
    # 404s and the System-tab card is not rendered, so the capability is
    # invisible and inert for normal users. See ml_forecast_lab/dev_branch.py.

    async def _restart_addon() -> bool:
        """Ask the Supervisor to restart this add-on. Best-effort.

        Returns True if the Supervisor accepted the request. Requires
        `hassio_api: true` in config.yaml and the SUPERVISOR_TOKEN env var
        (both present in the add-on runtime).
        """
        import aiohttp

        token = os.environ.get("SUPERVISOR_TOKEN", "")
        if not token:
            logger.warning("No SUPERVISOR_TOKEN; cannot self-restart.")
            return False
        url = "http://supervisor/addons/self/restart"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status in (200, 201):
                        return True
                    logger.error(
                        f"Supervisor restart returned {resp.status}: "
                        f"{(await resp.text())[:200]}"
                    )
                    return False
        except Exception as e:  # noqa: BLE001
            logger.error(f"Supervisor restart request failed: {e}")
            return False

    def _can_self_restart() -> bool:
        """Whether a Supervisor self-restart is available (token present)."""
        return bool(os.environ.get("SUPERVISOR_TOKEN"))

    async def _deferred_restart() -> None:
        """Restart the add-on a beat after the HTTP response is flushed.

        Run as a Starlette BackgroundTask (which executes only after the
        response is sent) so the JSON payload reaches the browser *before*
        the Supervisor kills the container. Restarting inline — before
        returning — races the response: the container dies mid-flight and
        the client sees a dropped connection, which surfaces as a JSON
        parse error rather than the success message. The short sleep adds
        margin for the response bytes to drain.
        """
        await asyncio.sleep(1.0)
        await _restart_addon()

    def _require_dev_mode():
        """Raise 404 unless developer_mode is enabled.

        404 (not 403) so the endpoints are indistinguishable from
        nonexistent when the option is off — nothing hints the capability
        exists.
        """
        from ml_forecast_lab import dev_branch
        if not dev_branch.developer_mode_enabled():
            raise HTTPException(status_code=404, detail="Not found")

    @app.get("/api/system/dev/branches")
    async def dev_list_branches():
        """List this repo's OPEN-PR branches to populate the System-tab dropdown.

        Only branches that currently back an open pull request are offered
        (plus the default branch), so merged/closed work drops off on its
        own. As a side effect, an installed overlay whose branch no longer
        has an open PR is treated as stale: its files are removed from the
        Pi so the next restart returns to the bundled release.

        Best-effort: on any GitHub failure (rate limit, no network) returns
        ``success: false`` with a message so the UI can fall back to the
        manual branch-name field rather than breaking — and crucially, no
        overlay is auto-removed when the PR list couldn't be fetched.
        """
        _require_dev_mode()
        from ml_forecast_lab import dev_branch

        token = os.environ.get("GITHUB_TOKEN", "") or None
        try:
            all_branches = await dev_branch.list_repo_branches(token=token)
            open_prs = await dev_branch.list_open_pr_branches(token=token)
        except dev_branch.DevBranchError as e:
            return JSONResponse(content={"success": False, "error": str(e),
                                         "branches": []})
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Branch listing failed: {e}")
            return JSONResponse(content={"success": False,
                                         "error": _safe_error(e), "branches": []})

        branches = dev_branch.compose_dev_branch_list(all_branches, open_prs)

        # Auto-cleanup: an installed overlay whose branch no longer has an
        # open PR is stale — drop its files from /data so the Pi reverts to
        # the bundled release on the next restart. Runs only after a
        # successful PR fetch (above), so a transient API failure can never
        # nuke a valid overlay.
        removed_overlay = None
        stale_overlay = None
        active = dev_branch.active_status() or {}
        active_branch = active.get("branch")
        if dev_branch.branch_is_closed(active_branch, open_prs):
            if dev_branch.is_overlay_running():
                # The overlay is not just installed, it is the code currently
                # executing: the s6 init script puts DEV_SRC on PYTHONPATH, so
                # this very module — and the Jinja template and static dirs
                # resolved at startup — live inside the directory revert()
                # deletes. Removing it underneath the running process makes the
                # next template render raise TemplateNotFound and every
                # function-local import raise ModuleNotFoundError (the imported
                # package's __path__ is pinned to the deleted directory, so it
                # does not fall back to the bundled /app copy). The page that
                # would tell the user to reload is itself the request that 500s.
                #
                # The manual Revert endpoint pairs the same delete with a
                # restart, which is why it is safe there. This path fires
                # unconditionally on page load, so it reports instead of acting.
                stale_overlay = {"branch": active_branch}
                logger.info(
                    "Dev overlay for %r has no open PR, but it is the running "
                    "code — leaving it in place; use Revert to restart onto "
                    "the bundled release.", active_branch,
                )
            else:
                try:
                    if dev_branch.revert():
                        removed_overlay = {"branch": active_branch}
                        active = {}
                        logger.warning(
                            "Auto-removed dev overlay for closed branch %r "
                            "(no open PR); Pi reverts to bundled on next restart.",
                            active_branch,
                        )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Auto-remove of closed overlay failed: {e}")

        # PR metadata (number/title) for the branches we kept, for nicer
        # dropdown labels.
        prs = {b: open_prs[b] for b in branches if b in open_prs}
        return JSONResponse(content={
            "success": True,
            "branches": branches,
            "pull_requests": prs,
            "current": active.get("branch"),
            "removed_overlay": removed_overlay,
            "stale_overlay": stale_overlay,
        })

    @app.post("/api/system/dev/install-branch")
    async def dev_install_branch(request: Request):
        """Fetch a branch of this repo and stage it as the boot overlay.

        On success the add-on restarts itself; on the next boot the s6
        script runs the overlaid branch instead of the bundled package.
        """
        _require_dev_mode()
        from ml_forecast_lab import dev_branch

        try:
            data = await request.json()
        except Exception:
            return JSONResponse(status_code=400,
                                content={"success": False, "error": "Invalid JSON"})
        branch = (data or {}).get("branch", "")
        try:
            branch = dev_branch.validate_branch(branch)
        except dev_branch.DevBranchError as e:
            return JSONResponse(status_code=400,
                                content={"success": False, "error": str(e)})

        import aiohttp

        # Resolve the branch head SHA (best-effort; non-fatal for install).
        sha = ""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    dev_branch.commit_api_url(branch),
                    headers={"Accept": "application/vnd.github+json"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        sha = (await resp.json()).get("sha", "") or ""
                    elif resp.status == 404:
                        return JSONResponse(
                            status_code=404,
                            content={"success": False,
                                     "error": f"Branch {branch!r} not found in "
                                              f"{dev_branch.REPO_OWNER}/{dev_branch.REPO_NAME}."},
                        )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not resolve SHA for {branch!r}: {e}")

        # Download the branch tarball.
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    dev_branch.tarball_url(branch),
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        return JSONResponse(
                            status_code=502,
                            content={"success": False,
                                     "error": f"Download failed (HTTP {resp.status}). "
                                              f"Check the branch name and that the Pi "
                                              f"has internet access."},
                        )
                    raw = await resp.read()
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=502,
                                content={"success": False,
                                         "error": f"Download failed: {_safe_error(e)}"})

        try:
            status = dev_branch.install_from_tarball_bytes(branch, raw, sha=sha)
        except dev_branch.DevBranchError as e:
            return JSONResponse(status_code=400,
                                content={"success": False, "error": str(e)})
        except Exception as e:  # noqa: BLE001
            logger.error(f"Overlay install failed: {e}", exc_info=True)
            return JSONResponse(status_code=500,
                                content={"success": False, "error": _safe_error(e)})

        # Respond first, restart after (BackgroundTask) — see _deferred_restart.
        restarting = _can_self_restart()
        return JSONResponse(
            content={
                "success": True,
                "status": status,
                "restarting": restarting,
                "message": (
                    f"Installed {branch}@{status['sha_short'] or '?'}. "
                    + ("Restarting the add-on now…" if restarting else
                       "Restart the add-on to run it (auto-restart unavailable).")
                ),
            },
            background=BackgroundTask(_deferred_restart) if restarting else None,
        )

    @app.get("/api/system/dev/install-stream")
    async def dev_install_stream(request: Request):
        """Fetch a branch, install any new dependencies, and restart —
        streaming live progress as Server-Sent Events.

        Backs the System-tab Developer card so the user sees download and
        pip progress in real time. Branches that add new Python
        dependencies (e.g. the foundation-model backends needing
        chronos-forecasting / granite-tsfm) get those installed into the
        live environment *before* the restart; on 32-bit ARM (no wheels)
        the dependency step is skipped with a warning. The install reuses
        the image's existing packages, so only genuinely-new distributions
        are fetched and core deps like torch are never disturbed.
        """
        _require_dev_mode()
        from ml_forecast_lab import dev_branch

        branch_raw = request.query_params.get("branch", "")

        def _sse(obj) -> str:
            return f"data: {json.dumps(obj)}\n\n"

        async def _gen():
            import aiohttp
            try:
                try:
                    branch = dev_branch.validate_branch(branch_raw)
                except dev_branch.DevBranchError as e:
                    yield _sse({"type": "error", "message": str(e)})
                    return

                yield _sse({"type": "step", "message": f"Resolving {branch}…"})
                sha = ""
                try:
                    async with aiohttp.ClientSession() as s:
                        async with s.get(
                            dev_branch.commit_api_url(branch),
                            headers={"Accept": "application/vnd.github+json"},
                            timeout=aiohttp.ClientTimeout(total=30),
                        ) as r:
                            if r.status == 200:
                                sha = (await r.json()).get("sha", "") or ""
                            elif r.status == 404:
                                yield _sse({"type": "error", "message":
                                            f"Branch {branch!r} not found in "
                                            f"{dev_branch.REPO_OWNER}/{dev_branch.REPO_NAME}."})
                                return
                except Exception as e:  # noqa: BLE001
                    yield _sse({"type": "log", "message": f"(could not resolve sha: {e})"})

                yield _sse({"type": "step", "message": "Downloading branch…"})
                try:
                    async with aiohttp.ClientSession() as s:
                        async with s.get(
                            dev_branch.tarball_url(branch),
                            timeout=aiohttp.ClientTimeout(total=180),
                        ) as r:
                            if r.status != 200:
                                yield _sse({"type": "error", "message":
                                            f"Download failed (HTTP {r.status}). Check the "
                                            f"branch name and the Pi's internet access."})
                                return
                            raw = await r.read()
                except Exception as e:  # noqa: BLE001
                    yield _sse({"type": "error",
                                "message": f"Download failed: {_safe_error(e)}"})
                    return

                yield _sse({"type": "step", "message": "Extracting overlay…"})
                try:
                    status = dev_branch.install_from_tarball_bytes(branch, raw, sha=sha)
                except dev_branch.DevBranchError as e:
                    yield _sse({"type": "error", "message": str(e)})
                    return
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Overlay install failed: {e}", exc_info=True)
                    yield _sse({"type": "error", "message": _safe_error(e)})
                    return
                yield _sse({"type": "log", "message":
                            f"Installed overlay {branch}@{status['sha_short'] or '?'}"})

                # Install any dependencies the branch adds but the image lacks.
                new_reqs = dev_branch.new_requirements(
                    dev_branch.read_branch_requirements())
                if new_reqs and not dev_branch.dependency_install_supported():
                    import platform
                    yield _sse({"type": "log", "message":
                                f"Skipping dependency install on {platform.machine()} "
                                f"(no 32-bit ARM wheels): {', '.join(new_reqs)}. "
                                f"Those backends will be unavailable."})
                elif new_reqs:
                    yield _sse({"type": "step", "message":
                                f"Installing {len(new_reqs)} new "
                                f"{'dependencies' if len(new_reqs) != 1 else 'dependency'}… "
                                f"(can take a few minutes)"})
                    yield _sse({"type": "log", "message": "$ pip install " + " ".join(new_reqs)})
                    rc = -1
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            *dev_branch.pip_install_command(new_reqs),
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.STDOUT,
                        )
                        async for bline in proc.stdout:
                            line = bline.decode("utf-8", "replace").rstrip()
                            if line:
                                yield _sse({"type": "log", "message": line})
                        rc = await proc.wait()
                    except Exception as e:  # noqa: BLE001
                        yield _sse({"type": "log", "message": f"pip error: {_safe_error(e)}"})
                    if rc != 0:
                        yield _sse({"type": "log", "message":
                                    f"Dependency install exited with code {rc}; those backends "
                                    f"may be unavailable, but the branch will still run."})
                    else:
                        yield _sse({"type": "log", "message": "Dependencies installed."})
                else:
                    yield _sse({"type": "log", "message": "No new dependencies to install."})

                # Restart so the overlay (and any new deps) take effect.
                if _can_self_restart():
                    yield _sse({"type": "restarting", "message":
                                f"Installed {branch}@{status['sha_short'] or '?'}. Restarting…"})
                    await asyncio.sleep(1.5)  # flush the event before the container dies
                    await _restart_addon()
                else:
                    yield _sse({"type": "done", "message":
                                f"Installed {branch}@{status['sha_short'] or '?'}. "
                                f"Restart the add-on to run it (auto-restart unavailable)."})
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001
                logger.error(f"install-stream failed: {e}", exc_info=True)
                yield _sse({"type": "error", "message": _safe_error(e)})

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/system/dev/revert")
    async def dev_revert(request: Request):
        """Remove the overlay and restart back onto the bundled release."""
        _require_dev_mode()
        from ml_forecast_lab import dev_branch
        existed = dev_branch.revert()
        # Respond first, restart after (BackgroundTask) — see _deferred_restart.
        restarting = existed and _can_self_restart()
        return JSONResponse(
            content={
                "success": True,
                "reverted": existed,
                "restarting": restarting,
                "message": (
                    ("Reverted to the bundled release. Restarting…" if restarting
                     else "Reverted to the bundled release. Restart to apply.")
                    if existed else "No developer overlay was installed."
                ),
            },
            background=BackgroundTask(_deferred_restart) if restarting else None,
        )

    @app.post("/api/experiment-settings")
    async def save_experiment_settings(request: Request):
        """
        Save per-experiment training settings (CV strategy, folds, recency weighting).

        Persists changes to mlfl.yaml without requiring add-on restart.
        Changes take effect on next training run.
        """
        import yaml

        try:
            data = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        exp_name = data.get("experiment")
        if not exp_name:
            return JSONResponse(content={"success": False, "error": "Missing experiment name"})

        # Allowed editable fields and their types/validators
        editable = {
            "cv_strategy": lambda v: v if v in ("walk_forward", "sliding_window") else None,
            "cv_folds": lambda v: int(v) if int(v) >= 2 else None,
            "cv_embargo_periods": lambda v: int(v) if int(v) >= 0 else None,
            "recency_half_life_days": lambda v: float(v) if float(v) >= 0 else None,
            "days_history": lambda v: int(v) if int(v) >= 1 else None,
            "interval_minutes": lambda v: int(v) if int(v) >= 1 else None,
            "future_periods": lambda v: int(v) if int(v) >= 1 else None,
            "source_is_cumulative": lambda v: bool(v),
            "reset_daily": lambda v: bool(v),
            "target_is_nonnegative": lambda v: bool(v),
            "debug_save_training_dumps": lambda v: bool(v),
            # idle_value: numeric (any sign, idle could conceivably be a
            # negative offset for e.g. heat-pump COP-style sensors), or
            # None to disable the fill. An empty string from the UI
            # input maps to None so the user can clear the override.
            "idle_value": lambda v: (
                None if v is None or v == "" else float(v)
            ),
            "log_transform": lambda v: bool(v),
            "forecast_every_minutes": lambda v: int(v) if int(v) >= 1 else None,
            "retrain_every_hours": lambda v: float(v) if float(v) >= 0.1 else None,
            "production_metric": lambda v: v if v in (
                "mae", "rmse", "mase", "seasonal_mase",
                "peak_weighted_mae", "pinball_q90",
            ) else None,
            "loss_fn": lambda v: v if v in ("mse", "mae", "huber", "tweedie") else None,
            "optimiser": lambda v: v if v in ("adamw", "adam") else None,
            # v2.41.0: daily_loss_weight / loss_balance validators
            # removed. The fields were inert since v2.40.14 but the API
            # still accepted and persisted them to YAML — exactly the
            # silent-misconfiguration pattern audit F11 flagged. POSTs
            # carrying them now get the standard unknown-field error.
            # v2.40.12: per-experiment early-stopping patience. null →
            # each backend uses its constructor default (20 neural, 50
            # tree). Set 1..500 to override uniformly.
            "patience": lambda v: int(v) if 1 <= int(v) <= 500 else None,
            "max_increment": lambda v: float(v) if float(v) > 0 else None,
            "conformal_coverage": lambda v: float(v) if 0.5 <= float(v) <= 0.99 else None,
            "country": lambda v: (str(v).strip().upper() or None) if v else None,
            "gap_handling": lambda v: v if v in ("ffill", "interpolate", "mask") else None,
            "gap_max_minutes": lambda v: int(v) if int(v) >= 1 else None,
            "outlier_method": lambda v: v if v in ("quantile", "mad", "off") else None,
            "outlier_quantile": lambda v: float(v) if 0.5 < float(v) < 1.0 else None,
            "outlier_lower": lambda v: v if v in ("auto", "zero", "symmetric", "off") else None,
            "include_sun_elevation": lambda v: bool(v),
            "include_clear_sky_irradiance": lambda v: bool(v),
            "output_activation": lambda v: v if v in (
                "auto", "linear", "softplus", "relu", "exp", "sigmoid", "zscore"
            ) else None,
            # External forecasts are managed as a list via the dedicated
            # /add-external-forecast and /remove-external-forecast endpoints,
            # not through this scalar-field settings form.
        }

        # Fields where None/null means "use global default" (valid, not an error)
        nullable_fields = {"forecast_every_minutes", "retrain_every_hours", "max_increment", "country", "patience"}

        updates = {}
        for field, validator in editable.items():
            if field in data:
                raw = data[field]
                if raw is None and field in nullable_fields:
                    updates[field] = None
                    continue
                try:
                    val = validator(raw)
                    if val is None:
                        return JSONResponse(content={
                            "success": False, "error": f"Invalid value for {field}"
                        })
                    updates[field] = val
                except (ValueError, TypeError) as e:
                    return JSONResponse(content={
                        "success": False, "error": f"Invalid {field}: {e}"
                    })

        if not updates:
            return JSONResponse(content={"success": False, "error": "No valid fields to update"})

        config_path = _find_config_path()

        if not config_path or not config_path.exists():
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        from ml_forecast_lab.config import atomic_yaml_write

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            # Find the experiment in the YAML
            experiments = yaml_data.get("experiments", [])
            found = False
            for exp in experiments:
                if exp.get("name") == exp_name:
                    for k, v in updates.items():
                        if v is None:
                            exp.pop(k, None)  # remove → falls back to global default
                        else:
                            exp[k] = v
                    found = True
                    break

            if not found:
                return JSONResponse(content={
                    "success": False, "error": f"Experiment '{exp_name}' not found in config"
                })

            atomic_yaml_write(config_path, yaml_data)

            # Also update the in-memory config if possible
            try:
                cfg = _load_config()
                for exp_cfg in cfg.experiments:
                    if exp_cfg.name == exp_name:
                        for k, v in updates.items():
                            setattr(exp_cfg, k, v)
                        break
            except Exception:
                pass  # Config will reload on next training run

            logger.info(f"Experiment '{exp_name}' settings updated: {updates}")
            return JSONResponse(content={"success": True})

        except Exception as e:
            logger.error(f"Failed to save experiment settings: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": _safe_error(e)})

    @app.get("/api/log")
    async def api_log(
        lines: int = 200,
        level: str = "all",
        search: str = "",
    ):
        """
        JSON log API with filtering.

        Parameters:
            lines: max lines to return (default 200)
            level: 'all', 'info', 'warning', 'error' (filters by level)
            search: text search filter
        """
        log_text = ""
        for log_path in [LOG_FILE.with_suffix(".log.1"), LOG_FILE]:
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        log_text += f.read()
                except Exception:
                    pass

        all_lines = log_text.splitlines()

        # Filter by level
        if level != "all":
            level_upper = level.upper()
            all_lines = [
                l for l in all_lines if f"- {level_upper} -" in l.upper()
            ]

        # Filter by search term
        if search:
            search_lower = search.lower()
            all_lines = [l for l in all_lines if search_lower in l.lower()]

        # Tail
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return {
            "total_lines": len(all_lines),
            "returned_lines": len(tail),
            "lines": tail,
        }

    @app.get("/debug_log", response_class=Response)
    async def download_log():
        """
        Download the full current log file.
        """
        if not LOG_FILE.exists():
            raise HTTPException(status_code=404, detail="No log file found")

        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return Response(
            content=content,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=mlfl.log"},
        )


    # ========== Training Tab: SSE + Routes ==========

    @app.get("/training")
    async def training_page(request: Request):
        """Redirect old Training tab URL to Dashboard (preserved for bookmarks)."""
        return RedirectResponse(url=f"{_get_base_path(request)}/", status_code=301)

    @app.get("/experiment/{name}/training-stream")
    async def training_stream(name: str, request: Request):
        """
        Server-Sent Events endpoint for live training metrics.

        Streams TrainingEvent objects as JSON. Replays history on connect
        so late-joining clients catch up, then streams live events until
        a pipeline_end event is received.

        Pass ?no_replay=1 to skip history replay (e.g. when the client
        already replayed via the /api/training/history endpoint).
        """
        import asyncio as _aio
        from ml_forecast_lab.training_events import TrainingEventBus

        skip_replay = request.query_params.get("no_replay") == "1"
        event_bus = TrainingEventBus.get_instance()
        loop = _aio.get_running_loop()
        q = event_bus.subscribe(name, loop)

        async def _generate():
            try:
                # Replay history for reconnecting clients (unless already done)
                if not skip_replay:
                    for ev in event_bus.get_history(name):
                        yield f"data: {json.dumps(ev.to_dict())}\n\n"

                # Stream live events
                while True:
                    try:
                        event = await _aio.wait_for(q.get(), timeout=30.0)
                    except _aio.TimeoutError:
                        # Send keep-alive comment
                        yield ": keepalive\n\n"
                        continue

                    yield f"data: {json.dumps(event.to_dict())}\n\n"

                    if event.event_type == "pipeline_end":
                        break
            except _aio.CancelledError:
                pass
            finally:
                event_bus.unsubscribe(name, q)

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/experiment/{name}/run-pipeline")
    async def run_pipeline(name: str, request: Request):
        """
        Trigger the full training pipeline for an experiment.

        Accepts optional JSON body:
          {"steps": ["benchmark"]}                     — default: benchmark only
          {"steps": ["benchmark", "covariate_analysis"]} — benchmark then covariate analysis

        Experiments are queued and run one at a time to avoid memory
        exhaustion on constrained hardware (e.g. RPi).
        Returns 202 Accepted. Progress is streamed via the SSE endpoint.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        appstate = app.state.appstate

        if appstate.is_benchmark_running(name):
            return JSONResponse(
                status_code=409,
                content={"error": "Pipeline already running for this experiment"},
            )
        if appstate.get_queue_position(name) > 0:
            return JSONResponse(
                status_code=409,
                content={"error": "Experiment is already queued"},
            )

        steps = ["benchmark"]
        try:
            body = await request.json()
            steps = body.get("steps", ["benchmark"])
        except Exception:
            pass

        if not getattr(appstate, 'benchmark_callback', None):
            raise HTTPException(status_code=501, detail="Benchmark callback not registered")

        # Add to queue
        appstate.training_queue.append({"name": name, "steps": steps})
        position = appstate.get_queue_position(name)
        logger.info(f"Queued pipeline for {name} (position {position})")

        # Kick the queue processor
        import asyncio as _aio
        _aio.ensure_future(_process_training_queue(app))

        return JSONResponse(
            status_code=202,
            content={
                "message": "Pipeline queued" if position > 1 else "Pipeline started",
                "experiment": name,
                "steps": steps,
                "queue_position": position,
                "status": "queued" if position > 1 else "running",
            },
        )

    async def _process_training_queue(app):
        """Process the training queue one experiment at a time."""
        appstate = app.state.appstate
        if appstate._queue_processing:
            return  # Another processor is already running
        appstate._queue_processing = True
        try:
            while appstate.training_queue:
                item = appstate.training_queue.pop(0)
                name = item["name"]
                steps = item["steps"]

                # Skip if experiment was deleted while queued
                if name not in appstate.experiment_statuses:
                    continue

                from ml_forecast_lab.training_events import TrainingEventBus
                TrainingEventBus.get_instance().clear_history(name)
                appstate.start_benchmark(name)

                import asyncio as _aio

                async def _run_pipeline(exp_name, exp_steps):
                    try:
                        if "benchmark" in exp_steps:
                            await appstate.benchmark_callback(exp_name)
                        if "covariate_analysis" in exp_steps and appstate.covariate_analysis_callback:
                            await appstate.covariate_analysis_callback(exp_name, "all")
                    except Exception as e:
                        logger.error(f"Pipeline failed for {exp_name}: {e}", exc_info=True)
                    finally:
                        appstate.end_benchmark(exp_name)

                task = _aio.create_task(_run_pipeline(name, steps))
                appstate._pipeline_tasks[name] = task
                task.add_done_callback(
                    lambda t, n=name: appstate._pipeline_tasks.pop(n, None)
                )

                # Wait for this pipeline to finish before starting the next
                try:
                    await task
                except _aio.CancelledError:
                    logger.info(f"Pipeline for {name} was cancelled, moving to next in queue")
                except Exception:
                    pass
        finally:
            appstate._queue_processing = False

    @app.get("/api/training/history/{name}")
    async def training_history(name: str):
        """Return all training events for an experiment as JSON."""
        from ml_forecast_lab.training_events import TrainingEventBus

        event_bus = TrainingEventBus.get_instance()
        history = event_bus.get_history(name)
        return JSONResponse(content=[ev.to_dict() for ev in history])

    return app
