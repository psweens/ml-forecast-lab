"""
Configuration module for ML Forecast Lab.

Provides dataclasses for experiment configuration, covariate specification,
and application settings, with YAML loading capabilities.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


def _atomic_yaml_write(config_path: Path, data: dict) -> None:
    """Write *data* to *config_path* atomically via write-to-temp + rename.

    ``open('w')`` truncates a file **immediately**, so if the process is
    killed before ``yaml.dump()`` completes (e.g. OOM SIGKILL during
    tuning), the config file is left empty or corrupt.  Writing to a
    temporary file in the **same directory** and then calling
    ``os.replace()`` is an atomic operation on POSIX — the file either
    contains the old data or the new data, never a half-written state.
    """
    config_path = Path(config_path)
    dir_ = config_path.parent
    try:
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix='.tmp', prefix='.mlfl_')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=False)
        os.replace(tmp, config_path)
    except BaseException:
        # Clean up the temp file if something goes wrong before rename
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@dataclass
class SubtractCfg:
    """Configuration for a single load-subtract sensor.

    Represents one signal to subtract from the target before training /
    forecasting — e.g. EV charging or an iBoost solar-divert dump to remove
    those contributions from a whole-house load signal so the model learns
    only the baseline household pattern.

    Robustness fields encode the checklist:

    - ``source`` makes cumulative semantics explicit (no inference)
    - ``on_missing`` decides what to do with ``unavailable`` / ``unknown`` /
      gap rows — **never silently coerced to 0** unless you ask for it
    - ``scale`` lets you fix a unit mismatch (e.g. Wh → kWh = 0.001) at
      config time rather than hiding it in preprocessing
    - ``max_fraction_of_load`` + ``max_fraction_violation_pct`` are the
      fail-fast guard for unit bugs or the subtract sensor measuring more
      than the parent
    """

    entity_id: str
    """Home Assistant sensor entity_id to subtract."""

    source: str = "auto"
    """Cumulative semantics of the sensor. One of:

    - ``cumulative_daily``: resets every day (e.g. ``*_today`` energy sensors)
    - ``cumulative_monotonic``: monotonically increasing (e.g. utility meter)
    - ``interval``: already per-interval values (kWh / interval or W)
    - ``auto``: infer from the target's ``source_is_cumulative`` /
      ``reset_daily`` — only safe when the subtract sensor has the same
      semantics as the parent load sensor

    Pick explicitly; ``auto`` is the convenience default but leaves a bug
    surface when you mix sensor types."""

    on_missing: str = "zero"
    """Policy for unavailable / gap rows in the subtract sensor:

    - ``zero``: fill with 0.0 (common for EV / iBoost — missing often means
      "wasn't on"). Audit log records the filled count so you can spot
      pathological cases.
    - ``drop``: drop the row from the load series. Shrinks the training
      set but avoids fabricated zeros if "missing" does not mean "idle".
    - ``error``: raise ``ValueError``. Strictest — use when any gap is a
      data-pipeline bug you want to surface.
    """

    scale: Optional[float] = None
    """Optional multiplier applied before subtraction — use to fix unit
    mismatches (e.g. Wh → kWh: ``scale: 0.001``). ``None`` = no scaling."""

    max_fraction_of_load: float = 1.0
    """Per-row ceiling on ``subtract / load``. Rows where the subtract total
    exceeds this fraction of the parent load are counted as violations.
    Set > 1.0 only if you have a legitimate reason to believe subtract can
    momentarily exceed parent (net metering, sensor latency, etc.)."""

    max_fraction_violation_pct: float = 5.0
    """Maximum percentage of rows allowed to violate ``max_fraction_of_load``
    before ``apply_load_subtract`` raises. This is the fail-fast guard for
    unit bugs and double-counted signals — a 0.1 % violation rate is noise,
    a 50 % rate almost certainly means Wh vs kWh."""

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_sources = {
            "auto", "cumulative_daily", "cumulative_monotonic", "interval",
        }
        if self.source not in valid_sources:
            raise ValueError(
                f"source must be one of {sorted(valid_sources)}, "
                f"got {self.source!r}"
            )
        valid_on_missing = {"zero", "drop", "error"}
        if self.on_missing not in valid_on_missing:
            raise ValueError(
                f"on_missing must be one of {sorted(valid_on_missing)}, "
                f"got {self.on_missing!r}"
            )
        if self.max_fraction_of_load < 0:
            raise ValueError(
                f"max_fraction_of_load must be >= 0, "
                f"got {self.max_fraction_of_load}"
            )
        if not 0.0 <= self.max_fraction_violation_pct <= 100.0:
            raise ValueError(
                f"max_fraction_violation_pct must be in [0, 100], "
                f"got {self.max_fraction_violation_pct}"
            )
        if not self.entity_id:
            raise ValueError("entity_id must be non-empty")


@dataclass
class CovariateCfg:
    """Configuration for a single covariate (external feature)."""

    entity: str
    """Home Assistant sensor entity_id."""

    role: str = 'lagged'
    """Feature role: 'future' (known in advance), 'lagged' (historical only), or 'both'."""

    scale: Optional[float] = None
    """Optional scaling factor; if None, no scaling applied."""

    scaling: Optional[str] = None
    """Optional scaling strategy name: 'standard', 'minmax', or None."""

    transform: Optional[str] = None
    """Optional transformation: 'log', 'sqrt', 'box_cox', or None."""

    aggregation: str = 'mean'
    """Aggregation method for resampling: 'mean', 'sum', 'max', 'min', 'last'."""

    is_binary: bool = False
    """Whether this is a binary (0/1) feature."""

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_roles = {'future', 'lagged', 'both', 'concurrent'}
        if self.role not in valid_roles:
            raise ValueError(
                f'role must be one of {valid_roles}, got {self.role!r}'
            )
        valid_transforms = {None, 'log', 'sqrt', 'box_cox'}
        if self.transform not in valid_transforms:
            raise ValueError(
                f'transform must be one of {valid_transforms}, got {self.transform!r}'
            )
        valid_agg = {'mean', 'sum', 'max', 'min', 'last'}
        if self.aggregation not in valid_agg:
            raise ValueError(
                f'aggregation must be one of {valid_agg}, got {self.aggregation!r}'
            )


@dataclass
class ExperimentCfg:
    """Configuration for a single sensor prediction experiment."""

    name: str
    """Unique experiment identifier."""

    target_entity: str
    """Home Assistant sensor entity_id to predict."""

    covariates: List[CovariateCfg] = field(default_factory=list)
    """List of covariate configurations."""

    days_history: int = 14
    """Number of days of historical data to use for training."""

    interval_minutes: int = 30
    """Sampling interval in minutes."""

    source_is_cumulative: bool = False
    """Whether target sensor reports cumulative values requiring conversion to intervals."""

    reset_daily: bool = False
    """If True, cumulative sensor resets daily (e.g., 'today' energy sensors)."""

    max_increment: Optional[float] = None
    """Maximum allowed increment for cumulative conversion; exceeds indicate anomaly or reset."""

    models_enabled: List[str] = field(
        default_factory=lambda: ['lightgbm', 'xgboost', 'lstm', 'cnn']
    )
    """List of model types to train."""

    cv_strategy: str = 'walk_forward'
    """Cross-validation strategy: 'walk_forward' or 'sliding_window'."""

    cv_folds: int = 5
    """Number of cross-validation folds."""

    cv_embargo_periods: int = 2
    """Gap between training and test sets (in periods) to avoid temporal leakage."""

    metrics: List[str] = field(default_factory=lambda: ['mae', 'rmse', 'mase'])
    """Standard metrics to compute."""

    custom_metrics: Optional[Dict[str, str]] = None
    """Custom metrics as {name: 'Python expression'} using y_true, y_pred."""

    production_model: Optional[str] = None
    """Which model to use in production; if None, auto-select best by production_metric."""

    selected_model: Optional[str] = None
    """Model the Results-tab UI highlights and the Forecast Accuracy analytics
    filter to by default. Persisted across add-on restarts so a user's
    ``/select-model`` click survives reboots. Distinct from
    ``production_model``: the latter drives what actually gets retrained and
    published to HA; this only drives what the UI shows. Without this field,
    the in-memory selection was lost on every restart and fell back to
    whichever model the next benchmark cycle ranked first, which
    appeared to users as "I chose XGBoost but the page forgets"."""

    production_metric: str = 'rmse'
    """Metric to use for automatic model selection."""

    publish_prefix: str = 'mlfl_'
    """Prefix for published Home Assistant sensor entities."""

    publish_interval: bool = True
    """Whether to publish interval values (for cumulative inputs)."""

    publish_cumulative: bool = False
    """Whether to publish reconstructed cumulative values."""

    publish_daily_cumulative: bool = False
    """Whether to publish daily cumulative totals."""

    country: Optional[str] = None
    """Country code for holiday features ('GB', 'US', etc.); None = no holidays."""

    units: str = ''
    """Units of the target variable (e.g. 'kWh', 'W', 'L')."""

    output_units: Optional[str] = None
    """Optional units for output; if different from input, a conversion is applied."""

    log_transform: bool = False
    """Whether to apply log transform to target before modelling."""

    output_activation: str = 'auto'
    """Output-layer activation for neural models. One of:

    - ``auto`` (default): LSTM → ``zscore``; other neural backends →
      ``softplus`` when ``source_is_cumulative`` else ``linear``
    - ``linear``: unbounded output, suitable for signed targets (temperature, deltas)
    - ``softplus``: smooth non-negative output in (0, ∞), suitable for energy / power / count
    - ``relu``: hard non-negative output in [0, ∞); can produce dead units if many
      training targets are exactly zero
    - ``exp``: positive output in (0, ∞) for strictly-positive quantities that vary by
      orders of magnitude; applies ``torch.exp`` with a clamp to prevent overflow
    - ``sigmoid``: bounded output in (0, s) where s is a learned buffer scaled from
      training-data maximum; use for quantities with a hard physical ceiling
      (battery state-of-charge, humidity percent)
    - ``zscore``: target z-score normalisation during training with linear output head;
      stats computed per-horizon from training data, denormalised at inference. Keeps
      gradients O(1) regardless of target magnitude and lets the network learn signed
      residuals without activation saturation. Honoured by all PyTorch neural backends
      (LSTM, CNN, DLinear, N-BEATS, N-HiTS, TiDE, TSMixer, SparseTSF, PatchTST,
      iTransformer, Crossformer, TimesNet). Predictions are floored at zero after
      denormalisation. Superseded by RevIN when ``use_revin=True`` on the backend
      — the two schemes are mutually exclusive and RevIN owns the scale.

    Tree-based models (lightgbm/xgboost) ignore this field."""

    use_revin: bool = True
    """Reversible Instance Normalization (Kim et al. 2022,
    https://openreview.net/forum?id=cGDAkQo1C0p). When True (default),
    every PyTorch neural backend except N-BEATS and N-HiTS applies per-window
    per-channel normalisation at the input of the network and reverses it at
    the output, handling distribution shift on non-stationary series without
    a retrain. Published transformer / MLP-mixer benchmarks (PatchTST,
    iTransformer, TimesNet, TiDE, TSMixer, SparseTSF, Crossformer) all
    depend on RevIN in the authors' reference code — leaving it on is the
    faithful replication.

    N-BEATS and N-HiTS ignore this flag: their doubly-residual backcast-
    subtraction stacking already handles instance-level normalisation, and
    stacking RevIN on top would double-normalise. Tree-based models ignore
    this field entirely.

    When True, the ``output_activation='zscore'`` path becomes a no-op (RevIN
    already provides per-window scale normalisation). Set False to fall back
    to dataset-level channel stats + the zscore denormalisation path, if you
    need published-parity with papers that explicitly disable RevIN."""

    future_covariate_features: List[str] = field(default_factory=list)
    """Feature names (as they appear in the engineered feature matrix) that
    contain KNOWN-FUTURE values for each forecast horizon step.

    Currently only consumed by the TiDE backend's temporal-decoder path
    (Das et al. 2023): if this list is non-empty AND the runner supplies
    a ``future_covariates`` array at fit time, TiDE routes the named
    features through a feature-projection block and combines them with the
    decoder state per horizon step via the paper's temporal decoder.

    Typical contents for forecasting use cases:
    - Calendar features: hour-of-day, day-of-week, day-of-year, holiday flag
    - Externally-forecast weather: Solcast GHI (p10/p50/p90), Open-Meteo
      temperature / cloud cover / wind
    - Known-future schedule: EV charging plan, occupancy calendar

    Do NOT include lags of the target, rolling stats, or any feature
    derived from the true future value — the whole point is that these
    values are knowable at forecast-issue time without peeking."""

    subtract: List[str] = field(default_factory=list)
    """DEPRECATED stub — prefer ``load_subtract`` with per-sensor config.

    Kept only so existing YAML with ``subtract: [...]`` doesn't crash on load.
    ``load_config`` emits a deprecation warning and does NOT wire this list
    into preprocessing. Migrate to ``load_subtract`` for a robust pipeline."""

    load_subtract: List[SubtractCfg] = field(default_factory=list)
    """Sensors to subtract from the target before feature build / training.

    Use case: the GivTCP whole-house load sensor includes EV charging and
    iBoost solar-divert dumps, both of which are noise for a model trying to
    learn *baseline household* load patterns. Subtracting them produces a
    cleaner training signal.

    Each entry is a ``SubtractCfg`` — see its docstring for the robustness
    semantics (source type, missing-data policy, unit-scale, fail-fast
    fraction guard). Subtraction is applied in ``_fetch_and_preprocess``
    after cumulative→interval conversion and before outlier clipping, so
    outlier bounds reflect the adjusted (baseline) signal rather than
    spikes from the subtracted component.

    Double-subtraction is the consumer's problem: if the downstream system
    (e.g. predbat) also subtracts the same sensors, don't do both."""

    mode: str = 'lab'
    """Operational mode: 'lab' (benchmark all models) or 'production' (forecast with best model)."""

    clear_forecast_log_on_retrain: bool = True
    """Whether to prune forecast_log rows older than the latest retrain
    timestamp when the champion is promoted.

    Forecasts logged under an older set of weights (even under the same
    model_name) pool into the stability metric and inflate cross-run
    disagreement — the "I retrained and now stability looks terrible"
    pattern. Clearing pre-retrain rows keeps the metric honest. Set to
    False if you want to preserve the full history for offline analysis
    and are willing to read the stability chart with that in mind."""

    stability_focus: str = 'per_moment'
    """Which stability metric drives the Forecast Accuracy verdict chip.

    - ``per_moment`` (default): chip reads median cross-cycle CV of
      predictions at the same target moment. Right when the downstream
      consumer cares about *when* demand hits (HVAC setpoints, pre-heat
      timing, battery dispatch).
    - ``daily_total``: chip reads median cross-cycle CV of daily-total
      predictions (cumulative sensors only). Right when the downstream
      consumer only integrates over the day — e.g. Predbat iBoost
      deciding how much to heat a hot-water tank, EV daily charging
      budget, solar-export daily planning. For those use cases a
      ±50% per-moment swing may be fine as long as the daily total is
      stable, so reporting per-moment "poor" gives the wrong verdict.

    The Layer 3 accordion still shows both metrics regardless; only
    the Layer 1 chip and headline follow this setting."""

    max_age: int = 365
    """Maximum days to keep in SQLite cache."""

    future_periods: int = 48
    """Number of future periods to forecast."""

    publish_name: Optional[str] = None
    """Override name for published HA entities."""

    database: bool = False
    """Whether to cache history in SQLite."""

    model_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Per-model hyperparameter overrides specific to this experiment.
    Takes precedence over global model_overrides. Keys are model names."""

    forecast_every_minutes: Optional[int] = None
    """How often to run inference and publish sensors for this experiment.
    Falls back to AppConfig.forecast_every_minutes if None."""

    retrain_every_hours: Optional[float] = None
    """How often to retrain the model from scratch for this experiment.
    Falls back to AppConfig.retrain_every_hours if None."""

    loss_fn: str = 'mse'
    """Training loss for neural models: 'mse', 'mae', or 'huber'."""

    optimiser: str = 'adamw'
    """Optimiser for neural models: 'adamw' (default, decoupled weight decay as
    used by every published time-series transformer paper) or 'adam' (classic
    Adam; weight decay is tied to the adaptive learning rate, which means
    frequently-updated parameters receive less effective regularisation). Both
    share the same ``learning_rate`` and ``weight_decay=1e-4``; the difference
    is purely in how weight decay composes with the adaptive update. Ignored
    by tree models."""

    daily_loss_weight: float = 0.0
    """Weight λ for the cumulative-trajectory loss term added to the per-interval
    loss during neural training. 0.0 disables (interval loss only — default).

    The daily term penalises error in the cumulative forecast curve at every
    horizon step (not just the endpoint), so the SHAPE of the predicted
    cumulative trajectory must match the actual cumulative trajectory. With
    ``future_periods=48`` and ``interval_minutes=30`` this is the 24 h
    daily-cumulative curve, directly aligned with what users evaluate on
    cumulative-origin targets such as ``sensor.mixergy_demand_today`` or
    daily energy-usage sensors.

    History: v2.16 used a mean-over-horizons constraint (just the endpoint).
    v2.18 replaced it with the trajectory formulation above after experiments
    on Mixergy demand showed the mean-only version was too weak to affect
    training measurably — the mean is already matched by any unbiased model,
    regardless of the curve shape.

    Applied to torch neural backends only; silently ignored by tree models.
    Typical useful range: 0.1–1.0 (stronger under MSE than MAE due to loss
    geometry)."""

    recency_half_life_days: float = 7.0
    """Half-life for exponential recency weighting in days. Recent samples receive
    higher weight during training so models prioritise current patterns.
    Set to 0 to disable recency weighting (all samples weighted equally)."""

    include_sun_elevation: bool = False
    """Include sun elevation angle (degrees above horizon) as a computed covariate.
    Deterministic from (lat, lon, timestamp); requires no external data source and
    is available for both training history and forecast horizon. Negative at night
    — a strong physical signal for diurnal patterns (solar PV, outdoor lighting,
    daytime load)."""

    include_clear_sky_irradiance: bool = False
    """Include clear-sky global horizontal irradiance (W/m²) as a computed covariate.
    Theoretical maximum solar energy under perfect clear-sky conditions, computed via
    pvlib's Ineichen model. Zero at night, peak at solar noon. Ideal for solar PV
    forecasting — turns the problem into predicting cloud-cover-driven attenuation
    rather than raw generation."""

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_modes = {'lab', 'production'}
        if self.mode not in valid_modes:
            raise ValueError(
                f'mode must be one of {valid_modes}, got {self.mode!r}'
            )
        valid_cv = {'walk_forward', 'sliding_window'}
        if self.cv_strategy not in valid_cv:
            raise ValueError(
                f'cv_strategy must be one of {valid_cv}, got {self.cv_strategy!r}'
            )
        valid_stability = {'per_moment', 'daily_total'}
        if self.stability_focus not in valid_stability:
            raise ValueError(
                f'stability_focus must be one of {valid_stability}, '
                f'got {self.stability_focus!r}'
            )
        if self.stability_focus == 'daily_total' and not self.source_is_cumulative:
            # Daily-total CV isn't meaningful for instantaneous sensors —
            # summing them over a day doesn't produce a physical quantity.
            raise ValueError(
                "stability_focus='daily_total' requires source_is_cumulative=True"
            )
        if self.cv_folds < 2:
            raise ValueError(f'cv_folds must be >= 2, got {self.cv_folds}')
        if self.cv_embargo_periods < 0:
            raise ValueError(
                f'cv_embargo_periods must be >= 0, got {self.cv_embargo_periods}'
            )
        if self.days_history < 1:
            raise ValueError(f'days_history must be >= 1, got {self.days_history}')
        if self.interval_minutes < 1:
            raise ValueError(
                f'interval_minutes must be >= 1, got {self.interval_minutes}'
            )
        if self.recency_half_life_days < 0:
            raise ValueError(
                f'recency_half_life_days must be >= 0, got {self.recency_half_life_days}'
            )
        if self.daily_loss_weight < 0:
            raise ValueError(
                f'daily_loss_weight must be >= 0, got {self.daily_loss_weight}'
            )
        valid_optimisers = {'adam', 'adamw'}
        if self.optimiser not in valid_optimisers:
            raise ValueError(
                f'optimiser must be one of {sorted(valid_optimisers)}, '
                f'got {self.optimiser!r}'
            )
        valid_activations = {
            'auto', 'linear', 'softplus', 'relu', 'exp', 'sigmoid', 'zscore',
        }
        if self.output_activation not in valid_activations:
            raise ValueError(
                f'output_activation must be one of {sorted(valid_activations)}, '
                f'got {self.output_activation!r}'
            )


@dataclass
class AppConfig:
    """Application-level configuration."""

    forecast_every_minutes: int = 30
    """How often to run inference with the cached model and publish sensors."""

    retrain_every_hours: float = 24.0
    """How often to retrain the production model from scratch."""

    update_every_minutes: int = 5
    """Legacy alias — used as forecast_every_minutes if forecast_every_minutes
    is at its default and this is explicitly set in YAML."""

    timezone: str = 'UTC'
    """Timezone for temporal features."""

    experiments: List[ExperimentCfg] = field(default_factory=list)
    """List of experiment configurations."""

    cpu_cores: int = 0
    """Number of CPU cores for model training. 0 = all available."""

    nice_priority: int = 10
    """Process priority for training (0=normal, 19=lowest). Default 10."""

    model_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Per-model hyperparameter overrides. Keys are model registry names."""

    def __post_init__(self) -> None:
        """Validate application configuration."""
        # Backward compat: if old update_every_minutes was set but new
        # forecast_every_minutes is at default, use the old value.
        if self.forecast_every_minutes == 30 and self.update_every_minutes != 5:
            self.forecast_every_minutes = self.update_every_minutes
        if self.forecast_every_minutes < 1:
            raise ValueError(
                f'forecast_every_minutes must be >= 1, got {self.forecast_every_minutes}'
            )
        if self.retrain_every_hours < 0.1:
            raise ValueError(
                f'retrain_every_hours must be >= 0.1, got {self.retrain_every_hours}'
            )
        if not self.experiments:
            logger.warning('No experiments configured')


def load_config(config_path: Path | str) -> AppConfig:
    """
    Load application configuration from a YAML file.

    Parameters
    ----------
    config_path : Path or str
        Path to YAML configuration file.

    Returns
    -------
    AppConfig
        Parsed application configuration.

    Raises
    ------
    FileNotFoundError
        If configuration file does not exist.
    ValueError
        If YAML is malformed or required fields are missing.

    Notes
    -----
    Expected YAML structure:

        update_every_minutes: 5
        timezone: Europe/London
        experiments:
          - name: solar_forecast
            target_entity: sensor.solar_generation_w
            source_is_cumulative: true
            reset_daily: true
            days_history: 30
            interval_minutes: 30
            units: W
            log_transform: false
            country: GB
            models_enabled: [lightgbm, xgboost]
            cv_strategy: walk_forward
            cv_folds: 5
            cv_embargo_periods: 2
            metrics: [mae, rmse, mase]
            covariates:
              - entity: sensor.cloud_cover_percent
                role: future
                aggregation: mean
              - entity: sensor.outdoor_temperature_c
                role: lagged
                scale: 0.1
                transform: null
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'Configuration file not found: {config_path}')

    logger.debug(f'Loading configuration from {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError('Configuration file must contain a YAML dictionary')

    # Parse experiments
    experiments_data = data.pop('experiments', [])
    experiments = []

    exp_fields = {f.name for f in dataclasses.fields(ExperimentCfg)}
    cov_fields = {f.name for f in dataclasses.fields(CovariateCfg)}
    sub_fields = {f.name for f in dataclasses.fields(SubtractCfg)}
    app_fields = {f.name for f in dataclasses.fields(AppConfig)} - {'experiments'}

    # Track whether we need to rewrite the YAML to clean deprecated fields
    _needs_migrate = False

    for exp_data in experiments_data:
        if not isinstance(exp_data, dict):
            raise ValueError(
                f'Each experiment must be a dictionary, got {type(exp_data)}'
            )

        # Migration: silently remove deprecated fields
        if 'horizons_minutes' in exp_data:
            exp_data.pop('horizons_minutes')
            _needs_migrate = True

        # Parse covariates
        covariates_data = exp_data.pop('covariates', [])
        covariates = []
        for cov in covariates_data:
            unknown_cov = set(cov) - cov_fields
            if unknown_cov:
                logger.warning(f'Ignoring unknown covariate fields: {unknown_cov}')
                cov = {k: v for k, v in cov.items() if k in cov_fields}
            covariates.append(CovariateCfg(**cov))

        # Parse load_subtract (robust, per-sensor config).
        #
        # Each entry must be a mapping with SubtractCfg fields. Plain string
        # entries are tolerated as a convenience (treated as entity_id with
        # defaults) but logged as an ambiguity — the defaults may not match
        # the sensor's actual semantics.
        load_subtract_data = exp_data.pop('load_subtract', [])
        load_subtract = []
        for sub in load_subtract_data:
            if isinstance(sub, str):
                logger.warning(
                    f'load_subtract entry {sub!r} is a bare string; '
                    f'using SubtractCfg defaults (source=auto, on_missing=zero). '
                    f'Prefer an explicit mapping with source/on_missing.'
                )
                sub = {'entity_id': sub}
            unknown_sub = set(sub) - sub_fields
            if unknown_sub:
                logger.warning(
                    f'Ignoring unknown load_subtract fields: {unknown_sub}'
                )
                sub = {k: v for k, v in sub.items() if k in sub_fields}
            load_subtract.append(SubtractCfg(**sub))

        # Deprecation: legacy `subtract: [str]` field is a stub that was never
        # wired into preprocessing. Warn loudly so users migrate to
        # load_subtract rather than silently ignoring what they set.
        if exp_data.get('subtract'):
            logger.warning(
                f"Experiment {exp_data.get('name', '?')!r}: field 'subtract' "
                f"is deprecated and has no effect. Migrate to 'load_subtract' "
                f"with explicit source/on_missing per sensor."
            )

        # Filter unknown experiment fields
        unknown_exp = set(exp_data) - exp_fields
        if unknown_exp:
            logger.warning(f'Ignoring unknown experiment fields: {unknown_exp}')
            exp_data = {k: v for k, v in exp_data.items() if k in exp_fields}

        exp = ExperimentCfg(
            **exp_data,
            covariates=covariates,
            load_subtract=load_subtract,
        )
        experiments.append(exp)

    # Extract model_overrides before filtering
    model_overrides = data.pop('model_overrides', {})
    if not isinstance(model_overrides, dict):
        logger.warning('model_overrides must be a dict; ignoring')
        model_overrides = {}

    # Filter unknown app-level fields
    unknown_app = set(data) - app_fields
    if unknown_app:
        logger.warning(f'Ignoring unknown app config fields: {unknown_app}')
        data = {k: v for k, v in data.items() if k in app_fields}

    app_config = AppConfig(
        **data, experiments=experiments, model_overrides=model_overrides,
    )
    logger.debug(
        f'Configuration loaded: {len(app_config.experiments)} experiment(s)'
    )

    # Auto-migrate: rewrite YAML to strip deprecated fields
    if _needs_migrate:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw = yaml.safe_load(f)
            for exp in raw.get('experiments', []):
                if isinstance(exp, dict):
                    exp.pop('horizons_minutes', None)
            _atomic_yaml_write(config_path, raw)
            logger.info('Migrated config: removed deprecated horizons_minutes')
        except Exception as e:
            logger.warning(f'Config migration failed (non-fatal): {e}')

    return app_config


def save_model_overrides(
    config_path: Path | str,
    model_name: str,
    overrides: Dict[str, Any] | None,
) -> None:
    """
    Persist per-model hyperparameter overrides to YAML.

    Parameters
    ----------
    config_path : Path or str
        Path to the YAML configuration file.
    model_name : str
        Model registry name (e.g. 'lstm', 'lightgbm').
    overrides : dict or None
        Parameter overrides to save. If None or empty, removes the
        model's entry (i.e. resets to defaults).
    """
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    mo = data.setdefault('model_overrides', {})
    if overrides:
        mo[model_name] = overrides
    else:
        mo.pop(model_name, None)

    # Clean up empty section
    if not mo:
        data.pop('model_overrides', None)

    _atomic_yaml_write(config_path, data)


def save_experiment_model_params(
    config_path: Path | str,
    experiment_name: str,
    model_name: str,
    params: Dict[str, Any] | None,
) -> None:
    """
    Save per-experiment model hyperparameter overrides.

    Unlike global ``model_overrides``, these are stored inside the
    experiment's config and take precedence during training.

    Parameters
    ----------
    config_path : Path or str
        Path to the YAML configuration file.
    experiment_name : str
        Experiment name to update.
    model_name : str
        Model registry name.
    params : dict or None
        Parameter overrides to save. None or empty removes the entry.
    """
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    for exp in data.get('experiments', []):
        if exp.get('name') != experiment_name:
            continue
        mp = exp.setdefault('model_params', {})
        if params:
            mp[model_name] = params
        else:
            mp.pop(model_name, None)
        if not mp:
            exp.pop('model_params', None)
        break

    _atomic_yaml_write(config_path, data)


def save_experiment_field(
    config_path: Path | str,
    experiment_name: str,
    field: str,
    value: Any,
) -> None:
    """
    Set a single field on an experiment in the YAML config.

    Parameters
    ----------
    config_path : Path or str
        Path to the YAML configuration file.
    experiment_name : str
        Experiment name to update.
    field : str
        Field name to set (e.g. 'production_model', 'mode').
    value : Any
        Value to write.
    """
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    for exp in data.get('experiments', []):
        if exp.get('name') == experiment_name:
            exp[field] = value
            break

    _atomic_yaml_write(config_path, data)


def remove_experiment_covariate(
    config_path: Path | str,
    experiment_name: str,
    entity_id: str,
) -> bool:
    """
    Remove a covariate from an experiment's config.

    `entity_id` can be either the full entity ID (e.g. ``sensor.current_charge``)
    or its short suffix (``current_charge``) — the covariate-analysis UI uses the
    short form because that's what becomes the dataframe column name.

    Returns True if a covariate was removed, False if not found.
    """
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    def _matches(cov: dict, target: str) -> bool:
        ent = cov.get('entity') or ''
        if not ent:
            return False
        return ent == target or ent.split('.')[-1] == target

    removed = False
    for exp in data.get('experiments', []):
        if exp.get('name') != experiment_name:
            continue
        covs = exp.get('covariates', [])
        original_len = len(covs)
        exp['covariates'] = [c for c in covs if not _matches(c, entity_id)]
        removed = len(exp['covariates']) < original_len
        break

    if removed:
        _atomic_yaml_write(config_path, data)

    return removed


def clear_experiment_covariates(
    config_path: Path | str,
    experiment_name: str,
) -> int:
    """
    Remove ALL covariates from an experiment's config.

    Returns the number of covariates that were removed.
    """
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    removed = 0
    for exp in data.get('experiments', []):
        if exp.get('name') != experiment_name:
            continue
        removed = len(exp.get('covariates', []))
        exp['covariates'] = []
        break

    if removed > 0:
        _atomic_yaml_write(config_path, data)

    return removed


def add_experiment_covariate(
    config_path: Path | str,
    experiment_name: str,
    covariate: Dict[str, Any],
) -> bool:
    """
    Add a covariate to an experiment's config.

    Parameters
    ----------
    config_path : Path or str
        Path to the YAML configuration file.
    experiment_name : str
        Experiment name to update.
    covariate : dict
        Covariate fields (must include 'entity' at minimum).

    Returns
    -------
    bool
        True if added successfully, False if experiment not found or
        covariate with that entity already exists.

    Raises
    ------
    ValueError
        If the covariate dict contains invalid fields.
    """
    cov_fields = {f.name for f in dataclasses.fields(CovariateCfg)}
    unknown = set(covariate) - cov_fields
    if unknown:
        raise ValueError(f'Unknown covariate fields: {unknown}')
    if 'entity' not in covariate or not covariate['entity']:
        raise ValueError('Covariate must include a non-empty "entity" field')

    # Validate by constructing (raises ValueError on bad role/aggregation/etc.)
    CovariateCfg(**covariate)

    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    for exp in data.get('experiments', []):
        if exp.get('name') != experiment_name:
            continue
        covs = exp.setdefault('covariates', [])
        # Reject duplicate entity
        if any(c.get('entity') == covariate['entity'] for c in covs):
            return False
        # Strip None values so YAML stays clean
        clean = {k: v for k, v in covariate.items() if v is not None}
        covs.append(clean)
        _atomic_yaml_write(config_path, data)
        return True

    return False


def add_experiment_load_subtract(
    config_path: Path | str,
    experiment_name: str,
    subtract: Dict[str, Any],
) -> bool:
    """
    Add a load-subtract sensor to an experiment's config.

    Parameters
    ----------
    config_path : Path or str
        Path to the YAML configuration file.
    experiment_name : str
        Experiment name to update.
    subtract : dict
        SubtractCfg fields (must include 'entity_id' at minimum).

    Returns
    -------
    bool
        True if added successfully, False if experiment not found or
        a subtract with that entity_id already exists.

    Raises
    ------
    ValueError
        If the subtract dict contains invalid fields or fails
        ``SubtractCfg.__post_init__`` validation.
    """
    sub_fields = {f.name for f in dataclasses.fields(SubtractCfg)}
    unknown = set(subtract) - sub_fields
    if unknown:
        raise ValueError(f'Unknown load_subtract fields: {unknown}')
    if 'entity_id' not in subtract or not subtract['entity_id']:
        raise ValueError(
            'load_subtract entry must include a non-empty "entity_id" field'
        )

    # Validate by constructing (raises ValueError on bad source/on_missing/etc.)
    SubtractCfg(**subtract)

    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    for exp in data.get('experiments', []):
        if exp.get('name') != experiment_name:
            continue
        subs = exp.setdefault('load_subtract', [])
        # Reject duplicate entity_id (bare strings treated as entity_id too)
        for existing in subs:
            existing_id = (
                existing if isinstance(existing, str)
                else existing.get('entity_id')
            )
            if existing_id == subtract['entity_id']:
                return False
        # Strip None values so YAML stays clean
        clean = {k: v for k, v in subtract.items() if v is not None}
        subs.append(clean)
        _atomic_yaml_write(config_path, data)
        return True

    return False


def remove_experiment_load_subtract(
    config_path: Path | str,
    experiment_name: str,
    entity_id: str,
) -> bool:
    """
    Remove a load-subtract sensor from an experiment's config.

    ``entity_id`` can be the full entity ID (``sensor.ev_energy_today``) or
    its short suffix (``ev_energy_today``).

    Returns True if an entry was removed, False if not found.
    """
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    def _matches(sub, target: str) -> bool:
        ent = sub if isinstance(sub, str) else (sub.get('entity_id') or '')
        if not ent:
            return False
        return ent == target or ent.split('.')[-1] == target

    removed = False
    for exp in data.get('experiments', []):
        if exp.get('name') != experiment_name:
            continue
        subs = exp.get('load_subtract', [])
        original_len = len(subs)
        exp['load_subtract'] = [s for s in subs if not _matches(s, entity_id)]
        removed = len(exp['load_subtract']) < original_len
        break

    if removed:
        _atomic_yaml_write(config_path, data)

    return removed


def clear_experiment_load_subtract(
    config_path: Path | str,
    experiment_name: str,
) -> int:
    """
    Remove ALL load-subtract entries from an experiment's config.

    Returns the number of entries that were removed.
    """
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    removed = 0
    for exp in data.get('experiments', []):
        if exp.get('name') != experiment_name:
            continue
        removed = len(exp.get('load_subtract', []))
        exp['load_subtract'] = []
        break

    if removed > 0:
        _atomic_yaml_write(config_path, data)

    return removed


_EXP_NAME_RE = re.compile(r'^[a-z][a-z0-9_]{0,63}$')


def create_experiment(
    config_path: Path | str,
    experiment: Dict[str, Any],
) -> None:
    """
    Create a new experiment in the YAML config.

    Only ``name`` and ``target_entity`` are required; all other fields
    get ``ExperimentCfg`` defaults on the next ``load_config()`` call.

    Parameters
    ----------
    config_path : Path or str
        Path to the YAML configuration file.
    experiment : dict
        Experiment fields. Must include 'name' and 'target_entity'.

    Raises
    ------
    ValueError
        If name is invalid, duplicate, or target_entity is missing.
    """
    name = experiment.get('name', '')
    if not _EXP_NAME_RE.match(name):
        raise ValueError(
            f'Experiment name must match [a-z][a-z0-9_]{{0,63}}, got {name!r}'
        )
    if not experiment.get('target_entity'):
        raise ValueError('target_entity is required')

    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    exps = data.setdefault('experiments', [])
    if any(e.get('name') == name for e in exps):
        raise ValueError(f'Experiment {name!r} already exists')

    exps.append(experiment)

    _atomic_yaml_write(config_path, data)


def delete_experiment(
    config_path: Path | str,
    experiment_name: str,
) -> bool:
    """
    Remove an experiment from the YAML config.

    Returns True if an experiment was removed, False if not found.
    """
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    exps = data.get('experiments', [])
    original_len = len(exps)
    data['experiments'] = [e for e in exps if e.get('name') != experiment_name]

    if len(data['experiments']) < original_len:
        _atomic_yaml_write(config_path, data)
        return True

    return False
