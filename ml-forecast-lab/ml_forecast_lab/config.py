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


def atomic_yaml_write(config_path: Path, data: dict) -> None:
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
    forecasting — e.g. EV charging or a solar-divert dump to remove those
    contributions from a whole-house load signal so the model learns only
    the baseline household pattern.

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

    - ``zero``: fill with 0.0 (common for switchable loads like EV chargers
      or solar-divert dumps — missing often means "wasn't on"). Audit log
      records the filled count so you can spot pathological cases.
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

    transform: Optional[str] = None
    """Optional transformation: 'log', 'sqrt', 'box_cox', or None."""

    aggregation: str = 'mean'
    """Aggregation method for resampling: 'mean', 'sum', 'max', 'min', 'last'."""

    is_binary: bool = False
    """Whether this is a binary (0/1) feature."""

    future_attribute: str = 'forecast'
    """For role='future' / 'both': the HA entity attribute that contains
    the known-future forecast (e.g. weather entities expose ``forecast``,
    Solcast exposes ``detailedForecast``). Ignored when role='lagged'."""

    future_value_key: Optional[str] = None
    """For role='future' / 'both': the key inside each forecast-list entry
    that contains the value (e.g. ``temperature`` for Met.no weather,
    ``pv_estimate`` for Solcast). When None, the resolver tries common
    keys (value, pv_estimate, state, temperature, cloud_coverage,
    wind_speed) in order. Ignored when the attribute is a flat
    ``{iso_dt: value}`` mapping."""

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

    target_is_nonnegative: bool = False
    """Whether the target is physically non-negative (PV power, irradiance, demand
    quantities). When True, the ``output_activation='auto'`` resolution defaults
    to ``'softplus'`` instead of ``'linear'`` — keeping predictions on the
    correct side of zero is worth more than the (small) loss of unbounded
    range. Cumulative targets always enable this implicitly via
    ``source_is_cumulative``; set this directly for non-cumulative non-negative
    targets like ``predbat.pv_power`` (v2.37+ PF8)."""

    debug_save_training_dumps: bool = False
    """If True, every retrain dumps the training inputs (features, channels,
    hyperparameters) and the immediate post-retrain forecast (raw + physical
    values) to ``<config_dir>/debug/<experiment>/<timestamp>/``. Last 5 dumps
    per experiment are kept; older ones are rotated out. Default OFF — enable
    only when diagnosing a regression so a maintainer can inspect the exact
    production training surface offline. ~0.5–2 MB per dump on a 30-min PV
    target with 22 channels and 48-step past window."""

    idle_value: Optional[float] = None
    """Value to fill NaN gaps with when the sensor is idle / unavailable
    (e.g. an EV charger between sessions, a solar pump in winter, an idle
    battery). Only applied when ``target_is_nonnegative=True``.

    - When ``None`` (default): current behaviour — solar targets with
      physics features get the v2.37.3 night-fill (NaN → 0 where sun is
      below horizon); other non-negative targets keep the original
      drop-on-NaN behaviour.
    - When set to a numeric value: the v2.37.3 solar night-fill uses this
      value instead of 0 (lets users with an inverter standby > 0
      override the default), AND non-solar non-negative targets fill
      ALL remaining NaN gaps with this value before dropna. The user
      asserts "this sensor is at <value> when it's not reporting" — use
      with care, a daytime EV-charger sensor failure would also be
      filled with the idle value rather than dropped.

    Typical usage:
      idle_value: 0     # EV charger / solar pump / non-solar PV
      idle_value: 0.005 # solar with measurable inverter standby"""

    models_enabled: List[str] = field(
        default_factory=lambda: ['lightgbm', 'xgboost', 'lstm', 'cnn']
    )
    """List of model types to train."""

    cv_strategy: str = 'walk_forward'
    """Cross-validation strategy: 'walk_forward' or 'sliding_window'."""

    cv_folds: int = 5
    """Number of cross-validation folds."""

    cv_embargo_periods: int = 2
    """Gap (in periods) between training and test sets to avoid temporal
    leakage from rolling / lag features that span the fold boundary.

    The pipeline's longest rolling window is ~36 h (72 steps at 30-min
    sampling); raising the embargo to that size eliminates the residual
    rolling spillover at the boundary. The conservative default ``2``
    preserves behaviour on small training fixtures and benchmark cycles
    where setting it to 72 would starve early folds of training rows.
    Increase manually when running on a long history (≥ 30 days)."""

    metrics: List[str] = field(
        default_factory=lambda: ['mae', 'rmse', 'mase', 'seasonal_mase']
    )
    """Standard metrics to compute."""

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

    production_metric: str = 'seasonal_mase'
    """Metric to use for automatic model selection. Default ``seasonal_mase``
    (MAE scaled by the same-time-yesterday baseline at the configured
    ``interval_minutes``) is the right comparison for the daily-seasonal HA
    sensors most users forecast. ``mase`` (1-step naive) is retained for
    backwards compatibility but understates skill on seasonal series."""

    publish_prefix: str = 'mlfl_'
    """Prefix for published Home Assistant sensor entities."""

    country: Optional[str] = None
    """Country code for holiday features ('GB', 'US', etc.); None = no holidays."""

    units: str = ''
    """Units of the target variable (e.g. 'kWh', 'W', 'L')."""

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

    subtract: List[str] = field(default_factory=list)
    """DEPRECATED stub — prefer ``load_subtract`` with per-sensor config.

    Kept only so existing YAML with ``subtract: [...]`` doesn't crash on load.
    ``load_config`` emits a deprecation warning and does NOT wire this list
    into preprocessing. Migrate to ``load_subtract`` for a robust pipeline."""

    load_subtract: List[SubtractCfg] = field(default_factory=list)
    """Sensors to subtract from the target before feature build / training.

    Use case: a whole-house load sensor includes switchable disturbances
    like EV charging and solar-divert dumps, both of which are noise for a
    model trying to learn *baseline household* load patterns. Subtracting
    them produces a cleaner training signal.

    Each entry is a ``SubtractCfg`` — see its docstring for the robustness
    semantics (source type, missing-data policy, unit-scale, fail-fast
    fraction guard). Subtraction is applied in ``_fetch_and_preprocess``
    after cumulative→interval conversion and before outlier clipping, so
    outlier bounds reflect the adjusted (baseline) signal rather than
    spikes from the subtracted component.

    Double-subtraction is the consumer's problem: if a downstream
    automation also subtracts the same sensors from its own load model,
    don't do both."""

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

    max_age: int = 365
    """Maximum days to keep in SQLite cache."""

    future_periods: int = 48
    """Number of future periods to forecast."""

    publish_name: Optional[str] = None
    """Override name for published HA entities."""

    # `database` was a per-experiment toggle gating the SQLite actuals
    # cache. Removed in v2.33.1 — actuals are cached unconditionally
    # because every Forecast Accuracy query depends on them, the cost
    # is negligible (~72 KB per experiment for a 30-day window), and
    # the flag was a foot-gun (disabling it silently broke the whole
    # Forecast Accuracy view). Old yamls carrying the field are
    # auto-migrated by load_config — the field is stripped from disk
    # and an INFO line records the migration.

    model_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Per-model hyperparameter overrides specific to this experiment.
    Takes precedence over global model_overrides. Keys are model names."""

    forecast_every_minutes: Optional[int] = None
    """How often to run inference and publish sensors for this experiment.
    Falls back to AppConfig.forecast_every_minutes if None."""

    retrain_every_hours: Optional[float] = None
    """How often to retrain the model from scratch for this experiment.
    Falls back to AppConfig.retrain_every_hours if None."""

    loss_fn: str = 'huber'
    """Training loss: 'mse', 'mae', 'huber', or 'tweedie' (tree backends).
    Default ``huber`` (smooth-L1) is appropriate for the typical HA target —
    quadratic near zero so gradients flow on small errors, linear in the
    tails so sensor spikes don't dominate. MSE is preserved for backwards
    compatibility but is rarely the right choice for spiky, near-zero
    series (power, occupancy, rainfall). ``tweedie`` is honoured only by
    LightGBM / XGBoost / CatBoost — neural backends fall back to Huber."""

    optimiser: str = 'adamw'
    """Optimiser for neural models: 'adamw' (default, decoupled weight decay as
    used by every published time-series transformer paper) or 'adam' (classic
    Adam; weight decay is tied to the adaptive learning rate, which means
    frequently-updated parameters receive less effective regularisation). Both
    share the same ``learning_rate`` and ``weight_decay=1e-4``; the difference
    is purely in how weight decay composes with the adaptive update. Ignored
    by tree models."""

    daily_loss_weight: float = 0.0
    """v2.40.14 DEPRECATED — kept on the model only so existing YAML
    configs continue to load. The underlying cumulative-trajectory loss
    term was removed (see CHANGELOG): measured to hurt the daily total
    in both sparse-demand and smooth-cumulative regimes, with a
    structural gradient asymmetry that systematically biased the model
    toward under-prediction at early horizon steps. Setting this field
    no longer affects training.

    Historical description below for context; ignore for new experiments.

    Original: Weight λ for the cumulative-trajectory loss term added
    to the per-interval loss during neural training. 0.0 disables
    (interval loss only — default).

    The daily term penalised error in the cumulative forecast curve at
    every horizon step (not just the endpoint), so the SHAPE of the
    predicted cumulative trajectory had to match the actual cumulative
    trajectory. With ``future_periods=48`` and ``interval_minutes=30``
    this is the 24 h daily-cumulative curve, directly aligned with
    what users evaluate on cumulative-origin targets such as
    ``sensor.energy_today`` or daily
    energy-usage sensors.

    History: v2.16 used a mean-over-horizons constraint (just the endpoint).
    v2.18 replaced it with the trajectory formulation above after experiments
    on a daily-cumulative demand target showed the mean-only version was too
    weak to affect training measurably — the mean is already matched by any
    unbiased model, regardless of the curve shape.

    Applied to torch neural backends only; silently ignored by tree models.

    v2.40.14: SETTING THIS FIELD HAS NO EFFECT. Retained on the model
    only so existing YAML configs load without error."""

    loss_balance: Optional[float] = None
    """v2.40.14 DEPRECATED — kept on the model only so existing YAML
    configs continue to load. The convex-blend cumulative-loss path
    was removed (see CHANGELOG): the harness measured the slider as a
    cliff (any α>0 → 50-95% daily-MAE degradation, flat across the
    α∈[0.1, 1.0] range) on BOTH sparse-demand and smooth-cumulative
    targets, with the same gradient-asymmetry mechanism identified in
    ``_cumulative_trajectory_loss``. Faster EMA decay softened the
    cliff but did not remove it. Setting this field no longer affects
    training."""

    recency_half_life_days: float = 0.0
    """Half-life for exponential recency weighting in days. ``0`` (default,
    post-audit) gives uniform sample weight — the right choice for the
    stable household / business sensors most users forecast, where the
    weekly-seasonal pattern from a fortnight ago is just as informative
    as yesterday's data. Set to a positive value (e.g. 7) only when the
    series has recently entered a new regime (heat pump install, EV
    delivery, schedule change). Pre-audit default was ``7``, which
    silently down-weighted older training rows by ~75%."""

    patience: Optional[int] = None
    """v2.40.12: per-experiment early-stopping patience (epochs / boosting
    rounds with no smoothed-val-loss improvement before training stops).

    ``None`` (default) means each backend uses its own constructor default
    (20 for every neural backend; 50 for LightGBM / XGBoost / CatBoost in
    the legacy pre-v2.40.12 code). Setting an integer here overrides the
    backend default uniformly across every neural and tree model in this
    experiment, so you get consistent stopping behaviour rather than the
    "neural runs cut off at 20, LightGBM at 50" asymmetry.

    Set higher (e.g. 60) to give the model more rope when val_loss is
    noisy; set lower (e.g. 10) for faster iteration on small experiments.
    Set very high (≥ epochs) to effectively disable early stopping — but
    note that disabling almost always hurts generalisation; the EMA-
    smoothed stop decision in v2.40.12 should already absorb most val-
    loss noise so the cases where disabling helps are rare."""

    conformal_coverage: float = 0.8
    """Nominal coverage for the conformal prediction interval published with
    every forecast. Default 0.8 (80%) — the band that catches the actual
    value 80% of the time under the residual-exchangeability assumption.

    Higher values (e.g. 0.9) widen the band; useful when the downstream
    automation must avoid false-negatives ("definitely not empty before
    6pm" needs > 90%). Lower values (e.g. 0.5) collapse the band to the
    median residual — useful only for diagnostic plots.

    Empirical coverage may differ from the nominal target when residuals
    are non-exchangeable across the forecast horizon (seasonal drift,
    regime change). The Results tab surfaces achieved coverage so users
    can diagnose calibration gaps."""

    quantiles: List[float] = field(default_factory=list)
    """Optional list of quantiles in (0, 1) for native-quantile training.
    Empty (default) trains a point forecast and wraps it in a post-hoc
    conformal band. Non-empty (e.g. [0.1, 0.5, 0.9]) routes the supported
    neural backends (DLinear) through a multi-quantile output head trained
    with the pinball loss, replacing the point loss for those backends.

    Backends without a quantile head (currently every backend except
    DLinear) fall back to the point-loss path and the conformal band
    continues to wrap their median prediction."""

    gap_handling: str = 'interpolate'
    """How to fill gaps after resampling:
    - ``ffill``: legacy behaviour — propagate the last observed value across
      every gap, large or small. Inserts artificial flat segments on a
      recorder outage that the model can over-fit.
    - ``interpolate`` (default): linear-interpolate gaps up to
      ``gap_max_minutes``; mark longer gaps as NaN so downstream dropna
      excludes them rather than imputing them with a stale value.
    - ``mask``: leave every gap as NaN (downstream dropna removes the row).
      Use when missing data should never be modelled."""

    gap_max_minutes: int = 90
    """Maximum gap (minutes) eligible for ``gap_handling='interpolate'``.
    Gaps longer than this are left as NaN regardless of method."""

    outlier_method: str = 'quantile'
    """Outlier-handling method:
    - ``quantile`` (default): clip upper tail at ``outlier_quantile``, lower
      tail per ``outlier_lower``.
    - ``mad``: Iglewicz-Hoaglin robust clip at ``median ± k · MAD`` with
      k=3.5. Less aggressive than the quantile clip on heavy-tailed but
      legitimate data (rainfall, occupancy).
    - ``off``: no clipping. Pair with a robust ``loss_fn`` (mae, huber) and
      a probabilistic target if your sensor genuinely has unbounded
      legitimate values."""

    outlier_quantile: float = 0.999
    """Upper-tail quantile for ``outlier_method='quantile'``. 0.999 trims the
    top 0.1% — less aggressive than the previous hardcoded 0.995 because HA
    sensor noise rarely needs a 0.5% top trim and legitimate peaks were
    being clipped. Lower this if your target has a clean upper bound."""

    outlier_lower: str = 'auto'
    """Lower bound for ``outlier_method='quantile'``:
    - ``auto`` (default): zero for cumulative sources, symmetric quantile
      otherwise. Matches the legacy ``positive_only=source_is_cumulative``
      logic.
    - ``zero``: clip at 0 — for non-negative quantities (power, energy).
    - ``symmetric``: clip at ``1 - outlier_quantile`` — for two-sided
      signed sensors (temperature delta, wind direction).
    - ``off``: no lower clip."""

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

    @property
    def effective_loss_balance(self) -> float:
        """v2.40.14: always 0.0 — the cumulative loss path is gone.

        Retained as a property so any caller / template that still
        references it gets a safe value rather than an AttributeError.
        """
        return 0.0

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
        if self.cv_folds < 2:
            raise ValueError(f'cv_folds must be >= 2, got {self.cv_folds}')
        if self.cv_folds > 20:
            # An upper guard is needed because the runner does not check the
            # absolute size of the test slice each fold — with cv_folds=1000
            # on 5000 rows you get 1000 folds of ~4 rows each, training stalls
            # for hours, and the UI shows no progress beyond "running".
            raise ValueError(
                f'cv_folds must be <= 20 to keep per-fold test slices large '
                f'enough for stable metrics, got {self.cv_folds}'
            )
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
        # v2.40.14: daily_loss_weight and loss_balance kept on the model
        # for YAML backwards-compat but are no-ops. Bounds validation
        # retained so a typo still surfaces at load.
        if self.daily_loss_weight < 0:
            raise ValueError(
                f'daily_loss_weight must be >= 0, got {self.daily_loss_weight}'
            )
        if self.loss_balance is not None and not (0.0 <= self.loss_balance <= 1.0):
            raise ValueError(
                f'loss_balance must be in [0, 1] or None, got {self.loss_balance}'
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

        # v2.37.6 added a warning here for nbeats / nhits / itransformer
        # — those three sliced ``x[:, :past_window_size, :]`` in their
        # forward pass and dropped the future block. v2.37.7 fixes all
        # three with an auxiliary future-feature head that adds a
        # per-horizon adjustment from the future block to the base
        # past-only forecast. Every neural backend in the registry now
        # consumes user future covariates, so the warning is no longer
        # needed and has been removed.


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


# Experiment-level YAML keys that older configs may carry but the code
# no longer reads. load_config strips them (with a log line) and
# rewrites the YAML so they don't linger as silent no-ops.
_DEPRECATED_EXPERIMENT_FIELDS = (
    'horizons_minutes',
    'database',
    'output_units',
    'custom_metrics',
    'stability_focus',
    'future_covariate_features',
)


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
            logger.warning(
                'Skipping malformed experiment entry: expected mapping, got %s',
                type(exp_data).__name__,
            )
            continue

        exp_name_for_err = exp_data.get('name', '<unnamed>') if isinstance(exp_data, dict) else '?'

        try:
            # Migration: silently remove deprecated fields.
            # 'database' removed in v2.33.1 (actuals always cached);
            # 'output_units', 'custom_metrics', 'stability_focus' and
            # 'future_covariate_features' removed in v2.41.0 — they were
            # parsed (and 'output_units' even shipped in the example
            # YAML) but consumed by nothing, silently absorbing
            # misconfiguration (audit F11).
            for _deprecated in _DEPRECATED_EXPERIMENT_FIELDS:
                if _deprecated in exp_data:
                    exp_data.pop(_deprecated)
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
        except (ValueError, TypeError, KeyError) as e:
            # One bad experiment must not kill the whole add-on. The remaining
            # experiments and the web UI must still come up so the user can
            # see the diagnostic and edit mlfl.yaml from inside HA.
            logger.error(
                'Experiment %r failed validation and will be skipped: %s',
                exp_name_for_err, e,
            )
            continue
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
            removed: list = []
            for exp in raw.get('experiments', []):
                if isinstance(exp, dict):
                    for fld in _DEPRECATED_EXPERIMENT_FIELDS:
                        if exp.pop(fld, None) is not None and fld not in removed:
                            removed.append(fld)
            atomic_yaml_write(config_path, raw)
            if removed:
                logger.info(
                    f'Migrated config: removed deprecated field(s): {", ".join(removed)}'
                )
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

    atomic_yaml_write(config_path, data)


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

    atomic_yaml_write(config_path, data)


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

    atomic_yaml_write(config_path, data)


def remove_experiment_covariate(
    config_path: Path | str,
    experiment_name: str,
    entity_id: str,
    role: Optional[str] = None,
    future_attribute: Optional[str] = None,
    future_value_key: Optional[str] = None,
) -> bool:
    """
    Remove a covariate from an experiment's config.

    `entity_id` can be either the full entity ID (e.g. ``sensor.current_charge``)
    or its short suffix (``current_charge``) — the covariate-analysis UI uses the
    short form because that's what becomes the dataframe column name.

    When the experiment configures the same entity multiple times (v2.38.2+
    allows this for e.g. ``cloud_coverage`` and ``temperature`` from the same
    ``weather.*`` entity), pass ``role`` / ``future_attribute`` /
    ``future_value_key`` to identify the specific row to remove.  Without
    those disambiguators a same-entity dedup would strip every matching row
    in one call, silently losing the other channels.

    Returns True if a covariate was removed, False if not found.
    """
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    def _entity_matches(cov: dict, target: str) -> bool:
        ent = cov.get('entity') or ''
        if not ent:
            return False
        return ent == target or ent.split('.')[-1] == target

    target_spec = {'entity': entity_id}
    if role is not None:
        target_spec['role'] = role
    if future_attribute is not None:
        target_spec['future_attribute'] = future_attribute
    if future_value_key is not None:
        target_spec['future_value_key'] = future_value_key

    removed = False
    for exp in data.get('experiments', []):
        if exp.get('name') != experiment_name:
            continue
        covs = exp.get('covariates', [])
        original_len = len(covs)
        same_entity = [c for c in covs if _entity_matches(c, entity_id)]
        if role is None and future_attribute is None and future_value_key is None and len(same_entity) > 1:
            # Caller didn't disambiguate but the entity has multiple rows
            # — refuse rather than silently stripping all of them. The UI
            # should always send the full (entity, role, future_attribute,
            # future_value_key) tuple in this case.
            logger.warning(
                'remove_experiment_covariate(%r, %r): entity is configured '
                '%d times — refusing to remove without role / future_attribute '
                '/ future_value_key disambiguation. Pass them explicitly.',
                experiment_name, entity_id, len(same_entity),
            )
            return False
        # When disambiguators given, match against the same (entity, role,
        # future_attribute, future_value_key) tuple as the add path uses.
        # Without disambiguators (single same-entity row), fall back to
        # entity-only match.
        if role is None and future_attribute is None and future_value_key is None:
            exp['covariates'] = [c for c in covs if not _entity_matches(c, entity_id)]
        else:
            # Normalise target's entity to the matching row's actual stored
            # form so the _same_covariate comparison succeeds regardless of
            # short-suffix vs full-entity input.
            kept = []
            removed_one = False
            for c in covs:
                if not removed_one and _entity_matches(c, entity_id):
                    cmp_target = dict(target_spec)
                    cmp_target['entity'] = c.get('entity', '')
                    if _same_covariate(c, cmp_target):
                        removed_one = True
                        continue
                kept.append(c)
            exp['covariates'] = kept
        removed = len(exp['covariates']) < original_len
        break

    if removed:
        atomic_yaml_write(config_path, data)

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
        atomic_yaml_write(config_path, data)

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
        # Reject only true duplicates — same (entity, role, source)
        # tuple. This allows the common pattern of consuming multiple
        # forecast attributes from a single weather entity (e.g.
        # weather.met_office_balsham gives cloud_coverage AND
        # temperature AND humidity from the same ``hourly`` service
        # forecast) as separate covariates.
        if any(_same_covariate(c, covariate) for c in covs):
            return False
        # Strip None values so YAML stays clean
        clean = {k: v for k, v in covariate.items() if v is not None}
        covs.append(clean)
        atomic_yaml_write(config_path, data)
        return True

    return False


def _same_covariate(a: dict, b: dict) -> bool:
    """Two covariates are the same configuration only if entity, role,
    AND the future-value source all match. A user who adds
    ``weather.met_office_balsham`` once for ``cloud_coverage`` and
    again for ``temperature`` is configuring two distinct covariates
    — they share an entity but differ in ``future_value_key``.

    The same disambiguation applies to ``role='lagged'`` because the
    v2.38.4 attribute-history path uses ``future_value_key`` to choose
    which weather attribute to pull historical numerics from — two
    lagged covariates from the same ``weather.*`` entity with
    different ``future_value_key`` values pull different signals."""
    if a.get('entity') != b.get('entity'):
        return False
    if a.get('role', 'lagged') != b.get('role', 'lagged'):
        return False
    # (attribute, value_key) routes to a specific forecast or attribute
    # metric for every role — different pairs = different covariates.
    if a.get('future_attribute', 'forecast') != b.get('future_attribute', 'forecast'):
        return False
    if a.get('future_value_key') != b.get('future_value_key'):
        return False
    return True


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
        atomic_yaml_write(config_path, data)
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
        atomic_yaml_write(config_path, data)

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
        atomic_yaml_write(config_path, data)

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

    atomic_yaml_write(config_path, data)


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
        atomic_yaml_write(config_path, data)
        return True

    return False
