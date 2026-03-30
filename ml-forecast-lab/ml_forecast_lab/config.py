"""
Configuration module for ML Forecast Lab.

Provides dataclasses for experiment configuration, covariate specification,
and application settings, with YAML loading capabilities.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


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

    horizons_minutes: List[int] = field(default_factory=lambda: [120, 480, 720])
    """Prediction horizons in minutes: [2h, 8h, 12h] by default."""

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

    metrics: List[str] = field(default_factory=lambda: ['mae', 'rmse', 'mape'])
    """Standard metrics to compute."""

    custom_metrics: Optional[Dict[str, str]] = None
    """Custom metrics as {name: 'Python expression'} using y_true, y_pred."""

    production_model: Optional[str] = None
    """Which model to use in production; if None, auto-select best by production_metric."""

    production_metric: str = 'mae'
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

    subtract: List[str] = field(default_factory=list)
    """Entity IDs to subtract from target (e.g. solar generation from grid import)."""

    mode: str = 'lab'
    """Operational mode: 'lab' (benchmark all models) or 'production' (forecast with best model)."""

    max_age: int = 365
    """Maximum days to keep in SQLite cache."""

    future_periods: int = 48
    """Number of future periods to forecast."""

    publish_name: Optional[str] = None
    """Override name for published HA entities."""

    database: bool = False
    """Whether to cache history in SQLite."""

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
        if self.cv_embargo_periods < 0:
            raise ValueError(
                f'cv_embargo_periods must be >= 0, got {self.cv_embargo_periods}'
            )
        if not self.horizons_minutes or not all(h > 0 for h in self.horizons_minutes):
            raise ValueError(
                f'horizons_minutes must be non-empty list of positive integers'
            )
        if self.days_history < 1:
            raise ValueError(f'days_history must be >= 1, got {self.days_history}')
        if self.interval_minutes < 1:
            raise ValueError(
                f'interval_minutes must be >= 1, got {self.interval_minutes}'
            )


@dataclass
class AppConfig:
    """Application-level configuration."""

    update_every_minutes: int = 5
    """Update frequency for model inference and forecasts."""

    timezone: str = 'UTC'
    """Timezone for temporal features."""

    experiments: List[ExperimentCfg] = field(default_factory=list)
    """List of experiment configurations."""

    hailo_enabled: bool = False
    """Whether to use Hailo accelerator (if available)."""

    cpu_cores: int = 0
    """Number of CPU cores for model training. 0 = all available."""

    nice_priority: int = 10
    """Process priority for training (0=normal, 19=lowest). Default 10."""

    def __post_init__(self) -> None:
        """Validate application configuration."""
        if self.update_every_minutes < 1:
            raise ValueError(
                f'update_every_minutes must be >= 1, got {self.update_every_minutes}'
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
        hailo_enabled: false
        experiments:
          - name: solar_forecast
            target_entity: sensor.solar_generation_w
            source_is_cumulative: true
            reset_daily: true
            days_history: 30
            interval_minutes: 30
            horizons_minutes: [120, 480, 1440]
            units: W
            log_transform: false
            country: GB
            models_enabled: [lightgbm, xgboost]
            cv_strategy: walk_forward
            cv_folds: 5
            cv_embargo_periods: 2
            metrics: [mae, rmse, mape]
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

    logger.info(f'Loading configuration from {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError('Configuration file must contain a YAML dictionary')

    # Parse experiments
    experiments_data = data.pop('experiments', [])
    experiments = []

    exp_fields = {f.name for f in dataclasses.fields(ExperimentCfg)}
    cov_fields = {f.name for f in dataclasses.fields(CovariateCfg)}
    app_fields = {f.name for f in dataclasses.fields(AppConfig)} - {'experiments'}

    for exp_data in experiments_data:
        if not isinstance(exp_data, dict):
            raise ValueError(
                f'Each experiment must be a dictionary, got {type(exp_data)}'
            )

        # Parse covariates
        covariates_data = exp_data.pop('covariates', [])
        covariates = []
        for cov in covariates_data:
            unknown_cov = set(cov) - cov_fields
            if unknown_cov:
                logger.warning(f'Ignoring unknown covariate fields: {unknown_cov}')
                cov = {k: v for k, v in cov.items() if k in cov_fields}
            covariates.append(CovariateCfg(**cov))

        # Filter unknown experiment fields
        unknown_exp = set(exp_data) - exp_fields
        if unknown_exp:
            logger.warning(f'Ignoring unknown experiment fields: {unknown_exp}')
            exp_data = {k: v for k, v in exp_data.items() if k in exp_fields}

        exp = ExperimentCfg(**exp_data, covariates=covariates)
        experiments.append(exp)

    # Filter unknown app-level fields
    unknown_app = set(data) - app_fields
    if unknown_app:
        logger.warning(f'Ignoring unknown app config fields: {unknown_app}')
        data = {k: v for k, v in data.items() if k in app_fields}

    app_config = AppConfig(**data, experiments=experiments)
    logger.info(
        f'Configuration loaded: {len(app_config.experiments)} experiment(s)'
    )
    return app_config
