"""ML Forecast Lab - Production machine learning forecasting framework."""

# Legacy imports (preserved for compatibility)
from .covariates import CovariateResolver
from .db import HistoryDB
from .ha_interface import (
    HAInterface,
    ensure_utc,
    normalise_history,
    parse_timestamp,
    state_to_float,
)
from .publishing import (
    daily_cumulative_series,
    dict_from_series,
    energy_already_used_today,
    make_entity_name,
    publish_forecasts,
)

# Core modules: configuration, preprocessing, and features
from .config import AppConfig, CovariateCfg, ExperimentCfg, load_config
from .features import (
    build_features,
    create_forecast_features,
    prepare_train_test,
    reshape_for_sequence,
)
from .preprocessing import (
    align_series,
    apply_log_transform,
    apply_transform,
    clip_outliers,
    cumulative_to_interval,
    invert_log_transform,
    invert_transform,
    power_to_energy,
    resample_to_grid,
    subtract_series,
)

__version__ = "1.21.1"

__all__ = [
    # Legacy
    "HAInterface",
    "HistoryDB",
    "CovariateResolver",
    "parse_timestamp",
    "ensure_utc",
    "normalise_history",
    "state_to_float",
    "make_entity_name",
    "dict_from_series",
    "daily_cumulative_series",
    "energy_already_used_today",
    "publish_forecasts",
    # Core config
    "AppConfig",
    "CovariateCfg",
    "ExperimentCfg",
    "load_config",
    # Core preprocessing
    "cumulative_to_interval",
    "resample_to_grid",
    "clip_outliers",
    "apply_log_transform",
    "invert_log_transform",
    "apply_transform",
    "invert_transform",
    "subtract_series",
    "power_to_energy",
    "align_series",
    # Core features
    "build_features",
    "prepare_train_test",
    "reshape_for_sequence",
    "create_forecast_features",
]
