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

# Core modules: configuration, preprocessing, and features
from .config import AppConfig, CovariateCfg, ExperimentCfg, SubtractCfg, load_config
from .features import (
    MISSING_SUFFIX,
    TARGET_MISSING_COLUMN,
    build_features,
    default_lag_windows,
    feature_warmup_rows,
    neural_covariate_columns,
    prepare_train_test,
    rebuild_fold_features,
    reshape_for_sequence,
)
from .preprocessing import (
    LoadSubtractError,
    align_series,
    causal_impute,
    apply_load_subtract,
    apply_log_transform,
    apply_transform,
    clip_outliers,
    cumulative_to_interval,
    invert_log_transform,
    invert_transform,
    power_to_energy,
    resample_to_grid,
    resolve_missingness,
    subtract_series,
)

__version__ = "2.51.0"

__all__ = [
    # Legacy
    "HAInterface",
    "HistoryDB",
    "CovariateResolver",
    "parse_timestamp",
    "ensure_utc",
    "normalise_history",
    "state_to_float",
    # Core config
    "AppConfig",
    "CovariateCfg",
    "ExperimentCfg",
    "SubtractCfg",
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
    "apply_load_subtract",
    "LoadSubtractError",
    "power_to_energy",
    "align_series",
    # Missingness: masking, indicators and causal imputation
    "causal_impute",
    "resolve_missingness",
    "MISSING_SUFFIX",
    "TARGET_MISSING_COLUMN",
    # Core features
    "build_features",
    "default_lag_windows",
    "feature_warmup_rows",
    "neural_covariate_columns",
    "prepare_train_test",
    "rebuild_fold_features",
    "reshape_for_sequence",
]
