"""
Models package for ML Forecast Lab.

Provides abstract base classes and factory infrastructure for forecast models.
Concrete model implementations (LightGBM, LSTM, XGBoost, etc.) are registered
here as they are added to the package.

Exports
-------
ForecastModel
    Abstract base class for all forecast models.
ModelResult
    Dataclass for storing training and inference results.
ModelRegistry
    Factory and registry for creating model instances.
get_registry
    Function to access the global model registry.
"""

import logging

from .base import ForecastModel, ModelResult
from .registry import ModelRegistry, get_registry

# Try importing concrete implementations if available
try:
    from .lightgbm_backend import LightGBMModel
except (ImportError, ModuleNotFoundError):
    LightGBMModel = None

try:
    from .xgboost_backend import XGBoostModel
except (ImportError, ModuleNotFoundError):
    XGBoostModel = None

try:
    from .lstm_backend import LSTMModel
except (ImportError, ModuleNotFoundError):
    LSTMModel = None

__all__ = [
    'ForecastModel',
    'ModelResult',
    'ModelRegistry',
    'get_registry',
    'LightGBMModel',
    'XGBoostModel',
    'LSTMModel',
]

logger = logging.getLogger(__name__)

# Package version — imported from parent package
from ml_forecast_lab import __version__
