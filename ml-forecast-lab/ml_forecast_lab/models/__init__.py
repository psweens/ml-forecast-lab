"""
Models package for ML Forecast Lab.

Provides abstract base classes and factory infrastructure for forecast models.
Concrete model implementations are registered here as they are added to the package.

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
_optional_imports = {
    'LightGBMModel': 'lightgbm_backend',
    'XGBoostModel': 'xgboost_backend',
    'CatBoostModel': 'catboost_backend',
    'LSTMModel': 'lstm_backend',
    'GRUModel': 'gru_backend',
    'CNNModel': 'cnn_backend',
    'DLinearModel': 'dlinear_backend',
    'NLinearModel': 'nlinear_backend',
    'FITSModel': 'fits_backend',
    'NBeatsModel': 'nbeats_backend',
    'NHiTSModel': 'nhits_backend',
    'TiDEModel': 'tide_backend',
    'TSMixerModel': 'tsmixer_backend',
    'TimeMixerModel': 'timemixer_backend',
    'SparseTSFModel': 'sparsetsf_backend',
    'PatchTSTModel': 'patchtst_backend',
    'iTransformerModel': 'itransformer_backend',
    'CrossformerModel': 'crossformer_backend',
    'TimesNetModel': 'timesnet_backend',
    'TFTModel': 'tft_backend',
    'TimeXerModel': 'timexer_backend',
    'ModernTCNModel': 'moderntcn_backend',
    'SeasonalNaiveModel': 'seasonal_naive_backend',
    'ARIMAModel': 'statsforecast_backend',
    'ETSModel': 'statsforecast_backend',
    'ThetaModel': 'statsforecast_backend',
    'ChronosBoltModel': 'chronos_bolt_backend',
    'TTMModel': 'ttm_backend',
}

for _cls_name, _module in _optional_imports.items():
    try:
        _mod = __import__(f'ml_forecast_lab.models.{_module}', fromlist=[_cls_name])
        globals()[_cls_name] = getattr(_mod, _cls_name)
    except (ImportError, ModuleNotFoundError):
        globals()[_cls_name] = None

__all__ = [
    'ForecastModel',
    'ModelResult',
    'ModelRegistry',
    'get_registry',
] + list(_optional_imports.keys())

logger = logging.getLogger(__name__)

# Package version — imported from parent package
from ml_forecast_lab import __version__
