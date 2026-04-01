"""
Abstract base class and data structures for forecast models.

Defines the ForecastModel ABC that all concrete model implementations
must inherit from, along with the ModelResult dataclass for storing
training and inference outcomes.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """
    Container for model training and inference results.

    Attributes
    ----------
    model_name : str
        Identifier of the model backend (e.g. 'lightgbm', 'lstm').
    predictions : np.ndarray
        Forecast predictions of shape (n_samples, horizon) or (n_samples,)
        for single-step predictions.
    train_time_seconds : float
        Wall-clock training time in seconds.
    inference_time_seconds : float
        Wall-clock inference time in seconds (for predict or predict_multi).
    metrics : dict[str, float]
        Calculated metrics (e.g. {'mae': 0.15, 'rmse': 0.22}).
    fold_metrics : list[dict[str, float]]
        Per-fold metrics for cross-validation. Each entry is a dictionary
        of metric names to values.
    hyperparameters : dict
        Current hyperparameter configuration of the model.
    """

    model_name: str
    predictions: np.ndarray
    train_time_seconds: float
    inference_time_seconds: float
    metrics: Dict[str, float] = field(default_factory=dict)
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return detailed representation of result."""
        return (
            f'ModelResult(model_name={self.model_name!r}, '
            f'predictions.shape={self.predictions.shape}, '
            f'train_time={self.train_time_seconds:.2f}s, '
            f'inference_time={self.inference_time_seconds:.4f}s, '
            f'metrics={self.metrics})'
        )


class ForecastModel(ABC):
    """
    Abstract base class for all time-series forecast models.

    All concrete implementations must:
    1. Implement the abstract methods (name, fit, predict, etc.)
    2. Properly validate and type-hint inputs
    3. Support serialisation/deserialisation
    4. Provide logging for important events

    The model receives FLAT feature matrices of shape (n_samples, n_features)
    from the feature engineering pipeline. Models that require sequence data
    (e.g. LSTM, GRU) should reshape internally using utilities from the
    features module.
    """

    def __init__(self) -> None:
        """Initialise base model."""
        self._is_fitted = False
        self._fit_timestamp: Optional[float] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the model identifier string.

        Returns
        -------
        str
            Unique model name (e.g. 'lightgbm', 'lstm', 'xgboost').
            Used as the key in the model registry.
        """
        pass

    @property
    def is_fitted(self) -> bool:
        """
        Return whether the model has been trained.

        Returns
        -------
        bool
            True if fit() has been called successfully.
        """
        return self._is_fitted

    @property
    def is_neural(self) -> bool:
        """Whether this model requires sequence (sliding-window) input."""
        return False

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Train the model on the provided data.

        Parameters
        ----------
        X_train : np.ndarray
            Training features of shape (n_samples, n_features).
        y_train : np.ndarray
            Training targets of shape (n_samples,) or (n_samples, 1).
        **kwargs : Any
            Optional model-specific parameters (e.g. validation set,
            early stopping patience, sample weights).

        Returns
        -------
        dict[str, Any]
            Training metadata including at minimum:
            - 'time_seconds': Wall-clock training duration
            - 'epochs': Number of training epochs (for neural models)
            - Any other backend-specific metrics

        Notes
        -----
        Implementation should:
        1. Validate input shapes and types
        2. Log training progress
        3. Set self._is_fitted = True upon success
        4. Return comprehensive training metadata
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate single-step-ahead forecast.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples,) or (n_samples, 1).

        Raises
        ------
        RuntimeError
            If model has not been fitted.

        Notes
        -----
        For multi-step forecasting, use predict_multi() instead.
        """
        if not self.is_fitted:
            raise RuntimeError(f'{self.name} model must be fitted before prediction')

    def predict_multi(
        self,
        X: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """
        Generate multi-step-ahead forecast.

        Default implementation performs recursive single-step predictions,
        re-feeding previous predictions as features. Subclasses can
        override for direct multi-output or attention-based approaches.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        horizon : int
            Number of steps to forecast ahead.

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples, horizon).

        Raises
        ------
        ValueError
            If horizon < 1.
        RuntimeError
            If model has not been fitted.

        Notes
        -----
        The recursive approach accumulates forecast error. For critical
        applications, consider direct multi-output models or ensembles.
        """
        if not self.is_fitted:
            raise RuntimeError(f'{self.name} model must be fitted before prediction')
        if horizon < 1:
            raise ValueError(f'horizon must be >= 1, got {horizon}')

        n_samples, n_features = X.shape
        predictions = np.zeros((n_samples, horizon), dtype=np.float32)

        X_current = X.copy()
        for h in range(horizon):
            y_pred = self.predict(X_current)
            if y_pred.ndim > 1:
                y_pred = y_pred.ravel()
            predictions[:, h] = y_pred

            # For recursive forecasting, shift features and append predictions
            # This is a naive approach; sophisticated models should override
            if h < horizon - 1:
                X_current = X_current.copy()
                # Shift lagged features if they exist in position
                if n_features > 1:
                    X_current[:, :-1] = X_current[:, 1:]
                X_current[:, 0] = y_pred

        logger.debug(
            f'{self.name} produced {horizon}-step forecast '
            f'of shape {predictions.shape}'
        )
        return predictions

    @abstractmethod
    def export_onnx(self, path: str) -> bool:
        """
        Export model to ONNX format for hardware deployment.

        ONNX (Open Neural Network Exchange) format enables deployment
        on specialised accelerators (e.g. Hailo NPU). Not all models
        support this format.

        Parameters
        ----------
        path : str
            File path where the ONNX model will be saved.

        Returns
        -------
        bool
            True if export succeeded, False if not supported by this model.

        Notes
        -----
        Unsupported models should return False without raising an exception.
        Concrete implementations should log details about unsupported features.
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Return all hyperparameters as a dictionary.

        Returns
        -------
        dict[str, Any]
            Hyperparameter names mapped to their current values.

        Notes
        -----
        Must return a deep copy to prevent external modification.
        """
        pass

    @abstractmethod
    def set_params(self, **kwargs: Any) -> None:
        """
        Update hyperparameters.

        Parameters
        ----------
        **kwargs : Any
            Hyperparameters to update. Invalid parameter names should
            raise ValueError.

        Raises
        ------
        ValueError
            If an unknown hyperparameter is provided.

        Notes
        -----
        Updates should not affect a fitted model; typically used before
        fitting. Implementations may log warnings if called on fitted models.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Serialise model state to disk.

        Parameters
        ----------
        path : str
            File path for saving. Should include appropriate file extension
            (e.g. '.pkl', '.joblib').

        Raises
        ------
        IOError
            If write fails.

        Notes
        -----
        Implementation should handle model-specific serialisation
        (e.g. LightGBM models may use native .txt format).
        The saved file must be loadable with load().
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Deserialise model state from disk.

        Parameters
        ----------
        path : str
            File path to the saved model.

        Raises
        ------
        IOError
            If read fails or file format is invalid.

        Notes
        -----
        After load(), the model should be in the same state as when save()
        was called, including self._is_fitted = True.
        """
        pass

    @abstractmethod
    def supports_hardware_accel(self) -> bool:
        """
        Return whether this model supports hardware acceleration (Hailo NPU).

        Returns
        -------
        bool
            True if the model can be optimised for Hailo NPU deployment,
            False otherwise.

        Notes
        -----
        Hardware support depends on model architecture and quantisation
        compatibility. This method provides a quick check before attempting
        ONNX export and compilation.
        """
        pass

    def _validate_fitted(self) -> None:
        """
        Raise RuntimeError if model is not fitted.

        Utility method for subclasses to enforce fit() before operations.

        Raises
        ------
        RuntimeError
            If self._is_fitted is False.
        """
        if not self.is_fitted:
            raise RuntimeError(f'{self.name} model must be fitted before this operation')

    def _validate_X(self, X: np.ndarray) -> None:
        """
        Validate input feature array.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix to validate.

        Raises
        ------
        TypeError
            If X is not a 2D numpy array.
        ValueError
            If X is empty.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f'X must be a numpy array, got {type(X).__name__}')
        if X.ndim != 2:
            raise TypeError(f'X must be 2D, got shape {X.shape}')
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f'X cannot be empty, got shape {X.shape}')

    def _validate_y(self, y: np.ndarray) -> np.ndarray:
        """
        Validate and flatten target array.

        Parameters
        ----------
        y : np.ndarray
            Target array of shape (n_samples,) or (n_samples, 1).

        Returns
        -------
        np.ndarray
            Flattened target array of shape (n_samples,).

        Raises
        ------
        TypeError
            If y is not a numpy array.
        ValueError
            If shape does not match expected dimensions.
        """
        if not isinstance(y, np.ndarray):
            raise TypeError(f'y must be a numpy array, got {type(y).__name__}')
        if y.ndim == 1:
            return y
        elif y.ndim == 2 and y.shape[1] == 1:
            return y.ravel()
        else:
            raise ValueError(f'y must be 1D or (n, 1), got shape {y.shape}')

    def __repr__(self) -> str:
        """Return string representation of model."""
        fitted_str = 'fitted' if self.is_fitted else 'unfitted'
        return f'{self.__class__.__name__}({self.name!r}, {fitted_str})'
