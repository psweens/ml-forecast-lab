"""
XGBoost forecasting model backend for ML Forecast Lab.

Implements extreme gradient boosting for time series forecasting
with support for multi-step recursive prediction and native serialisation.
"""

import json
import logging
import warnings
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

from .base import ForecastModel

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn(
        "XGBoost is not installed. XGBoostModel will not be functional. "
        "Install it with: pip install xgboost",
        ImportWarning
    )


class XGBoostModel(ForecastModel):
    """
    XGBoost-based forecasting model for time series prediction.

    Supports efficient gradient boosting training with early stopping and
    multi-step recursive forecasting for extended time horizons.
    """

    @property
    def name(self) -> str:
        """Return the model identifier string."""
        return "xgboost"

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        tree_method: str = "hist",
        verbose: int = 0,
    ):
        """
        Initialise XGBoost forecasting model.

        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds (default 500).
        max_depth : int
            Maximum tree depth (default 6).
        learning_rate : float
            Boosting learning rate (default 0.05).
        subsample : float
            Fraction of samples for training each iteration (default 0.8).
        colsample_bytree : float
            Fraction of features for training each iteration (default 0.8).
        reg_alpha : float
            L1 regularisation coefficient (default 0.1).
        reg_lambda : float
            L2 regularisation coefficient (default 1.0).
        tree_method : str
            Tree building method, 'hist' or 'exact' (default 'hist').
        verbose : int
            Logging level (default 0 for silent).
        """
        super().__init__()

        if not XGBOOST_AVAILABLE:
            raise RuntimeError(
                "XGBoost is not installed. Install with: pip install xgboost"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.tree_method = tree_method
        self.verbose = verbose

        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names_: Optional[list] = None
        self.training_metadata: Dict[str, Any] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Train the XGBoost model with optional early stopping.

        Uses the last 20% of training data for validation if eval_set
        is not provided in kwargs.

        Parameters
        ----------
        X_train : np.ndarray
            Training features of shape (n_samples, n_features).
        y_train : np.ndarray
            Training targets of shape (n_samples,).
        **kwargs : Any
            Optional parameters:
            - eval_set: Tuple of (X_val, y_val) for early stopping
            - feature_names: List of feature names (default: numeric indices)

        Returns
        -------
        dict[str, Any]
            Training metadata including best_iteration and feature_importances.

        Raises
        ------
        RuntimeError
            If XGBoost is not available.
        ValueError
            If input data is invalid.
        """
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost is not available")

        # Validate inputs
        self._validate_X(X_train)
        y_train = self._validate_y(y_train)

        if len(X_train) != len(y_train):
            raise ValueError("X and y must have the same number of samples")

        # Store feature names
        feature_names = kwargs.get("feature_names")
        if feature_names is None:
            self.feature_names_ = [f"feature_{i}" for i in range(X_train.shape[1])]
        else:
            self.feature_names_ = list(feature_names)

        # Create validation set if not provided
        # Extract sample weights if provided
        sample_weight = kwargs.get("sample_weight")

        eval_set = kwargs.get("eval_set")
        if eval_set is None:
            split_idx = int(len(X_train) * 0.8)
            X_train_split, X_val = X_train[:split_idx], X_train[split_idx:]
            y_train_split, y_val = y_train[:split_idx], y_train[split_idx:]
            w_train = sample_weight[:split_idx] if sample_weight is not None else None
        else:
            X_train_split, y_train_split = X_train, y_train
            X_val, y_val = eval_set
            w_train = sample_weight

        # Create XGBRegressor with specified hyperparameters
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            tree_method=self.tree_method,
            verbosity=self.verbose,
            random_state=42,
            early_stopping_rounds=50,
        )

        # Train with early stopping
        self.model.fit(
            X_train_split,
            y_train_split,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Extract training metadata
        best_iteration = self.model.best_iteration
        feature_importances = dict(
            zip(
                self.feature_names_,
                self.model.feature_importances_
            )
        )

        # Normalise feature importances to sum to 1
        total_importance = sum(feature_importances.values())
        if total_importance > 0:
            feature_importances = {
                k: v / total_importance for k, v in feature_importances.items()
            }

        self.training_metadata = {
            "best_iteration": best_iteration,
            "feature_importances": feature_importances,
            "num_features": len(self.feature_names_),
        }

        self._is_fitted = True
        logger.info(
            f"XGBoost model trained successfully. "
            f"Best iteration: {best_iteration}, "
            f"Number of features: {len(self.feature_names_)}"
        )

        return {
            "time_seconds": 0.0,  # Placeholder; actual timing should be added
            **self.training_metadata
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make single-step predictions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples,).

        Raises
        ------
        RuntimeError
            If model is not fitted.
        ValueError
            If input shape is invalid.
        """
        self._validate_fitted()
        self._validate_X(X)

        if self.model is None:
            raise RuntimeError("Model is not fitted")

        if X.shape[1] != len(self.feature_names_):
            raise ValueError(
                f"Expected {len(self.feature_names_)} features, "
                f"but got {X.shape[1]}"
            )

        predictions = self.model.predict(X)
        return np.array(predictions, dtype=np.float32)

    def export_onnx(self, path: str) -> bool:
        """
        Export the model to ONNX format.

        XGBoost tree-based models do not benefit from ONNX export for
        hardware acceleration (Hailo optimisation targets neural networks).
        Returns False to indicate export is not recommended.

        Parameters
        ----------
        path : str
            Path where the ONNX model should be saved.

        Returns
        -------
        bool
            False (ONNX export not recommended for tree models).
        """
        logger.info(
            "XGBoost tree models do not benefit from ONNX export. "
            "Consider using neural network models for hardware acceleration."
        )
        return False

    def supports_hardware_accel(self) -> bool:
        """
        Check if the model supports hardware acceleration.

        XGBoost tree-based models are not optimised for hardware
        acceleration frameworks like Hailo.

        Returns
        -------
        bool
            False (tree models don't support hardware acceleration).
        """
        return False

    def get_params(self) -> Dict[str, Any]:
        """
        Return all hyperparameters as a dictionary.

        Returns
        -------
        dict[str, Any]
            Hyperparameter names mapped to their current values.
        """
        return deepcopy({
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "tree_method": self.tree_method,
            "verbose": self.verbose,
        })

    def set_params(self, **kwargs: Any) -> None:
        """
        Update hyperparameters.

        Parameters
        ----------
        **kwargs : Any
            Hyperparameters to update.

        Raises
        ------
        ValueError
            If an unknown hyperparameter is provided.
        """
        valid_params = {
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "reg_alpha", "reg_lambda", "tree_method", "verbose"
        }

        for key, value in kwargs.items():
            if key not in valid_params:
                raise ValueError(f"Unknown parameter: {key}")
            setattr(self, key, value)

        if self.is_fitted:
            logger.warning(
                "Hyperparameters updated on a fitted model. "
                "Call fit() again to use new hyperparameters."
            )

    def save(self, path: str) -> None:
        """
        Save the trained model to disk using XGBoost native format.

        Parameters
        ----------
        path : str
            Path where the model should be saved.

        Raises
        ------
        IOError
            If write fails.
        RuntimeError
            If model is not fitted.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save unfitted model")

        try:
            self.model.save_model(path)

            # Also save metadata separately
            metadata_path = path + ".metadata.json"
            metadata = {
                "feature_names": self.feature_names_,
                "training_metadata": {
                    "best_iteration": self.training_metadata.get("best_iteration"),
                    "feature_importances": self.training_metadata.get(
                        "feature_importances", {}
                    ),
                    "num_features": self.training_metadata.get("num_features"),
                },
                "hyperparameters": self.get_params(),
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model saved successfully to {path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise IOError(f"Failed to save model to {path}: {e}")

    def load(self, path: str) -> None:
        """
        Load a trained model from disk using XGBoost native format.

        Parameters
        ----------
        path : str
            Path to the saved model file.

        Raises
        ------
        IOError
            If read fails or file format is invalid.
        """
        try:
            self.model = xgb.XGBRegressor()
            self.model.load_model(path)

            # Load metadata if available
            metadata_path = path + ".metadata.json"
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                self.feature_names_ = metadata.get("feature_names")
                self.training_metadata = metadata.get("training_metadata", {})

                # Restore hyperparameters
                hyperparams = metadata.get("hyperparameters", {})
                self.set_params(**hyperparams)

            except FileNotFoundError:
                logger.warning(f"Metadata file not found at {metadata_path}")

            self._is_fitted = True
            logger.info(f"Model loaded successfully from {path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise IOError(f"Failed to load model from {path}: {e}")
