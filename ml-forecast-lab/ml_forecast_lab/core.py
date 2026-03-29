"""
Core forecasting engine for ML Forecast Lab.

Provides the main ForecastingEngine class that orchestrates model training,
evaluation, and prediction across multiple forecasting algorithms.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ForecastingEngine:
    """
    Multi-model forecasting engine for time series prediction and benchmarking.

    Supports training and evaluation of multiple forecasting models with
    comprehensive metrics and model comparison capabilities.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise the forecasting engine.

        Args:
            config: Optional configuration dictionary for engine parameters.
        """
        self.config = config or {}
        self.models = {}
        self.training_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.results: Dict[str, Dict[str, float]] = {}

        logger.info("ForecastingEngine initialised")

    def load_data(
        self,
        data: pd.DataFrame,
        train_size: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and split time series data.

        Args:
            data: Input time series data
            train_size: Proportion of data for training (0.0 to 1.0)

        Returns:
            Tuple of (training_data, test_data)
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")

        if train_size <= 0 or train_size >= 1:
            raise ValueError("train_size must be between 0 and 1")

        split_point = int(len(data) * train_size)
        self.training_data = data.iloc[:split_point]
        self.test_data = data.iloc[split_point:]

        logger.info(
            f"Data split: training={len(self.training_data)}, "
            f"test={len(self.test_data)}"
        )

        return self.training_data, self.test_data

    def register_model(self, name: str, model: Any) -> None:
        """
        Register a forecasting model for training and evaluation.

        Args:
            name: Unique identifier for the model
            model: Model instance with fit and predict methods
        """
        if name in self.models:
            logger.warning(f"Overwriting existing model: {name}")

        self.models[name] = model
        logger.info(f"Model registered: {name}")

    def train_model(self, model_name: str) -> bool:
        """
        Train a registered model on loaded training data.

        Args:
            model_name: Name of the model to train

        Returns:
            True if training succeeded, False otherwise
        """
        if model_name not in self.models:
            logger.error(f"Model not found: {model_name}")
            return False

        if self.training_data is None:
            logger.error("No training data loaded")
            return False

        try:
            model = self.models[model_name]
            # Extract features and target from training data
            X = self.training_data.iloc[:, :-1]
            y = self.training_data.iloc[:, -1]

            model.fit(X, y)
            logger.info(f"Model trained successfully: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Error training model {model_name}: {e}")
            return False

    def evaluate_model(self, model_name: str) -> Optional[Dict[str, float]]:
        """
        Evaluate a trained model on test data.

        Args:
            model_name: Name of the model to evaluate

        Returns:
            Dictionary of evaluation metrics, or None if evaluation failed
        """
        if model_name not in self.models:
            logger.error(f"Model not found: {model_name}")
            return None

        if self.test_data is None:
            logger.error("No test data loaded")
            return None

        try:
            model = self.models[model_name]
            X = self.test_data.iloc[:, :-1]
            y_true = self.test_data.iloc[:, -1]

            y_pred = model.predict(X)

            # Calculate metrics
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            metrics = {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
            }

            self.results[model_name] = metrics
            logger.info(f"Model evaluated: {model_name} - RMSE: {rmse:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            return None

    def get_best_model(self, metric: str = "rmse") -> Optional[str]:
        """
        Identify the best performing model based on a specified metric.

        Args:
            metric: Metric to use for comparison (default: rmse)

        Returns:
            Name of the best model, or None if no results available
        """
        if not self.results:
            logger.warning("No evaluation results available")
            return None

        best_model = min(
            self.results.keys(),
            key=lambda m: self.results[m].get(metric, float("inf")),
        )

        logger.info(f"Best model: {best_model} ({metric}={self.results[best_model][metric]:.4f})")

        return best_model

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all trained and evaluated models.

        Returns:
            Dictionary containing model results and summary statistics
        """
        return {
            "models_trained": len(self.models),
            "models_evaluated": len(self.results),
            "results": self.results,
        }
