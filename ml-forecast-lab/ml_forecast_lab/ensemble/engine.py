"""
Ensemble engine for combining forecasts from multiple models.

Supports three strategies:
1. Simple average — unweighted mean of predictions
2. Weighted average — inverse-MAE weighting
3. Stacking — Ridge regression meta-learner trained on out-of-fold predictions

The engine operates on per-fold predictions retained by the BenchmarkRunner,
so stacking avoids data leakage by training only on out-of-fold data.
"""

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EnsembleStrategy(enum.Enum):
    """Available ensemble combination strategies."""

    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    STACKING = "stacking"


@dataclass
class EnsembleResult:
    """Result of a single ensemble strategy."""

    strategy: EnsembleStrategy
    member_models: List[str]
    predictions: np.ndarray  # shape (n_samples,) per fold
    weights: Optional[Dict[str, float]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)
    meta_model: Optional[Any] = None  # for stacking


class EnsembleEngine:
    """
    Combine predictions from multiple models using configurable strategies.

    Parameters
    ----------
    metric_registry : object
        Metric registry with a ``compute_all`` method.
    production_metric : str
        Primary metric name for ranking (e.g. 'mae').
    metrics : list[str]
        Metrics to compute on ensemble predictions.
    """

    def __init__(
        self,
        metric_registry: Any,
        production_metric: str = "mae",
        metrics: Optional[List[str]] = None,
    ) -> None:
        self.metric_registry = metric_registry
        self.production_metric = production_metric
        self.metrics = metrics or ["mae", "rmse", "mape"]

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    @staticmethod
    def simple_average(
        fold_predictions: Dict[str, List[np.ndarray]],
    ) -> List[np.ndarray]:
        """
        Compute unweighted mean of model predictions for each fold.

        Parameters
        ----------
        fold_predictions : dict[str, list[np.ndarray]]
            model_name -> list of per-fold prediction arrays.

        Returns
        -------
        list[np.ndarray]
            Combined predictions for each fold.
        """
        model_names = list(fold_predictions.keys())
        n_folds = len(fold_predictions[model_names[0]])
        combined = []
        for f in range(n_folds):
            arrays = [fold_predictions[m][f] for m in model_names]
            # Stack and mean — handles both 1D and 2D
            combined.append(np.mean(np.stack(arrays, axis=0), axis=0))
        return combined

    @staticmethod
    def weighted_average(
        fold_predictions: Dict[str, List[np.ndarray]],
        model_metrics: Dict[str, float],
    ) -> Tuple[List[np.ndarray], Dict[str, float]]:
        """
        Compute inverse-MAE-weighted mean of predictions.

        Parameters
        ----------
        fold_predictions : dict[str, list[np.ndarray]]
            model_name -> list of per-fold predictions.
        model_metrics : dict[str, float]
            model_name -> MAE (or other error metric).
            Lower is better; weights are proportional to 1/metric.

        Returns
        -------
        combined : list[np.ndarray]
            Weighted predictions for each fold.
        weights : dict[str, float]
            Normalised weight per model.
        """
        model_names = list(fold_predictions.keys())
        raw_weights = {}
        for m in model_names:
            metric_val = model_metrics.get(m, np.inf)
            if metric_val <= 0 or not np.isfinite(metric_val):
                metric_val = 1e-8
            raw_weights[m] = 1.0 / metric_val

        total = sum(raw_weights.values())
        weights = {m: w / total for m, w in raw_weights.items()}

        n_folds = len(fold_predictions[model_names[0]])
        combined = []
        for f in range(n_folds):
            weighted_sum = None
            for m in model_names:
                arr = fold_predictions[m][f] * weights[m]
                if weighted_sum is None:
                    weighted_sum = arr.copy()
                else:
                    weighted_sum += arr
            combined.append(weighted_sum)

        logger.debug(f"Weighted average weights: {weights}")
        return combined, weights

    @staticmethod
    def stacking(
        fold_predictions: Dict[str, List[np.ndarray]],
        fold_actuals: List[np.ndarray],
        alpha: float = 1.0,
    ) -> Tuple[List[np.ndarray], Dict[str, float], Any]:
        """
        Train a Ridge regression meta-learner on out-of-fold predictions.

        For each fold, a meta-learner is trained on predictions from all
        *other* folds, then predicts on the held-out fold. This is a
        leave-one-fold-out cross-validation of the meta-learner itself.

        Parameters
        ----------
        fold_predictions : dict[str, list[np.ndarray]]
            model_name -> list of per-fold prediction arrays.
        fold_actuals : list[np.ndarray]
            Per-fold actual values.
        alpha : float
            Ridge regularisation strength.

        Returns
        -------
        combined : list[np.ndarray]
            Stacked predictions for each fold.
        weights : dict[str, float]
            Meta-learner coefficients (approximate model importance).
        meta_model : sklearn.linear_model.Ridge
            Trained meta-model (from the full-data fit).
        """
        from sklearn.linear_model import Ridge

        model_names = list(fold_predictions.keys())
        n_folds = len(fold_actuals)
        n_models = len(model_names)

        combined = [None] * n_folds

        # Leave-one-fold-out for the meta-learner
        for test_fold in range(n_folds):
            # Build training data from all other folds
            X_meta_parts = []
            y_meta_parts = []
            for train_fold in range(n_folds):
                if train_fold == test_fold:
                    continue
                # Build feature matrix: each column is a model's predictions
                n_samples = len(fold_actuals[train_fold])
                fold_features = np.column_stack([
                    fold_predictions[m][train_fold].ravel()[:n_samples]
                    for m in model_names
                ])
                X_meta_parts.append(fold_features)
                y_meta_parts.append(fold_actuals[train_fold].ravel()[:n_samples])

            X_meta_train = np.vstack(X_meta_parts)
            y_meta_train = np.concatenate(y_meta_parts)

            # Build test feature matrix
            n_test = len(fold_actuals[test_fold])
            X_meta_test = np.column_stack([
                fold_predictions[m][test_fold].ravel()[:n_test]
                for m in model_names
            ])

            # Fit and predict
            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(X_meta_train, y_meta_train)
            combined[test_fold] = ridge.predict(X_meta_test)

        # Fit a final meta-model on all data for weight extraction
        all_X = np.vstack([
            np.column_stack([
                fold_predictions[m][f].ravel()[:len(fold_actuals[f])]
                for m in model_names
            ])
            for f in range(n_folds)
        ])
        all_y = np.concatenate([
            fold_actuals[f].ravel()[:len(fold_actuals[f])]
            for f in range(n_folds)
        ])
        final_ridge = Ridge(alpha=alpha, fit_intercept=True)
        final_ridge.fit(all_X, all_y)

        # Extract approximate weights from coefficients
        coefs = final_ridge.coef_
        abs_coefs = np.abs(coefs)
        total = abs_coefs.sum()
        weights = {
            model_names[i]: float(abs_coefs[i] / total) if total > 0 else 1.0 / n_models
            for i in range(n_models)
        }

        logger.debug(f"Stacking weights: {weights}")
        return combined, weights, final_ridge

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------

    def run_all(
        self,
        strategies: List[EnsembleStrategy],
        fold_predictions: Dict[str, List[np.ndarray]],
        fold_actuals: List[np.ndarray],
        model_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[EnsembleStrategy, EnsembleResult]:
        """
        Run all requested ensemble strategies and compute metrics.

        Parameters
        ----------
        strategies : list[EnsembleStrategy]
            Which strategies to evaluate.
        fold_predictions : dict[str, list[np.ndarray]]
            model_name -> list of per-fold prediction arrays.
        fold_actuals : list[np.ndarray]
            Per-fold actual values.
        model_metrics : dict[str, float], optional
            Per-model MAE (or production metric) for weighted averaging.

        Returns
        -------
        dict[EnsembleStrategy, EnsembleResult]
            Results keyed by strategy.
        """
        member_models = list(fold_predictions.keys())
        if len(member_models) < 2:
            logger.warning("Ensemble requires >= 2 models, skipping")
            return {}

        # Validate fold counts match
        n_folds = len(fold_actuals)
        for m, preds in fold_predictions.items():
            if len(preds) != n_folds:
                logger.warning(
                    f"Model {m} has {len(preds)} folds vs {n_folds} actuals, "
                    f"excluding from ensemble"
                )
                fold_predictions = {
                    k: v for k, v in fold_predictions.items() if len(v) == n_folds
                }
                member_models = list(fold_predictions.keys())
                break

        if len(member_models) < 2:
            logger.warning("After filtering, <2 models remain. Skipping ensemble.")
            return {}

        results: Dict[EnsembleStrategy, EnsembleResult] = {}
        metrics_to_compute = list(set(self.metrics + [self.production_metric]))

        for strategy in strategies:
            t0 = time.time()
            try:
                if strategy == EnsembleStrategy.SIMPLE_AVERAGE:
                    combined = self.simple_average(fold_predictions)
                    weights = {m: 1.0 / len(member_models) for m in member_models}
                    meta = None

                elif strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
                    if not model_metrics:
                        logger.warning("No model metrics for weighted average, using simple average")
                        combined = self.simple_average(fold_predictions)
                        weights = {m: 1.0 / len(member_models) for m in member_models}
                    else:
                        combined, weights = self.weighted_average(
                            fold_predictions, model_metrics,
                        )
                    meta = None

                elif strategy == EnsembleStrategy.STACKING:
                    combined, weights, meta = self.stacking(
                        fold_predictions, fold_actuals,
                    )

                else:
                    logger.warning(f"Unknown strategy {strategy}, skipping")
                    continue

                # Compute per-fold metrics
                fold_metrics_list = []
                for f in range(n_folds):
                    y_true = fold_actuals[f].ravel()
                    y_pred = combined[f].ravel()
                    # Align lengths (stacking/ravel may differ)
                    min_len = min(len(y_true), len(y_pred))
                    fm = self.metric_registry.compute_all(
                        metrics_to_compute,
                        y_true[:min_len],
                        y_pred[:min_len],
                    )
                    fold_metrics_list.append(fm)

                # Aggregate metrics (mean across folds)
                agg_metrics: Dict[str, float] = {}
                for metric_name in metrics_to_compute:
                    vals = [
                        fm.get(metric_name, np.nan)
                        for fm in fold_metrics_list if fm
                    ]
                    if vals:
                        agg_metrics[metric_name] = float(np.nanmean(vals))

                elapsed = time.time() - t0
                logger.info(
                    f"Ensemble {strategy.value}: "
                    f"{self.production_metric}={agg_metrics.get(self.production_metric, np.nan):.4f} "
                    f"({elapsed:.2f}s)"
                )

                results[strategy] = EnsembleResult(
                    strategy=strategy,
                    member_models=member_models,
                    predictions=combined[-1] if combined else np.array([]),
                    weights=weights,
                    metrics=agg_metrics,
                    fold_metrics=fold_metrics_list,
                    meta_model=meta,
                )

            except Exception as e:
                logger.error(f"Ensemble strategy {strategy.value} failed: {e}", exc_info=True)
                continue

        return results
