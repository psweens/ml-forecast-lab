"""
Benchmarking orchestrator for ML Forecast Lab.

Manages cross-validated model training and evaluation, computing metrics
across multiple folds and ranking models by performance.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml_forecast_lab.models.base import ForecastModel

from .metrics import MetricRegistry, get_metric_registry

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """
    Results for a single model across all cross-validation folds.

    Attributes
    ----------
    model_name : str
        Model identifier.
    metrics : dict[str, float]
        Aggregated metrics (mean across folds).
    fold_metrics : list[dict[str, float]]
        Per-fold metrics.
    train_times : list[float]
        Training time in seconds for each fold.
    inference_times : list[float]
        Inference time in seconds for each fold.
    """

    model_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)
    train_times: List[float] = field(default_factory=list)
    inference_times: List[float] = field(default_factory=list)
    fold_predictions: List[np.ndarray] = field(default_factory=list)
    fold_actuals: List[np.ndarray] = field(default_factory=list)
    fold_train_targets: List[np.ndarray] = field(default_factory=list)

    @property
    def mean_train_time(self) -> float:
        """Average training time across folds."""
        if not self.train_times:
            return np.nan
        return float(np.mean(self.train_times))

    @property
    def std_train_time(self) -> float:
        """Standard deviation of training time across folds."""
        if len(self.train_times) < 2:
            return np.nan
        return float(np.std(self.train_times))

    @property
    def mean_inference_time(self) -> float:
        """Average inference time across folds."""
        if not self.inference_times:
            return np.nan
        return float(np.mean(self.inference_times))

    @property
    def std_inference_time(self) -> float:
        """Standard deviation of inference time across folds."""
        if len(self.inference_times) < 2:
            return np.nan
        return float(np.std(self.inference_times))


@dataclass
class BenchmarkResult:
    """
    Complete benchmarking results for all models.

    Attributes
    ----------
    experiment_name : str
        Name of the experiment.
    model_results : dict[str, ModelResult]
        Results for each model.
    rankings : dict[str, int]
        Ranking of models (1 = best, based on production_metric).
    best_model : str
        Name of the best-performing model.
    best_metric_value : float
        Metric value of the best model.
    metric_used : str
        Name of the metric used for ranking.
    timestamp : datetime
        When the benchmark was run.
    cv_strategy : str
        Cross-validation strategy used ('walk_forward' or 'sliding_window').
    n_folds : int
        Number of cross-validation folds.
    """

    experiment_name: str
    model_results: Dict[str, ModelResult] = field(default_factory=dict)
    rankings: Dict[str, int] = field(default_factory=dict)
    best_model: str = ''
    best_metric_value: float = np.nan
    metric_used: str = ''
    timestamp: datetime = field(default_factory=datetime.now)
    cv_strategy: str = 'walk_forward'
    n_folds: int = 5

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a summary DataFrame.

        Returns
        -------
        pd.DataFrame
            Summary with columns: model, rank, metric_name, metric_mean,
            metric_std, train_time_mean, inference_time_mean.
        """
        rows = []
        for model_name, result in self.model_results.items():
            rank = self.rankings.get(model_name, -1)

            # Get mean and std for production metric
            metric_key = self.metric_used
            metric_values = [
                fm.get(metric_key, np.nan)
                for fm in result.fold_metrics
            ]
            metric_mean = float(np.nanmean(metric_values))
            metric_std = float(np.nanstd(metric_values)) if len(metric_values) > 1 else np.nan

            rows.append({
                'model': model_name,
                'rank': rank,
                'mean_rank': result.metrics.get('mean_rank', np.nan),
                'metric_name': self.metric_used,
                'metric_mean': metric_mean,
                'metric_std': metric_std,
                'train_time_mean': result.mean_train_time,
                'inference_time_mean': result.mean_inference_time,
            })

        return pd.DataFrame(rows).sort_values('rank')


class BenchmarkRunner:
    """
    Orchestrates cross-validated benchmarking of multiple forecast models.

    Manages train/test splits, feature building, model training, metric
    computation, and model ranking.
    """

    def __init__(
        self,
        experiment_cfg: Dict[str, Any],
        feature_builder: Callable,
        metric_registry: Optional[MetricRegistry] = None,
    ) -> None:
        """
        Initialise the benchmark runner.

        Parameters
        ----------
        experiment_cfg : dict[str, Any]
            Configuration dictionary with:
            - 'name': experiment name
            - 'cv_strategy': 'walk_forward' or 'sliding_window'
            - 'cv_folds': number of folds
            - 'production_metric': metric to use for ranking
            - (other fields used by feature_builder)
        feature_builder : Callable
            Function(df, config, purpose) that builds features from data.
            Purpose is 'train' or 'test'.
        metric_registry : MetricRegistry, optional
            Registry of evaluation metrics. If None, uses global registry.

        Raises
        ------
        ValueError
            If cv_strategy is invalid.
        """
        valid_strategies = {'walk_forward', 'sliding_window'}
        if experiment_cfg.get('cv_strategy', 'walk_forward') not in valid_strategies:
            raise ValueError(
                f'cv_strategy must be one of {valid_strategies}'
            )

        self.experiment_cfg = experiment_cfg
        self.feature_builder = feature_builder
        self.metric_registry = metric_registry or get_metric_registry()

        self.cv_strategy = experiment_cfg.get('cv_strategy', 'walk_forward')
        self.cv_folds = experiment_cfg.get('cv_folds', 5)
        self.production_metric = experiment_cfg.get('production_metric', 'mae')
        self.metrics = experiment_cfg.get('metrics', ['mae', 'rmse', 'mase'])

        logger.info(
            f'BenchmarkRunner initialised: '
            f'strategy={self.cv_strategy}, folds={self.cv_folds}, '
            f'metric={self.production_metric}'
        )

    def _prepare_train_test_splits(
        self, df: pd.DataFrame
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare train/test fold indices based on CV strategy.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            List of (train_indices, test_indices) tuples.

        Notes
        -----
        Walk-forward: Test set size = len(df) / (n_folds + 1).
        Sliding window: All folds have equal train/test sizes.
        """
        n_samples = len(df)

        if self.cv_strategy == 'walk_forward':
            # Grow train set, fixed test size
            test_size = n_samples // (self.cv_folds + 1)
            splits = []
            for fold in range(self.cv_folds):
                train_end = n_samples - test_size * (self.cv_folds - fold)
                test_start = train_end
                test_end = test_start + test_size

                train_idx = np.arange(train_end)
                test_idx = np.arange(test_start, test_end)
                splits.append((train_idx, test_idx))

        else:  # sliding_window
            # Fixed train/test sizes, window slides
            fold_size = n_samples // (self.cv_folds + 1)
            train_size = fold_size * self.cv_folds
            test_size = fold_size

            splits = []
            for fold in range(self.cv_folds):
                start = fold * fold_size
                train_idx = np.arange(start, start + train_size)
                test_idx = np.arange(
                    start + train_size, start + train_size + test_size
                )
                splits.append((train_idx, test_idx))

        logger.info(f'Prepared {len(splits)} CV folds using {self.cv_strategy}')
        return splits

    def run_single_model(
        self,
        df: pd.DataFrame,
        model: ForecastModel,
        fold_indices: List[Tuple[np.ndarray, np.ndarray]],
        epoch_callback: Any = None,
    ) -> ModelResult:
        """
        Run a single model across all CV folds.

        Parameters
        ----------
        df : pd.DataFrame
            Full time series data.
        model : ForecastModel
            Model instance with fit and predict methods.
        fold_indices : list[tuple[np.ndarray, np.ndarray]]
            List of (train_indices, test_indices) tuples.

        Returns
        -------
        ModelResult
            Aggregated results across folds.
        """
        model_result = ModelResult(model_name=model.name)

        for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
            logger.debug(
                f'Processing fold {fold_idx + 1}/{len(fold_indices)} '
                f'for model {model.name}'
            )

            # Split data
            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_test = df.iloc[test_idx].reset_index(drop=True)

            # Build features
            try:
                X_train = self.feature_builder(
                    df_train, self.experiment_cfg, purpose='train'
                )
                X_test = self.feature_builder(
                    df_test, self.experiment_cfg, purpose='test'
                )
            except Exception as e:
                logger.error(f'Feature building failed for fold {fold_idx}: {e}')
                model_result.fold_metrics.append({})
                model_result.train_times.append(np.nan)
                model_result.inference_times.append(np.nan)
                continue

            # Extract targets (assume last column or 'target' column)
            if 'target' in df_train.columns:
                y_train = df_train['target'].values
                y_test = df_test['target'].values
            else:
                y_train = df_train.iloc[:, -1].values
                y_test = df_test.iloc[:, -1].values

            # Get feature names from dataframe columns
            feat_names = [c for c in df_train.columns if c != 'target']

            # Generate time-decay sample weights (exponential recency weighting)
            n_train_samples = len(y_train)
            half_life_days = self.experiment_cfg.get('recency_half_life_days', 7.0)
            if half_life_days > 0:
                interval_min = self.experiment_cfg.get('interval_minutes', 30)
                half_life = max(1, half_life_days * (24 * 60 / interval_min))
                decay_rate = np.log(2) / half_life
                sample_weights = np.exp(decay_rate * np.arange(n_train_samples))
                sample_weights = sample_weights / sample_weights.sum() * n_train_samples
            else:
                sample_weights = None  # Equal weighting when disabled

            # Generate sliding window sequence data for neural models
            sequence_kwargs = {}
            horizon_steps = None
            window_size = None
            neural_cov_cols = None
            if model.is_neural:
                try:
                    from ml_forecast_lab.features import create_sliding_windows
                    target_col = 'target'
                    engineered = {
                        'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
                        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_holiday',
                    }
                    engineered.update(c for c in df_train.columns if c.startswith('y_lag_'))
                    neural_cov_cols = [c for c in df_train.columns if c not in engineered and c != target_col]

                    # Compute horizon steps from config
                    horizons_min = self.experiment_cfg.get('horizons_minutes', [interval_min])
                    horizon_steps = [h // interval_min for h in horizons_min]

                    if target_col in df_train.columns:
                        # Use original df slice (with DatetimeIndex) for temporal features
                        df_train_raw = df.iloc[train_idx]
                        window_size = min(48, len(df_train_raw) // 3)
                        if window_size >= 12:
                            seq_X, seq_y, channel_names = create_sliding_windows(
                                df_train_raw, target_col, window_size=window_size,
                                covariate_cols=neural_cov_cols if neural_cov_cols else None,
                                add_temporal=True,
                                horizon_steps=horizon_steps,
                            )
                            sequence_kwargs['sequence_data'] = seq_X
                            sequence_kwargs['channel_names'] = channel_names
                            y_train = seq_y
                            X_train = X_train[-len(seq_y):]
                            if sample_weights is not None:
                                sample_weights = sample_weights[-len(seq_y):]
                            logger.debug(
                                f'Sliding windows for {model.name}: '
                                f'{seq_X.shape[1]} steps × {seq_X.shape[2]} channels, '
                                f'horizons={horizon_steps}: {channel_names}'
                            )
                except Exception as e:
                    logger.debug(f'Sliding window creation failed: {e}')

            # Train model
            train_start = time.time()
            try:
                # Create fold-specific epoch callback
                fold_cb = None
                if epoch_callback is not None:
                    _fold_idx = fold_idx
                    _n_folds = len(fold_indices)
                    def fold_cb(**cb_data):
                        epoch_callback(fold=_fold_idx + 1, total_folds=_n_folds, **cb_data)
                model.fit(X_train, y_train, feature_names=feat_names,
                          sample_weight=sample_weights,
                          epoch_callback=fold_cb,
                          **sequence_kwargs)
            except Exception as e:
                logger.error(f'Model training failed for fold {fold_idx}: {e}')
                model_result.fold_metrics.append({})
                model_result.train_times.append(np.nan)
                model_result.inference_times.append(np.nan)
                continue

            train_time = time.time() - train_start

            # Predict — use predict_sequence for neural models
            inference_start = time.time()
            try:
                if 'sequence_data' in sequence_kwargs and model.is_neural and window_size:
                    try:
                        from ml_forecast_lab.features import create_sliding_windows
                        # Bridge fold boundary: prepend trailing train rows for context
                        n_bridge = min(window_size, len(train_idx))
                        bridge_idx = np.concatenate([train_idx[-n_bridge:], test_idx])
                        df_combined_test = df.iloc[bridge_idx]

                        seq_X_test, seq_y_test, _ = create_sliding_windows(
                            df_combined_test, 'target', window_size=window_size,
                            covariate_cols=neural_cov_cols if neural_cov_cols else None,
                            add_temporal=True,
                            horizon_steps=horizon_steps,
                        )
                        y_pred = model.predict_sequence(seq_X_test)
                        y_test = seq_y_test
                    except Exception:
                        y_pred = model.predict(X_test)
                else:
                    y_pred = model.predict(X_test)
            except Exception as e:
                logger.error(f'Model prediction failed for fold {fold_idx}: {e}')
                model_result.fold_metrics.append({})
                model_result.train_times.append(train_time)
                model_result.inference_times.append(np.nan)
                continue

            inference_time = time.time() - inference_start

            # Compute metrics — per-horizon for multi-output, then averaged for ranking
            metrics_to_compute = list(set(self.metrics + [self.production_metric]))
            fold_metrics = {}
            if y_pred.ndim == 2 and y_test.ndim == 2:
                # Per-horizon metrics (e.g. rmse_h4, rmse_h16)
                for h_idx, h_step in enumerate(horizon_steps or []):
                    h_metrics = self.metric_registry.compute_all(
                        metrics_to_compute,
                        y_test[:, h_idx],
                        y_pred[:, h_idx],
                        y_train=y_train[:, h_idx] if y_train.ndim == 2 else y_train,
                    )
                    for metric_name, value in h_metrics.items():
                        fold_metrics[f"{metric_name}_h{h_step}"] = value

                # Horizon-averaged metrics (un-suffixed) for Demšar ranking
                for metric_name in metrics_to_compute:
                    h_values = [
                        fold_metrics.get(f"{metric_name}_h{h}", np.nan)
                        for h in (horizon_steps or [])
                    ]
                    fold_metrics[metric_name] = float(np.nanmean(h_values))
            else:
                fold_metrics = self.metric_registry.compute_all(
                    metrics_to_compute, y_test, y_pred, y_train=y_train,
                )

            model_result.fold_metrics.append(fold_metrics)
            model_result.train_times.append(train_time)
            model_result.inference_times.append(inference_time)
            model_result.fold_predictions.append(y_pred.copy())
            model_result.fold_actuals.append(y_test.copy())
            model_result.fold_train_targets.append(y_train.copy())

        # Aggregate across folds — compute mean for all metrics
        if model_result.fold_metrics:
            metrics_to_compute = list(set(self.metrics + [self.production_metric]))
            for metric_name in metrics_to_compute:
                values = [
                    fm.get(metric_name, np.nan)
                    for fm in model_result.fold_metrics
                    if fm
                ]
                if values:
                    model_result.metrics[metric_name] = float(np.nanmean(values))

        logger.info(
            f'Model {model.name} completed: '
            f'metric={model_result.metrics.get(self.production_metric, np.nan):.4f}, '
            f'train_time={model_result.mean_train_time:.2f}s'
        )

        return model_result

    def run_benchmark(
        self,
        df: pd.DataFrame,
        models: Dict[str, ForecastModel],
    ) -> BenchmarkResult:
        """
        Run benchmark for all models using cross-validation.

        Parameters
        ----------
        df : pd.DataFrame
            Input time series data.
        models : dict[str, ForecastModel]
            Dictionary of {model_name: model_instance}.

        Returns
        -------
        BenchmarkResult
            Complete benchmarking results with rankings.
        """
        logger.info(
            f'Starting benchmark: experiment={self.experiment_cfg.get("name")}, '
            f'models={list(models.keys())}'
        )

        # Prepare CV splits
        fold_indices = self._prepare_train_test_splits(df)

        # Run each model
        result = BenchmarkResult(
            experiment_name=self.experiment_cfg.get('name', 'unnamed'),
            cv_strategy=self.cv_strategy,
            n_folds=self.cv_folds,
            metric_used=self.production_metric,
        )

        for model_name, model in models.items():
            try:
                model_result = self.run_single_model(df, model, fold_indices)
                result.model_results[model_name] = model_result
            except Exception as e:
                logger.error(f'Benchmark failed for model {model_name}: {e}')
                result.model_results[model_name] = ModelResult(model_name=model_name)

        # Composite ranking across folds (Demšar 2006)
        # Within each fold, rank models by each of {MAE, RMSE, MASE},
        # average the per-metric ranks to get a composite fold rank,
        # then average composite ranks across folds.
        _higher_better = {'r_squared', 'coverage'}
        ranking_metrics = [m for m in self.metrics if m not in _higher_better]
        if not ranking_metrics:
            ranking_metrics = [self.production_metric]

        model_names = list(result.model_results.keys())
        n_folds = max(
            len(mr.fold_metrics) for mr in result.model_results.values()
        ) if result.model_results else 0

        # Compute per-fold composite ranks
        fold_ranks = {name: [] for name in model_names}
        for fold_idx in range(n_folds):
            # For each ranking metric, rank models in this fold
            per_metric_ranks = {name: [] for name in model_names}
            for metric_name in ranking_metrics:
                higher_is_better = metric_name in _higher_better
                fold_values = {}
                for name in model_names:
                    mr = result.model_results[name]
                    if fold_idx < len(mr.fold_metrics) and mr.fold_metrics[fold_idx]:
                        fold_values[name] = mr.fold_metrics[fold_idx].get(
                            metric_name, np.inf if not higher_is_better else -np.inf
                        )
                    else:
                        fold_values[name] = np.inf if not higher_is_better else -np.inf

                sorted_fold = sorted(
                    fold_values.items(),
                    key=lambda x: x[1],
                    reverse=higher_is_better,
                )
                for rank, (name, _) in enumerate(sorted_fold):
                    per_metric_ranks[name].append(rank + 1)

            # Average across metrics to get composite fold rank
            for name in model_names:
                composite = float(np.mean(per_metric_ranks[name])) if per_metric_ranks[name] else float('inf')
                fold_ranks[name].append(composite)

        # Mean composite rank per model across folds
        mean_ranks = {
            name: float(np.mean(ranks)) if ranks else float('inf')
            for name, ranks in fold_ranks.items()
        }

        # Store mean_rank in ModelResult.metrics for UI access
        for name, mean_rank in mean_ranks.items():
            result.model_results[name].metrics['mean_rank'] = mean_rank

        # Final ranking: sort by mean rank (lower = better)
        sorted_models = sorted(mean_ranks.items(), key=lambda x: x[1])
        result.rankings = {name: rank + 1 for rank, (name, _) in enumerate(sorted_models)}

        if sorted_models:
            result.best_model = sorted_models[0][0]
            result.best_metric_value = result.model_results[sorted_models[0][0]].metrics.get(
                self.production_metric, np.nan
            )

        logger.info(
            f'Benchmark complete. Best model: {result.best_model} '
            f'({self.production_metric}={result.best_metric_value:.4f}, '
            f'mean_rank={mean_ranks.get(result.best_model, 0):.1f})'
        )
        for name in sorted_models:
            model_name = name[0]
            mr = result.model_results[model_name]
            logger.info(
                f'  #{result.rankings[model_name]} {model_name}: '
                f'{self.production_metric}={mr.metrics.get(self.production_metric, np.nan):.4f}, '
                f'mean_rank={mean_ranks[model_name]:.2f}, '
                f'fold_ranks={fold_ranks[model_name]}'
            )

        return result

    def get_best_model(self, result: BenchmarkResult) -> str:
        """
        Get the name of the best-performing model.

        Parameters
        ----------
        result : BenchmarkResult
            Benchmark results.

        Returns
        -------
        str
            Name of the best model.

        Raises
        ------
        ValueError
            If no models in result.
        """
        if not result.model_results:
            raise ValueError('No models in benchmark result')

        return result.best_model
