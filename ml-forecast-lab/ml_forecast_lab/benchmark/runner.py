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
        self.metrics = experiment_cfg.get('metrics', ['mae', 'rmse', 'mape'])

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

            # Generate time-decay sample weights (exponential, half-life = 30% of data)
            n_train_samples = len(y_train)
            half_life = max(1, int(n_train_samples * 0.3))
            decay_rate = np.log(2) / half_life
            sample_weights = np.exp(decay_rate * np.arange(n_train_samples))
            sample_weights = sample_weights / sample_weights.sum() * n_train_samples

            # Generate sliding window sequence data for LSTM/CNN
            # Includes target + covariates + temporal features (hour_sin/cos, dow_sin/cos, is_weekend)
            sequence_kwargs = {}
            if model.name in ('lstm', 'cnn'):
                try:
                    from ml_forecast_lab.features import create_sliding_windows
                    target_col = 'target'
                    # Find covariate columns: anything that's not target or an engineered feature
                    engineered = {
                        'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
                        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_holiday',
                    }
                    engineered.update(c for c in df_train.columns if c.startswith(('y_lag_', 'y_rolling_')))
                    cov_cols = [c for c in df_train.columns if c not in engineered and c != target_col]
                    if target_col in df_train.columns:
                        window_size = min(48, len(df_train) // 3)  # 24h at 30-min
                        if window_size >= 12:
                            seq_X, seq_y, channel_names = create_sliding_windows(
                                df_train, target_col, window_size=window_size,
                                covariate_cols=cov_cols if cov_cols else None,
                                add_temporal=True,
                            )
                            sequence_kwargs['sequence_data'] = seq_X
                            sequence_kwargs['channel_names'] = channel_names
                            # Use windowed targets for training
                            y_train = seq_y
                            X_train = X_train[-len(seq_y):]  # Align flat features
                            sample_weights = sample_weights[-len(seq_y):]
                            logger.debug(
                                f'Sliding windows for {model.name}: '
                                f'{seq_X.shape[1]} steps × {seq_X.shape[2]} channels: {channel_names}'
                            )
                except Exception as e:
                    logger.debug(f'Sliding window creation failed: {e}')

            # Train model
            train_start = time.time()
            try:
                model.fit(X_train, y_train, feature_names=feat_names,
                          sample_weight=sample_weights, **sequence_kwargs)
            except Exception as e:
                logger.error(f'Model training failed for fold {fold_idx}: {e}')
                model_result.fold_metrics.append({})
                model_result.train_times.append(np.nan)
                model_result.inference_times.append(np.nan)
                continue

            train_time = time.time() - train_start

            # Predict — use sequence data for LSTM/CNN if available
            inference_start = time.time()
            try:
                if 'sequence_data' in sequence_kwargs and model.name in ('lstm', 'cnn'):
                    # Create windowed test data matching training format
                    try:
                        target_col = 'target'
                        engineered = {
                            'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
                            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_holiday',
                        }
                        engineered.update(c for c in df_test.columns if c.startswith(('y_lag_', 'y_rolling_')))
                        cov_cols = [c for c in df_test.columns if c not in engineered and c != target_col]
                        if target_col in df_test.columns:
                            seq_X_test, seq_y_test, _ = create_sliding_windows(
                                df_test, target_col, window_size=sequence_kwargs['sequence_data'].shape[1],
                                covariate_cols=cov_cols if cov_cols else None,
                                add_temporal=True,
                            )
                            import torch
                            model._model.eval()
                            with torch.no_grad():
                                X_t = torch.FloatTensor(seq_X_test)
                                y_pred = model._model(X_t).numpy()
                                y_pred = np.clip(y_pred, 0.0, None).astype(np.float32)
                            y_test = seq_y_test
                        else:
                            y_pred = model.predict(X_test)
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

            # Compute all configured metrics
            metrics_to_compute = list(set(self.metrics + [self.production_metric]))
            fold_metrics = self.metric_registry.compute_all(
                metrics_to_compute,
                y_test,
                y_pred,
                y_train=y_train,
            )

            model_result.fold_metrics.append(fold_metrics)
            model_result.train_times.append(train_time)
            model_result.inference_times.append(inference_time)

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

        # Rank models by mean rank across folds (Demšar 2006)
        # Within each fold, rank models by production metric, then average ranks
        higher_is_better = self.production_metric in {'r_squared', 'coverage'}
        model_names = list(result.model_results.keys())
        n_folds = max(
            len(mr.fold_metrics) for mr in result.model_results.values()
        ) if result.model_results else 0

        # Compute per-fold ranks
        fold_ranks = {name: [] for name in model_names}
        for fold_idx in range(n_folds):
            fold_values = {}
            for name in model_names:
                mr = result.model_results[name]
                if fold_idx < len(mr.fold_metrics) and mr.fold_metrics[fold_idx]:
                    fold_values[name] = mr.fold_metrics[fold_idx].get(
                        self.production_metric, np.inf if not higher_is_better else -np.inf
                    )
                else:
                    fold_values[name] = np.inf if not higher_is_better else -np.inf

            sorted_fold = sorted(
                fold_values.items(),
                key=lambda x: x[1],
                reverse=higher_is_better,
            )
            for rank, (name, _) in enumerate(sorted_fold):
                fold_ranks[name].append(rank + 1)

        # Mean rank per model
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
