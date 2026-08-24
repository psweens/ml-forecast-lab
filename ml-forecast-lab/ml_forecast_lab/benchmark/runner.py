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

from ml_forecast_lab.models.base import ForecastModel, TrainingCancelled

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
    fold_train_metrics: List[Dict[str, float]] = field(default_factory=list)
    train_metrics: Dict[str, float] = field(default_factory=dict)
    training_history: Optional[Dict[str, List[float]]] = field(default=None)
    # Daily-cumulative metrics: same MAE/RMSE/MASE but computed on per-day
    # totals (each day's predictions summed, each day's actuals summed,
    # then compared). Better for use cases where the daily total matters
    # more than per-interval precision (e.g. daily energy / hot-water demand).
    daily_fold_metrics: List[Dict[str, float]] = field(default_factory=list)
    daily_metrics: Dict[str, float] = field(default_factory=dict)

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
    # Parallel ranking computed from daily-cumulative metrics. The primary
    # `rankings` (per-interval, h=1) still drives Promote / Tuning / sensor
    # publishing; `daily_rankings` is informational so users can see which
    # model wins on each criterion.
    daily_rankings: Dict[str, int] = field(default_factory=dict)
    best_model: str = ''
    best_metric_value: float = np.nan
    metric_used: str = ''
    timestamp: datetime = field(default_factory=datetime.now)
    cv_strategy: str = 'walk_forward'
    n_folds: int = 5
    # Models that failed at least one fold are excluded from the ranked
    # pool so they don't artificially take last-place "wins" off other
    # models. They still appear in ``model_results`` for diagnostic
    # surfaces; the UI shows them in a "did not complete" list rather
    # than at the bottom of the leaderboard.
    did_not_complete: List[str] = field(default_factory=list)
    # Models excluded from the daily-cumulative ranking specifically
    # (ranked in the per-interval leaderboard, but a fold's daily totals
    # weren't computable). Surfaced under the Daily table, not the main
    # leaderboard's DNC section.
    did_not_complete_daily: List[str] = field(default_factory=list)

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
        Walk-forward: train set grows; test set size = n // (n_folds + 1).
        Sliding window: fixed train and test sizes that slide forward; the
        stride is sized so the final fold's test window aligns with the end
        of the data (so the most recent rows are always evaluated).

        Both strategies honour ``cv_embargo_periods`` from the experiment
        config: that many rows immediately preceding each test window are
        excluded from training to prevent target leakage through lag /
        rolling features whose forecast horizon overlaps the test inputs.
        """
        n_samples = len(df)
        embargo = max(0, int(self.experiment_cfg.get('cv_embargo_periods', 0)))

        if self.cv_strategy == 'walk_forward':
            # Expanding train window, fixed test size. The final fold's test
            # window ends at n_samples so the leaderboard always reflects
            # the most-recent slice of the series.
            test_size = max(1, n_samples // (self.cv_folds + 1))
            splits: List[Tuple[np.ndarray, np.ndarray]] = []
            for fold in range(self.cv_folds):
                test_start = n_samples - test_size * (self.cv_folds - fold)
                test_end = min(test_start + test_size, n_samples)
                train_end = max(0, test_start - embargo)

                if test_start >= n_samples or train_end <= 0:
                    continue

                train_idx = np.arange(0, train_end)
                test_idx = np.arange(test_start, test_end)
                if len(train_idx) > 0 and len(test_idx) > 0:
                    splits.append((train_idx, test_idx))

        else:  # sliding_window
            # Fixed-size train and test windows that slide forward. The
            # previous implementation set train_size = test_size * cv_folds
            # with a stride of one test_size — that needs roughly
            # 2 * cv_folds * test_size rows to fit, so every fold past
            # the first produced out-of-range test indices on realistic
            # n_samples / cv_folds combinations. The new layout sizes the
            # train window to roughly half of what remains after a single
            # test slot, then distributes the leftover space as the stride
            # across the remaining folds, snapping the last fold's test to
            # the end of the data.
            test_size = max(1, n_samples // (self.cv_folds + 1))
            train_size = max(test_size, (n_samples - test_size - embargo) // 2)
            first_test_start = train_size + embargo
            last_test_start = max(first_test_start, n_samples - test_size)

            splits = []
            for fold in range(self.cv_folds):
                if self.cv_folds == 1:
                    test_start = first_test_start
                else:
                    # Linear interpolation of test_start across folds so the
                    # last fold's test window ends exactly at n_samples.
                    test_start = first_test_start + int(round(
                        fold * (last_test_start - first_test_start)
                        / (self.cv_folds - 1)
                    ))
                test_end = min(test_start + test_size, n_samples)
                train_end = max(0, test_start - embargo)
                train_start = max(0, train_end - train_size)

                if train_end <= train_start or test_end <= test_start:
                    continue
                if test_start >= n_samples:
                    continue

                train_idx = np.arange(train_start, train_end)
                test_idx = np.arange(test_start, test_end)
                splits.append((train_idx, test_idx))

        if not splits:
            raise ValueError(
                f'Could not create any CV splits for n={n_samples}, '
                f'folds={self.cv_folds}, embargo={embargo}, '
                f'strategy={self.cv_strategy}'
            )

        logger.info(
            f'Prepared {len(splits)} CV folds using {self.cv_strategy} '
            f'(embargo={embargo})'
        )
        return splits

    def run_single_model(
        self,
        df: pd.DataFrame,
        model: ForecastModel,
        fold_indices: List[Tuple[np.ndarray, np.ndarray]],
        epoch_callback: Any = None,
        precomputed_sequences: dict = None,
        cancel_event: Any = None,
        future_features_df: Optional[pd.DataFrame] = None,
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
        n_folds = len(fold_indices)

        # Wrap the caller's epoch callback so a set cancel_event raises
        # TrainingCancelled from inside the backend's epoch loop —
        # _emit_epoch re-raises it and fit() unwinds at the next epoch
        # boundary. This is the only way to stop an executor thread
        # that asyncio.Task.cancel() cannot reach (audit F10).
        if cancel_event is not None:
            _user_cb = epoch_callback

            def epoch_callback(**cb_data):  # noqa: F811 — deliberate shadow
                if cancel_event.is_set():
                    raise TrainingCancelled(
                        f'{model.name}: cancelled via stop-training'
                    )
                if _user_cb is not None:
                    _user_cb(**cb_data)

        for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
            if cancel_event is not None and cancel_event.is_set():
                raise TrainingCancelled(
                    f'{model.name}: cancelled before fold {fold_idx + 1}'
                )
            fold_start_time = time.time()
            logger.info(
                f'  [fold {fold_idx + 1}/{n_folds}] {model.name}: '
                f'train={len(train_idx)} test={len(test_idx)}'
            )

            # Capture DatetimeIndex BEFORE the reset so we can group test
            # predictions by date for daily-cumulative metrics. Falls back to
            # an empty DatetimeIndex when df has no DatetimeIndex (e.g. test
            # fixtures with RangeIndex).
            if isinstance(df.index, pd.DatetimeIndex):
                test_timestamps = df.index[test_idx]
                train_timestamps = df.index[train_idx]
            else:
                test_timestamps = pd.DatetimeIndex([])
                train_timestamps = pd.DatetimeIndex([])

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
                logger.error(
                    f'Feature building failed for model={model.name} fold={fold_idx + 1}/{len(fold_indices)}: {e}',
                    exc_info=True,
                )
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
            # Fallback matches the ExperimentCfg default (0.0 = disabled);
            # a 7.0 fallback here silently enabled recency weighting for
            # any caller building a partial cfg dict (audit F18).
            half_life_days = self.experiment_cfg.get('recency_half_life_days', 0.0)
            if half_life_days > 0:
                interval_min = self.experiment_cfg.get('interval_minutes', 30)
                half_life = max(1, half_life_days * (24 * 60 / interval_min))
                decay_rate = np.log(2) / half_life
                sample_weights = np.exp(decay_rate * np.arange(n_train_samples))
                sample_weights = sample_weights / sample_weights.sum() * n_train_samples
            else:
                sample_weights = None  # Equal weighting when disabled

            # Generate sliding window sequence data for neural models.
            # When `precomputed_sequences` is provided (e.g. from tuning,
            # where the fold split is identical across all trials), skip
            # the expensive create_sliding_windows call and reuse the
            # pre-built arrays directly.
            sequence_kwargs = {}
            horizon_steps = None
            window_size = None
            neural_cov_cols = None
            if model.is_neural:
                pc = precomputed_sequences or {}
                pc_fold = pc.get(fold_idx)
                if pc_fold:
                    # Reuse pre-computed sliding windows
                    sequence_kwargs['sequence_data'] = pc_fold['seq_X']
                    sequence_kwargs['channel_names'] = pc_fold.get('channel_names', [])
                    y_train = pc_fold['seq_y']
                    X_train = X_train[-len(y_train):]
                    if sample_weights is not None:
                        sample_weights = sample_weights[-len(y_train):]
                    window_size = pc_fold.get('window_size')
                    neural_cov_cols = pc_fold.get('neural_cov_cols', [])
                    horizon_steps = pc_fold.get('horizon_steps')
                    logger.debug(
                        f'Using precomputed sliding windows for {model.name}: '
                        f'{pc_fold["seq_X"].shape}'
                    )
                else:
                    try:
                        from ml_forecast_lab.features import create_sliding_windows
                        target_col = 'target'
                        engineered = {
                            'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
                            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_holiday',
                        }
                        engineered.update(c for c in df_train.columns if c.startswith('y_lag_'))
                        neural_cov_cols = [c for c in df_train.columns if c not in engineered and c != target_col]

                        # Use DENSE horizons matching the production training and
                        # holdout-chart paths so the leaderboard, holdout chart,
                        # and live forecasts all evaluate the SAME model
                        # architecture.
                        future_periods = int(self.experiment_cfg.get('future_periods', 48))
                        horizon_steps = list(range(1, future_periods + 1))

                        if target_col in df_train.columns:
                            # Use original df slice (with DatetimeIndex) for temporal features
                            df_train_raw = df.iloc[train_idx]
                            # Match holdout/production path: cap window at 48
                            # (24h at 30-min). Larger windows hurt small-fold CV
                            # by reducing effective sample size.
                            window_size = min(48, len(df_train_raw) // 3)
                            if window_size >= 12:
                                # v2.41.0 (audit F8): pass the same
                                # future-known features production
                                # training uses, so the leaderboard
                                # ranks the architecture that actually
                                # gets deployed. Before this, CV
                                # trained past-only windows while
                                # _retrain_and_cache trained extended
                                # ones — model selection was made on a
                                # different input pipeline than the
                                # champion runs in production.
                                seq_X, seq_y, channel_names = create_sliding_windows(
                                    df_train_raw, target_col, window_size=window_size,
                                    covariate_cols=neural_cov_cols if neural_cov_cols else None,
                                    add_temporal=True,
                                    horizon_steps=horizon_steps,
                                    future_features_df=future_features_df,
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
                                    f'dense horizons 1..{future_periods}: {channel_names}'
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
            except TrainingCancelled:
                # Not a model failure — propagate so the orchestrator
                # stops the whole run instead of moving to the next fold.
                raise
            except Exception as e:
                logger.error(
                    f'Model training failed for model={model.name} fold={fold_idx + 1}/{len(fold_indices)}: {e}',
                    exc_info=True,
                )
                model_result.fold_metrics.append({})
                model_result.train_times.append(np.nan)
                model_result.inference_times.append(np.nan)
                continue

            train_time = time.time() - train_start

            # --- Train predictions for overfitting diagnostics ---
            # Reduce 2D neural outputs to h=1 to match the test path so train
            # and test metrics are computed on the same horizon.
            y_pred_train = None
            y_train_metric = y_train  # 1D for tree, will be reduced to h=1 for neural
            try:
                if 'sequence_data' in sequence_kwargs and model.is_neural and window_size:
                    yp = model.predict_sequence(sequence_kwargs['sequence_data'])
                    if yp.ndim == 2:
                        y_pred_train = yp[:, 0]
                    else:
                        y_pred_train = yp
                    if y_train.ndim == 2:
                        y_train_metric = y_train[:, 0]
                else:
                    y_pred_train = model.predict(X_train)
            except Exception:
                pass  # Never break benchmark for diagnostics

            # Predict — use predict_sequence for neural models
            inference_start = time.time()
            try:
                if 'sequence_data' in sequence_kwargs and model.is_neural and window_size:
                    try:
                        from ml_forecast_lab.features import create_sliding_windows
                        # Bridge fold boundary: prepend the rows
                        # IMMEDIATELY preceding the test fold for
                        # context. Using train_idx's tail here (as
                        # pre-v2.41.0) skipped the embargo rows, so
                        # every bridged window spanned a
                        # cv_embargo_periods-sized time discontinuity
                        # (audit F14). The embargo only has to keep
                        # those rows out of TRAINING; test-input
                        # context may legitimately include them.
                        test_start = int(test_idx[0])
                        n_bridge = min(window_size, test_start)
                        bridge_idx = np.concatenate([
                            np.arange(test_start - n_bridge, test_start),
                            test_idx,
                        ])
                        df_combined_test = df.iloc[bridge_idx]

                        if future_features_df is not None and horizon_steps:
                            # Extended-window parity (audit F8): the
                            # model was trained on windows of
                            # window_size + future_periods steps, so
                            # test windows must have the same shape.
                            # Dense horizons shrink coverage by
                            # (future_periods − 1) rows at the fold
                            # tail; predictions align with the HEAD of
                            # the test fold (window i's h=1 label is
                            # test row i), so y_test / timestamps are
                            # head-trimmed — a tail trim here would
                            # shift every pair by future_periods − 1.
                            seq_X_test, _, _ = create_sliding_windows(
                                df_combined_test, 'target', window_size=window_size,
                                covariate_cols=neural_cov_cols if neural_cov_cols else None,
                                add_temporal=True,
                                horizon_steps=horizon_steps,
                                future_features_df=future_features_df,
                            )
                            y_pred_full = model.predict_sequence(seq_X_test)
                            if y_pred_full.ndim == 2:
                                y_pred = y_pred_full[:, 0]
                            else:
                                y_pred = y_pred_full
                            if len(y_pred) < len(y_test):
                                y_test = y_test[:len(y_pred)]
                                test_timestamps = test_timestamps[:len(y_pred)]
                        else:
                            # Use horizon_steps=[1] for the test windows so we get
                            # ONE window per test row (full coverage). The model
                            # was trained with dense horizons so y_pred still has
                            # shape (n, future_periods); we take the h=1 column
                            # for ranking. This matches the holdout-chart path.
                            seq_X_test, _, _ = create_sliding_windows(
                                df_combined_test, 'target', window_size=window_size,
                                covariate_cols=neural_cov_cols if neural_cov_cols else None,
                                add_temporal=True,
                                horizon_steps=[1],
                            )
                            y_pred_full = model.predict_sequence(seq_X_test)
                            # Reduce to h=1 (1D) so the metric path treats this
                            # like every other model — same shape as tree models.
                            if y_pred_full.ndim == 2:
                                y_pred = y_pred_full[:, 0]
                            else:
                                y_pred = y_pred_full
                            # y_test stays as the original 1D test fold values.
                            # Make sure shapes match (drop any leading rows the
                            # window builder couldn't fit, which shouldn't happen
                            # with horizon_steps=[1] but be defensive).
                            if len(y_pred) != len(y_test):
                                y_test = y_test[-len(y_pred):]
                    except Exception:
                        y_pred = model.predict(X_test)
                else:
                    y_pred = model.predict(X_test)
            except Exception as e:
                logger.error(
                    f'Model prediction failed for model={model.name} fold={fold_idx + 1}/{len(fold_indices)}: {e}',
                    exc_info=True,
                )
                model_result.fold_metrics.append({})
                model_result.train_times.append(train_time)
                model_result.inference_times.append(np.nan)
                continue

            inference_time = time.time() - inference_start

            # Compute metrics on h=1 (next-step) predictions only.
            # For tree models, y_pred / y_test are already 1D.
            # For neural models, the test path above reduced y_pred to its
            # h=1 column, and we use the original 1D y_test from the fold.
            # This guarantees tree and neural models are scored on the
            # SAME prediction horizon, with the SAME number of samples.
            metrics_to_compute = list(set(self.metrics + [self.production_metric]))
            # Defensively flatten any leftover singleton 2D arrays
            yt_metric = y_test.ravel() if y_test.ndim == 2 and y_test.shape[1] == 1 else y_test
            yp_metric = y_pred.ravel() if y_pred.ndim == 2 and y_pred.shape[1] == 1 else y_pred
            yt_train_metric = y_train if y_train.ndim == 1 else y_train[:, 0]

            # Invert log-transform before scoring so leaderboard numbers are
            # in the sensor's own units, not log(y+1). The model continues to
            # train in log space (that's what the data here is in); only the
            # comparison y_test vs y_pred is moved back to the user's space.
            if self.experiment_cfg.get('log_transform'):
                yt_metric = np.expm1(yt_metric)
                yp_metric = np.expm1(yp_metric)
                yt_train_metric = np.expm1(yt_train_metric)

            interval_minutes = int(self.experiment_cfg.get('interval_minutes', 30))
            season = max(1, 1440 // max(interval_minutes, 1))

            fold_metrics = self.metric_registry.compute_all(
                metrics_to_compute, yt_metric, yp_metric,
                y_train=yt_train_metric, season=season,
            )

            # --- Train metrics for overfitting table ---
            # Computed on h=1 only, matching the test path (y_pred_train and
            # y_train_metric were already reduced to h=1 above for neural models).
            fold_train_m = {}
            if y_pred_train is not None:
                try:
                    if self.experiment_cfg.get('log_transform'):
                        y_train_for_metric = np.expm1(y_train_metric)
                        y_pred_train_for_metric = np.expm1(y_pred_train)
                    else:
                        y_train_for_metric = y_train_metric
                        y_pred_train_for_metric = y_pred_train
                    fold_train_m = self.metric_registry.compute_all(
                        metrics_to_compute,
                        y_train_for_metric, y_pred_train_for_metric,
                        y_train=y_train_for_metric, season=season,
                    )
                except Exception:
                    pass

            # --- Daily-cumulative metrics ---
            # Group test predictions and actuals by date, sum each day, then
            # compute MAE/RMSE/MASE on the daily totals. Rewards models that
            # capture daily totals well even when their per-interval
            # predictions are noisy (zero-mean noise cancels out in the sum).
            # Use case: hot-water / energy demand where the daily total is
            # what matters more than 30-minute precision.
            daily_fold_m: dict = {}
            try:
                if len(test_timestamps) > 0 and len(train_timestamps) > 0:
                    # Align timestamp slices to the metric arrays. The neural
                    # bridge can leave yt_metric / yt_train_metric slightly
                    # shorter than the raw fold (window_size rows are bridge
                    # context, not part of the test fold itself).
                    test_dt = pd.DatetimeIndex(test_timestamps[-len(yt_metric):])
                    train_dt = pd.DatetimeIndex(train_timestamps[-len(yt_train_metric):])

                    daily_test = pd.DataFrame(
                        {'y_test': yt_metric, 'y_pred': yp_metric},
                        index=test_dt,
                    ).groupby(lambda ts: ts.date()).sum()

                    daily_train = pd.DataFrame(
                        {'y_train': yt_train_metric},
                        index=train_dt,
                    ).groupby(lambda ts: ts.date()).sum()

                    if len(daily_test) >= 2 and len(daily_train) >= 2:
                        # Daily totals have season=1 (compared day-over-day,
                        # not interval-over-interval).
                        daily_fold_m = self.metric_registry.compute_all(
                            metrics_to_compute,
                            daily_test['y_test'].values,
                            daily_test['y_pred'].values,
                            y_train=daily_train['y_train'].values,
                            season=1,
                        )
                    else:
                        # Fold legitimately too short to compute daily totals
                        # (e.g. test span covers <2 distinct dates on a small
                        # walk-forward step). Mark with a sentinel so the
                        # ranking layer can distinguish 'errored on this fold'
                        # (empty {}) from 'metric not computable here' — the
                        # model still ranked normally on every fold where
                        # daily totals were computable.
                        daily_fold_m = {'__skipped__': True}
            except Exception as e:
                logger.debug(
                    f'Daily metric computation failed for fold {fold_idx}: {e}'
                )

            model_result.fold_metrics.append(fold_metrics)
            model_result.fold_train_metrics.append(fold_train_m)
            model_result.daily_fold_metrics.append(daily_fold_m)
            model_result.train_times.append(train_time)
            model_result.inference_times.append(inference_time)
            model_result.fold_predictions.append(np.asarray(y_pred).copy())
            model_result.fold_actuals.append(np.asarray(y_test).copy())
            model_result.fold_train_targets.append(np.asarray(y_train_metric).copy())

            # Capture loss curves (last fold overwrites previous)
            if hasattr(model, '_training_history') and model._training_history:
                hist = model._training_history
                tl = hist.get('train_loss', [])
                vl = hist.get('val_loss', [])
                if tl and vl and tl != vl:
                    model_result.training_history = {
                        'train_loss': [float(v) for v in tl],
                        'val_loss': [float(v) for v in vl],
                    }

            fold_metric_val = fold_metrics.get(self.production_metric, np.nan)
            logger.info(
                f'  [fold {fold_idx + 1}/{n_folds}] {model.name} done: '
                f'{self.production_metric}={fold_metric_val:.4f} '
                f'(train={train_time:.1f}s, infer={inference_time:.2f}s, '
                f'total={time.time() - fold_start_time:.1f}s)'
            )

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

        # Aggregate daily-cumulative metrics across folds
        if model_result.daily_fold_metrics:
            metrics_to_compute = list(set(self.metrics + [self.production_metric]))
            for metric_name in metrics_to_compute:
                values = [
                    fm.get(metric_name, np.nan)
                    for fm in model_result.daily_fold_metrics
                    if fm
                ]
                values = [v for v in values if not np.isnan(v)]
                if values:
                    model_result.daily_metrics[metric_name] = float(np.nanmean(values))

        # Aggregate train metrics across folds
        if model_result.fold_train_metrics:
            for metric_name in list(set(self.metrics + [self.production_metric])):
                values = [
                    fm.get(metric_name, np.nan)
                    for fm in model_result.fold_train_metrics
                    if fm
                ]
                if values:
                    model_result.train_metrics[metric_name] = float(np.nanmean(values))

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

        n_models = len(models)
        for model_idx, (model_name, model) in enumerate(models.items(), start=1):
            logger.info(f'[model {model_idx}/{n_models}] Running {model_name}')
            model_start = time.time()
            try:
                model_result = self.run_single_model(df, model, fold_indices)
                result.model_results[model_name] = model_result
                logger.info(
                    f'[model {model_idx}/{n_models}] {model_name} finished '
                    f'in {time.time() - model_start:.1f}s'
                )
            except Exception as e:
                logger.error(f'Benchmark failed for model {model_name}: {e}', exc_info=True)
                result.model_results[model_name] = ModelResult(model_name=model_name)

        # Composite mean-rank across folds (Demšar-style averaging step;
        # see docs/RANKING_NOTES.md for why the full Demšar test does
        # not apply to CV folds of one series). Computed twice: once on
        # per-interval (h=1) metrics for the primary leaderboard, once
        # on daily-cumulative metrics for the secondary leaderboard.
        # Bootstrap CIs over fold resamples express the within-experiment
        # uncertainty.
        (
            interval_mean_ranks, interval_ranks,
            interval_rank_cis, dnc_interval,
        ) = self._compute_composite_ranks(
            result.model_results, metric_source='fold_metrics',
        )
        (
            daily_mean_ranks, daily_ranks,
            daily_rank_cis, dnc_daily,
        ) = self._compute_composite_ranks(
            result.model_results, metric_source='daily_fold_metrics',
        )

        # Store both mean ranks (and CIs) on ModelResult.metrics for UI
        # access. DNC models get inf so legacy callers that read
        # metrics['mean_rank'] still see them as unranked.
        for name in result.model_results:
            result.model_results[name].metrics['mean_rank'] = (
                interval_mean_ranks.get(name, float('inf'))
            )
            result.model_results[name].metrics['mean_rank_daily'] = (
                daily_mean_ranks.get(name, float('inf'))
            )
            ci = interval_rank_cis.get(name)
            if ci is not None:
                result.model_results[name].metrics['mean_rank_low'] = ci[0]
                result.model_results[name].metrics['mean_rank_high'] = ci[1]
            ci_d = daily_rank_cis.get(name)
            if ci_d is not None:
                result.model_results[name].metrics['mean_rank_daily_low'] = ci_d[0]
                result.model_results[name].metrics['mean_rank_daily_high'] = ci_d[1]

        # Primary ranking still drives Promote / Tuning / sensor publishing.
        # Interval DNCs go in did_not_complete (main leaderboard); daily-only
        # DNCs go in did_not_complete_daily (surfaced under the Daily table)
        # so a per-interval-ranked model isn't shown as not-completed.
        # Pre-v2.39.3 the daily list was discarded entirely.
        result.rankings = interval_ranks
        result.daily_rankings = daily_ranks
        result.did_not_complete = dnc_interval
        result.did_not_complete_daily = sorted(set(dnc_daily) - set(dnc_interval))
        sorted_models = sorted(interval_mean_ranks.items(), key=lambda x: x[1])
        mean_ranks = interval_mean_ranks  # for downstream logging

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
            daily_rank_str = (
                f', daily_rank=#{daily_ranks[model_name]}'
                if daily_ranks.get(model_name) else ''
            )
            logger.info(
                f'  #{result.rankings[model_name]} {model_name}: '
                f'{self.production_metric}={mr.metrics.get(self.production_metric, np.nan):.4f}, '
                f'mean_rank={mean_ranks[model_name]:.2f}'
                f'{daily_rank_str}'
            )

        return result

    def _compute_composite_ranks(
        self,
        model_results: Dict[str, ModelResult],
        metric_source: str,
        bootstrap_ci: bool = True,
        bootstrap_iters: int = 1000,
        bootstrap_seed: int = 0,
    ) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, Tuple[float, float]], List[str]]:
        """
        Compute a Demšar-style composite mean rank across CV folds, plus
        bootstrap uncertainty bands and a list of models excluded for
        incomplete folds.

        Note on the "Demšar" attribution: Demšar (2006) describes a full
        Friedman + post-hoc procedure intended for ranking K algorithms
        across N **independent** datasets. Here we apply only the
        mean-rank averaging step across CV folds of one series — folds
        are not independent, so the Friedman / Nemenyi test does not
        apply. We surface bootstrap CIs over fold resamples to express
        the within-experiment uncertainty honestly. See
        ``docs/RANKING_NOTES.md`` for the full caveat.

        Within each fold, models are ranked independently by each
        ranking metric (typically MAE, RMSE, MASE). Per-metric ranks are
        averaged to give a composite fold rank, then averaged across
        folds to give the model's mean composite rank.

        Models that did not complete every fold (a fold raised inside
        feature build / fit / predict, or the model errored at the outer
        level) are returned in the ``did_not_complete`` list and
        excluded from the ranked pool, so they don't artificially take
        last-place "wins" off the models that did complete.

        Parameters
        ----------
        model_results : dict[str, ModelResult]
            All model results to rank.
        metric_source : str
            Attribute name on `ModelResult` to read per-fold metrics
            from. Use ``'fold_metrics'`` for per-interval ranking and
            ``'daily_fold_metrics'`` for daily-cumulative ranking.
        bootstrap_ci : bool
            Whether to compute bootstrap CIs on mean ranks. Skipped when
            there are fewer than 2 folds (a CI off a single fold is
            meaningless).
        bootstrap_iters : int
            Number of bootstrap resamples. Default 1000.
        bootstrap_seed : int
            RNG seed for reproducibility.

        Returns
        -------
        mean_ranks : dict[str, float]
            Model name → mean composite rank (lower = better). Only
            includes rankable models (those that completed every fold).
        integer_ranks : dict[str, int]
            Model name → 1-indexed final rank derived from
            ``mean_ranks``.
        rank_cis : dict[str, tuple[float, float]]
            Model name → (low, high) 95% bootstrap CI on mean rank.
            Empty for any model when ``bootstrap_ci`` is False or there
            are <2 valid folds.
        did_not_complete : list[str]
            Names of models excluded from the rankable pool because
            they failed at least one fold (or all folds).
        """
        _higher_better = {'r_squared', 'coverage'}
        ranking_metrics = [m for m in self.metrics if m not in _higher_better]
        # Always let the configured selection metric steer the composite, even
        # if the user's `metrics` list omits it — otherwise a peak-aware
        # `production_metric` (e.g. peak_weighted_mae) would be computed but
        # never influence which model wins. It is also weighted below.
        prod_metric = self.production_metric
        if (prod_metric and prod_metric not in _higher_better
                and prod_metric not in ranking_metrics):
            ranking_metrics.append(prod_metric)
        if not ranking_metrics:
            ranking_metrics = [self.production_metric]

        # The selection metric carries a heavier vote in the composite so a
        # deliberately-chosen metric is decisive rather than diluted 1-of-N
        # (the default list has three L1-family metrics — mae/mase/seasonal_mase
        # — that co-rank, which would otherwise drown out a single peak-aware
        # metric). Weight ~1.5x(other metrics) gives the selection metric a
        # clear majority once there are >=3 ranking metrics (the common case),
        # and collapses to a no-op for 1-2 metrics so the equal-weight
        # behaviour — and the existing rank tests — are preserved.
        _prod_weight = max(1, (len(ranking_metrics) - 1) * 3 // 2)
        _metric_weight = {
            m: (_prod_weight if m == prod_metric else 1)
            for m in ranking_metrics
        }

        model_names = list(model_results.keys())
        if not model_names:
            return {}, {}, {}, []

        n_folds = max(
            len(getattr(mr, metric_source, [])) for mr in model_results.values()
        )

        # Identify did-not-complete models up front so they're excluded
        # from BOTH the ranking and from the per-fold rank sort below.
        # A model is rankable for this metric_source if every fold's
        # entry is either non-empty (real metrics) or the ``__skipped__``
        # sentinel emitted by run_single_model when the metric isn't
        # computable on that fold (e.g. daily totals on a fold spanning
        # <2 distinct dates) — that's a metric-level skip, not a model
        # failure. Truly empty ``{}`` means the fold's fit/predict
        # raised and counts as DNC.
        def _is_failure(fm: dict) -> bool:
            return not fm  # empty dict counts as failure

        rankable_names: List[str] = []
        did_not_complete: List[str] = []
        for name in model_names:
            fm_list = getattr(model_results[name], metric_source, [])
            complete = (
                len(fm_list) == n_folds
                and not any(_is_failure(fm) for fm in fm_list)
            )
            if complete:
                rankable_names.append(name)
            else:
                did_not_complete.append(name)

        if not rankable_names:
            return {}, {}, {}, did_not_complete

        # Compute per-fold composite ranks restricted to rankable models.
        # This is the array we both average (for mean_rank) and bootstrap
        # (for the CI). Shape: {model: [composite_rank_fold_0, ...]}.
        # We track per-fold rank vectors aligned by index so the bootstrap
        # below can do a PAIRED resample — drawing a fold-id matrix once
        # per iteration and applying the SAME ids to every model. Without
        # pairing, fold-to-fold correlation (hard folds hurt everyone) is
        # destroyed and the CIs overstate ties.
        fold_ranks: Dict[str, List[float]] = {name: [] for name in rankable_names}
        # Maps fold_idx → set of model names that had a rank computed for
        # that fold. Used to keep the paired bootstrap matrix aligned.
        ranked_fold_ids: List[int] = []
        for fold_idx in range(n_folds):
            # Skip folds where EVERY rankable model is on the sentinel
            # (metric not computable for any) — they contribute no rank.
            all_skipped = all(
                getattr(model_results[name], metric_source, [])[fold_idx].get(
                    '__skipped__'
                )
                for name in rankable_names
            )
            if all_skipped:
                continue
            per_metric_ranks: Dict[str, Dict[str, int]] = {
                name: {} for name in rankable_names
            }
            for metric_name in ranking_metrics:
                higher_is_better = metric_name in _higher_better
                fold_values: Dict[str, float] = {}
                for name in rankable_names:
                    fm_list = getattr(model_results[name], metric_source, [])
                    fm = fm_list[fold_idx]
                    if fm.get('__skipped__'):
                        # This model legitimately has no metric here;
                        # exclude from this metric's rank-sort rather
                        # than punishing it with last place.
                        continue
                    worst = -np.inf if higher_is_better else np.inf
                    val = fm.get(metric_name, worst)
                    # A metric that computed to NaN (e.g. seasonal_mase on a
                    # fold whose window is entirely flat) is unrankable: every
                    # comparison against NaN is False, so `sorted` below would
                    # silently leave dict insertion order intact and hand out
                    # ranks that encode registry order rather than accuracy.
                    # Map it to the same worst-case sentinel a missing metric
                    # gets, so the all-sentinel guard can drop the metric for
                    # this fold. Latent while every metric was one equal vote
                    # of N; not latent now the selection metric carries the
                    # majority (see the weighting above).
                    if val is None or val != val:
                        val = worst
                    fold_values[name] = val

                if not fold_values or all(
                    np.isinf(v) for v in fold_values.values()
                ):
                    continue

                sorted_fold = sorted(
                    fold_values.items(),
                    key=lambda x: x[1],
                    reverse=higher_is_better,
                )
                for r, (name, _) in enumerate(sorted_fold):
                    per_metric_ranks[name][metric_name] = r + 1

            fold_recorded = False
            for name in rankable_names:
                rd = per_metric_ranks[name]
                if rd:
                    # Weighted mean of this fold's per-metric ranks, giving the
                    # selection metric the heavier vote defined above.
                    num = sum(rank * _metric_weight[m] for m, rank in rd.items())
                    den = sum(_metric_weight[m] for m in rd)
                    fold_ranks[name].append(float(num) / float(den))
                    fold_recorded = True
                else:
                    # Pad with NaN so the per-fold arrays stay aligned
                    # across models for the paired bootstrap. The mean
                    # / bootstrap layer will skip NaN entries per model.
                    fold_ranks[name].append(float('nan'))
            if fold_recorded:
                ranked_fold_ids.append(fold_idx)

        # Mean composite rank per model across the folds with valid data.
        # A model that passed the completeness check but ended up with NO
        # ranked folds (every fold was '__skipped__' for it, or every
        # fold's metrics were all-inf) gets demoted to DNC for this
        # metric_source — emitting integer ranks for inf-mean models
        # would just be dict-insertion-order noise.
        mean_ranks: Dict[str, float] = {}
        for name in list(rankable_names):
            ranks = np.asarray(fold_ranks[name], dtype=float)
            valid = ranks[~np.isnan(ranks)]
            if not valid.size:
                rankable_names.remove(name)
                did_not_complete.append(name)
                continue
            mean_ranks[name] = float(np.mean(valid))

        # Paired bootstrap CI: resample fold IDs with replacement ONCE
        # per iteration, then apply the SAME ids to every model. Joint
        # resampling preserves the across-model fold structure
        # (model A's rank in fold k IS comparable to model B's in fold
        # k), which is what the leaderboard's 'tied with #1' chip wants
        # to compare. Pre-v2.39.3 each model drew its own ids in
        # sequence — independent draws inflated marginal CIs and the
        # tie-detection over-reported ties.
        rank_cis: Dict[str, Tuple[float, float]] = {}
        # Number of folds where any model produced a rank (the bootstrap
        # operates on this aligned matrix).
        n_ranked_folds = len(ranked_fold_ids)
        if bootstrap_ci and n_ranked_folds >= 2:
            rng = np.random.default_rng(bootstrap_seed)
            # Single (bootstrap_iters, n_ranked_folds) index matrix —
            # the same fold-id columns reused across every model.
            idx_matrix = rng.integers(
                0, n_ranked_folds,
                size=(bootstrap_iters, n_ranked_folds),
            )
            for name in rankable_names:
                ranks_arr = np.asarray(fold_ranks[name], dtype=float)
                if ranks_arr.size < 2:
                    continue
                resampled = ranks_arr[idx_matrix]  # shape (iters, n_folds)
                # nanmean tolerates per-model NaN entries (skipped folds)
                with np.errstate(invalid='ignore'):
                    resampled_means = np.nanmean(resampled, axis=1)
                resampled_means = resampled_means[~np.isnan(resampled_means)]
                if resampled_means.size == 0:
                    continue
                low = float(np.quantile(resampled_means, 0.025))
                high = float(np.quantile(resampled_means, 0.975))
                rank_cis[name] = (low, high)

        # Final integer ranking over the rankable pool only
        sorted_models = sorted(mean_ranks.items(), key=lambda x: x[1])
        integer_ranks = {name: idx + 1 for idx, (name, _) in enumerate(sorted_models)}
        return mean_ranks, integer_ranks, rank_cis, did_not_complete

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
