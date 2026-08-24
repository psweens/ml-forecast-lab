"""
Evaluation metrics for forecasting model benchmarking.

Provides standard accuracy metrics (MAE, RMSE, MAPE, etc.) and a metric
registry system for flexible metric computation and custom metric registration.
All metrics handle NaN values gracefully and return float results.
"""

import inspect
import logging
import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Minimum number of above-median intervals needed before `peak_weighted_mae`
# will estimate its weighting scale from the active part of a zero-heavy
# series. Below this the sample is too thin to be meaningful and the metric
# falls back to plain MAE.
_PEAK_MIN_ACTIVE = 8


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Mean absolute error. Returns np.nan if all values are NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = np.abs(y_true - y_pred)
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))

    if not np.any(valid_mask):
        return np.nan

    return float(np.mean(errors[valid_mask]))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Root mean square error. Returns np.nan if all values are NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = (y_true - y_pred) ** 2
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))

    if not np.any(valid_mask):
        return np.nan

    return float(np.sqrt(np.mean(errors[valid_mask])))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Skips zero values in the denominator to avoid division by zero.

    Parameters
    ----------
    y_true : np.ndarray
        True values (non-zero).
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Mean absolute percentage error. Returns np.nan if no valid samples.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Valid mask: not NaN and y_true != 0
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred)) & (y_true != 0)

    if not np.any(valid_mask):
        return np.nan

    errors = np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask])
    return float(np.mean(errors) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.

    Uses the symmetric definition: 2 * |y_true - y_pred| / (|y_true| + |y_pred|).

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Symmetric MAPE (0-200 range). Returns np.nan if all values are NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))

    if not np.any(valid_mask):
        return np.nan

    numerator = 2.0 * np.abs(y_true[valid_mask] - y_pred[valid_mask])
    denominator = np.abs(y_true[valid_mask]) + np.abs(y_pred[valid_mask])

    # Avoid division by zero
    nonzero_mask = denominator != 0
    if not np.any(nonzero_mask):
        return np.nan

    errors = numerator[nonzero_mask] / denominator[nonzero_mask]
    return float(np.mean(errors) * 100.0)


def mase(
    y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray
) -> float:
    """
    Calculate Mean Absolute Scaled Error.

    Scales errors by the mean absolute error of a naive forecast (using
    the last training value as the forecast for all test values).

    Parameters
    ----------
    y_true : np.ndarray
        True test values.
    y_pred : np.ndarray
        Predicted test values.
    y_train : np.ndarray
        Training values used to compute the naive forecast scale.

    Returns
    -------
    float
        Mean absolute scaled error. Returns np.nan if denominator is zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))

    if not np.any(valid_mask):
        return np.nan

    # Compute naive forecast scale (MAE of naive 1-step-ahead forecast)
    if len(y_train) < 2:
        return np.nan

    naive_errors = np.abs(np.diff(y_train))
    naive_scale = np.nanmean(naive_errors)

    if np.isnan(naive_scale) or naive_scale == 0:
        return np.nan

    # Compute MAE of actual forecast
    abs_errors = np.abs(y_true[valid_mask] - y_pred[valid_mask])
    return float(np.mean(abs_errors) / naive_scale)


def seasonal_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season: int = 48,
) -> float:
    """
    Seasonal MASE — MAE scaled by the seasonal-naive baseline error.

    The seasonal baseline forecast is ``ŷ_t = y_{t-season}`` (e.g. same time
    yesterday at season=48 with 30-min sampling). For HA sensors with a
    dominant daily cycle this is the meaningful comparison; the 1-step
    naive used by classical MASE under-states the baseline because
    consecutive 30-min values are heavily correlated.

    Falls back to 1-step naive when ``len(y_train) <= season``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(valid_mask):
        return np.nan

    if len(y_train) < 2:
        return np.nan

    if season < 1 or len(y_train) <= season:
        naive_errors = np.abs(np.diff(y_train))
    else:
        naive_errors = np.abs(y_train[season:] - y_train[:-season])

    naive_scale = np.nanmean(naive_errors)
    if np.isnan(naive_scale) or naive_scale == 0:
        return np.nan

    abs_errors = np.abs(y_true[valid_mask] - y_pred[valid_mask])
    return float(np.mean(abs_errors) / naive_scale)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination).

    R² = 1 - (SS_res / SS_tot), where SS_res is residual sum of squares
    and SS_tot is total sum of squares.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        R² score (range typically -inf to 1.0). Returns np.nan if invalid.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))

    if not np.any(valid_mask):
        return np.nan

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    mean_y = np.mean(y_true_valid)
    ss_tot = np.sum((y_true_valid - mean_y) ** 2)
    ss_res = np.sum((y_true_valid - y_pred_valid) ** 2)

    if ss_tot == 0:
        return np.nan

    return float(1.0 - (ss_res / ss_tot))


def pinball_loss(
    y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.5
) -> float:
    """
    Calculate pinball loss for quantile regression.

    Loss = sum((quantile - I(y_true < y_pred)) * (y_true - y_pred))
    where I(·) is the indicator function.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted quantile values.
    quantile : float, optional
        Quantile level (default 0.5 for median). Must be in [0, 1].

    Returns
    -------
    float
        Average pinball loss. Returns np.nan if all values are NaN.

    Raises
    ------
    ValueError
        If quantile is not in [0, 1].
    """
    if not 0 <= quantile <= 1:
        raise ValueError(f'quantile must be in [0, 1], got {quantile}')

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))

    if not np.any(valid_mask):
        return np.nan

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    errors = y_true_valid - y_pred_valid
    loss = np.where(
        errors >= 0, quantile * errors, (quantile - 1) * errors
    )

    return float(np.mean(loss))


def peak_weighted_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error that weights high-actual intervals more heavily.

    Plain MAE / RMSE / MASE reward the model that runs through the *middle* of
    a spiky series: chasing a peak and missing its timing is penalised twice
    (a false alarm where the model reached up, and a miss where the spike
    actually landed), so a flat conditional-mean forecast scores best. That is
    exactly the "it flattens my spikes" failure for bursty loads (hot water,
    EV, appliances).

    This metric weights each interval by how far its *actual* value sits above
    the typical level, so the score is dominated by how well the model tracks
    the peaks rather than the quiet baseline. Direction matches MAE (lower is
    better), so it slots into the existing rank machinery unchanged.

    The weight is ``1 + clip((y_true - median) / spread, 0, 9)``: an interval
    at the median weighs 1, a tall peak weighs up to 10, and the cap stops a
    single freak spike from dominating.

    ``spread`` is normally ``p90 - median``. On a zero-heavy target — a load
    that is off most of the time, which is precisely what this metric exists
    for — the p90 still sits inside the flat baseline (both are 0.0), so the
    scale is taken from the distribution *above* the median instead. Real
    30-minute hot-water and EV loads sit at 92-96% zeros, well past the point
    where the p90 collapses. Without the second anchor this metric silently
    degrades to plain MAE on exactly its intended targets.

    Only a genuinely constant series, or one with too few active intervals to
    estimate a scale from, falls back to plain MAE.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Peak-weighted mean absolute error. ``np.nan`` if all values are NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(valid_mask):
        return np.nan

    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]
    abs_err = np.abs(yt - yp)

    median = float(np.median(yt))
    spread = float(np.quantile(yt, 0.9)) - median
    if spread <= 0:
        # Zero-heavy target: the p90 sits inside the flat baseline. Anchor the
        # scale on the active part rather than collapsing to plain MAE.
        active = yt[yt > median]
        if active.size >= _PEAK_MIN_ACTIVE:
            spread = float(np.quantile(active, 0.9)) - median
    if not np.isfinite(spread) or spread <= 0:
        # Flat or degenerate target — nothing to up-weight, behave like MAE.
        return float(np.mean(abs_err))

    excess = np.clip((yt - median) / spread, 0.0, 9.0)
    weights = 1.0 + excess
    return float(np.sum(weights * abs_err) / np.sum(weights))


def pinball_q90(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Upper-quantile (0.9) pinball loss — under-shooting a value costs 9x more
    than over-shooting it.

    A first-class probabilistic / peak-emphasis score: it rewards a forecast
    that reaches *up* toward the highs rather than splitting the difference.
    Useful as an evaluation column or as the champion-selection metric for
    users who care most about not missing peaks (capacity, peak demand).
    Lower is better. See :func:`pinball_loss` for the underlying definition.
    """
    return pinball_loss(y_true, y_pred, quantile=0.9)


def coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """
    Calculate prediction interval coverage rate.

    Computes the proportion of true values that fall within the
    [lower, upper] prediction interval bounds.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    lower : np.ndarray
        Lower interval bounds.
    upper : np.ndarray
        Upper interval bounds.

    Returns
    -------
    float
        Coverage proportion in [0, 1]. Returns np.nan if all values are NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    valid_mask = ~(
        np.isnan(y_true) | np.isnan(lower) | np.isnan(upper)
    )

    if not np.any(valid_mask):
        return np.nan

    y_true_valid = y_true[valid_mask]
    lower_valid = lower[valid_mask]
    upper_valid = upper[valid_mask]

    in_interval = (y_true_valid >= lower_valid) & (y_true_valid <= upper_valid)
    return float(np.mean(in_interval))


class MetricRegistry:
    """
    Registry for managing evaluation metrics.

    Supports registering standard and custom metrics, computing metrics
    individually or in batch, and managing metric definitions.
    """

    def __init__(self) -> None:
        """Initialise the metric registry with standard metrics."""
        self._metrics: Dict[str, Callable] = {}
        self._register_standard_metrics()

    def _register_standard_metrics(self) -> None:
        """Register all standard metrics."""
        self.register('mae', mae)
        self.register('rmse', rmse)
        self.register('mape', mape)
        self.register('smape', smape)
        self.register('mase', mase)
        self.register('seasonal_mase', seasonal_mase)
        self.register('r_squared', r_squared)
        self.register('pinball_loss', pinball_loss)
        self.register('peak_weighted_mae', peak_weighted_mae)
        self.register('pinball_q90', pinball_q90)
        self.register('coverage', coverage)
        logger.info('Registered 11 standard metrics')

    def register(self, name: str, func: Callable) -> None:
        """
        Register a metric function.

        Parameters
        ----------
        name : str
            Unique metric identifier.
        func : Callable
            Function that computes the metric. Should accept y_true and y_pred
            as numpy arrays and return a float.

        Raises
        ------
        ValueError
            If name is empty or already registered.
        """
        if not name:
            raise ValueError('Metric name cannot be empty')

        if name in self._metrics:
            logger.warning(f'Overwriting existing metric: {name}')

        self._metrics[name] = func
        logger.debug(f'Registered metric: {name}')

    def compute(
        self, name: str, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any
    ) -> float:
        """
        Compute a single metric.

        Parameters
        ----------
        name : str
            Metric name.
        y_true : np.ndarray
            True values.
        y_pred : np.ndarray
            Predicted values (or second required positional arg for some metrics).
        **kwargs : Any
            Additional arguments passed to the metric function.
            For coverage metric: pass lower and upper as keyword arguments.
            For pinball_loss: pass quantile as keyword argument.
            For mase: pass y_train as keyword argument.

        Returns
        -------
        float
            Computed metric value.

        Raises
        ------
        ValueError
            If metric name is not registered.
        """
        if name not in self._metrics:
            raise ValueError(
                f'Metric {name!r} not registered. '
                f'Available: {list(self._metrics.keys())}'
            )

        metric_func = self._metrics[name]
        # Only pass kwargs that the metric function actually accepts
        sig = inspect.signature(metric_func)
        accepted_params = set(sig.parameters.keys())
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k in accepted_params
        }
        return metric_func(y_true, y_pred, **filtered_kwargs)

    def compute_all(
        self,
        names: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Compute multiple metrics in batch.

        Parameters
        ----------
        names : list[str]
            List of metric names to compute.
        y_true : np.ndarray
            True values.
        y_pred : np.ndarray
            Predicted values.
        **kwargs : Any
            Additional arguments passed to metric functions.

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to computed values.

        Raises
        ------
        ValueError
            If any metric name is not registered.
        """
        results = {}
        for name in names:
            try:
                results[name] = self.compute(name, y_true, y_pred, **kwargs)
            except Exception as e:
                logger.warning(f'Failed to compute metric {name!r}: {e}')
                results[name] = np.nan

        return results

    def register_custom(self, name: str, expression_str: str) -> None:
        """
        Register a custom metric from a Python expression string.

        The expression has access to:
        - y_true: true values as numpy array
        - y_pred: predicted values as numpy array
        - np: numpy module

        Example: "np.mean(np.abs(y_true - y_pred) * np.where(y_true > 0, 2.0, 1.0))"

        Parameters
        ----------
        name : str
            Unique metric identifier.
        expression_str : str
            Python expression string for computing the metric.

        Raises
        ------
        ValueError
            If expression is invalid or contains disallowed operations.
        SyntaxError
            If expression has invalid Python syntax.
        """
        if not name:
            raise ValueError('Metric name cannot be empty')

        # Basic security check: prevent imports and dangerous operations
        forbidden_patterns = [
            r'\bimport\b', r'\bfrom\b', r'\b__\w+__\b',
            r'\bopen\b', r'\bexec\b', r'\beval\b', r'\bcompile\b'
        ]

        for pattern in forbidden_patterns:
            if re.search(pattern, expression_str):
                raise ValueError(
                    f'Expression contains forbidden operation: {pattern}'
                )

        def custom_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            """Dynamically created custom metric function."""
            try:
                from asteval import Interpreter
                aeval = Interpreter(usersyms={
                    'y_true': y_true, 'y_pred': y_pred, 'np': np,
                    'float': float, 'int': int, 'abs': abs, 'sum': sum,
                    'min': min, 'max': max, 'len': len,
                })
                result = aeval(expression_str)
                if aeval.error:
                    raise RuntimeError('; '.join(str(e) for e in aeval.error))
                return float(result)
            except Exception as e:
                logger.error(
                    f'Error evaluating custom metric {name!r}: {e}'
                )
                raise

        self.register(name, custom_metric)
        logger.info(f'Registered custom metric: {name}')

    def list_available(self) -> List[str]:
        """
        List all available metric names.

        Returns
        -------
        list[str]
            Sorted list of registered metric names.
        """
        return sorted(self._metrics.keys())


_global_registry: Optional[MetricRegistry] = None


def get_metric_registry() -> MetricRegistry:
    """
    Get the global metric registry instance.

    Returns
    -------
    MetricRegistry
        Global singleton instance with standard metrics pre-registered.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = MetricRegistry()
    return _global_registry
