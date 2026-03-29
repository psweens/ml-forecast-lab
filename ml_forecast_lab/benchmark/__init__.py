"""
Benchmarking engine for ML Forecast Lab.

Provides comprehensive benchmarking infrastructure including:
- Evaluation metrics with flexible registry
- Cross-validated benchmark runner
- Statistical model comparison tests

Key classes:
- MetricRegistry: Register and compute evaluation metrics
- BenchmarkRunner: Orchestrate cross-validated model evaluation
- BenchmarkResult: Results container with rankings
- ModelResult: Per-model results and timing information

Functions:
- get_metric_registry(): Get global metric registry instance
- diebold_mariano_test(): Statistical test for model comparison
- model_confidence_set(): Identify statistically equivalent models
- paired_comparison_table(): Pairwise model comparison matrix
"""

from .comparison import (
    diebold_mariano_test,
    model_confidence_set,
    paired_comparison_table,
)
from .metrics import (
    MetricRegistry,
    coverage,
    get_metric_registry,
    mae,
    mape,
    mase,
    pinball_loss,
    r_squared,
    rmse,
    smape,
)
from .runner import (
    BenchmarkResult,
    BenchmarkRunner,
    ModelResult,
)

__all__ = [
    # Metrics module
    'mae',
    'rmse',
    'mape',
    'smape',
    'mase',
    'r_squared',
    'pinball_loss',
    'coverage',
    'MetricRegistry',
    'get_metric_registry',
    # Runner module
    'BenchmarkRunner',
    'BenchmarkResult',
    'ModelResult',
    # Comparison module
    'diebold_mariano_test',
    'model_confidence_set',
    'paired_comparison_table',
]
