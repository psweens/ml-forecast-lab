"""
FastAPI web application for ML Forecast Lab.

Provides dashboard and API endpoints for monitoring and managing forecasting
experiments, model benchmarking, and production deployment.
"""

import json
import logging
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ml_forecast_lab import __version__ as APP_VERSION

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

logger = logging.getLogger(__name__)


# Data models
class MetricValue(BaseModel):
    """Single metric value with mean and std across folds."""

    mean: float
    std: float


class ModelResult(BaseModel):
    """Results for a single model in a benchmark."""

    name: str
    mae: MetricValue
    rmse: MetricValue
    mase: MetricValue
    train_time_seconds: float
    rank: int
    mean_rank: float = 0.0
    is_production: bool = False
    fold_results: Optional[List[Dict[str, float]]] = None
    train_mae: Optional[MetricValue] = None
    train_rmse: Optional[MetricValue] = None
    training_history: Optional[Dict[str, List[float]]] = None
    # Daily-cumulative metrics. Same MAE/RMSE/MASE methodology, but
    # computed on per-day totals (each day's predictions summed, each
    # day's actuals summed, then compared). Better for use cases where
    # daily totals matter more than 30-minute precision (e.g. hot-water
    # or energy demand). The Daily Rank is computed via the same Demšar
    # composite as the primary rank but using the daily metrics; it is
    # purely informational and does NOT drive Promote/Tuning workflows.
    daily_mae: Optional[MetricValue] = None
    daily_rmse: Optional[MetricValue] = None
    daily_mase: Optional[MetricValue] = None
    daily_rank: Optional[int] = None
    daily_mean_rank: Optional[float] = None


class BenchmarkResult(BaseModel):
    """Complete benchmark run results for an experiment."""

    experiment_name: str
    timestamp: str
    status: str  # 'running', 'completed', 'failed'
    models: List[ModelResult]
    best_model_name: Optional[str] = None
    error_message: Optional[str] = None


class ExperimentStatus(BaseModel):
    """Status of an experiment."""

    name: str
    target_entity: str
    mode: str  # 'lab' or 'production'
    best_model: Optional[str] = None
    selected_model: Optional[str] = None  # User's chosen model (defaults to best)
    last_benchmark_timestamp: Optional[str] = None
    last_benchmark_status: str = "pending"
    last_error: Optional[str] = None  # Human-readable error from last failed cycle
    next_forecast_in_seconds: Optional[int] = None
    next_retrain_in_seconds: Optional[int] = None
    next_update_in_seconds: Optional[int] = None  # Legacy alias for forecast
    publish_entity: Optional[str] = None  # e.g. "sensor.mlfl_solar_forecast"


class ForecastPoint(BaseModel):
    """Single forecast point."""

    timestamp: str
    actual: Optional[float] = None
    predicted_mean: Optional[float] = None
    predicted_lower: Optional[float] = None
    predicted_upper: Optional[float] = None


class ForecastData(BaseModel):
    """Forecast data for charting."""

    experiment_name: str
    horizon_minutes: int
    points: List[ForecastPoint]
    model_name: Optional[str] = None


class ModelPrediction(BaseModel):
    """Predictions from a single model on holdout data.

    Note: trace colours are no longer set here. The frontend (Plotly)
    auto-assigns colours from a colorway in trace order.
    """

    model_name: str
    timestamps: List[str]
    actuals: List[Optional[float]]
    predictions: List[Optional[float]]


class LabForecastData(BaseModel):
    """Multi-model prediction data for lab mode visualisation."""

    experiment_name: str
    holdout_start: str
    holdout_end: str
    model_predictions: List[ModelPrediction]


class CovariateAnalysisCellResult(BaseModel):
    """Result for one model × one covariate configuration."""
    mae: float
    rmse: float
    mase: float = float('nan')
    change_pct: Optional[float] = None      # MAE % change vs baseline
    rmse_change_pct: Optional[float] = None  # RMSE % change vs baseline
    mase_change_pct: Optional[float] = None  # MASE % change vs baseline


class CovariateAnalysisResult(BaseModel):
    """Full covariate analysis results."""
    experiment_name: str
    timestamp: str
    status: str  # 'running', 'completed', 'failed'
    baseline_label: str  # "All covariates"
    covariate_labels: List[str]  # ["No covariates", "Without charge", ...]
    model_names: List[str]
    # results[covariate_label][model_name] = CovariateAnalysisCellResult
    results: Dict[str, Dict[str, CovariateAnalysisCellResult]]
    recommendations: List[Dict[str, str]]  # [{"icon": "✓", "text": "...", "color": "green"}, ...]
    total_runs: int = 0
    completed_runs: int = 0


class TuningTrialResult(BaseModel):
    """Result for one hyperparameter tuning trial."""
    trial_id: int
    params: Dict[str, Any]
    mae: float
    rmse: float
    mase: float
    duration_seconds: float = 0.0
    status: str = "completed"  # completed, pruned, failed


class TuningResult(BaseModel):
    """Hyperparameter tuning results for one model."""
    experiment_name: str
    model_name: str
    timestamp: str
    status: str  # running, completed, failed
    search_strategy: str = "tpe"
    n_trials: int = 30
    completed_trials: int = 0
    trials: List[TuningTrialResult] = []
    best_trial_id: Optional[int] = None
    best_params: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    error_message: Optional[str] = None
    # Holdout comparison: default vs tuned predictions
    holdout_timestamps: Optional[List[str]] = None
    holdout_actuals: Optional[List[Optional[float]]] = None
    holdout_default: Optional[List[Optional[float]]] = None  # predictions with default params
    holdout_tuned: Optional[List[Optional[float]]] = None    # predictions with best tuned params
    default_mae: Optional[float] = None


class FeatureImportanceData(BaseModel):
    """Feature importance from a trained model."""

    model_name: str
    features: List[Dict[str, Any]]  # [{"name": "hour_of_day", "importance": 0.25}, ...]


class EnsembleMethodResult(BaseModel):
    """Result for one ensemble strategy."""

    strategy: str  # "simple_average", "weighted_average", "stacking"
    display_name: str
    member_models: List[str]
    mae: float
    rmse: float
    mase: float
    weights: Optional[Dict[str, float]] = None


class EnsembleResultData(BaseModel):
    """Complete ensemble results for an experiment."""

    experiment_name: str
    timestamp: str
    status: str  # "running", "completed", "failed"
    methods: List[EnsembleMethodResult]
    best_strategy: Optional[str] = None
    improvement_pct: Optional[float] = None
    best_individual_model: Optional[str] = None
    best_individual_metric: Optional[float] = None
    best_individual_mae: Optional[float] = None
    best_individual_rmse: Optional[float] = None
    best_individual_mase: Optional[float] = None


class ModelInfo(BaseModel):
    """Information about an available model backend."""

    name: str
    display_name: str
    description: str


class HealthStatus(BaseModel):
    """System health status."""

    status: str
    service: str
    version: str
    timestamp: str
    experiments_total: int
    experiments_lab: int
    experiments_production: int


# In-memory state management
class AppState:
    """Simple in-memory state for benchmark results and experiment status."""

    def __init__(self):
        """Initialise state."""
        self.benchmark_results: Dict[str, BenchmarkResult] = {}
        self.experiment_statuses: Dict[str, ExperimentStatus] = {}
        self.forecast_data: Dict[str, ForecastData] = {}
        self.lab_forecast_data: Dict[str, LabForecastData] = {}
        self.feature_importances: Dict[str, List[FeatureImportanceData]] = {}
        self.covariate_analysis_results: Dict[str, CovariateAnalysisResult] = {}
        self.covariate_analysis_callback = None  # Set by main app for triggering
        self.ensemble_results: Dict[str, EnsembleResultData] = {}
        self.ensemble_callback = None  # Set by main app for triggering
        self.benchmark_callback = None  # Set by main app for triggering
        self.tuning_results: Dict[str, TuningResult] = {}
        self.tuning_callback = None  # Set by main app for triggering
        # Trigger an immediate retrain for one experiment. Used by the
        # apply-tuning and apply-covariate-best endpoints so the user
        # doesn't have to wait for the next scheduled retrain cycle.
        self.retrain_callback = None  # Set by main app
        self.stop_training_callback = None  # Set by main app
        self.running_benchmarks: set = set()
        self.last_update: Optional[datetime] = None
        self.next_update_seconds: Optional[int] = None

    def start_benchmark(self, experiment_name: str):
        """Mark benchmark as running."""
        self.running_benchmarks.add(experiment_name)

    def end_benchmark(self, experiment_name: str):
        """Mark benchmark as completed."""
        self.running_benchmarks.discard(experiment_name)

    def is_benchmark_running(self, experiment_name: str) -> bool:
        """Check if benchmark is running."""
        return experiment_name in self.running_benchmarks


def create_app(config_path: Optional[Path] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Parameters
    ----------
    config_path : Optional[Path]
        Path to configuration file (not currently used by web app directly)

    Returns
    -------
    FastAPI
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="ML Forecast Lab",
        description="Multi-model ML forecasting and benchmarking system",
        version=APP_VERSION,
    )

    # Initialize state
    state = AppState()
    app.state.appstate = state

    # Setup template and static paths
    template_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"

    # Create directories if they don't exist
    template_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)

    templates = Jinja2Templates(directory=str(template_dir))

    # Custom Jinja filters
    def _humanise_name(value: str) -> str:
        """Convert snake_case experiment names to Title Case for display."""
        return value.replace("_", " ").title()

    templates.env.filters["humanise"] = _humanise_name

    # Mount static files
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # CORS middleware for external dashboard integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ========== Ingress support ==========

    def _get_base_path(request: Request) -> str:
        """Get the ingress base path from HA proxy headers, or empty string."""
        return request.headers.get("X-Ingress-Path", "")

    def _find_config_path() -> Optional[Path]:
        """Locate the mlfl.yaml config file."""
        import glob as _glob
        for p in [
            Path("/addon_configs/ml_forecast_lab/mlfl.yaml"),
            Path("/config/mlfl.yaml"),
        ]:
            if p.exists():
                return p
        for match in _glob.glob("/addon_configs/*_ml_forecast_lab/mlfl.yaml"):
            return Path(match)
        return None

    # ---- Model parameter schema (type, default, display label) ----

    MODEL_PARAM_SCHEMA: Dict[str, Dict[str, dict]] = {
        "lightgbm": {
            "n_estimators": {"type": "int", "default": 500, "label": "Number of trees", "min": 10, "max": 5000},
            "max_depth": {"type": "int", "default": 6, "label": "Max tree depth", "min": 1, "max": 20},
            "learning_rate": {"type": "float", "default": 0.05, "label": "Learning rate", "min": 0.001, "max": 1.0, "step": 0.001},
            "num_leaves": {"type": "int", "default": 31, "label": "Max leaves", "min": 2, "max": 256},
            "min_child_samples": {"type": "int", "default": 10, "label": "Min samples per leaf", "min": 1, "max": 100},
            "subsample": {"type": "float", "default": 0.8, "label": "Row subsample ratio", "min": 0.1, "max": 1.0, "step": 0.05},
            "colsample_bytree": {"type": "float", "default": 0.8, "label": "Column subsample ratio", "min": 0.1, "max": 1.0, "step": 0.05},
            "reg_alpha": {"type": "float", "default": 0.1, "label": "L1 regularisation", "min": 0.0, "max": 10.0, "step": 0.01},
            "reg_lambda": {"type": "float", "default": 0.1, "label": "L2 regularisation", "min": 0.0, "max": 10.0, "step": 0.01},
        },
        "xgboost": {
            "n_estimators": {"type": "int", "default": 500, "label": "Number of trees", "min": 10, "max": 5000},
            "max_depth": {"type": "int", "default": 6, "label": "Max tree depth", "min": 1, "max": 20},
            "learning_rate": {"type": "float", "default": 0.05, "label": "Learning rate", "min": 0.001, "max": 1.0, "step": 0.001},
            "subsample": {"type": "float", "default": 0.8, "label": "Row subsample ratio", "min": 0.1, "max": 1.0, "step": 0.05},
            "colsample_bytree": {"type": "float", "default": 0.8, "label": "Column subsample ratio", "min": 0.1, "max": 1.0, "step": 0.05},
            "reg_alpha": {"type": "float", "default": 0.1, "label": "L1 regularisation", "min": 0.0, "max": 10.0, "step": 0.01},
            "reg_lambda": {"type": "float", "default": 1.0, "label": "L2 regularisation", "min": 0.0, "max": 10.0, "step": 0.01},
        },
        "lstm": {
            "hidden_size": {"type": "int", "default": 64, "label": "Hidden size", "min": 8, "max": 512},
            "num_layers": {"type": "int", "default": 2, "label": "LSTM layers", "min": 1, "max": 8},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"]},
        },
        "cnn": {
            "n_filters": {"type": "int", "default": 32, "label": "Filters per layer", "min": 8, "max": 256},
            "kernel_size": {"type": "int", "default": 3, "label": "Kernel size", "min": 2, "max": 15},
            "n_layers": {"type": "int", "default": 4, "label": "Conv layers", "min": 1, "max": 10},
            "dilation_base": {"type": "int", "default": 2, "label": "Dilation base", "min": 1, "max": 4},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"]},
        },
        "neuralprophet": {
            "n_lags": {"type": "int", "default": 12, "label": "Autoregressive lags", "min": 1, "max": 100},
            "n_forecasts": {"type": "int", "default": 1, "label": "Forecast steps", "min": 1, "max": 48},
            "learning_rate": {"type": "float", "default": 0.01, "label": "Learning rate", "min": 1e-5, "max": 0.1, "step": 1e-4},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512},
            "yearly_seasonality": {"type": "bool", "default": False, "label": "Yearly seasonality"},
            "weekly_seasonality": {"type": "bool", "default": True, "label": "Weekly seasonality"},
            "daily_seasonality": {"type": "bool", "default": True, "label": "Daily seasonality"},
            "n_changepoints": {"type": "int", "default": 10, "label": "Trend changepoints", "min": 0, "max": 50},
        },
        "dlinear": {
            "kernel_size": {"type": "int", "default": 25, "label": "Decomposition kernel", "min": 3, "max": 101},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"]},
        },
        "nbeats": {
            "hidden_size": {"type": "int", "default": 64, "label": "Hidden size", "min": 8, "max": 512},
            "n_stacks": {"type": "int", "default": 2, "label": "Stacks", "min": 1, "max": 8},
            "blocks_per_stack": {"type": "int", "default": 2, "label": "Blocks per stack", "min": 1, "max": 8},
            "n_fc_layers": {"type": "int", "default": 4, "label": "FC layers per block", "min": 1, "max": 8},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"]},
        },
        "nhits": {
            "hidden_size": {"type": "int", "default": 64, "label": "Hidden size", "min": 8, "max": 512},
            "n_stacks": {"type": "int", "default": 3, "label": "Stacks", "min": 1, "max": 8},
            "blocks_per_stack": {"type": "int", "default": 1, "label": "Blocks per stack", "min": 1, "max": 8},
            "n_fc_layers": {"type": "int", "default": 4, "label": "FC layers per block", "min": 1, "max": 8},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"]},
        },
        "tide": {
            "hidden_size": {"type": "int", "default": 64, "label": "Hidden size", "min": 8, "max": 512},
            "encoder_layers": {"type": "int", "default": 2, "label": "Encoder layers", "min": 1, "max": 8},
            "decoder_layers": {"type": "int", "default": 2, "label": "Decoder layers", "min": 1, "max": 8},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"]},
        },
        "tsmixer": {
            "n_mixer_layers": {"type": "int", "default": 4, "label": "Mixer layers", "min": 1, "max": 12},
            "hidden": {"type": "int", "default": 64, "label": "Hidden size", "min": 8, "max": 512},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"]},
        },
        "sparsetsf": {
            "period_len": {"type": "int", "default": 48, "label": "Period length", "min": 2, "max": 336},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"]},
        },
        "patchtst": {
            "patch_len": {"type": "int", "default": 8, "label": "Patch length", "min": 2, "max": 48},
            "stride": {"type": "int", "default": 4, "label": "Stride", "min": 1, "max": 24},
            "d_model": {"type": "int", "default": 32, "label": "Model dimension", "min": 8, "max": 256},
            "n_heads": {"type": "int", "default": 4, "label": "Attention heads", "min": 1, "max": 16},
            "n_encoder_layers": {"type": "int", "default": 2, "label": "Encoder layers", "min": 1, "max": 8},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"]},
        },
        "itransformer": {
            "d_model": {"type": "int", "default": 32, "label": "Model dimension", "min": 8, "max": 256},
            "n_heads": {"type": "int", "default": 4, "label": "Attention heads", "min": 1, "max": 16},
            "n_encoder_layers": {"type": "int", "default": 2, "label": "Encoder layers", "min": 1, "max": 8},
            "dim_feedforward": {"type": "int", "default": 64, "label": "Feedforward dimension", "min": 16, "max": 512},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"]},
        },
        "crossformer": {
            "seg_len": {"type": "int", "default": 6, "label": "Segment length", "min": 2, "max": 48},
            "d_model": {"type": "int", "default": 32, "label": "Model dimension", "min": 8, "max": 256},
            "n_heads": {"type": "int", "default": 4, "label": "Attention heads", "min": 1, "max": 16},
            "n_layers": {"type": "int", "default": 2, "label": "Encoder layers", "min": 1, "max": 8},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"]},
        },
        "timesnet": {
            "d_model": {"type": "int", "default": 16, "label": "Model dimension", "min": 8, "max": 256},
            "n_layers": {"type": "int", "default": 2, "label": "TimesBlock layers", "min": 1, "max": 8},
            "top_k": {"type": "int", "default": 3, "label": "Top-K periods", "min": 1, "max": 10},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout", "min": 0.0, "max": 0.8, "step": 0.05},
            "learning_rate": {"type": "float", "default": 2e-4, "label": "Learning rate", "min": 1e-6, "max": 0.01, "step": 1e-5},
            "batch_size": {"type": "int", "default": 64, "label": "Batch size", "min": 8, "max": 512},
            "loss_fn": {"type": "select", "default": "mse", "label": "Loss function", "options": ["mse", "mae", "huber"]},
        },
    }

    # ========== HTML Routes ==========

    @app.get("/", response_class=Response)
    async def dashboard(request: Request):
        """
        Main dashboard showing all experiments, their status, and current best models.
        """
        experiments = list(app.state.appstate.experiment_statuses.values())

        # Build lightweight training summaries for running experiments
        from ml_forecast_lab.training_events import TrainingEventBus
        training_summaries: Dict[str, Dict] = {}
        event_bus = TrainingEventBus.get_instance()
        for exp_name in app.state.appstate.running_benchmarks:
            history = event_bus.get_history(exp_name)
            if history:
                _cur = ""
                _done = 0
                _total = 0
                for ev in history:
                    if ev.event_type == "pipeline_start":
                        import re as _re
                        m = _re.search(r"(\d+) model", ev.message or "")
                        if m:
                            _total = int(m.group(1))
                    elif ev.event_type == "model_start":
                        _cur = ev.model_name
                    elif ev.event_type == "model_end":
                        _done += 1
                training_summaries[exp_name] = {
                    "current_model": _cur,
                    "completed_models": _done,
                    "total_models": _total,
                    "progress_pct": round(_done / _total * 100) if _total else 0,
                }

        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "request": request,
                "base_path": _get_base_path(request),
                "active_page": "dashboard",
                "version": APP_VERSION,
                "experiments": experiments,
                "total_experiments": len(experiments),
                "lab_experiments": sum(
                    1 for e in experiments if e.mode == "lab"
                ),
                "production_experiments": sum(
                    1 for e in experiments if e.mode == "production"
                ),
                "training_summaries": training_summaries,
                "running_experiments": app.state.appstate.running_benchmarks,
            },
        )

    # Model catalog (shared between Models page and experiment detail)
    MODEL_CATALOG = [
        {"name": "cnn", "display_name": "CNN", "model_type": "PyTorch",
         "description": "WaveNet-style dilated causal convolutions with residual connections.",
         "speed": "🔶 Moderate", "best_for": "Periodic/seasonal signals"},
        {"name": "crossformer", "display_name": "Crossformer", "model_type": "PyTorch",
         "description": "Segment embedding with temporal + cross-variable attention.",
         "speed": "🔶 Moderate", "best_for": "Joint temporal + cross-variate modelling"},
        {"name": "dlinear", "display_name": "DLinear", "model_type": "PyTorch",
         "description": "Decomposition-Linear: separate linear layers for trend and seasonal.",
         "speed": "⚡ Fast", "best_for": "Simple baseline — surprisingly competitive"},
        {"name": "itransformer", "display_name": "iTransformer", "model_type": "PyTorch",
         "description": "Inverted Transformer: attention across variables.",
         "speed": "🔶 Moderate", "best_for": "Cross-variate correlations"},
        {"name": "lightgbm", "display_name": "LightGBM", "model_type": "Tree",
         "description": "Gradient boosting framework optimised for speed and memory efficiency.",
         "speed": "⚡ Very Fast", "best_for": "Default choice — fast and accurate"},
        {"name": "lstm", "display_name": "LSTM", "model_type": "PyTorch",
         "description": "2-layer LSTM with temporal attention and multi-horizon output head.",
         "speed": "🔶 Moderate", "best_for": "Complex temporal patterns"},
        {"name": "nbeats", "display_name": "N-BEATS", "model_type": "PyTorch",
         "description": "Neural Basis Expansion with doubly-residual stacking.",
         "speed": "🔶 Moderate", "best_for": "Pure time-series without covariates"},
        {"name": "neuralprophet", "display_name": "NeuralProphet", "model_type": "PyTorch",
         "description": "Neural forecasting with trend decomposition and automatic seasonality.",
         "speed": "🔶 Moderate", "best_for": "Strong seasonality + covariates"},
        {"name": "nhits", "display_name": "N-HiTS", "model_type": "PyTorch",
         "description": "Hierarchical interpolation with multi-rate temporal downsampling.",
         "speed": "🔶 Moderate", "best_for": "Multi-scale temporal patterns"},
        {"name": "patchtst", "display_name": "PatchTST", "model_type": "PyTorch",
         "description": "Channel-independent Patch Transformer with encoder.",
         "speed": "🔶 Moderate", "best_for": "Long-range dependencies"},
        {"name": "sparsetsf", "display_name": "SparseTSF", "model_type": "PyTorch",
         "description": "Period-based sparse cross-period linear model.",
         "speed": "⚡ Fast", "best_for": "Strong daily/weekly periodicity"},
        {"name": "tide", "display_name": "TiDE", "model_type": "PyTorch",
         "description": "Time-series Dense Encoder with residual MLP encoder-decoder.",
         "speed": "🔶 Moderate", "best_for": "Efficient long-horizon forecasting"},
        {"name": "timesnet", "display_name": "TimesNet", "model_type": "PyTorch",
         "description": "FFT period detection with 2D inception convolutions.",
         "speed": "🔶 Moderate", "best_for": "Multi-periodic signals"},
        {"name": "tsmixer", "display_name": "TSMixer", "model_type": "PyTorch",
         "description": "Alternating time-mixing and feature-mixing MLP layers.",
         "speed": "🔶 Moderate", "best_for": "Multivariate cross-channel patterns"},
        {"name": "xgboost", "display_name": "XGBoost", "model_type": "Tree",
         "description": "Extreme gradient boosting with L1/L2 regularisation.",
         "speed": "⚡ Fast", "best_for": "When LightGBM overfits"},
    ]

    @app.get("/models", response_class=Response)
    async def models_page(request: Request):
        """Models configuration page with per-model hyperparameter editing."""
        import yaml

        models_list = MODEL_CATALOG

        # Load enabled models and overrides from config
        models_enabled = []
        model_overrides = {}
        config_path = _find_config_path()
        if config_path:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f) or {}
                exps = yaml_data.get("experiments", [])
                if exps:
                    models_enabled = exps[0].get("models_enabled", [])
                model_overrides = yaml_data.get("model_overrides", {})
            except Exception:
                pass

        return templates.TemplateResponse(
            request=request,
            name="models.html",
            context={
                "request": request,
                "base_path": _get_base_path(request),
                "active_page": "models",
                "version": APP_VERSION,
                "models": models_list,
                "models_enabled": models_enabled,
                "model_overrides": model_overrides,
                "param_schema": MODEL_PARAM_SCHEMA,
            },
        )

    @app.get("/api/models/params")
    async def get_all_model_params():
        """Return parameter schema, defaults, and current overrides for all models."""
        import yaml
        model_overrides = {}
        config_path = _find_config_path()
        if config_path:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f) or {}
                model_overrides = yaml_data.get("model_overrides", {})
            except Exception:
                pass

        result = {}
        for model_name, schema in MODEL_PARAM_SCHEMA.items():
            defaults = {k: v["default"] for k, v in schema.items()}
            overrides = model_overrides.get(model_name, {})
            current = {**defaults, **overrides}
            result[model_name] = {
                "defaults": defaults,
                "overrides": overrides,
                "current": current,
                "schema": schema,
            }
        return JSONResponse(content=result)

    @app.post("/api/models/params")
    async def save_model_params(request: Request):
        """
        Save hyperparameter overrides for a model.
        Body: {"model_name": "lstm", "params": {"hidden_size": 128, "dropout": 0.3}}
        """
        import yaml
        from ml_forecast_lab.config import save_model_overrides

        try:
            data = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        model_name = data.get("model_name")
        params = data.get("params", {})

        if not model_name or model_name not in MODEL_PARAM_SCHEMA:
            return JSONResponse(content={"success": False, "error": f"Unknown model: {model_name}"})

        schema = MODEL_PARAM_SCHEMA[model_name]

        # Validate and cast param values
        validated = {}
        for k, v in params.items():
            if k not in schema:
                return JSONResponse(content={"success": False, "error": f"Unknown param '{k}' for {model_name}"})
            spec = schema[k]
            try:
                if spec["type"] == "int":
                    v = int(v)
                elif spec["type"] == "float":
                    v = float(v)
                elif spec["type"] == "bool":
                    v = bool(v)
                elif spec["type"] == "select":
                    v = str(v)
                    if "options" in spec and v not in spec["options"]:
                        return JSONResponse(content={"success": False, "error": f"Invalid value '{v}' for {k}"})
            except (ValueError, TypeError) as e:
                return JSONResponse(content={"success": False, "error": f"Invalid type for {k}: {e}"})
            validated[k] = v

        # Only store values that differ from defaults
        overrides = {k: v for k, v in validated.items() if v != schema[k]["default"]}

        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        try:
            save_model_overrides(config_path, model_name, overrides)
            defaults = {k: v["default"] for k, v in schema.items()}
            current = {**defaults, **overrides}
            logger.info(f"Saved {len(overrides)} override(s) for {model_name}")
            return JSONResponse(content={"success": True, "overrides": overrides, "current": current})
        except Exception as e:
            logger.error(f"Failed to save model params: {e}")
            return JSONResponse(content={"success": False, "error": str(e)})

    @app.post("/api/models/params/reset")
    async def reset_model_params(request: Request):
        """Reset a model's params to defaults by removing its overrides."""
        from ml_forecast_lab.config import save_model_overrides

        try:
            data = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        model_name = data.get("model_name")
        if not model_name or model_name not in MODEL_PARAM_SCHEMA:
            return JSONResponse(content={"success": False, "error": f"Unknown model: {model_name}"})

        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        try:
            save_model_overrides(config_path, model_name, None)
            defaults = {k: v["default"] for k, v in MODEL_PARAM_SCHEMA[model_name].items()}
            logger.info(f"Reset {model_name} to defaults")
            return JSONResponse(content={"success": True, "defaults": defaults})
        except Exception as e:
            logger.error(f"Failed to reset model params: {e}")
            return JSONResponse(content={"success": False, "error": str(e)})

    @app.get("/experiment/{name}", response_class=Response)
    async def experiment_detail(request: Request, name: str):
        """
        Experiment detail page with model comparison, forecast charts, and metrics.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        exp_status = app.state.appstate.experiment_statuses[name]
        benchmark_result = app.state.appstate.benchmark_results.get(name)
        forecast_data = app.state.appstate.forecast_data.get(name)
        lab_forecast = app.state.appstate.lab_forecast_data.get(name)
        feature_imps = app.state.appstate.feature_importances.get(name, [])
        covariate_analysis = app.state.appstate.covariate_analysis_results.get(name)
        ensemble_result = app.state.appstate.ensemble_results.get(name)
        is_running = app.state.appstate.is_benchmark_running(name)

        # Embed training event history so the page can restore live
        # progress without a separate fetch (same pattern as training_page).
        from ml_forecast_lab.training_events import TrainingEventBus
        embedded_history: Dict[str, list] = {}
        event_bus = TrainingEventBus.get_instance()
        exp_history = event_bus.get_history(name)
        if exp_history:
            embedded_history[name] = [ev.to_dict() for ev in exp_history]

        # Build a lightweight training summary for the header bar
        training_summary: Optional[Dict] = None
        if is_running and exp_history:
            _cur_model = ""
            _completed = 0
            _total = 0
            _fold = 0
            _total_folds = 0
            _epoch = 0
            _total_epochs = 0
            for ev in exp_history:
                if ev.event_type == "pipeline_start":
                    import re as _re
                    m = _re.search(r"(\d+) model", ev.message or "")
                    if m:
                        _total = int(m.group(1))
                elif ev.event_type == "model_start":
                    _cur_model = ev.model_name
                elif ev.event_type == "model_end":
                    _completed += 1
                elif ev.event_type == "epoch":
                    _fold = ev.fold
                    _total_folds = ev.total_folds
                    _epoch = ev.epoch
                    _total_epochs = ev.total_epochs
            training_summary = {
                "current_model": _cur_model,
                "completed_models": _completed,
                "total_models": _total,
                "fold": _fold,
                "total_folds": _total_folds,
                "epoch": _epoch,
                "total_epochs": _total_epochs,
            }

        # Get units, per-experiment models_enabled, and full config from config
        units = ""
        exp_models_enabled: list = []
        exp_config = None
        try:
            from ml_forecast_lab.config import load_config as _lc
            import glob as _g
            for p in ["/addon_configs/ml_forecast_lab/mlfl.yaml", "/config/mlfl.yaml"] + \
                      _g.glob("/addon_configs/*_ml_forecast_lab/mlfl.yaml"):
                cfg_path = Path(p)
                if cfg_path.exists():
                    cfg = _lc(cfg_path)
                    for exp in cfg.experiments:
                        if exp.name == name:
                            units = exp.units or ""
                            exp_models_enabled = list(exp.models_enabled)
                            exp_config = exp
                    break
        except Exception:
            pass

        return templates.TemplateResponse(
            request=request,
            name="experiment.html",
            context={
                "request": request,
                "base_path": _get_base_path(request),
                "active_page": "dashboard",
                "version": APP_VERSION,
                "experiment": exp_status,
                "benchmark_result": benchmark_result,
                "forecast_data": forecast_data,
                "lab_forecast": lab_forecast,
                "feature_importances": feature_imps,
                "is_running": is_running,
                "models": benchmark_result.models if benchmark_result else [],
                "best_model": benchmark_result.best_model_name
                if benchmark_result
                else None,
                "covariate_analysis": covariate_analysis,
                "ensemble_result": ensemble_result,
                "units": units,
                "models_json": [m.model_dump() for m in (benchmark_result.models if benchmark_result else [])],
                "embedded_history": embedded_history,
                "training_summary": training_summary,
                "model_catalog": MODEL_CATALOG,
                "exp_models_enabled": exp_models_enabled,
                "exp_config": exp_config,
                "tuning_result": app.state.appstate.tuning_results.get(name),
                "param_defaults": {m: {p: s["default"] for p, s in schema.items()}
                                   for m, schema in MODEL_PARAM_SCHEMA.items()},
            },
        )

    # ========== API Routes ==========

    @app.post("/experiment/{name}/run-benchmark")
    async def run_benchmark(name: str):
        """
        Trigger a benchmark run for an experiment (async, returns 202 Accepted).
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        if app.state.appstate.is_benchmark_running(name):
            return JSONResponse(
                status_code=409,
                content={"error": "Benchmark already running for this experiment"},
            )

        # Mark as running
        app.state.appstate.start_benchmark(name)

        # In a real application, this would trigger an async task queue
        # For now, we return 202 Accepted and expect the main loop to handle it
        return JSONResponse(
            status_code=202,
            content={
                "message": "Benchmark run accepted",
                "experiment": name,
                "status": "queued",
            },
        )

    @app.post("/api/benchmarks/run-all")
    async def run_all_benchmarks():
        """Trigger benchmark runs for all experiments."""
        queued = []
        skipped = []
        for name, status in app.state.appstate.experiment_statuses.items():
            if app.state.appstate.is_benchmark_running(name):
                skipped.append(name)
            else:
                app.state.appstate.start_benchmark(name)
                if app.state.appstate.benchmark_callback:
                    try:
                        app.state.appstate.benchmark_callback(name)
                    except Exception:
                        pass
                queued.append(name)
        return JSONResponse(
            status_code=202,
            content={
                "message": f"Queued {len(queued)} benchmark(s)",
                "queued": queued,
                "skipped": skipped,
            },
        )

    @app.post("/experiment/{name}/promote/{model_name}")
    async def promote_model(name: str, model_name: str):
        """
        Promote a model to production.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        exp_status = app.state.appstate.experiment_statuses[name]
        benchmark_result = app.state.appstate.benchmark_results.get(name)

        if not benchmark_result:
            raise HTTPException(
                status_code=400, detail="No benchmark results available"
            )

        # Verify model exists in results
        if not any(m.name == model_name for m in benchmark_result.models):
            raise HTTPException(status_code=404, detail="Model not found in results")

        # Update in-memory status
        exp_status.best_model = model_name
        exp_status.selected_model = model_name
        exp_status.mode = "production"

        if benchmark_result:
            for model in benchmark_result.models:
                model.is_production = model.name == model_name

        # Persist to YAML so promotion survives restarts
        from ml_forecast_lab.config import save_experiment_field
        config_path = _find_config_path()
        if config_path:
            try:
                save_experiment_field(config_path, name, "production_model", model_name)
                save_experiment_field(config_path, name, "mode", "production")
            except Exception as e:
                logger.warning(f"Failed to persist promotion to YAML: {e}")

        return JSONResponse(
            content={
                "message": f"Model {model_name} promoted to production",
                "experiment": name,
                "model": model_name,
            }
        )

    @app.post("/experiment/{name}/select-model")
    async def select_model(name: str, request: Request):
        """Set the user's selected model for this experiment."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        model_name = body.get("model_name")
        if not model_name:
            return JSONResponse(content={"success": False, "error": "model_name required"})

        # Validate model exists in benchmark results
        benchmark_result = app.state.appstate.benchmark_results.get(name)
        if benchmark_result and not any(m.name == model_name for m in benchmark_result.models):
            return JSONResponse(content={"success": False, "error": "Model not in results"})

        app.state.appstate.experiment_statuses[name].selected_model = model_name
        return JSONResponse(content={"success": True, "selected_model": model_name})

    @app.post("/experiment/{name}/apply-covariate-best")
    async def apply_covariate_best(name: str):
        """
        Apply the winning covariate configuration from the latest deep
        analysis run, then trigger a background retrain.

        Determines which covariate set has the lowest average MAE across
        all tested models. The winner is one of:
            * "All covariates" — no changes needed
            * "No covariates"  — all covariates removed
            * "Without X"      — single covariate X removed

        Returns a JSON payload describing the action taken.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        result = app.state.appstate.covariate_analysis_results.get(name)
        if not result or result.status != "completed" or not result.results:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "No completed covariate analysis results"},
            )

        # Score every covariate label by its average MAE across models
        label_scores: Dict[str, float] = {}
        for label in result.covariate_labels:
            cells = result.results.get(label, {})
            maes = [
                cell.mae for cell in cells.values()
                if cell is not None and cell.mae is not None
                and not (isinstance(cell.mae, float) and (cell.mae != cell.mae))  # NaN check
            ]
            if maes:
                label_scores[label] = sum(maes) / len(maes)

        if not label_scores:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "No valid metrics in covariate analysis results"},
            )

        best_label = min(label_scores, key=label_scores.get)
        best_score = label_scores[best_label]
        baseline_score = label_scores.get(result.baseline_label, best_score)

        from ml_forecast_lab.config import (
            remove_experiment_covariate,
            clear_experiment_covariates,
        )
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(
                content={"success": False, "error": "Config file not found"},
            )

        action = "no_change"
        removed_covariates: List[str] = []
        try:
            if best_label == result.baseline_label:
                # All covariates already optimal
                action = "no_change"
            elif best_label == "No covariates":
                n_removed = clear_experiment_covariates(config_path, name)
                action = "cleared"
                removed_covariates = [f"{n_removed} covariate(s)"]
            elif best_label.startswith("Without "):
                short_name = best_label[len("Without "):]
                if remove_experiment_covariate(config_path, name, short_name):
                    action = "removed"
                    removed_covariates = [short_name]
                else:
                    return JSONResponse(content={
                        "success": False,
                        "error": f"Could not find covariate '{short_name}' to remove",
                    })
            else:
                return JSONResponse(content={
                    "success": False,
                    "error": f"Unsupported best label: {best_label}",
                })

            # Trigger a background retrain so the user sees the new
            # configuration take effect on the live forecast sensor without
            # waiting for the next scheduled retrain cycle. Skipped when
            # action == "no_change" — nothing changed, no retrain needed.
            retrain_scheduled = False
            if action != "no_change" and app.state.appstate.retrain_callback:
                import asyncio as _aio
                _aio.create_task(app.state.appstate.retrain_callback(name))
                retrain_scheduled = True
                logger.info(
                    f"Scheduled immediate retrain for {name} after apply-covariate-best"
                )

            improvement_pct = None
            if baseline_score and baseline_score > 0:
                improvement_pct = (baseline_score - best_score) / baseline_score * 100

            logger.info(
                f"apply-covariate-best ({name}): winner='{best_label}' "
                f"action={action} removed={removed_covariates} "
                f"improvement={improvement_pct:+.1f}% retraining={retrain_scheduled}"
                if improvement_pct is not None else
                f"apply-covariate-best ({name}): winner='{best_label}' "
                f"action={action} retraining={retrain_scheduled}"
            )

            return JSONResponse(content={
                "success": True,
                "action": action,
                "best_label": best_label,
                "best_mae": round(best_score, 6),
                "baseline_mae": round(baseline_score, 6),
                "improvement_pct": round(improvement_pct, 2) if improvement_pct is not None else None,
                "removed": removed_covariates,
                "retraining": retrain_scheduled,
            })
        except Exception as e:
            logger.error(f"Failed to apply covariate-best: {e}", exc_info=True)
            return JSONResponse(content={"success": False, "error": str(e)})

    @app.post("/experiment/{name}/remove-covariate")
    async def remove_covariate(name: str, request: Request):
        """Remove a covariate from experiment config."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        entity_id = body.get("entity_id")
        if not entity_id:
            return JSONResponse(content={"success": False, "error": "entity_id required"})

        from ml_forecast_lab.config import remove_experiment_covariate
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        try:
            removed = remove_experiment_covariate(config_path, name, entity_id)
            if removed:
                logger.info(f"Removed covariate {entity_id} from {name}")
                return JSONResponse(content={"success": True, "entity_id": entity_id})
            else:
                return JSONResponse(content={"success": False, "error": "Covariate not found"})
        except Exception as e:
            logger.error(f"Failed to remove covariate: {e}")
            return JSONResponse(content={"success": False, "error": str(e)})

    @app.post("/experiment/{name}/add-covariate")
    async def add_covariate(name: str, request: Request):
        """Add a covariate to an experiment's config."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        entity = body.get("entity")
        if not entity:
            return JSONResponse(content={"success": False, "error": "entity is required"})

        cov_dict = {"entity": entity}
        for opt_field in ("role", "aggregation", "scale", "is_binary"):
            if opt_field in body and body[opt_field] is not None:
                cov_dict[opt_field] = body[opt_field]

        from ml_forecast_lab.config import add_experiment_covariate
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        try:
            added = add_experiment_covariate(config_path, name, cov_dict)
            if added:
                logger.info(f"Added covariate {entity} to {name}")
                return JSONResponse(content={"success": True, "entity": entity})
            else:
                return JSONResponse(content={"success": False, "error": "Covariate already exists or experiment not found"})
        except ValueError as e:
            return JSONResponse(content={"success": False, "error": str(e)})
        except Exception as e:
            logger.error(f"Failed to add covariate: {e}")
            return JSONResponse(content={"success": False, "error": str(e)})

    @app.post("/experiment/{name}/stop-training")
    async def stop_training(name: str):
        """Stop a running training/tuning task for an experiment."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        cb = app.state.appstate.stop_training_callback
        if not cb:
            return JSONResponse(content={"success": False, "error": "Stop not available"})

        try:
            stopped = await cb(name)
            if stopped:
                logger.info(f"Stopped training for {name}")
                return JSONResponse(content={"success": True})
            else:
                return JSONResponse(content={"success": False, "error": "No running task for this experiment"})
        except Exception as e:
            logger.error(f"Failed to stop training for {name}: {e}")
            return JSONResponse(content={"success": False, "error": str(e)})

    @app.post("/api/experiments/create")
    async def create_experiment_route(request: Request):
        """Create a new experiment."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        name = body.get("name", "")
        target_entity = body.get("target_entity", "")

        if not name or not target_entity:
            return JSONResponse(content={"success": False, "error": "name and target_entity are required"})

        exp_dict = {"name": name, "target_entity": target_entity}
        for opt in ("source_is_cumulative", "reset_daily"):
            if opt in body:
                exp_dict[opt] = bool(body[opt])

        from ml_forecast_lab.config import create_experiment
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        try:
            create_experiment(config_path, exp_dict)
        except ValueError as e:
            return JSONResponse(content={"success": False, "error": str(e)})
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            return JSONResponse(content={"success": False, "error": str(e)})

        # Register in-memory so it appears immediately
        from ml_forecast_lab.web.app import ExperimentStatus
        app.state.appstate.experiment_statuses[name] = ExperimentStatus(
            name=name,
            target_entity=target_entity,
            mode="lab",
        )

        logger.info(f"Created experiment '{name}' targeting {target_entity}")
        return JSONResponse(content={"success": True, "redirect": f"/experiment/{name}"})

    @app.post("/api/experiments/{name}/delete")
    async def delete_experiment_route(name: str):
        """Delete an experiment."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        from ml_forecast_lab.config import delete_experiment
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        try:
            removed = delete_experiment(config_path, name)
            if not removed:
                return JSONResponse(content={"success": False, "error": "Experiment not found in config"})
        except Exception as e:
            logger.error(f"Failed to delete experiment: {e}")
            return JSONResponse(content={"success": False, "error": str(e)})

        # Remove from in-memory state
        app.state.appstate.experiment_statuses.pop(name, None)
        app.state.appstate.benchmark_results.pop(name, None)
        app.state.appstate.forecast_data.pop(name, None)
        app.state.appstate.lab_forecast_data.pop(name, None)
        app.state.appstate.feature_importances.pop(name, None)
        app.state.appstate.covariate_analysis_results.pop(name, None)
        app.state.appstate.tuning_results.pop(name, None)

        logger.info(f"Deleted experiment '{name}'")
        return JSONResponse(content={"success": True, "redirect": "/"})

    # ---- HA entity search (cached) ----
    _entity_cache: Dict[str, Any] = {"data": [], "ts": 0.0}

    @app.get("/api/ha/entities")
    async def ha_entities(request: Request):
        """Search HA entities for the covariate / target entity picker."""
        import time
        import aiohttp as _aiohttp

        q = (request.query_params.get("q") or "").lower().strip()
        now = time.time()

        # Refresh cache every 60 seconds
        if now - _entity_cache["ts"] > 60:
            ha_url = os.environ.get("HA_URL", "http://supervisor/core")
            ha_token = os.environ.get("SUPERVISOR_TOKEN", "")
            try:
                async with _aiohttp.ClientSession() as sess:
                    headers = {"Authorization": f"Bearer {ha_token}"}
                    async with sess.get(f"{ha_url}/api/states", headers=headers, timeout=_aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            states = await resp.json()
                            _entity_cache["data"] = [
                                {
                                    "entity_id": s["entity_id"],
                                    "friendly_name": s.get("attributes", {}).get("friendly_name", ""),
                                    "state": str(s.get("state", "")),
                                }
                                for s in states
                                if isinstance(s, dict) and "entity_id" in s
                            ]
                            _entity_cache["ts"] = now
            except Exception as e:
                logger.debug(f"Entity cache refresh failed: {e}")
                # Use stale cache or empty list

        entities = _entity_cache["data"]
        if q:
            entities = [
                e for e in entities
                if q in e["entity_id"].lower() or q in e["friendly_name"].lower()
            ]

        return JSONResponse(content=entities[:50])

    @app.post("/experiment/{name}/toggle-mode")
    async def toggle_mode(name: str):
        """
        Toggle experiment between lab and production mode.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        exp_status = app.state.appstate.experiment_statuses[name]
        old_mode = exp_status.mode
        new_mode = "production" if old_mode == "lab" else "lab"
        exp_status.mode = new_mode

        # Persist to YAML
        from ml_forecast_lab.config import save_experiment_field
        config_path = _find_config_path()
        if config_path:
            try:
                save_experiment_field(config_path, name, "mode", new_mode)
                if new_mode == "production" and exp_status.selected_model:
                    save_experiment_field(config_path, name, "production_model",
                                         exp_status.selected_model)
            except Exception as e:
                logger.warning(f"Failed to persist mode toggle: {e}")

        return JSONResponse(
            content={
                "message": f"Switched {name} to {new_mode} mode",
                "experiment": name,
                "mode": new_mode,
            }
        )

    @app.post("/experiment/{name}/run-covariate-analysis")
    async def run_covariate_analysis(name: str, request: Request):
        """
        Trigger a covariate analysis (async, returns 202 Accepted).
        Body (optional): {"model": "lightgbm"} or {"model": "all"}
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        existing = app.state.appstate.covariate_analysis_results.get(name)
        if existing and existing.status == "running":
            return JSONResponse(
                status_code=409,
                content={"error": "Covariate analysis already running"},
            )

        # Parse optional model selection
        selected_model = "all"
        try:
            body = await request.json()
            selected_model = body.get("model", "all")
        except Exception:
            pass

        # Trigger via callback if available
        import asyncio
        if app.state.appstate.covariate_analysis_callback:
            asyncio.create_task(app.state.appstate.covariate_analysis_callback(name, selected_model))

        return JSONResponse(
            status_code=202,
            content={
                "message": "Covariate analysis started",
                "experiment": name,
                "model": selected_model,
                "status": "running",
            },
        )

    @app.get("/experiment/{name}/covariate-analysis")
    async def get_covariate_analysis(name: str):
        """Get covariate analysis results as JSON."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        result = app.state.appstate.covariate_analysis_results.get(name)
        if not result:
            raise HTTPException(status_code=404, detail="No covariate analysis results")
        return result.model_dump()

    # ========== Tuning endpoints ==========

    @app.post("/experiment/{name}/run-tuning")
    async def run_tuning(name: str, request: Request):
        """Start hyperparameter tuning. Body: {model_name, n_trials?, strategy?}"""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        # Check if tuning is already running
        existing = app.state.appstate.tuning_results.get(name)
        if existing and existing.status == "running":
            return JSONResponse(
                status_code=409,
                content={"error": "Tuning already running for this experiment"},
            )

        try:
            body = await request.json()
        except Exception:
            body = {}

        model_name = body.get("model_name")
        if not model_name:
            return JSONResponse(
                status_code=400,
                content={"error": "model_name is required"},
            )

        n_trials = body.get("n_trials", 30)
        strategy = body.get("strategy", "tpe")

        # Get the parameter schema for this model
        schema = MODEL_PARAM_SCHEMA.get(model_name, {})
        if not schema:
            return JSONResponse(
                status_code=400,
                content={"error": f"No parameter schema for model: {model_name}"},
            )

        if not app.state.appstate.tuning_callback:
            raise HTTPException(status_code=501, detail="Tuning callback not registered")

        import asyncio as _aio
        _aio.create_task(
            app.state.appstate.tuning_callback(
                name, model_name, n_trials, strategy, schema
            )
        )

        return JSONResponse(
            status_code=202,
            content={"message": "Tuning started", "model": model_name,
                      "n_trials": n_trials, "strategy": strategy},
        )

    @app.get("/experiment/{name}/tuning")
    async def get_tuning(name: str):
        """Get current tuning state (for polling)."""
        result = app.state.appstate.tuning_results.get(name)
        if not result:
            return JSONResponse(status_code=404, content={"error": "No tuning results"})
        return JSONResponse(content=result.model_dump())

    @app.post("/experiment/{name}/apply-tuning")
    async def apply_tuning(name: str):
        """Apply best tuning params and promote the model to production."""
        result = app.state.appstate.tuning_results.get(name)
        if not result or not result.best_params:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "No tuning results to apply"},
            )

        from ml_forecast_lab.config import (
            save_experiment_model_params,
            save_experiment_field,
        )
        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(
                content={"success": False, "error": "Config file not found"},
            )

        try:
            # 1. Save tuned params per-experiment
            save_experiment_model_params(
                config_path, name, result.model_name, result.best_params
            )

            # 2. Promote the tuned model to production
            exp_status = app.state.appstate.experiment_statuses.get(name)
            if exp_status:
                exp_status.best_model = result.model_name
                exp_status.selected_model = result.model_name
                exp_status.mode = "production"

            benchmark = app.state.appstate.benchmark_results.get(name)
            if benchmark:
                for m in benchmark.models:
                    m.is_production = m.name == result.model_name

            save_experiment_field(config_path, name, "production_model", result.model_name)
            save_experiment_field(config_path, name, "mode", "production")

            # 3. Trigger an immediate retrain so the user doesn't have to
            # wait for the next scheduled retrain cycle to see the tuned
            # params take effect on the live forecast sensor.
            retrain_scheduled = False
            if app.state.appstate.retrain_callback:
                import asyncio as _aio
                _aio.create_task(app.state.appstate.retrain_callback(name))
                retrain_scheduled = True
                logger.info(f"Scheduled immediate retrain for {name} after apply-tuning")

            logger.info(
                f"Applied tuned params and promoted {result.model_name} "
                f"for experiment {name}"
            )
            return JSONResponse(content={
                "success": True, "model": result.model_name,
                "params": result.best_params, "promoted": True,
                "retraining": retrain_scheduled,
            })
        except Exception as e:
            logger.error(f"Failed to apply tuning: {e}")
            return JSONResponse(content={"success": False, "error": str(e)})

    @app.get("/experiment/{name}/results")
    async def get_results(name: str):
        """
        Get latest benchmark results as JSON.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        result = app.state.appstate.benchmark_results.get(name)
        if not result:
            raise HTTPException(status_code=404, detail="No benchmark results yet")

        return result.model_dump()

    @app.get("/experiment/{name}/forecast")
    async def get_forecast(name: str):
        """
        Get latest forecast data for charting.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        forecast = app.state.appstate.forecast_data.get(name)
        if not forecast:
            raise HTTPException(status_code=404, detail="No forecast data yet")

        return forecast.model_dump()

    @app.get("/api/status")
    async def health_check() -> HealthStatus:
        """
        Health check and overall status.
        """
        experiments = app.state.appstate.experiment_statuses.values()
        lab_count = sum(1 for e in experiments if e.mode == "lab")
        prod_count = sum(1 for e in experiments if e.mode == "production")

        return HealthStatus(
            status="healthy",
            service="ml-forecast-lab",
            version=APP_VERSION,
            timestamp=datetime.now(timezone.utc).isoformat(),
            experiments_total=len(list(experiments)),
            experiments_lab=lab_count,
            experiments_production=prod_count,
        )

    # ========== Log Routes ==========

    LOG_FILE = Path("/data/ml_forecast_lab/logs/mlfl.log")

    @app.get("/log", response_class=Response)
    async def view_log(request: Request, lines: int = 500):
        """
        View recent log output in styled template.
        """
        log_text = ""
        for log_path in [LOG_FILE.with_suffix(".log.1"), LOG_FILE]:
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        log_text += f.read()
                except Exception as e:
                    log_text += f"\n[Error reading {log_path}: {e}]\n"

        all_lines = log_text.splitlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return templates.TemplateResponse(
            request=request,
            name="logs.html",
            context={
                "request": request,
                "base_path": _get_base_path(request),
                "active_page": "logs",
                "version": APP_VERSION,
                "log_content": "\n".join(tail),
            },
        )

    @app.get("/status")
    async def status_page(request: Request):
        """Redirect old status page to /system."""
        return RedirectResponse(url=f"{_get_base_path(request)}/system", status_code=301)

    @app.post("/api/models/toggle")
    async def toggle_model(request: Request):
        """
        Toggle a model on/off in the config. Updates models_enabled in mlfl.yaml.
        Body: {"model_name": "lstm", "enabled": true}
        """
        import yaml
        import glob as _glob

        try:
            data = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        model_name = data.get("model_name")
        enabled = data.get("enabled", True)

        if not model_name:
            return JSONResponse(content={"success": False, "error": "model_name required"})

        # Find config file
        config_path = None
        for p in [Path("/addon_configs/ml_forecast_lab/mlfl.yaml"), Path("/config/mlfl.yaml")]:
            if p.exists():
                config_path = p
                break
        for match in _glob.glob("/addon_configs/*_ml_forecast_lab/mlfl.yaml"):
            config_path = Path(match)
            break

        if not config_path or not config_path.exists():
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            # Update models_enabled in all experiments
            for exp in yaml_data.get("experiments", []):
                models = exp.get("models_enabled", [])
                if enabled and model_name not in models:
                    models.append(model_name)
                elif not enabled and model_name in models:
                    models.remove(model_name)
                exp["models_enabled"] = models

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Model {model_name} {'enabled' if enabled else 'disabled'}")
            return JSONResponse(content={"success": True})

        except Exception as e:
            logger.error(f"Failed to toggle model: {e}")
            return JSONResponse(content={"success": False, "error": str(e)})

    @app.post("/api/experiment/{exp_name}/models/toggle")
    async def toggle_experiment_model(exp_name: str, request: Request):
        """
        Toggle a model on/off for a specific experiment.
        Body: {"model_name": "lstm", "enabled": true}
        """
        import yaml
        import glob as _glob

        try:
            data = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        model_name = data.get("model_name")
        enabled = data.get("enabled", True)

        if not model_name:
            return JSONResponse(content={"success": False, "error": "model_name required"})

        config_path = _find_config_path()
        if not config_path:
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            # Update models_enabled only for the target experiment
            for exp in yaml_data.get("experiments", []):
                if exp.get("name") != exp_name:
                    continue
                models = exp.get("models_enabled", [])
                if enabled and model_name not in models:
                    models.append(model_name)
                elif not enabled and model_name in models:
                    models.remove(model_name)
                exp["models_enabled"] = models
                break

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Model {model_name} {'enabled' if enabled else 'disabled'} for {exp_name}")
            return JSONResponse(content={"success": True})

        except Exception as e:
            logger.error(f"Failed to toggle model for {exp_name}: {e}")
            return JSONResponse(content={"success": False, "error": str(e)})

    @app.get("/settings")
    async def settings_page(request: Request):
        """Redirect old settings page to /system."""
        return RedirectResponse(url=f"{_get_base_path(request)}/system", status_code=301)

    @app.get("/system", response_class=Response)
    async def system_page(request: Request):
        """
        Unified system page: health, hardware, settings, experiments.
        Replaces the former separate /status and /settings pages.
        """
        import yaml

        experiment_statuses = list(app.state.appstate.experiment_statuses.values())
        lab_count = sum(1 for e in experiment_statuses if e.mode == "lab")
        prod_count = sum(1 for e in experiment_statuses if e.mode == "production")

        health = {
            "status": "healthy",
            "version": APP_VERSION,
            "experiments_total": len(experiment_statuses),
            "experiments_lab": lab_count,
            "experiments_production": prod_count,
        }

        # Hardware info
        cpu_count = os.cpu_count() or 4
        try:
            cpu_model = platform.processor() or platform.machine()
        except Exception:
            cpu_model = platform.machine()

        try:
            import psutil
            mem = psutil.virtual_memory()
            memory_total_gb = round(mem.total / (1024**3), 1)
            memory_used_gb = round(mem.used / (1024**3), 1)
            memory_percent = mem.percent
        except ImportError:
            try:
                with open("/proc/meminfo") as f:
                    meminfo = {line.split(":")[0]: int(line.split()[1]) for line in f if len(line.split()) >= 2}
                memory_total_gb = round(meminfo.get("MemTotal", 0) / (1024**2), 1)
                memory_used_gb = round((meminfo.get("MemTotal", 0) - meminfo.get("MemAvailable", 0)) / (1024**2), 1)
                memory_percent = round(memory_used_gb / max(memory_total_gb, 0.1) * 100, 1)
            except Exception:
                memory_total_gb = memory_used_gb = memory_percent = 0
        except Exception:
            memory_total_gb = memory_used_gb = memory_percent = 0

        try:
            disk = shutil.disk_usage("/data")
            disk_total_gb = round(disk.total / (1024**3), 1)
            disk_used_gb = round(disk.used / (1024**3), 1)
            disk_percent = round(disk.used / disk.total * 100, 1)
        except Exception:
            disk_total_gb = disk_used_gb = disk_percent = 0

        system_info = {
            "cpu_cores": cpu_count,
            "cpu_model": cpu_model,
            "memory_total_gb": memory_total_gb,
            "memory_used_gb": memory_used_gb,
            "memory_percent": memory_percent,
            "disk_total_gb": disk_total_gb,
            "disk_used_gb": disk_used_gb,
            "disk_percent": disk_percent,
        }

        # Config data
        config_data = {
            "forecast_every_minutes": 30,
            "retrain_every_hours": 24.0,
            "update_every_minutes": 30,
            "timezone": "UTC",
            "cpu_cores": 0,
            "nice_priority": 10,
        }
        config_path_str = "unknown"
        experiment_configs = []
        cp = _find_config_path()
        if cp:
            config_path_str = str(cp)
        try:
            from ml_forecast_lab.config import load_config as _load_config
            cfg = _load_config(config_path_str)
            config_data = {
                "forecast_every_minutes": cfg.forecast_every_minutes,
                "retrain_every_hours": cfg.retrain_every_hours,
                "update_every_minutes": cfg.forecast_every_minutes,
                "timezone": cfg.timezone,
                "cpu_cores": cfg.cpu_cores,
                "nice_priority": cfg.nice_priority,
            }
            experiment_configs = cfg.experiments
        except Exception:
            pass

        return templates.TemplateResponse(
            request=request,
            name="system.html",
            context={
                "request": request,
                "base_path": _get_base_path(request),
                "active_page": "system",
                "version": APP_VERSION,
                "health": health,
                "system": system_info,
                "config": config_data,
                "config_path": config_path_str,
                "experiment_statuses": experiment_statuses,
                "experiment_configs": experiment_configs,
            },
        )

    @app.post("/api/settings")
    async def save_settings(request: Request):
        """
        Save settings back to mlfl.yaml.
        """
        import yaml

        try:
            data = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        # Find config file
        config_path = None
        for p in [Path("/addon_configs/ml_forecast_lab/mlfl.yaml"), Path("/config/mlfl.yaml")]:
            if p.exists():
                config_path = p
                break
        import glob as _glob
        for match in _glob.glob("/addon_configs/*_ml_forecast_lab/mlfl.yaml"):
            config_path = Path(match)
            break

        if not config_path or not config_path.exists():
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        try:
            # Read existing YAML
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            # Update fields
            if "forecast_every_minutes" in data:
                yaml_data["forecast_every_minutes"] = int(data["forecast_every_minutes"])
                yaml_data["update_every_minutes"] = int(data["forecast_every_minutes"])
            elif "update_every_minutes" in data:
                yaml_data["update_every_minutes"] = int(data["update_every_minutes"])
            if "retrain_every_hours" in data:
                yaml_data["retrain_every_hours"] = float(data["retrain_every_hours"])
            if "timezone" in data:
                yaml_data["timezone"] = str(data["timezone"])
            if "cpu_cores" in data:
                yaml_data["cpu_cores"] = int(data["cpu_cores"])
            if "nice_priority" in data:
                yaml_data["nice_priority"] = int(data["nice_priority"])

            # Write back
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Settings saved to {config_path}")
            return JSONResponse(content={"success": True})

        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return JSONResponse(content={"success": False, "error": str(e)})

    @app.post("/api/experiment-settings")
    async def save_experiment_settings(request: Request):
        """
        Save per-experiment training settings (CV strategy, folds, recency weighting).

        Persists changes to mlfl.yaml without requiring add-on restart.
        Changes take effect on next training run.
        """
        import yaml

        try:
            data = await request.json()
        except Exception:
            return JSONResponse(content={"success": False, "error": "Invalid JSON"})

        exp_name = data.get("experiment")
        if not exp_name:
            return JSONResponse(content={"success": False, "error": "Missing experiment name"})

        # Allowed editable fields and their types/validators
        editable = {
            "cv_strategy": lambda v: v if v in ("walk_forward", "sliding_window") else None,
            "cv_folds": lambda v: int(v) if int(v) >= 2 else None,
            "recency_half_life_days": lambda v: float(v) if float(v) >= 0 else None,
            "days_history": lambda v: int(v) if int(v) >= 1 else None,
            "interval_minutes": lambda v: int(v) if int(v) >= 1 else None,
            "future_periods": lambda v: int(v) if int(v) >= 1 else None,
            "source_is_cumulative": lambda v: bool(v),
            "reset_daily": lambda v: bool(v),
            "log_transform": lambda v: bool(v),
            "forecast_every_minutes": lambda v: int(v) if int(v) >= 1 else None,
            "retrain_every_hours": lambda v: float(v) if float(v) >= 0.1 else None,
            "production_metric": lambda v: v if v in ("mae", "rmse", "mase") else None,
            "loss_fn": lambda v: v if v in ("mse", "mae", "huber") else None,
            "max_increment": lambda v: float(v) if float(v) > 0 else None,
        }

        # Fields where None/null means "use global default" (valid, not an error)
        nullable_fields = {"forecast_every_minutes", "retrain_every_hours", "max_increment"}

        updates = {}
        for field, validator in editable.items():
            if field in data:
                raw = data[field]
                if raw is None and field in nullable_fields:
                    updates[field] = None
                    continue
                try:
                    val = validator(raw)
                    if val is None:
                        return JSONResponse(content={
                            "success": False, "error": f"Invalid value for {field}"
                        })
                    updates[field] = val
                except (ValueError, TypeError) as e:
                    return JSONResponse(content={
                        "success": False, "error": f"Invalid {field}: {e}"
                    })

        if not updates:
            return JSONResponse(content={"success": False, "error": "No valid fields to update"})

        # Find config file
        config_path = None
        for p in [Path("/addon_configs/ml_forecast_lab/mlfl.yaml"), Path("/config/mlfl.yaml")]:
            if p.exists():
                config_path = p
                break
        import glob as _glob
        for match in _glob.glob("/addon_configs/*_ml_forecast_lab/mlfl.yaml"):
            config_path = Path(match)
            break

        if not config_path or not config_path.exists():
            return JSONResponse(content={"success": False, "error": "Config file not found"})

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            # Find the experiment in the YAML
            experiments = yaml_data.get("experiments", [])
            found = False
            for exp in experiments:
                if exp.get("name") == exp_name:
                    for k, v in updates.items():
                        if v is None:
                            exp.pop(k, None)  # remove → falls back to global default
                        else:
                            exp[k] = v
                    found = True
                    break

            if not found:
                return JSONResponse(content={
                    "success": False, "error": f"Experiment '{exp_name}' not found in config"
                })

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

            # Also update the in-memory config if possible
            try:
                cfg = _load_config()
                for exp_cfg in cfg.experiments:
                    if exp_cfg.name == exp_name:
                        for k, v in updates.items():
                            setattr(exp_cfg, k, v)
                        break
            except Exception:
                pass  # Config will reload on next training run

            logger.info(f"Experiment '{exp_name}' settings updated: {updates}")
            return JSONResponse(content={"success": True})

        except Exception as e:
            logger.error(f"Failed to save experiment settings: {e}")
            return JSONResponse(content={"success": False, "error": str(e)})

    @app.get("/api/log")
    async def api_log(
        lines: int = 200,
        level: str = "all",
        search: str = "",
    ):
        """
        JSON log API with filtering.

        Parameters:
            lines: max lines to return (default 200)
            level: 'all', 'info', 'warning', 'error' (filters by level)
            search: text search filter
        """
        log_text = ""
        for log_path in [LOG_FILE.with_suffix(".log.1"), LOG_FILE]:
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        log_text += f.read()
                except Exception:
                    pass

        all_lines = log_text.splitlines()

        # Filter by level
        if level != "all":
            level_upper = level.upper()
            all_lines = [
                l for l in all_lines if f"- {level_upper} -" in l.upper()
            ]

        # Filter by search term
        if search:
            search_lower = search.lower()
            all_lines = [l for l in all_lines if search_lower in l.lower()]

        # Tail
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return {
            "total_lines": len(all_lines),
            "returned_lines": len(tail),
            "lines": tail,
        }

    @app.get("/debug_log", response_class=Response)
    async def download_log():
        """
        Download the full current log file.
        """
        if not LOG_FILE.exists():
            raise HTTPException(status_code=404, detail="No log file found")

        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return Response(
            content=content,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=mlfl.log"},
        )

    @app.get("/dashboard_yaml", response_class=Response)
    async def download_dashboard():
        """
        Download the auto-generated ApexCharts dashboard YAML.
        Import this into HA via Settings > Dashboards > Add > From YAML.
        """
        import glob

        dashboard_paths = [
            *[Path(d) / "mlfl_dashboard.yaml" for d in glob.glob("/addon_configs/*_ml_forecast_lab")],
            Path("/addon_configs/ml_forecast_lab/mlfl_dashboard.yaml"),
        ]

        for path in dashboard_paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                return Response(
                    content=content,
                    media_type="text/yaml; charset=utf-8",
                    headers={"Content-Disposition": "attachment; filename=mlfl_dashboard.yaml"},
                )

        raise HTTPException(status_code=404, detail="Dashboard YAML not generated yet")

    @app.get("/api/models")
    async def list_models() -> List[ModelInfo]:
        """
        List available model backends.
        """
        return [
            ModelInfo(
                name="lightgbm",
                display_name="LightGBM",
                description="Gradient boosting framework for fast training",
            ),
            ModelInfo(
                name="xgboost",
                display_name="XGBoost",
                description="Extreme gradient boosting",
            ),
            ModelInfo(
                name="lstm",
                display_name="LSTM",
                description="Long short-term memory neural network",
            ),
            ModelInfo(
                name="cnn",
                display_name="CNN",
                description="Convolutional neural network",
            ),
        ]

    # ========== Training Tab: SSE + Routes ==========

    @app.get("/training")
    async def training_page(request: Request):
        """Redirect old Training tab URL to Dashboard (preserved for bookmarks)."""
        return RedirectResponse(url=f"{_get_base_path(request)}/", status_code=301)

    @app.get("/experiment/{name}/training-stream")
    async def training_stream(name: str, request: Request):
        """
        Server-Sent Events endpoint for live training metrics.

        Streams TrainingEvent objects as JSON. Replays history on connect
        so late-joining clients catch up, then streams live events until
        a pipeline_end event is received.

        Pass ?no_replay=1 to skip history replay (e.g. when the client
        already replayed via the /api/training/history endpoint).
        """
        import asyncio as _aio
        from ml_forecast_lab.training_events import TrainingEventBus

        skip_replay = request.query_params.get("no_replay") == "1"
        event_bus = TrainingEventBus.get_instance()
        loop = _aio.get_running_loop()
        q = event_bus.subscribe(name, loop)

        async def _generate():
            try:
                # Replay history for reconnecting clients (unless already done)
                if not skip_replay:
                    for ev in event_bus.get_history(name):
                        yield f"data: {json.dumps(ev.to_dict())}\n\n"

                # Stream live events
                while True:
                    try:
                        event = await _aio.wait_for(q.get(), timeout=30.0)
                    except _aio.TimeoutError:
                        # Send keep-alive comment
                        yield ": keepalive\n\n"
                        continue

                    yield f"data: {json.dumps(event.to_dict())}\n\n"

                    if event.event_type == "pipeline_end":
                        break
            except _aio.CancelledError:
                pass
            finally:
                event_bus.unsubscribe(name, q)

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/experiment/{name}/run-pipeline")
    async def run_pipeline(name: str, request: Request):
        """
        Trigger the full training pipeline for an experiment.

        Accepts optional JSON body:
          {"steps": ["benchmark"]}                     — default: benchmark only
          {"steps": ["benchmark", "covariate_analysis"]} — benchmark then covariate analysis

        Returns 202 Accepted. Progress is streamed via the SSE endpoint.
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        if app.state.appstate.is_benchmark_running(name):
            return JSONResponse(
                status_code=409,
                content={"error": "Pipeline already running for this experiment"},
            )

        steps = ["benchmark"]
        try:
            body = await request.json()
            steps = body.get("steps", ["benchmark"])
        except Exception:
            pass

        import asyncio as _aio

        if not getattr(app.state.appstate, 'benchmark_callback', None):
            raise HTTPException(status_code=501, detail="Benchmark callback not registered")

        # Clear stale event history and mark as running so the UI can
        # detect an active pipeline when the Training tab is re-opened.
        from ml_forecast_lab.training_events import TrainingEventBus
        TrainingEventBus.get_instance().clear_history(name)
        app.state.appstate.start_benchmark(name)

        # Run the full pipeline as a background task
        async def _pipeline():
            try:
                # Benchmark step
                if "benchmark" in steps:
                    await app.state.appstate.benchmark_callback(name)
                # Ensemble step
                if "ensemble" in steps and getattr(app.state.appstate, 'ensemble_callback', None):
                    await app.state.appstate.ensemble_callback(name)
                # Covariate analysis step
                if "covariate_analysis" in steps and app.state.appstate.covariate_analysis_callback:
                    await app.state.appstate.covariate_analysis_callback(name, "all")
            except Exception as e:
                logger.error(f"Pipeline failed for {name}: {e}", exc_info=True)
            finally:
                app.state.appstate.end_benchmark(name)

        _task = _aio.create_task(_pipeline())
        # Track for stop-training support (if the main app registered the dict)
        if hasattr(app.state.appstate, 'stop_training_callback'):
            # The main app tracks via _running_tasks; register so the
            # _stop_training_trigger callback can find and cancel this task.
            # We store on appstate as a simple dict keyed by experiment name.
            if not hasattr(app.state.appstate, '_pipeline_tasks'):
                app.state.appstate._pipeline_tasks = {}
            app.state.appstate._pipeline_tasks[name] = _task
            _task.add_done_callback(
                lambda t, n=name: app.state.appstate._pipeline_tasks.pop(n, None)
            )

        return JSONResponse(
            status_code=202,
            content={
                "message": "Pipeline started",
                "experiment": name,
                "steps": steps,
                "status": "running",
            },
        )

    @app.get("/api/training/history/{name}")
    async def training_history(name: str):
        """Return all training events for an experiment as JSON."""
        from ml_forecast_lab.training_events import TrainingEventBus

        event_bus = TrainingEventBus.get_instance()
        history = event_bus.get_history(name)
        return JSONResponse(content=[ev.to_dict() for ev in history])

    # ========== Ensemble endpoints ==========

    @app.post("/experiment/{name}/run-ensemble")
    async def run_ensemble(name: str, request: Request):
        """
        Trigger ensemble strategies on existing benchmark results.

        Body (all optional):
            strategies: list[str]  — e.g. ["simple_average", "stacking"]
            selected_models: list[str]  — explicit model names to include
            top_n: int  — include only the top N models by production metric
        """
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        if not app.state.appstate.ensemble_callback:
            raise HTTPException(status_code=501, detail="Ensemble callback not registered")

        strategies = None
        selected_models = None
        top_n = None
        try:
            body = await request.json()
            strategies = body.get("strategies")
            selected_models = body.get("selected_models")
            top_n = body.get("top_n")
        except Exception:
            pass

        import asyncio as _aio
        _aio.create_task(app.state.appstate.ensemble_callback(
            name, strategies,
            selected_models=selected_models, top_n=top_n,
        ))

        return JSONResponse(
            status_code=202,
            content={"message": "Ensemble started", "experiment": name},
        )

    @app.get("/experiment/{name}/ensemble-models")
    async def get_ensemble_models(name: str):
        """Return list of models available for ensemble with their metrics."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        completed_models = getattr(app.state.appstate, '_benchmark_models', {}).get(name)
        runner = getattr(app.state.appstate, '_benchmark_runners', {}).get(name)
        if not completed_models:
            return JSONResponse(content={"models": [], "production_metric": "mae"})

        production_metric = runner.production_metric if runner else "mae"
        models = []
        for m_name, mr in completed_models.items():
            if mr.fold_predictions and mr.fold_actuals:
                models.append({
                    "name": m_name,
                    "metric_name": production_metric,
                    "metric_value": float(mr.metrics.get(production_metric, float('inf'))),
                })
        # Sort by metric (best first)
        models.sort(key=lambda x: x["metric_value"])
        return JSONResponse(content={"models": models, "production_metric": production_metric})

    @app.get("/experiment/{name}/ensemble")
    async def get_ensemble_results(name: str):
        """Get ensemble results as JSON."""
        if name not in app.state.appstate.experiment_statuses:
            raise HTTPException(status_code=404, detail="Experiment not found")

        result = app.state.appstate.ensemble_results.get(name)
        if not result:
            raise HTTPException(status_code=404, detail="No ensemble results yet")
        return result.model_dump()

    return app
