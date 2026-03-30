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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    mape: MetricValue
    train_time_seconds: float
    rank: int
    is_production: bool = False
    fold_results: Optional[List[Dict[str, float]]] = None


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
    last_benchmark_timestamp: Optional[str] = None
    last_benchmark_status: str = "pending"
    next_update_in_seconds: Optional[int] = None


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
    """Predictions from a single model on holdout data."""

    model_name: str
    timestamps: List[str]
    actuals: List[Optional[float]]
    predictions: List[float]
    color: str = "#00d4ff"


class LabForecastData(BaseModel):
    """Multi-model prediction data for lab mode visualisation."""

    experiment_name: str
    holdout_start: str
    holdout_end: str
    model_predictions: List[ModelPrediction]


class FeatureImportanceData(BaseModel):
    """Feature importance from a trained model."""

    model_name: str
    features: List[Dict[str, Any]]  # [{"name": "hour_of_day", "importance": 0.25}, ...]


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
        version="0.2.0",
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

    # ========== HTML Routes ==========

    @app.get("/", response_class=Response)
    async def dashboard(request: Request):
        """
        Main dashboard showing all experiments, their status, and current best models.
        """
        experiments = list(app.state.appstate.experiment_statuses.values())
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "request": request,
                "base_path": _get_base_path(request),
                "active_page": "dashboard",
                "version": "0.3.2",
                "experiments": experiments,
                "total_experiments": len(experiments),
                "lab_experiments": sum(
                    1 for e in experiments if e.mode == "lab"
                ),
                "production_experiments": sum(
                    1 for e in experiments if e.mode == "production"
                ),
            },
        )

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
        is_running = app.state.appstate.is_benchmark_running(name)

        return templates.TemplateResponse(
            request=request,
            name="experiment.html",
            context={
                "request": request,
                "base_path": _get_base_path(request),
                "active_page": "dashboard",
                "version": "0.3.2",
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

        # Update status
        exp_status.best_model = model_name
        exp_status.mode = "production"

        # Mark all models as not production, then mark selected
        if benchmark_result:
            for model in benchmark_result.models:
                model.is_production = model.name == model_name

        return JSONResponse(
            content={
                "message": f"Model {model_name} promoted to production",
                "experiment": name,
                "model": model_name,
            }
        )

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

        return JSONResponse(
            content={
                "message": f"Switched {name} to {new_mode} mode",
                "experiment": name,
                "mode": new_mode,
            }
        )

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
            version="0.2.0",
            timestamp=datetime.utcnow().isoformat(),
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
                "version": "0.3.2",
                "log_content": "\n".join(tail),
            },
        )

    @app.get("/status", response_class=Response)
    async def status_page(request: Request):
        """
        System status page with styled template.
        """
        experiments = list(app.state.appstate.experiment_statuses.values())
        lab_count = sum(1 for e in experiments if e.mode == "lab")
        prod_count = sum(1 for e in experiments if e.mode == "production")

        health = {
            "status": "healthy",
            "version": "0.3.2",
            "experiments_total": len(experiments),
            "experiments_lab": lab_count,
            "experiments_production": prod_count,
        }

        models_list = [
            {"name": "lightgbm", "display_name": "LightGBM", "model_type": "Tree",
             "description": "Gradient boosting framework optimised for speed and memory efficiency. Builds trees leaf-wise for faster convergence.",
             "speed": "⚡ Very Fast (~0.5s/fold)", "hardware_accel": "No (CPU only)", "best_for": "Default choice — fast and accurate"},
            {"name": "xgboost", "display_name": "XGBoost", "model_type": "Tree",
             "description": "Extreme gradient boosting with L1/L2 regularisation. Builds trees level-wise with robust handling of missing values.",
             "speed": "⚡ Fast (~1s/fold)", "hardware_accel": "No (CPU only)", "best_for": "When LightGBM overfits"},
            {"name": "lstm", "display_name": "LSTM", "model_type": "Neural",
             "description": "Long short-term memory network with gated cells for learning temporal dependencies. Pure NumPy implementation.",
             "speed": "🐢 Slow (~30s/fold)", "hardware_accel": "Yes (Hailo NPU)", "best_for": "Complex temporal patterns"},
            {"name": "cnn", "display_name": "CNN", "model_type": "Neural",
             "description": "1D convolutional network that detects local patterns in sequences using sliding filters. Pure NumPy implementation.",
             "speed": "🐢 Moderate (~6s/fold)", "hardware_accel": "Yes (Hailo NPU)", "best_for": "Periodic/seasonal signals"},
        ]

        return templates.TemplateResponse(
            request=request,
            name="status.html",
            context={
                "request": request,
                "base_path": _get_base_path(request),
                "active_page": "status",
                "version": "0.3.2",
                "health": health,
                "experiments": experiments,
                "models": models_list,
            },
        )

    @app.get("/settings", response_class=Response)
    async def settings_page(request: Request):
        """
        Settings page with system info, resource limits, and experiment config.
        """
        # Gather system information
        import psutil
        cpu_count = os.cpu_count() or 4
        try:
            cpu_model = platform.processor() or platform.machine()
        except Exception:
            cpu_model = platform.machine()

        try:
            mem = psutil.virtual_memory()
            memory_total_gb = round(mem.total / (1024**3), 1)
            memory_used_gb = round(mem.used / (1024**3), 1)
            memory_percent = mem.percent
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

        # Get current config from app state
        config_data = {
            "update_every_minutes": 360,
            "timezone": "UTC",
            "hailo_enabled": False,
            "cpu_cores": 0,
            "nice_priority": 10,
        }

        # Try to read from the main app's config
        config_path = "unknown"
        for p in [
            Path("/addon_configs/ml_forecast_lab/mlfl.yaml"),
            Path("/config/mlfl.yaml"),
        ]:
            if p.exists():
                config_path = str(p)
                break
        # Also check hashed paths
        import glob
        for match in glob.glob("/addon_configs/*_ml_forecast_lab/mlfl.yaml"):
            config_path = match
            break

        try:
            from ml_forecast_lab.config import load_config as _load_config
            cfg = _load_config(config_path)
            config_data = {
                "update_every_minutes": cfg.update_every_minutes,
                "timezone": cfg.timezone,
                "hailo_enabled": cfg.hailo_enabled,
                "cpu_cores": cfg.cpu_cores,
                "nice_priority": cfg.nice_priority,
            }
            experiments = cfg.experiments
        except Exception:
            experiments = []

        return templates.TemplateResponse(
            request=request,
            name="settings.html",
            context={
                "request": request,
                "base_path": _get_base_path(request),
                "active_page": "settings",
                "version": "0.3.7",
                "system": system_info,
                "config": config_data,
                "config_path": config_path,
                "experiments": experiments,
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
            if "update_every_minutes" in data:
                yaml_data["update_every_minutes"] = int(data["update_every_minutes"])
            if "timezone" in data:
                yaml_data["timezone"] = str(data["timezone"])
            if "hailo_enabled" in data:
                yaml_data["hailo_enabled"] = bool(data["hailo_enabled"])
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

    return app
