"""
FastAPI web application for ML Forecast Lab.

Provides dashboard and API endpoints for monitoring and managing forecasting
experiments, model benchmarking, and production deployment.
"""

import json
import logging
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
        version="0.1.0",
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
            "dashboard.html",
            {
                "request": request,
                "base_path": _get_base_path(request),
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
        is_running = app.state.appstate.is_benchmark_running(name)

        return templates.TemplateResponse(
            "experiment.html",
            {
                "request": request,
                "base_path": _get_base_path(request),
                "experiment": exp_status,
                "benchmark_result": benchmark_result,
                "forecast_data": forecast_data,
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

        # Mark all models as not production, then mark selected one as production
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
            version="0.1.0",
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
        View recent log output in the browser.
        Returns the last N lines of the log file as plain text.
        """
        log_text = ""
        # Read current log + first rotated backup
        for log_path in [LOG_FILE.with_suffix(".log.1"), LOG_FILE]:
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        log_text += f.read()
                except Exception as e:
                    log_text += f"\n[Error reading {log_path}: {e}]\n"

        # Return last N lines
        all_lines = log_text.splitlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return Response(
            content="\n".join(tail),
            media_type="text/plain; charset=utf-8",
        )

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
