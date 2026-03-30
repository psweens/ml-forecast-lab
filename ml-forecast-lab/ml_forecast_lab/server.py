"""
FastAPI server for ML Forecast Lab.

Provides HTTP API endpoints for model management, forecasting,
and system health monitoring.
"""

import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uvicorn import Config, Server

logger = logging.getLogger(__name__)


# Request/Response models
class PredictionRequest(BaseModel):
    """Request model for making predictions."""

    model_name: str
    data: list


class PredictionResponse(BaseModel):
    """Response model for prediction results."""

    model_name: str
    predictions: list
    confidence: Optional[float] = None


class HealthResponse(BaseModel):
    """Response model for health checks."""

    status: str
    service: str
    version: str


class ModelInfo(BaseModel):
    """Information about a registered model."""

    name: str
    model_type: Optional[str] = None
    status: str


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="ML Forecast Lab API",
        description="Multi-model ML forecasting and benchmarking API for Home Assistant",
        version="0.2.0",
    )

    # State to hold the forecasting engine
    app.state.engine = None

    @app.on_event("startup")
    async def startup_event():
        """Initialise application on startup."""
        logger.info("ML Forecast Lab API starting up...")
        # Import here to avoid circular imports
        from ml_forecast_lab.core import ForecastingEngine

        app.state.engine = ForecastingEngine()

    @app.get("/", tags=["info"])
    async def root() -> Dict[str, Any]:
        """Root endpoint providing API information."""
        return {
            "name": "ML Forecast Lab",
            "version": "0.2.0",
            "description": "Multi-model ML forecasting and benchmarking system",
            "endpoints": {
                "health": "/health",
                "models": "/api/models",
                "predict": "/api/predict",
            },
        }

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            service="ml-forecast-lab",
            version="0.2.0",
        )

    @app.get("/api/models", tags=["models"])
    async def list_models() -> Dict[str, Any]:
        """List all registered models."""
        if app.state.engine is None:
            raise HTTPException(status_code=500, detail="Engine not initialised")

        models = [
            ModelInfo(name=name, status="ready")
            for name in app.state.engine.models.keys()
        ]

        return {
            "count": len(models),
            "models": models,
        }

    @app.post("/api/predict", response_model=PredictionResponse, tags=["predict"])
    async def predict(request: PredictionRequest) -> PredictionResponse:
        """
        Generate a forecast using the specified model.

        Args:
            request: Prediction request containing model name and input data

        Returns:
            Prediction response with forecasted values
        """
        if app.state.engine is None:
            raise HTTPException(status_code=500, detail="Engine not initialised")

        try:
            model = app.state.engine.models.get(request.model_name)
            if not model:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model not found: {request.model_name}",
                )

            predictions = model.predict([request.data])

            return PredictionResponse(
                model_name=request.model_name,
                predictions=predictions.tolist(),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    @app.get("/api/status", tags=["status"])
    async def get_status() -> Dict[str, Any]:
        """Get current system status and statistics."""
        if app.state.engine is None:
            raise HTTPException(status_code=500, detail="Engine not initialised")

        summary = app.state.engine.get_summary()

        return {
            "status": "operational",
            "engine": summary,
        }

    return app


async def run_server():
    """Start the ML Forecast Lab API server."""
    app = create_app()

    port = int(os.getenv("PORT", 5052))
    host = os.getenv("HOST", "0.0.0.0")
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    logger.info(f"Starting ML Forecast Lab server on {host}:{port}")

    config = Config(
        app=app,
        host=host,
        port=port,
        log_level=log_level,
    )
    server = Server(config)

    await server.serve()


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_server())
