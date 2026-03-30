"""
ML Forecast Lab entry point.

This module serves as the main entry point for the ML Forecast Lab application.
It initialises the application environment and starts the forecasting service.
"""

import asyncio
import logging
import logging.handlers
import os
import sys
from pathlib import Path

# Configure logging — console + rotating file
LOG_DIR = Path("/data/ml_forecast_lab/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "mlfl.log"
LOG_FORMAT = "%(asctime)s %(levelname)-5s %(message)s"
LOG_FORMAT_FILE = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Root logger setup
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)

# Console handler — short format for HA addon logs
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
root_logger.addHandler(console_handler)

# Rotating file handler — detailed format with module name for debugging
file_handler = logging.handlers.RotatingFileHandler(
    str(LOG_FILE),
    maxBytes=5 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT_FILE, datefmt="%Y-%m-%d %H:%M:%S"))
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)




async def main():
    """Main application entry point."""
    logger.info("Starting ML Forecast Lab...")

    # Import here to allow proper logging setup
    try:
        from ml_forecast_lab.main import MLForecastLabApp

        # Create and run the main application
        app = MLForecastLabApp()
        await app.run()

    except ImportError as e:
        logger.error(f"Failed to import main application: {e}")
        logger.info("Running in stub mode with basic HTTP server...")
        await stub_server()


async def stub_server():
    """
    Stub HTTP server for initial development and testing.
    Provides a basic endpoint for health checks.
    """
    from fastapi import FastAPI
    from uvicorn import Config, Server

    app = FastAPI(
        title="ML Forecast Lab",
        description="Multi-model ML forecasting system",
        version="0.2.0",
    )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "ml-forecast-lab",
            "version": "0.2.0",
        }

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "ML Forecast Lab",
            "version": "0.2.0",
            "description": "Multi-model ML forecasting and benchmarking system",
            "endpoints": {
                "health": "/health",
                "api": "/api",
            },
        }

    config = Config(
        app=app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5052)),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
    server = Server(config)

    logger.info("Starting ML Forecast Lab server on 0.0.0.0:5052...")
    await server.serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received shutdown signal, terminating...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
