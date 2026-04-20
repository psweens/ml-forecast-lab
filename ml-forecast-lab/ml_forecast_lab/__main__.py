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


def _parse_log_level(raw):
    """
    Parse a LOG_LEVEL env var into a Python logging level.

    The Home Assistant add-on base image (hassio-addons/ubuntu-base, via
    bashio + s6-overlay) exports LOG_LEVEL as a *bashio* level which can be:

    * a numeric string ("0".."8") corresponding to bashio's own levels
      (TRACE=8, DEBUG=7, INFO=6, NOTICE=5, WARNING=4, ERROR=3, FATAL=2, OFF=0),
    * a bashio level name (TRACE / NOTICE / FATAL) that Python's `logging`
      module doesn't natively understand,
    * a standard Python level name (DEBUG / INFO / WARNING / ERROR / CRITICAL),
    * or sometimes a value with stray whitespace / newlines from however the
      env var was written into /var/run/s6/container_environment/.

    Python's `logging.setLevel` only accepts the standard names or actual
    integer levels — passing it `'5\\n'` raises ValueError. This helper
    normalises everything into a Python logging int and falls back to INFO
    on any unrecognised input rather than crashing the add-on at startup.
    """
    if not raw:
        return logging.INFO
    s = str(raw).strip()
    if not s:
        return logging.INFO

    # Bashio numeric levels (most common case from the add-on supervisor)
    bashio_numeric = {
        "8": logging.DEBUG,     # TRACE → DEBUG (Python has no TRACE)
        "7": logging.DEBUG,
        "6": logging.INFO,
        "5": logging.INFO,      # NOTICE → INFO
        "4": logging.WARNING,
        "3": logging.ERROR,
        "2": logging.CRITICAL,  # FATAL → CRITICAL
        "1": logging.CRITICAL,
        "0": logging.CRITICAL,  # OFF → CRITICAL (effectively silent)
    }
    if s in bashio_numeric:
        return bashio_numeric[s]

    # Plain integer (Python's own level numbers, e.g. "10", "20", "30")
    try:
        return int(s)
    except ValueError:
        pass

    # Bashio string level names that Python doesn't recognise
    upper = s.upper()
    bashio_string = {
        "TRACE": logging.DEBUG,
        "NOTICE": logging.INFO,
        "FATAL": logging.CRITICAL,
        "OFF": logging.CRITICAL,
    }
    if upper in bashio_string:
        return bashio_string[upper]

    # Standard Python level names
    if upper in {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"}:
        return upper

    # Anything else — fall back to INFO rather than crash
    return logging.INFO


# Configure logging — console + rotating file
LOG_DIR = Path("/data/ml_forecast_lab/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "mlfl.log"
LOG_FORMAT = "%(asctime)s %(levelname)-5s %(phase)-7s %(message)s"
LOG_FORMAT_FILE = "%(asctime)s %(levelname)-5s %(phase)-7s [%(name)s] %(message)s"
LOG_LEVEL = _parse_log_level(os.getenv("LOG_LEVEL"))


# Phase tags derived from module name so every log line is greppable by
# subsystem (e.g. `grep '\[BENCH\]' mlfl.log`). Keep tags <=5 chars so the
# padded column stays at 7 including brackets.
_PHASE_PREFIX_MAP = (
    ("ml_forecast_lab.benchmark", "BENCH"),
    ("ml_forecast_lab.models",    "MODEL"),
    ("ml_forecast_lab.web",       "WEB"),
    ("ml_forecast_lab.ha_interface", "HA"),
    ("ml_forecast_lab.preprocessing", "PREP"),
    ("ml_forecast_lab.features",  "FEAT"),
    ("ml_forecast_lab.covariates", "COV"),
    ("ml_forecast_lab.publishing", "PUB"),
    ("ml_forecast_lab.solar_physics", "SOLAR"),
    ("ml_forecast_lab.training_events", "TRAIN"),
    ("ml_forecast_lab.dashboard", "DASH"),
    ("ml_forecast_lab.config",    "CFG"),
    ("ml_forecast_lab.db",        "DB"),
    ("ml_forecast_lab.main",      "APP"),
)


class _PhaseFormatter(logging.Formatter):
    """Formatter that injects a short [PHASE] tag derived from the logger name."""

    def format(self, record):
        record.phase = _phase_for(record.name)
        return super().format(record)


def _phase_for(name: str) -> str:
    for prefix, tag in _PHASE_PREFIX_MAP:
        if name == prefix or name.startswith(prefix + "."):
            return f"[{tag}]"
    return "[MLFL]"


# Root logger setup
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)

# Console handler — short format for HA addon logs
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(_PhaseFormatter(LOG_FORMAT, datefmt="%H:%M:%S"))
root_logger.addHandler(console_handler)

# Rotating file handler — detailed format with module name for debugging
file_handler = logging.handlers.RotatingFileHandler(
    str(LOG_FILE),
    maxBytes=5 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
file_handler.setFormatter(_PhaseFormatter(LOG_FORMAT_FILE, datefmt="%Y-%m-%d %H:%M:%S"))
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
        logger.error(f"Failed to import main application: {e}", exc_info=True)
        logger.info("Running in stub mode with basic HTTP server...")
        await stub_server()


async def stub_server():
    """
    Stub HTTP server for initial development and testing.
    Provides a basic endpoint for health checks.
    """
    from fastapi import FastAPI
    from uvicorn import Config, Server

    from ml_forecast_lab import __version__ as APP_VERSION

    app = FastAPI(
        title="ML Forecast Lab",
        description="Multi-model ML forecasting system",
        version=APP_VERSION,
    )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "ml-forecast-lab",
            "version": APP_VERSION,
        }

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "ML Forecast Lab",
            "version": APP_VERSION,
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
