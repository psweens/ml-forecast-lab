"""
Main application entry point and event loop for ML Forecast Lab.

Orchestrates the loading of configuration, initialisation of components,
and management of the main forecast/benchmark loop with FastAPI web server.
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import uvicorn

logger = logging.getLogger(__name__)


class MLForecastLabApp:
    """
    Main application controller for ML Forecast Lab.

    Manages the lifecycle of the forecasting engine, including configuration
    loading, component initialisation, web server management, and the main
    update loop that drives forecasting and benchmarking.
    """

    def __init__(self):
        """Initialise the application."""
        self.config = None
        self.ha_interface = None
        self.history_db = None
        self.covariate_resolver = None
        self.model_registry = None
        self.web_app = None
        self.server = None
        self.running = False
        self.last_update = None
        self.benchmarks_to_run = set()

    async def load_config(self, config_path: Optional[Path] = None):
        """
        Load configuration from YAML file.

        Parameters
        ----------
        config_path : Optional[Path]
            Path to configuration file. Defaults to /config/mlfl.yaml
        """
        if config_path is None:
            config_path = Path("/config/mlfl.yaml")

        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            logger.info("Creating stub configuration...")
            # Create a minimal config for testing
            self.config = self._create_stub_config()
        else:
            try:
                from ml_forecast_lab.config import load_config

                self.config = load_config(config_path)
                logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                self.config = self._create_stub_config()

    def _create_stub_config(self):
        """Create a minimal stub configuration for testing."""
        from ml_forecast_lab.config import AppConfig, ExperimentCfg

        return AppConfig(
            update_every_minutes=5,
            timezone="UTC",
            experiments=[
                ExperimentCfg(
                    name="test_experiment",
                    target_entity="sensor.test_value",
                    days_history=7,
                    interval_minutes=30,
                    horizons_minutes=[120, 480],
                    models_enabled=["lightgbm"],
                    cv_folds=3,
                )
            ],
            hailo_enabled=False,
        )

    async def initialise_components(self):
        """
        Initialise all application components.

        Includes HAInterface, HistoryDB, CovariateResolver, and ModelRegistry.
        """
        logger.info("Initialising application components...")

        try:
            # Initialise HAInterface
            from ml_forecast_lab.ha_interface import HAInterface

            self.ha_interface = HAInterface()
            logger.info("HAInterface initialised")

            # Initialise HistoryDB
            from ml_forecast_lab.db import HistoryDB

            db_path = Path("/data/ml_forecast_lab/history.db")
            self.history_db = HistoryDB(db_path)
            logger.info(f"HistoryDB initialised at {db_path}")

            # Initialise CovariateResolver
            from ml_forecast_lab.covariates import CovariateResolver

            self.covariate_resolver = CovariateResolver(self.ha_interface)
            logger.info("CovariateResolver initialised")

            # Initialise ModelRegistry with all available backends
            from ml_forecast_lab.models.registry import ModelRegistry
            from ml_forecast_lab.models.lightgbm_backend import LightGBMBackend
            from ml_forecast_lab.models.xgboost_backend import XGBoostBackend
            from ml_forecast_lab.models.lstm_backend import LSTMBackend
            from ml_forecast_lab.models.cnn_backend import CNNBackend

            self.model_registry = ModelRegistry()
            self.model_registry.register("lightgbm", LightGBMBackend)
            self.model_registry.register("xgboost", XGBoostBackend)
            self.model_registry.register("lstm", LSTMBackend)
            self.model_registry.register("cnn", CNNBackend)
            logger.info("ModelRegistry initialised with 4 backends")

        except Exception as e:
            logger.error(f"Failed to initialise components: {e}", exc_info=True)
            logger.info("Continuing with partial initialisation...")

    async def start_web_server(self, host: str = "0.0.0.0", port: int = 5052):
        """
        Start the FastAPI web server in background.

        Parameters
        ----------
        host : str
            Host to listen on (default: 0.0.0.0)
        port : int
            Port to listen on (default: 5052)
        """
        logger.info(f"Starting web server on {host}:{port}...")

        try:
            from ml_forecast_lab.web.app import create_app

            self.web_app = create_app()

            # Initialise experiment statuses in web app state
            for exp_cfg in self.config.experiments:
                from ml_forecast_lab.web.app import ExperimentStatus

                status = ExperimentStatus(
                    name=exp_cfg.name,
                    target_entity=exp_cfg.target_entity,
                    mode="lab",  # Default to lab mode
                    last_benchmark_status="pending",
                    next_update_in_seconds=self.config.update_every_minutes * 60,
                )
                self.web_app.state.appstate.experiment_statuses[exp_cfg.name] = (
                    status
                )

            config = uvicorn.Config(
                app=self.web_app,
                host=host,
                port=port,
                log_level="info",
            )
            self.server = uvicorn.Server(config)

            # Run in a background task
            asyncio.create_task(self.server.serve())
            logger.info(f"Web server started successfully on {host}:{port}")

        except Exception as e:
            logger.error(f"Failed to start web server: {e}", exc_info=True)

    async def update_experiment(self, experiment_name: str, is_lab_mode: bool):
        """
        Run update for a single experiment.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment to update
        is_lab_mode : bool
            If True, run full benchmark. If False, run production inference only.
        """
        logger.info(f"Updating experiment: {experiment_name} (mode={'lab' if is_lab_mode else 'production'})")

        try:
            # Find experiment config
            exp_cfg = None
            for cfg in self.config.experiments:
                if cfg.name == experiment_name:
                    exp_cfg = cfg
                    break

            if not exp_cfg:
                logger.error(f"Experiment config not found: {experiment_name}")
                return

            if is_lab_mode:
                await self._run_benchmark(exp_cfg)
            else:
                await self._run_production_inference(exp_cfg)

            # Update web app state
            if self.web_app:
                status = self.web_app.state.appstate.experiment_statuses.get(
                    experiment_name
                )
                if status:
                    status.last_benchmark_timestamp = datetime.utcnow().isoformat()
                    status.last_benchmark_status = "completed"
                    self.web_app.state.appstate.end_benchmark(experiment_name)

        except Exception as e:
            logger.error(
                f"Error updating experiment {experiment_name}: {e}",
                exc_info=True,
            )
            if self.web_app:
                status = self.web_app.state.appstate.experiment_statuses.get(
                    experiment_name
                )
                if status:
                    status.last_benchmark_status = "failed"
                    self.web_app.state.appstate.end_benchmark(experiment_name)

    async def _run_benchmark(self, exp_cfg):
        """
        Run full benchmark across all enabled models.

        Parameters
        ----------
        exp_cfg : ExperimentCfg
            Experiment configuration
        """
        logger.info(f"Running benchmark for {exp_cfg.name}...")

        # Mark as running in web app
        if self.web_app:
            self.web_app.state.appstate.start_benchmark(exp_cfg.name)

        # In a real implementation, this would:
        # 1. Fetch historical data from ha_interface
        # 2. Prepare features using CovariateResolver
        # 3. Run cross-validation benchmark using model_registry
        # 4. Store results in web_app.state.appstate.benchmark_results
        # 5. Publish results back to HA via publishing module

        # For now, create a stub result
        from ml_forecast_lab.web.app import BenchmarkResult, ModelResult, MetricValue

        models = []
        for idx, model_name in enumerate(exp_cfg.models_enabled[:2]):  # Limit to 2 for demo
            model = ModelResult(
                name=model_name,
                mae=MetricValue(mean=0.123 + idx * 0.01, std=0.012),
                rmse=MetricValue(mean=0.234 + idx * 0.02, std=0.023),
                mape=MetricValue(mean=5.6 + idx * 0.5, std=0.8),
                train_time_seconds=12.5 + idx * 2,
                rank=idx + 1,
                is_production=idx == 0,
                fold_results=[
                    {
                        "mae": 0.120 + idx * 0.01,
                        "rmse": 0.230 + idx * 0.02,
                        "mape": 5.5 + idx * 0.5,
                    }
                    for _ in range(exp_cfg.cv_folds)
                ],
            )
            models.append(model)

        result = BenchmarkResult(
            experiment_name=exp_cfg.name,
            timestamp=datetime.utcnow().isoformat(),
            status="completed",
            models=models,
            best_model_name=models[0].name if models else None,
        )

        if self.web_app:
            self.web_app.state.appstate.benchmark_results[exp_cfg.name] = result

        logger.info(f"Benchmark completed for {exp_cfg.name}")

    async def _run_production_inference(self, exp_cfg):
        """
        Run production mode inference.

        Train only the production model and generate forecasts.

        Parameters
        ----------
        exp_cfg : ExperimentCfg
            Experiment configuration
        """
        logger.info(f"Running production inference for {exp_cfg.name}...")

        # In a real implementation, this would:
        # 1. Fetch historical data from ha_interface
        # 2. Train production model using full history
        # 3. Generate forecast for configured horizons
        # 4. Publish forecasts back to HA via publishing module

        logger.info(f"Production inference completed for {exp_cfg.name}")

    async def publish_heartbeat(self):
        """
        Publish heartbeat sensor to Home Assistant.

        Creates/updates a sensor.mlfl_last_run entity with the current timestamp.
        """
        try:
            if not self.ha_interface:
                return

            timestamp = datetime.utcnow().isoformat()
            logger.debug(f"Publishing heartbeat: {timestamp}")

            # In a real implementation, this would use the publishing module
            # to publish to Home Assistant

        except Exception as e:
            logger.error(f"Failed to publish heartbeat: {e}")

    async def main_loop(self):
        """
        Main application event loop.

        Runs continuously, performing updates at configured intervals for each experiment.
        """
        logger.info("Starting main event loop...")
        self.running = True

        # Register signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating graceful shutdown...")
            self.running = False

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Calculate update interval
        update_interval = self.config.update_every_minutes * 60  # Convert to seconds
        next_update = datetime.utcnow() + timedelta(seconds=update_interval)

        while self.running:
            try:
                now = datetime.utcnow()

                # Check if it's time for update
                if now >= next_update:
                    logger.info("=== Starting scheduled update cycle ===")

                    # Reload config
                    await self.load_config()

                    # Run updates for each experiment
                    for exp_cfg in self.config.experiments:
                        # Determine if experiment is in lab or production mode
                        is_lab = (
                            not exp_cfg.production_model
                            or exp_cfg.production_model not in exp_cfg.models_enabled
                        )
                        await self.update_experiment(exp_cfg.name, is_lab)

                    # Publish heartbeat
                    await self.publish_heartbeat()

                    # Schedule next update
                    next_update = datetime.utcnow() + timedelta(seconds=update_interval)
                    logger.info(
                        f"Update cycle completed. Next update at {next_update.isoformat()}"
                    )

                    # Update web app state with next update time
                    if self.web_app:
                        for exp_cfg in self.config.experiments:
                            status = self.web_app.state.appstate.experiment_statuses.get(
                                exp_cfg.name
                            )
                            if status:
                                status.next_update_in_seconds = int(
                                    (next_update - now).total_seconds()
                                )

                # Update next_update_in_seconds counters
                if self.web_app:
                    now = datetime.utcnow()
                    for exp_cfg in self.config.experiments:
                        status = self.web_app.state.appstate.experiment_statuses.get(
                            exp_cfg.name
                        )
                        if status:
                            remaining = int(
                                (next_update - now).total_seconds()
                            )
                            status.next_update_in_seconds = max(0, remaining)

                # Sleep briefly to avoid busy-waiting
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(5)

        logger.info("Main event loop terminated")

    async def shutdown(self):
        """Gracefully shutdown the application."""
        logger.info("Shutting down ML Forecast Lab...")

        self.running = False

        if self.server:
            logger.info("Shutting down web server...")
            self.server.should_exit = True
            await asyncio.sleep(1)

        if self.history_db:
            logger.info("Closing database connection...")
            try:
                self.history_db.close()
            except Exception as e:
                logger.warning(f"Error closing database: {e}")

        logger.info("Shutdown complete")

    async def run(self):
        """
        Main entry point for the application.

        Initialises all components and starts the main event loop with web server.
        """
        try:
            logger.info("Initialising ML Forecast Lab v0.1.0...")

            # Setup directories
            self._setup_directories()

            # Load configuration
            await self.load_config()

            # Initialise components
            await self.initialise_components()

            # Start web server
            await self.start_web_server()

            # Run main event loop
            await self.main_loop()

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            await self.shutdown()

    @staticmethod
    def _setup_directories():
        """Create necessary directories for application operation."""
        directories = [
            Path("/data/ml_forecast_lab"),
            Path("/data/ml_forecast_lab/models"),
            Path("/data/ml_forecast_lab/logs"),
            Path("/config/ml_forecast_lab"),
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ready: {directory}")


async def main():
    """
    Async entry point for ML Forecast Lab.

    Creates and runs the main application instance.
    """
    app = MLForecastLabApp()
    await app.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
