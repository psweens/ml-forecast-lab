"""
Main application entry point and event loop for ML Forecast Lab.

Orchestrates the loading of configuration, initialisation of components,
and management of the main forecast/benchmark loop with FastAPI web server.
"""

import asyncio
import dataclasses
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
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

        Searches multiple paths in order:
        1. Explicit config_path (if provided)
        2. /addon_configs/ml_forecast_lab/mlfl.yaml
        3. /config/mlfl.yaml
        4. Bundled mlfl.yaml (for development)
        5. Falls back to stub config
        """
        import glob

        search_paths = []
        if config_path is not None:
            search_paths.append(Path(config_path))
        search_paths.extend([
            Path("/addon_configs/ml_forecast_lab/mlfl.yaml"),
            Path("/config/mlfl.yaml"),
            Path(__file__).parent.parent / "mlfl.yaml",
        ])

        # Also check HA's hashed slug paths (e.g. /addon_configs/47b4bbf0_ml_forecast_lab/)
        for match in glob.glob("/addon_configs/*_ml_forecast_lab/mlfl.yaml"):
            search_paths.insert(0, Path(match))

        found_path = None
        for p in search_paths:
            if p.exists():
                found_path = p
                break

        if found_path is None:
            logger.warning(
                f"Configuration file not found in any of: "
                f"{[str(p) for p in search_paths]}"
            )
            logger.info("Creating stub configuration...")
            self.config = self._create_stub_config()
        else:
            try:
                from ml_forecast_lab.config import load_config

                self.config = load_config(found_path)
                logger.info(f"Configuration loaded from {found_path}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}", exc_info=True)
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
            from ml_forecast_lab.models.lightgbm_backend import LightGBMModel
            from ml_forecast_lab.models.xgboost_backend import XGBoostModel
            from ml_forecast_lab.models.lstm_backend import LSTMModel
            from ml_forecast_lab.models.cnn_backend import CNNModel

            self.model_registry = ModelRegistry()
            self.model_registry.register("lightgbm", LightGBMModel)
            self.model_registry.register("xgboost", XGBoostModel)
            self.model_registry.register("lstm", LSTMModel)
            self.model_registry.register("cnn", CNNModel)
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
                    mode=exp_cfg.mode,
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

    async def _fetch_and_preprocess(self, exp_cfg) -> pd.DataFrame:
        """
        Fetch history and preprocess for an experiment.

        Returns DataFrame with DatetimeIndex and 'y' column containing the
        preprocessed target values, ready for feature engineering.
        """
        from ml_forecast_lab.ha_interface import normalise_history
        from ml_forecast_lab.preprocessing import (
            cumulative_to_interval,
            resample_to_grid,
            clip_outliers,
            apply_log_transform,
        )

        now = datetime.now(timezone.utc)
        start = now - timedelta(days=exp_cfg.days_history)
        freq = f"{exp_cfg.interval_minutes}min"
        table_name = self.history_db.safe_table_name(exp_cfg.target_entity) if self.history_db else None

        # --- Fetch from HA API ---
        raw_records = await self.ha_interface.get_history(
            exp_cfg.target_entity, start, now
        )
        df = normalise_history(raw_records)

        if df.empty:
            raise ValueError(
                f"No history data returned for {exp_cfg.target_entity}"
            )

        logger.info(
            f"Fetched {len(df)} records for {exp_cfg.target_entity}"
        )

        # --- Store in SQLite cache ---
        if exp_cfg.database and self.history_db and table_name:
            inserted = self.history_db.store_history(table_name, df)
            logger.debug(f"Cached {inserted} new records in SQLite")

        # --- Set DatetimeIndex ---
        df = df.set_index("ds").sort_index()
        series = df["value"]

        # --- Cumulative to interval ---
        if exp_cfg.source_is_cumulative:
            series = cumulative_to_interval(
                series,
                interval_minutes=exp_cfg.interval_minutes,
                reset_daily=exp_cfg.reset_daily,
                max_increment=exp_cfg.max_increment,
            )

        # --- Resample to regular grid ---
        series = resample_to_grid(series, freq=freq, method="mean")

        # --- Clip outliers ---
        series = clip_outliers(series, positive_only=exp_cfg.source_is_cumulative)

        # --- Optional log transform ---
        if exp_cfg.log_transform:
            series = apply_log_transform(series)

        # --- Build DataFrame ---
        result = pd.DataFrame({"y": series}, index=series.index)
        result = result.dropna()

        logger.info(
            f"Preprocessed {exp_cfg.target_entity}: "
            f"{len(result)} samples at {freq} intervals"
        )
        return result

    async def _run_benchmark(self, exp_cfg):
        """
        Run full benchmark across all enabled models using cross-validation.
        """
        from ml_forecast_lab.features import build_features
        from ml_forecast_lab.benchmark.runner import BenchmarkRunner
        from ml_forecast_lab.benchmark.metrics import get_metric_registry
        from ml_forecast_lab.web.app import (
            BenchmarkResult as WebBenchmarkResult,
            ModelResult as WebModelResult,
            MetricValue,
        )

        logger.info(f"Running benchmark for {exp_cfg.name}...")

        if self.web_app:
            self.web_app.state.appstate.start_benchmark(exp_cfg.name)

        # 1. Fetch and preprocess data
        df = await self._fetch_and_preprocess(exp_cfg)

        if len(df) < exp_cfg.cv_folds * 10:
            raise ValueError(
                f"Insufficient data for benchmark: {len(df)} samples "
                f"(need at least {exp_cfg.cv_folds * 10})"
            )

        # 2. Build features on full dataset
        features_df = build_features(
            df,
            target_col="y",
            interval_minutes=exp_cfg.interval_minutes,
            country=exp_cfg.country,
        )

        # 3. Combine features + target, drop NaN from lag warmup
        combined = features_df.copy()
        combined["target"] = df["y"]
        combined = combined.dropna()

        feature_cols = [c for c in combined.columns if c != "target"]
        logger.info(
            f"Feature matrix: {len(combined)} samples, "
            f"{len(feature_cols)} features"
        )

        # 4. Create feature_builder callback for BenchmarkRunner
        # BenchmarkRunner splits by index and passes df subsets here.
        # Features are already pre-built so we just extract numpy arrays.
        def feature_builder(df_sub, config, purpose="train"):
            cols = [c for c in df_sub.columns if c != "target"]
            X = df_sub[cols].values.astype(np.float32)
            # Replace any remaining NaN with 0 for model safety
            X = np.nan_to_num(X, nan=0.0)
            return X

        # 5. Instantiate models
        models = {}
        for model_name in exp_cfg.models_enabled:
            try:
                models[model_name] = self.model_registry.create(model_name)
            except Exception as e:
                logger.warning(
                    f"Skipping model {model_name}: {e}"
                )

        if not models:
            raise ValueError("No models could be created for benchmark")

        # 6. Run benchmark
        exp_cfg_dict = dataclasses.asdict(exp_cfg)
        metric_registry = get_metric_registry()
        runner = BenchmarkRunner(exp_cfg_dict, feature_builder, metric_registry)
        bench_result = runner.run_benchmark(combined, models)

        # 7. Generate holdout predictions from each model for visualisation
        #    Use last 20% of data as holdout, train on first 80%
        from ml_forecast_lab.web.app import (
            ModelPrediction,
            LabForecastData,
            FeatureImportanceData,
        )

        holdout_frac = 0.2
        split_idx = int(len(combined) * (1 - holdout_frac))
        train_part = combined.iloc[:split_idx]
        holdout_part = combined.iloc[split_idx:]

        X_train_hold = train_part[feature_cols].values.astype(np.float32)
        X_train_hold = np.nan_to_num(X_train_hold, nan=0.0)
        y_train_hold = train_part["target"].values.astype(np.float32)

        X_holdout = holdout_part[feature_cols].values.astype(np.float32)
        X_holdout = np.nan_to_num(X_holdout, nan=0.0)
        y_holdout = holdout_part["target"].values
        holdout_timestamps = [
            ts.isoformat() for ts in holdout_part.index
        ]

        MODEL_COLORS = {
            "lightgbm": "#2ecc71",
            "xgboost": "#3498db",
            "lstm": "#f39c12",
            "cnn": "#e74c3c",
        }

        model_predictions = []
        feature_importance_list = []

        for model_name in exp_cfg.models_enabled:
            try:
                # Create fresh model and train on 80% split
                model = self.model_registry.create(model_name)
                model.fit(X_train_hold, y_train_hold)

                # Predict on holdout
                y_pred = model.predict(X_holdout)
                if y_pred.ndim > 1:
                    y_pred = y_pred.ravel()

                model_predictions.append(ModelPrediction(
                    model_name=model_name,
                    timestamps=holdout_timestamps,
                    actuals=[float(v) if not np.isnan(v) else None for v in y_holdout],
                    predictions=[float(v) for v in y_pred],
                    color=MODEL_COLORS.get(model_name, "#00d4ff"),
                ))

                # Extract feature importances from tree models
                if hasattr(model, 'training_metadata') and model.training_metadata:
                    importances = model.training_metadata.get("feature_importances", {})
                    if importances:
                        sorted_features = sorted(
                            importances.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:20]
                        feature_importance_list.append(FeatureImportanceData(
                            model_name=model_name,
                            features=[
                                {"name": name, "importance": float(imp)}
                                for name, imp in sorted_features
                            ],
                        ))

                logger.info(f"Generated holdout predictions for {model_name}")

            except Exception as e:
                logger.warning(f"Failed to generate holdout predictions for {model_name}: {e}")

        # Store lab forecast data
        if model_predictions and self.web_app:
            lab_forecast = LabForecastData(
                experiment_name=exp_cfg.name,
                holdout_start=holdout_timestamps[0] if holdout_timestamps else "",
                holdout_end=holdout_timestamps[-1] if holdout_timestamps else "",
                model_predictions=model_predictions,
            )
            self.web_app.state.appstate.lab_forecast_data[exp_cfg.name] = lab_forecast
            logger.info(
                f"Stored lab predictions: {len(model_predictions)} models, "
                f"{len(holdout_timestamps)} holdout points"
            )

        if feature_importance_list and self.web_app:
            self.web_app.state.appstate.feature_importances[exp_cfg.name] = feature_importance_list

        # 8. Convert runner.BenchmarkResult -> web.app.BenchmarkResult
        web_models = []
        for rank_idx, (model_name, runner_model_result) in enumerate(
            sorted(
                bench_result.model_results.items(),
                key=lambda x: bench_result.rankings.get(x[0], 999),
            )
        ):
            rank = bench_result.rankings.get(model_name, rank_idx + 1)

            # Extract per-metric means and stds from fold_metrics
            fold_metrics_list = runner_model_result.fold_metrics
            metric_means = {}
            metric_stds = {}
            for metric_name in exp_cfg.metrics:
                values = [
                    fm.get(metric_name, np.nan)
                    for fm in fold_metrics_list
                    if fm
                ]
                metric_means[metric_name] = float(np.nanmean(values)) if values else 0.0
                metric_stds[metric_name] = float(np.nanstd(values)) if len(values) > 1 else 0.0

            web_model = WebModelResult(
                name=model_name,
                mae=MetricValue(
                    mean=metric_means.get("mae", 0.0),
                    std=metric_stds.get("mae", 0.0),
                ),
                rmse=MetricValue(
                    mean=metric_means.get("rmse", 0.0),
                    std=metric_stds.get("rmse", 0.0),
                ),
                mape=MetricValue(
                    mean=metric_means.get("mape", 0.0),
                    std=metric_stds.get("mape", 0.0),
                ),
                train_time_seconds=runner_model_result.mean_train_time,
                rank=rank,
                is_production=(model_name == bench_result.best_model),
                fold_results=[fm for fm in fold_metrics_list if fm],
            )
            web_models.append(web_model)

        web_result = WebBenchmarkResult(
            experiment_name=exp_cfg.name,
            timestamp=datetime.utcnow().isoformat(),
            status="completed",
            models=web_models,
            best_model_name=bench_result.best_model,
        )

        if self.web_app:
            self.web_app.state.appstate.benchmark_results[exp_cfg.name] = web_result

        logger.info(
            f"Benchmark completed for {exp_cfg.name}: "
            f"best={bench_result.best_model} "
            f"({exp_cfg.production_metric}={bench_result.best_metric_value:.4f})"
        )

    async def _run_production_inference(self, exp_cfg):
        """
        Run production mode: train best model on full data and publish forecasts.
        """
        from ml_forecast_lab.features import build_features, create_forecast_features
        from ml_forecast_lab.publishing import publish_forecasts

        logger.info(f"Running production inference for {exp_cfg.name}...")

        # 1. Fetch and preprocess
        df = await self._fetch_and_preprocess(exp_cfg)

        # 2. Build features
        features_df = build_features(
            df,
            target_col="y",
            interval_minutes=exp_cfg.interval_minutes,
            country=exp_cfg.country,
        )
        combined = features_df.copy()
        combined["target"] = df["y"]
        combined = combined.dropna()

        feature_cols = [c for c in combined.columns if c != "target"]

        X = combined[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)
        y = combined["target"].values.astype(np.float32)

        # 3. Determine production model
        prod_model_name = exp_cfg.production_model
        if not prod_model_name:
            # Try to get best model from last benchmark
            if self.web_app:
                bench = self.web_app.state.appstate.benchmark_results.get(
                    exp_cfg.name
                )
                if bench and bench.best_model_name:
                    prod_model_name = bench.best_model_name

        if not prod_model_name:
            prod_model_name = exp_cfg.models_enabled[0]
            logger.info(
                f"No production model set, defaulting to {prod_model_name}"
            )

        # 4. Create and train model on full data
        model = self.model_registry.create(prod_model_name)
        logger.info(f"Training {prod_model_name} on {len(X)} samples...")
        train_start = time.time()
        model.fit(X, y)
        train_time = time.time() - train_start
        logger.info(f"Training completed in {train_time:.1f}s")

        # 5. Create forecast features for future horizons
        n_lags = 12  # Must match build_features default
        last_ts = combined.index[-1]
        lag_values = y[-n_lags:]

        forecast_features = create_forecast_features(
            last_timestamp=last_ts,
            interval_minutes=exp_cfg.interval_minutes,
            horizons_minutes=exp_cfg.horizons_minutes,
            n_lags=n_lags,
            lag_values=lag_values,
            country=exp_cfg.country,
        )

        # Align forecast feature columns to training columns
        for col in feature_cols:
            if col not in forecast_features.columns:
                forecast_features[col] = 0.0
        forecast_features = forecast_features[feature_cols]

        X_forecast = forecast_features.values.astype(np.float32)
        X_forecast = np.nan_to_num(X_forecast, nan=0.0)

        # 6. Predict
        y_pred = model.predict(X_forecast)
        logger.info(
            f"Forecast generated: {len(y_pred)} points, "
            f"range [{y_pred.min():.2f}, {y_pred.max():.2f}]"
        )

        # 7. Build forecast DataFrame for publishing
        ds_future = forecast_features.index
        yhat_interval = pd.DataFrame({
            "ds": ds_future,
            "yhat": y_pred,
        })

        # 8. Publish to Home Assistant
        exp_cfg_dict = dataclasses.asdict(exp_cfg)
        success = await publish_forecasts(
            experiment_cfg=exp_cfg_dict,
            iface=self.ha_interface,
            app_config={},
            ds_future=ds_future,
            yhat_interval=yhat_interval,
            yhat_level=0.95,
        )

        if success:
            logger.info(f"Production forecasts published for {exp_cfg.name}")
        else:
            logger.warning(
                f"Some forecast publishes failed for {exp_cfg.name}"
            )

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

            await self.ha_interface.set_state(
                "sensor.mlfl_last_run",
                timestamp,
                attributes={
                    "friendly_name": "ML Forecast Lab Last Run",
                    "icon": "mdi:clock-check",
                    "experiments": len(self.config.experiments) if self.config else 0,
                },
            )

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

        # Calculate update interval — run first cycle immediately
        update_interval = self.config.update_every_minutes * 60  # Convert to seconds
        next_update = datetime.utcnow()

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
                        is_lab = exp_cfg.mode == "lab"
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
            logger.info("Initialising ML Forecast Lab v0.2.0...")

            # Setup directories
            self._setup_directories()

            # Load configuration
            await self.load_config()

            # Generate dashboard YAML
            self._generate_dashboard()

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

    def _generate_dashboard(self):
        """Generate ApexCharts dashboard YAML from current config."""
        if not self.config or not self.config.experiments:
            return

        try:
            from ml_forecast_lab.dashboard import generate_dashboard

            # Write to addon config dir (same location as mlfl.yaml)
            import glob
            config_dirs = glob.glob("/addon_configs/*_ml_forecast_lab")
            if config_dirs:
                output_path = Path(config_dirs[0]) / "mlfl_dashboard.yaml"
            else:
                output_path = Path("/addon_configs/ml_forecast_lab/mlfl_dashboard.yaml")

            generate_dashboard(self.config.experiments, output_path)
            logger.info(f"Dashboard YAML generated at {output_path}")
        except Exception as e:
            logger.warning(f"Failed to generate dashboard YAML: {e}")

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
