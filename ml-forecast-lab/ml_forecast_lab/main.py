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
        self._update_running = False
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

            # Register NeuralProphet if available
            try:
                from ml_forecast_lab.models.neuralprophet_backend import NeuralProphetModel
                self.model_registry.register("neuralprophet", NeuralProphetModel)
            except Exception as e:
                logger.debug(f"NeuralProphet not available: {e}")

            logger.info(f"ModelRegistry initialised with {len(self.model_registry.list_available())} backends")

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
                log_level="warning",
            )
            self.server = uvicorn.Server(config)

            # Register deep analysis callback
            async def _deep_analysis_trigger(experiment_name: str):
                exp_cfg = None
                for cfg in self.config.experiments:
                    if cfg.name == experiment_name:
                        exp_cfg = cfg
                        break
                if exp_cfg:
                    try:
                        await self._run_deep_analysis(exp_cfg)
                    except Exception as e:
                        logger.error(f"Deep analysis failed: {e}", exc_info=True)

            self.web_app.state.appstate.deep_analysis_callback = _deep_analysis_trigger

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

        df = pd.DataFrame(columns=["ds", "value"])

        # --- Try SQLite cache first ---
        if exp_cfg.database and self.history_db and table_name:
            cached_df = self.history_db.get_history(table_name)
            if not cached_df.empty:
                # Rename 'y' back to 'value' for consistency
                cached_df = cached_df.rename(columns={"y": "value"})
                # Ensure tz-naive for comparison (SQLite stores naive, start is tz-aware)
                start_naive = start.replace(tzinfo=None)
                cached_df = cached_df[cached_df["ds"] >= start_naive]
                if len(cached_df) > 0:
                    df = cached_df
                    logger.info(
                        f"  Loaded {len(df)} cached records for {exp_cfg.target_entity}"
                    )

        # --- Fetch delta from HA API ---
        if len(df) > 0:
            # Only fetch records newer than our latest cached record
            last_cached = df["ds"].max()
            # Ensure tz-aware for HA API
            if hasattr(last_cached, 'tzinfo') and last_cached.tzinfo is None:
                last_cached = last_cached.tz_localize("UTC")
            fetch_start = last_cached
        else:
            fetch_start = start

        raw_records = await self.ha_interface.get_history(
            exp_cfg.target_entity, fetch_start, now
        )
        new_df = normalise_history(raw_records)

        if not new_df.empty:
            if len(df) > 0:
                # Merge: append new records, deduplicate by timestamp
                df = pd.concat([df, new_df], ignore_index=True)
                df = df.drop_duplicates(subset=["ds"], keep="last").sort_values("ds").reset_index(drop=True)
                logger.info(
                    f"  Fetched {len(new_df)} new records from HA API "
                    f"(total: {len(df)})"
                )
            else:
                df = new_df
                logger.info(
                    f"  Fetched {len(df)} records for {exp_cfg.target_entity} "
                    f"({exp_cfg.days_history} days, {start.strftime('%d %b')} to {now.strftime('%d %b %H:%M')})"
                )

        if df.empty:
            raise ValueError(
                f"No history data for {exp_cfg.target_entity}"
            )

        # --- Store in SQLite cache ---
        if exp_cfg.database and self.history_db and table_name:
            inserted = self.history_db.store_history(table_name, df)
            if inserted > 0:
                logger.info(f"  Cached {inserted} new records in SQLite")

            # Cleanup old records beyond max_age
            oldest = now - timedelta(days=exp_cfg.max_age)
            self.history_db.cleanup(table_name, oldest)

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

        # --- Fetch covariates ---
        if exp_cfg.covariates and self.covariate_resolver:
            logger.info(f"  Fetching {len(exp_cfg.covariates)} covariate(s)...")
            for cov_cfg in exp_cfg.covariates:
                try:
                    # Build dict for CovariateResolver (expects entity_id, not entity)
                    cov_dict = {
                        "entity_id": cov_cfg.entity,
                        "name": cov_cfg.entity.split(".")[-1],  # sensor.current_charge → current_charge
                        "binary": cov_cfg.is_binary,
                    }

                    cov_series = await self.covariate_resolver.fetch_history(
                        cov_dict, start, now, freq
                    )

                    if cov_series.empty:
                        logger.warning(f"    No data for covariate {cov_cfg.entity}, skipping")
                        continue

                    # Apply scaling factor if configured
                    if cov_cfg.scale is not None:
                        cov_series = cov_series * cov_cfg.scale

                    # Apply transform if configured
                    if cov_cfg.transform is not None:
                        from ml_forecast_lab.preprocessing import apply_transform
                        cov_series = apply_transform(cov_series, cov_cfg.transform)

                    # Align to target index and merge
                    cov_name = cov_dict["name"]
                    cov_aligned = cov_series.reindex(result.index, method="ffill")
                    # Back-fill any leading NaNs and forward-fill trailing ones
                    cov_aligned = cov_aligned.ffill().bfill()
                    result[cov_name] = cov_aligned

                    valid_count = result[cov_name].notna().sum()
                    logger.info(
                        f"    ✓ {cov_cfg.entity} → '{cov_name}': "
                        f"{len(cov_series)} raw → {valid_count} aligned"
                        f"{f', scaled ×{cov_cfg.scale}' if cov_cfg.scale else ''}"
                    )

                except Exception as e:
                    logger.warning(f"    ✗ Failed to fetch {cov_cfg.entity}: {e}")

        result = result.dropna()

        # Rich data summary
        y = result["y"]
        logger.info(
            f"  Preprocessed: {len(result)} samples at {freq} intervals"
        )
        logger.info(
            f"  Data range: {result.index[0].strftime('%d %b %H:%M')} → "
            f"{result.index[-1].strftime('%d %b %H:%M')}"
        )
        logger.info(
            f"  Target stats: mean={y.mean():.3f}, std={y.std():.3f}, "
            f"min={y.min():.3f}, max={y.max():.3f}, zeros={int((y == 0).sum())}/{len(y)}"
        )
        if exp_cfg.source_is_cumulative:
            logger.info(
                f"  Cumulative→interval conversion: reset_daily={exp_cfg.reset_daily}, "
                f"max_increment={exp_cfg.max_increment}"
            )

        return result

    def _update_web_benchmark(self, exp_cfg, model_results, rankings, best_model_name, status="running"):
        """
        Update web app state with current benchmark progress.

        Called after each model completes so the UI updates progressively.
        """
        from ml_forecast_lab.web.app import (
            BenchmarkResult as WebBenchmarkResult,
            ModelResult as WebModelResult,
            MetricValue,
        )

        web_models = []
        for model_name, runner_model_result in sorted(
            model_results.items(),
            key=lambda x: rankings.get(x[0], 999),
        ):
            rank = rankings.get(model_name, 0)
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

            web_models.append(WebModelResult(
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
                is_production=(model_name == best_model_name),
                fold_results=[fm for fm in fold_metrics_list if fm],
            ))

        web_result = WebBenchmarkResult(
            experiment_name=exp_cfg.name,
            timestamp=datetime.utcnow().isoformat(),
            status=status,
            models=web_models,
            best_model_name=best_model_name,
        )

        self.web_app.state.appstate.benchmark_results[exp_cfg.name] = web_result

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

        logger.info(f"")
        logger.info(f"{'=' * 60}")
        logger.info(f"  BENCHMARK: {exp_cfg.name}")
        logger.info(f"  Target: {exp_cfg.target_entity}")
        logger.info(f"  Models: {', '.join(exp_cfg.models_enabled)}")
        logger.info(f"  Covariates: {len(exp_cfg.covariates)}" + (
            f" ({', '.join(c.entity.split('.')[-1] for c in exp_cfg.covariates)})" if exp_cfg.covariates else ""
        ))
        logger.info(f"  CV: {exp_cfg.cv_strategy}, {exp_cfg.cv_folds} folds, metric={exp_cfg.production_metric}")
        logger.info(f"{'=' * 60}")

        if self.web_app:
            self.web_app.state.appstate.start_benchmark(exp_cfg.name)

        # 1. Fetch and preprocess data
        df = await self._fetch_and_preprocess(exp_cfg)

        if len(df) < exp_cfg.cv_folds * 10:
            raise ValueError(
                f"Insufficient data for benchmark: {len(df)} samples "
                f"(need at least {exp_cfg.cv_folds * 10})"
            )

        # 2. Build temporal + lag features from target
        features_df = build_features(
            df,
            target_col="y",
            interval_minutes=exp_cfg.interval_minutes,
            country=exp_cfg.country,
        )

        # 3. Combine features + covariates + target, drop NaN from lag warmup
        combined = features_df.copy()
        combined["target"] = df["y"]

        # Add covariate columns from df (they were merged in _fetch_and_preprocess)
        covariate_cols = [c for c in df.columns if c != "y"]
        for col in covariate_cols:
            combined[col] = df[col]

        combined = combined.dropna()

        feature_cols = [c for c in combined.columns if c != "target"]
        n_cov = len(covariate_cols)
        logger.info(
            f"  Feature matrix: {len(combined)} samples, "
            f"{len(feature_cols)} features ({len(feature_cols) - n_cov} temporal + {n_cov} covariates)"
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

        # 6. Run benchmark model-by-model, updating web UI after each
        exp_cfg_dict = dataclasses.asdict(exp_cfg)
        metric_registry = get_metric_registry()
        runner = BenchmarkRunner(exp_cfg_dict, feature_builder, metric_registry)
        fold_indices = runner._prepare_train_test_splits(combined)

        completed_models = {}
        rankings = {}

        for model_idx, (model_name, model) in enumerate(models.items(), 1):
            logger.info(f"")
            logger.info(f"  [{model_idx}/{len(models)}] Benchmarking: {model_name}")

            model_result = await asyncio.get_event_loop().run_in_executor(
                None, runner.run_single_model, combined, model, fold_indices
            )
            completed_models[model_name] = model_result

            # Log model result summary
            mae_val = model_result.metrics.get("mae", np.nan)
            rmse_val = model_result.metrics.get("rmse", np.nan)
            logger.info(
                f"  ✓ {model_name}: MAE={mae_val:.4f}, RMSE={rmse_val:.4f}, "
                f"time={model_result.mean_train_time:.1f}s/fold"
            )

            # Rank completed models so far
            metric_values = {
                n: mr.metrics.get(runner.production_metric, np.inf)
                for n, mr in completed_models.items()
            }
            sorted_models = sorted(metric_values.items(), key=lambda x: x[1])
            rankings = {n: rank + 1 for rank, (n, _) in enumerate(sorted_models)}

            # Update web UI progressively
            if self.web_app:
                self._update_web_benchmark(
                    exp_cfg, completed_models, rankings,
                    sorted_models[0][0] if sorted_models else None,
                    status="running",
                )

        # Final ranking
        best_model_name = sorted_models[0][0] if sorted_models else None
        best_metric_value = sorted_models[0][1] if sorted_models else np.nan

        # Build a BenchmarkResult-compatible object for downstream use
        from ml_forecast_lab.benchmark.runner import BenchmarkResult as RunnerBenchmarkResult
        bench_result = RunnerBenchmarkResult(
            experiment_name=exp_cfg.name,
            model_results=completed_models,
            rankings=rankings,
            best_model=best_model_name or "",
            best_metric_value=best_metric_value,
            metric_used=runner.production_metric,
            cv_strategy=runner.cv_strategy,
            n_folds=runner.cv_folds,
        )

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
            "neuralprophet": "#9b59b6",
        }

        def _generate_holdout_predictions():
            """Run holdout predictions in thread pool to avoid blocking."""
            _model_predictions = []
            _feature_importance_list = []

            for m_name in exp_cfg.models_enabled:
                try:
                    m = self.model_registry.create(m_name)
                    m.fit(X_train_hold, y_train_hold, feature_names=feature_cols)

                    y_p = m.predict(X_holdout)
                    if y_p.ndim > 1:
                        y_p = y_p.ravel()

                    _model_predictions.append(ModelPrediction(
                        model_name=m_name,
                        timestamps=holdout_timestamps,
                        actuals=[float(v) if not np.isnan(v) else None for v in y_holdout],
                        predictions=[float(v) for v in y_p],
                        color=MODEL_COLORS.get(m_name, "#00d4ff"),
                    ))

                    if hasattr(m, 'training_metadata') and m.training_metadata:
                        importances = m.training_metadata.get("feature_importances", {})
                        if importances:
                            sorted_feats = sorted(
                                importances.items(), key=lambda x: x[1], reverse=True
                            )[:20]
                            _feature_importance_list.append(FeatureImportanceData(
                                model_name=m_name,
                                features=[
                                    {"name": name, "importance": float(imp)}
                                    for name, imp in sorted_feats
                                ],
                            ))

                    logger.info(f"Generated holdout predictions for {m_name}")

                except Exception as e:
                    logger.warning(f"Failed holdout predictions for {m_name}: {e}")

            return _model_predictions, _feature_importance_list

        model_predictions, feature_importance_list = await asyncio.get_event_loop().run_in_executor(
            None, _generate_holdout_predictions
        )

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

        # 8. Final web state update (mark as completed)
        if self.web_app:
            self._update_web_benchmark(
                exp_cfg, completed_models, rankings,
                best_model_name,
                status="completed",
            )

        # Final results summary table
        logger.info(f"")
        logger.info(f"  {'─' * 56}")
        logger.info(f"  {'Model':<12} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'Time':>8} {'Rank':>6}")
        logger.info(f"  {'─' * 56}")
        for m_name in sorted(completed_models.keys(), key=lambda x: rankings.get(x, 99)):
            mr = completed_models[m_name]
            rank = rankings.get(m_name, 0)
            marker = " ★" if m_name == best_model_name else ""
            logger.info(
                f"  {m_name:<12} "
                f"{mr.metrics.get('mae', np.nan):>8.4f} "
                f"{mr.metrics.get('rmse', np.nan):>8.4f} "
                f"{mr.metrics.get('mape', np.nan):>8.2f} "
                f"{mr.mean_train_time:>7.1f}s "
                f"{'#' + str(rank):>5}{marker}"
            )
        logger.info(f"  {'─' * 56}")
        logger.info(f"  Best model: {best_model_name} ({runner.production_metric}={best_metric_value:.4f})")
        logger.info(f"{'=' * 60}")
        logger.info(f"")

    async def _run_production_inference(self, exp_cfg):
        """
        Run production mode: train best model on full data, generate a full
        forecast curve, and publish results as HA sensor entities.
        """
        from ml_forecast_lab.features import build_features, create_forecast_features

        logger.info(f"")
        logger.info(f"{'=' * 60}")
        logger.info(f"  PRODUCTION: {exp_cfg.name}")
        logger.info(f"  Target: {exp_cfg.target_entity}")
        logger.info(f"{'=' * 60}")

        # 1. Fetch and preprocess
        df = await self._fetch_and_preprocess(exp_cfg)

        # 2. Build features + covariates
        features_df = build_features(
            df,
            target_col="y",
            interval_minutes=exp_cfg.interval_minutes,
            country=exp_cfg.country,
        )
        combined = features_df.copy()
        combined["target"] = df["y"]

        # Add covariate columns
        covariate_cols = [c for c in df.columns if c != "y"]
        for col in covariate_cols:
            combined[col] = df[col]

        combined = combined.dropna()

        feature_cols = [c for c in combined.columns if c != "target"]

        X = combined[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)
        y = combined["target"].values.astype(np.float32)

        # 3. Determine production model
        prod_model_name = exp_cfg.production_model
        if not prod_model_name:
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

        # 4. Train model on full data (in thread pool)
        model = self.model_registry.create(prod_model_name)
        logger.info(f"Training {prod_model_name} on {len(X)} samples...")
        train_start = time.time()
        await asyncio.get_event_loop().run_in_executor(
            None, model.fit, X, y
        )
        train_time = time.time() - train_start
        logger.info(f"Training completed in {train_time:.1f}s")

        # 5. Generate full forecast curve at regular intervals
        n_lags = 12
        last_ts = combined.index[-1]
        lag_values = y[-n_lags:]
        future_periods = getattr(exp_cfg, 'future_periods', 48)

        # Build future timestamps at regular intervals
        future_minutes = [
            exp_cfg.interval_minutes * (i + 1)
            for i in range(future_periods)
        ]

        forecast_features = create_forecast_features(
            last_timestamp=last_ts,
            interval_minutes=exp_cfg.interval_minutes,
            horizons_minutes=future_minutes,
            n_lags=n_lags,
            lag_values=lag_values,
            country=exp_cfg.country,
        )

        # Align columns to training features
        # For covariates, use last known value (forward-fill from training data)
        for col in feature_cols:
            if col not in forecast_features.columns:
                if col in covariate_cols and col in combined.columns:
                    # Use last known covariate value
                    forecast_features[col] = float(combined[col].iloc[-1])
                else:
                    forecast_features[col] = 0.0
        forecast_features = forecast_features[feature_cols]

        X_forecast = forecast_features.values.astype(np.float32)
        X_forecast = np.nan_to_num(X_forecast, nan=0.0)

        # 6. Predict full curve (in thread pool)
        def _predict():
            return model.predict(X_forecast)

        y_pred = await asyncio.get_event_loop().run_in_executor(
            None, _predict
        )
        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        logger.info(
            f"Forecast curve: {len(y_pred)} points over "
            f"{future_periods * exp_cfg.interval_minutes / 60:.0f}h, "
            f"range [{y_pred.min():.3f}, {y_pred.max():.3f}]"
        )

        ds_future = forecast_features.index
        publish_name = exp_cfg.publish_name or exp_cfg.name
        prefix = exp_cfg.publish_prefix
        units = exp_cfg.units or ""

        # 7. Publish main forecast sensor with full curve in attributes
        forecast_list = [
            {"datetime": ts.isoformat(), "value": round(float(val), 4)}
            for ts, val in zip(ds_future, y_pred)
        ]

        # Recent actuals for context (last 24h)
        recent_n = min(int(24 * 60 / exp_cfg.interval_minutes), len(combined))
        recent_actuals = [
            {"datetime": ts.isoformat(), "value": round(float(val), 4)}
            for ts, val in zip(
                combined.index[-recent_n:],
                combined["target"].values[-recent_n:],
            )
        ]

        base_entity = f"sensor.{prefix}{publish_name}"
        next_val = round(float(y_pred[0]), 4)

        await self.ha_interface.set_state(
            f"{base_entity}_forecast",
            str(next_val),
            attributes={
                "friendly_name": f"{publish_name} Forecast",
                "unit_of_measurement": units,
                "icon": "mdi:chart-timeline-variant-shimmer",
                "device_class": "power_factor" if units == "%" else None,
                "state_class": "measurement",
                "model": prod_model_name,
                "train_time_seconds": round(train_time, 1),
                "forecast_periods": future_periods,
                "interval_minutes": exp_cfg.interval_minutes,
                "last_trained": datetime.utcnow().isoformat(),
                "forecast": forecast_list,
                "recent_actuals": recent_actuals,
            },
        )
        logger.info(f"Published forecast curve to {base_entity}_forecast")

        # 8. Publish per-horizon scalar sensors
        for horizon_mins in exp_cfg.horizons_minutes:
            # Find the forecast point closest to this horizon
            target_ts = last_ts + pd.Timedelta(minutes=horizon_mins)
            idx = (ds_future - target_ts).abs().argmin()
            horizon_val = round(float(y_pred[idx]), 4)

            # Format horizon label
            if horizon_mins >= 1440:
                h_label = f"{horizon_mins // 1440}d"
            elif horizon_mins >= 60:
                h_label = f"{horizon_mins // 60}h"
            else:
                h_label = f"{horizon_mins}m"

            await self.ha_interface.set_state(
                f"{base_entity}_{h_label}",
                str(horizon_val),
                attributes={
                    "friendly_name": f"{publish_name} +{h_label}",
                    "unit_of_measurement": units,
                    "icon": "mdi:clock-fast",
                    "horizon_minutes": horizon_mins,
                    "forecast_timestamp": ds_future[idx].isoformat(),
                    "model": prod_model_name,
                },
            )

        logger.info(
            f"Published {len(exp_cfg.horizons_minutes)} horizon sensors "
            f"for {exp_cfg.name}"
        )

        # 9. Update web app status
        if self.web_app:
            status = self.web_app.state.appstate.experiment_statuses.get(
                exp_cfg.name
            )
            if status:
                status.best_model = prod_model_name
                status.mode = "production"

        logger.info(f"")
        logger.info(f"  {'─' * 50}")
        logger.info(f"  Production inference complete")
        logger.info(f"  Model: {prod_model_name}, trained in {train_time:.1f}s")
        logger.info(f"  Forecast: {len(y_pred)} points, {future_periods * exp_cfg.interval_minutes / 60:.0f}h ahead")
        logger.info(f"  Next interval: {next_val} {units}")
        for h_mins in exp_cfg.horizons_minutes:
            h_label = f"+{h_mins // 60}h" if h_mins >= 60 else f"+{h_mins}m"
            target_ts = last_ts + pd.Timedelta(minutes=h_mins)
            idx = (ds_future - target_ts).abs().argmin()
            logger.info(f"    {h_label}: {round(float(y_pred[idx]), 4)} {units}")
        logger.info(f"  {'─' * 50}")
        logger.info(f"{'=' * 60}")
        logger.info(f"")

    async def _run_update_cycle(self, next_update: datetime):
        """
        Run a full update cycle in the background.

        This runs as an asyncio task so the web server remains responsive
        during model training.
        """
        self._update_running = True
        try:
            logger.info("=== Starting scheduled update cycle ===")

            # Reload config
            await self.load_config()

            # Run updates for each experiment
            for exp_cfg in self.config.experiments:
                is_lab = exp_cfg.mode == "lab"
                await self.update_experiment(exp_cfg.name, is_lab)

            # Publish heartbeat
            await self.publish_heartbeat()

            logger.info(
                f"Update cycle completed. Next update at {next_update.isoformat()}"
            )

            # Update web app state with next update time
            if self.web_app:
                now = datetime.utcnow()
                for exp_cfg in self.config.experiments:
                    status = self.web_app.state.appstate.experiment_statuses.get(
                        exp_cfg.name
                    )
                    if status:
                        status.next_update_in_seconds = int(
                            (next_update - now).total_seconds()
                        )

        except Exception as e:
            logger.error(f"Error in update cycle: {e}", exc_info=True)
        finally:
            self._update_running = False

    async def _run_deep_analysis(self, exp_cfg):
        """
        Run deep covariate analysis: all models × all covariate combinations.

        Tests each model with:
        1. All covariates (baseline)
        2. No covariates (control)
        3. Each covariate dropped one at a time

        Generates recommendations based on MAE impact.
        """
        from ml_forecast_lab.features import build_features
        from ml_forecast_lab.benchmark.metrics import get_metric_registry
        from ml_forecast_lab.web.app import (
            DeepAnalysisResult,
            DeepAnalysisCellResult,
        )

        logger.info("")
        logger.info(f"{'=' * 60}")
        logger.info(f"  DEEP ANALYSIS: {exp_cfg.name}")
        logger.info(f"  Models: {', '.join(exp_cfg.models_enabled)}")
        logger.info(f"  Covariates: {len(exp_cfg.covariates)}")
        logger.info(f"{'=' * 60}")

        # Update status
        if self.web_app:
            cov_names = [c.entity.split(".")[-1] for c in exp_cfg.covariates]
            covariate_labels = ["All covariates", "No covariates"] + [
                f"Without {name}" for name in cov_names
            ]
            total_runs = len(exp_cfg.models_enabled) * len(covariate_labels)

            self.web_app.state.appstate.deep_analysis_results[exp_cfg.name] = DeepAnalysisResult(
                experiment_name=exp_cfg.name,
                timestamp=datetime.utcnow().isoformat(),
                status="running",
                baseline_label="All covariates",
                covariate_labels=covariate_labels,
                model_names=exp_cfg.models_enabled,
                results={},
                recommendations=[],
                total_runs=total_runs,
                completed_runs=0,
            )

        # Fetch and preprocess with all covariates
        df_full = await self._fetch_and_preprocess(exp_cfg)
        covariate_cols = [c for c in df_full.columns if c != "y"]

        # Build base features
        features_base = build_features(
            df_full, target_col="y",
            interval_minutes=exp_cfg.interval_minutes,
            country=exp_cfg.country,
        )

        # Define covariate configurations to test
        configs = []

        # 1. All covariates
        configs.append(("All covariates", covariate_cols[:]))

        # 2. No covariates
        configs.append(("No covariates", []))

        # 3. Drop one at a time
        for cov_col in covariate_cols:
            remaining = [c for c in covariate_cols if c != cov_col]
            configs.append((f"Without {cov_col}", remaining))

        results = {}
        completed = 0

        for config_label, cov_cols_to_use in configs:
            results[config_label] = {}

            for model_name in exp_cfg.models_enabled:
                try:
                    # Build combined feature matrix
                    combined = features_base.copy()
                    combined["target"] = df_full["y"]

                    # Add only the specified covariates
                    for col in cov_cols_to_use:
                        combined[col] = df_full[col]

                    combined = combined.dropna()

                    feature_cols = [c for c in combined.columns if c != "target"]
                    X = combined[feature_cols].values.astype(np.float32)
                    X = np.nan_to_num(X, nan=0.0)
                    y_all = combined["target"].values.astype(np.float32)

                    # Simple train/test split (80/20)
                    split = int(len(X) * 0.8)
                    X_tr, X_te = X[:split], X[split:]
                    y_tr, y_te = y_all[:split], y_all[split:]

                    # Train and evaluate
                    model = self.model_registry.create(model_name)

                    def _train_and_eval():
                        model.fit(X_tr, y_tr)
                        y_pred = model.predict(X_te)
                        if y_pred.ndim > 1:
                            y_pred_flat = y_pred.ravel()
                        else:
                            y_pred_flat = y_pred
                        mae_val = float(np.mean(np.abs(y_te - y_pred_flat)))
                        rmse_val = float(np.sqrt(np.mean((y_te - y_pred_flat) ** 2)))
                        return mae_val, rmse_val

                    mae_val, rmse_val = await asyncio.get_event_loop().run_in_executor(
                        None, _train_and_eval
                    )

                    results[config_label][model_name] = DeepAnalysisCellResult(
                        mae=round(mae_val, 4),
                        rmse=round(rmse_val, 4),
                    )

                    completed += 1
                    logger.info(
                        f"  [{completed}/{len(configs) * len(exp_cfg.models_enabled)}] "
                        f"{config_label} × {model_name}: MAE={mae_val:.4f}"
                    )

                    # Update progress
                    if self.web_app:
                        da = self.web_app.state.appstate.deep_analysis_results[exp_cfg.name]
                        da.completed_runs = completed
                        da.results = results

                except Exception as e:
                    logger.warning(f"  Deep analysis failed for {config_label} × {model_name}: {e}")
                    results[config_label][model_name] = DeepAnalysisCellResult(mae=np.nan, rmse=np.nan)
                    completed += 1

        # Compute % change vs baseline and generate recommendations
        baseline = results.get("All covariates", {})
        for config_label in results:
            for model_name in results[config_label]:
                cell = results[config_label][model_name]
                baseline_mae = baseline.get(model_name, DeepAnalysisCellResult(mae=np.nan, rmse=np.nan)).mae
                if baseline_mae > 0 and not np.isnan(cell.mae) and not np.isnan(baseline_mae):
                    cell.change_pct = round((cell.mae - baseline_mae) / baseline_mae * 100, 1)

        # Generate recommendations
        recommendations = []
        no_cov = results.get("No covariates", {})

        # Overall covariate value
        for model_name in exp_cfg.models_enabled:
            base_mae = baseline.get(model_name, DeepAnalysisCellResult(mae=np.nan, rmse=np.nan)).mae
            no_cov_mae = no_cov.get(model_name, DeepAnalysisCellResult(mae=np.nan, rmse=np.nan)).mae
            if not np.isnan(base_mae) and not np.isnan(no_cov_mae):
                improvement = (no_cov_mae - base_mae) / no_cov_mae * 100
                if improvement > 5:
                    recommendations.append({
                        "icon": "✓",
                        "text": f"Covariates improve {model_name} by {improvement:.1f}% — keep them",
                        "color": "#2ecc71",
                    })
                elif improvement < -5:
                    recommendations.append({
                        "icon": "⚠",
                        "text": f"Covariates hurt {model_name} by {abs(improvement):.1f}% — consider removing",
                        "color": "#f39c12",
                    })
                else:
                    recommendations.append({
                        "icon": "○",
                        "text": f"Covariates have minimal impact on {model_name} ({improvement:+.1f}%)",
                        "color": "#b0b0b0",
                    })

        # Per-covariate recommendations (using best tree model)
        best_tree = "lightgbm" if "lightgbm" in exp_cfg.models_enabled else exp_cfg.models_enabled[0]
        for cov_col in covariate_cols:
            label = f"Without {cov_col}"
            dropped = results.get(label, {})
            drop_mae = dropped.get(best_tree, DeepAnalysisCellResult(mae=np.nan, rmse=np.nan)).mae
            base_mae = baseline.get(best_tree, DeepAnalysisCellResult(mae=np.nan, rmse=np.nan)).mae
            if not np.isnan(drop_mae) and not np.isnan(base_mae) and base_mae > 0:
                impact = (drop_mae - base_mae) / base_mae * 100
                if impact > 5:
                    recommendations.append({
                        "icon": "✓",
                        "text": f"{cov_col} is important — dropping it increases {best_tree} MAE by {impact:.1f}%",
                        "color": "#2ecc71",
                    })
                elif impact < -2:
                    recommendations.append({
                        "icon": "✗",
                        "text": f"{cov_col} is harmful — dropping it improves {best_tree} MAE by {abs(impact):.1f}%",
                        "color": "#e74c3c",
                    })
                else:
                    recommendations.append({
                        "icon": "○",
                        "text": f"{cov_col} has marginal impact on {best_tree} ({impact:+.1f}%)",
                        "color": "#b0b0b0",
                    })

        # Store final results
        if self.web_app:
            da = self.web_app.state.appstate.deep_analysis_results[exp_cfg.name]
            da.status = "completed"
            da.results = results
            da.recommendations = recommendations
            da.completed_runs = completed

        logger.info("")
        logger.info(f"  Deep Analysis Complete")
        logger.info(f"  {'─' * 50}")
        for rec in recommendations:
            logger.info(f"  {rec['icon']} {rec['text']}")
        logger.info(f"  {'─' * 50}")
        logger.info(f"{'=' * 60}")

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
                if now >= next_update and not self._update_running:
                    # Run update cycle in background so web server stays responsive
                    next_update = datetime.utcnow() + timedelta(seconds=update_interval)
                    asyncio.create_task(self._run_update_cycle(next_update))

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
            from ml_forecast_lab import __version__
            logger.info("")
            logger.info("╔══════════════════════════════════════════════╗")
            logger.info(f"║  ML Forecast Lab v{__version__:<27}║")
            logger.info("║  Multi-model ML forecasting for HA          ║")
            logger.info("╚══════════════════════════════════════════════╝")
            logger.info("")

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
