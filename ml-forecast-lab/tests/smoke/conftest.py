"""Smoke-test fixtures: spin up the FastAPI app against a tmp config + data dir.

These tests boot the real `create_app()` factory but with synthetic state — no
HA REST round-trips, no model training, no /addon_configs writes. Goal is to
catch UI regressions cheaply and serve as a release gate.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---- Config fixtures ----

MINIMAL_MLFL_YAML = textwrap.dedent(
    """\
    timezone: "Europe/London"
    update_every_minutes: 360

    experiments:
      - name: "smoke_demand"
        target_entity: "sensor.smoke_demand"
        mode: "lab"
        source_is_cumulative: false
        interval_minutes: 30
        days_history: 14
        max_age: 30
        future_periods: 48
        covariates:
          - entity: "sensor.smoke_temperature"
            role: "lagged"
            aggregation: "mean"
        models_enabled:
          - "lightgbm"
          - "xgboost"
        loss_fn: "mse"
        cv_strategy: "walk_forward"
        cv_folds: 3
        cv_embargo_periods: 1
        metrics: ["mae", "rmse", "mase"]
        production_metric: "rmse"
        production_model: null
        publish_prefix: "mlfl_"
        publish_name: "smoke_demand"
        units: "%"
        output_units: "%"
        country: "GB"
    """
)


@pytest.fixture
def mlfl_config(tmp_path: Path) -> Path:
    """Write a minimal valid mlfl.yaml into tmp_path and return its path."""
    cfg_path = tmp_path / "mlfl.yaml"
    cfg_path.write_text(MINIMAL_MLFL_YAML)
    return cfg_path


# ---- App + client fixtures ----

@pytest.fixture
def app(mlfl_config: Path):
    """Build the FastAPI app pointed at the tmp config.

    `create_app(config_path=...)` honours the override (added in the path
    discovery refactor), so no monkey-patching of /addon_configs/ is needed.
    """
    from ml_forecast_lab.web.app import create_app
    return create_app(config_path=mlfl_config)


@pytest.fixture
def client(app) -> TestClient:
    """Synchronous test client — fine for HTML route checks and JSON APIs."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def seeded_experiment(app):
    """Register an experiment in AppState (the runtime-managed dict).

    Routes like ``/experiment/{name}`` 404 unless the name is in
    ``app.state.appstate.experiment_statuses`` — that dict is normally
    populated by the main app at startup. Tests that need a viewable
    experiment without going through the create flow use this.
    """
    from ml_forecast_lab.web.app import ExperimentStatus
    name = "smoke_demand"
    app.state.appstate.experiment_statuses[name] = ExperimentStatus(
        name=name,
        target_entity="sensor.smoke_demand",
        mode="lab",
    )
    return name
