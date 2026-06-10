"""POST /experiment/{name}/remove-covariate must accept the full
disambiguator tuple (entity, role, future_attribute, future_value_key)
and forward it so multi-row entities like ``weather.*`` (configured
with separate ``key:`` channels for temperature / precipitation / etc.)
can be removed individually.

Pre-v2.40.14 the UI button only passed the entity string, the backend
saw N matches, and ``remove_experiment_covariate`` refused with
"Covariate not found" — exactly the failure mode reported in v2.40.13.
"""

import textwrap

import pytest
import yaml


MULTI_ROW_WEATHER_YAML = textwrap.dedent(
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
          - entity: "weather.balsham"
            role: "future"
            aggregation: "mean"
            future_attribute: "hourly"
            future_value_key: "temperature"
          - entity: "weather.balsham"
            role: "future"
            aggregation: "mean"
            future_attribute: "hourly"
            future_value_key: "precipitation"
          - entity: "weather.balsham"
            role: "future"
            aggregation: "mean"
            future_attribute: "hourly"
            future_value_key: "uv_index"
        models_enabled:
          - "lightgbm"
        loss_fn: "mse"
        cv_strategy: "walk_forward"
        cv_folds: 3
        cv_embargo_periods: 1
        metrics: ["mae"]
        production_metric: "mae"
        production_model: null
        publish_prefix: "mlfl_"
        publish_name: "smoke_demand"
        units: "C"
        country: "GB"
    """
)


@pytest.fixture
def multi_row_config(tmp_path):
    cfg = tmp_path / "mlfl.yaml"
    cfg.write_text(MULTI_ROW_WEATHER_YAML)
    return cfg


@pytest.fixture
def multi_row_app(multi_row_config):
    from ml_forecast_lab.web.app import create_app
    return create_app(config_path=multi_row_config)


@pytest.fixture
def multi_row_client(multi_row_app):
    from fastapi.testclient import TestClient
    with TestClient(multi_row_app) as c:
        yield c


@pytest.fixture
def multi_row_seeded(multi_row_app):
    from ml_forecast_lab.web.app import ExperimentStatus
    name = "smoke_demand"
    multi_row_app.state.appstate.experiment_statuses[name] = ExperimentStatus(
        name=name, target_entity="sensor.smoke_demand", mode="lab",
    )
    return name


def test_remove_one_of_three_same_entity_rows(
    multi_row_client, multi_row_seeded, multi_row_config,
):
    """Remove the precipitation row only; temperature + uv_index survive."""
    resp = multi_row_client.post(
        f"/experiment/{multi_row_seeded}/remove-covariate",
        json={
            "entity_id": "weather.balsham",
            "role": "future",
            "future_attribute": "hourly",
            "future_value_key": "precipitation",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("success") is True, body

    written = yaml.safe_load(multi_row_config.read_text())
    covs = written["experiments"][0]["covariates"]
    keys = sorted(c["future_value_key"] for c in covs)
    assert keys == ["temperature", "uv_index"], (
        f"Expected only temperature + uv_index to survive; got {keys}"
    )


def test_remove_without_disambiguators_refuses_when_multiple_rows(
    multi_row_client, multi_row_seeded, multi_row_config,
):
    """Bare entity_id (legacy frontend) must NOT silently strip every
    matching row — refuses cleanly so the user can retry with the
    disambiguators the v2.40.14 frontend now sends."""
    resp = multi_row_client.post(
        f"/experiment/{multi_row_seeded}/remove-covariate",
        json={"entity_id": "weather.balsham"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("success") is False
    # Config untouched.
    written = yaml.safe_load(multi_row_config.read_text())
    assert len(written["experiments"][0]["covariates"]) == 3
