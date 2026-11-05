"""v2.57.0: ``/add-covariate`` accepts ``history_source`` and
``future_extra_entities`` (a list or a comma/space-separated string), and the
experiment page offers both on the Add Covariate form and shows them on a
configured covariate's row.
"""

import textwrap

import pytest
import yaml


YAML = textwrap.dedent(
    """\
    timezone: "Europe/London"
    update_every_minutes: 360
    experiments:
      - name: "smoke_demand"
        target_entity: "sensor.smoke_demand"
        mode: "lab"
        interval_minutes: 30
        days_history: 14
        max_age: 30
        future_periods: 48
        models_enabled:
          - "lightgbm"
        covariates:
          - entity: "sensor.day_forecast"
            role: "future"
            future_attribute: "forecast"
            future_value_key: "power"
            future_extra_entities: ["sensor.day_forecast_tomorrow"]
            history_source: "stored_forecast"
    """
)


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "mlfl.yaml"
    p.write_text(YAML)
    return p


@pytest.fixture
def client(cfg):
    from fastapi.testclient import TestClient
    from ml_forecast_lab.web.app import create_app, ExperimentStatus
    app = create_app(config_path=cfg)
    app.state.appstate.experiment_statuses["smoke_demand"] = ExperimentStatus(
        name="smoke_demand", target_entity="sensor.smoke_demand", mode="lab",
    )
    with TestClient(app) as c:
        yield c


def _covs(cfg):
    return yaml.safe_load(cfg.read_text())["experiments"][0]["covariates"]


def test_add_with_history_source_and_extra_sensors(client, cfg):
    r = client.post("/experiment/smoke_demand/add-covariate", json={
        "entity": "sensor.other_forecast", "role": "future",
        "future_attribute": "forecast", "future_value_key": "power",
        "future_extra_entities": "sensor.other_forecast_d2, sensor.other_forecast_d3",
        "history_source": "stored_forecast",
    })
    assert r.json()["success"] is True, r.json()
    added = _covs(cfg)[-1]
    assert added["history_source"] == "stored_forecast"
    assert added["future_extra_entities"] == [
        "sensor.other_forecast_d2", "sensor.other_forecast_d3"]


def test_stored_forecast_needs_a_forecast_role(client, cfg):
    r = client.post("/experiment/smoke_demand/add-covariate", json={
        "entity": "sensor.temperature", "role": "lagged",
        "history_source": "stored_forecast",
    })
    assert r.json()["success"] is False
    assert len(_covs(cfg)) == 1


def test_page_offers_and_shows_the_fields(client):
    r = client.get("/experiment/smoke_demand")
    assert r.status_code == 200
    assert 'id="cov-history-source-input"' in r.text
    assert 'id="cov-extra-entities-input"' in r.text
    assert "history: stored forecasts" in r.text
    assert "+1 sensor<" in r.text
