"""v2.56.0: an external forecast's unit through the web routes.

``/add-external-forecast`` accepts ``unit``, and ``/external-forecast-unit``
sets or clears it on one configured forecast, addressed by its source key, so
the comparison tab's warning can fix a forecast's unit without removing it
(removal deletes its captured history). The experiment page offers the Unit
field on the add form.
"""

import textwrap

import pytest
import yaml


TWO_SOLCAST_YAML = textwrap.dedent(
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
        external_forecasts:
          - entity_id: "sensor.solcast"
            mode: "attribute"
            attribute: "detailedForecast"
            value_key: "pv_estimate"
            label: "Solcast"
          - entity_id: "sensor.solcast"
            mode: "attribute"
            attribute: "detailedForecast"
            value_key: "pv_estimate90"
            label: "Solcast P90"
    """
)

P50 = "sensor.solcast|detailedForecast|pv_estimate"
P90 = "sensor.solcast|detailedForecast|pv_estimate90"


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "mlfl.yaml"
    p.write_text(TWO_SOLCAST_YAML)
    return p


@pytest.fixture
def app(cfg):
    from ml_forecast_lab.web.app import create_app
    return create_app(config_path=cfg)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


@pytest.fixture
def seeded(app):
    from ml_forecast_lab.web.app import ExperimentStatus
    name = "smoke_demand"
    app.state.appstate.experiment_statuses[name] = ExperimentStatus(
        name=name, target_entity="sensor.smoke_demand", mode="lab",
    )
    return name


def _units(cfg):
    exts = yaml.safe_load(cfg.read_text())["experiments"][0]["external_forecasts"]
    return [e.get("unit") for e in exts]


def test_set_unit_on_one_forecast_and_clear_it(client, seeded, cfg):
    r = client.post(f"/experiment/{seeded}/external-forecast-unit",
                    json={"source": P90, "unit": "kw"})
    assert r.status_code == 200 and r.json() == {"success": True}
    assert _units(cfg) == [None, "kW"]

    r = client.post(f"/experiment/{seeded}/external-forecast-unit",
                    json={"source": P90, "unit": ""})
    assert r.json()["success"] is True
    assert _units(cfg) == [None, None]


def test_set_unit_rejects_bad_input(client, seeded, cfg):
    r = client.post(f"/experiment/{seeded}/external-forecast-unit",
                    json={"source": P50, "unit": "volts"})
    assert r.json()["success"] is False
    r = client.post(f"/experiment/{seeded}/external-forecast-unit",
                    json={"source": "sensor.unknown", "unit": "kW"})
    assert r.json()["success"] is False
    r = client.post(f"/experiment/{seeded}/external-forecast-unit", json={"unit": "kW"})
    assert r.json()["success"] is False
    assert _units(cfg) == [None, None]
    r = client.post("/experiment/not_an_experiment/external-forecast-unit",
                    json={"source": P50, "unit": "kW"})
    assert r.status_code == 404


def test_add_with_unit(client, seeded, cfg):
    r = client.post(f"/experiment/{seeded}/add-external-forecast", json={
        "entity_id": "sensor.predbat_pv_today", "mode": "attribute",
        "attribute": "detailedForecast", "value_key": "pv_estimateCL",
        "unit": "kW",
    })
    assert r.json()["success"] is True, r.json()
    assert _units(cfg) == [None, None, "kW"]
    r = client.post(f"/experiment/{seeded}/add-external-forecast", json={
        "entity_id": "sensor.other", "unit": "furlongs",
    })
    assert r.json()["success"] is False


def test_page_offers_the_unit_field(client, seeded):
    r = client.get(f"/experiment/{seeded}")
    assert r.status_code == 200
    assert 'id="ext-unit-input"' in r.text
    assert "setExternalUnit" in r.text
