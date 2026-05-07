"""Settings round-trip — global and per-experiment YAML edits via API."""

import yaml


def test_save_global_settings(client, mlfl_config):
    """POST /api/settings updates fields in mlfl.yaml."""
    resp = client.post(
        "/api/settings",
        json={
            "forecast_every_minutes": 30,
            "retrain_every_hours": 12,
            "timezone": "America/New_York",
        },
    )
    assert resp.status_code == 200
    assert resp.json().get("success") is True

    written = yaml.safe_load(mlfl_config.read_text())
    assert written["update_every_minutes"] == 30
    assert written["forecast_every_minutes"] == 30
    assert written["retrain_every_hours"] == 12
    assert written["timezone"] == "America/New_York"


def test_save_settings_validates_int_types(client, mlfl_config):
    """Bad types in numeric fields surface as success=False, not 500."""
    resp = client.post("/api/settings", json={"forecast_every_minutes": "not_a_number"})
    assert resp.status_code == 200
    assert resp.json().get("success") is False


def test_per_experiment_settings_round_trip(client, seeded_experiment, mlfl_config):
    """POST /api/experiment-settings persists per-exp training config."""
    resp = client.post(
        "/api/experiment-settings",
        json={
            "experiment": seeded_experiment,
            "cv_folds": 4,
            "cv_strategy": "sliding_window",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("success") is True, body

    written = yaml.safe_load(mlfl_config.read_text())
    exp = next(e for e in written["experiments"] if e["name"] == seeded_experiment)
    assert exp["cv_folds"] == 4
    assert exp["cv_strategy"] == "sliding_window"


def test_settings_redirect_to_system(client):
    """`/settings` 301-redirects to `/system` (legacy URL preserved)."""
    resp = client.get("/settings", follow_redirects=False)
    assert resp.status_code == 301
    assert "/system" in resp.headers["location"]
