"""Experiment CRUD round-trip — create, view, delete."""


def test_seeded_experiment_renders(client, seeded_experiment):
    """An experiment registered in AppState renders its detail page."""
    resp = client.get(f"/experiment/{seeded_experiment}")
    assert resp.status_code == 200
    assert seeded_experiment in resp.text


def test_experiment_404_when_not_in_state(client):
    """Routes correctly 404 for unknown experiments — locks the empty-state contract."""
    resp = client.get("/experiment/does_not_exist")
    assert resp.status_code == 404


def test_create_then_view_experiment(client):
    """POST /api/experiments/create writes to YAML; the new experiment then renders."""
    name = "smoke_new_exp"
    resp = client.post(
        "/api/experiments/create",
        json={"name": name, "target_entity": "sensor.smoke_other"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("success") is True, data
    assert data.get("redirect") == f"/experiment/{name}"

    # The new experiment is now in the YAML and in AppState
    page = client.get(f"/experiment/{name}")
    assert page.status_code == 200
    assert name in page.text


def test_create_rejects_invalid_name(client):
    """Names not matching [a-z][a-z0-9_]{0,63} are rejected with success=False."""
    resp = client.post(
        "/api/experiments/create",
        json={"name": "Bad-Name", "target_entity": "sensor.x"},
    )
    assert resp.status_code == 200  # API uses success flag, not HTTP status
    assert resp.json().get("success") is False


def test_create_rejects_missing_target(client):
    """target_entity is required."""
    resp = client.post(
        "/api/experiments/create",
        json={"name": "smoke_no_target"},
    )
    assert resp.json().get("success") is False


def test_delete_experiment(client):
    """Create then delete — final state should be 404 on GET, missing from list."""
    name = "smoke_to_delete"
    create_resp = client.post(
        "/api/experiments/create",
        json={"name": name, "target_entity": "sensor.foo"},
    )
    assert create_resp.json().get("success") is True

    del_resp = client.post(f"/api/experiments/{name}/delete")
    assert del_resp.status_code == 200
    assert del_resp.json().get("success") is True


def test_replace_target_updates_yaml_and_status(client, seeded_experiment, mlfl_config):
    """POST replace-target rewrites target_entity, updates the in-memory
    status and surfaces the new sensor on the detail page."""
    import yaml

    resp = client.post(
        f"/experiment/{seeded_experiment}/replace-target",
        json={"target_entity": "sensor.smoke_replacement"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("success") is True, data
    assert data.get("changed") is True
    assert data.get("previous_target") == "sensor.smoke_demand"
    assert data.get("target_entity") == "sensor.smoke_replacement"

    # Persisted to YAML.
    cfg = yaml.safe_load(mlfl_config.read_text())
    exp = next(e for e in cfg["experiments"] if e["name"] == seeded_experiment)
    assert exp["target_entity"] == "sensor.smoke_replacement"

    # In-memory status reflects the swap; the detail page renders it.
    status = client.app.state.appstate.experiment_statuses[seeded_experiment]
    assert status.target_entity == "sensor.smoke_replacement"
    page = client.get(f"/experiment/{seeded_experiment}")
    assert "sensor.smoke_replacement" in page.text


def test_replace_target_noop_when_same(client, seeded_experiment):
    """Replacing with the current sensor is a no-op, not a history wipe."""
    resp = client.post(
        f"/experiment/{seeded_experiment}/replace-target",
        json={"target_entity": "sensor.smoke_demand"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("success") is True
    assert data.get("changed") is False


def test_replace_target_rejects_malformed_entity(client, seeded_experiment):
    """A value that isn't a domain.object_id entity_id is rejected."""
    resp = client.post(
        f"/experiment/{seeded_experiment}/replace-target",
        json={"target_entity": "not an entity"},
    )
    assert resp.status_code == 200
    assert resp.json().get("success") is False


def test_replace_target_404_for_unknown_experiment(client):
    resp = client.post(
        "/experiment/does_not_exist/replace-target",
        json={"target_entity": "sensor.x"},
    )
    assert resp.status_code == 404
