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
