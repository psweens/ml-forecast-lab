"""Model parameter and toggle round-trips."""


def test_get_model_params_returns_schema_for_all_models(client):
    """GET /api/models/params returns schema for every model in MODEL_PARAM_SCHEMA."""
    resp = client.get("/api/models/params")
    assert resp.status_code == 200
    data = resp.json()
    # Spot-check a handful from each category
    for model in ("lightgbm", "xgboost", "lstm", "cnn", "tft", "patchtst"):
        assert model in data, f"{model} missing from /api/models/params"
        assert "defaults" in data[model]
        assert "current" in data[model]
        assert "schema" in data[model]


def test_save_model_params_round_trip(client):
    """POST overrides → GET reads them back as 'overrides'."""
    resp = client.post(
        "/api/models/params",
        json={"model_name": "lightgbm", "params": {"n_estimators": 250, "max_depth": 4}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("success") is True, body
    assert body["overrides"]["n_estimators"] == 250
    assert body["overrides"]["max_depth"] == 4

    follow = client.get("/api/models/params")
    assert follow.json()["lightgbm"]["overrides"]["n_estimators"] == 250


def test_save_model_params_rejects_unknown_model(client):
    resp = client.post(
        "/api/models/params",
        json={"model_name": "not_a_real_model", "params": {}},
    )
    assert resp.json().get("success") is False


def test_save_model_params_rejects_unknown_param(client):
    resp = client.post(
        "/api/models/params",
        json={"model_name": "lightgbm", "params": {"not_a_real_param": 99}},
    )
    assert resp.json().get("success") is False


def test_save_model_params_strips_defaults(client):
    """Values matching defaults are NOT persisted as overrides."""
    # Default n_estimators for lightgbm is 500 — saving 500 should produce no override
    resp = client.post(
        "/api/models/params",
        json={"model_name": "lightgbm", "params": {"n_estimators": 500}},
    )
    assert resp.json().get("success") is True
    assert "n_estimators" not in resp.json()["overrides"]


def test_reset_model_params(client):
    """POST /api/models/params/reset wipes overrides for a model."""
    # Set then reset
    client.post(
        "/api/models/params",
        json={"model_name": "xgboost", "params": {"n_estimators": 999}},
    )
    resp = client.post("/api/models/params/reset", json={"model_name": "xgboost"})
    assert resp.status_code == 200
    assert resp.json().get("success") is True

    follow = client.get("/api/models/params")
    assert follow.json()["xgboost"]["overrides"] == {}


def test_global_model_toggle(client):
    """POST /api/models/toggle disables a model across all experiments."""
    resp = client.post(
        "/api/models/toggle",
        json={"model_name": "lstm", "enabled": False},
    )
    assert resp.status_code == 200
    assert resp.json().get("success") is True


def test_per_experiment_model_toggle(client, seeded_experiment):
    """POST /api/experiment/{name}/models/toggle works for a real experiment."""
    resp = client.post(
        f"/api/experiment/{seeded_experiment}/models/toggle",
        json={"model_name": "xgboost", "enabled": False},
    )
    assert resp.status_code == 200
    assert resp.json().get("success") is True
