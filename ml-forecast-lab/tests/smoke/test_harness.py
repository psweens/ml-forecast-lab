"""Validates the smoke-test harness itself — boots the app and walks the
top-level pages. If any of these regress, the rest of the smoke suite is
meaningless.
"""


def test_app_boots(app):
    """Factory returns a FastAPI app with all expected routes mounted."""
    route_paths = {getattr(r, "path", None) for r in app.routes}
    expected = {"/", "/models", "/system", "/training", "/log"}
    missing = expected - route_paths
    assert not missing, f"Missing top-level routes: {missing}"


def test_homepage_renders(client):
    """`GET /` returns 200 with the dashboard shell."""
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    assert "ML Forecast Lab" in body


def test_models_page_renders(client):
    """`GET /models` lists the registered model backends."""
    resp = client.get("/models")
    assert resp.status_code == 200
    assert "ML Forecast Lab" in resp.text


def test_system_page_renders(client):
    """`GET /system` returns the system info / settings page."""
    resp = client.get("/system")
    assert resp.status_code == 200


def test_training_page_renders(client):
    """`GET /training` returns the training history view."""
    resp = client.get("/training")
    assert resp.status_code == 200


def test_log_page_renders_when_no_log_file(client):
    """`/log` must not 500 when the addon log file doesn't exist (dev/test envs)."""
    resp = client.get("/log")
    assert resp.status_code == 200


def test_api_status_returns_json(client):
    """`/api/status` returns valid JSON describing the app state."""
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
