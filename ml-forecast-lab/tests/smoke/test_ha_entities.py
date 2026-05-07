"""HA entity picker — verifies graceful empty-state when HA isn't reachable.

In production this route caches /api/states from the HA supervisor every
60s. Tests with no HA available rely on the route's fallback path
(catches the request error, returns the stale-or-empty cache).
"""

import pytest


@pytest.fixture(autouse=True)
def _refuse_ha_fast(monkeypatch):
    """Point HA_URL at a port that refuses immediately (no 10s DNS hang)."""
    monkeypatch.setenv("HA_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("SUPERVISOR_TOKEN", "test_token")


def test_ha_entities_returns_empty_list_when_ha_unreachable(client):
    """Route must not raise when the HA REST API is unreachable."""
    resp = client.get("/api/ha/entities")
    assert resp.status_code == 200
    assert resp.json() == []


def test_ha_entities_search_query_param(client):
    """Empty query against an empty cache still returns 200 + empty list."""
    resp = client.get("/api/ha/entities?q=temp")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
