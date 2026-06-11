"""Gating contract for the developer-mode branch overlay.

The feature must be invisible and inert for normal users: with
`developer_mode` off (the default), the System page shows no Developer
card and the endpoints 404 (indistinguishable from nonexistent). With it
on, the endpoints exist and validate input. The endpoints read
DEVELOPER_MODE live per-request, so toggling the env var is enough — no
app rebuild needed.
"""

import pytest


def test_system_page_hides_dev_card_by_default(client, monkeypatch):
    monkeypatch.delenv("DEVELOPER_MODE", raising=False)
    resp = client.get("/system")
    assert resp.status_code == 200
    # The card's section heading must not appear when dev mode is off.
    assert ">Developer<" not in resp.text
    assert "/api/system/dev/install-branch" not in resp.text


def test_dev_endpoints_404_when_disabled(client, monkeypatch):
    monkeypatch.delenv("DEVELOPER_MODE", raising=False)
    assert client.post("/api/system/dev/install-branch",
                       json={"branch": "main"}).status_code == 404
    assert client.post("/api/system/dev/revert").status_code == 404


def test_system_page_shows_dev_card_when_enabled(client, monkeypatch):
    monkeypatch.setenv("DEVELOPER_MODE", "true")
    resp = client.get("/system")
    assert resp.status_code == 200
    assert ">Developer<" in resp.text
    assert "install-branch" in resp.text


def test_install_rejects_bad_branch_when_enabled(client, monkeypatch):
    """With dev mode on, a malformed branch is rejected at validation —
    a 400 (not a 404, and no network fetch attempted)."""
    monkeypatch.setenv("DEVELOPER_MODE", "true")
    resp = client.post("/api/system/dev/install-branch",
                       json={"branch": "../etc/passwd"})
    assert resp.status_code == 400
    assert resp.json()["success"] is False


def test_branches_endpoint_404_when_disabled(client, monkeypatch):
    monkeypatch.delenv("DEVELOPER_MODE", raising=False)
    assert client.get("/api/system/dev/branches").status_code == 404


def test_branches_endpoint_returns_list_when_enabled(client, monkeypatch):
    """With dev mode on, the endpoint returns the parsed branch list.

    The GitHub fetch is stubbed so the test is deterministic and offline.
    """
    monkeypatch.setenv("DEVELOPER_MODE", "true")
    from ml_forecast_lab import dev_branch

    async def _fake_list(token=None):
        return ["main", "claude/feature-a", "claude/feature-b"]

    monkeypatch.setattr(dev_branch, "list_repo_branches", _fake_list)
    resp = client.get("/api/system/dev/branches")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["branches"][0] == "main"
    assert "claude/feature-a" in body["branches"]


def test_branches_endpoint_reports_error_gracefully(client, monkeypatch):
    """A GitHub failure surfaces as success:false with an empty list so the
    UI can fall back to manual entry — never a 500."""
    monkeypatch.setenv("DEVELOPER_MODE", "true")
    from ml_forecast_lab import dev_branch

    async def _boom(token=None):
        raise dev_branch.DevBranchError("rate limited")

    monkeypatch.setattr(dev_branch, "list_repo_branches", _boom)
    resp = client.get("/api/system/dev/branches")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["branches"] == []
    assert "rate limited" in body["error"]


def test_revert_when_enabled_reports_no_overlay(client, monkeypatch, tmp_path):
    """Revert with nothing installed is a clean no-op (no restart)."""
    monkeypatch.setenv("DEVELOPER_MODE", "true")
    from ml_forecast_lab import dev_branch
    monkeypatch.setattr(dev_branch, "DEV_SRC_DIR", tmp_path / "nope")
    monkeypatch.setattr(dev_branch, "ACTIVE_MARKER", tmp_path / "nope" / "ACTIVE.json")
    resp = client.post("/api/system/dev/revert")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["reverted"] is False
    assert body["restarting"] is False
