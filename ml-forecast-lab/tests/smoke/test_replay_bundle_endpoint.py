"""/experiment/{name}/replay-bundle (v2.52.2).

The route returns the bundle the main app's capture callback produces as a
zip attachment. The capture itself (and its ML imports) lives behind the
callback, so this gate needs only the web stack.
"""
from __future__ import annotations


def test_returns_zip_attachment(client, seeded_experiment):
    calls = []

    async def _capture(name):
        calls.append(name)
        return b"PK\x05\x06" + b"\x00" * 18  # empty zip

    client.app.state.appstate.replay_bundle_callback = _capture
    r = client.post(f"/experiment/{seeded_experiment}/replay-bundle")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    disp = r.headers["content-disposition"]
    assert disp.startswith('attachment; filename="mlfl-replay-smoke_demand-')
    assert disp.endswith('.zip"')
    assert r.content.startswith(b"PK")
    assert calls == [seeded_experiment]


def test_unknown_experiment_is_404(client):
    r = client.post("/experiment/nope/replay-bundle")
    assert r.status_code == 404


def test_unavailable_without_callback(client, seeded_experiment):
    client.app.state.appstate.replay_bundle_callback = None
    r = client.post(f"/experiment/{seeded_experiment}/replay-bundle")
    assert r.status_code == 503


def test_capture_failure_is_reported(client, seeded_experiment):
    async def _boom(name):
        raise ValueError("No history data for sensor.smoke_demand")

    client.app.state.appstate.replay_bundle_callback = _boom
    r = client.post(f"/experiment/{seeded_experiment}/replay-bundle")
    assert r.status_code == 500
    assert r.json()["success"] is False
