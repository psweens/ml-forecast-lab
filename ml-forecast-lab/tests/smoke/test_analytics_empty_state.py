"""Analytics endpoints — empty-state contracts (no db, no benchmark).

These routes back the dashboard charts. The smoke goal is just that they
return clean 4xx/5xx codes (not 500) when their preconditions aren't met,
since most HA users hit them before any benchmark has run.
"""

import pytest

ANALYTICS_PATHS = [
    "/forecast-accuracy",
    "/forecast-log-stats",
    "/forecast-trajectory",
    "/forecast-evolution",
    "/forecast-stability",
    "/covariate-analysis",
    "/tuning",
    "/results",
    "/forecast",
]


@pytest.mark.parametrize("subpath", ANALYTICS_PATHS)
def test_analytics_404_for_unknown_experiment(client, subpath):
    """Every analytics sub-route 404s when the experiment doesn't exist."""
    resp = client.get(f"/experiment/does_not_exist{subpath}")
    assert resp.status_code == 404, f"{subpath} returned {resp.status_code}"


@pytest.mark.parametrize("subpath", ANALYTICS_PATHS)
def test_analytics_doesnt_500_for_seeded_experiment_without_db(
    client, seeded_experiment, subpath
):
    """With an experiment but no history_db / no benchmark, routes must return
    4xx/5xx with a clean JSON detail — not crash with 500.
    """
    resp = client.get(f"/experiment/{seeded_experiment}{subpath}")
    # Acceptable: 200 (empty data), 404 (no benchmark), 503 (no db)
    assert resp.status_code in (200, 404, 503), (
        f"{subpath} returned unexpected {resp.status_code}: {resp.text[:200]}"
    )
