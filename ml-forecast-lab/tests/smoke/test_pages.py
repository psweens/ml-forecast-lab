"""All top-level pages render without 500s. Cheap regression catcher — if any
template variable goes missing or a route raises, this fails first.
"""

import pytest

# (path, expected status, optional substring check)
TOP_LEVEL_PAGES = [
    ("/", 200, "ML Forecast Lab"),
    ("/models", 200, "ML Forecast Lab"),
    ("/system", 200, None),
    ("/log", 200, None),
    ("/training", 200, None),  # redirects to /
    ("/api/status", 200, None),
    # Empty-state contract: debug log returns 404 with a JSON detail when
    # nothing's been generated yet. Locking it in catches accidental 500s
    # in that branch.
    ("/debug_log", 404, None),
]


@pytest.mark.parametrize("path,expected_status,must_contain", TOP_LEVEL_PAGES)
def test_top_level_page_renders(client, path, expected_status, must_contain):
    """Each top-level URL returns its expected status and contains expected markers."""
    resp = client.get(path)
    assert resp.status_code == expected_status, (
        f"{path} returned {resp.status_code}: {resp.text[:200]}"
    )
    if must_contain:
        assert must_contain in resp.text, f"{path} missing expected '{must_contain}'"


def test_models_page_lists_all_backends(client):
    """Models page must list all 28 backends from MODEL_CATALOG."""
    resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.text
    # Spot-check one model from each category
    expected_models = [
        "LightGBM", "XGBoost", "CatBoost",  # tree
        "LSTM", "CNN", "TFT", "PatchTST", "iTransformer", "TimeMixer",  # neural
        "TimeXer", "ModernTCN",  # 2024 architectures
        "Chronos-Bolt", "Granite TTM",  # zero-shot foundation
        "Seasonal Naive",  # baseline
        "ARIMA", "ETS", "Theta",  # statsforecast
    ]
    missing = [m for m in expected_models if m not in body]
    assert not missing, f"Models page missing: {missing}"


def test_models_page_category_headings(client):
    """Models page groups backends under per-category headings, including a
    dedicated Foundation Models section for the zero-shot backends."""
    body = client.get("/models").text
    for heading in ("Tree Models", "Neural Models", "Foundation Models",
                    "Classical Models", "Baselines"):
        assert heading in body, f"missing category heading: {heading!r}"
    # The foundation backends sit under the Foundation Models heading.
    foundation_idx = body.index("Foundation Models")
    assert foundation_idx < body.index("Chronos-Bolt")
    assert foundation_idx < body.index("Granite TTM")
    # …and the Foundation heading itself comes after the Tree heading.
    assert body.index("Tree Models") < foundation_idx


def test_models_page_alphabetical(client):
    """Per project convention (v2.27.4), MODEL_CATALOG is sorted alphabetically."""
    resp = client.get("/models")
    body = resp.text
    # CatBoost must appear before LightGBM (alphabetical), even though
    # LightGBM was added to MODEL_CATALOG first.
    cat_idx = body.find("CatBoost")
    lgb_idx = body.find("LightGBM")
    assert cat_idx > 0 and lgb_idx > 0
    assert cat_idx < lgb_idx, "MODEL_CATALOG no longer rendering alphabetically"
