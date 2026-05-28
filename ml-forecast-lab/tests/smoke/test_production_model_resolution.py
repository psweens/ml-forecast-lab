"""UI 'production model' display must match the inference path's choice.

Bug: the UI fell back to ``selected_model or best_model``, but inference
falls back to ``production_model or best_model``. When a model Y is
Promoted (writes ``production_model=Y`` to YAML) and a later benchmark
crowns a different winner Z, the UI showed Z while inference still ran Y.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


PROMOTED_XGBOOST_YAML = textwrap.dedent(
    """\
    timezone: "Europe/London"
    update_every_minutes: 360
    experiments:
      - name: "smoke_demand"
        target_entity: "sensor.smoke_demand"
        mode: "production"
        source_is_cumulative: false
        interval_minutes: 30
        days_history: 14
        max_age: 30
        future_periods: 48
        covariates: []
        models_enabled:
          - "lightgbm"
          - "xgboost"
        loss_fn: "mse"
        cv_strategy: "walk_forward"
        cv_folds: 3
        cv_embargo_periods: 1
        metrics: ["mae", "rmse", "mase"]
        production_metric: "rmse"
        production_model: "xgboost"
        publish_prefix: "mlfl_"
        publish_name: "smoke_demand"
        units: "%"
        output_units: "%"
        country: "GB"
    """
)


@pytest.fixture
def mlfl_with_promoted_xgboost(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "mlfl.yaml"
    cfg_path.write_text(PROMOTED_XGBOOST_YAML)
    return cfg_path


@pytest.fixture
def app_promoted(mlfl_with_promoted_xgboost: Path):
    from ml_forecast_lab.web.app import create_app
    return create_app(config_path=mlfl_with_promoted_xgboost)


@pytest.fixture
def client_promoted(app_promoted) -> TestClient:
    with TestClient(app_promoted) as c:
        yield c


@pytest.fixture
def seeded_with_lightgbm_winner(app_promoted):
    """Production_model=xgboost lives in YAML (earlier Promote), but the latest
    benchmark crowned lightgbm. selected_model is None (user never clicked
    Select on Results tab).
    """
    from ml_forecast_lab.web.app import (
        BenchmarkResult,
        ExperimentStatus,
        MetricValue,
        ModelResult,
    )
    name = "smoke_demand"
    app_promoted.state.appstate.experiment_statuses[name] = ExperimentStatus(
        name=name,
        target_entity="sensor.smoke_demand",
        mode="production",
        best_model="lightgbm",
        selected_model=None,
    )
    metric = MetricValue(mean=0.1, std=0.02)
    app_promoted.state.appstate.benchmark_results[name] = BenchmarkResult(
        experiment_name=name,
        timestamp="2026-05-28T00:00:00Z",
        status="completed",
        models=[
            ModelResult(
                name="lightgbm",
                mae=metric, rmse=metric, mase=metric,
                train_time_seconds=1.0, rank=1, mean_rank=1.0,
            ),
            ModelResult(
                name="xgboost",
                mae=metric, rmse=metric, mase=metric,
                train_time_seconds=1.0, rank=2, mean_rank=2.0,
            ),
        ],
        best_model_name="lightgbm",
    )
    return name


def test_prod_model_js_var_matches_inference(
    client_promoted, seeded_with_lightgbm_winner,
):
    """The rendered HTML's ``var prodModel`` (template line 5739) must equal
    the model inference actually runs. Inference uses
    ``production_model or best_model_name`` → ``xgboost``. The UI must agree.
    """
    name = seeded_with_lightgbm_winner
    resp = client_promoted.get(f"/experiment/{name}")
    assert resp.status_code == 200, resp.text

    m = re.search(r'var\s+prodModel\s*=\s*"([^"]+)"', resp.text)
    assert m, "prodModel JS variable not found in rendered HTML"
    ui_says = m.group(1)

    assert ui_says == "xgboost", (
        f"UI shows prodModel='{ui_says}' but inference would run "
        f"production_model='xgboost' (from YAML). The UI fallback chain "
        f"is missing production_model."
    )


def test_publishing_button_shows_promoted_model(
    client_promoted, seeded_with_lightgbm_winner,
):
    """The 'Publishing X' button (template line 50) reflects what the
    inference path is actually deploying. Same fallback chain — covers
    the second visible site, in case future edits break only one."""
    name = seeded_with_lightgbm_winner
    resp = client_promoted.get(f"/experiment/{name}")
    assert resp.status_code == 200

    # ``model_display`` filter maps "xgboost" → "XGBoost", "lightgbm" → "LightGBM".
    assert re.search(r'Publishing\s+XGBoost', resp.text), (
        "Publishing button does not show 'Publishing XGBoost' — UI is "
        "showing the latest leaderboard winner instead of the promoted model."
    )
    # Sanity: the wrong (latest-winner) model is NOT shown in the publish button.
    assert not re.search(r'Publishing\s+LightGBM', resp.text), (
        "Publishing button still shows 'Publishing LightGBM' (the latest "
        "leaderboard winner) — the fix didn't reach this template site."
    )


def test_selected_model_still_wins_when_explicitly_set(
    app_promoted, client_promoted,
):
    """If the user has explicitly clicked Select on the Results tab,
    ``selected_model`` continues to take precedence over both
    ``production_model`` and ``best_model``. The three-way fallback must
    not regress this — only fix the case where ``selected_model`` is None.
    """
    from ml_forecast_lab.web.app import (
        BenchmarkResult,
        ExperimentStatus,
        MetricValue,
        ModelResult,
    )
    name = "smoke_demand"
    app_promoted.state.appstate.experiment_statuses[name] = ExperimentStatus(
        name=name,
        target_entity="sensor.smoke_demand",
        mode="production",
        best_model="lightgbm",
        selected_model="lightgbm",  # user explicitly picked lightgbm
    )
    metric = MetricValue(mean=0.1, std=0.02)
    app_promoted.state.appstate.benchmark_results[name] = BenchmarkResult(
        experiment_name=name,
        timestamp="2026-05-28T00:00:00Z",
        status="completed",
        models=[
            ModelResult(
                name="lightgbm",
                mae=metric, rmse=metric, mase=metric,
                train_time_seconds=1.0, rank=1, mean_rank=1.0,
            ),
        ],
        best_model_name="lightgbm",
    )
    # YAML still has production_model=xgboost (a stale Promote).
    resp = client_promoted.get(f"/experiment/{name}")
    assert resp.status_code == 200

    m = re.search(r'var\s+prodModel\s*=\s*"([^"]+)"', resp.text)
    assert m and m.group(1) == "lightgbm", (
        "selected_model should win when explicitly set, but UI shows "
        f"prodModel='{m.group(1) if m else None}'"
    )
