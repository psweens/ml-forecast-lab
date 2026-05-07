"""Promotion + mode-toggle flows. Tests both the empty-state contracts
(404 / 400 for unknown exp or missing benchmark) and the success path with
a seeded benchmark result.
"""

import yaml


def test_promote_404_for_unknown_experiment(client):
    """Promoting an experiment that doesn't exist 404s, doesn't 500."""
    resp = client.post("/experiment/does_not_exist/promote/lightgbm")
    assert resp.status_code == 404


def test_promote_400_when_no_benchmark(client, seeded_experiment):
    """Promoting before benchmarking returns 400 (not 500)."""
    resp = client.post(f"/experiment/{seeded_experiment}/promote/lightgbm")
    assert resp.status_code == 400
    assert "benchmark" in resp.json()["detail"].lower()


def test_promote_success_path(client, seeded_experiment, app, mlfl_config):
    """Seeding a benchmark result lets promote succeed and persist to YAML."""
    from ml_forecast_lab.web.app import (
        BenchmarkResult,
        ModelResult,
        MetricValue,
    )
    metric = MetricValue(mean=0.1, std=0.02)
    bench = BenchmarkResult(
        experiment_name=seeded_experiment,
        timestamp="2026-05-07T00:00:00Z",
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
    app.state.appstate.benchmark_results[seeded_experiment] = bench

    resp = client.post(f"/experiment/{seeded_experiment}/promote/lightgbm")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "lightgbm"
    assert body["experiment"] == seeded_experiment

    # Verify YAML persistence
    written = yaml.safe_load(mlfl_config.read_text())
    exp = next(e for e in written["experiments"] if e["name"] == seeded_experiment)
    assert exp["mode"] == "production"
    assert exp["production_model"] == "lightgbm"


def test_toggle_mode_flips_lab_to_production(client, seeded_experiment, mlfl_config):
    """POST /toggle-mode flips lab→production and persists."""
    resp = client.post(f"/experiment/{seeded_experiment}/toggle-mode")
    assert resp.status_code == 200

    written = yaml.safe_load(mlfl_config.read_text())
    exp = next(e for e in written["experiments"] if e["name"] == seeded_experiment)
    assert exp["mode"] == "production"


def test_toggle_mode_404_for_unknown(client):
    resp = client.post("/experiment/does_not_exist/toggle-mode")
    assert resp.status_code == 404


def test_promote_404_for_unknown_model(client, seeded_experiment, app):
    """Promoting a model not in benchmark results 404s the model name."""
    from ml_forecast_lab.web.app import (
        BenchmarkResult, ModelResult, MetricValue,
    )
    metric = MetricValue(mean=0.1, std=0.02)
    app.state.appstate.benchmark_results[seeded_experiment] = BenchmarkResult(
        experiment_name=seeded_experiment,
        timestamp="2026-05-07T00:00:00Z",
        status="completed",
        models=[
            ModelResult(
                name="lightgbm",
                mae=metric, rmse=metric, mase=metric,
                train_time_seconds=1.0, rank=1,
            ),
        ],
    )
    resp = client.post(f"/experiment/{seeded_experiment}/promote/not_a_real_model")
    assert resp.status_code == 404
