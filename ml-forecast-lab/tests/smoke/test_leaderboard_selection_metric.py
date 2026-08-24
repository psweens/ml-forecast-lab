"""The leaderboard must show the metric that actually decides the champion.

Rankings come from `production_metric`, but the table only ever displayed MAE,
RMSE and MASE — so choosing Peak-weighted MAE reordered the leaderboard with
every visible column contradicting it. The same gap applied to the DEFAULT
`seasonal_mase`, since the "Interval MASE" column is the 1-step `mase`.

These render the real page through the real route. An earlier attempt at this
column keyed off a template variable that does not exist in that context: the
header short-circuited and never rendered, while the matching cell raised
UndefinedError. "The page still returns 200" would not have caught the first
half of that — so these assert the column is PRESENT, not merely that nothing
exploded.
"""

from ml_forecast_lab.web.app import BenchmarkResult, MetricValue, ModelResult


def _seed_results(app, name, selection_metric, value=0.4242):
    """Attach a finished benchmark carrying a selection metric.

    The page reads `benchmark_results[name].models`, not the ExperimentStatus.
    """
    app.state.appstate.benchmark_results[name] = BenchmarkResult(
        experiment_name=name,
        timestamp="2026-08-24T00:00:00",
        status="completed",
        best_model_name="lightgbm",
        models=[
            ModelResult(
                name="lightgbm",
                mae=MetricValue(mean=1.0, std=0.1),
                rmse=MetricValue(mean=2.0, std=0.2),
                mase=MetricValue(mean=0.9, std=0.05),
                selection_metric=selection_metric,
                selection_value=MetricValue(mean=value, std=0.0101),
                train_time_seconds=1.5,
                rank=1,
                mean_rank=1.0,
            )
        ],
    )


def test_peak_weighted_mae_column_is_rendered(client, seeded_experiment, app):
    _seed_results(app, seeded_experiment, "peak_weighted_mae")
    html = client.get(f"/experiment/{seeded_experiment}").text
    assert "Peak Weighted Mae" in html, (
        "the column header for the ranking metric is missing — the leaderboard "
        "reorders with nothing on screen explaining why"
    )
    assert "0.4242" in html, "the ranking metric's value is not rendered"


def test_default_seasonal_mase_column_is_rendered(client, seeded_experiment, app):
    """The default ranks models too, and was equally invisible."""
    _seed_results(app, seeded_experiment, "seasonal_mase", value=0.7777)
    html = client.get(f"/experiment/{seeded_experiment}").text
    assert "Seasonal Mase" in html
    assert "0.7777" in html


def test_no_duplicate_column_when_the_metric_is_already_shown(client, seeded_experiment, app):
    """Ranking on plain MAE must not add a second MAE column."""
    _seed_results(app, seeded_experiment, "mae")
    html = client.get(f"/experiment/{seeded_experiment}").text
    body = html[html.index("<th>Model</th>"):html.index("</thead>", html.index("<th>Model</th>"))]
    assert body.count("Interval MAE") == 1


def test_page_still_renders_without_a_selection_metric(client, seeded_experiment, app):
    """Back-compatibility: results persisted before this field existed."""
    app.state.appstate.benchmark_results[seeded_experiment] = BenchmarkResult(
        experiment_name=seeded_experiment,
        timestamp="2026-08-24T00:00:00",
        status="completed",
        models=[
            ModelResult(
                name="lightgbm",
                mae=MetricValue(mean=1.0, std=0.1),
                rmse=MetricValue(mean=2.0, std=0.2),
                mase=MetricValue(mean=0.9, std=0.05),
                train_time_seconds=1.5, rank=1, mean_rank=1.0,
            )
        ],
    )
    assert client.get(f"/experiment/{seeded_experiment}").status_code == 200
