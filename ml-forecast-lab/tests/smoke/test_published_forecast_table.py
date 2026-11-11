"""The Results tab's Published forecast table (v2.58.0).

The table was first nested inside the Daily Cumulative Accuracy block, so it
only rendered when at least one model had daily metrics — a benchmark whose
folds spanned under two dates showed nothing, although the published scores
had chosen the winner. These render the real page and assert the table and
the comparison with the production model are present without any daily
metrics.
"""

from ml_forecast_lab.web.app import BenchmarkResult, MetricValue, ModelResult


def _row(name, crps, candidate=True, eligible=True):
    return dict(name=name, crps=crps, rmse=3.0, mae=1.5, bias_pct=-1.0,
                period_total_rmse=13.0, coverage=0.78, eligible=eligible,
                candidate=candidate, n_origins=100, interval_width=4.0,
                amplitude_ratio=0.6)


def _seed(app, name, comparison):
    mv = MetricValue(mean=1.0, std=0.1)
    app.state.appstate.benchmark_results[name] = BenchmarkResult(
        experiment_name=name, timestamp="2026-11-11T00:00:00", status="completed",
        best_model_name="lightgbm",
        models=[ModelResult(name=n, mae=mv, rmse=mv, mase=mv, train_time_seconds=1.0,
                            rank=r, mean_rank=float(r))
                for r, n in enumerate(["seasonal_profile", "lightgbm"], 1)],
        published_scores=[_row("baseline_profile_mean", 1.62, candidate=False),
                          _row("seasonal_profile", 1.6789), _row("lightgbm", 1.7961)],
        published_evaluation_days=29.4, published_enough_evidence=True,
        selection_metric="crps", production_comparison=comparison,
    )


def test_table_renders_without_daily_metrics(client, seeded_experiment, app):
    _seed(app, seeded_experiment, None)
    html = client.get(f"/experiment/{seeded_experiment}").text
    assert "Published forecast" in html
    assert "1.6789" in html and "1.7961" in html
    assert "Reference: typical day (mean)" in html


def test_consistent_improvement_over_production_is_stated(client, seeded_experiment, app):
    _seed(app, seeded_experiment, dict(
        top="seasonal_profile", production="lightgbm", same=False,
        improvement=0.058, n_days=25, verdict="better",
    ))
    html = client.get(f"/experiment/{seeded_experiment}").text
    assert "Seasonal Profile scored 5.8% better than your production model, LightGBM" in html
    assert "promoting it should help" in html


def test_noise_level_difference_is_stated(client, seeded_experiment, app):
    _seed(app, seeded_experiment, dict(
        top="seasonal_profile", production="lightgbm", same=False,
        improvement=0.003, n_days=25, verdict="indistinguishable",
    ))
    html = client.get(f"/experiment/{seeded_experiment}").text
    assert "0.3% better" in html and "within the day-to-day variation" in html


def test_production_model_on_top_is_stated(client, seeded_experiment, app):
    _seed(app, seeded_experiment, dict(
        top="lightgbm", production="lightgbm", same=True,
        improvement=0.0, n_days=None, verdict=None,
    ))
    html = client.get(f"/experiment/{seeded_experiment}").text
    assert "Your production model, LightGBM, is the best model in this table" in html


def test_leaderboard_crps_column_shows_the_published_score(client, seeded_experiment, app):
    """With CRPS selection the leaderboard's ranking column showed
    'nan ± nan': it read a per-fold 'crps' that does not exist."""
    _seed(app, seeded_experiment, None)
    br = app.state.appstate.benchmark_results[seeded_experiment]
    for m, v in zip(br.models, (0.8198, 0.8459)):
        m.selection_metric = "crps"
        m.selection_value = MetricValue(mean=v, std=0.0)
    html = client.get(f"/experiment/{seeded_experiment}").text
    assert "0.8459" in html and "nan ± nan" not in html
    assert "CRPS of each model's published forecast" in html
