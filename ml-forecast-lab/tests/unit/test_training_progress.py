"""Live training-progress summary (v2.40.3).

Regression for the "models complete reads higher than the total" bug
(e.g. 9/5 → 180%): completions must be counted only within the latest
pipeline_start window and clamped to the declared total, so stale or
replayed events from an earlier run on the same history can't inflate it.
"""
from ml_forecast_lab.training_events import TrainingEvent, summarise_history


def _ev(event_type, **kw):
    return TrainingEvent(event_type=event_type, experiment_name="e", **kw)


def _run(n_models, completed):
    evs = [_ev("pipeline_start", message=f"Starting benchmark with {n_models} model(s)")]
    for i in range(completed):
        evs.append(_ev("model_start", model_name=f"m{i}"))
        evs.append(_ev("model_end", model_name=f"m{i}"))
    return evs


def test_single_run_counts_correctly():
    s = summarise_history(_run(5, 3))
    assert s["total_models"] == 5
    assert s["completed_models"] == 3
    assert s["progress_pct"] == 60


def test_stale_prior_run_does_not_inflate_count():
    """A completed prior run's events followed by a new run must report only
    the new run's progress — not 5 (old) + 4 (new) = 9/5."""
    history = _run(5, 5) + _run(5, 4)  # full run, then a re-run 4-done
    s = summarise_history(history)
    assert s["completed_models"] == 4, "must count only the latest run"
    assert s["total_models"] == 5
    assert s["progress_pct"] == 80


def test_completed_clamped_to_total():
    """Even if extra model_end events slip in after pipeline_start (e.g.
    overlapping emissions), the reported count never exceeds the total and
    the percentage never exceeds 100."""
    history = [_ev("pipeline_start", message="Starting benchmark with 5 model(s)")]
    for i in range(9):  # 9 model_end against a declared 5
        history.append(_ev("model_end", model_name=f"m{i}"))
    s = summarise_history(history)
    assert s["completed_models"] == 5
    assert s["progress_pct"] == 100


def test_current_model_and_epoch_tracked():
    history = _run(3, 1) + [
        _ev("model_start", model_name="lstm"),
        _ev("epoch", model_name="lstm", fold=4, total_folds=5, epoch=23, total_epochs=100),
    ]
    s = summarise_history(history)
    assert s["current_model"] == "lstm"
    assert s["fold"] == 4 and s["total_folds"] == 5
    assert s["epoch"] == 23 and s["total_epochs"] == 100
    assert s["completed_models"] == 1  # only m0 finished


def test_empty_history_is_safe():
    s = summarise_history([])
    assert s["total_models"] == 0
    assert s["completed_models"] == 0
    assert s["progress_pct"] == 0
