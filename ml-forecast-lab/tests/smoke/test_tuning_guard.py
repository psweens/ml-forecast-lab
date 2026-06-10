"""Tuning-guard contracts.

The tuning endpoint must reject requests for models with no tunable
hyperparameters (the classical auto-models — ARIMA, ETS, Theta, the
Seasonal Naive baseline — and the zero-shot foundation models —
Chronos-Bolt, TTM — whose pretrained weights are frozen) — otherwise
Optuna spins on no-op trials and the UI looks hung. Locks the guard in.
"""

import pytest


@pytest.mark.parametrize("model_name", [
    "arima", "ets", "theta", "seasonal_naive", "chronos_bolt", "ttm",
])
def test_tuning_rejected_for_auto_models(client, seeded_experiment, model_name):
    """Models with all-non-tunable params get a clear 400, not a silent hang."""
    resp = client.post(
        f"/experiment/{seeded_experiment}/run-tuning",
        json={"model_name": model_name},
    )
    assert resp.status_code == 400, (
        f"{model_name} should reject tuning at the API; got {resp.status_code}"
    )
    body = resp.json()
    assert "no tunable" in body["error"].lower()


@pytest.mark.parametrize("model_name", [
    "lightgbm", "xgboost", "lstm", "cnn", "timexer", "moderntcn",
])
def test_tuning_accepts_models_with_tunable_params(
    client, seeded_experiment, app, model_name
):
    """Real ML models with searchable hyperparameters still pass the guard."""
    # Stub the tuning callback so the API guard returns success without
    # kicking off a real Optuna study.
    async def _noop_callback(*args, **kwargs):
        pass
    app.state.appstate.tuning_callback = _noop_callback

    resp = client.post(
        f"/experiment/{seeded_experiment}/run-tuning",
        json={"model_name": model_name},
    )
    assert resp.status_code == 202, (
        f"{model_name} should accept tuning; got {resp.status_code}: {resp.text}"
    )


def test_tuning_rejects_unknown_model(client, seeded_experiment):
    """Unknown model name returns 400 with the existing 'No parameter schema' message."""
    resp = client.post(
        f"/experiment/{seeded_experiment}/run-tuning",
        json={"model_name": "not_a_real_model"},
    )
    assert resp.status_code == 400
