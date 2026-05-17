"""
End-to-end pipeline integration tests for the user-reported PV
forecast collapse.

Each test reproduces a single user-facing configuration by running
the exact code path the addon uses in production:

    1. build_features → create_sliding_windows (with extended_window)
       — mirrors _retrain_and_cache.
    2. model.fit(X, y, **seq_kwargs)
       — same kwargs the addon passes.
    3. build_inference_window (with future_features_df)
       — mirrors _forecast_with_cached.
    4. model.predict_sequence(...)
       — same call the live forecast publish uses.
    5. log_transform inverse (np.expm1, clip ≥ 0)
       — mirrors the publish-time post-processing.

What we assert: the resulting forecast vector is NOT flat-zero,
NOT flat-constant, and has at least one horizon position that
exceeds a meaningful fraction of the training peak. These three
assertions together pin the user's reported failure mode (flat 0 /
flat 0.7 / flat anything) against future regressions.

The PV target is synthesised at the same Watt scale as
``predbat.pv_power`` (peak ~4 kW, instantaneous, integer-quantised,
with cloud noise and a small sensor-dropout rate) so the training
distribution matches production. Covariates are intentionally
omitted in the primary test — the user reported the bug persists
after they removed covariates, so the failure mode lives in the
core train+predict path, not the covariate plumbing.

If any test in this file fails, the failing case localises the
bug: same config + same code path → reproducer in CI.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from ml_forecast_lab.features import (
    build_features, build_inference_window, compute_known_future_features,
    create_sliding_windows,
)
from ml_forecast_lab.models.nlinear_backend import NLinearModel
from ml_forecast_lab.main import (
    _apply_output_activation, _resolve_output_activation,
)
from tests.synthetic.datasets import make_realistic_pv, GB_LAT, GB_LON


@pytest.fixture(scope="module")
def pv_data_no_covariates() -> pd.DataFrame:
    """365 days of realistic PV at 30-min interval, no covariates.

    Mirrors the user's predbat.pv_power target after they removed
    the met_office / carlton_green covariates from the experiment.
    """
    d = make_realistic_pv(seed=0)
    # Strip covariate columns — user reported the bug without them.
    return d.df[['y']].copy()


class _FakeExpCfg:
    """Minimal ExperimentCfg stand-in that has every attribute the
    production code reads. We use a plain class (not the real
    dataclass) so we can flip fields without re-instantiating, but
    every attribute name matches main.py's expectations.
    """
    def __init__(self, **overrides):
        self.name = "optimised_solar_under_test"
        self.target_entity = "predbat.pv_power"
        self.source_is_cumulative = False
        self.target_is_nonnegative = True
        self.reset_daily = False
        self.log_transform = True
        self.output_activation = "auto"
        self.daily_loss_weight = 0.25
        self.optimiser = "adam"
        self.loss_fn = "huber"
        self.interval_minutes = 30
        self.future_periods = 96
        self.country = "GB"
        self.models_enabled = ["nlinear"]
        self.production_model = "nlinear"
        self.covariates = []
        self.model_params = {}
        for k, v in overrides.items():
            setattr(self, k, v)


def _train_extended_window_nlinear(
    combined: pd.DataFrame,
    exp_cfg: _FakeExpCfg,
    epochs: int = 30,
    seed: int = 0,
    learning_rate: float = 5e-4,
):
    """Run the exact _retrain_and_cache neural training path.

    Returns the fitted model plus the seq_kwargs needed for the
    matching inference call (extended_window, past_window_size,
    future_feature_cols, channel_names).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    target_col = 'target'
    engineered = {
        'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_holiday',
    }
    engineered.update(c for c in combined.columns if c.startswith('y_lag_'))
    raw_cov_cols = [
        c for c in combined.columns if c not in engineered and c != target_col
    ]
    window_size = min(48, len(combined) // 3)
    future_periods = exp_cfg.future_periods
    horizon_steps = list(range(1, future_periods + 1))

    future_features_df = compute_known_future_features(
        combined.index,
        add_temporal=True,
        country=exp_cfg.country,
        solar_lat_lon=(GB_LAT, GB_LON),
        include_sun_elevation=False,   # user removed these covariates
        include_clear_sky_ghi=False,
    )
    seq_X, seq_y, channel_names = create_sliding_windows(
        combined, target_col, window_size=window_size,
        covariate_cols=raw_cov_cols if raw_cov_cols else None,
        add_temporal=True, horizon_steps=horizon_steps,
        future_features_df=future_features_df,
    )
    seq_kwargs = {
        'sequence_data': seq_X,
        'channel_names': channel_names,
        'extended_window': True,
        'past_window_size': window_size,
        'future_feature_cols': list(future_features_df.columns),
    }

    model = NLinearModel(
        epochs=epochs, batch_size=64, learning_rate=learning_rate,
    )
    _apply_output_activation(model, exp_cfg)
    if hasattr(model, 'daily_loss_weight'):
        model.set_params(daily_loss_weight=exp_cfg.daily_loss_weight)
    if hasattr(model, 'optimiser'):
        model.set_params(optimiser=exp_cfg.optimiser)

    X_flat = np.zeros((seq_X.shape[0], 1), dtype=np.float32)
    model.fit(
        X_flat, seq_y,
        sequence_data=seq_X,
        past_window_size=window_size,
        extended_window=True,
        channel_names=channel_names,
    )
    return model, seq_kwargs, window_size, future_features_df


def _live_forecast(
    combined: pd.DataFrame,
    model,
    seq_kwargs: dict,
    window_size: int,
    exp_cfg: _FakeExpCfg,
) -> np.ndarray:
    """Run the exact _forecast_with_cached inference path."""
    target_col = 'target'
    engineered = {
        'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_holiday',
    }
    engineered.update(c for c in combined.columns if c.startswith('y_lag_'))
    raw_cov_cols = [
        c for c in combined.columns if c not in engineered and c != target_col
    ]
    last_ts = combined.index[-1]
    future_index = pd.date_range(
        start=last_ts + pd.Timedelta(minutes=exp_cfg.interval_minutes),
        periods=exp_cfg.future_periods,
        freq=f'{exp_cfg.interval_minutes}min',
    )
    future_features_df = compute_known_future_features(
        future_index, add_temporal=True,
        country=exp_cfg.country,
        solar_lat_lon=(GB_LAT, GB_LON),
        include_sun_elevation=False,
        include_clear_sky_ghi=False,
    )
    seq_X_prod, _ = build_inference_window(
        combined, target_col, window_size=window_size,
        covariate_cols=raw_cov_cols if raw_cov_cols else None,
        add_temporal=True,
        future_features_df=future_features_df,
    )
    multi_pred = model.predict_sequence(seq_X_prod).ravel()
    y_pred = multi_pred[:exp_cfg.future_periods].astype(np.float32)
    if exp_cfg.log_transform:
        y_pred = np.expm1(y_pred).astype(np.float32)
        y_pred = np.maximum(y_pred, 0.0)
    return y_pred


def _assert_forecast_not_collapsed(
    y_pred: np.ndarray,
    target_peak_w: float,
    label: str,
) -> None:
    """Three assertions pinning the user's collapse signatures."""
    assert y_pred.shape[0] > 0
    assert np.all(np.isfinite(y_pred)), (
        f"{label}: forecast contains NaN/inf"
    )
    # Signature 1: flat zero everywhere (dying-ReLU / log_transform inverse
    # of constant 0).
    assert np.max(y_pred) > 0.0, (
        f"{label}: forecast is identically zero "
        f"(max={np.max(y_pred):.4f}, min={np.min(y_pred):.4f}, "
        f"mean={np.mean(y_pred):.4f}). This is the user's primary "
        f"failure signature — model produces flat zero across every "
        f"forecast step."
    )
    # Signature 2: flat at a non-zero constant (mean collapse).
    assert np.std(y_pred) > 0.05 * np.max(y_pred), (
        f"{label}: forecast is flat-constant "
        f"(std/max = {np.std(y_pred) / max(np.max(y_pred), 1e-9):.4f}). "
        f"Even small variation should pass this — failure indicates "
        f"the model has collapsed to a single value across the horizon."
    )
    # Signature 3: peak too small to be a meaningful PV forecast.
    # A 4 kW PV system with 96-step horizon should show a clear bell
    # at any reasonable training. 10% of the training peak is a very
    # loose lower bound that still catches the user's flat-0.7 case
    # (0.7 vs target ~3500 = 0.02% — would fail at 10%).
    assert np.max(y_pred) > 0.10 * target_peak_w, (
        f"{label}: forecast peak ({np.max(y_pred):.1f}) is < 10% of "
        f"training-data peak ({target_peak_w:.1f}). This is the "
        f"flat-mean collapse signature — model predicting a small "
        f"constant well below the actual peak."
    )


# ------------------------------------------------------------------
# Primary user-reported configuration: PV / target_is_nonnegative=True /
# log_transform=True / no covariates / extended_window
# ------------------------------------------------------------------
def test_pv_forecast_no_covariates_log_transform_extended_window(
    pv_data_no_covariates,
):
    """Reproduces the user's exact post-covariate-removal config.

    target_is_nonnegative=True + log_transform=True + extended_window
    → output_activation auto-resolves to softplus (v2.37.1) and the
    forecast must not collapse.
    """
    exp_cfg = _FakeExpCfg(
        target_is_nonnegative=True,
        log_transform=True,
    )
    # Build the same combined dataframe _retrain_and_cache builds.
    df = pv_data_no_covariates.copy()
    if exp_cfg.log_transform:
        df['y'] = np.log1p(np.maximum(df['y'].values, 0.0))
    features_df = build_features(
        df, target_col='y', interval_minutes=exp_cfg.interval_minutes,
        country=exp_cfg.country,
    )
    combined = features_df.copy()
    combined['target'] = df['y']
    combined = combined.dropna()

    # Resolver sanity check — should be softplus after v2.37.1.
    resolved = _resolve_output_activation(exp_cfg, 'nlinear')
    assert resolved == 'softplus', (
        f"v2.37.1 should auto-resolve to softplus for "
        f"target_is_nonnegative=True; got {resolved!r}"
    )

    model, seq_kwargs, window_size, _ = _train_extended_window_nlinear(
        combined, exp_cfg, epochs=20,
    )
    y_pred = _live_forecast(
        combined, model, seq_kwargs, window_size, exp_cfg,
    )
    # Peak of the original (non-log) physical target for the assertion bound.
    pv_peak = float(np.max(np.expm1(combined['target'].values)))
    _assert_forecast_not_collapsed(y_pred, pv_peak, label="pv_log_softplus")


def test_pv_forecast_no_covariates_no_log_transform(pv_data_no_covariates):
    """Same as above but without log_transform.

    User reported the collapse persists with log_transform=False as
    well, so this case must also pass.
    """
    exp_cfg = _FakeExpCfg(
        target_is_nonnegative=True,
        log_transform=False,
    )
    df = pv_data_no_covariates.copy()
    features_df = build_features(
        df, target_col='y', interval_minutes=exp_cfg.interval_minutes,
        country=exp_cfg.country,
    )
    combined = features_df.copy()
    combined['target'] = df['y']
    combined = combined.dropna()

    model, seq_kwargs, window_size, _ = _train_extended_window_nlinear(
        combined, exp_cfg, epochs=20,
    )
    y_pred = _live_forecast(
        combined, model, seq_kwargs, window_size, exp_cfg,
    )
    pv_peak = float(np.max(combined['target'].values))
    _assert_forecast_not_collapsed(y_pred, pv_peak, label="pv_nolog_softplus")


def test_pv_forecast_linear_activation_no_nonneg_flag(pv_data_no_covariates):
    """Pre-v2.37 default: target_is_nonnegative=False, log_transform=True.

    output_activation should auto-resolve to 'linear' here. This was the
    user's pre-v2.37 state — produced flat ~0.7 (mean collapse). After
    v2.37 PF1-PF9 fixes, this configuration should ALSO produce a
    non-collapsed forecast — the PF1-PF9 fixes target this exact case.
    """
    exp_cfg = _FakeExpCfg(
        target_is_nonnegative=False,
        log_transform=True,
    )
    df = pv_data_no_covariates.copy()
    df['y'] = np.log1p(np.maximum(df['y'].values, 0.0))
    features_df = build_features(
        df, target_col='y', interval_minutes=exp_cfg.interval_minutes,
        country=exp_cfg.country,
    )
    combined = features_df.copy()
    combined['target'] = df['y']
    combined = combined.dropna()

    resolved = _resolve_output_activation(exp_cfg, 'nlinear')
    assert resolved == 'linear'

    model, seq_kwargs, window_size, _ = _train_extended_window_nlinear(
        combined, exp_cfg, epochs=20,
    )
    y_pred = _live_forecast(
        combined, model, seq_kwargs, window_size, exp_cfg,
    )
    pv_peak = float(np.max(np.expm1(combined['target'].values)))
    _assert_forecast_not_collapsed(y_pred, pv_peak, label="pv_linear_pre_v237")


# ------------------------------------------------------------------
# Channel-parity test: train and inference must produce IDENTICAL
# channel ordering. A mismatch silently corrupts the forecast.
# ------------------------------------------------------------------
def test_train_and_inference_window_channel_parity(pv_data_no_covariates):
    """The cached training and live-inference paths must produce the
    SAME channel_names in the SAME order.

    A divergence here means the model is reading mis-labelled
    channels at inference time — the forecast would publish but be
    silently nonsense. This is a known historical regression class
    (the channel-parity guard in _forecast_with_cached was added in
    v2.35.2).
    """
    exp_cfg = _FakeExpCfg(target_is_nonnegative=True, log_transform=True)
    df = pv_data_no_covariates.copy()
    df['y'] = np.log1p(np.maximum(df['y'].values, 0.0))
    features_df = build_features(
        df, target_col='y', interval_minutes=exp_cfg.interval_minutes,
        country=exp_cfg.country,
    )
    combined = features_df.copy()
    combined['target'] = df['y']
    combined = combined.dropna()

    model, seq_kwargs, window_size, _ = _train_extended_window_nlinear(
        combined, exp_cfg, epochs=2,
    )
    train_channel_names = list(seq_kwargs['channel_names'])

    # Inference window
    target_col = 'target'
    engineered = {
        'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_holiday',
    }
    engineered.update(c for c in combined.columns if c.startswith('y_lag_'))
    raw_cov_cols = [
        c for c in combined.columns if c not in engineered and c != target_col
    ]
    last_ts = combined.index[-1]
    future_index = pd.date_range(
        start=last_ts + pd.Timedelta(minutes=exp_cfg.interval_minutes),
        periods=exp_cfg.future_periods,
        freq=f'{exp_cfg.interval_minutes}min',
    )
    future_features_df = compute_known_future_features(
        future_index, add_temporal=True,
        country=exp_cfg.country,
        solar_lat_lon=(GB_LAT, GB_LON),
        include_sun_elevation=False,
        include_clear_sky_ghi=False,
    )
    _, inference_channel_names = build_inference_window(
        combined, target_col, window_size=window_size,
        covariate_cols=raw_cov_cols if raw_cov_cols else None,
        add_temporal=True,
        future_features_df=future_features_df,
    )
    inference_channel_names = list(inference_channel_names)

    assert train_channel_names == inference_channel_names, (
        f"Channel-name mismatch between training and inference paths:\n"
        f"  train     = {train_channel_names}\n"
        f"  inference = {inference_channel_names}\n"
        f"Model would predict from mis-labelled channels at inference."
    )


# ------------------------------------------------------------------
# Past-only (non-extended) parity check: the benchmark path that
# the user has confirmed works on their data must keep working.
# This is the regression-test version of the holdout chart.
# ------------------------------------------------------------------
def test_pv_forecast_past_only_path_still_works(pv_data_no_covariates):
    """The benchmark/holdout training path uses past-only windows
    (NO extended_window). The user's holdout chart shows this path
    produces good predictions — so it must continue to.

    If extended_window IS the bug, this test passing while the
    extended tests above fail proves the localisation.
    """
    exp_cfg = _FakeExpCfg(
        target_is_nonnegative=True,
        log_transform=True,
    )
    df = pv_data_no_covariates.copy()
    if exp_cfg.log_transform:
        df['y'] = np.log1p(np.maximum(df['y'].values, 0.0))
    features_df = build_features(
        df, target_col='y', interval_minutes=exp_cfg.interval_minutes,
        country=exp_cfg.country,
    )
    combined = features_df.copy()
    combined['target'] = df['y']
    combined = combined.dropna()

    target_col = 'target'
    engineered = {
        'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_holiday',
    }
    engineered.update(c for c in combined.columns if c.startswith('y_lag_'))
    raw_cov_cols = [
        c for c in combined.columns if c not in engineered and c != target_col
    ]
    window_size = 48
    horizon_steps = list(range(1, exp_cfg.future_periods + 1))

    # Past-only — no future_features_df. Matches the benchmark path.
    torch.manual_seed(0)
    np.random.seed(0)
    seq_X, seq_y, channel_names = create_sliding_windows(
        combined, target_col, window_size=window_size,
        covariate_cols=raw_cov_cols if raw_cov_cols else None,
        add_temporal=True, horizon_steps=horizon_steps,
    )

    model = NLinearModel(epochs=20, batch_size=64)
    _apply_output_activation(model, exp_cfg)
    X_flat = np.zeros((seq_X.shape[0], 1), dtype=np.float32)
    model.fit(
        X_flat, seq_y,
        sequence_data=seq_X,
        channel_names=channel_names,
    )

    # Inference: build a past-only window of the last 48 rows.
    last_window, _ = build_inference_window(
        combined, target_col, window_size=window_size,
        covariate_cols=raw_cov_cols if raw_cov_cols else None,
        add_temporal=True,
    )
    multi_pred = model.predict_sequence(last_window).ravel()
    y_pred = multi_pred[:exp_cfg.future_periods].astype(np.float32)
    if exp_cfg.log_transform:
        y_pred = np.expm1(y_pred).astype(np.float32)
        y_pred = np.maximum(y_pred, 0.0)

    pv_peak = float(np.max(np.expm1(combined['target'].values)))
    _assert_forecast_not_collapsed(y_pred, pv_peak, label="pv_past_only")


# ------------------------------------------------------------------
# Save → load → predict roundtrip: the production cache writes the
# model to disk, the addon may restart, then a forecast cycle loads
# the model back from disk and calls predict_sequence. If save/load
# loses ANY state (past_window_size, channel_mean, y_mean, etc.) the
# reloaded model produces a wrong forecast — the in-memory tests
# above would not catch it because they predict directly from the
# freshly-trained model object.
# ------------------------------------------------------------------
def test_pv_forecast_save_load_predict_roundtrip(
    pv_data_no_covariates, tmp_path,
):
    """Train → save → reload → predict. Reloaded model must produce
    a forecast that is numerically identical (within float tolerance)
    to the in-memory model's forecast on the same input.

    A divergence here means model.save / model.load loses state —
    the user's flat-zero forecast could be caused by this if the
    cache loaded after addon restart is missing past_window_size,
    a y_mean offset, or similar.
    """
    exp_cfg = _FakeExpCfg(
        target_is_nonnegative=True,
        log_transform=True,
    )
    df = pv_data_no_covariates.copy()
    df['y'] = np.log1p(np.maximum(df['y'].values, 0.0))
    features_df = build_features(
        df, target_col='y', interval_minutes=exp_cfg.interval_minutes,
        country=exp_cfg.country,
    )
    combined = features_df.copy()
    combined['target'] = df['y']
    combined = combined.dropna()

    model, seq_kwargs, window_size, _ = _train_extended_window_nlinear(
        combined, exp_cfg, epochs=15,
    )
    y_pred_in_memory = _live_forecast(
        combined, model, seq_kwargs, window_size, exp_cfg,
    )

    # Save → reload (mirrors _retrain_and_cache's cache["model"].save(...)
    # plus the post-restart _forecast_with_cached model.load(...)).
    model_bin = tmp_path / "model.bin"
    model.save(str(model_bin))

    reloaded = NLinearModel()
    reloaded.load(str(model_bin))
    y_pred_reloaded = _live_forecast(
        combined, reloaded, seq_kwargs, window_size, exp_cfg,
    )

    # Numerical equivalence within float tolerance. If the reloaded
    # model output diverges from the in-memory model's output, save/load
    # has lost state.
    assert y_pred_reloaded.shape == y_pred_in_memory.shape
    max_diff = float(np.max(np.abs(y_pred_reloaded - y_pred_in_memory)))
    assert max_diff < 1e-3, (
        f"save/load roundtrip changes the forecast: "
        f"max abs diff = {max_diff:.6f}\n"
        f"  in-memory peak={np.max(y_pred_in_memory):.2f}, "
        f"min={np.min(y_pred_in_memory):.2f}\n"
        f"  reloaded peak ={np.max(y_pred_reloaded):.2f}, "
        f"min={np.min(y_pred_reloaded):.2f}\n"
        f"This pattern matches the user's flat-zero forecast if some "
        f"state (past_window_size, y_mean, channel_mean) is dropped "
        f"during load."
    )

    # And the reloaded model must still pass the collapse-signature
    # checks — it's the production-relevant assertion.
    pv_peak = float(np.max(np.expm1(combined['target'].values)))
    _assert_forecast_not_collapsed(
        y_pred_reloaded, pv_peak, label="pv_reloaded_from_disk"
    )


# ------------------------------------------------------------------
# Time-of-day inference robustness: the user's screenshot was taken
# mid-morning with current PV ~0.9W. The PF2 anchor mechanism uses
# the last past observation, so different inference times produce
# different anchors. Test that the forecast does not collapse
# regardless of which time-of-day we infer at.
# ------------------------------------------------------------------
@pytest.mark.parametrize(
    "inference_hour",
    [3, 8, 11, 14, 17, 22],
    ids=["pre-dawn", "morning-ramp", "mid-morning", "afternoon", "evening", "post-dusk"],
)
def test_pv_forecast_anchor_robust_across_time_of_day(
    pv_data_no_covariates, inference_hour,
):
    """The user's flat-zero screenshot was taken at mid-morning with
    anchor ≈ 0.9 W (near zero). The PF2 anchor mechanism makes the
    model's output sensitive to the anchor value: when anchor is
    small, the linear head needs strictly positive output to predict
    daytime values. Test that the forecast survives across a sweep
    of inference times-of-day.

    If a specific time-of-day produces a collapsed forecast, the
    failure localises to the anchor / RevIN interaction.
    """
    exp_cfg = _FakeExpCfg(
        target_is_nonnegative=True,
        log_transform=True,
    )
    df = pv_data_no_covariates.copy()
    df['y'] = np.log1p(np.maximum(df['y'].values, 0.0))
    features_df = build_features(
        df, target_col='y', interval_minutes=exp_cfg.interval_minutes,
        country=exp_cfg.country,
    )
    combined = features_df.copy()
    combined['target'] = df['y']
    combined = combined.dropna()

    # Train once on the full series, then slice the dataframe to end
    # at the requested hour-of-day so build_inference_window picks up
    # an anchor at that hour.
    model, seq_kwargs, window_size, _ = _train_extended_window_nlinear(
        combined, exp_cfg, epochs=15,
    )

    # Slice combined so its last row is at the chosen UTC hour.
    matching = combined.index.hour == inference_hour
    candidate_idx = np.where(matching)[0]
    # Use a candidate far enough from the start to have a full past
    # window available.
    candidate_idx = candidate_idx[candidate_idx >= window_size + 96]
    assert candidate_idx.size > 0, (
        f"No timestamp at hour={inference_hour} with full window. "
        f"Synthetic data shorter than expected?"
    )
    last_idx = int(candidate_idx[-1])
    combined_sliced = combined.iloc[: last_idx + 1].copy()
    y_pred = _live_forecast(
        combined_sliced, model, seq_kwargs, window_size, exp_cfg,
    )

    pv_peak = float(np.max(np.expm1(combined['target'].values)))
    _assert_forecast_not_collapsed(
        y_pred, pv_peak, label=f"pv_anchor_at_hour_{inference_hour:02d}"
    )


# ------------------------------------------------------------------
# User-tuned learning rate: the user's YAML has
# ``model_params.nlinear.learning_rate: 0.009480`` — 20× the v2.37
# default of 5e-4. Tuning was done against the pre-v2.37 NLinear,
# so the value may diverge or destabilise the new normalised /
# anchored architecture. This test reproduces the user's tuned-LR
# path and asserts the model still produces a non-collapsed forecast.
# ------------------------------------------------------------------
def test_pv_forecast_user_tuned_high_learning_rate(pv_data_no_covariates):
    """User's tuned ``learning_rate: 0.009480`` reproduces. If this
    test collapses while the same config with the default LR (5e-4)
    passes, the user's tuned hyperparameters are the culprit and
    they should reset ``model_params.nlinear``.
    """
    exp_cfg = _FakeExpCfg(
        target_is_nonnegative=True,
        log_transform=True,
        model_params={'nlinear': {'learning_rate': 0.009480163569127946}},
    )
    df = pv_data_no_covariates.copy()
    df['y'] = np.log1p(np.maximum(df['y'].values, 0.0))
    features_df = build_features(
        df, target_col='y', interval_minutes=exp_cfg.interval_minutes,
        country=exp_cfg.country,
    )
    combined = features_df.copy()
    combined['target'] = df['y']
    combined = combined.dropna()

    user_lr = exp_cfg.model_params['nlinear']['learning_rate']
    model, seq_kwargs, window_size, _ = _train_extended_window_nlinear(
        combined, exp_cfg, epochs=20, learning_rate=user_lr,
    )
    y_pred = _live_forecast(
        combined, model, seq_kwargs, window_size, exp_cfg,
    )
    pv_peak = float(np.max(np.expm1(combined['target'].values)))
    _assert_forecast_not_collapsed(y_pred, pv_peak, label="pv_tuned_lr")
