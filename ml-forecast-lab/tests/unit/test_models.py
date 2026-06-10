"""Tests for model backends (fit/predict roundtrip on synthetic data)."""

import numpy as np
import pytest


class TestLightGBM:
    def test_fit_predict_roundtrip(self):
        from ml_forecast_lab.models.lightgbm_backend import LightGBMModel
        rng = np.random.default_rng(42)
        X = rng.random((200, 10)).astype(np.float32)
        y = X[:, 0] * 2 + X[:, 1] + rng.normal(0, 0.1, 200).astype(np.float32)
        model = LightGBMModel()
        model.fit(X[:160], y[:160])
        preds = model.predict(X[160:])
        assert preds.shape == (40,)
        assert not np.any(np.isnan(preds))

    def test_supports_sample_weight(self):
        from ml_forecast_lab.models.lightgbm_backend import LightGBMModel
        rng = np.random.default_rng(42)
        X = rng.random((200, 5)).astype(np.float32)
        y = X[:, 0] + rng.normal(0, 0.1, 200).astype(np.float32)
        weights = np.ones(200, dtype=np.float32)
        model = LightGBMModel()
        model.fit(X, y, sample_weight=weights)
        assert model.is_fitted


class TestXGBoost:
    def test_fit_predict_roundtrip(self):
        from ml_forecast_lab.models.xgboost_backend import XGBoostModel
        rng = np.random.default_rng(42)
        X = rng.random((200, 10)).astype(np.float32)
        y = X[:, 0] * 2 + rng.normal(0, 0.1, 200).astype(np.float32)
        model = XGBoostModel()
        model.fit(X[:160], y[:160])
        preds = model.predict(X[160:])
        assert preds.shape == (40,)
        assert not np.any(np.isnan(preds))


class TestLSTM:
    def test_fit_predict_with_sequence_data(self):
        from ml_forecast_lab.models.lstm_backend import LSTMModel
        rng = np.random.default_rng(42)
        # Simulate sliding window data: (samples, window_size, channels)
        seq_data = rng.random((100, 24, 3)).astype(np.float32)
        y = rng.random(100).astype(np.float32)
        X_flat = rng.random((100, 10)).astype(np.float32)
        model = LSTMModel(hidden_size=16, num_layers=1, epochs=5, patience=3)
        result = model.fit(X_flat, y, sequence_data=seq_data)
        assert model.is_fitted
        assert "best_val_loss" in result

    def test_z_score_standardisation_stored(self):
        """With RevIN off, dataset-level z-score stats are fitted and stored."""
        from ml_forecast_lab.models.lstm_backend import LSTMModel
        rng = np.random.default_rng(42)
        seq_data = rng.random((100, 24, 3)).astype(np.float32)
        y = rng.random(100).astype(np.float32)
        X_flat = rng.random((100, 10)).astype(np.float32)
        model = LSTMModel(
            hidden_size=16, num_layers=1, epochs=3, patience=2,
            use_revin=False,
        )
        model.fit(X_flat, y, sequence_data=seq_data)
        assert model._channel_mean is not None
        assert model._channel_std is not None
        assert model._channel_mean.shape == (3,)

    def test_z_score_skipped_when_revin_enabled(self):
        """RevIN handles per-window normalisation, so the dataset-level z-score
        path must NOT run — applying both would double-normalise and wash out
        the per-instance signal RevIN relies on.
        """
        from ml_forecast_lab.models.lstm_backend import LSTMModel
        rng = np.random.default_rng(42)
        seq_data = rng.random((100, 24, 3)).astype(np.float32)
        y = rng.random(100).astype(np.float32)
        X_flat = rng.random((100, 10)).astype(np.float32)
        model = LSTMModel(
            hidden_size=16, num_layers=1, epochs=3, patience=2,
            use_revin=True,
        )
        model.fit(X_flat, y, sequence_data=seq_data)
        assert model._channel_mean is None
        assert model._channel_std is None

    def test_predict_clips_negative(self):
        from ml_forecast_lab.models.lstm_backend import LSTMModel
        rng = np.random.default_rng(42)
        seq_data = rng.random((100, 12, 2)).astype(np.float32)
        y = rng.random(100).astype(np.float32)
        X_flat = rng.random((100, 10)).astype(np.float32)
        model = LSTMModel(hidden_size=8, num_layers=1, epochs=3, patience=2)
        model.fit(X_flat, y, sequence_data=seq_data)
        preds = model.predict(X_flat[:10])
        assert (preds >= 0).all()


class TestCNN:
    def test_fit_predict_with_sequence_data(self):
        from ml_forecast_lab.models.cnn_backend import CNNModel
        rng = np.random.default_rng(42)
        seq_data = rng.random((100, 24, 3)).astype(np.float32)
        y = rng.random(100).astype(np.float32)
        X_flat = rng.random((100, 10)).astype(np.float32)
        model = CNNModel(n_filters=8, n_layers=2, epochs=5, patience=3)
        result = model.fit(X_flat, y, sequence_data=seq_data)
        assert model.is_fitted
        assert "best_val_loss" in result

    def test_z_score_standardisation_stored(self):
        """With RevIN off, dataset-level z-score stats are fitted and stored."""
        from ml_forecast_lab.models.cnn_backend import CNNModel
        rng = np.random.default_rng(42)
        seq_data = rng.random((100, 24, 3)).astype(np.float32)
        y = rng.random(100).astype(np.float32)
        X_flat = rng.random((100, 10)).astype(np.float32)
        model = CNNModel(
            n_filters=8, n_layers=2, epochs=3, patience=2,
            use_revin=False,
        )
        model.fit(X_flat, y, sequence_data=seq_data)
        assert model._channel_mean is not None
        assert model._channel_std is not None
        assert model._channel_mean.shape == (3,)

    def test_z_score_skipped_when_revin_enabled(self):
        """RevIN handles per-window normalisation; applying dataset-level
        z-score on top would double-normalise.
        """
        from ml_forecast_lab.models.cnn_backend import CNNModel
        rng = np.random.default_rng(42)
        seq_data = rng.random((100, 24, 3)).astype(np.float32)
        y = rng.random(100).astype(np.float32)
        X_flat = rng.random((100, 10)).astype(np.float32)
        model = CNNModel(
            n_filters=8, n_layers=2, epochs=3, patience=2,
            use_revin=True,
        )
        model.fit(X_flat, y, sequence_data=seq_data)
        assert model._channel_mean is None
        assert model._channel_std is None


class TestNLinear:
    def test_softplus_predict_non_negative(self):
        """NLinear with softplus head must produce non-negative predictions
        even when trained on data that crosses zero — the activation owns
        the floor so downstream log/expm1 inversion can't amplify a small
        negative bias into a kW-scale spurious peak."""
        from ml_forecast_lab.models.nlinear_backend import NLinearModel
        rng = np.random.default_rng(42)
        seq_data = rng.standard_normal((200, 24, 3)).astype(np.float32)
        y = rng.standard_normal((200, 8)).astype(np.float32) * 0.1
        X_flat = np.zeros((200, 72), dtype=np.float32)
        model = NLinearModel(
            epochs=4, patience=3, output_activation='softplus',
            use_revin=False,
        )
        model.fit(X_flat, y, sequence_data=seq_data)
        preds = model.predict_sequence(seq_data[:10])
        assert preds.shape == (10, 8)
        assert (preds >= 0).all(), f"softplus must floor at 0, got min={preds.min()}"


class TestInferenceWindowAlignment:
    """The inference window builder must produce a window whose LAST
    timestep is df.index[-1] (the most recent observation), and whose
    channels match the training-time ordering exactly.

    Pre-fix, the production forecast path called create_sliding_windows
    with horizon_steps=[1] on the last (window_size + 1) rows. That
    reserved the final row as an unused y-label and produced a window
    ending at df.iloc[-2] — a one-interval misalignment that shifted
    every published forecast prediction one interval later than the
    model intended. This test pins the corrected contract."""

    @staticmethod
    def _make_df(n=200, interval_min=30):
        import pandas as pd
        idx = pd.date_range('2026-05-15 00:00', periods=n, freq=f'{interval_min}min')
        return pd.DataFrame({
            'target': (idx.hour + idx.minute / 60.0).astype(np.float32),
            'cov_a': np.linspace(0, 1, n, dtype=np.float32),
            'cov_b': np.linspace(10, 20, n, dtype=np.float32),
        }, index=idx)

    def test_window_ends_at_last_index(self):
        from ml_forecast_lab.features import build_inference_window
        df = self._make_df()
        X, _ = build_inference_window(
            df, 'target', window_size=48,
            covariate_cols=['cov_a', 'cov_b'], add_temporal=True,
        )
        assert X.shape == (1, 48, 8)  # target + 2 cov + 5 temporal
        # The last timestep of the window's target channel MUST equal
        # df['target'].iloc[-1]. Pre-fix this assertion held on
        # df['target'].iloc[-2] instead, leaking the off-by-one into
        # production.
        assert X[0, -1, 0] == df['target'].iloc[-1]
        assert X[0, 0, 0] == df['target'].iloc[-48]

    def test_channel_order_matches_create_sliding_windows(self):
        """Inference channel ordering must match what was cached at
        training time. If these two helpers ever drift, the parity
        guard in _forecast_with_cached fires and forecasts stop
        publishing until retrain — so they MUST stay in lockstep."""
        from ml_forecast_lab.features import (
            build_inference_window, create_sliding_windows,
        )
        df = self._make_df()
        _, ch_inf = build_inference_window(
            df, 'target', window_size=48,
            covariate_cols=['cov_a', 'cov_b'], add_temporal=True,
        )
        _, _, ch_train = create_sliding_windows(
            df, 'target', window_size=48,
            covariate_cols=['cov_a', 'cov_b'], add_temporal=True,
            horizon_steps=[1],
        )
        assert ch_inf == ch_train

    def test_temporal_channels_reflect_actual_last_timestamp(self):
        """The window's last temporal-feature row must encode the hour
        at df.index[-1] (the model's anchor), not df.index[-2]. A
        one-step misalignment here would make the model interpret
        the inference as if 'now' were 30 minutes earlier — exactly
        the failure mode the original bug produced."""
        import pandas as pd
        from ml_forecast_lab.features import build_inference_window
        df = self._make_df()
        X, ch = build_inference_window(
            df, 'target', window_size=48,
            covariate_cols=None, add_temporal=True,
        )
        hour_sin_idx = ch.index('hour_sin')
        expected_hour_sin = float(np.sin(
            2 * np.pi * df.index[-1].hour / 24
        ))
        assert abs(float(X[0, -1, hour_sin_idx]) - expected_hour_sin) < 1e-6

    def test_too_few_rows_raises(self):
        from ml_forecast_lab.features import build_inference_window
        df = self._make_df(n=20)
        with __import__('pytest').raises(ValueError, match='at least 48 rows'):
            build_inference_window(df, 'target', window_size=48)


class TestChannelParityFilter:
    """The neural sliding-window builder must produce the SAME channel
    ordering at training and at inference, otherwise the model predicts
    from mis-labelled channels and the published forecast is silently
    wrong (e.g. peaks at the wrong hour of day).

    The filter in main.py:_retrain_and_cache + _forecast_with_cached is
    deterministic given the input DataFrame's column order, so the
    behavioural contract is: same df.columns + same exp_cfg → identical
    channel_names. This test pins that contract."""

    @staticmethod
    def _build_combined(target_col_order=None):
        """Replicate main.py's combined-frame construction so the test
        exercises the exact filter that production runs."""
        import pandas as pd
        from ml_forecast_lab.features import build_features
        idx = pd.date_range('2024-01-01', periods=500, freq='30min')
        y = np.sin(np.linspace(0, 20 * np.pi, 500)).astype(np.float32) + 1
        df = pd.DataFrame({'y': y}, index=idx)
        df['external_temperature'] = 18.0 + 2 * np.cos(np.linspace(0, 8 * np.pi, 500))
        df['sun_elevation'] = np.clip(
            60 * np.cos((idx.hour + idx.minute / 60.0 - 12) * np.pi / 12), 0, None
        )
        df['clear_sky_ghi'] = df['sun_elevation'] * 15
        if target_col_order is not None:
            # Re-order to simulate a different fetch order at inference time
            df = df[['y'] + [c for c in target_col_order if c in df.columns]]
        features_df = build_features(df, target_col='y', interval_minutes=30)
        combined = features_df.copy()
        combined['target'] = df['y']
        for col in [c for c in df.columns if c != 'y']:
            combined[col] = df[col]
        return combined.dropna()

    @staticmethod
    def _raw_cov_cols(combined):
        """Replicates the production filter used in both train and infer."""
        engineered = {
            'hour_of_day', 'day_of_week', 'is_weekend', 'month',
            'day_of_month', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'is_holiday',
        }
        engineered.update(c for c in combined.columns if c.startswith('y_lag_'))
        return [c for c in combined.columns
                if c not in engineered and c != 'target']

    def test_same_df_columns_produce_same_channel_order(self):
        a = self._build_combined()
        b = self._build_combined()
        assert self._raw_cov_cols(a) == self._raw_cov_cols(b)

    def test_reordered_covariates_produce_different_channels(self):
        """If the covariate fetch loop produces columns in a different
        order between train and inference — even with the exact same set
        of covariates — the channel ordering diverges. This is the
        scenario the channel-name parity guard in _forecast_with_cached
        is designed to catch."""
        train = self._raw_cov_cols(self._build_combined())
        # Inference fetched the same three covariates but in a different
        # order (sun_elevation and clear_sky_ghi swapped).
        infer = self._raw_cov_cols(self._build_combined(
            target_col_order=['external_temperature', 'clear_sky_ghi', 'sun_elevation']
        ))
        assert train != infer, (
            "If this assertion passes, the filter is order-stable and the "
            "parity guard is belt-and-braces. If it fails, the guard is "
            "load-bearing — keep it in."
        )


class TestResolveOutputActivation:
    """Tests for the 'auto' output_activation resolution rule (main.py)."""

    @staticmethod
    def _cfg(**overrides):
        from ml_forecast_lab.config import ExperimentCfg
        defaults = dict(
            name='t', target_entity='sensor.t',
            output_activation='auto',
            source_is_cumulative=False,
            log_transform=False,
        )
        defaults.update(overrides)
        return ExperimentCfg(**defaults)

    def test_lstm_always_zscore(self):
        from ml_forecast_lab.main import _resolve_output_activation
        assert _resolve_output_activation(self._cfg(), 'lstm') == 'zscore'
        assert _resolve_output_activation(
            self._cfg(log_transform=True), 'lstm'
        ) == 'zscore'

    def test_cumulative_source_picks_linear(self):
        """v2.41.0: 'auto' resolves to 'linear' even for cumulative /
        non-negative targets. The softplus auto-pick collapsed to flat
        zero once the cumulative-loss term stopped masking it (see
        tests/integration/test_pv_forecast_pipeline.py); non-negativity
        is now enforced by the publish-time clamp instead."""
        from ml_forecast_lab.main import _resolve_output_activation
        cfg = self._cfg(source_is_cumulative=True)
        assert _resolve_output_activation(cfg, 'nlinear') == 'linear'

    def test_log_transform_alone_picks_linear(self):
        """``log_transform=True`` alone does NOT trigger softplus — that's an
        explicit-opt-in for the user. ``log_transform`` only changes the
        target space; the post-hoc ``max(0, expm1(.))`` already enforces the
        physical floor for ``'linear'``. Softplus is reserved for cases
        where the user explicitly wants the network's output activation to
        own the non-negativity contract (e.g. cumulative-with-reset
        sensors where ``softplus`` is also Auto's choice).
        """
        from ml_forecast_lab.main import _resolve_output_activation
        cfg = self._cfg(log_transform=True)
        for name in ('nlinear', 'dlinear', 'tide', 'patchtst'):
            assert _resolve_output_activation(cfg, name) == 'linear', (
                f"{name} with log_transform=True (only) should stay 'linear'"
            )

    def test_neither_flag_picks_linear(self):
        from ml_forecast_lab.main import _resolve_output_activation
        cfg = self._cfg()
        assert _resolve_output_activation(cfg, 'nlinear') == 'linear'

    def test_explicit_activation_is_honoured(self):
        from ml_forecast_lab.main import _resolve_output_activation
        # Explicit choice wins over the auto rule, even if log_transform is on.
        cfg = self._cfg(log_transform=True, output_activation='linear')
        assert _resolve_output_activation(cfg, 'nlinear') == 'linear'



class TestEarlyStopStep:
    """v2.40.12: ``ForecastModel._step_early_stop`` is the shared
    helper every backend uses for early-stopping bookkeeping. The
    refinements over the pre-v2.40.12 strict-``<`` comparison are:

    1. ``min_delta`` margin — improvements smaller than 0.1 % (default)
       don't reset patience. Stops the "lucky tiny epoch" from
       extending training pointlessly past a real plateau.
    2. EMA-smoothed val_loss for the STOP decision — one noisy epoch
       doesn't reset patience and one lucky one doesn't extend it. The
       checkpoint still tracks the raw val_loss so the saved weights
       are the truly best ones.

    These tests pin both refinements.
    """

    def _step(self, **kw):
        from ml_forecast_lab.models.base import ForecastModel
        return ForecastModel._step_early_stop(**kw)

    def test_first_epoch_seeds_ema_and_resets_patience(self):
        out = self._step(
            val_loss=2.0,
            best_val_loss=float("inf"),
            best_val_loss_smoothed=float("inf"),
            val_loss_ema=None,
            patience_counter=99,
        )
        # First epoch: ema seeded to val_loss; raw + smoothed both
        # become "best"; patience resets.
        assert out["val_loss_ema"] == 2.0
        assert out["best_val_loss"] == 2.0
        assert out["best_val_loss_smoothed"] == 2.0
        assert out["patience_counter"] == 0
        assert out["checkpoint_best"] is True

    def test_tiny_improvement_below_min_delta_does_not_reset_patience(self):
        # Best smoothed is 2.0. A new val_loss producing smoothed
        # 1.99999 is technically less but only by 0.0005 % — below the
        # 0.1 % min_delta threshold. Patience must NOT reset.
        out = self._step(
            val_loss=1.99999, best_val_loss=2.0,
            best_val_loss_smoothed=2.0,
            val_loss_ema=1.99999,
            patience_counter=5,
            min_delta=1e-3,
            ema_alpha=1.0,  # no smoothing to keep maths trivial
        )
        assert out["patience_counter"] == 6, (
            "Tiny improvement (<min_delta) should NOT reset patience"
        )
        # Raw best still updates so the checkpoint catches it
        assert out["best_val_loss"] == 1.99999
        assert out["checkpoint_best"] is True
        # Smoothed best stays put
        assert out["best_val_loss_smoothed"] == 2.0

    def test_real_improvement_above_min_delta_resets_patience(self):
        # 0.5 % improvement, comfortably above min_delta=0.1 %.
        out = self._step(
            val_loss=1.99, best_val_loss=2.0,
            best_val_loss_smoothed=2.0,
            val_loss_ema=1.99,
            patience_counter=5,
            min_delta=1e-3,
            ema_alpha=1.0,
        )
        assert out["patience_counter"] == 0
        assert out["best_val_loss"] == 1.99
        assert out["best_val_loss_smoothed"] == 1.99

    def test_no_improvement_increments_patience(self):
        out = self._step(
            val_loss=2.5, best_val_loss=2.0,
            best_val_loss_smoothed=2.0,
            val_loss_ema=2.5,
            patience_counter=3,
            min_delta=1e-3,
            ema_alpha=1.0,
        )
        assert out["patience_counter"] == 4
        # Raw best unchanged (2.5 > 2.0)
        assert out["best_val_loss"] == 2.0
        assert out["checkpoint_best"] is False
        assert out["best_val_loss_smoothed"] == 2.0

    def test_ema_smooths_a_single_spike(self):
        # A one-epoch spike (2.0 → 5.0) with smoothing α=0.3 produces
        # EMA ≈ 0.3*5 + 0.7*2 = 2.9. Best raw checkpoint NOT updated
        # (5.0 > 2.0). Patience increments because smoothed didn't beat
        # best_smoothed (2.0).
        out = self._step(
            val_loss=5.0, best_val_loss=2.0,
            best_val_loss_smoothed=2.0,
            val_loss_ema=2.0,
            patience_counter=0,
            min_delta=1e-3,
            ema_alpha=0.3,
        )
        assert abs(out["val_loss_ema"] - 2.9) < 1e-9
        assert out["checkpoint_best"] is False
        assert out["patience_counter"] == 1

    def test_ema_lets_modest_noise_through_genuine_trend(self):
        # Noisy-but-trending-down path: 1.8 / 1.9 / 1.6 / 1.7 / 1.4 / 1.5
        # (mean step −0.06, alternating ±0.15 noise). With α=0.3 the EMA
        # descends monotonically. With the pre-v2.40.12 strict-``<``
        # comparison this path would have left patience accumulating on
        # every up-tick (1.9 > 1.8, 1.7 > 1.6, 1.5 > 1.4) — final
        # patience ≥ 1. With EMA + min_delta, patience stays at 0.
        ema = None
        best_raw = float("inf")
        best_smoothed = float("inf")
        patience = 0
        path = [1.8, 1.9, 1.6, 1.7, 1.4, 1.5]
        for v in path:
            out = self._step(
                val_loss=v, best_val_loss=best_raw,
                best_val_loss_smoothed=best_smoothed,
                val_loss_ema=ema,
                patience_counter=patience,
                min_delta=1e-3, ema_alpha=0.3,
            )
            ema = out["val_loss_ema"]
            best_raw = out["best_val_loss"]
            best_smoothed = out["best_val_loss_smoothed"]
            patience = out["patience_counter"]
        assert patience == 0, (
            f"EMA should rescue a noisy-but-genuinely-trending-down "
            f"path; got patience={patience} on path {path}"
        )

    def test_ema_alpha_one_recovers_legacy_behaviour(self):
        # α=1.0 means no smoothing: val_loss_ema == val_loss. Combined
        # with min_delta=0 this is the pre-v2.40.12 strict-``<`` path.
        out = self._step(
            val_loss=2.0001,
            best_val_loss=2.0, best_val_loss_smoothed=2.0,
            val_loss_ema=2.0,
            patience_counter=4,
            min_delta=0.0, ema_alpha=1.0,
        )
        # 2.0001 > 2.0 (no improvement, strict <)
        assert out["patience_counter"] == 5
        assert out["best_val_loss_smoothed"] == 2.0


class TestApplyPatience:
    """v2.40.12: ``_apply_patience`` plumbs the per-experiment Setting
    onto a backend's ``self.patience`` attribute. Called from every training-setup
    site (benchmark CV, holdout, production retrain, tuning) so
    backend default asymmetries (20 neural vs 50 tree) collapse to
    one uniform value when the user sets it."""

    def _make_cfg(self, patience=None):
        from types import SimpleNamespace
        return SimpleNamespace(patience=patience)

    def _make_model(self, patience=20):
        from types import SimpleNamespace
        return SimpleNamespace(patience=patience)

    def test_none_setting_leaves_backend_default(self):
        from ml_forecast_lab.main import _apply_patience
        cfg = self._make_cfg(patience=None)
        model = self._make_model(patience=20)
        _apply_patience(model, cfg)
        assert model.patience == 20, (
            "When the experiment Setting is None, the backend default "
            "must be preserved (was about to be overwritten to None)."
        )

    def test_explicit_setting_overrides_backend_default(self):
        from ml_forecast_lab.main import _apply_patience
        cfg = self._make_cfg(patience=35)
        model = self._make_model(patience=20)  # neural default
        _apply_patience(model, cfg)
        assert model.patience == 35

    def test_per_model_override_wins_over_experiment_setting(self):
        from ml_forecast_lab.main import _apply_patience
        cfg = self._make_cfg(patience=35)
        model = self._make_model(patience=20)
        _apply_patience(model, cfg, overrides={'patience': 99})
        # When the caller already pinned patience via overrides, the
        # experiment-level Setting must NOT clobber it.
        assert model.patience == 20

    def test_skipped_silently_on_backend_without_patience_attr(self):
        from ml_forecast_lab.main import _apply_patience
        from types import SimpleNamespace
        cfg = self._make_cfg(patience=35)
        # E.g. a hypothetical backend that doesn't do early stopping.
        model = SimpleNamespace()
        _apply_patience(model, cfg)
        # Must not raise, must not set a phantom attribute.
        assert not hasattr(model, 'patience')

    def test_uniform_across_neural_and_tree_defaults(self):
        """The whole point of the Setting: set it once, every backend
        in the experiment uses the same value regardless of its own
        constructor default (20 for neural, 50 for tree)."""
        from ml_forecast_lab.main import _apply_patience
        cfg = self._make_cfg(patience=40)
        neural = self._make_model(patience=20)
        tree = self._make_model(patience=50)
        _apply_patience(neural, cfg)
        _apply_patience(tree, cfg)
        assert neural.patience == 40
        assert tree.patience == 40
