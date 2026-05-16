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

    def test_cumulative_source_picks_softplus(self):
        from ml_forecast_lab.main import _resolve_output_activation
        cfg = self._cfg(source_is_cumulative=True)
        assert _resolve_output_activation(cfg, 'nlinear') == 'softplus'

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
