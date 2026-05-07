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
