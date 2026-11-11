"""
Tweedie tree models are level-corrected and stop on their own metric
(v2.58.0).

Pins two findings of the 2026-11 intermittent-demand investigation.

A Tweedie objective boosts in log space, and an early-stopped log-space fit
has a lower arithmetic mean than the data it was fitted to: about 17 % lower
on an intermittent target, where a squared-error fit matches the training
mean at every round. The tree backends now scale a Tweedie model's
predictions by mean(y) / mean(fitted) on the training split.

LightGBM stopped every objective on validation RMSE. Tweedie is now
stopped on its deviance; Huber deliberately stays on RMSE, because on a
skewed target its own metric keeps improving while the level slides
towards the median.

The Tuning tab's tree ranges no longer reach a learning rate and tree count
whose product leaves the model effectively unboosted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab.features import build_features
from ml_forecast_lab.models.base import LEVEL_SCALE_BOUNDS, log_link_level_scale
from tests.synthetic import intermittent_demand as gen

lgb = pytest.importorskip("lightgbm")
from ml_forecast_lab.models.lightgbm_backend import LightGBMModel  # noqa: E402


def _xy(days=30, seed=0):
    y, _ = gen.slot_truth("2026-11-02", days, seed=seed)
    y.index = y.index.tz_localize(None)
    df = pd.DataFrame({"y": y})
    frame = build_features(df, "y", 30).assign(target=df["y"]).dropna()
    cols = [c for c in frame.columns if c != "target"]
    return (frame[cols].to_numpy(np.float32), frame["target"].to_numpy(np.float32), cols)


class TestLevelScale:
    def test_ratio_of_means(self):
        assert log_link_level_scale([1.0, 3.0], [1.0, 1.0]) == pytest.approx(2.0)
        assert log_link_level_scale([2.0, 2.0], [2.5, 2.5]) == pytest.approx(0.8)

    def test_clipped_and_safe(self):
        lo, hi = LEVEL_SCALE_BOUNDS
        assert log_link_level_scale([10.0], [1.0]) == hi
        assert log_link_level_scale([1.0], [10.0]) == lo
        assert log_link_level_scale([0.0, 0.0], [1.0, 1.0]) == 1.0
        assert log_link_level_scale([1.0], [0.0]) == 1.0
        assert log_link_level_scale([np.nan], [1.0]) == 1.0


class TestLightGBM:
    def test_tweedie_fit_matches_the_training_mean(self):
        X, y, cols = _xy()
        m = LightGBMModel(loss_fn="tweedie")
        m.fit(X, y, feature_names=cols)
        k = int(len(X) * 0.8)
        scale = m.training_metadata["level_scale"]
        raw = m.model.predict(X[:k])
        assert raw.mean() < 0.95 * y[:k].mean(), "the uncorrected fit should sit low"
        assert scale > 1.05
        assert m.predict(X[:k]).mean() == pytest.approx(y[:k].mean(), rel=1e-3)

    def test_other_objectives_are_not_scaled(self):
        X, y, cols = _xy()
        for loss in ("huber", "mse"):
            m = LightGBMModel(loss_fn=loss)
            m.fit(X, y, feature_names=cols)
            assert "level_scale" not in m.training_metadata
            np.testing.assert_allclose(m.predict(X[:50]), m.model.predict(X[:50]), rtol=1e-6)

    def test_scale_survives_save_and_load(self, tmp_path):
        X, y, cols = _xy()
        m = LightGBMModel(loss_fn="tweedie")
        m.fit(X, y, feature_names=cols)
        path = str(tmp_path / "m.pkl")
        m.save(path)
        again = LightGBMModel()
        again.load(path)
        assert again.training_metadata["level_scale"] == m.training_metadata["level_scale"]
        np.testing.assert_allclose(again.predict(X[:50]), m.predict(X[:50]), rtol=1e-6)

    def test_a_model_saved_before_the_correction_predicts_unscaled(self):
        X, y, cols = _xy()
        m = LightGBMModel(loss_fn="tweedie")
        m.fit(X, y, feature_names=cols)
        m.training_metadata.pop("level_scale")
        np.testing.assert_allclose(m.predict(X[:50]), m.model.predict(X[:50]), rtol=1e-6)

    @pytest.mark.parametrize("loss,metric", [
        ("tweedie", "tweedie"), ("huber", "rmse"), ("mse", "rmse"), ("mae", "rmse"),
    ])
    def test_early_stopping_metric(self, loss, metric):
        X, y, cols = _xy(days=12)
        m = LightGBMModel(loss_fn=loss, n_estimators=5)
        m.fit(X, y, feature_names=cols)
        assert m.model.params["metric"] == metric


class TestXGBoost:
    def test_tweedie_fit_matches_the_training_mean(self):
        pytest.importorskip("xgboost")
        from ml_forecast_lab.models.xgboost_backend import XGBoostModel
        X, y, cols = _xy()
        m = XGBoostModel(loss_fn="tweedie")
        m.fit(X, y, feature_names=cols)
        k = int(len(X) * 0.8)
        assert "level_scale" in m.training_metadata
        assert m.predict(X[:k]).mean() == pytest.approx(y[:k].mean(), rel=1e-3)
        plain = XGBoostModel(loss_fn="mse")
        plain.fit(X, y, feature_names=cols)
        assert "level_scale" not in plain.training_metadata


class TestTuningRanges:
    @pytest.mark.parametrize("backend", ["lightgbm", "xgboost", "catboost"])
    def test_ranges_cannot_leave_a_tree_model_unboosted(self, backend):
        from ml_forecast_lab.web import app as web_app
        import inspect
        src = inspect.getsource(web_app)
        start = src.index(f'        "{backend}": {{')
        block = src[start:src.index("        },", start)]
        assert '"label": "Number of trees", "min": 50' in block
        assert '"label": "Learning rate", "min": 0.01' in block
