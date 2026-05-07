"""Tests for configuration module."""

import pytest
import yaml
from pathlib import Path

from ml_forecast_lab.config import AppConfig, ExperimentCfg, CovariateCfg, load_config


class TestExperimentCfg:
    def test_defaults(self):
        cfg = ExperimentCfg(name="test", target_entity="sensor.test")
        assert cfg.interval_minutes == 30
        assert cfg.cv_folds == 5
        # Default flipped from mae → rmse pre-v2.27 (uses RMSE for ranking;
        # tracks MAE/RMSE/MASE for display) but the test wasn't updated.
        assert cfg.production_metric == "rmse"
        assert not cfg.source_is_cumulative

    def test_models_enabled_default(self):
        cfg = ExperimentCfg(name="test", target_entity="sensor.test")
        assert "lightgbm" in cfg.models_enabled
        assert "xgboost" in cfg.models_enabled

    def test_invalid_cv_strategy(self):
        with pytest.raises(ValueError):
            cfg = ExperimentCfg(
                name="test", target_entity="sensor.test",
                cv_strategy="invalid_strategy",
            )
            cfg.__post_init__()


class TestSelectedModel:
    """Persistence field for the Results-tab UI selection across restarts."""

    def test_default_is_none(self):
        cfg = ExperimentCfg(name="t", target_entity="sensor.t")
        assert cfg.selected_model is None

    def test_accepts_explicit_value(self):
        cfg = ExperimentCfg(
            name="t", target_entity="sensor.t",
            selected_model="xgboost",
        )
        assert cfg.selected_model == "xgboost"

    def test_roundtrips_through_yaml(self, tmp_path):
        import yaml
        config_data = {
            "experiments": [{
                "name": "t", "target_entity": "sensor.t",
                "selected_model": "xgboost",
            }],
        }
        p = tmp_path / "mlfl.yaml"
        p.write_text(yaml.dump(config_data))
        cfg = load_config(p)
        assert cfg.experiments[0].selected_model == "xgboost"


class TestStabilityFocus:
    def test_default_is_per_moment(self):
        cfg = ExperimentCfg(name="t", target_entity="sensor.t")
        assert cfg.stability_focus == "per_moment"

    def test_accepts_daily_total_on_cumulative(self):
        cfg = ExperimentCfg(
            name="t", target_entity="sensor.t",
            source_is_cumulative=True, stability_focus="daily_total",
        )
        assert cfg.stability_focus == "daily_total"

    def test_rejects_invalid_value(self):
        with pytest.raises(ValueError, match="stability_focus"):
            ExperimentCfg(
                name="t", target_entity="sensor.t",
                stability_focus="both",  # not in the allowed set
            )

    def test_rejects_daily_total_on_instantaneous(self):
        # Summing an instantaneous sensor over a day isn't a physical
        # quantity, so daily_total focus doesn't make sense there.
        with pytest.raises(ValueError, match="requires source_is_cumulative"):
            ExperimentCfg(
                name="t", target_entity="sensor.t",
                source_is_cumulative=False,
                stability_focus="daily_total",
            )


class TestClearForecastLogOnRetrain:
    def test_default_is_true(self):
        cfg = ExperimentCfg(name="t", target_entity="sensor.t")
        assert cfg.clear_forecast_log_on_retrain is True

    def test_can_opt_out(self):
        cfg = ExperimentCfg(
            name="t", target_entity="sensor.t",
            clear_forecast_log_on_retrain=False,
        )
        assert cfg.clear_forecast_log_on_retrain is False


class TestCovariateCfg:
    def test_valid_roles(self):
        for role in ('future', 'lagged', 'both', 'concurrent'):
            cfg = CovariateCfg(entity="sensor.test", role=role)
            assert cfg.role == role

    def test_invalid_role(self):
        with pytest.raises(ValueError, match="role must be one of"):
            cfg = CovariateCfg(entity="sensor.test", role="invalid")
            cfg.__post_init__()


class TestLoadConfig:
    def test_load_valid_config(self, tmp_path):
        config_data = {
            "update_every_minutes": 60,
            "experiments": [{
                "name": "test_exp",
                "target_entity": "sensor.test",
                "source_is_cumulative": True,
                "models_enabled": ["lightgbm"],
            }],
        }
        config_path = tmp_path / "mlfl.yaml"
        config_path.write_text(yaml.dump(config_data))
        cfg = load_config(config_path)
        assert len(cfg.experiments) == 1
        assert cfg.experiments[0].name == "test_exp"
        assert cfg.experiments[0].source_is_cumulative is True

    def test_missing_file(self, tmp_path):
        """Should handle missing file gracefully or raise."""
        with pytest.raises(Exception):
            load_config(tmp_path / "nonexistent.yaml")

    def test_unknown_fields_tolerated(self, tmp_path):
        """Config parser should tolerate unknown YAML fields."""
        config_data = {
            "update_every_minutes": 60,
            "unknown_field": "should_not_crash",
            "experiments": [{
                "name": "test",
                "target_entity": "sensor.test",
                "also_unknown": True,
            }],
        }
        config_path = tmp_path / "mlfl.yaml"
        config_path.write_text(yaml.dump(config_data))
        cfg = load_config(config_path)
        assert cfg.experiments[0].name == "test"
