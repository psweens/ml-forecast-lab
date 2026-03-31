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
        assert cfg.production_metric == "mae"
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
