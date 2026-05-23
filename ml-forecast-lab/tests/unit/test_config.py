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
        # production_metric is seasonal_mase post-H-1 (vs same-time-yesterday
        # baseline, matching the daily seasonality of typical HA sensors).
        assert cfg.production_metric == "seasonal_mase"
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


class TestRemoveExperimentCovariateDisambiguation:
    """v2.39.3 bug 3: remove_experiment_covariate must mirror the
    v2.38.2 add path's (entity, role, future_attribute, future_value_key)
    matching tuple. Removing without disambiguators when the same entity
    is configured multiple times silently stripped every matching row
    in one shot — a user clicking × on the 'temperature' row would lose
    the sibling 'cloud_coverage' row too."""

    def _write(self, tmp_path, covs):
        from ml_forecast_lab.config import atomic_yaml_write
        path = tmp_path / "mlfl.yaml"
        atomic_yaml_write(path, {
            "experiments": [{
                "name": "e1",
                "target_entity": "sensor.t",
                "covariates": covs,
            }],
        })
        return path

    def test_disambiguated_removal_keeps_sibling_row(self, tmp_path):
        from ml_forecast_lab.config import remove_experiment_covariate
        path = self._write(tmp_path, [
            {"entity": "weather.x", "role": "future",
             "future_attribute": "hourly", "future_value_key": "temperature"},
            {"entity": "weather.x", "role": "future",
             "future_attribute": "hourly", "future_value_key": "cloud_coverage"},
        ])
        removed = remove_experiment_covariate(
            path, "e1", "weather.x",
            role="future",
            future_attribute="hourly",
            future_value_key="temperature",
        )
        assert removed is True
        import yaml as _yaml
        with open(path) as f:
            data = _yaml.safe_load(f)
        covs = data["experiments"][0]["covariates"]
        assert len(covs) == 1
        assert covs[0]["future_value_key"] == "cloud_coverage"

    def test_undisambiguated_removal_with_multiple_same_entity_refuses(
        self, tmp_path, caplog,
    ):
        from ml_forecast_lab.config import remove_experiment_covariate
        path = self._write(tmp_path, [
            {"entity": "weather.x", "role": "future",
             "future_value_key": "temperature"},
            {"entity": "weather.x", "role": "future",
             "future_value_key": "cloud_coverage"},
        ])
        removed = remove_experiment_covariate(path, "e1", "weather.x")
        assert removed is False
        import yaml as _yaml
        with open(path) as f:
            data = _yaml.safe_load(f)
        # No data lost — both covariates still present.
        assert len(data["experiments"][0]["covariates"]) == 2

    def test_undisambiguated_removal_works_when_only_one_same_entity(
        self, tmp_path,
    ):
        """Backward-compat: the common case (one row per entity) still
        works with the legacy (config_path, exp, entity) signature."""
        from ml_forecast_lab.config import remove_experiment_covariate
        path = self._write(tmp_path, [
            {"entity": "sensor.foo", "role": "lagged"},
        ])
        assert remove_experiment_covariate(path, "e1", "sensor.foo") is True
        import yaml as _yaml
        with open(path) as f:
            data = _yaml.safe_load(f)
        assert data["experiments"][0]["covariates"] == []


class TestSameCovariateRespectsLaggedValueKey:
    """v2.39.3 bug N6: two lagged covariates of the same weather entity
    with different future_value_key values resolve to different
    attribute-history signals (covariates.py:139), so the dedup must
    treat them as distinct rather than blocking the second add."""

    def test_lagged_with_distinct_value_keys_are_not_dedup(self):
        from ml_forecast_lab.config import _same_covariate
        a = {"entity": "weather.x", "role": "lagged",
             "future_value_key": "temperature"}
        b = {"entity": "weather.x", "role": "lagged",
             "future_value_key": "cloud_coverage"}
        assert _same_covariate(a, b) is False

    def test_lagged_with_same_value_keys_still_dedup(self):
        from ml_forecast_lab.config import _same_covariate
        a = {"entity": "weather.x", "role": "lagged",
             "future_value_key": "temperature"}
        b = {"entity": "weather.x", "role": "lagged",
             "future_value_key": "temperature"}
        assert _same_covariate(a, b) is True
