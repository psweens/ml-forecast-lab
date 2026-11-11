"""
Automatic versus configured ``production_metric`` (v2.58.0).

v2.58.0 selects non-negative targets on CRPS of the published forecast. A
first cut applied that to every non-negative experiment whose metric was
``mae``, ``mase`` or ``seasonal_mase`` — including experiments where the
user had chosen that metric, silently overriding it. Pins:

- an unset ``production_metric`` is automatic: it resolves to
  ``seasonal_mase`` for every consumer and is flagged as not explicit;
- a configured metric is flagged explicit and honoured;
- the derived flag is not a YAML key;
- the settings endpoint treats null as "remove the key" (automatic);
- a replay bundle's captured config resolves the metric the same way.
"""
from __future__ import annotations

import dataclasses

import pytest
import yaml

from ml_forecast_lab.benchmark.runner import BenchmarkRunner
from ml_forecast_lab.config import (
    DEFAULT_PRODUCTION_METRIC, ExperimentCfg, load_config,
)
from ml_forecast_lab.replay import _jsonable, _load_config


def _cfg(**kw):
    base = dict(name="e", target_entity="sensor.x")
    base.update(kw)
    return ExperimentCfg(**base)


def _write(tmp_path, exp):
    p = tmp_path / "mlfl.yaml"
    p.write_text(yaml.safe_dump({"experiments": [exp]}))
    return p


class TestResolution:
    def test_unset_is_automatic(self):
        c = _cfg()
        assert c.production_metric == DEFAULT_PRODUCTION_METRIC
        assert c.production_metric_explicit is False

    @pytest.mark.parametrize("metric", ["seasonal_mase", "mae", "rmse", "crps"])
    def test_configured_is_explicit(self, metric):
        c = _cfg(production_metric=metric)
        assert c.production_metric == metric
        assert c.production_metric_explicit is True

    def test_flag_reaches_the_runner_through_asdict(self):
        auto = BenchmarkRunner(
            dataclasses.asdict(_cfg(source_is_cumulative=True)), lambda *a, **k: None,
        )
        pinned = BenchmarkRunner(
            dataclasses.asdict(_cfg(source_is_cumulative=True, production_metric="mae")),
            lambda *a, **k: None,
        )
        assert auto.selects_on_published_forecast() is True
        assert pinned.selects_on_published_forecast() is False
        assert pinned.median_seeking_override() is True


class TestYaml:
    def test_absent_key_is_automatic(self, tmp_path):
        exp = load_config(_write(tmp_path, {"name": "e", "target_entity": "sensor.x"})).experiments[0]
        assert exp.production_metric_explicit is False

    def test_present_key_is_explicit_even_at_the_default_value(self, tmp_path):
        exp = load_config(_write(tmp_path, {
            "name": "e", "target_entity": "sensor.x",
            "production_metric": "seasonal_mase",
        })).experiments[0]
        assert exp.production_metric_explicit is True

    def test_flag_is_not_a_yaml_key(self, tmp_path, caplog):
        exp = load_config(_write(tmp_path, {
            "name": "e", "target_entity": "sensor.x",
            "production_metric_explicit": True,
        })).experiments[0]
        assert exp.production_metric_explicit is False
        assert "production_metric_explicit" in caplog.text


class TestReplayRoundTrip:
    @pytest.mark.parametrize("metric,explicit", [(None, False), ("mae", True)])
    def test_captured_config_resolves_the_same_way(self, metric, explicit):
        kw = {} if metric is None else {"production_metric": metric}
        captured = _jsonable(_cfg(**kw))
        _, exp = _load_config({}, captured)
        assert exp.production_metric_explicit is explicit
        assert exp.production_metric == (metric or DEFAULT_PRODUCTION_METRIC)
