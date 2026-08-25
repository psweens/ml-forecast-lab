"""Shared cache tables are pruned at the longest retention (v2.50.0).

SQLite cache tables are keyed by entity, not by experiment, so two
experiments on the same sensor share one table. It used to be pruned at
whichever experiment happened to touch it last: an experiment with
`max_age: 30` alongside one with `max_age: 365` truncated the shared table
to 30 days every cycle, and since Home Assistant's recorder has its own
purge window those rows were unrecoverable.
"""
from __future__ import annotations

import pytest

from ml_forecast_lab.config import (
    AppConfig,
    CovariateCfg,
    ExperimentCfg,
    ExternalForecastCfg,
)
from ml_forecast_lab.covariates import CovariateResolver, cov_cache_raw_key
from ml_forecast_lab.db import HistoryDB
from ml_forecast_lab.main import _DEFAULT_CACHE_MAX_AGE_DAYS, MLForecastLabApp


@pytest.fixture
def app(tmp_db):
    a = MLForecastLabApp()
    a.history_db = HistoryDB(tmp_db)
    return a


def _exp(name, target, max_age, covariates=None, externals=None):
    return ExperimentCfg(
        name=name,
        target_entity=target,
        max_age=max_age,
        covariates=covariates or [],
        external_forecasts=externals or [],
    )


class TestSharedTables:
    def test_shared_target_prunes_at_the_larger_max_age(self, app):
        app.config = AppConfig(experiments=[
            _exp("short", "sensor.rate", 30),
            _exp("long", "sensor.rate", 365),
        ])
        table = app.history_db.safe_table_name("sensor.rate")

        assert app._retention_days_for_table(table) == 365

    def test_shared_covariate_prunes_at_the_larger_max_age(self, app):
        cov = [CovariateCfg(entity="sensor.temp", role="lagged")]
        app.config = AppConfig(experiments=[
            _exp("a", "sensor.one", 14, covariates=cov),
            _exp("b", "sensor.two", 400, covariates=cov),
        ])
        table = app.history_db.safe_table_name(
            cov_cache_raw_key("sensor.temp", None)
        )

        assert app._retention_days_for_table(table) == 400

    def test_entity_that_is_a_target_here_and_a_covariate_there(self, app):
        """Targets and covariates live in different namespaces, so each
        table takes the retention of the experiments that actually read
        it — the target table is not stretched by covariate use."""
        app.config = AppConfig(experiments=[
            _exp("target_side", "sensor.pv", 10),
            _exp("cov_side", "sensor.other", 500, covariates=[
                CovariateCfg(entity="sensor.pv", role="lagged"),
            ]),
        ])

        target_table = app.history_db.safe_table_name("sensor.pv")
        cov_table = app.history_db.safe_table_name(
            cov_cache_raw_key("sensor.pv", None)
        )

        assert app._retention_days_for_table(target_table) == 10
        assert app._retention_days_for_table(cov_table) == 500

    def test_state_mode_external_shares_the_target_namespace(self, app):
        app.config = AppConfig(experiments=[
            _exp("a", "sensor.x", 10),
            _exp("b", "sensor.y", 200, externals=[
                ExternalForecastCfg(entity_id="sensor.x", mode="state"),
            ]),
        ])
        table = app.history_db.safe_table_name("sensor.x")

        assert app._retention_days_for_table(table) == 200

    def test_attribute_mode_external_is_not_an_entity_keyed_table(self, app):
        """Attribute-mode externals log into `external_forecast_log`, which
        has its own retention setting — they must not claim a stake in the
        entity-keyed cache."""
        app.config = AppConfig(experiments=[
            _exp("a", "sensor.other", 5, externals=[
                ExternalForecastCfg(entity_id="sensor.solcast", mode="attribute"),
            ]),
        ])
        table = app.history_db.safe_table_name("sensor.solcast")

        assert app._retention_days_for_table(table) == _DEFAULT_CACHE_MAX_AGE_DAYS

    def test_distinct_attribute_keys_get_independent_retention(self, app):
        app.config = AppConfig(experiments=[
            _exp("a", "sensor.one", 20, covariates=[
                CovariateCfg(
                    entity="weather.met", role="future",
                    future_value_key="temperature",
                ),
            ]),
            _exp("b", "sensor.two", 300, covariates=[
                CovariateCfg(
                    entity="weather.met", role="future",
                    future_value_key="cloud_coverage",
                ),
            ]),
        ])
        db = app.history_db
        temp = db.safe_table_name(cov_cache_raw_key("weather.met", "temperature"))
        cloud = db.safe_table_name(
            cov_cache_raw_key("weather.met", "cloud_coverage")
        )

        assert temp != cloud
        assert app._retention_days_for_table(temp) == 20
        assert app._retention_days_for_table(cloud) == 300


class TestKeyAgreement:
    """The retention map computes covariate table names itself. If it
    disagrees with the resolver, every covariate table silently falls back
    to the default retention."""

    @pytest.mark.parametrize("entity,value_key", [
        ("sensor.simple", None),
        ("sensor.simple", "temperature"),
        ("weather.met_office_balsham", "cloud_coverage"),
        ("sensor.with-hyphen", None),
        ("sensor.trailing_underscore_", "wind_speed"),
    ])
    def test_map_key_matches_resolver_key(self, app, entity, value_key):
        resolver = CovariateResolver(None, history_db=app.history_db)

        assert app.history_db.safe_table_name(
            cov_cache_raw_key(entity, value_key)
        ) == resolver._cov_cache_table(entity, value_key)


class TestFallbacks:
    def test_unknown_table_gets_the_default(self, app):
        app.config = AppConfig(experiments=[_exp("a", "sensor.x", 30)])

        assert app._retention_days_for_table(
            "sensor_never_referenced"
        ) == _DEFAULT_CACHE_MAX_AGE_DAYS

    def test_no_config_gets_the_default(self, app):
        app.config = None

        assert app._retention_days_for_table(
            "sensor_x"
        ) == _DEFAULT_CACHE_MAX_AGE_DAYS

    def test_stub_config_gets_the_default(self, app):
        app.config = app._create_stub_config()

        assert app._retention_days_for_table(
            "sensor_not_in_stub"
        ) == _DEFAULT_CACHE_MAX_AGE_DAYS

    def test_no_history_db_gets_the_default(self, app):
        app.history_db = None
        app.config = AppConfig(experiments=[_exp("a", "sensor.x", 30)])

        assert app._retention_days_for_table(
            "sensor_x"
        ) == _DEFAULT_CACHE_MAX_AGE_DAYS

    def test_empty_table_name_gets_the_default(self, app):
        assert app._retention_days_for_table(
            None
        ) == _DEFAULT_CACHE_MAX_AGE_DAYS
        assert app._retention_days_for_table(
            ""
        ) == _DEFAULT_CACHE_MAX_AGE_DAYS

    @pytest.mark.parametrize("bad", [0, -5])
    def test_non_positive_max_age_is_clamped_not_honoured(self, app, bad):
        """`max_age` is unvalidated in config.py. Left alone, a hand-edited
        0 puts the cut-off in the future and deletes the whole table."""
        app.config = AppConfig(experiments=[_exp("a", "sensor.x", bad)])
        table = app.history_db.safe_table_name("sensor.x")

        assert app._retention_days_for_table(table) >= 1

    def test_experiment_with_no_target_does_not_crash(self, app):
        class _Bare:
            max_age = 90

        app.config = AppConfig(experiments=[_Bare()])

        assert app._retention_days_for_table(
            "sensor_x"
        ) == _DEFAULT_CACHE_MAX_AGE_DAYS


class TestLiveConfig:
    def test_retention_follows_a_config_reload(self, app):
        """`load_config` rebinds a new AppConfig on a timer and on every
        Settings edit. A value captured at construction would ignore the
        edit until restart."""
        table = app.history_db.safe_table_name("sensor.rate")
        app.config = AppConfig(experiments=[_exp("a", "sensor.rate", 30)])
        assert app._retention_days_for_table(table) == 30

        app.config = AppConfig(experiments=[_exp("a", "sensor.rate", 730)])

        assert app._retention_days_for_table(table) == 730

    def test_resolver_is_wired_to_the_live_bound_method(self, app):
        """The resolver must hold the bound method, not a snapshot."""
        resolver = CovariateResolver(
            None,
            history_db=app.history_db,
            retention_provider=app._retention_days_for_table,
        )
        table = resolver._cov_cache_table("sensor.temp", None)
        app.config = AppConfig(experiments=[
            _exp("a", "sensor.t", 30, covariates=[
                CovariateCfg(entity="sensor.temp", role="lagged"),
            ]),
        ])
        assert resolver._retention_days(table) == 30

        app.config = AppConfig(experiments=[
            _exp("a", "sensor.t", 730, covariates=[
                CovariateCfg(entity="sensor.temp", role="lagged"),
            ]),
        ])

        assert resolver._retention_days(table) == 730
