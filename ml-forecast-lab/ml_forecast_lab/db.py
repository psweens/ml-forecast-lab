"""
SQLite history cache for ML Forecast Lab.

Stores historical entity states with efficient bulk insert and retrieval,
and automatic cleanup of old records.
"""

import functools
import logging
import math
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# v2.44.x: the Forecast Comparison tab treats a source as "warming up"
# (results inconclusive) until it has at least this many distinct days of
# overlapping data. Surfaced per-source and as a top-level flag so the UI
# can grey provisional lines, badge "n/7 days", and suppress the headline
# verdict until the comparison clears the gate.
EXTERNAL_COMPARISON_WARMUP_DAYS = 7

# Absolute sanity ceiling for a logged forecast value. No real HA sensor
# produces ~1e12; a value beyond this is a log-inversion blow-up (np.expm1 of a
# diverged log-space prediction → ~1e30 / inf), not a forecast. Guarded at the
# WRITE path (log_forecast) so a blow-up can never enter forecast_log and
# corrupt every analytics tab that reads/aggregates it — and reused by the
# read-side corruption filter as the no-actuals fallback cap.
FORECAST_ABS_SANITY = 1e12


def _locked(fn):
    """Serialise SQLite access via ``self._lock``.

    The connection is shared across threads with ``check_same_thread=False``,
    which CPython's sqlite3 module allows only when callers wrap each use of
    the connection / cursor in an external lock. ``self._lock`` is an
    ``RLock`` so methods that delegate to other locked helpers don't
    deadlock.
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return wrapper


class HistoryDB:
    """SQLite database for caching HA entity history."""

    def __init__(self, path: str = "/config/mlfl.db"):
        """
        Initialise database connection.

        Args:
            path: Path to SQLite database file
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False so asyncio.to_thread offloads can use
        # the connection; writes are serialized by _lock. WAL lets the
        # offloaded readers proceed while the publish-cycle writer owns
        # the connection, which is the whole point of moving the big
        # accuracy scans off the event loop.
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.Error as e:
            logger.warning(f"Could not enable WAL mode: {e}")
        self._lock = threading.RLock()
        self._ensure_schema_versions_table()
        logger.info(f"HistoryDB initialised at {self.path}")

    def _ensure_schema_versions_table(self) -> None:
        """Bookkeeping table for schema migrations.

        Each row records that a versioned migration has been applied. Future
        migrations should check this table (via ``_applied_versions()``) rather
        than re-inspecting ``PRAGMA table_info`` for every relevant table.
        """
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_versions (
                    version    INTEGER PRIMARY KEY,
                    applied_at TEXT    NOT NULL
                )
                """
            )
            self.conn.commit()

    def _applied_versions(self) -> set:
        """Return the set of schema versions already applied."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT version FROM schema_versions")
        return {row[0] for row in cursor.fetchall()}

    def _record_version(self, version: int) -> None:
        """Mark *version* as applied (idempotent)."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO schema_versions (version, applied_at) VALUES (?, ?)",
            (version, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
        )
        self.conn.commit()

    def safe_table_name(self, entity_id: str) -> str:
        """
        Convert entity ID to safe SQL table name.

        Args:
            entity_id: Entity ID (e.g. 'sensor.temperature')

        Returns:
            SQL-safe table name (e.g. 'sensor_temperature')
        """
        # Replace dots and hyphens with underscores, remove invalid chars
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", entity_id)
        # Ensure starts with letter or underscore
        safe = re.sub(r"^[0-9]", "_", safe)
        if not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]{0,127}', safe):
            raise ValueError(f"Invalid table name after sanitisation: {safe!r}")
        return safe

    @_locked
    def ensure_table(self, table_name: str) -> None:
        """
        Create table if it doesn't exist.

        Args:
            table_name: SQL table name
        """
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ds TEXT NOT NULL UNIQUE,
                value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ds ON {table_name}(ds)")
        self.conn.commit()
        logger.debug(f"Ensured table: {table_name}")

    @_locked
    def store_history(self, table_name: str, df: pd.DataFrame) -> int:
        """
        Bulk insert history records (ignore duplicates).

        Args:
            table_name: SQL table name
            df: DataFrame with columns ['ds', 'value']
               - ds: datetime or ISO string
               - value: float or None

        Returns:
            Number of rows inserted
        """
        if df.empty:
            return 0

        self.ensure_table(table_name)

        # Ensure ds is string ISO format
        df_copy = df.copy()
        if hasattr(df_copy["ds"], "dt"):
            df_copy["ds"] = df_copy["ds"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        else:
            df_copy["ds"] = pd.to_datetime(df_copy["ds"]).dt.strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )

        # Convert NaN to None for NULL insertion
        df_copy["value"] = df_copy["value"].where(pd.notna(df_copy["value"]), None)

        records = df_copy[["ds", "value"]].values.tolist()

        cursor = self.conn.cursor()
        try:
            cursor.executemany(
                f"INSERT OR IGNORE INTO {table_name} (ds, value) VALUES (?, ?)",
                records,
            )
            self.conn.commit()
            inserted = cursor.rowcount
            logger.debug(f"Inserted {inserted} records into {table_name}")
            return inserted
        except sqlite3.Error as e:
            logger.error(f"Error inserting into {table_name}: {e}", exc_info=True)
            self.conn.rollback()
            return 0

    @_locked
    def get_history(self, table_name: str) -> pd.DataFrame:
        """
        Retrieve all stored history for a table.

        Args:
            table_name: SQL table name

        Returns:
            DataFrame with columns ['ds', 'y'] (value renamed to y for consistency)
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute(f"SELECT ds, value FROM {table_name} ORDER BY ds")
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            # Table doesn't exist
            return pd.DataFrame(columns=["ds", "y"])

        if not rows:
            return pd.DataFrame(columns=["ds", "y"])

        df = pd.DataFrame(rows, columns=["ds", "value"])
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.rename(columns={"value": "y"})

        return df

    @_locked
    def cleanup(self, table_name: str, oldest_datetime: datetime) -> int:
        """
        Delete records older than specified datetime.

        Args:
            table_name: SQL table name
            oldest_datetime: Records before this are deleted

        Returns:
            Number of rows deleted
        """
        oldest_str = oldest_datetime.strftime("%Y-%m-%d %H:%M:%S")

        cursor = self.conn.cursor()
        try:
            cursor.execute(
                f"DELETE FROM {table_name} WHERE ds < ?", (oldest_str,)
            )
            self.conn.commit()
            deleted = cursor.rowcount
            logger.info(f"Deleted {deleted} old records from {table_name}")
            return deleted
        except sqlite3.Error as e:
            logger.error(f"Error cleaning up {table_name}: {e}", exc_info=True)
            self.conn.rollback()
            return 0

    # ------------------------------------------------------------------
    # Forecast evolution log
    # ------------------------------------------------------------------

    @_locked
    def ensure_forecast_log_table(self) -> None:
        """Create the forecast_log table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecast_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment   TEXT    NOT NULL,
                model_name   TEXT    NOT NULL,
                issued_at    TEXT    NOT NULL,
                target_dt    TEXT    NOT NULL,
                lead_minutes INTEGER NOT NULL,
                predicted    REAL    NOT NULL,
                forecast_type TEXT   NOT NULL DEFAULT 'cached',
                upper        REAL,
                lower        REAL,
                model_version TEXT
            )
        """)
        # Migrate pre-existing tables that don't have the upper/lower
        # columns. Idempotency on a clean install is preserved by
        # checking schema_versions first; on an existing install with no
        # schema_versions row we still fall back to PRAGMA table_info so
        # the migration is robust across the rollout boundary.
        applied = self._applied_versions()
        cursor.execute("PRAGMA table_info(forecast_log)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        migrated: list = []
        if 1 not in applied:
            if "upper" not in existing_cols:
                cursor.execute("ALTER TABLE forecast_log ADD COLUMN upper REAL")
                migrated.append("upper")
            if "lower" not in existing_cols:
                cursor.execute("ALTER TABLE forecast_log ADD COLUMN lower REAL")
                migrated.append("lower")
            # `model_version` distinguishes weight regimes of a model that
            # keeps the same name across retrains. Without it, the stability
            # metric silently pools predictions from weights v1 and v2 under
            # model_name='lgb' and reports that as "run-to-run disagreement".
            # NULL is a legitimate legacy marker — rows predating this
            # migration have no version and are filtered out once a versioned
            # cohort appears for that experiment (see the analytics queries).
            if "model_version" not in existing_cols:
                cursor.execute(
                    "ALTER TABLE forecast_log ADD COLUMN model_version TEXT"
                )
                migrated.append("model_version")
            self._record_version(1)
        if migrated:
            logger.info(
                f"Migrated forecast_log to schema v1: added column(s) {migrated}"
            )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_flog_exp_target "
            "ON forecast_log(experiment, target_dt)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_flog_exp_issued "
            "ON forecast_log(experiment, issued_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_flog_exp_model_ver "
            "ON forecast_log(experiment, model_name, model_version)"
        )
        self.conn.commit()
        logger.debug("Ensured forecast_log table")

    @_locked
    def log_forecast(
        self,
        experiment: str,
        issued_at: datetime,
        targets: list,
        predictions: list,
        model_name: str,
        forecast_type: str = "cached",
        upper_bounds: Optional[list] = None,
        lower_bounds: Optional[list] = None,
        model_version: Optional[str] = None,
    ) -> int:
        """
        Bulk-insert a forecast snapshot into the log.

        Parameters
        ----------
        experiment : str
            Experiment name.
        issued_at : datetime
            Wall-clock UTC time the forecast was produced.
        targets : list of datetime or Timestamp
            Future timestamps being predicted.
        predictions : list of float
            Predicted values (same length as targets).
        model_name : str
            Name of the model that produced the forecast.
        forecast_type : str
            'retrain' or 'cached'.
        upper_bounds, lower_bounds : list of float, optional
            Conformal interval bounds (same length as predictions).
            Pass None when no intervals are available — stored as NULL
            so the coverage query can tell calibrated rows from legacy
            point-only rows.
        model_version : str, optional
            Opaque version tag distinguishing weight regimes of the same
            ``model_name``. Typically the ISO-timestamp of the last
            training completion. When provided, analytics queries will
            default to filtering on this so post-retrain cycles don't
            pool with pre-retrain ones. Pass None on legacy/uninitialised
            rows — they'll sort as their own implicit cohort.

        Returns
        -------
        int
            Number of rows inserted.
        """
        issued_str = issued_at.strftime("%Y-%m-%d %H:%M:%S")
        n = len(targets)
        if upper_bounds is not None and len(upper_bounds) != n:
            raise ValueError("upper_bounds length must match targets")
        if lower_bounds is not None and len(lower_bounds) != n:
            raise ValueError("lower_bounds length must match targets")
        rows = []
        n_dropped = 0
        for i, (ts, val) in enumerate(zip(targets, predictions)):
            fval = float(val)
            # WRITE-PATH BLOW-UP GUARD. forecast_log is the single source every
            # analytics tab reads (Forecast Accuracy, Comparison, trajectory,
            # evolution, stability — several aggregate `predicted` directly in
            # SQL via AVG/MAX). A non-finite or absurd value (a log-inversion
            # divergence: np.expm1 of a diverged log-space prediction → ~1e30 /
            # inf) would corrupt every one of those, so it never enters the log.
            # Normal forecasts are already clamped far below this upstream, so
            # this only ever fires on a pathological divergence — defence in
            # depth, not the primary clamp.
            if not math.isfinite(fval) or abs(fval) > FORECAST_ABS_SANITY:
                n_dropped += 1
                continue
            target_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            lead_min = int((ts - issued_at).total_seconds() / 60)
            upper_val = float(upper_bounds[i]) if upper_bounds is not None else None
            lower_val = float(lower_bounds[i]) if lower_bounds is not None else None
            # Drop a non-finite / absurd band to NULL rather than the whole row
            # (the point forecast is still usable; only the interval is bad).
            if upper_val is not None and (
                not math.isfinite(upper_val) or abs(upper_val) > FORECAST_ABS_SANITY
            ):
                upper_val = None
            if lower_val is not None and (
                not math.isfinite(lower_val) or abs(lower_val) > FORECAST_ABS_SANITY
            ):
                lower_val = None
            rows.append((
                experiment, model_name, issued_str, target_str,
                lead_min, fval, forecast_type, upper_val, lower_val,
                model_version,
            ))
        if n_dropped:
            logger.warning(
                "log_forecast[%s/%s]: dropped %d non-finite/absurd predicted "
                "value(s) (>|%.0e|) before insert — a log-inversion blow-up "
                "never reaches the analytics tabs.",
                experiment, model_name, n_dropped, FORECAST_ABS_SANITY,
            )
        cursor = self.conn.cursor()
        try:
            cursor.executemany(
                "INSERT INTO forecast_log "
                "(experiment, model_name, issued_at, target_dt, lead_minutes, "
                "predicted, forecast_type, upper, lower, model_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self.conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f"Error logging forecast for {experiment}: {e}", exc_info=True)
            self.conn.rollback()
            return 0

    def probe_forecast_rows(
        self,
        experiment: str,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        max_age_days: int = 30,
    ) -> bool:
        """Cheap EXISTS check for forecast_log rows matching a filter.

        Used by the web layer to pick the narrowest filter that has any
        data *before* running the expensive accuracy query, instead of
        calling the full query up to three times when the strict filter
        returns empty. Served by idx_flog_exp_model_ver — O(log N).

        We also require ``target_dt <= now`` so a freshly-retrained
        cohort (only future-targeting predictions, no actuals possible
        yet) doesn't look "alive" to the probe — without this, the
        widening ladder won't kick in for the first ~horizon-worth of
        time after every retrain, and the user sees an empty chart even
        though older versions of the same model would render fine.

        False positives (rows exist in forecast_log but none land in
        the accuracy INNER JOIN with actuals) still trigger one
        redundant full query, but that was the baseline cost anyway.
        """
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cutoff_str = (
            datetime.utcnow() - pd.Timedelta(days=max_age_days)
        ).strftime("%Y-%m-%d %H:%M:%S")
        where = ["experiment = ?", "issued_at >= ?", "target_dt <= ?"]
        params: list = [experiment, cutoff_str, now_str]
        if model_name:
            where.append("model_name = ?")
            params.append(model_name)
        if model_version:
            where.append("model_version = ?")
            params.append(model_version)
        try:
            with self._lock:
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT EXISTS(SELECT 1 FROM forecast_log WHERE "
                    + " AND ".join(where) + ")",
                    params,
                )
                row = cursor.fetchone()
                return bool(row and row[0])
        except sqlite3.Error as e:
            logger.warning(
                f"probe_forecast_rows({experiment}) failed: {e}",
                exc_info=True,
            )
            # On probe failure, claim rows exist so the caller still
            # runs the full query with the strict filter — falling
            # back to a correctness-preserving path rather than
            # silently widening the scope.
            return True

    def _materialise_actuals_grid(
        self,
        cursor,
        actuals_table: str,
        interval_sec: int,
        since_str: Optional[str] = None,
        increment: bool = False,
    ) -> bool:
        """(Re)build the indexed TEMP table the analytics joins read from.

        Creates ``_mlfl_actuals_grid_tmp(grid_dt, value)`` — raw actuals
        snapped to the interval grid — and, when ``increment`` is set,
        additionally ``_mlfl_actuals_vals_tmp(grid_dt, value)`` holding
        the adjacency-guarded per-interval deltas (NULL across gaps).

        Every analytics query used to inline the grid aggregation as a
        WITH-clause CTE. SQLite executes those as a co-routine that is
        re-scanned for EVERY outer forecast_log row, making the joins
        O(N_forecasts × N_actuals) with per-row strftime work — measured
        at 78 s (conformal) / 240 s (accuracy) on one experiment-month
        of forecast_log vs ~0.4 s with the materialised, indexed temp
        table (audit F3/F4; the v2.39.3 coverage fix pioneered the
        pattern, this generalises it). ``since_str`` bounds the actuals
        scan: join keys satisfy target_dt >= issued_at >= cutoff, so
        earlier actuals can never match and scanning them only adds
        cost that grows with ``max_age`` (up to 365 days) for no result.

        Callers run under ``self._lock`` (single shared connection), so
        the fixed table names cannot race. Returns False on failure —
        callers should bail out with their empty-result shape. The temp
        tables are dropped and rebuilt at the start of every call (and
        die with the connection), so callers may leave them in place
        after use; ``_drop_actuals_grid`` exists for callers that want
        to reclaim the memory eagerly.
        """
        try:
            cursor.execute("DROP TABLE IF EXISTS _mlfl_actuals_grid_tmp")
            cursor.execute("DROP TABLE IF EXISTS _mlfl_actuals_vals_tmp")
            # Built lazily by the daily-cumulative accuracy path for
            # non-cumulative sensors — drop any stale copy on every rebuild.
            cursor.execute("DROP TABLE IF EXISTS _mlfl_actuals_cum_tmp")
            since_sql = "WHERE SUBSTR(ds, 1, 19) >= ?" if since_str else ""
            since_params = (since_str,) if since_str else ()
            cursor.execute(
                f"""
                CREATE TEMP TABLE _mlfl_actuals_grid_tmp AS
                SELECT
                    strftime('%Y-%m-%d %H:%M:%S',
                        (CAST(strftime('%s', SUBSTR(ds, 1, 19)) AS INTEGER) / ?) * ?,
                        'unixepoch') AS grid_dt,
                    AVG(value) AS value
                FROM {actuals_table}
                {since_sql}
                GROUP BY grid_dt
                """,
                (interval_sec, interval_sec, *since_params),
            )
            cursor.execute(
                "CREATE INDEX _mlfl_actuals_grid_tmp_idx "
                "ON _mlfl_actuals_grid_tmp(grid_dt)"
            )
            if increment:
                cursor.execute(
                    """
                    CREATE TEMP TABLE _mlfl_actuals_vals_tmp AS
                    SELECT grid_dt,
                        CASE
                          WHEN CAST(strftime('%s', grid_dt) AS INTEGER)
                               - CAST(strftime('%s', LAG(grid_dt) OVER (ORDER BY grid_dt)) AS INTEGER)
                               = ?
                          THEN value - LAG(value) OVER (ORDER BY grid_dt)
                          ELSE NULL
                        END AS value
                    FROM _mlfl_actuals_grid_tmp
                    """,
                    (interval_sec,),
                )
                cursor.execute(
                    "CREATE INDEX _mlfl_actuals_vals_tmp_idx "
                    "ON _mlfl_actuals_vals_tmp(grid_dt)"
                )
            return True
        except sqlite3.Error as e:
            logger.warning(f"Failed to build temp actuals grid: {e}")
            return False

    @staticmethod
    def _drop_actuals_grid(cursor) -> None:
        """Drop the temp tables built by ``_materialise_actuals_grid``."""
        try:
            cursor.execute("DROP TABLE IF EXISTS _mlfl_actuals_grid_tmp")
            cursor.execute("DROP TABLE IF EXISTS _mlfl_actuals_vals_tmp")
            cursor.execute("DROP TABLE IF EXISTS _mlfl_actuals_cum_tmp")
        except sqlite3.Error:
            pass

    def get_forecast_accuracy(
        self,
        experiment: str,
        actuals_table: str,
        max_age_days: int = 30,
        interval_minutes: int = 30,
        evaluation_mode: str = "raw",
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        day_offset_hours: Optional[float] = None,
        cumulative_source: bool = True,
    ) -> dict:
        """
        Compute forecast accuracy by lead time.

        Joins forecast_log predictions against the actuals table and
        returns MAE/RMSE/ME grouped by lead-time buckets, plus revision
        improvement data (first vs last forecast for each target).

        Parameters
        ----------
        experiment : str
            Experiment name.
        actuals_table : str
            SQL-safe table name for the actuals.
        max_age_days : int
            Only consider forecasts issued in the last N days.
        interval_minutes : int
            Resampling grid interval in minutes. Raw actuals are snapped
            to the nearest grid boundary before joining against forecast
            targets (which are already grid-aligned).
        evaluation_mode : str
            "raw" — evaluate against stored cumulative/point values.
            "increment" — evaluate against per-interval deltas (LAG-based
            diff of both forecasts and actuals). Only meaningful for
            cumulative sensors where raw-value errors mostly reflect the
            sensor's shape through the day rather than model skill.
        model_name : str, optional
            Restrict to predictions from one model. Without this, a
            champion swap mid-window mixes old and new model residuals
            and misattributes the transition to either model.

        Returns
        -------
        dict with keys: lead_time_curve, revision_improvement,
              total_logged, actuals_matched, date_range
        """
        # Serialize cursor use because this method may be called from a
        # thread pool worker (asyncio.to_thread) while the event loop
        # issues writes on the same connection. RLock so nested helpers
        # that also lock don't deadlock.
        with self._lock:
            if evaluation_mode == "daily_cumulative":
                return self._get_forecast_accuracy_daily_cumulative_locked(
                    experiment, actuals_table, max_age_days,
                    interval_minutes, model_name, model_version,
                    day_offset_hours=day_offset_hours,
                    cumulative_source=cumulative_source,
                )
            return self._get_forecast_accuracy_locked(
                experiment, actuals_table, max_age_days, interval_minutes,
                evaluation_mode, model_name, model_version,
            )

    def _get_forecast_accuracy_locked(
        self,
        experiment: str,
        actuals_table: str,
        max_age_days: int,
        interval_minutes: int,
        evaluation_mode: str,
        model_name: Optional[str],
        model_version: Optional[str],
    ) -> dict:
        cursor = self.conn.cursor()
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cutoff_str = (
            datetime.utcnow() - pd.Timedelta(days=max_age_days)
        ).strftime("%Y-%m-%d %H:%M:%S")

        # Check actuals table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (actuals_table,),
        )
        if not cursor.fetchone():
            return {
                "error": "No actuals data available yet",
                "empty_reason": "no_actuals",
            }

        interval_sec = interval_minutes * 60
        bucket_min = max(1, int(interval_minutes))
        increment = evaluation_mode == "increment"
        # Optional filters, spliced directly into forecast_vals so both
        # the lead-time curve and revision query apply them. Combined
        # into a single clause to keep parameter ordering simple.
        _filter_clauses = []
        _filter_params: list = []
        if model_name:
            _filter_clauses.append("model_name = ?")
            _filter_params.append(model_name)
        if model_version:
            _filter_clauses.append("model_version = ?")
            _filter_params.append(model_version)
        model_filter_sql = (" AND " + " AND ".join(_filter_clauses)) if _filter_clauses else ""
        model_filter_param = tuple(_filter_params)

        # Build value-extraction CTEs according to mode. In "raw" mode we
        # just pass the stored value through; in "increment" mode we
        # convert the *actuals* (raw cumulative readings) into per-
        # interval deltas via LAG so the join lands in delta space.
        #
        # The *forecast* side is ALREADY in delta space for cumulative
        # sensors — `forecast_log.predicted` is written from y_pred,
        # which is the model's per-interval delta output (see
        # main.py:5068; the HA cumulative sensor is built downstream by
        # cumsumming y_pred and is never logged). So the forecast CTE
        # must PASS predicted through unchanged in BOTH modes. The
        # trajectory function (db.py:1040-1053, 1084) gets this right;
        # this function used to take a second LAG diff on the forecast,
        # producing a 2nd-difference compared against a 1st-difference
        # actual — a perfect model then scored MAE ≈ typical demand and
        # the subsequent `fv.value >= 0` filter silently dropped any row
        # where the 2nd difference went negative. v2.40.7 fix.
        #
        # Midnight resets on daily-cumulative sensors produce a large
        # negative actuals increment (e.g. 0 - 85 = -85). Increment mode
        # filters `av.value >= 0` to drop those rows; safe because the
        # mode is gated on source_is_cumulative upstream.
        #
        # We also null out the actuals delta when the previous grid row
        # is not exactly one interval earlier — otherwise an HA outage
        # causes e.g. a 2-hour span to be treated as a single-interval
        # demand, inflating MAE with data-availability artefacts rather
        # than model error. The adjacency check compares unix-epoch
        # seconds of the stringified grid_dt.
        # v2.41.0 (audit F4): the grid-aligned actuals — and, in
        # increment mode, their adjacency-guarded deltas — are
        # materialised ONCE into indexed temp tables instead of being
        # inlined as WITH-clause CTEs in every query below. The CTE
        # form executed as a co-routine re-scanned per forecast_log
        # row: O(N_forecasts × N_actuals) per query, three queries per
        # call, 240 s measured on one experiment-month while holding
        # the DB lock. ``actuals_vals`` stays as a trivial CTE alias so
        # the query text reads the same in both modes; SQLite flattens
        # it onto the indexed temp table.
        if not self._materialise_actuals_grid(
            cursor, actuals_table, interval_sec, cutoff_str,
            increment=increment,
        ):
            return {"error": "could not build actuals grid"}

        if increment:
            actuals_vals_cte = (
                "actuals_vals AS "
                "(SELECT grid_dt, value FROM _mlfl_actuals_vals_tmp)"
            )
            # Only actuals can be NULL (adjacency guard) or negative
            # (midnight reset). Forecasts are passthrough so neither
            # check applies on the fv side — applying ≥0 to the forecast
            # was the silent half of the previous bug, hiding rows where
            # the spurious 2nd difference went negative.
            mode_filter = (
                "AND av.value IS NOT NULL AND av.value >= 0"
            )
        else:
            actuals_vals_cte = (
                "actuals_vals AS "
                "(SELECT grid_dt, value FROM _mlfl_actuals_grid_tmp)"
            )
            mode_filter = ""

        # Forecast passthrough — predicted is already a per-interval
        # delta when source_is_cumulative (the only case where
        # increment mode is offered).
        forecast_vals_cte = (
            "forecast_vals AS ("
            "  SELECT experiment, model_name, model_version,"
            "         issued_at, target_dt, lead_minutes,"
            "         predicted AS value"
            "  FROM forecast_log"
            "  WHERE experiment = ? AND target_dt <= ? AND issued_at >= ?"
            f"  {model_filter_sql}"
            ")"
        )
        forecast_vals_params = (
            experiment, now_str, cutoff_str, *model_filter_param,
        )

        # --- Lead-time accuracy curve ---
        # Bucket lead_minutes into `interval_minutes`-sized bins so the
        # chart resolution matches the forecast grid. Hardcoding a 30-min
        # bucket loses resolution on fine-grained experiments (e.g. 5-min
        # interval with a 1h horizon collapses to only 3 buckets, which
        # also drives MAE→RMSE convergence within each large bucket).
        # Actuals are stored with raw irregular HA timestamps, while
        # forecast targets are grid-aligned. Snap actuals to the grid
        # (floor to nearest interval boundary) and average before joining.
        try:
            cursor.execute(f"""
                WITH {actuals_vals_cte},
                {forecast_vals_cte}
                SELECT
                    CAST((fv.lead_minutes / ?) * ? AS INTEGER) AS lead_bucket,
                    AVG(ABS(fv.value - av.value)) AS mae,
                    SQRT(AVG((fv.value - av.value) * (fv.value - av.value))) AS rmse,
                    AVG(fv.value - av.value) AS me,
                    COUNT(*) AS n
                FROM forecast_vals fv
                INNER JOIN actuals_vals av
                    ON av.grid_dt = fv.target_dt
                WHERE 1=1 {mode_filter}
                GROUP BY lead_bucket
                ORDER BY lead_bucket
            """, (
                *forecast_vals_params,
                bucket_min, bucket_min,
            ))
            lead_rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Forecast accuracy query failed: {e}", exc_info=True)
            return {"error": str(e)}

        lead_time_curve = {
            "lead_minutes": [r[0] for r in lead_rows],
            "mae": [round(r[1], 4) for r in lead_rows],
            "rmse": [round(r[2], 4) for r in lead_rows],
            "me": [round(r[3], 4) for r in lead_rows],
            "sample_count": [r[4] for r in lead_rows],
        }

        # v2.34.0: per-cohort breakdown of the same lead-time curve.
        # The pooled `lead_time_curve` above stays unchanged (still
        # meaningful as a headline error number) and is what the
        # verdict-card chip + headline reads. The per-cohort
        # decomposition feeds the multi-trace lead-time chart so the
        # user can see how each retrain's error curve compares.
        #
        # Grouped by (lead_bucket, model_name, model_version) — every
        # distinct cohort gets its own row per bucket. Skipped when
        # the caller has already pinned a single cohort via the
        # filter (only one cohort in the data, multi-trace would be
        # a single line, runtime saved).
        cohorts: list = []
        if not (model_name and model_version):
            try:
                cursor.execute(f"""
                    WITH {actuals_vals_cte},
                    {forecast_vals_cte}
                    SELECT
                        fv.model_name,
                        fv.model_version,
                        CAST((fv.lead_minutes / ?) * ? AS INTEGER) AS lead_bucket,
                        AVG(ABS(fv.value - av.value)) AS mae,
                        SQRT(AVG((fv.value - av.value) * (fv.value - av.value))) AS rmse,
                        AVG(fv.value - av.value) AS me,
                        COUNT(*) AS n
                    FROM forecast_vals fv
                    INNER JOIN actuals_vals av
                        ON av.grid_dt = fv.target_dt
                    WHERE 1=1 {mode_filter}
                    GROUP BY fv.model_name, fv.model_version, lead_bucket
                    ORDER BY fv.model_name, fv.model_version, lead_bucket
                """, (
                    *forecast_vals_params,
                    bucket_min, bucket_min,
                ))
                cohort_rows = cursor.fetchall()
            except sqlite3.Error as e:
                logger.warning(f"Per-cohort lead-time query failed: {e}")
                cohort_rows = []

            # Reshape: rows → list of cohort dicts, each with a
            # lead_time_curve in the same shape as the pooled one.
            cohort_map: dict = {}
            for mn, mv, lb, mae, rmse, me_val, n in cohort_rows:
                key = (mn, mv)
                if key not in cohort_map:
                    cohort_map[key] = {
                        "model_name": mn,
                        "model_version": mv,
                        "lead_time_curve": {
                            "lead_minutes": [],
                            "mae": [],
                            "rmse": [],
                            "me": [],
                            "sample_count": [],
                        },
                    }
                ltc = cohort_map[key]["lead_time_curve"]
                ltc["lead_minutes"].append(int(lb))
                ltc["mae"].append(round(mae, 4))
                ltc["rmse"].append(round(rmse, 4))
                ltc["me"].append(round(me_val, 4))
                ltc["sample_count"].append(int(n))
            cohorts = list(cohort_map.values())

        # --- Revision improvement ---
        # Compare the FIRST forecast for each target_dt vs the LAST.
        # Only include targets that were actually re-forecast (>= 2
        # distinct issuances). Previously this used `first_pred != last_pred`
        # which wrongly excluded targets whose re-forecasts happened to
        # produce numerically identical predictions (legitimate case),
        # while INCLUDING no-op single-forecast targets only when the
        # CASE-aggregation happened to line up — the practical effect was
        # that Samples collapsed to a tiny fraction of actual re-forecast
        # targets. The explicit COUNT-based filter is correct.
        try:
            cursor.execute(f"""
                WITH {actuals_vals_cte},
                {forecast_vals_cte},
                ranked AS (
                    SELECT
                        fv.target_dt,
                        fv.lead_minutes,
                        fv.value AS predicted,
                        av.value AS actual,
                        ROW_NUMBER() OVER (PARTITION BY fv.target_dt ORDER BY fv.issued_at ASC) AS rn_first,
                        ROW_NUMBER() OVER (PARTITION BY fv.target_dt ORDER BY fv.issued_at DESC) AS rn_last,
                        COUNT(*) OVER (PARTITION BY fv.target_dt) AS n_forecasts
                    FROM forecast_vals fv
                    INNER JOIN actuals_vals av
                        ON av.grid_dt = fv.target_dt
                    WHERE 1=1 {mode_filter}
                ),
                first_last AS (
                    SELECT target_dt,
                           MAX(CASE WHEN rn_first = 1 THEN predicted END) AS first_pred,
                           MAX(CASE WHEN rn_last  = 1 THEN predicted END) AS last_pred,
                           MAX(actual) AS actual,
                           MAX(n_forecasts) AS n_forecasts
                    FROM ranked
                    WHERE rn_first = 1 OR rn_last = 1
                    GROUP BY target_dt
                    HAVING n_forecasts >= 2
                )
                SELECT
                    AVG(ABS(first_pred - actual)) AS first_mae,
                    AVG(ABS(last_pred  - actual)) AS last_mae,
                    AVG(first_pred - actual) AS first_me,
                    AVG(last_pred  - actual) AS last_me,
                    COUNT(*) AS n
                FROM first_last
            """, (
                *forecast_vals_params,
            ))
            rev_row = cursor.fetchone()
        except sqlite3.Error as e:
            logger.warning(f"Revision improvement query failed: {e}")
            rev_row = None

        revision = {}
        if rev_row and rev_row[4] > 0:
            first_mae = round(rev_row[0], 4)
            last_mae = round(rev_row[1], 4)
            improvement = round((1 - last_mae / first_mae) * 100, 1) if first_mae > 0 else 0
            revision = {
                "first_forecast_mae": first_mae,
                "latest_forecast_mae": last_mae,
                "first_forecast_me": round(rev_row[2], 4),
                "latest_forecast_me": round(rev_row[3], 4),
                "improvement_pct": improvement,
                "sample_count": rev_row[4],
            }

        # --- Summary stats ---
        cursor.execute(
            "SELECT COUNT(*), MIN(issued_at), MAX(issued_at) "
            f"FROM forecast_log WHERE experiment = ? AND issued_at >= ?{model_filter_sql}",
            (experiment, cutoff_str, *model_filter_param),
        )
        stats_row = cursor.fetchone()

        # --- Normalisation baseline ---
        # Mean |actual| across the same window in evaluation_mode. Used by
        # the UI to report MAE as a % of "typical interval demand" without
        # needing the caller to know units. In increment mode we diff
        # actuals with the same adjacency guard as the accuracy query so
        # the baseline matches what errors are measured against.
        typical = None
        try:
            if increment:
                cursor.execute(
                    "SELECT AVG(ABS(value)) FROM _mlfl_actuals_vals_tmp "
                    "WHERE value IS NOT NULL AND value >= 0"
                )
            else:
                cursor.execute(
                    f"SELECT AVG(ABS(value)) FROM {actuals_table} "
                    "WHERE SUBSTR(ds, 1, 19) >= ?",
                    (cutoff_str,),
                )
            row = cursor.fetchone()
            if row and row[0] is not None:
                typical = round(float(row[0]), 4)
        except sqlite3.Error as e:
            logger.warning(f"Typical-demand baseline query failed: {e}")

        # Classify why the lead-time curve might be empty so the UI can
        # render a precise hint instead of substring-matching the error
        # string. Branches:
        #   "ok"          — at least one bucket populated.
        #   "warming_up"  — no rows logged at all in this window
        #                   (production just started, or champion
        #                   rotated and the filter widened to a fresh
        #                   cohort).
        #   "no_overlap"  — rows exist but no forecast target_dt matched
        #                   the actuals grid (sensor stopped reporting,
        #                   increment mode + outages, etc.).
        total_logged = int(stats_row[0]) if stats_row else 0
        if lead_rows:
            empty_reason = "ok"
        elif total_logged == 0:
            empty_reason = "warming_up"
        else:
            empty_reason = "no_overlap"

        return {
            "experiment": experiment,
            "evaluation_mode": "increment" if increment else "raw",
            "lead_time_curve": lead_time_curve,
            "cohorts": cohorts,
            "revision_improvement": revision,
            "typical_interval_demand": typical,
            "total_logged": total_logged,
            "empty_reason": empty_reason,
            "date_range": {
                "from": stats_row[1] if stats_row else None,
                "to": stats_row[2] if stats_row else None,
            },
        }

    def _get_forecast_accuracy_daily_cumulative_locked(
        self,
        experiment: str,
        actuals_table: str,
        max_age_days: int = 30,
        interval_minutes: int = 30,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        day_offset_hours: Optional[float] = None,
        cumulative_source: bool = True,
    ) -> dict:
        """Lead-time accuracy in **daily-cumulative space**.

        Two flavours, selected by ``cumulative_source``:

        * ``cumulative_source=True`` (daily-reset cumulative sensors, e.g.
          ``sensor.energy_today``): the actuals table already holds the
          running daily total, so it is read directly — the ``seed`` and
          the comparison value are raw readings.
        * ``cumulative_source=False`` (instantaneous sensors, e.g. a kW
          power reading): there is no cumulative actual to read, so the
          running daily total is BUILT here by cumulatively summing the
          per-interval actual values within each local day (reset at
          midnight). The forecast side is identical — ``forecast_log``
          stores the per-interval predicted value for both sensor kinds,
          so summing it within the day gives the predicted running total.
          This makes the per-interval ↔ cumulative accuracy switch
          meaningful on every experiment, not just cumulative ones.

        For each (issued_at, target_dt, predicted) row we compute the
        ``predicted_cumulative`` at target_dt:

            predicted_cumulative = seed
                                  + Σ (per-interval predictions within
                                       target_dt's local day, in
                                       chronological order up to and
                                       including this target_dt)

        where ``seed`` is the actual cumulative reading at issued_at
        when target_dt lands in the SAME local day as issued_at, and
        0 otherwise (because the daily-reset sensor restarts at
        midnight, so the prior day's accumulation is irrelevant).

        This is then compared against the raw cumulative actual at
        target_dt — for a daily-reset sensor that reading IS the
        demand-so-far on that day. Errors integrate per-interval
        errors, so the lead-time MAE curve typically grows with lead.
        That's expected: this metric answers "how close is the
        predicted day-so-far to actual day-so-far", not "how close is
        each per-interval delta".

        ``forecast_log.predicted`` is logged as per-interval delta for
        cumulative sensors (see main.py:5068 and the comment in
        ``_get_forecast_accuracy_locked``); this function relies on
        that invariant.
        """
        from datetime import datetime, timedelta
        cursor = self.conn.cursor()
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cutoff_str = (
            datetime.utcnow() - timedelta(days=max_age_days)
        ).strftime("%Y-%m-%d %H:%M:%S")
        interval_sec = max(60, int(interval_minutes) * 60)
        bucket_min = max(1, int(interval_minutes))

        _filter_clauses = []
        _filter_params: list = []
        if model_name:
            _filter_clauses.append("model_name = ?")
            _filter_params.append(model_name)
        if model_version:
            _filter_clauses.append("model_version = ?")
            _filter_params.append(model_version)
        model_filter_sql = (
            " AND " + " AND ".join(_filter_clauses)
        ) if _filter_clauses else ""
        model_filter_param = tuple(_filter_params)

        # v2.40.10: TZ-aware day bucketing. Forecasts are stored in
        # UTC, but HA's daily-reset sensors (sensor.<x>_today) reset
        # at LOCAL midnight. A UTC ``SUBSTR(target_dt, 1, 10)`` lumps
        # all of a UTC day into one bucket, but for a viewer in BST
        # (UTC+1) the "last same-UTC-day target" sits at 23:30 UTC =
        # 00:30 local — right after the local-midnight reset — so the
        # actual_cumulative reading at that target is back near zero
        # and the End-of-day card reads ~0% actual every cycle.
        # ``day_offset_hours`` shifts the day-bucketing key so it
        # follows local midnight instead. Mirrors what the stability
        # function does at db.py:2169-2180.
        off = float(day_offset_hours or 0.0)
        off_seconds = int(off * 3600)
        if off_seconds == 0:
            target_day_expr = "SUBSTR(fl.target_dt, 1, 10)"
            issued_day_expr = "SUBSTR(fl.issued_at, 1, 10)"
        else:
            # ``off_seconds`` is a Python int derived from a numeric
            # endpoint param — safe to interpolate directly. Inlining
            # it (rather than bind-param) avoids shuffling extra
            # placeholders into two separate queries with different day-
            # expression counts.
            target_day_expr = (
                "strftime('%Y-%m-%d', "
                f"CAST(strftime('%s', fl.target_dt) AS INTEGER) + {off_seconds}, "
                "'unixepoch')"
            )
            issued_day_expr = (
                "strftime('%Y-%m-%d', "
                f"CAST(strftime('%s', fl.issued_at) AS INTEGER) + {off_seconds}, "
                "'unixepoch')"
            )

        # v2.41.0 (audit F4): indexed temp table instead of a per-query
        # co-routine CTE — see _materialise_actuals_grid.
        if not self._materialise_actuals_grid(
            cursor, actuals_table, interval_sec, cutoff_str,
        ):
            return {"error": "could not build actuals grid"}

        # The joins below read the running-daily-total ACTUAL from
        # ``actuals_rel``. For a cumulative-source sensor that IS the raw
        # grid (the sensor already reports the running total). For an
        # instantaneous sensor we build it here by cumulatively summing
        # the per-interval actual values within each local day (same day
        # key + offset as the forecast side), so the seed and comparison
        # land in the same running-total space as the summed predictions.
        actuals_rel = "_mlfl_actuals_grid_tmp"
        # Cumulative readings are non-negative by construction, so the
        # raw path keeps the ``>= 0`` sanity guard. A summed-demand series
        # can legitimately be small/zero but a hard ``>= 0`` would wrongly
        # drop signed sensors, so the built path only requires non-NULL.
        ag_guard = "AND ag.value >= 0" if cumulative_source else ""
        typ_guard = "AND value >= 0" if cumulative_source else ""
        if not cumulative_source:
            if off_seconds == 0:
                grid_day_expr = "SUBSTR(grid_dt, 1, 10)"
            else:
                grid_day_expr = (
                    "strftime('%Y-%m-%d', "
                    f"CAST(strftime('%s', grid_dt) AS INTEGER) + {off_seconds}, "
                    "'unixepoch')"
                )
            try:
                cursor.execute("DROP TABLE IF EXISTS _mlfl_actuals_cum_tmp")
                cursor.execute(
                    f"""
                    CREATE TEMP TABLE _mlfl_actuals_cum_tmp AS
                    SELECT grid_dt,
                        SUM(value) OVER (
                            PARTITION BY {grid_day_expr}
                            ORDER BY grid_dt
                        ) AS value
                    FROM _mlfl_actuals_grid_tmp
                    WHERE value IS NOT NULL
                    """
                )
                cursor.execute(
                    "CREATE INDEX _mlfl_actuals_cum_tmp_idx "
                    "ON _mlfl_actuals_cum_tmp(grid_dt)"
                )
                actuals_rel = "_mlfl_actuals_cum_tmp"
            except sqlite3.Error as e:
                logger.warning(
                    "Failed to build cumulative actuals grid (non-cumulative "
                    "source); daily-cumulative accuracy unavailable: %s", e,
                )
                return {"error": "could not build cumulative actuals grid"}

        sql = f"""
            WITH forecast_base AS (
                SELECT
                    fl.experiment, fl.model_name, fl.model_version,
                    fl.issued_at, fl.target_dt, fl.lead_minutes,
                    fl.predicted,
                    strftime('%Y-%m-%d %H:%M:%S',
                        (CAST(strftime('%s', SUBSTR(fl.issued_at, 1, 19)) AS INTEGER) / ?) * ?,
                        'unixepoch'
                    ) AS issued_grid,
                    {issued_day_expr} AS issued_day,
                    {target_day_expr} AS target_day
                FROM forecast_log fl
                WHERE fl.experiment = ?
                  AND fl.target_dt <= ?
                  AND fl.issued_at >= ?
                  {model_filter_sql}
            ),
            forecast_seeded AS (
                SELECT
                    fb.*,
                    CASE WHEN fb.target_day = fb.issued_day
                         THEN COALESCE(seed_a.value, 0)
                         ELSE 0
                    END AS seed_value
                FROM forecast_base fb
                LEFT JOIN {actuals_rel} seed_a
                    ON seed_a.grid_dt = fb.issued_grid
            ),
            forecast_cum AS (
                SELECT
                    fs.*,
                    fs.seed_value + SUM(predicted) OVER (
                        PARTITION BY experiment, model_name, model_version,
                                     issued_at, target_day
                        ORDER BY target_dt
                    ) AS predicted_cumulative
                FROM forecast_seeded fs
            )
            SELECT
                CAST((fc.lead_minutes / ?) * ? AS INTEGER) AS lead_bucket,
                AVG(ABS(fc.predicted_cumulative - ag.value)) AS mae,
                SQRT(AVG((fc.predicted_cumulative - ag.value)
                       * (fc.predicted_cumulative - ag.value))) AS rmse,
                AVG(fc.predicted_cumulative - ag.value) AS me,
                COUNT(*) AS n
            FROM forecast_cum fc
            INNER JOIN {actuals_rel} ag ON ag.grid_dt = fc.target_dt
            WHERE ag.value IS NOT NULL {ag_guard}
            GROUP BY lead_bucket
            ORDER BY lead_bucket
        """

        # Param order matches CTE order. Day-bucketing offset (if any)
        # is inlined into target_day_expr / issued_day_expr above, so
        # no day params here. The actuals grid is a pre-built temp
        # table (no params).
        #  forecast_base seed grid: (interval_sec, interval_sec)
        #  forecast_base WHERE: (experiment, now_str, cutoff_str,
        #                        *model_filter_param)
        #  outer SELECT: (bucket_min, bucket_min)
        params = (
            interval_sec, interval_sec,
            experiment, now_str, cutoff_str, *model_filter_param,
            bucket_min, bucket_min,
        )
        try:
            cursor.execute(sql, params)
            lead_rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(
                f"Daily-cumulative accuracy query failed: {e}",
                exc_info=True,
            )
            return {"error": str(e)}

        lead_time_curve = {
            "lead_minutes": [r[0] for r in lead_rows],
            "mae": [round(r[1], 4) for r in lead_rows],
            "rmse": [round(r[2], 4) for r in lead_rows],
            "me": [round(r[3], 4) for r in lead_rows],
            "sample_count": [r[4] for r in lead_rows],
        }

        # End-of-day headline: for each (issued_at, target_day) where
        # target_day == issued_day, take the LAST target's predicted
        # cumulative vs the actual cumulative at that target. That's
        # "the forecast's predicted day-so-far at the latest target it
        # made for today" — the most meaningful single number for a
        # daily-total forecaster.
        end_of_day = {"sample_count": 0}
        try:
            cursor.execute(
                f"""
                WITH forecast_base AS (
                    SELECT
                        fl.experiment, fl.model_name, fl.model_version,
                        fl.issued_at, fl.target_dt, fl.lead_minutes,
                        fl.predicted,
                        strftime('%Y-%m-%d %H:%M:%S',
                            (CAST(strftime('%s', SUBSTR(fl.issued_at, 1, 19)) AS INTEGER) / ?) * ?,
                            'unixepoch'
                        ) AS issued_grid,
                        {issued_day_expr} AS issued_day,
                        {target_day_expr} AS target_day
                    FROM forecast_log fl
                    WHERE fl.experiment = ?
                      AND fl.target_dt <= ?
                      AND fl.issued_at >= ?
                      AND {target_day_expr} = {issued_day_expr}
                      {model_filter_sql}
                ),
                forecast_seeded AS (
                    SELECT fb.*,
                        COALESCE(seed_a.value, 0) AS seed_value
                    FROM forecast_base fb
                    LEFT JOIN {actuals_rel} seed_a
                        ON seed_a.grid_dt = fb.issued_grid
                ),
                forecast_cum AS (
                    SELECT fs.*,
                        fs.seed_value + SUM(predicted) OVER (
                            PARTITION BY experiment, model_name, model_version,
                                         issued_at, target_day
                            ORDER BY target_dt
                        ) AS predicted_cumulative
                    FROM forecast_seeded fs
                ),
                last_target_per_issuance AS (
                    SELECT issued_at, target_day,
                           MAX(target_dt) AS last_target_dt
                    FROM forecast_cum
                    GROUP BY issued_at, target_day
                )
                SELECT
                    AVG(ABS(fc.predicted_cumulative - ag.value)) AS mae,
                    AVG(fc.predicted_cumulative - ag.value) AS me,
                    AVG(fc.predicted_cumulative) AS mean_predicted,
                    AVG(ag.value) AS mean_actual,
                    COUNT(*) AS n
                FROM forecast_cum fc
                INNER JOIN last_target_per_issuance lt
                    ON lt.issued_at = fc.issued_at
                   AND lt.target_day = fc.target_day
                   AND lt.last_target_dt = fc.target_dt
                INNER JOIN {actuals_rel} ag ON ag.grid_dt = fc.target_dt
                WHERE ag.value IS NOT NULL {ag_guard}
                """,
                params[:-2],  # outer query has no lead_bucket binding
            )
            row = cursor.fetchone()
            if row and row[4]:
                end_of_day = {
                    "mae": round(row[0], 4),
                    "me": round(row[1], 4),
                    "mean_predicted": round(row[2], 4),
                    "mean_actual": round(row[3], 4),
                    "sample_count": int(row[4]),
                }
        except sqlite3.Error as e:
            logger.warning(f"End-of-day headline query failed: {e}")

        # typical_interval_demand: in daily-cumulative mode we report
        # the typical magnitude of the END-OF-DAY actual so the
        # verdict-card's nmae normalises against a meaningful scale
        # (e.g. ~50 kWh, not the per-interval ~0.5 kWh used by
        # increment mode).
        typical = 0.0
        try:
            cursor.execute(
                f"""
                WITH daily_max AS (
                    SELECT SUBSTR(grid_dt, 1, 10) AS day,
                           MAX(value) AS day_max
                    FROM {actuals_rel}
                    WHERE value IS NOT NULL {typ_guard}
                    GROUP BY day
                )
                SELECT AVG(day_max) FROM daily_max
                """
            )
            t = cursor.fetchone()
            typical = float(t[0]) if t and t[0] is not None else 0.0
        except sqlite3.Error as e:
            logger.warning(f"typical_interval_demand query failed: {e}")

        # Stats / window range for the header.
        total_logged = 0
        date_from = date_to = None
        try:
            cursor.execute(
                """
                SELECT COUNT(*), MIN(issued_at), MAX(issued_at)
                FROM forecast_log
                WHERE experiment = ?
                  AND issued_at >= ?
                """,
                (experiment, cutoff_str),
            )
            row = cursor.fetchone()
            if row:
                total_logged = int(row[0])
                date_from, date_to = row[1], row[2]
        except sqlite3.Error:
            pass

        return {
            "evaluation_mode": "daily_cumulative",
            "lead_time_curve": lead_time_curve,
            "cohorts": [],  # per-cohort breakdown deferred to a follow-up
            "revision_improvement": {},
            "typical_interval_demand": typical,
            "end_of_day": end_of_day,
            "total_logged": total_logged,
            "actuals_matched": sum(lead_time_curve["sample_count"]),
            "model_name": model_name,
            "model_version": model_version,
            "date_range": {"from": date_from, "to": date_to},
        }

    @_locked
    def get_forecast_trajectory(
        self,
        experiment: str,
        actuals_table: str,
        target_dt: Optional[str] = None,
        interval_minutes: int = 30,
        max_age_days: int = 14,
        source_is_cumulative: bool = False,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> dict:
        """
        Return every forecast ever issued for a single target_dt, plus
        the actual value at that target, so the UI can plot a
        "prediction walking toward truth" trajectory.

        Parameters
        ----------
        experiment : str
        actuals_table : str
            SQL-safe table name for the actuals.
        target_dt : str or None
            Target timestamp (YYYY-MM-DD HH:MM:SS). If None, picks the
            most recent target with an actual value AND at least two
            distinct forecast issuances — that's the one most worth
            plotting.
        interval_minutes : int
            Actuals are snapped to this grid before matching target_dt.
        max_age_days : int
            Only consider forecasts/targets within this window.
        source_is_cumulative : bool
            When True, the model predicts per-interval deltas even
            though the underlying sensor stores cumulative values. The
            raw stored actual (e.g. 17% fill) is then in a different
            space from the predicted delta (≈0–1%/interval) and
            plotting them together is misleading. This flag tells the
            query to return the actual as a per-interval delta
            (``value − value[t−interval]``, adjacency-guarded) so both
            series live on the same axis.
        model_name : str, optional
            Restrict forecast candidates to one model. Matches the
            accuracy / stability endpoints.

        Returns
        -------
        dict with keys:
            ``target_dt``: the target being shown (ISO string)
            ``actual``: float or None — in the prediction space
            ``forecasts``: [{issued_at, predicted, lead_minutes, model_name}, ...]
                           ordered by issued_at ascending
            ``available_targets``: list of recent target_dts that have an
                           actual + >=2 issuances, newest first — populates
                           the dropdown in the UI.
            ``actual_space``: "delta" or "raw" — labels the units of
                           ``actual`` and ``target_meta[i].actual``.
        """
        cursor = self.conn.cursor()
        interval_sec = max(60, int(interval_minutes) * 60)
        cutoff_str = (
            datetime.utcnow() - pd.Timedelta(days=max_age_days)
        ).strftime("%Y-%m-%d %H:%M:%S")
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (actuals_table,),
        )
        if not cursor.fetchone():
            return {"error": "No actuals data available yet"}

        _clauses = []
        _params: list = []
        if model_name:
            _clauses.append("fl.model_name = ?")
            _params.append(model_name)
        if model_version:
            _clauses.append("fl.model_version = ?")
            _params.append(model_version)
        model_filter_sql = (" AND " + " AND ".join(_clauses)) if _clauses else ""
        model_filter_param = tuple(_params)

        # v2.41.0 (audit F4): indexed temp tables instead of per-query
        # co-routine CTEs (see _materialise_actuals_grid). For
        # cumulative sources the deltas table puts actuals in the same
        # space as the per-interval predictions, with the same
        # adjacency guard as get_forecast_accuracy's increment mode.
        if not self._materialise_actuals_grid(
            cursor, actuals_table, interval_sec, cutoff_str,
            increment=source_is_cumulative,
        ):
            return {"error": "could not build actuals grid"}
        if source_is_cumulative:
            actuals_vals_cte = (
                "actuals_vals AS "
                "(SELECT grid_dt, value FROM _mlfl_actuals_vals_tmp)"
            )
            actual_space = "delta"
        else:
            actuals_vals_cte = (
                "actuals_vals AS "
                "(SELECT grid_dt, value FROM _mlfl_actuals_grid_tmp)"
            )
            actual_space = "raw"

        # Candidate targets for the dropdown: recent targets with an
        # actual value on the grid AND >=2 distinct issuances. These are
        # the ones where a trajectory plot is meaningful (one-shot
        # targets give a single dot and teach nothing).
        # We also compute max |predicted − actual| across all issuances
        # for each target — now in the same (delta / raw) space so
        # "biggest miss" sort is meaningful for cumulative sensors.
        try:
            cursor.execute(f"""
                WITH {actuals_vals_cte}
                SELECT fl.target_dt,
                       COUNT(DISTINCT fl.issued_at) AS n_iss,
                       MAX(ABS(fl.predicted - av.value)) AS max_abs_err,
                       MAX(av.value) AS actual
                FROM forecast_log fl
                INNER JOIN actuals_vals av ON av.grid_dt = fl.target_dt
                WHERE fl.experiment = ?
                  AND fl.target_dt <= ?
                  AND fl.issued_at >= ?
                  AND av.value IS NOT NULL
                  {model_filter_sql}
                GROUP BY fl.target_dt
                HAVING n_iss >= 2
                ORDER BY fl.target_dt DESC
                LIMIT 48
            """, (
                experiment, now_str, cutoff_str,
                *model_filter_param,
            ))
            candidate_rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Trajectory candidates query failed: {e}", exc_info=True)
            return {"error": str(e)}

        available_targets = [r[0] for r in candidate_rows]
        target_meta = [
            {
                "target_dt": r[0],
                "n_issuances": int(r[1]),
                "max_abs_error": round(float(r[2]), 4) if r[2] is not None else None,
                "actual": round(float(r[3]), 4) if r[3] is not None else None,
            }
            for r in candidate_rows
        ]
        if not available_targets:
            return {
                "experiment": experiment,
                "target_dt": None,
                "actual": None,
                "forecasts": [],
                "available_targets": [],
                "target_meta": [],
                "actual_space": actual_space,
            }

        # If caller didn't pick a target, default to the most recent one
        # we have a useful trajectory for.
        if not target_dt or target_dt not in available_targets:
            target_dt = available_targets[0]

        # Fetch all forecasts for the chosen target. The `fl.` aliases in
        # model_filter_sql carry over intact — single-table here but
        # SQLite doesn't mind qualified names without a join.
        try:
            cursor.execute(
                "SELECT issued_at, predicted, lead_minutes, model_name "
                "FROM forecast_log fl "
                f"WHERE experiment = ? AND target_dt = ?{model_filter_sql} "
                "ORDER BY issued_at ASC",
                (experiment, target_dt, *model_filter_param),
            )
            forecast_rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Trajectory fetch failed: {e}", exc_info=True)
            return {"error": str(e)}

        forecasts = [
            {
                "issued_at": r[0],
                "predicted": round(float(r[1]), 4),
                "lead_minutes": int(r[2]),
                "model_name": r[3],
            }
            for r in forecast_rows
        ]

        # Fetch the actual for this target (in the prediction space).
        try:
            cursor.execute(f"""
                WITH {actuals_vals_cte}
                SELECT av.value FROM actuals_vals av
                WHERE av.grid_dt = ?
            """, (target_dt,))
            actual_row = cursor.fetchone()
        except sqlite3.Error as e:
            logger.warning(f"Trajectory actual lookup failed: {e}")
            actual_row = None

        actual = (
            round(float(actual_row[0]), 4)
            if actual_row and actual_row[0] is not None
            else None
        )

        return {
            "experiment": experiment,
            "target_dt": target_dt,
            "actual": actual,
            "forecasts": forecasts,
            "available_targets": available_targets,
            "target_meta": target_meta,
            "actual_space": actual_space,
        }

    @_locked
    def get_conformal_quantiles(
        self,
        experiment: str,
        actuals_table: str,
        level: float = 0.8,
        model_name: Optional[str] = None,
        interval_minutes: int = 30,
        max_age_days: int = 14,
        min_samples: int = 10,
        model_version: Optional[str] = None,
    ) -> dict:
        """
        Compute per-lead-time conformal nonconformity quantiles from
        historical forecasts vs actuals in forecast_log.

        For a symmetric band built from ABSOLUTE residuals, coverage is
        P(|y − ŷ| ≤ q̂) — exactly the quantile level used — so a (1−α)
        band takes the (1−α)-th quantile of |residual| at each lead
        bucket: for an 80%-band (level=0.8), the 80th percentile. (The
        (1−α/2) rule belongs to the two-sided SIGNED-residual
        construction; applying it to |residual| — as this method did
        before v2.41.0 — systematically over-covered: nominal-80% bands
        realised ~90% coverage and were ~1.5× wider than calibrated.)

        This is split conformal prediction with a rolling residual
        buffer (capped at ``max_age_days``), not ACI / Gibbs-Candès
        (2021) or any other adaptive variant — there is no α_t update
        step, no exponential reweighting. Using deployed forecast /
        actual pairs as the calibration sample avoids the cost of
        refitting the model on a held-out window every production
        cycle. The tradeoff is that residuals aren't strictly
        exchangeable (trend, weekly seasonality, regime shifts,
        retrain boundaries), so finite-sample coverage is approximate
        rather than guaranteed. The realised-coverage diagnostic
        (``get_forecast_coverage``) — including the hour-of-day and
        weekday/weekend breakdowns — is the user-visible signal that
        the bands need recalibrating. For diagnostic intervals on a
        home-automation sensor this is an acceptable simplification.

        Parameters
        ----------
        experiment : str
        actuals_table : str
        level : float
            Desired coverage, in (0, 1). 0.8 produces the 80th-percentile
            absolute-residual band.
        model_name : str, optional
            Restrict to residuals from one model (usually the current
            champion) so interval width reflects its behaviour, not a
            different model's.
        interval_minutes : int
            Grid for aligning actuals to target_dt and bucketing leads.
        max_age_days : int
            Residual lookback window.
        min_samples : int
            Minimum residuals per lead bucket before reporting a
            quantile; buckets below this are omitted and the caller
            falls back (see ``fallback_quantile`` in the return dict).

        Returns
        -------
        dict with keys:
            ``quantiles``: {lead_bucket (int, minutes): q (float)}
            ``fallback_quantile``: float or None — use when a lead bucket
                is missing from ``quantiles`` (the ``level``-quantile
                computed across ALL buckets, giving a safe default).
            ``sample_counts``: {lead_bucket: int}
            ``total_samples``: int
            ``level``: float (echoed)
        """
        cursor = self.conn.cursor()
        interval_sec = max(60, int(interval_minutes) * 60)
        bucket_min = max(1, int(interval_minutes))
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cutoff_str = (
            datetime.utcnow() - pd.Timedelta(days=max_age_days)
        ).strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (actuals_table,),
        )
        if not cursor.fetchone():
            return {
                "quantiles": {},
                "fallback_quantile": None,
                "sample_counts": {},
                "total_samples": 0,
                "level": level,
            }

        # The actuals grid is a pre-built temp table (no interval params
        # in the query itself — see the materialise call below).
        params = [bucket_min, bucket_min,
                  experiment, now_str, cutoff_str]
        _clauses = []
        if model_name:
            _clauses.append("fl.model_name = ?")
            params.append(model_name)
        if model_version:
            _clauses.append("fl.model_version = ?")
            params.append(model_version)
        model_filter = ("AND " + " AND ".join(_clauses)) if _clauses else ""

        # v2.34.0: SELECT now carries (model_name, model_version) so
        # the Python aggregation below can partition residuals by
        # cohort and pick one winning cohort per lead bucket.
        # Conformal quantiles calibrated against mixed-cohort residuals
        # don't correspond to any actually-published band; the bands
        # the user sees on the chart come from one specific weight
        # regime, so the calibration target should too.
        # v2.41.0 (audit F3): indexed temp table instead of a
        # co-routine CTE re-scanned per forecast row — this call sits on
        # the forecast-publish path, so its cost directly delays sensor
        # publishing. The cutoff also bounds the actuals scan, which
        # previously covered the full max_age retention window.
        if not self._materialise_actuals_grid(
            cursor, actuals_table, interval_sec, cutoff_str,
        ):
            return {
                "quantiles": {},
                "fallback_quantile": None,
                "sample_counts": {},
                "total_samples": 0,
                "level": level,
            }
        try:
            cursor.execute(f"""
                SELECT
                    CAST((fl.lead_minutes / ?) * ? AS INTEGER) AS lead_bucket,
                    ABS(fl.predicted - ag.value) AS abs_residual,
                    fl.model_name,
                    fl.model_version
                FROM forecast_log fl
                INNER JOIN _mlfl_actuals_grid_tmp ag ON ag.grid_dt = fl.target_dt
                WHERE fl.experiment = ?
                  AND fl.target_dt <= ?
                  AND fl.issued_at >= ?
                  {model_filter}
            """, tuple(params))
            rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Conformal quantile query failed: {e}", exc_info=True)
            return {
                "quantiles": {},
                "fallback_quantile": None,
                "sample_counts": {},
                "total_samples": 0,
                "level": level,
            }

        if not rows:
            return {
                "quantiles": {},
                "fallback_quantile": None,
                "sample_counts": {},
                "total_samples": 0,
                "level": level,
            }

        df = pd.DataFrame(
            rows,
            columns=["lead_bucket", "abs_residual", "model_name", "model_version"],
        )
        df = df.dropna(subset=["lead_bucket", "abs_residual"])

        # Pick one cohort per lead_bucket — the one with the most
        # residuals, breaking ties by newest model_version. Filters
        # df down to residuals from the winning (lead_bucket, cohort)
        # combinations before quantile estimation.
        cohort_counts = (
            df.groupby(["lead_bucket", "model_name", "model_version"], dropna=False)
              .size()
              .reset_index(name="n")
              .sort_values(
                  ["lead_bucket", "n", "model_version"],
                  ascending=[True, False, False],
              )
        )
        winners = cohort_counts.drop_duplicates("lead_bucket")[
            ["lead_bucket", "model_name", "model_version"]
        ]
        df = df.merge(
            winners,
            on=["lead_bucket", "model_name", "model_version"],
            how="inner",
        )

        # Coverage of a |residual| band equals the quantile level itself
        # (see docstring) — use `level` directly, with a light
        # finite-sample bump (the split-CP ceil((n+1)·level)/n
        # correction, bounded at 1.0) applied per group at lookup time
        # would require per-bucket n; the global rolling buffer is large
        # enough (min_samples gate) that the plain quantile is within
        # one residual of the corrected one.
        q = max(0.0, min(1.0, level))

        counts = df.groupby("lead_bucket").size()
        # Usable buckets have enough samples to estimate a stable
        # quantile; buckets below min_samples fall back to the global
        # quantile so short-lead predictions still get a band.
        usable = counts[counts >= min_samples].index
        quantiles = (
            df[df["lead_bucket"].isin(usable)]
            .groupby("lead_bucket")["abs_residual"]
            .quantile(q)
            .to_dict()
        )
        fallback = float(df["abs_residual"].quantile(q)) if len(df) > 0 else None

        return {
            "quantiles": {int(k): float(v) for k, v in quantiles.items()},
            "fallback_quantile": fallback,
            "sample_counts": {int(k): int(v) for k, v in counts.items()},
            "total_samples": int(len(df)),
            "level": level,
        }

    @_locked
    def get_forecast_coverage(
        self,
        experiment: str,
        actuals_table: str,
        interval_minutes: int = 30,
        max_age_days: int = 30,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        tz: Optional[str] = None,
        nominal: float = 0.8,
    ) -> dict:
        """
        Compute empirical coverage of published interval forecasts.

        Coverage is the fraction of actuals that fell inside
        [lower, upper] among forecast_log rows with both bounds stored.
        A well-calibrated 80% band should land at ~0.80; systematic
        deviation is the user-visible signal that intervals need
        recalibration.

        Only rows with non-NULL upper AND lower are considered — legacy
        point-only rows are excluded automatically.

        Returns
        -------
        dict with keys:
            ``by_lead``: {lead_minutes: [...], coverage: [...], n: [...]}
            ``by_hour_of_day``: {hour: [0..23], coverage: [...], n: [...]}
                Coverage bucketed by local hour-of-day (using ``tz``;
                falls back to UTC if not provided). Regime-shift signal
                that the per-lead view doesn't show — e.g. weekday
                evenings being systematically under-covered.
            ``by_weekday_weekend``: {bucket: ["weekday", "weekend"], coverage: [...], n: [...]}
                Two-bucket split. The conformal residual buffer pools
                weekday and weekend cycles together, so consistent
                divergence here is the cleanest signal that
                exchangeability is failing.
            ``worst_bucket``: {kind, label, coverage, n} | None
                The most-mis-covered bucket across the breakdowns,
                where ``kind`` is ``'hour_of_day'``,
                ``'weekday_weekend'``, or ``'lead'`` and the row has
                at least 20 observations to keep tiny buckets out of
                the user's verdict chip.
            ``overall``: {coverage: float, n: int} or empty dict
            ``level``: float (nominal level read from ``level`` kwarg
                of the conformal run; here inferred by the caller and
                echoed in the UI).
            ``tz``: str (echoed; what the hour-of-day uses)
        """
        cursor = self.conn.cursor()
        interval_sec = max(60, int(interval_minutes) * 60)
        bucket_min = max(1, int(interval_minutes))
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cutoff_str = (
            datetime.utcnow() - pd.Timedelta(days=max_age_days)
        ).strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (actuals_table,),
        )
        if not cursor.fetchone():
            return {
                "by_lead": {"lead_minutes": [], "coverage": [], "n": []},
                "by_hour_of_day": {"hour": [], "coverage": [], "n": []},
                "by_weekday_weekend": {"bucket": [], "coverage": [], "n": []},
                "worst_bucket": None,
                "overall": {},
                "tz": tz or "UTC",
            }

        # Optional filters — coverage of a rotated-out model (or
        # pre-retrain weights of the current model) tells the user
        # little about the current champion's calibration.
        _clauses = []
        _params: list = []
        if model_name:
            _clauses.append("fl.model_name = ?")
            _params.append(model_name)
        if model_version:
            _clauses.append("fl.model_version = ?")
            _params.append(model_version)
        model_filter_sql = (" AND " + " AND ".join(_clauses)) if _clauses else ""
        model_filter_param = tuple(_params)

        # v2.39.3: materialise the actuals_grid aggregation ONCE into a
        # TEMP table, then reuse it across the three subsequent queries
        # (per-lead, overall, breakdown). v2.41.0: moved to the shared
        # _materialise_actuals_grid helper, which also bounds the
        # actuals scan to the forecast cutoff (join keys satisfy
        # target_dt >= issued_at >= cutoff, so earlier actuals can
        # never match).
        if not self._materialise_actuals_grid(
            cursor, actuals_table, interval_sec, cutoff_str,
        ):
            return {
                "by_lead": {"lead_minutes": [], "coverage": [], "n": []},
                "by_hour_of_day": {"hour": [], "coverage": [], "n": []},
                "by_weekday_weekend": {"bucket": [], "coverage": [], "n": []},
                "worst_bucket": None,
                "overall": {},
                "tz": tz or "UTC",
            }

        # v2.34.0: per-lead coverage is computed per cohort and one
        # winner picked per bucket (most rows, ties broken by newest
        # version). Previously, when the caller's filter widened
        # across versions or models, the coverage rate at a given lead
        # would blend rows from differently-calibrated weight regimes
        # — meaningful as an aggregate, but the verdict-card chip
        # uses this number to judge whether THE current bands are
        # well calibrated, which the blended value misrepresents.
        try:
            cursor.execute(f"""
                WITH per_cohort AS (
                    SELECT
                        CAST((fl.lead_minutes / ?) * ? AS INTEGER) AS lead_bucket,
                        fl.model_name,
                        fl.model_version,
                        AVG(CASE WHEN ag.value BETWEEN fl.lower AND fl.upper
                                 THEN 1.0 ELSE 0.0 END) AS coverage,
                        COUNT(*) AS n
                    FROM forecast_log fl
                    INNER JOIN _mlfl_actuals_grid_tmp ag ON ag.grid_dt = fl.target_dt
                    WHERE fl.experiment = ?
                      AND fl.target_dt <= ?
                      AND fl.issued_at >= ?
                      AND fl.upper IS NOT NULL
                      AND fl.lower IS NOT NULL
                      {model_filter_sql}
                    GROUP BY lead_bucket, fl.model_name, fl.model_version
                ),
                ranked AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY lead_bucket
                        ORDER BY n DESC, model_version DESC
                    ) AS rn
                    FROM per_cohort
                )
                SELECT lead_bucket, coverage, n
                FROM ranked
                WHERE rn = 1
                ORDER BY lead_bucket
            """, (
                bucket_min, bucket_min,
                experiment, now_str, cutoff_str,
                *model_filter_param,
            ))
            by_lead_rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Coverage query failed: {e}", exc_info=True)
            try:
                cursor.execute("DROP TABLE IF EXISTS _mlfl_actuals_grid_tmp")
            except sqlite3.Error:
                pass
            return {
                "by_lead": {"lead_minutes": [], "coverage": [], "n": []},
                "by_hour_of_day": {"hour": [], "coverage": [], "n": []},
                "by_weekday_weekend": {"bucket": [], "coverage": [], "n": []},
                "worst_bucket": None,
                "overall": {},
                "tz": tz or "UTC",
            }

        by_lead = {
            "lead_minutes": [int(r[0]) for r in by_lead_rows],
            "coverage": [round(float(r[1]), 4) for r in by_lead_rows],
            "n": [int(r[2]) for r in by_lead_rows],
        }

        # v2.34.0: overall coverage picks the dominant cohort the same
        # way the per-lead query does, so the headline number that
        # feeds the verdict-card chip reflects ONE weight regime's
        # calibration. Mixing regimes here previously produced a
        # number that didn't correspond to any actually-published
        # band, which made calibration drift look better or worse
        # than it really was for the current champion.
        try:
            cursor.execute(f"""
                WITH per_cohort AS (
                    SELECT
                        fl.model_name,
                        fl.model_version,
                        AVG(CASE WHEN ag.value BETWEEN fl.lower AND fl.upper
                                 THEN 1.0 ELSE 0.0 END) AS coverage,
                        COUNT(*) AS n
                    FROM forecast_log fl
                    INNER JOIN _mlfl_actuals_grid_tmp ag ON ag.grid_dt = fl.target_dt
                    WHERE fl.experiment = ?
                      AND fl.target_dt <= ?
                      AND fl.issued_at >= ?
                      AND fl.upper IS NOT NULL
                      AND fl.lower IS NOT NULL
                      {model_filter_sql}
                    GROUP BY fl.model_name, fl.model_version
                )
                SELECT coverage, n
                FROM per_cohort
                ORDER BY n DESC, model_version DESC
                LIMIT 1
            """, (
                experiment, now_str, cutoff_str,
                *model_filter_param,
            ))
            overall_row = cursor.fetchone()
        except sqlite3.Error as e:
            logger.warning(f"Overall coverage query failed: {e}")
            overall_row = None

        overall = {}
        if overall_row and overall_row[1]:
            overall = {
                "coverage": round(float(overall_row[0]), 4),
                "n": int(overall_row[1]),
            }

        # Regime-shift breakdowns. The conformal bands are calibrated
        # against a single rolling residual buffer that pools across
        # hours-of-day and weekday/weekend cycles. Household series
        # routinely violate exchangeability across those dimensions
        # (load shapes differ between Tuesday morning and Sunday
        # evening), so systematic divergence shows up here even when
        # the overall coverage looks fine. We compute these in Python
        # rather than SQL so the TZ conversion (HA's local time, not
        # UTC) doesn't depend on the SQLite build's strftime support.
        # The cohort-winner logic from the per-lead query is preserved:
        # we restrict to rows from the dominant (model_name,
        # model_version) cohort so each bucket reflects ONE weight
        # regime, not a blend.
        by_hour_of_day = {"hour": [], "coverage": [], "n": []}
        by_weekday_weekend = {"bucket": [], "coverage": [], "n": []}
        try:
            cursor.execute(f"""
                WITH joined AS (
                    SELECT
                        fl.target_dt,
                        fl.model_name,
                        fl.model_version,
                        CASE WHEN ag.value BETWEEN fl.lower AND fl.upper
                             THEN 1.0 ELSE 0.0 END AS in_band
                    FROM forecast_log fl
                    INNER JOIN _mlfl_actuals_grid_tmp ag ON ag.grid_dt = fl.target_dt
                    WHERE fl.experiment = ?
                      AND fl.target_dt <= ?
                      AND fl.issued_at >= ?
                      AND fl.upper IS NOT NULL
                      AND fl.lower IS NOT NULL
                      {model_filter_sql}
                )
                SELECT j.target_dt, j.in_band
                FROM joined j
                INNER JOIN (
                    SELECT model_name, model_version, COUNT(*) AS n
                    FROM joined
                    GROUP BY model_name, model_version
                    ORDER BY n DESC, model_version DESC
                    LIMIT 1
                ) w
                ON j.model_name = w.model_name
                  AND (j.model_version = w.model_version
                       OR (j.model_version IS NULL AND w.model_version IS NULL))
            """, (
                experiment, now_str, cutoff_str,
                *model_filter_param,
            ))
            cohort_rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.warning(f"Coverage cohort query failed: {e}")
            cohort_rows = []

        worst_bucket: Optional[dict] = None
        if cohort_rows:
            df_cov = pd.DataFrame(cohort_rows, columns=["target_dt", "in_band"])
            df_cov["target_dt"] = pd.to_datetime(
                df_cov["target_dt"], errors="coerce", utc=True,
            )
            df_cov = df_cov.dropna(subset=["target_dt"])
            if tz:
                try:
                    df_cov["local_dt"] = df_cov["target_dt"].dt.tz_convert(tz)
                except Exception as _e:
                    # Unknown tz string — fall back silently to UTC so
                    # the diagnostic still works; the UI will say "UTC"
                    logger.debug(f"Unknown tz {tz!r}: {_e}; falling back to UTC")
                    df_cov["local_dt"] = df_cov["target_dt"]
                    tz = "UTC"
            else:
                df_cov["local_dt"] = df_cov["target_dt"]

            hour_grp = df_cov.groupby(df_cov["local_dt"].dt.hour)["in_band"].agg(
                ["mean", "count"]
            )
            for h, row in hour_grp.iterrows():
                by_hour_of_day["hour"].append(int(h))
                by_hour_of_day["coverage"].append(round(float(row["mean"]), 4))
                by_hour_of_day["n"].append(int(row["count"]))

            # weekday/weekend split — weekday is Mon-Fri (dt.dayofweek < 5)
            is_weekend = df_cov["local_dt"].dt.dayofweek >= 5
            for label, mask in (("weekday", ~is_weekend), ("weekend", is_weekend)):
                rows = df_cov[mask]
                if len(rows) == 0:
                    continue
                by_weekday_weekend["bucket"].append(label)
                by_weekday_weekend["coverage"].append(round(float(rows["in_band"].mean()), 4))
                by_weekday_weekend["n"].append(int(len(rows)))

            # Worst bucket across all three breakdowns. Minimum 20 obs
            # so a stale single-bucket outlier doesn't dominate the
            # verdict chip. "Worst" means biggest absolute deviation
            # from the nominal level — passed in via the ``nominal``
            # kwarg (default 0.8 matching the published-band default).
            MIN_N = 20
            candidates = []
            for h, cov, n in zip(by_hour_of_day["hour"], by_hour_of_day["coverage"], by_hour_of_day["n"]):
                if n >= MIN_N:
                    candidates.append({
                        "kind": "hour_of_day", "label": f"hour {h:02d}",
                        "coverage": cov, "n": n,
                    })
            for label, cov, n in zip(by_weekday_weekend["bucket"], by_weekday_weekend["coverage"], by_weekday_weekend["n"]):
                if n >= MIN_N:
                    candidates.append({
                        "kind": "weekday_weekend", "label": label,
                        "coverage": cov, "n": n,
                    })
            for lead, cov, n in zip(by_lead["lead_minutes"], by_lead["coverage"], by_lead["n"]):
                if n >= MIN_N:
                    candidates.append({
                        "kind": "lead", "label": f"+{int(lead)}min",
                        "coverage": cov, "n": n,
                    })
            if candidates:
                # Pick the bucket whose coverage is furthest from the
                # nominal level the bands were calibrated at. Caller
                # passes ``nominal=exp_cfg.conformal_coverage`` (default
                # 0.8); the comparison uses |dev| so over-conservative
                # (e.g. 99% on a nominal 80%) and under-conservative
                # (e.g. 60% on the same target) buckets are both
                # candidates for "worst". Pre-v2.39.3 this picked
                # min(coverage), which silently ignored over-covered
                # buckets even when their |deviation| was larger.
                worst_bucket = max(
                    candidates,
                    key=lambda c: abs(c["coverage"] - nominal),
                )

        try:
            cursor.execute("DROP TABLE IF EXISTS _mlfl_actuals_grid_tmp")
        except sqlite3.Error:
            # TEMP tables are connection-scoped and dropped when the
            # connection closes — failure to explicitly drop is harmless.
            pass

        return {
            "by_lead": by_lead,
            "by_hour_of_day": by_hour_of_day,
            "by_weekday_weekend": by_weekday_weekend,
            "worst_bucket": worst_bucket,
            "overall": overall,
            "tz": tz or "UTC",
        }

    @_locked
    def get_forecast_evolution(
        self,
        experiment: str,
        actuals_table: str,
        n_cycles: int = 12,
        interval_minutes: int = 30,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        source_is_cumulative: bool = False,
    ) -> dict:
        """
        Return the last ``n_cycles`` forecast snapshots plus actuals.

        Each cycle is one forecast issuance — a series of (target_dt,
        predicted) pairs emitted together at the same ``issued_at``.
        The chart overlays them against the resampled actual curve so
        the user can see how predictions evolve as lead time shrinks.

        Parameters
        ----------
        experiment : str
            Experiment name.
        actuals_table : str
            SQL-safe table name for the actuals.
        n_cycles : int
            Number of most-recent forecast issuances to return.
        interval_minutes : int
            Grid interval used to snap actuals to regular bins before
            returning.
        model_name : str, optional
            Restrict to one model. Without this, a champion rotation
            inside the window can mix predictions from two different
            models on the same chart — the "Latest run" line and the
            fan band become incoherent.
        model_version : str, optional
            Restrict to one weight regime of ``model_name``. Same
            motivation as above for retrains under the current champion.

        Returns
        -------
        dict with keys:
            ``cycles``: list of {issued_at, targets: [iso], predictions: [float]}
            ``actuals``: {targets: [iso], values: [float]}
            ``time_range``: {from, to} (ISO strings, bounding window)
        """
        cursor = self.conn.cursor()
        interval_sec = max(60, int(interval_minutes) * 60)

        # Check actuals table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (actuals_table,),
        )
        if not cursor.fetchone():
            return {"error": "No actuals data available yet"}

        # Optional model/version filter — splice into both the issuance
        # query and the per-cycle rows query so cycles and rows agree on
        # the cohort. Without this the fan can mix rotated-out
        # predictions with the current champion.
        _clauses = []
        _params: list = []
        if model_name:
            _clauses.append("model_name = ?")
            _params.append(model_name)
        if model_version:
            _clauses.append("model_version = ?")
            _params.append(model_version)
        model_filter_sql = (" AND " + " AND ".join(_clauses)) if _clauses else ""
        model_filter_param = tuple(_params)

        # Find the last N distinct issuance timestamps for this experiment
        # under the chosen model/version filter.
        try:
            cursor.execute(
                "SELECT DISTINCT issued_at FROM forecast_log "
                f"WHERE experiment = ?{model_filter_sql} "
                "ORDER BY issued_at DESC LIMIT ?",
                (experiment, *model_filter_param, int(n_cycles)),
            )
            issued_ats = [r[0] for r in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.warning(f"Forecast evolution issuance query failed: {e}")
            return {"error": str(e)}

        if not issued_ats:
            return {
                "cycles": [],
                "actuals": {"targets": [], "values": []},
                "time_range": {"from": None, "to": None},
            }

        # Fetch all rows for those issuances in one query, applying the
        # same filter so we never re-introduce rotated-out rows that
        # happen to share an issued_at with the kept cohort.
        placeholders = ",".join("?" * len(issued_ats))
        try:
            cursor.execute(
                f"SELECT issued_at, target_dt, predicted "
                f"FROM forecast_log "
                f"WHERE experiment = ? AND issued_at IN ({placeholders})"
                f"{model_filter_sql} "
                f"ORDER BY issued_at ASC, target_dt ASC",
                (experiment, *issued_ats, *model_filter_param),
            )
            rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.warning(f"Forecast evolution rows query failed: {e}")
            return {"error": str(e)}

        cycles_map: dict = {}
        min_target = None
        max_target = None
        for issued_at, target_dt, predicted in rows:
            cycle = cycles_map.setdefault(
                issued_at, {"issued_at": issued_at, "targets": [], "predictions": []}
            )
            cycle["targets"].append(target_dt)
            cycle["predictions"].append(float(predicted))
            if min_target is None or target_dt < min_target:
                min_target = target_dt
            if max_target is None or target_dt > max_target:
                max_target = target_dt

        # Pull actuals covering the same time window, snapped to the grid.
        # When the source is cumulative, the model emits per-interval
        # deltas to forecast_log, so the actuals need to be diffed to the
        # same space — otherwise the fan chart plots raw cumulative
        # values against delta predictions on one axis and the "Measured"
        # line spikes up to the daily total while predictions hug zero.
        # Adjacency guard: only diff against the previous bin if it's
        # exactly one interval back (mirrors get_forecast_trajectory).
        # Reset guard: clamp negative deltas to 0 so daily-reset sensors
        # don't produce a huge negative spike at midnight.
        actuals_targets: list = []
        actuals_values: list = []
        if min_target and max_target:
            if source_is_cumulative:
                actuals_sql = f"""
                    WITH actuals_grid AS (
                        SELECT
                            strftime('%Y-%m-%d %H:%M:%S',
                                (CAST(strftime('%s', SUBSTR(ds, 1, 19)) AS INTEGER) / ?) * ?,
                                'unixepoch') AS grid_dt,
                            AVG(value) AS value
                        FROM {actuals_table}
                        WHERE SUBSTR(ds, 1, 19) >= ?
                          AND SUBSTR(ds, 1, 19) <= ?
                        GROUP BY grid_dt
                    )
                    SELECT grid_dt,
                        CASE
                            WHEN CAST(strftime('%s', grid_dt) AS INTEGER)
                                 - CAST(strftime('%s', LAG(grid_dt) OVER (ORDER BY grid_dt)) AS INTEGER)
                                 = ?
                            THEN MAX(0, value - LAG(value) OVER (ORDER BY grid_dt))
                            ELSE NULL
                        END AS value
                    FROM actuals_grid
                    ORDER BY grid_dt ASC
                """
                actuals_params: tuple = (
                    interval_sec, interval_sec,
                    min_target, max_target,
                    interval_sec,
                )
            else:
                actuals_sql = f"""
                    SELECT
                        strftime('%Y-%m-%d %H:%M:%S',
                            (CAST(strftime('%s', SUBSTR(ds, 1, 19)) AS INTEGER) / ?) * ?,
                            'unixepoch') AS grid_dt,
                        AVG(value) AS value
                    FROM {actuals_table}
                    WHERE SUBSTR(ds, 1, 19) >= ?
                      AND SUBSTR(ds, 1, 19) <= ?
                    GROUP BY grid_dt
                    ORDER BY grid_dt ASC
                """
                actuals_params = (
                    interval_sec, interval_sec, min_target, max_target,
                )
            try:
                cursor.execute(actuals_sql, actuals_params)
                for grid_dt, value in cursor.fetchall():
                    if value is None:
                        continue
                    actuals_targets.append(grid_dt)
                    actuals_values.append(float(value))
            except sqlite3.Error as e:
                logger.warning(f"Forecast evolution actuals query failed: {e}")

        # Chronological cycle order (oldest first) so callers can fade
        # older traces and emphasize newer ones by index.
        cycles = [cycles_map[k] for k in sorted(cycles_map.keys())]

        return {
            "cycles": cycles,
            "actuals": {"targets": actuals_targets, "values": actuals_values},
            "time_range": {"from": min_target, "to": max_target},
        }

    @_locked
    def get_forecast_stability(
        self,
        experiment: str,
        max_age_days: int = 30,
        source_is_cumulative: bool = False,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        day_offset_hours: Optional[float] = None,
    ) -> dict:
        """
        Return self-consistency metrics across forecast issuances.

        Stability is the complement of accuracy: accuracy asks "how
        close is the prediction to reality?", stability asks "how much
        does the prediction for the same future target swing from one
        issuance to the next?" A stable model produces similar
        forecasts each cycle; an unstable one oscillates wildly
        issuance-to-issuance even when the target doesn't move.

        Parameters
        ----------
        experiment : str
            Experiment name.
        max_age_days : int
            Only consider forecasts issued in the last N days.
        source_is_cumulative : bool
            When True, also compute per-day-total stability by summing
            per-interval deltas within each cycle × calendar-day pair.
            For instantaneous sources this doesn't make physical sense.
        model_name : str, optional
            Restrict to one model. Without this, runs from a rotated-
            out model mix with the current champion and inflate the
            cross-run disagreement metric.

        Returns
        -------
        dict with keys:
            ``per_timestep``: per target_dt, across-cycle mean / std /
                cv_pct / n_cycles (only target_dts with >=2 cycles)
            ``daily_totals``: when source_is_cumulative, per
                calendar-day summary of across-cycle daily-total
                stability
            ``summary``: median CVs + total cycles analysed + date range
        """
        cursor = self.conn.cursor()
        cutoff_str = (
            datetime.utcnow() - pd.Timedelta(days=max_age_days)
        ).strftime("%Y-%m-%d %H:%M:%S")
        # Combined name + version filter. Without the version split,
        # cycles under the same model_name but different weight regimes
        # (i.e. around a retrain) pool together and inflate cross-run
        # disagreement — the original motivation for versioning.
        _clauses = []
        _params: list = []
        if model_name:
            _clauses.append("model_name = ?")
            _params.append(model_name)
        if model_version:
            _clauses.append("model_version = ?")
            _params.append(model_version)
        model_filter_sql = (" AND " + " AND ".join(_clauses)) if _clauses else ""
        model_filter_param = tuple(_params)

        # --- Per-timestep cross-cycle stability ---
        # SQLite has no STDDEV; compute it via the sum-of-squares
        # identity: Var(X) = E[X^2] - E[X]^2. The subtraction can go
        # very slightly negative on perfectly-constant columns due to
        # float rounding, so clamp before SQRT.
        #
        # v2.34.0: partition by (model_name, model_version) so cross-
        # cohort pooling is impossible. The previous query grouped by
        # target_dt alone — when the caller's filter widened (e.g.
        # `model=catboost` with no version pin, or `model=all`), a
        # single target_dt could pool predictions from old + new
        # weight regimes, mixing tuning runs together and inflating
        # the std as "model instability" when it was actually
        # "different weights making different predictions". With this
        # change, each (target_dt, model_name, model_version) cohort
        # produces its own mean/std; ROW_NUMBER picks one cohort per
        # target_dt — preferring the one with most data, breaking
        # ties by most recent version. The output shape is unchanged
        # so the caller stays oblivious.
        try:
            cursor.execute(
                f"""
                WITH per_cohort AS (
                    SELECT
                        target_dt,
                        model_name,
                        model_version,
                        AVG(predicted)              AS mean_p,
                        AVG(predicted * predicted) AS mean_pp,
                        COUNT(DISTINCT issued_at)  AS n_cycles
                    FROM forecast_log
                    WHERE experiment = ?
                      AND issued_at >= ?
                      {model_filter_sql}
                    GROUP BY target_dt, model_name, model_version
                    HAVING n_cycles >= 2
                ),
                ranked AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY target_dt
                        ORDER BY n_cycles DESC, model_version DESC
                    ) AS rn
                    FROM per_cohort
                )
                SELECT
                    target_dt,
                    mean_p,
                    CASE WHEN mean_pp - mean_p * mean_p > 0
                         THEN SQRT(mean_pp - mean_p * mean_p)
                         ELSE 0 END AS std_p,
                    n_cycles
                FROM ranked
                WHERE rn = 1
                ORDER BY target_dt ASC
                """,
                (experiment, cutoff_str, *model_filter_param),
            )
            rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.warning(f"Forecast stability per-timestep query failed: {e}")
            return {"error": str(e)}

        target_dts: list = []
        means: list = []
        stds: list = []
        cv_pcts: list = []
        n_cycles_list: list = []
        for target_dt, mean_p, std_p, n_cycles in rows:
            mean_p = float(mean_p or 0)
            std_p = float(std_p or 0)
            # CV undefined when mean ≈ 0. If std is also ≈ 0 the
            # forecast is genuinely flat and stable (CV = 0). If std
            # is non-zero we have predictions oscillating around zero,
            # which the CV ratio can't describe — skip those rows so
            # they don't pollute the median and don't surface as a
            # misleading "perfectly stable" point.
            if abs(mean_p) < 1e-9:
                if std_p < 1e-9:
                    cv = 0.0
                else:
                    continue
            else:
                cv = 100.0 * std_p / abs(mean_p)
            target_dts.append(target_dt)
            means.append(round(mean_p, 4))
            stds.append(round(std_p, 4))
            cv_pcts.append(round(cv, 2))
            n_cycles_list.append(int(n_cycles))

        import statistics as _st
        median_step_cv = round(_st.median(cv_pcts), 2) if cv_pcts else 0.0

        # --- Daily-total stability (cumulative-source experiments only) ---
        daily_totals: list = []
        if source_is_cumulative:
            # Full-coverage gate: a cycle only contributes to a day's
            # daily-total spread if it forecast EVERY bin of that day.
            # Otherwise a 08:00 issuance (covers Mon 08:30→23:30, ≈32
            # bins of Mon) gets compared against a prior-day issuance
            # that covered all 48 bins of Mon — the partial cycle's
            # SUM is mechanically smaller, and that coverage gap
            # masquerades as model disagreement. For a daily-cumulative
            # sensor where overnight carries most of the demand, this
            # artefact alone can produce 50–70% CV. Gating to cycles
            # with n_bins = MAX(n_bins) for that day keeps only the
            # apples-to-apples comparisons.
            #
            # Day boundary follows HA's local midnight, not UTC — the
            # physical reset happens at HA local midnight (because
            # that's when the `_today` counter resets), so "day X"
            # should integrate 00:00→24:00 local. We approximate the
            # local day by shifting target_dt by `day_offset_hours`
            # (computed in Python from the HA time zone) before taking
            # the YYYY-MM-DD prefix. A static offset is exact away from
            # DST transitions and ±1h off for the transition day only,
            # which is an acceptable trade for keeping the bucketing in
            # pure SQLite.
            off = float(day_offset_hours or 0.0)
            off_seconds = int(off * 3600)
            if off_seconds == 0:
                day_expr = "SUBSTR(target_dt, 1, 10)"
                day_params: tuple = ()
            else:
                day_expr = (
                    "strftime('%Y-%m-%d',"
                    " CAST(strftime('%s', target_dt) AS INTEGER) + ?,"
                    " 'unixepoch')"
                )
                day_params = (off_seconds,)
            try:
                # v2.34.0: per_cycle_day now also tracks model identity
                # so the day_max + full_cov + final aggregation each
                # respect cohort boundaries. A retrain mid-window
                # produced two regimes' worth of daily totals being
                # pooled into a single day's std — meaningless and
                # alarming. Now each (day, model_name, model_version)
                # cohort produces its own mean/std; ROW_NUMBER picks
                # the cohort with most cycles per day (ties broken by
                # most recent version) and returns its scalars.
                cursor.execute(
                    f"""
                    WITH per_cycle_day AS (
                        SELECT
                            issued_at,
                            model_name,
                            model_version,
                            {day_expr}                   AS day,
                            SUM(predicted)               AS daily_total,
                            COUNT(*)                     AS n_bins
                        FROM forecast_log
                        WHERE experiment = ?
                          AND issued_at >= ?
                          {model_filter_sql}
                        GROUP BY issued_at, model_name, model_version, day
                    ),
                    day_cohort_max AS (
                        SELECT day, model_name, model_version,
                               MAX(n_bins) AS max_bins
                        FROM per_cycle_day
                        GROUP BY day, model_name, model_version
                    ),
                    full_cov AS (
                        SELECT pcd.*
                        FROM per_cycle_day pcd
                        INNER JOIN day_cohort_max dcm
                          ON dcm.day = pcd.day
                         AND dcm.model_name = pcd.model_name
                         AND (
                             (dcm.model_version IS NULL AND pcd.model_version IS NULL)
                             OR dcm.model_version = pcd.model_version
                         )
                         AND pcd.n_bins = dcm.max_bins
                    ),
                    per_cohort AS (
                        SELECT
                            day,
                            model_name,
                            model_version,
                            AVG(daily_total)                AS mean_total,
                            AVG(daily_total * daily_total) AS mean_tt,
                            COUNT(DISTINCT issued_at)       AS n_cycles,
                            MAX(n_bins)                     AS max_bins_in_day
                        FROM full_cov
                        GROUP BY day, model_name, model_version
                        HAVING n_cycles >= 2
                    ),
                    ranked AS (
                        SELECT *, ROW_NUMBER() OVER (
                            PARTITION BY day
                            ORDER BY n_cycles DESC, model_version DESC
                        ) AS rn
                        FROM per_cohort
                    )
                    SELECT
                        day,
                        mean_total,
                        mean_tt,
                        n_cycles,
                        max_bins_in_day
                    FROM ranked
                    WHERE rn = 1
                    ORDER BY day ASC
                    """,
                    (*day_params, experiment, cutoff_str, *model_filter_param),
                )
                for day, mean_total, mean_tt, n_cycles, max_bins in cursor.fetchall():
                    mean_total = float(mean_total or 0)
                    var = float(mean_tt or 0) - mean_total * mean_total
                    std_total = (var ** 0.5) if var > 0 else 0.0
                    cv_day = (100.0 * std_total / abs(mean_total)) if abs(mean_total) > 1e-9 else 0.0
                    daily_totals.append({
                        "day": day,
                        "mean_total": round(mean_total, 3),
                        "std_total": round(std_total, 3),
                        "cv_pct": round(cv_day, 2),
                        "n_cycles": int(n_cycles),
                        "max_bins_in_day": int(max_bins),
                    })
            except sqlite3.Error as e:
                logger.warning(f"Forecast stability daily-total query failed: {e}")

        daily_cvs = [d["cv_pct"] for d in daily_totals]
        median_daily_cv = round(_st.median(daily_cvs), 2) if daily_cvs else None

        # --- Summary / cycle count ---
        try:
            cursor.execute(
                "SELECT COUNT(DISTINCT issued_at), MIN(issued_at), MAX(issued_at) "
                f"FROM forecast_log WHERE experiment = ? AND issued_at >= ?{model_filter_sql}",
                (experiment, cutoff_str, *model_filter_param),
            )
            cyc_row = cursor.fetchone()
        except sqlite3.Error:
            cyc_row = (0, None, None)

        # v2.34.0: per-cohort breakdown of run-to-run std.
        # The main `per_timestep` arrays above carry the
        # dominant-cohort value per target_dt (post-Commit-1
        # partitioning). The `cohorts` array gives the full
        # per-(model_name, model_version) breakdown so the
        # frontend multi-trace stability chart can render one
        # line per cohort. Skipped when the caller has pinned
        # a specific cohort (only one cohort in data, multi-
        # trace would be redundant).
        stability_cohorts: list = []
        if not (model_name and model_version):
            try:
                cursor.execute(
                    f"""
                    SELECT
                        target_dt, model_name, model_version,
                        AVG(predicted) AS mean_p,
                        AVG(predicted * predicted) AS mean_pp,
                        COUNT(DISTINCT issued_at) AS n_cycles
                    FROM forecast_log
                    WHERE experiment = ?
                      AND issued_at >= ?
                      {model_filter_sql}
                    GROUP BY target_dt, model_name, model_version
                    HAVING n_cycles >= 2
                    ORDER BY model_name, model_version, target_dt
                    """,
                    (experiment, cutoff_str, *model_filter_param),
                )
                cohort_rows = cursor.fetchall()
            except sqlite3.Error as e:
                logger.warning(f"Per-cohort stability query failed: {e}")
                cohort_rows = []

            cohort_map: dict = {}
            for tdt, mn, mv, mean_p, mean_pp, nc in cohort_rows:
                mean_p = float(mean_p or 0)
                mean_pp = float(mean_pp or 0)
                var = mean_pp - mean_p * mean_p
                std_p = (var ** 0.5) if var > 0 else 0.0
                if abs(mean_p) < 1e-9:
                    if std_p < 1e-9:
                        cv = 0.0
                    else:
                        continue
                else:
                    cv = 100.0 * std_p / abs(mean_p)
                key = (mn, mv)
                if key not in cohort_map:
                    cohort_map[key] = {
                        "model_name": mn,
                        "model_version": mv,
                        "per_timestep": {
                            "target_dt": [],
                            "mean": [],
                            "std": [],
                            "cv_pct": [],
                            "n_cycles": [],
                        },
                    }
                pt = cohort_map[key]["per_timestep"]
                pt["target_dt"].append(tdt)
                pt["mean"].append(round(mean_p, 4))
                pt["std"].append(round(std_p, 4))
                pt["cv_pct"].append(round(cv, 2))
                pt["n_cycles"].append(int(nc))
            stability_cohorts = list(cohort_map.values())

        return {
            "experiment": experiment,
            "per_timestep": {
                "target_dt": target_dts,
                "mean": means,
                "std": stds,
                "cv_pct": cv_pcts,
                "n_cycles": n_cycles_list,
            },
            "cohorts": stability_cohorts,
            "daily_totals": daily_totals,
            "summary": {
                "median_step_cv_pct": median_step_cv,
                "median_daily_cv_pct": median_daily_cv,
                "total_cycles": int(cyc_row[0]) if cyc_row else 0,
                "steps_analysed": len(target_dts),
                "date_range": {
                    "from": cyc_row[1] if cyc_row else None,
                    "to": cyc_row[2] if cyc_row else None,
                },
            },
        }

    @_locked
    def get_retrain_events(
        self,
        experiment: str,
        days: int = 30,
        model_name: Optional[str] = None,
    ) -> list:
        """Return distinct retrain events for *experiment* in the window.

        A retrain is detected as a new ``model_version`` value appearing in
        ``forecast_log`` — every retrain stamps a fresh version on every
        subsequent log row. The earliest ``issued_at`` per (model_name,
        model_version) approximates the moment the retrain finished.

        Returns a list of dicts (ordered by first_seen ascending) shaped
        ``{model_name, model_version, first_seen, n_forecasts}``. Rows
        with a NULL model_version (legacy / pre-tag forecasts) are
        excluded.
        """
        cursor = self.conn.cursor()
        cutoff = (datetime.utcnow() - pd.Timedelta(days=days)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        params: list = [experiment, cutoff]
        extra = ""
        if model_name:
            extra = " AND model_name = ?"
            params.append(model_name)
        cursor.execute(
            f"""
            SELECT model_name, model_version,
                   MIN(issued_at) AS first_seen,
                   COUNT(*)       AS n_forecasts
            FROM forecast_log
            WHERE experiment = ?
              AND issued_at >= ?
              AND model_version IS NOT NULL
              AND model_version != ''
              {extra}
            GROUP BY model_name, model_version
            ORDER BY first_seen ASC
            """,
            params,
        )
        return [
            {
                "model_name": row["model_name"],
                "model_version": row["model_version"],
                "first_seen": row["first_seen"],
                "n_forecasts": int(row["n_forecasts"]),
            }
            for row in cursor.fetchall()
        ]

    @_locked
    def get_forecast_log_stats(
        self,
        experiment: str,
        default_model: Optional[str] = None,
        default_version: Optional[str] = None,
    ) -> dict:
        """Per-cohort row counts + totals for the debug log-stats panel.

        Moved out of the web handler in v2.41.0 (audit F4): the handler
        used ``db.conn.cursor()`` directly, bypassing both the lock
        discipline this class documents and the thread offload — the
        only call site in the codebase that touched the shared
        connection without ``self._lock``.

        Returns ``{"cohorts": [...], "totals": {...},
        "targets_with_multi_issuances": int | None}``.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT model_name,
                   COALESCE(model_version, '(null)') AS mv,
                   COUNT(*) AS n,
                   MIN(issued_at) AS first_issued_at,
                   MAX(issued_at) AS last_issued_at,
                   MAX(target_dt) AS last_target_dt
            FROM forecast_log
            WHERE experiment = ?
            GROUP BY model_name, model_version
            ORDER BY last_issued_at DESC
            """,
            (experiment,),
        )
        cohorts = [
            {
                "model_name": r[0],
                "model_version": None if r[1] == "(null)" else r[1],
                "rows": int(r[2]),
                "first_issued_at": r[3],
                "last_issued_at": r[4],
                "last_target_dt": r[5],
            }
            for r in cursor.fetchall()
        ]
        cursor.execute(
            "SELECT COUNT(*), MIN(issued_at), MAX(issued_at) "
            "FROM forecast_log WHERE experiment = ?",
            (experiment,),
        )
        total_row = cursor.fetchone()
        targets_with_multi_issuances = None
        if default_model and default_version:
            cursor.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT target_dt
                    FROM forecast_log
                    WHERE experiment = ?
                      AND model_name = ?
                      AND model_version = ?
                    GROUP BY target_dt
                    HAVING COUNT(DISTINCT issued_at) >= 2
                )
                """,
                (experiment, default_model, default_version),
            )
            targets_with_multi_issuances = int(cursor.fetchone()[0])
        return {
            "cohorts": cohorts,
            "totals": {
                "rows": int(total_row[0]) if total_row else 0,
                "first_issued_at": total_row[1] if total_row else None,
                "last_issued_at": total_row[2] if total_row else None,
            },
            "targets_with_multi_issuances": targets_with_multi_issuances,
        }

    @_locked
    def cleanup_forecast_log(
        self,
        experiment: str,
        oldest_datetime: datetime,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        exclude_model_name: Optional[str] = None,
        exclude_model_version: Optional[str] = None,
    ) -> int:
        """Delete forecast log entries older than the specified datetime.

        Filters
        -------
        model_name / model_version : optional positive filters to delete only
            rows matching the given cohort. Omit either to leave that
            dimension unfiltered.
        exclude_model_name / exclude_model_version : optional negative
            filters used by the promote path — delete only rows that do
            **not** belong to the new champion cohort, so analytics for
            the incoming model survive a champion switch.
        """
        oldest_str = oldest_datetime.strftime("%Y-%m-%d %H:%M:%S")
        where = ["experiment = ?", "issued_at < ?"]
        params: list = [experiment, oldest_str]
        if model_name is not None:
            where.append("model_name = ?")
            params.append(model_name)
        if model_version is not None:
            where.append("model_version = ?")
            params.append(model_version)
        if exclude_model_name is not None:
            # Exclude rows that match BOTH name and (if given) version.
            if exclude_model_version is not None:
                where.append(
                    "NOT (model_name = ? AND model_version = ?)"
                )
                params.extend([exclude_model_name, exclude_model_version])
            else:
                where.append("model_name != ?")
                params.append(exclude_model_name)
        sql = "DELETE FROM forecast_log WHERE " + " AND ".join(where)
        cursor = self.conn.cursor()
        try:
            cursor.execute(sql, params)
            self.conn.commit()
            deleted = cursor.rowcount
            if deleted:
                logger.info(f"Pruned {deleted} old forecast_log rows for {experiment}")
            return deleted
        except sqlite3.Error as e:
            logger.error(f"Error pruning forecast_log for {experiment}: {e}", exc_info=True)
            self.conn.rollback()
            return 0

    @_locked
    def delete_forecast_log(self, experiment: str) -> int:
        """Delete all forecast log entries for an experiment."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "DELETE FROM forecast_log WHERE experiment = ?", (experiment,)
            )
            self.conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f"Error deleting forecast_log for {experiment}: {e}", exc_info=True)
            self.conn.rollback()
            return 0

    # ------------------------------------------------------------------
    # External forecast log (third-party forecast trajectories)
    # ------------------------------------------------------------------
    # Mirrors forecast_log but for a forecast NOT produced by this add-on
    # (an external HA sensor — Solcast, a utility curve, etc.). Used by the
    # per-experiment "External Comparison" tab to score this add-on's
    # forecast head-to-head against the external one. Only the
    # ``attribute`` external-forecast mode writes here (it has a real
    # lead-time / trajectory shape); the ``state`` mode reuses the ordinary
    # cached-history table (store_history) since it is just a time-series.

    @_locked
    def ensure_external_forecast_log_table(self) -> None:
        """Create the external_forecast_log table if it doesn't exist.

        ``source`` is the external entity_id, so a single experiment can be
        compared against several third-party forecasts at once (each is its
        own cohort in this table).
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS external_forecast_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment   TEXT    NOT NULL,
                source       TEXT    NOT NULL DEFAULT '',
                issued_at    TEXT    NOT NULL,
                target_dt    TEXT    NOT NULL,
                lead_minutes INTEGER NOT NULL,
                value        REAL    NOT NULL,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Migrate pre-existing tables (created before multi-source support)
        # that lack the ``source`` column.
        cursor.execute("PRAGMA table_info(external_forecast_log)")
        cols = {row[1] for row in cursor.fetchall()}
        if "source" not in cols:
            cursor.execute(
                "ALTER TABLE external_forecast_log "
                "ADD COLUMN source TEXT NOT NULL DEFAULT ''"
            )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_extflog_exp_src_target "
            "ON external_forecast_log(experiment, source, target_dt)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_extflog_exp_issued "
            "ON external_forecast_log(experiment, issued_at)"
        )
        # One-time cleanup of trajectories logged before content-change
        # detection: each forecast cycle used to re-log an unchanged external
        # trajectory as a fresh issuance, so every target had a ~one-cycle-old
        # "latest" issuance and the reported lead collapsed to a single cycle
        # (e.g. a source that refreshes a few times a day reading as 15 min).
        # Collapse runs of identical issuances down to their earliest one —
        # exactly what the capture-time de-duplication now produces — so
        # existing data reports true leads without losing history or restarting
        # the warming-up window. Guarded by schema_versions so it runs once.
        if 2 not in self._applied_versions():
            try:
                removed = self._collapse_external_duplicates(cursor)
                if removed:
                    logger.info(
                        "Collapsed %d duplicate external forecast issuance "
                        "row(s) to their original issue time", removed,
                    )
            except sqlite3.Error as e:
                logger.error(
                    "External forecast de-duplication migration failed: %s",
                    e, exc_info=True,
                )
            self._record_version(2)
        self.conn.commit()
        logger.debug("Ensured external_forecast_log table")

    def _collapse_external_duplicates(self, cursor) -> int:
        """Collapse consecutive identical external-forecast issuances to the
        earliest of each run, returning the number of rows deleted.

        Mirrors the capture-time content-change test (see
        ``main._capture_external_forecast``): an issuance whose values match
        the last *kept* trajectory over their overlapping targets — within the
        same tolerance, ``max(1e-4, 0.005·|value|)`` — is a re-log of the same
        forecast, not a new one, so it is dropped. The earliest issuance of a
        run is kept, preserving the true (longest) lead.
        """
        cursor.execute(
            "SELECT DISTINCT experiment, source FROM external_forecast_log"
        )
        total_removed = 0
        for experiment, source in cursor.fetchall():
            total_removed += self._collapse_source_duplicates(
                cursor, experiment, source,
            )
        return total_removed

    def _collapse_source_duplicates(self, cursor, experiment, source) -> int:
        """De-duplicate one source's logged issuances in place. See
        ``_collapse_external_duplicates`` for the matching rule."""
        cursor.execute(
            "SELECT DISTINCT issued_at FROM external_forecast_log "
            "WHERE experiment = ? AND source = ? ORDER BY issued_at ASC",
            (experiment, source),
        )
        issuances = [r[0] for r in cursor.fetchall()]
        if len(issuances) < 2:
            return 0
        kept: Optional[dict] = None  # last kept issuance, target_dt → value
        duplicates: list = []
        for iss in issuances:
            cursor.execute(
                "SELECT target_dt, value FROM external_forecast_log "
                "WHERE experiment = ? AND source = ? AND issued_at = ?",
                (experiment, source, iss),
            )
            traj = {r[0]: round(float(r[1]), 4) for r in cursor.fetchall()}
            if kept is not None:
                overlap = [k for k in traj if k in kept]
                if overlap and all(
                    abs(traj[k] - kept[k]) <= max(1e-4, 0.005 * abs(traj[k]))
                    for k in overlap
                ):
                    duplicates.append(iss)
                    continue
            kept = traj
        removed = 0
        for iss in duplicates:
            cursor.execute(
                "DELETE FROM external_forecast_log "
                "WHERE experiment = ? AND source = ? AND issued_at = ?",
                (experiment, source, iss),
            )
            removed += cursor.rowcount
        return removed

    @_locked
    def get_last_external_issued_at(
        self, experiment: str, source: str,
    ) -> Optional[datetime]:
        """Most recent ``issued_at`` already logged for an external source,
        or ``None`` if none. Used to de-duplicate captures: if the source's
        update time hasn't advanced past this, the trajectory is unchanged
        and re-logging it would just bloat the log and misreport its lead."""
        try:
            self.ensure_external_forecast_log_table()
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT MAX(issued_at) FROM external_forecast_log "
                "WHERE experiment = ? AND source = ?",
                (experiment, source),
            )
            row = cursor.fetchone()
        except sqlite3.Error as e:
            logger.error(
                "Error reading last external issued_at for %s/%s: %s",
                experiment, source, e,
            )
            return None
        if not row or not row[0]:
            return None
        try:
            return datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return None

    @_locked
    def get_last_external_trajectory(
        self, experiment: str, source: str,
    ) -> dict:
        """The most-recently-issued trajectory for an external source as a
        ``{target_dt_str: value}`` map. Used to detect whether the source's
        forecast *content* has changed (a new issuance) — a robust,
        source-agnostic alternative to trusting HA ``last_updated``, which on
        many integrations bumps far more often than the forecast actually
        refreshes."""
        try:
            self.ensure_external_forecast_log_table()
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT MAX(issued_at) FROM external_forecast_log "
                "WHERE experiment = ? AND source = ?",
                (experiment, source),
            )
            row = cursor.fetchone()
            if not row or not row[0]:
                return {}
            cursor.execute(
                "SELECT target_dt, value FROM external_forecast_log "
                "WHERE experiment = ? AND source = ? AND issued_at = ?",
                (experiment, source, row[0]),
            )
            return {r[0]: r[1] for r in cursor.fetchall()}
        except sqlite3.Error as e:
            logger.error(
                "Error reading last external trajectory for %s/%s: %s",
                experiment, source, e,
            )
            return {}

    def log_external_forecast(
        self,
        experiment: str,
        source: str,
        issued_at: datetime,
        targets: list,
        values: list,
    ) -> int:
        """Bulk-insert an external forecast trajectory snapshot.

        ``source`` is the external entity_id (which third-party forecast).
        Same shape as ``log_forecast`` minus the model columns: each row is
        one (source, issued_at, target_dt, lead_minutes, value). Non-finite
        values are skipped so a partially-NaN external trajectory doesn't
        poison the comparison join. ``issued_at`` should be the source's own
        update time (HA ``last_updated``) so ``lead_minutes`` reflects the
        forecast's true age, not when the add-on snapshotted it.

        Returns the number of rows inserted.
        """
        import math
        self.ensure_external_forecast_log_table()
        issued_str = issued_at.strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        for ts, val in zip(targets, values):
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fval):
                continue
            target_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            lead_min = int((ts - issued_at).total_seconds() / 60)
            rows.append((experiment, source, issued_str, target_str, lead_min, fval))
        if not rows:
            return 0
        cursor = self.conn.cursor()
        try:
            cursor.executemany(
                "INSERT INTO external_forecast_log "
                "(experiment, source, issued_at, target_dt, lead_minutes, value) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            self.conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(
                f"Error logging external forecast for {experiment}: {e}",
                exc_info=True,
            )
            self.conn.rollback()
            return 0

    @_locked
    def cleanup_external_forecast_log(
        self, experiment: str, oldest_datetime: datetime,
    ) -> int:
        """Delete external_forecast_log rows issued before *oldest_datetime*."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='external_forecast_log'"
        )
        if not cursor.fetchone():
            return 0
        oldest_str = oldest_datetime.strftime("%Y-%m-%d %H:%M:%S")
        try:
            cursor.execute(
                "DELETE FROM external_forecast_log "
                "WHERE experiment = ? AND issued_at < ?",
                (experiment, oldest_str),
            )
            self.conn.commit()
            deleted = cursor.rowcount
            if deleted:
                logger.info(
                    f"Pruned {deleted} old external_forecast_log rows "
                    f"for {experiment}"
                )
            return deleted
        except sqlite3.Error as e:
            logger.error(
                f"Error pruning external_forecast_log for {experiment}: {e}",
                exc_info=True,
            )
            self.conn.rollback()
            return 0

    @_locked
    def delete_external_forecast_log(self, experiment: str) -> int:
        """Delete all external_forecast_log entries for an experiment."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='external_forecast_log'"
        )
        if not cursor.fetchone():
            return 0
        try:
            cursor.execute(
                "DELETE FROM external_forecast_log WHERE experiment = ?",
                (experiment,),
            )
            self.conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(
                f"Error deleting external_forecast_log for {experiment}: {e}",
                exc_info=True,
            )
            self.conn.rollback()
            return 0

    @_locked
    def delete_external_forecast_source(self, experiment: str, source: str) -> int:
        """Delete external_forecast_log rows for one source (entity) of an
        experiment — used when a third-party forecast is removed so a later
        re-add starts clean."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='external_forecast_log'"
        )
        if not cursor.fetchone():
            return 0
        try:
            cursor.execute(
                "DELETE FROM external_forecast_log "
                "WHERE experiment = ? AND source = ?",
                (experiment, source),
            )
            self.conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(
                f"Error deleting external_forecast_log source {source} "
                f"for {experiment}: {e}", exc_info=True,
            )
            self.conn.rollback()
            return 0

    @_locked
    def get_external_forecast_comparison(
        self,
        experiment: str,
        actuals_table: str,
        externals: list,
        max_age_days: int = 30,
        interval_minutes: int = 30,
        evaluation_mode: str = "raw",
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        analysis_mode: str = "per_interval",
        target_unit: Optional[str] = None,
        excluded_dates: Optional[list] = None,
        reset_at: Optional[str] = None,
        day_offset_hours: Optional[float] = None,
    ) -> dict:
        """Score this add-on's forecast against one or more external forecasts.

        ``externals`` is a list of specs, each a dict with ``entity``,
        ``table`` (state-mode cached-history table or None), ``mode``,
        ``scale``, ``is_cumulative`` (None = auto-detect), ``label`` and
        ``unit`` (HA unit_of_measurement, for unit-aware conversion).

        Unit-aware: each series is converted to a common per-interval ENERGY
        canonical using its HA unit (power → ×interval_hours; cumulative
        energy → differenced) so a cumulative kWh sensor lines up with an
        instantaneous kW target. When a unit isn't a recognised power/energy
        unit the series is left in its raw evaluation space and the
        scale-mismatch guard flags it.

        ``analysis_mode``: ``per_interval`` (per-bin demand, in the target's
        native unit) or ``cumulative`` (running daily total in kWh / the
        target unit). The lead-time curve is always per-interval.

        ``excluded_dates``: local calendar dates (``YYYY-MM-DD``) to drop from
        every series so a day of corrupt sensor data can't pollute the
        comparison. ``reset_at``: a UTC "restart" floor — data before it is
        ignored entirely, so the head-to-head starts fresh from that moment.
        ``day_offset_hours``: UTC→HA-local offset so the excluded-date match
        follows the same local day the rest of the tab buckets by.
        """
        import numpy as np

        now = datetime.utcnow()
        cutoff = now - pd.Timedelta(days=max_age_days)
        # "Restart comparison" floor: ignore everything before reset_at so the
        # comparison begins fresh from the restart point (the underlying logs
        # and the Forecast Accuracy tab are untouched). Whichever of the
        # rolling window / reset floor is more recent wins.
        if reset_at:
            try:
                reset_ts = pd.Timestamp(str(reset_at).replace("Z", "").strip())
                if pd.notna(reset_ts) and reset_ts > pd.Timestamp(cutoff):
                    cutoff = reset_ts.to_pydatetime()
            except (ValueError, TypeError):
                logger.debug("Ignoring unparseable comparison reset_at %r", reset_at)
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        bucket_min = max(1, int(interval_minutes))
        freq = f"{bucket_min}min"
        interval_hours = bucket_min / 60.0
        target_cumulative = evaluation_mode == "increment"
        if analysis_mode not in ("per_interval", "cumulative"):
            analysis_mode = "per_interval"

        # Unit dimension classification → (dimension, scale-to-base) where
        # base power = kW and base energy = kWh.
        _POWER = {"w": 0.001, "watt": 0.001, "watts": 0.001, "kw": 1.0, "mw": 1000.0}
        _ENERGY = {"wh": 0.001, "kwh": 1.0, "mwh": 1000.0}

        def _classify_unit(u):
            if not u:
                return ("unknown", 1.0)
            k = str(u).strip().lower()
            if k in _POWER:
                return ("power", _POWER[k])
            if k in _ENERGY:
                return ("energy", _ENERGY[k])
            return ("other", 1.0)

        t_dim, t_base = _classify_unit(target_unit)
        unit_aware = t_dim in ("power", "energy")

        cursor = self.conn.cursor()

        def _table_exists(name: Optional[str]) -> bool:
            if not name:
                return False
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (name,),
            )
            return cursor.fetchone() is not None

        def _iso_utc(ts) -> str:
            ts = pd.Timestamp(ts)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            return ts.isoformat()

        def _raw_grid(rows) -> "pd.Series":
            """Grid-align a (ds, value) frame to a Series (last value per bin);
            NO differencing — canonicalisation handles that."""
            df = pd.DataFrame(rows, columns=["ds", "value"])
            if df.empty:
                return pd.Series(dtype="float64")
            df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["ds"]).sort_values("ds")
            if df.empty:
                return pd.Series(dtype="float64")
            df["grid"] = df["ds"].dt.floor(freq)
            return df.groupby("grid")["value"].last().sort_index()

        def _looks_cumulative(s) -> bool:
            """Heuristic: a running total is (almost) monotonic non-decreasing
            between resets, so the vast majority of consecutive diffs are >= 0.
            A per-interval signal has ~50% negative diffs."""
            sd = s.dropna() if s is not None else None
            if sd is None or len(sd) < 6:
                return False
            d = sd.diff().dropna()
            if d.empty:
                return False
            return float((d >= 0).mean()) >= 0.8

        def _diff_reset(s):
            full = pd.date_range(s.index.min(), s.index.max(), freq=freq)
            d = s.reindex(full).diff()
            d[d < 0] = np.nan  # daily-reset / rollover guard
            return d

        def _to_canon(s, dim, base, is_cum):
            """Raw grid series → per-interval ENERGY (kWh/interval) when the
            unit is power/energy; else returns the series unchanged (best
            effort) so the scale guard can flag it."""
            if s is None or s.empty:
                return s
            s = s.astype(float)
            if base and base != 1.0:
                s = s * base
            if dim == "power":
                return s * interval_hours
            if dim == "energy":
                return _diff_reset(s) if is_cum else s
            return s  # unknown dimension — left raw

        def _legacy_eval(s, is_cum):
            """Pre-unit-aware behaviour: per-interval deltas for cumulative
            sources, raw otherwise."""
            if s is None or s.empty:
                return s
            return _diff_reset(s.astype(float)) if is_cum else s

        def _cumsum_daily(s):
            if s is None or s.empty:
                return s
            df = s.to_frame("v")
            df["day"] = df.index.normalize()
            df["c"] = df["v"].fillna(0.0).groupby(df["day"]).cumsum()
            return df["c"]

        def _native_pi(canon):
            """Per-interval canonical → the target's native per-interval unit
            for display (kW for a power target; kWh/interval otherwise)."""
            if unit_aware and t_dim == "power" and canon is not None and not canon.empty:
                return canon / interval_hours
            return canon

        def _display(canon):
            if analysis_mode == "cumulative":
                return _cumsum_daily(canon)
            return _native_pi(canon)

        def _canon_factor(dim, base):
            """Multiplier turning a per-interval VALUE into per-interval energy
            (power values still need ×interval_hours)."""
            if not unit_aware:
                return 1.0
            if dim == "power":
                return base * interval_hours
            if dim == "energy":
                return base
            return 1.0

        def _median_lead(df):
            if df is None or df.empty or "lead_minutes" not in df.columns \
                    or "grid" not in df.columns:
                return None
            rep = df.groupby("grid")["lead_minutes"].last().dropna()
            return round(float(rep.median()), 1) if not rep.empty else None

        externals = externals or []
        # Display unit label for the axes.
        if analysis_mode == "cumulative":
            display_unit = "kWh" if unit_aware else (
                ((target_unit or "") + " (cumulative)").strip()
            )
        else:
            if unit_aware:
                display_unit = (target_unit or "kW") if t_dim == "power" else "kWh/interval"
            else:
                display_unit = target_unit or ""

        result: dict = {
            "configured": bool(externals),
            "evaluation_mode": evaluation_mode,
            "analysis_mode": analysis_mode,
            "interval_minutes": bucket_min,
            "unit_aware": unit_aware,
            "display_unit": display_unit,
            "lead_unit": "kWh/interval" if unit_aware else (target_unit or ""),
            "overlay": {"ds": [], "actual": [], "app": [], "externals": []},
            "comparisons": [],
            "lead_time": None,
            "skill": None,
            "date_range": {},
            "app_points": 0,
            "grid_points": 0,
        }

        if not _table_exists(actuals_table):
            result["error"] = "No actuals data available yet"
            result["empty_reason"] = "no_actuals"
            return result

        # --- actuals → canonical per-interval ---
        cursor.execute(
            f"SELECT ds, value FROM {actuals_table} WHERE ds >= ? ORDER BY ds",
            (cutoff_str,),
        )
        actual_raw = _raw_grid(cursor.fetchall())
        if unit_aware:
            actual_canon = _to_canon(actual_raw, t_dim, t_base, target_cumulative)
        else:
            actual_canon = _legacy_eval(actual_raw, target_cumulative)
        actual_disp = _display(actual_canon)
        # Chart-only view of the actuals. Identical to actual_disp unless the
        # recorder-quiet carry-forward below has something to add.
        actual_disp_overlay = actual_disp

        # Guard against grossly non-physical logged forecast values. A
        # log-transform inversion that overflows (np.expm1 of a diverged
        # log-space prediction → ~1e30) gets logged verbatim; a single such
        # point dwarfs every real value, flattening the charts and exploding
        # the MAE/ranking. A forecast more than CORRUPT_FACTOR× the largest
        # actual ever seen (or beyond an absolute sanity ceiling when there
        # are no actuals to scale against) isn't a "big miss" — it's corrupt
        # data, so drop it here. Legitimate unit/scale differences are handled
        # separately by the scale-mismatch guard.
        CORRUPT_FACTOR = 1e4
        ABS_SANITY = FORECAST_ABS_SANITY
        _amax = float(actual_raw.abs().max()) if (actual_raw is not None and len(actual_raw)) else 0.0
        _corrupt_cap = min(_amax * CORRUPT_FACTOR, ABS_SANITY) if _amax > 0 else ABS_SANITY

        def _drop_corrupt(df, col):
            """Drop rows whose value is non-finite or beyond the sanity cap.

            Returns (clean_df, info) where info is None when nothing was
            dropped, else {count, max_value, last_ts} describing the dropped
            points so the UI can SURFACE that a blow-up happened (rather than
            silently hiding the only evidence)."""
            if df is None or df.empty or col not in df.columns:
                return df, None
            v = pd.to_numeric(df[col], errors="coerce")
            ok = np.isfinite(v) & (v.abs() <= _corrupt_cap)
            n_bad = int((~ok).sum())
            if not n_bad:
                return df, None
            bad = df[~ok]
            bv = pd.to_numeric(bad[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            max_abs = float(bv.abs().max()) if bv.notna().any() else None
            last_ts = None
            if "target_dt" in bad.columns:
                tt = pd.to_datetime(bad["target_dt"], errors="coerce").dropna()
                if len(tt):
                    last_ts = _iso_utc(tt.max())
            info = {
                "count": n_bad,
                "max_value": (round(max_abs, 4) if (max_abs is not None and np.isfinite(max_abs)) else None),
                "last_ts": last_ts,
            }
            return df[ok], info

        # Accumulates dropped-corruption info so the response can flag it.
        corrupt = {"app": None, "externals": []}

        # --- app forecast: every logged row in the window ---
        _filt = ""
        _params: list = [experiment, cutoff_str, now_str]
        if model_name:
            _filt += " AND model_name = ?"
            _params.append(model_name)
        if model_version:
            _filt += " AND model_version = ?"
            _params.append(model_version)
        cursor.execute(
            "SELECT target_dt, issued_at, lead_minutes, predicted "
            "FROM forecast_log "
            "WHERE experiment = ? AND issued_at >= ? AND target_dt <= ?" + _filt,
            _params,
        )
        fdf = pd.DataFrame(
            cursor.fetchall(),
            columns=["target_dt", "issued_at", "lead_minutes", "predicted"],
        )
        if not fdf.empty:
            fdf["target_dt"] = pd.to_datetime(fdf["target_dt"], errors="coerce")
            fdf["issued_at"] = pd.to_datetime(fdf["issued_at"], errors="coerce")
            fdf["predicted"] = pd.to_numeric(fdf["predicted"], errors="coerce")
            fdf["lead_minutes"] = pd.to_numeric(fdf["lead_minutes"], errors="coerce")
            fdf = fdf.dropna(subset=["target_dt", "issued_at", "predicted"])
            fdf, _capp = _drop_corrupt(fdf, "predicted")
            if _capp:
                corrupt["app"] = _capp
                logger.warning(
                    "external-comparison[%s]: dropped %d non-physical app "
                    "forecast value(s) (max ≈ %s at %s, cap %.3g — likely a "
                    "log-transform inversion overflow); flagged in the UI.",
                    experiment, _capp["count"], _capp["max_value"],
                    _capp["last_ts"], _corrupt_cap,
                )
            fdf["grid"] = fdf["target_dt"].dt.floor(freq)
            fdf = fdf.sort_values("issued_at")
        app_raw = (
            fdf.groupby("grid")["predicted"].last().sort_index()
            if not fdf.empty else pd.Series(dtype="float64")
        )
        # The app forecast is already per-interval (delta for cumulative
        # targets; the instantaneous value otherwise) → never differenced.
        app_factor = _canon_factor(t_dim, t_base)
        app_canon = (app_raw * app_factor) if (unit_aware and not app_raw.empty) else app_raw
        app_disp = _display(app_canon)
        app_points = int(app_disp.notna().sum()) if app_disp is not None else 0
        app_median_lead = _median_lead(fdf)

        ext_table_exists = _table_exists("external_forecast_log")

        # --- resolve each external → canonical + display ---
        ext_items = []
        for spec in externals:
            entity = spec.get("entity")
            if not entity:
                continue
            mode = spec.get("mode", "state") or "state"
            scale = spec.get("scale")
            # Composite source key (entity + attribute + value_key) so the
            # same entity can supply several distinct forecasts.
            source = spec.get("source") or entity
            label = spec.get("label") or (entity.split(".")[-1] if entity else "External")
            e_dim, e_base = _classify_unit(spec.get("unit"))
            edf = pd.DataFrame(
                columns=["target_dt", "issued_at", "lead_minutes", "value", "grid"]
            )
            update_min = None
            if mode == "attribute":
                if ext_table_exists:
                    cursor.execute(
                        "SELECT target_dt, issued_at, lead_minutes, value "
                        "FROM external_forecast_log "
                        "WHERE experiment = ? AND source = ? "
                        "AND issued_at >= ? AND target_dt <= ?",
                        (experiment, source, cutoff_str, now_str),
                    )
                    edf = pd.DataFrame(
                        cursor.fetchall(),
                        columns=["target_dt", "issued_at", "lead_minutes", "value"],
                    )
                if not edf.empty:
                    edf["target_dt"] = pd.to_datetime(edf["target_dt"], errors="coerce")
                    edf["issued_at"] = pd.to_datetime(edf["issued_at"], errors="coerce")
                    edf["value"] = pd.to_numeric(edf["value"], errors="coerce")
                    edf["lead_minutes"] = pd.to_numeric(edf["lead_minutes"], errors="coerce")
                    edf = edf.dropna(subset=["target_dt", "issued_at", "value"])
                    if scale is not None:
                        edf["value"] = edf["value"] * float(scale)
                    edf, _ce = _drop_corrupt(edf, "value")
                    if _ce:
                        _ce["label"] = label
                        corrupt["externals"].append(_ce)
                        logger.warning(
                            "external-comparison[%s]: dropped %d non-physical "
                            "value(s) from external %s (max ≈ %s).",
                            experiment, _ce["count"], source, _ce["max_value"],
                        )
                    edf["grid"] = edf["target_dt"].dt.floor(freq)
                    edf = edf.sort_values("issued_at")
                ext_raw = (
                    edf.groupby("grid")["value"].last().sort_index()
                    if not edf.empty else pd.Series(dtype="float64")
                )
            else:  # state mode
                table = spec.get("table")
                srows = []
                if _table_exists(table):
                    cursor.execute(
                        f"SELECT ds, value FROM {table} WHERE ds >= ? ORDER BY ds",
                        (cutoff_str,),
                    )
                    srows = cursor.fetchall()
                ext_raw = _raw_grid(srows)
                if scale is not None and not ext_raw.empty:
                    ext_raw = ext_raw * float(scale)
                try:
                    ts = pd.to_datetime(
                        pd.DataFrame(srows, columns=["ds", "value"])["ds"],
                        errors="coerce",
                    ).dropna().sort_values()
                    if len(ts) >= 2:
                        diffs = ts.diff().dropna().dt.total_seconds() / 60.0
                        if len(diffs):
                            update_min = round(float(diffs.median()), 1)
                except Exception:
                    update_min = None

            # Resolve cumulative: explicit override wins; else auto-detect.
            override = spec.get("is_cumulative")
            if override is None:
                is_cum = _looks_cumulative(ext_raw) if e_dim != "power" else False
                auto_cumulative = bool(is_cum)
            else:
                is_cum = bool(override)
                auto_cumulative = False

            if unit_aware and e_dim in ("power", "energy"):
                ext_canon = _to_canon(ext_raw, e_dim, e_base, is_cum)
                ext_convertible = True
            else:
                # Unknown unit (or non-unit-aware target): best-effort eval
                # space; the scale guard flags if it doesn't line up.
                ext_canon = _legacy_eval(ext_raw, is_cum)
                ext_convertible = unit_aware is False
            ext_disp = _display(ext_canon)
            ext_items.append({
                "entity": entity, "label": label, "mode": mode,
                "is_cumulative": is_cum, "auto_cumulative": auto_cumulative,
                "dim": e_dim, "base": e_base, "convertible": ext_convertible,
                "canon": ext_canon, "disp": ext_disp, "edf": edf,
                "update_min": update_min,
            })

        # --- carry the actual forward to the comparison horizon -----------
        # HA's recorder dedups unchanged states, so a sensor whose value has
        # plateaued writes no new rows — its cached series ends at the last
        # *change*, not at wall-clock now. The canonical trigger is a
        # cumulative daily total once generation stops for the day (e.g. a PV
        # "energy today" sensor on a cloudy afternoon): the running total holds
        # flat, the recorder goes quiet, and the cached actual stops mid-day.
        # The app forecast is logged live every cycle and runs to ~now, so
        # without this the actual line "just stops" mid-window while the
        # forecasts march on. (A complete past day doesn't show this: its
        # plateau is bracketed by the next midnight-reset row, so the per-day
        # cumsum fills the internal gap — only the current partial day, whose
        # plateau reaches the right edge, truncates.)
        #
        # Mirror the forecast pipeline's recorder-quiet carry-forward
        # (main.py): hold the last cached value flat up to the latest bin any
        # forecast/external covers (capped at now). The per-day cumsum and the
        # diff reset guard handle a midnight reset inside the carried span, and
        # a resumed real reading supersedes the hold (its first diff trips the
        # reset guard). Bounding to the forecast extent — not unconditionally
        # to now — keeps a long-stale series (the rolling window can be 30 days)
        # from growing a fabricated flat tail, and makes this a no-op whenever
        # the actuals are already current.
        last_valid = actual_raw.last_valid_index() if actual_raw is not None else None
        if last_valid is not None:
            horizon = None
            for s in [app_disp] + [it["disp"] for it in ext_items]:
                if s is not None and not s.empty:
                    m = s.index.max()
                    horizon = m if horizon is None else max(horizon, m)
            if horizon is not None:
                now_floor = pd.Timestamp(now).floor(freq)
                if horizon > now_floor:
                    horizon = now_floor
                if horizon > last_valid:
                    fill_idx = pd.date_range(
                        last_valid + pd.Timedelta(freq), horizon, freq=freq
                    )
                    if len(fill_idx):
                        held = pd.Series(float(actual_raw.loc[last_valid]), index=fill_idx)
                        carried_raw = pd.concat([actual_raw, held])
                        carried_raw = carried_raw[
                            ~carried_raw.index.duplicated(keep="first")
                        ].sort_index()
                        # DISPLAY ONLY. The held values are inferred, not
                        # observed, so they are kept in a separate series that
                        # feeds the chart overlay and nothing else.
                        #
                        # Rebinding actual_canon / actual_disp here — as this
                        # originally did — silently routes them into every
                        # scoring path downstream: the metrics block, the
                        # head-to-head winner election, the daily-error series,
                        # the lead curve and the skill table. The held points
                        # are non-NaN, so they also defeat the dropna() filters
                        # that previously excluded exactly those bins. On a
                        # sensor that has genuinely died the add-on would then
                        # be scoring its forecast against a flat line it
                        # invented, which flatters or penalises it at random.
                        if unit_aware:
                            carried_canon = _to_canon(
                                carried_raw, t_dim, t_base, target_cumulative
                            )
                        else:
                            carried_canon = _legacy_eval(carried_raw, target_cumulative)
                        actual_disp_overlay = _display(carried_canon)

        # --- drop user-excluded days from every series & frame ---
        # A day flagged as corrupt is removed from the actuals, this add-on's
        # forecast AND every external before any overlay/metric/ranking is
        # computed, so one bad day can't pollute the head-to-head. Matching is
        # on the HA-local calendar date (same day key the rest of the tab uses
        # via day_offset_hours).
        excluded_set = {str(d).strip() for d in (excluded_dates or []) if str(d).strip()}
        if excluded_set:
            _off = pd.Timedelta(seconds=int(float(day_offset_hours or 0.0) * 3600))

            def _drop_excluded_series(s):
                if s is None or s.empty:
                    return s
                local_days = pd.Index(s.index + _off).strftime("%Y-%m-%d")
                return s[~local_days.isin(excluded_set)]

            def _drop_excluded_frame(df):
                if df is None or df.empty or "target_dt" not in df.columns:
                    return df
                local_days = (
                    pd.to_datetime(df["target_dt"], errors="coerce") + _off
                ).dt.strftime("%Y-%m-%d")
                return df[~local_days.isin(excluded_set)]

            actual_canon = _drop_excluded_series(actual_canon)
            actual_disp = _drop_excluded_series(actual_disp)
            actual_disp_overlay = _drop_excluded_series(actual_disp_overlay)
            app_canon = _drop_excluded_series(app_canon)
            app_disp = _drop_excluded_series(app_disp)
            fdf = _drop_excluded_frame(fdf)
            # app_points / app_median_lead were computed pre-filter — refresh.
            app_points = int(app_disp.notna().sum()) if app_disp is not None else 0
            app_median_lead = _median_lead(fdf)
            for it in ext_items:
                it["canon"] = _drop_excluded_series(it["canon"])
                it["disp"] = _drop_excluded_series(it["disp"])
                it["edf"] = _drop_excluded_frame(it["edf"])

        result["excluded_dates"] = sorted(excluded_set)
        result["reset_at"] = (str(reset_at).strip() or None) if reset_at else None

        # --- shared overlay grid (union within the window) ---
        idx = pd.DatetimeIndex([])
        for s in [actual_disp_overlay, app_disp] + [it["disp"] for it in ext_items]:
            if s is not None and not s.empty:
                idx = idx.union(s.index)
        if len(idx):
            lo, hi = pd.Timestamp(cutoff), pd.Timestamp(now)
            idx = idx[(idx >= lo) & (idx <= hi)].sort_values()

        def _col(s) -> list:
            if s is None or s.empty or len(idx) == 0:
                return [None] * len(idx)
            r = s.reindex(idx)
            return [None if pd.isna(v) else round(float(v), 4) for v in r.values]

        result["grid_points"] = int(len(idx))
        result["app_points"] = app_points
        result["overlay"] = {
            "ds": [_iso_utc(t) for t in idx],
            "actual": _col(actual_disp_overlay),
            "app": _col(app_disp),
            "externals": [
                {"entity": it["entity"], "label": it["label"],
                 "mode": it["mode"], "values": _col(it["disp"])}
                for it in ext_items
            ],
        }
        if len(idx):
            result["date_range"] = {"start": _iso_utc(idx.min()), "end": _iso_utc(idx.max())}

        # --- typical magnitude (for "% of typical") in the display space ---
        # Mean |actual| over the window; the same normaliser for every source
        # so "% of typical" is comparable across rows.
        _ad_win = actual_disp.reindex(idx) if (actual_disp is not None and len(idx)) else actual_disp
        typical = (
            round(float(_ad_win.abs().mean()), 4)
            if (_ad_win is not None and _ad_win.notna().any()) else 0.0
        )

        def _pct(mae):
            return round(mae / typical * 100.0, 1) if typical and typical > 1e-9 else None

        def _err(df, col):
            e = (df[col] - df["actual"]).astype(float)
            mae = float(e.abs().mean())
            return {
                "mae": round(mae, 4),
                "rmse": round(float(np.sqrt((e ** 2).mean())), 4),
                "bias": round(float(e.mean()), 4),
                "pct": _pct(mae),
            }

        def _daily_err(pred_canon, common_index):
            """Daily-total MAE/bias on the CANONICAL per-interval energy,
            restricted to the supplied (already 2- or 3-way intersected)
            index so every series sums over identical bins each day — no
            partial-day coverage bias."""
            if pred_canon is None or actual_canon is None or len(common_index) == 0:
                return None
            m = pd.DataFrame({
                "a": actual_canon.reindex(common_index),
                "p": pred_canon.reindex(common_index),
            }).dropna()
            if m.empty:
                return None
            day = m.index.normalize()
            a_d = m["a"].groupby(day).sum()
            p_d = m["p"].groupby(day).sum()
            de = (p_d - a_d).astype(float)
            return {
                "mae": round(float(de.abs().mean()), 4),
                "bias": round(float(de.mean()), 4),
                "days": int(de.shape[0]),
            }

        def _metrics_block(pred_disp, pred_canon):
            """Standalone accuracy of one source vs the actual on their OWN
            2-way overlap (for the row display): mae/rmse/bias/% of typical,
            daily MAE/bias, distinct days logged, and the warming-up flag."""
            al = pd.DataFrame({"actual": actual_disp, "pred": pred_disp})
            if len(idx):
                al = al.reindex(idx)
            common = al.dropna(subset=["actual", "pred"])
            if common.empty:
                return None
            e = (common["pred"] - common["actual"]).astype(float)
            mae = float(e.abs().mean())
            day = common.index.normalize()
            days_logged = int(len(set(day)))
            return {
                "mae": round(mae, 4),
                "rmse": round(float(np.sqrt((e ** 2).mean())), 4),
                "bias": round(float(e.mean()), 4),
                "pct": _pct(mae),
                "n": int(len(common)),
                "daily": _daily_err(pred_canon, common.index),
                "days_logged": days_logged,
                "warming": days_logged < EXTERNAL_COMPARISON_WARMUP_DAYS,
            }

        # --- per-external head-to-head + timing (on the display series) ---
        for it in ext_items:
            aligned = pd.DataFrame(
                {"actual": actual_disp, "app": app_disp, "external": it["disp"]}
            )
            if len(idx):
                aligned = aligned.reindex(idx)
            common = aligned.dropna(subset=["actual", "app", "external"])
            n_common = int(len(common))
            h2h = None
            scale_ratio = None
            scale_mismatch = False
            if n_common >= 1:
                app_m = _err(common, "app")
                ext_m = _err(common, "external")
                if app_m["mae"] < ext_m["mae"]:
                    winner = "app"
                elif ext_m["mae"] < app_m["mae"]:
                    winner = "external"
                else:
                    winner = "tie"
                impr = None
                if ext_m["mae"] > 0:
                    impr = round((ext_m["mae"] - app_m["mae"]) / ext_m["mae"] * 100.0, 1)
                h2h = {
                    "n": n_common, "app": app_m, "external": ext_m,
                    "winner": winner, "app_mae_improvement_pct": impr,
                    "daily": {
                        "app": _daily_err(app_canon, common.index),
                        "external": _daily_err(it["canon"], common.index),
                    },
                }
                # Scale-mismatch guard: after unit-aware conversion a genuine
                # like-for-like sits near ratio 1; a large gap means the units
                # couldn't be reconciled (unknown unit, or a quantity we can't
                # bridge) — the head-to-head isn't meaningful.
                mean_actual = float(common["actual"].abs().mean())
                mean_ext = float(common["external"].abs().mean())
                if mean_actual > 1e-9 and mean_ext > 1e-9:
                    scale_ratio = round(mean_ext / mean_actual, 2)
                    scale_mismatch = scale_ratio > 4.0 or scale_ratio < 0.25
            ext_points = int(it["disp"].notna().sum()) if it["disp"] is not None else 0
            timing = {
                "app_median_lead_minutes": app_median_lead,
                "external_median_lead_minutes": (
                    _median_lead(it["edf"]) if it["mode"] == "attribute" else None
                ),
                "external_contemporaneous": it["mode"] != "attribute",
                "external_update_minutes": it["update_min"],
                "external_points": ext_points,
                "app_points": app_points,
                "grid_points": int(len(idx)),
                "external_stale": bool(app_points > 0 and ext_points < 0.5 * app_points),
            }
            # Standalone row metrics (own 2-way overlap with actual) + the
            # warming-up gate. days_logged counts distinct calendar days with
            # overlapping data; below EXTERNAL_COMPARISON_WARMUP_DAYS the row
            # is flagged provisional/inconclusive.
            it["scale_mismatch"] = scale_mismatch  # consumed by the skill block
            ext_block = _metrics_block(it["disp"], it["canon"])
            result["comparisons"].append({
                "entity": it["entity"], "label": it["label"], "mode": it["mode"],
                "head_to_head": h2h, "timing": timing, "n": n_common,
                "scale_ratio": scale_ratio, "scale_mismatch": scale_mismatch,
                "auto_cumulative": it["auto_cumulative"],
                "metrics": ext_block,
                "days_logged": (ext_block["days_logged"] if ext_block else 0),
                "warming": (ext_block["warming"] if ext_block else True),
            })

        # --- app reference row (standalone) + the top-level warming-up gate ---
        app_self = _metrics_block(app_disp, app_canon)
        result["app_self"] = app_self
        result["app_days_logged"] = app_self["days_logged"] if app_self else 0
        result["typical"] = typical
        result["warmup_days"] = EXTERNAL_COMPARISON_WARMUP_DAYS
        # Surface any non-physical forecast values we excluded so the blowup
        # is VISIBLE (a stale long-horizon blowup is otherwise masked by the
        # latest-per-target / h=1 views and shows up nowhere else).
        if corrupt["app"] or corrupt["externals"]:
            result["corrupt"] = corrupt

        # Top-level warming gate: inconclusive while the add-on OR any
        # configured external still lacks the threshold days of overlap.
        _warm = [bool(app_self and app_self["warming"])]
        for _c in result["comparisons"]:
            if _c.get("metrics"):
                _warm.append(bool(_c["warming"]))
        result["warming_up"] = any(_warm) if (app_self or result["comparisons"]) else False

        # --- combined lead-time (app + attribute externals), always
        #     per-interval canonical energy so horizons are comparable ---
        def _leadcurve(traj, value_col, is_cum, factor):
            # Per-bucket error SUMS (abs / squared / signed) + count, so the
            # skill block can aggregate exact MAE / RMSE / bias over any band
            # via Σ/Σn (the per-bucket means alone can't be re-pooled).
            empty = pd.DataFrame(columns=["abs_sum", "sq_sum", "signed_sum", "count"])
            if traj is None or traj.empty or value_col not in traj.columns:
                return empty
            t = traj[["grid", "issued_at", "lead_minutes", value_col]].copy()
            if is_cum:
                t = t.sort_values(["issued_at", "grid"])
                t["val"] = t.groupby("issued_at")[value_col].diff()
                t.loc[t["val"] < 0, "val"] = np.nan
                t = t.dropna(subset=["val"])
                vcol = "val"
            else:
                vcol = value_col
            t = t.join(actual_canon.rename("actual"), on="grid")
            t = t.dropna(subset=["actual", vcol])
            if t.empty:
                return empty
            t["err"] = t[vcol] * factor - t["actual"]
            t["abs_e"] = t["err"].abs()
            t["sq_e"] = t["err"] * t["err"]
            t["bucket"] = (t["lead_minutes"] // bucket_min) * bucket_min
            return t.groupby("bucket").agg(
                abs_sum=("abs_e", "sum"),
                sq_sum=("sq_e", "sum"),
                signed_sum=("err", "sum"),
                count=("err", "size"),
            )

        attr_items = [it for it in ext_items if it["mode"] == "attribute"]
        if attr_items:
            app_curve = _leadcurve(fdf, "predicted", False, _canon_factor(t_dim, t_base))
            ext_curves = [
                (it, _leadcurve(it["edf"], "value", it["is_cumulative"],
                                _canon_factor(it["dim"], it["base"])))
                for it in attr_items
            ]
            buckets = set(app_curve.index)
            for _it, c in ext_curves:
                buckets |= set(c.index)
            buckets = sorted(buckets)
            if buckets:
                def _ser(curve, key):
                    out = []
                    for b in buckets:
                        if b in curve.index:
                            cnt = int(curve.loc[b, "count"])
                            if key == "mae":
                                out.append(round(float(curve.loc[b, "abs_sum"]) / cnt, 4) if cnt else None)
                            else:
                                out.append(cnt)
                        else:
                            out.append(None if key == "mae" else 0)
                    return out
                result["lead_time"] = {
                    "lead_minutes": [int(b) for b in buckets],
                    "app_mae": _ser(app_curve, "mae"),
                    "app_n": _ser(app_curve, "n"),
                    "externals": [
                        {"entity": it["entity"], "label": it["label"],
                         "mae": _ser(c, "mae"), "n": _ser(c, "n")}
                        for it, c in ext_curves
                    ],
                }

            # --- "Same lead time" (skill) scoring ----------------------------
            # Score the app + each trajectory external over the lead band they
            # all cover, so the head-to-head isolates model skill from update
            # frequency. The MAE over the band is the sample-weighted mean of
            # the per-bucket MAEs: Σ(mae_b·n_b) / Σ(n_b). State-mode sources
            # (lead 0 only) and scale-mismatched ones can't take part and are
            # listed as excluded.
            excluded = []
            for it, c in ext_curves:
                if it.get("scale_mismatch"):
                    excluded.append({"label": it["label"], "reason": "scale mismatch"})
                elif c.empty:
                    excluded.append({"label": it["label"], "reason": "no overlapping forecasts yet"})
            for it in ext_items:
                if it["mode"] != "attribute":
                    excluded.append({"label": it["label"], "reason": "state-mode (nowcast only)"})
            eligible = [(it, c) for it, c in ext_curves
                        if not it.get("scale_mismatch") and not c.empty]
            band = set(app_curve.index)
            for it, c in eligible:
                band &= set(c.index)
            band = sorted(band)
            if eligible and band:
                # Typical magnitude in the canonical (per-interval) space the
                # lead curves live in, for "% of typical" at matched lead.
                _ac = (actual_canon.reindex(idx)
                       if (actual_canon is not None and len(idx)) else actual_canon)
                typical_canon = (float(_ac.abs().mean())
                                 if (_ac is not None and _ac.notna().any()) else None)

                def _band_stats(curve):
                    a = s = g = 0.0
                    den = 0
                    for b in band:
                        if b in curve.index:
                            a += float(curve.loc[b, "abs_sum"])
                            s += float(curve.loc[b, "sq_sum"])
                            g += float(curve.loc[b, "signed_sum"])
                            den += int(curve.loc[b, "count"])
                    if den <= 0:
                        return {"mae": None, "rmse": None, "bias": None, "pct": None, "n": 0}
                    mae = a / den
                    return {
                        "mae": round(mae, 4),
                        "rmse": round((s / den) ** 0.5, 4),
                        "bias": round(g / den, 4),
                        "pct": (round(mae / typical_canon * 100.0, 1)
                                if typical_canon and typical_canon > 1e-9 else None),
                        "n": den,
                    }
                skill_rows = [dict({"entity": None, "label": "ML Forecast Lab",
                                    "app": True}, **_band_stats(app_curve))]
                for it, c in eligible:
                    skill_rows.append(dict({"entity": it["entity"], "label": it["label"],
                                            "app": False}, **_band_stats(c)))
                result["skill"] = {
                    "available": True,
                    "lead_band_minutes": [int(band[0]), int(band[-1])],
                    "unit": result.get("lead_unit", ""),
                    "rows": skill_rows,
                    "excluded": excluded,
                }
            elif excluded:
                result["skill"] = {
                    "available": False,
                    "reason": ("no common lead band" if eligible else
                               "needs a trajectory (attribute-mode) source"),
                    "excluded": excluded,
                }

        return result

    # ------------------------------------------------------------------
    # Benchmark results persistence
    # ------------------------------------------------------------------

    @_locked
    def ensure_benchmark_table(self) -> None:
        """Create the benchmark_results + benchmark_history tables."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                experiment TEXT PRIMARY KEY,
                data       TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        # Append-only retention table — every save_benchmark_result writes
        # one row here, capped to RETAIN_PER_EXP rows per experiment so
        # the Results tab can offer a "Previous runs" dropdown without
        # the disk usage growing unbounded on an SD card.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment TEXT NOT NULL,
                ran_at     TEXT NOT NULL,
                data       TEXT NOT NULL
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_bhist_exp_ran "
            "ON benchmark_history(experiment, ran_at)"
        )
        self.conn.commit()
        logger.debug("Ensured benchmark_results + benchmark_history tables")

    BENCHMARK_HISTORY_RETAIN_PER_EXP = 5

    @_locked
    def save_benchmark_result(self, experiment: str, json_data: str) -> None:
        """
        Upsert a benchmark result as a JSON blob, and append a copy to
        ``benchmark_history`` (capped at ``BENCHMARK_HISTORY_RETAIN_PER_EXP``
        rows per experiment) so the Results tab can offer a "Previous runs"
        dropdown.

        Parameters
        ----------
        experiment : str
            Experiment name (primary key).
        json_data : str
            JSON-serialised BenchmarkResult.
        """
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO benchmark_results "
                "(experiment, data, updated_at) VALUES (?, ?, ?)",
                (experiment, json_data, now_str),
            )
            cursor.execute(
                "INSERT INTO benchmark_history (experiment, ran_at, data) "
                "VALUES (?, ?, ?)",
                (experiment, now_str, json_data),
            )
            # Trim oldest rows so the retention cap holds. Two-step
            # because SQLite doesn't support DELETE with LIMIT/OFFSET in
            # all builds; the subquery is bounded and indexed.
            cursor.execute(
                "DELETE FROM benchmark_history WHERE id IN ("
                "  SELECT id FROM benchmark_history WHERE experiment = ?"
                "  ORDER BY ran_at DESC LIMIT -1 OFFSET ?"
                ")",
                (experiment, self.BENCHMARK_HISTORY_RETAIN_PER_EXP),
            )
            self.conn.commit()
            logger.debug(f"Saved benchmark result for {experiment}")
        except sqlite3.Error as e:
            logger.error(f"Error saving benchmark result for {experiment}: {e}", exc_info=True)
            self.conn.rollback()

    @_locked
    def load_benchmark_history(
        self, experiment: str, limit: int = 5,
    ) -> list:
        """Return up to *limit* previous benchmark runs for *experiment*.

        Each item is a dict with ``ran_at`` (ISO-ish UTC string) and
        ``data`` (JSON-serialised BenchmarkResult). Ordered newest first.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT ran_at, data FROM benchmark_history "
                "WHERE experiment = ? ORDER BY ran_at DESC LIMIT ?",
                (experiment, int(limit)),
            )
            return [
                {"ran_at": row["ran_at"], "data": row["data"]}
                for row in cursor.fetchall()
            ]
        except sqlite3.Error as e:
            logger.error(
                "Error loading benchmark history for %s: %s",
                experiment, e, exc_info=True,
            )
            return []

    @_locked
    def load_all_benchmark_results(self) -> dict:
        """
        Load all stored benchmark results.

        Returns
        -------
        dict
            Mapping of experiment name → JSON string.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT experiment, data FROM benchmark_results")
            return {row[0]: row[1] for row in cursor.fetchall()}
        except sqlite3.Error as e:
            logger.error(f"Error loading benchmark results: {e}", exc_info=True)
            return {}

    @_locked
    def delete_benchmark_result(self, experiment: str) -> None:
        """Delete stored benchmark result + history for an experiment."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "DELETE FROM benchmark_results WHERE experiment = ?",
                (experiment,),
            )
            cursor.execute(
                "DELETE FROM benchmark_history WHERE experiment = ?",
                (experiment,),
            )
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error deleting benchmark result for {experiment}: {e}", exc_info=True)
            self.conn.rollback()

    @_locked
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("HistoryDB closed")
