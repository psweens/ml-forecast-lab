"""
SQLite history cache for ML Forecast Lab.

Stores historical entity states with efficient bulk insert and retrieval,
and automatic cleanup of old records.
"""

import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


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

        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        logger.info(f"HistoryDB initialised at {self.path}")

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
            logger.error(f"Error inserting into {table_name}: {e}")
            self.conn.rollback()
            return 0

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
            logger.error(f"Error cleaning up {table_name}: {e}")
            self.conn.rollback()
            return 0

    # ------------------------------------------------------------------
    # Forecast evolution log
    # ------------------------------------------------------------------

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
                lower        REAL
            )
        """)
        # Migrate pre-existing tables that don't have the upper/lower
        # columns. PRAGMA table_info returns rows (cid, name, type, ...).
        cursor.execute("PRAGMA table_info(forecast_log)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        if "upper" not in existing_cols:
            cursor.execute("ALTER TABLE forecast_log ADD COLUMN upper REAL")
        if "lower" not in existing_cols:
            cursor.execute("ALTER TABLE forecast_log ADD COLUMN lower REAL")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_flog_exp_target "
            "ON forecast_log(experiment, target_dt)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_flog_exp_issued "
            "ON forecast_log(experiment, issued_at)"
        )
        self.conn.commit()
        logger.debug("Ensured forecast_log table")

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
        for i, (ts, val) in enumerate(zip(targets, predictions)):
            target_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            lead_min = int((ts - issued_at).total_seconds() / 60)
            upper_val = float(upper_bounds[i]) if upper_bounds is not None else None
            lower_val = float(lower_bounds[i]) if lower_bounds is not None else None
            rows.append((
                experiment, model_name, issued_str, target_str,
                lead_min, float(val), forecast_type, upper_val, lower_val,
            ))
        cursor = self.conn.cursor()
        try:
            cursor.executemany(
                "INSERT INTO forecast_log "
                "(experiment, model_name, issued_at, target_dt, lead_minutes, "
                "predicted, forecast_type, upper, lower) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self.conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f"Error logging forecast for {experiment}: {e}")
            self.conn.rollback()
            return 0

    def get_forecast_accuracy(
        self,
        experiment: str,
        actuals_table: str,
        max_age_days: int = 30,
        interval_minutes: int = 30,
        evaluation_mode: str = "raw",
        model_name: Optional[str] = None,
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
            return {"error": "No actuals data available yet"}

        interval_sec = interval_minutes * 60
        bucket_min = max(1, int(interval_minutes))
        increment = evaluation_mode == "increment"
        # Optional model filter, spliced directly into forecast_vals so
        # both the lead-time curve and revision query apply it.
        model_filter_sql = " AND model_name = ?" if model_name else ""
        model_filter_param = (model_name,) if model_name else ()

        # Build value-extraction CTEs according to mode. In "raw" mode we
        # just pass the stored value through; in "increment" mode we take
        # LAG-based diffs so error reflects per-interval demand rather
        # than cumulative shape. The same actuals_grid CTE feeds both.
        #
        # Midnight resets on daily-cumulative sensors produce a large
        # negative increment (e.g. 0 - 85 = -85). Increment mode filters
        # `value >= 0` at the join to drop those rows; this is only safe
        # because the mode is gated on source_is_cumulative upstream.
        #
        # We also null out the delta when the previous grid row is not
        # exactly one interval earlier — otherwise an HA outage causes
        # e.g. a 2-hour span to be treated as a single-interval demand,
        # inflating MAE with data-availability artefacts rather than
        # model error. The adjacency check compares unix-epoch seconds
        # of the stringified grid_dt.
        if increment:
            actuals_vals_cte = (
                "actuals_vals AS ("
                "  SELECT grid_dt,"
                "    CASE"
                "      WHEN CAST(strftime('%s', grid_dt) AS INTEGER)"
                "           - CAST(strftime('%s', LAG(grid_dt) OVER (ORDER BY grid_dt)) AS INTEGER)"
                "           = ?"
                "      THEN value - LAG(value) OVER (ORDER BY grid_dt)"
                "      ELSE NULL"
                "    END AS value"
                "  FROM actuals_grid"
                ")"
            )
            forecast_vals_cte = (
                "forecast_vals AS ("
                "  SELECT experiment, model_name, issued_at, target_dt, lead_minutes,"
                "    CASE"
                "      WHEN CAST(strftime('%s', target_dt) AS INTEGER)"
                "           - CAST(strftime('%s', LAG(target_dt) OVER ("
                "               PARTITION BY issued_at ORDER BY target_dt)) AS INTEGER)"
                "           = ?"
                "      THEN predicted - LAG(predicted) OVER ("
                "          PARTITION BY issued_at ORDER BY target_dt)"
                "      ELSE NULL"
                "    END AS value"
                "  FROM forecast_log"
                "  WHERE experiment = ? AND target_dt <= ? AND issued_at >= ?"
                f"  {model_filter_sql}"
                ")"
            )
            mode_filter = (
                "AND fv.value IS NOT NULL AND av.value IS NOT NULL"
                "  AND fv.value >= 0 AND av.value >= 0"
            )
        else:
            actuals_vals_cte = (
                "actuals_vals AS (SELECT grid_dt, value FROM actuals_grid)"
            )
            forecast_vals_cte = (
                "forecast_vals AS ("
                "  SELECT experiment, model_name, issued_at, target_dt, lead_minutes,"
                "         predicted AS value"
                "  FROM forecast_log"
                "  WHERE experiment = ? AND target_dt <= ? AND issued_at >= ?"
                f"  {model_filter_sql}"
                ")"
            )
            mode_filter = ""

        # Parameter prefixes for the value-extraction CTEs. Increment
        # mode takes an adjacency interval for each LAG (actuals and
        # forecasts); raw mode takes neither.
        if increment:
            actuals_vals_params = (interval_sec,)
            forecast_vals_params = (
                interval_sec, experiment, now_str, cutoff_str, *model_filter_param,
            )
        else:
            actuals_vals_params = ()
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
                WITH actuals_grid AS (
                    SELECT
                        strftime('%Y-%m-%d %H:%M:%S',
                            (CAST(strftime('%s', SUBSTR(ds, 1, 19)) AS INTEGER) / ?) * ?,
                            'unixepoch') AS grid_dt,
                        AVG(value) AS value
                    FROM {actuals_table}
                    GROUP BY grid_dt
                ),
                {actuals_vals_cte},
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
                interval_sec, interval_sec,
                *actuals_vals_params,
                *forecast_vals_params,
                bucket_min, bucket_min,
            ))
            lead_rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Forecast accuracy query failed: {e}")
            return {"error": str(e)}

        lead_time_curve = {
            "lead_minutes": [r[0] for r in lead_rows],
            "mae": [round(r[1], 4) for r in lead_rows],
            "rmse": [round(r[2], 4) for r in lead_rows],
            "me": [round(r[3], 4) for r in lead_rows],
            "sample_count": [r[4] for r in lead_rows],
        }

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
                WITH actuals_grid AS (
                    SELECT
                        strftime('%Y-%m-%d %H:%M:%S',
                            (CAST(strftime('%s', SUBSTR(ds, 1, 19)) AS INTEGER) / ?) * ?,
                            'unixepoch') AS grid_dt,
                        AVG(value) AS value
                    FROM {actuals_table}
                    GROUP BY grid_dt
                ),
                {actuals_vals_cte},
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
                interval_sec, interval_sec,
                *actuals_vals_params,
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
                cursor.execute(f"""
                    WITH actuals_grid AS (
                        SELECT
                            strftime('%Y-%m-%d %H:%M:%S',
                                (CAST(strftime('%s', SUBSTR(ds, 1, 19)) AS INTEGER) / ?) * ?,
                                'unixepoch') AS grid_dt,
                            AVG(value) AS value
                        FROM {actuals_table}
                        WHERE SUBSTR(ds, 1, 19) >= ?
                        GROUP BY grid_dt
                    ),
                    deltas AS (
                        SELECT
                            CASE
                              WHEN CAST(strftime('%s', grid_dt) AS INTEGER)
                                 - CAST(strftime('%s', LAG(grid_dt) OVER (ORDER BY grid_dt)) AS INTEGER)
                                 = ?
                              THEN value - LAG(value) OVER (ORDER BY grid_dt)
                              ELSE NULL
                            END AS d
                        FROM actuals_grid
                    )
                    SELECT AVG(ABS(d)) FROM deltas WHERE d IS NOT NULL AND d >= 0
                """, (interval_sec, interval_sec, cutoff_str, interval_sec))
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

        return {
            "experiment": experiment,
            "evaluation_mode": "increment" if increment else "raw",
            "lead_time_curve": lead_time_curve,
            "revision_improvement": revision,
            "typical_interval_demand": typical,
            "total_logged": stats_row[0] if stats_row else 0,
            "date_range": {
                "from": stats_row[1] if stats_row else None,
                "to": stats_row[2] if stats_row else None,
            },
        }

    def get_forecast_trajectory(
        self,
        experiment: str,
        actuals_table: str,
        target_dt: Optional[str] = None,
        interval_minutes: int = 30,
        max_age_days: int = 14,
        source_is_cumulative: bool = False,
        model_name: Optional[str] = None,
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

        model_filter_sql = " AND fl.model_name = ?" if model_name else ""
        model_filter_param = (model_name,) if model_name else ()

        # Build the actuals_vals CTE according to source type. For
        # cumulative sources we need to diff so the actual lives in the
        # same space as the per-interval predictions stored in
        # forecast_log. The adjacency guard (delta only when prior grid
        # row is exactly one interval earlier) mirrors the increment-
        # mode logic in get_forecast_accuracy.
        if source_is_cumulative:
            actuals_vals_cte = (
                "actuals_vals AS ("
                "  SELECT grid_dt,"
                "    CASE"
                "      WHEN CAST(strftime('%s', grid_dt) AS INTEGER)"
                "           - CAST(strftime('%s', LAG(grid_dt) OVER (ORDER BY grid_dt)) AS INTEGER)"
                "           = ?"
                "      THEN value - LAG(value) OVER (ORDER BY grid_dt)"
                "      ELSE NULL"
                "    END AS value"
                "  FROM actuals_grid"
                ")"
            )
            actuals_vals_params = (interval_sec,)
            actual_space = "delta"
        else:
            actuals_vals_cte = (
                "actuals_vals AS (SELECT grid_dt, value FROM actuals_grid)"
            )
            actuals_vals_params = ()
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
                WITH actuals_grid AS (
                    SELECT
                        strftime('%Y-%m-%d %H:%M:%S',
                            (CAST(strftime('%s', SUBSTR(ds, 1, 19)) AS INTEGER) / ?) * ?,
                            'unixepoch') AS grid_dt,
                        AVG(value) AS value
                    FROM {actuals_table}
                    GROUP BY grid_dt
                ),
                {actuals_vals_cte}
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
                interval_sec, interval_sec,
                *actuals_vals_params,
                experiment, now_str, cutoff_str,
                *model_filter_param,
            ))
            candidate_rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Trajectory candidates query failed: {e}")
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

        # Fetch all forecasts for the chosen target.
        try:
            cursor.execute(
                "SELECT issued_at, predicted, lead_minutes, model_name "
                "FROM forecast_log "
                f"WHERE experiment = ? AND target_dt = ?{(' AND model_name = ?' if model_name else '')} "
                "ORDER BY issued_at ASC",
                (experiment, target_dt, *model_filter_param),
            )
            forecast_rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Trajectory fetch failed: {e}")
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
                WITH actuals_grid AS (
                    SELECT
                        strftime('%Y-%m-%d %H:%M:%S',
                            (CAST(strftime('%s', SUBSTR(ds, 1, 19)) AS INTEGER) / ?) * ?,
                            'unixepoch') AS grid_dt,
                        AVG(value) AS value
                    FROM {actuals_table}
                    GROUP BY grid_dt
                ),
                {actuals_vals_cte}
                SELECT av.value FROM actuals_vals av
                WHERE av.grid_dt = ?
            """, (
                interval_sec, interval_sec,
                *actuals_vals_params,
                target_dt,
            ))
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

    def get_conformal_quantiles(
        self,
        experiment: str,
        actuals_table: str,
        level: float = 0.8,
        model_name: Optional[str] = None,
        interval_minutes: int = 30,
        max_age_days: int = 14,
        min_samples: int = 10,
    ) -> dict:
        """
        Compute per-lead-time conformal nonconformity quantiles from
        historical forecasts vs actuals in forecast_log.

        For a symmetric (1−α) band with α = 1−level, we take the
        (1−α/2)-th quantile of |residual| at each lead bucket. For an
        80%-band (level=0.8), that is the 90th percentile.

        Using deployed forecast/actual pairs as the calibration sample
        (adaptive / online conformal) avoids the cost of refitting the
        model on a held-out window every production cycle. The tradeoff
        is that residuals aren't strictly exchangeable (temporal drift,
        model retrains), so finite-sample coverage is approximate rather
        than guaranteed. For diagnostic intervals on a home-automation
        sensor this is an acceptable simplification.

        Parameters
        ----------
        experiment : str
        actuals_table : str
        level : float
            Desired coverage, in (0, 1). 0.8 produces the 90th-percentile
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

        params = [interval_sec, interval_sec, bucket_min, bucket_min,
                  experiment, now_str, cutoff_str]
        model_filter = ""
        if model_name:
            model_filter = "AND fl.model_name = ?"
            params.append(model_name)

        try:
            cursor.execute(f"""
                WITH actuals_grid AS (
                    SELECT
                        strftime('%Y-%m-%d %H:%M:%S',
                            (CAST(strftime('%s', SUBSTR(ds, 1, 19)) AS INTEGER) / ?) * ?,
                            'unixepoch') AS grid_dt,
                        AVG(value) AS value
                    FROM {actuals_table}
                    GROUP BY grid_dt
                )
                SELECT
                    CAST((fl.lead_minutes / ?) * ? AS INTEGER) AS lead_bucket,
                    ABS(fl.predicted - ag.value) AS abs_residual
                FROM forecast_log fl
                INNER JOIN actuals_grid ag ON ag.grid_dt = fl.target_dt
                WHERE fl.experiment = ?
                  AND fl.target_dt <= ?
                  AND fl.issued_at >= ?
                  {model_filter}
            """, tuple(params))
            rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Conformal quantile query failed: {e}")
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

        df = pd.DataFrame(rows, columns=["lead_bucket", "abs_residual"])
        df = df.dropna()

        alpha = max(0.0, min(1.0, 1.0 - level))
        q = 1.0 - alpha / 2.0

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

    def get_forecast_coverage(
        self,
        experiment: str,
        actuals_table: str,
        interval_minutes: int = 30,
        max_age_days: int = 30,
        model_name: Optional[str] = None,
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
            ``overall``: {coverage: float, n: int} or empty dict
            ``level``: float (nominal level read from ``level`` kwarg
                of the conformal run; here inferred by the caller and
                echoed in the UI).
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
            return {"by_lead": {"lead_minutes": [], "coverage": [], "n": []}, "overall": {}}

        # Optional model filter — coverage of a rotated-out model tells
        # the user little about the current champion's calibration.
        model_filter_sql = " AND fl.model_name = ?" if model_name else ""
        model_filter_param = (model_name,) if model_name else ()

        try:
            cursor.execute(f"""
                WITH actuals_grid AS (
                    SELECT
                        strftime('%Y-%m-%d %H:%M:%S',
                            (CAST(strftime('%s', SUBSTR(ds, 1, 19)) AS INTEGER) / ?) * ?,
                            'unixepoch') AS grid_dt,
                        AVG(value) AS value
                    FROM {actuals_table}
                    GROUP BY grid_dt
                )
                SELECT
                    CAST((fl.lead_minutes / ?) * ? AS INTEGER) AS lead_bucket,
                    AVG(CASE WHEN ag.value BETWEEN fl.lower AND fl.upper
                             THEN 1.0 ELSE 0.0 END) AS coverage,
                    COUNT(*) AS n
                FROM forecast_log fl
                INNER JOIN actuals_grid ag ON ag.grid_dt = fl.target_dt
                WHERE fl.experiment = ?
                  AND fl.target_dt <= ?
                  AND fl.issued_at >= ?
                  AND fl.upper IS NOT NULL
                  AND fl.lower IS NOT NULL
                  {model_filter_sql}
                GROUP BY lead_bucket
                ORDER BY lead_bucket
            """, (
                interval_sec, interval_sec,
                bucket_min, bucket_min,
                experiment, now_str, cutoff_str,
                *model_filter_param,
            ))
            by_lead_rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Coverage query failed: {e}")
            return {"by_lead": {"lead_minutes": [], "coverage": [], "n": []}, "overall": {}}

        by_lead = {
            "lead_minutes": [int(r[0]) for r in by_lead_rows],
            "coverage": [round(float(r[1]), 4) for r in by_lead_rows],
            "n": [int(r[2]) for r in by_lead_rows],
        }

        try:
            cursor.execute(f"""
                WITH actuals_grid AS (
                    SELECT
                        strftime('%Y-%m-%d %H:%M:%S',
                            (CAST(strftime('%s', SUBSTR(ds, 1, 19)) AS INTEGER) / ?) * ?,
                            'unixepoch') AS grid_dt,
                        AVG(value) AS value
                    FROM {actuals_table}
                    GROUP BY grid_dt
                )
                SELECT
                    AVG(CASE WHEN ag.value BETWEEN fl.lower AND fl.upper
                             THEN 1.0 ELSE 0.0 END) AS coverage,
                    COUNT(*) AS n
                FROM forecast_log fl
                INNER JOIN actuals_grid ag ON ag.grid_dt = fl.target_dt
                WHERE fl.experiment = ?
                  AND fl.target_dt <= ?
                  AND fl.issued_at >= ?
                  AND fl.upper IS NOT NULL
                  AND fl.lower IS NOT NULL
                  {model_filter_sql}
            """, (
                interval_sec, interval_sec,
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

        return {"by_lead": by_lead, "overall": overall}

    def get_forecast_evolution(
        self,
        experiment: str,
        actuals_table: str,
        n_cycles: int = 12,
        interval_minutes: int = 30,
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

        # Find the last N distinct issuance timestamps for this experiment
        try:
            cursor.execute(
                "SELECT DISTINCT issued_at FROM forecast_log "
                "WHERE experiment = ? "
                "ORDER BY issued_at DESC LIMIT ?",
                (experiment, int(n_cycles)),
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

        # Fetch all rows for those issuances in one query
        placeholders = ",".join("?" * len(issued_ats))
        try:
            cursor.execute(
                f"SELECT issued_at, target_dt, predicted "
                f"FROM forecast_log "
                f"WHERE experiment = ? AND issued_at IN ({placeholders}) "
                f"ORDER BY issued_at ASC, target_dt ASC",
                (experiment, *issued_ats),
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

        # Pull actuals covering the same time window, snapped to the grid
        actuals_targets: list = []
        actuals_values: list = []
        if min_target and max_target:
            try:
                cursor.execute(
                    f"""
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
                    """,
                    (interval_sec, interval_sec, min_target, max_target),
                )
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

    def get_forecast_stability(
        self,
        experiment: str,
        max_age_days: int = 30,
        source_is_cumulative: bool = False,
        model_name: Optional[str] = None,
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
        model_filter_sql = " AND model_name = ?" if model_name else ""
        model_filter_param = (model_name,) if model_name else ()

        # --- Per-timestep cross-cycle stability ---
        # SQLite has no STDDEV; compute it via the sum-of-squares
        # identity: Var(X) = E[X^2] - E[X]^2. The subtraction can go
        # very slightly negative on perfectly-constant columns due to
        # float rounding, so clamp before SQRT.
        try:
            cursor.execute(
                f"""
                WITH per_target AS (
                    SELECT
                        target_dt,
                        AVG(predicted)              AS mean_p,
                        AVG(predicted * predicted) AS mean_pp,
                        COUNT(DISTINCT issued_at)  AS n_cycles
                    FROM forecast_log
                    WHERE experiment = ?
                      AND issued_at >= ?
                      {model_filter_sql}
                    GROUP BY target_dt
                    HAVING n_cycles >= 2
                )
                SELECT
                    target_dt,
                    mean_p,
                    CASE WHEN mean_pp - mean_p * mean_p > 0
                         THEN SQRT(mean_pp - mean_p * mean_p)
                         ELSE 0 END AS std_p,
                    n_cycles
                FROM per_target
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
            try:
                cursor.execute(
                    f"""
                    WITH per_cycle_day AS (
                        SELECT
                            issued_at,
                            SUBSTR(target_dt, 1, 10)  AS day,
                            SUM(predicted)            AS daily_total,
                            COUNT(*)                  AS n_bins
                        FROM forecast_log
                        WHERE experiment = ?
                          AND issued_at >= ?
                          {model_filter_sql}
                        GROUP BY issued_at, day
                    )
                    SELECT
                        day,
                        AVG(daily_total)                  AS mean_total,
                        AVG(daily_total * daily_total)   AS mean_tt,
                        COUNT(DISTINCT issued_at)         AS n_cycles,
                        MAX(n_bins)                       AS max_bins_in_day
                    FROM per_cycle_day
                    GROUP BY day
                    HAVING n_cycles >= 2
                    ORDER BY day ASC
                    """,
                    (experiment, cutoff_str, *model_filter_param),
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

        return {
            "experiment": experiment,
            "per_timestep": {
                "target_dt": target_dts,
                "mean": means,
                "std": stds,
                "cv_pct": cv_pcts,
                "n_cycles": n_cycles_list,
            },
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

    def cleanup_forecast_log(self, experiment: str, oldest_datetime: datetime) -> int:
        """Delete forecast log entries older than the specified datetime."""
        oldest_str = oldest_datetime.strftime("%Y-%m-%d %H:%M:%S")
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "DELETE FROM forecast_log WHERE experiment = ? AND issued_at < ?",
                (experiment, oldest_str),
            )
            self.conn.commit()
            deleted = cursor.rowcount
            if deleted:
                logger.info(f"Pruned {deleted} old forecast_log rows for {experiment}")
            return deleted
        except sqlite3.Error as e:
            logger.error(f"Error pruning forecast_log for {experiment}: {e}")
            self.conn.rollback()
            return 0

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
            logger.error(f"Error deleting forecast_log for {experiment}: {e}")
            self.conn.rollback()
            return 0

    # ------------------------------------------------------------------
    # Benchmark results persistence
    # ------------------------------------------------------------------

    def ensure_benchmark_table(self) -> None:
        """Create the benchmark_results table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                experiment TEXT PRIMARY KEY,
                data       TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        self.conn.commit()
        logger.debug("Ensured benchmark_results table")

    def save_benchmark_result(self, experiment: str, json_data: str) -> None:
        """
        Upsert a benchmark result as a JSON blob.

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
            self.conn.commit()
            logger.debug(f"Saved benchmark result for {experiment}")
        except sqlite3.Error as e:
            logger.error(f"Error saving benchmark result for {experiment}: {e}")
            self.conn.rollback()

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
            logger.error(f"Error loading benchmark results: {e}")
            return {}

    def delete_benchmark_result(self, experiment: str) -> None:
        """Delete stored benchmark result for an experiment."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "DELETE FROM benchmark_results WHERE experiment = ?",
                (experiment,),
            )
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error deleting benchmark result for {experiment}: {e}")
            self.conn.rollback()

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("HistoryDB closed")
