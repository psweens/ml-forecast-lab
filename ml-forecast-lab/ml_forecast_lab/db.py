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
                forecast_type TEXT   NOT NULL DEFAULT 'cached'
            )
        """)
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

        Returns
        -------
        int
            Number of rows inserted.
        """
        issued_str = issued_at.strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        for ts, val in zip(targets, predictions):
            target_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            lead_min = int((ts - issued_at).total_seconds() / 60)
            rows.append((
                experiment, model_name, issued_str, target_str,
                lead_min, float(val), forecast_type,
            ))
        cursor = self.conn.cursor()
        try:
            cursor.executemany(
                "INSERT INTO forecast_log "
                "(experiment, model_name, issued_at, target_dt, lead_minutes, predicted, forecast_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
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
    ) -> dict:
        """
        Compute forecast accuracy by lead time.

        Joins forecast_log predictions against the actuals table and
        returns MAE/RMSE grouped by lead-time buckets, plus revision
        improvement data (first vs last forecast for each target).

        Parameters
        ----------
        experiment : str
            Experiment name.
        actuals_table : str
            SQL-safe table name for the actuals.
        max_age_days : int
            Only consider forecasts issued in the last N days.

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

        # --- Lead-time accuracy curve ---
        # Bucket lead_minutes into 30-min bins for cleaner charts
        try:
            cursor.execute(f"""
                SELECT
                    CAST((fl.lead_minutes / 30) * 30 AS INTEGER) AS lead_bucket,
                    AVG(ABS(fl.predicted - a.value)) AS mae,
                    SQRT(AVG((fl.predicted - a.value) * (fl.predicted - a.value))) AS rmse,
                    COUNT(*) AS n
                FROM forecast_log fl
                INNER JOIN {actuals_table} a
                    ON SUBSTR(a.ds, 1, 19) = fl.target_dt
                WHERE fl.experiment = ?
                  AND fl.target_dt <= ?
                  AND fl.issued_at >= ?
                GROUP BY lead_bucket
                ORDER BY lead_bucket
            """, (experiment, now_str, cutoff_str))
            lead_rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Forecast accuracy query failed: {e}")
            return {"error": str(e)}

        lead_time_curve = {
            "lead_minutes": [r[0] for r in lead_rows],
            "mae": [round(r[1], 4) for r in lead_rows],
            "rmse": [round(r[2], 4) for r in lead_rows],
            "sample_count": [r[3] for r in lead_rows],
        }

        # --- Revision improvement ---
        # Compare the FIRST forecast for each target_dt vs the LAST
        try:
            cursor.execute(f"""
                WITH ranked AS (
                    SELECT
                        fl.target_dt,
                        fl.lead_minutes,
                        fl.predicted,
                        a.value AS actual,
                        ROW_NUMBER() OVER (PARTITION BY fl.target_dt ORDER BY fl.issued_at ASC) AS rn_first,
                        ROW_NUMBER() OVER (PARTITION BY fl.target_dt ORDER BY fl.issued_at DESC) AS rn_last
                    FROM forecast_log fl
                    INNER JOIN {actuals_table} a
                        ON SUBSTR(a.ds, 1, 19) = fl.target_dt
                    WHERE fl.experiment = ?
                      AND fl.target_dt <= ?
                      AND fl.issued_at >= ?
                ),
                first_last AS (
                    SELECT target_dt,
                           MAX(CASE WHEN rn_first = 1 THEN predicted END) AS first_pred,
                           MAX(CASE WHEN rn_last  = 1 THEN predicted END) AS last_pred,
                           MAX(actual) AS actual
                    FROM ranked
                    WHERE rn_first = 1 OR rn_last = 1
                    GROUP BY target_dt
                    HAVING first_pred IS NOT NULL AND last_pred IS NOT NULL
                       AND first_pred != last_pred
                )
                SELECT
                    AVG(ABS(first_pred - actual)) AS first_mae,
                    AVG(ABS(last_pred  - actual)) AS last_mae,
                    COUNT(*) AS n
                FROM first_last
            """, (experiment, now_str, cutoff_str))
            rev_row = cursor.fetchone()
        except sqlite3.Error as e:
            logger.warning(f"Revision improvement query failed: {e}")
            rev_row = None

        revision = {}
        if rev_row and rev_row[2] > 0:
            first_mae = round(rev_row[0], 4)
            last_mae = round(rev_row[1], 4)
            improvement = round((1 - last_mae / first_mae) * 100, 1) if first_mae > 0 else 0
            revision = {
                "first_forecast_mae": first_mae,
                "latest_forecast_mae": last_mae,
                "improvement_pct": improvement,
                "sample_count": rev_row[2],
            }

        # --- Summary stats ---
        cursor.execute(
            "SELECT COUNT(*), MIN(issued_at), MAX(issued_at) "
            "FROM forecast_log WHERE experiment = ? AND issued_at >= ?",
            (experiment, cutoff_str),
        )
        stats_row = cursor.fetchone()

        return {
            "experiment": experiment,
            "lead_time_curve": lead_time_curve,
            "revision_improvement": revision,
            "total_logged": stats_row[0] if stats_row else 0,
            "date_range": {
                "from": stats_row[1] if stats_row else None,
                "to": stats_row[2] if stats_row else None,
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
