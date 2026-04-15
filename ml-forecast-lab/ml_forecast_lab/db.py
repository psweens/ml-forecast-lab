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
        interval_minutes: int = 30,
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
        interval_minutes : int
            Resampling grid interval in minutes. Raw actuals are snapped
            to the nearest grid boundary before joining against forecast
            targets (which are already grid-aligned).

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

        # --- Lead-time accuracy curve ---
        # Bucket lead_minutes into `interval_minutes`-sized bins so the
        # chart resolution matches the forecast grid. Hardcoding a 30-min
        # bucket loses resolution on fine-grained experiments (e.g. 5-min
        # interval with a 1h horizon collapses to only 3 buckets, which
        # also drives MAE→RMSE convergence within each large bucket).
        # Actuals are stored with raw irregular HA timestamps, while
        # forecast targets are grid-aligned. Snap actuals to the grid
        # (floor to nearest interval boundary) and average before joining.
        bucket_min = max(1, int(interval_minutes))
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
                    AVG(ABS(fl.predicted - ag.value)) AS mae,
                    SQRT(AVG((fl.predicted - ag.value) * (fl.predicted - ag.value))) AS rmse,
                    COUNT(*) AS n
                FROM forecast_log fl
                INNER JOIN actuals_grid ag
                    ON ag.grid_dt = fl.target_dt
                WHERE fl.experiment = ?
                  AND fl.target_dt <= ?
                  AND fl.issued_at >= ?
                GROUP BY lead_bucket
                ORDER BY lead_bucket
            """, (
                interval_sec, interval_sec,
                bucket_min, bucket_min,
                experiment, now_str, cutoff_str,
            ))
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
                ranked AS (
                    SELECT
                        fl.target_dt,
                        fl.lead_minutes,
                        fl.predicted,
                        ag.value AS actual,
                        ROW_NUMBER() OVER (PARTITION BY fl.target_dt ORDER BY fl.issued_at ASC) AS rn_first,
                        ROW_NUMBER() OVER (PARTITION BY fl.target_dt ORDER BY fl.issued_at DESC) AS rn_last,
                        COUNT(*) OVER (PARTITION BY fl.target_dt) AS n_forecasts
                    FROM forecast_log fl
                    INNER JOIN actuals_grid ag
                        ON ag.grid_dt = fl.target_dt
                    WHERE fl.experiment = ?
                      AND fl.target_dt <= ?
                      AND fl.issued_at >= ?
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
                    COUNT(*) AS n
                FROM first_last
            """, (interval_sec, interval_sec, experiment, now_str, cutoff_str))
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

        # --- Per-timestep cross-cycle stability ---
        # SQLite has no STDDEV; compute it via the sum-of-squares
        # identity: Var(X) = E[X^2] - E[X]^2. The subtraction can go
        # very slightly negative on perfectly-constant columns due to
        # float rounding, so clamp before SQRT.
        try:
            cursor.execute(
                """
                WITH per_target AS (
                    SELECT
                        target_dt,
                        AVG(predicted)              AS mean_p,
                        AVG(predicted * predicted) AS mean_pp,
                        COUNT(DISTINCT issued_at)  AS n_cycles
                    FROM forecast_log
                    WHERE experiment = ?
                      AND issued_at >= ?
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
                (experiment, cutoff_str),
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
            cv = (100.0 * std_p / abs(mean_p)) if abs(mean_p) > 1e-9 else 0.0
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
                    """
                    WITH per_cycle_day AS (
                        SELECT
                            issued_at,
                            SUBSTR(target_dt, 1, 10)  AS day,
                            SUM(predicted)            AS daily_total,
                            COUNT(*)                  AS n_bins
                        FROM forecast_log
                        WHERE experiment = ?
                          AND issued_at >= ?
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
                    (experiment, cutoff_str),
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
                "FROM forecast_log WHERE experiment = ? AND issued_at >= ?",
                (experiment, cutoff_str),
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
