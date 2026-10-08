"""v2.52.4: ``HistoryDB.log_external_forecast`` ran without the connection lock.

It was the one public ``HistoryDB`` method on the shared ``sqlite3``
connection without ``self._lock``. Concurrent publish cycles interleaved its
``executemany`` with other writes ("cannot start a transaction within a
transaction", "no more rows available"), and its ``rollback()`` on failure
could discard another thread's uncommitted write.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

from ml_forecast_lab.db import HistoryDB

EXP = "demand"


class _LockProbe:
    """Proxies the sqlite3 connection; records whether the calling thread
    held ``HistoryDB._lock`` at each ``executemany``."""

    def __init__(self, conn, lock):
        self._conn, self._lock = conn, lock
        self.held: list[bool] = []

    def cursor(self):
        probe, cur = self, self._conn.cursor()

        class _Cursor:
            def executemany(self, *a, **kw):
                probe.held.append(probe._lock._is_owned())
                return cur.executemany(*a, **kw)

            def __getattr__(self, name):
                return getattr(cur, name)

        return _Cursor()

    def __getattr__(self, name):
        return getattr(self._conn, name)


class TestExternalForecastLock:
    def test_insert_runs_under_the_connection_lock(self, tmp_path):
        db = HistoryDB(tmp_path / "history.db")
        db.ensure_external_forecast_log_table()
        probe = _LockProbe(db.conn, db._lock)
        db.conn = probe
        issued = datetime.now(timezone.utc).replace(tzinfo=None)
        n = db.log_external_forecast(
            EXP, "sensor.external", issued,
            [issued + timedelta(minutes=30 * (i + 1)) for i in range(3)],
            [1.0, 2.0, 3.0],
        )
        assert n == 3
        assert probe.held == [True]

    def test_concurrent_writers_do_not_interleave(self, tmp_path):
        """Forecast and external-forecast logging run on worker threads in
        the same publish cycle; none of them may fail or lose rows."""
        db = HistoryDB(tmp_path / "history.db")
        db.ensure_forecast_log_table()
        db.ensure_external_forecast_log_table()
        issued = datetime.now(timezone.utc).replace(tzinfo=None)
        targets = [issued + timedelta(minutes=30 * (i + 1)) for i in range(48)]
        errors: list[BaseException] = []

        def external(k):
            try:
                for j in range(20):
                    assert db.log_external_forecast(
                        EXP, f"sensor.ext_{k}", issued + timedelta(seconds=j),
                        targets, [1.0] * len(targets),
                    ) == len(targets)
            except BaseException as e:  # noqa: BLE001 — surfaced below
                errors.append(e)

        def internal(k):
            try:
                for j in range(20):
                    db.log_forecast(
                        f"{EXP}_{k}", issued + timedelta(seconds=j), targets,
                        [1.0] * len(targets), "lightgbm",
                    )
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=f, args=(k,))
                   for k in range(3) for f in (external, internal)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert not errors
        n = db.conn.execute(
            "SELECT COUNT(*) FROM external_forecast_log").fetchone()[0]
        assert n == 3 * 20 * len(targets)
