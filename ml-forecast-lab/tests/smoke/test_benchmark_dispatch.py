"""v2.55.1: Run All Benchmarks and run-benchmark actually run the benchmark.

Both routes used to call the async ``benchmark_callback`` without awaiting
it, so only a coroutine was built ("coroutine ... was never awaited"):
nothing ran, and the running flag set ahead of the call was never cleared.
Every later run-benchmark / run-pipeline / retrain / rollback for that
experiment was refused until the app restarted, and Stop could not clear
it. Both routes now enqueue on the same serial training queue as
run-pipeline. These tests pin that the callback runs, the flag always
clears, Run All runs one experiment at a time with the rest shown as
queued, nothing runs twice, and Stop works on running and queued entries.
"""

from __future__ import annotations

import asyncio
import gc
import textwrap
import threading
import time
import warnings
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

EXPS = ("exp_a", "exp_b", "exp_c")


def _yaml() -> str:
    blocks = "".join(
        textwrap.indent(textwrap.dedent(f"""\
            - name: "{n}"
              target_entity: "sensor.{n}"
              mode: "lab"
              interval_minutes: 30
              days_history: 14
              future_periods: 48
              models_enabled: ["lightgbm"]
              publish_prefix: "mlfl_"
              publish_name: "{n}"
        """), "  ")
        for n in EXPS
    )
    return 'timezone: "Europe/London"\nupdate_every_minutes: 360\n\nexperiments:\n' + blocks


class _Bench:
    """Async stand-ins for main.py's benchmark and stop-training triggers.

    The benchmark trigger mirrors ``_benchmark_trigger``: global lock, then
    try/finally end_benchmark. ``raise_early`` mimics a failure before that
    try (``load_config`` raising), which the queue's own finally must cover.
    """

    def __init__(self, appstate):
        self.appstate = appstate
        self.ran: list = []
        self.done: list = []
        self.active = 0
        self.max_active = 0
        self.hold = False
        self.raise_early: set = set()
        self.gates: dict = {}
        self._lock = None

    def gate(self, name) -> threading.Event:
        return self.gates.setdefault(name, threading.Event())

    def release_all(self):
        self.hold = False
        for n in EXPS:
            self.gate(n).set()

    async def benchmark(self, name):
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if name in self.raise_early:
                raise RuntimeError("load_config failed")
            try:
                self.appstate.start_benchmark(name)
                self.ran.append(name)
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                while self.hold and not self.gate(name).is_set():
                    await asyncio.sleep(0.005)
                await asyncio.sleep(0.01)
            finally:
                self.active -= 1
                self.done.append(name)
                self.appstate.end_benchmark(name)

    async def stop(self, name) -> bool:
        task = self.appstate._pipeline_tasks.get(name)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except BaseException:
                pass
            self.appstate.end_benchmark(name)
            return True
        return False


def _wait(pred, timeout=3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def _unawaited(ws) -> list:
    return [str(w.message) for w in ws
            if issubclass(w.category, RuntimeWarning) and "never awaited" in str(w.message)]


def _card(grid: str, name: str) -> str:
    for part in grid.split('<div class="experiment-card">')[1:]:
        if f"deleteExperiment('{name}'" in part:
            return part
    raise AssertionError(f"no dashboard card for {name}")


@pytest.fixture
def bench(tmp_path: Path):
    from ml_forecast_lab.web.app import ExperimentStatus, create_app

    cfg = tmp_path / "mlfl.yaml"
    cfg.write_text(_yaml())
    app = create_app(config_path=cfg)
    appstate = app.state.appstate
    for n in EXPS:
        appstate.experiment_statuses[n] = ExperimentStatus(
            name=n, target_entity=f"sensor.{n}", mode="lab",
        )
    with TestClient(app) as client:
        b = _Bench(appstate)
        b.client = client
        appstate.benchmark_callback = b.benchmark
        appstate.stop_training_callback = b.stop
        yield b
        b.release_all()
        _wait(lambda: b.active == 0 and not appstate._pipeline_tasks, timeout=2.0)


def _running(b, name) -> bool:
    return b.appstate.is_benchmark_running(name)


# --- run-benchmark --------------------------------------------------------

def test_run_benchmark_runs_callback_and_clears_flag(bench):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        r = bench.client.post("/experiment/exp_a/run-benchmark")
        gc.collect()
    assert r.status_code == 202, r.text
    assert r.json()["experiment"] == "exp_a" and r.json()["status"] == "queued"
    assert not _unawaited(ws), _unawaited(ws)
    assert _wait(lambda: bench.ran == ["exp_a"]), bench.ran
    assert _wait(lambda: not _running(bench, "exp_a"))
    # The experiment is free again: a second run is accepted and runs.
    assert bench.client.post("/experiment/exp_a/run-benchmark").status_code == 202
    assert _wait(lambda: bench.ran == ["exp_a", "exp_a"]), bench.ran


def test_run_benchmark_refuses_while_running(bench):
    bench.hold = True
    bench.client.post("/experiment/exp_a/run-benchmark")
    assert _wait(lambda: bench.ran == ["exp_a"])
    assert _running(bench, "exp_a")
    assert bench.client.post("/experiment/exp_a/run-benchmark").status_code == 409
    assert bench.client.post("/experiment/exp_a/run-pipeline", json={}).status_code == 409
    bench.release_all()
    assert _wait(lambda: not _running(bench, "exp_a"))


def test_flag_clears_when_callback_fails_before_its_own_finally(bench):
    bench.raise_early = {"exp_a"}
    bench.client.post("/experiment/exp_a/run-benchmark")
    assert _wait(lambda: not bench.appstate._pipeline_tasks)
    assert not _running(bench, "exp_a")
    assert bench.ran == []


def test_unknown_experiment_and_missing_callback(bench):
    assert bench.client.post("/experiment/nope/run-benchmark").status_code == 404
    bench.appstate.benchmark_callback = None
    assert bench.client.post("/experiment/exp_a/run-benchmark").status_code == 501
    assert bench.client.post("/api/benchmarks/run-all").status_code == 501
    assert not bench.appstate.running_benchmarks
    assert not bench.appstate.training_queue


# --- run-all ----------------------------------------------------------------

def test_run_all_runs_every_experiment_once_serially(bench):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        r = bench.client.post("/api/benchmarks/run-all")
        gc.collect()
    assert r.status_code == 202
    assert r.json() == {
        "message": "Queued 3 benchmark(s)", "queued": list(EXPS), "skipped": [],
    }
    assert not _unawaited(ws), _unawaited(ws)
    assert _wait(lambda: bench.done == list(EXPS), timeout=5), bench.done
    assert bench.ran == list(EXPS)
    assert bench.max_active == 1
    assert _wait(lambda: not bench.appstate.running_benchmarks)


def test_run_all_shows_one_running_and_the_rest_queued(bench):
    bench.hold = True
    bench.client.post("/api/benchmarks/run-all")
    assert _wait(lambda: bench.ran == ["exp_a"])
    assert bench.appstate.running_benchmarks == {"exp_a"}
    grid = bench.client.get("/api/dashboard/grid").text
    assert "Stop Training" in _card(grid, "exp_a")
    assert "Queued (#1)" in _card(grid, "exp_b")
    assert "Queued (#2)" in _card(grid, "exp_c")


def test_run_all_skips_running_and_queued_without_duplicates(bench):
    bench.hold = True
    bench.client.post("/experiment/exp_a/run-pipeline", json={"steps": ["benchmark"]})
    assert _wait(lambda: bench.ran == ["exp_a"])
    bench.client.post("/experiment/exp_b/run-pipeline", json={"steps": ["benchmark"]})
    r = bench.client.post("/api/benchmarks/run-all").json()
    assert r["skipped"] == ["exp_a", "exp_b"] and r["queued"] == ["exp_c"], r
    # A second Run All while the first is in flight queues nothing.
    r = bench.client.post("/api/benchmarks/run-all").json()
    assert r["queued"] == [] and r["skipped"] == list(EXPS), r
    bench.release_all()
    assert _wait(lambda: sorted(bench.done) == sorted(EXPS), timeout=5), bench.done
    _wait(lambda: False, timeout=0.3)  # a duplicate run would land here
    assert bench.ran == list(EXPS), bench.ran


# --- stop -------------------------------------------------------------------

def test_stop_cancels_the_running_benchmark_and_the_queue_moves_on(bench):
    bench.hold = True
    bench.client.post("/api/benchmarks/run-all")
    assert _wait(lambda: bench.ran == ["exp_a"])
    r = bench.client.post("/experiment/exp_a/stop-training").json()
    assert r == {"success": True}, r
    assert not _running(bench, "exp_a")
    assert _wait(lambda: bench.ran == ["exp_a", "exp_b"]), bench.ran
    assert _running(bench, "exp_b")


def test_stop_removes_a_queued_benchmark(bench):
    bench.hold = True
    bench.client.post("/api/benchmarks/run-all")
    assert _wait(lambda: bench.ran == ["exp_a"])
    r = bench.client.post("/experiment/exp_c/stop-training").json()
    assert r == {"success": True, "was_queued": True}, r
    bench.release_all()
    assert _wait(lambda: bench.done == ["exp_a", "exp_b"], timeout=5), bench.done
    _wait(lambda: False, timeout=0.3)
    assert "exp_c" not in bench.ran
