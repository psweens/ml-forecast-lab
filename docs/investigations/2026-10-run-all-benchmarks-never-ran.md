# Run All Benchmarks never ran (v2.55.1)

**Status:** Shipped in v2.55.1 (issue #120). Both web routes that start a
benchmark built a coroutine and dropped it, so no benchmark ran and the
experiment stayed flagged as running. The fix, why it goes through the
training queue rather than scheduling the callback directly, and what is
left on the same path.

The pinning tests: `ml-forecast-lab/tests/smoke/test_benchmark_dispatch.py`
(web stack only). All nine fail on v2.55.0.

```bash
cd ml-forecast-lab/
python -m pytest tests/smoke/test_benchmark_dispatch.py -v
```

---

## Defect

`POST /experiment/{name}/run-benchmark` and `POST /api/benchmarks/run-all`
(the dashboard's **Run All Benchmarks** button) called
`appstate.start_benchmark(name)` and then `appstate.benchmark_callback(name)`
with no `await`. The callback is `main.py`'s `async def _benchmark_trigger`,
so the call returned an unawaited coroutine
(`RuntimeWarning: coroutine '..._benchmark_trigger' was never awaited`,
printed once per call site per process). The route answered 202 "queued".

The matching `end_benchmark` lives in the trigger's `finally`, which never
ran. Until the app restarted, the experiment:

- answered 409 to run-benchmark and run-pipeline, and was listed under
  `skipped` by run-all;
- was refused by Retrain and Roll back ("Training already running" /
  "Training is in progress");
- rendered as running on the dashboard (Stop Training in place of its
  buttons, delete disabled, 10 s grid polling) and on the experiment page;
- could not be cleared by Stop Training: `_stop_training_trigger` found no
  task in `_running_tasks` or `_pipeline_tasks` and returned False without
  touching the flag.

Scheduled retrains were not blocked (they gate on `_running_tasks` and
`_training_lock`, not `running_benchmarks`). A lab experiment's next
scheduled retrain cleared the flag as a side effect, because
`update_experiment` calls `end_benchmark` after `_run_benchmark`; a
production experiment stayed stuck until restart.

The v2.30.0 CHANGELOG line "the route now invokes `benchmark_callback`" and
the comment above the call described the intent, not the behaviour: the bare
call never ran anything. The smoke harness did not catch it because no test
wired an async callback and checked that its body ran.

## Fix

Both routes now enqueue `{"name", "steps": ["benchmark"]}` on
`appstate.training_queue` and spawn `_process_training_queue`, the path the
experiment page's **Run Pipeline** button already used:

- `_run_pipeline` awaits the callback and clears the flag in its own
  `finally`, so the flag clears even when the trigger fails before its own
  `try` (`load_config` raising) or returns early (experiment absent from the
  reloaded config: `_benchmark_trigger` has no `else` for `exp_cfg is None`).
- One benchmark runs at a time. Only the running experiment is flagged; the
  rest show **Queued (#n)**, and clicking that removes them via
  `remove_from_queue`. Stop Training cancels the running one through
  `_pipeline_tasks`.
- Both routes skip experiments already running *or queued*, so overlapping
  run-pipeline / run-all requests no longer run an experiment twice.
- With no callback wired both routes answer 501, as run-pipeline does.
- run-pipeline's own queue kick now goes through `appstate.spawn` instead of
  a bare `ensure_future`, so the processor task is strongly referenced and a
  failure is logged.

### Why not `create_task(benchmark_callback(name))`

That was the issue's proposal. Prototyped and tested against the same
behaviour checks, it fixes the repro but:

- run-all marks every experiment running at once and parks N tasks on
  `_training_lock`, which is FIFO, so a scheduled retrain that arrives
  during a run-all waits behind the whole batch (the same reason tune-all
  takes the lock per model);
- it still leaves the flag stuck when the trigger raises before its
  `finally` or finds no `exp_cfg`;
- an experiment already queued by run-pipeline runs twice, and the two
  tasks share one `_pipeline_tasks[name]` slot, so the first one's done
  callback drops the second and Stop cannot find it.

## Left on the same path (not changed)

- `_benchmark_trigger` returns silently when the experiment is not in the
  reloaded config (`main.py`, `if exp_cfg:` with no `else`). Harmless now
  that the queue clears the flag, but there is no log line.
  `experiment_statuses` is built at startup and not reconciled when
  `mlfl.yaml` is edited by hand.
- `_stop_training_trigger` checks `_running_tasks` before `_pipeline_tasks`.
  If a UI benchmark for X is waiting behind a scheduled retrain of X, Stop
  cancels the retrain and the benchmark still runs.
- `delete_experiment_route` does not discard the name from
  `running_benchmarks` or `training_queue`. The queue processor skips deleted
  names, and `_run_pipeline` clears the flag when the run ends, so this only
  mattered while the flag could stick.
- run-pipeline reports `queue_position` 1 / "Pipeline started" when the
  queue processor is idle, even if a scheduled retrain holds the training
  lock and the run is actually waiting.
- The dashboard swaps run-all's JSON body into `#status-message` as text.
