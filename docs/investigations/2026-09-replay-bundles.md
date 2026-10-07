# Replay bundles: capture/replay design and what building them found (v2.52.2)

**Status:** Shipped in v2.52.2. This is the design rationale and
verification record behind that release's CHANGELOG entry: what a bundle
records and why, how replay decides "identical", and the existing defects
the forecast capture exposed.

The pinning tests are `ml-forecast-lab/tests/unit/test_replay.py` (training
frame) and `tests/unit/test_replay_forecast.py` (forecast stage, real
LightGBM and NLinear models).

```bash
cd ml-forecast-lab/
pytest tests/unit/test_replay.py tests/unit/test_replay_forecast.py -v
python -m ml_forecast_lab.replay bundle.zip --trust-model   # a user's bundle
```

---

## TL;DR

- A bundle records every response the pipeline receives at its two I/O
  boundaries (HA interface, history database) for two stages, each run on
  its own shadow app with writes suppressed:
  - the **training frame**: `_prepare_training_frame` → `_build_training_windows`;
  - the **forecast**: `_compute_cached_forecast` → `_conformal_bands`, using
    the production model saved from memory.
- Replay drives the same production methods against the recordings at the
  captured instant and diffs every stage.
- On the same host fingerprint everything must be bit-identical. Across
  hosts the rule depends on the backend family (below).
- Building the forecast stage exposed three existing save/restore defects
  and one latent `KeyError`. They are outside this release and are tracked
  separately (see "Defects found"). The bundle's `model_roundtrip` and
  `disk` checks report them on a user's machine.

## Why record at the I/O boundary

The per-retrain debug dump (`debug_dump.py`) captures `combined` after
`_supervised_frame`. By then resampling, feature construction and
missingness resolution have run, which is where most "forecasts are wrong"
regressions lived (`2026-08-missingness-masking.md`). It records nothing of
the forecast's own fetch, future covariates or bands.

Recording responses instead of intermediate frames means replay runs the
*current* code on the *user's* inputs. A diff therefore localises a change
to a stage and a column rather than just flagging the output.

Replay must not be a reimplementation, or it drifts from production. It
calls extracted production methods instead:
- `_prepare_training_frame` and `_build_training_windows` (training frame);
- `_compute_cached_forecast` and `_conformal_bands` (forecast);
- `build_model_registry`, `_cache_meta` and `_cache_entry_from_meta` (model
  cache).

The live paths call the same methods, and the refactor changed no live
behaviour. That is pinned by the unchanged unit and integration suites and
by `TestComputePublishSplit`.

## What is recorded, and the traps

| Boundary | Methods | Notes |
| --- | --- | --- |
| HA | `get_history`, `get_state`, `get_config`, `api_call` | `get_config` is trimmed to latitude/longitude. `api_call` is allow-listed to `POST weather.get_forecasts?return_response` only, because it is also HA's write path. |
| History DB | `get_history`, `get_conformal_quantiles` | `safe_table_name` is pure, so it passes through. `store_history` and `cleanup` are suppressed. The conformal key is every bound argument with defaults applied; since v2.52.4 `source_is_cumulative` joins it only when true, so bundles recorded earlier keep their keys. A cumulative experiment's bundle from v2.52.3 or earlier recorded raw-counter quantiles, which the current tree no longer requests: its forecast stage stops at `UnrecordedCall` (exit 2). |
| Model | `model.save` of a deep copy of the live model | The live object is never touched from the capture thread, because XGBoost's `save_model` sets and clears Booster attributes. |

**Lossless recording.** The recorder hands the capture pipeline the
*decoded recorded value*, the same one replay will serve. Otherwise a lossy
codec produces a capture that is wrong yet still "replays identically":
- Conformal quantiles are keyed by lead-bucket `int`s, which JSON turns into
  strings. The band lookup `quantiles.get(int(b))` then silently falls back
  to the pooled quantile for every lead. They travel as `[int, value]` pairs
  (`TestForecastRoundTrip::test_lead_bucket_quantiles_survive_the_bundle`).
- Cache timestamps travel as epoch nanoseconds. `Timestamp.isoformat()`
  drops `.000000` on whole-second rows, and `pd.to_datetime` without a
  format infers one shape from the first row and raises on the other. HA's
  `last_changed` has microsecond resolution, so about 1 row in 10⁶ is
  whole-second. On a fast sensor with hundreds of thousands of cached rows,
  that made capture crash on the target table, or, on a covariate table
  (where `fetch_history` swallows the error and falls back to a full HA
  fetch), record a different frame from the one the add-on trained on
  (`TestHistoryCodec`, plus the jittered stub timestamps in every
  end-to-end test).
- Recorded failures replay with the same exception class name and
  `str()`, because the pipeline stores them as `f"{type(e).__name__}: {e}"`
  (`fetch_error`, `bands_error`) (`TestRecordedErrors`).

**Swallowed misses.** The pipeline deliberately catches many I/O errors:
`fetch_future`, `_get_site_location`, the band computation's caller and
covariate fetches. An unrecorded request would therefore replay as "no
covariate" or "no band" and look like a real data gap. Both recorder and
replay log every miss even when the exception is swallowed, and replay
exits 2 (`TestStrictness`). The same fix closed a gap in the first
iteration, where only the target `get_history` path was strict.

**One shadow app per stage, one call log per stage.** A second
`_fetch_and_preprocess` on the same app diverges. The covariate resolver's
`_backfill_horizon` state, together with the suppressed cache writes, turns
the second fetch into a delta fetch.

**The forecast stage uses `cache["exp_cfg"]`,** the retrain-time snapshot,
not the live config, which may have been edited since
(`TestCaptureSideEffects::test_forecast_uses_the_trained_config_not_a_later_edit`).

**History trimming.** Cache tables are keyed by entity, so they can hold
another experiment's longer history. Rows older than one day before the
stage's window start are dropped before recording. Every consumer filters to
`ds >= start` before any decision, including the covariate backfill check,
so replay is unaffected (`TestHistoryTrim`).

## Comparison policy

| Host fingerprint | Backend family | Rule |
| --- | --- | --- |
| Same machine, CPU (`/proc/cpuinfo` model, numpy dispatch features, torch CPU capability), Python, library versions, torch threads | any | bit-identical |
| Different | lightgbm, xgboost, catboost, seasonal_naive, daily_profile | bit-identical: order-fixed sums and comparisons; recursive lag feedback makes any drift large, not small |
| Different | torch and foundation models | \|Δ\| ≤ 1e-4 × max(\|expected array\|, \|raw model output\|). The output term stops a forecast clamped to zero (night-time PV) collapsing the tolerance to ~0. |
| Different | arima, ets, theta | informational: statsforecast refits with a discrete order search at predict time |

Model *inputs* (`X_rows` for trees, the inference window for neural models)
are always compared exactly. A mismatch there points at input construction
(pvlib/holidays versions, libm) or a code change, not at the model.

Replay sets torch's intra-op thread count to the capture's, and
`HF_HUB_OFFLINE=1`. Foundation-model weights are not in the bundle, and a
changed download would otherwise look like a regression.

## Trust

Several backends load `model.bin` with `pickle` (LightGBM, CatBoost,
statsforecast, seasonal_naive, daily_profile, chronos_bolt, ttm), as does
`torch.load` below 2.6. Loading a bundled model therefore runs code from
whoever made the bundle. The forecast stage only runs with `--trust-model`,
and a bundle from a public issue belongs in a throwaway container or VM.

Bundle entry names are checked before extraction: absolute names, `..` and
backslashes are refused (`TestBundleSafety`). Everything else (frames,
forecast arrays) is JSON or `np.load(allow_pickle=False)`.

A pickle-free export (LightGBM `model_to_string`, CatBoost `.cbm`, neural
`state_dict` under `weights_only=True`) would remove the need for trust. It
is a possible follow-up.

## Defects found while building the forecast stage (not fixed here)

These are real on `main` and outside a diagnostics release, so they are
tracked as separate tasks.
- **A: XGBoost sidecar name.** `_persist_cached_model` saves to
  `model.bin.tmp`, so the metadata sidecar is written as
  `model.bin.tmp.metadata.json`. `load` looks for `model.bin.metadata.json`.
  A restored XGBoost champion then raises `TypeError` on every forecast
  until the next retrain.
- **A0: XGBoost float32 importances.** When importances are left
  un-normalised, `json.dump` raises. Persist aborts before the rename, and
  restore never checks the restored `model_name` against the configured
  champion, so a restart can serve an older model from another backend.
  Capture reports this as `model_save_failed`, with the disk check still
  run.
- **B: torch ≥ 2.6 `weights_only=True`.** It rejects the numpy channel
  statistics that N-BEATS and NHiTS always store, and that any
  `use_revin=False` model stores. Restore then fails, and the add-on
  retrains after every restart.
- **C14: tree forecast with sun elevation but no clear-sky irradiance.**
  `future_solar.loc[ts, 'clear_sky_ghi']` raises `KeyError`, because
  `compute_solar_features` emits only the requested columns.

## Known limits

- The forecast stage replays from the capture's own pinned `now`. The live
  tick that published the user's forecast ran at a different instant, so
  the published forecast itself is not a comparison target.
- Capture runs a forecast and a model save/load on the Pi alongside the live
  ticks. statsforecast refits and Chronos/TTM may load weights, so allow
  seconds, not milliseconds.
- Cross-host tolerances were reasoned from how each backend computes, not
  measured on a Pi-to-x86 pair. The first real bundle will calibrate them.
