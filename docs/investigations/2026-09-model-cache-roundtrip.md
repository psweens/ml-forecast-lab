# Production model cache round trip (v2.52.3)

**Status:** Shipped in v2.52.3. Defect analysis, fix rationale and
verification record behind that release's CHANGELOG entry.

The pinning tests are in
`ml-forecast-lab/tests/unit/test_cache_roundtrip.py`. They drive the real
`_retrain_and_cache` → `_persist_cached_model` → `_restore_cached_models` /
`_rollback_cached_model` → `_forecast_with_cached` path and assert that the
post-restore forecast is **bit-identical** to the one the live model
published.

```bash
cd ml-forecast-lab/
python -m pytest tests/unit/test_cache_roundtrip.py -v   # ~12 s
```

---

## TL;DR

| Defect (live before v2.52.3) | Observed | Fix |
| --- | --- | --- |
| **A.** `XGBoostModel.save` wrote metadata to `path + ".metadata.json"`; the persist saves to `model.bin.tmp` and renames only that file | The sidecar was stranded at `model.bin.tmp.metadata.json`. `load` warned, left `feature_names_=None` and reported success, so the retrain was deferred and every tick raised `TypeError: object of type 'NoneType' has no len()`. `previous/` never held the sidecar | Metadata rides inside the model file as a booster attribute (`mlfl_metadata`), written as explicit UBJSON |
| **A0.** All-zero XGBoost importances skipped normalisation and stayed `np.float32` | `json.dump` raised inside `save`; the persist aborted before its rename and disk kept the previous generation, possibly a different backend, which restore served silently | Importances are cast to `float`; restore warns when the loaded `model_name` differs from `production_model` |
| **B.** Neural `load()` called `torch.load(path, map_location="cpu")` | On torch ≥ 2.6 (`weights_only=True` by default), every checkpoint carrying numpy stats was refused and forced a retrain on each restart | `models.base.load_torch_checkpoint`, which stays `weights_only` with the numpy globals allowlisted |

## A: the single-file save contract

`_persist_cached_model`, the `previous/` archive, `_rollback_cached_model`
and `_reset_cached_model` all handle `model.bin` **by name**. A backend that
writes anything beside `path` loses it at the first of those steps. The fix
restores the contract instead of teaching four call sites about sidecars:
**`save(path)` writes exactly `path`**. The test
`test_save_writes_exactly_the_file_it_was_given` pins it at the persist
layer: after a retrain, the cache directory holds exactly `cache_meta.json`
and `model.bin`.

`save` writes `booster.save_raw(raw_format="ubj")` rather than calling
`save_model(path)`, because `save_model` picks its format from the file
extension, and the persist writes to `.tmp`. On xgboost 2.0.x that meant the
deprecated binary format; on 2.1 it means UBJSON with a warning.

**Legacy caches.** A cache persisted by v2.52.2 or earlier holds an intact
booster without metadata. `load` falls back, in order, to:

1. the embedded attribute;
2. a `path + ".metadata.json"` sidecar (files saved outside the persist);
3. the booster's own `num_features()`, with `fit()`'s default names.

`predict` needs only the column count. Column alignment comes from
`cache_meta.json`'s `feature_cols`, not from the model. Verified
bit-identical for main files written as UBJSON, JSON and the deprecated
binary format (`test_cache_written_before_the_fix_still_restores` covers the
2.1 default).

## A0: how common all-zero importances are

All-zero importances look like an edge case but are the normal outcome for
XGBoost's default configuration on most sensors (see *Open finding* below).
The normalised path was always safe: `np.float32 / np.float32` summed from
the Python int `0` promotes to `np.float64`, a `float` subclass that
`json.dump` accepts. Only the skipped-normalisation branch leaked
`np.float32`.

### Why restore warns instead of rejecting a mismatch

`_rollback_cached_model` swaps disk generations without touching
`production_model`. A user who rolls back to a different backend therefore
leaves `cache_meta.json.model_name != production_model` **legitimately**.
Rejecting the mismatch at restore would undo that rollback on every
restart. Restore therefore serves the model and logs one warning naming
both backends; the next scheduled retrain trains the configured one.

## B: why an allowlist rather than tensors at save time

Converting the stats to tensors or lists in `save` would still need the
allowlist to load the caches already on users' disks, so it adds work
without removing any. The helper allowlists:

- `numpy.core.multiarray._reconstruct` (arrays);
- `numpy.core.multiarray.scalar` (numpy scalars);
- `np.ndarray` and `np.dtype`;
- the dtype classes for float16/32/64, int32/64 and bool. A dtype reduces to
  its own class (`numpy.dtype[float32]`), not to `np.dtype`.

Any other global is still refused (`test_loading_stays_weights_only`). On
torch < 2.5 there is no scoped `safe_globals`. The helper then passes
`weights_only=False`, which is that version's own default, so behaviour
there is unchanged.

All 22 torch-checkpoint backends store `channel_mean` / `channel_std` /
`y_mean` / `y_std`. The channel stats are numpy whenever RevIN is off, and
always for `nbeats` / `nhits`, which have no RevIN; a z-score head adds
numpy `y_mean` / `y_std`. With RevIN on (the default) every other backend
skips both, even with a z-score head, which is why most users never hit B.
`test_every_neural_checkpoint_round_trips_its_numpy_stats` discovers the
backends from `WIRED_BACKENDS`, so a new one is covered automatically.
`test_no_backend_calls_torch_load_directly` keeps them routed through the
helper. The pre-existing `test_models.py` save/load tests all run with
RevIN defaults, which put no numpy in the checkpoint; that is why CI never
saw B.

## Verification record

The new suite was run against the v2.52.1 sources, with only the hard-coded
`/data/ml_forecast_lab/models` gate removed from `_restore_cached_models`
(it made every restore a no-op outside the container, masking the causes).
Each failure matched its defect:

- The XGBoost round trip restored "successfully", then raised
  `TypeError: object of type 'NoneType' has no len()` (A).
- `nbeats`, `nhits` and `lstm` (RevIN off, z-score) failed at restore, as
  did all 22 cases of the checkpoint sweep (B).
- The constant-target XGBoost persist raised
  `Object of type float32 is not JSON serializable`, leaving no
  `cache_meta.json` (A0).
- `lightgbm` and default `nlinear` (RevIN on) passed, as controls.

Since v2.52.2 that gate is derived from `_cached_model_dir(...).parent`, so
monkeypatching `_cached_model_dir` alone redirects the whole cache and the
suite needs no other patching.

## Open finding (not fixed here): XGBoost's default loss grows no splits

With the backend defaults (`loss_fn='huber'` → `reg:pseudohubererror`,
xgboost's `huber_slope=1`, `min_child_weight=1`), a clean daily sinusoid of
standard deviation ~7 trains **30 of 30 trees with zero splits**
(`best_iteration=0`). The published forecast is the flat intercept. The
same data with `reg:squarederror` grows 32 splits and a forecast standard
deviation of 6.78. Scaling the target to about ±0.5 also restores splits. Neither
`min_child_weight=0` nor `huber_slope=std(y)` does, so the mechanism is more
than the hessian floor alone and needs its own investigation.

`ExperimentCfg.loss_fn` also defaults to `huber`, so by default both the
benchmark and the production retrain train XGBoost this way. Any XGBoost
model on a sensor whose values vary by more than about a unit is likely a
constant. A fix changes benchmark results, so it ships as its own minor
release. The round-trip tests train XGBoost with `loss_fn: mse` so their
bit-identity comparison is not between two constants.

A related wiring gap turned up alongside it. `_apply_experiment_neural_params`
passes the experiment's `loss_fn` to tree backends so that benchmark and
production train under the same objective. But `_retrain_and_cache`, the
holdout refit and `_run_production_inference` still gate `loss_fn` on
`is_neural` inline. With a non-default `loss_fn` (e.g. `mse`), a tree champion
is benchmarked under that objective and served under its backend default.
