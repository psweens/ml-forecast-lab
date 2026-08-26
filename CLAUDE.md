# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML Forecast Lab is a Home Assistant app (add-on) that forecasts any HA sensor: it benchmarks
~29 time-series model backends on the sensor's history, promotes a winner, retrains it on
schedule, and publishes calibrated forecasts (with 80% conformal bands) back to HA as sensors.
Target hardware is a Raspberry Pi 5 (8 GB, no GPU, ARM64), which drives most performance
decisions. The intended user mindset is "benchmark once, run forever".

This is an add-on *repository* (root `repository.yaml` makes it installable in HA); the entire
application lives under `ml-forecast-lab/`. The root holds only store metadata and the
top-level README.

## Running Tests

Run everything from `ml-forecast-lab/` — there is no pytest config file, so the rootdir and
test imports resolve from the working directory.

```bash
cd ml-forecast-lab
pip install -r requirements.txt -r tests/requirements-dev.txt   # full deps (torch is large)

python -m pytest tests/unit/ -v --tb=short          # unit suite (what CI runs; ~30 s)
python -m pytest tests/smoke/ -v --tb=short         # FastAPI UI/API gate (~10 s, web deps only)
python -m pytest tests/integration/ -v --tb=short   # real train→predict→publish path (~3–4 min)

# single test / class
python -m pytest tests/unit/test_missingness_masking.py::TestOrdering -v

# fast all-backends config/shape check without paying training cost
python tests/dryrun_pipeline.py
```

The integration suite is slow — save its output to a file and grep that, rather than piping to
grep and re-running when the pattern was wrong. CI (`.github/workflows/tests.yml`) runs the
three suites as separate jobs; the smoke job installs only the web stack, so smoke tests must
not import ML backends. `pandas` is pinned `<3.0` and `numpy` `<2.0` — match those locally or
pandas-3 behaviour differences will bite. There is no linter or formatter configured; match the
surrounding style. `tests/synthetic/` holds research harness scripts (`run_phase*.py`), not
pytest suites — they produced the reports in `docs/investigations/`.

### Training dumps

When an experiment sets `debug_save_training_dumps: true`, each retrain writes a bundle to
`<config_dir>/debug/<experiment>/<UTC-iso>/`: `meta.json` (hyperparams, channel order, flags),
`training.parquet` (the exact combined frame that fed `fit()`), `sliding_window.npz` (neural
tensors), and `forecast.parquet` (the immediate post-retrain forecast). This is the production
training surface captured offline — built for the "synthetic tests pass but production fails"
regression signature. There is no replay tool; examine the bundles directly.

### Debugging journal

`docs/investigations/*.md` records what past investigations found (neural-PV root causes,
prototype fixes, phase observations). Read the relevant one before debugging a "forecasts are
wrong/flat/zero" report, and add a new file when an investigation learns something a future
session would want. Deep design rationale lives there too — see the CHANGELOG rules below.

## Release mechanics

The version lives in **two places that must agree**: `ml-forecast-lab/config.yaml` (`version:`)
and `ml_forecast_lab/__init__.py` (`__version__`). `release.yml` refuses a tag whose name
(`v<version>`) doesn't match `config.yaml`, and the **tag annotation body becomes the GitHub
release notes** — write it at tag time. HA Supervisor reads `config.yaml` off `main` to offer
users the update, and updating the add-on restarts it; the model-cache `schema_version` gate in
`_restore_cached_models` then decides whether cached models survive or retrain. Bump that schema
version whenever a change alters what a cached model would be fed at inference.

Release commits follow `v<version>: lowercase one-line summary`.

## CHANGELOG conventions (read before writing an entry)

The CHANGELOG is **user-facing release notes for HA users, not a design record**. House style:

- Sections `### Added` / `### Fixed` / `### Changed` only, in that order.
- Each item opens with a **bold declarative sentence stating the user-visible outcome**, then a
  few short lines: the symptom the user saw → what changed → what they'll notice after updating.
  5–20 lines per item; a whole release should stay well under ~80 lines.
- Deep rationale, post-mortems, measurement methodology, and review findings go in
  `docs/investigations/*.md`, linked from the CHANGELOG item, never inlined into it.
- Non-marketing, past-tense-diagnostic tone; name the exact wrong behaviour before the fix.
- The duplicate `## 2.49.2` heading is a known pre-existing quirk — leave it.

## Architecture

### Core modules (`ml_forecast_lab/`)

| Module | Role |
|--------|------|
| `main.py` | `MLForecastLabApp` — all orchestration: fetch, benchmark, retrain, forecast ticks, publishing (~9k lines; grep it, don't read it) |
| `preprocessing.py` | Resampling to the grid, cumulative→interval, outlier clip, load-subtract, **`resolve_missingness`** (the single row-selection point) |
| `features.py` | `build_features` (lags/rolling/temporal), sliding-window builders, warm-up arithmetic, neural channel selection |
| `covariates.py` | `CovariateResolver` — covariate history fetch/cache and future-forecast resolution |
| `benchmark/runner.py` | `BenchmarkRunner` — CV folds (positional indices), per-fold training/eval for all backends |
| `models/` | Registry of ~29 backends behind one interface; `is_neural` splits tabular vs sequence handling; optional deps degrade gracefully (chronos/ttm absent on armv7) |
| `db.py` | `HistoryDB` (SQLite): history cache, forecast log, accuracy/coverage analytics. Tables are keyed by **entity**, so experiments sharing a sensor share a table; retention is the max across them |
| `web/app.py` | FastAPI UI + API (what the smoke tests boot) |
| `ha_interface.py` | HA REST history/state access, `normalise_history` |
| `solar_physics.py` | Deterministic sun-elevation / clear-sky-GHI features via pvlib |
| `config.py` | `ExperimentCfg` / `CovariateCfg` dataclasses and YAML load/validation |
| `debug_dump.py` | The per-retrain training/forecast bundle dumper described above |
| `dev_branch.py` | Maintainer-only overlay for running a git branch inside the add-on — not a user feature |

### The data pipeline (order is load-bearing)

1. `_fetch_and_preprocess` — HA history + SQLite cache → resample to a **complete time grid
   with NaN preserved**. It must never drop rows: lags and rolling windows are positional
   (`target.shift(k)`), so a punctured index silently redefines every lag.
2. `build_features` over that unbroken grid.
3. `_supervised_frame` → `preprocessing.resolve_missingness` — the single row-selection point.
   The role rule: a missing **label** excludes the row and is **never imputed**; a missing
   **feature** is masked, causally imputed (expanding median over strictly prior observations;
   hold-forward for binary), and flagged with a `<name>_missing` indicator. Target-derived
   features share one aggregate `y_missing`. It also returns a **window frame** (imputed
   complete grid + per-row label mask) that all sequence-model sliding windows are built from.
4. Six consumers of that frame: benchmark, production inference, retrain-and-cache, the cached
   forecast tick, hyperparameter tuning, and covariate analysis. Tree backends get engineered
   features + a recursive multi-step forecast; neural backends get sliding windows with dense
   multi-horizon heads ("extended window" = past block + future-known-features block).

### Invariants that tests pin (don't relax them; update tests deliberately if a spec changes)

- **Gap-free experiments are bit-identical across releases**: same feature matrix, row count,
  and sliding windows. Anything data-dependent (indicator columns, window drops) must be a
  no-op when the data has no gaps.
- Labels are never imputed anywhere — training, scoring, conformal calibration, or window
  targets (windows whose horizon labels are unmeasured are dropped, not filled).
- The inference-time indicator set is **pinned from the model cache** (`missing_indicators`),
  never re-derived from the `_missing` name suffix — a real sensor can legitimately be called
  `*_missing`.
- `np.nan_to_num` at feature-matrix boundaries goes through `_nan_to_num_guarded` only; a bare
  call is a silent bug-absorber (that's how the v2.27.10 regression survived a release).
- Coverage/diagnostic thresholds are fixed constants, not config knobs.

## Testing conventions

- New behaviour gets a dedicated `tests/unit/test_<feature>.py` whose module docstring names the
  version and the defect it pins.
- The house harness for pipeline tests (`_StubHA` / `_make_app` / `_recorder_rows`) lives in
  `tests/unit/test_covariate_coverage.py` and `test_missingness_masking.py` — copy it, don't
  reinvent. Log assertions use `caplog.at_level(..., logger="ml_forecast_lab.main")`.
- Source-contract tests (asserting a code snippet exists in `main.py`) are used deliberately to
  pin wiring that unit tests can't reach; expect to update them when refactoring those lines.
- Comments in this codebase record constraints and version markers (`v2.xx.x:` / audit refs),
  not narration — keep that idiom.

## Documentation audiences

`ml-forecast-lab/README.md` and `DOCS.md` render inside HA's app UI (Info/Documentation tabs) —
write them for HA users. `docs/` (MODEL_GUIDE, RANKING_NOTES, investigations/) is the
maintainer/design layer. Match depth to shelf.
