# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A Home Assistant add-on repository (the root `repository.yaml` makes it installable in HA). The
actual application lives entirely under `ml-forecast-lab/` — a single HA add-on that benchmarks
~29 time-series model backends on any HA sensor's history, promotes a winner, retrains it on
schedule, and publishes calibrated forecasts back to HA. Target hardware is a Raspberry Pi 5
(8 GB, no GPU), which drives most performance decisions. All development happens in
`ml-forecast-lab/`; the root holds only the add-on-store metadata and top-level README.

## Commands

Run everything from `ml-forecast-lab/` (there is no pytest config file; the rootdir is inferred
from the working directory, and test imports resolve against the package there).

```bash
cd ml-forecast-lab
pip install -r requirements.txt -r tests/requirements-dev.txt   # full deps (torch is large)

python -m pytest tests/unit/ -v --tb=short          # unit suite (what CI runs; ~30 s)
python -m pytest tests/smoke/ -v --tb=short         # FastAPI UI/API gate (~10 s, no ML deps needed)
python -m pytest tests/integration/ -v --tb=short   # real train→predict→publish path (~3–4 min)

# single test
python -m pytest tests/unit/test_missingness_masking.py::TestOrdering -v
```

CI (`.github/workflows/tests.yml`) runs those three suites as separate jobs; the smoke job
installs only the web stack, so smoke tests must not import ML backends. `pandas` is pinned
`<3.0` and `numpy` `<2.0` — match those locally or pandas-3 behaviour differences will bite.
There is no linter or formatter configured; match the surrounding style.

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
  `docs/investigations/*.md` (existing pattern — see `2026-05-neural-pv.md`), linked from the
  CHANGELOG item, never inlined into it.
- Non-marketing, past-tense-diagnostic tone; name the exact wrong behaviour before the fix.
- The duplicate `## 2.49.2` heading is a known pre-existing quirk — leave it.

## Architecture

`ml_forecast_lab/main.py` (~9k lines) holds `MLForecastLabApp` and the whole orchestration; grep
it rather than reading it. The data pipeline order is load-bearing:

1. `_fetch_and_preprocess` — HA history + SQLite cache (`db.py`, tables keyed by **entity**, so
   experiments sharing a sensor share a table and retention is the max across them) → resample to
   a **complete time grid with NaN preserved**. It must never drop rows: lags and rolling windows
   are positional (`target.shift(k)`), so a punctured index silently redefines every lag.
2. `build_features` (`features.py`) over that unbroken grid.
3. `_supervised_frame` → `preprocessing.resolve_missingness` — the single row-selection point.
   The role rule: a missing **label** excludes the row and is **never imputed**; a missing
   **feature** is masked, causally imputed (expanding median over strictly prior observations;
   hold-forward for binary), and flagged with a `<name>_missing` indicator. Target-derived
   features share one aggregate `y_missing`. It also returns a **window frame** (imputed complete
   grid + per-row label mask) that all sequence-model sliding windows are built from.
4. Six consumers of that frame: benchmark (`benchmark/runner.py` — fold indices are positional
   into the supervised frame; windows are sliced from the window frame by timestamp), production
   inference, retrain-and-cache, the cached forecast tick, hyperparameter tuning, and covariate
   analysis.

`models/` is a registry of ~29 backends behind one interface; `is_neural` decides tabular
(engineered features, recursive multi-step forecast) vs sequence (sliding windows, dense
multi-horizon heads, "extended window" = past block + future-known-features block). Backends
degrade gracefully when optional deps are missing (e.g. chronos/ttm are excluded on armv7).
`covariates.py` (`CovariateResolver`) fetches/caches covariate history and future forecasts.
`web/app.py` is the FastAPI UI the smoke tests boot. `dev_branch.py` is a maintainer-only
overlay for running a git branch inside the add-on — not a user feature.

### Invariants that tests pin (don't relax them; update tests deliberately if a spec changes)

- **Gap-free experiments are bit-identical across releases**: same feature matrix, row count,
  and sliding windows. Anything data-dependent (indicator columns, window drops) must be a no-op
  when the data has no gaps.
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

`ml-forecast-lab/README.md` and `DOCS.md` render inside HA's add-on UI (Info/Documentation
tabs) — write them for HA users. `docs/` (MODEL_GUIDE, RANKING_NOTES, investigations/) is the
maintainer/design layer. Match depth to shelf.
