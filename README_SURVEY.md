# README_SURVEY.md

Survey of the repository state, produced before drafting the public GitHub README.

## One-sentence purpose

A Home Assistant add-on that benchmarks 24 forecasting model backends on a user-selected HA sensor, promotes the winner, retrains it on schedule, and publishes the forecast (with calibrated 80% prediction bands) back to Home Assistant as companion sensor entities.

## Repository tree

```
ml-forecast-lab/                            (repo root — HA add-on repository)
├── README.md                               existing public README (already substantial — see "Existing READMEs")
├── SECURITY.md                             vulnerability reporting policy (private disclosure via GitHub Security)
├── LICENSE                                 MIT, © 2026 Paul Sweeney
├── repository.yaml                         HA add-on repository manifest
├── logo.png                                2127 × 2127 PNG (root copy — duplicate of ml-forecast-lab/logo.png)
├── icon.png                                2127 × 2127 PNG (root copy — duplicate of ml-forecast-lab/icon.png)
├── docs/
│   └── MODEL_GUIDE.md                      114 lines — "which of the 24 backends should I enable?" decision flow
├── .github/workflows/
│   ├── tests.yml                           smoke + unit pytest gates (PR + main)
│   ├── validate.yml                        version consistency + syntax check
│   └── release.yml                         tag-driven GitHub Release publish
└── ml-forecast-lab/                        (the add-on itself)
    ├── README.md                           HA store **Info** tab
    ├── DOCS.md                             HA store **Documentation** tab — full config reference
    ├── CHANGELOG.md                        4 901 lines, 175 versions, latest 2.34.6
    ├── config.yaml                         HA add-on manifest (name, version, ingress, arch list)
    ├── build.yaml                          base image per arch (hassio-addons/ubuntu-base:9.0.5)
    ├── Dockerfile                          two-stage Python venv → ubuntu-base runtime
    ├── requirements.txt                    20 pinned dependencies (numpy, pandas, torch, lightgbm, xgboost, catboost, statsforecast, fastapi, plotly, optuna, pvlib, …)
    ├── mlfl.yaml                           bundled example config (generic household-load template)
    ├── logo.png / icon.png                 duplicates of the root assets
    ├── translations/en.yaml                HA add-on options translation (just log_level)
    ├── rootfs/etc/s6-overlay/…/run         s6-overlay init script — finds /addon_configs path, launches app
    ├── ml_forecast_lab/                    Python package
    │   ├── __init__.py                     __version__ = "2.34.6"; re-exports for legacy callers
    │   ├── __main__.py                     entrypoint
    │   ├── main.py                         6 150 lines — orchestration / lab + production cycles
    │   ├── config.py                       1 332 lines — load_config, dataclasses for every YAML field
    │   ├── ha_interface.py                 HA REST/WS client; history fetch, sensor publish
    │   ├── db.py                           SQLite cache (actuals, forecast log, retrain rollbacks)
    │   ├── preprocessing.py                cumulative→interval, gap fill, outlier clip, load-subtract
    │   ├── features.py                     565 lines — lags, temporal encodings, rolling stats
    │   ├── covariates.py                   resolves lagged + future-attribute covariate signals
    │   ├── solar_physics.py                sun-elevation + clear-sky GHI (pvlib Ineichen)
    │   ├── training_events.py              event-log helpers
    │   ├── benchmark/                      cross-validation, ranking, pairwise comparison
    │   │   ├── runner.py                   963 lines — walk-forward / sliding-window CV, Demšar composite rank
    │   │   ├── metrics.py                  MAE / RMSE / MAPE / SMAPE / MASE / seasonal-MASE / pinball / coverage
    │   │   └── comparison.py               paired-t pairwise comparison
    │   ├── models/                         24 backend adapters (see "Model backends" below) + registry
    │   └── web/                            FastAPI + Jinja2 + HTMX web UI
    │       ├── app.py                      3 679 lines — every page, endpoint, HTMX partial
    │       ├── templates/*.html            8 templates (base, dashboard, experiment, models, training, logs, system, plus 2 partials)
    │       └── static/
    │           ├── style.css               1 048 lines — dark theme, navy + cyan + magenta palette
    │           ├── htmx.min.js             vendored
    │           ├── plotly-basic.min.js     vendored
    │           └── icon.png                same 2127 × 2127 brand mark
    └── tests/
        ├── conftest.py                     pytest config
        ├── dryrun_pipeline.py              ad-hoc dry-run harness
        ├── requirements-dev.txt
        ├── smoke/                          61 tests — boots FastAPI app against tmp mlfl.yaml; release gate
        │   └── test_*.py                   8 modules covering empty state, experiment lifecycle, HA entities, harness, model config, pages, promote flow, settings, tuning guard
        └── unit/                           124 tests on isolated modules
            └── test_*.py                   benchmark / config / db / features / forecast analytics / load-subtract / models / preprocessing
```

Total tests reported in the existing public README: 185 (61 smoke + 124 unit).

## Model backends (24, from the registry)

Tree: `lightgbm`, `xgboost`, `catboost` · Recurrent: `lstm`, `gru` · Convolutional: `cnn`, `timesnet` · Linear / MLP: `dlinear`, `nlinear`, `tsmixer`, `timemixer`, `tide`, `sparsetsf` · Frequency-domain: `fits` · N-BEATS family: `nbeats`, `nhits` · Transformers: `patchtst`, `itransformer`, `crossformer`, `tft` · Classical (statsforecast): `arima`, `ets`, `theta` · Baseline: `seasonal_naive`.

## Branding assets

| Asset | Path(s) | Size | Format | Notes |
|---|---|---|---|---|
| Logo / icon (same image) | `logo.png`, `icon.png`, `ml-forecast-lab/logo.png`, `ml-forecast-lab/icon.png`, `ml-forecast-lab/ml_forecast_lab/web/static/icon.png` | 2127 × 2127 px | PNG, RGBA | Conical flask outline with an upward-trending line/arrow rising out of it. Logo and icon are identical — there is no separate wordmark, banner, or screenshot in the repo. |
| Hero / banner | — | — | — | None. |
| Screenshots | — | — | — | None. The per-add-on README has commented-out placeholders (`<!-- ![Dashboard](images/dashboard.png) -->`) anticipating an `ml-forecast-lab/images/` directory that has not been created. |

## Colour palette (from `ml_forecast_lab/web/static/style.css`)

| Role | Hex | Note |
|---|---|---|
| `--bg-primary` | `#1a1a2e` | Deep navy — page background |
| `--bg-secondary` | `#16213e` | Surface / panel |
| `--bg-tertiary` | `#0f3460` | Borders / accents on dark |
| `--accent-primary` | `#e94560` | Pink/magenta — headings, primary accents, matches the arrow in the logo |
| `--accent-secondary` | `#00d4ff` | Cyan — links, active nav, brand text, matches the flask in the logo |
| `--text-primary` | `#e0e0e0` | Body text |
| `--text-secondary` | `#b0b0b0` | Muted text |
| `--border` | `#2d3561` | Hairlines |
| `--success` | `#2ecc71` | Green |
| `--warning` | `#f39c12` | Amber |
| `--error` | `#e74c3c` | Red |

The README badge URLs hard-code `#41bdf5` (a Home Assistant brand blue) rather than the in-app palette — that is a deliberate alignment with the HA add-on store look.

## Existing READMEs and marketing copy

| File | Length | Audience | Status |
|---|---|---|---|
| `README.md` (repo root) | 146 lines | GitHub readers, mixed evaluators / installers | Already well-written. Has a logo + tagline header, 5 badges, "What this repository is" framing, install button, an ASCII architecture diagram inside a `<details>`, a docs-routing table, a Development section, MIT licence line. Contains hard-coded badge URLs with a comment explaining they were frozen while the repo was private. |
| `ml-forecast-lab/README.md` | 125 lines | HA store **Info** tab | Install + first-experiment walkthrough. |
| `ml-forecast-lab/DOCS.md` | 338 lines | HA store **Documentation** tab | Full configuration reference, web-UI tour, ops, troubleshooting. |
| `ml-forecast-lab/CHANGELOG.md` | 4 901 lines | HA store **Changelog** tab | 175 versions of release notes. |
| `docs/MODEL_GUIDE.md` | 114 lines | Users picking model backends | Decision tree by data shape + target characteristics. |
| `SECURITY.md` | 51 lines | Reporters | Private disclosure via GitHub Security advisories. |
| `repository.yaml` description | one line | HA store discovery | "Multi-model machine learning forecasting and benchmarking for Home Assistant" |
| `config.yaml` description | one line | HA add-on metadata | "Multi-model ML forecasting and benchmarking for Home Assistant" |

### Distilled taglines already used

- **Repo README header tagline:** "Multi-model machine learning forecasting for Home Assistant."
- **Repo README subhead:** "Train, benchmark, and deploy time-series models for any HA sensor — with academic-standard evaluation built in."
- **Add-on store description:** "Multi-model ML forecasting and benchmarking for Home Assistant"
- **Mindset framing the project uses:** "benchmark once, run forever."

## Actual feature surface (what a user can do today)

Grounded in the existing READMEs, DOCS.md, and source structure — no extrapolation:

1. Install via the HA add-on store, by adding the repository URL.
2. Configure one or more experiments either via the web UI (no YAML editing required) or by hand-editing `mlfl.yaml` — both paths are kept consistent (UI rewrites the YAML atomically).
3. Forecast any HA recorder-backed sensor. The target sensor can be cumulative (with explicit `source_is_cumulative` + `reset_daily` handling), and supports daily-reset energy semantics, log transform, outlier clipping, gap interpolation, and load-subtract (subtract one or more sensors from the target before training).
4. Add covariates — either **lagged** (historical only) or **future** (read forecast attributes from weather / Solcast / forecast.solar entities). Supports per-covariate scale, transform (`log` / `sqrt` / `box_cox`), aggregation, and binary indicator handling.
5. Two solar-physics covariates (sun elevation, clear-sky GHI via pvlib Ineichen) usable without any external API.
6. Benchmark every enabled backend with walk-forward or sliding-window cross-validation (default 5 folds, configurable embargo).
7. Composite Demšar rank across MAE / RMSE / MASE; per-fold pairwise comparison via paired-t test; PSI-based train/test drift verdict; always-on "vs Seasonal Naive" skill chip.
8. Bayesian hyperparameter tuning per model via Optuna TPE; one-click "Tune All Enabled" sweep.
9. **Promote winner to production** — flips mode, retrains on schedule (default 24 h), publishes forecast sensors every 30 min (configurable per experiment).
10. Published companion sensors per experiment: `_forecast`, `_interval` (cumulative sources only), `_cumulative`, `_upper_<pct>`, `_lower_<pct>`, `_forecast_accuracy`, `_last_benchmark`, `_last_retrain`. The interval forecast carries the full future curve as a `[{datetime, value}, …]` attribute.
11. Calibrated conformal prediction bands (default 80%; configurable). Bands appear after ~10 deployed predictions vs actuals have accumulated.
12. **Forecast Accuracy tab** — per-horizon error, bias, conformal coverage, retrain-history chip strip, "Compare with previous run" diff across the last five benchmarks.
13. **Covariate Analysis** — automatic search across covariate combinations to identify which signals genuinely help.
14. Roll back one generation of the production model atomically; bounded to single-generation to limit SD-card writes.
15. CPU-core cap (`OMP_NUM_THREADS` / `torch.set_num_threads`) and process `nice` priority — actually applied, configurable from the System page.
16. Logs categorised by 13 phase tags (`[APP]`, `[CFG]`, `[HA]`, `[DB]`, `[PREP]`, `[FEAT]`, `[COV]`, `[SOLAR]`, `[MODEL]`, `[BENCH]`, `[TRAIN]`, `[PUB]`, `[WEB]`) for grep-friendly triage. Rotating file log at `/data/ml_forecast_lab/logs/mlfl.log` (5 MB × 5).

## Stack

- **Language:** Python 3.11 (CI), 3.10 syntax-check.
- **Web framework:** FastAPI + Uvicorn, Jinja2 templates, HTMX for partials, Plotly (basic build) for charts.
- **ML:** PyTorch ≥ 2.0 (neural backends), LightGBM ≥ 4, XGBoost ≥ 2, CatBoost ≥ 1.2, statsforecast ≥ 1.7 (classical), scikit-learn ≥ 1.3, scipy ≥ 1.10, Optuna ≥ 3.5 (tuning), pvlib ≥ 0.10 (solar physics), holidays ≥ 0.40.
- **Data:** numpy < 2, pandas ≥ 2, SQLite via stdlib `sqlite3` (no ORM).
- **Storage:** SQLite database under `/data/ml_forecast_lab/` for actuals cache, forecast log, and per-experiment model artefacts.
- **Container base:** `ghcr.io/hassio-addons/ubuntu-base:9.0.5` per architecture; two-stage Dockerfile with a builder venv compiling native extensions on `aarch64` / `armv7`.
- **Process supervision:** s6-overlay (the HA add-on standard).
- **Sandbox eval:** `asteval` for user-supplied `custom_metrics` expressions.

## Distribution

- Public GitHub repository: `https://github.com/psweens/ml-forecast-lab`.
- Installed by adding `https://github.com/psweens/ml-forecast-lab` as a repository in the Home Assistant add-on store (Settings → Add-ons → Add-on store → ⋮ → Repositories), then installing the **ML Forecast Lab** entry that appears.
- The existing root README also exposes a "My Home Assistant" one-click install button (`my.home-assistant.io/redirect/supervisor_add_addon_repository`).
- First build on a Pi 5 takes **10–15 minutes** (compiles LightGBM / XGBoost / PyTorch native extensions); subsequent updates use the cached image.

## Contribution surface (realistic)

There is no `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `FUNDING.yml`, or `.github/ISSUE_TEMPLATE/`. The existing per-add-on README's Support section frames the project as:

- "**First public release of a project that was developed in a private repository.** You are amongst the first external users — please open issues for anything that surprises you, including documentation gaps."
- "Maintained on a best-effort basis as a side project."

Realistic contribution shape, inferred from the repository:

- **Bug reports** are the most actionable. The README explicitly asks for add-on version + relevant `mlfl.yaml` section + last 50 log lines.
- **Documentation gaps** are called out as first-class issues.
- **Tested configurations** for specific HA setups (sensor types, regimes, solar setups) would be useful — the project's value depends on knowing which model wins on which kind of data, and the only person who has run it long-term is the author.
- **Model backend tuning** (better default hyperparams for the 24 backends on Pi-scale data) is high-value contribution surface because the bench harness is already built.
- **PRs to core orchestration** (`main.py`, 6 150 lines) are higher-risk and would need scope agreement first.
- **Security issues** route through GitHub's private vulnerability reporting (`SECURITY.md`).
- **No translations contribution path today** — `translations/en.yaml` only carries the HA add-on options strings (one field: `log_level`); the web UI itself is hard-coded English.

## Licence, version, supported HA / architectures

- **Licence:** MIT (`LICENSE`), © 2026 Paul Sweeney. Repository-root attribution names the author as Dr Paul W. Sweeney, University of Cambridge.
- **Current version:** `2.34.6` (`ml-forecast-lab/config.yaml`, `ml_forecast_lab/__init__.py`; the Validate workflow enforces these match).
- **Stage:** `stable` (declared in `config.yaml`).
- **Stability framing in docs:** "first public release of a project that was developed in a private repository" — so effectively stable codebase, fresh-to-public status.
- **Supported HA architectures:** `aarch64`, `amd64`, `armv7`.
- **Supported HA versions:** not pinned in `config.yaml` (no `homeassistant: ">= x.y.z"` floor); requires HA Supervisor (uses ingress + `homeassistant_api`).
- **Hardware target:** Raspberry Pi 5 with 8 GB RAM is the explicit design point; works on amd64 / armv7 too.
- **No GPU support today.** Hailo NPU is explicitly called out as not wired in.

## Existing visual style choices the README should echo

- **Palette:** the in-UI dark navy (`#1a1a2e`) + cyan (`#00d4ff`) + magenta (`#e94560`) is the canonical app palette and matches the logo. The README's existing badges use HA's brand blue `#41bdf5` instead — fine as-is; mixing the two is the current author choice and shouldn't be "corrected".
- **Tone in existing prose:** dry, specific, slightly British, occasionally wry. "Benchmark once, run forever." "If your fancy transformer can't beat naive, the issue isn't the model — it's the data." "Maintained on a best-effort basis as a side project." Avoids hype words. Heavy use of tables for reference material.
- **Existing README format:** logo + tagline + subhead in a centred div; badge row underneath; horizontal rule; H2 sections; ASCII architecture diagram inside `<details>`. The author already favours specifics over rhetoric (every metric named, every backend listed).
- **No emojis used anywhere in the existing docs.** Continue this.

## Assets I needed but didn't find

These would strengthen the README but are not in the repo:

1. **A dashboard screenshot.** The single most useful asset for the 60% of readers who are evaluating "what does this actually give me?". Proposed: a capture of the experiment dashboard with one experiment promoted to production, the rank table visible, ideally with a chart in view. ~1600 px wide, PNG, dark UI (already the app theme).
2. **A Forecast Accuracy tab screenshot.** Demonstrates the diagnostic depth that distinguishes this from simpler forecasting add-ons. Same format.
3. **(Optional) A short GIF or MP4** of the "Run Pipeline → Promote → published sensor appears in HA" loop. High evaluator-conversion value; not essential.
4. **A horizontal banner / OG-image variant of the logo.** The current asset is a 1:1 circular mark; a 1280 × 640 PNG with the mark + wordmark in the app palette would improve social-card embeds. Not blocking — the current README opens with the 1:1 logo at 180 px and that reads fine.

None of these should be invented as ASCII / Mermaid placeholders — README_DESIGN.md will recommend including them if/when produced, and structuring the README so it still reads sensibly without them.
