# DOC_SURVEY.md — ML Forecast Lab documentation inventory (Phase 1)

Scope: every file that a Home Assistant power user might land on when discovering, installing, configuring, or running this add-on. British English. File paths absolute from the repository root.

This is the **survey** — no judgement, no recommendations. Phase 2 (`DOC_AUDIT.md`) will grade everything against the install/configure/use journey.

---

## 1. The full inventory

### 1.1 User-facing documentation

| Path | Lines | Last commit | Role |
|---|---:|---|---|
| `README.md` | 281 | 2026-05-07 | Repository-root README. Renders on the GitHub project page. |
| `docs/MODEL_GUIDE.md` | 114 | 2026-05-07 | "Which of the 24 backends should I enable?" decision guide. |
| `docs/CONFIG_GUIDE.md` | 306 | 2026-04-20 | "Configuration module" reference — written as a Python API guide. |
| `docs/PREPROCESSING_GUIDE.md` | 429 | 2026-04-20 | "Preprocessing module" reference — written as a Python API guide. |
| `docs/FEATURES_GUIDE.md` | 479 | 2026-04-20 | "Features module" reference — written as a Python API guide. |
| `ml-forecast-lab/CHANGELOG.md` | 4 268 | 2026-05-13 | Per-release notes. v0.1.0 → v2.31.0. |
| `ml-forecast-lab/mlfl.yaml` | 166 | 2026-04-20 | Example user config, heavily commented. |
| `LICENSE` | 21 | 2026-05-11 | MIT, © Dr Paul W. Sweeney. |

### 1.2 Add-on store metadata

| Path | Role |
|---|---|
| `repository.yaml` | Multi-add-on repository metadata (name, url, maintainer, description). |
| `ml-forecast-lab/config.yaml` | Add-on manifest. Sets version, slug, ingress, arch list, schema, panel title/icon. **Current version: 2.32.0.** |
| `ml-forecast-lab/build.yaml` | Base-image pins per architecture. |
| `ml-forecast-lab/translations/en.yaml` | 8-line translation file. Covers `log_level` and a stub `models` section. |
| `ml-forecast-lab/Dockerfile` | Build instructions (not user docs but referenced by install-time behaviour). |

### 1.3 Images

| Path | Dimensions | Purpose by convention | Actual use |
|---|---:|---|---|
| `icon.png` (root) | 2127×2127 | HA add-on icon should be 128×128 PNG. | Repo-root copy; embedded in README at `width=180`. |
| `logo.png` (root) | 2127×2127 | HA add-on logo banner, conventionally ~250×100 PNG. | Repo-root copy. Byte-identical to `icon.png`. |
| `ml-forecast-lab/icon.png` | 2127×2127 | The icon the supervisor actually uses when rendering the add-on tile. | Byte-identical to root copy. |
| `ml-forecast-lab/logo.png` | 2127×2127 | The logo the supervisor renders above the Info tab. | Byte-identical to root copy. |
| `ml-forecast-lab/ml_forecast_lab/web/static/icon.png` | 2127×2127 | Web-UI favicon / brand mark. | Byte-identical to the others. |

No screenshot files exist anywhere in the repository. The README contains three `<!-- TODO: screenshot -->` placeholders (lines 52–54).

### 1.4 Internal / planning docs at the repository root

These exist in the repository root but are not user documentation — they are audit and proposal artefacts from earlier development phases. A casual visitor to the GitHub page will see them in the file listing.

| Path | Lines | Role |
|---|---:|---|
| `AUDIT_PROMPT.md` | 583 | The prompt for an earlier latent-defect audit. |
| `SURVEY.md` | 901 | Phase 1 ML methodology recon (preprocessing / loss). |
| `ML_AUDIT.md` | 1 069 | Phase 2 ML methodology findings. |
| `IMPROVEMENTS.md` | 692 | Product / UX wave proposal (shipped as v2.31.0). |
| `ml-forecast-lab/MODELS_CREATED.txt` | 328 | Stale "creation summary" for the LightGBM/XGBoost backends, dated `2026-03-29`. References paths from a different developer environment (`/sessions/serene-determined-ride/...`). |

### 1.5 What is conspicuously absent

| Expected | Status |
|---|---|
| `ml-forecast-lab/README.md` — the file the HA supervisor renders on the **Info** tab of the add-on store panel. | **Missing.** |
| `ml-forecast-lab/DOCS.md` — the file the supervisor renders on the **Documentation** tab. | **Missing.** |
| Any screenshot of the dashboard, experiment detail, or forecast accuracy view. | **Missing.** Three TODOs in `README.md`. |
| `docs/SCREENSHOTS/` or similar. | **Missing.** |
| Contributing guide (`CONTRIBUTING.md`). | **Missing.** |
| Issue / PR templates under `.github/`. | **Missing.** |
| Security policy (`SECURITY.md`). | **Missing.** |

---

## 2. Section-by-section content of each user-facing file

### 2.1 `README.md` (repository root)

Heading hierarchy:

| Level | Heading | What it covers |
|---|---|---|
| H1 | ML Forecast Lab | Centred title block. 5 badges (release, licence, tests, HA add-on, arch). Tag-line, one-paragraph "what it is" sentence, nav links. |
| H2 | About | Two paragraphs. The "trains every enabled backend, ranks them on identical folds, you promote the winner" workflow. Coins the term "benchmark once, run forever". |
| H2 | Installation | Two sub-paths. "Open in HA" deep-link badge; "Manual" 4-step instructions for adding the repository URL. Note about 10–15 min first build on Pi 5. Architecture support line. |
| H2 | Screenshots | Three `<!-- TODO: -->` placeholder comments. **No image actually rendered.** |
| H2 | Quick start | Six numbered steps from "create `mlfl.yaml`" to "watch it" via the Forecast Accuracy tab. References `homeassistant.local:5052`. |
| H2 | Features | Two-column HTML table — 24 backends, evaluation, tuning, decoupled retrain/forecast cycles, conformal bands, accuracy tracking, load subtract, covariate analysis. Two `<details>` blocks for "Auto-generated features and built-in physics" and "Architecture" (ASCII diagram). |
| H2 | Configuration | Minimal `mlfl.yaml` example (`mixergy_demand` experiment). 12-row option table (`target_entity` → `cv_folds`). Defers full schema to `docs/CONFIG_GUIDE.md`. |
| H2 | Documentation | Four bulleted links into `docs/`. |
| H2 | Troubleshooting | Five `<details>` accordions: "not enough data", "sensors don't appear", "forecasts collapse to zero overnight", "Database not available", "ARM build slow". Closing paragraph names the log tags `[BENCH] [MODEL] [WEB] [HA] [PREP]`. |
| H2 | Development | Three `pytest` commands. Names 185 tests, 61 smoke tests. References `tests.yml`. |
| H2 | Licence | One-line MIT + author. |

Tone: informed enthusiast. Uses some ML-shorthand without unpacking it for non-ML readers ("composite Demšar score", "walk-forward CV with embargo gaps", "conformal bands", "pinball loss", "Diebold-Mariano"). Assumes a confident reader.

### 2.2 `docs/MODEL_GUIDE.md`

Heading hierarchy: H1 → 6 × H2.

- **The 24 backends at a glance** — 24-row table (family / backend name / strength / weakness / speed).
- **Decision flow** — start with `seasonal_naive`; then branch by data volume (<2 weeks → trees only; 2 weeks – 2 months → add LSTM, CNN, DLinear, NLinear; 2–6 months → add NHiTS, PatchTST, TiDE, TSMixer; >6 months → TFT, Crossformer, TimeMixer). Then branch by target characteristics (strong seasonality, noisy/sparse, covariate-driven, univariate, solar).
- **Speed tradeoffs on a Pi 5** — Fast / Medium / Slow / Very slow tiers, with per-fold times. Total benchmark time estimate.
- **Pragmatic starter sets** — three YAML snippets (minimum viable, decent data, classical / univariate).
- **After the benchmark** — Demšar rank reading, naive-baseline interpretation.
- **Tuning** — one paragraph on the Tuning tab and the "Apply Tuned Params, Promote & Retrain" button.

Tone: practical, user-facing, mostly free of unexplained jargon. The most HA-user-friendly file in the doc set.

### 2.3 `docs/CONFIG_GUIDE.md`

Heading hierarchy: H1 → "Overview" / "Core Classes" / "YAML Configuration" / "Validation" / "Best Practices" / "Error Handling".

- **Overview** — frames the file as "The `config` module provides dataclasses and YAML loading for ML Forecast Lab experiment configuration". The first sentence is about Python classes, not about a YAML file.
- **Core Classes** — `CovariateCfg`, `ExperimentCfg`, `AppConfig`. Each documented as a Python dataclass with `from ml_forecast_lab.config import ...` examples and Python instantiation snippets.
- **YAML Configuration** — a single 60-line YAML example with comments.
- **Validation** — Python-level `try / except ValueError` patterns.
- **Best Practices** — six bullets, mixes user-facing advice ("use sensible defaults", "set embargo_periods") with Python-developer advice ("document custom metrics").
- **Error Handling** — Python `try / except` snippets again.

Tone: written for a Python developer who has imported the package, not for an HA user who is editing `mlfl.yaml` through the HA File editor / Studio Code Server.

### 2.4 `docs/PREPROCESSING_GUIDE.md`

Heading hierarchy: H1 → "Overview" / "Key Principle" / "Core Functions" / "Usage Patterns" / "Error Handling" / "Logging" / "Performance Notes".

Same shape as `CONFIG_GUIDE.md`: every section is a Python-function reference with the function signature, parameter docs, and Python code examples. Functions documented: `cumulative_to_interval`, `resample_to_grid`, `clip_outliers`, `apply_transform`, `apply_log_transform`, `subtract_series`, `power_to_energy`, `align_series`.

Tone: written for a developer importing `ml_forecast_lab.preprocessing`. No mention of HA entity types, the Settings tab UI, or how a user would *trigger* these via `mlfl.yaml`.

### 2.5 `docs/FEATURES_GUIDE.md`

H1 → "Overview" / "Core Functions" / "Usage Patterns" / "Best Practices" / "Performance Notes".

Opens with "The `features` module provides unified temporal feature engineering shared across all **14 model backends** (tree and neural)" — quoting that figure verbatim.

Functions documented: `build_features`, `prepare_train_test`, `reshape_for_sequence`, `create_forecast_features`, `is_holiday`. End-to-end usage patterns include `from sklearn.preprocessing import StandardScaler` and **TensorFlow / Keras** code (`from tensorflow.keras.models import Sequential`, `LSTM`, `Conv1D`).

Tone: developer reference. Almost no overlap with what an HA user editing `mlfl.yaml` actually needs.

### 2.6 `ml-forecast-lab/mlfl.yaml`

A heavily commented example config file. Effectively doubles as documentation by virtue of inline comments. Structure:

- Top banner: explains lab vs production mode, typical workflow.
- `timezone`, `update_every_minutes` globals.
- One concrete experiment (`mixergy_demand`) fully filled in with every field annotated.
- Source characteristics, training window, forecast window.
- Covariates with three roles illustrated.
- 21 lines of `models_enabled` with 14 model names listed (4 active, 10 commented out with paper citations).
- Neural / CV / metrics / custom metrics / production / publishing sections.
- A second experiment (`mixergy_heat_energy`) entirely commented out as a "add a second experiment" template.

Header says *"Place this file at /config/mlfl.yaml on your HA instance."*

### 2.7 `ml-forecast-lab/CHANGELOG.md`

4 268 lines, 165 versioned sections (`## 2.31.0` … `## 0.1.0`). Standard Keep-a-Changelog-ish layout per version: subsections like Functionality / Fixed / Internal / Security / Frontend / etc. Newest entry is `## 2.31.0`. The current `config.yaml` version is `2.32.0` — i.e. there is no `2.32.0` entry yet.

### 2.8 `ml-forecast-lab/MODELS_CREATED.txt`

Plain-text internal artefact dated `2026-03-29`. Titled "ML FORECAST LAB - LIGHTGBM & XGBOOST MODEL BACKENDS". Reads like a developer hand-off note: line counts of two backend files, hyperparameter lists, "production readiness checklist", "future enhancements". The "Files Location" section at the bottom contains hard-coded sandbox paths (`/sessions/serene-determined-ride/mnt/PredAI/...`).

### 2.9 `ml-forecast-lab/translations/en.yaml`

8 lines. Contains a `log_level` entry (matching the one option declared in `config.yaml`) and an orphan `models` entry that has no counterpart anywhere in `config.yaml`'s `schema:`.

---

## 3. The install / configure / first-use path a user must follow

Reading the docs as written, the end-to-end journey is:

1. Discover the add-on on GitHub (the project's `README.md` renders there). Read the About → Installation sections.
2. Click the "Open Home Assistant and add this repository" badge **or** follow the four-step manual repository-add.
3. Wait 10–15 minutes for the Pi 5 first build (mentioned in a callout note).
4. Open the add-on page in HA and click Install.
5. Before starting the add-on, create `/addon_configs/ml_forecast_lab/mlfl.yaml` (or `/config/mlfl.yaml` as fallback) — README, Quick-start step 1. The user is told the minimal viable config is `name`, `target_entity`, `models_enabled` and to "see Configuration below".
6. The Configuration section gives a worked `mixergy_demand` example and a 12-row option table, then defers to `docs/CONFIG_GUIDE.md` "Full schema in …".
7. Start the add-on. Web UI is at `http://homeassistant.local:5052` **or** via Open Web UI. (See §4.6 below for why this URL is wrong.)
8. Run the benchmark, pick a model from the Models tab, promote to production. The promoted model retrains every 24 h and publishes sensors every 30 min.
9. Watch progress on the Forecast Accuracy tab.

The shortest happy path that a user can actually follow without leaving the repo's documentation: about 20 minutes of reading + one external dependency (HA recorder retention long enough to give ~30 days of history).

---

## 4. Cross-reference of docs against code

### 4.1 Number of model backends

| Source | Figure quoted |
|---|---|
| Code (`main.py` lines 348–375) | 4 core + 20 optional = **24** backends actually registered. |
| `README.md` | "24 model backends, benchmarked on identical splits." |
| `docs/MODEL_GUIDE.md` | 24-row table, 5 references to "24 backends". |
| `docs/FEATURES_GUIDE.md` line 5 | "all **14** model backends (tree and neural)" — **stale**. |
| `ml-forecast-lab/mlfl.yaml` | 14 backends in the YAML comment block (4 enabled + 10 commented out). |

### 4.2 Configuration options — code vs documentation

#### 4.2.1 `ExperimentCfg` fields actually exposed by the code

Defined in `ml-forecast-lab/ml_forecast_lab/config.py` lines 200–637. Forty-seven fields in total:

`name`, `target_entity`, `covariates`, `days_history`, `interval_minutes`, `source_is_cumulative`, `reset_daily`, `max_increment`, `models_enabled`, `cv_strategy`, `cv_folds`, `cv_embargo_periods`, `metrics`, `custom_metrics`, `production_model`, `selected_model`, `production_metric`, `publish_prefix`, `country`, `units`, `output_units`, `log_transform`, `output_activation`, `use_revin`, `future_covariate_features`, `subtract` (deprecated stub), `load_subtract`, `mode`, `clear_forecast_log_on_retrain`, `stability_focus`, `max_age`, `future_periods`, `publish_name`, `database`, `model_params`, `forecast_every_minutes`, `retrain_every_hours`, `loss_fn`, `optimiser`, `daily_loss_weight`, `recency_half_life_days`, `conformal_coverage`, `quantiles`, `gap_handling`, `gap_max_minutes`, `outlier_method`, `outlier_quantile`, `outlier_lower`, `include_sun_elevation`, `include_clear_sky_irradiance`.

Documented in `docs/CONFIG_GUIDE.md`: `name`, `target_entity`, `covariates`, `days_history`, `interval_minutes`, `horizons_minutes` (**no longer in code — removed via auto-migration**), `source_is_cumulative`, `reset_daily`, `max_increment`, `log_transform`, `subtract` (**deprecated stub — present in code but never wired into preprocessing**), `country`, `units`, `output_units`, `cv_strategy`, `cv_folds`, `cv_embargo_periods`, `models_enabled`, `metrics`, `custom_metrics`, `production_model`, `production_metric`, `publish_prefix`.

Documented in `README.md` table: `target_entity`, `mode`, `source_is_cumulative`, `reset_daily`, `interval_minutes`, `days_history`, `future_periods`, `forecast_every_minutes`, `retrain_every_hours`, `models_enabled`, `cv_strategy`, `cv_folds`.

Documented in `mlfl.yaml` example comments: most of the daily-life knobs including `max_increment`, `future_periods`, `loss_fn`, `production_metric`, `publish_name`, `units`, `output_units`, `country`, `database`.

#### 4.2.2 Configuration fields completely undocumented

Present in the code, found in no markdown doc and absent from the example `mlfl.yaml`:

- `selected_model` — separate from `production_model`; tracks which model the UI highlights.
- `output_activation` — neural-net output head; 7 valid values; affects the prediction range.
- `use_revin` — Reversible Instance Normalisation toggle (default `True`).
- `future_covariate_features` — the list of feature names the TiDE temporal decoder routes through its future path.
- `load_subtract` — entire feature exists. It is referenced as a feature bullet in the README ("Load subtract"), but the per-sensor schema (`entity_id`, `source`, `on_missing`, `scale`, `max_fraction_of_load`, `max_fraction_violation_pct`) is undocumented. Compare with the deprecated `subtract` stub which **is** documented.
- `clear_forecast_log_on_retrain` — affects forecast-stability metric interpretation.
- `stability_focus` — `per_moment` vs `daily_total`; drives which metric the Forecast Accuracy verdict chip reads.
- `max_age` — SQLite cache retention (365 days default).
- `daily_loss_weight` — cumulative-trajectory loss weight for neural models.
- `recency_half_life_days` — exponential recency weighting (default 0; v2.x had 7).
- `conformal_coverage` — 80% default; README mentions "80% conformal bands" but not how to change them.
- `quantiles` — multi-quantile pinball-loss head (currently only the DLinear backend).
- `gap_handling`, `gap_max_minutes` — gap-fill behaviour (default `interpolate`, 90 min).
- `outlier_method`, `outlier_quantile`, `outlier_lower` — outlier-handling pipeline; default has changed (used to be 0.995 quantile, now 0.999 — per the field docstring).
- `loss_fn` (default `huber`) — mentioned in `mlfl.yaml` only, with no explanation of `mse` vs `mae` vs `huber` vs `tweedie`.
- `optimiser` (`adamw` vs `adam`) — neural-only knob.
- `include_sun_elevation`, `include_clear_sky_irradiance` — these are the gate for the solar-physics path. README's "Forecasts collapse to zero overnight" troubleshooting names them, but the option semantics are documented nowhere.
- `model_params` (per-experiment hyperparameter overrides).
- `clear_forecast_log_on_retrain`.

#### 4.2.3 `CovariateCfg` fields

Code (lines 145–197): `entity`, `role`, `scale`, `transform`, `aggregation`, `is_binary`, `future_attribute`, `future_value_key`. Valid roles: `future`, `lagged`, `both`, `concurrent`.

Documented in `CONFIG_GUIDE.md`: first six only. `future_attribute` and `future_value_key` (required to wire up Met.no weather or Solcast attributes) are **undocumented**.

CHANGELOG 2.31.0 entry A6 says *"Future / Both options removed from the covariate Role dropdown"* — but the code still accepts those values without warning. Also adds `concurrent` to the valid set, which is documented nowhere.

#### 4.2.4 `SubtractCfg` (load-subtract)

Code: 6 fields with explicit validation and a multi-paragraph docstring per field (`config.py` 50–143). Documented: **nowhere**. The README's Features section says "Load subtract. Subtract one HA sensor from another before modelling (e.g. net-of-solar demand) with a robustness layer for missing covariate data" — that is the entire user-visible documentation of this entire feature.

#### 4.2.5 `AppConfig` (top-level) fields

Code (lines 640–684): `forecast_every_minutes`, `retrain_every_hours`, `update_every_minutes` (legacy alias), `timezone`, `experiments`, `cpu_cores`, `nice_priority`, `model_overrides`.

`docs/CONFIG_GUIDE.md` documents: `update_every_minutes` (legacy alias only — not the canonical pair), `timezone`, `experiments`.

`README.md` mentions `timezone` and the experiments list; mentions `forecast_every_minutes` and `retrain_every_hours` in the table; says nothing about `cpu_cores`, `nice_priority`, or `model_overrides`. CHANGELOG 2.31.0 A1 promises `cpu_cores` and `nice_priority` are surfaced on the System page UI — but the YAML field that controls them is not documented at all.

### 4.3 Documented fields that no longer exist

| Doc | Field | Status |
|---|---|---|
| `docs/CONFIG_GUIDE.md` | `horizons_minutes` (4 references) | Removed. `load_config()` strips it on load and auto-migrates the YAML (`config.py` lines 773–774, 867–869). The README and `mlfl.yaml` example use `future_periods` instead. |
| `docs/CONFIG_GUIDE.md` | `subtract: [str]` documented as a working feature | Deprecated stub. `config.py` line 359 marks it as never-wired-into-preprocessing and `load_config` emits a deprecation warning. Real feature is `load_subtract`. |
| `docs/FEATURES_GUIDE.md` | `create_forecast_features` function | Function does **not** exist in `features.py`. The module exposes `is_holiday`, `build_features`, `prepare_train_test`, `reshape_for_sequence`, `create_sliding_windows` — but not `create_forecast_features`. |
| `docs/FEATURES_GUIDE.md` | TensorFlow / Keras (`from tensorflow.keras.models import Sequential`, `LSTM`, `Conv1D`) | The project uses PyTorch (`torch>=2.0.0` in `requirements.txt`), not TF. No TF / Keras anywhere in the code. |
| `docs/FEATURES_GUIDE.md` | "shared across all 14 model backends" | 24 backends are now registered (`main.py` 348–375). |

### 4.4 Features in the code that the docs never mention

Discovered by reading `main.py`, `web/app.py`, `CHANGELOG.md` 2.30.0–2.31.0:

- **Companion lifecycle sensors.** Each production experiment publishes `sensor.{prefix}{name}_last_benchmark` and `sensor.{prefix}{name}_last_retrain` (CHANGELOG 2.31.0 A5, code in `main.py` line 5936). README mentions only the `_forecast` / `_upper_80` / `_lower_80` sensors.
- **`_interval` and `_cumulative` companion sensors.** `main.py` lines 3990–4119 publish `sensor.{prefix}{name}_forecast`, `_interval`, `_cumulative`, `_upper_{pct}`, `_lower_{pct}`, `_forecast_accuracy`. README documents `_forecast` + `_upper_80` + `_lower_80` only. `CONFIG_GUIDE.md` documents `_forecast`, `_cumulative`, `_interval` but not the conformal-band or accuracy sensors.
- **The Lovelace dashboard YAML download.** CHANGELOG 2.31.0 B5 — auto-generated Lovelace YAML downloadable from System and per-experiment pages. Not in any doc.
- **One-click rollback.** CHANGELOG 2.31.0 A4 — `Roll back` button on production experiment headers swaps current ↔ previous model. Mentioned in CHANGELOG only.
- **Pre-flight Data sanity check.** CHANGELOG 2.31.0 C1 — Settings tab. Catches "your sensor has a 14-day flatline" before benchmarking. Mentioned in CHANGELOG only.
- **Tune-all sweep.** CHANGELOG 2.31.0 C4 — Bayesian tuning across every enabled model. README mentions per-model tuning ("Bayesian optimisation (Optuna TPE) per-model with composite-rank trial selection") but not the sweep across-all feature.
- **Pairwise model comparison matrix / paired-t test.** CHANGELOG 2.31.0 C3 — Results tab. README mentions Diebold-Mariano in features list; doesn't connect to the UI surface.
- **Skill-vs-Seasonal-Naive chip.** CHANGELOG 2.31.0 D2 — always-on chip on Results. Not in docs.
- **Retrain-history chip strip.** CHANGELOG 2.31.0 D3 — Forecast Accuracy tab. Not in docs.
- **Training-window vs test-window drift verdict (PSI).** CHANGELOG 2.31.0 D4 — Results tab. Not in docs.
- **Quick-preset chips (Fast / Balanced / Thorough).** CHANGELOG 2.31.0 C2 — Models tab. Cross-references `docs/MODEL_GUIDE.md`. Not surfaced in `MODEL_GUIDE.md` itself.
- **The HTMX-driven partial dashboard refresh.** CHANGELOG 2.31.0 B2 — invisible to users by design but explains the no-page-flicker behaviour.
- **CPU-cores and nice-priority controls actually applied.** CHANGELOG 2.31.0 A1 — System page. The README doesn't mention these even exist as knobs.
- **Conformal coverage configurability.** README/CONFIG_GUIDE say "80% bands" as if hardcoded; code exposes `conformal_coverage` (0–1).
- **Native-quantile training via `quantiles: [...]`** — only DLinear currently. Not in docs.
- **Solar-physics covariates as opt-in fields.** README troubleshooting section names them, but the field semantics are documented nowhere.

### 4.5 Features in the docs that the code does not back

- `docs/CONFIG_GUIDE.md` documents `horizons_minutes: [120, 480, 1440]` in two example blocks. A user copying this into `mlfl.yaml` will trigger the silent auto-migration in `load_config` (the field is stripped, the YAML is rewritten). Not destructive but confusing — the user will lose the field and may think they configured something they didn't.
- `docs/FEATURES_GUIDE.md` code examples use `from tensorflow.keras.models import Sequential`. A user copy-pasting these will get `ModuleNotFoundError`.
- `docs/CONFIG_GUIDE.md` calls out `subtract: [str]` as a feature for "calculating net import/export". A user setting this will get a deprecation warning in the log and **no effect on training** — the field has been a no-op stub since v2.30.0.

### 4.6 Other concrete mismatches with the running code

| Doc claim | Code reality |
|---|---|
| `README.md` line 61, `mlfl.yaml` line 13: "Open the Web UI (`http://homeassistant.local:5052`)". | `config.yaml` declares `ingress: true` and `ingress_port: 5052` but no `ports:` mapping — the direct host port is NOT exposed. CHANGELOG 2.30.0 explicitly notes "Removed the direct port 5052 exposure from `config.yaml`. The web UI is now reached exclusively through Home Assistant's authenticated ingress proxy." The user must click "Open Web UI" or use HA's panel. |
| `mlfl.yaml` line 4: "Place this file at `/config/mlfl.yaml` on your HA instance." | Primary path is `/addon_configs/ml_forecast_lab/mlfl.yaml`; `/config/mlfl.yaml` is the *secondary* fallback (`main.py` lines 219–223). README states the canonical primary path; `mlfl.yaml`'s own header gives only the fallback. |
| `README.md` Quick start step 5: "publishes `sensor.mlfl_<experiment_name>` companion sensors with `_lower_80` / `_upper_80` conformal bands." | The conformal-band sensor names embed the coverage level dynamically — `_upper_80` and `_lower_80` are emitted when `conformal_coverage` is at its default 0.8, but `_upper_90` / `_lower_50` etc. when the user changes the setting (`main.py` 4154–4158). |
| `README.md` line 200, table: "`models_enabled` … Backends to benchmark (24 available) — Default: LightGBM, XGBoost". | Code default (`config.py` line 228): `['lightgbm', 'xgboost', 'lstm', 'cnn']`. README's stated default is wrong. |
| `README.md` line 197, table: "`days_history` … Default: 30". | Code default (`config.py` line 212): `14`. |
| `docs/CONFIG_GUIDE.md` line 100: "`production_metric` (str, default `'mae'`)". | Code default (`config.py` line 270): `'seasonal_mase'`. The default was deliberately changed to better reward daily-seasonal HA targets; CONFIG_GUIDE is stale. |
| `README.md` final paragraph of Troubleshooting names log tags `[BENCH] [MODEL] [WEB] [HA] [PREP]`. | Actual phase tags (`__main__.py` lines 94–109): `BENCH, MODEL, WEB, HA, PREP, FEAT, COV, PUB, SOLAR, TRAIN, DASH, CFG, DB, APP, MLFL`. The README sample is correct but partial — users running `grep` for the others (`[FEAT]`, `[COV]`, `[CFG]`, `[DB]`) get hits the README never names. |
| `README.md`: "First build takes 10–15 minutes on a Raspberry Pi 5 — LightGBM, XGBoost, and **PyTorch** all compile native extensions for `aarch64`." | `requirements.txt` confirms PyTorch usage — internally consistent. **But** the same sentence in `docs/FEATURES_GUIDE.md` code examples uses **TensorFlow**. The two docs disagree on which DL framework the project uses. |
| `ml-forecast-lab/config.yaml` `version: 2.32.0`. | The most recent `CHANGELOG.md` entry is `## 2.31.0`. 2.32.0 has no changelog stanza. |
| `ml-forecast-lab/translations/en.yaml` has a `models` translation key. | The add-on schema in `config.yaml` only declares `log_level`. The `models` key is orphaned — HA renders nothing for it. |

### 4.7 Documented options vs the example `mlfl.yaml`

`mlfl.yaml` uses some fields the markdown docs never describe:

- `loss_fn: "mse"` — neural-model training loss.
- `max_age: 365` — SQLite cache retention.
- `future_periods: 96` — used in README minimal example but not in `CONFIG_GUIDE.md`.
- `database: true` — cache flag.
- `publish_name: "mixergy_demand"` — referenced in README only as a sentence fragment.
- Custom-metric block with `peak_hour_mae` — README mentions custom metrics in one line; the asteval sandbox (code line `metrics.py:545`) is not documented anywhere.

---

## 5. Screenshots, images, and rendering

- **Add-on store icon / logo.** `ml-forecast-lab/icon.png` and `ml-forecast-lab/logo.png` exist but are both 2127×2127 PNGs (~150 kB). HA add-on convention is 128×128 for the icon and a wider banner (~250×100) for the logo. The supervisor will down-scale, but the on-disk asset is much larger than typical add-ons ship, and `icon.png` and `logo.png` are byte-for-byte identical — there is no separate banner image.
- **README screenshots.** Section "Screenshots" (line 47–54) contains three HTML comments marked `<!-- TODO: -->`. Nothing renders. A user reading the GitHub README sees just the heading "Screenshots" followed by the next section.
- **In-doc images.** `README.md` line 3 embeds `logo.png` at `width=180`. That's the only image rendered anywhere in the doc set.
- **Web UI screenshots embedded in docs.** None. The user has no visual preview of the dashboard, the experiment detail page, the Models tab rank table, the Forecast Accuracy verdict, or the Tuning chart before installing.
- **External screenshot host (Imgur, GitHub user-content URLs).** None found.

---

## 6. Render targets — where each file ends up

| File | GitHub project README | HA add-on store **Info** tab | HA add-on store **Documentation** tab | HA add-on store **Changelog** tab |
|---|:---:|:---:|:---:|:---:|
| `README.md` (repo root) | yes | no | no | no |
| `ml-forecast-lab/README.md` | n/a | **would render here — but the file does not exist** | n/a | no |
| `ml-forecast-lab/DOCS.md` | n/a | n/a | **would render here — but the file does not exist** | no |
| `ml-forecast-lab/CHANGELOG.md` | no | no | no | yes |
| `docs/*.md` | linked from `README.md` | no | no | no |
| `ml-forecast-lab/config.yaml`'s `description:` field | no | yes (one line fallback) | no | no |

A HA user opening the add-on in the store currently sees:

- **Info tab**: the one-line fallback description from `config.yaml` (`"Multi-model ML forecasting and benchmarking for Home Assistant"`) and the version, arch list, icon. No installation steps, no quick-start, no Configuration, no screenshots — because none of the project's content is in a file the supervisor renders.
- **Documentation tab**: same fallback or blank.
- **Changelog tab**: the full 4 268-line CHANGELOG.md.

The README content the author wrote is only visible if the user navigates from the store panel to the GitHub repo URL — which is a click most users won't make.

---

## 7. Tone / convention spot-checks

- British English is generally consistent across the doc set (`behaviour`, `serialisation`, `optimiser`, `colour`, `licence` in the README badge). `docs/FEATURES_GUIDE.md` mixes `normalisation` (BE) and `Sequence` (no preference). `docs/PREPROCESSING_GUIDE.md` uses `optimization` (US) once at line 415. No major dialect issues but some inconsistency.
- The README and `MODEL_GUIDE.md` use the warmer "you" voice. The `CONFIG_GUIDE` / `PREPROCESSING_GUIDE` / `FEATURES_GUIDE` trio uses passive, reference-manual voice.
- Models are referred to by both short slug (`lightgbm`) and proper noun (`LightGBM`) interchangeably — usually consistent within a file.

---

## 8. Summary observations (no judgement — Phase 2 grades these)

1. **The HA add-on store panels are documentation-empty.** The repo has 1 700+ lines of user-facing markdown, but the supervisor renders none of it on the Info / Documentation tabs — there is no `ml-forecast-lab/README.md` or `DOCS.md`.
2. **The `docs/` folder is a developer reference**, not a user guide. Three of the four files (`CONFIG_GUIDE`, `PREPROCESSING_GUIDE`, `FEATURES_GUIDE`) document Python classes / functions intended for someone importing the package, not someone editing `mlfl.yaml` in the HA File editor.
3. **Configuration documentation lags the code significantly.** Roughly 18 fields in `ExperimentCfg` are completely undocumented, and several documented fields no longer exist or are no-op stubs. The entire `SubtractCfg` schema (a feature called out as a headline bullet in the README) is undocumented.
4. **There is no screenshot anywhere in the documentation set** — three `<!-- TODO -->` placeholders in the README and nothing else.
5. **The Web UI access URL given in two places (`README.md`, `mlfl.yaml`) is wrong** since v2.30.0 — direct port 5052 access has been removed.
6. **The example `mlfl.yaml` header points to the secondary fallback path** rather than the primary `/addon_configs/...` location.
7. **`MODELS_CREATED.txt`** in the addon directory is an internal hand-off note with sandbox paths and a "future enhancements" wishlist. It is not user documentation but lives in the user-facing tree.
8. **CHANGELOG covers 2.31.0; `config.yaml` is at 2.32.0.** Either a missing entry or an unintended bump.
9. **Numbers stated as "default" in README and `CONFIG_GUIDE.md` differ from the dataclass defaults** for at least `days_history`, `models_enabled`, and `production_metric`.
10. **Several major features shipped in 2.30 / 2.31 (lifecycle sensors, rollback, sanity check, tune-all, drift verdict, Lovelace YAML download)** are CHANGELOG-only — no mention in README, no mention in `docs/`.

End of survey. Awaiting confirmation before producing `DOC_AUDIT.md`.
