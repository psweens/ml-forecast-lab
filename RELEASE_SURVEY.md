# RELEASE_SURVEY.md — ML Forecast Lab

Phase 1 (survey) output. No judgements yet — that is `RELEASE_GATE.md`.
British English.

---

## 1. Repo tree (one-line role per file)

### Root

| Path | Role |
|---|---|
| `LICENSE` | MIT licence, © 2026 Paul Sweeney. |
| `README.md` | Top-level GitHub landing page; pitches the add-on and points at the per-add-on docs. |
| `repository.yaml` | HA add-on **repository manifest** — declares this repo as a multi-add-on store (currently one add-on). |
| `icon.png`, `logo.png` | Top-level repository branding (identical bytes; 2127×2127 PNG). |
| `.gitignore` | Standard Python/IDE ignores; excludes `.env`, `*.db`, `*.log`. |
| `AUDIT_PROMPT.md` | Internal dev prompt for a latent-defect audit. Not user-facing. |
| `docs/MODEL_GUIDE.md` | Practical "which of the 24 backends to enable" guide, linked from both READMEs. |
| `.github/workflows/tests.yml` | CI — smoke + unit test jobs on every PR/push to `main`. |
| `.github/workflows/validate.yml` | CI — config-version vs `__init__.__version__` consistency + Python `py_compile` smoke. |
| `.github/workflows/release.yml` | CI — on push of a `v*` tag, verifies `config.yaml` version matches and creates a GitHub Release. |

### Add-on directory (`ml-forecast-lab/`)

| Path | Role |
|---|---|
| `config.yaml` | HA add-on manifest — name, slug, version (`2.34.4`), arch list, ingress, permissions. |
| `build.yaml` | Per-arch `build_from` base images (hassio-addons ubuntu-base 9.0.5). |
| `Dockerfile` | Two-stage build: builder installs Python deps into venv, runtime stage copies venv + rootfs + app code. |
| `requirements.txt` | 20 runtime Python deps (numpy, pandas, torch, fastapi, lightgbm, xgboost, statsforecast, plotly, optuna, pvlib, …). |
| `README.md` | HA store **Info** tab — install + first forecast. |
| `DOCS.md` | HA store **Documentation** tab — full config reference, published sensors, operations, troubleshooting. |
| `CHANGELOG.md` | HA store **Changelog** tab — 173 versioned entries; current head is `2.34.4`. |
| `icon.png`, `logo.png` | Add-on artwork (identical bytes to root, 2127×2127). |
| `mlfl.yaml` | **Bundled example user-config**; copied to `addon_configs/ml_forecast_lab/mlfl.yaml` on first boot if no config is present. References the author's `mixergy_*` sensors as illustrative examples. |
| `translations/en.yaml` | One-string HA add-on UI translation (`log_level` field label). |
| `rootfs/etc/s6-overlay/s6-rc.d/init-mlforecastlab/{run,up,type}` | s6-overlay v3 service definition (`type=longrun`, run script bootstraps `/data/ml_forecast_lab/{models,logs}` and `exec python3 -m ml_forecast_lab`). |
| `rootfs/etc/s6-overlay/s6-rc.d/user/contents.d/init-mlforecastlab` | Empty bundle-membership marker for the `user` s6 bundle. |

### Application package (`ml-forecast-lab/ml_forecast_lab/`)

Total: ~30,700 LOC Python.

| Path | Role | LOC |
|---|---|---|
| `__init__.py` | Public re-exports; declares `__version__ = "2.34.4"`. | 70 |
| `__main__.py` | Process entry. Configures logging (bashio LOG_LEVEL parsing, rotating file at `/data/.../logs/mlfl.log`), launches `MLForecastLabApp.run()`. | 227 |
| `main.py` | Orchestrator: config load, component init, web server start, scheduled retrain/forecast loop. | 6,166 |
| `config.py` | YAML dataclasses (`AppConfig`, `ExperimentCfg`, `CovariateCfg`, `SubtractCfg`) + atomic YAML write. | 1,331 |
| `db.py` | SQLite history cache (actuals, forecast log, benchmark blobs, conformal residuals). | 2,442 |
| `ha_interface.py` | HA REST client (history fetch, state publish). Bearer auth from `SUPERVISOR_TOKEN`. | 424 |
| `covariates.py` | Resolves lagged + future covariates from HA. | 290 |
| `preprocessing.py` | Cumulative-to-interval, gap fill, outlier clip, load-subtract pipeline. | 941 |
| `features.py` | Lags + temporal + holiday + solar features. | 565 |
| `solar_physics.py` | `pvlib` clear-sky GHI + sun-elevation features. | 97 |
| `training_events.py` | In-memory event bus used by the Training tab. | 152 |
| `dashboard.py` | Generates a Lovelace ApexCharts YAML at `/addon_configs/<slug>/mlfl_dashboard.yaml`. | 250 |
| `benchmark/{runner,metrics,comparison}.py` | CV runner, scoring, Demšar ranking, paired-t. | 1,890 |
| `models/base.py` | Shared neural training scaffolding (RevIN, early stop, recency weighting). | 921 |
| `models/registry.py` | Backend slug → class registry. | 330 |
| `models/{lightgbm,xgboost,catboost}_backend.py` | Tree backends. | ~1,200 |
| `models/{lstm,gru,cnn,timesnet}_backend.py` | Recurrent + convolutional. | ~2,100 |
| `models/{dlinear,nlinear,tsmixer,timemixer,tide,sparsetsf,fits}_backend.py` | Linear / MLP / freq-domain. | ~3,200 |
| `models/{nbeats,nhits}_backend.py` | N-BEATS family. | ~1,000 |
| `models/{patchtst,itransformer,crossformer,tft}_backend.py` | Transformers. | ~2,100 |
| `models/{statsforecast,seasonal_naive}_backend.py` | Classical + naive baselines. | ~600 |
| `web/app.py` | FastAPI app — routes, HTMX fragments, ingress support, settings writeback. | 3,709 |
| `web/templates/{base,dashboard,experiment,models,system,training,logs}.html` | Jinja2 templates for the UI. | ~10,000 (mostly `experiment.html`) |
| `web/static/{style.css,htmx.min.js,plotly-basic.min.js,icon.png}` | Frontend assets (htmx 1.9.10, plotly basic build). | (vendored libs) |

### Tests (`ml-forecast-lab/tests/`)

| Path | Role |
|---|---|
| `dryrun_pipeline.py` | Manual end-to-end harness used during development. |
| `conftest.py`, `__init__.py` | Shared fixtures. |
| `requirements-dev.txt` | pytest, httpx, fastapi test client. |
| `smoke/` | 9 FastAPI smoke tests (boot `create_app()` against a tmp `mlfl.yaml`, walk the golden flows). Designed as the release gate per `README.md`. |
| `unit/` | 8 module-level unit suites (`config`, `db`, `features`, `preprocessing`, `models`, `benchmark`, `load_subtract`, `forecast_analytics`). |

---

## 2. License

- **File present**: `LICENSE` at root (1 069 bytes).
- **Type**: MIT. Standard SPDX text, © 2026 Paul Sweeney.
- **In-tree references**: README footer says "MIT © Dr Paul W. Sweeney, University of Cambridge". `config.yaml` does not embed a SPDX field (HA add-on config has no required licence field).

---

## 3. Version scheme and current version

- **Scheme**: SemVer-style `MAJOR.MINOR.PATCH` (no `v` prefix inside files; tags use `v` prefix).
- **Current version**: `2.34.4`, present in:
  - `ml-forecast-lab/config.yaml:2` (`version: 2.34.4`)
  - `ml-forecast-lab/ml_forecast_lab/__init__.py:36` (`__version__ = "2.34.4"`)
  - `ml-forecast-lab/CHANGELOG.md:3` (top entry)
  - `README.md:129` (static release badge)
- **Version consistency enforced by CI**: `.github/workflows/validate.yml` greps both files on every PR.
- **Release-gating CI**: `.github/workflows/release.yml` fires on `v*` tag push, verifies `config.yaml` version matches, and creates a GitHub Release with the tag annotation as the body.
- **Git tags pushed to remote**: latest is `v2.8.5`. There is **no `v2.34.4` tag** on the remote. Local repo has no tags at all. So the 26 versions between `v2.9.x` and `v2.34.4` were never released via the `release.yml` workflow.

---

## 4. Repository structure

**Multi-add-on repository layout** (single add-on inside it).

- Root `repository.yaml` declares this as a HA add-on store (loads when the user adds the URL to *Settings → Add-ons → Repositories*).
- The add-on lives in `ml-forecast-lab/` (the directory name == add-on slug-ish; the actual slug `ml_forecast_lab` is set inside `ml-forecast-lab/config.yaml`).
- HA add-on stores require: root `repository.yaml` (or `.json`), and each add-on as a subdirectory containing its own `config.yaml`. Both are present.

---

## 5. Distribution channel

- **Intended**: Personal Home Assistant repository, installed by users adding `https://github.com/psweens/ml-forecast-lab` via *Settings → Add-ons → ⋮ → Repositories*. The "Open your HA instance and show the add-on repository dialog" badge in `README.md` is the one-click variant.
- **Not** the HA community add-ons store (`hassio-addons` organisation) — no PR referenced, no inclusion in `hassio-addons/repository`.
- **Not HACS** — HACS is for integrations + frontend modules, not HA add-ons.
- **No image registry**. `config.yaml` does not set `image:`, so the supervisor builds the Docker image locally on the user's HA host from the `Dockerfile`. No ghcr.io / dockerhub publication step exists in CI.

---

## 6. Public-facing surfaces

| Surface | Where | Status |
|---|---|---|
| Root README | `README.md` | Present. Pitches the add-on, lists 24 backends, gives the one-click install badge. |
| Add-on README | `ml-forecast-lab/README.md` | Present. Loads as the **Info** tab in the HA store. Install + first forecast. |
| Add-on docs | `ml-forecast-lab/DOCS.md` | Present. Loads as the **Documentation** tab. Full config reference + ops. |
| Changelog | `ml-forecast-lab/CHANGELOG.md` | Present. 173 entries, current head matches version. |
| Model guide | `docs/MODEL_GUIDE.md` | Present. |
| Issue templates | `.github/ISSUE_TEMPLATE/` | **Absent.** |
| PR template | `.github/pull_request_template.md` | **Absent.** |
| Contributing guide | `CONTRIBUTING.md` | **Absent.** |
| Code of conduct | `CODE_OF_CONDUCT.md` | **Absent.** |
| Security policy | `SECURITY.md` | **Absent.** |
| Support / contact channel | `README.md`'s `## Support` section | Points at GitHub issues. |
| Screenshots | Suggested in add-on README but currently `<!-- commented out -->`. | None bundled. |
| Translations | `ml-forecast-lab/translations/en.yaml` | One field only (`log_level`). |

---

## 7. CI/CD

| Workflow | Trigger | What it does | Publishes |
|---|---|---|---|
| `tests.yml` | push/PR to `main` | Smoke (~30 s, FastAPI test client) + Unit (~10 min, full deps). | nothing |
| `validate.yml` | push/PR to `main` | Version consistency check (`config.yaml` vs `__init__.py`); `py_compile` smoke; parses `mlfl.yaml`. | nothing |
| `release.yml` | push of `v*` tag | Verifies `config.yaml` version matches the tag; creates GitHub Release with tag-annotation body. | GitHub Release object only |

- **Image registry**: none. Per §5, no Docker image is published — the supervisor builds locally on each user's HA host from the `Dockerfile`.
- **Multi-arch build matrix**: not run in CI. The Dockerfile is per-arch via `BUILD_FROM`; user-side supervisor selects the arch.
- **Tag/release reality**: latest remote tag is `v2.8.5` (no `v2.34.x`). The `release.yml` workflow has been dormant since the v2.8 series.

---

## 8. Secrets, tokens, API keys

### Current tree

| Finding | Evidence |
|---|---|
| **Only secret consumed**: `SUPERVISOR_TOKEN` env var, provided by the HA supervisor at runtime when `homeassistant_api: true`. | `ml_forecast_lab/ha_interface.py:210` (`self.ha_key = ha_key or os.environ.get("SUPERVISOR_TOKEN", "")`); `web/app.py:1977`. |
| No hard-coded keys, no `.env` in tree, no `credentials.json`. | `find` returns nothing matching `*secret*`, `*credentials*`, `*.env`. |
| `.gitignore` excludes `.env` and `*.db` / `*.log`. | `.gitignore:27,32,33`. |

### Git history

| Finding | Evidence |
|---|---|
| No `.env`, `credentials*`, `secret*`, or token-like file ever added/removed. | `git rev-list --all | xargs git diff-tree --name-only` → no matches. |
| Sample of full-history diff scanned for the patterns `API_KEY=`, `password =`, `secret =`, `access_token =` → all hits are either documentation strings, Python parameter names (`ha_key`, `ha_token`), or vendored plotly minified JS noise. | Manual scan. |
| Default sample config (`mlfl.yaml`) contains author-style example sensor IDs (`sensor.mixergy_demand_today`, `sensor.current_charge`, `sensor.external_temperature`) but no real device IDs / hostnames / IPs / coordinates / personal data. | `mlfl.yaml:36-167`. |

### Telemetry / phone-home

| Finding | Evidence |
|---|---|
| **None.** No outbound HTTP destinations other than HA's own supervisor + REST API. | `grep -rn "session\.get|ClientSession" ml_forecast_lab/` returns only `ha_interface.py` (supervisor) and `web/app.py:1979` (also supervisor). |
| The word "analytics" in the codebase refers to the add-on's internal forecast-accuracy analytics, not external trackers. | `web/app.py:124, 436, 479`. |
| No Sentry, PostHog, GA, Datadog, Mixpanel, Amplitude SDKs in deps. | `requirements.txt`. |

---

## 9. Third-party dependencies and licences

### Python (`requirements.txt`)

| Package | Pinned range | Licence | MIT-compatible? |
|---|---|---|---|
| `numpy` | `>=1.24,<2.0` | BSD-3 | ✓ |
| `pandas` | `>=2.0,<3.0` | BSD-3 | ✓ |
| `PyYAML` | `>=6.0,<7.0` | MIT | ✓ |
| `aiohttp` | `>=3.8,<4.0` | Apache-2.0 | ✓ |
| `lightgbm` | `>=4.0,<5.0` | MIT | ✓ |
| `xgboost` | `>=2.0,<3.0` | Apache-2.0 | ✓ |
| `catboost` | `>=1.2,<2.0` | Apache-2.0 | ✓ |
| `torch` | `>=2.0` | BSD-3 | ✓ |
| `statsforecast` | `>=1.7,<2.0` | Apache-2.0 | ✓ |
| `holidays` | `>=0.40,<1.0` | MIT | ✓ |
| `asteval` | `>=1.0,<2.0` | MIT | ✓ |
| `fastapi` | `>=0.100,<1.0` | MIT | ✓ |
| `uvicorn[standard]` | `>=0.23,<1.0` | BSD-3 | ✓ |
| `Jinja2` | `>=3.1,<4.0` | BSD-3 | ✓ |
| `python-multipart` | `>=0.0.5,<1.0` | Apache-2.0 | ✓ |
| `plotly` | `>=5.0,<6.0` | MIT | ✓ |
| `scikit-learn` | `>=1.3,<2.0` | BSD-3 | ✓ |
| `scipy` | `>=1.10,<2.0` | BSD-3 | ✓ |
| `optuna` | `>=3.5,<4.0` | MIT | ✓ |
| `pvlib` | `>=0.10,<1.0` | BSD-3 | ✓ |

No GPL/AGPL/LGPL/copyleft dependencies. All compatible with the MIT licence on this project.

### Vendored frontend JS (`ml_forecast_lab/web/static/`)

| File | Version (from preamble) | Licence | Compatible? |
|---|---|---|---|
| `htmx.min.js` | `1.9.10` | 0BSD / Zero-clause BSD | ✓ |
| `plotly-basic.min.js` | (Plotly.js Basic distribution) | MIT | ✓ |

No vendored licence files are bundled alongside (`THIRD_PARTY_LICENSES` / `NOTICE` absent), but neither licence requires inclusion of the original licence text when distributing minified builds.

### Docker base image

- `ghcr.io/hassio-addons/ubuntu-base:9.0.5` (aarch64, armv7) and `ghcr.io/hassio-addons/ubuntu-base/amd64:9.0.5` (amd64) — community-add-ons project, Apache-2.0.
- Note the asymmetry: `amd64` has the per-arch repo suffix, `aarch64` / `armv7` do not. (Flagged in the gate review — needs runtime verification of whether the no-suffix manifest exists for those arches.)

---

## 10. Hardcoded paths, URLs, usernames, dev-machine artefacts

### Paths

| Path | File | Justified? |
|---|---|---|
| `/data/ml_forecast_lab/{models,logs,history.db}` | `__main__.py:83`, `main.py:328,6137-6139`, `rootfs/.../run:9-11` | Yes — `/data` is the canonical HA add-on persistent dir. |
| `/addon_configs/ml_forecast_lab/mlfl.yaml` | `main.py:220`, `web/app.py:467`, `dashboard.py:6122` | Yes — HA `addon_config:rw` mount; canonical path. |
| `/addon_configs/[8-hex-prefix]_ml_forecast_lab/...` | `main.py:229-233`, `web/app.py:467`, `rootfs/.../run:21` | Yes — covers the supervisor's hashed-slug variant. Anchored to 8 hex chars to prevent a fork hijack. |
| `/config/mlfl.yaml` and `/config/ml_forecast_lab` | `main.py:221, 6140`, `web/app.py:467`, `db.py:41` (default) | Legacy fallback. `homeassistant_config:ro` is mounted; the `mkdir(/config/ml_forecast_lab)` in `_setup_directories` may fail on a fresh install — **flagged for runtime verification**. |
| `/share/ml_forecast_lab` mentioned in `DOCS.md:23` | `DOCS.md` | Documentation only; no code currently writes here (writes go to `/data`). Minor doc drift. |

### URLs

| URL | Where | Notes |
|---|---|---|
| `http://supervisor/core` | `ha_interface.py:209`, `web/app.py:1976` | HA supervisor API — correct. |
| `http://homeassistant.local:5052/experiment/<name>` | `dashboard.py:124` (inside the auto-generated Lovelace YAML's markdown card) | **Stale.** Direct port 5052 was removed in v2.30.0 per `DOCS.md:336`. The downloadable Lovelace YAML would point users at a dead URL. |
| `https://github.com/psweens/ml-forecast-lab` | `config.yaml:6`, `repository.yaml:3`, README badges | Correct. |
| Various paper references on `arxiv.org`, `openreview.net` | Model backend docstrings | Reference citations, not active fetches. |

### Personal data

- "Dr Paul W. Sweeney" / "psweens" / "University of Cambridge" appear in licence header, README footer, repository URL, and an internal footer link — all legitimate authorship attribution.
- No email addresses, IPs, hostnames, postcodes, or coordinates in code or example config.

### Dev-machine artefacts

- No `node_modules/`, `.venv/`, `__pycache__/`, `.DS_Store`, `Thumbs.db`, IDE-local settings.
- `AUDIT_PROMPT.md` at the repo root is an internal dev prompt template; it survived a "remove stale internal audit documents" cleanup that took out four similar docs in v2.33.2.
- No `TODO` / `FIXME` / `XXX` / `HACK` in production paths (only one match across the whole `ml_forecast_lab/` tree, which is in a model docstring, not as a workaround marker).
- No `console.log` / `debugger` in templates. Two `console.warn` and a `console.error` in `experiment.html` are user-facing diagnostic aids retained on purpose.

---

## 11. Misc observations relevant to the gate

1. `config.yaml` declares only `log_level` as a user-facing option. Everything else lives in `mlfl.yaml` inside the addon-config dir. Defensible — typical for ML add-ons.
2. `panel_icon: mdi:chart-timeline-variant-shimmer` and `panel_title: ML Forecast Lab` are set; the add-on installs as a sidebar entry on top of the standard add-on store entry.
3. `homeassistant_api: true` is declared; the supervisor token is supplied to the container and used in `ha_interface.py` and `web/app.py`. Justified usage.
4. `ingress: true`, `ingress_port: 5052`. Web server in `main.py:389` binds `0.0.0.0:5052` — direct exposure of that port is gone (no `ports:` field in `config.yaml`), so the only path in is the authenticated ingress proxy. CORS middleware is intentionally not installed per a comment at `web/app.py:442-445`.
5. Map: `addon_config:rw` (writeable, used for `mlfl.yaml` + generated Lovelace YAML), `homeassistant_config:ro` (read-only, used to read legacy `/config/mlfl.yaml`). No `share:rw`, no `media:rw`, no `usb`, no `host_network`, no `privileged` — minimal surface.
6. Icon vs logo: `icon.png` and `logo.png` in `ml-forecast-lab/` are byte-identical. HA convention is that `icon.png` is the small square (typically 250×250) and `logo.png` is a wider banner (typically 250×100 or similar). Both at 2127×2127 will render but are oversized and don't distinguish the two slots.
7. `init: false` is required because the chosen base image already runs s6-overlay as PID 1. Correct.
8. s6-overlay v3 service is wired: `type=longrun` + `run` + empty `contents.d/<name>` membership marker. The companion `up` file is harmless for a longrun (only oneshots read `up`). The Dockerfile chmods `run` but not `up`; only `run` needs the +x bit for a longrun, so this is fine.
9. The `mlfl.yaml` example references the author's `mixergy_*` sensors. Useful as illustration but the comments around it are clear that users edit it. The first-boot init script does `cp /app/mlfl.yaml "${ADDON_CONFIG}/mlfl.yaml"` only when no config exists, so the author's example sensors are landed verbatim until the user replaces them.
10. README's release/licence/tests badges are deliberately **static** because shields.io can't read a private repo's API; commented dynamic equivalents are present for the public flip. The repo is still private at the time of this survey.
