# RELEASE_GATE.md — ML Forecast Lab

# DECISION: **GO-WITH-CAVEATS**

No blockers. Six warnings of substance, listed below as a `v2.34.5` punch list. The add-on can be flipped public; the warnings should be cleared in the first patch release.

The two items I initially flagged as candidate blockers — the `build_from` manifest for the non-amd64 arches, and the `_setup_directories` `mkdir(/config/ml_forecast_lab)` against a `homeassistant_config:ro` mount — both check out:

- `ghcr.io/hassio-addons/ubuntu-base:9.0.5` is a real multi-arch manifest containing arm64, amd64, and armv7 variants (verified by anonymous GHCR pull). The asymmetry in `build.yaml` is cosmetic — both the unsuffixed manifest and the `/amd64`-suffixed manifest exist on the registry.
- Under current Home Assistant supervisor conventions (HA 2023.6+), `addon_config:rw` mounts the add-on's own config to `/config` inside the container, and `homeassistant_config:ro` maps HA core's config to `/homeassistant`. So `/config/ml_forecast_lab` is writeable; the `mkdir` succeeds.

Both still belong on the runtime-verification checklist below — confirm on a real HAOS instance before the public flip.

British English throughout.

---

## A. Legal & licensing

| # | Item | Status | Evidence | Fix if not passing | Confidence |
|---|---|---|---|---|---|
| A1 | `LICENSE` file present and appropriate | **PASS** | `LICENSE` (MIT, © 2026 Paul Sweeney). Standard SPDX text. | — | High |
| A2 | Bundled / vendored code licence-compatible | **PASS** | `web/static/htmx.min.js` is htmx 1.9.10 (0BSD); `web/static/plotly-basic.min.js` is Plotly.js Basic (MIT). Both are compatible with this project's MIT licence and neither requires attribution text in a minified build. | — | High |
| A3 | All dependency licences recorded and compatible | **PASS** | All 20 entries in `ml-forecast-lab/requirements.txt` are MIT / BSD / Apache-2.0 (see `RELEASE_SURVEY.md` §9). No GPL/AGPL/LGPL/copyleft transitive deps in the direct list. | — | High (direct deps); medium (transitive — not enumerated) |
| A4 | No copyrighted assets used without permission | **PASS** | `icon.png` / `logo.png` are author-supplied artwork (identical bytes at root and under `ml-forecast-lab/`). No third-party screenshots bundled; README's `<!-- ![Dashboard] -->` references are commented placeholders. The Home Assistant logo in `ha-shield` is the official HA badge linked from `home-assistant.io`, not bundled. | — | High |
| A5 | No code copied from incompatibly-licensed sources | **PASS** | Model backend files cite their academic papers but are independent implementations on top of PyTorch / scikit-learn primitives — no copy-paste from GPL reference implementations is visible (`grep -r "GPL\|AGPL\|LGPL"` returns nothing in `ml_forecast_lab/`). Not exhaustively traced. | — | Medium-high |

---

## B. Secrets & privacy

| # | Item | Status | Evidence | Fix if not passing | Confidence |
|---|---|---|---|---|---|
| B1 | No API keys / tokens / passwords / `.env` in current tree | **PASS** | The only secret consumed is `SUPERVISOR_TOKEN` from the supervisor-supplied env (`ha_interface.py:210`, `web/app.py:1977`). `.gitignore` covers `.env`, `*.db`, `*.log`. | — | High |
| B2 | No secrets in git history | **PASS** | `git rev-list --all` shows no `*.env`, `*credentials*`, or `*secret*` file ever in the tree. Full-history grep for `API_KEY=`, `password =`, `secret =`, `access_token =` returns only docstring noise and `SUPERVISOR_TOKEN` plumbing. | — | High |
| B3 | No personal / production data in samples or fixtures | **PASS (with note)** | Bundled `mlfl.yaml` uses the author's `sensor.mixergy_*` / `sensor.current_charge` / `sensor.external_temperature` as illustrative entities. No hostnames, IPs, emails, coordinates, or device identifiers leak. Smoke fixtures use `sensor.smoke_*`. | — (note: an entirely generic example like `sensor.power_consumption_w` would be tidier; not a release blocker) | High |
| B4 | If add-on collects / transmits anything, it's disclosed | **PASS** | Only outbound traffic is to `http://supervisor/core` (HA's own REST API) — disclosed implicitly by `homeassistant_api: true`. No third-party endpoints contacted (`grep -rn "session\.get\|ClientSession"` only finds the supervisor calls). | — | High |
| B5 | Telemetry, if any, is documented, opt-in, and minimal | **PASS (N/A)** | No telemetry / analytics / phone-home of any kind. No Sentry, GA, PostHog, Datadog, Mixpanel, or Amplitude SDK in `requirements.txt`. | — | High |

---

## C. HA add-on contract

| # | Item | Status | Evidence | Fix if not passing | Confidence |
|---|---|---|---|---|---|
| C1 | `config.yaml` valid, schema correct, version present, slug sensible | **PASS** | Parses with `yaml.safe_load`. `version: 2.34.4`, `slug: ml_forecast_lab`, `name: ML Forecast Lab`, `description: …` all present. `arch:` is a list (not a scalar). Options schema is a valid bashio enum. | — | High |
| C2 | Dockerfile builds cleanly for ARM64 | **NEEDS RUNTIME VERIFICATION** | `Dockerfile:1-78` is a two-stage build pulling `ghcr.io/hassio-addons/ubuntu-base:9.0.5` (verified multi-arch on GHCR: arm64 + amd64 + armv7 are all present). Static `RUN apt-get install python3 build-essential gcc g++ gfortran cmake git` looks correct for aarch64 wheel-less builds (torch / lightgbm / xgboost compile). | If build fails, supervisor surfaces the error in the install log — fix per failure. Apt-get layer reasonable. | Medium (static analysis only) |
| C3 | Multi-arch build if intended for wider audience | **PASS** | `config.yaml:10-13` declares `aarch64`, `amd64`, `armv7`. `build.yaml` provides `build_from` per arch (verified above). HA supervisor builds the relevant arch locally on each user's host. | — | High |
| C4 | `run` / s6 services correct; add-on starts cleanly on fresh install | **PASS (needs runtime verification)** | `rootfs/etc/s6-overlay/s6-rc.d/init-mlforecastlab/type` = `longrun`. `run` is `bashio`-aware, creates `/data/ml_forecast_lab/{models,logs}`, copies bundled `mlfl.yaml` to `addon_config` dir if absent, then `exec python3 -m ml_forecast_lab`. Bash syntax-checks clean. Empty `contents.d/init-mlforecastlab` membership marker correctly registers the service in the `user` bundle. Dockerfile chmods the `run` script. | — | Medium-high (correctness of script; runtime confirmation in checklist) |
| C5 | Permissions in `config.yaml` are all used and justified | **PASS** | `homeassistant_api: true` — used (`ha_interface.py:210` Bearer token; `web/app.py` entity search). `ingress: true` + `ingress_port: 5052` — used (`main.py:389` binds 0.0.0.0:5052). `map: [addon_config:rw, homeassistant_config:ro]` — addon_config used for `mlfl.yaml` writes, homeassistant_config:ro used for the legacy `/config/mlfl.yaml` fallback path. No `host_network`, no `privileged`, no `usb`, no `gpio`. Surface is minimal. | — | High |
| C6 | Ingress works; if direct port, it's documented | **PASS** | `config.yaml` has no `ports:` block — the only public path is HA ingress. `_get_base_path` (`web/app.py:449-451`) reads `X-Ingress-Path` for URL rewriting. Templates use `BASE_PATH` for every link. Comment at `web/app.py:442-445` documents the deliberate no-CORS posture: same-origin only via the supervisor's authenticated proxy. | — | High |
| C7 | No data loss on update; persistence model respected | **PASS** | All user state lives under `/data/ml_forecast_lab/` (models, logs, SQLite) which is the HA-supervisor-persistent add-on data volume. `mlfl.yaml` lives in `addon_config` which is preserved across updates. Atomic YAML writes via `config.py:24-47` (`os.replace` after temp file in same dir) prevent corruption on mid-write SIGKILL. Retrain archives the previous champion under `<model_dir>/previous/` and exposes a rollback. | — | High |
| C8 | `repository.yaml` correct if multi-add-on repo | **PASS** | Root `repository.yaml` declares the store: `name`, `url`, `maintainer`, `description`. Add-on lives in `ml-forecast-lab/` with its own `config.yaml`. Standard layout. | — | High |

---

## D. Install & first-run

| # | Item | Status | Evidence | Fix if not passing | Confidence |
|---|---|---|---|---|---|
| D1 | Fresh-install path works end-to-end from the README alone | **PASS (needs runtime verification)** | `README.md:39-48` walks Settings → Add-ons → Repositories → install. Also offers a one-click `my.home-assistant.io` install badge that pre-fills the repo dialog (`README.md:31`, `[openhainstall-link]`). Stages: (1) supervisor pulls repo, (2) reads `repository.yaml`, (3) shows add-on, (4) on Install → builds Dockerfile, (5) on Start → s6 runs `init-mlforecastlab/run`, (6) Python boots and serves web UI via ingress. | — | Medium (no live install attempted in this gate) |
| D2 | Sensible defaults — new user gets to "working" without expert config | **PASS** | Add-on boots with the bundled `mlfl.yaml` example experiment if the user hasn't created their own; or per the add-on README, the user can land in the dashboard with an empty config and use **Add Experiment** to create one. The schema in `config.py` defaults every non-required field. `models_enabled` defaults to `[lightgbm, xgboost, lstm, cnn]` — a sensible starter mix. | — | High |
| D3 | Failure on misconfiguration is graceful with a useful error | **PASS** | `load_config` (`main.py:248-289`) catches YAML errors, logs them, and either falls back to the existing in-memory config (on reload) or a stub config (on first load) rather than crashing. The web UI's `_safe_error` (`web/app.py:23-30`) redacts filesystem paths from user-visible error strings. | — | High |
| D4 | No assumed external services without documenting them | **PASS** | The only required service is Home Assistant itself (REST API via supervisor). `pvlib` clear-sky computation is local. No internet calls. Optional Met.no / Solcast covariates are documented in `DOCS.md` (lines 85-100). | — | High |

---

## E. Safety & data integrity

| # | Item | Status | Evidence | Fix if not passing | Confidence |
|---|---|---|---|---|---|
| E1 | Writes to `/data` are atomic where they need to be | **PASS** | `config.py:24-47` `atomic_yaml_write`: write-to-temp-in-same-dir + `os.replace`. Used 21 times across `config.py` and `web/app.py`. Rollback path (`main.py:3600-3620`) is a three-step `shutil.move` swap with a `_swap_tmp` intermediate, idempotent across retries. SQLite is journalled (WAL) by default. | — | High |
| E2 | No destructive operations without confirmation | **PASS** | `deleteExperiment` (`dashboard.html:161-182`, mirrored in `experiment.html:4755`) routes through `mlfl.confirm()` modal before POSTing `/api/experiments/<name>/delete`. The confirm copy spells out the consequences ("Removes the entry from mlfl.yaml and clears its forecast log and benchmark history. The sensor history cache and cached model weights are kept on disk."). Rollback button likewise gated. | — | High |
| E3 | Model files, user config, history survive restart / update | **PASS** | All under `/data/ml_forecast_lab/` (persistent add-on volume) and `addon_config/` (persistent config volume). `_restore_cached_models` (`main.py:5972`) rehydrates on boot. `load_all_benchmark_results` rehydrates benchmarks from SQLite. The schema-migration pattern at `db.py` is additive. | — | High |
| E4 | No code that could brick HA (filesystem writes outside `/data`, modifying HA config) | **PASS** | Filesystem writes are confined to `/data/ml_forecast_lab/*` and the add-on's own `addon_config` mount (under `/config/` inside the container — the addon's own writable directory under current HA conventions, not the HA core config). The `homeassistant_config:ro` mount is read-only at the supervisor level — the code only ever opens `/config/mlfl.yaml` for **reading** as a legacy fallback (`web/app.py:467`, `main.py:221`). No `os.system`, no `subprocess`, no shell-out. Asteval custom metrics use a sandboxed interpreter (`benchmark/metrics.py:545-551`). | — | High |

---

## F. Documentation honesty

| # | Item | Status | Evidence | Fix if not passing | Confidence |
|---|---|---|---|---|---|
| F1 | README describes what the add-on actually does; no aspirational features as present | **PASS** | Claims map to code: "24 backends" (registered in `models/registry.py`, all 24 backend files present); "calibrated 80% conformal bands" (`web/app.py` conformal endpoints, `db.py` conformal residual tables); "auto-generated Lovelace YAML" (`dashboard.py`, exposed via `/dashboard_yaml` route at `web/app.py:3499`); "promotion / retrain / rollback" (corresponding endpoints under `/experiment/<name>/`). The Hailo NPU is explicitly noted as **not wired in yet** in the add-on README. | — | High |
| F2 | Known limitations listed | **PASS** | Add-on README §"Is this for me?" covers history (≥2 weeks), disk (~2 GB), no GPU needed (and no NPU yet). DOCS troubleshooting section lists cold-start delays for conformal bands, first-build time, OOM mitigations, asteval expression gotchas. | — | High |
| F3 | System requirements accurate | **PASS** | "8 GB recommended" for Pi 5 is realistic for 24 backends + torch + lightgbm + xgboost. "First build 10-15 minutes on Pi 5" is a real estimate for the compile load (torch wheel for aarch64 + lightgbm + xgboost native build). Architectures listed match `config.yaml:10-13`. | — | High |
| F4 | "Beta" / "experimental" label if appropriate | **WARNING** | Version `2.34.4` and active CHANGELOG signal a mature project, but per the README badge comment block (`README.md:121-128`) the repo is **still private** and no `v2.34.x` tag has been pushed publicly. This will be a true first public release. New users have no other users' deployment experience to draw on. | Add a single line near the top of the add-on README acknowledging "First public release of a previously private project. Please open a GitHub issue if anything's off." or similar. Optional: `stage: experimental` in `config.yaml` (currently `stage: stable`) — but `stage: stable` is defensible given the long internal history. Recommend the README sentence rather than the stage flip. | High |

---

## G. Maintainability signal

| # | Item | Status | Evidence | Fix if not passing | Confidence |
|---|---|---|---|---|---|
| G1 | Contact path: GitHub issues enabled, or a stated channel | **PASS** | Add-on README `## Support` section at line 118-120 directs users to GitHub issues with a clear "include version + mlfl.yaml + last 50 log lines" template-in-prose. The repo URL is in `config.yaml:6` and `repository.yaml:3`. | — | High |
| G2 | Reasonable response expectation set | **WARNING** | No `SECURITY.md`, no `CONTRIBUTING.md`, no stated triage cadence. The README's Support section gives an issue checklist but no "I respond within …" pledge. For a free community add-on this is acceptable but absent. | One-line "Best-effort response; this is a side-project" in README, plus a thin `SECURITY.md` with a private-disclosure email/handle. Both can ship in v2.34.5. | High |
| G3 | CHANGELOG present with at least the current release described | **PASS** | `ml-forecast-lab/CHANGELOG.md` has 173 entries; the current `## 2.34.4` head describes the two fixes (convergence-chart X-axis bunching, trajectory ticks, lead-time legend reposition) in clear, user-readable prose. | — | High |
| G4 | Version bump and tag in place | **WARNING** | Version is consistently bumped to `2.34.4` across `config.yaml`, `__init__.py`, and `CHANGELOG.md` head (CI-enforced by `validate.yml`). **But the latest git tag on the remote is `v2.8.5`** — no tag exists for any 2.9–2.34 release. The `release.yml` workflow has been dormant for that entire range. For a HA add-on this is not strictly fatal (the supervisor reads `config.yaml`'s `version` to detect updates, not git tags), but the GitHub Releases page will be empty, the release-shield's `release-link` (`README.md:130`) will 404 for visitors clicking through, and the workflow is untested with the recent code. | Push a `v2.34.4` annotated tag (the release workflow expects an annotated message body) before flipping public, so the Releases page has one entry. Cadence afterwards: tag every release. | High |

---

## H. Distribution channel compliance

| # | Item | Status | Evidence | Fix if not passing | Confidence |
|---|---|---|---|---|---|
| H1 | HA add-on repository conventions met (icon/logo, panel_icon, name/description, version format) | **PASS (with note)** | `config.yaml` has `name`, `description`, `version` (SemVer), `slug`, `panel_icon` (`mdi:chart-timeline-variant-shimmer`), `panel_title`, `stage: stable`, `arch:` list. `icon.png` + `logo.png` are present at both repo root and add-on subdirectory. **Note**: `icon.png` and `logo.png` have identical MD5 sums (both 2127×2127). HA convention is `icon.png` square (~250×250) and `logo.png` wider banner — they should be different files. Cosmetic only; the supervisor accepts identical bytes. | Provide a 250×250 `icon.png` and a wider `logo.png` banner. Recommend before public flip but won't block install. | High |
| H2 | HACS rules met if targeting HACS | **PASS (N/A)** | HA add-ons are not distributed through HACS. | — | High |
| H3 | No conflicts with the slug of existing add-ons | **PASS** | The slug `ml_forecast_lab` is distinctive — no existing community add-on uses it (no other forecasting add-on in the `hassio-addons` or `home-assistant/addons` repos has this name). The folder name `ml-forecast-lab/` (kebab-case) and the slug `ml_forecast_lab` (snake-case) follow HA convention. | — | Medium-high |

---

## I. Polish floor

| # | Item | Status | Evidence | Fix if not passing | Confidence |
|---|---|---|---|---|---|
| I1 | No broken links in docs | **WARNING** | The auto-generated Lovelace YAML produced by `dashboard.py` (downloadable from the System page per `DOCS.md:222`) contains two broken references: **(a)** `dashboard.py:124` embeds `http://homeassistant.local:5052/experiment/<name>` in a markdown card, but direct port 5052 was removed in v2.30.0 (per `DOCS.md:336`); **(b)** `dashboard.py:39, 51` reference `_upper_95` / `_lower_95` sensors, but with the default `conformal_coverage: 0.8` the add-on actually publishes `_upper_80` / `_lower_80` (per `DOCS.md:201-202`). A user who downloads and imports the YAML will see entities-not-found + a dead link. | Either: fix `dashboard.py` to interpolate `conformal_coverage` into the sensor names and to omit the port-5052 link (point at the ingress entry point or remove the markdown card); or, if the feature is unused, remove the `/dashboard_yaml` route and the `_generate_dashboard` call at `main.py:5965`. | High |
| I2 | No TODO/FIXME/XXX in user-facing code paths | **PASS** | `grep -rnE "(TODO\|FIXME\|XXX\|HACK)"` across `ml_forecast_lab/` returns zero matches in production paths. | — | High |
| I3 | No debug logging at INFO level by default | **NOTE** | `logger.info` appears 311× vs `logger.debug` 74× across `ml_forecast_lab/`. The add-on does periodic operations (forecast every 30 min, retrain every 24 h), so steady-state info volume is bounded but not minimal. The rotating file handler (`__main__.py:137-143`) caps log file at 5 MB × 5 backups, so disk impact is bounded. Not severe enough to gate. | Optional pre-1.0 review: re-classify periodic operational announcements (`Refreshed models_enabled for …`, `Configuration reloaded (no changes) from …`) to DEBUG. Several are already at DEBUG per the comment at `main.py:253-256`. Audit the remainder. | High |
| I4 | No dev-only features exposed | **NOTE** | `AUDIT_PROMPT.md` at the repo root is an internal dev prompt template (a survivor of a "remove four stale internal audit documents" cleanup in v2.33.2 — see `b1184e67`). Not user-facing, but visible in the GitHub repo when public. | Delete `AUDIT_PROMPT.md` or move it into a `.github/dev-notes/` location with a clear "internal use" caveat. Same `__main__.py:169-217` `stub_server` path is dev/debug; reachable only if the main import fails, so practically harmless. | High |
| I5 | No `console.log` left in frontend | **PASS** | `grep -c "console\.log\|console\.debug\|console\.info"` across `web/templates/*.html` returns 0. Two `console.warn` and one `console.error` exist in `experiment.html` and `logs.html` — legitimate user-facing diagnostics retained on purpose. | — | High |

---

## Summary tally

| Severity | Count |
|---|---|
| BLOCKER | 0 |
| WARNING | 4 (F4 experimental label, G2 contact expectation, G4 missing v2.34.4 tag, I1 broken Lovelace YAML) |
| NOTE | 4 (B3 example sensors, H1 identical icon/logo, I3 INFO log volume, I4 `AUDIT_PROMPT.md`) |
| PASS | rest |

---

# First-release punch list (target: `v2.34.5`)

Order by user impact, lightest fixes first.

1. **Push `v2.34.4` git tag** (G4). Annotated, body = the changelog entry. This wakes the release workflow and gives the GitHub Releases page its first entry. ~2 min.
2. **Remove `AUDIT_PROMPT.md`** from the repo root (I4). Internal dev artefact. ~1 min.
3. **Differentiate `icon.png` and `logo.png`** (H1). Produce a 250×250 square `icon.png` and a wider banner `logo.png`. Update both the repo-root and `ml-forecast-lab/` copies. ~15-30 min.
4. **Fix the auto-generated Lovelace YAML** (I1). In `dashboard.py`:
   - Drop the `homeassistant.local:5052` markdown card or replace its URL with a note pointing at the add-on's ingress sidebar entry.
   - Pass the experiment's `conformal_coverage` through to the series-name lookup so the generated YAML names match the actually-published sensors (`_upper_80`/`_lower_80` by default).
   - Or remove the feature entirely if it's not earning its keep. ~30-60 min.
5. **Add a one-line "first public release" caveat** to the top of `ml-forecast-lab/README.md` (F4) — sets expectations that pre-2.34.4 versions have no external installs to draw bug reports from. ~5 min.
6. **Add a thin `SECURITY.md` and a one-line "best-effort" support note** to the README (G2). Sets a maintenance expectation that matches reality. ~10 min.

Optional, ship anytime in the 2.34.x series:

7. Audit INFO-level logging once (I3) — re-classify steady-state periodic announcements to DEBUG.
8. Swap the bundled `mlfl.yaml` example sensors from the author's `mixergy_*` to a generic `sensor.power_consumption_w` example (B3 note).

---

# 30-minute runtime verification

The static gate above flags two items that **must** be confirmed on a real HAOS instance before the public flip, plus a smoke-walk of the install-and-first-forecast path.

### Prereqs
- HAOS 2024.x or later running on aarch64 (Raspberry Pi 5 / 8 GB) **or** amd64 (a VM is fine for amd64; ideally also a Pi for aarch64). HA's supervisor handles the build locally so no GHCR push step is needed.
- A target sensor with at least 14 days of recorder history. A `sensor.processor_use_percent` or any always-on numeric sensor on the HA host works.

### Steps (target: 30 min including build wait)

1. **(5 min) Repo install.** Settings → Add-ons → Add-on store → ⋮ → Repositories → paste `https://github.com/psweens/ml-forecast-lab` → Add. **Expected:** the *ML Forecast Lab* add-on appears in the store. *Fail signal:* "Could not load add-on info" — `repository.yaml` or `config.yaml` is malformed.

2. **(10-15 min) Install build.** Click Install on the add-on card. **Expected:** the build log streams compile output for aarch64 (LightGBM, XGBoost, PyTorch native wheels). Completes without `manifest unknown` for the `build_from` image. *Fail signal:* a `pull failed` or `manifest unknown` error → the `build_from` path is wrong for the host's arch.

3. **(1 min) Container starts.** Start the add-on. Open the **Log** tab. **Expected within 30 s:** the banner box (`╔══ ML Forecast Lab v2.34.4 ══╗`), a `Directory ready: /data/ml_forecast_lab/...` (DEBUG, surface with log_level: debug if you want to see it), `HistoryDB initialised at /data/ml_forecast_lab/history.db`, and `Starting web server on 0.0.0.0:5052...`. *Fail signal:* a `Fatal error: …` with `Read-only file system` would mean my analysis of the `addon_config:rw → /config` mount was wrong — the fix is to retarget the `/config/ml_forecast_lab` directory in `_setup_directories` (`main.py:6140`) to `/data/ml_forecast_lab/config` or drop it.

4. **(1 min) Ingress reachable.** Click **Open Web UI** on the add-on page. **Expected:** the dashboard renders inside the HA chrome at `/api/hassio_ingress/<token>/`. Static assets (htmx, plotly, icon, style.css) load 200 with the `X-Ingress-Path` base prefixed correctly. *Fail signal:* 404 on `/static/*` → `_get_base_path` not being read or templates not using `BASE_PATH`.

5. **(2 min) Create an experiment via the UI.** Click **Add Experiment**. Pick the test sensor as `target_entity`, leave the rest as defaults. Submit. **Expected:** dashboard card appears for the new experiment; under the hood, `addon_config/mlfl.yaml` now exists with one experiment. *Fail signal:* "Failed to write config" toast → atomic-write path under unexpected mount permissions.

6. **(5 min) Run the benchmark.** Open the new experiment → Run Pipeline. With defaults (`lightgbm, xgboost, lstm, cnn`) and 14 days of 30-min data, expect 3-6 minutes on a Pi 5. **Expected:** the Results tab shows a rank table with all four models; logs show `[BENCH]` lines and a `BENCH benchmark completed` final line. *Fail signal:* any backend OOM or "Not enough data" — note which one and which message.

7. **(1 min) Promote and observe production cycle.** Click **Publish** on the top-ranked model. The mode flips to `production`. Wait one forecast cycle (default 30 min — for the verification you can edit `mlfl.yaml` to `forecast_every_minutes: 1` via the Settings tab to shorten this). **Expected:** within one cycle, `sensor.mlfl_<name>_forecast` appears in HA's Developer Tools → States, with attributes including the future curve. *Fail signal:* sensor missing → `[HA]` log lines for publish errors.

8. **(1 min) Restart the add-on.** Click Restart. **Expected:** the dashboard, the experiment, and the promoted-model state all survive. The retrain timer remembers where it was. *Fail signal:* experiment vanishes → state isn't being read back from `addon_config/mlfl.yaml`.

If steps 1-8 all pass, the gate's NEEDS-RUNTIME-VERIFICATION items convert to PASS.

---

# Post-release watch list

These are the items below the v2.34.5 punch-list line — pick them up across the first few patch releases as time permits. None of them gates the release.

- **Tag-and-release cadence**. Make sure every version bump in `config.yaml` from now on is followed by an annotated `v<version>` tag push within the same merge. The `release.yml` workflow needs it, and the README's release badge will start to drift visually once the dynamic-badge swap is done (see `README.md:128-138`).
- **Repo public-visibility flip**. Once public, swap the **static** shields (release-shield, licence-shield, tests-shield in `README.md:129-133`) for the **dynamic** equivalents already in the comment block (`README.md:135-138`). Cosmetic, but the README explicitly cites this as a known limitation of running against a private repo.
- **Issue / PR templates**. Add minimal `.github/ISSUE_TEMPLATE/bug_report.md` and `.github/ISSUE_TEMPLATE/feature_request.md` so users land in a structured form (add-on version + `mlfl.yaml` excerpt + last 50 log lines). Mirrors what the README's `## Support` already asks for, just enforced by GitHub.
- **`CONTRIBUTING.md`**. One page: how to run the tests (already documented in root README's Development section), branching policy, version-bump rules (which file to edit, who runs the tag).
- **Bundled example `mlfl.yaml` neutrality**. Swap `sensor.mixergy_*` for a generic example like `sensor.power_consumption_w` so first-time users don't accidentally land author-specific entity references in their config (cosmetic; first start through the UI replaces it anyway).
- **Log-volume audit**. Re-classify periodic operational INFOs (`Refreshed models_enabled …`, `Configuration reloaded (no changes) …`) to DEBUG. Should reduce steady-state log lines per hour by ~3-5×.
- **`stage: stable` revisit**. Six months after public flip, with real installs running, reconsider whether `stage: stable` (`config.yaml:7`) is honest. If issues surface, drop to `stage: experimental` for the next patch and back to `stable` once they settle.
- **Hailo NPU**. The add-on README mentions it as "optional and not wired in yet." If wiring it stays out of scope for the public release window, leave the mention as-is. If it lands, update README + DOCS + add the NPU detection to `_apply_runtime_resources`.
- **Translations beyond `en.yaml`**. `ml-forecast-lab/translations/en.yaml` carries one string. Not worth filling out until there's user demand for de/fr/nl, but worth keeping in mind once you start fielding non-English issues.
- **Optional `image:` field for prebuilt images**. The add-on currently builds locally on each user's host (10-15 min first install on a Pi 5). If install friction shows up in issues, you can add a CI step to publish prebuilt images to GHCR per arch and add `image: ghcr.io/psweens/{arch}-ml-forecast-lab` + `codenotary` to `config.yaml`. Not needed for v1.
