<div align="center">

<img src="logo.png" alt="ML Forecast Lab" width="180">

# ML Forecast Lab

**Multi-model machine learning forecasting for Home Assistant.**

Train, benchmark, and deploy time-series models for any HA sensor — with academic-standard evaluation built in.

[![Latest release][release-shield]][release-link]
[![Licence][licence-shield]][licence-link]
[![Tests][tests-shield]][tests-link]
[![Home Assistant add-on][ha-shield]][ha-link]
[![Architectures][arch-shield]][release-link]

</div>

---

## What this repository is

A Home Assistant add-on for HA power users who want to forecast a sensor — not data scientists who want a fresh framework. Plug in any sensor, the add-on benchmarks 24 model backends on your data, picks the winner, retrains it on schedule, and publishes forecasts back to HA with calibrated 80% bands. Designed for the Pi 5 sweet spot: 8 GB RAM, no GPU, ARM64.

The intended mindset is **benchmark once, run forever**. After the first benchmark you click Promote, and the add-on takes care of retraining + publishing — re-benchmark only when your sensor's behaviour drifts or you want to try newer architectures.

If you're browsing on GitHub, the install button below is the fastest path. The per-add-on [README](ml-forecast-lab/README.md) and [DOCS](ml-forecast-lab/DOCS.md) render on HA's **Info** and **Documentation** tabs once installed.

## Install

[![Open your Home Assistant instance and show the add add-on repository dialog with this repository pre-filled.][openhainstall-shield]][openhainstall-link]

Or add the repository manually: **Settings → Add-ons → Add-on store → ⋮ → Repositories**, then paste `https://github.com/psweens/ml-forecast-lab`.

First build takes 10–15 minutes on a Raspberry Pi 5. Subsequent updates use the cached image.

**Supported architectures:** `aarch64` (Raspberry Pi 4/5), `amd64` (x86-64 servers), `armv7`.

## What it does

ML Forecast Lab trains every enabled forecasting backend on your sensor's history, ranks them on identical cross-validation folds with a composite Demšar score across MAE / RMSE / MASE, and shows you which one wins. You promote the winner to **production**, and the add-on retrains it on schedule and publishes forecasts back to Home Assistant as companion sensors with calibrated 80% conformal prediction bands.

24 backends are wired in: tree (LightGBM, XGBoost, CatBoost), recurrent (LSTM, GRU), convolutional (CNN, TimesNet), linear / MLP (DLinear, NLinear, TSMixer, TimeMixer, TiDE, SparseTSF), N-BEATS family (N-BEATS, N-HiTS), transformers (PatchTST, iTransformer, Crossformer, TFT), classical (AutoARIMA, AutoETS, AutoTheta), frequency-domain (FITS), and a Seasonal Naive baseline. See [`docs/MODEL_GUIDE.md`](docs/MODEL_GUIDE.md) for picking the right ones.

<details>
<summary><b>Architecture</b></summary>

```
                     mlfl.yaml
                         │
                         ▼
         ┌──────────────┐    ┌──────────────────┐
         │ HA Interface │───▶│ History Database │
         │ (API client) │    │     (SQLite)     │
         └──────────────┘    └────────┬─────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │ Feature Engineer │
                            │ lags + temporal +│
                            │   covariates     │
                            └────────┬─────────┘
                                     │
                ┌────────────────────┼─────────────────────┐
                ▼                    ▼                     │
       ┌─────────────────┐   ┌─────────────────┐           │
       │ Cross-Validator │   │ Model Registry  │           │
       │ walk-fwd or SW  │   │ 24 backends     │           │
       └────────┬────────┘   │ tree+neural+cls │           │
                │            └─────────────────┘           │
                ▼                                          │
      ┌──────────────────┐         ┌──────────────────┐
      │   Benchmarker    │────────▶│   Model Cache    │
      │ Demšar ranking   │         │ (per experiment) │
      └────────┬─────────┘         └────────┬─────────┘
               │                            │
               ▼                            ▼
      ┌──────────────────┐         ┌──────────────────┐
      │      Web UI      │         │ Forecast Cycle   │
      │   (HA ingress)   │         │ (every 30 min)   │
      └──────────────────┘         └────────┬─────────┘
                                            ▼
                                   ┌──────────────────┐
                                   │   HA Sensor      │
                                   │   Publisher      │
                                   └──────────────────┘
```

Retrain (default 24 h) trains all enabled models from scratch and refreshes the cache. Forecast (default 30 min) uses the cached model for fast inference without retraining.

</details>

## Documentation

| Where it lives | What it covers |
|---|---|
| [`ml-forecast-lab/README.md`](ml-forecast-lab/README.md) | HA store **Info** tab — what the add-on is, hardware requirements, install, minimal `mlfl.yaml`, first forecast. |
| [`ml-forecast-lab/DOCS.md`](ml-forecast-lab/DOCS.md) | HA store **Documentation** tab — full configuration reference, published sensors, web-UI tour, operations, troubleshooting. |
| [`ml-forecast-lab/CHANGELOG.md`](ml-forecast-lab/CHANGELOG.md) | HA store **Changelog** tab — per-version release notes. |
| [`docs/MODEL_GUIDE.md`](docs/MODEL_GUIDE.md) | Practical "which of the 24 backends should I enable?" with starter sets keyed to data volume, target shape, and Pi compute budget. |

## Development

Tests run locally without an HA instance:

```bash
cd ml-forecast-lab
pip install -r requirements.txt -r tests/requirements-dev.txt
pytest tests/                # 185 tests, ~10s
pytest tests/smoke/          # 61 smoke tests, ~2s — release gate
pytest tests/unit/           # 124 unit tests
```

Both suites are wired into GitHub Actions on every PR and main push (`tests.yml`). The smoke suite boots the FastAPI app against a tmp `mlfl.yaml` and walks the eight golden user flows without needing trained models — designed as a fast release gate that catches UI/API regressions before they ship.

## Licence

[MIT](LICENSE) © Dr Paul W. Sweeney, University of Cambridge.

<!-- Badges -->
<!-- Badge URLs.
     The release / licence / tests badges are STATIC text while the
     repo remains private (shields.io is unauthenticated and can't
     read private-repo GitHub APIs — every dynamic badge rendered as
     "repo not found"). Refresh `release-shield` on each version bump;
     `tests-shield` only needs a manual flip if the suite ever starts
     failing. When the repo is flipped to public, swap these back to
     the dynamic equivalents (commented below). -->
[release-shield]: https://img.shields.io/badge/release-v2.34.0-41bdf5?style=flat-square
[release-link]: https://github.com/psweens/ml-forecast-lab/releases/latest
[licence-shield]: https://img.shields.io/badge/licence-MIT-41bdf5?style=flat-square
[licence-link]: LICENSE
[tests-shield]: https://img.shields.io/badge/tests-passing-41bdf5?style=flat-square
[tests-link]: https://github.com/psweens/ml-forecast-lab/actions/workflows/tests.yml
<!-- Dynamic equivalents (re-enable once the repo is public):
[release-shield]: https://img.shields.io/github/v/release/psweens/ml-forecast-lab?style=flat-square&color=41bdf5
[licence-shield]: https://img.shields.io/github/license/psweens/ml-forecast-lab?style=flat-square&color=41bdf5
[tests-shield]: https://img.shields.io/github/actions/workflow/status/psweens/ml-forecast-lab/tests.yml?branch=main&style=flat-square&label=tests&color=41bdf5
-->

[ha-shield]: https://img.shields.io/badge/Home%20Assistant-add--on-41bdf5?style=flat-square&logo=home-assistant&logoColor=white
[ha-link]: https://www.home-assistant.io/
[arch-shield]: https://img.shields.io/badge/arch-aarch64%20%7C%20amd64%20%7C%20armv7-41bdf5?style=flat-square
[openhainstall-shield]: https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg
[openhainstall-link]: https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fpsweens%2Fml-forecast-lab
