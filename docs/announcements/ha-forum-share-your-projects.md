# HA Community Forum — Share your Projects

**Target board:** https://community.home-assistant.io/c/projects/9
**Status:** Draft for first post.

Tips for posting:
- Upload `docs/images/dashboard.png` directly to Discourse (don't hot-link from GitHub — the forum re-hosts and indexes inline images).
- Don't tag `@anyone`. Don't cross-link Reddit in the same post; do those separately.
- Discourse strips most BBCode but supports standard markdown and `<details>` blocks.
- Reply to the first 2–3 comments within a day — early engagement materially affects how Discourse surfaces the thread.

---

## Title

> **ML Forecast Lab — benchmark 24 ML models on any HA sensor, deploy the winner automatically**

(Alternative if the first feels too long: *"New add-on: ML Forecast Lab — multi-model forecasting for any HA sensor"*)

---

## Body

Hi all,

I've just opened the repo on an add-on I've been building privately for a while: **ML Forecast Lab**. It trains every enabled forecasting model on one of your HA sensors, ranks them on identical cross-validation folds, and lets you promote the winner to production with one click — after which it retrains on schedule and publishes forecasts back to HA as companion sensors with 80% prediction bands.

![Dashboard with three production experiments](upload://dashboard.png)

The mindset is **benchmark once, run forever**. You point it at a sensor, leave it overnight, click Promote on the winner, and from then on it just keeps a fresh forecast in HA. Re-benchmark when the sensor's behaviour changes or you want to try newer architectures.

### What you can forecast

Anything that's a numeric `sensor.*` with a few weeks of history. Some examples that work today:

- Solar / PV production (1h or 30min)
- Indoor temperature / humidity per room
- Daily and weekly energy consumption
- Heat-pump COP and flow temperature
- Battery state-of-charge trajectory
- Tank temperatures, well-pump runtime, anything seasonal

The point isn't a hand-tuned PV model — it's that *one* add-on covers all of these without you picking the algorithm.

### What's under the hood

24 backends are wired in and benchmarked on the same folds:

- **Trees:** LightGBM, XGBoost, CatBoost
- **Recurrent:** LSTM, GRU
- **Convolutional:** CNN, TimesNet
- **Linear / MLP:** DLinear, NLinear, TSMixer, TimeMixer, TiDE, SparseTSF
- **N-BEATS family:** N-BEATS, N-HiTS
- **Transformers:** PatchTST, iTransformer, Crossformer, TFT
- **Classical:** AutoARIMA, AutoETS, AutoTheta (via Nixtla `statsforecast`)
- **Frequency-domain:** FITS
- **Baseline:** Seasonal Naive

Ranking is a composite Demšar score across MAE / RMSE / MASE so no single metric can hand a backend a fake win, and the published bands are conformal — calibrated on held-out folds rather than the asymptotic Gaussian intervals most libraries default to.

### Hardware reality

Built and tuned for the **Pi 5 / 8 GB RAM / no GPU / ARM64** sweet spot. Also runs on `amd64` and `armv7`. First build is 10–15 minutes on a Pi 5; subsequent updates use the cached image. A full 24-backend benchmark on a year of hourly data finishes overnight; the production retrain cycle on the promoted winner is minutes.

### Install

Add the repo: **Settings → Add-ons → Add-on store → ⋮ → Repositories**, paste:

```
https://github.com/psweens/ml-forecast-lab
```

Then install **ML Forecast Lab** from the store. Full docs render on the Info / Documentation tabs once installed.

### Status & what I'm looking for

It's a first public release after 175 internal versions in a private repo, so the codebase is stable but the public user base is tiny. What would actually help right now:

- **Try it on one sensor** and tell me what surprised you — good or bad.
- **Which backend won** on your data (the UI shows the full ranking) — I'm collecting these to sharpen the defaults guide.
- **Bug reports** with the last 50 lines of the add-on log; phase tags (`[BENCH]`, `[MODEL]`, `[HA]`, `[PUB]`) make triage fast.

GitHub: https://github.com/psweens/ml-forecast-lab
MIT licensed, no cloud, no fee, no telemetry.

Happy to answer questions in the thread.

— Paul

---

## After posting

- Note the topic URL here for tracking.
- Reply to the first wave of comments same day.
- After 48h, if engagement is healthy, write the r/homeassistant variant (don't reuse the same body verbatim — Reddit prefers more compressed openings).
- If a question gets asked twice, fold the answer into `ml-forecast-lab/DOCS.md` rather than just replying inline.
