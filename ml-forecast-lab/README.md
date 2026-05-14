# ML Forecast Lab

**Multi-model machine learning forecasting for Home Assistant.**

Train and benchmark 24 forecasting backends on any HA sensor, then promote the winner. The add-on retrains it on schedule and publishes forecasts back to Home Assistant as companion sensors with calibrated 80% prediction bands.

The intended workflow is **benchmark once, run forever**. After the initial benchmark, production mode is set-and-forget; re-benchmark only when the sensor's behaviour drifts or you want to try newer architectures.

## Is this for me?

You're the right user if you have:

- A Raspberry Pi 5 (8 GB recommended) **or** an amd64 / armv7 HA host.
- At least **2 weeks** of recorder history for the sensor you want to forecast (4+ weeks is better — see the data-volume guidance in `docs/MODEL_GUIDE.md`).
- ~2 GB of free disk space for the add-on image plus model cache.
- Comfort editing a YAML file with the HA File editor / Studio Code Server.

You do **not** need:

- A GPU. Everything runs on CPU. A Hailo NPU is optional and not wired in yet.
- Existing ML knowledge. The defaults work for typical household sensors; the only choice you must make is which sensor to forecast.

## What you get

Once installed, you'll see an **ML Forecast Lab** entry in the HA sidebar that opens the web UI through ingress.

<!-- Screenshots — add files under ml-forecast-lab/images/ and uncomment.
     Suggested captures:
     - dashboard.png  : dashboard with one experiment in lab mode
     - experiment.png : Models tab showing a rank table
     - accuracy.png   : Forecast Accuracy tab with the verdict chip
-->
<!-- ![Dashboard](images/dashboard.png) -->
<!-- ![Experiment Models tab](images/experiment.png) -->
<!-- ![Forecast Accuracy](images/accuracy.png) -->

## Install

1. In Home Assistant, go to **Settings → Add-ons → Add-on store**.
2. Click the **⋮** menu (top right) → **Repositories**.
3. Add `https://github.com/psweens/ml-forecast-lab` and close the dialog.
4. **ML Forecast Lab** now appears in the store. Click **Install**.
5. First build takes **10–15 minutes on a Pi 5** — LightGBM, XGBoost, and PyTorch all compile native extensions for `aarch64`. Subsequent updates use the cached image.
6. **Do not start the add-on yet.** Create `mlfl.yaml` first (see below) so the add-on has an experiment to load on boot.

**Supported architectures:** `aarch64` (Raspberry Pi 4/5), `amd64` (x86-64 servers), `armv7`.

## Minimal configuration

Create `/addon_configs/ml_forecast_lab/mlfl.yaml` with one experiment. The minimum viable example:

```yaml
timezone: "Europe/London"

experiments:
  - name: household_load
    target_entity: sensor.power_consumption_w
    mode: lab                # start in lab to benchmark; switch to production once you've picked a winner
    interval_minutes: 30
    days_history: 30
    future_periods: 48       # 48 steps × 30 min = 24 h horizon

    models_enabled:
      - seasonal_naive       # baseline — always include
      - lightgbm
      - xgboost
      - lstm
      - dlinear
```

That is enough to run. Every other field has a sensible default. The full reference, including covariates, cumulative-sensor handling, solar-physics features, and load-subtract, is in [DOCS.md](DOCS.md).

If your target is an **energy-today** style sensor that resets at midnight, add:

```yaml
    source_is_cumulative: true
    reset_daily: true
    max_increment: 5.0       # max kWh you'd realistically see in one 30-min slot
```

## First forecast

1. **Start the add-on.** Click **Open Web UI** on the add-on page (the direct port 5052 is no longer exposed — access is via HA ingress).
2. **Run the benchmark.** Open your experiment → **Run Pipeline**. Every enabled model trains with walk-forward cross-validation. With 5 models and 30 days of history on a Pi 5 you'll see results in 5–15 minutes.
3. **Pick the winner.** The composite Demšar rank highlights one model. Override from the Models tab if you want, then click **Promote to Production**.
4. **Switch the experiment to production.** Edit `mlfl.yaml` and set `mode: production`. The add-on retrains every 24 h and publishes:

   - `sensor.mlfl_household_load_forecast` — next-interval forecast value.
   - `sensor.mlfl_household_load_upper_80` / `_lower_80` — conformal prediction bands (calibrated against recent residuals; appear after ~10 forecasts have been compared against actuals).
   - `sensor.mlfl_household_load_cumulative` — integrated forecast curve.
   - `sensor.mlfl_household_load_forecast_accuracy` — running accuracy summary.
   - `sensor.mlfl_household_load_last_benchmark` / `_last_retrain` — ISO timestamps you can trigger automations on.

5. **Watch it.** The Forecast Accuracy tab compares every logged prediction against the actual once it arrives. Bias, per-horizon error, and conformal coverage are tracked automatically.

## Next

- [DOCS.md](DOCS.md) — full configuration reference, web-UI tour, operations (logs, backup, rollback, reset), and the long-form troubleshooting list.
- [docs/MODEL_GUIDE.md](https://github.com/psweens/ml-forecast-lab/blob/main/docs/MODEL_GUIDE.md) — which of the 24 backends to enable for your data shape and Pi compute budget.
- [CHANGELOG.md](CHANGELOG.md) — release notes.

## Support

Open an issue at https://github.com/psweens/ml-forecast-lab/issues — please include the add-on version, the relevant section of `mlfl.yaml`, and the last 50 lines of the add-on log.
