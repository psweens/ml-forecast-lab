# Model Guide — Picking Backends to Benchmark

ML Forecast Lab ships with 24 model backends. You don't need all of them — that's a lot of compute for a Pi, and many overlap in behaviour. This guide is a practical "which should I enable?" pre-flight that takes about 5 minutes to read.

The short version: **start with `lightgbm`, `xgboost`, `lstm`, and `cnn`.** Add more once you've seen how those do on your data.

## The 24 backends at a glance

| Family | Backend | Strength | Weakness | Speed |
|---|---|---|---|---|
| Tree | `lightgbm` | All-rounder, handles tabular features beautifully | Can over-rely on calendar features if covariates weak | Fast |
| Tree | `xgboost` | Like LightGBM but slightly different bias | Same | Fast |
| Tree | `catboost` | Robust to default hyperparams, handles categorical natively | Slow on Pi | Slow |
| Recurrent | `lstm` | Captures long temporal dependencies, well-understood | Can be unstable on small data, slow vs newer linear baselines | Medium |
| Recurrent | `gru` | Lighter than LSTM, often comparable | Same caveats as LSTM | Medium |
| Convolutional | `cnn` | WaveNet-style dilated causal convs, strong on cyclical patterns | Less interpretable than trees | Medium |
| Convolutional | `timesnet` | 2D-vision backbone for time series, captures multi-period seasonality | Heavy, ~2-3× slower than CNN | Slow |
| Linear / MLP | `dlinear` | Decomposition-Linear (Zeng 2023): simple, surprisingly competitive | Limited capacity | Very fast |
| Linear / MLP | `nlinear` | Variant of DLinear, sometimes wins on stationary series | Same limited capacity | Very fast |
| Linear / MLP | `tsmixer` | All-MLP mixer, strong on multivariate | Newer — less battle-tested | Fast |
| Linear / MLP | `timemixer` | Multi-scale TSMixer variant | Same | Fast |
| Linear / MLP | `tide` | Time-series Dense Encoder (Das 2023), strong on covariate-heavy targets | Tuning-sensitive | Fast |
| Linear / MLP | `sparsetsf` | Sparse TS Forecasting, parameter-efficient | Newer paper | Very fast |
| Frequency-domain | `fits` | ~10k params, frequency-domain interpolation | Niche — wins on highly seasonal targets | Very fast |
| N-BEATS | `nbeats` | Neural basis expansion — strong empirical record | Heavy, slow training | Slow |
| N-BEATS | `nhits` | Neural hierarchical interpolation, often beats N-BEATS | Same | Slow |
| Transformer | `patchtst` | Patch-based transformer, often top of academic benchmarks | Slow, needs more data | Slow |
| Transformer | `itransformer` | Variable-as-token inversion, competitive on multivariate | Same | Slow |
| Transformer | `crossformer` | Cross-variable attention, multivariate-focused | Heavy | Slow |
| Transformer | `tft` | Temporal Fusion Transformer, interpretable variable selection | Heaviest in the catalogue, very slow | Very slow |
| Classical | `arima` (statsforecast) | Strong baseline on stationary univariate series | Univariate only — ignores covariates | Medium |
| Classical | `ets` (statsforecast) | Exponential smoothing, good seasonal decomposition | Univariate only | Medium |
| Classical | `theta` (statsforecast) | M3 competition winner, low-complexity | Univariate only | Fast |
| Baseline | `seasonal_naive` | "Tomorrow looks like last week" — sanity check | Trivial — but if it wins, your sophisticated models are wasted | Trivial |

## Decision flow

**Always include `seasonal_naive`.** It costs nothing and tells you whether the problem is even worth modelling. If your fancy transformer can't beat naive, the issue isn't the model — it's the data, the covariates, or the target framing.

**Then pick by data shape:**

- **<2 weeks of history** → just trees (`lightgbm`, `xgboost`) and `seasonal_naive`. Neural models will overfit.
- **2 weeks – 2 months** → add `lstm`, `cnn`, `dlinear`, `nlinear`. Skip the heavy transformers.
- **2 months – 6 months** → add `nhits`, `patchtst`, `tide`, `tsmixer`. This is the sweet spot for the modern architectures.
- **>6 months** → also try `tft`, `crossformer`, `timemixer` if you want to invest the compute.

**Then pick by target characteristics:**

- **Strong daily / weekly seasonality** (e.g. household load, water demand): trees + `nhits` + `fits` are usually winners.
- **Noisy, sparse, low SNR** (e.g. EV charging, intermittent appliances): trees + `seasonal_naive`. Neural models often struggle here — the noise overwhelms the signal.
- **Covariate-driven** (e.g. heating ~ outside temp, solar ~ irradiance): `tide`, `tft`, `tsmixer` shine when the target is mostly explained by external features.
- **Univariate, no good covariates**: `arima` and `ets` (statsforecast) are surprisingly hard to beat. Don't underestimate classical baselines.
- **Solar generation specifically**: tree models with the built-in solar-physics covariates (`include_clear_sky_irradiance`, `include_sun_elevation`) typically win. Pure neural backends without those covariates struggle to learn the day/night structure.

## Speed tradeoffs on a Pi 5

A typical 60-day, 30-min experiment with default hyperparameters takes roughly:

| Tier | Backends | Per-fold time |
|---|---|---|
| Fast | `seasonal_naive`, `dlinear`, `nlinear`, `theta`, `fits`, `sparsetsf` | < 5s |
| Medium | `lightgbm`, `xgboost`, `cnn`, `gru`, `tsmixer`, `timemixer` | 5–30s |
| Slow | `lstm`, `nbeats`, `nhits`, `patchtst`, `itransformer`, `tide`, `arima`, `ets`, `catboost`, `timesnet` | 30s–2min |
| Very slow | `tft`, `crossformer` | 2–10min |

With 5 CV folds, multiply each by 5. A "throw everything at it" benchmark with all 24 backends enabled takes 1-2 hours on a Pi 5. A more reasonable setup with 6-8 selected backends finishes in 10-20 minutes.

## Pragmatic starter sets

**The "minimum viable" set** (good for any sensor, fast enough for a Pi):
```yaml
models_enabled:
  - seasonal_naive   # baseline sanity check
  - lightgbm         # tree all-rounder
  - xgboost          # second tree opinion
  - lstm             # neural reference
  - dlinear          # linear baseline — often surprisingly good
```

**The "I have decent data and want a real benchmark" set:**
```yaml
models_enabled:
  - seasonal_naive
  - lightgbm
  - xgboost
  - lstm
  - cnn
  - nhits            # often the best modern neural for covariate-rich targets
  - patchtst         # transformer reference
  - tide             # if you have strong covariates
```

**The "univariate target, classical baselines" set:**
```yaml
models_enabled:
  - seasonal_naive
  - arima
  - ets
  - theta
  - lightgbm         # tree comparison even on univariate
  - dlinear          # linear comparison
```

## After the benchmark

Open the **Models** tab — composite Demšar rank ranks across MAE / RMSE / MASE simultaneously. The winner is the model whose rank is most consistent across folds and metrics, not just the lowest single-metric error.

If two models tie within rank 1.0–1.4, prefer the simpler / faster one. The trees (`lightgbm`, `xgboost`) usually edge out the heavy transformers on consumer-grade data unless you have months of high-resolution history.

If `seasonal_naive` wins or comes close, that's actionable information: your problem is mostly periodic, and the "fancy" models aren't earning their compute. Stick with naive or trees.

## Tuning

Once you've identified a winner, the **Tuning** tab runs Bayesian optimisation (Optuna TPE) on its hyperparameters. Default tuning budget is 40 trials per model — enough to rank hyperparameter configurations without wasting Pi compute. The "Apply Tuned Params, Promote & Retrain" button does what it says: writes the tuned params back to `mlfl.yaml`, promotes the model to production, and kicks off an immediate retrain so the new sensors start publishing within minutes.
