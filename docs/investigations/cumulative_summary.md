# All neural backends on cumulative-with-daily-reset target

Dataset: ``make_cumulative_daily_reset(0)`` — interval form (per-30min household demand), trained with ``target_is_nonnegative`` implied for the harness. True peak interval hour: 19 (evening). Reset at midnight. Ideal flatness ≈ 1.0.

``daily_total_mape`` = how far the model's predicted day-total (sum of intervals over 24h) is from truth. This is the cumulative-target user's key metric because the addon re-cumsums interval predictions for display.

|backend|verdict|peak_truth|peak_pred|flatness|MAE|daily_total_mape|
|---|---|---:|---:|---:|---:|---:|
|nlinear|OK|19|19|0.57|0.0291|27.7%|
|dlinear|OK|19|19|0.65|0.0230|27.0%|
|sparsetsf|PHASE_OFF|19|15|0.46|0.0515|56.4%|
|fits|OK|19|19|0.49|0.0337|44.2%|
|tsmixer|OK|19|19|0.64|0.0215|28.4%|
|timemixer|OK|19|19|0.62|0.0209|27.8%|
|tide|OK|19|18|0.63|0.0288|27.7%|
|lstm|COLLAPSED|19|18|0.23|0.0401|36.8%|
|gru|COLLAPSED|19|18|0.17|0.0440|37.9%|
|cnn|COLLAPSED|19|18|0.19|0.0395|36.6%|
|patchtst|COLLAPSED|19|19|0.22|0.0406|26.8%|
|itransformer|OK|19|19|0.58|0.0260|32.2%|
|crossformer|OK|19|19|0.37|0.0336|29.8%|
|timesnet|OK|19|19|0.70|0.0218|27.4%|
|tft|OK|19|19|0.61|0.0247|30.5%|
|nbeats|OK|19|19|0.74|0.0410|35.3%|
|nhits|OK|19|19|0.78|0.0385|33.8%|
