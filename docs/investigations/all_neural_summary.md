# All neural backends on `realistic_pv` (v2.36.0 path)

Dataset: synthetic Watt-scale PV. extended_window=True, use_revin=True (defaults).

Verdict: OK / PHASE_OFF (peak hour off >1h) / COLLAPSED (flat<0.3) / OVER_VARY (flat>1.5) / HEAVILY_BROKEN (both) / FAILED.

|backend|verdict|peak_truth|peak_pred|flatness|MAE|night MAE|day MAE|
|---|---|---:|---:|---:|---:|---:|---:|
|nlinear|OK|11|12|0.70|198.7|72.2|488.2|
|dlinear|OK|11|12|0.66|166.7|25.3|514.1|
|sparsetsf|OK|11|12|0.71|178.9|51.5|507.3|
|fits|OK|11|11|0.50|289.2|99.2|854.0|
|tsmixer|OK|11|12|0.78|171.2|63.0|407.5|
|timemixer|OK|11|12|0.69|190.5|100.5|497.5|
|tide|OK|11|12|0.70|188.2|52.5|512.4|
|lstm|HEAVILY_BROKEN|11|1|0.08|528.4|363.0|1226.0|
|gru|HEAVILY_BROKEN|11|8|0.06|503.9|292.9|1245.1|
|cnn|COLLAPSED|11|10|0.03|526.3|352.0|1223.4|
|patchtst|HEAVILY_BROKEN|11|13|0.13|489.8|310.9|1090.3|
|itransformer|OK|11|12|0.58|220.7|80.4|634.7|
|crossformer|COLLAPSED|11|11|0.25|444.5|232.3|1095.2|
|timesnet|OK|11|12|0.57|230.8|79.1|645.1|
|tft|OK|11|12|0.74|158.6|77.5|420.8|
|nbeats|COLLAPSED|11|10|0.11|452.6|194.9|1219.1|
|nhits|HEAVILY_BROKEN|11|3|0.08|492.7|168.0|1455.5|
