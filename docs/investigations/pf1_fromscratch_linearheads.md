# PF1 from-scratch retrain — linear-head backends

Dataset: `realistic_pv`. Noise-free signal peaks at hour 12 (UTC); empirical truth peak hour varies per holdout due to cloud noise.

|backend|baseline MAE|baseline flat|baseline peak|PF1 MAE|PF1 flat|PF1 peak|peak fixed?|flatness lift|
|---|---:|---:|---:|---:|---:|---:|---|---|
|nlinear|129.2|0.82|12|223.3|0.96|12|yes|1.18x|
|dlinear|81.9|0.88|12|99.8|0.87|12|yes|0.99x|
|sparsetsf|107.3|0.81|12|169.8|0.72|12|yes|0.89x|
|fits|156.5|0.70|12|156.5|0.70|12|yes|1.00x|
|tsmixer|103.2|0.85|12|130.0|0.83|12|yes|0.97x|
|timemixer|127.8|0.79|12|132.3|0.78|12|yes|0.99x|
|tide|104.2|0.87|12|202.1|0.91|12|yes|1.05x|
|itransformer|136.3|0.76|12|146.1|0.78|12|yes|1.02x|
|timesnet|128.0|0.86|12|140.1|0.85|12|yes|0.99x|
|tft|124.3|0.79|12|135.6|0.80|12|yes|1.01x|
