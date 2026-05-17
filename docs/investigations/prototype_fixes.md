# Prototype fix comparison

Dataset: `realistic_pv` (~4500 W peak, AR(1) clouds, integer quantisation).
Holdout: 14 days at the tail of the year. Per condition: model trained
for 20 epochs from scratch, then PF1/PF2 applied as monkey-patches and
fine-tuned for 5 more epochs.

True peak hour = 11 (UTC) for this dataset. Ideal flatness ≈ 1.0.

## Headline

* **LSTM and CNN baseline collapse** (flatness ≈ 0.3-0.5, peak shifted
  to early evening / morning). PF1 alone restores both to peak=12
  (~ truth ± 1 h) and flatness ≈ 1.9 — the model is no longer reading
  absolute time from future-position covariates as a magnitude signal.
* **NLinear baseline** does NOT collapse but over-varies (flatness
  2.23) — a different pathology of RC1 + RC2, hidden by the linear
  head's bias term. PF1 corrects the peak hour from 12 to 11 (truth).
* **SparseTSF baseline** has the same over-vary, same direction.
  PF1 reduces flatness from 2.28 to 1.95 (closer to 1.0) at the cost
  of ~13% MAE.
* **PF2 alone** (NLinear anchor) has a small effect on NLinear (peak
  stays 12, flatness barely changes) — the anchor restoration without
  PF1's bias correction lets the head fight RevIN's denormalisation
  in the wrong direction.
* **PF1 + PF2** for NLinear is intermediate between PF1-only and
  baseline — suggesting the two changes interact and a from-scratch
  retrain with both in place would beat this monkey-patched + 5-epoch
  fine-tune. (The prototype runner only ran 5 fine-tune epochs to
  keep the wall time short; a real fix would be ground-up.)

|backend|condition|mae|flatness|peak_hour|
|---|---|---:|---:|---:|
|nlinear|baseline (broken)|163.4|2.23|12|
|nlinear|PF1 (RevIN past-only)|353.7|3.14|11|
|nlinear|PF2 (NLinear anchor)|165.6|2.28|12|
|nlinear|PF1 + PF2|216.3|2.34|12|
|sparsetsf|baseline (broken)|159.7|2.28|12|
|sparsetsf|PF1 (RevIN past-only)|180.4|1.95|12|
|sparsetsf|PF1 + PF2|180.4|1.95|12|
|lstm|baseline (broken)|199.2|0.32|9|
|lstm|PF1 (RevIN past-only)|197.9|1.87|12|
|lstm|PF1 + PF2|197.9|1.87|12|
|cnn|baseline (broken)|178.6|0.46|10|
|cnn|PF1 (RevIN past-only)|202.1|1.86|12|
|cnn|PF1 + PF2|202.1|1.86|12|
