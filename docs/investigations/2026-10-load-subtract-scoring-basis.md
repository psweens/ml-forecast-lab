# Load subtract: forecasts scored against the gross target (v2.55.0)

## What was wrong

`load_subtract` removes the subtracted sensors from the target inside
`_fetch_and_preprocess`, so training and inference work on the net series.
Every consumer that compares a forecast with what happened kept reading the
raw target instead:

| Consumer | Read before | Effect |
|---|---|---|
| `_cumulative` seed (`_publish_forecast_sensors`) | `get_state(target_entity)` | Net forecast added to a gross running total |
| `_forecast_accuracy` sensor | `safe_table_name(target_entity)` | MAE measured the subtracted loads |
| Conformal bands (`_conformal_bands`) | same | Residual quantiles sized from the subtracted loads |
| Forecast Accuracy, Comparison, trajectory, evolution routes (`web/app.py`) | same | Same, in the UI |

The training basis and the scoring basis were different series, so every
figure in the second column was dominated by the subtracted share of the
load rather than by forecast error.

## Measured on one production install

A `house_load` experiment: target is a daily-reset house-load counter, with
`load_subtract` removing an electric boiler (`zero`), a hot-water immersion
(`drop`) and a car charger (`zero`). Late October, so space heating was
running.

- At 20:11 the cumulative sensor reported `seeded_with: 49.7` kWh while use
  after subtraction was about 13 kWh. Its state was therefore about 37 kWh
  above the forecast it was meant to carry.
- On a morning when the inverter's daily counter reset three hours late,
  the seed was the previous day's gross total and the published day-ahead
  figure was 72 kWh against about 35 kWh of net use.
- Forecast Comparison reported a daily bias of −25 kWh/day for the app's
  own forecast and −16 to −22 kWh/day for three external forecasts. The
  actual was gross; none of the forecasts were.
- The accuracy sensor's next-interval MAE was 0.72 kWh against a per-bin
  net load of roughly 0.25–0.5 kWh. Its rise over two weeks tracked the
  heating season widening the gross–net gap, not a change in the model.
- Daily totals rebuilt from the four counters put the app's day-ahead
  forecast of net use at a median absolute error of 3.6 kWh/day and a
  median bias of −2.9 kWh/day over 12 days: a real, much smaller error
  that the gross scoring hid.

## Fix

`_record_net_actuals` runs right after `apply_load_subtract` in the fetch
path, on the series before outlier clipping and the log transform (what
the sensor measured once the loads are taken out):

- For a cumulative target it stores today's use so far (local day, from
  `AppConfig.timezone`) for the seed, and writes every grid bin's running
  total since local midnight to `net_actuals_<experiment>`. An unmeasured
  bin (NaN label, e.g. `on_missing: drop`) is written as NULL.
- For a non-cumulative target it writes each measured bin's value.
- Writes use `HistoryDB.upsert_history`, because the current bin and the
  day's total change between fetches. The first fetch of a run writes the
  whole fetched window (so history logged before the upgrade is scored
  correctly at once); later fetches rewrite the last two days.
- Retention is the experiment's `max_age`, via
  `_retention_tables_for_experiment`.

`HistoryDB.actuals_table_for(exp_cfg)` returns that table when
`load_subtract` is set and the raw target's table otherwise. Every reader
listed above now goes through it. Experiments without `load_subtract` read
exactly what they read before.

The table stores the shape the existing readers already handle for a
daily-reset counter, so no query changed shape:

- Increment mode differences consecutive rows; the midnight restart is a
  drop below 90% of the previous reading and is read as a reset, as for a
  raw counter.
- The Comparison tab grid-aligns and differences the rows, discarding the
  negative diff at the restart, as for a raw counter.
- The daily-cumulative accuracy path reads the running total directly.

One reader rule differs. For a raw counter a bin with no readings is zero
use, because the recorder only logs changes. A net table has a row for
every bin, so a missing bin can only be an unmeasured label.
`_materialise_actuals_increments` therefore sets bins holding a NULL row
to NULL for tables with the `net_actuals_` prefix, and they are excluded
like the window's first bin. Labels are never imputed.

## Not changed here, worth a follow-up

Two further sources of disagreement between the training label and what
was used, found while replaying the same install's history through
`cumulative_to_interval`:

- With `max_increment` unset, the per-reading spike cap defaults to the
  95th percentile of reading-to-reading rises (0.2 kWh on that counter).
  It clips real use whenever updates are spread out, and the subtract
  sensors always use the default because their cap is not configurable.
  Setting the target's cap to 0.5 halved the daily label error on clean
  days (0.68 → 0.36 kWh). A cap on power × elapsed time would fix the
  cause.
- `apply_load_subtract` clips each bin at zero after subtracting. When a
  subtracted sensor reports late, the catch-up bin's negative result is
  clipped and that energy is never subtracted. On the two days the
  immersion used 26–30 kWh this overstated net use by 7.6 and 9.5 kWh.

Replay bundles captured before v2.55.0 recorded conformal queries against
the raw target's table, so for a `load_subtract` experiment their
forecast stage now reports an unrecorded call.
