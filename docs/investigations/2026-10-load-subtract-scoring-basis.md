# Load subtract: scoring basis and cumulative training labels (v2.55.0)

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

## Training labels

Replaying 28 days of the same install's history through
`cumulative_to_interval` and `apply_load_subtract` found two places where
the training label departed from the use the counters recorded. Both are
fixed in the same release. Unlike the scoring fix above, these change what
models are trained on.

### The default increment cap was a size

With `max_increment` unset, each reading's rise was capped at the 95th
percentile of reading-to-reading rises (0.2 kWh on the house counter). A
size cap cuts the largest 5% of rises by construction, so it removed real
use wherever a large genuine rise built up between two readings: a counter
that polls irregularly, or a change-only counter whose use arrives in
bursts. The subtract sensors always used the default because their cap is
not configurable.

The default is now a rate (`_rate_based_cap`): twice the 99th percentile of
the counter's observed rates (rise ÷ time since the previous reading),
times that reading's own elapsed time, never below the median rise so the
counter's resolution step always passes. A glitch recovery runs hundreds of
times faster than normal use and is still caught. With fewer than 20
measured rates the cap falls back to the old 95th-percentile size. Rows
spanning more than 1.5 intervals stay exempt, as before.

A rate cap on its own overcounted the house counter by 2.4 kWh/day. Every
drop used to count as a restart, so each time the counter stepped back by
0.1 kWh the next reading scored its whole value as one interval's use, and
the tight size cap had been hiding that by cutting it to 0.2 kWh. Restart
detection therefore changed too:

- A restart is a drop below `COUNTER_RESTART_RATIO` (0.9) of the previous
  reading, the threshold `HistoryDB._COUNTER_RESTART_RATIO` and HA's
  `total_increasing` use. A test pins the two constants equal.
- Between restarts a reading's use is its rise above the highest reading
  since the restart, so a dip and its recovery net to zero. On a counter
  that never dips this equals the plain difference.

- With `reset_daily`, a reading crossing UTC midnight was also a restart
  if it rose by less than 10% of the counter's mean. A counter that resets
  at local midnight has already restarted by UTC midnight whenever local
  time is ahead of UTC, so in BST the first reading after 00:00 UTC scored
  its whole value — the use from 00:00 to 01:00 local, counted a second
  time. Only a drop across midnight is now a restart (any drop, so a daily
  counter whose previous total was small is still caught). On a synthetic
  0.4 kW night / 8 kW evening load at 0.01 kWh resolution, the rate cap
  with the old midnight rule overcounted by 0.4 kWh/day; the old size cap
  hid most of it (+0.26 kWh/day).

The restart rules apply to every cumulative source, including those that
set `max_increment`; the explicit cap keeps its fixed per-reading meaning.

Clean-day daily label error against the counter's own daily rise (mean
absolute error / mean bias), measured with the rate cap and the dip rule
but before the midnight rule changed:

| Counter | 95th-percentile size cap | Rate cap + dip rule |
|---|---|---|
| House load (inverter, daily reset) | 1.18 / −1.18 kWh | 0.34 / +0.34 kWh |
| Hot-water demand (%, daily reset) | 10.6 / −10.6 points | 0 / 0 points |
| Hot-water heat energy | 0.65 kWh | 0.01 kWh |
| Electric boiler | 0.15 kWh | 0.12 kWh |
| Car charger | ≈0.02 kWh | ≈0.03 kWh |

The remaining +0.34 kWh/day on the house counter matches the BST midnight
double count in sign and size (the 28 days were almost all BST, and that
hour's night-time use is 0.3–0.5 kWh), but the replay has not been re-run
with the midnight fix to confirm it. The install's own hot-water demand experiment
sets `max_increment: 100`, so its labels were already right; the 10.6-point
error is what the default would have produced, and what a subtract sensor
on that counter would have seen.

### Clipping after subtraction lost the subtracted energy

`apply_load_subtract` clipped each bin at zero. Over 21 days, 244 of 1008
half-hours went below zero, about 157 kWh in total, much of it while the
inverter's counter froze and then caught up in one step while the
subtracted sensors kept counting. A clipped bin's deficit was never
subtracted, so net labels overstated use: by 7.6 and 9.5 kWh on the two
days the immersion used 26–30 kWh. Cross-correlating the house counter with
each subtracted sensor showed no consistent lag, so shifting the series
would not have fixed it.

For a cumulative target, `_carry_deficit` now carries a negative bin's
deficit into the following bins of the same local day (`_local_dates`, from
`AppConfig.timezone`) until it is absorbed. Each day's net total is then
the parent's use minus the subtracted sensors' use whenever that is
non-negative. Unmeasured bins pass through as NaN and keep the deficit for
the next measured bin; `on_missing: drop` rows neither create nor absorb
one. A deficit still left at midnight is dropped and reported in the
fetch log's audit line, with the amount carried.

Trade-off: daily totals become exact, but within the day the deficit comes
off the next bins with use rather than the bins where the parent should
have recorded it. Non-cumulative (power) targets still clip each bin: a
sampled power reading has no later catch-up step to absorb the deficit.

### Cached models

A cached model is fed lag features built from these labels, so one trained
under the old rule would be served inputs on a different scale. Cache
metadata records `LABEL_RULE_VERSION` (2), and `_restore_cached_models`
ignores an older rule's cache for experiments with a cumulative target or a
`load_subtract`, which then retrain once. `_rollback_cached_model` refuses
a `previous/` generation from an older rule for the same experiments; the
first retrain after the update puts exactly such a model there, and
without the check a rollback would serve it until the next restart
rejected it. `CACHE_SCHEMA_VERSION` is
unchanged because bumping it would retrain every experiment. The gate is a
superset: a `load_subtract` experiment whose target and subtract sensors are
all interval sources retrains needlessly.

## Replay bundles

Replay bundles captured before v2.55.0 recorded conformal queries against
the raw target's table, so for a `load_subtract` experiment their
forecast stage now reports an unrecorded call.
