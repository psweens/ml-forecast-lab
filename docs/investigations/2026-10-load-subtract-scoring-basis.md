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

- It writes each grid bin's value — its use, for a cumulative target — to
  `net_actuals_<experiment>`. For a cumulative target an unmeasured bin
  (NaN label, e.g. `on_missing: drop`) is written as NULL, and today's use
  so far (local day, from `AppConfig.timezone`) is kept for the seed. For a
  non-cumulative target only measured bins are written.
- Writes use `HistoryDB.upsert_history`, because the current bin, and any
  bin a late reading or a carried deficit reaches, change between fetches.
  The first fetch of a run writes the whole fetched window (so history
  logged before the upgrade is scored correctly at once); later fetches
  rewrite the last two days.
- Only bins the fetch measured in full are written, because the first
  fetch after every restart rewrites the window over rows an earlier run
  wrote:
  - nothing, when a configured subtract sensor's fetch failed (net would be
    partly subtracted);
  - nothing before every subtract sensor's history starts (the leading
    gap in the `apply_load_subtract` audit). The target comes from the
    app's cache, but the subtract sensors are fetched from HA, whose
    recorder keeps less: with the defaults (`days_history` 14, recorder 10
    days) the oldest days would be written with nothing subtracted. Before
    this guard, a restart reverted correct rows on those days to gross
    (a probe day went from 28.5 to 47.5 kWh);
  - for a cumulative target, nothing before the first local midnight after
    that: the window opens mid-day, its first reading has no rise, and the
    deficit carried from earlier that day is missing.
- Retention is the experiment's `max_age`, via
  `_retention_tables_for_experiment`.

`HistoryDB.actuals_table_for(exp_cfg)` returns that table when
`load_subtract` is set and the raw target's table otherwise. Every reader
listed above now goes through it. Experiments without `load_subtract` read
exactly what they read before.

Readers recognise the `net_actuals_` prefix and take the rows as they are:

- Increment mode (accuracy, conformal bands, coverage, trajectory,
  evolution) builds its per-bin table from the rows directly
  (`_materialise_net_increments`) instead of differencing a counter.
- The Comparison tab does not difference them, and skips the
  recorder-quiet carry-forward, which would draw a bin that was never
  measured.
- Daily-cumulative accuracy sums the rows within each day, split by HA's
  time zone like the forecasts. Once a bin of the day is NULL or has no row,
  the running total is unknown for the rest of that day.

A bin with a NULL value or without a row is unknown, never zero use as for
a change-only raw counter. Labels are never imputed.

An earlier draft stored each bin's running total since local midnight, the
shape of a daily-reset counter, so the existing counter readers could read
it unchanged. Review found that this made every reader infer the day
boundary from the data:
- the increment reader saw a restart only where a day's first bin fell
  below 90% of the previous day's total. On a daily-interval experiment
  every bin is its own day, so a day using at least 90% of the day before
  scored the difference (11.9 kWh as 1.9). Simulated over 20 days, 9 days
  were scored wrong and 4 excluded, and a perfect forecast scored an MAE of
  5.4 kWh/day;
- the daily-cumulative view split days by HA's time zone while the totals
  restarted at `AppConfig.timezone` midnight (default UTC). With HA on
  UTC+1, perfect forecasts scored an end-of-day MAE of about 1 kWh;
- bins with no row, left by a write gap longer than the rewritten tail,
  were scored as zero use.

Per-bin rows remove the inference.

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

The rate estimate leaves out the first rise after a restart, and any rise
the next reading takes back. In an earlier draft it didn't, and a glitch's
own rate set the 99th percentile wherever glitches were at least about 1% of
readings, or the counter had fewer than about 100 of them. A lifetime
counter that briefly read 0 and recovered then scored its whole value in one
bin. With one glitch and 51 half-hourly readings, the bin was 4,117 kWh
(main: 1.7). A 15-minute counter glitching once a day for 14 days totalled
53,718 kWh against 335 kWh recorded.

A lifetime counter's drop that its next rise undoes is a glitch and its
readings are ignored (`_undone_drops`): the first rise after a drop below
90% reaches 90% of the reading before the drop, which a counter that
genuinely restarted cannot do in one reading. The recovery then scores only
the use since the last good reading. Those sweeps now total what was
recorded (23.4 against 23.4; 334.7 against 334.7), and a drop to 0 lasting
3 hours (a gap row, exempt from the cap) no longer adds 4,443 kWh, as it did
on main. Daily counters are excluded from this rule, since one reading after
the midnight reset can genuinely carry most of a day's use (a charger
reporting its session when it ends); their glitch recovery is the first rise
after a restart, kept out of the rate estimate, so the cap still holds it.

A rate cap on its own overcounted the house counter by 2.4 kWh/day. Every
drop used to count as a restart, so each time the counter stepped back by
0.1 kWh the next reading scored its whole value as one interval's use, and
the tight size cap had been hiding that by cutting it to 0.2 kWh. Restart
detection therefore changed too:

- A restart is a drop below `COUNTER_RESTART_RATIO` (0.9) of the previous
  reading, the threshold `HistoryDB._COUNTER_RESTART_RATIO` and HA's
  `total_increasing` use. A test pins the two constants equal.
- Between restarts a reading's use is its rise above the level already
  scored, so a dip and its recovery net to zero. On a counter that never
  dips this equals the plain difference. The level is the highest reading
  since the restart less any capped-off excess, so a dip is measured from
  what was scored (`_score_from_accepted_level`); a dip deeper than an
  interval's plausible use (the cap's rate times one interval, or
  `max_increment`) re-baselines the counter instead. An earlier draft
  measured from the raw high. After one corrupted reading (+1,000 kWh on a
  12,345 kWh meter, under the 10% restart drop), or a 5% step down, every
  later reading scored 0 until the counter passed the old high: 20 of 28
  days of zero labels, including the most recent bins feeding inference
  lags. Both cases now total the recorded use (811.5 against 811.5 kWh;
  810.6 for the step). The scan only runs when a counter dips and also has a
  capped reading or a deep dip; otherwise the vectorised running-high
  result stands unchanged.

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
