# Cumulative targets: bands, coverage and the accuracy sensor (v2.52.4)

**Status:** Shipped in v2.52.4. Defect analysis, fix rationale and the
limits that remain, behind that release's CHANGELOG entries.

The pinning tests:
- `ml-forecast-lab/tests/unit/test_cumulative_conformal.py`: bands,
  coverage, accuracy sensor scale and the replay key;
- `tests/unit/test_increment_actuals.py`: the increment actuals equal
  the training label;
- `tests/unit/test_accuracy_sensor_state.py`: the frozen accuracy state;
- `tests/unit/test_external_forecast_lock.py`: the database lock.

```bash
cd ml-forecast-lab/
python -m pytest tests/unit/test_cumulative_conformal.py tests/unit/test_increment_actuals.py \
  tests/unit/test_accuracy_sensor_state.py tests/unit/test_external_forecast_lock.py -v
```

---

## TL;DR

| Defect (live up to v2.52.3) | Observed | Fix |
| --- | --- | --- |
| **A.** `get_conformal_quantiles` joined a cumulative experiment's logged per-interval deltas to the raw counter grid | 80% bands tens of kWh wide around a forecast of a fraction of a kWh. On a daily-reset counter the quantile tracks the day's total | Join the actuals increments (`increment=True`) |
| **B.** `get_forecast_coverage` made the same join, in all three of its queries | Accuracy-tab coverage tested A's bands against the counter too, so it was measured on A's scale and could not expose A | Same join |
| **C.** The `_forecast_accuracy` sensor called `get_forecast_accuracy` in its default `raw` mode | The sensor measured the counter level, not the model. The web Accuracy tab already forced `increment` for cumulative sources, so the chart and the sensor disagreed | The sensor passes `evaluation_mode="increment"` for a cumulative source |
| **D.** The increment actuals differenced bin means and nulled any bin whose previous bin had no reading | A counter that only logs changes was scored on a small, biased subset: no quiet bin, no first draw after a quiet spell, no reset bin. Simulated bands about three times too wide | The increment actuals follow the training label's rule |
| **E.** The sensor state was `lead_time_curve["mae"][0]`, the most negative lead bucket | After one stale-frame tick, every experiment's accuracy state stayed identical on every publish, cumulative or not | The state is the first bucket with a non-negative lead |
| **F.** `log_external_forecast` was the one public `HistoryDB` method on the shared connection without `@_locked` | "cannot start a transaction within a transaction" and "no more rows available" from concurrent publish cycles | `@_locked` |

## A–C: which actuals a cumulative residual is taken against

For `source_is_cumulative`, the model is trained on per-interval
increments, and `forecast_log.predicted`, `upper` and `lower` hold those
increments. The actuals table holds what Home Assistant recorded: the
counter. Before v2.52.4 the residual was therefore
`|forecast increment − counter value|`. Its 80th percentile is roughly
the counter's typical level, which on a daily-reset kWh counter is a
large fraction of the day's total. A simulated daily-reset counter gave
an 80% quantile of about 43 kWh against about 0.25 kWh from the true
per-interval residuals.

`_materialise_actuals_grid(..., increment=True)` builds the per-interval
actuals the Accuracy tab's increment mode reads, `_mlfl_actuals_vals_tmp`.
Bands, coverage and the sensor now read the same relation, through one
helper (`HistoryDB._actuals_join`) so the call sites cannot disagree.
What that relation holds was itself wrong (D, below).

The raw path is byte-for-byte the pre-v2.52.4 SQL: `_actuals_join(False)`
returns the old relation and an empty guard. Non-cumulative bands,
coverage and accuracy are unchanged, and `CACHE_SCHEMA_VERSION` is not
bumped, because nothing a model is fed has changed.

The new `source_is_cumulative` parameter is last in both signatures,
because the web routes call them positionally.

### Replay key

`replay._conformal_key` binds every argument with defaults applied. It
adds `source_is_cumulative` only when it is true, so every
non-cumulative bundle recorded before v2.52.4 keeps its key. A cumulative
bundle from v2.52.3 or earlier recorded raw-counter quantiles. The
current tree no longer requests those, so its forecast stage stops at
`UnrecordedCall` (exit 2) rather than replaying a different band
silently.

## D: the increment actuals are the training label

A cumulative experiment's label is `cumulative_to_interval` followed by a
sum resample. Each reading contributes its rise over the previous
reading, a drop counts the reading itself (a reset), and the bin's label
is the sum. A bin with no readings is a zero increment. Since v2.40.5 the
rise across a gap of more than one interval counts in full in the bin
where it was recorded, because HA's recorder stores only state changes and
a quiet stretch is real zero use, not missing data.

The increment actuals predate v2.40.5 and kept the older rule. They took
`AVG(counter in bin t) − AVG(counter in bin t−1)` and set it to NULL
unless bin t−1 had a reading. Against a change-only counter this:
- never scored a quiet bin, because it had no grid row;
- nulled the first bin after every quiet stretch, often the day's main
  draw, and filtered out the bin containing the daily reset;
- compared multi-reading bins on a difference of means, half a bin out
  of phase with the label.

In simulation (a 13-day daily-reset counter with six draws a day,
recorded on change only), 19 of 576 bins were scorable. A model that
forecast the training label exactly got an 80% quantile of 0.36 kWh
instead of 0. A realistic model's band realised 94% coverage against a
nominal 80%.

The increment for bin t is now `last(t) − last reading before t`. When
that is negative, the counter has restarted and the increment is
`last(t)`. A bin with no readings, between the window's first and last
reading, is 0. That equals the training label exactly, except where
training caps a spike (`max_increment`, by default the 95th percentile
of per-reading increments): the analytics score against what the counter
measured. The window's first bin has no earlier reading and is NULL;
bins after the last reading are absent until a reading arrives.
`test_increment_actuals.py` compares the relation with the output of
`cumulative_to_interval` and `resample_to_grid` on a change-only counter.

**Trade-off.** A recorder outage on a meter that normally reports every
interval is now scored the way training sees it: zeros through the
outage, then the whole outage's use in one interval. Before, those bins
were dropped. Recorder data alone cannot tell an outage from a quiet
stretch. Training already treats both as quiet, and the published band
has to cover quiet intervals, so the analytics follow training. The
v2.40-era test that pinned the old guard is now
`test_increment_mode_scores_an_actuals_gap_like_training`, which pins the
training rule.

## E: the frozen accuracy state

`get_forecast_accuracy` buckets leads with SQLite integer division, which
truncates toward zero, so bucket 0 holds leads in `(-interval, interval)`
and the curve is ordered by bucket. In steady state the smallest logged
lead is `interval − tick phase`, which falls in bucket 0.

`_compute_cached_forecast` falls back to the cached frame when the fresh
fetch fails ("Fresh data fetch failed, using cached data"), during a Home
Assistant outage for example. The forecast it logs then starts at the
cached frame's end, hours before the issue time. Those targets already
have actuals, so they join, and their bucket (minus several hours) sorts
first. Index 0 then pointed at a bucket with one or a few samples, and
the state stayed at that value on every publish until the rows aged out
of the 30-day window. Every experiment issued during the same outage
froze together.

The state now reads the first bucket with `lead_minutes >= 0`. The
hindcast buckets stay in the `lead_hours` / `mae` attributes, and the
sensor reports `status: accumulating` until a next-interval sample
exists.

## F: the external-forecast lock

`HistoryDB` serialises the shared `sqlite3` connection with an `RLock`
taken by `@_locked`. `log_external_forecast` ran its `executemany` and
`commit` outside it, so two publish cycles (or a cycle and the analytics
thread) could interleave on the connection. On failure it also called
`self.conn.rollback()`, which could discard another thread's uncommitted
write. `ensure_external_forecast_log_table` was already locked, which is
why a single-threaded test passes on the old code. The pinning test
checks lock ownership inside `executemany` and runs external and internal
writers concurrently.

## Not changed here

- **Hindcast rows still reach the band and coverage queries.** They feed
  the pooled `fallback_quantile` and the coverage figures. Skipping
  `log_forecast` when the tick used a stale frame, or filtering
  `lead_minutes > -interval` in the analytics, would remove them, but it
  changes non-cumulative bands and every replay bundle's expected output,
  so it belongs in its own change. The stale-frame tick also publishes
  timestamps that are already in the past.
- **Band lookup offset.** `_conformal_bands` uses
  `issued_ref = ds_future[0] − interval`, so step 1 looks up bucket
  `interval`. Logged step-1 leads are `interval − tick phase`, which lands
  in bucket 0. Bands therefore read one bucket later than the residuals
  they were calibrated from.
- **`load_subtract` experiments.** The model forecasts load net of the
  subtracted sources, while the actuals table holds the gross counter.
  Increment mode removes the counter-level error but leaves that bias.
- **Lower-band clamp.** The lower band is clamped at zero only when
  `source_is_cumulative`. A non-negative target such as PV power can
  publish a negative lower band. Keying the clamp on
  `target_is_nonnegative` changes published bands and replay output.
- **A counter glitch.** A single bad reading of 0 on a lifetime counter
  counts as a restart (increment 0), and the next bin's rebound counts
  as one interval's use. Training caps that rebound with
  `max_increment`. The analytics have no equivalent cap, so one glitch
  inflates the accuracy figures until it leaves the window.
- **The training spike cap itself.** With `max_increment` unset,
  `cumulative_to_interval` caps each reading's rise at the 95th
  percentile of all rises, so the top 5% of readings train on a clipped
  label. The analytics score against the uncapped counter. This has not
  been measured on real data.
