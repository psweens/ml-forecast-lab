# Covariates trained on stored forecasts (v2.57.0)

**Status:** Added in v2.57.0: `history_source: stored_forecast` and
`future_extra_entities` on covariates. Fixed in the same release: the
Forecast Comparison capture stored values beyond a source's own span.

## The mismatch

A `role: future` / `both` covariate has two series:

- **history**, used for training: the entity's recorded state, or with
  `future_value_key` that attribute, cached by `CovariateResolver`;
- **future values**, used at forecast time: the entity's forecast attribute,
  read live by `fetch_future`.

For a weather covariate, history is the observed value and the future is a
forecast of it. Training on observations and forecasting from forecasts is
the "perfect prognosis" set-up: the model learns how the target responds to
the true covariate, then is fed a noisier one. It is the usual choice when
no archive of past forecasts exists.

For many forecast entities the two series are not even the same quantity.
An entity whose state is a total over its whole forecast horizon, or a daily
total, while its attribute holds per-period values, trains the model on the
state and feeds it the per-period entries. The column carries a relationship
that inference never reproduces.

## Stored forecasts as history

The app already archives forecasts: the Forecast Comparison tab logs each
attribute-mode external's trajectory to `external_forecast_log` (issued at,
target, value), de-duplicated by content. `history_source: stored_forecast`
uses the same store as a covariate's history.

### What each bin's value is

`stored_forecast_on_grid`: each grid bin takes its value from the latest
trajectory issued **at or before the bin's start**, interpolated linearly in
time between that trajectory's own points (as `fetch_future` resolves a live
forecast onto the horizon). A bin outside that trajectory's span is NaN, not
filled from an older or a later one. Bins before the first stored trajectory
are NaN too, so they are masked and imputed with a `_missing` indicator like
any covariate whose history starts part-way into the window.

This is causal: every value was available before its period began.

The trade-off is lead. Training values come from the forecast in force at
the bin (lead from zero to the source's refresh interval). At forecast time
the covariate's values run up to the experiment's horizon. A model trained
on short-lead values will trust long-lead ones as much, which matters in
proportion to how much a source's error grows with lead. For sources that
refresh rarely, the forecast "in force" is already hours old, and the gap is
small. A lead-matched history (the forecast issued `h` ahead, for each
horizon `h`) is not possible for the recursive tree path, which uses one
value per row, and is left for a later change.

### Where trajectories come from

- **The covariate's own capture.** `fetch_future` calls
  `_store_live_forecast` whenever it reads a stored-forecast covariate, i.e.
  on every production forecast cycle. Only the source's own points are
  stored, from two hours before the capture on (an earlier point only serves
  to interpolate the first bins), never the interpolated or edge-filled
  horizon. `HistoryDB.store_covariate_forecast` writes them under
  `covariate:<source key>` with an empty `experiment`, so experiments sharing
  a forecast share its history, and the comparison tab, which reads by exact
  source, never scores them. It applies the comparison capture's content
  rule: a trajectory matching the last stored one on every shared target is
  the same issuance and is not stored again.
- **The comparison's archive.** The source key is the comparison's own key
  for the same entity, attribute and value key
  (`config.stored_forecast_source`). `get_stored_forecast_history` reads both
  namespaces, from any experiment, preferring the covariate's rows on an
  exact duplicate. A covariate added today therefore has history going back
  as far as the comparison stored the same forecast (by default 60 days).

### Cost

The history is read every forecast cycle, like any covariate's. A year of a
half-hourly source refreshed about every 72 minutes is ~700k stored rows, and
reading them all and computing the grid took 3.5 s on the development
machine, holding the shared database lock. Only a few rows per trajectory
can matter: trajectory `i` serves bins until issuance `i+1`, so
`_stored_forecast_rows` reads each trajectory's targets up to that issuance
plus two of the source's steps (the median spacing of its latest trajectory),
through a temporary issuance table `CROSS JOIN`ed onto the
`(source, issued_at, target_dt)` index so each range scan stops at its
cut-off. That is ~39k rows for the same year. `stored_forecast_on_grid` is
vectorised (one `searchsorted` over an (issuance, seconds) key), and runs
outside the lock. Together: 0.35–0.45 s. Randomised tests check both the
row reduction and the vectorised rule against a brute-force reference.

### Retention and backfill

Stored forecasts used as history are training data, so they cannot live on
the comparison's 60-day window. At each retrain,
`MLForecastLabApp._maintain_stored_forecasts` calls
`HistoryDB.maintain_stored_forecasts` for each such source:

- **backfill:** comparison rows for the source issued before the covariate's
  earliest own row are copied under the covariate's key, one per
  (issued_at, target). Before they age out of the comparison, or are deleted
  with it when that external is removed, they become the covariate's.
  Rows issued after the covariate started capturing are not copied, since
  the covariate's capture stores the same forecasts;
- **prune:** the covariate's rows issued more than the longest
  `days_history` among the experiments using that source, plus a day, are
  deleted. The comparison's own rows are left to that tab.

### Known limitations

- Trajectories are only stored while some experiment using the covariate is
  in production, and comparison rows only while its experiment is.
- Comparison rows stored before v2.57.0 include values held beyond a
  source's span (see below). Under the latest-issuance rule they are used
  only for a bin no newer trajectory covers, e.g. after the app was down
  across the end of a source's span.
- Until the store covers the training window, only the stored part teaches
  the model what the forecast means. Forecasting always has the forecast, so
  the masked stretch only trains the "missing" branch of the indicator.

## Forecasts across several sensors

Some integrations publish one forecast as several sensors, one per day.
`fetch_future` reads one entity, and `align_future_points` holds the edge
values beyond its points, so a covariate on the first day's sensor read
every later period as that sensor's final entry.

`future_extra_entities` lists the other sensors. `fetch_future_points` reads
the same `future_attribute` / `future_value_key` from each, normalises them
to naive UTC and merges with `combine_first`, so the main entity's value
wins wherever both cover a time. The weather service types
(`hourly` / `daily` / `twice_daily`) already return the whole forecast and
take no extra entities. A stored-forecast covariate stores the merged
trajectory.

## The comparison stored values a source never forecast

`_capture_external_forecast` resolved each attribute-mode external onto the
experiment's horizon with `fetch_future` and logged every target, including
those beyond the source's last point (and before its first), which
`fetch_future` fills with the edge value. A source covering less than the
horizon was then scored at leads it does not cover, against its last value
repeated, and the "Same lead time" band (the intersection of the leads each
source has) included them.

The capture now reads the source's own points (`fetch_future_points`),
resolves them onto the grid (`align_future_points`) and keeps only targets
inside the points' span (`_within_points_span`). It makes one HA read per
external, as before. Rows logged before v2.57.0 are not rewritten.

## Replay

`get_stored_forecast_history` is in the training-frame path, so
`RecordingHistoryDB` records its result on the grid it was read on (the
`get_history` shape, not the raw trajectories) and `ReplayHistoryDB` serves
it back. `store_covariate_forecast` is a write and is suppressed on both
sides. Extra entities are read through `get_state`, which the HA recorder
already records.

## Tests

- `tests/unit/test_stored_forecast_covariates.py`: config validation and
  normalisation; the grid rule (latest issuance at or before a bin,
  interpolation, no fill outside the chosen trajectory, no value before the
  first, duplicates); storage de-duplication, reads across both namespaces
  and sources, backfill and prune; extra sensors merged with the main one
  winning, and the old hold-last behaviour without them; the live trajectory
  stored only for stored-forecast covariates and only from its own points;
  history routed to the store; the training frame carrying the stored values
  and nothing before the first issuance; retention as the longest
  `days_history` sharing a source; the comparison capture logging only a
  source's span; replay round trip.
- `tests/smoke/test_stored_forecast_covariate_route.py`: the add route, the
  role check, and the form fields and row labels.
