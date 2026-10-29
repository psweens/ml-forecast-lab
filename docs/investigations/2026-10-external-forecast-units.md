# Attribute-mode external forecasts read in their sensor's unit (v2.56.0)

**Status:** Fixed in v2.56.0. External forecasts gain a `unit` setting, the
Forecast Comparison tab flags a forecast whose values look to be in another
unit from its sensor, and a one-click fix sets the unit without dropping the
forecast's captured history.

## What happened

`/experiment/{name}/external-forecast-comparison` passed each external's HA
`unit_of_measurement` to `HistoryDB.get_external_forecast_comparison`, which
converts every series to energy per interval: power × interval hours, energy
as it is. In `attribute` mode that unit belongs to the sensor's **state**,
not to the trajectory entries the comparison scores.

A common layout for solar and other energy forecasts, and the one in the
DOCS example, has:

- a state that is the day's total, in kWh;
- trajectory entries that are average power over each period, in kW.

The entries were therefore taken as kWh per interval. Against a kW target on
a 30-minute grid, such an external came out at 1/0.5 = 2× its real size. The
same holds against a cumulative kWh target, which is also compared in energy
per interval. The scale-mismatch guard only trips beyond 4× or below 0.25×,
so nothing flagged it.

## Effect

For every external in that layout, on an experiment with an interval under
an hour:

- the scale ratio (external mean ÷ actual mean) sat near 1/interval_hours
  times the forecast's own bias, about 1.8–2.2 on a 30-minute grid;
- per-interval error and RMSE were roughly doubled;
- the daily bias was close to the mean daily total, since doubling adds one
  day's total;
- the verdict and ranking favoured the app over forecasts that, scored in
  the right unit at matched lead, could be better than it.

Because values are logged raw and converted at read time, the logged
history itself was never wrong; only its scoring was.

The "Best available" view also compares the app's latest issuance (median
lead near 0) with each external's latest, which for a source that updates a
few times a day is typically hours old. That is by design and labelled on the
tab; "Same lead time" is the like-for-like view.

## Fix

### `unit` on each external forecast

`ExternalForecastCfg.unit`, one of `EXTERNAL_FORECAST_UNITS` (W, kW, MW, Wh,
kWh, MWh), case-insensitive and stored in canonical spelling. When set it
replaces the sensor's unit in the comparison and is trusted. An unrecognised
unit in YAML is dropped with a warning rather than failing the config load;
the add and set routes reject it.

### Flagging a likely mix-up

With `unit` unset, `suspect_external_unit` tests the head-to-head scale
ratio `r`. Reading power as energy per interval scales a forecast by
`g = 1/interval_hours`; reading energy per interval as power scales it by
`g = interval_hours`. The forecast is flagged when `r` is on the same side of
1 as `g` and nearer `g` than 1 in log terms, `|ln r| > |ln g| / 2`. That is
the log midpoint, so no constant is tuned. On a 30-minute grid the threshold
is √2 ≈ 1.41 for an energy-unit sensor, and 1/√2 for a power-unit one.

It applies only to `attribute` mode (a state-mode sensor's unit describes the
value scored), to non-cumulative power or energy series (a cumulative series
has no power reading), and when the interval is not one hour (the two readings
then coincide).

A flagged forecast reports `unit_suspect` and `suggested_unit` (the same
multiplier in the other dimension: kWh→kW, Wh→W, kW→kWh), and is:

- excluded from the "Same lead time" ranking (reason "unit looks wrong");
- shown as excluded in the "Best available" table and verdict, like a scale
  mismatch;
- explained in the comparison-basis note with a **Use kW** button.

### Why flag rather than correct

Applying the suggested unit automatically would also "correct" a forecast
that is read correctly but really runs 1.4–2.8× high, making a broken
forecast look reasonable. Flagging costs one click for the common case and
hides nothing: once a unit is set it is trusted, so a genuinely biased
forecast is scored as one.

The residual false positive is a correctly read forecast more than √2 high
(30-minute grid) with its unit unset. Setting its unit clears the flag.

### Setting the unit keeps history

`/remove-external-forecast` deletes the source's rows from
`external_forecast_log`, so removing and re-adding a forecast to change its
unit would discard its captured trajectories (up to
`external_forecast_retention_days`, default 60). `POST
/experiment/{name}/external-forecast-unit` instead updates the one entry
whose `source_key` matches (`set_experiment_external_forecast_unit`). The
comparison now returns each external's `source` for this. Values are logged
raw and converted at read time, so the whole history is rescored under the
new unit.

## Tests

- `tests/unit/test_external_forecast_units.py`: config parsing, canonical
  spelling and tolerant load; `set_experiment_external_forecast_unit` on two
  forecasts from one entity; the mix-up test across 15-minute, 30-minute,
  1-hour and 2-hour grids, both directions and every exemption; the
  comparison end to end (kW entries on a kWh sensor are flagged at ratio
  ≈ 2.1 and excluded from the skill ranking; an explicit kW scores the true
  error; an explicit unit on a genuinely doubled forecast is scored, not
  flagged; a correct sensor unit gives unchanged results).
- `tests/smoke/test_external_forecast_unit_route.py`: the set-unit and add
  routes, bad input, and the add form's Unit field.
