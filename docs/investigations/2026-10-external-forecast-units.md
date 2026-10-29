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

The Solcast attribute format is the common case, and the one in the DOCS
example:

- the state is the day's total, in kWh;
- each `detailedForecast` entry (`pv_estimate`, `pv_estimate10`,
  `pv_estimate90`, or Predbat's `pv_estimateCL`) is average power over the
  period, in kW.

The entries were therefore taken as kWh per interval. Against a kW target on
a 30-minute grid, each external came out at 1/0.5 = 2× its real size. The
same holds against a cumulative kWh target, which is also compared in energy
per interval. The scale-mismatch guard only trips beyond 4× or below 0.25×,
so nothing flagged it.

## Measured on a production solar experiment

30-minute interval, kW target, two trajectory externals in the Solcast
format (raw Solcast `pv_estimate`, and a calibrated `pv_estimateCL`), 30-day
window:

| | App | Calibrated | Raw |
|---|---|---|---|
| Scale ratio (external mean ÷ actual mean) | — | 1.79 | 1.81 |
| "Best available" MAE (kW) | 0.285 | 0.593 | 0.589 |
| Daily bias (kWh/day) | — | +4.4 | +4.4 |

A daily bias of about the mean daily generation is what doubling gives. The
tab declared the app about 50% better than both.

Re-scored by hand with the entries in kW, on the half-hours the app scores
(those where the target reported, daylight in practice), 16 days, the
forecast in force at each lead, about 250 half-hours per lead:

| Lead | App | Calibrated | Raw |
|---|---|---|---|
| 0 | **0.17** | 0.21 | 0.20 |
| 1 h | 0.23 | 0.22 | **0.19** |
| 4 h | 0.23 | 0.21 | **0.20** |
| 8 h | 0.27 | **0.20** | 0.21 |
| 24 h | 0.30 | **0.19** | 0.19 |

So the headline was inverted beyond the current half-hour. A check on the 48
targets the trajectory route exposes, scored against the app's own actuals,
gave the same ordering, so the hand resampling of the actuals is not the
cause.

The "Best available" view also compares the app's latest issuance (median
lead 0) with the externals' latest (median lead 90–140 min). That is by
design and labelled on the tab; "Same lead time" is the like-for-like view.

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
