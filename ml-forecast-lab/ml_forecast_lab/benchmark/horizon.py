"""
Rolling-origin evaluation of the forecast that is published.

The cross-validated leaderboard scores each backend one step ahead with
measured lags. A tree backend publishes something else: a
``future_periods``-step recursive forecast in which every lag beyond the
first step is its own output. This module scores that forecast — from many
issue times, per lead — together with a predictive distribution, so a model
is ranked on what users receive.

Three things it does that the one-step metrics cannot:

- **A proper scoring rule.** CRPS, computed from a quantile grid, rewards a
  forecast for placing probability where the outcomes fall. MAE is minimised
  by the median, so on a non-negative right-skewed target (most intervals
  near zero, occasional large events) it ranks a constant near zero above a
  correctly-levelled forecast. CRPS and RMSE do not.
- **Reference forecasts.** Zero, constant mean, a seasonal profile by
  slot-of-day × weekday/weekend (mean and median) and the same slot on the
  previous day are scored on the same origins, so "better than nothing" is
  measured rather than assumed.
- **Period totals.** For a target that resets each day, the error in the
  remaining-in-day total as a function of issue time.

Everything here is pure numpy/pandas and backend-agnostic: a forecaster is
any ``fit(history)`` / ``predict(history, future_index)`` pair.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Quantile grid the predictive distribution is carried on. CRPS is the
# integral of the pinball loss over all levels; nineteen evenly spaced
# levels approximate it to well inside the sampling noise of any window
# this app evaluates.
QUANTILE_LEVELS: np.ndarray = np.round(np.arange(0.05, 0.96, 0.05), 2)

# Fewer scored days than this and differences between forecasters on a
# noisy target are not evidence: rankings are reported, not acted on.
MIN_EVALUATION_DAYS = 14

# Fewer out-of-sample residuals than this in a lead bucket and the bucket
# borrows the pooled residuals of every lead.
MIN_RESIDUALS_PER_BUCKET = 30


# --------------------------------------------------------------------------
# Scoring rules
# --------------------------------------------------------------------------

def pinball(y: np.ndarray, q: np.ndarray, level: float) -> np.ndarray:
    """Element-wise pinball loss of quantile forecast ``q`` at ``level``."""
    err = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return np.where(err >= 0, level * err, (level - 1.0) * err)


def crps_from_quantiles(
    y: np.ndarray, quantiles: np.ndarray,
    levels: Sequence[float] = QUANTILE_LEVELS,
) -> np.ndarray:
    """Element-wise CRPS from a quantile grid.

    ``quantiles`` has shape ``y.shape + (len(levels),)``. CRPS equals twice
    the pinball loss integrated over the level; the mean over an even grid
    is its quadrature. A degenerate distribution (every quantile equal to
    one point) gives the absolute error, as CRPS should.
    """
    y = np.asarray(y, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    lv = np.asarray(levels, dtype=float)
    err = y[..., None] - q
    loss = np.where(err >= 0, lv * err, (lv - 1.0) * err)
    return 2.0 * loss.mean(axis=-1)


# --------------------------------------------------------------------------
# Local-time keys
# --------------------------------------------------------------------------

def local_index(index: pd.DatetimeIndex, tz: Optional[str]) -> pd.DatetimeIndex:
    """``index`` (tz-naive UTC, or tz-aware) in the local zone ``tz``."""
    aware = index.tz_localize('UTC') if index.tz is None else index
    return aware.tz_convert(tz) if tz else aware


def slot_keys(
    index: pd.DatetimeIndex, interval_minutes: int, tz: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """(slot-of-day, is-weekend) in local time for each timestamp."""
    loc = local_index(index, tz)
    slot = (loc.hour * 60 + loc.minute) // max(1, int(interval_minutes))
    return np.asarray(slot, dtype=int), np.asarray(loc.dayofweek >= 5)


# --------------------------------------------------------------------------
# Forecasters
# --------------------------------------------------------------------------

class Forecaster:
    """A point forecaster, optionally with its own predictive quantiles.

    ``predict`` returns the point forecast for ``future_index``;
    ``predict_quantiles`` returns an array of shape
    ``(len(future_index), len(QUANTILE_LEVELS))`` or ``None``, in which
    case the evaluator builds the distribution from the forecaster's own
    earlier out-of-sample residuals.
    """

    name = 'forecaster'

    def fit(self, history: pd.Series) -> None:  # pragma: no cover - interface
        pass

    def predict(self, history: pd.Series, future_index: pd.DatetimeIndex) -> np.ndarray:
        raise NotImplementedError

    def predict_quantiles(
        self, history: pd.Series, future_index: pd.DatetimeIndex,
    ) -> Optional[np.ndarray]:
        return None


class ZeroBaseline(Forecaster):
    name = 'baseline_zero'

    def predict(self, history, future_index):
        return np.zeros(len(future_index))

    def predict_quantiles(self, history, future_index):
        return np.zeros((len(future_index), len(QUANTILE_LEVELS)))


class MeanBaseline(Forecaster):
    """Constant at the history mean; quantiles are the history's own."""

    name = 'baseline_mean'

    def predict(self, history, future_index):
        return np.full(len(future_index), float(np.nanmean(history.to_numpy())))

    def predict_quantiles(self, history, future_index):
        q = np.nanquantile(history.to_numpy(dtype=float), QUANTILE_LEVELS)
        return np.tile(q, (len(future_index), 1))


class SeasonalProfileBaseline(Forecaster):
    """Profile by local slot-of-day × weekday/weekend.

    ``statistic`` is ``'mean'`` or ``'median'`` for the point forecast; the
    quantiles are the empirical quantiles of the same group either way. A
    group with no history falls back to the slot across both day types,
    then to the whole history.
    """

    def __init__(self, interval_minutes: int, tz: Optional[str], statistic: str = 'mean'):
        if statistic not in ('mean', 'median'):
            raise ValueError(f"statistic must be 'mean' or 'median', got {statistic!r}")
        self.interval_minutes = int(interval_minutes)
        self.tz = tz
        self.statistic = statistic
        self.name = f'baseline_profile_{statistic}'

    def _groups(self, history: pd.Series):
        h = history.dropna()
        slot, weekend = slot_keys(h.index, self.interval_minutes, self.tz)
        return h.to_numpy(dtype=float), slot, weekend

    def _values(self, history, future_index):
        vals, slot, weekend = self._groups(history)
        f_slot, f_weekend = slot_keys(future_index, self.interval_minutes, self.tz)
        point = np.empty(len(future_index))
        quant = np.empty((len(future_index), len(QUANTILE_LEVELS)))
        cache: Dict[Tuple[int, bool], Tuple[float, np.ndarray]] = {}
        stat = np.mean if self.statistic == 'mean' else np.median
        for i, key in enumerate(zip(f_slot.tolist(), f_weekend.tolist())):
            if key not in cache:
                sel = vals[(slot == key[0]) & (weekend == key[1])]
                if sel.size == 0:
                    sel = vals[slot == key[0]]
                if sel.size == 0:
                    sel = vals
                cache[key] = (float(stat(sel)), np.quantile(sel, QUANTILE_LEVELS))
            point[i], quant[i] = cache[key]
        return point, quant

    def predict(self, history, future_index):
        return self._values(history, future_index)[0]

    def predict_quantiles(self, history, future_index):
        return self._values(history, future_index)[1]


class PreviousDayBaseline(Forecaster):
    """The same slot on the most recent measured day."""

    name = 'baseline_previous_day'

    def __init__(self, interval_minutes: int):
        self.steps_per_day = max(1, 1440 // max(1, int(interval_minutes)))

    def predict(self, history, future_index):
        vals = history.ffill().to_numpy(dtype=float)
        spd = self.steps_per_day
        last_day = vals[-spd:] if len(vals) >= spd else np.resize(vals, spd)
        return np.array([last_day[i % spd] for i in range(len(future_index))])


class RecursiveTreeForecaster(Forecaster):
    """A tabular backend scored through its published recursive forecast.

    ``model_factory`` returns a fresh, unfitted backend with the app's
    ``fit(X, y, feature_names=...)`` / ``predict(X)`` interface. Features
    come from :func:`features.build_features` and the forecast from
    :func:`features.recursive_tree_forecast`, the same construction the
    production forecast uses.
    """

    def __init__(self, model_factory: Callable[[], object], interval_minutes: int,
                 name: str = 'model', country: Optional[str] = None):
        self.model_factory = model_factory
        self.interval_minutes = int(interval_minutes)
        self.country = country
        self.name = name
        self.model = None
        self.feature_cols: List[str] = []

    def fit(self, history: pd.Series) -> None:
        from ml_forecast_lab.features import build_features
        df = pd.DataFrame({'y': history})
        feats = build_features(
            df, target_col='y', interval_minutes=self.interval_minutes,
            country=self.country,
        )
        frame = feats.assign(target=df['y']).dropna()
        self.feature_cols = [c for c in frame.columns if c != 'target']
        X = frame[self.feature_cols].to_numpy(dtype=np.float32)
        y = frame['target'].to_numpy(dtype=np.float32)
        self.model = self.model_factory()
        self.model.fit(X, y, feature_names=self.feature_cols)

    def predict(self, history, future_index):
        from ml_forecast_lab.features import recursive_tree_forecast
        out = recursive_tree_forecast(
            self.model.predict, self.feature_cols, history,
            periods=len(future_index), interval_minutes=self.interval_minutes,
            country=self.country,
        )
        return out.to_numpy()


def default_baselines(interval_minutes: int, tz: Optional[str]) -> List[Forecaster]:
    """The reference forecasts every evaluation includes."""
    return [
        ZeroBaseline(),
        MeanBaseline(),
        SeasonalProfileBaseline(interval_minutes, tz, 'mean'),
        SeasonalProfileBaseline(interval_minutes, tz, 'median'),
        PreviousDayBaseline(interval_minutes),
    ]


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

@dataclass
class HorizonScores:
    """One forecaster's scores over a set of issue times."""

    name: str
    n_origins: int
    n_points: int
    mae: float
    rmse: float
    bias_pct: float
    crps: float
    coverage: float
    interval_width: float
    amplitude_ratio: float
    period_total_rmse: float = float('nan')
    by_lead: pd.DataFrame = field(repr=False, default=None)
    period_total: pd.DataFrame = field(repr=False, default=None)
    # Per-origin mean CRPS and squared error, for paired comparisons.
    origin_crps: np.ndarray = field(repr=False, default=None)
    origin_sq_err: np.ndarray = field(repr=False, default=None)
    origin_times: pd.DatetimeIndex = field(repr=False, default=None)

    def summary(self) -> dict:
        return {
            'name': self.name, 'n_origins': self.n_origins,
            'mae': self.mae, 'rmse': self.rmse, 'bias_pct': self.bias_pct,
            'crps': self.crps, 'coverage': self.coverage,
            'interval_width': self.interval_width,
            'amplitude_ratio': self.amplitude_ratio,
            'period_total_rmse': self.period_total_rmse,
        }


@dataclass
class HorizonEvaluation:
    scores: Dict[str, HorizonScores]
    evaluation_days: float
    enough_evidence: bool
    coverage_level: float

    def table(self) -> pd.DataFrame:
        rows = [s.summary() for s in self.scores.values()]
        return pd.DataFrame(rows).set_index('name').sort_values('crps')


# How a point-only forecaster's earlier errors are grouped before their
# quantiles are added to a new forecast.
#   'lead'  — by lead bucket: how far ahead the forecast was.
#   'level' — by the size of the point forecast: errors made when the
#             forecast was high are applied where it is high again.
#   'slot'  — by local time of day (three-hour blocks) and weekday/weekend.
RESIDUAL_CONDITIONS = ('lead', 'level', 'slot')
LEVEL_BINS = 4


def level_conditioned_offsets(
    predicted: np.ndarray, residual: np.ndarray, new_predicted: np.ndarray,
    levels: Sequence[float], bins: int = LEVEL_BINS,
    min_samples: int = MIN_RESIDUALS_PER_BUCKET,
) -> Optional[np.ndarray]:
    """
    Quantile offsets for new forecasts from past signed errors, grouped by
    the size of the forecast.

    ``residual`` is ``actual − predicted`` for past forecasts. Past
    forecasts are split into ``bins`` equal-count groups by their value;
    each new forecast takes the error quantiles of the group it falls in.
    On a non-negative skewed target the errors made when little was
    forecast are small and one-sided, and those made when a lot was
    forecast are large — one pooled set of quantiles is too wide for the
    first and too narrow for the second. A group with fewer than
    ``min_samples`` errors uses the pooled quantiles.

    Returns an array of shape ``(len(new_predicted), len(levels))`` to add
    to the new forecasts, or ``None`` when there are fewer than
    ``min_samples`` past errors in total. This is the one construction
    behind both the published asymmetric band and the evaluation's
    ``'level'`` distribution.
    """
    predicted = np.asarray(predicted, dtype=float).ravel()
    residual = np.asarray(residual, dtype=float).ravel()
    ok = np.isfinite(predicted) & np.isfinite(residual)
    predicted, residual = predicted[ok], residual[ok]
    if residual.size < min_samples:
        return None
    new_predicted = np.asarray(new_predicted, dtype=float)
    lv = np.asarray(levels, dtype=float)
    pooled = np.quantile(residual, lv)
    edges = np.unique(
        np.quantile(predicted, np.linspace(0, 1, bins + 1)[1:-1])
    )
    past_bin = np.searchsorted(edges, predicted, side='right')
    new_bin = np.searchsorted(edges, new_predicted, side='right')
    out = np.empty((new_predicted.size, lv.size))
    for b in np.unique(new_bin):
        e = residual[past_bin == b]
        out[new_bin == b] = np.quantile(e, lv) if e.size >= min_samples else pooled
    return out


def _residual_keys(
    condition: str, point: np.ndarray, origin_pos: np.ndarray, horizon: int,
    lead_bucket: int, future_times: List[pd.DatetimeIndex],
    interval_minutes: int, tz: Optional[str],
) -> Callable[[int, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Return ``keys(i, prior)`` → (keys of origin *i*'s cells, keys of the
    prior origins' cells), integers that group like with like."""
    n_orig = point.shape[0]
    if condition == 'lead':
        row = np.arange(horizon) // max(1, lead_bucket)
        fixed = np.tile(row, (n_orig, 1))
        return lambda i, prior: (fixed[i], fixed[prior])
    if condition == 'slot':
        per_block = max(1, 180 // max(1, int(interval_minutes)))
        fixed = np.empty((n_orig, horizon), dtype=int)
        for i, fut in enumerate(future_times):
            slot, weekend = slot_keys(fut, interval_minutes, tz)
            fixed[i] = slot // per_block + 100 * weekend.astype(int)
        return lambda i, prior: (fixed[i], fixed[prior])
    raise ValueError(
        f'condition must be one of {RESIDUAL_CONDITIONS}, got {condition!r}'
    )


def _residual_quantiles(
    point: np.ndarray, actual: np.ndarray, usable: np.ndarray,
    origin_pos: np.ndarray, horizon: int, lead_bucket: int,
    nonnegative: bool, condition: str = 'level',
    future_times: Optional[List[pd.DatetimeIndex]] = None,
    interval_minutes: int = 30, tz: Optional[str] = None,
) -> np.ndarray:
    """Predictive quantiles for a point-only forecaster.

    For origin *i* the distribution of a cell is the point forecast plus
    the empirical quantiles of the signed errors this forecaster made in
    like cells (see ``RESIDUAL_CONDITIONS``) at origins whose scored
    targets were all measured before origin *i* was issued. Signed, not
    absolute: on a skewed target the errors above and below a mean forecast
    are not the same size. A group with too few earlier errors borrows the
    pooled ones; origins with too few earlier errors at all get NaN and are
    left out of the probabilistic scores for every forecaster alike.
    """
    n_orig, H = point.shape
    L = len(QUANTILE_LEVELS)
    out = np.full((n_orig, H, L), np.nan)
    err = np.where(usable, actual - point, np.nan)
    if condition not in RESIDUAL_CONDITIONS:
        raise ValueError(
            f'condition must be one of {RESIDUAL_CONDITIONS}, got {condition!r}'
        )
    keys = None if condition == 'level' else _residual_keys(
        condition, point, origin_pos, horizon, lead_bucket,
        future_times, interval_minutes, tz,
    )
    for i in range(n_orig):
        # Earlier origins whose horizon ended at or before this issue time.
        prior = np.flatnonzero(origin_pos + H <= origin_pos[i])
        if prior.size == 0:
            continue
        prior_err = err[prior]
        if condition == 'level':
            offsets = level_conditioned_offsets(
                point[prior], prior_err, point[i], QUANTILE_LEVELS,
            )
            if offsets is not None:
                out[i] = point[i, :, None] + offsets
            continue
        pooled = prior_err[np.isfinite(prior_err)]
        if pooled.size < MIN_RESIDUALS_PER_BUCKET:
            continue
        pooled_q = np.quantile(pooled, QUANTILE_LEVELS)
        own_keys, prior_keys = keys(i, prior)
        for k in np.unique(own_keys):
            cols = np.flatnonzero(own_keys == k)
            e = prior_err[prior_keys == k]
            e = e[np.isfinite(e)]
            q = (
                np.quantile(e, QUANTILE_LEVELS)
                if e.size >= MIN_RESIDUALS_PER_BUCKET else pooled_q
            )
            out[i, cols, :] = point[i, cols, None] + q[None, :]
    if nonnegative:
        out = np.where(np.isfinite(out), np.maximum(out, 0.0), out)
    return out


def _period_total_errors(
    point: np.ndarray, actual: np.ndarray, usable: np.ndarray,
    origin_times: pd.DatetimeIndex, future_times: List[pd.DatetimeIndex],
    tz: Optional[str],
) -> pd.DataFrame:
    """Remaining-in-day total error per origin, keyed by local issue time.

    The remaining total runs from the first forecast slot to the end of
    that slot's local day. An origin is scored only when every slot of
    that span is inside the horizon and measured.
    """
    rows = []
    for i, fut in enumerate(future_times):
        loc = local_index(fut, tz)
        same_day = np.asarray(loc.date == loc[0].date())
        if same_day.all() or not usable[i][same_day].all():
            # The horizon ends before local midnight, or a label is missing.
            continue
        rows.append({
            'issue_hour': loc[0].hour + loc[0].minute / 60.0,
            'forecast': float(point[i][same_day].sum()),
            'actual': float(actual[i][same_day].sum()),
        })
    if not rows:
        return pd.DataFrame(columns=['issue_hour', 'n', 'rmse', 'bias'])
    df = pd.DataFrame(rows)
    df['err'] = df['forecast'] - df['actual']
    g = df.groupby('issue_hour')['err']
    return pd.DataFrame({
        'n': g.size(),
        'rmse': g.apply(lambda e: float(np.sqrt(np.mean(e ** 2)))),
        'bias': g.mean(),
    }).reset_index()


def evaluate_forecasters(
    series: pd.Series,
    forecasters: Sequence[Forecaster],
    *,
    interval_minutes: int,
    horizon: int,
    first_origin: int,
    origin_stride: int,
    refit_every: int,
    tz: Optional[str] = None,
    nonnegative: bool = False,
    coverage_level: float = 0.8,
    lead_bucket: Optional[int] = None,
    residual_condition: str = 'level',
    origin_positions: Optional[Sequence[int]] = None,
    precomputed: Optional[Dict[str, np.ndarray]] = None,
) -> HorizonEvaluation:
    """
    Score forecasters from a sequence of issue times on one series.

    Parameters
    ----------
    series : pd.Series
        Target on a complete grid (NaN where unmeasured). Labels are never
        imputed: a NaN target is left out of every score.
    horizon : int
        Steps per forecast — the experiment's ``future_periods``.
    first_origin : int
        Position of the first issue time; everything before it is history
        for the first fit.
    origin_stride : int
        Steps between issue times. Choose one that is not a multiple of a
        day so issue times cover the whole day.
    refit_every : int
        Refit each forecaster every this many origins (1 = every origin).
        Between refits the forecaster predicts from updated history with
        the model it has, as a production model does between retrains.
    lead_bucket : int, optional
        Steps per lead bucket for residual quantiles and the per-lead
        table. Defaults to six hours.
    residual_condition : {'lead', 'level', 'slot'}
        How a point-only forecaster's earlier errors are grouped when its
        distribution is built (``RESIDUAL_CONDITIONS``).
    origin_positions : sequence of int, optional
        Explicit issue positions, replacing ``first_origin`` /
        ``origin_stride``. Position *p* forecasts rows ``p .. p+horizon-1``
        from the rows before *p*.
    precomputed : dict[str, np.ndarray], optional
        Point forecasts already made elsewhere, one ``(n_origins, horizon)``
        array per name, aligned with the origins. A NaN cell means "no
        forecast" and is left out of every forecaster's scores. This is how
        the benchmark scores backends it trained in its own folds.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError('series must have a DatetimeIndex')
    n = len(series)
    if lead_bucket is None:
        lead_bucket = max(1, 360 // max(1, int(interval_minutes)))
    if origin_positions is not None:
        origin_pos = np.asarray(sorted(set(int(p) for p in origin_positions)), dtype=int)
        origin_pos = origin_pos[(origin_pos >= 1) & (origin_pos + horizon <= n)]
    else:
        origin_pos = np.arange(first_origin, n - horizon + 1, origin_stride)
    if origin_pos.size == 0:
        raise ValueError('series too short for one forecast horizon after first_origin')
    values = series.to_numpy(dtype=float)
    actual = np.stack([values[p:p + horizon] for p in origin_pos])
    usable = np.isfinite(actual)
    future_times = [series.index[p:p + horizon] for p in origin_pos]
    origin_times = pd.DatetimeIndex([series.index[p - 1] for p in origin_pos])
    lo_i = int(np.argmin(np.abs(QUANTILE_LEVELS - (1 - coverage_level) / 2)))
    hi_i = int(np.argmin(np.abs(QUANTILE_LEVELS - (1 + coverage_level) / 2)))

    raw: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for fc in forecasters:
        point = np.empty((len(origin_pos), horizon))
        own_q: Optional[np.ndarray] = np.empty(
            (len(origin_pos), horizon, len(QUANTILE_LEVELS))
        )
        for i, p in enumerate(origin_pos):
            history = series.iloc[:p]
            if i % max(1, refit_every) == 0:
                fc.fit(history)
            pt = np.asarray(fc.predict(history, future_times[i]), dtype=float)
            if nonnegative:
                pt = np.maximum(pt, 0.0)
            point[i] = pt
            if own_q is not None:
                q = fc.predict_quantiles(history, future_times[i])
                if q is None:
                    own_q = None
                else:
                    own_q[i] = np.maximum(q, 0.0) if nonnegative else q
        quant = own_q if own_q is not None else _residual_quantiles(
            point, actual, usable, origin_pos, horizon, lead_bucket, nonnegative,
            condition=residual_condition, future_times=future_times,
            interval_minutes=interval_minutes, tz=tz,
        )
        raw[fc.name] = (point, quant)

    for name, point in (precomputed or {}).items():
        point = np.asarray(point, dtype=float)
        if point.shape != (len(origin_pos), horizon):
            raise ValueError(
                f'precomputed[{name!r}] has shape {point.shape}, expected '
                f'{(len(origin_pos), horizon)}'
            )
        if nonnegative:
            point = np.where(np.isfinite(point), np.maximum(point, 0.0), point)
        raw[name] = (point, _residual_quantiles(
            point, actual, usable, origin_pos, horizon, lead_bucket, nonnegative,
            condition=residual_condition, future_times=future_times,
            interval_minutes=interval_minutes, tz=tz,
        ))

    # One scored set for everyone: a (origin, lead) cell counts only when
    # its label was measured and every forecaster has a distribution there.
    prob_ok = usable.copy()
    for point, quant in raw.values():
        prob_ok &= np.isfinite(quant).all(axis=-1)
    scored_origins = prob_ok.any(axis=1)
    steps_per_day = max(1, 1440 // max(1, int(interval_minutes)))
    if scored_origins.any():
        idx = np.flatnonzero(scored_origins)
        span = origin_pos[idx[-1]] + horizon - origin_pos[idx[0]]
        evaluation_days = span / steps_per_day
    else:
        evaluation_days = 0.0

    truth_profile_sd = float(np.nanstd(np.where(prob_ok, actual, np.nan)))
    leads = np.arange(horizon) // lead_bucket
    scores: Dict[str, HorizonScores] = {}
    for name, (point, quant) in raw.items():
        m = prob_ok
        err = np.where(m, point - actual, np.nan)
        crps = np.where(m, crps_from_quantiles(actual, quant), np.nan)
        inside = np.where(
            m, (actual >= quant[..., lo_i]) & (actual <= quant[..., hi_i]), np.nan,
        )
        width = np.where(m, quant[..., hi_i] - quant[..., lo_i], np.nan)
        act_sum = float(np.nansum(np.where(m, actual, np.nan)))
        pred_sum = float(np.nansum(np.where(m, point, np.nan)))
        by_lead = pd.DataFrame({
            'lead_start': np.unique(leads) * lead_bucket + 1,
            'mae': [float(np.nanmean(np.abs(err[:, leads == b]))) for b in np.unique(leads)],
            'rmse': [float(np.sqrt(np.nanmean(err[:, leads == b] ** 2))) for b in np.unique(leads)],
            'crps': [float(np.nanmean(crps[:, leads == b])) for b in np.unique(leads)],
        })
        # Origins with no scored cell are all-NaN rows; they are dropped
        # just below, so their means are computed only where one exists.
        has = m.any(axis=1)
        o_crps = np.full(len(origin_pos), np.nan)
        o_sq = np.full(len(origin_pos), np.nan)
        o_crps[has] = np.nanmean(crps[has], axis=1)
        o_sq[has] = np.nanmean(err[has] ** 2, axis=1)
        period_total = _period_total_errors(
            point, actual, prob_ok, origin_times, future_times, tz,
        )
        if len(period_total):
            pt_rmse = float(np.sqrt(
                (period_total['rmse'] ** 2 * period_total['n']).sum()
                / period_total['n'].sum()
            ))
        else:
            pt_rmse = float('nan')
        scores[name] = HorizonScores(
            name=name,
            n_origins=int(scored_origins.sum()),
            n_points=int(m.sum()),
            mae=float(np.nanmean(np.abs(err))) if m.any() else float('nan'),
            rmse=float(np.sqrt(np.nanmean(err ** 2))) if m.any() else float('nan'),
            bias_pct=(100.0 * (pred_sum / act_sum - 1.0)) if act_sum > 0 else float('nan'),
            crps=float(np.nanmean(crps)) if m.any() else float('nan'),
            coverage=float(np.nanmean(inside)) if m.any() else float('nan'),
            interval_width=float(np.nanmean(width)) if m.any() else float('nan'),
            amplitude_ratio=(
                float(np.nanstd(np.where(m, point, np.nan))) / truth_profile_sd
                if truth_profile_sd > 0 else float('nan')
            ),
            by_lead=by_lead,
            period_total_rmse=pt_rmse,
            period_total=period_total,
            origin_crps=o_crps[scored_origins],
            origin_sq_err=o_sq[scored_origins],
            origin_times=origin_times[scored_origins],
        )
    return HorizonEvaluation(
        scores=scores,
        evaluation_days=float(evaluation_days),
        enough_evidence=evaluation_days >= MIN_EVALUATION_DAYS,
        coverage_level=coverage_level,
    )


# A model whose remaining-in-day totals are this much worse than a constant
# at the history mean is not selected, whatever its CRPS. Slot-level CRPS on
# a target that sits near zero most of the time barely separates forecasts
# by level — two forecasts with the same quantiles and point forecasts 40 %
# apart score the same — while the period total separates them at once.
PERIOD_TOTAL_TOLERANCE = 1.25
GUARDRAIL_REFERENCE = 'baseline_mean'


def select_by_crps(
    evaluation: HorizonEvaluation, candidates: Sequence[str],
) -> Tuple[Optional[str], List[dict]]:
    """
    Pick the candidate with the lowest CRPS among those that pass the
    period-total guardrail.

    Returns ``(winner, rows)``. ``rows`` lists every scored forecaster,
    candidates and reference forecasts alike, best CRPS first, each with
    ``eligible`` (passed the guardrail) and ``candidate`` flags. The winner
    is ``None`` when no candidate has a finite CRPS. If every candidate
    fails the guardrail the lowest-CRPS candidate still wins — a benchmark
    must name a model — and its row says it was not eligible.
    """
    ref = evaluation.scores.get(GUARDRAIL_REFERENCE)
    limit = (
        PERIOD_TOTAL_TOLERANCE * ref.period_total_rmse
        if ref is not None and np.isfinite(ref.period_total_rmse) else float('inf')
    )
    rows = []
    for name, sc in evaluation.scores.items():
        row = sc.summary()
        total = sc.period_total_rmse
        row['eligible'] = bool(not np.isfinite(total) or total <= limit)
        row['candidate'] = name in candidates
        rows.append(row)
    rows.sort(key=lambda r: (not np.isfinite(r['crps']), r['crps']))
    scored = [r for r in rows if r['candidate'] and np.isfinite(r['crps'])]
    eligible = [r for r in scored if r['eligible']]
    pool = eligible or scored
    return (pool[0]['name'] if pool else None), rows


def paired_difference(
    candidate: np.ndarray, incumbent: np.ndarray, origin_times: pd.DatetimeIndex,
    n_boot: int = 2000, seed: int = 0,
) -> dict:
    """
    Paired loss difference (candidate − incumbent) with a day-block bootstrap.

    Per-origin losses of two forecasters on the same origins are strongly
    correlated — both miss the same unforecastable events — so the paired
    difference is far less noisy than either score. Origins issued on the
    same day overlap in their targets, so days, not origins, are resampled.

    Returns the mean difference, its 95 % interval and a verdict:
    ``'better'`` (interval entirely below zero), ``'worse'`` or
    ``'indistinguishable'``.
    """
    d = np.asarray(candidate, dtype=float) - np.asarray(incumbent, dtype=float)
    keep = np.isfinite(d)
    d = d[keep]
    days = np.asarray(pd.DatetimeIndex(origin_times)[keep].normalize())
    uniq = np.unique(days)
    if d.size == 0 or uniq.size < 2:
        return {'mean': float('nan'), 'lo': float('nan'), 'hi': float('nan'),
                'n_days': int(uniq.size), 'verdict': 'indistinguishable'}
    groups = [d[days == u] for u in uniq]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        means[b] = np.concatenate([groups[k] for k in pick]).mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    verdict = 'better' if hi < 0 else 'worse' if lo > 0 else 'indistinguishable'
    return {'mean': float(d.mean()), 'lo': float(lo), 'hi': float(hi),
            'n_days': int(uniq.size), 'verdict': verdict}
