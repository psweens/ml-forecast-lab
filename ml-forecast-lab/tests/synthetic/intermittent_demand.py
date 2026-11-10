"""
Synthetic intermittent-demand counter.

A daily-resetting cumulative sensor whose demand is mostly a small constant
background with occasional large events at habitual times of day — the
shape of any metered consumable drawn in bursts. Statistics the generator
is calibrated to, per 30-minute slot:

  - ~90 % of slots non-zero (background in 0.1 steps, ~0.2 per slot)
  - events (>= 1 per slot) in ~20 % of slots carrying ~85 % of demand
  - median 0.2, mean ~1.25, 99th percentile ~14
  - weekday peaks 07-09 h and 17-20 h local, weekend morning peak 07-11 h,
    highest totals on Sunday

The profile is defined in LOCAL time (``TZ``) and the counter resets at
local midnight, so a window that spans a daylight-saving change exercises
both the reset handling and the calendar features.

``slot_truth`` returns the per-slot ground truth; ``recorder_rows`` turns it
into change-only history rows in the shape Home Assistant's history API
returns, with optional outages. Used by ``run_intermittent_demand.py`` and
by the unit tests that pin target construction.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

TZ = "Europe/London"
STEP = 0.1


def event_probability(local_hour: int, weekend: bool) -> float:
    """P(event in a 30-minute slot) by local hour."""
    h = local_hour
    if h < 6:
        return 0.004
    if not weekend:
        if h == 6:
            return 0.15
        if h < 9:
            return 0.70
        if h < 12:
            return 0.16
        if h < 17:
            return 0.12
        if h < 20:
            return 0.50
        if h < 22:
            return 0.22
        return 0.06
    if h == 6:
        return 0.06
    if h < 11:
        return 0.50
    if h < 17:
        return 0.20
    if h < 20:
        return 0.42
    if h < 22:
        return 0.20
    return 0.06


def _event_scale(local_hour: int, weekend: bool, sunday: bool) -> float:
    morning_end = 11 if weekend else 9
    if 7 <= local_hour < morning_end:
        peak = 1.9
    elif 17 <= local_hour < 20:
        peak = 1.25
    else:
        peak = 0.8
    return peak * (1.25 if sunday else 1.0)


def slot_truth(start, days: int, seed: int = 0) -> Tuple[pd.Series, pd.Series]:
    """Per-slot demand and its event probability on a tz-aware UTC grid."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 48, freq="30min", tz="UTC")
    local = idx.tz_convert(TZ)
    y = np.zeros(len(idx))
    p = np.zeros(len(idx))
    for i, t in enumerate(local):
        weekend = t.dayofweek >= 5
        p[i] = event_probability(t.hour, weekend)
        background = STEP * rng.poisson(2.0)
        event = 0.0
        if rng.random() < p[i]:
            scale = _event_scale(t.hour, weekend, t.dayofweek == 6)
            event = min(1.0 + scale * rng.lognormal(1.0, 0.8), 30.0)
        y[i] = round(background + event, 1)
    return pd.Series(y, index=idx), pd.Series(p, index=idx)


def recorder_rows(
    y: pd.Series,
    seed: int = 1,
    outages: Iterable[Tuple[pd.Timestamp, pd.Timestamp]] = (),
) -> List[dict]:
    """Change-only history rows of a counter that resets at local midnight.

    Each slot's demand arrives as one to five increments at random offsets
    inside the slot. Rows whose timestamp falls inside an ``outages`` span
    are withheld, as when the sensor was unavailable: the counter keeps
    counting, so the first row after the span carries the whole rise.
    """
    rng = np.random.default_rng(seed)
    events: List[Tuple[pd.Timestamp, float]] = []
    total, day = 0.0, None
    for t, v in y.items():
        local_day = t.tz_convert(TZ).date()
        if day is not None and local_day != day:
            total = 0.0
            events.append((t + pd.Timedelta(seconds=2), 0.0))
        day = local_day
        n = int(round(v / STEP))
        if n == 0:
            continue
        k = min(n, int(rng.integers(1, 6)))
        offsets = np.sort(rng.uniform(60, 1740, size=k))
        cuts = (
            np.sort(rng.choice(np.arange(1, n), size=k - 1, replace=False))
            if k > 1 else np.array([], dtype=int)
        )
        parts = np.diff(np.concatenate([[0], cuts, [n]]))
        for off, c in zip(offsets, parts):
            total = round(total + STEP * c, 1)
            events.append((t + pd.Timedelta(seconds=float(off)), total))
    outages = list(outages)
    return [
        {"last_changed": ts.isoformat(), "state": f"{val:.1f}"}
        for ts, val in events
        if not any(a <= ts <= b for a, b in outages)
    ]


def expected_demand(index: pd.DatetimeIndex) -> pd.Series:
    """E[demand] per slot for a tz-aware UTC index — the profile a forecast
    with no information beyond the calendar should reproduce."""
    rng = np.random.default_rng(12345)
    draws = rng.lognormal(1.0, 0.8, size=200_000)
    cache: dict = {}
    local = index.tz_convert(TZ)
    out = np.zeros(len(index))
    for i, t in enumerate(local):
        weekend = t.dayofweek >= 5
        scale = _event_scale(t.hour, weekend, t.dayofweek == 6)
        if scale not in cache:
            cache[scale] = float(np.minimum(1.0 + scale * draws, 30.0).mean())
        out[i] = STEP * 2.0 + event_probability(t.hour, weekend) * cache[scale]
    return pd.Series(out, index=index)
