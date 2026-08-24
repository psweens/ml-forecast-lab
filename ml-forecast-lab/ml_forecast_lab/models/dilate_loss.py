"""
DILATE loss — shape + temporal training objective for sharp, spiky targets.

Plain MSE / MAE / Huber over a multi-step horizon push a neural forecaster
toward the conditional mean, which on a spiky series (hot water, EV, appliance
demand) is a smooth envelope — the "it flattens my spikes" failure. A spike
predicted one step early is penalised twice by a point loss (a false alarm at
the predicted step + a miss at the true step), so the model learns not to
reach up at all.

DILATE (Le Guen & Thome, NeurIPS 2019, "Shape and Time Distortion Loss") fixes
this by scoring **shape** and **timing** separately:

    L_DILATE = alpha * L_shape + (1 - alpha) * L_temporal

* ``L_shape`` is the soft-DTW (Cuturi & Blondel 2017) between the predicted and
  target horizon vectors — an *alignment-invariant* shape match, so a correctly
  shaped but slightly mistimed spike is cheap.
* ``L_temporal`` penalises *how far* that optimal alignment drifts from the
  diagonal (using the soft alignment matrix = the gradient of soft-DTW w.r.t.
  the pairwise cost), so the model is still pushed toward the right time —
  it just isn't double-penalised for being one step off.

A known, useful property: DILATE-trained outputs are visibly **sharper** than
MSE-trained ones, which is exactly what we want for peak-bearing loads.

Implementation notes
--------------------
* CPU-only (Pi target): pure-PyTorch autograd recursion, vectorised over the
  batch, with a **Sakoe-Chiba band** so the cost is O(H · band) rather than
  O(H²). Gradients come from autograd (no hand-written backward to get wrong);
  the soft alignment for the temporal term is obtained with
  ``torch.autograd.grad(..., create_graph=True)``.
* The loss expects the per-window **horizon vector** ``(batch, H)``. With
  ``H < 2`` there is no shape/time to speak of, so it degrades to MAE.
* Best paired with RevIN / z-score normalised targets (which every neural
  backend here uses), so the soft-min temperature ``gamma`` is on a stable
  scale regardless of the sensor's units.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

# Default Sakoe-Chiba band half-width (in horizon steps) when the caller does
# not pin ``band``. DILATE only needs to tolerate a *small* misalignment — a
# few steps either side — so the warping window is a small constant, not a
# fraction of the horizon. The previous ``H // 2`` default made the band as
# wide as half the horizon, which (a) computed almost the full H×H lattice
# (the dominant cost, doubled again by the second-order temporal term) and
# (b) let a spike drift up to ±H/2 steps and still count as "aligned", which
# defeats the temporal term. A small cap is both far cheaper (≈4–5× fewer
# cells at a 48-step horizon) and makes the timing penalty meaningful.
# Override per call / per backend with ``dilate_band``.
_DEFAULT_BAND_CAP = 8

# Floor for the per-window cost scale. A window that is entirely flat has zero
# variance; without a floor the normalisation would divide by ~0.
_MIN_COST_SCALE = 1e-8


def _soft_dtw_value_scalar(
    D: "torch.Tensor", gamma: float, band: int
) -> "torch.Tensor":
    """Reference soft-DTW (cell-by-cell Python recursion).

    Kept as the readable, obviously-correct definition that
    :func:`_soft_dtw_value` (the vectorised production path) is parity-tested
    against. Same contract: ``D`` is ``(B, N, M)`` pairwise cost, returns a
    ``(B,)`` soft-DTW value differentiable w.r.t. ``D``; only cells with
    ``|i - j| <= band`` are evaluated.
    """
    B, N, M = D.shape
    BIG = 1e10  # finite "infinity": exp(-BIG/gamma) underflows to 0 cleanly
    zeros = torch.zeros(B, device=D.device, dtype=D.dtype)
    big = torch.full((B,), BIG, device=D.device, dtype=D.dtype)

    # R is the accumulated-cost lattice, keyed (i, j) → (B,) tensor, 1-indexed.
    # Stored as a dict so the autograd graph is built functionally (in-place
    # writes into a single tensor would break differentiation).
    R: dict = {(0, 0): zeros}
    for i in range(1, N + 1):
        R[(i, 0)] = big
    for j in range(1, M + 1):
        R[(0, j)] = big

    for i in range(1, N + 1):
        jlo = max(1, i - band)
        jhi = min(M, i + band)
        for j in range(jlo, jhi + 1):
            r0 = R.get((i - 1, j - 1), big)
            r1 = R.get((i - 1, j), big)
            r2 = R.get((i, j - 1), big)
            stacked = torch.stack([r0, r1, r2], dim=0)  # (3, B)
            soft_min = -gamma * torch.logsumexp(-stacked / gamma, dim=0)
            R[(i, j)] = D[:, i - 1, j - 1] + soft_min

    return R[(N, M)]


@functools.lru_cache(maxsize=64)
def _band_plan(N: int, band: int):
    """Precompute the per-anti-diagonal index plan for a banded soft-DTW sweep.

    Geometry only (no data, no autograd), cached per ``(N, band)``. The
    soft-DTW recurrence ``R[i,j] = D[i,j] + softmin(R[i-1,j-1], R[i-1,j],
    R[i,j-1])`` has no dependency *within* an anti-diagonal ``s = i + j`` — each
    cell only needs diagonals ``s-1`` and ``s-2`` — so a whole diagonal can be
    computed in one vectorised tensor op. That turns the ``O(N·band)`` Python
    loop of :func:`_soft_dtw_value_scalar` into ``~2N`` iterations while still
    touching only the banded cells (so the autograd graph stays banded too).

    Returns ``(steps, final_pos)`` where ``steps`` has one entry per diagonal
    ``s = 2..2N`` of ``(ii, jj, gd, gu, gl)`` Long index tensors:

    * ``ii, jj`` index this diagonal's cells into ``D`` (``i-1``, ``j-1``);
    * ``gd, gu, gl`` gather the three neighbours from the previous two
      diagonals' R-vectors — a missing neighbour points at a padded "BIG"
      column (index ``K_prev``), so out-of-band / boundary cells read as +inf.

    ``final_pos`` is the index of cell ``(N, N)`` in the last diagonal.
    """
    def diag_cells(s):
        # valid i on diagonal s: in-bounds AND within the band |2i - s| <= band
        ilo = max(1, s - N, -((-(s - band)) // 2))   # max(1, s-N, ceil((s-band)/2))
        ihi = min(N, s - 1, (s + band) // 2)         # min(N, s-1, floor((s+band)/2))
        return list(range(ilo, ihi + 1)) if ilo <= ihi else []

    cells = {0: [0], 1: []}                          # diag 0 = origin; diag 1 = boundary
    pos = {0: {0: 0}, 1: {}}
    for s in range(2, 2 * N + 1):
        cells[s] = diag_cells(s)
        pos[s] = {i: p for p, i in enumerate(cells[s])}

    steps = []
    for s in range(2, 2 * N + 1):
        K2, K1 = len(cells[s - 2]), len(cells[s - 1])
        p2, p1 = pos[s - 2], pos[s - 1]
        ii, jj, gd, gu, gl = [], [], [], [], []
        for i in cells[s]:
            j = s - i
            ii.append(i - 1)
            jj.append(j - 1)
            gd.append(p2.get(i - 1, K2))             # (i-1, j-1) on diag s-2
            gu.append(p1.get(i - 1, K1))             # (i-1, j)   on diag s-1
            gl.append(p1.get(i, K1))                 # (i,   j-1) on diag s-1
        steps.append((
            torch.tensor(ii, dtype=torch.long),
            torch.tensor(jj, dtype=torch.long),
            torch.tensor(gd, dtype=torch.long),
            torch.tensor(gu, dtype=torch.long),
            torch.tensor(gl, dtype=torch.long),
        ))
    return steps, pos[2 * N][N]


def _soft_dtw_value(
    D: "torch.Tensor", gamma: float, band: int
) -> "torch.Tensor":
    """Soft-DTW discrepancy for a batch of pairwise cost matrices.

    Vectorised banded anti-diagonal sweep — numerically identical to
    :func:`_soft_dtw_value_scalar` (parity-tested), but ``~2N`` Python
    iterations instead of ``O(N·band)``, which matters for the second-order
    autograd graph DILATE's temporal term builds. Only banded cells are
    touched, so the graph stays ``O(N·band)``.

    Parameters
    ----------
    D : torch.Tensor
        Pairwise cost, shape ``(B, N, M)`` (square ``N == M`` for DILATE).
    gamma : float
        Soft-min temperature.
    band : int
        Sakoe-Chiba band half-width (``|i - j| <= band``).

    Returns
    -------
    torch.Tensor
        Shape ``(B,)`` soft-DTW value, differentiable w.r.t. ``D``.
    """
    B, N, M = D.shape
    if N != M:  # DILATE always passes a square cost; be safe for odd callers
        return _soft_dtw_value_scalar(D, gamma, band)

    steps, final_pos = _band_plan(int(N), int(band))
    BIG = 1e10
    device, dtype = D.device, D.dtype

    # Two rolling diagonals, built functionally (no in-place writes) so the
    # autograd graph — including the create_graph=True second-order pass of the
    # temporal term — flows through cleanly. Diag 0 is the origin (0,0)=0;
    # diag 1 is the all-BIG boundary (represented as an empty vector — any
    # gather into it lands on the padded BIG column).
    R_prev2 = torch.zeros(B, 1, device=device, dtype=dtype)   # diagonal s-2
    R_prev1 = torch.zeros(B, 0, device=device, dtype=dtype)   # diagonal s-1
    big_col = torch.full((B, 1), BIG, device=device, dtype=dtype)

    for (ii, jj, gd, gu, gl) in steps:
        ii = ii.to(device); jj = jj.to(device)
        gd = gd.to(device); gu = gu.to(device); gl = gl.to(device)
        # Pad each previous diagonal with a trailing BIG column; a "missing"
        # gather index equals K_prev and so selects that +inf sentinel.
        Rpp = torch.cat([R_prev2, big_col], dim=1)
        Rp = torch.cat([R_prev1, big_col], dim=1)
        Dd = D[:, ii, jj]                                     # (B, K)
        neigh = torch.stack([
            Rpp.index_select(1, gd),                         # (i-1, j-1)
            Rp.index_select(1, gu),                          # (i-1, j)
            Rp.index_select(1, gl),                          # (i,   j-1)
        ], dim=0)                                            # (3, B, K)
        soft_min = -gamma * torch.logsumexp(-neigh / gamma, dim=0)
        R_cur = Dd + soft_min                                # (B, K)
        R_prev2, R_prev1 = R_prev1, R_cur

    return R_prev1[:, final_pos]


def dilate_per_sample(
    y_pred: "torch.Tensor",
    y_true: "torch.Tensor",
    alpha: float = 0.5,
    gamma: float = 0.01,
    band: Optional[int] = None,
) -> "torch.Tensor":
    """Per-sample DILATE loss over a forecast horizon.

    Parameters
    ----------
    y_pred, y_true : torch.Tensor
        Horizon vectors, shape ``(B, H)`` (or ``(B,)`` / ``(B, 1)`` for a
        single step, in which case the loss degrades to MAE).
    alpha : float, default 0.5
        Shape/time mix. ``1.0`` is pure soft-DTW (shape only); lower values
        add the temporal-distortion penalty.
    gamma : float, default 0.01
        Soft-min temperature for soft-DTW.
    band : int, optional
        Sakoe-Chiba band half-width (max timing tolerance, in steps).
        ``None`` → ``min(max(1, H // 2), _DEFAULT_BAND_CAP)`` — a small
        constant window rather than half the horizon, so the loss is cheap
        (≈O(H · band)) and the timing term stays meaningful.

    Returns
    -------
    torch.Tensor
        Shape ``(B,)`` loss, lower = better, differentiable w.r.t. ``y_pred``.
    """
    if y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(-1)
        y_true = y_true.unsqueeze(-1)
    B, H = y_pred.shape[0], y_pred.shape[-1]

    if H < 2:
        # No shape/time structure — fall back to MAE so the call is always valid.
        return torch.abs(y_pred - y_true).reshape(B, -1).mean(dim=-1)

    if band is None:
        # Small constant window (not H // 2): keeps the soft-DTW banded for
        # real (≈4–5× fewer cells at a 48-step horizon, halved again under
        # the second-order temporal term) and keeps the timing penalty
        # meaningful. Override via ``dilate_band``.
        band = min(max(1, H // 2), _DEFAULT_BAND_CAP)

    # Pairwise squared cost: D[b, i, j] = (pred_i - true_j)^2.
    diff = y_pred.unsqueeze(2) - y_true.unsqueeze(1)  # (B, H, H)
    D = diff * diff

    # Normalise the cost to a per-window scale before it reaches the softmin.
    #
    # `gamma` is the soft-min temperature — the thing that makes soft-DTW
    # differentiable, and the only reason to prefer it over plain DTW. It is
    # applied directly to D, so its meaning depends entirely on D's magnitude.
    # The module was written assuming targets arrive z-scored; they do not.
    # Every neural backend here denormalises inside its forward pass (RevIN),
    # the target z-score is skipped whenever RevIN is on, and there is no
    # pipeline-level scaler — so D arrives in raw sensor units.
    #
    # Measured at H=96, band=8, as the ratio of loss(gamma=1.0) to
    # loss(gamma=0.001) — i.e. how much gamma still does:
    #
    #     amplitude       1     ratio 77.0     gamma dominates
    #     amplitude     100     ratio  0.994
    #     amplitude    1e+04    ratio  1.0000  gamma is inert
    #     amplitude    1e+06    ratio  1.0000  gamma is inert
    #
    # Above roughly amplitude 100 the exponentials saturate, the soft-min is a
    # hard min, and DILATE has quietly become plain DTW. Most HA power sensors
    # report watts and most energy sensors watt-hours, so that is the common
    # case, not the edge case. The loss value itself also spans 7.5e-3 to
    # 1.04e+08 for the identical relative error.
    #
    # Dividing by a detached per-window variance fixes both: the loss is exactly
    # scale-invariant (0.259418 at every amplitude tested) and gamma keeps a
    # consistent effect across the whole range.
    #
    # Detached, so this is a normalisation rather than a gradient path.
    scale = y_true.detach().var(dim=-1, keepdim=True, unbiased=False)
    scale = scale.clamp_min(_MIN_COST_SCALE).unsqueeze(-1)  # (B, 1, 1)
    D = D / scale

    # Shape term — normalise by H so gamma stays meaningful across horizons.
    shape = _soft_dtw_value(D, gamma, band) / float(H)  # (B,)

    if alpha >= 1.0:
        return shape

    # Temporal term — the soft alignment is exactly d(soft-DTW)/dD. Weight it
    # by the squared off-diagonal distance so drift from "on time" is penalised.
    # create_graph=True keeps it differentiable for the optimiser step; if no
    # graph exists (e.g. under torch.no_grad in validation) fall back to shape.
    try:
        align = torch.autograd.grad(
            shape.sum(), D, create_graph=True, retain_graph=True,
        )[0]  # (B, H, H)
    except RuntimeError:
        return shape

    idx = torch.arange(H, device=D.device, dtype=D.dtype)
    omega = (idx.view(H, 1) - idx.view(1, H)) ** 2 / float(H * H)  # (H, H)
    temporal = (align * omega.unsqueeze(0)).sum(dim=(1, 2))  # (B,)
    return alpha * shape + (1.0 - alpha) * temporal
