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


def _soft_dtw_value(
    D: "torch.Tensor", gamma: float, band: int
) -> "torch.Tensor":
    """Soft-DTW discrepancy for a batch of pairwise cost matrices.

    Parameters
    ----------
    D : torch.Tensor
        Pairwise cost, shape ``(B, N, M)`` with ``D[b, i, j]`` the cost of
        matching predicted step ``i`` to target step ``j``.
    gamma : float
        Soft-min temperature. Smaller → closer to hard DTW; larger → smoother
        (and more convex) surrogate.
    band : int
        Sakoe-Chiba band half-width. Only cells with ``|i - j| <= band`` are
        evaluated, bounding cost at O(N · band).

    Returns
    -------
    torch.Tensor
        Shape ``(B,)`` soft-DTW value, differentiable w.r.t. ``D``.
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
