"""Unit tests for the DILATE (shape + time) training loss.

These lock down the properties that make DILATE the right objective for spiky
targets: it is alignment-tolerant (a slightly mistimed spike is cheap), it
prefers a sharp-but-mistimed forecast over a flat one (where MSE would prefer
the flat line), and it produces finite gradients for the optimiser.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml_forecast_lab.models.dilate_loss import dilate_per_sample, _soft_dtw_value


def _bump(n, centre, width=2, height=1.0):
    v = np.zeros(n, dtype=np.float32)
    for k in range(width):
        if 0 <= centre + k < n:
            v[centre + k] = height
    return v


def test_softdtw_tolerates_time_shift_where_mse_does_not():
    """Soft-DTW between a bump and its one-step shift is much smaller than the
    squared error — the alignment-invariance that stops the double-penalty."""
    n = 16
    true = torch.tensor(_bump(n, 6)).unsqueeze(0)
    shifted = torch.tensor(_bump(n, 7)).unsqueeze(0)

    diff = shifted.unsqueeze(2) - true.unsqueeze(1)
    D = diff * diff
    sdtw = float(_soft_dtw_value(D, gamma=0.01, band=n)[0])
    mse = float(((shifted - true) ** 2).sum())

    assert sdtw < mse, f"soft-DTW {sdtw} should be < MSE {mse} for a 1-step shift"


def test_dilate_prefers_sharp_shifted_over_flat():
    """The core property: a sharp spike one step early beats a flat line that
    hedges through the middle — the opposite of what MSE rewards."""
    n = 24
    true = torch.tensor(_bump(n, 10)).unsqueeze(0)
    sharp_shifted = torch.tensor(_bump(n, 11)).unsqueeze(0)
    flat = torch.full((1, n), float(true.mean()))

    # Pure shape term (alpha=1) — the robust, parameter-free comparison.
    d_sharp = float(dilate_per_sample(sharp_shifted, true, alpha=1.0)[0])
    d_flat = float(dilate_per_sample(flat, true, alpha=1.0)[0])
    assert d_sharp < d_flat, (
        f"DILATE should prefer the sharp spike ({d_sharp}) over the flat "
        f"line ({d_flat})"
    )

    # MSE, by contrast, prefers the flat line — the failure DILATE fixes.
    mse_sharp = float(((sharp_shifted - true) ** 2).mean())
    mse_flat = float(((flat - true) ** 2).mean())
    assert mse_flat < mse_sharp


def test_dilate_full_objective_runs_and_is_finite():
    """The shape+time objective (alpha<1, needs the soft-alignment gradient)
    must produce a finite per-sample loss."""
    n = 20
    true = torch.tensor(_bump(n, 8)).unsqueeze(0)
    pred = torch.tensor(_bump(n, 9)).unsqueeze(0).requires_grad_(True)
    loss = dilate_per_sample(pred, true, alpha=0.5, gamma=0.01)
    assert loss.shape == (1,)
    assert torch.isfinite(loss).all()


def test_dilate_gradients_flow():
    n = 16
    true = torch.tensor(_bump(n, 6)).unsqueeze(0)
    pred = torch.zeros(1, n, requires_grad=True)
    loss = dilate_per_sample(pred, true, alpha=0.5, gamma=0.01).mean()
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0.0


def test_dilate_degenerates_to_mae_for_single_step():
    pred = torch.tensor([[2.0]])
    true = torch.tensor([[5.0]])
    loss = dilate_per_sample(pred, true)
    assert float(loss[0]) == pytest.approx(3.0)


def test_dilate_batched_shape():
    n = 12
    true = torch.stack([torch.tensor(_bump(n, 4)), torch.tensor(_bump(n, 7))])
    pred = torch.stack([torch.tensor(_bump(n, 5)), torch.tensor(_bump(n, 7))])
    loss = dilate_per_sample(pred, true, alpha=1.0)
    assert loss.shape == (2,)
    # Exact-match sample (index 1) has lower loss than the shifted one (index 0).
    assert float(loss[1]) <= float(loss[0])


def test_dilate_default_band_is_capped():
    """The default band is a small constant, not H // 2 — so long horizons
    stay banded (cheap) instead of computing almost the full lattice."""
    from ml_forecast_lab.models.dilate_loss import _DEFAULT_BAND_CAP
    assert _DEFAULT_BAND_CAP < 24  # smaller than H//2 for the typical 48-step horizon


def test_dilate_band_cap_still_disciplines_far_drift():
    """With the capped band, a spike drifted *beyond* the band costs more than
    one within it — the timing term stays meaningful on a long horizon (the old
    H//2 band would have let a spike drift half the horizon and still align)."""
    from ml_forecast_lab.models.dilate_loss import _DEFAULT_BAND_CAP
    n = 64
    true = torch.tensor(_bump(n, 20)).unsqueeze(0)
    near = torch.tensor(_bump(n, 20 + 2)).unsqueeze(0)                       # within band
    far = torch.tensor(_bump(n, 20 + _DEFAULT_BAND_CAP + 6)).unsqueeze(0)    # beyond band
    d_near = float(dilate_per_sample(near, true, alpha=1.0)[0])
    d_far = float(dilate_per_sample(far, true, alpha=1.0)[0])
    assert d_far > d_near


def test_dilate_long_horizon_full_objective_runs():
    """The full shape+time objective runs and back-props on a realistic
    horizon (regression guard for the second-order temporal path at scale)."""
    n = 96
    true = torch.tensor(_bump(n, 40)).unsqueeze(0)
    pred = torch.tensor(_bump(n, 43)).unsqueeze(0).requires_grad_(True)
    loss = dilate_per_sample(pred, true, alpha=0.5).mean()
    loss.backward()
    assert torch.isfinite(loss).all()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0.0
