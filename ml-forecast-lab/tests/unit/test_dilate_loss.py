"""Unit tests for the DILATE (shape + time) training loss.

These lock down the properties that make DILATE the right objective for spiky
targets: it is alignment-tolerant (a slightly mistimed spike is cheap), it
prefers a sharp-but-mistimed forecast over a flat one (where MSE would prefer
the flat line), and it produces finite gradients for the optimiser.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml_forecast_lab.models.dilate_loss import (
    dilate_per_sample,
    _soft_dtw_value,
    _soft_dtw_value_scalar,
)


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


def test_softdtw_vectorised_matches_scalar_value():
    """The vectorised banded sweep is numerically identical to the reference
    cell-by-cell recursion (validated in numpy to 0.0; this guards the torch
    port — buffers/pad/gather — in CI)."""
    torch.manual_seed(0)
    for H, band in [(8, 3), (16, 8), (24, 8), (12, 11), (5, 1)]:
        D = (torch.rand(3, H, H) ** 2) * 4.0
        v = _soft_dtw_value(D, 0.01, band)
        s = _soft_dtw_value_scalar(D, 0.01, band)
        assert torch.allclose(v, s, atol=1e-5), (H, band, float((v - s).abs().max()))


def test_softdtw_vectorised_matches_scalar_grad():
    """Gradients match the reference too (the vectorised path uses only
    out-of-place autograd ops, so first- and second-order grads are correct
    by construction; this pins the first-order parity)."""
    torch.manual_seed(1)
    H, band = 16, 8
    base = torch.rand(2, H, H) ** 2
    Dv = base.clone().requires_grad_(True)
    Ds = base.clone().requires_grad_(True)
    _soft_dtw_value(Dv, 0.05, band).sum().backward()
    _soft_dtw_value_scalar(Ds, 0.05, band).sum().backward()
    assert torch.allclose(Dv.grad, Ds.grad, atol=1e-5)


def test_dilate_long_horizon_full_objective_runs():
    """The full shape+time objective runs and back-props on a realistic
    horizon (regression guard for the second-order temporal path at scale)."""
    n = 96
    true = torch.tensor(_bump(n, 40)).unsqueeze(0)
    # Use a magnitude mismatch at the SAME location (an under-shot peak) so the
    # gradient is genuinely non-zero. A slightly-shifted but equal-height spike
    # would warp-align within the band at ~zero shape cost — a flat minimum
    # where a zero gradient is correct, not a back-prop failure.
    pred = torch.tensor(_bump(n, 40, height=0.4)).unsqueeze(0).requires_grad_(True)
    loss = dilate_per_sample(pred, true, alpha=0.5).mean()
    loss.backward()
    assert torch.isfinite(loss).all()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# Scale invariance
# --------------------------------------------------------------------------- #
# `gamma` (the soft-min temperature) is applied directly to the pairwise cost,
# so its meaning depends on that cost's magnitude. The module was written
# assuming z-scored targets, but every neural backend here denormalises inside
# its forward pass (RevIN), the target z-score is skipped whenever RevIN is on,
# and there is no pipeline-level scaler — so the cost arrives in raw sensor
# units and spans ~11 orders of magnitude across plausible sensors.
def _spiky(B, H, amp, seed=0):
    torch.manual_seed(seed)
    yt = torch.zeros(B, H)
    yt[:, ::12] = amp
    return yt


@pytest.mark.parametrize("amp", [1.0, 1e3, 1e6])
def test_loss_is_invariant_to_target_scale(amp):
    """The same forecast error, expressed in different units, must score the
    same. Otherwise every hyperparameter is unit-dependent."""
    B, H = 8, 96
    yt = _spiky(B, H, amp)
    yp = yt * 0.8                                   # 20% under-forecast at any scale
    v = dilate_per_sample(yp, yt, band=8).mean().item()
    ref_yt = _spiky(B, H, 1.0)
    ref = dilate_per_sample(ref_yt * 0.8, ref_yt, band=8).mean().item()
    assert v == pytest.approx(ref, rel=1e-4), (
        f"loss at amplitude {amp} is {v}, but {ref} at amplitude 1"
    )


@pytest.mark.parametrize("amp", [1.0, 1e2, 1e4, 1e6])
def test_gamma_still_smooths_at_realistic_sensor_magnitudes(amp):
    """`gamma` is the soft-min temperature — the thing that makes soft-DTW
    differentiable and distinguishes it from plain DTW. It is applied directly
    to the pairwise cost, so if that cost is left in raw sensor units the
    exponentials saturate and the soft-min collapses to a hard min.

    Measured on the unnormalised loss: changing gamma by three orders of
    magnitude changes the loss by a factor of 77 at amplitude 1, but by a
    factor of 1.0000 from amplitude 100 upward — gamma is a dead parameter for
    any sensor reading in watts or watt-hours, and DILATE silently degrades to
    hard DTW on most real HA sensors.
    """
    B, H = 8, 96
    yt = torch.zeros(B, H)
    yt[:, ::12] = amp
    yp = torch.zeros(B, H)
    yp[:, 2::12] = amp                      # same spikes, mistimed by 2 steps

    smoothed = dilate_per_sample(yp, yt, gamma=1.0, band=8).mean().item()
    sharp = dilate_per_sample(yp, yt, gamma=0.001, band=8).mean().item()
    assert abs(smoothed - sharp) > 1e-6 * max(1.0, abs(sharp)), (
        f"gamma has no effect at amplitude {amp} (smoothed={smoothed}, "
        f"sharp={sharp}) — the soft-min has collapsed to a hard min"
    )


def test_gradients_stay_finite_at_realistic_sensor_magnitudes():
    """A Wh-scale hot-water sensor is an ordinary case, not an edge case."""
    B, H = 8, 96
    for amp in (1.0, 3e3, 1e5, 1e7):
        yt = _spiky(B, H, amp)
        torch.manual_seed(2)
        yp = (yt + torch.randn(B, H) * amp * 0.1).requires_grad_(True)
        loss = dilate_per_sample(yp, yt, band=8).mean()
        loss.backward()
        assert torch.isfinite(loss).item(), f"loss non-finite at amplitude {amp}"
        assert torch.isfinite(yp.grad).all().item(), f"grad non-finite at amplitude {amp}"


def test_flat_window_does_not_divide_by_zero():
    """A window with zero variance has no scale to normalise by."""
    B, H = 4, 48
    yt = torch.full((B, H), 5.0)
    yp = (yt + 0.5).requires_grad_(True)
    loss = dilate_per_sample(yp, yt, band=8).mean()
    loss.backward()
    assert torch.isfinite(loss).item()
    assert torch.isfinite(yp.grad).all().item()


# --------------------------------------------------------------------------- #
# Flat windows must not dominate the batch
# --------------------------------------------------------------------------- #
# The per-window variance normalisation divides the cost matrix by var(y_true).
# A flat window has variance exactly 0, so flooring at a small CONSTANT turned
# it into a 1e8 amplifier. On a spiky load — the workload this loss exists for —
# most windows ARE flat between bursts, so those windows became the entire loss
# and the gradient from the real peaks was drowned out. Measured at H=96,
# band=8 before the fix:
#
#     normal spiky window          var 7.6e+00   loss 1.4e-01
#     all-zero window, pred 0.1    var 0.0e+00   loss 1.0e+06
#     1 spiky + 4 flat windows     batch mean 0.094 -> 80.0
#
# Flooring against the batch's own scale fixed the magnitude but left the loss
# able to go NEGATIVE on flat windows (a soft-min with a large effective gamma
# dips below zero), so a batch mean could be lowered just by adding flat
# windows. Flat windows now fall back to a normalised point loss instead.
def _spiky_and_flat(n_flat, H=96, seed=0):
    torch.manual_seed(seed)
    spiky_t = torch.zeros(1, H)
    spiky_t[0, ::12] = 10.0
    spiky_p = spiky_t + torch.randn(1, H) * 1.0
    yt = torch.cat([spiky_t] + [torch.zeros(1, H)] * n_flat)
    yp = torch.cat([spiky_p] + [torch.full((1, H), 1e-3)] * n_flat)
    return yp, yt


@pytest.mark.parametrize("n_flat", [1, 4, 16, 31])
def test_flat_windows_do_not_dominate_the_batch(n_flat):
    yp, yt = _spiky_and_flat(n_flat)
    per = dilate_per_sample(yp, yt, band=8)
    spiky, flats = per[0], per[1:]
    assert float(flats.max()) < float(spiky), (
        f"a flat window scored {float(flats.max()):.3e} against the real "
        f"window's {float(spiky):.3e} — the batch is measuring the gaps"
    )


@pytest.mark.parametrize("n_flat", [1, 4, 16, 31])
def test_loss_is_never_negative(n_flat):
    """A negative per-window loss lets the batch mean be driven down simply by
    including more flat windows, which is not a signal about the forecast."""
    yp, yt = _spiky_and_flat(n_flat)
    per = dilate_per_sample(yp, yt, band=8)
    assert float(per.min()) >= 0.0, f"negative loss: {float(per.min()):.3e}"


def test_peaks_still_drive_the_gradient():
    yp, yt = _spiky_and_flat(15)
    yp = yp.requires_grad_(True)
    dilate_per_sample(yp, yt, band=8).mean().backward()
    g = yp.grad
    assert torch.isfinite(g).all()
    assert float(g[0].abs().sum()) > float(g[1:].abs().sum()), (
        "fifteen flat windows carry more gradient than the one real window"
    )


def test_an_entirely_flat_batch_falls_back_to_mae():
    """No window has any shape or timing, so there is nothing for soft-DTW to
    measure and no scale to normalise against. Normalising anyway divided by
    the absolute floor and returned ~1e6 for a trivially small error."""
    H = 96
    yt = torch.zeros(4, H)
    yp = torch.full((4, H), 0.1)
    got = float(dilate_per_sample(yp, yt, band=8).mean())
    assert got == pytest.approx(0.1, abs=1e-6), f"expected plain MAE 0.1, got {got}"


def test_flat_fallback_does_not_break_scale_invariance():
    H = 96
    vals = []
    for amp in (1.0, 1e3, 1e6):
        yt = torch.zeros(4, H)
        yt[:, ::12] = amp
        vals.append(float(dilate_per_sample(yt * 0.8, yt, band=8).mean()))
    assert vals[0] == pytest.approx(vals[1], rel=1e-4)
    assert vals[0] == pytest.approx(vals[2], rel=1e-4)


class TestValidationUsesTheSameFormulaAsTraining:
    """Validation runs the whole loss under `torch.no_grad()` — every neural
    backend does. The temporal term needs a graph, and the old
    `except RuntimeError: return shape` was written as a safety net; under
    no_grad it was the ONLY branch that ever ran.

    So training returned `alpha*shape + (1-alpha)*temporal` and validation
    returned bare `shape`. At the shipped alpha=0.5 that inflates validation by
    ~2x, pinning the epoch chart's val curve at almost exactly twice train from
    epoch one for every DILATE run — indistinguishable from severe overfitting,
    and it never improves, because it is an artefact rather than a gap.
    """

    @staticmethod
    def _batch(H=48, seed=0):
        torch.manual_seed(seed)
        yt = torch.zeros(4, H)
        yt[:, 12:14] = 3.0
        yt[:, 36:38] = 3.0
        return yt * 0.7 + torch.randn(4, H) * 0.05, yt

    def test_val_equals_train_on_identical_input(self):
        base, yt = self._batch()
        yp = base.clone().requires_grad_(True)
        train = float(dilate_per_sample(yp, yt, band=8).mean().detach())
        with torch.no_grad():
            val = float(dilate_per_sample(base.clone(), yt, band=8).mean())
        assert val == pytest.approx(train, rel=1e-6), (
            f"val {val:.6f} vs train {train:.6f} (ratio {val/train:.4f}) — the "
            f"two paths are computing different formulas"
        )

    def test_training_still_gets_gradients(self):
        base, yt = self._batch()
        yp = base.clone().requires_grad_(True)
        dilate_per_sample(yp, yt, band=8).mean().backward()
        assert torch.isfinite(yp.grad).all()
        assert float(yp.grad.norm()) > 0

    def test_no_grad_path_does_not_leak_into_autograd(self):
        """The temporal term is recovered on a detached copy; nothing from it
        may attach to the graph the optimiser steps on."""
        base, yt = self._batch()
        with torch.no_grad():
            out = dilate_per_sample(base.clone(), yt, band=8)
        assert out.requires_grad is False

    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75])
    def test_holds_at_every_shape_time_mix(self, alpha):
        base, yt = self._batch()
        yp = base.clone().requires_grad_(True)
        train = float(dilate_per_sample(yp, yt, alpha=alpha, band=8).mean().detach())
        with torch.no_grad():
            val = float(dilate_per_sample(base.clone(), yt, alpha=alpha, band=8).mean())
        assert val == pytest.approx(train, rel=1e-6)
