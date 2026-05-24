"""Interval↔cumulative loss balance (v2.40).

Covers the two modes of ForecastModel._composite_horizon_loss:
  - loss_balance=None → legacy additive L = L_interval + λ·L_daily (unchanged);
  - loss_balance=α    → EMA-normalised convex blend
                        L = (1-α)·L_interval/ema_i + α·L_daily/ema_d.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from ml_forecast_lab.models.base import ForecastModel


class _Dummy:
    """Minimal carrier for the loss method's only instance state."""
    def __init__(self, loss_balance=None):
        self.loss_balance = loss_balance
        self._loss_ema = None


def _loss(obj, yp, yt, daily_weight=0.0, w=None):
    crit = torch.nn.MSELoss(reduction="none")
    return ForecastModel._composite_horizon_loss(obj, yp, yt, crit, w, daily_weight)


def _mk(batch=4, H=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    yp = torch.rand(batch, H, generator=g, requires_grad=True)
    yt = torch.rand(batch, H, generator=g)
    return yp, yt


def test_legacy_interval_only_matches_plain_mse():
    """loss_balance=None, daily_weight=0 → exactly the per-interval mean MSE."""
    yp, yt = _mk()
    loss, _ = _loss(_Dummy(loss_balance=None), yp, yt, daily_weight=0.0)
    expected = torch.nn.functional.mse_loss(yp, yt)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_legacy_additive_matches_interval_plus_lambda_daily():
    """loss_balance=None, daily_weight=λ → L = interval + λ·daily (the
    pre-v2.40 formula), byte-for-byte."""
    yp, yt = _mk()
    lam = 0.5
    loss, _ = _loss(_Dummy(loss_balance=None), yp, yt, daily_weight=lam)

    interval = torch.nn.functional.mse_loss(yp, yt)
    H = yp.size(1)
    cum = ((yp.cumsum(1) - yt.cumsum(1)) ** 2).mean(dim=1) / float(H)
    daily = cum.mean()
    expected = interval + lam * daily
    assert torch.allclose(loss, expected, atol=1e-6)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_blend_first_call_normalises_both_terms_to_one(alpha):
    """On the first blended call the EMA seeds to each term's own value, so
    both normalised terms equal 1.0 and the blend is (1-α)·1 + α·1 = 1.0
    regardless of α. Confirms the magnitude normalisation makes α a balance
    of *influence*, not raw scale."""
    yp, yt = _mk()
    loss, _ = _loss(_Dummy(loss_balance=alpha), yp, yt)
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-5)


def test_blend_alpha_extremes_track_their_own_term():
    """After the EMA has moved, α=0 must equal interval/ema_i and α=1 must
    equal daily/ema_d — i.e. the far ends are pure interval / pure cumulative."""
    # Seed EMA with one batch, then evaluate a different batch.
    obj0 = _Dummy(loss_balance=0.0)
    obj1 = _Dummy(loss_balance=1.0)
    yp_seed, yt_seed = _mk(seed=1)
    _loss(obj0, yp_seed, yt_seed)
    _loss(obj1, yp_seed, yt_seed)

    yp, yt = _mk(seed=2)
    loss0, _ = _loss(obj0, yp, yt)
    loss1, _ = _loss(obj1, yp, yt)

    interval = torch.nn.functional.mse_loss(yp, yt)
    H = yp.size(1)
    daily = (((yp.cumsum(1) - yt.cumsum(1)) ** 2).mean(dim=1) / float(H)).mean()
    assert torch.allclose(loss0, interval / obj0._loss_ema["interval"], atol=1e-5)
    assert torch.allclose(loss1, daily / obj1._loss_ema["daily"], atol=1e-5)


def test_blend_ema_updates_under_grad_but_not_under_no_grad():
    """EMA must advance on training calls (grad enabled) and stay frozen on
    validation calls (under torch.no_grad), so val-loss stays comparable for
    early stopping."""
    obj = _Dummy(loss_balance=0.5)
    yp, yt = _mk(seed=3)
    _loss(obj, yp, yt)  # seeds EMA
    seeded = dict(obj._loss_ema)

    # Training call with different data → EMA should move.
    yp2, yt2 = _mk(seed=4)
    _loss(obj, yp2 * 3.0, yt2)  # inflate to force a change
    assert obj._loss_ema["interval"] != seeded["interval"]

    moved = dict(obj._loss_ema)
    # Validation call under no_grad → EMA must NOT move.
    yp3, yt3 = _mk(seed=5)
    with torch.no_grad():
        ypv = yp3.detach() * 10.0
        _loss(obj, ypv, yt3)
    assert obj._loss_ema == moved


def test_blend_single_horizon_collapses_to_interval():
    """H=1 has no cumulative trajectory, so the blend returns the raw
    interval loss regardless of α (no divide-by-EMA, no daily term)."""
    g = torch.Generator().manual_seed(6)
    yp = torch.rand(4, 1, generator=g, requires_grad=True)
    yt = torch.rand(4, 1, generator=g)
    loss, _ = _loss(_Dummy(loss_balance=0.7), yp, yt)
    assert torch.allclose(loss, torch.nn.functional.mse_loss(yp, yt), atol=1e-6)


def test_blend_alpha_clamped_to_unit_interval():
    """Defensive clamp: out-of-range α behaves like the nearest bound."""
    yp, yt = _mk(seed=7)
    lo, _ = _loss(_Dummy(loss_balance=-5.0), yp, yt)
    hi, _ = _loss(_Dummy(loss_balance=9.0), yp, yt)
    # First-call normalisation → both still 1.0; the point is no crash / no nan.
    assert torch.isfinite(lo) and torch.isfinite(hi)


def test_blend_loss_is_differentiable():
    """The blended loss must support backward() so training actually runs."""
    yp, yt = _mk(seed=8)
    loss, _ = _loss(_Dummy(loss_balance=0.6), yp, yt)
    loss.backward()
    assert yp.grad is not None and torch.isfinite(yp.grad).all()


def test_end_to_end_neural_fit_uses_blend_path():
    """Integration: a real backend trained with loss_balance set must run
    the blend path (populating _loss_ema) and produce finite predictions —
    confirms the attribute plumbing + training loop work together."""
    from ml_forecast_lab.models.nlinear_backend import NLinearModel

    rng = np.random.default_rng(0)
    n, H = 240, 6
    X_seq = rng.standard_normal((n, 12, 2)).astype(np.float32)
    y = rng.standard_normal((n, H)).astype(np.float32)
    X_flat = np.zeros((n, 1), dtype=np.float32)

    model = NLinearModel(epochs=2, batch_size=64, output_activation="linear")
    # Mirror what _apply_experiment_neural_params does before fit().
    model.loss_balance = 1.0  # pure cumulative
    model._loss_ema = None
    model.fit(X_flat, y, sequence_data=X_seq, past_window_size=12)

    assert model._loss_ema is not None, "blend path should have seeded the EMA"
    assert "interval" in model._loss_ema and "daily" in model._loss_ema
    pred = model.predict_sequence(X_seq[:5])
    assert pred.shape == (5, H)
    assert np.all(np.isfinite(pred))
