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


def test_blend_alpha_zero_is_raw_interval_loss():
    """α=0 (the default) short-circuits to the raw interval loss — byte-for-
    byte identical to the legacy interval-only path, with no EMA rescale —
    so making every neural experiment default to α=0 changes nothing."""
    obj = _Dummy(loss_balance=0.0)
    yp, yt = _mk()
    loss, _ = _loss(obj, yp, yt)
    assert torch.allclose(loss, torch.nn.functional.mse_loss(yp, yt), atol=1e-6)
    assert obj._loss_ema is None, "α=0 must not touch the EMA"


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75, 1.0])
def test_blend_first_call_normalises_both_terms_to_one(alpha):
    """For α>0, the first blended call seeds the EMA to each term's own
    value, so both normalised terms equal 1.0 and the blend is
    (1-α)·1 + α·1 = 1.0 regardless of α. Confirms the magnitude
    normalisation makes α a balance of *influence*, not raw scale."""
    yp, yt = _mk()
    loss, _ = _loss(_Dummy(loss_balance=alpha), yp, yt)
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-5)


def test_blend_alpha_extremes_track_their_own_term():
    """α=0 returns the raw interval loss (no EMA); α=1, once the EMA has
    moved, equals daily/ema_d — the far ends are pure interval / pure
    cumulative."""
    obj0 = _Dummy(loss_balance=0.0)
    obj1 = _Dummy(loss_balance=1.0)
    yp_seed, yt_seed = _mk(seed=1)
    _loss(obj0, yp_seed, yt_seed)
    _loss(obj1, yp_seed, yt_seed)  # seeds obj1's EMA

    yp, yt = _mk(seed=2)
    loss0, _ = _loss(obj0, yp, yt)
    loss1, _ = _loss(obj1, yp, yt)

    interval = torch.nn.functional.mse_loss(yp, yt)
    H = yp.size(1)
    daily = (((yp.cumsum(1) - yt.cumsum(1)) ** 2).mean(dim=1) / float(H)).mean()
    assert torch.allclose(loss0, interval, atol=1e-6)  # raw interval at α=0
    assert obj0._loss_ema is None
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


def test_apply_loss_balance_wires_neural_model():
    """v2.40.2 regression: the helper every training path uses must set
    loss_balance (and reset the EMA) on neural models — the bug was that
    the benchmark / retrain paths set daily_loss_weight by hand and never
    set loss_balance, so the slider was a no-op on the models that produce
    the user's results."""
    from ml_forecast_lab.main import _apply_loss_balance
    from ml_forecast_lab.config import ExperimentCfg

    class _Neural:
        is_neural = True
        loss_balance = None
        _loss_ema = {"interval": 1.0, "daily": 1.0}

    cfg = ExperimentCfg(name="e", target_entity="sensor.x", loss_balance=0.8)
    m = _Neural()
    _apply_loss_balance(m, cfg)
    assert m.loss_balance == pytest.approx(0.8)
    assert m._loss_ema is None  # reset so the run normalises afresh


def test_apply_loss_balance_noop_for_tree_model():
    from ml_forecast_lab.main import _apply_loss_balance
    from ml_forecast_lab.config import ExperimentCfg

    class _Tree:
        is_neural = False
        loss_balance = "untouched"

    cfg = ExperimentCfg(name="e", target_entity="sensor.x", loss_balance=0.8)
    t = _Tree()
    _apply_loss_balance(t, cfg)
    assert t.loss_balance == "untouched"


def test_apply_loss_balance_respects_overrides():
    """A swept / pinned loss_balance in overrides must not be clobbered."""
    from ml_forecast_lab.main import _apply_loss_balance
    from ml_forecast_lab.config import ExperimentCfg

    class _Neural:
        is_neural = True
        loss_balance = 0.3

    cfg = ExperimentCfg(name="e", target_entity="sensor.x", loss_balance=0.8)
    m = _Neural()
    _apply_loss_balance(m, cfg, overrides={"loss_balance": 0.3})
    assert m.loss_balance == pytest.approx(0.3)


def test_effective_loss_balance_resolution():
    """The slider's displayed/used α (config.effective_loss_balance) is the
    single source of truth: explicit value wins; else migrate from the
    effective daily_loss_weight (incl. the PF9 non-negative auto-default);
    else per-interval (0.0)."""
    from ml_forecast_lab.config import ExperimentCfg

    def cfg(**kw):
        return ExperimentCfg(name="e", target_entity="sensor.x", **kw)

    # Explicit slider value wins outright.
    assert cfg(loss_balance=0.7).effective_loss_balance == pytest.approx(0.7)
    # Signed target, nothing set → per-interval default.
    assert cfg().effective_loss_balance == 0.0
    # Explicit additive weight migrates λ→α=λ/(1+λ): 1.0 → 0.5.
    assert cfg(daily_loss_weight=1.0).effective_loss_balance == pytest.approx(0.5)
    # Non-negative target with no weight keeps PF9's λ=0.5 → α=1/3.
    assert cfg(target_is_nonnegative=True).effective_loss_balance == pytest.approx(0.5 / 1.5)
    # Cumulative source likewise inherits the PF9 default.
    assert cfg(source_is_cumulative=True).effective_loss_balance == pytest.approx(0.5 / 1.5)
    # Explicit loss_balance overrides any daily_loss_weight.
    assert cfg(daily_loss_weight=1.0, loss_balance=0.0).effective_loss_balance == 0.0


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
