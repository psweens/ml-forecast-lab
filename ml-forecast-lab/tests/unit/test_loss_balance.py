"""v2.40.14: the cumulative-loss path and its slider were removed.

This file used to pin the interval+cumulative blend (legacy additive
`L = L_interval + λ·L_daily` AND the EMA-normalised convex blend
`L = (1-α)·L_interval/ema_i + α·L_daily/ema_d`). Both paths are gone.

The empirical case for removal is recorded in
`scripts/LOSS_COMPARISON_FINDINGS.md`:

  - α-cliff measured on BOTH sparse-demand and smooth-cumulative
    profiles (any α > 0 → 50–95 % daily-MAE degradation, flat across
    α ∈ [0.1, 1.0]).
  - Faster EMA decay (β = 0.9 vs production 0.99) softened by ~25 %
    but did not remove the cliff → mechanism in the loss, not the
    normaliser.
  - Gradient analysis of ``_cumulative_trajectory_loss``: early
    horizon steps have H × more cumsum-error terms summed into their
    gradient than late ones → systematic under-prediction at early
    horizon → exactly the failure mode the term was supposed to fix.

These tests now pin the POST-removal behaviour: everything is a
no-op, ``_composite_horizon_loss`` returns interval-only, and the
``daily_weight`` argument is ignored.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch required",
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _make_model():
    """Return a concrete neural backend instance.

    NLinear is the lightest neural backend that owns the composite-loss
    method via ForecastModel; instantiating one gives us a real ``self``
    for the bound method without monkey-patching abstract classes.
    """
    from ml_forecast_lab.models.nlinear_backend import NLinearModel
    return NLinearModel(epochs=1, patience=1)


def _pair(shape=(8, 24)):
    torch.manual_seed(0)
    return torch.randn(*shape, requires_grad=True), torch.randn(*shape)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

class TestCompositeHorizonLossPostRemoval:
    """``_composite_horizon_loss`` is now interval-only — the
    ``daily_weight`` arg is silently ignored and ``self.loss_balance``
    / ``self._loss_ema`` are never consulted."""

    def test_returns_pure_interval_mse(self):
        model = _make_model()
        yp, yt = _pair()
        crit = nn.MSELoss(reduction='none')
        loss, per_sample = model._composite_horizon_loss(
            yp, yt, crit, None, daily_weight=0.0,
        )
        assert torch.isclose(loss, ((yp - yt) ** 2).mean())
        assert per_sample.shape == yp.shape

    def test_daily_weight_is_ignored(self):
        model = _make_model()
        yp, yt = _pair()
        crit = nn.MSELoss(reduction='none')
        l0, _ = model._composite_horizon_loss(yp, yt, crit, None, 0.0)
        l5, _ = model._composite_horizon_loss(yp, yt, crit, None, 0.5)
        l_huge, _ = model._composite_horizon_loss(yp, yt, crit, None, 1e6)
        assert torch.isclose(l0, l5)
        assert torch.isclose(l0, l_huge)

    def test_loss_balance_attribute_is_ignored(self):
        """Setting ``self.loss_balance`` to anything must not change
        the loss — the convex-blend path was the slider's
        implementation, and it's gone."""
        model = _make_model()
        yp, yt = _pair()
        crit = nn.MSELoss(reduction='none')
        l_none, _ = model._composite_horizon_loss(yp, yt, crit, None, 0.0)
        model.loss_balance = 0.7
        l_07, _ = model._composite_horizon_loss(yp, yt, crit, None, 0.0)
        model.loss_balance = 1.0
        l_10, _ = model._composite_horizon_loss(yp, yt, crit, None, 0.0)
        assert torch.isclose(l_none, l_07)
        assert torch.isclose(l_none, l_10)

    def test_loss_remains_differentiable(self):
        model = _make_model()
        yp, yt = _pair()
        crit = nn.MSELoss(reduction='none')
        loss, _ = model._composite_horizon_loss(yp, yt, crit, None, 0.0)
        loss.backward()
        assert yp.grad is not None and torch.isfinite(yp.grad).all().item()


class TestEffectiveLossBalanceProperty:
    """``ExperimentCfg.effective_loss_balance`` is retained as a
    property so any caller / template still referencing it gets ``0.0``
    rather than an AttributeError."""

    def _cfg(self, **overrides):
        from ml_forecast_lab.config import ExperimentCfg
        defaults = dict(
            name="x", target_entity="sensor.x", mode="lab",
            interval_minutes=30, days_history=14, max_age=30,
            future_periods=48, source_is_cumulative=False,
            metrics=["mae"], production_metric="mae",
        )
        defaults.update(overrides)
        return ExperimentCfg(**defaults)

    def test_unset_returns_zero(self):
        cfg = self._cfg()
        assert cfg.effective_loss_balance == 0.0

    def test_loss_balance_set_returns_zero(self):
        cfg = self._cfg(loss_balance=0.7)
        assert cfg.effective_loss_balance == 0.0

    def test_legacy_daily_loss_weight_set_returns_zero(self):
        cfg = self._cfg(daily_loss_weight=2.0)
        assert cfg.effective_loss_balance == 0.0

    def test_cumulative_source_no_longer_auto_engages(self):
        """Pre-v2.40.14 the auto-default for cumulative / non-negative
        targets was λ = 0.5 → α ≈ 0.33. Now it's just 0.0 — the slider
        was the trigger and the slider is gone."""
        cfg = self._cfg(source_is_cumulative=True)
        assert cfg.effective_loss_balance == 0.0


class TestApplyLossBalanceStub:
    """``_apply_loss_balance`` is retained as a defensive no-op stub —
    pins ``model.loss_balance = 0`` and clears any stale EMA, but does
    nothing else. The 5 call sites in main.py still call it for
    backwards-compat with old checkpoints."""

    def test_pins_loss_balance_zero_on_neural(self):
        from ml_forecast_lab.main import _apply_loss_balance
        from types import SimpleNamespace
        model = _make_model()
        model.loss_balance = 0.8       # simulate a stale value
        model._loss_ema = {"interval": 1.0, "daily": 1.0}
        cfg = SimpleNamespace(
            source_is_cumulative=False, target_is_nonnegative=False,
            daily_loss_weight=0.0, loss_balance=None,
            effective_loss_balance=0.0,
        )
        _apply_loss_balance(model, cfg)
        assert model.loss_balance == 0.0
        assert model._loss_ema is None

    def test_noop_on_tree_model(self):
        from ml_forecast_lab.main import _apply_loss_balance
        from types import SimpleNamespace
        tree = SimpleNamespace(is_neural=False)
        cfg = SimpleNamespace(effective_loss_balance=0.0)
        _apply_loss_balance(tree, cfg)   # must not raise
