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


class TestDeprecatedYamlKeysStripped:
    """v2.41.0: daily_loss_weight / loss_balance are no longer fields on
    ExperimentCfg. Old YAMLs carrying them load fine — load_config strips
    the keys (with a log line) and rewrites the file, same migration path
    as horizons_minutes / database / output_units."""

    def test_old_yaml_with_dead_knobs_loads_and_migrates(self, tmp_path):
        import yaml
        from ml_forecast_lab.config import load_config

        cfg_path = tmp_path / "mlfl.yaml"
        cfg_path.write_text(yaml.dump({
            "timezone": "UTC",
            "experiments": [{
                "name": "legacy",
                "target_entity": "sensor.t",
                "daily_loss_weight": 0.5,
                "loss_balance": 0.3,
            }],
        }))
        cfg = load_config(cfg_path)
        assert len(cfg.experiments) == 1
        exp = cfg.experiments[0]
        assert not hasattr(exp, "daily_loss_weight")
        assert not hasattr(exp, "loss_balance")
        # the YAML itself was rewritten without the dead keys
        raw = yaml.safe_load(cfg_path.read_text())
        assert "daily_loss_weight" not in raw["experiments"][0]
        assert "loss_balance" not in raw["experiments"][0]

    def test_settings_api_rejects_dead_knobs(self):
        """The /api/experiment-settings validator map must no longer
        accept the removed fields (they used to validate + persist while
        affecting nothing — audit F11's silent-misconfiguration shape)."""
        import inspect
        from ml_forecast_lab.web import app as web_app
        src = inspect.getsource(web_app)
        assert '"daily_loss_weight": lambda' not in src
        assert '"loss_balance": lambda' not in src
