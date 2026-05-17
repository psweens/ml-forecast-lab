"""
Regression tests pinning the two confirmed neural-PV root causes from
docs/investigations/2026-05-neural-pv.md.

After v2.37 these tests cover the FIXED behaviour:

* PF1 — ``_RevIN.normalize(x, past_window_size=W)`` produces a per-window
  mean within 5% of the past-block mean for the target channel,
  regardless of how many future positions follow.
* PF2 — A freshly-trained NLinear with an extended-window input
  outputs a value that sits close to the last past observation (it
  uses that as its anchor and the linear residual is small under a
  near-deterministic input).

The third test demonstrates the PF1 prototype is equivalent to the
production fix.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ml_forecast_lab.features import (
    compute_known_future_features, create_sliding_windows,
)
from ml_forecast_lab.models.base import _RevIN
from ml_forecast_lab.models.nlinear_backend import _NLinearNet, NLinearModel

from tests.synthetic.datasets import make_realistic_pv, GB_LAT, GB_LON


WINDOW = 48
HORIZON = 48


@pytest.fixture(scope="module")
def extended_window_tensor():
    """A real extended window tensor from realistic_pv."""
    d = make_realistic_pv(0)
    horizon_steps = list(range(1, HORIZON + 1))
    future_df = compute_known_future_features(
        d.df.index, add_temporal=True, country="GB",
        solar_lat_lon=(GB_LAT, GB_LON),
        include_sun_elevation=True, include_clear_sky_ghi=True,
    )
    X, y, ch = create_sliding_windows(
        d.df, "y", window_size=WINDOW,
        covariate_cols=["sun_elevation", "clear_sky_ghi"],
        add_temporal=True,
        horizon_steps=horizon_steps,
        future_features_df=future_df,
    )
    return X, y, ch


def test_revin_past_only_extended_window_mean_unbiased(extended_window_tensor):
    """PF1: RevIN's per-window mean must match the past mean when
    past_window_size is passed.

    Pre-v2.37 behaviour (no past_window_size kwarg) is also covered
    via a sub-assertion: it still computes whole-window stats, so old
    saved checkpoints with no past_window_size context behave as
    before.
    """
    X, _, ch = extended_window_tensor
    target_ch = 0
    past_means = X[:, :WINDOW, target_ch].mean(axis=1)
    # Pick a sample with non-trivial past mean (otherwise the bias is
    # invisible by construction).
    candidates = np.where(past_means > float(np.quantile(past_means, 0.9)))[0]
    assert candidates.size > 0
    i = int(candidates[0])
    sample = torch.from_numpy(X[i: i + 1])
    past_mean = float(X[i, :WINDOW, target_ch].mean())

    # PF1 path — past_window_size provided.
    revin = _RevIN(X.shape[2], target_channel=target_ch, affine=False)
    revin.normalize(sample, past_window_size=WINDOW)
    revin_mean_pf1 = float(revin._mean[0, 0, target_ch].item())
    bias_pf1 = abs(revin_mean_pf1 - past_mean) / max(abs(past_mean), 1e-6)
    assert bias_pf1 < 0.05, (
        f"PF1 broken: with past_window_size={WINDOW}, RevIN mean "
        f"{revin_mean_pf1:.2f} vs past mean {past_mean:.2f}; "
        f"relative bias {bias_pf1:.3f} > 5%"
    )

    # Legacy path — no past_window_size: still computes whole-window
    # stats (preserves behaviour for old checkpoints).
    revin_legacy = _RevIN(X.shape[2], target_channel=target_ch, affine=False)
    revin_legacy.normalize(sample)  # no past_window_size
    revin_mean_legacy = float(revin_legacy._mean[0, 0, target_ch].item())
    bias_legacy = abs(revin_mean_legacy - past_mean) / max(abs(past_mean), 1e-6)
    assert bias_legacy > 0.3, (
        f"Legacy path no longer biased — did the past_window_size=None "
        f"branch change? Expected ~50% bias as before. Got {bias_legacy:.3f}"
    )


def test_nlinear_anchor_carries_last_past_observation(extended_window_tensor):
    """PF2: NLinear's anchor is the last PAST target observation when
    past_window_size is set.

    Verified by comparing NLinear forward output behaviour: with a
    near-zero (untrained) linear head, the model's output should
    approximately equal the anchor value. We use a manually
    constructed network with a near-zero weight matrix to isolate the
    anchor logic.
    """
    X, _, ch = extended_window_tensor
    target_ch = 0
    seq_len = X.shape[1]
    n_channels = X.shape[2]

    # Pick a sample whose last past row has non-trivial target value.
    last_past = X[:, WINDOW - 1, target_ch]
    candidates = np.where(last_past > 100.0)[0]
    assert candidates.size > 0
    i = int(candidates[0])

    # PF2 path — past_window_size set.
    net = _NLinearNet(
        seq_len=seq_len, n_channels=n_channels, n_horizons=HORIZON,
        output_activation="linear", sigmoid_scale=1.0,
        use_revin=False, target_channel=target_ch,
        past_window_size=WINDOW,
    )
    # Zero out the linear head so output reduces to (0 + anchor).
    with torch.no_grad():
        net.linear.weight.zero_()
        net.linear.bias.zero_()
    sample = torch.from_numpy(X[i: i + 1])
    net.eval()
    with torch.no_grad():
        out = net(sample).cpu().numpy().ravel()
    expected = float(X[i, WINDOW - 1, target_ch])
    # All horizon outputs should equal the past-end anchor (within numerical noise).
    assert np.allclose(out, expected, atol=1.0), (
        f"PF2 broken: with past_window_size={WINDOW}, NLinear output "
        f"{out[:3]}... should equal the last past observation {expected:.2f}, "
        f"but the broadcasted anchor differs."
    )

    # Legacy path — past_window_size=None: anchor reverts to literal
    # last row which is a future-position zero, so output should be ~0.
    net_legacy = _NLinearNet(
        seq_len=seq_len, n_channels=n_channels, n_horizons=HORIZON,
        output_activation="linear", sigmoid_scale=1.0,
        use_revin=False, target_channel=target_ch,
    )
    with torch.no_grad():
        net_legacy.linear.weight.zero_()
        net_legacy.linear.bias.zero_()
    net_legacy.eval()
    with torch.no_grad():
        out_legacy = net_legacy(sample).cpu().numpy().ravel()
    assert np.allclose(out_legacy, 0.0, atol=1.0), (
        f"Legacy NLinear no longer anchors at the literal last row — "
        f"backward compatibility broken? Expected ~0, got {out_legacy[:3]}"
    )


def test_nlinear_pf7_head_input_dim_drops_future_target_slots():
    """PF7: NLinear's head input dimension excludes the future-position
    target-channel slots when past_window_size < seq_len.

    Past-only path (past_window_size = None or == seq_len) keeps the
    original flat ``seq_len * n_channels`` shape so saved checkpoints
    from before v2.37 load and run identically.
    """
    seq_len = 96
    past = 48
    n_channels = 8
    # PF7 extended-window net.
    net_ext = _NLinearNet(
        seq_len=seq_len, n_channels=n_channels, n_horizons=24,
        output_activation="linear", use_revin=False,
        past_window_size=past,
    )
    # Past block contributes past * n_channels; future block contributes
    # (seq_len - past) * (n_channels - 1) — target channel slots are
    # omitted (PF7).
    expected_in = past * n_channels + (seq_len - past) * (n_channels - 1)
    assert net_ext.linear.in_features == expected_in, (
        f"PF7 head input dim mismatch: expected {expected_in}, got "
        f"{net_ext.linear.in_features}"
    )

    # Legacy path: unchanged.
    net_legacy = _NLinearNet(
        seq_len=seq_len, n_channels=n_channels, n_horizons=24,
        output_activation="linear", use_revin=False,
        past_window_size=None,
    )
    assert net_legacy.linear.in_features == seq_len * n_channels, (
        f"Legacy NLinear head input dim changed — backward compatibility "
        f"broken. Expected {seq_len * n_channels}, got "
        f"{net_legacy.linear.in_features}"
    )


def test_nlinear_end_to_end_fit_predict_with_past_window_size(
    extended_window_tensor,
):
    """Sanity check that a freshly-trained NLinear with PF1+PF2+PF7
    end-to-end produces finite predictions of the right shape on an
    extended-window input.
    """
    X, y, _ = extended_window_tensor
    # Use a tiny subset so the test is fast.
    X_small = X[:200]
    y_small = y[:200]
    model = NLinearModel(epochs=3, batch_size=64, use_revin=True,
                         output_activation="linear")
    X_flat = np.zeros((X_small.shape[0], 1), dtype=np.float32)
    model.fit(X_flat, y_small, sequence_data=X_small, past_window_size=WINDOW)
    pred = model.predict_sequence(X_small[:5])
    assert pred.shape == (5, HORIZON)
    assert np.all(np.isfinite(pred))


def test_pf4_nbeats_past_only_backcast(extended_window_tensor):
    """PF4: N-BEATS only feeds the past slice to its backcast stack
    when past_window_size is set; the future block (which has
    zero-target placeholders) is ignored.

    Asserts that the trained module's internal ``past_window_size``
    attribute matches and that the flat_input_size matches past * C
    rather than seq_len * C.
    """
    from ml_forecast_lab.models.nbeats_backend import _NBeatsNet
    X, _, _ = extended_window_tensor
    seq_len, n_channels = X.shape[1], X.shape[2]
    net = _NBeatsNet(
        seq_len=seq_len, n_channels=n_channels, hidden_size=16,
        n_stacks=1, blocks_per_stack=1, n_fc_layers=2,
        n_horizons=HORIZON, output_activation="linear",
        past_window_size=WINDOW,
    )
    assert net.flat_input_size == WINDOW * n_channels, (
        f"PF4 broken: N-BEATS flat_input_size {net.flat_input_size} != "
        f"past*C = {WINDOW * n_channels}"
    )
    # Forward should still work on a (1, seq_len, n_channels) input
    # because forward slices the past block off internally.
    sample = torch.from_numpy(X[:1])
    net.eval()
    with torch.no_grad():
        out = net(sample)
    assert out.shape == (1, HORIZON)
    assert torch.isfinite(out).all()

    # Legacy path: when past_window_size is None, flat_input_size
    # equals seq_len * n_channels (whole-window).
    net_legacy = _NBeatsNet(
        seq_len=seq_len, n_channels=n_channels, hidden_size=16,
        n_stacks=1, blocks_per_stack=1, n_fc_layers=2,
        n_horizons=HORIZON, output_activation="linear",
    )
    assert net_legacy.flat_input_size == seq_len * n_channels


def test_pf4_nhits_past_only_input():
    """PF4: N-HiTS shortens its effective sequence length to
    past_window_size when set. Past-only path unchanged.
    """
    from ml_forecast_lab.models.nhits_backend import _NHiTSNet
    seq_len = 96
    past = 48
    n_channels = 8
    net = _NHiTSNet(
        seq_len=seq_len, n_channels=n_channels, hidden_size=16,
        n_stacks=2, blocks_per_stack=1, pool_kernels=[4, 2],
        n_fc_layers=2, n_horizons=24, output_activation="linear",
        past_window_size=past,
    )
    assert net.past_window_size == past
    # Each block's seq_len should match past, not seq_len.
    for block in net.blocks:
        assert block.seq_len == past, (
            f"PF4 broken: N-HiTS block seq_len {block.seq_len} != "
            f"past {past}"
        )
    # Forward returns shape (1, n_horizons).
    sample = torch.zeros(1, seq_len, n_channels)
    sample[:, :past, 0] = torch.linspace(0, 1, past)
    net.eval()
    with torch.no_grad():
        out = net(sample)
    assert out.shape == (1, 24)


def test_pf8_output_activation_resolves_softplus_for_nonneg_target():
    """PF8: output_activation='auto' resolves to 'softplus' when
    target_is_nonnegative is set, mirrors the existing
    source_is_cumulative behaviour for non-cumulative non-negative
    targets like PV power.
    """
    from dataclasses import dataclass
    from ml_forecast_lab.main import _resolve_output_activation

    @dataclass
    class _Fake:
        output_activation: str = "auto"
        source_is_cumulative: bool = False
        target_is_nonnegative: bool = False

    # Default (signed) → linear.
    assert _resolve_output_activation(_Fake(), "nlinear") == "linear"

    # source_is_cumulative → softplus (existing behaviour preserved).
    assert _resolve_output_activation(
        _Fake(source_is_cumulative=True), "nlinear"
    ) == "softplus"

    # target_is_nonnegative → softplus (the PF8 behaviour).
    assert _resolve_output_activation(
        _Fake(target_is_nonnegative=True), "nlinear"
    ) == "softplus"

    # LSTM always picks zscore regardless.
    assert _resolve_output_activation(
        _Fake(target_is_nonnegative=True), "lstm"
    ) == "zscore"

    # Explicit override is honoured.
    assert _resolve_output_activation(
        _Fake(output_activation="relu", target_is_nonnegative=True), "nlinear"
    ) == "relu"


def test_pf9_daily_loss_weight_resolves_to_half_on_nonneg_target():
    """PF9: daily_loss_weight defaults to 0.5 for non-negative neural
    targets when the user leaves it at 0.0. Explicit non-zero is
    honoured as-is. Signed targets stay at 0.0.
    """
    from dataclasses import dataclass
    from ml_forecast_lab.main import _resolve_daily_loss_weight

    @dataclass
    class _Fake:
        daily_loss_weight: float = 0.0
        source_is_cumulative: bool = False
        target_is_nonnegative: bool = False

    # Signed default — stays 0 (no implicit weight).
    assert _resolve_daily_loss_weight(_Fake()) == 0.0
    # source_is_cumulative → 0.5
    assert _resolve_daily_loss_weight(
        _Fake(source_is_cumulative=True)
    ) == 0.5
    # target_is_nonnegative → 0.5
    assert _resolve_daily_loss_weight(
        _Fake(target_is_nonnegative=True)
    ) == 0.5
    # Explicit user value wins.
    assert _resolve_daily_loss_weight(
        _Fake(daily_loss_weight=1.5, target_is_nonnegative=True)
    ) == 1.5
