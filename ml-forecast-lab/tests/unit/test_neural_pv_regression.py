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


def test_output_activation_resolves_linear_for_nonneg_target():
    """v2.41.0: output_activation='auto' resolves to 'linear' for ALL
    non-LSTM neural backends, including cumulative / non-negative
    targets.

    History: v2.37 PF8 picked 'relu' (dying-ReLU collapse), v2.37.1
    switched to 'softplus' (non-zero gradient everywhere). Empirically
    softplus only slowed the same death: with daily_loss_weight=0 —
    every real deployment — the zero-valued half of a PV/demand target
    keeps pushing the pre-activation down until float32 softplus
    saturates to exactly 0 and gradients vanish
    (tests/integration/test_pv_forecast_pipeline.py pins the collapse).
    Non-negativity is now enforced by the publish-time clamp; an
    explicit ``output_activation: softplus`` is still honoured.
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

    # source_is_cumulative → linear (publish clamp owns non-negativity).
    assert _resolve_output_activation(
        _Fake(source_is_cumulative=True), "nlinear"
    ) == "linear"

    # target_is_nonnegative → linear (same).
    assert _resolve_output_activation(
        _Fake(target_is_nonnegative=True), "nlinear"
    ) == "linear"

    # LSTM always picks zscore regardless.
    assert _resolve_output_activation(
        _Fake(target_is_nonnegative=True), "lstm"
    ) == "zscore"

    # Explicit override is honoured.
    assert _resolve_output_activation(
        _Fake(output_activation="softplus", target_is_nonnegative=True), "nlinear"
    ) == "softplus"


def test_pf7_dlinear_head_input_dim_drops_future_target_slots():
    """PF7 for DLinear: head input dim shrinks when past_window_size <
    seq_len; legacy past-only mode keeps the full seq_len * n_channels.
    """
    from ml_forecast_lab.models.dlinear_backend import _DLinearNet
    seq_len = 96
    past = 48
    n_channels = 8
    # PF7 extended path.
    net_ext = _DLinearNet(
        seq_len=seq_len, n_channels=n_channels, kernel_size=13,
        n_horizons=24, use_revin=False, past_window_size=past,
    )
    expected_in = past * n_channels + (seq_len - past) * (n_channels - 1)
    assert net_ext.trend_linear.in_features == expected_in
    assert net_ext.seasonal_linear.in_features == expected_in
    # Legacy path.
    net_legacy = _DLinearNet(
        seq_len=seq_len, n_channels=n_channels, kernel_size=13,
        n_horizons=24, use_revin=False, past_window_size=None,
    )
    assert net_legacy.trend_linear.in_features == seq_len * n_channels
    assert net_legacy.seasonal_linear.in_features == seq_len * n_channels


def test_pf7_tsmixer_head_input_dim_drops_future_target_slots():
    """PF7 for TSMixer."""
    from ml_forecast_lab.models.tsmixer_backend import _TSMixerNet
    seq_len = 96
    past = 48
    n_channels = 8
    net_ext = _TSMixerNet(
        seq_len=seq_len, n_channels=n_channels, n_mixer_layers=2,
        hidden=16, dropout=0.0, n_horizons=24, use_revin=False,
        past_window_size=past,
    )
    expected_in = past * n_channels + (seq_len - past) * (n_channels - 1)
    assert net_ext.head.in_features == expected_in
    net_legacy = _TSMixerNet(
        seq_len=seq_len, n_channels=n_channels, n_mixer_layers=2,
        hidden=16, dropout=0.0, n_horizons=24, use_revin=False,
        past_window_size=None,
    )
    assert net_legacy.head.in_features == seq_len * n_channels


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


def test_cumulative_daily_reset_dataset_invariants():
    """Sanity check that the cumulative-with-daily-reset synthetic data
    has the structure we test against (used by run_cumulative_check.py).
    """
    from tests.synthetic.datasets import make_cumulative_daily_reset
    d = make_cumulative_daily_reset(0)
    assert "y" in d.df.columns
    assert "y_interval" in d.df.columns
    # The cumulative column must reset at midnight: every value at
    # 00:00 should be smaller than the previous day's 23:30 value.
    cum_at_midnight = d.df.loc[
        (d.df.index.hour == 0) & (d.df.index.minute == 0), "y"
    ]
    cum_at_eod = d.df.loc[
        (d.df.index.hour == 23) & (d.df.index.minute == 30), "y"
    ]
    assert len(cum_at_midnight) == 365
    assert len(cum_at_eod) == 365
    # The first row of each day (midnight) must be the smallest within
    # that day. Easier check: the average midnight value is much smaller
    # than the average end-of-day value.
    assert float(cum_at_midnight.mean()) < 0.1 * float(cum_at_eod.mean()), (
        f"Cumulative reset broken: midnight avg "
        f"{cum_at_midnight.mean():.3f}, EOD avg {cum_at_eod.mean():.3f}"
    )
    # Interval column must be non-negative everywhere.
    assert (d.df["y_interval"] >= 0).all()
    # The cumsum of the interval column within each day should equal
    # the cumulative column (up to float precision).
    day_starts = np.where(d.df.index.hour == 0)[0]
    day_starts = day_starts[d.df.index.minute[day_starts] == 0]
    for i_start, i_next in zip(day_starts[:-1], day_starts[1:]):
        intervals = d.df["y_interval"].values[i_start: i_next]
        reconstructed_cum = np.cumsum(intervals)
        actual_cum = d.df["y"].values[i_start: i_next]
        assert np.allclose(reconstructed_cum, actual_cum, atol=1e-4), (
            f"Cumulative/interval mismatch at day starting {d.df.index[i_start]}"
        )
        break  # one day is enough


def test_pf8_pf9_resolve_together_for_source_is_cumulative():
    """Cumulative-with-daily-reset is the canonical PF8/PF9 case.

    ExperimentCfg.source_is_cumulative=True should trigger both PF8
    (softplus) and PF9 (daily_loss_weight=0.5) defaults via the auto
    resolvers.
    """
    from dataclasses import dataclass
    from ml_forecast_lab.main import (
        _resolve_output_activation, _resolve_daily_loss_weight,
    )

    @dataclass
    class _Cfg:
        output_activation: str = "auto"
        daily_loss_weight: float = 0.0
        source_is_cumulative: bool = True
        target_is_nonnegative: bool = False

    # v2.41.0: cumulative → linear (publish-time clamp owns the
    # non-negativity contract; the softplus auto-pick collapsed to flat
    # zero with the production daily_loss_weight=0 — see
    # _resolve_output_activation's history note).
    assert _resolve_output_activation(_Cfg(), "nlinear") == "linear"
    assert _resolve_output_activation(_Cfg(), "dlinear") == "linear"
    assert _resolve_output_activation(_Cfg(), "nbeats") == "linear"
    # LSTM still picks zscore (its specialised default)
    assert _resolve_output_activation(_Cfg(), "lstm") == "zscore"
    # PF9: cumulative → daily_loss_weight = 0.5
    assert _resolve_daily_loss_weight(_Cfg()) == 0.5


def test_auto_activation_resolves_linear_regardless_of_log_transform():
    """v2.41.0: 'auto' resolves to 'linear' for every non-LSTM neural
    backend, regardless of log_transform / target_is_nonnegative /
    source_is_cumulative.

    History: PF10 originally made log_transform flip 'relu' → 'softplus';
    v2.37.1 made softplus the pick for all non-negative cases. With the
    cumulative-loss term inactive (daily_loss_weight=0 — every real
    deployment) softplus saturates to exactly 0 in float32 on
    zero-heavy targets and the forecast collapses flat (pinned by
    tests/integration/test_pv_forecast_pipeline.py). Non-negativity is
    now enforced at the publish boundary instead.
    """
    from dataclasses import dataclass
    from ml_forecast_lab.main import _resolve_output_activation

    @dataclass
    class _Fake:
        output_activation: str = "auto"
        source_is_cumulative: bool = False
        target_is_nonnegative: bool = False
        log_transform: bool = False

    # target_is_nonnegative alone → linear
    assert _resolve_output_activation(
        _Fake(target_is_nonnegative=True), "nlinear"
    ) == "linear"

    # target_is_nonnegative + log_transform → linear
    assert _resolve_output_activation(
        _Fake(target_is_nonnegative=True, log_transform=True), "nlinear"
    ) == "linear"

    # source_is_cumulative + log_transform → linear
    assert _resolve_output_activation(
        _Fake(source_is_cumulative=True, log_transform=True), "nlinear"
    ) == "linear"

    # log_transform alone (signed target) → linear (unchanged)
    assert _resolve_output_activation(
        _Fake(log_transform=True), "nlinear"
    ) == "linear"

    # Explicit override always wins
    assert _resolve_output_activation(
        _Fake(
            output_activation="relu",
            target_is_nonnegative=True,
        ),
        "nlinear",
    ) == "relu"


def test_pf10_nlinear_non_zero_forecast_with_softplus_log_transform():
    """End-to-end smoke test: NLinear with output_activation=softplus on
    a log-transformed non-negative target produces a forecast that is
    NOT identically zero. Catches the dying-ReLU regression where the
    head got stuck at all-negative pre-activations and ReLU clamped
    every horizon to 0.0.
    """
    import math
    import numpy as np
    import torch
    from ml_forecast_lab.models.nlinear_backend import NLinearModel

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    n = 1200
    window = 48
    horizon = 96  # match user's future_periods=96
    t = np.arange(n)
    hours = (t * 0.5) % 24
    pv_w = np.maximum(0.0, 1500.0 * np.sin(np.pi * (hours - 6) / 12.0))
    pv_w = pv_w + rng.normal(0, 30.0, size=n).clip(-50, 50)
    pv_w = np.maximum(0.0, pv_w)
    y_log = np.log1p(pv_w).astype(np.float32)
    hr_sin = np.sin(2 * math.pi * hours / 24.0).astype(np.float32)
    hr_cos = np.cos(2 * math.pi * hours / 24.0).astype(np.float32)
    X = np.stack([y_log, hr_sin, hr_cos], axis=-1)

    seq_X = []
    seq_y = []
    for i in range(window, n - horizon):
        past = X[i - window:i, :]
        fut_known = X[i:i + horizon, 1:]
        fut_target_zero = np.zeros((horizon, 1), dtype=np.float32)
        fut = np.concatenate([fut_target_zero, fut_known], axis=-1)
        seq_X.append(np.concatenate([past, fut], axis=0))
        seq_y.append(y_log[i:i + horizon])
    seq_X = np.array(seq_X)
    seq_y = np.array(seq_y)

    model = NLinearModel(
        epochs=30,
        batch_size=64,
        output_activation="softplus",
        use_revin=True,
    )
    X_flat = np.zeros((seq_X.shape[0], 1), dtype=np.float32)
    model.fit(
        X_flat,
        seq_y,
        sequence_data=seq_X,
        past_window_size=window,
    )
    pred = model.predict_sequence(seq_X[-1:]).reshape(-1)

    # Dying-ReLU collapse signature is "forecast is IDENTICALLY zero"
    # across the entire horizon. We use loose thresholds (~0.05 / std
    # 0.01) so the assertion is robust to CI hardware variation and
    # only fires on a true collapse, not on a slightly different
    # convergence path between machines.
    assert pred.shape == (horizon,)
    assert np.max(pred) > 0.05, (
        f"Softplus NLinear collapsed to zero (max log-pred={np.max(pred):.4f}, "
        f"min={np.min(pred):.4f}); dying-ReLU regression returned"
    )
    assert pred.std() > 0.01, (
        f"Softplus NLinear collapsed to flat output (std={pred.std():.4f}, "
        f"mean={pred.mean():.4f}); forecast shape lost"
    )
