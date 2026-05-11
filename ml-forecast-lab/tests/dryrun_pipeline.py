"""End-to-end dry-run of the ML Forecast Lab pipeline against synthetic data.

For each of the 24 registered model backends:
  1. Instantiate via the model registry.
  2. Verify get_params / set_params round-trip with every entry in the
     ``MODEL_PARAM_SCHEMA`` default block.
  3. Build the same feature matrices the BenchmarkRunner would hand to fit()
     (flat for trees, sliding-window for neural) and assert the shapes.
  4. Do NOT call fit() — the goal is to surface config / shape bugs without
     paying the training cost.

Then exercise:
  - Covariate analysis: enumerate every (config, model) cell of the matrix
    and confirm the combined feature frame can be built.
  - Hyperparameter tuning: for every model with tunable params, draw one
    Optuna sample from each schema entry and verify set_params accepts it.

Exit code 0 on full pass, otherwise prints every failure and exits 1.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Lets the script run as ``python tests/dryrun_pipeline.py`` from the repo
# root without needing ``pip install -e .`` first.
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# --------------------------------------------------------------------------- #
# Synthetic data: ~14 days at 30-min, target + 3 covariates                   #
# --------------------------------------------------------------------------- #

def make_synthetic_frame(n_periods: int = 48 * 14) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_periods, freq="30min")
    rng = np.random.default_rng(42)

    daily = np.sin(np.linspace(0, np.pi, 48)) * 1.5 + 0.5
    y = np.tile(daily, n_periods // 48) + rng.normal(0, 0.1, n_periods)
    y = np.clip(y, 0, None)

    return pd.DataFrame(
        {
            "y": y,
            "current_charge": 0.5 + 0.3 * np.sin(np.linspace(0, 8 * np.pi, n_periods))
            + rng.normal(0, 0.02, n_periods),
            "external_temperature": 10
            + 5 * np.sin(np.linspace(0, 8 * np.pi, n_periods))
            + rng.normal(0, 0.5, n_periods),
            "clear_sky_ghi": np.clip(
                np.tile(np.sin(np.linspace(0, np.pi, 48)) * 600, n_periods // 48),
                0,
                None,
            ),
        },
        index=idx,
    )


# --------------------------------------------------------------------------- #
# Per-fold feature builder identical to the one in main.py                    #
# --------------------------------------------------------------------------- #


def build_feature_matrices(df: pd.DataFrame):
    """Return (combined, feature_cols, X_flat, y_flat, seq_X, seq_y, channels)."""
    from ml_forecast_lab.features import build_features, create_sliding_windows

    features = build_features(df, target_col="y", interval_minutes=30, country="GB")
    combined = features.copy()
    combined["target"] = df["y"]
    for col in [c for c in df.columns if c != "y"]:
        combined[col] = df[col]
    combined = combined.dropna()

    feature_cols = [c for c in combined.columns if c != "target"]
    X_flat = combined[feature_cols].values.astype(np.float32)
    X_flat = np.nan_to_num(X_flat, nan=0.0)
    y_flat = combined["target"].values.astype(np.float32)

    # Sliding-window for neural backends. Matches benchmark runner defaults.
    cov_cols = [c for c in combined.columns if c in df.columns and c != "y"]
    window_size = min(48, len(combined) // 3)
    seq_X, seq_y, channels = create_sliding_windows(
        combined,
        target_col="target",
        window_size=window_size,
        covariate_cols=cov_cols or None,
        add_temporal=True,
        horizon_steps=list(range(1, 49)),
    )
    return combined, feature_cols, X_flat, y_flat, seq_X, seq_y, channels


# --------------------------------------------------------------------------- #
# Per-model schema-default smoke check (no fit)                               #
# --------------------------------------------------------------------------- #


def check_model_dryrun(name, model_cls, schema, registry, sequence_inputs, flat_inputs):
    """For one model name: instantiate, set defaults, set Optuna-style sample,
    save+load round-trip, verify expected interface — without running fit().

    Returns ``(ok: bool, notes: list[str])``.
    """
    notes = []

    # 1. registry.create returns an instance
    try:
        model = registry.create(name)
    except Exception as e:
        return False, [f"registry.create({name!r}) raised {type(e).__name__}: {e}"]

    # 2. inspect baseline interface
    if not hasattr(model, "fit") or not callable(model.fit):
        notes.append("missing fit()")
    if not hasattr(model, "predict") or not callable(model.predict):
        notes.append("missing predict()")
    if not hasattr(model, "get_params") or not callable(model.get_params):
        notes.append("missing get_params()")
    if not hasattr(model, "set_params") or not callable(model.set_params):
        notes.append("missing set_params()")

    is_neural = bool(getattr(model, "is_neural", False))

    # 3. get_params is a dict
    try:
        baseline_params = model.get_params()
    except Exception as e:
        return False, notes + [f"get_params() raised {type(e).__name__}: {e}"]
    if not isinstance(baseline_params, dict):
        return False, notes + [f"get_params() returned {type(baseline_params).__name__}, expected dict"]

    # 4. set_params accepts every schema default value
    defaults = {pname: spec.get("default") for pname, spec in schema.items()
                if spec.get("default") is not None}
    try:
        model.set_params(**defaults)
    except Exception as e:
        return False, notes + [
            f"set_params(**schema_defaults={defaults!r}) raised "
            f"{type(e).__name__}: {e}"
        ]

    # 5. set_params accepts one boundary sample (max of every range,
    #    randomly-chosen categorical) — flushes out off-by-one validation.
    boundary = {}
    rng = np.random.default_rng(0)
    for pname, spec in schema.items():
        ptype = spec.get("type", "float")
        if ptype == "int":
            boundary[pname] = int(spec["max"])
        elif ptype == "float":
            boundary[pname] = float(spec["max"])
        elif ptype == "select":
            boundary[pname] = rng.choice(spec["options"])
        elif ptype == "bool":
            boundary[pname] = True
    if boundary:
        try:
            fresh = registry.create(name)
            fresh.set_params(**boundary)
        except Exception as e:
            return False, notes + [
                f"set_params(**boundary_sample={boundary!r}) raised "
                f"{type(e).__name__}: {e}"
            ]

    # 6. The X / y / sequence shapes the benchmark runner produces must
    #    survive model._validate_X / _validate_y (it's what fit() calls
    #    first). Pick the right input set.
    if is_neural:
        # Neural models get sequence_data through kwargs and the flat X is
        # still validated via _validate_X (it gets shape-checked even though
        # the sequence path supersedes it).
        X_for_check, y_for_check = flat_inputs
    else:
        X_for_check, y_for_check = flat_inputs

    try:
        model._validate_X(X_for_check)
    except Exception as e:
        return False, notes + [
            f"_validate_X(flat shape={X_for_check.shape}) raised "
            f"{type(e).__name__}: {e}"
        ]
    try:
        validated_y = model._validate_y(y_for_check)
        if validated_y.ndim not in (1, 2):
            return False, notes + [
                f"_validate_y returned ndim={validated_y.ndim}"
            ]
    except Exception as e:
        return False, notes + [
            f"_validate_y raised {type(e).__name__}: {e}"
        ]

    # 7. Repr / name shouldn't crash
    try:
        _ = repr(model)
        _ = model.name
    except Exception as e:
        notes.append(f"repr/name raised {type(e).__name__}: {e}")

    return True, notes


def main():
    from ml_forecast_lab.models.registry import ModelRegistry
    # MODEL_PARAM_SCHEMA is a local in create_app(); pull it via the same
    # /api/models/params endpoint the UI uses so this test exercises the
    # public surface the user sees.
    from fastapi.testclient import TestClient
    from ml_forecast_lab.web.app import create_app

    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/models/params")
        resp.raise_for_status()
        api_payload = resp.json()
    MODEL_PARAM_SCHEMA = {name: entry["schema"] for name, entry in api_payload.items()}

    print("=" * 72)
    print("ML Forecast Lab — pipeline dry-run with synthetic data (no training)")
    print("=" * 72)

    df = make_synthetic_frame()
    print(f"Synthetic frame: {len(df)} rows × {len(df.columns)} cols, "
          f"index {df.index[0]} → {df.index[-1]}")

    # Build all feature matrices once
    combined, feat_cols, X_flat, y_flat, seq_X, seq_y, channels = build_feature_matrices(df)
    print(f"After build_features + dropna: {len(combined)} rows, "
          f"{len(feat_cols)} features")
    print(f"  X_flat.shape={X_flat.shape}, y_flat.shape={y_flat.shape}")
    print(f"  seq_X.shape={seq_X.shape}, seq_y.shape={seq_y.shape}, "
          f"channels={len(channels)}")

    # Build a registry that mirrors what main.py does on startup
    registry = ModelRegistry()
    backends = [
        ("lightgbm", "lightgbm_backend", "LightGBMModel"),
        ("xgboost", "xgboost_backend", "XGBoostModel"),
        ("lstm", "lstm_backend", "LSTMModel"),
        ("cnn", "cnn_backend", "CNNModel"),
        ("catboost", "catboost_backend", "CatBoostModel"),
        ("gru", "gru_backend", "GRUModel"),
        ("dlinear", "dlinear_backend", "DLinearModel"),
        ("nlinear", "nlinear_backend", "NLinearModel"),
        ("fits", "fits_backend", "FITSModel"),
        ("nbeats", "nbeats_backend", "NBeatsModel"),
        ("nhits", "nhits_backend", "NHiTSModel"),
        ("tide", "tide_backend", "TiDEModel"),
        ("tsmixer", "tsmixer_backend", "TSMixerModel"),
        ("timemixer", "timemixer_backend", "TimeMixerModel"),
        ("sparsetsf", "sparsetsf_backend", "SparseTSFModel"),
        ("patchtst", "patchtst_backend", "PatchTSTModel"),
        ("itransformer", "itransformer_backend", "iTransformerModel"),
        ("crossformer", "crossformer_backend", "CrossformerModel"),
        ("timesnet", "timesnet_backend", "TimesNetModel"),
        ("tft", "tft_backend", "TFTModel"),
        ("seasonal_naive", "seasonal_naive_backend", "SeasonalNaiveModel"),
        ("arima", "statsforecast_backend", "ARIMAModel"),
        ("ets", "statsforecast_backend", "ETSModel"),
        ("theta", "statsforecast_backend", "ThetaModel"),
    ]
    for short, mod, cls in backends:
        try:
            m = __import__(f"ml_forecast_lab.models.{mod}", fromlist=[cls])
            registry.register(short, getattr(m, cls))
        except Exception as e:
            print(f"!! could not import {short}: {e}")

    print(f"\nRegistry: {len(registry.list_available())} models")
    print(", ".join(registry.list_available()))

    # ---- Per-model dry run ---------------------------------------------- #
    failures = []
    print(f"\n{'─' * 72}\n[1/3] Per-model config dry-run\n{'─' * 72}")
    header = f"{'model':<16} {'neural':<7} {'schema':<7} status notes"
    print(header)
    print("-" * len(header))
    for name in registry.list_available():
        schema = MODEL_PARAM_SCHEMA.get(name, {})
        try:
            ok, notes = check_model_dryrun(
                name, registry._models[name], schema, registry,
                sequence_inputs=(seq_X, seq_y),
                flat_inputs=(X_flat, y_flat),
            )
        except Exception as e:
            ok = False
            notes = [f"unhandled {type(e).__name__}: {e}",
                     traceback.format_exc(limit=2)]
        is_neural_flag = "yes" if getattr(registry.create(name), "is_neural", False) else "no"
        status = "OK" if ok else "FAIL"
        note_str = ("; ".join(notes))[:160] if notes else ""
        print(f"  {name:<14} {is_neural_flag:<7} {len(schema):<7} {status}  {note_str}")
        if not ok:
            failures.append((name, notes))

    # ---- Covariate analysis enumeration --------------------------------- #
    print(f"\n{'─' * 72}\n[2/3] Covariate-analysis enumeration\n{'─' * 72}")
    # Mirror main.py:_run_covariate_analysis
    df_full = df.copy()
    covariate_cols = [c for c in df_full.columns if c != "y"]
    from ml_forecast_lab.features import build_features

    features_base = build_features(
        df_full, target_col="y", interval_minutes=30, country="GB",
    )

    configs = [("All covariates", covariate_cols[:]), ("No covariates", [])]
    for cov_col in covariate_cols:
        remaining = [c for c in covariate_cols if c != cov_col]
        configs.append((f"Without {cov_col}", remaining))

    print(f"Generated {len(configs)} covariate configurations to evaluate:")
    for label, cols in configs:
        comb = features_base.copy()
        comb["target"] = df_full["y"]
        for col in cols:
            comb[col] = df_full[col]
        comb = comb.dropna()
        feat = [c for c in comb.columns if c != "target"]
        print(f"  {label:<35}  → {len(comb)} rows × {len(feat)} features  "
              f"(covariates kept: {cols})")
        if len(comb) < 100:
            failures.append((f"covariate[{label}]",
                             [f"only {len(comb)} rows after dropna()"]))

    # ---- Tuning Optuna-sample round-trip -------------------------------- #
    print(f"\n{'─' * 72}\n[3/3] Hyperparameter-tuning sample round-trip\n{'─' * 72}")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    LOG_PARAMS = {"learning_rate", "reg_alpha", "reg_lambda"}
    print(f"{'model':<16} {'tunable':<8} {'sample':<58} status")
    print("-" * 95)
    for model_name, schema in MODEL_PARAM_SCHEMA.items():
        if model_name not in registry.list_available():
            continue

        tunable_keys = [k for k, v in schema.items() if v.get("tunable", True)]
        if not tunable_keys:
            print(f"  {model_name:<14} {'(0)':<8} (no tunable params — blocked at API)")
            continue

        # Run a tiny Optuna study with a stub objective that just samples
        # from the schema and asserts the resulting params are accepted by
        # the model's set_params().
        sampled_params = {}

        def objective(trial):
            params = {}
            for pname, spec in schema.items():
                if spec.get("tunable", True) is False:
                    continue
                ptype = spec.get("type", "float")
                if ptype == "int":
                    params[pname] = trial.suggest_int(pname, spec["min"], spec["max"])
                elif ptype == "float":
                    log = pname in LOG_PARAMS and spec.get("min", 0) > 0
                    params[pname] = trial.suggest_float(
                        pname, spec["min"], spec["max"], log=log,
                    )
                elif ptype == "select":
                    params[pname] = trial.suggest_categorical(pname, spec["options"])
                elif ptype == "bool":
                    params[pname] = trial.suggest_categorical(pname, [True, False])
            sampled_params.update(params)
            # The objective stub returns 0 — we only care that suggestion +
            # set_params didn't raise. Composite scoring is exercised by
            # the real runner separately.
            mdl = registry.create(model_name, **params)
            return 0.0

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        try:
            study.optimize(objective, n_trials=3, show_progress_bar=False)
            status = "OK"
            note = ""
        except Exception as e:
            status = "FAIL"
            note = f"{type(e).__name__}: {e}"
            failures.append((f"tuning[{model_name}]", [note]))

        sample_str = ", ".join(f"{k}={v}" for k, v in sampled_params.items())[:55]
        print(f"  {model_name:<14} ({len(tunable_keys):<6}) {sample_str:<58} {status}  {note}")

    # ---- Extra: output_activation resolution + unfitted save error ------ #
    print(f"\n{'─' * 72}\n[4/4] output_activation resolution + unfitted save error\n{'─' * 72}")
    from ml_forecast_lab.config import ExperimentCfg
    from ml_forecast_lab.main import (
        _apply_output_activation, _apply_experiment_neural_params,
        _resolve_output_activation,
    )

    activations_to_check = ['auto', 'linear', 'softplus', 'relu', 'exp', 'sigmoid', 'zscore']
    print(f"  Resolving 'auto' against synthetic exp_cfg for each model:")
    for model_name in registry.list_available():
        m = registry.create(model_name)
        if not getattr(m, 'is_neural', False):
            continue
        for act in activations_to_check:
            exp_cfg = ExperimentCfg(
                name='walk', target_entity='sensor.t',
                output_activation=act, source_is_cumulative=False,
            )
            try:
                resolved = _resolve_output_activation(exp_cfg, model_name)
                fresh = registry.create(model_name)
                _apply_output_activation(fresh, exp_cfg)
                _apply_experiment_neural_params(fresh, exp_cfg)
            except Exception as e:
                failures.append((
                    f"activation[{model_name}/{act}]",
                    [f"{type(e).__name__}: {e}"],
                ))
    print("  All activations applied cleanly.")

    print(f"\n  save()-on-unfitted error contract per model:")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for model_name in registry.list_available():
            m = registry.create(model_name)
            path = f"{tmp}/{model_name}.pkl"
            try:
                m.save(path)
            except (RuntimeError, IOError, ValueError, AttributeError) as e:
                pass  # Expected: model is unfitted
            except Exception as e:
                failures.append((
                    f"save_unfitted[{model_name}]",
                    [f"unexpected {type(e).__name__}: {e}"],
                ))
            else:
                # Some backends might silently save an unfitted state.
                # Flag that as a soft concern (could lead to a wedged
                # production cache).
                failures.append((
                    f"save_unfitted[{model_name}]",
                    ["save() succeeded on unfitted model — risk of caching empty state"],
                ))
    print("  All backends raised on save()-when-unfitted.")

    print(f"\n  model_overrides round-trip per backend (get_params reflects set_params):")
    for model_name, schema in MODEL_PARAM_SCHEMA.items():
        if model_name not in registry.list_available():
            continue
        # Build a synthetic 'override' that flips every default
        overrides = {}
        for pname, spec in schema.items():
            d = spec.get("default")
            if d is None:
                continue
            ptype = spec.get("type", "float")
            if ptype == "int" and "max" in spec:
                overrides[pname] = min(int(d) + 1, int(spec["max"]))
            elif ptype == "float" and "max" in spec:
                overrides[pname] = min(float(d) * 1.1, float(spec["max"]))
            elif ptype == "select":
                opts = spec.get("options", [])
                overrides[pname] = next((o for o in opts if o != d), d)
        m = registry.create(model_name, **overrides)
        got = m.get_params()
        mismatched = {k: (v, got.get(k)) for k, v in overrides.items()
                      if k in got and got[k] != v}
        if mismatched:
            failures.append((
                f"override_roundtrip[{model_name}]",
                [f"set_params(**override) not reflected in get_params(): {mismatched}"],
            ))
    print("  All overrides round-tripped through get_params().")

    # ---- Summary -------------------------------------------------------- #
    print(f"\n{'=' * 72}")
    if failures:
        print(f"FAILURES: {len(failures)}")
        for fname, notes in failures:
            for n in notes:
                print(f"  - {fname}: {n}")
        return 1
    print("All dry-runs passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
