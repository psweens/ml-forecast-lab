"""
CatBoost forecasting model backend for ML Forecast Lab.

Implements Yandex's CatBoost gradient boosting library as a third tabular
backend alongside LightGBM and XGBoost. CatBoost's ordered boosting and
permutation-driven categorical handling often make it a strong default on
noisy or covariate-rich smart-home signals.
"""

import logging
import pickle
import warnings
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

from .base import ForecastModel

logger = logging.getLogger(__name__)

try:
    from catboost import CatBoostRegressor, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn(
        "CatBoost is not installed. CatBoostModel will not be functional. "
        "Install it with: pip install catboost",
        ImportWarning,
    )


class CatBoostModel(ForecastModel):
    """
    CatBoost-based forecasting model for time-series prediction.

    Uses ordered boosting (which prevents target leakage during training on
    categorical data) and oblivious symmetric trees (which are faster at
    inference than depth-wise trees). Supports early stopping on a tail
    validation slice and recursive multi-step forecasting via the inherited
    ``predict_multi``.
    """

    @property
    def name(self) -> str:
        return "catboost"

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        subsample: float = 0.8,
        colsample_bylevel: float = 0.8,
        min_data_in_leaf: int = 10,
        bootstrap_type: str = "Bernoulli",
        verbose: int = 0,
        loss_fn: str = 'huber',
        tweedie_variance_power: float = 1.5,
        huber_delta: float = 1.0,
        patience: int = 50,
    ) -> None:
        super().__init__()
        if not CATBOOST_AVAILABLE:
            raise RuntimeError(
                "CatBoost is not installed. Install with: pip install catboost"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.colsample_bylevel = colsample_bylevel
        self.min_data_in_leaf = min_data_in_leaf
        # Bernoulli supports subsample; Bayesian does not but is more robust on
        # tiny datasets. Kept configurable for parity with CatBoost upstream.
        self.bootstrap_type = bootstrap_type
        self.verbose = verbose
        self.loss_fn = loss_fn
        self.tweedie_variance_power = tweedie_variance_power
        self.huber_delta = huber_delta
        # v2.40.12: previously hardcoded at the training site.
        self.patience = patience

        self.model: Optional[CatBoostRegressor] = None
        self.feature_names_: Optional[list] = None
        self.training_metadata: Dict[str, Any] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Train CatBoost with a tail validation slice and early stopping."""
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost is not available")

        self._validate_X(X_train)
        y_train = self._validate_y(y_train)
        if len(X_train) != len(y_train):
            raise ValueError("X and y must have the same number of samples")

        feature_names = kwargs.get("feature_names")
        self.feature_names_ = (
            list(feature_names) if feature_names is not None
            else [f"feature_{i}" for i in range(X_train.shape[1])]
        )

        sample_weight = kwargs.get("sample_weight")
        eval_set = kwargs.get("eval_set")
        if eval_set is None:
            split_idx = int(len(X_train) * 0.8)
            X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
            y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
            w_tr = sample_weight[:split_idx] if sample_weight is not None else None
        else:
            X_tr, y_tr = X_train, y_train
            X_val, y_val = eval_set
            w_tr = sample_weight

        # CatBoost native loss-function names. mse → RMSE (CatBoost's
        # squared-error implementation), mae → MAE, huber takes a delta
        # parameter, tweedie takes a variance_power.
        if self.loss_fn == "mse":
            loss_function = "RMSE"
        elif self.loss_fn == "mae":
            loss_function = "MAE"
        elif self.loss_fn in ("tweedie", "dilate"):
            # 'dilate' is a neural shape+time loss with no tree analogue → use
            # the peak-appropriate Tweedie objective for spiky targets.
            loss_function = f"Tweedie:variance_power={float(self.tweedie_variance_power):.3f}"
        else:  # huber (default) and anything unknown
            loss_function = f"Huber:delta={float(self.huber_delta):.3f}"

        params: Dict[str, Any] = {
            "iterations": self.n_estimators,
            "depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "l2_leaf_reg": self.l2_leaf_reg,
            "min_data_in_leaf": self.min_data_in_leaf,
            "loss_function": loss_function,
            # Pin RMSE as the eval metric so early stopping and the
            # epoch callback always have a stable key to read, even when
            # loss_function is Huber / Tweedie / MAE.
            "eval_metric": "RMSE",
            "verbose": self.verbose,
            "allow_writing_files": False,
            "random_seed": 42,
        }
        # CatBoost rejects subsample / colsample if bootstrap_type is "Bayesian";
        # only forward those when the bootstrap actually supports them.
        if self.bootstrap_type in ("Bernoulli", "MVS", "Poisson"):
            params["bootstrap_type"] = self.bootstrap_type
            params["subsample"] = self.subsample
        else:
            params["bootstrap_type"] = self.bootstrap_type
        params["rsm"] = self.colsample_bylevel  # CatBoost calls it rsm

        self.model = CatBoostRegressor(**params)

        train_pool = Pool(X_tr, label=y_tr, weight=w_tr,
                          feature_names=self.feature_names_)
        val_pool = Pool(X_val, label=y_val,
                        feature_names=self.feature_names_)

        # v2.40.12: patience_limit reads self.patience (was hardcoded
        # 50); min_delta margin prevents micro-improvements from
        # resetting patience.
        epoch_callback = kwargs.get("epoch_callback")
        best_val_loss = float('inf')
        patience_counter = 0
        patience_limit = int(self.patience)
        min_delta = float(getattr(self, 'min_delta', 1e-3))

        cat_callbacks = []
        _outer = self

        if epoch_callback is not None:
            class _EpochCB:
                """Emit per-iteration metrics to the training event bus."""
                def after_iteration(self_cb, info):
                    nonlocal best_val_loss, patience_counter
                    val_metrics = info.metrics.get("validation", {})
                    rmse_vals = val_metrics.get("RMSE", [])
                    val_loss = rmse_vals[-1] if rmse_vals else None
                    if val_loss is not None:
                        # v2.40.12: min_delta margin.
                        if val_loss < best_val_loss * (1.0 - min_delta):
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        # v2.40.13: cap displayed counter at limit.
                        _outer._emit_epoch(epoch_callback,
                            model_name=_outer.name,
                            epoch=info.iteration + 1,
                            total_epochs=_outer.n_estimators,
                            train_loss=val_loss,
                            val_loss=val_loss,
                            lr=_outer.learning_rate,
                            patience_counter=min(patience_counter, patience_limit),
                            patience_limit=patience_limit,
                            best_val_loss=best_val_loss)
                    return True  # continue training
            cat_callbacks.append(_EpochCB())

        self.model.fit(
            train_pool,
            eval_set=val_pool,
            # v2.40.13: was hardcoded 50 — only the Python progress
            # callback was fixed in v2.40.12. The LIBRARY's early-stop
            # param wasn't, so CatBoost trained past the v2.40.12
            # patience and the Python counter went past patience_limit
            # in the UI.
            early_stopping_rounds=patience_limit,
            use_best_model=True,
            verbose=False,
            callbacks=cat_callbacks if cat_callbacks else None,
        )

        best_iteration = int(self.model.get_best_iteration() or 0)
        best_val = float(
            self.model.get_best_score().get("validation", {}).get("RMSE", 0.0)
        )

        importances = self.model.get_feature_importance(type="FeatureImportance")
        feature_importances = dict(zip(self.feature_names_, importances))
        total = sum(feature_importances.values())
        if total > 0:
            feature_importances = {k: v / total for k, v in feature_importances.items()}

        self.training_metadata = {
            "best_iteration": best_iteration,
            "feature_importances": feature_importances,
            "num_features": len(self.feature_names_),
        }

        self._is_fitted = True
        logger.info(
            f"CatBoost trained: best_iteration={best_iteration}, "
            f"best_val_rmse={best_val:.6f}, n_features={len(self.feature_names_)}"
        )
        return {"time_seconds": 0.0, **self.training_metadata}

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._validate_fitted()
        self._validate_X(X)
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        if X.shape[1] != len(self.feature_names_):
            raise ValueError(
                f"Expected {len(self.feature_names_)} features, got {X.shape[1]}"
            )
        return np.asarray(self.model.predict(X), dtype=np.float32)

    def get_params(self) -> Dict[str, Any]:
        return deepcopy({
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "l2_leaf_reg": self.l2_leaf_reg,
            "subsample": self.subsample,
            "colsample_bylevel": self.colsample_bylevel,
            "min_data_in_leaf": self.min_data_in_leaf,
            "bootstrap_type": self.bootstrap_type,
            "verbose": self.verbose,
            "loss_fn": self.loss_fn,
            "tweedie_variance_power": self.tweedie_variance_power,
            "huber_delta": self.huber_delta,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {
            "n_estimators", "max_depth", "learning_rate", "l2_leaf_reg",
            "subsample", "colsample_bylevel", "min_data_in_leaf",
            "bootstrap_type", "verbose",
            "loss_fn", "tweedie_variance_power", "huber_delta",
        }
        for k, v in kwargs.items():
            if k not in valid:
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)
        if self.is_fitted:
            logger.warning(
                "Hyperparameters updated on a fitted model; call fit() again."
            )

    def save(self, path: str) -> None:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save unfitted model")
        try:
            with open(path, "wb") as f:
                pickle.dump(
                    {
                        "model": self.model,
                        "feature_names": self.feature_names_,
                        "training_metadata": self.training_metadata,
                        "hyperparameters": self.get_params(),
                    },
                    f,
                )
            logger.info(f"Saved CatBoost model to {path}")
        except Exception as e:
            logger.error(f"Failed to save model to {path}: {e}", exc_info=True)
            raise IOError(f"Failed to save model to {path}: {e}")

    def load(self, path: str) -> None:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.feature_names_ = data["feature_names"]
            self.training_metadata = data["training_metadata"]
            self.set_params(**data.get("hyperparameters", {}))
            self._is_fitted = True
            logger.info(f"Loaded CatBoost model from {path}")
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}", exc_info=True)
            raise IOError(f"Failed to load model from {path}: {e}")
