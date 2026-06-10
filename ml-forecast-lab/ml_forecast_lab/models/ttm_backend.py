"""
IBM Granite TTM (Tiny Time Mixer) zero-shot foundation-model backend.

Wraps IBM's TTM (Ekambaram et al. 2024, NeurIPS,
https://arxiv.org/abs/2401.03955; weights at
https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2): a
pretrained MLP-mixer-based time-series foundation model in the 1–5M
parameter range — by far the smallest of the published foundation
models, and the one that actually fits the Pi-5 compute envelope as a
first-class citizen rather than a compromise.

Zero-shot operation: ``fit()`` caches the training tail (used to extend
short inference windows up to the model's fixed context length) and
loads the pretrained weights; forecasts condition the frozen model on
each window's recent target history. No gradient steps are taken on the
user's data. Like the classical and Chronos backends, TTM consumes the
target series only — covariate channels of the window are ignored.

TTM checkpoints have a fixed (context_length, prediction_length)
geometry — e.g. 512→96 for the r2 main revision. The granite-tsfm
``get_model`` helper selects the closest published revision for the
requested geometry; horizons beyond the checkpoint's prediction length
are produced autoregressively (forecast, append, re-forecast).

Weights are downloaded from the Hugging Face Hub on first use (~5–20 MB)
and cached locally; subsequent fits and forecasts are fully offline.

Requires the optional ``granite-tsfm`` package. The backend is skipped
at registration time when the package is missing (same pattern as
catboost / statsforecast).
"""

import importlib.util
import logging
import pickle
import time
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

from .base import ForecastModel

logger = logging.getLogger(__name__)

# Availability is probed with find_spec rather than an actual import:
# importing `tsfm_public` drags the whole transformers stack into memory
# (~300 MB settled RSS on top of torch), and this module is imported at
# app startup by the registry even when the user never enables the
# backend. The heavy import is deferred to _load_model(), i.e. the first
# actual fit/predict.
TSFM_AVAILABLE = importlib.util.find_spec("tsfm_public") is not None


def _resolve_get_model():
    """Deferred import of granite-tsfm's get_model helper.

    get_model moved to the package root in newer granite-tsfm releases;
    the toolkit path is the long-standing import. Try both.
    """
    try:
        from tsfm_public import get_model
    except ImportError:
        from tsfm_public.toolkit.get_model import get_model
    return get_model


class TTMModel(ForecastModel):
    """
    IBM Granite TTM zero-shot foundation model.

    No training happens on the user's data: ``fit()`` caches the target
    tail and loads the pretrained checkpoint; ``predict_sequence()``
    conditions the frozen model on each window's target history. TTM
    applies its own per-window standard scaling internally, so raw
    target values are passed straight through.

    The loaded checkpoint is cached at class level keyed by
    (model path, context length, prediction length) so repeated fits
    across CV folds and retrain cycles reuse the weights.
    """

    # Class-level model cache: weights load once per process.
    _MODEL_CACHE: Dict[tuple, Any] = {}

    # Inference batch size — bounds peak memory on small devices.
    _BATCH_SIZE = 64

    def __init__(
        self,
        model_path: str = "ibm-granite/granite-timeseries-ttm-r2",
        context_length: int = 512,
        prediction_length: int = 96,
        train_history: int = 1024,
        target_channel: int = 0,
        sequence_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not TSFM_AVAILABLE:
            raise RuntimeError(
                "granite-tsfm is not installed. "
                "Install with: pip install granite-tsfm"
            )
        self.model_path = str(model_path)
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.train_history = int(train_history)
        self.target_channel = int(target_channel)
        self.sequence_length = sequence_length

        self._train_tail: Optional[np.ndarray] = None
        self._n_horizons: int = 1
        # Extended-window contract (same as seasonal_naive / chronos_bolt):
        # condition only on the past block of each window.
        self._past_window_size: Optional[int] = None

    @property
    def name(self) -> str:
        return "ttm"

    @property
    def is_neural(self) -> bool:
        # Consume sliding windows so the target series can be pulled from
        # channel ``target_channel``.
        return True

    @property
    def model_family(self) -> str:
        return 'foundation'

    @classmethod
    def _load_model(cls, model_path: str, context_length: int,
                    prediction_length: int):
        """Load (or fetch from cache) the pretrained TTM checkpoint.

        Split out as a classmethod so tests can monkeypatch the loader or
        inject a fake model into ``_MODEL_CACHE`` without touching the
        network.
        """
        key = (model_path, context_length, prediction_length)
        if key not in cls._MODEL_CACHE:
            _ttm_get_model = _resolve_get_model()  # deferred heavy import
            logger.info(
                f"Loading TTM checkpoint {model_path!r} "
                f"(context={context_length}, pred={prediction_length})..."
            )
            t0 = time.time()
            try:
                model = _ttm_get_model(
                    model_path,
                    context_length=context_length,
                    prediction_length=prediction_length,
                )
            except TypeError:
                # Older granite-tsfm get_model signatures — fall back to
                # the checkpoint's default geometry.
                model = _ttm_get_model(model_path)
            model.eval()
            cls._MODEL_CACHE[key] = model
            logger.info(
                f"TTM checkpoint {model_path!r} ready in {time.time() - t0:.1f}s"
            )
        return cls._MODEL_CACHE[key]

    def _get_model(self):
        try:
            return self._load_model(
                self.model_path, self.context_length, self.prediction_length,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load TTM checkpoint {self.model_path!r}: {e}. "
                f"First use needs internet access to download the pretrained "
                f"weights from the Hugging Face Hub (cached locally afterwards)."
            ) from e

    def _reshape_to_sequences(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        seq_len = self.sequence_length or n_features
        n_features_per_step = (
            n_features // seq_len if n_features % seq_len == 0 else 1
        )
        return X[:, :seq_len * n_features_per_step].reshape(
            n_samples, seq_len, n_features_per_step,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            **kwargs: Any) -> Dict[str, Any]:
        """Cache the target tail and ensure the pretrained weights are loaded.

        Zero-shot: nothing is trained. The tail extends short inference
        windows up to the checkpoint's fixed context length.
        """
        self._validate_X(X_train)
        y_train = self._validate_y(y_train)
        start_time = time.time()

        if y_train.ndim == 2 and y_train.shape[1] > 1:
            self._n_horizons = y_train.shape[1]
            tail_source = y_train[:, 0]
        else:
            self._n_horizons = 1
            tail_source = y_train.ravel()

        keep = max(self.train_history, self.context_length, 16)
        self._train_tail = np.asarray(tail_source[-keep:], dtype=np.float32)

        extended = bool(kwargs.get('extended_window', False))
        pw = kwargs.get('past_window_size')
        self._past_window_size = int(pw) if extended and pw is not None else None

        # Load weights now so a missing download fails the benchmark fit
        # step with a clear message instead of stalling the forecast cycle.
        self._get_model()

        self._is_fitted = True
        elapsed = time.time() - start_time
        logger.info(
            f"ttm fit (zero-shot, tail-cache only): "
            f"model={self.model_path}, tail={len(self._train_tail)}, "
            f"context_length={self.context_length}"
        )
        return {"time_seconds": elapsed, "epochs": 0, "best_val_loss": 0.0}

    def _context_from_window(self, window: np.ndarray) -> np.ndarray:
        """Build an exactly-context_length conditioning series from one window.

        TTM checkpoints take fixed-length input, so the window's past
        target slice is extended with the cached training tail and, if
        still short, left-padded with the earliest available value (edge
        padding keeps the internal std-scaler statistics sane, unlike
        zero padding which would drag the window mean down).
        """
        target = window[:, self.target_channel]
        if (self._past_window_size is not None
                and self._past_window_size < len(target)):
            target = target[: self._past_window_size]
        if (self._train_tail is not None
                and len(target) < self.context_length):
            target = np.concatenate([self._train_tail, target])
        if len(target) >= self.context_length:
            target = target[-self.context_length:]
        else:
            pad = np.full(
                self.context_length - len(target),
                target[0] if len(target) else 0.0,
                dtype=np.float32,
            )
            target = np.concatenate([pad, target])
        return np.asarray(target, dtype=np.float32)

    def _forecast_contexts(self, contexts: np.ndarray, h: int) -> np.ndarray:
        """Forecast h steps for (n, context_length) contexts, batched.

        Horizons beyond the checkpoint's native prediction length are
        generated autoregressively: forecast a block, append it to the
        context, re-forecast.
        """
        import torch
        model = self._get_model()
        n = contexts.shape[0]
        out = np.zeros((n, h), dtype=np.float32)
        for start in range(0, n, self._BATCH_SIZE):
            ctx = contexts[start:start + self._BATCH_SIZE].copy()
            produced = 0
            while produced < h:
                x = torch.tensor(ctx, dtype=torch.float32).unsqueeze(-1)
                with torch.no_grad():
                    result = model(past_values=x)
                block = result.prediction_outputs[:, :, 0].cpu().numpy()
                take = min(block.shape[1], h - produced)
                out[start:start + ctx.shape[0], produced:produced + take] = (
                    block[:, :take]
                )
                produced += take
                if produced < h:
                    # Roll the context forward with the new predictions.
                    ctx = np.concatenate([ctx, block[:, :take]], axis=1)
                    ctx = ctx[:, -self.context_length:]
        return out

    def predict_sequence(self, X: np.ndarray) -> np.ndarray:
        self._validate_fitted()
        if X.ndim != 3:
            raise ValueError(f"Expected 3-D windowed input, got shape {X.shape}")
        contexts = np.stack(
            [self._context_from_window(X[i]) for i in range(X.shape[0])]
        )
        return self._forecast_contexts(contexts, self._n_horizons)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._validate_fitted()
        self._validate_X(X)
        X_seq = self._reshape_to_sequences(X)
        contexts = np.stack(
            [self._context_from_window(X_seq[i]) for i in range(X_seq.shape[0])]
        )
        preds = self._forecast_contexts(contexts, max(self._n_horizons, 1))
        return preds[:, 0]

    def get_params(self) -> Dict[str, Any]:
        return deepcopy({
            "model_path": self.model_path,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "train_history": self.train_history,
            "target_channel": self.target_channel,
            "sequence_length": self.sequence_length,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"model_path", "context_length", "prediction_length",
                 "train_history", "target_channel", "sequence_length"}
        for k, v in kwargs.items():
            if k not in valid:
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)

    def save(self, path: str) -> None:
        """Persist params + tail. The pretrained weights live in the HF
        cache, not the model file — load() re-attaches them lazily."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "params": self.get_params(),
                    "train_tail": self._train_tail,
                    "n_horizons": self._n_horizons,
                    "past_window_size": self._past_window_size,
                }, f)
            logger.info(f"Saved ttm to {path}")
        except Exception as e:
            raise IOError(f"Failed to save model to {path}: {e}")

    def load(self, path: str) -> None:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.set_params(**data["params"])
            self._train_tail = data.get("train_tail")
            self._n_horizons = data.get("n_horizons", 1)
            self._past_window_size = data.get("past_window_size")
            self._is_fitted = True
            logger.info(f"Loaded ttm from {path}")
        except Exception as e:
            raise IOError(f"Failed to load model from {path}: {e}")
