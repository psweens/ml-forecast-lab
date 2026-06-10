"""
Chronos-Bolt zero-shot foundation-model backend for ML Forecast Lab.

Wraps Amazon's Chronos-Bolt (Ansari et al. 2024,
https://arxiv.org/abs/2403.07815; Bolt variant 2024,
https://github.com/amazon-science/chronos-forecasting): a pretrained
time-series foundation model that forecasts zero-shot — no training on
the user's data at all. ``fit()`` only caches the training tail (for
context-stitching on short windows) and loads the pretrained weights;
every forecast conditions the frozen model on the window's recent
history.

Why this matters for HA users: zero-shot models produce sensible
forecasts from day one, before there is enough history to train any
supervised backend. They are also immune to the overfitting failure
modes of small-data neural training. The trade-off is that they consume
the target series only — covariate channels are ignored, like the
classical backends.

The Bolt family is the CPU-efficient direct-multistep variant of
Chronos: ``bolt-tiny`` (9M params) runs comfortably on a Pi 5. Weights
are downloaded from the Hugging Face Hub on first use (~30 MB for tiny)
and cached locally; subsequent fits and forecasts are fully offline.

Requires the optional ``chronos-forecasting`` package. The backend is
skipped at registration time when the package is missing (same pattern
as catboost / statsforecast).
"""

import logging
import pickle
import time
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np

from .base import ForecastModel

logger = logging.getLogger(__name__)

try:
    from chronos import BaseChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    BaseChronosPipeline = None  # type: ignore[assignment]
    warnings.warn(
        "chronos-forecasting is not installed. ChronosBoltModel will not be "
        "functional. Install it with: pip install chronos-forecasting",
        ImportWarning,
    )


class ChronosBoltModel(ForecastModel):
    """
    Chronos-Bolt zero-shot foundation model.

    No training happens on the user's data: ``fit()`` caches the target
    tail and loads the pretrained pipeline; ``predict_sequence()``
    conditions the frozen model on each window's target history and
    returns the model's probabilistic mean forecast.

    The pipeline is cached at class level keyed by (model id, device) so
    repeated fits across CV folds and retrain cycles reuse the loaded
    weights instead of re-reading them from disk.
    """

    # Class-level pipeline cache: weights load once per process.
    _PIPELINE_CACHE: Dict[tuple, Any] = {}

    # Inference batch size — bounds peak memory on small devices while
    # still amortising the transformer forward pass across windows.
    _BATCH_SIZE = 32

    def __init__(
        self,
        model_name: str = "amazon/chronos-bolt-tiny",
        context_length: int = 512,
        device: str = "cpu",
        train_history: int = 512,
        target_channel: int = 0,
        sequence_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not CHRONOS_AVAILABLE:
            raise RuntimeError(
                "chronos-forecasting is not installed. "
                "Install with: pip install chronos-forecasting"
            )
        self.model_name = str(model_name)
        self.context_length = int(context_length)
        self.device = str(device)
        self.train_history = int(train_history)
        self.target_channel = int(target_channel)
        self.sequence_length = sequence_length

        self._train_tail: Optional[np.ndarray] = None
        self._n_horizons: int = 1
        # In extended-window mode the window's future positions hold zero
        # placeholders on the target channel — the context must stop at
        # past_window_size (same contract as seasonal_naive).
        self._past_window_size: Optional[int] = None

    @property
    def name(self) -> str:
        return "chronos_bolt"

    @property
    def is_neural(self) -> bool:
        # Consume sliding windows so the target series can be pulled from
        # channel ``target_channel`` — same rationale as the classical
        # backends.
        return True

    @property
    def model_family(self) -> str:
        return 'foundation'

    @classmethod
    def _load_pipeline(cls, model_name: str, device: str):
        """Load (or fetch from cache) the pretrained Chronos pipeline.

        Split out as a classmethod so tests can monkeypatch the loader or
        inject a fake pipeline into ``_PIPELINE_CACHE`` without touching
        the network.
        """
        key = (model_name, device)
        if key not in cls._PIPELINE_CACHE:
            import torch
            logger.info(f"Loading Chronos pipeline {model_name!r} on {device}...")
            t0 = time.time()
            cls._PIPELINE_CACHE[key] = BaseChronosPipeline.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.float32,
            )
            logger.info(
                f"Chronos pipeline {model_name!r} ready in {time.time() - t0:.1f}s"
            )
        return cls._PIPELINE_CACHE[key]

    def _get_pipeline(self):
        try:
            return self._load_pipeline(self.model_name, self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Chronos pipeline {self.model_name!r}: {e}. "
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

        Zero-shot: nothing is trained. The tail is kept so short inference
        windows can be stitched up to a useful context length, mirroring
        the statsforecast backends.
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

        keep = max(self.train_history, 16)
        self._train_tail = np.asarray(tail_source[-keep:], dtype=np.float32)

        # Extended-window contract (same as seasonal_naive): predictions
        # must only condition on the past block of each window.
        extended = bool(kwargs.get('extended_window', False))
        pw = kwargs.get('past_window_size')
        self._past_window_size = int(pw) if extended and pw is not None else None

        # Load weights now (not at first forecast) so a missing download
        # fails the benchmark fit step with a clear message instead of
        # stalling the forecast cycle later.
        self._get_pipeline()

        self._is_fitted = True
        elapsed = time.time() - start_time
        logger.info(
            f"chronos_bolt fit (zero-shot, tail-cache only): "
            f"model={self.model_name}, tail={len(self._train_tail)}, "
            f"context_length={self.context_length}"
        )
        return {"time_seconds": elapsed, "epochs": 0, "best_val_loss": 0.0}

    def _context_from_window(self, window: np.ndarray) -> np.ndarray:
        """Build the conditioning context from one (seq_len, n_channels) window."""
        target = window[:, self.target_channel]
        if (self._past_window_size is not None
                and self._past_window_size < len(target)):
            target = target[: self._past_window_size]
        # Stitch the cached training tail in front of short windows so the
        # model sees enough history to pick up the seasonal pattern.
        if (self._train_tail is not None
                and len(target) < self.context_length):
            target = np.concatenate([self._train_tail, target])
        # Cap at context_length — Bolt truncates internally anyway, this
        # just keeps the tensors small.
        if len(target) > self.context_length:
            target = target[-self.context_length:]
        return np.asarray(target, dtype=np.float32)

    def _forecast_batch(self, contexts: List[np.ndarray], h: int) -> np.ndarray:
        """Run the pipeline over a list of contexts, returning (n, h) means."""
        import torch
        pipeline = self._get_pipeline()
        out = np.zeros((len(contexts), h), dtype=np.float32)
        for start in range(0, len(contexts), self._BATCH_SIZE):
            chunk = contexts[start:start + self._BATCH_SIZE]
            tensors = [torch.tensor(c, dtype=torch.float32) for c in chunk]
            # predict_quantiles returns (quantiles, mean): quantiles is
            # (batch, h, n_quantile_levels), mean is (batch, h). The mean
            # is Bolt's native point forecast head.
            _, mean = pipeline.predict_quantiles(
                tensors,
                prediction_length=h,
                quantile_levels=[0.1, 0.5, 0.9],
            )
            out[start:start + len(chunk)] = (
                mean.detach().cpu().numpy().astype(np.float32)
            )
        return out

    def predict_sequence(self, X: np.ndarray) -> np.ndarray:
        self._validate_fitted()
        if X.ndim != 3:
            raise ValueError(f"Expected 3-D windowed input, got shape {X.shape}")
        contexts = [self._context_from_window(X[i]) for i in range(X.shape[0])]
        preds = self._forecast_batch(contexts, self._n_horizons)
        return preds

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._validate_fitted()
        self._validate_X(X)
        X_seq = self._reshape_to_sequences(X)
        contexts = [self._context_from_window(X_seq[i]) for i in range(X_seq.shape[0])]
        preds = self._forecast_batch(contexts, max(self._n_horizons, 1))
        return preds[:, 0]

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantile_levels: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Native probabilistic forecast: (n_samples, n_horizons, n_quantiles).

        Chronos-Bolt is trained on quantile loss, so these bands come from
        the model itself rather than post-hoc residual calibration. Not
        consumed by the benchmark pipeline (which calibrates conformal
        bands uniformly across backends) — exposed for direct use, same
        as DLinear's optional quantile head.
        """
        import torch
        self._validate_fitted()
        if X.ndim != 3:
            raise ValueError(f"Expected 3-D windowed input, got shape {X.shape}")
        levels = quantile_levels or [0.1, 0.5, 0.9]
        pipeline = self._get_pipeline()
        chunks = []
        contexts = [self._context_from_window(X[i]) for i in range(X.shape[0])]
        for start in range(0, len(contexts), self._BATCH_SIZE):
            chunk = contexts[start:start + self._BATCH_SIZE]
            tensors = [torch.tensor(c, dtype=torch.float32) for c in chunk]
            quantiles, _ = pipeline.predict_quantiles(
                tensors,
                prediction_length=self._n_horizons,
                quantile_levels=levels,
            )
            chunks.append(quantiles.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)

    def get_params(self) -> Dict[str, Any]:
        return deepcopy({
            "model_name": self.model_name,
            "context_length": self.context_length,
            "device": self.device,
            "train_history": self.train_history,
            "target_channel": self.target_channel,
            "sequence_length": self.sequence_length,
        })

    def set_params(self, **kwargs: Any) -> None:
        valid = {"model_name", "context_length", "device", "train_history",
                 "target_channel", "sequence_length"}
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
            logger.info(f"Saved chronos_bolt to {path}")
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
            logger.info(f"Loaded chronos_bolt from {path}")
        except Exception as e:
            raise IOError(f"Failed to load model from {path}: {e}")
