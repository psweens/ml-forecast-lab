"""
Thread-safe training event bus for streaming live training metrics.

The training loops run in executor threads (via run_in_executor) while
SSE endpoints are async in the FastAPI event loop. This module bridges
the two using asyncio.Queue with call_soon_threadsafe.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrainingEvent:
    """A single training lifecycle event."""

    event_type: str
    """One of: pipeline_start, pipeline_end, model_start, model_end,
    fold_start, fold_end, epoch, early_stop."""

    experiment_name: str
    model_name: str = ""
    fold: int = 0
    total_folds: int = 0
    epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    best_val_loss: float = 0.0
    learning_rate: float = 0.0
    patience_counter: int = 0
    patience_limit: int = 0
    elapsed_seconds: float = 0.0
    message: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class TrainingEventBus:
    """Singleton event bus bridging training threads to async SSE streams."""

    _instance: Optional[TrainingEventBus] = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Tuple[asyncio.Queue, asyncio.AbstractEventLoop]]] = {}
        self._lock = threading.Lock()
        self._history: Dict[str, List[TrainingEvent]] = {}

    @classmethod
    def get_instance(cls) -> TrainingEventBus:
        """Return the global singleton, creating it if needed."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def subscribe(
        self, experiment_name: str, loop: asyncio.AbstractEventLoop,
    ) -> asyncio.Queue:
        """
        Register an async subscriber for an experiment's training events.

        Called from the async SSE endpoint. Returns a Queue that will
        receive TrainingEvent objects via call_soon_threadsafe.
        """
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.setdefault(experiment_name, []).append((q, loop))
        logger.debug(f"SSE subscriber added for {experiment_name}")
        return q

    def unsubscribe(self, experiment_name: str, q: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        with self._lock:
            subs = self._subscribers.get(experiment_name, [])
            self._subscribers[experiment_name] = [
                (sq, sl) for sq, sl in subs if sq is not q
            ]
        logger.debug(f"SSE subscriber removed for {experiment_name}")

    def publish(self, event: TrainingEvent) -> None:
        """
        Publish an event from a training thread.

        Uses call_soon_threadsafe to bridge into each subscriber's
        event loop without blocking the training thread.
        """
        with self._lock:
            self._history.setdefault(event.experiment_name, []).append(event)
            subscribers = list(self._subscribers.get(event.experiment_name, []))

        for q, loop in subscribers:
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception:
                pass  # Subscriber's loop may have closed

    def get_history(self, experiment_name: str) -> List[TrainingEvent]:
        """Return all events for an experiment (for SSE reconnection replay)."""
        with self._lock:
            return list(self._history.get(experiment_name, []))

    def clear_history(self, experiment_name: str) -> None:
        """Clear event history when a new pipeline run starts."""
        with self._lock:
            self._history.pop(experiment_name, None)
