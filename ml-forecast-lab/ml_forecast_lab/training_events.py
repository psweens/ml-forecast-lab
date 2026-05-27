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

    # Per-subscriber queue cap. Sized to comfortably hold one full benchmark
    # pipeline's worth of telemetry (~28 models × 5 folds × ~50 epochs ≈ 7k
    # events). Slow / disconnected clients see their oldest events dropped
    # rather than driving the add-on into OOM on a Pi.
    SUBSCRIBER_QUEUE_MAX = 8192

    # Per-experiment history cap. clear_history() empties it at every
    # pipeline_start, but a stuck consumer + a long run without a clear can
    # still pile events up. Same order of magnitude as the queue cap.
    HISTORY_MAX = 8192

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

        Called from the async SSE endpoint. Returns a bounded Queue that
        will receive TrainingEvent objects via call_soon_threadsafe.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=self.SUBSCRIBER_QUEUE_MAX)
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

    def _enqueue(self, q: asyncio.Queue, event: TrainingEvent) -> None:
        """put_nowait with oldest-drop fallback so a slow client doesn't OOM us."""
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            try:
                q.get_nowait()  # drop oldest
                q.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    def publish(self, event: TrainingEvent) -> None:
        """
        Publish an event from a training thread.

        Uses call_soon_threadsafe to bridge into each subscriber's
        event loop without blocking the training thread. History and
        subscriber queues are bounded with oldest-drop semantics so a
        stalled consumer cannot grow memory without limit.
        """
        with self._lock:
            history = self._history.setdefault(event.experiment_name, [])
            history.append(event)
            # Cap history with a slice rather than per-event popleft because
            # appends are common and the list is rarely near the cap.
            if len(history) > self.HISTORY_MAX:
                del history[: len(history) - self.HISTORY_MAX]
            subscribers = list(self._subscribers.get(event.experiment_name, []))

        for q, loop in subscribers:
            try:
                loop.call_soon_threadsafe(self._enqueue, q, event)
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


def summarise_history(events: List[TrainingEvent]) -> Dict[str, Any]:
    """Collapse an experiment's event history into a live-progress summary.

    Completions are counted ONLY within the most recent ``pipeline_start``
    window (the counter resets on every ``pipeline_start``) and clamped to
    the declared total. This is what stops the progress from reading past
    the total — e.g. ``9/5`` / ``180%`` — when stale or replayed events from
    an earlier run linger in the same history, or a re-run lands on the same
    stream.
    """
    import re

    current_model = ""
    completed = 0
    total = 0
    fold = total_folds = epoch = total_epochs = 0
    for ev in events:
        if ev.event_type == "pipeline_start":
            m = re.search(r"(\d+) model", ev.message or "")
            total = int(m.group(1)) if m else 0
            completed = 0          # only the latest run's completions count
            current_model = ""
        elif ev.event_type == "model_start":
            current_model = ev.model_name
        elif ev.event_type == "model_end":
            completed += 1
        elif ev.event_type == "epoch":
            fold = ev.fold
            total_folds = ev.total_folds
            epoch = ev.epoch
            total_epochs = ev.total_epochs
    if total:
        completed = min(completed, total)   # never report past the total
    return {
        "current_model": current_model,
        "completed_models": completed,
        "total_models": total,
        "progress_pct": min(100, round(completed / total * 100)) if total else 0,
        "fold": fold,
        "total_folds": total_folds,
        "epoch": epoch,
        "total_epochs": total_epochs,
    }
