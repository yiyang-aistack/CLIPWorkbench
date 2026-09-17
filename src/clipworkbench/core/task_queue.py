"""In-memory async task queue for tracking training jobs.

Used by the FastAPI layer to run training in the background
and report progress via /status/{task_id}.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime
from typing import Any

from clipworkbench.core.schema import TrainStatus


class TaskQueue:
    """Thread-safe in-memory task store (no external broker needed)."""

    def __init__(self) -> None:
        self._tasks: dict[str, TrainStatus] = {}
        self._lock = threading.Lock()

    def create(self) -> TrainStatus:
        """Create a new training task and return its status tracker."""
        task_id = str(uuid.uuid4())
        status = TrainStatus(
            task_id=task_id,
            status="queued",
            start_time=datetime.now().isoformat(),
        )
        with self._lock:
            self._tasks[task_id] = status
        return status

    def get(self, task_id: str) -> TrainStatus | None:
        with self._lock:
            return self._tasks.get(task_id)

    def update(self, task_id: str, **fields: Any) -> None:
        with self._lock:
            status = self._tasks.get(task_id)
            if status is None:
                return
            for key, value in fields.items():
                setattr(status, key, value)

    def list_all(self) -> list[TrainStatus]:
        with self._lock:
            return list(self._tasks.values())

    def delete(self, task_id: str) -> None:
        with self._lock:
            self._tasks.pop(task_id, None)


# Module-level singleton (imported by API routes)
task_queue = TaskQueue()
