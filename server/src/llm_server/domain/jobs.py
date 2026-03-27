from __future__ import annotations

from enum import Enum


class AsyncJobLifecycle(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @classmethod
    def from_status(cls, status: str | None) -> "AsyncJobLifecycle | None":
        if not isinstance(status, str):
            return None
        normalized = status.strip().lower()
        for candidate in cls:
            if candidate.value == normalized:
                return candidate
        return None
