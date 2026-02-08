# policy/src/llm_policy/types/artifact_read.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ArtifactReadResult(Generic[T]):
    """
    Generic "artifact read" wrapper for file-based artifact lanes.

    Use this when you want:
      - a resolved path
      - a simple ok flag
      - a typed payload (contracts model, policy model, etc.)
      - a human-readable error string (optional)

    Conventions:
      - ok=True means payload is valid/usable
      - ok=False means payload may be fail-closed and error should be inspected
    """
    ok: bool
    artifact: T
    resolved_path: str
    error: Optional[str] = None