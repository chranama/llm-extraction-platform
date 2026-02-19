# server/src/llm_server/services/backends/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class GenerateTimings:
    # best-effort; backends can leave as None
    total_ms: Optional[float] = None
    queue_ms: Optional[float] = None
    backend_ms: Optional[float] = None


@dataclass(frozen=True)
class GenerateUsage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass(frozen=True)
class GenerateResult:
    text: str
    usage: GenerateUsage = GenerateUsage()
    timings: GenerateTimings = GenerateTimings()
    raw: Optional[Dict[str, Any]] = None  # backend response for debugging/telemetry


@runtime_checkable
class LLMBackend(Protocol):
    """
    Common backend interface.

    IMPORTANT: Keep this compatible with existing route code that expects:
        backend.generate(prompt=..., max_new_tokens=..., temperature=..., top_p=..., top_k=..., stop=...)
    """

    backend_name: str  # "transformers" | "llamacpp" | "remote"
    model_id: str

    def ensure_loaded(self) -> None:
        """Optional: load heavy resources. Should be idempotent."""
        ...

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        # extra kwargs for future extension (safe to ignore)
        **kwargs: Any,
    ) -> str:
        """Return text only (route layer already handles token counting + caching)."""
        ...

    def generate_rich(
        self,
        *,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> GenerateResult:
        """Optional: richer return for later telemetry. Default can wrap generate()."""
        ...