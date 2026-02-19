# simulations/traffic/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional, Protocol

from pydantic import BaseModel, Field

Endpoint = Literal["generate", "extract"]
PromptSize = Literal["short", "medium", "long"]


class TrafficConfig(BaseModel):
    """
    Single source of truth used by:
      - CLI (main.py) to construct config
      - runner.py to schedule + connect + write outputs
      - scenarios to build deterministic RequestSpec payloads
    """

    # identity
    run_id: str = Field(..., min_length=1)
    scenario: str = Field(..., min_length=1)

    # connection
    base_url: str = Field(default="http://127.0.0.1:8000")
    api_key: str = Field(..., min_length=1)
    timeout_s: float = Field(default=20.0, gt=0.0)

    # load/scheduling
    duration_s: float = Field(default=20.0, gt=0.0)
    rps: float = Field(default=2.0, gt=0.0)
    max_in_flight: int = Field(default=4, ge=1)

    # determinism
    seed: int = Field(default=13, ge=0)

    # scenario knobs
    model: Optional[str] = None
    cache: bool = Field(default=True)

    # demo-a
    prompt_size: PromptSize = Field(default="short")

    # demo-b
    schema_id: str = Field(default="ticket_v1")
    repair: bool = Field(default=True)

    # generation parameters (used by both generate/extract)
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop: Optional[list[str]] = None


class RequestSpec(BaseModel):
    idx: int
    endpoint: Endpoint
    payload: Dict[str, Any]
    tags: Dict[str, Any] = Field(default_factory=dict)


class Event(BaseModel):
    run_id: str
    scenario: str

    idx: int
    endpoint: Endpoint

    started_at_unix: float
    elapsed_ms: float

    ok: bool
    status: Optional[int] = None

    cached: Optional[bool] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    error: Optional[str] = None
    error_payload: Optional[Any] = None

    model: Optional[str] = None
    tags: Dict[str, Any] = Field(default_factory=dict)


class Summary(BaseModel):
    run_id: str
    scenario: str

    duration_s: float
    sent: int
    completed: int

    ok: bool
    error_rate: float

    avg_latency_ms: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None

    cache_hit_rate: Optional[float] = None

    total_prompt_tokens: Optional[int] = None
    total_completion_tokens: Optional[int] = None

    breakdown: Dict[str, Any] = Field(default_factory=dict)


# -------------------------
# Hooks (midstream actions)
# -------------------------

class ScenarioHook(Protocol):
    """
    Midstream side-effect executed during traffic.

    Runner behavior (back-compat):
      - Prefer calling: hook.run(client, cfg, repo_root=<Path>)
      - Fallback to:     hook.run(client, cfg)

    Required fields:
      - name: stable identifier for event tags / logs
      - at_s: seconds since scenario start to trigger the hook
    """
    name: str
    at_s: float

    def run(self, client: Any, cfg: TrafficConfig, repo_root: Any = None) -> None: ...


@dataclass(frozen=True)
class Scenario:
    """
    Scenario is a deterministic request factory (+ optional setup + optional hooks).
    The runner owns timing; scenario owns payload content.
    """
    name: str
    endpoint: Endpoint
    build_requests: Any  # Callable[[TrafficConfig], Iterable[RequestSpec]]

    # Called once before traffic starts (optional)
    setup: Any = None

    # Hooks executed midstream by runner (optional)
    hooks: Optional[list[ScenarioHook]] = None