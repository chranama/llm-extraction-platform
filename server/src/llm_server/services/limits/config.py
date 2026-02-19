# server/src/llm_server/services/limits/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _truthy(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(x: Any, default: int) -> int:
    if x is None or isinstance(x, bool):
        return default
    try:
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float) -> float:
    if x is None or isinstance(x, bool):
        return default
    try:
        return float(x)
    except Exception:
        return default


def _get_attr_or_key(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _get_nested(obj: Any, path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        cur = _get_attr_or_key(cur, part)
        if cur is None:
            return None
    return cur


def _env(name: str) -> str | None:
    v = os.environ.get(name)
    if v is None:
        return None
    s = v.strip()
    return s if s else None


# ---------------------------------------
# Generate Gate
# ---------------------------------------

@dataclass(frozen=True)
class GenerateGateConfig:
    enabled: bool
    max_concurrent: int
    max_queue: int
    timeout_seconds: float
    fail_fast: bool
    count_queued_as_in_flight: bool


GENERATE_GATE_DEFAULTS = GenerateGateConfig(
    enabled=True,
    max_concurrent=2,
    max_queue=32,
    timeout_seconds=30.0,
    fail_fast=True,
    count_queued_as_in_flight=False,
)


def load_generate_gate_config(settings: Any | None = None) -> GenerateGateConfig:
    """
    Reads nested config from server.yaml-like settings shape:
      limits.generate_gate.*

    Env vars override:
      GENERATE_GATE_ENABLED
      MAX_CONCURRENT_GENERATIONS
      MAX_GENERATE_QUEUE
      GENERATE_TIMEOUT_S
      GENERATE_GATE_FAIL_FAST
      GENERATE_GATE_COUNT_QUEUED_AS_IN_FLIGHT
    """
    if settings is None:
        from llm_server.core.config import get_settings  # late import to avoid cycles
        settings = get_settings()

    enabled_raw = _get_nested(settings, "limits.generate_gate.enabled")
    max_conc_raw = _get_nested(settings, "limits.generate_gate.max_concurrent")
    max_queue_raw = _get_nested(settings, "limits.generate_gate.max_queue")
    timeout_raw = _get_nested(settings, "limits.generate_gate.timeout_seconds")
    fail_fast_raw = _get_nested(settings, "limits.generate_gate.fail_fast")
    count_queued_raw = _get_nested(settings, "limits.generate_gate.count_queued_as_in_flight")

    enabled = _truthy(enabled_raw) if enabled_raw is not None else GENERATE_GATE_DEFAULTS.enabled
    max_concurrent = _as_int(max_conc_raw, GENERATE_GATE_DEFAULTS.max_concurrent)
    max_queue = _as_int(max_queue_raw, GENERATE_GATE_DEFAULTS.max_queue)
    timeout_seconds = _as_float(timeout_raw, GENERATE_GATE_DEFAULTS.timeout_seconds)
    fail_fast = _truthy(fail_fast_raw) if fail_fast_raw is not None else GENERATE_GATE_DEFAULTS.fail_fast
    count_queued_as_in_flight = (
        _truthy(count_queued_raw) if count_queued_raw is not None else GENERATE_GATE_DEFAULTS.count_queued_as_in_flight
    )

    # env overrides
    v = _env("GENERATE_GATE_ENABLED")
    if v is not None:
        enabled = _truthy(v)

    v = _env("MAX_CONCURRENT_GENERATIONS")
    if v is not None:
        max_concurrent = _as_int(v, max_concurrent)

    v = _env("MAX_GENERATE_QUEUE")
    if v is not None:
        max_queue = _as_int(v, max_queue)

    v = _env("GENERATE_TIMEOUT_S")
    if v is not None:
        timeout_seconds = _as_float(v, timeout_seconds)

    v = _env("GENERATE_GATE_FAIL_FAST")
    if v is not None:
        fail_fast = _truthy(v)

    v = _env("GENERATE_GATE_COUNT_QUEUED_AS_IN_FLIGHT")
    if v is not None:
        count_queued_as_in_flight = _truthy(v)

    # safety clamps
    if max_concurrent <= 0:
        max_concurrent = 1
    if max_queue < 0:
        max_queue = 0
    if timeout_seconds < 0:
        timeout_seconds = GENERATE_GATE_DEFAULTS.timeout_seconds
    if 0 < timeout_seconds < 0.5:
        timeout_seconds = 0.5

    return GenerateGateConfig(
        enabled=bool(enabled),
        max_concurrent=int(max_concurrent),
        max_queue=int(max_queue),
        timeout_seconds=float(timeout_seconds),
        fail_fast=bool(fail_fast),
        count_queued_as_in_flight=bool(count_queued_as_in_flight),
    )


# ---------------------------------------
# Early Reject (middleware)
# ---------------------------------------

@dataclass(frozen=True)
class GenerateEarlyRejectConfig:
    enabled: bool
    reject_queue_depth_gte: int
    reject_in_flight_gte: int
    routes: tuple[str, ...]


GENERATE_EARLY_REJECT_DEFAULTS = GenerateEarlyRejectConfig(
    enabled=True,
    reject_queue_depth_gte=0,   # 0 => disabled check
    reject_in_flight_gte=0,     # 0 => disabled check
    routes=("/v1/generate", "/v1/generate/batch"),
)


def load_generate_early_reject_config(settings: Any | None = None) -> GenerateEarlyRejectConfig:
    """
    Reads nested config from server.yaml-like settings shape:
      limits.generate_early_reject.*

    Env vars override:
      GENERATE_EARLY_REJECT_ENABLED
      GENERATE_EARLY_REJECT_QUEUE_DEPTH_GTE
      GENERATE_EARLY_REJECT_IN_FLIGHT_GTE
      GENERATE_EARLY_REJECT_ROUTES   (comma-separated)
    """
    if settings is None:
        from llm_server.core.config import get_settings  # late import to avoid cycles
        settings = get_settings()

    enabled_raw = _get_nested(settings, "limits.generate_early_reject.enabled")
    rq_raw = _get_nested(settings, "limits.generate_early_reject.reject_queue_depth_gte")
    rf_raw = _get_nested(settings, "limits.generate_early_reject.reject_in_flight_gte")
    routes_raw = _get_nested(settings, "limits.generate_early_reject.routes")

    enabled = _truthy(enabled_raw) if enabled_raw is not None else GENERATE_EARLY_REJECT_DEFAULTS.enabled
    reject_queue_depth_gte = _as_int(rq_raw, GENERATE_EARLY_REJECT_DEFAULTS.reject_queue_depth_gte)
    reject_in_flight_gte = _as_int(rf_raw, GENERATE_EARLY_REJECT_DEFAULTS.reject_in_flight_gte)

    routes: list[str] = []
    if isinstance(routes_raw, str):
        routes_raw = [routes_raw]
    if isinstance(routes_raw, (list, tuple)):
        for r in routes_raw:
            if isinstance(r, str) and r.strip():
                routes.append(r.strip())

    if not routes:
        routes = list(GENERATE_EARLY_REJECT_DEFAULTS.routes)

    # env overrides
    v = _env("GENERATE_EARLY_REJECT_ENABLED")
    if v is not None:
        enabled = _truthy(v)

    v = _env("GENERATE_EARLY_REJECT_QUEUE_DEPTH_GTE")
    if v is not None:
        reject_queue_depth_gte = _as_int(v, reject_queue_depth_gte)

    v = _env("GENERATE_EARLY_REJECT_IN_FLIGHT_GTE")
    if v is not None:
        reject_in_flight_gte = _as_int(v, reject_in_flight_gte)

    v = _env("GENERATE_EARLY_REJECT_ROUTES")
    if v is not None:
        parts = [p.strip() for p in v.split(",")]
        routes = [p for p in parts if p]

    # safety clamps
    if reject_queue_depth_gte < 0:
        reject_queue_depth_gte = 0
    if reject_in_flight_gte < 0:
        reject_in_flight_gte = 0

    return GenerateEarlyRejectConfig(
        enabled=bool(enabled),
        reject_queue_depth_gte=int(reject_queue_depth_gte),
        reject_in_flight_gte=int(reject_in_flight_gte),
        routes=tuple(routes),
    )