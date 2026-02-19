# src/llm_server/core/limits.py
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Final, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from llm_server.core.config import get_settings
from llm_server.core.errors import AppError
from llm_server.core.metrics import LLM_GUARD_TRIPS

logger = logging.getLogger("llm_server.limits")

HEAVY_PREFIXES: Final = ("/v1/generate", "/v1/extract")


# ----------------------------
# Concurrency guard (Phase 0)
# ----------------------------

def _read_max_concurrency() -> int:
    """
    Read max concurrency from settings once at startup.

    Notes:
    - Phase 0 behavior: queue (await); do not reject.
    - Value is effectively frozen at process startup (middleware init).
    """
    s = get_settings()
    try:
        n = int(getattr(s, "max_concurrent_requests", 2))
    except Exception:
        n = 2
    return max(1, min(64, n))


# ----------------------------
# Memory guard (Phase 0)
# ----------------------------

def _read_container_limit_bytes(settings) -> Optional[int]:
    """
    Best-effort container memory limit.
    For Phase 0, prefer explicit env/config (CONTAINER_MEMORY_BYTES).
    """
    v = getattr(settings, "container_memory_bytes", None)
    if v is None:
        v = os.getenv("CONTAINER_MEMORY_BYTES")
    if v is None:
        return None
    try:
        n = int(str(v).strip())
        return n if n > 0 else None
    except Exception:
        return None


def _read_rss_bytes() -> Optional[int]:
    """
    Best-effort process RSS without adding hard dependency for Phase 0.

    - If psutil is available, use it.
    - Otherwise return None (guard becomes a no-op).
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return None

    try:
        p = psutil.Process()
        return int(p.memory_info().rss)
    except Exception:
        return None


def _read_mem_guard_enabled(settings) -> bool:
    v = getattr(settings, "mem_guard_enabled", False)
    try:
        return bool(v)
    except Exception:
        return False


def _read_mem_guard_rss_pct(settings) -> float:
    v = getattr(settings, "mem_guard_rss_pct", 0.85)
    try:
        f = float(v)
    except Exception:
        f = 0.85
    # clamp sanity
    if f < 0.10:
        return 0.10
    if f > 0.99:
        return 0.99
    return f


def _maybe_trip_memory_guard(request: Request, settings) -> None:
    """
    If enabled and we can compute RSS% of container limit, reject with 503
    before we hit the OOM killer.

    IMPORTANT: keep logic fast and best-effort. If anything fails, do nothing.
    """
    if not _read_mem_guard_enabled(settings):
        return

    limit = _read_container_limit_bytes(settings)
    if not limit:
        return

    rss = _read_rss_bytes()
    if rss is None:
        return

    pct = rss / float(limit)
    threshold = _read_mem_guard_rss_pct(settings)

    if pct < threshold:
        return

    # Emit metric *before* raising so we capture even if handler fails later.
    try:
        route = getattr(getattr(request, "state", None), "route", None) or request.url.path
        LLM_GUARD_TRIPS.labels(kind="mem_rss", route=str(route)).inc()
    except Exception:
        pass

    rid = getattr(getattr(request, "state", None), "request_id", None)
    logger.warning(
        "memory_guard_trip",
        extra={
            "request_id": rid,
            "path": request.url.path,
            "rss_bytes": int(rss),
            "limit_bytes": int(limit),
            "rss_pct": round(pct, 4),
            "threshold_pct": round(threshold, 4),
        },
    )

    raise AppError(
        code="server_overloaded",
        message="Service temporarily unavailable (memory guard)",
        status_code=503,
        extra={
            "stage": "mem_guard",
            "path": request.url.path,
            "rss_bytes": int(rss),
            "limit_bytes": int(limit),
            "rss_pct": round(pct, 4),
            "threshold_pct": round(threshold, 4),
        },
    )


class _GuardMiddleware(BaseHTTPMiddleware):
    """
    Phase 0 guards:
      1) Concurrency: queue heavy requests (POST on HEAVY_PREFIXES)
      2) Memory: reject with 503 when RSS approaches container limit
    """

    def __init__(self, app):
        super().__init__(app)
        self._settings = get_settings()
        self._max_concurrent = _read_max_concurrency()
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

        logger.info(
            "limits_init",
            extra={
                "max_concurrent_requests": self._max_concurrent,
                "mem_guard_enabled": _read_mem_guard_enabled(self._settings),
                "mem_guard_rss_pct": _read_mem_guard_rss_pct(self._settings),
                "container_memory_bytes": _read_container_limit_bytes(self._settings),
            },
        )

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if request.method == "POST" and path.startswith(HEAVY_PREFIXES):
            # Trip memory guard *before* queueing so we shed early under pressure.
            _maybe_trip_memory_guard(request, self._settings)

            t0 = time.perf_counter()
            async with self._semaphore:
                wait_ms = (time.perf_counter() - t0) * 1000.0

                if wait_ms > 5:
                    rid = getattr(getattr(request, "state", None), "request_id", None)
                    logger.info(
                        "concurrency_wait",
                        extra={
                            "request_id": rid,
                            "path": path,
                            "wait_ms": round(wait_ms, 2),
                            "max_concurrent": self._max_concurrent,
                        },
                    )

                # Check again right before the heavy work begins.
                _maybe_trip_memory_guard(request, self._settings)

                return await call_next(request)

        return await call_next(request)


def setup(app) -> None:
    """Install phase-0 guard middleware."""
    app.add_middleware(_GuardMiddleware)