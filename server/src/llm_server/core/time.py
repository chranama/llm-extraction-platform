# server/src/llm_server/core/time.py
from __future__ import annotations

import time
from typing import Optional

from starlette.requests import Request


# ---------------------------------------------------------------------------
# Canonical request timing (wall-clock, request lifecycle)
# ---------------------------------------------------------------------------

def request_start_ts(request: Request) -> Optional[float]:
    """
    Return the canonical request start timestamp (wall clock seconds since epoch).

    This MUST match what RequestContextMiddleware sets:
        request.state.start_ts = time.time()

    Returns None if unavailable.
    """
    try:
        ts = getattr(getattr(request, "state", None), "start_ts", None)
        if isinstance(ts, (int, float)) and ts > 0:
            return float(ts)
    except Exception:
        pass
    return None


def request_latency_ms(request: Request) -> Optional[float]:
    """
    Canonical request latency in milliseconds.

    Definition:
        time from RequestContextMiddleware arrival
        -> to the moment this function is called.

    Uses wall-clock time (time.time()) because start_ts was set with time.time().

    Returns:
        float latency_ms >= 0.0
        or None if start_ts is unavailable.
    """
    start_ts = request_start_ts(request)
    if start_ts is None:
        return None

    try:
        elapsed = (time.time() - start_ts) * 1000.0
        return max(0.0, float(elapsed))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Monotonic helpers (internal spans only; NOT request SLO baseline)
# ---------------------------------------------------------------------------

def monotonic_now() -> float:
    """
    High-resolution monotonic clock (seconds).

    Use for:
      - internal spans (queue wait, execution time, backend call timing)
      - metrics timing (Prometheus histograms)

    DO NOT mix with request.state.start_ts for SLO math.
    """
    return time.perf_counter()


def monotonic_elapsed_ms(start: float) -> float:
    """
    Elapsed milliseconds from a monotonic start timestamp.
    """
    try:
        return max(0.0, (time.perf_counter() - float(start)) * 1000.0)
    except Exception:
        return 0.0