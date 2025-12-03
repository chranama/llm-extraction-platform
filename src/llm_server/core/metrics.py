# src/llm_server/core/metrics.py
from __future__ import annotations

from typing import Callable

from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, Gauge
from prometheus_client import REGISTRY
from starlette.middleware.base import BaseHTTPMiddleware

from llm_server.core.config import settings


# -----------------------------------------
# LLM Token Counters
# -----------------------------------------
LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Total tokens processed",
    ["direction", "model_id"],
)

# -----------------------------------------
# Request Metrics (existing)
# -----------------------------------------
REQUEST_LATENCY = Histogram(
    "llm_api_request_latency_seconds",
    "Request latency in seconds",
    ["route", "model_id", "cached"],
)

REQUEST_COUNT = Counter(
    "llm_api_request_total",
    "Total requests",
    ["route", "model_id", "cached"],
)

# -----------------------------------------
# NEW Redis Metrics
# -----------------------------------------

# Hits: Redis returned a cached value
LLM_REDIS_HITS = Counter(
    "llm_redis_hits_total",
    "Redis cache hits",
    ["model_id", "kind"],   # kind = "single" or "batch"
)

# Misses: Redis returned nothing (None)
LLM_REDIS_MISSES = Counter(
    "llm_redis_misses_total",
    "Redis cache misses",
    ["model_id", "kind"],
)

# Latency to call Redis GET operations
LLM_REDIS_LATENCY = Histogram(
    "llm_redis_latency_seconds",
    "Latency of Redis cache GET operations",
    ["model_id", "kind"],
)

# Redis enabled status (0 or 1)
LLM_REDIS_ENABLED = Gauge(
    "llm_redis_enabled",
    "Whether Redis caching is enabled (1=yes, 0=no)",
)
# Set this at import time
LLM_REDIS_ENABLED.set(1 if settings.redis_enabled else 0)


# -----------------------------------------
# Middleware to record request metrics
# -----------------------------------------
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        route = getattr(request.state, "route", "unknown")
        model_id = getattr(request.state, "model_id", "unknown")
        cached = getattr(request.state, "cached", False)

        # Convert boolean → "true"/"false"
        cached_label = "true" if cached else "false"

        with REQUEST_LATENCY.labels(
            route=route,
            model_id=model_id,
            cached=cached_label,
        ).time():
            response = await call_next(request)

        REQUEST_COUNT.labels(
            route=route,
            model_id=model_id,
            cached=cached_label,
        ).inc()

        return response


# -----------------------------------------
# Setup entry point for main.py
# -----------------------------------------
def setup(app: FastAPI):
    """
    Register Prometheus collectors and install middleware.
    """
    # Prevent duplicate registration during reloads
    for collector in [
        LLM_TOKENS,
        REQUEST_LATENCY,
        REQUEST_COUNT,
        LLM_REDIS_HITS,
        LLM_REDIS_MISSES,
        LLM_REDIS_LATENCY,
        LLM_REDIS_ENABLED,
    ]:
        try:
            REGISTRY.register(collector)
        except ValueError:
            pass

    app.add_middleware(MetricsMiddleware)