# server/src/llm_server/core/logging.py
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from llm_server.core.time import request_latency_ms


# -----------------------------
# JSON log formatter
# -----------------------------

class JsonFormatter(logging.Formatter):
    """
    Minimal JSON formatter.

    Standard fields:
      - ts, level, logger, message

    Plus selected extras if present on the LogRecord:
      - request_id, method, path, status_code, latency_ms, client_ip
      - route, model_id, api_key_id, api_key_role
      - error_type, error_message
      - cached
    """

    _EXTRA_KEYS = [
        "request_id",
        "method",
        "path",
        "status_code",
        "latency_ms",
        "client_ip",
        "route",
        "model_id",
        "api_key_id",
        "api_key_role",
        "error_type",
        "error_message",
        "cached",
    ]

    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key in self._EXTRA_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                base[key] = value

        # If there is exception info, include a short form
        if record.exc_info:
            base.setdefault("error_type", record.exc_info[0].__name__)
            base.setdefault("error_message", str(record.exc_info[1]))

        return json.dumps(base, default=str)


# -----------------------------
# Request logging middleware
# -----------------------------

access_logger = logging.getLogger("llm_server.access")
error_logger = logging.getLogger("llm_server.error")


def _best_request_id(request: Request) -> Optional[str]:
    """
    Canonical request id lookup.

    MUST be set upstream by RequestContextMiddleware.
    We do not generate IDs here.
    """
    try:
        rid = getattr(getattr(request, "state", None), "request_id", None)
        if isinstance(rid, str) and rid.strip():
            return rid.strip()
    except Exception:
        pass
    return None


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Per-request structured logging.

    Assumes:
      - request.state.start_ts is set upstream
      - request.state.request_id is set upstream (from header or generated once)

    This middleware:
      - Logs a structured "request" record on success
      - Logs a structured "request_error" record on unhandled exceptions
      - Propagates X-Request-ID response header (canonical casing)
      - NEVER generates or overrides request_id
      - Uses canonical request latency from core/time.py (start_ts baseline)
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = _best_request_id(request)

        client_ip: Optional[str] = None
        if request.client:
            client_ip = request.client.host

        try:
            response = await call_next(request)
        except Exception:
            latency_ms = request_latency_ms(request)

            error_logger.exception(
                "request_error",
                extra={
                    "request_id": rid,
                    "method": request.method,
                    "path": request.url.path,
                    "client_ip": client_ip,
                    "latency_ms": latency_ms,
                    "model_id": getattr(request.state, "model_id", None),
                    "cached": getattr(request.state, "cached", None),
                },
            )
            raise

        latency_ms = request_latency_ms(request)

        # Canonical casing: X-Request-ID
        if rid and "X-Request-ID" not in response.headers:
            response.headers["X-Request-ID"] = rid

        extra: Dict[str, Any] = {
            "request_id": rid,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "client_ip": client_ip,
            "latency_ms": latency_ms,
        }

        model_id = getattr(request.state, "model_id", None)
        cached = getattr(request.state, "cached", None)
        route = getattr(request.state, "route", None)

        if isinstance(route, str) and route:
            extra["route"] = route
        if model_id is not None:
            extra["model_id"] = model_id
        if cached is not None:
            extra["cached"] = cached

        access_logger.info("request", extra=extra)
        return response


# -----------------------------
# Setup
# -----------------------------

def _configure_root_logging() -> None:
    """
    Configure root + uvicorn loggers to use JSON formatting.

    Called once from main.create_app().
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    # Root logger
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)

    # Uvicorn loggers
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)


def setup(app: FastAPI) -> None:
    """
    Called from main.create_app().

    - Configures logging
    - Adds RequestLoggingMiddleware

    IMPORTANT:
      - RequestContextMiddleware must be added LAST in main.py so it runs FIRST.
      - This middleware assumes request.state.request_id and start_ts are already set.
    """
    _configure_root_logging()
    app.add_middleware(RequestLoggingMiddleware)