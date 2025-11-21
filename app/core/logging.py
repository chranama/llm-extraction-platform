# app/core/logging.py
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


# We DO NOT use "uvicorn.access" to avoid its strict tuple-based formatter.
ACCESS_LOGGER_NAME = "llm.access"
DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"


def _configure_logger() -> logging.Logger:
    """
    Create a dedicated, idempotent logger for access logs.
    - Simple formatter (no tuple unpacking).
    - Does not propagate to root to avoid double-logging.
    """
    logger = logging.getLogger(ACCESS_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


access_logger = _configure_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Adds a request_id to request.state, logs start/end with method, path,
    status_code, client, and latency, and sets X-Request-ID on the response.
    """
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        method = request.method
        path = request.url.path
        client: Optional[str] = request.client.host if request.client else None

        access_logger.info(f"[{request_id}] started method={method} path={path} client={client}")
        t0 = time.perf_counter()
        status_code: Optional[int] = None

        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            # Log the exception and re-raise so default handlers still apply
            access_logger.exception(f"[{request_id}] unhandled_exception path={path}: {exc}")
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            # Try to set X-Request-ID if we have a response object
            try:
                # When an exception occurs before response creation, this may fail; ignore.
                response.headers["X-Request-ID"] = request_id  # type: ignore[name-defined]
            except Exception:
                pass

            # Log completion line
            if status_code is None:
                # In case of exception before response is produced
                access_logger.info(
                    f"[{request_id}] finished method={method} path={path} status=ERROR "
                    f"latency_ms={elapsed_ms:.2f} client={client}"
                )
            else:
                content_length = None
                try:
                    content_length = response.headers.get("content-length")  # type: ignore[name-defined]
                except Exception:
                    pass

                access_logger.info(
                    f"[{request_id}] finished method={method} path={path} status={status_code} "
                    f"latency_ms={elapsed_ms:.2f} client={client} content_length={content_length}"
                )


def setup(app):
    """
    Attach the RequestLoggingMiddleware. Idempotent if called once at startup.
    """
    app.add_middleware(RequestLoggingMiddleware)