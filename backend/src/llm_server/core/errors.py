# src/llm_server/core/errors.py
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import HTTPException as FastAPIHTTPException

logger = logging.getLogger("llm.errors")


class AppError(Exception):
    """
    Raise this for application/business logic errors with a stable code.
    Example:
        raise AppError(code="quota_exhausted", message="Monthly quota exhausted", status_code=402)
    """
    def __init__(self, *, code: str, message: str, status_code: int = 400, extra: Optional[Dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.extra = extra or {}
        super().__init__(message)


def _request_id(request: Request) -> Optional[str]:
    return getattr(getattr(request, "state", None), "request_id", None)


def _to_json_error(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    extra: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    payload: Dict[str, Any] = {
        "error": {"code": code, "message": message},
        "request_id": _request_id(request),
    }
    if extra:
        payload["error"].update(extra)

    resp = JSONResponse(payload, status_code=status_code)

    rid = _request_id(request)
    if rid:
        resp.headers["X-Request-ID"] = rid

    return resp


async def handle_fastapi_http_exception(request: Request, exc: FastAPIHTTPException):
    """
    Catches `fastapi.HTTPException`.
    `detail` may be a str or a dict we previously raised.
    """
    detail: Union[str, Dict[str, Any]] = exc.detail
    if isinstance(detail, dict):
        # Prefer our own structure if caller already provided it
        code = detail.get("code", "http_error")
        message = detail.get("message", "HTTP error")
        extra = {k: v for k, v in detail.items() if k not in {"code", "message"}}
        return _to_json_error(request, status_code=exc.status_code, code=code, message=message, extra=extra)
    # Otherwise wrap it
    return _to_json_error(request, status_code=exc.status_code, code="http_error", message=str(detail))


async def handle_starlette_http_exception(request: Request, exc: StarletteHTTPException):
    """
    Catches Starlette's HTTPException (e.g., 404 from router not found).
    """
    if exc.status_code == 404:
        return _to_json_error(request, status_code=404, code="not_found", message="Route not found")
    return _to_json_error(request, status_code=exc.status_code, code="http_error", message=str(exc.detail))


async def handle_validation_error(request: Request, exc: RequestValidationError):
    """
    Catches Pydantic/validation errors (422).
    """
    # Log full error for server-side debugging
    logger.debug("validation_error: %s", exc.errors())
    return _to_json_error(
        request,
        status_code=422,
        code="validation_error",
        message="Request validation failed",
        extra={"fields": exc.errors()},
    )


async def handle_app_error(request: Request, exc: AppError):
    """
    Our explicit app/business logic errors.
    """
    # Log at INFO (expected client-side issues) or WARNING if you prefer
    logger.info("app_error code=%s msg=%s", exc.code, exc.message)
    return _to_json_error(
        request,
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        extra=exc.extra,
    )


async def handle_unhandled_exception(request: Request, exc: Exception):
    """
    Final safety net: avoid leaking internals.
    """
    logger.exception("unhandled_exception path=%s", request.url.path, exc_info=exc)
    return _to_json_error(
        request,
        status_code=500,
        code="internal_error",
        message="An unexpected error occurred",
    )


def setup(app: FastAPI) -> None:
    """
    Register all handlers on the FastAPI app.
    Call from main.py early in setup.
    """
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(FastAPIHTTPException, handle_fastapi_http_exception)
    app.add_exception_handler(StarletteHTTPException, handle_starlette_http_exception)
    app.add_exception_handler(AppError, handle_app_error)
    app.add_exception_handler(Exception, handle_unhandled_exception)