# server/src/llm_server/core/errors.py
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, Request
from fastapi import HTTPException as FastAPIHTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

import llm_server.db.session as db_session  # module import so tests can patch session wiring
from llm_server.core.time import request_latency_ms
from llm_server.services.llm_runtime.inference import write_failure_log

logger = logging.getLogger("llm.errors")


# ---------------------------------------------------------------------------
# AppError
# ---------------------------------------------------------------------------

class AppError(FastAPIHTTPException):
    """
    Canonical application error.

    Always serializes into:
      { code, message, extra?, request_id? }
    """

    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.extra = extra

        detail: Dict[str, Any] = {"code": code, "message": message}
        if isinstance(extra, dict) and extra:
            detail["extra"] = extra

        super().__init__(status_code=status_code, detail=detail)


# ---------------------------------------------------------------------------
# Best-effort request metadata helpers
# ---------------------------------------------------------------------------

def _request_id(request: Request) -> Optional[str]:
    rid = getattr(getattr(request, "state", None), "request_id", None)
    return rid if isinstance(rid, str) and rid.strip() else None


def _best_route_label(request: Request) -> str:
    route = getattr(getattr(request, "state", None), "route", None)
    if isinstance(route, str) and route.strip():
        return route.strip()
    return request.url.path


def _best_model_id(request: Request) -> str:
    mid = getattr(getattr(request, "state", None), "model_id", None)
    if isinstance(mid, str) and mid.strip():
        return mid.strip()
    return "unknown"


def _best_cached(request: Request) -> Optional[bool]:
    c = getattr(getattr(request, "state", None), "cached", None)
    return c if isinstance(c, bool) else None


def _best_client_host(request: Request) -> Optional[str]:
    try:
        return request.client.host if request.client else None
    except Exception:
        return None


def _best_api_key_value(request: Request) -> str:
    v = getattr(getattr(request, "state", None), "api_key", None)
    if isinstance(v, str) and v.strip():
        return v.strip()
    return ""


# ---------------------------------------------------------------------------
# Failure logging (hard rules: never raise, never block)
# ---------------------------------------------------------------------------

async def _best_effort_log_failure(
    request: Request,
    *,
    status_code: int,
    error_code: str,
    error_stage: Optional[str],
) -> None:
    try:
        SessionLocal = db_session.get_sessionmaker()
        async with SessionLocal() as session:
            await write_failure_log(
                session,
                api_key=_best_api_key_value(request),
                request_id=_request_id(request),
                route=_best_route_label(request),
                client_host=_best_client_host(request),
                model_id=_best_model_id(request),
                latency_ms=request_latency_ms(request),
                status_code=int(status_code),
                error_code=str(error_code),
                error_stage=error_stage.strip() if isinstance(error_stage, str) and error_stage.strip() else None,
                cached=_best_cached(request),
                commit=True,
            )
    except Exception:
        return


# ---------------------------------------------------------------------------
# Canonical JSON error response
# ---------------------------------------------------------------------------

def _to_json_error(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    extra: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    payload: Dict[str, Any] = {"code": code, "message": message}

    if isinstance(extra, dict) and extra:
        payload["extra"] = extra

    rid = _request_id(request)
    if rid:
        payload["request_id"] = rid

    resp = JSONResponse(payload, status_code=status_code)
    if rid:
        # Canonical casing: X-Request-ID
        resp.headers["X-Request-ID"] = rid
    return resp


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

async def handle_fastapi_http_exception(request: Request, exc: FastAPIHTTPException):
    detail: Union[str, Dict[str, Any]] = exc.detail

    if isinstance(detail, dict):
        code = str(detail.get("code", "http_error"))
        message = str(detail.get("message", "HTTP error"))

        extra = detail.get("extra")
        if not isinstance(extra, dict):
            extra = {k: v for k, v in detail.items() if k not in {"code", "message", "extra"}}

        await _best_effort_log_failure(
            request,
            status_code=exc.status_code,
            error_code=code,
            error_stage="http_exception",
        )

        return _to_json_error(
            request,
            status_code=exc.status_code,
            code=code,
            message=message,
            extra=extra or None,
        )

    await _best_effort_log_failure(
        request,
        status_code=exc.status_code,
        error_code="http_error",
        error_stage="http_exception",
    )

    return _to_json_error(
        request,
        status_code=exc.status_code,
        code="http_error",
        message=str(detail),
    )


async def handle_starlette_http_exception(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        await _best_effort_log_failure(
            request,
            status_code=404,
            error_code="not_found",
            error_stage="router",
        )
        return _to_json_error(
            request,
            status_code=404,
            code="not_found",
            message="Route not found",
        )

    await _best_effort_log_failure(
        request,
        status_code=exc.status_code,
        error_code="http_error",
        error_stage="router",
    )

    return _to_json_error(
        request,
        status_code=exc.status_code,
        code="http_error",
        message=str(exc.detail),
    )


async def handle_validation_error(request: Request, exc: RequestValidationError):
    logger.debug("validation_error: %s", exc.errors())

    await _best_effort_log_failure(
        request,
        status_code=422,
        error_code="validation_error",
        error_stage="request_validation",
    )

    return _to_json_error(
        request,
        status_code=422,
        code="validation_error",
        message="Request validation failed",
        extra={
            "stage": "request_validation",
            "fields": exc.errors(),
        },
    )


async def handle_app_error(request: Request, exc: AppError):
    logger.info("app_error code=%s msg=%s", exc.code, exc.message)

    extra = exc.extra if isinstance(exc.extra, dict) else None

    stage: Optional[str] = None
    if isinstance(extra, dict):
        st = extra.get("stage")
        if isinstance(st, str) and st.strip():
            stage = st.strip()

    if stage is None:
        st2 = getattr(getattr(request, "state", None), "error_stage", None)
        if isinstance(st2, str) and st2.strip():
            stage = st2.strip()

    if stage is None:
        stage = "app_error"

    await _best_effort_log_failure(
        request,
        status_code=exc.status_code,
        error_code=exc.code,
        error_stage=stage,
    )

    return _to_json_error(
        request,
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        extra=extra,
    )


async def handle_unhandled_exception(request: Request, exc: Exception):
    logger.exception("unhandled_exception path=%s", request.url.path, exc_info=exc)

    safe_extra: Dict[str, Any] = {
        "stage": "unhandled_exception",
        "path": request.url.path,
        "method": request.method,
        "exc_type": type(exc).__name__,
    }

    st = getattr(getattr(request, "state", None), "error_stage", None)
    if isinstance(st, str) and st.strip():
        safe_extra["stage"] = st.strip()

    await _best_effort_log_failure(
        request,
        status_code=500,
        error_code="internal_error",
        error_stage=str(safe_extra.get("stage")),
    )

    return _to_json_error(
        request,
        status_code=500,
        code="internal_error",
        message="An unexpected error occurred",
        extra=safe_extra,
    )


def setup(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(FastAPIHTTPException, handle_fastapi_http_exception)
    app.add_exception_handler(StarletteHTTPException, handle_starlette_http_exception)
    app.add_exception_handler(AppError, handle_app_error)
    app.add_exception_handler(Exception, handle_unhandled_exception)