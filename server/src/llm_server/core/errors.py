from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, Request
from fastapi import HTTPException as FastAPIHTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

import llm_server.db.session as db_session  # module import so tests can patch session wiring
from llm_server.services.inference import write_failure_log

logger = logging.getLogger("llm.errors")


# ---------------------------------------------------------------------------
# AppError
# ---------------------------------------------------------------------------

class AppError(FastAPIHTTPException):
    """
    Canonical application error.

    Always serializes into:
      { code, message, extra?, request_id? }

    By subclassing HTTPException, FastAPI always renders it,
    even in tests without custom exception handlers.
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
    return getattr(getattr(request, "state", None), "request_id", None)


def _best_effort_latency_ms(request: Request) -> Optional[float]:
    """
    Compute latency for failures using request.state.start_ts
    (expected to be set by RequestContextMiddleware).
    """
    start_ts = getattr(getattr(request, "state", None), "start_ts", None)
    try:
        if isinstance(start_ts, (int, float)) and start_ts > 0:
            return max(0.0, (time.time() - float(start_ts)) * 1000.0)
    except Exception:
        pass
    return None


def _best_route_label(request: Request) -> str:
    """
    Prefer request.state.route if handlers set it; otherwise fall back to URL path.
    """
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
    """
    Best-effort API key attribution.

    get_api_key() sets request.state.api_key; failures before auth will be empty.
    """
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
    """
    Insert an InferenceLog row for failures.

    HARD RULES:
      - MUST NOT raise
      - MUST NOT delay error response
    """
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
                latency_ms=_best_effort_latency_ms(request),
                status_code=int(status_code),
                error_code=str(error_code),
                error_stage=error_stage.strip() if isinstance(error_stage, str) and error_stage.strip() else None,
                cached=_best_cached(request),
                commit=True,
            )
    except Exception:
        # Absolute last-ditch safety: swallow everything
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
    """
    Canonical error envelope.

    Shape:
      {
        code: str,
        message: str,
        extra?: dict,
        request_id?: str
      }
    """
    payload: Dict[str, Any] = {"code": code, "message": message}

    if isinstance(extra, dict) and extra:
        payload["extra"] = extra

    rid = _request_id(request)
    if rid:
        payload["request_id"] = rid

    resp = JSONResponse(payload, status_code=status_code)
    if rid:
        resp.headers["X-Request-ID"] = rid
    return resp


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

async def handle_fastapi_http_exception(request: Request, exc: FastAPIHTTPException):
    """
    Handles fastapi.HTTPException.

    detail may be:
      - str
      - dict
      - canonical {code, message, extra}
    """
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
    """
    Handles Starlette router errors (e.g. 404).
    """
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
    """
    Handles Pydantic / request validation errors (422).
    """
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
    """
    Handles explicit application / business logic errors.
    """
    logger.info("app_error code=%s msg=%s", exc.code, exc.message)

    extra = exc.extra if isinstance(exc.extra, dict) else None

    # Stage resolution priority:
    #   1) extra.stage
    #   2) request.state.error_stage
    #   3) "app_error"
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
    """
    Final safety net.

    Never leak internals, but DO emit stable telemetry signals.
    """
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


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def setup(app: FastAPI) -> None:
    """
    Register all exception handlers.
    Call once during app startup.
    """
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(FastAPIHTTPException, handle_fastapi_http_exception)
    app.add_exception_handler(StarletteHTTPException, handle_starlette_http_exception)
    app.add_exception_handler(AppError, handle_app_error)
    app.add_exception_handler(Exception, handle_unhandled_exception)