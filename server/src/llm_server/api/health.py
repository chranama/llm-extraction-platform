# server/src/llm_server/api/health.py
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.core.config import get_settings
from llm_server.core.redis import get_redis_from_request
from llm_server.db.session import get_session
from llm_server.services.llm_registry import MultiModelManager

logger = logging.getLogger("llm_server.api.health")
router = APIRouter(tags=["health"])


# -----------------------------------------------------------------------------
# Settings + basic status helpers
# -----------------------------------------------------------------------------


def _settings(request: Request):
    return getattr(request.app.state, "settings", None) or get_settings()


def _effective_model_load_mode(request: Request) -> str:
    mode = getattr(request.app.state, "model_load_mode", None)
    if isinstance(mode, str) and mode.strip():
        return mode.strip().lower()
    return str(getattr(_settings(request), "model_load_mode", "lazy")).strip().lower()


def _llm_state(llm) -> str:
    if llm is None:
        return "not initialized"

    fn = getattr(llm, "is_loaded", None)
    if callable(fn):
        try:
            return "loaded" if fn() else "not loaded"
        except Exception:
            return "unknown"

    for attr in ("loaded", "is_ready", "ready"):
        if hasattr(llm, attr):
            try:
                return "loaded" if bool(getattr(llm, attr)) else "not loaded"
            except Exception:
                return "unknown"

    for attr in ("model", "_model", "pipeline", "_pipeline"):
        if hasattr(llm, attr):
            try:
                return "loaded" if getattr(llm, attr) is not None else "not loaded"
            except Exception:
                return "unknown"

    return "unknown"


def _model_required_for_readyz(request: Request) -> bool:
    s = _settings(request)
    return bool(getattr(s, "require_model_ready", False))

def _model_readiness_mode(request: Request) -> str:
    s = _settings(request)
    v = getattr(s, "model_readiness_mode", None)
    if isinstance(v, str) and v.strip():
        return v.strip().lower()
    return "generate"


# -----------------------------------------------------------------------------
# Infra checks
# -----------------------------------------------------------------------------


async def _db_check(session: AsyncSession) -> Tuple[bool, str]:
    try:
        await session.execute(text("SELECT 1"))
        return True, "ok"
    except Exception:
        logger.exception("DB check failed")
        return False, "error"


async def _redis_check(request: Request) -> Tuple[bool, str]:
    s = _settings(request)
    if not bool(getattr(s, "redis_enabled", False)):
        return True, "disabled"

    try:
        redis = get_redis_from_request(request)
        if redis is None:
            return False, "not initialized"

        pong = await redis.ping()
        return (pong is True), ("ok" if pong is True else f"unexpected: {pong}")
    except Exception:
        logger.exception("Redis check failed")
        return False, "error"


# -----------------------------------------------------------------------------
# Llama-server dependency checks (only when backend=llamacpp is in use)
# -----------------------------------------------------------------------------


def _summarize_default_backend(request: Request) -> Dict[str, Any]:
    """
    Best-effort summary of the runtime-selected default backend without forcing loads.
    """
    llm = getattr(request.app.state, "llm", None)

    # MultiModelManager: check the actual default backend object
    if isinstance(llm, MultiModelManager):
        default_id = getattr(llm, "default_id", None)
        backend_obj = None
        try:
            if isinstance(default_id, str) and default_id in llm:
                backend_obj = llm[default_id]
        except Exception:
            backend_obj = None

        backend_name = getattr(backend_obj, "backend_name", None)
        return {
            "registry": "multimodel",
            "default_model_id": default_id,
            "backend": backend_name,
            "backend_obj": backend_obj,
        }

    # Single backend
    backend_name = getattr(llm, "backend_name", None) if llm is not None else None
    model_id = getattr(llm, "model_id", None) if llm is not None else None
    return {
        "registry": "single",
        "default_model_id": model_id,
        "backend": backend_name,
        "backend_obj": llm,
    }


def _llamacpp_dependency_check(backend_obj: Any) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Synchronous readiness check for llama-server.
    This should be fast (GET /health).

    Returns: (ok, status_str, details)
    """
    try:
        fn = getattr(backend_obj, "is_ready", None)
        if callable(fn):
            ok, details = fn()
            okb = bool(ok)
            return okb, ("ok" if okb else "not ready"), (details if isinstance(details, dict) else {"details": details})

        # Fallback: if backend has client.health()
        client = getattr(backend_obj, "_client", None)
        health_fn = getattr(client, "health", None) if client is not None else None
        if callable(health_fn):
            data = health_fn()
            okb = bool(isinstance(data, dict) and data.get("status") == "ok")
            return okb, ("ok" if okb else "not ready"), {"health": data}

        return False, "missing health check", {"reason": "llamacpp backend lacks is_ready() / client.health()"}
    except Exception as e:
        return False, "error", {"error": repr(e)}


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@router.get("/healthz")
async def healthz():
    """
    Liveness probe.
    Must always be fast and must never fail due to infra or model state.
    """
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Readiness probe.
    Determines whether the service can receive traffic.

    NEW (Phase 1):
      - If the runtime default backend is `llamacpp`, also require llama-server /health to be ok.
    """
    s = _settings(request)

    db_ok, db_status = await _db_check(session)
    redis_ok, redis_status = await _redis_check(request)

    llm = getattr(request.app.state, "llm", None)
    model_loaded = bool(getattr(request.app.state, "model_loaded", False))
    model_error = getattr(request.app.state, "model_error", None)

    mode = _effective_model_load_mode(request)
    llm_status = "disabled" if mode == "off" else _llm_state(llm)

    require_model = _model_required_for_readyz(request)

    model_ok = True
    if require_model:
        model_ok = bool(model_loaded and model_error is None)

    # --- llama-server dependency (only if selected backend is llamacpp) ---
    dep_ok = True
    dep_status = "n/a"
    dep_details: Dict[str, Any] = {}

    backend_summary = _summarize_default_backend(request)
    default_backend = backend_summary.get("backend")
    backend_obj = backend_summary.get("backend_obj")

    if default_backend == "llamacpp":
        ok, st, details = _llamacpp_dependency_check(backend_obj)
        dep_ok, dep_status, dep_details = ok, st, details

    ready = bool(db_ok and redis_ok and model_ok and dep_ok)

    payload: Dict[str, Any] = {
        "status": "ready" if ready else "not ready",
        "db": db_status,
        "redis": redis_status,
        "db_instance": getattr(s, "db_instance", "unknown"),
        "model_load_mode": mode,
        "require_model_ready": bool(require_model),
        "model_loaded": bool(model_loaded),
        "model_error": model_error,
        "llm": llm_status,
        # New: dependency truth (only meaningful when backend=llamacpp)
        "default_model_id": backend_summary.get("default_model_id"),
        "default_backend": default_backend,
        "dependency": {
            "llama_server": {
                "required": bool(default_backend == "llamacpp"),
                "status": dep_status,
                "ok": bool(dep_ok),
                **(dep_details if isinstance(dep_details, dict) else {"details": dep_details}),
            }
        },
    }
    return JSONResponse(payload, status_code=200 if ready else 503)


@router.get("/modelz")
async def modelz(request: Request):
    """
    Model-only readiness.

    For in-process backends (transformers): reflects model_loaded/model_error.
    For external backends (llamacpp/remote): reflects backend ability to generate.
    """
    s = _settings(request)

    llm = getattr(request.app.state, "llm", None)
    mode = _effective_model_load_mode(request)

    model_loaded = bool(getattr(request.app.state, "model_loaded", False))
    model_error = getattr(request.app.state, "model_error", None)
    llm_status = "disabled" if mode == "off" else _llm_state(llm)

    backend_summary = _summarize_default_backend(request)
    default_backend = backend_summary.get("backend")
    backend_obj = backend_summary.get("backend_obj")

    readiness_mode = _model_readiness_mode(request)

    # Defaults
    ready = False
    reason = None
    dep_details: Dict[str, Any] = {}

    if readiness_mode == "off":
        ready = True
        reason = "skipped (model_readiness_mode=off)"
    elif default_backend in ("llamacpp", "remote"):
        # External backend: prove readiness via probe or generate
        try:
            if readiness_mode == "probe":
                fn = getattr(backend_obj, "is_ready", None)
                if callable(fn):
                    ok, details = fn()
                    ready = bool(ok)
                    dep_details = details if isinstance(details, dict) else {"details": details}
                else:
                    ready = False
                    reason = "backend missing is_ready()"
            else:  # "generate" (default)
                fn = getattr(backend_obj, "can_generate", None)
                if callable(fn):
                    ok, details = fn()
                    ready = bool(ok)
                    dep_details = details if isinstance(details, dict) else {"details": details}
                else:
                    # fallback to health probe
                    fn2 = getattr(backend_obj, "is_ready", None)
                    if callable(fn2):
                        ok, details = fn2()
                        ready = bool(ok)
                        dep_details = details if isinstance(details, dict) else {"details": details}
                        reason = "can_generate missing; fell back to is_ready"
                    else:
                        ready = False
                        reason = "backend missing can_generate() and is_ready()"
        except Exception as e:
            ready = False
            reason = f"error: {repr(e)}"
    else:
        # In-process backend: old semantics
        ready = bool(model_loaded and model_error is None)

    payload: Dict[str, Any] = {
        "status": "ready" if ready else "not ready",
        "db_instance": getattr(s, "db_instance", "unknown"),
        "model_load_mode": mode,
        "model_readiness_mode": readiness_mode,
        # Keep legacy fields for compatibility (but interpret carefully for external)
        "model_loaded": bool(model_loaded),
        "model_error": model_error,
        "llm": llm_status,
        "default_model_id": backend_summary.get("default_model_id"),
        "default_backend": default_backend,
        "reason": reason,
        "dependency": {
            "backend": {
                "status": "ok" if ready else "not ready",
                "details": dep_details,
            }
        },
    }
    return JSONResponse(payload, status_code=200 if ready else 503)