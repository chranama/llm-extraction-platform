# server/src/llm_server/api/health.py
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.db.session import get_session
from llm_server.services.api_deps.core.model_load_mode import effective_model_load_mode_from_request
from llm_server.services.api_deps.health.infra import db_check, redis_check, settings_from_request
from llm_server.services.api_deps.health.readiness import model_ready_check_async
from llm_server.services.api_deps.health.snapshots import (
    assessed_gate_snapshot,
    deployment_metadata_snapshot,
    generate_gate_snapshot,
    policy_summary,
)
from llm_server.services.api_deps.routing.models import (
    model_flags_from_app_state,
    per_model_readiness_mode,
    resolve_default_model_id_and_backend_obj,
)

logger = logging.getLogger("llm_server.api.health")
router = APIRouter(tags=["health"])


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
    s = settings_from_request(request)
    return bool(getattr(s, "require_model_ready", False))


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

    Includes best-effort snapshots:
      - policy snapshot
      - generate gate snapshot
      - assessed gate snapshot (sourced from models.yaml)
      - deployment metadata snapshot
    """
    s = settings_from_request(request)

    db_ok, db_status = await db_check(session)
    redis_ok, redis_status = await redis_check(request)

    llm = getattr(request.app.state, "llm", None)
    mode = effective_model_load_mode_from_request(request)
    llm_status = "disabled" if mode == "off" else _llm_state(llm)

    model_loaded, model_error, loaded_model_id, runtime_default = model_flags_from_app_state(request)

    runtime_loader_present = bool(getattr(request.app.state, "runtime_model_loader", None) is not None)
    models_config_present = bool(getattr(request.app.state, "models_config", None) is not None)
    registry_present = bool(getattr(request.app.state, "llm", None) is not None)

    default_model_id, default_backend, _ = resolve_default_model_id_and_backend_obj(request)
    readiness_mode = per_model_readiness_mode(request, model_id=default_model_id)

    require_model = _model_required_for_readyz(request)

    model_ok = True
    model_status = "skipped"
    model_details: Dict[str, Any] = {}

    if require_model and mode != "off":
        ok, st, details = await model_ready_check_async(request)
        model_ok = bool(ok)
        model_status = st
        model_details = details if isinstance(details, dict) else {"details": details}

    ready = bool(db_ok and redis_ok and model_ok)

    payload: Dict[str, Any] = {
        "status": "ready" if ready else "not ready",
        "db": db_status,
        "redis": redis_status,
        "db_instance": getattr(s, "db_instance", "unknown"),
        "model_load_mode": mode,
        "model_readiness_mode": readiness_mode,
        "require_model_ready": bool(require_model),
        # state (structured -> legacy mirrored)
        "model_loaded": bool(model_loaded),
        "model_error": model_error,
        "loaded_model_id": loaded_model_id,
        "llm": llm_status,
        # runtime visibility
        "runtime": {
            "runtime_model_loader_present": runtime_loader_present,
            "models_config_loaded": models_config_present,
            "registry_initialized": registry_present,
            "runtime_default_model_id": runtime_default,
        },
        # backend selection
        "default_model_id": default_model_id,
        "default_backend": default_backend,
        # unified model readiness when required
        "model": {
            "required": bool(require_model and mode != "off"),
            "status": model_status,
            "ok": bool(model_ok),
            "details": model_details,
        },
        # snapshots (best-effort)
        "policy": policy_summary(request),
        "generate_gate": generate_gate_snapshot(),
        "assessed_gate": assessed_gate_snapshot(request),
        "deployment": deployment_metadata_snapshot(request),
    }
    return JSONResponse(payload, status_code=200 if ready else 503)


@router.get("/modelz")
async def modelz(request: Request):
    """
    Model-only readiness.

    Includes best-effort snapshots:
      - policy snapshot
      - generate gate snapshot
      - assessed gate snapshot (sourced from models.yaml)
      - deployment metadata snapshot
    """
    s = settings_from_request(request)

    llm = getattr(request.app.state, "llm", None)
    mode = effective_model_load_mode_from_request(request)
    llm_status = "disabled" if mode == "off" else _llm_state(llm)

    model_loaded, model_error, loaded_model_id, runtime_default = model_flags_from_app_state(request)

    default_model_id, default_backend, _ = resolve_default_model_id_and_backend_obj(request)
    readiness_mode = per_model_readiness_mode(request, model_id=default_model_id)

    ok, st, details = await model_ready_check_async(request)

    payload: Dict[str, Any] = {
        "status": "ready" if ok else "not ready",
        "db_instance": getattr(s, "db_instance", "unknown"),
        "model_load_mode": mode,
        "model_readiness_mode": readiness_mode,
        # state (structured -> legacy mirrored)
        "model_loaded": bool(model_loaded),
        "model_error": model_error,
        "loaded_model_id": loaded_model_id,
        "runtime_default_model_id": runtime_default,
        "llm": llm_status,
        "default_model_id": default_model_id,
        "default_backend": default_backend,
        "reason": None if ok else st,
        "dependency": {
            "backend": {
                "status": "ok" if ok else "not ready",
                "details": details if isinstance(details, dict) else {"details": details},
            }
        },
        # snapshots (best-effort)
        "policy": policy_summary(request),
        "generate_gate": generate_gate_snapshot(),
        "assessed_gate": assessed_gate_snapshot(request),
        "deployment": deployment_metadata_snapshot(request),
    }
    return JSONResponse(payload, status_code=200 if ok else 503)