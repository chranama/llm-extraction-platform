from __future__ import annotations

from typing import Any

from fastapi import Request

from llm_server.core.errors import AppError
from llm_server.services.api_deps.core.model_load_mode import (
    effective_model_load_mode_from_request,
)
from llm_server.services.api_deps.core.settings import settings_from_request
from llm_server.services.api_deps.enforcement.capabilities import (
    deployment_capabilities,
    effective_capabilities,
)
from llm_server.services.api_deps.routing.models import allowed_model_ids
from llm_server.services.llm_runtime.llm_registry import MultiModelManager


def _unique_ids(default_id: str, allowed: list[str]) -> list[str]:
    out: list[str] = []
    if isinstance(default_id, str) and default_id.strip():
        out.append(default_id.strip())
    for mid in allowed:
        s = str(mid).strip()
        if s and s not in out:
            out.append(s)
    return out


def _single_backend_payload(
    *, request: Request, llm: Any, default_model: str, load_mode: str
) -> list[dict[str, Any]]:
    model_id = str(getattr(llm, "model_id", "") or default_model or "").strip() or "default"
    backend_name = getattr(llm, "backend_name", None)
    loaded = bool(getattr(request.app.state, "model_loaded", False)) if load_mode == "eager" else None
    return [
        {
            "id": model_id,
            "default": True,
            "backend": str(backend_name) if isinstance(backend_name, str) else type(llm).__name__,
            "capabilities": effective_capabilities(model_id, request=request),
            "load_mode": load_mode,
            "loaded": loaded,
        }
    ]


def list_models_payload(*, request: Request, llm: Any) -> dict[str, Any]:
    load_mode = effective_model_load_mode_from_request(request)
    settings = settings_from_request(request)
    default_model = str(getattr(settings, "model_id", "") or "").strip()
    allowed = allowed_model_ids(request=request)
    dep_caps = deployment_capabilities(request)

    if load_mode == "off":
        ids = _unique_ids(default_model, allowed)
        models = [
            {
                "id": mid,
                "default": mid == default_model,
                "backend": None,
                "capabilities": effective_capabilities(mid, request=request),
                "load_mode": "off",
                "loaded": False,
            }
            for mid in ids
        ]
        return {
            "default_model": default_model,
            "models": models,
            "deployment_capabilities": dep_caps,
        }

    if llm is None:
        raise AppError(
            code="llm_unavailable",
            message="Model runtime is not initialized",
            status_code=503,
        )

    if isinstance(llm, MultiModelManager):
        models: list[dict[str, Any]] = []
        try:
            for st in llm.status():
                mid = str(getattr(st, "model_id", "") or "").strip()
                if not mid:
                    continue
                models.append(
                    {
                        "id": mid,
                        "default": mid == llm.default_id,
                        "backend": getattr(st, "backend", None),
                        "capabilities": effective_capabilities(mid, request=request),
                        "load_mode": getattr(st, "load_mode", None) or "unknown",
                        "loaded": getattr(st, "loaded", None),
                    }
                )
        except Exception:
            for mid in llm.list_models():
                backend = llm[mid]
                backend_name = getattr(backend, "backend_name", None)
                models.append(
                    {
                        "id": mid,
                        "default": mid == llm.default_id,
                        "backend": str(backend_name) if isinstance(backend_name, str) else type(backend).__name__,
                        "capabilities": effective_capabilities(mid, request=request),
                        "load_mode": "unknown",
                        "loaded": None,
                    }
                )

        return {
            "default_model": llm.default_id,
            "models": models,
            "deployment_capabilities": dep_caps,
        }

    return {
        "default_model": getattr(llm, "model_id", None) or default_model,
        "models": _single_backend_payload(
            request=request,
            llm=llm,
            default_model=default_model,
            load_mode=load_mode,
        ),
        "deployment_capabilities": dep_caps,
    }
