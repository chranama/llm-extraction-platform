from __future__ import annotations

from fastapi import Request

from llm_server.services.api_deps.core.model_load_mode import (
    effective_model_load_mode_from_request,
)
from llm_server.runtime.routing import (
    model_flags_from_app_state,
    resolve_default_model_id_and_backend_obj,
)


def compute_public_models_status(request: Request) -> dict[str, object]:
    mode = effective_model_load_mode_from_request(request)
    model_loaded, model_error, loaded_model_id, runtime_default_model_id = (
        model_flags_from_app_state(request)
    )
    default_model_id, default_backend, _ = resolve_default_model_id_and_backend_obj(request)

    status = "ok"
    if mode != "off":
        if model_error:
            status = "degraded"
        elif not model_loaded:
            status = "degraded"

    return {
        "status": status,
        "model_load_mode": mode,
        "model_loaded": bool(model_loaded),
        "loaded_model_id": loaded_model_id,
        "runtime_default_model_id": runtime_default_model_id,
        "model_error": model_error,
        "default_model_id": default_model_id,
        "default_backend": default_backend,
    }
