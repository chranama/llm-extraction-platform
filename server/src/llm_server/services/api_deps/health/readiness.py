# server/src/llm_server/services/api_deps/health/readiness.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from fastapi import Request

from llm_server.services.api_deps.core.model_load_mode import effective_model_load_mode_from_request
from llm_server.services.api_deps.health.external_probes import (
    external_backend_generate_check_async,
    llamacpp_dependency_check_async,
    remote_probe_async,
)
from llm_server.services.api_deps.routing.models import (
    model_flags_from_app_state,
    resolve_default_model_id_and_backend_obj,
    per_model_readiness_mode,
)


async def model_ready_check_async(request: Request) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Unified model readiness check used by BOTH /modelz and /readyz.

    Rules:
      - model_load_mode=off => ready (skipped)
      - readiness_mode (per-model meta w/ settings fallback):
          - off => ready (skipped)
          - probe => external: probe/health; in-process: reflect loaded/error only
          - generate => external: can_generate/is_ready; in-process: reflect loaded/error only

    IMPORTANT:
      - This endpoint never triggers weight loading.
    """
    mode = effective_model_load_mode_from_request(request)
    if mode == "off":
        return True, "skipped (model_load_mode=off)", {"mode": mode}

    model_id, backend_name, backend_obj = resolve_default_model_id_and_backend_obj(request)
    readiness_mode = per_model_readiness_mode(request, model_id=model_id)

    if readiness_mode == "off":
        return True, "skipped (readiness_mode=off)", {"readiness_mode": readiness_mode, "model_id": model_id, "backend": backend_name}

    model_loaded, model_error, loaded_model_id, runtime_default = model_flags_from_app_state(request)

    # External backend readiness
    if backend_name in ("llamacpp", "remote"):
        if backend_obj is None:
            return False, "backend missing", {"model_id": model_id, "backend": backend_name}

        if readiness_mode == "probe":
            if backend_name == "llamacpp":
                ok, st, details = await llamacpp_dependency_check_async(backend_obj)
                return ok, st, {"model_id": model_id, "backend": backend_name, "readiness_mode": readiness_mode, **details}

            ok, st, details = await remote_probe_async(backend_obj)
            return ok, st, {"model_id": model_id, "backend": backend_name, "readiness_mode": readiness_mode, **details}

        # readiness_mode == "generate"
        ok, st, details = await external_backend_generate_check_async(backend_obj)
        return ok, st, {"model_id": model_id, "backend": backend_name, "readiness_mode": readiness_mode, **details}

    # In-process backend readiness: reflect runtime truth only.
    ok_inproc = bool(model_loaded and model_error is None)
    status = "ok" if ok_inproc else "not ready"
    details: Dict[str, Any] = {
        "model_id": model_id,
        "backend": backend_name,
        "readiness_mode": readiness_mode,
        "model_loaded": bool(model_loaded),
        "model_error": model_error,
        "loaded_model_id": loaded_model_id,
        "runtime_default_model_id": runtime_default,
    }
    return ok_inproc, status, details