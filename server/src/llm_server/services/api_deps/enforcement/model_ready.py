# server/src/llm_server/services/api_deps/enforcement/model_ready.py
from __future__ import annotations

from typing import Any

from fastapi import Request, status

from llm_server.core.errors import AppError
from llm_server.services.api_deps.core.llm_access import get_runtime_model_loader
from llm_server.services.api_deps.core.model_load_mode import effective_model_load_mode_from_request


def _detect_backend_name(backend_obj: Any) -> str | None:
    try:
        v = getattr(backend_obj, "backend_name", None)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    except Exception:
        pass
    return None


async def require_inprocess_loaded_if_needed(
    *,
    request: Request,
    model_id: str,
    backend_obj: Any,
) -> None:
    """
    Canonical enforcement for model_load_mode semantics.

    Rules:
      - backend=transformers (in-process):
          * mode == "off"  -> require explicit admin hot-load AND loaded_model_id must match model_id
          * mode == "lazy" -> allow first-request hot-load via RuntimeModelLoader.load_model()
          * mode == "eager"-> allow first-request hot-load (best-effort) if startup didn't load
      - backend in ("llamacpp","remote"): always allowed (weights live outside server/)

    IMPORTANT:
      - This function is async because lazy/eager may load weights (off-loop inside loader).
      - This function is the enforcement boundary. Call it AFTER resolve_model() and capability checks.
    """
    mode = effective_model_load_mode_from_request(request)
    backend = _detect_backend_name(backend_obj)

    # External backends always allowed regardless of model_load_mode
    if backend in ("llamacpp", "remote"):
        return

    # Unknown backend: treat as in-process to be safe
    if backend != "transformers":
        backend = "transformers"

    loaded_flag = bool(getattr(request.app.state, "model_loaded", False))
    loaded_model_id = getattr(request.app.state, "loaded_model_id", None)
    loaded_model_id = loaded_model_id.strip() if isinstance(loaded_model_id, str) and loaded_model_id.strip() else None

    # Strict "off": must be explicitly loaded (admin) and match selected model
    if mode == "off":
        if loaded_flag and (loaded_model_id == model_id):
            return
        raise AppError(
            code="model_not_loaded",
            message="Model weights are not loaded. Admin load required.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            extra={
                "model_id": model_id,
                "backend": backend,
                "model_load_mode": mode,
                "model_loaded": loaded_flag,
                "loaded_model_id": loaded_model_id,
            },
        )

    # Lazy/eager: allow first-request load (single authoritative path)
    loader = get_runtime_model_loader(request)
    try:
        if loaded_flag and (loaded_model_id == model_id):
            return
        await loader.load_model(model_id)
    except AppError:
        raise
    except Exception as e:
        raise AppError(
            code="model_load_failed",
            message="Failed to load model",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            extra={"model_id": model_id, "backend": backend, "model_load_mode": mode, "error": repr(e)},
        ) from e