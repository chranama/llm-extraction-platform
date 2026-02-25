# server/src/llm_server/services/api_deps/core/llm_access.py
from __future__ import annotations

from typing import Any

from fastapi import Request

from llm_server.core.errors import AppError
from llm_server.services.api_deps.core.model_load_mode import effective_model_load_mode_from_request
from llm_server.services.llm_runtime.llm_build import build_llm_from_settings
from llm_server.services.llm_runtime.llm_loader import RuntimeModelLoader


def get_runtime_model_loader(request: Request) -> RuntimeModelLoader:
    """
    Retrieve RuntimeModelLoader from app.state. Defensive fallback for tests.
    """
    loader = getattr(request.app.state, "runtime_model_loader", None)
    if isinstance(loader, RuntimeModelLoader):
        return loader

    loader = RuntimeModelLoader(request.app.state)
    request.app.state.runtime_model_loader = loader
    return loader


def get_llm(request: Request) -> Any:
    """
    Retrieve the registry/backend object.

    Important:
      - Always builds the registry object if missing (even when model_load_mode == "off").
      - Raises 503 if app.state.model_error is set.
    """
    mode = effective_model_load_mode_from_request(request)

    model_error = getattr(request.app.state, "model_error", None)
    if model_error:
        raise AppError(
            code="llm_unavailable",
            message="LLM is unavailable due to startup/model error",
            status_code=503,
            extra={"model_error": model_error, "model_load_mode": mode},
        )

    llm = getattr(request.app.state, "llm", None)
    if llm is None:
        llm = build_llm_from_settings()
        request.app.state.llm = llm
    return llm