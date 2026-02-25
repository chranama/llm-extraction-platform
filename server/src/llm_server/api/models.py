# server/src/llm_server/api/models.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from llm_server.services.api_deps.core.llm_access import get_llm
from llm_server.services.llm_runtime.inference import set_request_meta

from llm_server.services.api_deps.models.status import compute_public_models_status
from llm_server.services.api_deps.models.listing import list_models_payload

router = APIRouter()


class ModelInfo(BaseModel):
    id: str
    default: bool
    backend: Optional[str] = None
    capabilities: Optional[Dict[str, bool]] = None
    load_mode: Optional[str] = None
    loaded: Optional[bool] = None


class ModelsResponse(BaseModel):
    default_model: str
    models: List[ModelInfo]
    deployment_capabilities: Dict[str, bool]


class ModelsStatusResponse(BaseModel):
    """
    Minimal runtime status for non-admin clients.
    Keep this intentionally small and stable (no deep registry internals).
    """
    status: str  # "ok" | "degraded"
    model_load_mode: str
    model_loaded: bool
    loaded_model_id: Optional[str] = None
    runtime_default_model_id: Optional[str] = None
    model_error: Optional[str] = None

    default_model_id: Optional[str] = None
    default_backend: Optional[str] = None


@router.get("/v1/models/status", response_model=ModelsStatusResponse)
async def models_status(request: Request) -> ModelsStatusResponse:
    """
    Public, minimal runtime model status (non-admin).
    Intentionally does NOT expose registry internals.
    """
    set_request_meta(request, route="/v1/models/status", model_id="models", cached=False)
    payload = compute_public_models_status(request)
    return ModelsStatusResponse(**payload)


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(request: Request) -> ModelsResponse:
    set_request_meta(request, route="/v1/models", model_id="models", cached=False)

    llm: Any = get_llm(request)
    payload = list_models_payload(request=request, llm=llm)

    # Coerce to Pydantic API contract classes
    items = [ModelInfo(**x) for x in payload["models"]]
    return ModelsResponse(
        default_model=str(payload["default_model"] or ""),
        models=items,
        deployment_capabilities=payload["deployment_capabilities"],
    )