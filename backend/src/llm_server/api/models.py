# src/llm_server/api/models.py
from __future__ import annotations

import os
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel

from llm_server.api.deps import get_llm  # ✅ canonical location
from llm_server.core.config import get_settings
from llm_server.core.errors import AppError
from llm_server.services.inference import set_request_meta  # ✅ request.state helper
from llm_server.services.llm import MultiModelManager

router = APIRouter()


class ModelInfo(BaseModel):
    id: str
    default: bool
    backend: Optional[str] = None


class ModelsResponse(BaseModel):
    default_model: str
    models: List[ModelInfo]


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(
    request: Request,
    llm: Any = Depends(get_llm),
) -> ModelsResponse:
    """
    Return the list of available models and which one is the default.

    If MODEL_LOAD_MODE=off, we avoid touching the model and return the allowed list
    derived from settings.
    """
    # Canonical request.state instrumentation
    set_request_meta(request, route="/v1/models", model_id="models", cached=False)

    s = get_settings()

    mode = os.getenv("MODEL_LOAD_MODE", "lazy").strip().lower()
    if mode == "off":
        # No model load; just reflect config
        default_model = s.model_id
        set_request_meta(request, route="/v1/models", model_id=default_model, cached=False)

        return ModelsResponse(
            default_model=default_model,
            models=[ModelInfo(id=m, default=(m == default_model), backend=None) for m in s.all_model_ids],
        )

    if llm is None:
        raise AppError(
            code="llm_unavailable",
            message="LLM is not initialized",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # Multi-model setup
    if isinstance(llm, MultiModelManager):
        set_request_meta(request, route="/v1/models", model_id=llm.default_id, cached=False)

        items: List[ModelInfo] = []
        for model_id, backend in llm.models.items():
            items.append(
                ModelInfo(
                    id=model_id,
                    default=(model_id == llm.default_id),
                    backend=backend.__class__.__name__,
                )
            )
        return ModelsResponse(default_model=llm.default_id, models=items)

    # Single-model setup
    model_id = getattr(llm, "model_id", s.model_id)
    set_request_meta(request, route="/v1/models", model_id=model_id, cached=False)

    return ModelsResponse(
        default_model=model_id,
        models=[ModelInfo(id=model_id, default=True, backend=llm.__class__.__name__)],
    )