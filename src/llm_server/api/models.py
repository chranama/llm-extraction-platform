# src/llm_server/api/models.py
from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from llm_server.core.config import settings
from llm_server.api.generate import get_llm
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
def list_models(
    request: Request,
    llm: Any = Depends(get_llm),
) -> ModelsResponse:
    """
    Return the list of available models and which one is the default.

    - If we're using MultiModelManager, expose all registered models.
    - Otherwise, expose the single local model_id.
    """
    # Multi-model setup
    if isinstance(llm, MultiModelManager):
        items: List[ModelInfo] = []
        for model_id, backend in llm.models.items():
            items.append(
                ModelInfo(
                    id=model_id,
                    default=(model_id == llm.default_id),
                    backend=backend.__class__.__name__,
                )
            )
        return ModelsResponse(
            default_model=llm.default_id,
            models=items,
        )

    # Single-model setup
    model_id = getattr(llm, "model_id", settings.model_id)
    backend_name = llm.__class__.__name__ if llm is not None else None

    return ModelsResponse(
        default_model=model_id,
        models=[
            ModelInfo(
                id=model_id,
                default=True,
                backend=backend_name,
            )
        ],
    )