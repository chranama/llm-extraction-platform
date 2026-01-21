from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel

from llm_server.api.deps import get_llm  # canonical location
from llm_server.core.config import get_settings
from llm_server.core.errors import AppError
from llm_server.services.inference import set_request_meta  # request.state helper
from llm_server.services.llm import MultiModelManager

router = APIRouter()


class ModelInfo(BaseModel):
    id: str
    default: bool
    backend: Optional[str] = None


class ModelsResponse(BaseModel):
    default_model: str
    models: List[ModelInfo]


def _settings_from_request(request: Request):
    return getattr(request.app.state, "settings", None) or get_settings()


def _effective_model_load_mode(request: Request) -> str:
    # Prefer lifespan-computed mode; fallback to settings; fallback derived default
    mode = getattr(request.app.state, "model_load_mode", None)
    if isinstance(mode, str) and mode.strip():
        return mode.strip().lower()

    s = _settings_from_request(request)
    raw = getattr(s, "model_load_mode", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()

    env = str(getattr(s, "env", "dev")).strip().lower()
    return "eager" if env == "prod" else "lazy"


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(
    request: Request,
    llm: Any = Depends(get_llm),
) -> ModelsResponse:
    """
    Return the list of available models and which one is the default.

    If mode == "off", do NOT touch the model. Just reflect settings.
    """
    set_request_meta(request, route="/v1/models", model_id="models", cached=False)

    s = _settings_from_request(request)
    mode = _effective_model_load_mode(request)

    if mode == "off":
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