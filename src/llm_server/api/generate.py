# src/llm_server/api/generate.py

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select

from llm_server.db.models import InferenceLog, CompletionCache
from llm_server.db.session import async_session_maker
from llm_server.core.config import settings
from llm_server.api.deps import get_api_key

router = APIRouter()


# -------------------------------
# Schemas
# -------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: list[str] | None = None


class StreamRequest(GenerateRequest):
    pass


# -------------------------------
# LLM dependency
# -------------------------------

def get_llm(request: Request) -> Any:
    """
    Accessor used as a FastAPI dependency and imported by health.py.

    The test suite may override this dependency to inject DummyModelManager.
    """
    return getattr(request.app.state, "llm", None)


# -------------------------------
# Helpers
# -------------------------------

def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:32]


def fingerprint_params(body: GenerateRequest) -> str:
    params = body.model_dump(exclude={"prompt"}, exclude_none=True)
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:32]


def get_model_id(llm: Any) -> str:
    if llm is not None and hasattr(llm, "model_id"):
        return getattr(llm, "model_id")
    return getattr(settings, "llm_model", None) or "unknown"


# -------------------------------
# Generate endpoint
# -------------------------------

@router.post("/v1/generate")
async def generate(
    request: Request,
    body: GenerateRequest,
    api_key=Depends(get_api_key),
    llm: Any = Depends(get_llm),
):
    model_id = get_model_id(llm)

    prompt_hash = hash_prompt(body.prompt)
    params_fp = fingerprint_params(body)

    start = time.time()

    async with async_session_maker() as session:

        # ---- 1. Check cache ----
        cached = await session.execute(
            select(CompletionCache).where(
                CompletionCache.model_id == model_id,
                CompletionCache.prompt_hash == prompt_hash,
                CompletionCache.params_fingerprint == params_fp,
            )
        )
        cached = cached.scalar_one_or_none()

        if cached:
            output = cached.output
            latency = (time.time() - start) * 1000

            log = InferenceLog(
                api_key=api_key.key,
                route="/v1/generate",
                client_host=request.client.host if request.client else None,
                model_id=model_id,
                params_json=body.model_dump(exclude={"prompt"}, exclude_none=True),
                prompt=body.prompt,
                output=output,
                latency_ms=latency,
            )

            session.add(log)
            await session.commit()

            return {
                "model": model_id,
                "output": output,
                "cached": True,
            }

        # ---- 2. Run model ----
        result = llm.generate(
            prompt=body.prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            stop=body.stop,
        )

        output = result if isinstance(result, str) else str(result)

        latency = (time.time() - start) * 1000

        # ---- 3. Save cache ----
        cache = CompletionCache(
            model_id=model_id,
            prompt=body.prompt,
            prompt_hash=prompt_hash,
            params_fingerprint=params_fp,
            output=output,
        )
        session.add(cache)

        # ---- 4. Save log ----
        log = InferenceLog(
            api_key=api_key.key,
            route="/v1/generate",
            client_host=request.client.host if request.client else None,
            model_id=model_id,
            params_json=body.model_dump(exclude={"prompt"}, exclude_none=True),
            prompt=body.prompt,
            output=output,
            latency_ms=latency,
        )
        session.add(log)

        await session.commit()

        return {
            "model": model_id,
            "output": output,
            "cached": False,
        }


# -------------------------------
# Stream endpoint
# -------------------------------

async def _sse_event_generator(llm: Any, body: StreamRequest) -> AsyncGenerator[str, None]:
    for chunk in llm.stream(
        prompt=body.prompt,
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        stop=body.stop,
    ):
        yield f"data: {chunk}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/v1/stream")
async def stream(
    request: Request,
    body: StreamRequest,
    api_key=Depends(get_api_key),
    llm: Any = Depends(get_llm),
):
    generator = _sse_event_generator(llm, body)

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
    )