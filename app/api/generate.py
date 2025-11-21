# app/api/generate.py
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Iterator, Optional, List

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import CompletionCache, InferenceLog, ApiKey
from app.db.session import get_session
from app.api.deps import require_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

# mirror request/response models for type safety
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None

class GenerateResponse(BaseModel):
    output: str
    model: str

def _params_fingerprint(d: dict) -> str:
    payload = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


@router.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: Request,
    body: GenerateRequest,
    session: AsyncSession = Depends(get_session),
    key: ApiKey = Depends(require_api_key),
):
    """
    Non-streaming generation that:
      - checks cache
      - calls external LLM service
      - logs inference + persists cache
    """
    request_id = getattr(request.state, "request_id", "unknown")
    client_host = request.client.host if request.client else None
    api_key_value = key.key
    logger.info("[%s] /v1/generate received (prompt_len=%d)", request_id, len(body.prompt))

    # params for cache (exclude prompt)
    params = dict(
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        repetition_penalty=body.repetition_penalty,
        stop=body.stop or [],
    )
    fp = _params_fingerprint(params)
    prompt_hash = hashlib.sha256(body.prompt.encode("utf-8")).hexdigest()[:32]

    # 1) cache lookup
    try:
        q = select(CompletionCache).where(
            CompletionCache.model_id == settings.model_id,
            CompletionCache.prompt_hash == prompt_hash,
            CompletionCache.params_fingerprint == fp,
        )
        cached = (await session.execute(q)).scalar_one_or_none()
    except Exception as e:
        logger.exception("[%s] cache lookup failed: %s", request_id, e)
        cached = None

    if cached:
        output = cached.output
        try:
            session.add(
                InferenceLog(
                    api_key=api_key_value,
                    request_id=request_id,
                    route="/v1/generate",
                    client_host=client_host,
                    model_id=settings.model_id or "unknown",
                    params_json=params,
                    prompt=body.prompt,
                    output=output,
                    latency_ms=0.0,
                )
            )
            await session.commit()
        except Exception as e:
            logger.exception("[%s] DB write failed (cache-hit log): %s", request_id, e)

        logger.info("[%s] /v1/generate cache-hit (output_len=%d)", request_id, len(output))
        return GenerateResponse(output=output, model=settings.model_id or "unknown")

    # 2) call external LLM
    latency_ms = None
    try:
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=settings.http_client_timeout) as client:
            resp = await client.post(
                f"{settings.llm_service_url.rstrip('/')}/generate",
                json=body.model_dump(),
            )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        output = data["output"]
        model_id = data.get("model") or (settings.model_id or "unknown")
        latency_ms = (time.perf_counter() - t0) * 1000.0
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[%s] llm service call failed: %s", request_id, e)
        raise HTTPException(status_code=502, detail="LLM upstream error")

    # Optional token counts — skip or compute later if you want
    prompt_tokens = None
    completion_tokens = None

    # 3) persist cache + log
    try:
        session.add(
            CompletionCache(
                model_id=model_id,
                prompt=body.prompt,
                prompt_hash=prompt_hash,
                params_fingerprint=fp,
                output=output,
            )
        )
        session.add(
            InferenceLog(
                api_key=api_key_value,
                request_id=request_id,
                route="/v1/generate",
                client_host=client_host,
                model_id=model_id,
                params_json=params,
                prompt=body.prompt,
                output=output,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
        await session.commit()
    except Exception as e:
        logger.exception("[%s] DB write failed: %s", request_id, e)

    logger.info(
        "[%s] /v1/generate complete (latency_ms=%.1f, output_len=%d)",
        request_id, latency_ms or -1.0, len(output),
    )
    return GenerateResponse(output=output, model=model_id)


@router.post("/stream")
async def stream_text(body: GenerateRequest, key: ApiKey = Depends(require_api_key)) -> StreamingResponse:
    """
    Proxy streaming from LLM service; forwards SSE frames as-is.
    """
    async def event_source():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{settings.llm_service_url.rstrip('/')}/stream",
                    json=body.model_dump(),
                ) as r:
                    r.raise_for_status()
                    async for chunk in r.aiter_bytes():
                        # pass through the SSE bytes
                        yield chunk
        except httpx.HTTPError as e:
            # emit a final SSE error-ish frame
            yield f"data: [ERROR] {e}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")