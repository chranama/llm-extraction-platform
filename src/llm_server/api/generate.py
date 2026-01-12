# src/llm_server/api/generate.py
from __future__ import annotations

import os
import hashlib
import json
import time
from functools import lru_cache
from typing import Any, AsyncGenerator, List

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from transformers import AutoTokenizer

from llm_server.api.deps import get_api_key
from llm_server.core.config import settings
from llm_server.core.errors import AppError
from llm_server.core.redis import get_redis_from_request, redis_get, redis_set
from llm_server.core.metrics import LLM_TOKENS
from llm_server.db.models import CompletionCache, InferenceLog
from llm_server.db.session import async_session_maker
from llm_server.services.llm import MultiModelManager, build_llm_from_settings
from llm_server.api.deps import get_llm

router = APIRouter()

# How long Redis entries should live, in seconds
REDIS_TTL_SECONDS = 3600


# -------------------------------
# Schemas
# -------------------------------


class GenerateRequest(BaseModel):
    prompt: str

    # Optional override to pick a non-default model (when multi-model is enabled)
    model: str | None = Field(
        default=None,
        description="Optional model id override for multi-model routing",
    )

    # NEW: allow disabling caching for single-generate
    cache: bool = True

    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: list[str] | None = None


class StreamRequest(BaseModel):
    prompt: str
    model: str | None = None
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: list[str] | None = None

class BatchGenerateRequest(BaseModel):
    """
    Batch of generate requests.

    For v1, we keep this simple: same generation params for all prompts.
    """

    prompts: List[str]

    # Optional model override, applied to all prompts
    model: str | None = Field(
        default=None,
        description="Optional model id override for multi-model routing (applies to all prompts in the batch)",
    )

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int | None = None
    stop: list[str] | None = None
    cache: bool = True  # reuse existing caching semantics


class BatchGenerateResult(BaseModel):
    output: str
    cached: bool
    prompt_tokens: int
    completion_tokens: int


class BatchGenerateResponse(BaseModel):
    model: str
    results: List[BatchGenerateResult]


# -------------------------------
# LLM dependency + routing
# -------------------------------



def resolve_model(llm: Any, model_override: str | None) -> tuple[str, Any]:
    allowed = settings.all_model_ids

    if model_override is None:
        model_id = settings.model_id
    else:
        model_id = model_override
        if model_id not in allowed:
            raise AppError(
                code="model_not_allowed",
                message=f"Model '{model_id}' not allowed.",
                status_code=status.HTTP_400_BAD_REQUEST,
                extra={"allowed": allowed},
            )

    # Multi-model: MultiModelManager
    if isinstance(llm, MultiModelManager):
        if model_id not in llm:
            raise AppError(
                code="model_missing",
                message=f"Model '{model_id}' not found in LLM registry",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return model_id, llm[model_id]

    # (optional) Multi-model: dict
    if isinstance(llm, dict):
        if model_id not in llm:
            raise AppError(
                code="model_missing",
                message=f"Model '{model_id}' not found in LLM registry",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return model_id, llm[model_id]

    # Single-model mode
    return model_id, llm


# -------------------------------
# Helpers
# -------------------------------


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:32]


def fingerprint_params(body: GenerateRequest) -> str:
    """
    Fingerprint all generation params except:
    - prompt (handled separately)
    - model (we key by model_id in the DB, so no need to double-encode it)

    Cache key is effectively:
        (model_id, prompt_hash, params_fingerprint)
    """
    params = body.model_dump(exclude={"prompt", "model", "cache"}, exclude_none=True)
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:32]


def fingerprint_batch_params(body: BatchGenerateRequest) -> str:
    """
    Same idea as fingerprint_params, but for the batch request:
    - Excludes 'prompts' and 'model'; the model_id is already part of the key.
    """
    params = body.model_dump(exclude={"prompts", "model"}, exclude_none=True)
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:32]


def make_redis_key(model_id: str, prompt_hash: str, params_fp: str) -> str:
    return f"llm:cache:{model_id}:{prompt_hash}:{params_fp}"


# -------------------------------
# Token counting helpers
# -------------------------------


@lru_cache(maxsize=16)
def _get_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id, use_fast=True)


def count_tokens(model_id: str, prompt: str, completion: str | None) -> tuple[int | None, int | None]:
    try:
        tok = _get_tokenizer(model_id)

        prompt_ids = tok(prompt, add_special_tokens=False).input_ids
        prompt_tokens = len(prompt_ids)

        if completion:
            completion_ids = tok(completion, add_special_tokens=False).input_ids
            completion_tokens = len(completion_ids)
        else:
            completion_tokens = 0

        return prompt_tokens, completion_tokens
    except Exception:
        return None, None


# -------------------------------
# Generate endpoint (single)
# -------------------------------


@router.post("/v1/generate")
async def generate(
    request: Request,
    body: GenerateRequest,
    api_key=Depends(get_api_key),
    llm: Any = Depends(get_llm),
):
    model_id, model = resolve_model(llm, body.model)

    request.state.route = "/v1/generate"
    request.state.model_id = model_id
    request.state.cached = False

    request_id = getattr(request.state, "request_id", None)

    prompt_hash = hash_prompt(body.prompt)
    params_fp = fingerprint_params(body)
    redis_key = make_redis_key(model_id, prompt_hash, params_fp)

    start = time.time()
    redis = get_redis_from_request(request)

    async with async_session_maker() as session:
        # ---- 0) Redis cache (if enabled AND cache=True) ----
        if redis is not None and body.cache:
            raw = await redis_get(redis, redis_key, model_id=model_id, kind="single")
            if raw is not None:
                output: str | None = None
                try:
                    payload = json.loads(raw)
                    output = payload.get("output")
                    if output is None:
                        raise ValueError("Missing 'output' in Redis payload")
                except Exception:
                    output = None

                if output is not None:
                    latency_ms = (time.time() - start) * 1000
                    request.state.cached = True

                    prompt_tokens, completion_tokens = count_tokens(model_id, body.prompt, output)
                    if prompt_tokens is not None:
                        LLM_TOKENS.labels(direction="prompt", model_id=model_id).inc(prompt_tokens)
                    if completion_tokens is not None:
                        LLM_TOKENS.labels(direction="completion", model_id=model_id).inc(completion_tokens)

                    session.add(
                        InferenceLog(
                            api_key=api_key.key,
                            request_id=request_id,
                            route="/v1/generate",
                            client_host=request.client.host if request.client else None,
                            model_id=model_id,
                            params_json=body.model_dump(exclude={"prompt", "model"}, exclude_none=True),
                            prompt=body.prompt,
                            output=output,
                            latency_ms=latency_ms,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        )
                    )
                    await session.commit()

                    return {"model": model_id, "output": output, "cached": True}

        # ---- 1) DB CompletionCache (if cache=True) ----
        if body.cache:
            cached_row = await session.execute(
                select(CompletionCache).where(
                    CompletionCache.model_id == model_id,
                    CompletionCache.prompt_hash == prompt_hash,
                    CompletionCache.params_fingerprint == params_fp,
                )
            )
            cached = cached_row.scalar_one_or_none()

            if cached is not None:
                output = cached.output
                latency_ms = (time.time() - start) * 1000
                request.state.cached = True

                prompt_tokens, completion_tokens = count_tokens(model_id, body.prompt, output)
                if prompt_tokens is not None:
                    LLM_TOKENS.labels(direction="prompt", model_id=model_id).inc(prompt_tokens)
                if completion_tokens is not None:
                    LLM_TOKENS.labels(direction="completion", model_id=model_id).inc(completion_tokens)

                # Backfill Redis
                if redis is not None:
                    try:
                        await redis_set(redis, redis_key, json.dumps({"output": output}), ex=REDIS_TTL_SECONDS)
                    except Exception:
                        pass

                session.add(
                    InferenceLog(
                        api_key=api_key.key,
                        request_id=request_id,
                        route="/v1/generate",
                        client_host=request.client.host if request.client else None,
                        model_id=model_id,
                        params_json=body.model_dump(exclude={"prompt", "model"}, exclude_none=True),
                        prompt=body.prompt,
                        output=output,
                        latency_ms=latency_ms,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                )
                await session.commit()

                return {"model": model_id, "output": output, "cached": True}

        # ---- 2) Run model ----
        result = model.generate(
            prompt=body.prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            stop=body.stop,
        )

        output = result if isinstance(result, str) else str(result)
        latency_ms = (time.time() - start) * 1000
        request.state.cached = False

        prompt_tokens, completion_tokens = count_tokens(model_id, body.prompt, output)
        if prompt_tokens is not None:
            LLM_TOKENS.labels(direction="prompt", model_id=model_id).inc(prompt_tokens)
        if completion_tokens is not None:
            LLM_TOKENS.labels(direction="completion", model_id=model_id).inc(completion_tokens)

        # ---- 3) Save cache (DB + Redis) if cache=True ----
        if body.cache:
            session.add(
                CompletionCache(
                    model_id=model_id,
                    prompt=body.prompt,
                    prompt_hash=prompt_hash,
                    params_fingerprint=params_fp,
                    output=output,
                )
            )

            try:
                await session.flush()
            except IntegrityError:
                await session.rollback()

            if redis is not None:
                try:
                    await redis_set(redis, redis_key, json.dumps({"output": output}), ex=REDIS_TTL_SECONDS)
                except Exception:
                    pass

        # ---- 4) Save log ----
        session.add(
            InferenceLog(
                api_key=api_key.key,
                request_id=request_id,
                route="/v1/generate",
                client_host=request.client.host if request.client else None,
                model_id=model_id,
                params_json=body.model_dump(exclude={"prompt", "model"}, exclude_none=True),
                prompt=body.prompt,
                output=output,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
        await session.commit()

        return {"model": model_id, "output": output, "cached": False}


# -------------------------------
# Batch generate endpoint
# -------------------------------


@router.post("/v1/generate/batch", response_model=BatchGenerateResponse)
async def generate_batch(
    request: Request,
    body: BatchGenerateRequest,
    api_key=Depends(get_api_key),
    llm: Any = Depends(get_llm),
):
    model_id, model = resolve_model(llm, body.model)

    request.state.route = "/v1/generate/batch"
    request.state.model_id = model_id
    request.state.cached = False  # will be set at end

    request_id = getattr(request.state, "request_id", None)

    params_fp = fingerprint_batch_params(body)
    redis = get_redis_from_request(request)

    results: list[BatchGenerateResult] = []

    # If cache is disabled, the request is never "cached"
    all_cached = True if body.cache else False

    async with async_session_maker() as session:
        for prompt in body.prompts:
            item_start = time.time()
            prompt_hash = hash_prompt(prompt)
            redis_key = make_redis_key(model_id, prompt_hash, params_fp)

            output: str | None = None
            cached_flag = False

            # ---- 0) Redis lookup (if enabled AND cache=True) ----
            if redis is not None and body.cache:
                raw = await redis_get(redis, redis_key, model_id=model_id, kind="batch")
                if raw is not None:
                    try:
                        payload = json.loads(raw)
                        output = payload.get("output")
                        if output is None:
                            raise ValueError("Missing 'output' in Redis payload")
                        cached_flag = True  # <-- FIX: redis hit counts as cached
                    except Exception:
                        output = None
                        cached_flag = False

            # ---- 1) DB cache (CompletionCache) ----
            if output is None and body.cache:
                cached_row = await session.execute(
                    select(CompletionCache).where(
                        CompletionCache.model_id == model_id,
                        CompletionCache.prompt_hash == prompt_hash,
                        CompletionCache.params_fingerprint == params_fp,
                    )
                )
                cached = cached_row.scalar_one_or_none()

                if cached is not None:
                    output = cached.output
                    cached_flag = True

                    # Backfill Redis
                    if redis is not None:
                        try:
                            await redis_set(redis, redis_key, json.dumps({"output": output}), ex=REDIS_TTL_SECONDS)
                        except Exception:
                            pass

            # ---- 2) Run model if still no output ----
            if output is None:
                result = model.generate(
                    prompt=prompt,
                    max_new_tokens=body.max_new_tokens,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    top_k=body.top_k,
                    stop=body.stop,
                )
                output = result if isinstance(result, str) else str(result)
                cached_flag = False
                all_cached = False  # <-- any model call makes the request non-cached

                if body.cache:
                    session.add(
                        CompletionCache(
                            model_id=model_id,
                            prompt=prompt,
                            prompt_hash=prompt_hash,
                            params_fingerprint=params_fp,
                            output=output,
                        )
                    )

                    try:
                        await session.flush()
                    except IntegrityError:
                        await session.rollback()

                    if redis is not None:
                        try:
                            await redis_set(redis, redis_key, json.dumps({"output": output}), ex=REDIS_TTL_SECONDS)
                        except Exception:
                            pass

            # If cache is enabled, AND the per-item result was not cached, the whole request isn't cached.
            if body.cache and not cached_flag:
                all_cached = False

            latency_ms = (time.time() - item_start) * 1000

            prompt_tokens, completion_tokens = count_tokens(model_id, prompt, output)
            prompt_tokens_int = int(prompt_tokens or 0)
            completion_tokens_int = int(completion_tokens or 0)

            if prompt_tokens is not None:
                LLM_TOKENS.labels(direction="prompt", model_id=model_id).inc(prompt_tokens)
            if completion_tokens is not None:
                LLM_TOKENS.labels(direction="completion", model_id=model_id).inc(completion_tokens)

            session.add(
                InferenceLog(
                    api_key=api_key.key,
                    request_id=request_id,
                    route="/v1/generate/batch",
                    client_host=request.client.host if request.client else None,
                    model_id=model_id,
                    params_json=body.model_dump(exclude={"prompts", "model"}, exclude_none=True),
                    prompt=prompt,
                    output=output,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )

            results.append(
                BatchGenerateResult(
                    output=output,
                    cached=cached_flag,
                    prompt_tokens=prompt_tokens_int,
                    completion_tokens=completion_tokens_int,
                )
            )

        await session.commit()

    request.state.cached = all_cached
    return BatchGenerateResponse(model=model_id, results=results)


# -------------------------------
# Stream endpoint
# -------------------------------


async def _sse_event_generator(
    model: Any,
    body: StreamRequest,
) -> AsyncGenerator[str, None]:
    for chunk in model.stream(
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
    model_id, model = resolve_model(llm, body.model)
    request.state.route = "/v1/stream"
    request.state.model_id = model_id
    request.state.cached = False  # no caching for streams

    generator = _sse_event_generator(model, body)

    return StreamingResponse(generator, media_type="text/event-stream")