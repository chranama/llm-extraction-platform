# src/llm_server/api/generate.py
from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from typing import Any, AsyncGenerator, List

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from transformers import AutoTokenizer

from llm_server.db.models import InferenceLog, CompletionCache
from llm_server.db.session import async_session_maker
from llm_server.core.config import settings
from llm_server.api.deps import get_api_key
from llm_server.services.llm import build_llm_from_settings, MultiModelManager
from llm_server.core.metrics import (
    LLM_TOKENS,
    LLM_REDIS_HITS,
    LLM_REDIS_MISSES,
    LLM_REDIS_LATENCY,
)
from llm_server.core.redis import get_redis_from_request

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

    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: list[str] | None = None


class StreamRequest(GenerateRequest):
    pass


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


def get_llm(request: Request) -> Any:
    """
    Accessor used as a FastAPI dependency and imported by health.py.

    - If app.state.llm is already set (from lifespan startup), just return it.
    - If it's None (e.g. startup failed or was bypassed), lazily build it
      from settings so the API can still serve requests.
    """
    llm = getattr(request.app.state, "llm", None)

    if llm is None:
        # Lazy fallback initialization (no arguments; uses global settings)
        llm = build_llm_from_settings()
        request.app.state.llm = llm

    return llm


def resolve_model(request: Request, llm: Any, model_override: str | None) -> tuple[str, Any]:
    allowed = settings.all_model_ids

    if model_override is None:
        model_id = settings.model_id
    else:
        model_id = model_override
        if model_id not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_id}' not allowed. Allowed: {allowed}",
            )

    # Multi-model: MultiModelManager
    if isinstance(llm, MultiModelManager):
        if model_id not in llm:
            raise HTTPException(
                status_code=500,
                detail=f"Model '{model_id}' not found in LLM registry",
            )
        return model_id, llm[model_id]

    # (optional) Multi-model: dict
    if isinstance(llm, dict):
        if model_id not in llm:
            raise HTTPException(
                status_code=500,
                detail=f"Model '{model_id}' not found in LLM registry",
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
    params = body.model_dump(exclude={"prompt", "model"}, exclude_none=True)
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:32]


def fingerprint_batch_params(body: BatchGenerateRequest) -> str:
    """
    Same idea as fingerprint_params, but for the batch request:
    - Excludes 'prompts' and 'model'; the model_id is already part of the key.
    """
    params = body.model_dump(exclude={"prompts", "model"}, exclude_none=True)
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:32]


def make_redis_key(model_id: str, prompt_hash: str, params_fp: str) -> str:
    """
    Compose a Redis key for the completion cache.
    """
    return f"llm:cache:{model_id}:{prompt_hash}:{params_fp}"


# -------------------------------
# Token counting helpers
# -------------------------------


@lru_cache(maxsize=16)
def _get_tokenizer(model_id: str):
    """
    Lazily load and cache a tokenizer per model_id.

    This is independent from the runtime ModelManager so it also works
    when the model is remote (HttpLLMClient) as long as the HF repo
    is accessible locally.
    """
    return AutoTokenizer.from_pretrained(model_id, use_fast=True)


def count_tokens(model_id: str, prompt: str, completion: str | None) -> tuple[int | None, int | None]:
    """
    Best-effort token counting.

    Returns (prompt_tokens, completion_tokens). If anything goes wrong,
    returns (None, None) so logging doesn't break inference.
    """
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
        # Don’t let logging kill the request
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
    # Resolve which logical model id this request should use
    model_id, model = resolve_model(request, llm, body.model)

    # Tag request for logging/metrics middleware
    request.state.route = "/v1/generate"
    request.state.model_id = model_id

    # Best-effort request id (set by RequestLoggingMiddleware)
    request_id = getattr(request.state, "request_id", None)

    prompt_hash = hash_prompt(body.prompt)
    params_fp = fingerprint_params(body)
    redis_key = make_redis_key(model_id, prompt_hash, params_fp)

    start = time.time()

    redis = get_redis_from_request(request)

    async with async_session_maker() as session:
        # ---- 0. Redis cache (if enabled) ----
        if redis is not None:
            redis_start = time.perf_counter()
            raw = None
            try:
                raw = await redis.get(redis_key)
            finally:
                latency = time.perf_counter() - redis_start
                LLM_REDIS_LATENCY.labels(model_id=model_id, kind="single").observe(latency)

            if raw is not None:
                # Hit
                LLM_REDIS_HITS.labels(model_id=model_id, kind="single").inc()
                try:
                    payload = json.loads(raw)
                    output = payload.get("output")
                    if output is None:
                        # Fallback: treat as miss if malformed
                        raise ValueError("Missing 'output' in Redis payload")
                except Exception:
                    # Treat JSON errors as a miss but don't crash the request
                    output = None

                if output is not None:
                    latency_ms = (time.time() - start) * 1000
                    request.state.cached = True

                    # Token counting from cached output
                    prompt_tokens, completion_tokens = count_tokens(model_id, body.prompt, output)

                    # Token metrics
                    if prompt_tokens is not None:
                        LLM_TOKENS.labels(direction="prompt", model_id=model_id).inc(prompt_tokens)
                    if completion_tokens is not None:
                        LLM_TOKENS.labels(direction="completion", model_id=model_id).inc(completion_tokens)

                    # Log inference
                    log = InferenceLog(
                        api_key=api_key.key,
                        request_id=request_id,
                        route="/v1/generate",
                        client_host=request.client.host if request.client else None,
                        model_id=model_id,
                        params_json=body.model_dump(
                            exclude={"prompt", "model"},
                            exclude_none=True,
                        ),
                        prompt=body.prompt,
                        output=output,
                        latency_ms=latency_ms,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                    session.add(log)
                    await session.commit()

                    return {
                        "model": model_id,
                        "output": output,
                        "cached": True,
                    }
            else:
                LLM_REDIS_MISSES.labels(model_id=model_id, kind="single").inc()

        # ---- 1. DB CompletionCache ----
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
            latency_ms = (time.time() - start) * 1000

            request.state.cached = True

            # Token counting from cached output
            prompt_tokens, completion_tokens = count_tokens(model_id, body.prompt, output)

            # Token metrics
            if prompt_tokens is not None:
                LLM_TOKENS.labels(direction="prompt", model_id=model_id).inc(prompt_tokens)
            if completion_tokens is not None:
                LLM_TOKENS.labels(direction="completion", model_id=model_id).inc(completion_tokens)

            # Backfill Redis cache if available
            if redis is not None:
                try:
                    await redis.set(
                        redis_key,
                        json.dumps({"output": output}),
                        ex=REDIS_TTL_SECONDS,
                    )
                except Exception:
                    # Don't fail the request if Redis SET fails
                    pass

            log = InferenceLog(
                api_key=api_key.key,
                request_id=request_id,
                route="/v1/generate",
                client_host=request.client.host if request.client else None,
                model_id=model_id,
                params_json=body.model_dump(
                    exclude={"prompt", "model"},
                    exclude_none=True,
                ),
                prompt=body.prompt,
                output=output,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            session.add(log)
            await session.commit()

            return {
                "model": model_id,
                "output": output,
                "cached": True,
            }

        # ---- 2. Run model ----
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

        # Token counting from live output
        prompt_tokens, completion_tokens = count_tokens(model_id, body.prompt, output)

        # Token metrics
        if prompt_tokens is not None:
            LLM_TOKENS.labels(direction="prompt", model_id=model_id).inc(prompt_tokens)
        if completion_tokens is not None:
            LLM_TOKENS.labels(direction="completion", model_id=model_id).inc(completion_tokens)

        # ---- 3. Save cache (DB + Redis) ----
        cache = CompletionCache(
            model_id=model_id,
            prompt=body.prompt,
            prompt_hash=prompt_hash,
            params_fingerprint=params_fp,
            output=output,
        )
        session.add(cache)

        if redis is not None:
            try:
                await redis.set(
                    redis_key,
                    json.dumps({"output": output}),
                    ex=REDIS_TTL_SECONDS,
                )
            except Exception:
                pass

        # ---- 4. Save log ----
        log = InferenceLog(
            api_key=api_key.key,
            request_id=request_id,
            route="/v1/generate",
            client_host=request.client.host if request.client else None,
            model_id=model_id,
            params_json=body.model_dump(
                exclude={"prompt", "model"},
                exclude_none=True,
            ),
            prompt=body.prompt,
            output=output,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        session.add(log)

        await session.commit()

        return {
            "model": model_id,
            "output": output,
            "cached": False,
        }


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
    """
    Batch generation endpoint.

    - Reuses the same LLM client and model routing as /v1/generate
    - Executes prompts sequentially for now
    - Uses the same CompletionCache + InferenceLog patterns as /v1/generate
    - Adds Redis as a fast read-through cache per prompt
    """
    # Resolve model and tag request for metrics middleware
    model_id, model = resolve_model(request, llm, body.model)

    request.state.route = "/v1/generate/batch"
    request.state.model_id = model_id

    request_id = getattr(request.state, "request_id", None)

    params_fp = fingerprint_batch_params(body)
    redis = get_redis_from_request(request)

    results: list[BatchGenerateResult] = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    all_cached = True

    async with async_session_maker() as session:
        for prompt in body.prompts:
            item_start = time.time()
            prompt_hash = hash_prompt(prompt)
            redis_key = make_redis_key(model_id, prompt_hash, params_fp)

            output: str | None = None
            cached_flag = False

            # ---- 0. Redis lookup (if enabled) ----
            if redis is not None and body.cache:
                redis_start = time.perf_counter()
                raw = None
                try:
                    raw = await redis.get(redis_key)
                finally:
                    latency = time.perf_counter() - redis_start
                    LLM_REDIS_LATENCY.labels(model_id=model_id, kind="batch").observe(latency)

                if raw is not None:
                    LLM_REDIS_HITS.labels(model_id=model_id, kind="batch").inc()
                    try:
                        payload = json.loads(raw)
                        output = payload.get("output")
                        if output is None:
                            raise ValueError("Missing 'output' in Redis payload")
                    except Exception:
                        output = None
                else:
                    LLM_REDIS_MISSES.labels(model_id=model_id, kind="batch").inc()

            # ---- 1. DB cache (CompletionCache) ----
            if output is None and body.cache:
                cached_row = await session.execute(
                    select(CompletionCache).where(
                        CompletionCache.model_id == model_id,
                        CompletionCache.prompt_hash == prompt_hash,
                        CompletionCache.params_fingerprint == params_fp,
                    )
                )
                cached_row = cached_row.scalar_one_or_none()

                if cached_row:
                    output = cached_row.output
                    cached_flag = True

                    # Backfill Redis if available
                    if redis is not None:
                        try:
                            await redis.set(
                                redis_key,
                                json.dumps({"output": output}),
                                ex=REDIS_TTL_SECONDS,
                            )
                        except Exception:
                            pass

            # ---- 2. Run model if still no output ----
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
                all_cached = False

                if body.cache:
                    cache = CompletionCache(
                        model_id=model_id,
                        prompt=prompt,
                        prompt_hash=prompt_hash,
                        params_fingerprint=params_fp,
                        output=output,
                    )
                    session.add(cache)

                    if redis is not None:
                        try:
                            await redis.set(
                                redis_key,
                                json.dumps({"output": output}),
                                ex=REDIS_TTL_SECONDS,
                            )
                        except Exception:
                            pass

            latency_ms = (time.time() - item_start) * 1000

            # Token counting
            prompt_tokens, completion_tokens = count_tokens(model_id, prompt, output)

            if prompt_tokens is not None:
                LLM_TOKENS.labels(direction="prompt", model_id=model_id).inc(prompt_tokens)
                total_prompt_tokens += prompt_tokens
                prompt_tokens_int = prompt_tokens
            else:
                prompt_tokens_int = 0

            if completion_tokens is not None:
                LLM_TOKENS.labels(direction="completion", model_id=model_id).inc(completion_tokens)
                total_completion_tokens += completion_tokens
                completion_tokens_int = completion_tokens
            else:
                completion_tokens_int = 0

            # Log per prompt
            log = InferenceLog(
                api_key=api_key.key,
                request_id=request_id,
                route="/v1/generate/batch",
                client_host=request.client.host if request.client else None,
                model_id=model_id,
                params_json=body.model_dump(
                    exclude={"prompts", "model"},
                    exclude_none=True,
                ),
                prompt=prompt,
                output=output,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            session.add(log)

            results.append(
                BatchGenerateResult(
                    output=output,
                    cached=cached_flag,
                    prompt_tokens=prompt_tokens_int,
                    completion_tokens=completion_tokens_int,
                )
            )

        await session.commit()

    # For middleware metrics, treat the request as cached only if *all* items were cached
    request.state.cached = all_cached

    return BatchGenerateResponse(
        model=model_id,
        results=results,
    )


# -------------------------------
# Stream endpoint
# -------------------------------


async def _sse_event_generator(
    model: Any,
    body: StreamRequest,
) -> AsyncGenerator[str, None]:
    """
    Wrap the model.stream(...) iterator into an SSE stream.
    """
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
    # Resolve model id and tag request for logging/metrics
    model_id, model = resolve_model(request, llm, body.model)
    request.state.route = "/v1/stream"
    request.state.model_id = model_id
    request.state.cached = False  # no caching for streams

    # (Optionally: you could later log streaming requests keyed by model_id)
    generator = _sse_event_generator(model, body)

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
    )