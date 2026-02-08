# server/src/llm_server/api/generate.py
from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any, List

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from llm_server.api.deps import (
    fingerprint_pydantic,
    get_api_key,
    get_llm,
    make_cache_redis_key,
    require_capability,
    resolve_model,
    sha32,
)
from llm_server.core.redis import get_redis_from_request
from llm_server.io.policy_decisions import policy_generate_max_new_tokens_cap
import llm_server.db.session as db_session  # module import so tests can patch session wiring
from llm_server.services.inference import (
    CacheSpec,
    get_cached_output,
    record_token_metrics,
    set_request_meta,
    write_cache,
    write_inference_log,
)

router = APIRouter()

REDIS_TTL_SECONDS = 3600


class GenerateRequest(BaseModel):
    prompt: str
    model: str | None = Field(default=None, description="Optional model id override for multi-model routing")
    cache: bool = True
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: list[str] | None = None


class BatchGenerateRequest(BaseModel):
    prompts: List[str]
    model: str | None = Field(
        default=None,
        description="Optional model id override for multi-model routing (applies to all prompts in the batch)",
    )
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int | None = None
    stop: list[str] | None = None
    cache: bool = True


class BatchGenerateResult(BaseModel):
    output: str
    cached: bool
    prompt_tokens: int
    completion_tokens: int


class BatchGenerateResponse(BaseModel):
    model: str
    results: List[BatchGenerateResult]


@lru_cache(maxsize=16)
def _get_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id, use_fast=True)


def count_tokens(model_id: str, prompt: str, completion: str | None) -> tuple[int | None, int | None]:
    # Hard off-switch (tests can set TOKEN_COUNTING=0)
    if os.getenv("TOKEN_COUNTING", "1").strip().lower() in {"0", "false", "no", "off"}:
        return None, None

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


def _apply_generate_cap(request: Request, *, model_id: str, requested: int | None) -> int | None:
    """
    Apply v2 policy clamp to max_new_tokens.

    Behavior:
      - If no cap => return requested unchanged
      - If cap exists:
          - if requested is None: we set effective=cap (so clamp actually clamps)
          - else effective=min(requested, cap)
    """
    cap = policy_generate_max_new_tokens_cap(model_id, request=request)
    if cap is None:
        return requested

    try:
        cap_i = int(cap)
        if cap_i <= 0:
            return requested
    except Exception:
        return requested

    if requested is None:
        return cap_i

    try:
        req_i = int(requested)
        if req_i <= 0:
            return cap_i
        return min(req_i, cap_i)
    except Exception:
        return cap_i


@router.post("/v1/generate")
async def generate(
    request: Request,
    body: GenerateRequest,
    api_key=Depends(get_api_key),
    llm: Any = Depends(get_llm),
):
    model_id, model = resolve_model(llm, body.model, capability="generate", request=request)
    require_capability(model_id, "generate", request=request)
    set_request_meta(request, route="/v1/generate", model_id=model_id, cached=False)

    request_id = getattr(request.state, "request_id", None)

    # --- apply policy clamp BEFORE fingerprinting / cache keys ---
    effective_max_new_tokens = _apply_generate_cap(request, model_id=model_id, requested=body.max_new_tokens)

    prompt_hash = sha32(body.prompt)
    # exclude prompt/model/cache; fingerprint SHOULD include max_new_tokens (effective)
    params_fp = fingerprint_pydantic(
        body.model_copy(update={"max_new_tokens": effective_max_new_tokens}),
        exclude={"prompt", "model", "cache"},
    )
    redis_key = make_cache_redis_key(model_id, prompt_hash, params_fp)

    cache = CacheSpec(
        model_id=model_id,
        prompt=body.prompt,
        prompt_hash=prompt_hash,
        params_fp=params_fp,
        redis_key=redis_key,
        redis_ttl_seconds=REDIS_TTL_SECONDS,
    )

    start = time.time()
    redis = get_redis_from_request(request)

    async with db_session.get_sessionmaker()() as session:
        # ---- cache read (redis -> db) ----
        cached_out, cached_flag, _layer = await get_cached_output(
            session,
            redis,
            cache=cache,
            kind="single",
            enabled=bool(body.cache),
        )

        if isinstance(cached_out, str) and cached_flag:
            latency_ms = (time.time() - start) * 1000
            request.state.cached = True

            prompt_tokens, completion_tokens = count_tokens(model_id, body.prompt, cached_out)
            record_token_metrics(model_id, prompt_tokens, completion_tokens)

            await write_inference_log(
                session,
                api_key=api_key.key,
                request_id=request_id,
                route="/v1/generate",
                client_host=request.client.host if request.client else None,
                model_id=model_id,
                params_json=body.model_dump(exclude={"prompt", "model"}, exclude_none=True)
                | {"max_new_tokens": effective_max_new_tokens},
                prompt=body.prompt,
                output=cached_out,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                status_code=200,
                cached=True,
                error_code=None,
                error_stage=None,
                commit=True,
            )
            return {"model": model_id, "output": cached_out, "cached": True}

        # ---- run model ----
        result = model.generate(
            prompt=body.prompt,
            max_new_tokens=effective_max_new_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            stop=body.stop,
        )
        output = result if isinstance(result, str) else str(result)

        latency_ms = (time.time() - start) * 1000
        request.state.cached = False

        prompt_tokens, completion_tokens = count_tokens(model_id, body.prompt, output)
        record_token_metrics(model_id, prompt_tokens, completion_tokens)

        # ---- cache write ----
        await write_cache(
            session,
            redis,
            cache=cache,
            output=output,
            enabled=bool(body.cache),
        )

        # ---- log ----
        await write_inference_log(
            session,
            api_key=api_key.key,
            request_id=request_id,
            route="/v1/generate",
            client_host=request.client.host if request.client else None,
            model_id=model_id,
            params_json=body.model_dump(exclude={"prompt", "model"}, exclude_none=True)
            | {"max_new_tokens": effective_max_new_tokens},
            prompt=body.prompt,
            output=output,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            status_code=200,
            cached=False,
            error_code=None,
            error_stage=None,
            commit=True,
        )

        return {"model": model_id, "output": output, "cached": False}


@router.post("/v1/generate/batch", response_model=BatchGenerateResponse)
async def generate_batch(
    request: Request,
    body: BatchGenerateRequest,
    api_key=Depends(get_api_key),
    llm: Any = Depends(get_llm),
):
    model_id, model = resolve_model(llm, body.model, capability="generate", request=request)
    require_capability(model_id, "generate", request=request)
    set_request_meta(request, route="/v1/generate/batch", model_id=model_id, cached=False)

    request_id = getattr(request.state, "request_id", None)

    # Apply clamp once for the batch.
    # Note: Batch uses required int max_new_tokens; clamp may reduce it.
    effective_batch_max_new = _apply_generate_cap(request, model_id=model_id, requested=body.max_new_tokens)
    if effective_batch_max_new is None:
        effective_batch_max_new = body.max_new_tokens

    params_fp = fingerprint_pydantic(
        body.model_copy(update={"max_new_tokens": effective_batch_max_new}),
        exclude={"prompts", "model"},
    )
    redis = get_redis_from_request(request)

    results: list[BatchGenerateResult] = []
    all_cached = True if body.cache else False

    async with db_session.get_sessionmaker()() as session:
        for prompt in body.prompts:
            item_start = time.time()
            prompt_hash = sha32(prompt)
            redis_key = make_cache_redis_key(model_id, prompt_hash, params_fp)

            cache = CacheSpec(
                model_id=model_id,
                prompt=prompt,
                prompt_hash=prompt_hash,
                params_fp=params_fp,
                redis_key=redis_key,
                redis_ttl_seconds=REDIS_TTL_SECONDS,
            )

            out, cached_flag, _layer = await get_cached_output(
                session,
                redis,
                cache=cache,
                kind="batch",
                enabled=bool(body.cache),
            )

            if isinstance(out, str) and cached_flag:
                output = out
            else:
                result = model.generate(
                    prompt=prompt,
                    max_new_tokens=effective_batch_max_new,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    top_k=body.top_k,
                    stop=body.stop,
                )
                output = result if isinstance(result, str) else str(result)
                cached_flag = False
                all_cached = False

                await write_cache(
                    session,
                    redis,
                    cache=cache,
                    output=output,
                    enabled=bool(body.cache),
                )

            if body.cache and not cached_flag:
                all_cached = False

            latency_ms = (time.time() - item_start) * 1000

            prompt_tokens, completion_tokens = count_tokens(model_id, prompt, output)
            record_token_metrics(model_id, prompt_tokens, completion_tokens)

            await write_inference_log(
                session,
                api_key=api_key.key,
                request_id=request_id,
                route="/v1/generate/batch",
                client_host=request.client.host if request.client else None,
                model_id=model_id,
                params_json=body.model_dump(exclude={"prompts", "model"}, exclude_none=True)
                | {"max_new_tokens": effective_batch_max_new},
                prompt=prompt,
                output=output,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                status_code=200,
                cached=bool(cached_flag),
                error_code=None,
                error_stage=None,
                commit=False,
            )

            results.append(
                BatchGenerateResult(
                    output=output,
                    cached=bool(cached_flag),
                    prompt_tokens=int(prompt_tokens or 0),
                    completion_tokens=int(completion_tokens or 0),
                )
            )

        await session.commit()

    request.state.cached = all_cached
    return BatchGenerateResponse(model=model_id, results=results)