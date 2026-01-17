# src/llm_server/api/generate.py
from __future__ import annotations

import hashlib
import json
import os
import time
from functools import lru_cache
from typing import Any, List

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from llm_server.api.deps import get_api_key, get_llm
from llm_server.core.config import get_settings
from llm_server.core.errors import AppError
from llm_server.core.redis import get_redis_from_request
import llm_server.db.session as db_session  # module import so tests can patch session wiring
from llm_server.services.inference import (
    CacheSpec,
    get_cached_output,
    record_token_metrics,
    set_request_meta,
    write_cache,
    write_inference_log,
)
from llm_server.services.llm import MultiModelManager

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


def resolve_model(llm: Any, model_override: str | None) -> tuple[str, Any]:
    s = get_settings()
    allowed = s.all_model_ids

    if model_override is None:
        model_id = s.model_id
    else:
        model_id = model_override
        if model_id not in allowed:
            raise AppError(
                code="model_not_allowed",
                message=f"Model '{model_id}' not allowed.",
                status_code=status.HTTP_400_BAD_REQUEST,
                extra={"allowed": allowed},
            )

    if isinstance(llm, MultiModelManager):
        if model_id not in llm:
            raise AppError(
                code="model_missing",
                message=f"Model '{model_id}' not found in LLM registry",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return model_id, llm[model_id]

    if isinstance(llm, dict):
        if model_id not in llm:
            raise AppError(
                code="model_missing",
                message=f"Model '{model_id}' not found in LLM registry",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return model_id, llm[model_id]

    return model_id, llm


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:32]


def fingerprint_params(body: GenerateRequest) -> str:
    params = body.model_dump(exclude={"prompt", "model", "cache"}, exclude_none=True)
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:32]


def fingerprint_batch_params(body: BatchGenerateRequest) -> str:
    params = body.model_dump(exclude={"prompts", "model"}, exclude_none=True)
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:32]


def make_redis_key(model_id: str, prompt_hash: str, params_fp: str) -> str:
    return f"llm:cache:{model_id}:{prompt_hash}:{params_fp}"


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


@router.post("/v1/generate")
async def generate(
    request: Request,
    body: GenerateRequest,
    api_key=Depends(get_api_key),
    llm: Any = Depends(get_llm),
):
    model_id, model = resolve_model(llm, body.model)
    set_request_meta(request, route="/v1/generate", model_id=model_id, cached=False)

    request_id = getattr(request.state, "request_id", None)

    prompt_hash = hash_prompt(body.prompt)
    params_fp = fingerprint_params(body)
    redis_key = make_redis_key(model_id, prompt_hash, params_fp)

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
                params_json=body.model_dump(exclude={"prompt", "model"}, exclude_none=True),
                prompt=body.prompt,
                output=cached_out,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                commit=True,
            )
            return {"model": model_id, "output": cached_out, "cached": True}

        # ---- run model ----
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
            params_json=body.model_dump(exclude={"prompt", "model"}, exclude_none=True),
            prompt=body.prompt,
            output=output,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
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
    model_id, model = resolve_model(llm, body.model)
    set_request_meta(request, route="/v1/generate/batch", model_id=model_id, cached=False)

    request_id = getattr(request.state, "request_id", None)

    params_fp = fingerprint_batch_params(body)
    redis = get_redis_from_request(request)

    results: list[BatchGenerateResult] = []
    all_cached = True if body.cache else False

    async with db_session.get_sessionmaker()() as session:
        for prompt in body.prompts:
            item_start = time.time()
            prompt_hash = hash_prompt(prompt)
            redis_key = make_redis_key(model_id, prompt_hash, params_fp)

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
                    max_new_tokens=body.max_new_tokens,
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
                params_json=body.model_dump(exclude={"prompts", "model"}, exclude_none=True),
                prompt=prompt,
                output=output,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                commit=False,
            )

            results.append(
                BatchGenerateResult(
                    output=output,
                    cached=cached_flag,
                    prompt_tokens=int(prompt_tokens or 0),
                    completion_tokens=int(completion_tokens or 0),
                )
            )

        await session.commit()

    request.state.cached = all_cached
    return BatchGenerateResponse(model=model_id, results=results)