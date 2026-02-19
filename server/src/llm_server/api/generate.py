# server/src/llm_server/api/generate.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, List, Optional, Tuple

import anyio
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
from llm_server.core.time import monotonic_elapsed_ms, monotonic_now, request_latency_ms
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
from llm_server.services.limits.generate_gating import get_generate_gate

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
    requested_max_new_tokens: int | None = None
    effective_max_new_tokens: int | None = None
    policy_generate_max_new_tokens_cap: int | None = None
    clamped: bool | None = None


@lru_cache(maxsize=16)
def _get_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id, use_fast=True)


def _token_count_hf(model_id: str, prompt: str, completion: str | None) -> tuple[int | None, int | None]:
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


def _detect_backend_name(model: Any) -> str | None:
    try:
        v = getattr(model, "backend_name", None)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    except Exception:
        pass
    return None


def _llamacpp_tokenize(model: Any, texts: list[str]) -> list[list[int]] | None:
    try:
        client = getattr(model, "_client", None)
        tok_fn = getattr(client, "tokenize", None)
        if not callable(tok_fn):
            return None

        data = tok_fn(content=texts)
        toks = data.get("tokens") if isinstance(data, dict) else None
        if not isinstance(toks, list):
            return None

        if toks and all(isinstance(x, int) for x in toks):
            return [toks] if len(texts) == 1 else None

        if toks and all(isinstance(x, list) for x in toks):
            out: list[list[int]] = []
            for row in toks:
                if isinstance(row, list) and all(isinstance(x, int) for x in row):
                    out.append(row)
                else:
                    return None
            return out if len(out) == len(texts) else None

        if not toks:
            return [[] for _ in texts]

        return None
    except Exception:
        return None


def count_tokens_split(
    *,
    model: Any,
    model_id: str,
    prompt: str,
    completion: str | None,
    usage_from_backend: Any | None,
) -> tuple[int | None, int | None]:
    if os.getenv("TOKEN_COUNTING", "1").strip().lower() in {"0", "false", "no", "off"}:
        return None, None

    if isinstance(usage_from_backend, dict):
        pt = usage_from_backend.get("prompt_tokens")
        ct = usage_from_backend.get("completion_tokens")
        pt_i = int(pt) if isinstance(pt, int) else None
        ct_i = int(ct) if isinstance(ct, int) else None
        if pt_i is not None or ct_i is not None:
            return pt_i, ct_i

    if _detect_backend_name(model) == "llamacpp":
        texts: list[str] = [prompt]
        if completion is not None:
            texts.append(completion)

        toks = _llamacpp_tokenize(model, texts)
        if toks is not None:
            pt = len(toks[0]) if len(toks) >= 1 else None
            ct = len(toks[1]) if (completion is not None and len(toks) >= 2) else (0 if completion is not None else None)
            return pt, ct

        return None, None

    return _token_count_hf(model_id, prompt, completion)


def _normalize_positive_int(x: Any) -> Optional[int]:
    if x is None or isinstance(x, bool):
        return None
    try:
        i = int(x)
    except Exception:
        return None
    return i if i > 0 else None


def _apply_generate_cap(
    request: Request,
    *,
    model_id: str,
    requested: int | None,
) -> Tuple[int | None, int | None, bool]:
    cap_raw = policy_generate_max_new_tokens_cap(model_id, request=request)
    cap_i = _normalize_positive_int(cap_raw)
    req_i = _normalize_positive_int(requested)

    if cap_i is None:
        return requested, None, False

    if req_i is None:
        return cap_i, cap_i, False

    eff = min(req_i, cap_i)
    clamped = req_i > cap_i
    return eff, cap_i, clamped


async def _run_generate_rich_offloop(model: Any, **kwargs: Any) -> tuple[str, dict[str, Any] | None]:
    def _run() -> tuple[str, dict[str, Any] | None]:
        gen_rich = getattr(model, "generate_rich", None)
        if callable(gen_rich):
            r = gen_rich(**kwargs)
            text = str(getattr(r, "text", "") or "")
            usage_obj = getattr(r, "usage", None)
            usage_dict: dict[str, Any] | None = None
            if usage_obj is not None:
                try:
                    pt = getattr(usage_obj, "prompt_tokens", None)
                    ct = getattr(usage_obj, "completion_tokens", None)
                    tt = getattr(usage_obj, "total_tokens", None)
                    usage_dict = {
                        "prompt_tokens": int(pt) if isinstance(pt, int) else None,
                        "completion_tokens": int(ct) if isinstance(ct, int) else None,
                        "total_tokens": int(tt) if isinstance(tt, int) else None,
                    }
                except Exception:
                    usage_dict = None
            return text, usage_dict

        gen = getattr(model, "generate", None)
        if not callable(gen):
            return "", None
        out = gen(**kwargs)
        return (out if isinstance(out, str) else str(out)), None

    return await anyio.to_thread.run_sync(_run)


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

    effective_max_new_tokens, policy_cap, clamped = _apply_generate_cap(
        request,
        model_id=model_id,
        requested=body.max_new_tokens,
    )

    prompt_hash = sha32(body.prompt)
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

    redis = get_redis_from_request(request)

    requested_max_new_tokens = body.max_new_tokens
    params_with_effective = (
        body.model_dump(exclude={"prompt", "model"}, exclude_none=True)
        | {
            "requested_max_new_tokens": requested_max_new_tokens,
            "effective_max_new_tokens": effective_max_new_tokens,
            "policy_generate_max_new_tokens_cap": policy_cap,
            "clamped": clamped,
            "max_new_tokens": effective_max_new_tokens,
        }
    )

    async with db_session.get_sessionmaker()() as session:
        cached_out, cached_flag, _layer = await get_cached_output(
            session,
            redis,
            cache=cache,
            kind="single",
            enabled=bool(body.cache),
        )

        if isinstance(cached_out, str) and cached_flag:
            request.state.cached = True
            latency = request_latency_ms(request)

            prompt_tokens, completion_tokens = count_tokens_split(
                model=model,
                model_id=model_id,
                prompt=body.prompt,
                completion=cached_out,
                usage_from_backend=None,
            )
            record_token_metrics(model_id, prompt_tokens, completion_tokens)

            await write_inference_log(
                session,
                api_key=api_key.key,
                request_id=request_id,
                route="/v1/generate",
                client_host=request.client.host if request.client else None,
                model_id=model_id,
                params_json=params_with_effective,
                prompt=body.prompt,
                output=cached_out,
                latency_ms=latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                status_code=200,
                cached=True,
                error_code=None,
                error_stage=None,
                commit=True,
            )
            return {
                "model": model_id,
                "output": cached_out,
                "cached": True,
                "requested_max_new_tokens": requested_max_new_tokens,
                "effective_max_new_tokens": effective_max_new_tokens,
                "policy_generate_max_new_tokens_cap": policy_cap,
                "clamped": clamped,
            }

        gate = get_generate_gate()

        async def _do_generate() -> Any:
            return await _run_generate_rich_offloop(
                model,
                prompt=body.prompt,
                max_new_tokens=effective_max_new_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
                top_k=body.top_k,
                stop=body.stop,
            )

        result = await gate.run(
            _do_generate,
            request_id=str(request_id) if request_id is not None else None,
            model_id=model_id,
        )

        if isinstance(result, tuple) and len(result) == 2:
            output_text = result[0] if isinstance(result[0], str) else str(result[0])
            usage_from_backend = result[1] if isinstance(result[1], dict) else None
        else:
            output_text = result if isinstance(result, str) else str(result)
            usage_from_backend = None

        request.state.cached = False
        latency = request_latency_ms(request)

        prompt_tokens, completion_tokens = count_tokens_split(
            model=model,
            model_id=model_id,
            prompt=body.prompt,
            completion=output_text,
            usage_from_backend=usage_from_backend,
        )
        record_token_metrics(model_id, prompt_tokens, completion_tokens)

        await write_cache(
            session,
            redis,
            cache=cache,
            output=output_text,
            enabled=bool(body.cache),
        )

        await write_inference_log(
            session,
            api_key=api_key.key,
            request_id=request_id,
            route="/v1/generate",
            client_host=request.client.host if request.client else None,
            model_id=model_id,
            params_json=params_with_effective,
            prompt=body.prompt,
            output=output_text,
            latency_ms=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            status_code=200,
            cached=False,
            error_code=None,
            error_stage=None,
            commit=True,
        )

        return {
            "model": model_id,
            "output": output_text,
            "cached": False,
            "requested_max_new_tokens": requested_max_new_tokens,
            "effective_max_new_tokens": effective_max_new_tokens,
            "policy_generate_max_new_tokens_cap": policy_cap,
            "clamped": clamped,
        }


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

    effective_batch_max_new, policy_cap, clamped = _apply_generate_cap(
        request,
        model_id=model_id,
        requested=body.max_new_tokens,
    )
    if effective_batch_max_new is None:
        effective_batch_max_new = int(body.max_new_tokens)

    params_fp = fingerprint_pydantic(
        body.model_copy(update={"max_new_tokens": effective_batch_max_new}),
        exclude={"prompts", "model"},
    )
    redis = get_redis_from_request(request)

    results: list[BatchGenerateResult] = []
    all_cached = True if body.cache else False

    params_with_effective = (
        body.model_dump(exclude={"prompts", "model"}, exclude_none=True)
        | {
            "requested_max_new_tokens": body.max_new_tokens,
            "effective_max_new_tokens": effective_batch_max_new,
            "policy_generate_max_new_tokens_cap": policy_cap,
            "clamped": clamped,
            "max_new_tokens": effective_batch_max_new,
        }
    )

    gate = get_generate_gate()

    async with db_session.get_sessionmaker()() as session:
        for prompt in body.prompts:
            # Per-item latency is an internal span, not the request lifecycle baseline.
            item_t0 = monotonic_now()

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

            usage_from_backend: dict[str, Any] | None = None

            if isinstance(out, str) and cached_flag:
                output = out
            else:
                async def _do_generate_one() -> Any:
                    return await _run_generate_rich_offloop(
                        model,
                        prompt=prompt,
                        max_new_tokens=effective_batch_max_new,
                        temperature=body.temperature,
                        top_p=body.top_p,
                        top_k=body.top_k,
                        stop=body.stop,
                    )

                r = await gate.run(
                    _do_generate_one,
                    request_id=str(request_id) if request_id is not None else None,
                    model_id=model_id,
                )

                if isinstance(r, tuple) and len(r) == 2:
                    output = r[0] if isinstance(r[0], str) else str(r[0])
                    usage_from_backend = r[1] if isinstance(r[1], dict) else None
                else:
                    output = r if isinstance(r, str) else str(r)
                    usage_from_backend = None

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

            latency_ms_item = monotonic_elapsed_ms(item_t0)

            prompt_tokens, completion_tokens = count_tokens_split(
                model=model,
                model_id=model_id,
                prompt=prompt,
                completion=output,
                usage_from_backend=None if cached_flag else usage_from_backend,
            )
            record_token_metrics(model_id, prompt_tokens, completion_tokens)

            await write_inference_log(
                session,
                api_key=api_key.key,
                request_id=request_id,
                route="/v1/generate/batch",
                client_host=request.client.host if request.client else None,
                model_id=model_id,
                params_json=params_with_effective,
                prompt=prompt,
                output=output,
                latency_ms=latency_ms_item,
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
    return BatchGenerateResponse(
        model=model_id,
        results=results,
        requested_max_new_tokens=body.max_new_tokens,
        effective_max_new_tokens=effective_batch_max_new,
        policy_generate_max_new_tokens_cap=policy_cap,
        clamped=clamped,
    )