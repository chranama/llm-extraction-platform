# server/src/llm_server/api/generate.py
from __future__ import annotations

from typing import Any, List

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from llm_server.core.errors import AppError
from llm_server.core.redis import get_redis_from_request
from llm_server.core.time import monotonic_elapsed_ms, monotonic_now, request_latency_ms
import llm_server.db.session as db_session  # module import so tests can patch session wiring

from llm_server.services.api_deps.core.auth import get_api_key
from llm_server.services.api_deps.core.cache_keys import (
    fingerprint_pydantic,
    make_cache_redis_key,
    sha32,
)
from llm_server.services.api_deps.core.llm_access import get_llm
from llm_server.services.api_deps.core.model_load_mode import effective_model_load_mode_from_request
from llm_server.services.api_deps.enforcement.assessed_gate import require_assessed_gate
from llm_server.services.api_deps.enforcement.capabilities import require_capability
from llm_server.services.api_deps.enforcement.model_ready import require_inprocess_loaded_if_needed
from llm_server.services.api_deps.generate.generate_policy import apply_generate_cap
from llm_server.services.api_deps.generate.generate_runner import run_generate_rich_offloop
from llm_server.services.api_deps.generate.token_counting import count_tokens_split
from llm_server.services.api_deps.routing.models import resolve_model
from llm_server.services.limits.generate_gating import get_generate_gate
from llm_server.services.llm_runtime.inference import (
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
    requested_max_new_tokens: int | None = None
    effective_max_new_tokens: int | None = None
    policy_generate_max_new_tokens_cap: int | None = None
    clamped: bool | None = None


def _reject_if_mode_off(request: Request) -> None:
    """
    Enforce that /v1/generate* does not run when model_load_mode=off.

    NOTE:
      - In FastAPI, Depends(get_llm) would execute before entering the handler.
      - So we avoid Depends(get_llm) entirely and only call get_llm(request) after this check.
    """
    mode = effective_model_load_mode_from_request(request)
    if mode == "off":
        raise AppError(
            code="model_disabled",
            message="Model is disabled (model_load_mode=off).",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            extra={"model_load_mode": mode},
        )


@router.post("/v1/generate")
async def generate(
    request: Request,
    body: GenerateRequest,
    api_key=Depends(get_api_key),
):
    # IMPORTANT: avoid touching llm dependency in model_load_mode=off
    _reject_if_mode_off(request)

    llm: Any = get_llm(request)

    model_id, model = resolve_model(llm, body.model, capability="generate", request=request)
    require_capability(model_id, "generate", request=request)

    # NEW: assessed gate enforcement (policy-driven allow/block)
    require_assessed_gate(request=request, model_id=model_id, capability="generate")

    # Enforcement boundary (MODEL_LOAD_MODE semantics)
    await require_inprocess_loaded_if_needed(request=request, model_id=model_id, backend_obj=model)

    set_request_meta(request, route="/v1/generate", model_id=model_id, cached=False)
    request_id = getattr(request.state, "request_id", None)

    effective_max_new_tokens, policy_cap, clamped = apply_generate_cap(
        request,
        model_id=model_id,
        requested_max_new_tokens=body.max_new_tokens,
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
            return await run_generate_rich_offloop(
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
):
    # IMPORTANT: avoid touching llm dependency in model_load_mode=off
    _reject_if_mode_off(request)

    llm: Any = get_llm(request)

    model_id, model = resolve_model(llm, body.model, capability="generate", request=request)
    require_capability(model_id, "generate", request=request)

    # NEW: assessed gate enforcement (policy-driven allow/block)
    require_assessed_gate(request=request, model_id=model_id, capability="generate")

    # Enforcement boundary (MODEL_LOAD_MODE semantics)
    await require_inprocess_loaded_if_needed(request=request, model_id=model_id, backend_obj=model)

    set_request_meta(request, route="/v1/generate/batch", model_id=model_id, cached=False)
    request_id = getattr(request.state, "request_id", None)

    effective_batch_max_new, policy_cap, clamped = apply_generate_cap(
        request,
        model_id=model_id,
        requested_max_new_tokens=body.max_new_tokens,
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
                    return await run_generate_rich_offloop(
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