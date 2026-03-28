# server/src/llm_server/api/generate.py
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from llm_server.core.errors import AppError
from llm_server.core.redis import get_redis_from_request
from llm_server.core.tracing import bind_request_span, start_child_span
from llm_server.core.time import request_latency_ms
import llm_server.db.session as db_session  # module import so tests can patch session wiring

from llm_server.api.dependencies.auth import get_api_key
from llm_server.core.cache_keys import (
    fingerprint_pydantic,
    make_cache_redis_key,
    sha32,
)
from llm_server.core.model_load_mode import effective_model_load_mode_from_request
from llm_server.runtime.assessment import require_assessed_gate
from llm_server.runtime.capabilities import require_capability
from llm_server.runtime.model_loading import require_inprocess_loaded_if_needed
from llm_server.services.limits.generate_gating import get_generate_gate
from llm_server.services.llm_runtime.access import get_llm
from llm_server.services.llm_runtime.inference import (
    CacheSpec,
    get_cached_output,
    record_token_metrics,
    set_request_meta,
    write_cache,
    write_inference_log,
)
from llm_server.runtime.generation import (
    apply_generate_cap,
    count_tokens_split,
    run_generate_rich_offloop,
)
from llm_server.runtime.routing import resolve_model
from llm_server.telemetry.traces import trace_id_from_ctx, trace_job_id_from_ctx

router = APIRouter()

REDIS_TTL_SECONDS = 3600


class GenerateRequest(BaseModel):
    prompt: str
    model: str | None = Field(
        default=None, description="Optional model id override for multi-model routing"
    )
    cache: bool = True
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: list[str] | None = None


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
    bind_request_span(
        request,
        name="backend.generate",
        route="/v1/generate",
        attributes={"llm.requested_model_id": body.model},
    )

    llm: Any = get_llm(request)

    model_id, model = resolve_model(llm, body.model, capability="generate", request=request)
    bind_request_span(request, attributes={"llm.resolved_model_id": model_id})
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
    params_with_effective = body.model_dump(exclude={"prompt", "model"}, exclude_none=True) | {
        "requested_max_new_tokens": requested_max_new_tokens,
        "effective_max_new_tokens": effective_max_new_tokens,
        "policy_generate_max_new_tokens_cap": policy_cap,
        "clamped": clamped,
        "max_new_tokens": effective_max_new_tokens,
    }

    async with db_session.get_sessionmaker()() as session:
        with start_child_span(
            "generate.cache_lookup",
            request=request,
            attributes={"llm.resolved_model_id": model_id},
        ):
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
                trace_id=trace_id_from_ctx(request),
                job_id=trace_job_id_from_ctx(request),
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

        with start_child_span(
            "generate.model_call",
            request=request,
            attributes={"llm.resolved_model_id": model_id},
        ):
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
            trace_id=trace_id_from_ctx(request),
            job_id=trace_job_id_from_ctx(request),
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
