# server/src/llm_server/api/extract.py
from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from llm_server.core.errors import AppError
from llm_server.core.metrics import (
    EXTRACTION_CACHE_HITS,
    EXTRACTION_REPAIR,
    EXTRACTION_REQUESTS,
    EXTRACTION_VALIDATION_FAILURES,
)
from llm_server.core.schema_registry import (
    SchemaLoadError,
    SchemaNotFoundError,
    list_schemas,
    load_schema,
)
from llm_server.core.time import request_latency_ms
from llm_server.core.validation import (
    DependencyMissingError,
    JSONSchemaValidationError,
    validate_jsonschema,
)

import llm_server.db.session as db_session  # module import so tests can patch session wiring

from llm_server.services.api_deps.core.auth import get_api_key
from llm_server.services.api_deps.core.cache_keys import (
    fingerprint_pydantic,
    make_extract_redis_key,
    sha32,
)
from llm_server.services.api_deps.core.llm_access import get_llm
from llm_server.services.api_deps.enforcement.assessed_gate import require_assessed_gate
from llm_server.services.api_deps.enforcement.capabilities import require_capability
from llm_server.services.api_deps.enforcement.model_ready import require_inprocess_loaded_if_needed
from llm_server.services.api_deps.generate.generate_policy import apply_generate_cap
from llm_server.services.api_deps.generate.generate_runner import run_generate_rich_offloop
from llm_server.services.api_deps.generate.token_counting import count_tokens_split
from llm_server.services.api_deps.routing.models import resolve_model
from llm_server.services.api_deps.extract.constants import REDIS_TTL_SECONDS
from llm_server.services.api_deps.extract.prompts import build_extraction_prompt, build_repair_prompt
from llm_server.services.api_deps.extract.json_parse import validate_first_matching
from llm_server.services.api_deps.extract.truncation import maybe_raise_truncation_error
from llm_server.services.api_deps.extract.stage import failure_stage_for_app_error, set_stage

from llm_server.services.llm_runtime.inference import (
    CacheSpec,
    get_cached_output,
    record_token_metrics,
    set_request_meta,
    write_cache,
    write_inference_log,
)
from llm_server.services.limits.generate_gating import get_generate_gate

router = APIRouter()


class ExtractRequest(BaseModel):
    schema_id: str = Field(..., description="Schema id (e.g. ticket_v1, invoice_v1, receipt_v1)")
    text: str = Field(..., description="Raw text or OCR text to extract from")

    model: str | None = Field(default=None, description="Optional model id override for multi-model routing")

    max_new_tokens: int | None = 512
    temperature: float | None = 0.0

    cache: bool = True
    repair: bool = True


class ExtractResponse(BaseModel):
    schema_id: str
    model: str
    data: dict[str, Any]
    cached: bool
    repair_attempted: bool


@router.get("/v1/schemas")
async def schemas_index(api_key=Depends(get_api_key)):
    return [{"schema_id": s.schema_id, "title": s.title, "description": s.description} for s in list_schemas()]


@router.get("/v1/schemas/{schema_id}", response_model=dict)
async def schema_detail(schema_id: str, api_key=Depends(get_api_key)):
    try:
        schema = load_schema(schema_id)
    except SchemaNotFoundError as e:
        raise AppError(
            code=e.code,
            message=e.message,
            status_code=status.HTTP_404_NOT_FOUND,
            extra={"schema_id": e.schema_id},
        ) from e
    except SchemaLoadError as e:
        raise AppError(
            code=e.code,
            message=e.message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            extra={"schema_id": e.schema_id},
        ) from e

    return JSONResponse(content=schema)


@router.post("/v1/extract", response_model=ExtractResponse)
async def extract(
    request: Request,
    body: ExtractRequest,
    api_key=Depends(get_api_key),
    llm: Any = Depends(get_llm),
):
    stage = "start"
    set_stage(request, stage)

    request_id = getattr(request.state, "request_id", None)

    try:
        stage = "resolve_model"
        set_stage(request, stage)
        model_id, model = resolve_model(llm, body.model, capability="extract", request=request)

        stage = "require_capability"
        set_stage(request, stage)
        require_capability(model_id, "extract", request=request)

        # NEW: assessed gate enforcement (policy-driven allow/block)
        stage = "assessed_gate"
        set_stage(request, stage)
        require_assessed_gate(request=request, model_id=model_id, capability="extract")

        # Enforcement boundary (MODEL_LOAD_MODE semantics)
        stage = "enforcement"
        set_stage(request, stage)
        await require_inprocess_loaded_if_needed(request=request, model_id=model_id, backend_obj=model)

        set_request_meta(request, route="/v1/extract", model_id=model_id, cached=False)
        EXTRACTION_REQUESTS.labels(schema_id=body.schema_id, model_id=model_id).inc()

        stage = "load_schema"
        set_stage(request, stage)
        try:
            schema = load_schema(body.schema_id)
        except SchemaNotFoundError as e:
            raise AppError(
                code=e.code,
                message=e.message,
                status_code=status.HTTP_404_NOT_FOUND,
                extra={"schema_id": e.schema_id, "stage": "load_schema"},
            ) from e
        except SchemaLoadError as e:
            raise AppError(
                code=e.code,
                message=e.message,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                extra={"schema_id": e.schema_id, "stage": "load_schema"},
            ) from e

        # Apply the SAME generate cap policy as /v1/generate
        stage = "apply_policy_clamp"
        set_stage(request, stage)
        effective_max_new_tokens, policy_cap, clamped = apply_generate_cap(
            request,
            model_id=model_id,
            requested_max_new_tokens=body.max_new_tokens,
        )
        applied_cap = policy_cap

        stage = "build_cache_keys"
        set_stage(request, stage)
        prompt_hash = sha32(f"{body.schema_id}\n{body.text}")
        params_fp = fingerprint_pydantic(
            body.model_copy(update={"max_new_tokens": effective_max_new_tokens}),
            exclude={"text", "model", "cache", "repair"},
        )
        redis_key = make_extract_redis_key(model_id, prompt_hash, params_fp)

        cache = CacheSpec(
            model_id=model_id,
            prompt=body.text,
            prompt_hash=prompt_hash,
            params_fp=params_fp,
            redis_key=redis_key,
            redis_ttl_seconds=REDIS_TTL_SECONDS,
        )

        stage = "db_session_open"
        set_stage(request, stage)

        gate = get_generate_gate()

        async with db_session.get_sessionmaker()() as session:
            stage = "cache_read"
            set_stage(request, stage)
            cached_out, cached_flag, layer = await get_cached_output(
                session,
                None,
                cache=cache,
                kind="single",
                enabled=bool(body.cache),
            )

            if isinstance(cached_out, str) and cached_flag:
                stage = "cache_validate"
                set_stage(request, stage)
                data: dict[str, Any] | None
                try:
                    data_obj = json.loads(cached_out)
                    if not isinstance(data_obj, dict):
                        raise ValueError("Expected object")
                    validate_jsonschema(schema, data_obj)
                    data = data_obj
                except DependencyMissingError as e:
                    raise AppError(code=e.code, message=e.message, status_code=500, extra={"stage": stage}) from e
                except (JSONSchemaValidationError, Exception):
                    data = None

                if isinstance(data, dict):
                    EXTRACTION_CACHE_HITS.labels(
                        schema_id=body.schema_id,
                        model_id=model_id,
                        layer=(layer or "db"),
                    ).inc()

                    request.state.cached = True
                    latency = request_latency_ms(request)

                    stage = "log_cached"
                    set_stage(request, stage)
                    await write_inference_log(
                        session,
                        api_key=api_key.key,
                        request_id=request_id,
                        route="/v1/extract",
                        client_host=request.client.host if request.client else None,
                        model_id=model_id,
                        params_json={
                            "schema_id": body.schema_id,
                            "cache": True,
                            "repair": body.repair,
                            "requested_max_new_tokens": body.max_new_tokens,
                            "effective_max_new_tokens": effective_max_new_tokens,
                            "policy_generate_max_new_tokens_cap": policy_cap,
                            "clamped": clamped,
                        },
                        prompt=body.text,
                        output=json.dumps(data, ensure_ascii=False),
                        latency_ms=latency,
                        prompt_tokens=None,
                        completion_tokens=None,
                        status_code=200,
                        cached=True,
                        error_code=None,
                        error_stage=None,
                        commit=True,
                    )

                    return ExtractResponse(
                        schema_id=body.schema_id,
                        model=model_id,
                        data=data,
                        cached=True,
                        repair_attempted=False,
                    )

            stage = "build_prompt"
            set_stage(request, stage)
            prompt = build_extraction_prompt(body.schema_id, schema, body.text)

            stage = "model_generate"
            set_stage(request, stage)

            async def _do_generate() -> Any:
                return await run_generate_rich_offloop(
                    model,
                    prompt=prompt,
                    max_new_tokens=effective_max_new_tokens,
                    temperature=body.temperature,
                )

            gen_result = await gate.run(
                _do_generate,
                request_id=str(request_id) if request_id is not None else None,
                model_id=model_id,
            )

            if isinstance(gen_result, tuple) and len(gen_result) == 2:
                output = gen_result[0] if isinstance(gen_result[0], str) else str(gen_result[0])
                usage_from_backend = gen_result[1] if isinstance(gen_result[1], dict) else None
            else:
                output = gen_result if isinstance(gen_result, str) else str(gen_result)
                usage_from_backend = None

            stage = "truncation_check"
            set_stage(request, stage)
            maybe_raise_truncation_error(
                raw_output=output,
                effective_max_new_tokens=effective_max_new_tokens,
                applied_cap=applied_cap,
                stage=stage,
            )

            repair_attempted = False

            stage = "validate_output"
            set_stage(request, stage)
            try:
                data = validate_first_matching(schema, output)
            except AppError as e:
                st = failure_stage_for_app_error(e, is_repair=False)
                if st is not None:
                    EXTRACTION_VALIDATION_FAILURES.labels(schema_id=body.schema_id, model_id=model_id, stage=st).inc()

                if not body.repair:
                    if isinstance(e.extra, dict):
                        e.extra.setdefault("stage", st or "validate_output")
                    else:
                        e.extra = {"stage": st or "validate_output"}  # type: ignore[assignment]
                    raise

                repair_attempted = True
                EXTRACTION_REPAIR.labels(schema_id=body.schema_id, model_id=model_id, outcome="attempted").inc()

                stage = "repair_prompt"
                set_stage(request, stage)
                error_hint = json.dumps(
                    {"code": e.code, "message": e.message, **(e.extra or {})},
                    ensure_ascii=False,
                )

                repair_prompt = build_repair_prompt(
                    body.schema_id,
                    schema,
                    body.text,
                    bad_output=output,
                    error_hint=error_hint,
                )

                stage = "repair_generate"
                set_stage(request, stage)

                async def _do_repair() -> Any:
                    return await run_generate_rich_offloop(
                        model,
                        prompt=repair_prompt,
                        max_new_tokens=effective_max_new_tokens,
                        temperature=0.0,
                    )

                repair_result = await gate.run(
                    _do_repair,
                    request_id=str(request_id) if request_id is not None else None,
                    model_id=model_id,
                )

                if isinstance(repair_result, tuple) and len(repair_result) == 2:
                    repaired = repair_result[0] if isinstance(repair_result[0], str) else str(repair_result[0])
                    repair_usage_from_backend = repair_result[1] if isinstance(repair_result[1], dict) else None
                else:
                    repaired = repair_result if isinstance(repair_result, str) else str(repair_result)
                    repair_usage_from_backend = None

                stage = "repair_truncation_check"
                set_stage(request, stage)
                maybe_raise_truncation_error(
                    raw_output=repaired,
                    effective_max_new_tokens=effective_max_new_tokens,
                    applied_cap=applied_cap,
                    stage=stage,
                )

                stage = "repair_validate"
                set_stage(request, stage)
                try:
                    data = validate_first_matching(schema, repaired)
                    EXTRACTION_REPAIR.labels(schema_id=body.schema_id, model_id=model_id, outcome="success").inc()
                except AppError as e2:
                    st2 = failure_stage_for_app_error(e2, is_repair=True)
                    if st2 is not None:
                        EXTRACTION_VALIDATION_FAILURES.labels(schema_id=body.schema_id, model_id=model_id, stage=st2).inc()
                    EXTRACTION_REPAIR.labels(schema_id=body.schema_id, model_id=model_id, outcome="failure").inc()

                    if isinstance(e2.extra, dict):
                        e2.extra.setdefault("stage", st2 or "repair_validate")
                    else:
                        e2.extra = {"stage": st2 or "repair_validate"}  # type: ignore[assignment]
                    raise

                usage_from_backend = repair_usage_from_backend or usage_from_backend

            request.state.cached = False
            latency = request_latency_ms(request)

            prompt_tokens, completion_tokens = count_tokens_split(
                model=model,
                model_id=model_id,
                prompt=prompt,
                completion=json.dumps(data, ensure_ascii=False),
                usage_from_backend=usage_from_backend,
            )
            record_token_metrics(model_id, prompt_tokens, completion_tokens)

            stage = "cache_write"
            set_stage(request, stage)
            out_json = json.dumps(data, ensure_ascii=False)
            await write_cache(
                session,
                None,
                cache=cache,
                output=out_json,
                enabled=bool(body.cache),
            )

            stage = "log_uncached"
            set_stage(request, stage)
            await write_inference_log(
                session,
                api_key=api_key.key,
                request_id=request_id,
                route="/v1/extract",
                client_host=request.client.host if request.client else None,
                model_id=model_id,
                params_json={
                    "schema_id": body.schema_id,
                    "cache": body.cache,
                    "repair": body.repair,
                    "requested_max_new_tokens": body.max_new_tokens,
                    "effective_max_new_tokens": effective_max_new_tokens,
                    "policy_generate_max_new_tokens_cap": policy_cap,
                    "clamped": clamped,
                },
                prompt=body.text,
                output=out_json,
                latency_ms=latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                status_code=200,
                cached=False,
                error_code=None,
                error_stage=None,
                commit=True,
            )

            return ExtractResponse(
                schema_id=body.schema_id,
                model=model_id,
                data=data,
                cached=False,
                repair_attempted=repair_attempted,
            )

    except AppError:
        raise
    except Exception as e:
        raise AppError(
            code="internal_error",
            message="An unexpected error occurred",
            status_code=500,
            extra={"stage": stage, "exc_type": type(e).__name__},
        ) from e