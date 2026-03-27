from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

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
    load_schema,
)
from llm_server.core.time import request_latency_ms
from llm_server.core.validation import (
    DependencyMissingError,
    JSONSchemaValidationError,
    validate_jsonschema,
)
from llm_server.services.api_deps.core.cache_keys import (
    fingerprint_pydantic,
    make_extract_redis_key,
    sha32,
)
from llm_server.services.api_deps.enforcement.assessed_gate import require_assessed_gate
from llm_server.services.api_deps.enforcement.capabilities import require_capability
from llm_server.services.api_deps.enforcement.model_ready import require_inprocess_loaded_if_needed
from llm_server.services.api_deps.extract.constants import REDIS_TTL_SECONDS
from llm_server.services.api_deps.extract.json_parse import validate_first_matching
from llm_server.services.api_deps.extract.prompts import (
    build_extraction_prompt,
    build_repair_prompt,
)
from llm_server.services.api_deps.extract.stage import failure_stage_for_app_error, set_stage
from llm_server.services.api_deps.extract.truncation import maybe_raise_truncation_error
from llm_server.services.api_deps.generate.generate_policy import apply_generate_cap
from llm_server.services.api_deps.generate.generate_runner import run_generate_rich_offloop
from llm_server.services.api_deps.generate.token_counting import count_tokens_split
from llm_server.services.api_deps.routing.models import resolve_model
from llm_server.services.llm_runtime.inference import (
    CacheSpec,
    get_cached_output,
    record_token_metrics,
    set_request_meta,
    write_cache,
    write_inference_log,
)
from llm_server.services.limits.generate_gating import get_generate_gate
from llm_server.telemetry.traces import (
    record_trace_event_best_effort,
    set_trace_meta,
    trace_id_from_ctx,
    trace_job_id_from_ctx,
)


class InternalRequestContext:
    def __init__(
        self,
        *,
        app: Any,
        route: str,
        request_id: str | None,
        client_host: str | None = None,
        trace_id: str | None = None,
        trace_job_id: str | None = None,
    ):
        self.app = app
        self.state = SimpleNamespace(
            route=route,
            model_id="unknown",
            cached=False,
            request_id=request_id,
            trace_id=trace_id or request_id,
            trace_job_id=trace_job_id,
            api_key="",
            start_ts=None,
        )
        self.client = SimpleNamespace(host=client_host) if client_host else None
        self.url = SimpleNamespace(path=route)


@dataclass
class ExtractExecutionResult:
    schema_id: str
    model: str
    data: dict[str, Any]
    cached: bool
    repair_attempted: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    policy_generate_max_new_tokens_cap: int | None
    effective_max_new_tokens: int | None
    requested_max_new_tokens: int | None
    clamped: bool


def _client_host(ctx: Any) -> str | None:
    try:
        return ctx.client.host if getattr(ctx, "client", None) else None
    except Exception:
        return None


def set_api_key_meta(ctx: Any, api_key: str) -> None:
    try:
        ctx.state.api_key = api_key
    except Exception:
        pass


async def _trace(
    ctx: Any,
    *,
    event_name: str,
    route: str,
    stage: str | None,
    status: str,
    model_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    if event_name in {"extract.completed", "extract.failed"}:
        try:
            ctx.state.trace_terminal_emitted = True
        except Exception:
            pass
    await record_trace_event_best_effort(
        trace_id=trace_id_from_ctx(ctx),
        event_name=event_name,
        route=route,
        stage=stage,
        status=status,
        request_id=getattr(getattr(ctx, "state", None), "request_id", None),
        job_id=trace_job_id_from_ctx(ctx),
        model_id=model_id,
        details=details,
    )


def validate_extract_submission(
    *,
    ctx: Any,
    body: Any,
    llm: Any,
) -> tuple[str, Any]:
    model_id, model = resolve_model(llm, body.model, capability="extract", request=ctx)
    require_capability(model_id, "extract", request=ctx)
    require_assessed_gate(request=ctx, model_id=model_id, capability="extract")
    try:
        load_schema(body.schema_id)
    except SchemaNotFoundError as e:
        raise AppError(
            code=e.code,
            message=e.message,
            status_code=404,
            extra={"schema_id": e.schema_id, "stage": "load_schema"},
        ) from e
    except SchemaLoadError as e:
        raise AppError(
            code=e.code,
            message=e.message,
            status_code=500,
            extra={"schema_id": e.schema_id, "stage": "load_schema"},
        ) from e
    return model_id, model


async def execute_extract(
    *,
    ctx: Any,
    body: Any,
    api_key: Any,
    llm: Any,
    session: Any,
    redis: Any | None,
    route_label: str = "/v1/extract",
) -> ExtractExecutionResult:
    stage = "start"
    set_stage(ctx, stage)
    request_id = getattr(getattr(ctx, "state", None), "request_id", None)
    set_trace_meta(ctx)
    set_api_key_meta(ctx, getattr(api_key, "key", ""))
    await _trace(
        ctx,
        event_name="extract.accepted",
        route=route_label,
        stage=stage,
        status="accepted",
        details={
            "schema_id": body.schema_id,
            "requested_model_id": body.model,
            "cache": bool(body.cache),
            "repair": bool(body.repair),
            "requested_max_new_tokens": body.max_new_tokens,
        },
    )
    try:
        stage = "resolve_model"
        set_stage(ctx, stage)
        model_id, model = resolve_model(llm, body.model, capability="extract", request=ctx)
        await _trace(
            ctx,
            event_name="extract.model_resolved",
            route=route_label,
            stage=stage,
            status="ok",
            model_id=model_id,
            details={"schema_id": body.schema_id, "requested_model_id": body.model},
        )

        stage = "require_capability"
        set_stage(ctx, stage)
        require_capability(model_id, "extract", request=ctx)

        stage = "assessed_gate"
        set_stage(ctx, stage)
        require_assessed_gate(request=ctx, model_id=model_id, capability="extract")

        stage = "enforcement"
        set_stage(ctx, stage)
        await require_inprocess_loaded_if_needed(request=ctx, model_id=model_id, backend_obj=model)

        set_request_meta(ctx, route=route_label, model_id=model_id, cached=False)
        EXTRACTION_REQUESTS.labels(schema_id=body.schema_id, model_id=model_id).inc()

        stage = "load_schema"
        set_stage(ctx, stage)
        try:
            schema = load_schema(body.schema_id)
        except SchemaNotFoundError as e:
            raise AppError(
                code=e.code,
                message=e.message,
                status_code=404,
                extra={"schema_id": e.schema_id, "stage": "load_schema"},
            ) from e
        except SchemaLoadError as e:
            raise AppError(
                code=e.code,
                message=e.message,
                status_code=500,
                extra={"schema_id": e.schema_id, "stage": "load_schema"},
            ) from e

        stage = "apply_policy_clamp"
        set_stage(ctx, stage)
        effective_max_new_tokens, policy_cap, clamped = apply_generate_cap(
            ctx,
            model_id=model_id,
            requested_max_new_tokens=body.max_new_tokens,
        )
        applied_cap = policy_cap

        stage = "build_cache_keys"
        set_stage(ctx, stage)
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

        stage = "cache_read"
        set_stage(ctx, stage)
        cached_out, cached_flag, layer = await get_cached_output(
            session,
            redis,
            cache=cache,
            kind="single",
            enabled=bool(body.cache),
        )
        await _trace(
            ctx,
            event_name="extract.cache_lookup",
            route=route_label,
            stage=stage,
            status="hit" if cached_flag and isinstance(cached_out, str) else "miss",
            model_id=model_id,
            details={"cache_enabled": bool(body.cache), "hit": bool(cached_flag), "layer": layer},
        )
        if isinstance(cached_out, str) and cached_flag:
            stage = "cache_validate"
            set_stage(ctx, stage)
            data: dict[str, Any] | None
            try:
                data_obj = json.loads(cached_out)
                if not isinstance(data_obj, dict):
                    raise ValueError("Expected object")
                validate_jsonschema(schema, data_obj)
                data = data_obj
            except DependencyMissingError as e:
                raise AppError(
                    code=e.code, message=e.message, status_code=500, extra={"stage": stage}
                ) from e
            except (JSONSchemaValidationError, Exception):
                data = None

            if isinstance(data, dict):
                EXTRACTION_CACHE_HITS.labels(
                    schema_id=body.schema_id,
                    model_id=model_id,
                    layer=(layer or "db"),
                ).inc()
                ctx.state.cached = True
                await _trace(
                    ctx,
                    event_name="extract.validation_completed",
                    route=route_label,
                    stage=stage,
                    status="ok",
                    model_id=model_id,
                    details={"cached": True, "repair_attempted": False},
                )
                latency = request_latency_ms(ctx)
                await write_inference_log(
                    session,
                    api_key=api_key.key,
                    request_id=request_id,
                    trace_id=trace_id_from_ctx(ctx),
                    job_id=trace_job_id_from_ctx(ctx),
                    route=route_label,
                    client_host=_client_host(ctx),
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
                await _trace(
                    ctx,
                    event_name="extract.logged",
                    route=route_label,
                    stage="log_cached",
                    status="ok",
                    model_id=model_id,
                    details={"cached": True},
                )
                await _trace(
                    ctx,
                    event_name="extract.completed",
                    route=route_label,
                    stage="complete",
                    status="completed",
                    model_id=model_id,
                    details={"cached": True, "repair_attempted": False},
                )
                return ExtractExecutionResult(
                    schema_id=body.schema_id,
                    model=model_id,
                    data=data,
                    cached=True,
                    repair_attempted=False,
                    prompt_tokens=None,
                    completion_tokens=None,
                    policy_generate_max_new_tokens_cap=policy_cap,
                    effective_max_new_tokens=effective_max_new_tokens,
                    requested_max_new_tokens=body.max_new_tokens,
                    clamped=clamped,
                )

        stage = "build_prompt"
        set_stage(ctx, stage)
        prompt = build_extraction_prompt(body.schema_id, schema, body.text)

        stage = "model_generate"
        set_stage(ctx, stage)
        gate = get_generate_gate()

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
        await _trace(
            ctx,
            event_name="extract.generate_completed",
            route=route_label,
            stage=stage,
            status="ok",
            model_id=model_id,
            details={
                "requested_max_new_tokens": body.max_new_tokens,
                "effective_max_new_tokens": effective_max_new_tokens,
                "policy_generate_max_new_tokens_cap": policy_cap,
                "clamped": clamped,
            },
        )

        if isinstance(gen_result, tuple) and len(gen_result) == 2:
            output = gen_result[0] if isinstance(gen_result[0], str) else str(gen_result[0])
            usage_from_backend = gen_result[1] if isinstance(gen_result[1], dict) else None
        else:
            output = gen_result if isinstance(gen_result, str) else str(gen_result)
            usage_from_backend = None

        stage = "truncation_check"
        set_stage(ctx, stage)
        maybe_raise_truncation_error(
            raw_output=output,
            effective_max_new_tokens=effective_max_new_tokens,
            applied_cap=applied_cap,
            stage=stage,
        )

        repair_attempted = False
        stage = "validate_output"
        set_stage(ctx, stage)
        try:
            data = validate_first_matching(schema, output)
        except AppError as e:
            st = failure_stage_for_app_error(e, is_repair=False)
            if st is not None:
                EXTRACTION_VALIDATION_FAILURES.labels(
                    schema_id=body.schema_id, model_id=model_id, stage=st
                ).inc()
            if not body.repair:
                if isinstance(e.extra, dict):
                    e.extra.setdefault("stage", st or "validate_output")
                else:
                    e.extra = {"stage": st or "validate_output"}  # type: ignore[assignment]
                raise

            repair_attempted = True
            EXTRACTION_REPAIR.labels(
                schema_id=body.schema_id, model_id=model_id, outcome="attempted"
            ).inc()

            stage = "repair_prompt"
            set_stage(ctx, stage)
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
            set_stage(ctx, stage)

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
                repaired = (
                    repair_result[0] if isinstance(repair_result[0], str) else str(repair_result[0])
                )
                repair_usage_from_backend = (
                    repair_result[1] if isinstance(repair_result[1], dict) else None
                )
            else:
                repaired = repair_result if isinstance(repair_result, str) else str(repair_result)
                repair_usage_from_backend = None

            stage = "repair_truncation_check"
            set_stage(ctx, stage)
            maybe_raise_truncation_error(
                raw_output=repaired,
                effective_max_new_tokens=effective_max_new_tokens,
                applied_cap=applied_cap,
                stage=stage,
            )

            stage = "repair_validate"
            set_stage(ctx, stage)
            try:
                data = validate_first_matching(schema, repaired)
                EXTRACTION_REPAIR.labels(
                    schema_id=body.schema_id, model_id=model_id, outcome="success"
                ).inc()
            except AppError as e2:
                st2 = failure_stage_for_app_error(e2, is_repair=True)
                if st2 is not None:
                    EXTRACTION_VALIDATION_FAILURES.labels(
                        schema_id=body.schema_id, model_id=model_id, stage=st2
                    ).inc()
                EXTRACTION_REPAIR.labels(
                    schema_id=body.schema_id, model_id=model_id, outcome="failure"
                ).inc()
                if isinstance(e2.extra, dict):
                    e2.extra.setdefault("stage", st2 or "repair_validate")
                else:
                    e2.extra = {"stage": st2 or "repair_validate"}  # type: ignore[assignment]
                raise
            usage_from_backend = repair_usage_from_backend or usage_from_backend

        await _trace(
            ctx,
            event_name="extract.validation_completed",
            route=route_label,
            stage=stage,
            status="ok",
            model_id=model_id,
            details={"cached": False, "repair_attempted": repair_attempted},
        )

        ctx.state.cached = False
        latency = request_latency_ms(ctx)
        out_json = json.dumps(data, ensure_ascii=False)

        prompt_tokens, completion_tokens = count_tokens_split(
            model=model,
            model_id=model_id,
            prompt=prompt,
            completion=out_json,
            usage_from_backend=usage_from_backend,
        )
        record_token_metrics(model_id, prompt_tokens, completion_tokens)

        stage = "cache_write"
        set_stage(ctx, stage)
        await write_cache(
            session,
            redis,
            cache=cache,
            output=out_json,
            enabled=bool(body.cache),
        )
        await _trace(
            ctx,
            event_name="extract.cache_written",
            route=route_label,
            stage=stage,
            status="ok",
            model_id=model_id,
            details={"cache_enabled": bool(body.cache)},
        )

        stage = "log_uncached"
        set_stage(ctx, stage)
        await write_inference_log(
            session,
            api_key=api_key.key,
            request_id=request_id,
            trace_id=trace_id_from_ctx(ctx),
            job_id=trace_job_id_from_ctx(ctx),
            route=route_label,
            client_host=_client_host(ctx),
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
        await _trace(
            ctx,
            event_name="extract.logged",
            route=route_label,
            stage=stage,
            status="ok",
            model_id=model_id,
            details={"cached": False},
        )
        await _trace(
            ctx,
            event_name="extract.completed",
            route=route_label,
            stage="complete",
            status="completed",
            model_id=model_id,
            details={
                "cached": False,
                "repair_attempted": repair_attempted,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

        return ExtractExecutionResult(
            schema_id=body.schema_id,
            model=model_id,
            data=data,
            cached=False,
            repair_attempted=repair_attempted,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            policy_generate_max_new_tokens_cap=policy_cap,
            effective_max_new_tokens=effective_max_new_tokens,
            requested_max_new_tokens=body.max_new_tokens,
            clamped=clamped,
        )
    except AppError as e:
        details = {"error_code": e.code}
        if isinstance(e.extra, dict):
            details["error_stage"] = str(e.extra.get("stage") or stage)
        await _trace(
            ctx,
            event_name="extract.failed",
            route=route_label,
            stage=str(details.get("error_stage") or stage),
            status="failed",
            model_id=getattr(getattr(ctx, "state", None), "model_id", None),
            details=details,
        )
        raise
    except Exception:
        await _trace(
            ctx,
            event_name="extract.failed",
            route=route_label,
            stage=stage,
            status="failed",
            model_id=getattr(getattr(ctx, "state", None), "model_id", None),
            details={"error_code": "internal_error", "error_stage": stage},
        )
        raise
