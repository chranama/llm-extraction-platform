from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import llm_server.db.session as db_session

from llm_server.core.errors import AppError
from llm_server.domain.outcomes import RunOutcome
from llm_server.domain.runs import ExtractionRun, RunIdentity, RunPolicySnapshot
from llm_server.services.extract_execution import ExtractExecutionResult, execute_extract


@dataclass(frozen=True, slots=True)
class RunExtractResult:
    run: ExtractionRun
    response: ExtractExecutionResult


def build_extraction_run(
    *,
    ctx: Any,
    body: Any,
    route_label: str = "/v1/extract",
) -> ExtractionRun:
    state = getattr(ctx, "state", None)
    request_id = getattr(state, "request_id", None)
    trace_id = getattr(state, "trace_id", None) or request_id
    job_id = getattr(state, "trace_job_id", None)

    return ExtractionRun(
        identity=RunIdentity(
            request_id=request_id,
            trace_id=trace_id,
            job_id=job_id,
        ),
        route=route_label,
        schema_id=body.schema_id,
        requested_model_id=getattr(body, "model", None),
        cache_enabled=bool(getattr(body, "cache", True)),
        repair_enabled=bool(getattr(body, "repair", True)),
        requested_max_new_tokens=getattr(body, "max_new_tokens", None),
    )


def apply_extract_result(
    run: ExtractionRun,
    result: ExtractExecutionResult,
) -> ExtractionRun:
    policy = RunPolicySnapshot(
        generate_max_new_tokens_cap=result.policy_generate_max_new_tokens_cap
    )
    return (
        run.with_resolution(
            resolved_model_id=result.model,
            effective_max_new_tokens=result.effective_max_new_tokens,
        )
        .with_policy(policy)
        .with_outcome(
            RunOutcome.succeeded(
                cached=result.cached,
                repair_attempted=result.repair_attempted,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )
        )
    )


def apply_extract_error(run: ExtractionRun, error: AppError) -> ExtractionRun:
    extra = error.extra if isinstance(error.extra, dict) else {}
    error_stage = extra.get("stage")
    return run.with_outcome(
        RunOutcome.failed(
            error_code=error.code,
            error_stage=str(error_stage) if error_stage is not None else None,
        )
    )


async def run_extract(
    *,
    ctx: Any,
    body: Any,
    api_key: Any,
    llm: Any,
    session: Any,
    redis: Any | None,
    route_label: str = "/v1/extract",
) -> RunExtractResult:
    run = build_extraction_run(ctx=ctx, body=body, route_label=route_label)
    try:
        result = await execute_extract(
            ctx=ctx,
            body=body,
            api_key=api_key,
            llm=llm,
            session=session,
            redis=redis,
            route_label=route_label,
        )
    except AppError:
        # The error-aware run mapping is intentionally available for later
        # route/application integration work, but Phase 2.1.1 keeps current
        # external behavior unchanged by re-raising here.
        raise

    return RunExtractResult(
        run=apply_extract_result(run, result),
        response=result,
    )


async def run_extract_request(
    *,
    ctx: Any,
    body: Any,
    api_key: Any,
    llm: Any,
    redis: Any | None = None,
    route_label: str = "/v1/extract",
) -> RunExtractResult:
    async with db_session.get_sessionmaker()() as session:
        return await run_extract(
            ctx=ctx,
            body=body,
            api_key=api_key,
            llm=llm,
            session=session,
            redis=redis,
            route_label=route_label,
        )
