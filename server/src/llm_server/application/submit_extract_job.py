from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import llm_server.db.session as db_session

from llm_server.core.errors import AppError
from llm_server.core.tracing import current_trace_carrier, start_child_span
from llm_server.domain.jobs import AsyncJobLifecycle
from llm_server.domain.runs import ExtractionRun
from llm_server.services.extract_execution import validate_extract_submission
from llm_server.services.extract_jobs import (
    ExtractJobQueue,
    create_extract_job,
    job_poll_path,
    job_trace_id,
    queue_from_request,
)
from llm_server.telemetry.traces import (
    record_trace_event_best_effort,
    set_trace_meta,
    trace_id_from_ctx,
)

from .run_extract import build_extraction_run


@dataclass(frozen=True, slots=True)
class SubmitExtractJobResult:
    run: ExtractionRun
    job: Any


async def submit_extract_job(
    *,
    request: Any,
    body: Any,
    api_key: Any,
    llm: Any,
    session: Any,
    queue: ExtractJobQueue,
    route_label: str = "/v1/extract/jobs",
) -> SubmitExtractJobResult:
    trace_id = trace_id_from_ctx(request)
    await record_trace_event_best_effort(
        trace_id=trace_id,
        event_name="extract_job.submitted",
        route=route_label,
        stage="submitted",
        status="accepted",
        request_id=getattr(getattr(request, "state", None), "request_id", None),
        details={
            "schema_id": body.schema_id,
            "requested_model_id": body.model,
            "cache": bool(body.cache),
            "repair": bool(body.repair),
        },
    )

    resolved_model_id, _ = validate_extract_submission(ctx=request, body=body, llm=llm)
    request_id = getattr(getattr(request, "state", None), "request_id", None)

    run = build_extraction_run(ctx=request, body=body, route_label=route_label).with_resolution(
        resolved_model_id=resolved_model_id,
        effective_max_new_tokens=None,
    )

    job = await create_extract_job(
        session=session,
        queue=queue,
        api_key=api_key,
        request_id=request_id,
        trace_id=trace_id,
        otel_parent_context=current_trace_carrier(),
        body=body,
        resolved_model_id=resolved_model_id,
    )
    set_trace_meta(request, trace_id=trace_id, job_id=job.id)

    run = run.with_identity(job_id=job.id).with_job_lifecycle(AsyncJobLifecycle.QUEUED)

    await record_trace_event_best_effort(
        trace_id=trace_id,
        event_name="extract_job.persisted",
        route=route_label,
        stage="persisted",
        status="ok",
        request_id=getattr(getattr(request, "state", None), "request_id", None),
        job_id=job.id,
        model_id=job.resolved_model_id,
        details={"schema_id": job.schema_id},
    )
    await record_trace_event_best_effort(
        trace_id=trace_id,
        event_name="extract_job.queued",
        route=route_label,
        stage="queued",
        status="ok",
        request_id=getattr(getattr(request, "state", None), "request_id", None),
        job_id=job.id,
        model_id=job.resolved_model_id,
        details={"schema_id": job.schema_id},
    )
    return SubmitExtractJobResult(run=run, job=job)


async def submit_extract_job_request(
    *,
    request: Any,
    body: Any,
    api_key: Any,
    llm: Any,
    route_label: str = "/v1/extract/jobs",
) -> SubmitExtractJobResult:
    queue = queue_from_request(request)
    if queue is None:
        raise AppError(
            code="extract_job_queue_unavailable",
            message="Async extract jobs require Redis-backed queueing to be enabled.",
            status_code=503,
        )

    with start_child_span(
        "extract.job_submit",
        request=request,
        attributes={
            "llm.schema_id": body.schema_id,
            "llm.requested_model_id": getattr(body, "model", None),
        },
    ):
        async with db_session.get_sessionmaker()() as session:
            return await submit_extract_job(
                request=request,
                body=body,
                api_key=api_key,
                llm=llm,
                session=session,
                queue=queue,
                route_label=route_label,
            )


def submit_extract_job_response_payload(result: SubmitExtractJobResult) -> dict[str, Any]:
    job = result.job
    return {
        "job_id": job.id,
        "trace_id": job_trace_id(job),
        "status": job.status,
        "schema_id": job.schema_id,
        "model": job.resolved_model_id or job.requested_model_id or "unknown",
        "created_at": job.created_at.isoformat(),
        "poll_path": job_poll_path(job.id),
    }
