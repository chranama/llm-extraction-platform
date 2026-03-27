from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import llm_server.db.session as db_session

from llm_server.core.errors import AppError
from llm_server.domain.jobs import AsyncJobLifecycle
from llm_server.domain.outcomes import RunOutcome
from llm_server.domain.runs import ExtractionRun, RunIdentity
from llm_server.services.extract_jobs import (
    get_owned_extract_job,
    job_poll_path,
    job_trace_id,
    serialize_extract_job,
)
from llm_server.telemetry.traces import record_trace_event_best_effort, set_trace_meta


@dataclass(frozen=True, slots=True)
class PollExtractJobResult:
    run: ExtractionRun
    payload: dict[str, Any]


def build_extraction_run_from_job(*, job: Any, route_label: str) -> ExtractionRun:
    lifecycle = AsyncJobLifecycle.from_status(getattr(job, "status", None))
    run = ExtractionRun(
        identity=RunIdentity(
            request_id=getattr(job, "request_id", None),
            trace_id=job_trace_id(job),
            job_id=getattr(job, "id", None),
        ),
        route=route_label,
        schema_id=job.schema_id,
        requested_model_id=getattr(job, "requested_model_id", None),
        resolved_model_id=getattr(job, "resolved_model_id", None),
        cache_enabled=bool(getattr(job, "cache", True)),
        repair_enabled=bool(getattr(job, "repair", True)),
        requested_max_new_tokens=getattr(job, "max_new_tokens", None),
        job_lifecycle=lifecycle,
    )
    if lifecycle is AsyncJobLifecycle.SUCCEEDED:
        return run.with_outcome(
            RunOutcome.succeeded(
                cached=bool(getattr(job, "cached", False)),
                repair_attempted=bool(getattr(job, "repair_attempted", False)),
                prompt_tokens=getattr(job, "prompt_tokens", None),
                completion_tokens=getattr(job, "completion_tokens", None),
            )
        )
    if lifecycle is AsyncJobLifecycle.FAILED:
        return run.with_outcome(
            RunOutcome.failed(
                error_code=str(getattr(job, "error_code", "job_failed") or "job_failed"),
                error_stage=getattr(job, "error_stage", None),
            )
        )
    return run


async def poll_extract_job(
    *,
    request: Any,
    job_id: str,
    api_key: Any,
    session: Any,
) -> PollExtractJobResult:
    job = await get_owned_extract_job(session=session, api_key=api_key, job_id=job_id)
    if job is None:
        raise AppError(
            code="not_found",
            message="Job not found",
            status_code=404,
        )

    route_label = job_poll_path(job.id)
    run = build_extraction_run_from_job(job=job, route_label=route_label)
    set_trace_meta(request, trace_id=run.trace_id, job_id=run.job_id)
    await record_trace_event_best_effort(
        trace_id=run.trace_id,
        event_name="extract_job.status_polled",
        route=route_label,
        stage="status_poll",
        status="ok",
        request_id=getattr(getattr(request, "state", None), "request_id", None),
        job_id=run.job_id,
        model_id=run.resolved_model_id,
        details={"job_status": job.status, "schema_id": job.schema_id},
    )
    return PollExtractJobResult(run=run, payload=serialize_extract_job(job))


async def poll_extract_job_request(
    *,
    request: Any,
    job_id: str,
    api_key: Any,
) -> PollExtractJobResult:
    async with db_session.get_sessionmaker()() as session:
        return await poll_extract_job(
            request=request,
            job_id=job_id,
            api_key=api_key,
            session=session,
        )
