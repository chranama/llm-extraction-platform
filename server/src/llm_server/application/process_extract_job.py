from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_server.core.errors import AppError
from llm_server.core.tracing import record_error, start_child_span, start_consumer_span
from llm_server.db.models import ApiKey
from llm_server.domain.jobs import AsyncJobLifecycle
from llm_server.domain.outcomes import RunOutcome
from llm_server.domain.runs import ExtractionRun, RunIdentity
from llm_server.services.extract_execution import execute_extract, validate_extract_submission
from llm_server.services.extract_jobs import (
    EXTRACT_JOB_QUEUE_KEY,
    ExtractJobBody,
    ExtractJobQueue,
    claim_extract_job,
    complete_extract_job_failure,
    complete_extract_job_success,
    job_trace_id,
)
from llm_server.services.llm_runtime.access import get_llm
from llm_server.telemetry.traces import record_trace_event_best_effort, set_trace_meta

from .run_extract import apply_extract_error, apply_extract_result

logger = logging.getLogger("llm_server.application.process_extract_job")

WORKER_ROUTE = "/v1/extract/jobs/worker"


class WorkerRequestContext:
    def __init__(
        self,
        *,
        app: Any,
        route: str,
        request_id: str | None,
        client_host: str | None = None,
        trace_id: str | None = None,
        trace_job_id: str | None = None,
    ) -> None:
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


@dataclass(frozen=True, slots=True)
class ExtractJobProcessResult:
    job_id: str
    status: str
    error_code: str | None = None


def build_extract_job_run(*, job: Any, route_label: str = WORKER_ROUTE) -> ExtractionRun:
    return ExtractionRun(
        identity=RunIdentity(
            request_id=job.request_id,
            trace_id=job_trace_id(job),
            job_id=job.id,
        ),
        route=route_label,
        schema_id=job.schema_id,
        requested_model_id=job.requested_model_id,
        resolved_model_id=job.resolved_model_id,
        cache_enabled=bool(job.cache),
        repair_enabled=bool(job.repair),
        requested_max_new_tokens=job.max_new_tokens,
        job_lifecycle=AsyncJobLifecycle.RUNNING,
        outcome=RunOutcome.accepted(),
    )


def build_extract_job_body(job: Any) -> ExtractJobBody:
    return ExtractJobBody(
        schema_id=job.schema_id,
        text=job.text,
        model=job.requested_model_id,
        max_new_tokens=job.max_new_tokens,
        temperature=job.temperature,
        cache=job.cache,
        repair=job.repair,
    )


def build_worker_request_context(*, app: Any, run: ExtractionRun) -> WorkerRequestContext:
    return WorkerRequestContext(
        app=app,
        route=run.route,
        request_id=run.request_id,
        client_host=None,
        trace_id=run.trace_id,
        trace_job_id=run.job_id,
    )


async def process_extract_job_once(
    *,
    app: Any,
    sessionmaker: async_sessionmaker[AsyncSession],
    queue: ExtractJobQueue,
    timeout_seconds: int = 0,
) -> ExtractJobProcessResult | None:
    job_id = await queue.dequeue(timeout_seconds=timeout_seconds)
    if not job_id:
        return None

    async with sessionmaker() as session:
        job = await claim_extract_job(session=session, job_id=job_id)
        if job is None:
            logger.info(
                "extract_job_skip",
                extra={
                    "job_id": job_id,
                    "route": WORKER_ROUTE,
                    "error_message": "not_queued",
                },
            )
            return ExtractJobProcessResult(job_id=job_id, status="skipped")

        row = await session.execute(select(ApiKey).where(ApiKey.key == job.api_key))
        api_key = row.scalar_one()
        run = build_extract_job_run(job=job)
        ctx = build_worker_request_context(app=app, run=run)
        set_trace_meta(ctx, trace_id=run.trace_id, job_id=run.job_id)

        body = build_extract_job_body(job)
        llm = get_llm(ctx)
        worker_span: Any | None = None

        try:
            with start_consumer_span(
                "extract.job_worker",
                carrier=getattr(job, "otel_parent_context_json", None),
                attributes={
                    "llm.route": run.route,
                    "llm.request_id": run.request_id,
                    "llm.trace_id": run.trace_id,
                    "llm.job_id": run.job_id,
                    "llm.schema_id": run.schema_id,
                    "llm.requested_model_id": run.requested_model_id,
                    "llm.resolved_model_id": run.resolved_model_id,
                    "messaging.system": "redis",
                    "messaging.destination.name": EXTRACT_JOB_QUEUE_KEY,
                },
            ) as worker_span:
                await record_trace_event_best_effort(
                    trace_id=run.trace_id,
                    event_name="extract_job.worker_claimed",
                    route=WORKER_ROUTE,
                    stage="claim_job",
                    status="ok",
                    request_id=run.request_id,
                    job_id=run.job_id,
                    model_id=run.resolved_model_id,
                    details={"schema_id": run.schema_id},
                )
                validate_extract_submission(ctx=ctx, body=body, llm=llm)
                await record_trace_event_best_effort(
                    trace_id=run.trace_id,
                    event_name="extract_job.execution_started",
                    route=WORKER_ROUTE,
                    stage="execution_started",
                    status="ok",
                    request_id=run.request_id,
                    job_id=run.job_id,
                    model_id=run.resolved_model_id,
                    details={"schema_id": run.schema_id},
                )
                with start_child_span(
                    "extract.execute",
                    request=ctx,
                    attributes={
                        "llm.schema_id": run.schema_id,
                        "llm.requested_model_id": run.requested_model_id,
                        "llm.resolved_model_id": run.resolved_model_id,
                        "llm.job_id": run.job_id,
                    },
                ):
                    result = await execute_extract(
                        ctx=ctx,
                        body=body,
                        api_key=api_key,
                        llm=llm,
                        session=session,
                        redis=getattr(app.state, "redis", None),
                        route_label=WORKER_ROUTE,
                    )
                run = apply_extract_result(run, result).with_job_lifecycle(
                    AsyncJobLifecycle.SUCCEEDED
                )
                worker_span.set_attribute("llm.job_status", AsyncJobLifecycle.SUCCEEDED.value)

            await complete_extract_job_success(session=session, job_id=job_id, result=result)
            await record_trace_event_best_effort(
                trace_id=run.trace_id,
                event_name="extract_job.completed",
                route=WORKER_ROUTE,
                stage="complete_job",
                status="completed",
                request_id=run.request_id,
                job_id=run.job_id,
                model_id=result.model,
                details={
                    "schema_id": run.schema_id,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                },
            )
            logger.info(
                "extract_job_done",
                extra={
                    "job_id": run.job_id,
                    "request_id": run.request_id,
                    "trace_id": run.trace_id,
                    "route": run.route,
                    "model_id": result.model,
                },
            )
            return ExtractJobProcessResult(
                job_id=job_id,
                status=AsyncJobLifecycle.SUCCEEDED.value,
            )
        except AppError as error:
            run = apply_extract_error(run, error).with_job_lifecycle(AsyncJobLifecycle.FAILED)
            if worker_span is not None:
                record_error(worker_span, error)
                worker_span.set_attribute("llm.job_status", AsyncJobLifecycle.FAILED.value)
            await complete_extract_job_failure(session=session, job_id=job_id, error=error)
            error_stage = (
                (error.extra or {}).get("stage") if isinstance(error.extra, dict) else None
            )
            await record_trace_event_best_effort(
                trace_id=run.trace_id,
                event_name="extract_job.failed",
                route=WORKER_ROUTE,
                stage=str(error_stage or "app_error"),
                status="failed",
                request_id=run.request_id,
                job_id=run.job_id,
                model_id=run.resolved_model_id,
                details={
                    "schema_id": run.schema_id,
                    "error_code": error.code,
                    "error_stage": error_stage,
                },
            )
            logger.info(
                "extract_job_done",
                extra={
                    "job_id": run.job_id,
                    "request_id": run.request_id,
                    "trace_id": run.trace_id,
                    "route": run.route,
                    "model_id": run.resolved_model_id,
                    "error_type": "app_error",
                    "error_message": error.code,
                },
            )
            return ExtractJobProcessResult(
                job_id=job_id,
                status=AsyncJobLifecycle.FAILED.value,
                error_code=error.code,
            )
