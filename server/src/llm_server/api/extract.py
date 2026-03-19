from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

import llm_server.db.session as db_session

from llm_server.core.errors import AppError
from llm_server.core.schema_registry import (
    SchemaLoadError,
    SchemaNotFoundError,
    list_schemas,
    load_schema,
)
from llm_server.services.api_deps.core.auth import get_api_key
from llm_server.services.api_deps.core.llm_access import get_llm
from llm_server.services.extract_execution import execute_extract
from llm_server.services.extract_jobs import (
    ExtractJobBody,
    create_extract_job,
    get_owned_extract_job,
    job_trace_id,
    job_poll_path,
    queue_from_request,
    serialize_extract_job,
    validate_extract_submission,
)
from llm_server.telemetry.traces import (
    record_trace_event_best_effort,
    set_trace_meta,
    trace_id_from_ctx,
)

router = APIRouter()


class ExtractRequest(BaseModel):
    schema_id: str = Field(..., description="Schema id (e.g. ticket_v1, invoice_v1, receipt_v1)")
    text: str = Field(..., description="Raw text or OCR text to extract from")
    model: str | None = Field(
        default=None, description="Optional model id override for multi-model routing"
    )
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


class ExtractJobSubmitResponse(BaseModel):
    job_id: str
    trace_id: str | None = None
    status: str
    schema_id: str
    model: str
    created_at: str
    poll_path: str


class ExtractJobStatusResponse(BaseModel):
    job_id: str
    trace_id: str | None = None
    status: str
    schema_id: str
    model: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    cached: bool | None = None
    repair_attempted: bool | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


@router.get("/v1/schemas")
async def schemas_index(api_key=Depends(get_api_key)):
    return [
        {"schema_id": s.schema_id, "title": s.title, "description": s.description}
        for s in list_schemas()
    ]


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

    return schema


@router.post("/v1/extract", response_model=ExtractResponse)
async def extract(
    request: Request,
    body: ExtractRequest,
    api_key=Depends(get_api_key),
    llm: Any = Depends(get_llm),
):
    set_trace_meta(request)
    async with db_session.get_sessionmaker()() as session:
        result = await execute_extract(
            ctx=request,
            body=body,
            api_key=api_key,
            llm=llm,
            session=session,
            redis=None,
            route_label="/v1/extract",
        )
        return ExtractResponse(
            schema_id=result.schema_id,
            model=result.model,
            data=result.data,
            cached=result.cached,
            repair_attempted=result.repair_attempted,
        )


@router.post(
    "/v1/extract/jobs",
    response_model=ExtractJobSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_extract_job(
    request: Request,
    body: ExtractJobBody,
    api_key=Depends(get_api_key),
    llm: Any = Depends(get_llm),
):
    set_trace_meta(request)
    trace_id = trace_id_from_ctx(request)
    await record_trace_event_best_effort(
        trace_id=trace_id,
        event_name="extract_job.submitted",
        route="/v1/extract/jobs",
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
    queue = queue_from_request(request)
    if queue is None:
        raise AppError(
            code="extract_job_queue_unavailable",
            message="Async extract jobs require Redis-backed queueing to be enabled.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    resolved_model_id, _ = validate_extract_submission(ctx=request, body=body, llm=llm)
    request_id = getattr(getattr(request, "state", None), "request_id", None)
    trace_id = trace_id_from_ctx(request)

    async with db_session.get_sessionmaker()() as session:
        job = await create_extract_job(
            session=session,
            queue=queue,
            api_key=api_key,
            request_id=request_id,
            trace_id=trace_id,
            body=body,
            resolved_model_id=resolved_model_id,
        )
        set_trace_meta(request, trace_id=trace_id, job_id=job.id)
        await record_trace_event_best_effort(
            trace_id=trace_id,
            event_name="extract_job.persisted",
            route="/v1/extract/jobs",
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
            route="/v1/extract/jobs",
            stage="queued",
            status="ok",
            request_id=getattr(getattr(request, "state", None), "request_id", None),
            job_id=job.id,
            model_id=job.resolved_model_id,
            details={"schema_id": job.schema_id},
        )
        return ExtractJobSubmitResponse(
            job_id=job.id,
            trace_id=job_trace_id(job),
            status=job.status,
            schema_id=job.schema_id,
            model=job.resolved_model_id or job.requested_model_id or "unknown",
            created_at=job.created_at.isoformat(),
            poll_path=job_poll_path(job.id),
        )


@router.get("/v1/extract/jobs/{job_id}", response_model=ExtractJobStatusResponse)
async def get_extract_job_status(
    request: Request,
    job_id: str,
    api_key=Depends(get_api_key),
):
    async with db_session.get_sessionmaker()() as session:
        job = await get_owned_extract_job(session=session, api_key=api_key, job_id=job_id)
        if job is None:
            raise AppError(
                code="not_found",
                message="Job not found",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        trace_id = job_trace_id(job)
        set_trace_meta(request, trace_id=trace_id, job_id=job.id)
        await record_trace_event_best_effort(
            trace_id=trace_id,
            event_name="extract_job.status_polled",
            route=job_poll_path(job.id),
            stage="status_poll",
            status="ok",
            request_id=getattr(getattr(request, "state", None), "request_id", None),
            job_id=job.id,
            model_id=job.resolved_model_id,
            details={"job_status": job.status, "schema_id": job.schema_id},
        )
        return ExtractJobStatusResponse(**serialize_extract_job(job))
