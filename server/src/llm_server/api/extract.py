from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from llm_server.application.poll_extract_job import poll_extract_job_request
from llm_server.application.run_extract import run_extract_request
from llm_server.application.submit_extract_job import (
    submit_extract_job_request,
    submit_extract_job_response_payload,
)
from llm_server.core.errors import AppError
from llm_server.core.tracing import bind_request_span
from llm_server.core.schema_registry import (
    SchemaLoadError,
    SchemaNotFoundError,
    list_schemas,
    load_schema,
)
from llm_server.api.dependencies.auth import get_api_key
from llm_server.services.llm_runtime.access import get_llm
from llm_server.services.extract_jobs import (
    ExtractJobBody,
    serialize_extract_job,
)
from llm_server.telemetry.traces import (
    set_trace_meta,
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
    bind_request_span(
        request,
        name="backend.extract",
        route="/v1/extract",
        attributes={
            "llm.schema_id": body.schema_id,
            "llm.requested_model_id": body.model,
        },
    )
    result = await run_extract_request(
        ctx=request,
        body=body,
        api_key=api_key,
        llm=llm,
        redis=None,
        route_label="/v1/extract",
    )
    response = result.response
    return ExtractResponse(
        schema_id=response.schema_id,
        model=response.model,
        data=response.data,
        cached=response.cached,
        repair_attempted=response.repair_attempted,
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
    bind_request_span(
        request,
        name="backend.extract_jobs.submit",
        route="/v1/extract/jobs",
        attributes={
            "llm.schema_id": body.schema_id,
            "llm.requested_model_id": body.model,
        },
    )
    result = await submit_extract_job_request(
        request=request,
        body=body,
        api_key=api_key,
        llm=llm,
        route_label="/v1/extract/jobs",
    )
    return ExtractJobSubmitResponse(**submit_extract_job_response_payload(result))


@router.get("/v1/extract/jobs/{job_id}", response_model=ExtractJobStatusResponse)
async def get_extract_job_status(
    request: Request,
    job_id: str,
    api_key=Depends(get_api_key),
):
    bind_request_span(
        request,
        name="backend.extract_jobs.poll",
        route="/v1/extract/jobs/{job_id}",
        attributes={"llm.job_id": job_id},
    )
    result = await poll_extract_job_request(
        request=request,
        job_id=job_id,
        api_key=api_key,
    )
    return ExtractJobStatusResponse(**result.payload)
