from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.core.errors import AppError
from llm_server.core.redis import get_redis_from_request
from llm_server.db.models import ApiKey, ExtractJob

EXTRACT_JOB_QUEUE_KEY = "llm:extract_jobs:queue"
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"


class ExtractJobBody(BaseModel):
    schema_id: str
    text: str
    model: str | None = None
    max_new_tokens: int | None = 512
    temperature: float | None = 0.0
    cache: bool = True
    repair: bool = True


class ExtractJobQueue(Protocol):
    async def enqueue(self, job_id: str) -> None: ...

    async def dequeue(self, timeout_seconds: int = 0) -> str | None: ...


class RedisExtractJobQueue:
    def __init__(self, redis: Any, *, queue_key: str = EXTRACT_JOB_QUEUE_KEY) -> None:
        self._redis = redis
        self._queue_key = queue_key

    async def enqueue(self, job_id: str) -> None:
        await self._redis.lpush(self._queue_key, job_id)

    async def dequeue(self, timeout_seconds: int = 0) -> str | None:
        item = await self._redis.brpop(self._queue_key, timeout=timeout_seconds)
        if item is None:
            return None
        if isinstance(item, (list, tuple)) and len(item) == 2:
            value = item[1]
        else:
            value = item
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return value if isinstance(value, str) and value.strip() else None


class InMemoryExtractJobQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def enqueue(self, job_id: str) -> None:
        await self._queue.put(job_id)

    async def dequeue(self, timeout_seconds: int = 0) -> str | None:
        try:
            if timeout_seconds <= 0:
                return self._queue.get_nowait()
            return await asyncio.wait_for(self._queue.get(), timeout=float(timeout_seconds))
        except (asyncio.QueueEmpty, asyncio.TimeoutError):
            return None


def queue_from_app(app: Any) -> ExtractJobQueue | None:
    q = getattr(app.state, "extract_job_queue", None)
    if q is not None:
        return q
    redis = getattr(app.state, "redis", None)
    if redis is None:
        return None
    q = RedisExtractJobQueue(redis)
    app.state.extract_job_queue = q
    return q


def queue_from_request(request: Any) -> ExtractJobQueue | None:
    q = getattr(request.app.state, "extract_job_queue", None)
    if q is not None:
        return q
    redis = get_redis_from_request(request)
    if redis is None:
        return None
    q = RedisExtractJobQueue(redis)
    request.app.state.extract_job_queue = q
    return q


def utc_now() -> datetime:
    return datetime.now(UTC)


def job_trace_id(job: ExtractJob) -> str | None:
    trace_id = getattr(job, "trace_id", None)
    if isinstance(trace_id, str) and trace_id.strip():
        return trace_id.strip()
    request_id = getattr(job, "request_id", None)
    if isinstance(request_id, str) and request_id.strip():
        return request_id.strip()
    return None


async def create_extract_job(
    *,
    session: AsyncSession,
    queue: ExtractJobQueue,
    api_key: ApiKey,
    request_id: str | None,
    trace_id: str | None,
    otel_parent_context: dict[str, str] | None,
    body: ExtractJobBody,
    resolved_model_id: str,
) -> ExtractJob:
    job = ExtractJob(
        id=uuid.uuid4().hex,
        status=STATUS_QUEUED,
        api_key=api_key.key,
        request_id=request_id,
        trace_id=trace_id or request_id,
        otel_parent_context_json=otel_parent_context,
        schema_id=body.schema_id,
        text=body.text,
        requested_model_id=body.model,
        resolved_model_id=resolved_model_id,
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        cache=bool(body.cache),
        repair=bool(body.repair),
    )
    session.add(job)
    await session.commit()
    await queue.enqueue(job.id)
    return job


async def get_owned_extract_job(
    *,
    session: AsyncSession,
    api_key: ApiKey,
    job_id: str,
) -> ExtractJob | None:
    row = await session.execute(
        select(ExtractJob).where(
            ExtractJob.id == job_id,
            ExtractJob.api_key == api_key.key,
        )
    )
    return row.scalar_one_or_none()


async def claim_extract_job(*, session: AsyncSession, job_id: str) -> ExtractJob | None:
    now = utc_now()
    result = await session.execute(
        sa.update(ExtractJob)
        .where(ExtractJob.id == job_id, ExtractJob.status == STATUS_QUEUED)
        .values(
            status=STATUS_RUNNING,
            started_at=now,
            attempt_count=ExtractJob.attempt_count + 1,
        )
        .returning(ExtractJob)
    )
    row = result.scalar_one_or_none()
    await session.commit()
    return row


async def complete_extract_job_success(
    *,
    session: AsyncSession,
    job_id: str,
    result: Any,
) -> None:
    await session.execute(
        sa.update(ExtractJob)
        .where(ExtractJob.id == job_id)
        .values(
            status=STATUS_SUCCEEDED,
            finished_at=utc_now(),
            resolved_model_id=result.model,
            result_json=result.data,
            cached=result.cached,
            repair_attempted=result.repair_attempted,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            error_code=None,
            error_message=None,
            error_stage=None,
        )
    )
    await session.commit()


async def complete_extract_job_failure(
    *,
    session: AsyncSession,
    job_id: str,
    error: AppError,
) -> None:
    extra = error.extra if isinstance(error.extra, dict) else {}
    await session.execute(
        sa.update(ExtractJob)
        .where(ExtractJob.id == job_id)
        .values(
            status=STATUS_FAILED,
            finished_at=utc_now(),
            error_code=error.code,
            error_message=error.message,
            error_stage=str(extra.get("stage")) if extra.get("stage") is not None else None,
        )
    )
    await session.commit()


def job_poll_path(job_id: str) -> str:
    return f"/v1/extract/jobs/{job_id}"


def serialize_extract_job(job: ExtractJob) -> dict[str, Any]:
    error = None
    if job.status == STATUS_FAILED:
        error = {
            "code": job.error_code,
            "message": job.error_message,
            "stage": job.error_stage,
        }
    return {
        "job_id": job.id,
        "trace_id": job_trace_id(job),
        "status": job.status,
        "schema_id": job.schema_id,
        "model": job.resolved_model_id or job.requested_model_id,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "cached": job.cached,
        "repair_attempted": job.repair_attempted,
        "result": job.result_json if job.status == STATUS_SUCCEEDED else None,
        "error": error,
    }
