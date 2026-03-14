from __future__ import annotations

from typing import Any, Mapping

import llm_server.db.session as db_session
from llm_server.db.models import RequestTraceEvent


def trace_id_from_ctx(ctx: Any) -> str | None:
    state = getattr(ctx, "state", None)
    trace_id = getattr(state, "trace_id", None)
    if isinstance(trace_id, str) and trace_id.strip():
        return trace_id.strip()
    request_id = getattr(state, "request_id", None)
    if isinstance(request_id, str) and request_id.strip():
        return request_id.strip()
    return None


def set_trace_meta(ctx: Any, *, trace_id: str | None = None, job_id: str | None = None) -> None:
    state = getattr(ctx, "state", None)
    if state is None:
        return
    if isinstance(trace_id, str) and trace_id.strip():
        state.trace_id = trace_id.strip()
    elif not isinstance(getattr(state, "trace_id", None), str):
        request_id = getattr(state, "request_id", None)
        if isinstance(request_id, str) and request_id.strip():
            state.trace_id = request_id.strip()
    if isinstance(job_id, str) and job_id.strip():
        state.trace_job_id = job_id.strip()


def trace_job_id_from_ctx(ctx: Any) -> str | None:
    job_id = getattr(getattr(ctx, "state", None), "trace_job_id", None)
    return job_id.strip() if isinstance(job_id, str) and job_id.strip() else None


def compact_details(raw: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[str(key)] = value
        elif isinstance(value, Mapping):
            nested = compact_details(value)
            if nested:
                out[str(key)] = nested
        elif isinstance(value, (list, tuple)):
            items = [x for x in value if isinstance(x, (str, int, float, bool))]
            if items:
                out[str(key)] = items
    return out or None


async def record_trace_event(
    session: Any,
    *,
    trace_id: str,
    event_name: str,
    route: str,
    stage: str | None,
    status: str,
    request_id: str | None = None,
    job_id: str | None = None,
    model_id: str | None = None,
    details: Mapping[str, Any] | None = None,
    commit: bool = True,
) -> RequestTraceEvent:
    row = RequestTraceEvent(
        trace_id=trace_id,
        event_name=event_name,
        route=route,
        stage=stage,
        status=status,
        request_id=request_id,
        job_id=job_id,
        model_id=model_id,
        details_json=compact_details(details),
    )
    session.add(row)
    if commit:
        await session.commit()
        await session.refresh(row)
    else:
        await session.flush()
    return row


async def record_trace_event_best_effort(
    *,
    trace_id: str | None,
    event_name: str,
    route: str,
    stage: str | None,
    status: str,
    request_id: str | None = None,
    job_id: str | None = None,
    model_id: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    if not isinstance(trace_id, str) or not trace_id.strip():
        return
    try:
        sessionmaker = db_session.get_sessionmaker()
        async with sessionmaker() as session:
            await record_trace_event(
                session,
                trace_id=trace_id.strip(),
                event_name=event_name,
                route=route,
                stage=stage,
                status=status,
                request_id=request_id,
                job_id=job_id,
                model_id=model_id,
                details=details,
                commit=True,
            )
    except Exception:
        return
