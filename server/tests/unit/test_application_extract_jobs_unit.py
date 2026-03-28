from __future__ import annotations

import importlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from llm_server.domain.jobs import AsyncJobLifecycle
from llm_server.domain.outcomes import RunStatus

submit_jobs_app = importlib.import_module("llm_server.application.submit_extract_job")
poll_jobs_app = importlib.import_module("llm_server.application.poll_extract_job")


def _request(*, request_id: str = "rid-1", trace_id: str = "trace-1"):
    return SimpleNamespace(state=SimpleNamespace(request_id=request_id, trace_id=trace_id))


def _body():
    return SimpleNamespace(
        schema_id="sroie_receipt_v1",
        text="Vendor: ACME\nTotal: 10.00",
        model="model-a",
        cache=True,
        repair=True,
        max_new_tokens=256,
        temperature=0.0,
    )


def _job(*, status: str = "queued"):
    return SimpleNamespace(
        id="job-1",
        status=status,
        api_key="proof-user-key",
        request_id="rid-1",
        trace_id="trace-1",
        otel_parent_context_json={"traceparent": "00-parent"},
        schema_id="sroie_receipt_v1",
        text="Vendor: ACME\nTotal: 10.00",
        requested_model_id="model-a",
        resolved_model_id="resolved-model",
        max_new_tokens=256,
        temperature=0.0,
        cache=True,
        repair=True,
        created_at=SimpleNamespace(isoformat=lambda: "2026-03-27T00:00:00+00:00"),
        started_at=None,
        finished_at=None,
        cached=False,
        repair_attempted=False,
        prompt_tokens=None,
        completion_tokens=None,
        result_json=None,
        error_code=None,
        error_message=None,
        error_stage=None,
    )


@pytest.mark.anyio
async def test_submit_extract_job_builds_queued_run(monkeypatch: pytest.MonkeyPatch):
    trace_events: list[dict[str, object]] = []
    create_calls: list[dict[str, object]] = []

    async def fake_record_trace_event_best_effort(**kwargs):
        trace_events.append(kwargs)

    async def fake_create_extract_job(**kwargs):
        create_calls.append(kwargs)
        return _job(status="queued")

    monkeypatch.setattr(
        submit_jobs_app,
        "record_trace_event_best_effort",
        fake_record_trace_event_best_effort,
        raising=True,
    )
    monkeypatch.setattr(
        submit_jobs_app,
        "validate_extract_submission",
        lambda **_: ("resolved-model", object()),
        raising=True,
    )
    monkeypatch.setattr(
        submit_jobs_app,
        "create_extract_job",
        fake_create_extract_job,
        raising=True,
    )
    monkeypatch.setattr(
        submit_jobs_app,
        "current_trace_carrier",
        lambda: {"traceparent": "00-parent"},
        raising=True,
    )
    monkeypatch.setattr(submit_jobs_app, "set_trace_meta", lambda *_a, **_k: None, raising=True)

    result = await submit_jobs_app.submit_extract_job(
        request=_request(),
        body=_body(),
        api_key=SimpleNamespace(key="proof-user-key"),
        llm=object(),
        session=object(),
        queue=object(),
    )

    assert result.run.trace_id == "trace-1"
    assert result.run.job_id == "job-1"
    assert result.run.resolved_model_id == "resolved-model"
    assert result.run.job_lifecycle is AsyncJobLifecycle.QUEUED
    assert result.run.outcome.status is RunStatus.ACCEPTED
    assert create_calls[0]["otel_parent_context"] == {"traceparent": "00-parent"}
    assert [item["event_name"] for item in trace_events] == [
        "extract_job.submitted",
        "extract_job.persisted",
        "extract_job.queued",
    ]


def test_build_extraction_run_from_job_maps_success_and_failure():
    success = _job(status="succeeded")
    success.cached = True
    success.repair_attempted = True
    success.prompt_tokens = 11
    success.completion_tokens = 22

    success_run = poll_jobs_app.build_extraction_run_from_job(
        job=success,
        route_label="/v1/extract/jobs/job-1",
    )
    assert success_run.job_lifecycle is AsyncJobLifecycle.SUCCEEDED
    assert success_run.outcome.status is RunStatus.SUCCEEDED
    assert success_run.outcome.cached is True

    failed = _job(status="failed")
    failed.error_code = "schema_validation_failed"
    failed.error_stage = "validate"

    failed_run = poll_jobs_app.build_extraction_run_from_job(
        job=failed,
        route_label="/v1/extract/jobs/job-1",
    )
    assert failed_run.job_lifecycle is AsyncJobLifecycle.FAILED
    assert failed_run.outcome.status is RunStatus.FAILED
    assert failed_run.outcome.error_code == "schema_validation_failed"


@pytest.mark.anyio
async def test_poll_extract_job_request_opens_child_span(monkeypatch: pytest.MonkeyPatch):
    span_calls: list[dict[str, object]] = []

    @contextmanager
    def fake_start_child_span(name: str, *, request=None, attributes=None):
        span_calls.append({"name": name, "request": request, "attributes": attributes})
        yield SimpleNamespace()

    async def fake_poll_extract_job(**kwargs):
        return SimpleNamespace(run=SimpleNamespace(job_id="job-1"), payload={"job_id": "job-1"})

    class _SessionContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        poll_jobs_app,
        "start_child_span",
        fake_start_child_span,
        raising=True,
    )
    monkeypatch.setattr(
        poll_jobs_app.db_session, "get_sessionmaker", lambda: lambda: _SessionContext()
    )
    monkeypatch.setattr(poll_jobs_app, "poll_extract_job", fake_poll_extract_job, raising=True)

    result = await poll_jobs_app.poll_extract_job_request(
        request=_request(),
        job_id="job-1",
        api_key=SimpleNamespace(key="proof-user-key"),
    )

    assert result.payload == {"job_id": "job-1"}
    assert span_calls == [
        {
            "name": "extract.job_poll",
            "request": _request(),
            "attributes": {"llm.job_id": "job-1"},
        }
    ]


@pytest.mark.anyio
async def test_poll_extract_job_adds_linked_status_span(monkeypatch: pytest.MonkeyPatch):
    span_calls: list[dict[str, object]] = []
    trace_events: list[dict[str, object]] = []

    @contextmanager
    def fake_start_child_span(name: str, *, request=None, attributes=None, links=None, **kwargs):
        span_calls.append(
            {
                "name": name,
                "request": request,
                "attributes": attributes,
                "links": links,
            }
        )
        yield SimpleNamespace()

    async def fake_get_owned_extract_job(**kwargs):
        return _job(status="succeeded")

    async def fake_record_trace_event_best_effort(**kwargs):
        trace_events.append(kwargs)

    monkeypatch.setattr(
        poll_jobs_app,
        "get_owned_extract_job",
        fake_get_owned_extract_job,
        raising=True,
    )
    monkeypatch.setattr(
        poll_jobs_app,
        "record_trace_event_best_effort",
        fake_record_trace_event_best_effort,
        raising=True,
    )
    monkeypatch.setattr(
        poll_jobs_app,
        "set_trace_meta",
        lambda *_a, **_k: None,
        raising=True,
    )
    monkeypatch.setattr(
        poll_jobs_app,
        "span_link_from_carrier",
        lambda carrier, **kwargs: "linked-parent",
        raising=True,
    )
    monkeypatch.setattr(
        poll_jobs_app,
        "start_child_span",
        fake_start_child_span,
        raising=True,
    )

    result = await poll_jobs_app.poll_extract_job(
        request=_request(request_id="poll-1", trace_id="trace-1"),
        job_id="job-1",
        api_key=SimpleNamespace(key="proof-user-key"),
        session=object(),
    )

    assert result.payload["job_id"] == "job-1"
    assert trace_events[0]["event_name"] == "extract_job.status_polled"
    assert span_calls == [
        {
            "name": "extract.job_poll.status",
            "request": _request(request_id="poll-1", trace_id="trace-1"),
            "attributes": {
                "llm.job_id": "job-1",
                "llm.schema_id": "sroie_receipt_v1",
                "llm.job_status": "succeeded",
            },
            "links": ["linked-parent"],
        }
    ]


@pytest.mark.anyio
async def test_submit_extract_job_request_opens_child_span(monkeypatch: pytest.MonkeyPatch):
    span_calls: list[dict[str, object]] = []

    @contextmanager
    def fake_start_child_span(name: str, *, request=None, attributes=None):
        span_calls.append({"name": name, "request": request, "attributes": attributes})
        yield SimpleNamespace()

    class _SessionContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def fake_submit_extract_job(**kwargs):
        return SimpleNamespace(run=SimpleNamespace(job_id="job-1"), job=_job(status="queued"))

    monkeypatch.setattr(
        submit_jobs_app,
        "start_child_span",
        fake_start_child_span,
        raising=True,
    )
    monkeypatch.setattr(
        submit_jobs_app, "queue_from_request", lambda request: object(), raising=True
    )
    monkeypatch.setattr(
        submit_jobs_app.db_session,
        "get_sessionmaker",
        lambda: lambda: _SessionContext(),
    )
    monkeypatch.setattr(
        submit_jobs_app,
        "submit_extract_job",
        fake_submit_extract_job,
        raising=True,
    )

    result = await submit_jobs_app.submit_extract_job_request(
        request=_request(),
        body=_body(),
        api_key=SimpleNamespace(key="proof-user-key"),
        llm=object(),
    )

    assert result.job.status == "queued"
    assert span_calls == [
        {
            "name": "extract.job_submit",
            "request": _request(),
            "attributes": {
                "llm.schema_id": "sroie_receipt_v1",
                "llm.requested_model_id": "model-a",
            },
        }
    ]
