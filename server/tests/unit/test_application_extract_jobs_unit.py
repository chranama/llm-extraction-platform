from __future__ import annotations

import importlib
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

    async def fake_record_trace_event_best_effort(**kwargs):
        trace_events.append(kwargs)

    async def fake_create_extract_job(**kwargs):
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
    monkeypatch.setattr(submit_jobs_app, "create_extract_job", fake_create_extract_job, raising=True)
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
