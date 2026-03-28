from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import llm_server.services.extract_jobs as extract_jobs


class _FakeSession:
    async def execute(self, _stmt):
        return SimpleNamespace(
            scalar_one=lambda: SimpleNamespace(key="proof-user-key"),
        )


class _FakeSessionContext:
    async def __aenter__(self):
        return _FakeSession()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeQueue:
    async def dequeue(self, timeout_seconds: int = 0):
        return "job-1"


def _job():
    return SimpleNamespace(
        id="job-1",
        status="running",
        api_key="proof-user-key",
        request_id="submit-request-1",
        trace_id="shared-trace-1",
        otel_parent_context_json={
            "traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
        },
        schema_id="sroie_receipt_v1",
        text="Vendor: ACME\nTotal: 10.00",
        requested_model_id="model-a",
        resolved_model_id="resolved-model",
        max_new_tokens=256,
        temperature=0.0,
        cache=True,
        repair=True,
    )


@pytest.mark.anyio
async def test_process_extract_job_once_continues_worker_trace(monkeypatch: pytest.MonkeyPatch):
    consumer_calls: list[dict[str, object]] = []
    child_calls: list[dict[str, object]] = []
    trace_events: list[str] = []
    job = _job()

    async def fake_claim_extract_job(*, session, job_id):
        assert job_id == "job-1"
        return job

    async def fake_record_trace_event_best_effort(**kwargs):
        trace_events.append(str(kwargs["event_name"]))

    async def fake_execute_extract(**kwargs):
        return SimpleNamespace(
            model="resolved-model",
            data={"id": "1"},
            cached=False,
            repair_attempted=False,
            prompt_tokens=11,
            completion_tokens=22,
        )

    async def fake_complete_extract_job_success(*, session, job_id, result):
        assert job_id == "job-1"
        assert result.model == "resolved-model"

    @contextmanager
    def fake_start_consumer_span(name: str, *, carrier=None, attributes=None):
        span = SimpleNamespace(set_attribute=lambda *args, **kwargs: None)
        consumer_calls.append(
            {
                "name": name,
                "carrier": carrier,
                "attributes": attributes,
            }
        )
        yield span

    @contextmanager
    def fake_start_child_span(name: str, *, request=None, attributes=None, **kwargs):
        child_calls.append(
            {
                "name": name,
                "request": request,
                "attributes": attributes,
            }
        )
        yield SimpleNamespace()

    monkeypatch.setattr(extract_jobs, "claim_extract_job", fake_claim_extract_job, raising=True)
    monkeypatch.setattr(
        extract_jobs,
        "record_trace_event_best_effort",
        fake_record_trace_event_best_effort,
        raising=True,
    )
    monkeypatch.setattr(
        extract_jobs,
        "execute_extract",
        fake_execute_extract,
        raising=True,
    )
    monkeypatch.setattr(
        extract_jobs,
        "complete_extract_job_success",
        fake_complete_extract_job_success,
        raising=True,
    )
    monkeypatch.setattr(extract_jobs, "get_llm", lambda ctx: object(), raising=True)
    monkeypatch.setattr(
        extract_jobs,
        "validate_extract_submission",
        lambda **_: ("resolved-model", object()),
        raising=True,
    )
    monkeypatch.setattr(
        extract_jobs,
        "start_consumer_span",
        fake_start_consumer_span,
        raising=True,
    )
    monkeypatch.setattr(
        extract_jobs,
        "start_child_span",
        fake_start_child_span,
        raising=True,
    )

    result = await extract_jobs.process_extract_job_once(
        app=SimpleNamespace(state=SimpleNamespace(redis=None)),
        sessionmaker=lambda: _FakeSessionContext(),
        queue=_FakeQueue(),
        timeout_seconds=1,
    )

    assert result is not None
    assert result.status == "succeeded"
    assert consumer_calls == [
        {
            "name": "extract.job_worker",
            "carrier": job.otel_parent_context_json,
            "attributes": {
                "llm.route": "/v1/extract/jobs/worker",
                "llm.request_id": "submit-request-1",
                "llm.trace_id": "shared-trace-1",
                "llm.job_id": "job-1",
                "llm.schema_id": "sroie_receipt_v1",
                "llm.requested_model_id": "model-a",
                "llm.resolved_model_id": "resolved-model",
                "messaging.system": "redis",
                "messaging.destination.name": extract_jobs.EXTRACT_JOB_QUEUE_KEY,
            },
        }
    ]
    assert len(child_calls) == 1
    assert child_calls[0]["name"] == "extract.execute"
    assert (
        getattr(getattr(child_calls[0]["request"], "state", None), "trace_id", None)
        == "shared-trace-1"
    )
    assert (
        getattr(getattr(child_calls[0]["request"], "state", None), "trace_job_id", None) == "job-1"
    )
    assert child_calls[0]["attributes"] == {
        "llm.schema_id": "sroie_receipt_v1",
        "llm.requested_model_id": "model-a",
        "llm.resolved_model_id": "resolved-model",
        "llm.job_id": "job-1",
    }
    assert trace_events == [
        "extract_job.worker_claimed",
        "extract_job.execution_started",
        "extract_job.completed",
    ]
