from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from llm_server.api import extract as extract_api


def _request():
    return SimpleNamespace(state=SimpleNamespace(request_id="rid-1", trace_id="trace-1"))


@pytest.mark.anyio
async def test_submit_extract_job_route_delegates_to_application_use_case(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[object] = []
    span_calls: list[dict[str, object]] = []

    def fake_set_trace_meta(request, **kwargs):
        calls.append(("set_trace_meta", request, kwargs))

    def fake_bind_request_span(request, **kwargs):
        span_calls.append({"request": request, "kwargs": deepcopy(kwargs)})

    async def fake_submit_extract_job_request(**kwargs):
        calls.append(("submit", kwargs))
        return SimpleNamespace(
            run=SimpleNamespace(job_id="job-1", trace_id="trace-1"),
            job=SimpleNamespace(),
        )

    monkeypatch.setattr(extract_api, "set_trace_meta", fake_set_trace_meta, raising=True)
    monkeypatch.setattr(extract_api, "bind_request_span", fake_bind_request_span, raising=True)
    monkeypatch.setattr(
        extract_api,
        "submit_extract_job_request",
        fake_submit_extract_job_request,
        raising=True,
    )
    monkeypatch.setattr(
        extract_api,
        "submit_extract_job_response_payload",
        lambda _result: {
            "job_id": "job-1",
            "trace_id": "trace-1",
            "status": "queued",
            "schema_id": "sroie_receipt_v1",
            "model": "resolved-model",
            "created_at": "2026-03-27T00:00:00+00:00",
            "poll_path": "/v1/extract/jobs/job-1",
        },
        raising=True,
    )

    response = await extract_api.submit_extract_job(
        _request(),
        extract_api.ExtractJobBody(schema_id="sroie_receipt_v1", text="id 1"),
        api_key=SimpleNamespace(key="proof-user-key"),
        llm=object(),
    )

    assert response.job_id == "job-1"
    assert response.status == "queued"
    assert response.poll_path == "/v1/extract/jobs/job-1"
    assert calls[1][0] == "submit"
    assert span_calls == [
        {
            "request": _request(),
            "kwargs": {
                "name": "backend.extract_jobs.submit",
                "route": "/v1/extract/jobs",
                "attributes": {
                    "llm.schema_id": "sroie_receipt_v1",
                    "llm.requested_model_id": None,
                },
            },
        }
    ]


@pytest.mark.anyio
async def test_get_extract_job_status_route_delegates_to_application_use_case(
    monkeypatch: pytest.MonkeyPatch,
):
    span_calls: list[dict[str, object]] = []

    async def fake_poll_extract_job_request(**kwargs):
        return SimpleNamespace(
            payload={
                "job_id": "job-1",
                "trace_id": "trace-1",
                "status": "succeeded",
                "schema_id": "sroie_receipt_v1",
                "model": "resolved-model",
                "created_at": "2026-03-27T00:00:00+00:00",
                "started_at": None,
                "finished_at": None,
                "cached": True,
                "repair_attempted": False,
                "result": {"id": "1"},
                "error": None,
            }
        )

    def fake_bind_request_span(request, **kwargs):
        span_calls.append({"request": request, "kwargs": deepcopy(kwargs)})

    monkeypatch.setattr(
        extract_api,
        "poll_extract_job_request",
        fake_poll_extract_job_request,
        raising=True,
    )
    monkeypatch.setattr(extract_api, "bind_request_span", fake_bind_request_span, raising=True)

    response = await extract_api.get_extract_job_status(
        _request(),
        "job-1",
        api_key=SimpleNamespace(key="proof-user-key"),
    )

    assert response.job_id == "job-1"
    assert response.trace_id == "trace-1"
    assert response.status == "succeeded"
    assert response.result == {"id": "1"}
    assert span_calls == [
        {
            "request": _request(),
            "kwargs": {
                "name": "backend.extract_jobs.poll",
                "route": "/v1/extract/jobs/{job_id}",
                "attributes": {"llm.job_id": "job-1"},
            },
        }
    ]
