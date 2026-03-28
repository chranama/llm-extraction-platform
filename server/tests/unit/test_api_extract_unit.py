from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from llm_server.api import extract as extract_api
from llm_server.application.run_extract import RunExtractResult
from llm_server.domain.outcomes import RunOutcome
from llm_server.domain.runs import ExtractionRun, RunIdentity
from llm_server.services.extract_execution import ExtractExecutionResult


def _request():
    return SimpleNamespace(state=SimpleNamespace(request_id="rid-1", trace_id="trace-1"))


def _body():
    return extract_api.ExtractRequest(
        schema_id="sroie_receipt_v1",
        text="Vendor: ACME\nTotal: 10.00",
        model="model-a",
        cache=True,
        repair=True,
        max_new_tokens=256,
        temperature=0.0,
    )


@pytest.mark.anyio
async def test_extract_route_delegates_to_application_use_case(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []
    span_calls: list[dict[str, object]] = []

    def fake_set_trace_meta(request, **kwargs):
        calls.append({"set_trace_meta": request, "kwargs": kwargs})

    def fake_bind_request_span(request, **kwargs):
        span_calls.append({"request": request, "kwargs": deepcopy(kwargs)})

    async def fake_run_extract_request(**kwargs):
        calls.append(kwargs)
        return RunExtractResult(
            run=ExtractionRun(
                identity=RunIdentity(request_id="rid-1", trace_id="trace-1"),
                route="/v1/extract",
                schema_id="sroie_receipt_v1",
                requested_model_id="model-a",
                outcome=RunOutcome.accepted(),
            ),
            response=ExtractExecutionResult(
                schema_id="sroie_receipt_v1",
                model="resolved-model",
                data={"company": "ACME"},
                cached=True,
                repair_attempted=False,
                prompt_tokens=10,
                completion_tokens=20,
                policy_generate_max_new_tokens_cap=512,
                effective_max_new_tokens=256,
                requested_max_new_tokens=256,
                clamped=False,
            ),
        )

    monkeypatch.setattr(extract_api, "set_trace_meta", fake_set_trace_meta, raising=True)
    monkeypatch.setattr(extract_api, "bind_request_span", fake_bind_request_span, raising=True)
    monkeypatch.setattr(extract_api, "run_extract_request", fake_run_extract_request, raising=True)

    response = await extract_api.extract(
        _request(),
        _body(),
        api_key=SimpleNamespace(key="proof-user-key"),
        llm=object(),
    )

    assert response.schema_id == "sroie_receipt_v1"
    assert response.model == "resolved-model"
    assert response.data == {"company": "ACME"}
    assert response.cached is True
    assert response.repair_attempted is False

    use_case_call = calls[1]
    assert use_case_call["route_label"] == "/v1/extract"
    assert use_case_call["redis"] is None
    assert span_calls == [
        {
            "request": _request(),
            "kwargs": {
                "name": "backend.extract",
                "route": "/v1/extract",
                "attributes": {
                    "llm.schema_id": "sroie_receipt_v1",
                    "llm.requested_model_id": "model-a",
                },
            },
        }
    ]
