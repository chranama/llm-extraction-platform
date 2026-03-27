from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from llm_server.core.errors import AppError
from llm_server.domain.outcomes import RunStatus
from llm_server.services.extract_execution import ExtractExecutionResult

run_extract_app = importlib.import_module("llm_server.application.run_extract")


def _ctx(
    *,
    request_id: str = "rid-1",
    trace_id: str | None = "trace-1",
    job_id: str | None = None,
):
    return SimpleNamespace(
        state=SimpleNamespace(
            request_id=request_id,
            trace_id=trace_id,
            trace_job_id=job_id,
        )
    )


def _body():
    return SimpleNamespace(
        schema_id="sroie_receipt_v1",
        model="model-a",
        cache=True,
        repair=True,
        max_new_tokens=512,
    )


def test_build_extraction_run_uses_request_context():
    run = run_extract_app.build_extraction_run(
        ctx=_ctx(request_id="rid-1", trace_id="trace-1", job_id="job-1"),
        body=_body(),
        route_label="/v1/extract",
    )

    assert run.request_id == "rid-1"
    assert run.trace_id == "trace-1"
    assert run.job_id == "job-1"
    assert run.route == "/v1/extract"
    assert run.schema_id == "sroie_receipt_v1"
    assert run.requested_model_id == "model-a"
    assert run.outcome.status is RunStatus.ACCEPTED


def test_build_extraction_run_falls_back_to_request_id_for_trace():
    run = run_extract_app.build_extraction_run(
        ctx=_ctx(request_id="rid-2", trace_id=None),
        body=_body(),
    )

    assert run.request_id == "rid-2"
    assert run.trace_id == "rid-2"


def test_apply_extract_error_maps_code_and_stage():
    run = run_extract_app.build_extraction_run(ctx=_ctx(), body=_body())
    error = AppError(
        code="schema_not_found",
        message="missing schema",
        status_code=404,
        extra={"stage": "load_schema"},
    )

    failed = run_extract_app.apply_extract_error(run, error)

    assert failed.outcome.status is RunStatus.FAILED
    assert failed.outcome.error_code == "schema_not_found"
    assert failed.outcome.error_stage == "load_schema"


@pytest.mark.anyio
async def test_run_extract_returns_run_with_result_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_execute_extract(**_: object) -> ExtractExecutionResult:
        return ExtractExecutionResult(
            schema_id="sroie_receipt_v1",
            model="resolved-model",
            data={"company": "ACME"},
            cached=True,
            repair_attempted=False,
            prompt_tokens=17,
            completion_tokens=33,
            policy_generate_max_new_tokens_cap=384,
            effective_max_new_tokens=256,
            requested_max_new_tokens=512,
            clamped=True,
        )

    monkeypatch.setattr(run_extract_app, "execute_extract", fake_execute_extract)

    result = await run_extract_app.run_extract(
        ctx=_ctx(),
        body=_body(),
        api_key=SimpleNamespace(key="proof-user-key"),
        llm=object(),
        session=object(),
        redis=None,
    )

    assert result.response.model == "resolved-model"
    assert result.run.request_id == "rid-1"
    assert result.run.trace_id == "trace-1"
    assert result.run.resolved_model_id == "resolved-model"
    assert result.run.effective_max_new_tokens == 256
    assert result.run.policy is not None
    assert result.run.policy.generate_max_new_tokens_cap == 384
    assert result.run.outcome.status is RunStatus.SUCCEEDED
    assert result.run.outcome.cached is True
    assert result.run.outcome.completion_tokens == 33
