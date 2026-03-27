from __future__ import annotations

from llm_server.domain.outcomes import RunOutcome, RunStatus
from llm_server.domain.runs import ExtractionRun, RunIdentity, RunPolicySnapshot


def test_extraction_run_tracks_identity_and_request_shape():
    run = ExtractionRun(
        identity=RunIdentity(request_id="rid-1", trace_id="trace-1", job_id="job-1"),
        route="/v1/extract",
        schema_id="sroie_receipt_v1",
        requested_model_id="model-a",
        cache_enabled=False,
        repair_enabled=True,
        requested_max_new_tokens=256,
    )

    assert run.request_id == "rid-1"
    assert run.trace_id == "trace-1"
    assert run.job_id == "job-1"
    assert run.schema_id == "sroie_receipt_v1"
    assert run.requested_model_id == "model-a"
    assert run.cache_enabled is False
    assert run.repair_enabled is True
    assert run.requested_max_new_tokens == 256
    assert run.outcome.status is RunStatus.ACCEPTED


def test_extraction_run_update_helpers_return_new_instances():
    original = ExtractionRun(
        identity=RunIdentity(request_id="rid-1", trace_id="trace-1"),
        route="/v1/extract",
        schema_id="sroie_receipt_v1",
    )

    updated = (
        original.with_resolution(
            resolved_model_id="resolved-model",
            effective_max_new_tokens=384,
        )
        .with_policy(RunPolicySnapshot(generate_max_new_tokens_cap=512))
        .with_outcome(
            RunOutcome.succeeded(
                cached=True,
                repair_attempted=False,
                prompt_tokens=12,
                completion_tokens=34,
            )
        )
    )

    assert original.resolved_model_id is None
    assert original.policy is None
    assert original.outcome.status is RunStatus.ACCEPTED

    assert updated.resolved_model_id == "resolved-model"
    assert updated.effective_max_new_tokens == 384
    assert updated.policy is not None
    assert updated.policy.generate_max_new_tokens_cap == 512
    assert updated.outcome.status is RunStatus.SUCCEEDED
    assert updated.outcome.cached is True
    assert updated.outcome.prompt_tokens == 12
