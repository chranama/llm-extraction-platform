from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_server.services.extract_jobs import process_extract_job_once

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _force_lazy_model_mode(app):
    app.state.settings.model_load_mode = "lazy"
    app.state.model_load_mode = "lazy"
    yield


@pytest.fixture
def llm_outputs():
    return ['{"id":"1"}']


@pytest.fixture
def schema_dir(tmp_path: Path, monkeypatch):
    schema = {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
        "additionalProperties": False,
    }
    (tmp_path / "a.json").write_text(json.dumps(schema), encoding="utf-8")
    monkeypatch.setenv("SCHEMAS_DIR", str(tmp_path))

    import llm_server.core.schema_registry as reg

    reg._SCHEMA_CACHE.clear()
    return tmp_path


@pytest.mark.anyio
async def test_submit_extract_job_returns_queued_and_status_visible(
    client, app, auth_headers, extract_job_queue, schema_dir
):
    app.state.extract_job_queue = extract_job_queue

    r = await client.post(
        "/v1/extract/jobs",
        headers=auth_headers,
        json={"schema_id": "a", "text": "id 1", "cache": False, "repair": True},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "queued"
    assert body["poll_path"].endswith(body["job_id"])

    s = await client.get(body["poll_path"], headers=auth_headers)
    assert s.status_code == 200, s.text
    status_body = s.json()
    assert status_body["job_id"] == body["job_id"]
    assert status_body["status"] == "queued"


@pytest.mark.anyio
async def test_non_owner_gets_404_for_extract_job(
    client, app, auth_headers, extract_job_queue, api_key, test_sessionmaker, schema_dir
):
    app.state.extract_job_queue = extract_job_queue

    r = await client.post(
        "/v1/extract/jobs",
        headers=auth_headers,
        json={"schema_id": "a", "text": "id 1", "cache": False, "repair": True},
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    from llm_server.db.models import ApiKey

    other = "other_key"
    async with test_sessionmaker() as session:
        session.add(ApiKey(key=other, active=True, quota_monthly=None, quota_used=0))
        await session.commit()

    s = await client.get(f"/v1/extract/jobs/{job_id}", headers={"X-API-Key": other})
    assert s.status_code == 404, s.text


@pytest.mark.anyio
async def test_worker_processes_extract_job_to_success(
    client, app, auth_headers, extract_job_queue, test_sessionmaker, schema_dir
):
    app.state.extract_job_queue = extract_job_queue

    r = await client.post(
        "/v1/extract/jobs",
        headers=auth_headers,
        json={"schema_id": "a", "text": "id 1", "cache": False, "repair": True},
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    result = await process_extract_job_once(
        app=app,
        sessionmaker=test_sessionmaker,
        queue=extract_job_queue,
        timeout_seconds=1,
    )
    assert result is not None
    assert result.status == "succeeded"

    s = await client.get(f"/v1/extract/jobs/{job_id}", headers=auth_headers)
    assert s.status_code == 200, s.text
    body = s.json()
    assert body["status"] == "succeeded"
    assert body["result"]["id"] == "1"


@pytest.mark.anyio
async def test_submission_fails_when_policy_blocks_extract(
    client, app, auth_headers, extract_job_queue, monkeypatch, tmp_path: Path, schema_dir
):
    app.state.extract_job_queue = extract_job_queue
    p = tmp_path / "policy.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": "policy_decision_v2",
                "generated_at": "2026-01-01T00:00:00Z",
                "policy": "extract_enablement",
                "pipeline": "extract_only",
                "status": "allow",
                "ok": True,
                "enable_extract": False,
                "generate_max_new_tokens_cap": None,
                "contract_errors": 0,
                "thresholds_profile": "default",
                "thresholds_version": "v1",
                "eval_run_dir": "/tmp/eval",
                "eval_task": "extract",
                "eval_run_id": "run-1",
                "model_id": None,
                "reasons": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))
    try:
        delattr(client.app.state, "policy_snapshot")
    except Exception:
        pass

    r = await client.post(
        "/v1/extract/jobs",
        headers=auth_headers,
        json={"schema_id": "a", "text": "id 1", "cache": False, "repair": True},
    )
    assert r.status_code == 400, r.text
    assert r.json()["code"] == "capability_not_supported"


@pytest.mark.anyio
async def test_worker_fails_job_when_policy_drifts_after_submission(
    client, app, auth_headers, extract_job_queue, test_sessionmaker, monkeypatch, tmp_path: Path, schema_dir
):
    app.state.extract_job_queue = extract_job_queue

    r = await client.post(
        "/v1/extract/jobs",
        headers=auth_headers,
        json={"schema_id": "a", "text": "id 1", "cache": False, "repair": True},
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    p = tmp_path / "policy.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": "policy_decision_v2",
                "generated_at": "2026-01-01T00:00:00Z",
                "policy": "extract_enablement",
                "pipeline": "extract_only",
                "status": "allow",
                "ok": True,
                "enable_extract": False,
                "generate_max_new_tokens_cap": None,
                "contract_errors": 0,
                "thresholds_profile": "default",
                "thresholds_version": "v1",
                "eval_run_dir": "/tmp/eval",
                "eval_task": "extract",
                "eval_run_id": "run-1",
                "model_id": None,
                "reasons": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))
    try:
        delattr(app.state, "policy_snapshot")
    except Exception:
        pass

    result = await process_extract_job_once(
        app=app,
        sessionmaker=test_sessionmaker,
        queue=extract_job_queue,
        timeout_seconds=1,
    )
    assert result is not None
    assert result.status == "failed"

    s = await client.get(f"/v1/extract/jobs/{job_id}", headers=auth_headers)
    assert s.status_code == 200, s.text
    body = s.json()
    assert body["status"] == "failed"
    assert body["error"]["code"] == "capability_not_supported"
