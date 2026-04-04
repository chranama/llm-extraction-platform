from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from llm_server.application.process_extract_job import process_extract_job_once

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _force_lazy_model_mode(app):
    app.state.settings.model_load_mode = "lazy"
    app.state.model_load_mode = "lazy"
    app.state.model_loaded = True
    yield


async def _mk_role_and_key(test_sessionmaker, *, role_name: str) -> str:
    from llm_server.db.models import ApiKey, RoleTable

    key = f"test_{uuid.uuid4().hex}"
    async with test_sessionmaker() as session:
        role = (
            await session.execute(select(RoleTable).where(RoleTable.name == role_name))
        ).scalar_one_or_none()
        if role is None:
            role = RoleTable(name=role_name)
            session.add(role)
            await session.flush()
        session.add(ApiKey(key=key, active=True, role_id=role.id, quota_monthly=None, quota_used=0))
        await session.commit()
    return key


@pytest.fixture
async def admin_headers(test_sessionmaker):
    return {"X-API-Key": await _mk_role_and_key(test_sessionmaker, role_name="admin")}


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
async def test_sync_extract_trace_visible_via_admin(
    client, auth_headers, admin_headers, schema_dir
):
    r = await client.post(
        "/v1/extract",
        headers=auth_headers,
        json={"schema_id": "a", "text": "id 1", "cache": False, "repair": True},
    )
    assert r.status_code == 200, r.text
    trace_id = r.headers["X-Request-ID"]

    detail = await client.get(f"/v1/admin/traces/{trace_id}", headers=admin_headers)
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["trace_id"] == trace_id
    assert body["request_kind"] == "sync_extract"
    names = [item["event_name"] for item in body["events"]]
    assert "extract.accepted" in names
    assert "extract.model_resolved" in names
    assert "extract.validation_completed" in names
    assert "extract.completed" in names

    logs = await client.get(f"/v1/admin/logs?request_id={trace_id}", headers=admin_headers)
    assert logs.status_code == 200, logs.text
    logs_body = logs.json()
    assert logs_body["total"] >= 1
    assert all(item["request_id"] == trace_id for item in logs_body["items"])


@pytest.mark.anyio
async def test_async_extract_trace_spans_submit_worker_and_poll(
    client,
    app,
    auth_headers,
    admin_headers,
    extract_job_queue,
    test_sessionmaker,
    schema_dir,
):
    app.state.extract_job_queue = extract_job_queue

    submit = await client.post(
        "/v1/extract/jobs",
        headers=auth_headers,
        json={"schema_id": "a", "text": "id 1", "cache": False, "repair": True},
    )
    assert submit.status_code == 202, submit.text
    submit_body = submit.json()
    trace_id = submit_body["trace_id"]
    assert trace_id

    result = await process_extract_job_once(
        app=app,
        sessionmaker=test_sessionmaker,
        queue=extract_job_queue,
        timeout_seconds=1,
    )
    assert result is not None
    assert result.status == "succeeded"

    status_r = await client.get(submit_body["poll_path"], headers=auth_headers)
    assert status_r.status_code == 200, status_r.text
    status_body = status_r.json()
    assert status_body["trace_id"] == trace_id

    detail = await client.get(f"/v1/admin/traces/{trace_id}", headers=admin_headers)
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["request_kind"] == "async_extract"
    assert body["status"] == "completed"
    names = [item["event_name"] for item in body["events"]]
    assert "extract_job.submitted" in names
    assert "extract_job.persisted" in names
    assert "extract_job.queued" in names
    assert "extract_job.worker_claimed" in names
    assert "extract_job.execution_started" in names
    assert "extract_job.completed" in names
    assert "extract_job.status_polled" in names


@pytest.mark.anyio
async def test_failed_sync_extract_trace_records_error_stage(
    client, auth_headers, admin_headers, monkeypatch, tmp_path: Path, schema_dir
):
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
        "/v1/extract",
        headers=auth_headers,
        json={"schema_id": "a", "text": "id 1", "cache": False, "repair": True},
    )
    assert r.status_code == 400, r.text
    trace_id = r.headers["X-Request-ID"]

    detail = await client.get(f"/v1/admin/traces/{trace_id}", headers=admin_headers)
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["status"] == "failed"
    failed = [item for item in body["events"] if item["event_name"] == "extract.failed"]
    assert failed
    assert failed[-1]["details"]["error_code"] == "capability_not_supported"

    replay = await client.get(
        f"/v1/admin/replay-cases/traces/{trace_id}",
        headers=admin_headers,
    )
    assert replay.status_code == 200, replay.text
    replay_body = replay.json()
    assert replay_body["source"]["kind"] == "trace"
    assert replay_body["source"]["trace_id"] == trace_id
    case = replay_body["cases"][0]
    assert case["case_id"] == f"trace:{trace_id}"
    assert case["replay_ready"] is False
    assert case["missing_fields"]
    assert any(field in {"text", "schema_id"} for field in case["missing_fields"])
    assert case["expectation"]["status"] == "failed"
    assert case["expectation"]["error_code"] == "capability_not_supported"


@pytest.mark.anyio
async def test_sync_extract_uses_gateway_trace_id_in_behind_gateway_mode(
    app_client, monkeypatch, api_key, admin_headers, schema_dir
):
    monkeypatch.setenv("EDGE_MODE", "behind_gateway")
    monkeypatch.setenv("MODEL_LOAD_MODE", "lazy")

    async with await app_client() as client:
        client.app.state.settings.model_load_mode = "lazy"
        client.app.state.model_load_mode = "lazy"
        client.app.state.model_loaded = True
        headers = {
            "X-API-Key": api_key,
            "X-Request-ID": "sync-request-1",
            "X-Trace-ID": "sync-trace-1",
            "X-Gateway-Proxy": "inference-serving-gateway",
        }
        r = await client.post(
            "/v1/extract",
            headers=headers,
            json={"schema_id": "a", "text": "id 1", "cache": False, "repair": True},
        )
        assert r.status_code == 200, r.text
        assert r.headers["X-Request-ID"] == "sync-request-1"
        assert r.headers["X-Trace-ID"] == "sync-trace-1"

        detail = await client.get("/v1/admin/traces/sync-trace-1", headers=admin_headers)
        assert detail.status_code == 200, detail.text
        body = detail.json()
        assert body["trace_id"] == "sync-trace-1"
        assert any(
            item["event_name"] == "extract.accepted" and item["request_id"] == "sync-request-1"
            for item in body["events"]
        )

        logs = await client.get(
            "/v1/admin/logs?request_id=sync-request-1",
            headers=admin_headers,
        )
        assert logs.status_code == 200, logs.text
        logs_body = logs.json()
        assert logs_body["total"] >= 1
        assert logs_body["items"][0]["trace_id"] == "sync-trace-1"
        assert logs_body["items"][0]["job_id"] is None


@pytest.mark.anyio
async def test_async_extract_preserves_gateway_trace_with_split_poll_request_id(
    app_client,
    monkeypatch,
    api_key,
    admin_headers,
    extract_job_queue,
    test_sessionmaker,
    schema_dir,
):
    monkeypatch.setenv("EDGE_MODE", "behind_gateway")
    monkeypatch.setenv("MODEL_LOAD_MODE", "lazy")

    async with await app_client() as client:
        client.app.state.settings.model_load_mode = "lazy"
        client.app.state.model_load_mode = "lazy"
        client.app.state.model_loaded = True
        client.app.state.extract_job_queue = extract_job_queue

        submit_headers = {
            "X-API-Key": api_key,
            "X-Request-ID": "submit-request-1",
            "X-Trace-ID": "shared-trace-1",
            "X-Gateway-Proxy": "inference-serving-gateway",
        }
        submit = await client.post(
            "/v1/extract/jobs",
            headers=submit_headers,
            json={"schema_id": "a", "text": "id 1", "cache": False, "repair": True},
        )
        assert submit.status_code == 202, submit.text
        assert submit.headers["X-Request-ID"] == "submit-request-1"
        assert submit.headers["X-Trace-ID"] == "shared-trace-1"
        submit_body = submit.json()
        assert submit_body["trace_id"] == "shared-trace-1"

        from llm_server.db.models import ExtractJob

        async with test_sessionmaker() as session:
            row = (
                await session.execute(
                    select(ExtractJob).where(ExtractJob.id == submit_body["job_id"])
                )
            ).scalar_one()
            assert row.request_id == "submit-request-1"
            assert row.trace_id == "shared-trace-1"

        result = await process_extract_job_once(
            app=client.app,
            sessionmaker=test_sessionmaker,
            queue=extract_job_queue,
            timeout_seconds=1,
        )
        assert result is not None
        assert result.status == "succeeded"

        poll_headers = {
            "X-API-Key": api_key,
            "X-Request-ID": "poll-request-1",
            "X-Trace-ID": "shared-trace-1",
            "X-Gateway-Proxy": "inference-serving-gateway",
        }
        status_r = await client.get(submit_body["poll_path"], headers=poll_headers)
        assert status_r.status_code == 200, status_r.text
        assert status_r.headers["X-Request-ID"] == "poll-request-1"
        assert status_r.headers["X-Trace-ID"] == "shared-trace-1"
        status_body = status_r.json()
        assert status_body["trace_id"] == "shared-trace-1"

        detail = await client.get("/v1/admin/traces/shared-trace-1", headers=admin_headers)
        assert detail.status_code == 200, detail.text
        body = detail.json()
        assert body["trace_id"] == "shared-trace-1"
        assert any(
            item["event_name"] == "extract_job.submitted"
            and item["request_id"] == "submit-request-1"
            for item in body["events"]
        )
        assert any(
            item["event_name"] == "extract_job.status_polled"
            and item["request_id"] == "poll-request-1"
            for item in body["events"]
        )

        logs = await client.get(
            f"/v1/admin/logs?trace_id=shared-trace-1&job_id={submit_body['job_id']}",
            headers=admin_headers,
        )
        assert logs.status_code == 200, logs.text
        logs_body = logs.json()
        assert logs_body["total"] >= 1
        assert any(
            item["trace_id"] == "shared-trace-1"
            and item["job_id"] == submit_body["job_id"]
            and item["route"] == "/v1/extract/jobs/worker"
            for item in logs_body["items"]
        )
