# tests/test_integration_generate.py
import pytest
from sqlalchemy import select
from llm_server.db.models import InferenceLog, CompletionCache

@pytest.mark.anyio
async def test_generate_happy_path(client, api_key, mock_model, db_session):
    body = {"prompt": "Integration test prompt", "max_new_tokens": 8}
    headers = {"X-API-Key": api_key}

    r = await client.post("/v1/generate", json=body, headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == "dummy/model"
    assert "[DUMMY COMPLETION" in data["output"]

    # DB logs present
    res = await db_session.execute(select(InferenceLog).where(InferenceLog.prompt == body["prompt"]))
    log = res.scalar_one_or_none()
    assert log is not None
    assert log.api_key == api_key

    # Cache entry created
    res = await db_session.execute(select(CompletionCache).where(CompletionCache.prompt == body["prompt"]))
    cache = res.scalar_one_or_none()
    assert cache is not None


@pytest.mark.anyio
async def test_generate_cache_hit_has_log(client, api_key, mock_model, db_session):
    body = {"prompt": "Cached prompt", "max_new_tokens": 8}
    headers = {"X-API-Key": api_key}

    # First request (fills cache)
    r1 = await client.post("/v1/generate", json=body, headers=headers)
    assert r1.status_code == 200

    # Second request (should hit cache and still log)
    r2 = await client.post("/v1/generate", json=body, headers=headers)
    assert r2.status_code == 200

    # There should be at least two logs for the same prompt now
    res = await db_session.execute(select(InferenceLog).where(InferenceLog.prompt == body["prompt"]))
    logs = res.scalars().all()
    assert len(logs) >= 2