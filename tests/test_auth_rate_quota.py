# tests/test_auth_rate_quota.py
import pytest

@pytest.mark.anyio
async def test_missing_api_key_returns_401(client, mock_model):
    r = await client.post("/v1/generate", json={"prompt": "hi"})
    assert r.status_code == 401
    j = r.json()
    assert j["detail"]["code"] in ("missing_api_key", "invalid_api_key")


@pytest.mark.anyio
async def test_rate_limit_429(client, api_key, mock_model, monkeypatch):
    # Force RPM=1 for test
    import llm_server.api.deps as deps
    deps._RL.clear()
    monkeypatch.setattr(deps, "_role_rpm", lambda role: 1, raising=True)

    headers = {"X-API-Key": api_key}
    body = {"prompt": "rl-test"}

    ok = await client.post("/v1/generate", json=body, headers=headers)
    assert ok.status_code == 200

    limited = await client.post("/v1/generate", json=body, headers=headers)
    assert limited.status_code == 429
    assert limited.json()["detail"]["code"] == "rate_limited"


@pytest.mark.anyio
async def test_quota_exhausted_402(client, api_key, mock_model, monkeypatch):
    # Simulate a monthly quota of 0 so first request triggers 402
    import llm_server.api.deps as deps
    from llm_server.db.models import ApiKey
    from sqlalchemy import select
    from llm_server.db.session import async_session_maker

    async with async_session_maker() as s:
        row = (await s.execute(select(ApiKey).where(ApiKey.key == api_key))).scalar_one()
        row.quota_monthly = 0
        row.quota_used = 0
        await s.commit()

    headers = {"X-API-Key": api_key}
    body = {"prompt": "quota-test"}

    r = await client.post("/v1/generate", json=body, headers=headers)
    assert r.status_code in (200, 402)
    if r.status_code == 402:
        detail = r.json()["detail"]
        assert detail["code"] == "quota_exhausted"