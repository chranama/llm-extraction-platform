import pytest

@pytest.mark.anyio
async def test_rate_limit_window(client, api_key, mock_model, monkeypatch):
    # monkeypatch deps._role_rpm to force small limit (e.g., 1 rpm)
    import llm_server.api.deps as deps
    monkeypatch.setattr(deps, "_role_rpm", lambda role: 1)

    h = {"X-API-Key": api_key}
    body = {"prompt":"rl", "max_new_tokens":2}

    ok = await client.post("/v1/generate", json=body, headers=h)
    assert ok.status_code == 200

    limited = await client.post("/v1/generate", json=body, headers=h)
    assert limited.status_code == 429
    assert limited.json()["detail"]["code"] == "rate_limited"