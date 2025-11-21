import pytest

@pytest.mark.anyio
async def test_generate_requires_key(client):
    r = await client.post("/v1/generate", json={"prompt": "hi"})
    assert r.status_code == 401
    body = r.json()
    assert body["detail"]["code"] in {"missing_api_key", "invalid_api_key"}

@pytest.mark.anyio
async def test_generate_with_key(client, api_key, mock_model):
    r = await client.post(
        "/v1/generate",
        headers={"X-API-Key": api_key},
        json={"prompt": "hi", "max_new_tokens": 8}
    )
    assert r.status_code == 200
    j = r.json()
    assert j["model"] == "dummy/model"
    assert "output" in j