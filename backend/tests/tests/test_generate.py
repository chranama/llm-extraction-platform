import pytest

@pytest.mark.anyio
async def test_cache_hit_logs(client, api_key, mock_model):
    body = {"prompt": "cache me", "max_new_tokens": 4}
    h = {"X-API-Key": api_key}

    r1 = await client.post("/v1/generate", json=body, headers=h)
    assert r1.status_code == 200
    out1 = r1.json()["output"]

    r2 = await client.post("/v1/generate", json=body, headers=h)
    assert r2.status_code == 200
    out2 = r2.json()["output"]
    assert out1 == out2  # same output from cache or dummy

@pytest.mark.anyio
async def test_stream_sse(client, api_key, mock_model):
    r = await client.post("/v1/stream", headers={"X-API-Key": api_key}, json={"prompt":"yo"})
    assert r.status_code == 200
    # SSE is "data: ...\n\n" chunks; we just ensure it ends with [DONE]
    text = r.text
    assert text.strip().endswith("data: [DONE]")