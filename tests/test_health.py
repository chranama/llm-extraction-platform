import pytest

@pytest.mark.anyio
async def test_healthz(client):
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

@pytest.mark.anyio
async def test_readyz(client):
    r = await client.get("/readyz")
    assert r.status_code == 200
    assert "status" in r.json()