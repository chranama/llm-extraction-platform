# tests/test_health_metrics.py
import pytest

@pytest.mark.anyio
async def test_healthz(client):
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.anyio
async def test_readyz(client):
    r = await client.get("/readyz")
    assert r.status_code == 200
    # with dummy model ensure_loaded() always passes, so expect "ready"
    assert r.json()["status"] in ("ready", "ok")


@pytest.mark.anyio
async def test_metrics(client):
    r = await client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "python_gc_objects_collected_total" in text