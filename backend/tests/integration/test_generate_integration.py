import pytest
from sqlalchemy import select

pytestmark = pytest.mark.integration

@pytest.mark.anyio
async def test_generate(client, auth_headers):
    r = await client.post("/v1/generate", headers=auth_headers, json={"prompt": "hi", "cache": False})
    assert r.status_code == 200
    assert "output" in r.json()

@pytest.mark.anyio
async def test_generate_log_written(client, auth_headers, test_sessionmaker):
    await client.post("/v1/generate", headers=auth_headers, json={"prompt": "hi", "cache": False})

    from llm_server.db.models import InferenceLog
    async with test_sessionmaker() as session:
        rows = (await session.execute(select(InferenceLog))).scalars().all()
        assert len(rows) >= 1