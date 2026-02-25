from __future__ import annotations

import pytest
from sqlalchemy import select

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _force_lazy_model_mode(app):
    app.state.settings.model_load_mode = "lazy"
    app.state.model_load_mode = "lazy"
    yield


@pytest.mark.anyio
async def test_models_reflect_generate_only(client):
    r = await client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()

    assert body["deployment_capabilities"]["generate"] is True
    assert body["deployment_capabilities"]["extract"] is False

    assert "default_model" in body and body["default_model"]

    default_id = body["default_model"]
    models = {m["id"]: m for m in body["models"]}
    assert default_id in models

    caps = models[default_id].get("capabilities") or {}
    assert caps.get("generate") is True
    assert caps.get("extract") is False


@pytest.mark.anyio
async def test_generate_works(client, auth_headers):
    r = await client.post(
        "/v1/generate", headers=auth_headers, json={"prompt": "hi", "cache": False}
    )
    assert r.status_code == 200
    assert str(r.json()["output"]).strip().lower() == "ok"


@pytest.mark.anyio
async def test_extract_is_disabled(client, auth_headers):
    r = await client.post(
        "/v1/extract",
        headers=auth_headers,
        json={"schema_id": "ticket_v1", "text": "hello"},
    )

    assert r.status_code == 501
    body = r.json()
    assert body["code"] == "capability_disabled"
    assert body["extra"]["capability"] == "extract"


@pytest.mark.anyio
async def test_generate_log_written(client, auth_headers, test_sessionmaker):
    await client.post(
        "/v1/generate", headers=auth_headers, json={"prompt": "hi", "cache": False}
    )

    from llm_server.db.models import InferenceLog

    async with test_sessionmaker() as session:
        rows = (await session.execute(select(InferenceLog))).scalars().all()
        assert len(rows) >= 1
