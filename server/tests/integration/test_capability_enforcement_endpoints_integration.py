from __future__ import annotations

import pytest

from llm_server.services.api_deps.enforcement import capabilities

pytestmark = pytest.mark.integration


@pytest.mark.anyio
async def test_deployment_disable_extract_returns_501(
    monkeypatch, client, auth_headers
):
    monkeypatch.setattr(
        client.app.state.settings, "enable_extract", False, raising=False
    )

    r = await client.post(
        "/v1/extract",
        headers=auth_headers,
        json={"schema_id": "ticket_v1", "text": "x", "cache": False, "repair": False},
    )

    assert r.status_code == 501, r.text
    body = r.json()
    assert body["code"] == "capability_disabled"


@pytest.mark.anyio
async def test_model_lacks_extract_returns_400(monkeypatch, client, auth_headers):
    monkeypatch.setattr(
        client.app.state.settings, "enable_extract", True, raising=False
    )
    monkeypatch.setattr(
        capabilities,
        "model_capabilities",
        lambda _model_id, request=None: {"extract": False, "generate": True},
    )

    r = await client.post(
        "/v1/extract",
        headers=auth_headers,
        json={"schema_id": "ticket_v1", "text": "x", "cache": False, "repair": False},
    )

    assert r.status_code == 400, r.text
    body = r.json()
    assert body["code"] == "capability_not_supported"
