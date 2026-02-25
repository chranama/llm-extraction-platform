from __future__ import annotations

import pytest

from llm_server.services.api_deps.enforcement import capabilities

pytestmark = pytest.mark.integration


@pytest.mark.anyio
async def test_models_endpoint_deployment_and_model_caps_reflected(monkeypatch, client):
    monkeypatch.setattr(
        client.app.state.settings, "model_load_mode", "off", raising=False
    )
    monkeypatch.setattr(client.app.state.settings, "model_id", "modelA", raising=False)
    monkeypatch.setattr(
        client.app.state.settings, "allowed_models", ["modelA", "modelB"], raising=False
    )
    monkeypatch.setattr(
        client.app.state.settings, "enable_generate", True, raising=False
    )
    monkeypatch.setattr(
        client.app.state.settings, "enable_extract", False, raising=False
    )

    monkeypatch.setattr(
        capabilities,
        "_model_capabilities_from_models_yaml",
        lambda model_id, request=None: {"extract": False}
        if model_id == "modelA"
        else None,
    )

    r = await client.get("/v1/models")
    assert r.status_code == 200
    payload = r.json()

    assert payload["deployment_capabilities"]["extract"] is False

    by_id = {m["id"]: m for m in payload["models"]}
    assert by_id["modelA"]["capabilities"]["extract"] is False
    assert by_id["modelB"]["capabilities"]["extract"] is False
