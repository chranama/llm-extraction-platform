from __future__ import annotations

import pytest

from llm_server.services.llm_runtime.llm_registry import MultiModelManager
from llm_server.services.llm_runtime.model_state import ModelStateStore

pytestmark = pytest.mark.integration


class _FakeModel:
    def __init__(self, output: str):
        self._output = output
        self.backend_name = "transformers"

    def ensure_loaded(self):
        return None

    def is_loaded(self):
        return True

    def generate(self, **kwargs):
        return self._output


@pytest.mark.anyio
async def test_extract_uses_default_for_capability_generate_uses_default(
    client, auth_headers
):
    mm = MultiModelManager(
        models={
            "default_gen": _FakeModel("GEN_OK"),
            "extractor": _FakeModel(
                '{"company":"Store","date":"2026-01-01","total":"1.00"}'
            ),
        },
        default_id="default_gen",
        model_meta={
            "default_gen": {"capabilities": ["generate"]},
            "extractor": {"capabilities": ["extract"]},
        },
    )

    client.app.state.llm = mm
    client.app.state.runtime_default_model_id = None
    client.app.state.settings.allowed_models = ["default_gen", "extractor"]
    client.app.state.settings.model_id = "default_gen"

    ms = ModelStateStore(client.app.state)
    ms.set_model_load_mode("lazy")
    ms.set_loaded_model_id("default_gen")
    ms.set_model_loaded(True)

    r1 = await client.post(
        "/v1/generate", headers=auth_headers, json={"prompt": "hi", "cache": False}
    )
    r2 = await client.post(
        "/v1/extract",
        headers=auth_headers,
        json={
            "schema_id": "sroie_receipt_v1",
            "text": "x",
            "cache": False,
            "repair": False,
        },
    )

    assert r1.status_code == 200, r1.text
    assert r1.json()["model"] == "default_gen"

    assert r2.status_code == 200, r2.text
    assert r2.json()["model"] == "extractor"
