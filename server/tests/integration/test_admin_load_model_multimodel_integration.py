from __future__ import annotations

import uuid

import pytest

from llm_server.services.llm_runtime.llm_registry import MultiModelManager
from llm_server.services.llm_runtime.model_state import ModelStateStore

pytestmark = pytest.mark.integration


class _Backend:
    backend_name = "transformers"

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._loaded = False
        self.ensure_loaded_calls = 0

    def ensure_loaded(self):
        self.ensure_loaded_calls += 1
        self._loaded = True

    def is_loaded(self):
        return self._loaded


async def _make_admin_headers(test_sessionmaker):
    from llm_server.db.models import ApiKey, RoleTable

    key = f"test_{uuid.uuid4().hex}"

    async with test_sessionmaker() as session:
        role = RoleTable(name="admin")
        session.add(role)
        await session.flush()

        session.add(
            ApiKey(
                key=key,
                active=True,
                role_id=role.id,
                quota_monthly=None,
                quota_used=0,
            )
        )
        await session.commit()

    return {"X-API-Key": key}


@pytest.mark.anyio
async def test_admin_models_load_multimodel_from_off(client, test_sessionmaker):
    headers = await _make_admin_headers(test_sessionmaker)

    mm = MultiModelManager(
        models={"m1": _Backend("m1"), "m2": _Backend("m2")},
        default_id="m1",
    )

    client.app.state.llm = mm
    ms = ModelStateStore(client.app.state)
    ms.set_model_load_mode("off")
    ms.set_model_loaded(False)
    ms.set_loaded_model_id(None)

    r = await client.post("/v1/admin/models/load", headers=headers, json={})

    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["ok"] is True
    assert payload["model_id"] == "m1"
    assert payload["loaded"] is True
    assert payload["load_mode"] == "off"

    assert mm.models["m1"].ensure_loaded_calls == 1
    assert mm.models["m2"].ensure_loaded_calls == 0
