from __future__ import annotations

import pytest

from llm_server.services.api_deps.core import auth

pytestmark = pytest.mark.integration


@pytest.mark.anyio
async def test_missing_api_key(client):
    r = await client.get("/v1/schemas")
    assert r.status_code == 401, r.text
    body = r.json()
    assert body["code"] == "missing_api_key"


@pytest.mark.anyio
async def test_invalid_api_key(client):
    r = await client.get("/v1/schemas", headers={"X-API-Key": "bad"})
    assert r.status_code == 401, r.text
    body = r.json()
    assert body["code"] in ("invalid_api_key", "unauthorized")


@pytest.mark.anyio
async def test_rate_limit(monkeypatch, client, auth_headers):
    monkeypatch.setattr(auth, "_role_rpm", lambda _role: 1)
    auth.clear_rate_limit_state()

    r1 = await client.get("/v1/schemas", headers=auth_headers)
    assert r1.status_code == 200, r1.text

    r2 = await client.get("/v1/schemas", headers=auth_headers)
    assert r2.status_code == 429, r2.text
    body = r2.json()
    assert body["code"] in ("rate_limited", "too_many_requests")
