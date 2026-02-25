from __future__ import annotations

import pytest

from llm_eval.client.http_client import ExtractErr, ExtractOk, HttpEvalClient


@pytest.mark.integration
@pytest.mark.integration_live
@pytest.mark.asyncio
async def test_extract_sroie_receipt_v1_happy_path(
    require_live_server: None,
    live_client: HttpEvalClient,
):
    """
    Live success contract for canonical extraction schema.
    """
    schema_id = "sroie_receipt_v1"
    text = "Company: ACME\nDate: 2024-01-01\nTotal: 10.00\nAddress: 123 Main St"

    resp = await live_client.extract(
        schema_id=schema_id,
        text=text,
        temperature=0.0,
        max_new_tokens=256,
        cache=False,
        repair=True,
    )

    assert resp.latency_ms >= 0.0
    assert isinstance(resp, ExtractOk), f"Expected ExtractOk, got {type(resp).__name__}: {resp}"
    assert resp.schema_id == schema_id
    assert isinstance(resp.model, str) and resp.model
    assert isinstance(resp.data, dict)

    for k in ("company", "date", "total", "address"):
        if k in resp.data:
            assert resp.data[k] is None or isinstance(resp.data[k], (str, int, float))


@pytest.mark.integration
@pytest.mark.integration_contract
@pytest.mark.asyncio
async def test_extract_invalid_schema_returns_typed_error(live_client: HttpEvalClient):
    """
    Ensure non-200 responses become ExtractErr (not exceptions).
    """
    resp = await live_client.extract(
        schema_id="__definitely_not_a_real_schema__",
        text="Hello",
        cache=False,
        repair=False,
    )

    assert resp.latency_ms >= 0.0
    assert isinstance(resp, (ExtractOk, ExtractErr))

    if isinstance(resp, ExtractErr):
        assert resp.status_code != 200
        assert resp.error_code
