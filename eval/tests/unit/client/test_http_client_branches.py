from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
import pytest

import llm_eval.client.http_client as hc


class _FakeAsyncClient:
    def __init__(self, responses: List[httpx.Response], timeout: float):
        self._responses = list(responses)
        self.timeout = timeout

    async def post(self, url: str, json: Dict[str, Any], headers: Dict[str, str]) -> httpx.Response:
        if not self._responses:
            raise AssertionError("No responses")
        return self._responses.pop(0)

    async def get(self, url: str, headers: Dict[str, str]) -> httpx.Response:
        if not self._responses:
            raise AssertionError("No responses")
        return self._responses.pop(0)

    async def aclose(self) -> None:
        return None


def _patch(monkeypatch: pytest.MonkeyPatch, responses: List[httpx.Response]):
    created = {"client": None}

    def _factory(*, timeout: float):
        c = _FakeAsyncClient(responses=responses, timeout=timeout)
        created["client"] = c
        return c

    monkeypatch.setattr(hc.httpx, "AsyncClient", _factory)
    return created


class _TimeSeq:
    def __init__(self, seq: list[float]):
        self.seq = seq

    def __call__(self) -> float:
        if not self.seq:
            raise AssertionError("time exhausted")
        return self.seq.pop(0)


def test_shorten_and_extract_helpers():
    assert hc.HttpEvalClient._shorten("abcdef", 4) == "a..."

    assert hc.HttpEvalClient._extract_text_from_generate_payload(None) == ""
    assert hc.HttpEvalClient._extract_text_from_generate_payload({"x": 1}) == "{'x': 1}"

    assert hc.HttpEvalClient._extract_model_from_payload({"data": {"model_id": "m1"}}) == "m1"


def test_extract_request_id_from_headers_and_body():
    req = httpx.Request("GET", "http://x")

    r_header = httpx.Response(500, request=req, headers={"X-Request-ID": "rid-h"}, json={})
    assert hc.HttpEvalClient._extract_request_id(r_header, {}) == "rid-h"

    r_body = httpx.Response(500, request=req, headers={}, json={"request_id": "rid-b"})
    assert hc.HttpEvalClient._extract_request_id(r_body, {"request_id": "rid-b"}) == "rid-b"

    r_none = httpx.Response(500, request=req, headers={}, json={})
    assert hc.HttpEvalClient._extract_request_id(r_none, {}) is None


@pytest.mark.parametrize(
    "exc,stage",
    [
        (httpx.ReadTimeout("r"), "read_timeout"),
        (httpx.WriteTimeout("w"), "write_timeout"),
        (httpx.PoolTimeout("p"), "pool_timeout"),
        (httpx.TimeoutException("t"), "timeout"),
    ],
)
def test_timeout_stage_variants(exc: BaseException, stage: str):
    assert hc.HttpEvalClient._timeout_stage(exc) == stage


@pytest.mark.asyncio
async def test_modelz_non_dict_json_and_generic_exception(monkeypatch: pytest.MonkeyPatch):
    responses = [httpx.Response(200, json=[1, 2, 3])]
    _patch(monkeypatch, responses)
    monkeypatch.setattr(hc.time, "time", _TimeSeq([1.0, 1.1]))

    c = hc.HttpEvalClient(base_url="http://svc", api_key="KEY")
    out = await c.modelz()
    assert isinstance(out, hc.ModelzOk)
    assert out.status == "ready"
    assert out.default_model_id is None

    # generic exception branch in modelz()
    async def _boom(url: str, headers: Dict[str, str]):
        raise RuntimeError("boom")

    client = c._get_client()
    monkeypatch.setattr(client, "get", _boom)
    monkeypatch.setattr(hc.time, "time", _TimeSeq([2.0, 2.2]))

    out2 = await c.modelz()
    assert isinstance(out2, hc.ModelzErr)
    assert out2.error_code == "transport_error"


@pytest.mark.asyncio
async def test_effective_helpers_return_none_on_error(monkeypatch: pytest.MonkeyPatch):
    c = hc.HttpEvalClient(base_url="http://svc", api_key="KEY")

    async def _err():
        return hc.ModelzErr(
            status_code=500, error_code="e", message="m", extra=None, latency_ms=1.0
        )

    monkeypatch.setattr(c, "modelz", _err)
    assert await c.effective_server_model_id() is None
    assert await c.effective_deployment_key() is None


@pytest.mark.asyncio
async def test_generate_error_branches(monkeypatch: pytest.MonkeyPatch):
    _patch(monkeypatch, [])
    c = hc.HttpEvalClient(base_url="http://svc", api_key="KEY")
    client = c._get_client()

    req = httpx.Request("POST", "http://svc/v1/generate")

    async def _timeout(url: str, json: Dict[str, Any], headers: Dict[str, str]):
        raise httpx.ReadTimeout("rt", request=req)

    monkeypatch.setattr(client, "post", _timeout)
    monkeypatch.setattr(hc.time, "time", _TimeSeq([0.0, 0.1]))
    out = await c.generate(prompt="p")
    assert isinstance(out, hc.GenerateErr)
    assert out.error_code == "timeout"

    async def _request_err(url: str, json: Dict[str, Any], headers: Dict[str, str]):
        raise httpx.RequestError("re", request=req)

    monkeypatch.setattr(client, "post", _request_err)
    monkeypatch.setattr(hc.time, "time", _TimeSeq([1.0, 1.2]))
    out2 = await c.generate(prompt="p")
    assert isinstance(out2, hc.GenerateErr)
    assert out2.error_code == "transport_error"

    async def _other(url: str, json: Dict[str, Any], headers: Dict[str, str]):
        raise ValueError("x")

    monkeypatch.setattr(client, "post", _other)
    monkeypatch.setattr(hc.time, "time", _TimeSeq([2.0, 2.2]))
    out3 = await c.generate(prompt="p")
    assert isinstance(out3, hc.GenerateErr)
    assert out3.error_code == "transport_error"


@pytest.mark.asyncio
async def test_extract_success_and_error_branches(monkeypatch: pytest.MonkeyPatch):
    responses = [httpx.Response(200, json=["not-dict"])]
    _patch(monkeypatch, responses)
    c = hc.HttpEvalClient(base_url="http://svc", api_key="KEY")
    monkeypatch.setattr(hc.time, "time", _TimeSeq([0.0, 0.1]))

    out = await c.extract(schema_id="s1", text="doc")
    assert isinstance(out, hc.ExtractOk)
    assert out.data == {}

    client = c._get_client()
    req = httpx.Request("POST", "http://svc/v1/extract")

    async def _timeout(url: str, json: Dict[str, Any], headers: Dict[str, str]):
        raise httpx.WriteTimeout("wt", request=req)

    monkeypatch.setattr(client, "post", _timeout)
    monkeypatch.setattr(hc.time, "time", _TimeSeq([1.0, 1.1]))
    out2 = await c.extract(schema_id="s1", text="doc")
    assert isinstance(out2, hc.ExtractErr)
    assert out2.error_code == "timeout"

    async def _request_err(url: str, json: Dict[str, Any], headers: Dict[str, str]):
        raise httpx.RequestError("re", request=req)

    monkeypatch.setattr(client, "post", _request_err)
    monkeypatch.setattr(hc.time, "time", _TimeSeq([2.0, 2.2]))
    out3 = await c.extract(schema_id="s1", text="doc")
    assert isinstance(out3, hc.ExtractErr)
    assert out3.error_code == "transport_error"

    async def _other(url: str, json: Dict[str, Any], headers: Dict[str, str]):
        raise RuntimeError("boom")

    monkeypatch.setattr(client, "post", _other)
    monkeypatch.setattr(hc.time, "time", _TimeSeq([3.0, 3.3]))
    out4 = await c.extract(schema_id="s1", text="doc")
    assert isinstance(out4, hc.ExtractErr)
    assert out4.error_code == "transport_error"
