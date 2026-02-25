from __future__ import annotations

import json
from typing import Any

import pytest

from llm_server.core.errors import AppError
from llm_server.services.backends import backend_api as mod


class _Resp:
    def __init__(self, *, status_code: int = 200, data: Any = None, text: str = ""):
        self.status_code = status_code
        self._data = data
        self.text = text

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


class _Client:
    def __init__(self, timeout=None):
        self.timeout = timeout
        self.closed = False
        self.last_get = None
        self.last_request = None
        self.raise_on_get = None
        self.raise_on_request = None
        self.get_resp = _Resp(data={"ok": True})
        self.req_resp = _Resp(data={"ok": True})

    def close(self):
        self.closed = True

    def get(self, url, headers=None, timeout=None):
        if self.raise_on_get is not None:
            raise self.raise_on_get
        self.last_get = (url, headers, timeout)
        return self.get_resp

    def request(self, method, url, headers=None, content=None, timeout=None):
        if self.raise_on_request is not None:
            raise self.raise_on_request
        self.last_request = (method, url, headers, content, timeout)
        return self.req_resp


def _make(monkeypatch, *, api_key: str | None = " key "):
    fake = _Client()
    monkeypatch.setattr(mod.httpx, "Client", lambda timeout: fake, raising=True)
    cfg = mod.OpenAICompatClientConfig(base_url="http://backend", api_key=api_key)
    c = mod.OpenAICompatClient(cfg)
    return c, fake


def test_init_requires_base_url():
    with pytest.raises(AppError) as e:
        mod.OpenAICompatClient(mod.OpenAICompatClientConfig(base_url=""))
    assert e.value.code == "backend_config_invalid"


def test_headers_include_content_type_and_auth(monkeypatch):
    c, _ = _make(monkeypatch, api_key="  token  ")
    h = c._headers()
    assert h["Content-Type"] == "application/json"
    assert h["Authorization"] == "Bearer token"


def test_request_json_get_success_non_dict_json_wrapped(monkeypatch):
    c, fake = _make(monkeypatch)
    fake.get_resp = _Resp(data=[1, 2, 3])
    out = c._request_json("GET", "v1/models")
    assert out == {"raw": [1, 2, 3]}
    assert fake.last_get is not None


def test_request_json_transport_error_maps_to_backend_unreachable(monkeypatch):
    c, fake = _make(monkeypatch)
    fake.raise_on_request = RuntimeError("network down")
    with pytest.raises(AppError) as e:
        c._request_json("POST", "/v1/completions", payload={"prompt": "x"})
    assert e.value.code == "backend_unreachable"


def test_request_json_http_error_uses_json_detail_or_text(monkeypatch):
    c, fake = _make(monkeypatch)
    fake.req_resp = _Resp(status_code=503, data={"err": "oops"}, text="backend bad")
    with pytest.raises(AppError) as e1:
        c._request_json("POST", "/v1/completions", payload={"prompt": "x"})
    assert e1.value.code == "backend_error"
    assert e1.value.extra["status_code"] == 503
    assert e1.value.extra["detail"] == {"err": "oops"}

    fake.req_resp = _Resp(status_code=500, data=ValueError("bad json"), text="plain text")
    with pytest.raises(AppError) as e2:
        c._request_json("POST", "/v1/completions", payload={"prompt": "x"})
    assert e2.value.code == "backend_error"
    assert e2.value.extra["detail"] == "plain text"


def test_request_json_invalid_json_maps_to_backend_bad_response(monkeypatch):
    c, fake = _make(monkeypatch)
    fake.get_resp = _Resp(status_code=200, data=ValueError("not json"), text="not-json")
    with pytest.raises(AppError) as e:
        c._request_json("GET", "/health")
    assert e.value.code == "backend_bad_response"


def test_completions_and_chat_payload_shape(monkeypatch):
    c, fake = _make(monkeypatch)
    fake.req_resp = _Resp(data={"choices": [{"text": "ok"}]})

    c.completions(prompt="p", max_tokens=7, temperature=0.2, top_p=0.8, top_k=10, stop=["."], model="m", extra={"x": 1})
    method, _, _, content, _ = fake.last_request
    payload = json.loads(content)
    assert method == "POST"
    assert payload["prompt"] == "p"
    assert payload["model"] == "m"
    assert payload["max_tokens"] == 7
    assert payload["x"] == 1

    c.chat_completions(messages=[{"role": "user", "content": "hi"}], model="m2")
    _, _, _, content2, _ = fake.last_request
    payload2 = json.loads(content2)
    assert payload2["messages"][0]["content"] == "hi"
    assert payload2["model"] == "m2"


def test_close_swallows_exceptions(monkeypatch):
    c, fake = _make(monkeypatch)

    def _boom():
        raise RuntimeError("close failed")

    fake.close = _boom
    c.close()
