from __future__ import annotations

import pytest

from llm_server.core.errors import AppError
from llm_server.services.backends import llamacpp_backend as mod


class _FakeClient:
    def __init__(self):
        self.health_value = {"status": "ok"}
        self.models_value = {"data": [{"id": "m-live", "owned_by": "llamacpp", "meta": {"ctx": 4096}}]}
        self.raw_map = {
            "/version": RuntimeError("no version"),
            "/v1/version": {"version": "1.0.0"},
        }
        self.chat_value = {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        self.completions_value = {"choices": [{"text": "pong"}]}
        self.tokenize_value = {"tokens": [1, 2, 3]}
        self.last_chat = None
        self.last_completions = None

    def health(self):
        return self.health_value

    def models(self):
        return self.models_value

    def raw_get(self, path: str):
        v = self.raw_map.get(path, RuntimeError("missing"))
        if isinstance(v, Exception):
            raise v
        return v

    def chat_completions(self, **kwargs):
        self.last_chat = kwargs
        return self.chat_value

    def completions(self, **kwargs):
        self.last_completions = kwargs
        return self.completions_value

    def tokenize(self, *, content):
        return self.tokenize_value


def _backend(monkeypatch):
    fake = _FakeClient()

    def _mk_client(cfg):
        return fake

    monkeypatch.setattr(mod, "OpenAICompatClient", _mk_client, raising=True)
    cfg = mod.LlamaCppBackendConfig(server_url="http://llama:8080", model_name="served-model")
    b = mod.LlamaCppBackend(model_id="m-local", cfg=cfg)
    return b, fake


def test_model_info_collects_runtime_details(monkeypatch):
    b, _ = _backend(monkeypatch)
    info = b.model_info()
    assert info["ok"] is True
    assert info["backend"] == "llamacpp"
    assert info["runtime"]["health"]["status"] == "ok"
    assert info["runtime"]["active_model_id"] == "m-live"
    assert info["runtime"]["server_version"]["version"] == "1.0.0"


def test_generate_rich_validates_prompt(monkeypatch):
    b, _ = _backend(monkeypatch)
    with pytest.raises(AppError) as e:
        b.generate_rich(prompt="")
    assert e.value.code == "invalid_request"


def test_generate_rich_builds_payload_and_extracts_usage(monkeypatch):
    b, fake = _backend(monkeypatch)
    out = b.generate_rich(prompt="hello", max_new_tokens=8, extra_k="v")
    assert out.text == "hello"
    assert out.usage.total_tokens == 5
    assert fake.last_chat["model"] == "served-model"
    assert fake.last_chat["max_tokens"] == 8
    assert fake.last_chat["extra"]["extra_k"] == "v"


def test_count_prompt_tokens_handles_shapes(monkeypatch):
    b, fake = _backend(monkeypatch)
    fake.tokenize_value = {"tokens": [11, 22]}
    assert b.count_prompt_tokens("abc") == 2

    fake.tokenize_value = {"tokens": [[1, 2, 3]]}
    assert b.count_prompt_tokens("abc") == 3

    fake.tokenize_value = {"tokens": ["bad"]}
    assert b.count_prompt_tokens("abc") is None
    assert b.count_prompt_tokens("") == 0


def test_is_ready_and_can_generate_paths(monkeypatch):
    b, fake = _backend(monkeypatch)
    ok, details = b.is_ready()
    assert ok is True
    assert details["health"]["status"] == "ok"

    ok2, details2 = b.can_generate()
    assert ok2 is True
    assert details2["sample"] == "pong"

    fake.completions_value = {"choices": [{"not_text": "x"}]}
    ok3, _ = b.can_generate()
    assert ok3 is False

    def _boom():
        raise RuntimeError("backend down")

    fake.health = _boom
    ok4, details4 = b.is_ready()
    assert ok4 is False
    assert "backend down" in details4["error"]
