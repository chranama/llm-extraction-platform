from __future__ import annotations

from llm_server.services.llm_runtime import llm_build as mod


class _FakeOpenAICompatClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_chat = None
        self.last_completion = None
        self.health_value = {"status": "ok"}
        self.models_value = {"data": [{"id": "remote-model"}]}
        self.raw_get_value = {"version": "fake"}
        self.chat_value = {
            "choices": [{"message": {"content": "chat out"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        self.completion_value = {
            "choices": [{"text": "completion out"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        }

    def health(self):
        return self.health_value

    def models(self):
        return self.models_value

    def raw_get(self, path):
        return self.raw_get_value

    def chat_completions(self, **kwargs):
        self.last_chat = kwargs
        return self.chat_value

    def completions(self, **kwargs):
        self.last_completion = kwargs
        return self.completion_value


def _backend(monkeypatch, *, request_mode="chat", provider="vllm"):
    holder = {}

    def _factory(cfg):
        client = _FakeOpenAICompatClient(cfg)
        holder["client"] = client
        return client

    monkeypatch.setattr(mod, "OpenAICompatClient", _factory, raising=True)
    backend = mod.RemoteBackend(
        model_id="local-id",
        base_url="http://remote:8000",
        remote_model_id="remote-id",
        request_mode=request_mode,
        provider=provider,
    )
    return backend, holder["client"]


def test_remote_backend_chat_generate_rich(monkeypatch):
    backend, client = _backend(monkeypatch, request_mode="chat", provider="vllm")

    result = backend.generate_rich(prompt="hello", max_new_tokens=8, temperature=0.0)

    assert result.text == "chat out"
    assert result.usage.total_tokens == 5
    assert client.last_chat["model"] == "remote-id"
    assert client.last_chat["messages"] == [{"role": "user", "content": "hello"}]


def test_remote_backend_completion_generate_rich(monkeypatch):
    backend, client = _backend(monkeypatch, request_mode="completion", provider="openai_compat")

    result = backend.generate_rich(prompt="hello", max_new_tokens=8, temperature=0.0)

    assert result.text == "completion out"
    assert result.usage.prompt_tokens == 4
    assert client.last_completion["model"] == "remote-id"
    assert client.last_completion["prompt"] == "hello"


def test_remote_backend_probe_and_model_info(monkeypatch):
    backend, _client = _backend(monkeypatch, request_mode="chat", provider="vllm")

    ok, details = backend.is_ready()
    assert ok is True
    assert details["provider"] == "vllm"

    ok2, details2 = backend.can_generate()
    assert ok2 is True
    assert details2["request_mode"] == "chat"

    info = backend.model_info()
    assert info["provider"] == "vllm"
    assert info["runtime"]["models_raw"]["data"][0]["id"] == "remote-model"
