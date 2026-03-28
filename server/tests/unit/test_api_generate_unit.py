from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

from llm_server.api import generate as generate_api


def _request():
    return SimpleNamespace(
        state=SimpleNamespace(request_id="rid-1", trace_id="trace-1", trace_job_id=None),
        client=SimpleNamespace(host="127.0.0.1"),
    )


def _body():
    return generate_api.GenerateRequest(
        prompt="hello",
        model="model-a",
        cache=True,
        max_new_tokens=256,
        temperature=0.0,
    )


@pytest.mark.anyio
async def test_generate_route_binds_request_span_and_opens_cache_lookup_span(
    monkeypatch: pytest.MonkeyPatch,
):
    span_bind_calls: list[dict[str, object]] = []
    child_span_calls: list[dict[str, object]] = []

    @contextmanager
    def fake_start_child_span(name: str, *, request=None, attributes=None):
        child_span_calls.append(
            {"name": name, "request": request, "attributes": deepcopy(attributes)}
        )
        yield SimpleNamespace()

    def fake_bind_request_span(request, **kwargs):
        span_bind_calls.append({"request": request, "kwargs": deepcopy(kwargs)})

    class _SessionContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(generate_api, "_reject_if_mode_off", lambda request: None, raising=True)
    monkeypatch.setattr(generate_api, "bind_request_span", fake_bind_request_span, raising=True)
    monkeypatch.setattr(generate_api, "start_child_span", fake_start_child_span, raising=True)
    monkeypatch.setattr(generate_api, "get_llm", lambda request: object(), raising=True)
    monkeypatch.setattr(
        generate_api,
        "resolve_model",
        lambda llm, model, capability, request: ("resolved-model", object()),
        raising=True,
    )
    monkeypatch.setattr(generate_api, "require_capability", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(generate_api, "require_assessed_gate", lambda *a, **k: None, raising=True)

    async def fake_require_inprocess_loaded_if_needed(**kwargs):
        return None

    monkeypatch.setattr(
        generate_api,
        "require_inprocess_loaded_if_needed",
        fake_require_inprocess_loaded_if_needed,
        raising=True,
    )
    monkeypatch.setattr(generate_api, "set_request_meta", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(
        generate_api, "apply_generate_cap", lambda *a, **k: (128, 256, False), raising=True
    )
    monkeypatch.setattr(generate_api, "sha32", lambda value: "sha", raising=True)
    monkeypatch.setattr(generate_api, "fingerprint_pydantic", lambda *a, **k: "fp", raising=True)
    monkeypatch.setattr(
        generate_api, "make_cache_redis_key", lambda *a, **k: "cache-key", raising=True
    )
    monkeypatch.setattr(generate_api, "get_redis_from_request", lambda request: None, raising=True)
    monkeypatch.setattr(
        generate_api.db_session, "get_sessionmaker", lambda: lambda: _SessionContext()
    )

    async def fake_get_cached_output(*args, **kwargs):
        return "cached-output", True, "db"

    monkeypatch.setattr(generate_api, "get_cached_output", fake_get_cached_output, raising=True)
    monkeypatch.setattr(generate_api, "request_latency_ms", lambda request: 12, raising=True)
    monkeypatch.setattr(generate_api, "count_tokens_split", lambda **kwargs: (10, 20), raising=True)
    monkeypatch.setattr(generate_api, "record_token_metrics", lambda *a, **k: None, raising=True)

    async def fake_write_inference_log(*args, **kwargs):
        return None

    monkeypatch.setattr(generate_api, "write_inference_log", fake_write_inference_log, raising=True)

    response = await generate_api.generate(
        _request(),
        _body(),
        api_key=SimpleNamespace(key="proof-user-key"),
    )

    assert response == {
        "model": "resolved-model",
        "output": "cached-output",
        "cached": True,
        "requested_max_new_tokens": 256,
        "effective_max_new_tokens": 128,
        "policy_generate_max_new_tokens_cap": 256,
        "clamped": False,
    }
    assert len(span_bind_calls) == 2
    assert span_bind_calls[0]["kwargs"] == {
        "name": "backend.generate",
        "route": "/v1/generate",
        "attributes": {"llm.requested_model_id": "model-a"},
    }
    assert span_bind_calls[1]["kwargs"] == {
        "attributes": {"llm.resolved_model_id": "resolved-model"},
    }
    assert child_span_calls == [
        {
            "name": "generate.cache_lookup",
            "request": span_bind_calls[0]["request"],
            "attributes": {"llm.resolved_model_id": "resolved-model"},
        }
    ]
