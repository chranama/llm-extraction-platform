from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_server.services.api_deps.generate import generate_runner as gr


def test_extract_usage_dict_paths():
    assert gr._extract_usage_dict(None) is None

    u = SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7)
    out = gr._extract_usage_dict(u)
    assert out == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}

    u2 = SimpleNamespace(prompt_tokens="x", completion_tokens=None, total_tokens=1)
    out2 = gr._extract_usage_dict(u2)
    assert out2 == {"prompt_tokens": None, "completion_tokens": None, "total_tokens": 1}


@pytest.mark.anyio
async def test_run_generate_rich_prefers_generate_rich():
    class _Model:
        @staticmethod
        def generate_rich(**kwargs):
            return SimpleNamespace(
                text="hello",
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            )

    text, usage = await gr.run_generate_rich_offloop(_Model(), prompt="p")
    assert text == "hello"
    assert usage["total_tokens"] == 7


@pytest.mark.anyio
async def test_run_generate_rich_fallback_to_generate():
    class _Model:
        @staticmethod
        def generate(**kwargs):
            return 123

    text, usage = await gr.run_generate_rich_offloop(_Model(), prompt="p")
    assert text == "123"
    assert usage is None


@pytest.mark.anyio
async def test_run_generate_rich_no_methods():
    text, usage = await gr.run_generate_rich_offloop(object(), prompt="p")
    assert text == ""
    assert usage is None
