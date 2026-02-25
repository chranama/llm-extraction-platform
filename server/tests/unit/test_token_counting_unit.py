from __future__ import annotations

import types

from llm_server.services.api_deps.generate import token_counting as tc


class _TokOut:
    def __init__(self, ids):
        self.input_ids = ids


class _Tok:
    def __init__(self):
        self.calls = []

    def __call__(self, text, add_special_tokens=False):
        self.calls.append((text, add_special_tokens))
        return _TokOut([1, 2, 3] if text else [])


class _LlamaClient:
    def __init__(self, tokens):
        self.tokens = tokens

    def tokenize(self, *, content):
        return {"tokens": self.tokens}


class _LlamaModel:
    backend_name = "llamacpp"

    def __init__(self, tokens):
        self._client = _LlamaClient(tokens)


def test_token_counting_enabled_toggle(monkeypatch):
    monkeypatch.setenv("TOKEN_COUNTING", "0")
    assert tc._token_counting_enabled() is False
    monkeypatch.setenv("TOKEN_COUNTING", "on")
    assert tc._token_counting_enabled() is True


def test_detect_backend_name():
    assert tc.detect_backend_name(types.SimpleNamespace(backend_name="  LlamaCPP  ")) == "llamacpp"
    assert tc.detect_backend_name(types.SimpleNamespace()) is None


def test_llamacpp_tokenize_shapes():
    m1 = _LlamaModel([10, 20])
    assert tc._llamacpp_tokenize(m1, ["hello"]) == [[10, 20]]
    assert tc._llamacpp_tokenize(m1, ["a", "b"]) is None

    m2 = _LlamaModel([[1], [2, 3]])
    assert tc._llamacpp_tokenize(m2, ["a", "b"]) == [[1], [2, 3]]

    m3 = _LlamaModel([])
    assert tc._llamacpp_tokenize(m3, ["a", "b"]) == [[], []]

    m4 = _LlamaModel([["bad"], [2]])
    assert tc._llamacpp_tokenize(m4, ["a", "b"]) is None


def test_token_count_hf_success_and_failure(monkeypatch):
    tok = _Tok()
    monkeypatch.setattr(tc, "_get_tokenizer", lambda _: tok, raising=True)
    p, c = tc._token_count_hf("hf-id", "prompt", "done")
    assert (p, c) == (3, 3)

    def _boom(_):
        raise RuntimeError("no tokenizer")

    monkeypatch.setattr(tc, "_get_tokenizer", _boom, raising=True)
    p2, c2 = tc._token_count_hf("hf-id", "prompt", None)
    assert (p2, c2) == (None, None)


def test_count_tokens_split_precedence(monkeypatch):
    monkeypatch.setenv("TOKEN_COUNTING", "1")
    tc._get_tokenizer.cache_clear()

    # usage wins when present
    p1, c1 = tc.count_tokens_split(
        model=object(),
        model_id="hf-id",
        prompt="p",
        completion="c",
        usage_from_backend={"prompt_tokens": 11, "completion_tokens": 7},
    )
    assert (p1, c1) == (11, 7)

    # llamacpp path
    p2, c2 = tc.count_tokens_split(
        model=_LlamaModel([[1, 2], [3]]),
        model_id="ignored",
        prompt="p",
        completion="c",
        usage_from_backend=None,
    )
    assert (p2, c2) == (2, 1)

    # hf fallback when not llamacpp
    monkeypatch.setattr(tc, "_token_count_hf", lambda *_: (5, 2), raising=True)
    p3, c3 = tc.count_tokens_split(
        model=types.SimpleNamespace(backend_name="transformers"),
        model_id="hf-id",
        prompt="p",
        completion="c",
        usage_from_backend=None,
    )
    assert (p3, c3) == (5, 2)


def test_count_tokens_split_disabled_and_llama_failure(monkeypatch):
    monkeypatch.setenv("TOKEN_COUNTING", "off")
    p0, c0 = tc.count_tokens_split(
        model=object(),
        model_id="x",
        prompt="p",
        completion="c",
        usage_from_backend=None,
    )
    assert (p0, c0) == (None, None)

    monkeypatch.setenv("TOKEN_COUNTING", "1")
    monkeypatch.setattr(tc, "_llamacpp_tokenize", lambda *_: None, raising=True)
    p1, c1 = tc.count_tokens_split(
        model=types.SimpleNamespace(backend_name="llamacpp"),
        model_id="x",
        prompt="p",
        completion="c",
        usage_from_backend=None,
    )
    assert (p1, c1) == (None, None)
