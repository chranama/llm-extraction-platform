# server/src/llm_server/services/generate/token_counting.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from transformers import AutoTokenizer


def _token_counting_enabled() -> bool:
    return os.getenv("TOKEN_COUNTING", "1").strip().lower() not in {"0", "false", "no", "off"}


def detect_backend_name(model: Any) -> str | None:
    """
    Best-effort backend identifier.

    We use this to avoid trying HF tokenizers for llama.cpp-based backends.
    """
    try:
        v = getattr(model, "backend_name", None)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    except Exception:
        pass
    return None


def _llamacpp_tokenize(model: Any, texts: list[str]) -> list[list[int]] | None:
    """
    Best-effort llama.cpp tokenize via the model's client.

    Expected contract (based on your current wrapper):
      - model._client.tokenize(content=[...]) -> {"tokens": [...]}
      - tokens can be:
          - list[int] for single input
          - list[list[int]] for batch inputs
    """
    try:
        client = getattr(model, "_client", None)
        tok_fn = getattr(client, "tokenize", None)
        if not callable(tok_fn):
            return None

        data = tok_fn(content=texts)
        toks = data.get("tokens") if isinstance(data, dict) else None
        if not isinstance(toks, list):
            return None

        # single -> list[int]
        if toks and all(isinstance(x, int) for x in toks):
            return [toks] if len(texts) == 1 else None

        # batch -> list[list[int]]
        if toks and all(isinstance(x, list) for x in toks):
            out: list[list[int]] = []
            for row in toks:
                if isinstance(row, list) and all(isinstance(x, int) for x in row):
                    out.append(row)
                else:
                    return None
            return out if len(out) == len(texts) else None

        if not toks:
            return [[] for _ in texts]

        return None
    except Exception:
        return None


@lru_cache(maxsize=16)
def _get_tokenizer(tokenizer_id: str):
    return AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)


def _token_count_hf(tokenizer_id: str, prompt: str, completion: str | None) -> tuple[int | None, int | None]:
    try:
        tok = _get_tokenizer(tokenizer_id)
        prompt_ids = tok(prompt, add_special_tokens=False).input_ids
        prompt_tokens = len(prompt_ids)

        if completion:
            completion_ids = tok(completion, add_special_tokens=False).input_ids
            completion_tokens = len(completion_ids)
        else:
            completion_tokens = 0

        return prompt_tokens, completion_tokens
    except Exception:
        return None, None


def count_tokens_split(
    *,
    model: Any,
    model_id: str,
    prompt: str,
    completion: str | None,
    usage_from_backend: Any | None,
) -> tuple[int | None, int | None]:
    """
    Count tokens split (prompt vs completion).

    Order:
      1) If TOKEN_COUNTING disabled -> (None, None)
      2) If backend usage dict provided -> use it
      3) If backend is llamacpp -> try llamacpp tokenize
      4) Else -> HF tokenizer fallback (best-effort)

    IMPORTANT:
      - For llama-server deployments, model_id may NOT be an HF tokenizer id.
        HF fallback is best-effort and may return (None, None).
      - If you later add tokenizer_id to models.yaml, prefer passing that into this module.
    """
    if not _token_counting_enabled():
        return None, None

    # 1) Prefer backend usage if present
    if isinstance(usage_from_backend, dict):
        pt = usage_from_backend.get("prompt_tokens")
        ct = usage_from_backend.get("completion_tokens")
        pt_i = int(pt) if isinstance(pt, int) else None
        ct_i = int(ct) if isinstance(ct, int) else None
        if pt_i is not None or ct_i is not None:
            return pt_i, ct_i

    # 2) llama.cpp tokenize (best-effort)
    if detect_backend_name(model) == "llamacpp":
        texts: list[str] = [prompt]
        if completion is not None:
            texts.append(completion)

        toks = _llamacpp_tokenize(model, texts)
        if toks is not None:
            pt = len(toks[0]) if len(toks) >= 1 else None
            ct = len(toks[1]) if (completion is not None and len(toks) >= 2) else (0 if completion is not None else None)
            return pt, ct

        return None, None

    # 3) HF fallback (best-effort)
    return _token_count_hf(model_id, prompt, completion)