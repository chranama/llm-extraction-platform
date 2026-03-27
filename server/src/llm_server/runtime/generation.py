from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import anyio
from fastapi import Request
from transformers import AutoTokenizer

from llm_server.io.policy_decisions import get_policy_snapshot
from llm_server.services.api_deps.core.policy_snapshot import snapshot_generate_cap


def normalize_positive_int(x: Any) -> int | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        i = int(x)
    except Exception:
        return None
    return i if i > 0 else None


def apply_generate_cap(
    request: Request,
    *,
    model_id: str,
    requested_max_new_tokens: int | None,
) -> tuple[int | None, int | None, bool]:
    del model_id
    try:
        snap = get_policy_snapshot(request)
    except Exception:
        snap = None

    cap_raw = snapshot_generate_cap(snap) if snap is not None else None
    cap_i = normalize_positive_int(cap_raw)
    req_i = normalize_positive_int(requested_max_new_tokens)

    if cap_i is None:
        return requested_max_new_tokens, None, False

    if req_i is None:
        return cap_i, cap_i, False

    eff = min(req_i, cap_i)
    clamped = req_i > cap_i
    return eff, cap_i, clamped


def _extract_usage_dict(usage_obj: Any) -> dict[str, Any] | None:
    if usage_obj is None:
        return None
    try:
        pt = getattr(usage_obj, "prompt_tokens", None)
        ct = getattr(usage_obj, "completion_tokens", None)
        tt = getattr(usage_obj, "total_tokens", None)
        return {
            "prompt_tokens": int(pt) if isinstance(pt, int) else None,
            "completion_tokens": int(ct) if isinstance(ct, int) else None,
            "total_tokens": int(tt) if isinstance(tt, int) else None,
        }
    except Exception:
        return None


async def run_generate_rich_offloop(model: Any, **kwargs: Any) -> tuple[str, dict[str, Any] | None]:
    def _run() -> tuple[str, dict[str, Any] | None]:
        gen_rich = getattr(model, "generate_rich", None)
        if callable(gen_rich):
            r = gen_rich(**kwargs)
            text = str(getattr(r, "text", "") or "")
            usage_dict = _extract_usage_dict(getattr(r, "usage", None))
            return text, usage_dict

        gen = getattr(model, "generate", None)
        if not callable(gen):
            return "", None

        out = gen(**kwargs)
        return (out if isinstance(out, str) else str(out)), None

    return await anyio.to_thread.run_sync(_run)


def _token_counting_enabled() -> bool:
    return os.getenv("TOKEN_COUNTING", "1").strip().lower() not in {"0", "false", "no", "off"}


def detect_backend_name(model: Any) -> str | None:
    try:
        v = getattr(model, "backend_name", None)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    except Exception:
        pass
    return None


def _llamacpp_tokenize(model: Any, texts: list[str]) -> list[list[int]] | None:
    try:
        client = getattr(model, "_client", None)
        tok_fn = getattr(client, "tokenize", None)
        if not callable(tok_fn):
            return None

        data = tok_fn(content=texts)
        toks = data.get("tokens") if isinstance(data, dict) else None
        if not isinstance(toks, list):
            return None

        if toks and all(isinstance(x, int) for x in toks):
            return [toks] if len(texts) == 1 else None

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


def _token_count_hf(
    tokenizer_id: str, prompt: str, completion: str | None
) -> tuple[int | None, int | None]:
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
    if not _token_counting_enabled():
        return None, None

    if isinstance(usage_from_backend, dict):
        pt = usage_from_backend.get("prompt_tokens")
        ct = usage_from_backend.get("completion_tokens")
        pt_i = int(pt) if isinstance(pt, int) else None
        ct_i = int(ct) if isinstance(ct, int) else None
        if pt_i is not None or ct_i is not None:
            return pt_i, ct_i

    if detect_backend_name(model) == "llamacpp":
        texts: list[str] = [prompt]
        if completion is not None:
            texts.append(completion)

        toks = _llamacpp_tokenize(model, texts)
        if toks is not None:
            pt = len(toks[0]) if len(toks) >= 1 else None
            ct = (
                len(toks[1])
                if (completion is not None and len(toks) >= 2)
                else (0 if completion is not None else None)
            )
            return pt, ct

        return None, None

    return _token_count_hf(model_id, prompt, completion)
