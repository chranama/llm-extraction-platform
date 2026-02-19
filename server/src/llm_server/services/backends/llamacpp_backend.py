# server/src/llm_server/services/backends/llamacpp_backend.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from llm_server.core.errors import AppError
from llm_server.services.backends.base import GenerateResult, GenerateTimings, GenerateUsage, LLMBackend
from llm_server.services.backends.backend_api import OpenAICompatClient, OpenAICompatClientConfig


@dataclass(frozen=True)
class LlamaCppBackendConfig:
    """
    OpenAI-compatible llama-server config.
    """
    server_url: str  # e.g. http://127.0.0.1:8081
    api_key: Optional[str] = None

    # HTTP client timeouts
    timeout_seconds: float = 60.0
    connect_timeout_seconds: float = 5.0

    # Optional "model" string for OpenAI compatibility.
    model_name: Optional[str] = None

    # Optional defaults (applied when request fields are None)
    default_temperature: float = 0.7
    default_top_p: float = 0.95


class LlamaCppBackend(LLMBackend):
    backend_name: str = "llamacpp"

    def __init__(self, *, model_id: str, cfg: LlamaCppBackendConfig) -> None:
        self.model_id = model_id
        self._cfg = cfg
        self._client = OpenAICompatClient(
            OpenAICompatClientConfig(
                base_url=cfg.server_url,
                api_key=cfg.api_key,
                timeout_seconds=float(cfg.timeout_seconds),
                connect_timeout_seconds=float(cfg.connect_timeout_seconds),
            )
        )

    def ensure_loaded(self) -> None:
        # External service; nothing to load in-process.
        return None

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        r = self.generate_rich(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            **kwargs,
        )
        return r.text

    def generate_rich(
        self,
        *,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> GenerateResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise AppError(code="invalid_request", message="prompt must be a non-empty string", status_code=400)

        # Apply backend defaults when missing
        temp = float(temperature) if isinstance(temperature, (int, float)) else float(self._cfg.default_temperature)
        tp = float(top_p) if isinstance(top_p, (int, float)) else float(self._cfg.default_top_p)

        max_tokens: int | None = None
        if isinstance(max_new_tokens, int) and max_new_tokens > 0:
            max_tokens = int(max_new_tokens)

        # Pass-through extra kwargs (drop None)
        extra: Dict[str, Any] = {k: v for k, v in kwargs.items() if v is not None}

        # Always send a model string (important for some servers)
        model = self._cfg.model_name or self.model_id

        messages = [{"role": "user", "content": prompt}]

        t0 = time.perf_counter()
        data = self._client.chat_completions(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=tp,
            top_k=top_k,
            stop=stop,
            model=model,
            extra=extra or None,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Extract text (OpenAI chat format)
        text = ""
        try:
            choices = data.get("choices") or []
            if isinstance(choices, list) and choices:
                c0 = choices[0] or {}
                if isinstance(c0, dict):
                    msg = c0.get("message") or {}
                    if isinstance(msg, dict):
                        text = str(msg.get("content") or "")
        except Exception:
            text = ""

        # Extract usage (if present)
        usage = data.get("usage") if isinstance(data, dict) else None
        u = GenerateUsage()
        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            tt = usage.get("total_tokens")
            u = GenerateUsage(
                prompt_tokens=int(pt) if isinstance(pt, int) else None,
                completion_tokens=int(ct) if isinstance(ct, int) else None,
                total_tokens=int(tt) if isinstance(tt, int) else None,
            )

        return GenerateResult(
            text=text,
            usage=u,
            timings=GenerateTimings(total_ms=dt_ms, backend_ms=dt_ms),
            raw=data if isinstance(data, dict) else None,
        )

    # ------------------------------------------------------------------
    # Optional helpers for your control-plane / telemetry story
    # ------------------------------------------------------------------

    def count_prompt_tokens(self, prompt: str) -> int | None:
        """
        Best-effort token counting using llama.cpp server /tokenize endpoint.

        Your server supports:
          POST /tokenize {"content":["some random input","another string"]}
          -> {"tokens":[...]}  OR {"tokens":[[...],[...]]}

        Returns:
          int token count, or None if unavailable.
        """
        if not isinstance(prompt, str) or not prompt:
            return 0

        try:
            data = self._client.tokenize(content=[prompt])
        except Exception:
            return None

        toks = data.get("tokens") if isinstance(data, dict) else None

        # Case A: {"tokens":[14689,4194,...]}
        if isinstance(toks, list) and toks and all(isinstance(x, int) for x in toks):
            return len(toks)

        # Case B: {"tokens":[[...],[...]]}
        if isinstance(toks, list) and toks and all(isinstance(x, list) for x in toks):
            first = toks[0]
            if isinstance(first, list) and all(isinstance(x, int) for x in first):
                return len(first)

        return None

    def is_ready(self) -> tuple[bool, dict[str, Any]]:
        """
        Best-effort readiness for llama-server.

        Your curl shows:
          GET /health -> {"status":"ok"}

        Returns:
          (ok, details)
        """
        try:
            data = self._client.health()
            ok = bool(isinstance(data, dict) and data.get("status") == "ok")
            return ok, {"health": data}
        except Exception as e:
            return False, {"error": repr(e)}
        
    def can_generate(self) -> tuple[bool, dict[str, Any]]:
        """
        Strong readiness: prove the backend can actually generate output.

        Uses POST /v1/completions with a tiny payload.
        """
        try:
            t0 = time.perf_counter()
            data = self._client.completions(
                prompt="ping",
                max_tokens=1,
                temperature=0.0,
                model=self._cfg.model_name or self.model_id,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0

            # Minimal sanity: has choices[0].text
            choices = data.get("choices") if isinstance(data, dict) else None
            ok = bool(
                isinstance(choices, list)
                and len(choices) > 0
                and isinstance(choices[0], dict)
                and isinstance(choices[0].get("text"), str)
            )

            return ok, {"latency_ms": dt_ms, "sample": (choices[0].get("text") if ok else None)}
        except Exception as e:
            return False, {"error": repr(e)}