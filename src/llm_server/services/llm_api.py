# src/llm_server/services/llm_api.py
from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

import httpx

from llm_server.core.config import settings


class HttpLLMClient:
    """
    Simple HTTP client for calling an external LLM service.

    This is used when `settings.model_mode == "remote"` in combination with
    `settings.llm_service_url`.

    It is intentionally *decoupled* from the local `ModelManager` in
    `llm_server.services.llm` to avoid circular imports.  It presents a
    compatible interface:

        - .model_id attribute
        - .generate(prompt=..., max_new_tokens=..., temperature=..., ...)
        - optional .stream(...) for future use
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        # Base URL of the remote LLM service (NOT the FastAPI gateway)
        self.base_url: str = base_url or settings.llm_service_url
        # Logical model identifier, used for logging / metrics
        self.model_id: str = model_id or settings.model_id
        # Per-request timeout in seconds
        self.timeout: int = timeout or settings.http_client_timeout

    # ------------------------------------------------------------
    # Non-streaming generate
    # ------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        """
        Call the remote LLM service's /v1/generate endpoint.

        The remote service is expected to implement a JSON API similar to
        this gateway's /v1/generate:

            POST {base_url}/v1/generate
            {
                "prompt": "...",
                "max_new_tokens": ...,
                "temperature": ...,
                "top_p": ...,
                "top_k": ...,
                "stop": ["..."]
            }

            -> { "model": "...", "output": "completion text", ... }
        """
        url = f"{self.base_url.rstrip('/')}/v1/generate"

        payload: Dict[str, Any] = {
            "prompt": prompt,
        }
        if max_new_tokens is not None:
            payload["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        if stop:
            payload["stop"] = stop

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Expect the remote service to return {"output": "..."}.
        output = data.get("output", "")
        if not isinstance(output, str):
            output = str(output)
        return output

    # ------------------------------------------------------------
    # Streaming generate (optional / stub)
    # ------------------------------------------------------------
    def stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> Iterator[str]:
        """
        Optional streaming interface.

        If your remote service exposes a streaming endpoint (e.g. SSE at
        /v1/stream), you can implement chunked iteration here. For now,
        this provides a simple non-streaming fallback by calling
        `generate()` and yielding the whole answer once.
        """
        # Simple fallback: one-shot generate
        text = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )
        if text:
            yield text