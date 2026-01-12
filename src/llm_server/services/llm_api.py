# src/llm_server/services/llm_api.py
from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

import httpx

from llm_server.core.config import settings
from llm_server.core.errors import AppError


class HttpLLMClient:
    """
    Simple HTTP client for calling an external LLM service.

    Presents a compatible interface:
      - .model_id attribute
      - .ensure_loaded() -> None (no-op; readiness compatibility)
      - .generate(...) -> str
      - .stream(...) -> Iterator[str] (fallback: single chunk)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.base_url: str = base_url or settings.llm_service_url
        self.model_id: str = model_id or settings.model_id
        self.timeout: int = int(timeout or settings.http_client_timeout)

    def ensure_loaded(self) -> None:
        # Remote client has nothing to preload; kept for interface parity.
        return None

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    @staticmethod
    def _preview(text: str | None, limit: int = 500) -> str:
        if not text:
            return ""
        t = text.strip()
        return t[:limit]

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        url = self._url("/v1/generate")

        payload: Dict[str, Any] = {"prompt": prompt}
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

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)

            # Normalize upstream non-2xx into AppError
            if resp.status_code >= 400:
                raise AppError(
                    code="upstream_error",
                    message="Upstream LLM service returned an error",
                    status_code=502,
                    extra={
                        "upstream_status": resp.status_code,
                        "upstream_body_preview": self._preview(resp.text),
                        "upstream_url": url,
                        "model_id": self.model_id,
                    },
                )

            try:
                data = resp.json()
            except Exception as e:
                raise AppError(
                    code="upstream_bad_response",
                    message="Upstream LLM service returned non-JSON response",
                    status_code=502,
                    extra={
                        "upstream_status": resp.status_code,
                        "upstream_body_preview": self._preview(resp.text),
                        "upstream_url": url,
                        "model_id": self.model_id,
                    },
                ) from e

            # Enforce response contract
            if not isinstance(data, dict):
                raise AppError(
                    code="upstream_bad_response",
                    message="Upstream LLM service returned invalid JSON payload",
                    status_code=502,
                    extra={
                        "upstream_status": resp.status_code,
                        "upstream_url": url,
                        "model_id": self.model_id,
                        "payload_type": type(data).__name__,
                    },
                )

            if "output" not in data:
                raise AppError(
                    code="upstream_bad_response",
                    message="Upstream LLM response missing required field 'output'",
                    status_code=502,
                    extra={
                        "upstream_status": resp.status_code,
                        "upstream_url": url,
                        "model_id": self.model_id,
                        "keys": sorted(list(data.keys()))[:50],
                    },
                )

            output = data.get("output")
            if output is None:
                return ""

            if not isinstance(output, str):
                output = str(output)

            return output

        except AppError:
            raise
        except httpx.TimeoutException as e:
            raise AppError(
                code="upstream_timeout",
                message="Upstream LLM service timed out",
                status_code=504,
                extra={"upstream_url": url, "model_id": self.model_id},
            ) from e
        except httpx.ConnectError as e:
            raise AppError(
                code="upstream_unreachable",
                message="Upstream LLM service is unreachable",
                status_code=502,
                extra={"upstream_url": url, "model_id": self.model_id},
            ) from e
        except httpx.RequestError as e:
            raise AppError(
                code="upstream_request_failed",
                message="Upstream LLM request failed",
                status_code=502,
                extra={"upstream_url": url, "model_id": self.model_id, "error": str(e)},
            ) from e

    def stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> Iterator[str]:
        # Fallback: one-shot generate
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