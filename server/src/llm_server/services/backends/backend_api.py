# server/src/llm_server/services/backends/backend_api.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import httpx

from llm_server.core.errors import AppError


@dataclass(frozen=True)
class OpenAICompatClientConfig:
    base_url: str  # e.g. "http://127.0.0.1:8080/"
    api_key: Optional[str] = None
    timeout_seconds: float = 60.0
    connect_timeout_seconds: float = 5.0


class OpenAICompatClient:
    """
    Generic OpenAI-compatible HTTP client.
    Supports llama-server and future OpenAI-like backends.
    """

    def __init__(self, cfg: OpenAICompatClientConfig) -> None:
        base = (cfg.base_url or "").strip()
        if not base:
            raise AppError(
                code="backend_config_invalid",
                message="OpenAICompatClient requires base_url",
                status_code=500,
                extra={"base_url": cfg.base_url},
            )
        if not base.endswith("/"):
            base += "/"

        self._cfg = cfg
        self._base = base

        timeout = httpx.Timeout(
            timeout=cfg.timeout_seconds,
            connect=cfg.connect_timeout_seconds,
            read=cfg.timeout_seconds,
            write=cfg.timeout_seconds,
        )

        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {"Content-Type": "application/json"}
        if isinstance(self._cfg.api_key, str) and self._cfg.api_key.strip():
            h["Authorization"] = f"Bearer {self._cfg.api_key.strip()}"
        return h

    def completions(
        self,
        *,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        model: str | None = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        POST /v1/completions
        Returns parsed JSON dict.
        """
        url = urljoin(self._base, "v1/completions")

        payload: Dict[str, Any] = {"prompt": prompt}
        if model:
            payload["model"] = model

        if isinstance(max_tokens, int) and max_tokens > 0:
            payload["max_tokens"] = int(max_tokens)

        if isinstance(temperature, (int, float)):
            payload["temperature"] = float(temperature)

        if isinstance(top_p, (int, float)):
            payload["top_p"] = float(top_p)

        if isinstance(top_k, int) and top_k > 0:
            payload["top_k"] = int(top_k)

        if stop:
            payload["stop"] = stop

        if extra and isinstance(extra, dict):
            payload.update(extra)

        t0 = time.perf_counter()
        try:
            resp = self._client.post(url, headers=self._headers(), content=json.dumps(payload))
        except Exception as e:
            raise AppError(
                code="backend_unreachable",
                message="Backend is unreachable",
                status_code=502,
                extra={"base_url": self._base, "error": repr(e)},
            ) from e

        dt_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code >= 400:
            detail: Any = None
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:1000]
            raise AppError(
                code="backend_error",
                message="Backend request failed",
                status_code=502,
                extra={
                    "base_url": self._base,
                    "endpoint": "/v1/completions",
                    "status_code": resp.status_code,
                    "latency_ms": dt_ms,
                    "detail": detail,
                },
            )

        try:
            return resp.json()
        except Exception as e:
            raise AppError(
                code="backend_bad_response",
                message="Backend returned invalid JSON",
                status_code=502,
                extra={
                    "base_url": self._base,
                    "endpoint": "/v1/completions",
                    "error": repr(e),
                    "text": resp.text[:500],
                },
            ) from e

    def chat_completions(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        model: str | None = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        POST /v1/chat/completions
        Returns parsed JSON dict.
        """
        url = urljoin(self._base, "v1/chat/completions")

        payload: Dict[str, Any] = {"messages": messages}
        if model:
            payload["model"] = model

        if isinstance(max_tokens, int) and max_tokens > 0:
            payload["max_tokens"] = int(max_tokens)

        if isinstance(temperature, (int, float)):
            payload["temperature"] = float(temperature)

        if isinstance(top_p, (int, float)):
            payload["top_p"] = float(top_p)

        if isinstance(top_k, int) and top_k > 0:
            payload["top_k"] = int(top_k)

        if stop:
            payload["stop"] = stop

        if extra and isinstance(extra, dict):
            payload.update(extra)

        t0 = time.perf_counter()
        try:
            resp = self._client.post(url, headers=self._headers(), content=json.dumps(payload))
        except Exception as e:
            raise AppError(
                code="backend_unreachable",
                message="Backend is unreachable",
                status_code=502,
                extra={"base_url": self._base, "error": repr(e)},
            ) from e

        dt_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code >= 400:
            detail: Any = None
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:1000]
            raise AppError(
                code="backend_error",
                message="Backend request failed",
                status_code=502,
                extra={
                    "base_url": self._base,
                    "endpoint": "/v1/chat/completions",
                    "status_code": resp.status_code,
                    "latency_ms": dt_ms,
                    "detail": detail,
                },
            )

        try:
            return resp.json()
        except Exception as e:
            raise AppError(
                code="backend_bad_response",
                message="Backend returned invalid JSON",
                status_code=502,
                extra={
                    "base_url": self._base,
                    "endpoint": "/v1/chat/completions",
                    "error": repr(e),
                    "text": resp.text[:500],
                },
            ) from e
        
    def tokenize(self, *, content: list[str]) -> Dict[str, Any]:
        """
        POST /tokenize  (llama.cpp server)
        Returns parsed JSON dict, typically: {"tokens":[...]} or {"tokens":[[...],[...]]}
        """
        url = urljoin(self._base, "tokenize")

        payload: Dict[str, Any] = {"content": content}

        t0 = time.perf_counter()
        try:
            resp = self._client.post(url, headers=self._headers(), content=json.dumps(payload))
        except Exception as e:
            raise AppError(
                code="backend_unreachable",
                message="Backend is unreachable",
                status_code=502,
                extra={"base_url": self._base, "error": repr(e)},
            ) from e

        dt_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code >= 400:
            detail: Any = None
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:1000]
            raise AppError(
                code="backend_error",
                message="Backend request failed",
                status_code=502,
                extra={
                    "base_url": self._base,
                    "endpoint": "/tokenize",
                    "status_code": resp.status_code,
                    "latency_ms": dt_ms,
                    "detail": detail,
                },
            )

        try:
            return resp.json()
        except Exception as e:
            raise AppError(
                code="backend_bad_response",
                message="Backend returned invalid JSON",
                status_code=502,
                extra={
                    "base_url": self._base,
                    "endpoint": "/tokenize",
                    "error": repr(e),
                    "text": resp.text[:500],
                },
            ) from e
        

    def health(self) -> Dict[str, Any]:
        """
        GET /health

        For llama-server this returns: {"status":"ok"} (per your curl).
        This is intentionally non-OpenAI but is the right readiness primitive.
        """
        url = urljoin(self._base, "health")
        t0 = time.perf_counter()
        try:
            resp = self._client.get(url, headers=self._headers())
        except Exception as e:
            raise AppError(
                code="backend_unreachable",
                message="Backend is unreachable",
                status_code=502,
                extra={"base_url": self._base, "endpoint": "/health", "error": repr(e)},
            ) from e

        dt_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code >= 400:
            detail: Any = None
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:1000]
            raise AppError(
                code="backend_error",
                message="Backend request failed",
                status_code=502,
                extra={
                    "base_url": self._base,
                    "endpoint": "/health",
                    "status_code": resp.status_code,
                    "latency_ms": dt_ms,
                    "detail": detail,
                },
            )

        try:
            data = resp.json()
            return data if isinstance(data, dict) else {"raw": data}
        except Exception as e:
            raise AppError(
                code="backend_bad_response",
                message="Backend returned invalid JSON",
                status_code=502,
                extra={
                    "base_url": self._base,
                    "endpoint": "/health",
                    "error": repr(e),
                    "text": resp.text[:500],
                },
            ) from e