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

    Notes:
      - All methods raise AppError on transport/HTTP/JSON failures.
      - Backends that need "never throw" behavior should wrap calls in try/except
        (e.g., LlamaCppBackend.model_info()).
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
            timeout=float(cfg.timeout_seconds),
            connect=float(cfg.connect_timeout_seconds),
            read=float(cfg.timeout_seconds),
            write=float(cfg.timeout_seconds),
        )

        self._client = httpx.Client(timeout=timeout)

    # -----------------------------
    # lifecycle
    # -----------------------------

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    # -----------------------------
    # internal helpers
    # -----------------------------

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {"Content-Type": "application/json"}
        if isinstance(self._cfg.api_key, str) and self._cfg.api_key.strip():
            h["Authorization"] = f"Bearer {self._cfg.api_key.strip()}"
        return h

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Low-level JSON request helper.

        - method: "GET" | "POST" | ...
        - path: "v1/models" or "/v1/models" (both ok)
        - payload: encoded as JSON if provided (for POST/PUT/etc.)
        - timeout_seconds: optional per-call override (rare)

        Returns:
          Parsed JSON dict (or {"raw": <json>} if backend returns a non-dict JSON value)

        Raises:
          AppError (backend_unreachable / backend_error / backend_bad_response)
        """
        p = (path or "").lstrip("/")
        url = urljoin(self._base, p)

        t0 = time.perf_counter()
        try:
            if method.upper() == "GET":
                resp = self._client.get(url, headers=self._headers(), timeout=timeout_seconds)
            else:
                body = json.dumps(payload or {})
                resp = self._client.request(method.upper(), url, headers=self._headers(), content=body, timeout=timeout_seconds)
        except Exception as e:
            raise AppError(
                code="backend_unreachable",
                message="Backend is unreachable",
                status_code=502,
                extra={"base_url": self._base, "endpoint": f"/{p}", "method": method.upper(), "error": repr(e)},
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
                    "endpoint": f"/{p}",
                    "method": method.upper(),
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
                    "endpoint": f"/{p}",
                    "method": method.upper(),
                    "error": repr(e),
                    "text": resp.text[:500],
                },
            ) from e

    # -----------------------------
    # OpenAI-compatible endpoints
    # -----------------------------

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
        """
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

        return self._request_json("POST", "/v1/completions", payload=payload)

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
        """
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

        return self._request_json("POST", "/v1/chat/completions", payload=payload)

    def models(self) -> Dict[str, Any]:
        """
        GET /v1/models

        llama.cpp returns something like:
          {"object":"list","data":[{"id":"...","owned_by":"llamacpp","meta":{...}}, ...], ...}
        """
        return self._request_json("GET", "/v1/models")

    # -----------------------------
    # llama.cpp specific endpoints
    # -----------------------------

    def tokenize(self, *, content: list[str]) -> Dict[str, Any]:
        """
        POST /tokenize  (llama.cpp server)
        Returns parsed JSON dict, typically: {"tokens":[...]} or {"tokens":[[...],[...]]}
        """
        payload: Dict[str, Any] = {"content": content}
        return self._request_json("POST", "/tokenize", payload=payload)

    def health(self) -> Dict[str, Any]:
        """
        GET /health

        For llama-server this returns: {"status":"ok"} (per your curl).
        """
        return self._request_json("GET", "/health")

    # -----------------------------
    # misc probing
    # -----------------------------

    def raw_get(self, path: str) -> Dict[str, Any]:
        """
        Best-effort helper for GET <path> returning JSON.
        Example: raw_get("/version") if a backend exposes it.

        Raises AppError on failure (backend should swallow if needed).
        """
        return self._request_json("GET", path)