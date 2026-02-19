# simulations/traffic/client.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class SimClientError(RuntimeError):
    """
    Deterministic error wrapper for HTTP failures.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        url: str | None = None,
        response_text: str | None = None,
        payload: Any | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.url = url
        self.response_text = response_text
        self.payload = payload
        self.cause = cause


@dataclass(frozen=True)
class ClientConfig:
    base_url: str
    api_key: str
    timeout_s: float = 30.0

    # Optional default model override sent to endpoints that accept "model"
    model: Optional[str] = None

    # Optional additional headers for scenario-level attribution (e.g. X-Sim-Scenario)
    extra_headers: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_env(
        *,
        base_url_env: str = "SIM_BASE_URL",
        api_key_env: str = "SIM_API_KEY",
        api_key_fallback_env: str = "API_KEY",
        timeout_env: str = "SIM_TIMEOUT_S",
        model_env: str = "SIM_MODEL",
        default_base_url: str = "http://127.0.0.1:8000",
    ) -> "ClientConfig":
        base = (os.getenv(base_url_env, "") or "").strip() or default_base_url

        key = (os.getenv(api_key_env, "") or "").strip()
        if not key:
            key = (os.getenv(api_key_fallback_env, "") or "").strip()
        if not key:
            raise SimClientError(
                f"Missing API key. Set {api_key_env} (preferred) or {api_key_fallback_env} in your environment."
            )

        timeout_raw = (os.getenv(timeout_env, "") or "").strip()
        timeout_s = 30.0
        if timeout_raw:
            try:
                timeout_s = float(timeout_raw)
            except Exception:
                timeout_s = 30.0

        model = (os.getenv(model_env, "") or "").strip() or None
        return ClientConfig(base_url=base.rstrip("/"), api_key=key, timeout_s=timeout_s, model=model)


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    json: Any
    text: str
    elapsed_ms: float


class SimClient:
    def __init__(self, cfg: ClientConfig):
        self.cfg = cfg

    # -------------------------
    # low-level request helpers
    # -------------------------

    def _full_url(self, path: str) -> str:
        p = path if path.startswith("/") else f"/{path}"
        return f"{self.cfg.base_url}{p}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        query: Dict[str, str] | None = None,
        headers: Dict[str, str] | None = None,
    ) -> HTTPResponse:
        url = self._full_url(path)
        if query:
            # tiny deterministic query encoder (no requests dependency)
            parts = []
            for k, v in query.items():
                if v is None:
                    continue
                parts.append(f"{k}={_url_escape(str(v))}")
            if parts:
                url = url + ("?" if "?" not in url else "&") + "&".join(parts)

        body_bytes: bytes | None = None
        base_headers: Dict[str, str] = {
            "X-API-Key": self.cfg.api_key,
            "Accept": "application/json",
            "User-Agent": "simulations/traffic-client",
        }

        # Apply config-level headers first, then per-call headers last (highest priority).
        if self.cfg.extra_headers:
            base_headers.update(self.cfg.extra_headers)
        if headers:
            base_headers.update(headers)

        if json_body is not None:
            base_headers["Content-Type"] = "application/json"
            body_bytes = json.dumps(json_body, ensure_ascii=False).encode("utf-8")

        req = Request(url=url, method=method.upper(), data=body_bytes, headers=base_headers)
        t0 = time.time()
        try:
            with urlopen(req, timeout=self.cfg.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                elapsed_ms = (time.time() - t0) * 1000
                try:
                    payload = json.loads(raw) if raw else None
                except Exception:
                    payload = None
                return HTTPResponse(
                    status=int(getattr(resp, "status", 200)),
                    json=payload,
                    text=raw,
                    elapsed_ms=elapsed_ms,
                )
        except HTTPError as e:
            # HTTPError is also a file-like object; read body if possible
            try:
                raw = e.read().decode("utf-8", errors="replace")
            except Exception:
                raw = ""
            raise SimClientError(
                f"HTTP {e.code} for {method.upper()} {url}",
                status=int(e.code),
                url=url,
                response_text=raw,
                payload=_best_effort_json(raw),
                cause=e,
            ) from e
        except URLError as e:
            raise SimClientError(
                f"Connection error for {method.upper()} {url}: {e}",
                status=None,
                url=url,
                response_text=None,
                payload=None,
                cause=e,
            ) from e
        except Exception as e:
            raise SimClientError(
                f"Unexpected error for {method.upper()} {url}: {type(e).__name__}: {e}",
                status=None,
                url=url,
                response_text=None,
                payload=None,
                cause=e,
            ) from e

    # -------------------------
    # public API helpers
    # -------------------------

    def get_admin_policy(self) -> HTTPResponse:
        return self._request("GET", "/v1/admin/policy")

    def post_admin_policy_reload(self) -> HTTPResponse:
        return self._request("POST", "/v1/admin/policy/reload")

    def post_admin_reload(self) -> HTTPResponse:
        return self._request("POST", "/v1/admin/reload")

    def post_admin_models_load(self, *, model_id: str | None = None) -> HTTPResponse:
        """
        Force-load model weights (admin override to lazy/off).

        Body:
          {"model_id": "..."}  # optional
        """
        body: Dict[str, Any] = {}
        if model_id:
            body["model_id"] = str(model_id)
        return self._request("POST", "/v1/admin/models/load", json_body=body or None)

    def post_admin_write_generate_slo(
        self,
        *,
        window_seconds: int = 300,
        route: str | None = None,
        model_id: str | None = None,
        out_path: str | None = None,
    ) -> HTTPResponse:
        """
        Trigger server-side computation + artifact write for runtime_generate_slo_v1.

        NOTE: out_path is interpreted by the SERVER process (typically relative to its CWD),
        so prefer absolute paths or ensure server CWD matches repo root in local dev.
        """
        q: Dict[str, str] = {"window_seconds": str(int(window_seconds))}
        if route:
            q["route"] = route
        if model_id:
            q["model_id"] = model_id
        if out_path:
            q["out_path"] = out_path
        return self._request("POST", "/v1/admin/slo/generate/write", query=q)

    def post_generate(
        self,
        *,
        prompt: str,
        model: str | None = None,
        cache: bool = True,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
    ) -> HTTPResponse:
        body: Dict[str, Any] = {"prompt": prompt, "cache": bool(cache)}
        mid = model or self.cfg.model
        if mid:
            body["model"] = mid
        if max_new_tokens is not None:
            body["max_new_tokens"] = int(max_new_tokens)
        if temperature is not None:
            body["temperature"] = float(temperature)
        if top_p is not None:
            body["top_p"] = float(top_p)
        if top_k is not None:
            body["top_k"] = int(top_k)
        if stop is not None:
            body["stop"] = list(stop)
        return self._request("POST", "/v1/generate", json_body=body)

    def post_extract(
        self,
        *,
        schema_id: str,
        text: str,
        model: str | None = None,
        max_new_tokens: int | None = 512,
        temperature: float | None = 0.0,
        cache: bool = True,
        repair: bool = True,
    ) -> HTTPResponse:
        body: Dict[str, Any] = {
            "schema_id": schema_id,
            "text": text,
            "cache": bool(cache),
            "repair": bool(repair),
        }
        mid = model or self.cfg.model
        if mid:
            body["model"] = mid
        if max_new_tokens is not None:
            body["max_new_tokens"] = int(max_new_tokens)
        if temperature is not None:
            body["temperature"] = float(temperature)
        return self._request("POST", "/v1/extract", json_body=body)

    # -------------------------
    # convenience helpers
    # -------------------------

    @staticmethod
    def explain_error(e: SimClientError) -> Dict[str, Any]:
        """
        Render a deterministic, JSON-serializable error summary for CLIs.
        """
        return {
            "ok": False,
            "error": str(e),
            "status": e.status,
            "url": e.url,
            "response_text": e.response_text,
            "payload": e.payload,
            "cause": f"{type(e.cause).__name__}: {e.cause}" if e.cause else None,
        }


def _best_effort_json(s: str) -> Any | None:
    try:
        return json.loads(s) if s else None
    except Exception:
        return None


def _url_escape(s: str) -> str:
    # minimal URL escaping for query params
    out = []
    for ch in s:
        o = ord(ch)
        if ("a" <= ch <= "z" or "A" <= ch <= "Z" or "0" <= ch <= "9" or ch in {"-", "_", ".", "~"}):
            out.append(ch)
        elif ch == " ":
            out.append("%20")
        else:
            out.append(f"%{o:02X}")
    return "".join(out)