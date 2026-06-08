# server/src/llm_server/services/llm_runtime/llm_build.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from llm_server.core.config import get_settings
from llm_server.core.errors import AppError
from llm_server.services.llm_runtime.llm_config import ModelSpec, load_models_config
from llm_server.services.llm_runtime.llm_registry import MultiModelManager

from llm_server.services.backends.base import GenerateResult, GenerateTimings, GenerateUsage
from llm_server.services.backends.backend_api import OpenAICompatClient, OpenAICompatClientConfig
from llm_server.services.backends.fake_backend import FakeBackend, FakeBackendConfig
from llm_server.services.backends.llamacpp_backend import LlamaCppBackend, LlamaCppBackendConfig
from llm_server.services.backends.transformers_backend import (
    TransformersBackend,
    TransformersBackendConfig,
)

# Gate config is the source-of-truth for total budget
from llm_server.services.limits.config import load_generate_gate_config

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

DEFAULT_STOPS: List[str] = ["\nUser:", "\nuser:", "User:", "###"]


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _get_attr_or_key(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _get_nested(obj: Any, path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        cur = _get_attr_or_key(cur, part)
        if cur is None:
            return None
    return cur


def _as_str(x: Any) -> Optional[str]:
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    return None


def _as_int(x: Any) -> Optional[int]:
    if x is None or isinstance(x, bool):
        return None
    try:
        return int(x)
    except Exception:
        return None


def _as_float(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _caps_meta(sp: Optional[ModelSpec]) -> Optional[list[str]]:
    """
    Preserve semantics:
      - None => unspecified (fail-open)
      - dict => True keys enabled
      - list/tuple/set => allowlist
      - str => single cap
    Returns stable sorted list[str] (or None).
    """
    if sp is None:
        return None

    caps = getattr(sp, "capabilities", None)
    if caps is None:
        return None

    def _norm_one(x: object) -> Optional[str]:
        if not isinstance(x, str):
            return None
        s = x.strip().lower()
        return s or None

    out: list[str] = []

    if isinstance(caps, dict):
        for k, v in caps.items():
            kk = _norm_one(k)
            if kk and bool(v):
                out.append(kk)
    elif isinstance(caps, (list, tuple, set)):
        for x in caps:
            s = _norm_one(x)
            if s:
                out.append(s)
    elif isinstance(caps, str):
        s = _norm_one(caps)
        if s:
            out.append(s)
    else:
        return None

    out = sorted(set(out))
    return out or None


def _normalize_backend_name(raw: Any) -> str:
    """
    Back-compat:
      - "local" => "transformers"
    New:
      - "transformers" | "llamacpp" | "remote" | "vllm" | "fake"
    """
    s = (str(raw or "")).strip().lower()
    if not s:
        return "transformers"
    if s == "local":
        return "transformers"
    if s in ("vllm", "openai", "openai_compat", "openai-compatible"):
        return "remote"
    if s in ("transformers", "llamacpp", "remote", "fake"):
        return s
    if s.startswith("llama"):
        return "llamacpp"
    return s


def _deployment_key_for_model(sp: ModelSpec) -> Optional[str]:
    """
    Prefer explicit per-model deployment_key in models.yaml.
    Allow env override for demos/tests.
    """
    env_override = _as_str(os.environ.get("DEPLOYMENT_KEY"))
    if env_override:
        return env_override
    v = getattr(sp, "deployment_key", None)
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


# ------------------------------------------------------------
# Timeout alignment helpers
# ------------------------------------------------------------


def _timeout_alignment_buffer_seconds() -> float:
    """
    Buffer so the upstream HTTP call times out BEFORE the gate budget expires.
    """
    raw = os.getenv("LLM_TIMEOUT_ALIGNMENT_BUFFER_SECONDS", "1.0")
    try:
        v = float(raw)
    except Exception:
        v = 1.0
    # sane bounds
    if v < 0.0:
        v = 0.0
    if v > 10.0:
        v = 10.0
    return v


def _aligned_backend_timeout_seconds(*, requested: float) -> float:
    """
    Align backend HTTP timeout to the GenerateGate total timeout budget.

    Rule:
      T_http <= max(1.0, gate_timeout - buffer)

    If gate config isn't enabled/available, returns requested.
    """
    req = float(requested) if requested and requested > 0 else 60.0

    try:
        gate_cfg = load_generate_gate_config(settings=None)
        gate_timeout = float(getattr(gate_cfg, "timeout_seconds", 0.0) or 0.0)
        gate_enabled = bool(getattr(gate_cfg, "enabled", True))
        if not gate_enabled or gate_timeout <= 0:
            return req

        buf = _timeout_alignment_buffer_seconds()
        budget = max(1.0, gate_timeout - buf)
        return float(min(req, budget))
    except Exception:
        # best-effort: never fail model construction due to alignment logic
        return req


def _aligned_connect_timeout_seconds(total_timeout_seconds: float) -> float:
    """
    Ensure connect timeout never exceeds total timeout (httpx will complain).
    """
    t = (
        float(total_timeout_seconds)
        if total_timeout_seconds and total_timeout_seconds > 0
        else 60.0
    )
    return float(min(5.0, t))


def _requested_timeout_seconds(cfg_block: Any, *, settings: Any) -> float:
    """
    Read per-backend timeout_seconds with a Settings fallback.
    """
    return float(
        _as_float(_get_nested(cfg_block, "timeout_seconds"))
        or float(getattr(settings, "http_client_timeout", 60) or 60)
    )


def _requested_connect_timeout_seconds(cfg_block: Any, *, total_timeout_seconds: float) -> float:
    """
    Prefer explicit connect_timeout_seconds, else align to <= total.
    """
    raw = _as_float(_get_nested(cfg_block, "connect_timeout_seconds"))
    if raw is not None and raw > 0:
        return float(min(float(raw), float(total_timeout_seconds)))
    return _aligned_connect_timeout_seconds(float(total_timeout_seconds))


# ------------------------------------------------------------
# Remote backend (OpenAI-compat)
# ------------------------------------------------------------


class RemoteBackend:
    """
    Generic OpenAI-compatible remote backend.

    This supports llama.cpp-compatible completion servers, vLLM, and other
    OpenAI-compatible model runtimes without loading model weights in the API
    process.
    """

    backend_name: str = "remote"

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        connect_timeout_seconds: float = 5.0,
        remote_model_id: str | None = None,
        request_mode: str = "completion",
        provider: str = "openai_compat",
        default_temperature: float = 0.7,
        default_top_p: float = 0.95,
    ) -> None:
        self.model_id = model_id
        self.remote_model_id = remote_model_id or model_id
        self.provider = (provider or "openai_compat").strip().lower() or "openai_compat"
        mode = (request_mode or "completion").strip().lower()
        self.request_mode = "chat" if mode in {"chat", "chat_completions"} else "completion"
        self.default_temperature = float(default_temperature)
        self.default_top_p = float(default_top_p)

        ct = (
            float(min(float(connect_timeout_seconds), float(timeout_seconds)))
            if timeout_seconds > 0
            else float(connect_timeout_seconds)
        )

        self._client = OpenAICompatClient(
            OpenAICompatClientConfig(
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=float(timeout_seconds),
                connect_timeout_seconds=float(ct),
            )
        )

    def ensure_loaded(self) -> None:
        return None

    def model_info(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": True,
            "backend": self.backend_name,
            "provider": self.provider,
            "model_id": self.model_id,
            "remote_model_id": self.remote_model_id,
            "request_mode": self.request_mode,
            "loaded": None,
            "runtime": {},
        }
        try:
            out["runtime"]["health"] = self._client.health()
        except Exception as e:
            out["runtime"]["health_error"] = repr(e)
        try:
            out["runtime"]["models_raw"] = self._client.models()
        except Exception as e:
            out["runtime"]["models_error"] = repr(e)
        for path in ("/version", "/v1/version", "/build", "/v1/build"):
            try:
                v = self._client.raw_get(path)
                if isinstance(v, dict) and v:
                    out["runtime"]["server_version"] = v
                    out["runtime"]["server_version_path"] = path
                    break
            except Exception:
                continue
        return out

    def is_ready(self) -> tuple[bool, dict[str, Any]]:
        try:
            data = self._client.health()
            ok = bool(
                isinstance(data, dict)
                and (
                    data.get("status") == "ok"
                    or data.get("ok") is True
                    or data.get("healthy") is True
                )
            )
            return ok, {"health": data, "provider": self.provider}
        except Exception as e:
            return False, {"error": repr(e), "provider": self.provider}

    def can_generate(self) -> tuple[bool, dict[str, Any]]:
        try:
            t0 = time.perf_counter()
            result = self.generate_rich(prompt="ping", max_new_tokens=1, temperature=0.0)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return bool(result.text), {
                "latency_ms": dt_ms,
                "sample": result.text[:80],
                "provider": self.provider,
                "request_mode": self.request_mode,
            }
        except Exception as e:
            return False, {
                "error": repr(e),
                "provider": self.provider,
                "request_mode": self.request_mode,
            }

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
        return self.generate_rich(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            **kwargs,
        ).text

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
            raise AppError(
                code="invalid_request", message="prompt must be a non-empty string", status_code=400
            )

        temp = (
            float(temperature)
            if isinstance(temperature, (int, float))
            else float(self.default_temperature)
        )
        tp = float(top_p) if isinstance(top_p, (int, float)) else float(self.default_top_p)
        extra: Dict[str, Any] = {k: v for k, v in kwargs.items() if v is not None}

        t0 = time.perf_counter()
        if self.request_mode == "chat":
            data = self._client.chat_completions(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temp,
                top_p=tp,
                top_k=top_k,
                stop=stop,
                model=self.remote_model_id,
                extra=extra or None,
            )
        else:
            data = self._client.completions(
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=temp,
                top_p=tp,
                top_k=top_k,
                stop=stop,
                model=self.remote_model_id,
                extra=extra or None,
            )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        text = ""
        try:
            choices = data.get("choices") or []
            if isinstance(choices, list) and choices:
                c0 = choices[0] or {}
                if isinstance(c0, dict):
                    if self.request_mode == "chat":
                        msg = c0.get("message") or {}
                        if isinstance(msg, dict):
                            text = str(msg.get("content") or "")
                    else:
                        text = str(c0.get("text") or "")
        except Exception:
            text = ""

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


# ------------------------------------------------------------
# Backend builder
# ------------------------------------------------------------


def _build_backend_for_model(*, sp: ModelSpec, settings: Any) -> Tuple[Any, Dict[str, Any]]:
    raw_backend_name = getattr(sp, "backend", None) or "transformers"
    backend_name = _normalize_backend_name(raw_backend_name)
    caps = _caps_meta(sp)
    load_mode = str(getattr(sp, "load_mode", "lazy") or "lazy")

    deployment_key = _deployment_key_for_model(sp)

    # Optional per-model readiness mode (parsed by llm_config.py / models.yaml)
    readiness_mode = getattr(sp, "readiness_mode", None)
    if isinstance(readiness_mode, str):
        readiness_mode = readiness_mode.strip().lower() or None
    else:
        readiness_mode = None

    transformers_cfg = _get_attr_or_key(sp, "transformers")
    llamacpp_cfg = _get_attr_or_key(sp, "llamacpp")
    remote_cfg = _get_attr_or_key(sp, "remote")

    # -------------------------
    # transformers (in-process)
    # -------------------------
    if backend_name == "transformers":
        hf_id = _as_str(_get_nested(transformers_cfg, "hf_id")) or sp.id
        device = _as_str(_get_nested(transformers_cfg, "device")) or "auto"
        dtype = _as_str(_get_nested(transformers_cfg, "dtype"))
        trc = bool(_get_nested(transformers_cfg, "trust_remote_code") or False)

        b = TransformersBackend(
            model_id=sp.id,
            cfg=TransformersBackendConfig(
                hf_id=hf_id,
                device=device,
                dtype=dtype,
                trust_remote_code=trc,
                default_temperature=float(
                    _as_float(_get_nested(transformers_cfg, "default_temperature")) or 0.7
                ),
                default_top_p=float(
                    _as_float(_get_nested(transformers_cfg, "default_top_p")) or 0.95
                ),
            ),
        )
        meta = {
            "backend": "transformers",
            "load_scope": "in_process",
            "capabilities": caps,
            "load_mode": load_mode,
            "hf_id": hf_id,
            "readiness_mode": readiness_mode,
            "deployment_key": deployment_key,
        }
        return b, meta

    # -------------------------
    # llamacpp (external llama-server)
    # -------------------------
    if backend_name == "llamacpp":
        server_url = _as_str(_get_nested(llamacpp_cfg, "server_url")) or _as_str(
            os.environ.get("LLAMA_SERVER_URL")
        )
        if not server_url:
            raise AppError(
                code="backend_config_invalid",
                message="llamacpp backend requires server_url (set models.yaml llamacpp.server_url or env LLAMA_SERVER_URL)",
                status_code=500,
                extra={"model_id": sp.id},
            )

        api_key = _as_str(_get_nested(llamacpp_cfg, "api_key")) or _as_str(
            os.environ.get("LLAMA_SERVER_API_KEY")
        )

        requested_timeout = _requested_timeout_seconds(llamacpp_cfg, settings=settings)
        timeout_seconds = _aligned_backend_timeout_seconds(requested=requested_timeout)
        connect_timeout_seconds = _requested_connect_timeout_seconds(
            llamacpp_cfg, total_timeout_seconds=timeout_seconds
        )

        model_name = _as_str(_get_nested(llamacpp_cfg, "model_name"))
        default_temperature = float(
            _as_float(_get_nested(llamacpp_cfg, "default_temperature")) or 0.7
        )
        default_top_p = float(_as_float(_get_nested(llamacpp_cfg, "default_top_p")) or 0.95)

        b = LlamaCppBackend(
            model_id=sp.id,
            cfg=LlamaCppBackendConfig(
                server_url=server_url,
                api_key=api_key,
                timeout_seconds=float(timeout_seconds),
                connect_timeout_seconds=float(connect_timeout_seconds),
                model_name=model_name,
                default_temperature=default_temperature,
                default_top_p=default_top_p,
            ),
        )
        meta = {
            "backend": "llamacpp",
            "load_scope": "external",
            "server_url": server_url,
            "capabilities": caps,
            # NOTE: this is the model's own mode from models.yaml, not Settings.model_load_mode
            "load_mode": load_mode,
            "timeout_seconds": float(timeout_seconds),
            "connect_timeout_seconds": float(connect_timeout_seconds),
            "readiness_mode": readiness_mode,
            "deployment_key": deployment_key,
        }
        return b, meta

    # -------------------------
    # remote (OpenAI-compat)
    # -------------------------
    if backend_name == "remote":
        base_url = _as_str(_get_nested(remote_cfg, "base_url")) or _as_str(
            getattr(settings, "llm_service_url", None)
        )
        if not base_url:
            raise AppError(
                code="remote_models_require_llm_service_url",
                message="remote backend requires Settings.llm_service_url or models.yaml remote.base_url",
                status_code=500,
                extra={"model_id": sp.id},
            )

        api_key = _as_str(_get_nested(remote_cfg, "api_key")) or _as_str(
            os.environ.get("REMOTE_BACKEND_API_KEY")
        )

        requested_timeout = _requested_timeout_seconds(remote_cfg, settings=settings)
        timeout_seconds = _aligned_backend_timeout_seconds(requested=requested_timeout)
        connect_timeout_seconds = _requested_connect_timeout_seconds(
            remote_cfg, total_timeout_seconds=timeout_seconds
        )

        remote_model_id = _as_str(_get_nested(remote_cfg, "model_id")) or _as_str(
            _get_nested(remote_cfg, "model_name")
        )
        provider = (
            (
                _as_str(_get_nested(remote_cfg, "provider"))
                or ("vllm" if str(raw_backend_name).strip().lower() == "vllm" else "openai_compat")
            )
            .strip()
            .lower()
        )
        request_mode = _as_str(_get_nested(remote_cfg, "request_mode")) or (
            "chat" if provider == "vllm" else "completion"
        )
        default_temperature = float(
            _as_float(_get_nested(remote_cfg, "default_temperature")) or 0.7
        )
        default_top_p = float(_as_float(_get_nested(remote_cfg, "default_top_p")) or 0.95)

        b = RemoteBackend(
            model_id=sp.id,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=float(timeout_seconds),
            connect_timeout_seconds=float(connect_timeout_seconds),
            remote_model_id=remote_model_id,
            request_mode=request_mode,
            provider=provider,
            default_temperature=default_temperature,
            default_top_p=default_top_p,
        )
        meta = {
            "backend": "remote",
            "provider": provider,
            "load_scope": "external",
            "base_url": base_url,
            "request_mode": request_mode,
            "capabilities": caps,
            # NOTE: this is the model's own mode from models.yaml, not Settings.model_load_mode
            "load_mode": load_mode,
            "timeout_seconds": float(timeout_seconds),
            "connect_timeout_seconds": float(connect_timeout_seconds),
            "readiness_mode": readiness_mode,
            "deployment_key": deployment_key,
        }
        return b, meta

    # -------------------------
    # fake (deterministic local proof backend)
    # -------------------------
    if backend_name == "fake":
        b = FakeBackend(
            model_id=sp.id,
            cfg=FakeBackendConfig(
                output_text=str(_get_nested(remote_cfg, "output_text") or "ping"),
            ),
        )
        meta = {
            "backend": "fake",
            "load_scope": "external",
            "capabilities": caps,
            "load_mode": load_mode,
            "readiness_mode": readiness_mode,
            "deployment_key": deployment_key,
        }
        return b, meta

    raise AppError(
        code="backend_config_invalid",
        message="Unknown backend in model spec",
        status_code=500,
        extra={
            "model_id": sp.id,
            "backend": backend_name,
            "allowed": ["transformers", "llamacpp", "remote", "vllm", "fake"],
        },
    )


# ------------------------------------------------------------
# Public builder (wiring)
# ------------------------------------------------------------


def build_llm_from_settings() -> Any:
    """
    Build model backend(s) based on models.yaml and profile selection.

    Key semantics:
      - Settings/ENV MODEL_LOAD_MODE controls *weight loading behavior elsewhere*,
        NOT whether models exist in the registry.
      - Per-model load_mode=off in models.yaml disables that model (excluded from registry).
      - ENABLE_MULTI_MODELS=0 returns a single backend (default model).
      - Otherwise returns MultiModelManager(models=..., default_id=..., model_meta=...).

    IMPORTANT:
      - This function must never trigger weight loading. It only constructs backends/clients.
    """
    cfg = load_models_config()
    s = get_settings()

    primary_id = cfg.primary_id
    spec_map: Dict[str, ModelSpec] = {sp.id: sp for sp in cfg.models}

    # Apply per-model load_mode=off filters (this is *model disable*, not server load mode).
    ordered_ids: List[str] = [
        mid
        for mid in cfg.model_ids
        if (
            spec_map.get(mid) is not None
            and str(getattr(spec_map[mid], "load_mode", "lazy")).lower() != "off"
        )
    ]
    if not ordered_ids:
        raise AppError(
            code="model_config_invalid",
            message="No enabled models after applying load_mode=off filters",
            status_code=500,
            extra={"primary_id": primary_id, "configured_ids": cfg.model_ids},
        )

    # Ensure primary/default is first and valid.
    if primary_id in ordered_ids:
        ordered_ids = [primary_id] + [x for x in ordered_ids if x != primary_id]
    else:
        primary_id = ordered_ids[0]

    multi_enabled = _truthy_env("ENABLE_MULTI_MODELS", default=False)
    if not multi_enabled:
        ordered_ids = [primary_id]

    # single-model shortcut
    if len(ordered_ids) == 1:
        sp = spec_map[primary_id]
        backend, _meta = _build_backend_for_model(sp=sp, settings=s)
        return backend

    models: Dict[str, Any] = {}
    meta: Dict[str, Dict[str, Any]] = {}

    for mid in ordered_ids:
        sp = spec_map.get(mid)
        if sp is None:
            continue
        backend, m = _build_backend_for_model(sp=sp, settings=s)
        models[mid] = backend
        meta[mid] = m

    return MultiModelManager(models=models, default_id=primary_id, model_meta=meta)
