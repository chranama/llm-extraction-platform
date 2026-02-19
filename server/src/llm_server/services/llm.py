# server/src/llm_server/services/llm.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from llm_server.core.config import get_settings
from llm_server.core.errors import AppError
from llm_server.services.llm_config import load_models_config, ModelSpec
from llm_server.services.llm_registry import MultiModelManager

from llm_server.services.backends.transformers_backend import TransformersBackend, TransformersBackendConfig
from llm_server.services.backends.llamacpp_backend import LlamaCppBackend, LlamaCppBackendConfig
from llm_server.services.backends.backend_api import OpenAICompatClient, OpenAICompatClientConfig

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
      - "transformers" | "llamacpp" | "remote"
    """
    s = (str(raw or "")).strip().lower()
    if not s:
        return "transformers"
    if s == "local":
        return "transformers"
    if s in ("transformers", "llamacpp", "remote"):
        return s
    if s.startswith("llama"):
        return "llamacpp"
    return s


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
    t = float(total_timeout_seconds) if total_timeout_seconds and total_timeout_seconds > 0 else 60.0
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
    Generic OpenAI-compatible remote completion backend.
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
    ) -> None:
        self.model_id = model_id
        self.remote_model_id = remote_model_id or model_id

        ct = float(min(float(connect_timeout_seconds), float(timeout_seconds))) if timeout_seconds > 0 else float(connect_timeout_seconds)

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
        data = self._client.completions(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            model=self.remote_model_id,
            extra=kwargs if kwargs else None,
        )
        try:
            choices = data.get("choices") or []
            if isinstance(choices, list) and choices:
                c0 = choices[0] or {}
                if isinstance(c0, dict):
                    return str(c0.get("text") or "")
        except Exception:
            pass
        return ""


# ------------------------------------------------------------
# Backend builder
# ------------------------------------------------------------

def _build_backend_for_model(*, sp: ModelSpec, settings: Any) -> Tuple[Any, Dict[str, Any]]:
    backend_name = _normalize_backend_name(getattr(sp, "backend", None) or "transformers")
    caps = _caps_meta(sp)
    load_mode = str(getattr(sp, "load_mode", "lazy") or "lazy")

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
                default_temperature=float(_as_float(_get_nested(transformers_cfg, "default_temperature")) or 0.7),
                default_top_p=float(_as_float(_get_nested(transformers_cfg, "default_top_p")) or 0.95),
            ),
        )
        meta = {"backend": "transformers", "capabilities": caps, "load_mode": load_mode, "hf_id": hf_id}
        return b, meta

    # -------------------------
    # llamacpp (external llama-server)
    # -------------------------
    if backend_name == "llamacpp":
        server_url = _as_str(_get_nested(llamacpp_cfg, "server_url")) or _as_str(os.environ.get("LLAMA_SERVER_URL"))
        if not server_url:
            raise AppError(
                code="backend_config_invalid",
                message="llamacpp backend requires server_url (set models.yaml llamacpp.server_url or env LLAMA_SERVER_URL)",
                status_code=500,
                extra={"model_id": sp.id},
            )

        api_key = _as_str(_get_nested(llamacpp_cfg, "api_key")) or _as_str(os.environ.get("LLAMA_SERVER_API_KEY"))

        requested_timeout = _requested_timeout_seconds(llamacpp_cfg, settings=settings)
        timeout_seconds = _aligned_backend_timeout_seconds(requested=requested_timeout)

        connect_timeout_seconds = _requested_connect_timeout_seconds(llamacpp_cfg, total_timeout_seconds=timeout_seconds)

        model_name = _as_str(_get_nested(llamacpp_cfg, "model_name"))
        default_temperature = float(_as_float(_get_nested(llamacpp_cfg, "default_temperature")) or 0.7)
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
            "server_url": server_url,
            "capabilities": caps,
            "load_mode": "remote_process",
            "timeout_seconds": float(timeout_seconds),
            "connect_timeout_seconds": float(connect_timeout_seconds),
        }
        return b, meta

    # -------------------------
    # remote (OpenAI-compat)
    # -------------------------
    if backend_name == "remote":
        base_url = _as_str(_get_nested(remote_cfg, "base_url")) or _as_str(getattr(settings, "llm_service_url", None))
        if not base_url:
            raise AppError(
                code="remote_models_require_llm_service_url",
                message="remote backend requires Settings.llm_service_url or models.yaml remote.base_url",
                status_code=500,
                extra={"model_id": sp.id},
            )

        api_key = _as_str(_get_nested(remote_cfg, "api_key")) or _as_str(os.environ.get("REMOTE_BACKEND_API_KEY"))

        requested_timeout = _requested_timeout_seconds(remote_cfg, settings=settings)
        timeout_seconds = _aligned_backend_timeout_seconds(requested=requested_timeout)

        connect_timeout_seconds = _requested_connect_timeout_seconds(remote_cfg, total_timeout_seconds=timeout_seconds)

        remote_model_id = _as_str(_get_nested(remote_cfg, "model_id")) or _as_str(_get_nested(remote_cfg, "model_name"))

        b = RemoteBackend(
            model_id=sp.id,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=float(timeout_seconds),
            connect_timeout_seconds=float(connect_timeout_seconds),
            remote_model_id=remote_model_id,
        )
        meta = {
            "backend": "remote",
            "base_url": base_url,
            "capabilities": caps,
            "load_mode": "remote",
            "timeout_seconds": float(timeout_seconds),
            "connect_timeout_seconds": float(connect_timeout_seconds),
        }
        return b, meta

    raise AppError(
        code="backend_config_invalid",
        message="Unknown backend in model spec",
        status_code=500,
        extra={"model_id": sp.id, "backend": backend_name, "allowed": ["transformers", "llamacpp", "remote"]},
    )


# ------------------------------------------------------------
# Public builder (wiring)
# ------------------------------------------------------------

def build_llm_from_settings() -> Any:
    """
    Build model backend(s) based on models.yaml and profile selection.

    Behavior:
      - If MODEL_LOAD_MODE=off => empty MultiModelManager
      - If ENABLE_MULTI_MODELS=0 => return single backend object (default model)
      - Else => MultiModelManager(models=..., default_id=..., model_meta=...)
    """
    cfg = load_models_config()
    s = get_settings()

    primary_id = cfg.primary_id
    spec_map: Dict[str, ModelSpec] = {sp.id: sp for sp in cfg.models}

    global_load_mode = (os.getenv("MODEL_LOAD_MODE") or getattr(s, "model_load_mode", None) or "").strip().lower()
    if global_load_mode == "off":
        return MultiModelManager(models={}, default_id=primary_id, model_meta={})

    ordered_ids: List[str] = [
        mid
        for mid in cfg.model_ids
        if (spec_map.get(mid) is not None and str(getattr(spec_map[mid], "load_mode", "lazy")).lower() != "off")
    ]
    if not ordered_ids:
        raise AppError(
            code="model_config_invalid",
            message="No enabled models after applying load_mode=off filters",
            status_code=500,
            extra={"primary_id": primary_id, "configured_ids": cfg.model_ids},
        )

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