# server/src/llm_server/core/config.py
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from pydantic_settings.sources import SettingsSourceCallable  # type: ignore
except Exception:
    SettingsSourceCallable = Callable[..., Dict[str, Any]]  # type: ignore

try:
    import yaml
except Exception:
    yaml = None


# ============================================================
# Path + YAML helpers
# ============================================================
def _app_root() -> Path:
    v = (os.environ.get("APP_ROOT") or "").strip()
    return Path(v).expanduser().resolve() if v else Path.cwd().resolve()


def _resolve_path(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (_app_root() / p).resolve()


def _deep_merge(base: Any, overlay: Any) -> Any:
    """
    Deep merge dictionaries: overlay wins.
    Non-dict types are replaced.
    """
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return overlay
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _select_profile_yaml(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports two shapes:

    1) legacy (no profiles):
       service:, server:, model:, ...

    2) profiled:
       base: {...}
       profiles:
         host: {...}
         docker: {...}

    Selection:
      APP_PROFILE env var selects profile (default: "host" if profiles exist).
    """
    if not isinstance(raw, dict):
        return {}

    if "profiles" not in raw and "base" not in raw:
        return raw

    base = raw.get("base") or {}
    profiles = raw.get("profiles") or {}
    if not isinstance(base, dict):
        base = {}
    if not isinstance(profiles, dict):
        profiles = {}

    profile = (os.getenv("APP_PROFILE") or "").strip() or "host"

    overlay = profiles.get(profile) or {}
    if not isinstance(overlay, dict):
        overlay = {}

    merged = _deep_merge(base, overlay)
    merged["_selected_profile"] = profile
    return merged


def _load_yaml_file(path: str) -> Dict[str, Any]:
    if yaml is None:
        return {}
    p = _resolve_path(path)
    if not p.exists():
        return {}
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _load_app_yaml(path: str) -> Dict[str, Any]:
    """
    Load config/server.yaml and map to Settings fields.
    Supports profiles via APP_PROFILE.

    IMPORTANT:
      - This function should be pure/read-only (no env mutation).
      - Env vars still override YAML via pydantic_settings source order.
    """
    raw = _load_yaml_file(path)
    cfg = _select_profile_yaml(raw)
    if not isinstance(cfg, dict):
        return {}

    def g(*keys, default=None):
        cur: Any = cfg
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    out: Dict[str, Any] = {}

    # service
    if (v := g("service", "name")) is not None:
        out["service_name"] = v
    if (v := g("service", "version")) is not None:
        out["version"] = v
    if (v := g("service", "debug")) is not None:
        out["debug"] = v
    if (v := g("service", "env")) is not None:
        out["env"] = v

    # server
    if (v := g("server", "host")) is not None:
        out["host"] = v
    if (v := g("server", "port")) is not None:
        out["port"] = v

    # api
    if (v := g("api", "cors_allowed_origins")) is not None:
        out["cors_allowed_origins"] = v

    # capabilities (support both shapes)
    if (v := g("capabilities", "generate")) is not None:
        out["enable_generate"] = v
    if (v := g("capabilities", "extract")) is not None:
        out["enable_extract"] = v
    if (v := g("capabilities", "enable_generate")) is not None:
        out["enable_generate"] = v
    if (v := g("capabilities", "enable_extract")) is not None:
        out["enable_extract"] = v

    # model
    if (v := g("model", "default_id")) is not None:
        out["model_id"] = v
    if (v := g("model", "allowed_models")) is not None:
        out["allowed_models"] = v
    if (v := g("model", "models_config_path")) is not None:
        out["models_config_path"] = v
    if (v := g("model", "dtype")) is not None:
        out["model_dtype"] = v
    if (v := g("model", "device")) is not None:
        out["model_device"] = v

    # runtime model toggles
    if (v := g("model", "model_load_mode")) is not None:
        out["model_load_mode"] = v
    if (v := g("model", "require_model_ready")) is not None:
        out["require_model_ready"] = v
    if (v := g("model", "token_counting")) is not None:
        out["token_counting"] = v

    # NEW: modelz semantics
    if (v := g("model", "model_readiness_mode")) is not None:
        out["model_readiness_mode"] = v

    # redis
    if (v := g("redis", "enabled")) is not None:
        out["redis_enabled"] = v
    if (v := g("redis", "url")) is not None:
        out["redis_url"] = v

    # http
    if (v := g("http", "llm_service_url")) is not None:
        out["llm_service_url"] = v
    if (v := g("http", "client_timeout_seconds")) is not None:
        out["http_client_timeout"] = v

    # limits (phase 0 knobs)
    if (v := g("limits", "max_concurrent_requests")) is not None:
        out["max_concurrent_requests"] = v
    if (v := g("limits", "mem_guard_enabled")) is not None:
        out["mem_guard_enabled"] = v
    if (v := g("limits", "mem_guard_rss_pct")) is not None:
        out["mem_guard_rss_pct"] = v

    # legacy limits
    if (v := g("limits", "rate_limit_rpm", "admin")) is not None:
        out["rate_limit_rpm_admin"] = v
    if (v := g("limits", "rate_limit_rpm", "default")) is not None:
        out["rate_limit_rpm_default"] = v
    if (v := g("limits", "rate_limit_rpm", "free")) is not None:
        out["rate_limit_rpm_free"] = v
    if (v := g("limits", "quota_auto_reset_days")) is not None:
        out["quota_auto_reset_days"] = v

    # cache
    if (v := g("cache", "api_key_cache_ttl_seconds")) is not None:
        out["api_key_cache_ttl_seconds"] = v

    return out


# ============================================================
# Runtime env coherence
# ============================================================
def _truthy(v: Any) -> str:
    return "1" if bool(v) else "0"


def _sync_runtime_env(s: "Settings") -> None:
    """
    Keep runtime env vars coherent with Settings.

    Use setdefault so real environment overrides (or test overrides) still win.
    """
    os.environ.setdefault("ENV", str(s.env))
    os.environ.setdefault("DEBUG", _truthy(s.debug))

    os.environ.setdefault("ENABLE_GENERATE", _truthy(s.enable_generate))
    os.environ.setdefault("ENABLE_EXTRACT", _truthy(s.enable_extract))

    os.environ.setdefault("REDIS_ENABLED", _truthy(s.redis_enabled))

    os.environ.setdefault("MODEL_LOAD_MODE", str(s.model_load_mode))
    os.environ.setdefault("REQUIRE_MODEL_READY", _truthy(s.require_model_ready))
    os.environ.setdefault("TOKEN_COUNTING", _truthy(s.token_counting))

    # NEW: modelz semantics
    os.environ.setdefault("MODEL_READINESS_MODE", str(s.model_readiness_mode))

    # Concurrency / soft guard knobs (best-effort; do not fight explicit env)
    os.environ.setdefault("MAX_CONCURRENT_REQUESTS", str(int(s.max_concurrent_requests)))
    os.environ.setdefault("MEM_GUARD_ENABLED", _truthy(s.mem_guard_enabled))
    os.environ.setdefault("MEM_GUARD_RSS_PCT", str(float(s.mem_guard_rss_pct)))

    # Explicit “which DB world am I in?” signal (host vs docker vs itest)
    os.environ.setdefault("DB_INSTANCE", str(s.db_instance))


# ============================================================
# Settings
# ============================================================
class Settings(BaseSettings):
    # --- config file path ---
    app_config_path: str = Field(default="config/server.yaml", validation_alias="APP_CONFIG_PATH")

    # --- service info ---
    service_name: str = "LLM Server"
    version: str = "0.1.0"
    debug: bool = False

    # --- server ---
    env: str = "dev"
    host: str = "0.0.0.0"
    port: int = 8000

    # --- database ---
    database_url: str = Field(
        default="postgresql+asyncpg://llm:llm@postgres:5432/llm",
        validation_alias="DATABASE_URL",
    )

    # A human-readable tag for “which database instance am I pointed at?”
    db_instance: str = Field(default="unknown", validation_alias="DB_INSTANCE")

    # --- CORS ---
    cors_allowed_origins: Any = Field(default_factory=lambda: ["*"])

    # --- capabilities ---
    enable_generate: bool = Field(default=True, validation_alias="ENABLE_GENERATE")
    enable_extract: bool = Field(default=True, validation_alias="ENABLE_EXTRACT")

    # --- model config ---
    model_id: str = Field(default="mistralai/Mistral-7B-v0.1")
    allowed_models: List[str] = Field(default_factory=list)

    models_config_path: Optional[str] = Field(
        default="config/models.generate-only.yaml",
        validation_alias="MODELS_YAML",
    )

    model_dtype: Literal["float16", "bfloat16", "float32"] = Field(default="float16")
    model_device: Optional[str] = Field(default=None)

    # --- runtime model behavior toggles ---
    model_load_mode: Literal["off", "lazy", "eager"] = Field(default="lazy", validation_alias="MODEL_LOAD_MODE")
    require_model_ready: bool = Field(default=False, validation_alias="REQUIRE_MODEL_READY")
    token_counting: bool = Field(default=True, validation_alias="TOKEN_COUNTING")

    # NEW: modelz semantics (especially for external backends like llamacpp/remote)
    # - off: /modelz always succeeds (dev convenience)
    # - probe: backend /health (fast)
    # - generate: backend can generate output (tiny completion)
    model_readiness_mode: Literal["off", "probe", "generate"] = Field(
        default="generate",
        validation_alias="MODEL_READINESS_MODE",
    )

    # --- Redis ---
    redis_url: Optional[str] = Field(default=None, validation_alias="REDIS_URL")
    redis_enabled: bool = Field(default=False, validation_alias="REDIS_ENABLED")

    # --- LLM service ---
    llm_service_url: str = Field(default="http://127.0.0.1:9001")
    http_client_timeout: int = Field(default=60)

    # --- Phase 0 guardrails ---
    max_concurrent_requests: int = Field(default=2, validation_alias="MAX_CONCURRENT_REQUESTS")
    mem_guard_enabled: bool = Field(default=False, validation_alias="MEM_GUARD_ENABLED")
    mem_guard_rss_pct: float = Field(default=0.85, validation_alias="MEM_GUARD_RSS_PCT")
    container_memory_bytes: Optional[int] = Field(default=None, validation_alias="CONTAINER_MEMORY_BYTES")

    # --- rate limits / quotas ---
    rate_limit_rpm_admin: int = 0
    rate_limit_rpm_default: int = 120
    rate_limit_rpm_free: int = 30
    quota_auto_reset_days: int = 30

    # --- API key cache ---
    api_key_cache_ttl_seconds: int = 10

    @property
    def all_model_ids(self) -> List[str]:
        return self.allowed_models or [self.model_id]

    @field_validator("max_concurrent_requests", mode="after")
    @classmethod
    def _validate_max_concurrency(cls, v: int) -> int:
        try:
            v = int(v)
        except Exception:
            return 2
        if v < 1:
            return 1
        if v > 64:
            return 64
        return v

    @field_validator("mem_guard_rss_pct", mode="after")
    @classmethod
    def _validate_mem_guard_pct(cls, v: float) -> float:
        try:
            f = float(v)
        except Exception:
            return 0.85
        if f < 0.10:
            return 0.10
        if f > 0.99:
            return 0.99
        return f

    @field_validator("container_memory_bytes", mode="after")
    @classmethod
    def _validate_container_mem_bytes(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        try:
            n = int(v)
            return n if n > 0 else None
        except Exception:
            return None

    @field_validator("require_model_ready", mode="after")
    @classmethod
    def derive_require_model_ready(cls, v: bool, info):
        if v:
            return True
        env = str(info.data.get("env", "dev")).strip().lower()
        return env == "prod"

    @field_validator("cors_allowed_origins", mode="after")
    @classmethod
    def normalize_cors_origins(cls, v: Any) -> List[str]:
        if v is None:
            return ["*"]
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()] or ["*"]
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return ["*"]
            if s == "*":
                return ["*"]
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()] or ["*"]
            except Exception:
                pass
            values = [item.strip() for item in s.split(",") if item.strip()]
            return values or ["*"]
        try:
            return [str(v).strip()] if str(v).strip() else ["*"]
        except Exception:
            return ["*"]

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: SettingsSourceCallable,
        env_settings: SettingsSourceCallable,
        dotenv_settings: SettingsSourceCallable,
        file_secret_settings: SettingsSourceCallable,
    ):
        def yaml_settings() -> Dict[str, Any]:
            path = os.getenv("APP_CONFIG_PATH", "config/server.yaml")
            return _load_app_yaml(path)

        return (init_settings, yaml_settings, dotenv_settings, env_settings, file_secret_settings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    _sync_runtime_env(s)
    return s