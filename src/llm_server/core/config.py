# src/llm_server/core/config.py
from __future__ import annotations

import json
from typing import Literal, Optional, List, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- service info ---
    service_name: str = "LLM Server"
    version: str = "0.1.0"
    debug: bool = False

    # --- server ---
    env: str = Field(default="dev")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # --- database ---
    database_url: str = Field(
        default="postgresql+asyncpg://llm:llm@postgres:5432/llm"
    )

    # --- CORS ---
    cors_allowed_origins: Any = Field(default_factory=lambda: ["*"])

    # --- model config ---
    # Default model served by the runtime
    model_id: str = Field(default="mistralai/Mistral-7B-v0.1")

    # Optional whitelist of allowed models for routing.
    # If this list is empty AND models.yaml is not present,
    # `model_id` is treated as the only allowed model.
    allowed_models: List[str] = Field(
        default_factory=list,
        description=(
            "Optional list of allowed model IDs for routing. "
            "If empty, only `model_id` is accepted (unless models.yaml is used)."
        ),
    )

    # Optional path to models.yaml (relative or absolute).
    models_config_path: Optional[str] = Field(
        default="models.yaml",
        description="Optional path to a YAML file with model routing config.",
    )

    model_dtype: Literal["float16", "bfloat16", "float32"] = Field(default="float16")
    model_device: Optional[str] = Field(
        default=None
    )  # 'cuda', 'mps', 'cpu' or None for auto-detect

    # --- Redis ---
    redis_url: Optional[str] = Field(default=None)
    redis_enabled: bool = Field(
        default=False,
        description="Set to True to require Redis in health/ready checks; False in tests/dev.",
    )

    # --- LLM service ---
    llm_service_url: str = Field(
        default="http://127.0.0.1:9001",
        description="URL of the LLM service (used for remote HTTP backends).",
    )
    http_client_timeout: int = Field(
        default=60,
        description="Timeout (seconds) for HTTP client requests to the LLM service",
    )

    # --- rate limits / quotas ---
    rate_limit_rpm_admin: int = 0
    rate_limit_rpm_default: int = 120
    rate_limit_rpm_free: int = 30
    quota_auto_reset_days: int = 30

    # --- API key cache ---
    api_key_cache_ttl_seconds: int = 10

    # ----------- Convenience properties -----------

    @property
    def all_model_ids(self) -> List[str]:
        """
        Effective list of allowed model IDs.

        - If allowed_models is non-empty, return that.
        - Otherwise, just [model_id].
        """
        return self.allowed_models or [self.model_id]

    @field_validator("cors_allowed_origins", mode="after")
    @classmethod
    def normalize_cors_origins(cls, v: Any) -> List[str]:
        # ... your existing CORS normalization ...
        if v is None:
            return ["*"]
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return ["*"]
            if s == "*":
                return ["*"]
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
            values = [item.strip() for item in s.split(",") if item.strip()]
            return values or ["*"]
        try:
            return [str(v).strip()] if str(v).strip() else ["*"]
        except Exception:
            return ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()