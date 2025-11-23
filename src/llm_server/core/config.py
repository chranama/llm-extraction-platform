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
    # Note: we accept Any here and normalize in a validator to avoid
    # pydantic raising on weird env/.env formats.
    cors_allowed_origins: Any = Field(default_factory=lambda: ["*"])

    # --- model config ---
    model_id: str = Field(default="mistralai/Mistral-7B-v0.1")
    model_dtype: Literal["float16", "bfloat16", "float32"] = Field(default="float16")
    model_device: Optional[str] = Field(
        default=None
    )  # 'cuda', 'mps', 'cpu' or None for auto-detect

    # --- Redis ---
# --- Redis ---
    redis_url: Optional[str] = Field(default=None)  # e.g. "redis://localhost:6379/0"
    redis_enabled: bool = Field(
        default=False,
        description="Set to True to require Redis in health/ready checks; False in tests/dev."
    )

    # --- LLM service ---
    llm_service_url: str = Field(
        default="http://127.0.0.1:9001",
        description="URL of the LLM service (can be overridden with LLM_SERVICE_URL env var)",
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

    # ----------- Validators -----------

    @field_validator("cors_allowed_origins", mode="after")
    @classmethod
    def normalize_cors_origins(cls, v: Any) -> List[str]:
        """
        Normalize CORS origins to a list[str].

        Accepts:
        - list[str] (already parsed)
        - JSON array in env: '["http://foo","http://bar"]'
        - Comma separated string: 'http://foo,http://bar'
        - "*" or empty string
        - Any other junk -> falls back to ["*"]
        """
        if v is None:
            return ["*"]

        # Already a list
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]

        # String from env/.env
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return ["*"]

            # "*" means wildcard
            if s == "*":
                return ["*"]

            # Try JSON list first
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass

            # Fallback: comma separated
            values = [item.strip() for item in s.split(",") if item.strip()]
            return values or ["*"]

        # Anything else -> stringify and wrap
        try:
            return [str(v).strip()] if str(v).strip() else ["*"]
        except Exception:
            return ["*"]

    # ----------- Settings config -----------

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",  # prevents crashing on unknown env keys
    )


settings = Settings()