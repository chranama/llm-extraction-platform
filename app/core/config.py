# app/core/config.py
from __future__ import annotations
from typing import Literal, Optional, List
from pydantic import Field
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
    database_url: str = Field(default="postgresql+asyncpg://llm:llm@postgres:5432/llm")

    # --- CORS ---
    cors_allowed_origins: List[str] = Field(default=["*"])

    # --- model config ---
    model_id: str = Field(default="mistralai/Mistral-7B-v0.1")
    model_dtype: Literal["float16", "bfloat16", "float32"] = Field(default="float16")
    model_device: Optional[str] = Field(
        default=None
    )  # 'cuda', 'mps', 'cpu' or None for auto-detect

    # --- Redis ---
    redis_url: Optional[str] = Field(default=None)  # e.g., "redis://localhost:6379/0"
    redis_enabled: bool = Field(default=True)

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

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",   # prevents hard-crashing on unknown env keys
    )

settings = Settings()