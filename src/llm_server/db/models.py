# app/db/models.py
from __future__ import annotations

from datetime import datetime, UTC
from typing import Optional, Dict, Any
import enum

from sqlalchemy import (
    String,
    Integer,
    Float,
    DateTime,
    Text,
    JSON,
    UniqueConstraint,
    Index,
    ForeignKey,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


# Optional enum for use in application code (not enforced by DB)
class Role(str, enum.Enum):
    admin = "admin"
    standard = "standard"
    free = "free"


class Base(DeclarativeBase):
    pass


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def utc_now() -> datetime:
    """Timezone-aware UTC timestamp (replacement for datetime.utcnow)."""
    return datetime.now(UTC)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

class RoleTable(Base):
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=func.now(),
        nullable=False,
        index=True,
    )


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    label: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    active: Mapped[bool] = mapped_column(default=True)

    # Quotas (optional)
    quota_monthly: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # NULL = unlimited
    quota_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    quota_reset_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # NEW: link each API key to an optional role row
    role_id: Mapped[Optional[int]] = mapped_column(ForeignKey("roles.id"), nullable=True)
    role: Mapped[Optional["RoleTable"]] = relationship("RoleTable")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,        # Python-side fallback
        server_default=func.now(),  # DB-side default
        nullable=False,
        index=True,
    )


class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # request context
    api_key: Mapped[Optional[str]] = mapped_column(String(128), index=True, nullable=True)
    request_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    route: Mapped[str] = mapped_column(String(64))  # e.g., /v1/generate or /v1/stream
    client_host: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # model context
    model_id: Mapped[str] = mapped_column(String(256), index=True)
    params_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # payload
    prompt: Mapped[str] = mapped_column(Text)
    output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # metrics
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_inflog_model_created", "model_id", "created_at"),
    )


class CompletionCache(Base):
    """
    Dedup cache: (model_id, prompt_hash, params_fingerprint) -> output
    Store full prompt for observability; index on the hash.
    """
    __tablename__ = "completion_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    model_id: Mapped[str] = mapped_column(String(256), index=True)

    # full prompt (kept for debugging / analytics)
    prompt: Mapped[str] = mapped_column(Text)

    # short hash of the prompt (e.g., sha256[:32])
    prompt_hash: Mapped[str] = mapped_column(String(64), index=True)

    # hash of the generation params (your fingerprint)
    params_fingerprint: Mapped[str] = mapped_column(String(128), index=True)

    output: Mapped[str] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint(
            "model_id",
            "prompt_hash",
            "params_fingerprint",
            name="uq_completion_key",
        ),
        Index("ix_cache_model_promptfp", "model_id", "params_fingerprint"),
    )