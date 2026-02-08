# server/src/llm_server/db/models.py
from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Any, Dict, Optional

import sqlalchemy as sa
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from llm_server.db.session import Base

# Prefer JSONB on Postgres; fallback to JSON elsewhere.
try:  # pragma: no cover
    from sqlalchemy.dialects.postgresql import JSONB as _JSONType
except Exception:  # pragma: no cover
    from sqlalchemy import JSON as _JSONType  # type: ignore[assignment]


# Optional enum for use in application code (not enforced by DB)
class Role(str, enum.Enum):
    admin = "admin"
    standard = "standard"
    free = "free"


def utc_now() -> datetime:
    """Timezone-aware UTC timestamp (replacement for datetime.utcnow)."""
    return datetime.now(UTC)


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

    # secret
    key: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)

    # metadata (admin UI / labeling)
    name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # Back-compat if older code/data used "label" (optional)
    label: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=sa.text("true"),
        nullable=False,
    )

    # Prefer disabled_at so admin can display "disabled" without deleting records.
    disabled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Quotas (optional)
    quota_monthly: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # NULL = unlimited
    quota_used: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default=sa.text("0"),
        nullable=False,
    )
    quota_reset_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Link each API key to an optional role row
    role_id: Mapped[Optional[int]] = mapped_column(ForeignKey("roles.id"), nullable=True)

    # async-friendly default: selectin to avoid surprise lazy loads
    role: Mapped[Optional["RoleTable"]] = relationship("RoleTable", lazy="selectin")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=func.now(),
        nullable=False,
        index=True,
    )


class InferenceLog(Base):
    """
    One row per request attempt (success OR failure).

    Notes:
      - Store status_code so telemetry can compute error rates deterministically.
      - error_code/error_stage are stable classification signals (safe to expose internally).
      - prompt/output are kept for admin observability; for failures, output can be null.
    """

    __tablename__ = "inference_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # --------------------
    # request context
    # --------------------
    api_key: Mapped[Optional[str]] = mapped_column(String(128), index=True, nullable=True)
    request_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    route: Mapped[str] = mapped_column(String(64), nullable=False)
    client_host: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # HTTP/result classification
    status_code: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=200,
        server_default=sa.text("200"),
        index=True,
    )

    # IMPORTANT: keep consistent with DB: NOT NULL DEFAULT false
    cached: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=sa.text("false"),
    )

    # Stable error signals (best-effort)
    error_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    error_stage: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)

    # --------------------
    # model context
    # --------------------
    model_id: Mapped[str] = mapped_column(String(256), index=True, nullable=False)
    params_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(_JSONType, nullable=True)

    # --------------------
    # payload (admin observability)
    # --------------------
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # --------------------
    # metrics
    # --------------------
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        # Common SLO window query: per-model time series
        Index("ix_inflog_model_created", "model_id", "created_at"),
        # Common SLO window query: per-route time series
        Index("ix_inflog_route_created", "route", "created_at"),
        # Common SLO window query: filter failures
        Index("ix_inflog_status_created", "status_code", "created_at"),
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

    model_id: Mapped[str] = mapped_column(String(256), index=True, nullable=False)

    # full prompt (kept for debugging / analytics)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)

    # short hash of the prompt (e.g., sha256[:32])
    prompt_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)

    # hash of the generation params (your fingerprint)
    params_fingerprint: Mapped[str] = mapped_column(String(128), index=True, nullable=False)

    output: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "model_id",
            "prompt_hash",
            "params_fingerprint",
            name="uq_completion_key",
        ),
        Index("ix_cache_model_promptfp", "model_id", "params_fingerprint"),
    )