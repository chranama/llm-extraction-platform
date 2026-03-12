"""add extract_jobs table

Revision ID: 9f0f3f2b5c21
Revises: bdd9204b32d2
Create Date: 2026-03-12 12:30:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

try:
    from sqlalchemy.dialects.postgresql import JSONB as _JSONType
except Exception:  # pragma: no cover
    from sqlalchemy import JSON as _JSONType  # type: ignore[assignment]


# revision identifiers, used by Alembic.
revision = "9f0f3f2b5c21"
down_revision = "3bf8c0c75291"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "extract_jobs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("api_key", sa.String(length=128), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("schema_id", sa.String(length=128), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("requested_model_id", sa.String(length=256), nullable=True),
        sa.Column("resolved_model_id", sa.String(length=256), nullable=True),
        sa.Column("max_new_tokens", sa.Integer(), nullable=True),
        sa.Column("temperature", sa.Float(), nullable=True),
        sa.Column("cache", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("repair", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result_json", _JSONType(), nullable=True),
        sa.Column("cached", sa.Boolean(), nullable=True),
        sa.Column("repair_attempted", sa.Boolean(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("error_stage", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_extract_jobs_api_key"), "extract_jobs", ["api_key"], unique=False)
    op.create_index(op.f("ix_extract_jobs_created_at"), "extract_jobs", ["created_at"], unique=False)
    op.create_index(op.f("ix_extract_jobs_request_id"), "extract_jobs", ["request_id"], unique=False)
    op.create_index(op.f("ix_extract_jobs_resolved_model_id"), "extract_jobs", ["resolved_model_id"], unique=False)
    op.create_index(op.f("ix_extract_jobs_status"), "extract_jobs", ["status"], unique=False)
    op.create_index("ix_extract_jobs_api_key_created", "extract_jobs", ["api_key", "created_at"], unique=False)
    op.create_index("ix_extract_jobs_status_created", "extract_jobs", ["status", "created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_extract_jobs_status_created", table_name="extract_jobs")
    op.drop_index("ix_extract_jobs_api_key_created", table_name="extract_jobs")
    op.drop_index(op.f("ix_extract_jobs_status"), table_name="extract_jobs")
    op.drop_index(op.f("ix_extract_jobs_resolved_model_id"), table_name="extract_jobs")
    op.drop_index(op.f("ix_extract_jobs_request_id"), table_name="extract_jobs")
    op.drop_index(op.f("ix_extract_jobs_created_at"), table_name="extract_jobs")
    op.drop_index(op.f("ix_extract_jobs_api_key"), table_name="extract_jobs")
    op.drop_table("extract_jobs")
