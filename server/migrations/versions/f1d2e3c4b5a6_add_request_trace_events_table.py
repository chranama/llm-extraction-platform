"""add request trace events table

Revision ID: f1d2e3c4b5a6
Revises: 3bf8c0c75291, 9f0f3f2b5c21
Create Date: 2026-03-14 13:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "f1d2e3c4b5a6"
down_revision = ("3bf8c0c75291", "9f0f3f2b5c21")
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "request_trace_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("event_name", sa.String(length=64), nullable=False),
        sa.Column("route", sa.String(length=64), nullable=False),
        sa.Column("stage", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("job_id", sa.String(length=64), nullable=True),
        sa.Column("model_id", sa.String(length=256), nullable=True),
        sa.Column("details_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_request_trace_events_created_at"),
        "request_trace_events",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_request_trace_events_job_id"), "request_trace_events", ["job_id"], unique=False
    )
    op.create_index(
        op.f("ix_request_trace_events_model_id"), "request_trace_events", ["model_id"], unique=False
    )
    op.create_index(
        op.f("ix_request_trace_events_request_id"),
        "request_trace_events",
        ["request_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_request_trace_events_status"), "request_trace_events", ["status"], unique=False
    )
    op.create_index(
        op.f("ix_request_trace_events_trace_id"), "request_trace_events", ["trace_id"], unique=False
    )
    op.create_index(
        "ix_trace_event_job_created", "request_trace_events", ["job_id", "created_at"], unique=False
    )
    op.create_index(
        "ix_trace_event_trace_created",
        "request_trace_events",
        ["trace_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_trace_event_trace_created", table_name="request_trace_events")
    op.drop_index("ix_trace_event_job_created", table_name="request_trace_events")
    op.drop_index(op.f("ix_request_trace_events_trace_id"), table_name="request_trace_events")
    op.drop_index(op.f("ix_request_trace_events_status"), table_name="request_trace_events")
    op.drop_index(op.f("ix_request_trace_events_request_id"), table_name="request_trace_events")
    op.drop_index(op.f("ix_request_trace_events_model_id"), table_name="request_trace_events")
    op.drop_index(op.f("ix_request_trace_events_job_id"), table_name="request_trace_events")
    op.drop_index(op.f("ix_request_trace_events_created_at"), table_name="request_trace_events")
    op.drop_table("request_trace_events")
