"""add trace_id to extract_jobs

Revision ID: c7a9d2f4e8b1
Revises: f1d2e3c4b5a6
Create Date: 2026-03-19 18:40:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "c7a9d2f4e8b1"
down_revision = "f1d2e3c4b5a6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("extract_jobs", sa.Column("trace_id", sa.String(length=64), nullable=True))
    op.create_index(op.f("ix_extract_jobs_trace_id"), "extract_jobs", ["trace_id"], unique=False)
    op.execute("UPDATE extract_jobs SET trace_id = request_id WHERE trace_id IS NULL")


def downgrade() -> None:
    op.drop_index(op.f("ix_extract_jobs_trace_id"), table_name="extract_jobs")
    op.drop_column("extract_jobs", "trace_id")
