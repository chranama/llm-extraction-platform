"""add trace_id and job_id to inference_logs

Revision ID: 6c4e2b1f9a7d
Revises: c7a9d2f4e8b1
Create Date: 2026-03-25 16:45:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "6c4e2b1f9a7d"
down_revision = "c7a9d2f4e8b1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("inference_logs", sa.Column("trace_id", sa.String(length=64), nullable=True))
    op.add_column("inference_logs", sa.Column("job_id", sa.String(length=64), nullable=True))
    op.create_index(op.f("ix_inference_logs_trace_id"), "inference_logs", ["trace_id"], unique=False)
    op.create_index(op.f("ix_inference_logs_job_id"), "inference_logs", ["job_id"], unique=False)
    op.create_index("ix_inflog_trace_created", "inference_logs", ["trace_id", "created_at"], unique=False)
    op.create_index("ix_inflog_job_created", "inference_logs", ["job_id", "created_at"], unique=False)
    op.execute("UPDATE inference_logs SET trace_id = request_id WHERE trace_id IS NULL")


def downgrade() -> None:
    op.drop_index("ix_inflog_job_created", table_name="inference_logs")
    op.drop_index("ix_inflog_trace_created", table_name="inference_logs")
    op.drop_index(op.f("ix_inference_logs_job_id"), table_name="inference_logs")
    op.drop_index(op.f("ix_inference_logs_trace_id"), table_name="inference_logs")
    op.drop_column("inference_logs", "job_id")
    op.drop_column("inference_logs", "trace_id")
