"""add otel parent context to extract_jobs

Revision ID: a91b7c3d4e5f
Revises: 6c4e2b1f9a7d
Create Date: 2026-03-28 12:20:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

try:  # pragma: no cover
    from sqlalchemy.dialects.postgresql import JSONB as _JSONType
except Exception:  # pragma: no cover
    _JSONType = sa.JSON


# revision identifiers, used by Alembic.
revision = "a91b7c3d4e5f"
down_revision = "6c4e2b1f9a7d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "extract_jobs",
        sa.Column("otel_parent_context_json", _JSONType, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("extract_jobs", "otel_parent_context_json")
