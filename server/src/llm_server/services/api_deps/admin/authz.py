# server/src/llm_server/services/api_deps/admin/authz.py
from __future__ import annotations

from fastapi import status
from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.core.errors import AppError
from llm_server.db.models import ApiKey
from llm_server.telemetry import queries as telem_q


async def ensure_admin(api_key: ApiKey, session: AsyncSession) -> None:
    """
    Reload ApiKey with its Role in the current async session and enforce admin role.
    """
    db_key = await telem_q.reload_key_with_role(session, api_key_id=api_key.id)
    role_name = db_key.role.name if db_key and db_key.role else None
    if role_name != "admin":
        raise AppError(code="forbidden", message="Admin privileges required", status_code=status.HTTP_403_FORBIDDEN)