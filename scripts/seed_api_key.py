# scripts/seed_api_key.py
from __future__ import annotations

import asyncio
import secrets
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import init_db, async_session_maker
from app.db.models import ApiKey, RoleTable, Role


async def upsert_role(session: AsyncSession, role_name: Optional[str]) -> Optional[RoleTable]:
    """Return RoleTable row by name; create it if it doesn't exist. If role_name is None, return None."""
    if not role_name:
        return None
    res = await session.execute(select(RoleTable).where(RoleTable.name == role_name))
    role = res.scalar_one_or_none()
    if role:
        return role
    role = RoleTable(name=role_name)
    session.add(role)
    await session.flush()  # populate role.id
    return role


async def main() -> None:
    # Ensure DB and tables exist
    await init_db()

    # Choose which role to assign to this key
    role_name = Role.standard.value  # "standard" | "admin" | "free"

    # Generate a 64-hex-char API key
    key = secrets.token_hex(32)

    async with async_session_maker() as s:
        role = await upsert_role(s, role_name)
        s.add(
            ApiKey(
                key=key,
                label="dev",
                active=True,
                role_id=role.id if role else None,
                # optionally set quotas here:
                # quota_monthly=10000,
                # quota_used=0,
            )
        )
        await s.commit()

    print("Seeded API key:", key)
    print("Role:", role_name)


if __name__ == "__main__":
    asyncio.run(main())