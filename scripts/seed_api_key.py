# scripts/seed_api_key.py
from __future__ import annotations

import argparse
import asyncio
import secrets
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.db.session import async_session_maker, engine
from llm_server.db.models import ApiKey, RoleTable, Role, Base


async def init_db() -> None:
    """
    Minimal DB initializer for scripts: ensures all tables exist.
    Mirrors what tests do in conftest.py (create_all on the engine).
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def upsert_role(session: AsyncSession, role_name: Optional[str]) -> Optional[RoleTable]:
    """Return RoleTable row by name; create it if it doesn't exist. If None, return None."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed a new API key.")

    parser.add_argument(
        "--role",
        type=str,
        default=Role.standard.value,
        choices=[r.value for r in Role],
        help="Role to assign to the key (default: standard)",
    )

    parser.add_argument(
        "--quota",
        type=int,
        default=None,
        help="Monthly quota to enforce (omit for unlimited)",
    )

    parser.add_argument(
        "--unlimited",
        action="store_true",
        help="Force unlimited usage (sets quota_monthly = NULL)",
    )

    parser.add_argument(
        "--label",
        type=str,
        default="dev",
        help="Optional label for the API key",
    )

    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # Ensure DB and tables exist
    await init_db()

    # Generate a 64-hex-char API key
    key = secrets.token_hex(32)

    # Determine quota logic
    if args.unlimited:
        quota_monthly = None
    else:
        quota_monthly = args.quota

    async with async_session_maker() as s:
        role = await upsert_role(s, args.role)

        api_key = ApiKey(
            key=key,
            label=args.label,
            active=True,
            role_id=role.id if role else None,
            quota_monthly=quota_monthly,
            quota_used=0,
        )

        s.add(api_key)
        await s.commit()

    print("\n✅ API key seeded successfully\n")
    print("Key:", key)
    print("Role:", args.role)
    if quota_monthly is None:
        print("Quota: UNLIMITED")
    else:
        print(f"Quota: {quota_monthly} tokens / month")
    print()


if __name__ == "__main__":
    asyncio.run(main())