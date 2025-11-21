# scripts/show_db.py
import asyncio
from sqlalchemy import select

from app.db.session import async_session_maker
from app.db.models import ApiKey, RoleTable


async def show():
    async with async_session_maker() as s:
        api_keys = (await s.execute(select(ApiKey))).scalars().all()
        roles = (await s.execute(select(RoleTable))).scalars().all()

        print("API Keys:")
        for a in api_keys:
            print(f"  id={a.id}, key={a.key}, active={a.active}, role_id={a.role_id}")

        print("\nRoles:")
        for r in roles:
            print(f"  id={r.id}, name={r.name}")


if __name__ == "__main__":
    asyncio.run(show())