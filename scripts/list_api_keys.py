# scripts/list_api_keys.py
import asyncio
from sqlalchemy import select

from llm_server.db.session import async_session_maker
from llm_server.db.models import ApiKey


async def main():
    async with async_session_maker() as s:
        rows = (await s.execute(select(ApiKey))).scalars().all()
        for r in rows:
            # Only print the key (no labels, ids, etc.)
            print(r.key)


if __name__ == "__main__":
    asyncio.run(main())