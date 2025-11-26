# scripts/list_api_keys.py
import asyncio
from sqlalchemy import select
from llm_server.db.session import async_session_maker
from llm_server.db.models import ApiKey, RoleTable

async def main():
    async with async_session_maker() as session:
        result = await session.execute(
            select(ApiKey, RoleTable.name)
            .join(RoleTable, ApiKey.role_id == RoleTable.id, isouter=True)
        )
        rows = result.all()

        if not rows:
            print("No API keys found.")
            return

        print("API Keys:")
        for key, role in rows:
            print("-" * 60)
            print(f"Key:            {key.key}")
            print(f"Label:          {key.label}")
            print(f"Active:         {key.active}")
            print(f"Role:           {role}")
            print(f"Quota monthly:  {key.quota_monthly}")
            print(f"Quota used:     {key.quota_used}")

asyncio.run(main())