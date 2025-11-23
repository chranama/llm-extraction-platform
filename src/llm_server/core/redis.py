from __future__ import annotations
from typing import Optional
from redis.asyncio import Redis, from_url

from llm_server.core.config import settings

async def init_redis() -> Optional[Redis]:
    if not settings.redis_enabled or not settings.redis_url:
        return None
    # decode_responses=True -> str in/out
    return from_url(settings.redis_url, decode_responses=True)

async def close_redis(client: Optional[Redis]) -> None:
    if client is not None:
        await client.aclose()