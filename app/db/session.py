# app/db/session.py
from __future__ import annotations

from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.models import Base


# Create the async engine from the configured URL
engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
)

# Async session factory we can use in dependencies *and* scripts
async_session_maker: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)



# FastAPI dependency (yield a session per request)
async def get_session():
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()