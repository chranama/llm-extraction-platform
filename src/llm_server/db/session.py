# src/llm_server/db/session.py
from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from llm_server.core.config import settings

# ----------------------------------------------------------------------
# Database configuration (Phase: settings-driven, single source of truth)
# ----------------------------------------------------------------------

DATABASE_URL: str = settings.database_url

# Create one async engine per process
engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    future=True,
)

# Session factory for creating AsyncSession instances
async_session_maker = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)

# Base class for ORM models (kept for backwards compatibility)
Base = declarative_base()


# ----------------------------------------------------------------------
# FastAPI dependency
# ----------------------------------------------------------------------

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an AsyncSession for FastAPI dependencies.

    Prefer injecting this in routes/services:
        session: AsyncSession = Depends(get_session)
    """
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            # In SQLAlchemy 2.x, the context manager handles closing,
            # but we keep this explicit for clarity and backwards safety.
            await session.close()


# Backwards-compatible alias for older imports & tests
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Backwards-compatible alias to get_session().
    """
    async for s in get_session():
        yield s