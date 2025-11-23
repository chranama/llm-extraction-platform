# src/llm_server/db/session.py
from __future__ import annotations

import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

# ----------------------------------------------------------------------
# Database configuration
# ----------------------------------------------------------------------

# Use DATABASE_URL from environment, with a sensible default.
# In tests, this is overridden to sqlite+aiosqlite:///./data/test_app.db
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/app.db")

# Async engine (works for both Postgres and SQLite async URLs)
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
)

# Session factory for creating AsyncSession instances
async_session_maker = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)

# Base class for ORM models
Base = declarative_base()


# ----------------------------------------------------------------------
# FastAPI dependencies
# ----------------------------------------------------------------------
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an AsyncSession for use as a FastAPI dependency.

    Typical usage:

        from fastapi import Depends
        from sqlalchemy.ext.asyncio import AsyncSession
        from llm_server.db.session import get_session

        @router.get("/healthz")
        async def healthz(db: AsyncSession = Depends(get_session)):
            ...

    Tests and routes can both depend on this.
    """
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


# Backwards-compatible alias for older imports & tests
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Backwards-compatible alias to get_session(), kept so older code/tests that
    import `get_async_session` still work.
    """
    async for s in get_session():
        yield s