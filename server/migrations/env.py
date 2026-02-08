# server/migrations/env.py
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

# -------------------------------------------------------------------
# Ensure imports work even when the package isn't installed yet.
# This helps when someone runs `alembic ...` in a fresh checkout.
# -------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent            # server/migrations
_SERVER_DIR = _THIS_DIR.parent                         # server/
_SRC_DIR = _SERVER_DIR / "src"                         # server/src

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ---- Import metadata only (no side effects) ----
from llm_server.db.models import Base  # noqa: E402

# ---- Optional fallback to app settings ----
try:
    from llm_server.core.config import settings  # noqa: E402
except Exception:  # pragma: no cover
    settings = None

# Alembic Config object
config = context.config

# Logging configuration from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata for autogenerate
target_metadata = Base.metadata


# -------------------------------------------------------------------
# Database URL resolution
# -------------------------------------------------------------------
def get_url() -> str:
    """
    Migration URL resolution order:

    1) DATABASE_URL env var (preferred; compose/k8s/jobs)
    2) settings.database_url (local/dev fallback)

    Must be an async URL for online mode, e.g.:
      postgresql+asyncpg://user:pass@host:5432/db
    """
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    if settings is not None:
        return settings.database_url

    raise RuntimeError("DATABASE_URL is not set and settings could not be loaded.")


# -------------------------------------------------------------------
# Offline mode
# -------------------------------------------------------------------
def run_migrations_offline() -> None:
    """
    Offline mode does not open a DB connection; it emits SQL.

    Note: Using an async URL here is generally fine for offline generation
    because Alembic won't connect, but it must still be a valid SQLAlchemy URL.
    """
    url = get_url()

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# -------------------------------------------------------------------
# Online mode (async)
# -------------------------------------------------------------------
def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable: AsyncEngine = create_async_engine(
        get_url(),
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as async_conn:
        await async_conn.run_sync(do_run_migrations)

    await connectable.dispose()


# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())