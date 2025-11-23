# migrations/env.py
from __future__ import annotations

from logging.config import fileConfig
import asyncio
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from alembic import context

# --- Load your app settings & metadata ---
from llm_server.core.config import settings
from llm_server.db.models import Base  # <-- target_metadata comes from here

# Alembic Config object, provides access to the .ini file values
config = context.config

# If you keep logging in alembic.ini, enable it:
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Use your app’s metadata for autogenerate
target_metadata = Base.metadata

# Always take the URL from your application settings
def get_url() -> str:
    return settings.database_url  # e.g., "sqlite+aiosqlite:///./data/app.db"


# ----- Offline mode (generates SQL without DB connection) -----
def run_migrations_offline() -> None:
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,           # detect column type changes
        compare_server_default=True, # detect server_default changes
    )

    with context.begin_transaction():
        context.run_migrations()


# ----- Online (Async) mode -----
def do_run_migrations(connection: Connection) -> None:
    """
    This is run in a synchronous context but uses the connection
    that Alembic created for us. We associate our metadata so
    `--autogenerate` can compare models -> database.
    """
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


# Entrypoint Alembic calls
if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())