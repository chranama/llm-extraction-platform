# tests/conftest.py
import os
import pytest
from datetime import datetime, timedelta

import anyio
import httpx
from httpx import ASGITransport
from asgi_lifespan import LifespanManager
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# IMPORTANT: set env vars before importing your app modules
os.environ.setdefault("ENV", "test")
os.environ.setdefault("DEBUG", "0")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/test_app.db")
os.environ.setdefault("CORS_ALLOWED_ORIGINS", "http://localhost:3000")
os.environ.setdefault("LLM_MODEL_ID", "dummy/model")
# we don't actually call a real external LLM in tests, but settings will still have a URL
os.environ.setdefault("LLM_SERVICE_URL", "http://dummy-llm")

from llm_server.main import create_app
from llm_server.db.session import engine, async_session_maker
from llm_server.db.models import Base, ApiKey, RoleTable


@pytest.fixture(scope="session")
def anyio_backend():
    # make pytest-anyio use asyncio
    return "asyncio"


@pytest.fixture(scope="session")
async def prepare_db():
    # Create all tables for tests; drop at the end
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def db_session(prepare_db) -> AsyncSession:
    async with async_session_maker() as s:
        yield s


@pytest.fixture
async def app():
    return create_app()


# ---------------------------------------------------------------------------
# Dummy LLM used by /v1/generate and /v1/stream
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_model():
    """
    In-process dummy LLM implementation used in tests.

    This is what the FastAPI dependency override will inject into the
    /v1/generate and /v1/stream endpoints instead of the real ModelManager.
    """

    class DummyModelManager:
        # Attributes that the API layer may read
        model_id = "dummy/model"

        def ensure_loaded(self):
            # No-op in tests
            return None

        def generate(
            self,
            prompt: str,
            max_new_tokens: int | None = None,
            temperature: float | None = None,
            top_p: float | None = None,
            top_k: int | None = None,
            stop: list[str] | None = None,
        ) -> str:
            # Simple deterministic behavior for assertions
            return f"[DUMMY COMPLETION for: {prompt}]"

        def stream(
            self,
            prompt: str,
            max_new_tokens: int | None = None,
            temperature: float | None = None,
            top_p: float | None = None,
            top_k: int | None = None,
            stop: list[str] | None = None,
        ):
            # Yield in one chunk; tests can assert on the full string
            yield f"[DUMMY COMPLETION for: {prompt}]"

    return DummyModelManager()


@pytest.fixture
async def client(app, mock_model):
    """
    Async test client wired to the FastAPI app using httpx's ASGITransport.
    Compatible with httpx>=0.27 which removed the `app=` parameter.

    Also overrides the `get_llm` dependency so that endpoints use `mock_model`
    instead of the real HF-backed ModelManager.
    """
    # Import here to avoid any circular import surprises
    from llm_server.api.generate import get_llm as generate_get_llm

    # Dependency override: whenever FastAPI sees Depends(get_llm),
    # it will call this lambda instead and get `mock_model`.
    app.dependency_overrides[generate_get_llm] = lambda: mock_model

    async with LifespanManager(app):
        transport = ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as c:
            yield c


@pytest.fixture
async def api_key(db_session: AsyncSession) -> str:
    # Ensure a role exists
    role = (
        await db_session.execute(
            select(RoleTable).where(RoleTable.name == "standard")
        )
    ).scalar_one_or_none()
    if not role:
        role = RoleTable(name="standard")
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

    # Seed an active API key with generous/empty quotas
    key_value = "test_api_key_123"
    row = (
        await db_session.execute(
            select(ApiKey).where(ApiKey.key == key_value)
        )
    ).scalar_one_or_none()
    if not row:
        row = ApiKey(
            key=key_value,
            label="tests",
            active=True,
            quota_monthly=None,  # unlimited in tests
            quota_used=0,
            quota_reset_at=None,
            role_id=role.id,
        )
        db_session.add(row)
        await db_session.commit()

    return key_value


# ---------------------------------------------------------------------------
# Make /readyz always see a "loaded" LLM (no heavy model loading in tests)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _readyz_llm_always_ok(monkeypatch):
    """
    Patch the health endpoint so that ensure_loaded() never fails or
    tries to load a real model.

    We override health.get_llm() to return a cheap dummy object.
    """
    from llm_server.api import health as health_mod

    class _ReadyDummyLLM:
        def ensure_loaded(self):
            # no-op: always succeeds
            return None

    # Ensure /readyz uses this dummy instead of a real LLM
    monkeypatch.setattr(
        health_mod,
        "get_llm",
        lambda: _ReadyDummyLLM(),
        raising=False,
    )
    yield