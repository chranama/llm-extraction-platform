# tests/conftest.py
import os
import asyncio
import pytest
from datetime import datetime, timedelta

import anyio
import httpx
from asgi_lifespan import LifespanManager
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# IMPORTANT: set env vars before importing your app modules
os.environ.setdefault("ENV", "test")
os.environ.setdefault("DEBUG", "0")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/test_app.db")
os.environ.setdefault("CORS_ALLOWED_ORIGINS", "http://localhost:3000")
os.environ.setdefault("LLM_MODEL_ID", "dummy/model")

from app.main import create_app
from app.db.session import engine, async_session_maker
from app.db.models import Base, ApiKey, RoleTable


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


@pytest.fixture
async def client(app):
    async with LifespanManager(app):
        async with httpx.AsyncClient(app=app, base_url="http://testserver") as c:
            yield c


@pytest.fixture
async def api_key(db_session: AsyncSession) -> str:
    # Ensure a role exists (optional)
    role = (await db_session.execute(select(RoleTable).where(RoleTable.name == "standard"))).scalar_one_or_none()
    if not role:
        role = RoleTable(name="standard")
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)

    # Seed an active API key with generous/empty quotas
    key_value = "test_api_key_123"
    row = (await db_session.execute(select(ApiKey).where(ApiKey.key == key_value))).scalar_one_or_none()
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


@pytest.fixture
def mock_model(monkeypatch):
    """
    Replace ModelManager with a deterministic dummy for tests.
    """
    from app.services import llm as llm_mod

    class DummyModelManager:
        model_id = "dummy/model"
        _tokenizer = object()
        _model = object()
        _device = "cpu"

        def ensure_loaded(self):  # no-op
            return None

        def generate(self, prompt: str, **kwargs) -> str:
            # Simple deterministic behavior
            return f"[DUMMY COMPLETION for: {prompt}]"

        def stream(self, prompt: str, **kwargs):
            for chunk in ["[DU", "MMY ", "STREAM ", "OK]"]:
                yield chunk

    # patch in both service module and where it's used by router
    monkeypatch.setattr(llm_mod, "ModelManager", DummyModelManager, raising=True)

    # also ensure existing singleton in generate.py uses the dummy
    import app.api.generate as gen
    gen._llm = DummyModelManager()

    return DummyModelManager