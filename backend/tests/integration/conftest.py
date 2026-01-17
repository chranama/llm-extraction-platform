# tests/integration/conftest.py
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

# ============================================================
# Paths
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parents[1]

if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

# ============================================================
# Config routing
# ============================================================
APP_TEST_YAML = "config/app.test.yaml"
os.environ.setdefault("APP_ROOT", str(REPO_ROOT))
os.environ.setdefault("APP_CONFIG_PATH", APP_TEST_YAML)

# ============================================================
# AnyIO backend
# ============================================================
@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"

# ============================================================
# Assert config file exists
# ============================================================
@pytest.fixture(scope="session", autouse=True)
def _assert_test_config_file_exists():
    root = Path(os.environ["APP_ROOT"])
    cfg = (root / os.environ["APP_CONFIG_PATH"]).resolve()
    assert cfg.exists(), f"Missing test config: {cfg}"

# ============================================================
# Per-test env isolation
# ============================================================
@pytest.fixture(autouse=True)
def _integration_env_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APP_ROOT", str(REPO_ROOT))
    monkeypatch.setenv("APP_CONFIG_PATH", APP_TEST_YAML)

    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("DEBUG", "0")
    monkeypatch.setenv("REDIS_ENABLED", "0")

    monkeypatch.setenv("MODEL_LOAD_MODE", "lazy")
    monkeypatch.setenv("REQUIRE_MODEL_READY", "0")
    monkeypatch.setenv("TOKEN_COUNTING", "0")

# ============================================================
# Assert actual settings object is test config
# ============================================================
@pytest.fixture(autouse=True)
def _assert_test_config_loaded_per_test():
    from llm_server.core.config import get_settings

    get_settings.cache_clear()
    s = get_settings()

    assert s.env == "test", f"Expected env=test, got {s.env}"
    assert "test" in s.service_name.lower(), f"service_name not test-like: {s.service_name}"

# ============================================================
# DB helpers
# ============================================================
def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL must be set for integration tests. "
            "Example: postgresql+asyncpg://llm:llm@127.0.0.1:5433/llm"
        )
    return url

# ============================================================
# Engine / Session
# ============================================================
@pytest.fixture
async def test_engine():
    engine = create_async_engine(
        _database_url(),
        echo=False,
        future=True,
        poolclass=NullPool,
        pool_pre_ping=True,
    )
    try:
        yield engine
    finally:
        await engine.dispose()

@pytest.fixture
def test_sessionmaker(test_engine):
    return async_sessionmaker(bind=test_engine, expire_on_commit=False, class_=AsyncSession)

# ============================================================
# Canonical DB injection + teardown reset
# ============================================================
@pytest.fixture(autouse=True)
async def patch_app_db_engine(test_engine, test_sessionmaker):
    """
    Canonical test hook: point llm_server.db.session at the per-test engine,
    then RESET it on teardown so no module-level state leaks across tests.
    """
    import llm_server.db.session as db_session

    db_session.set_engine_for_tests(test_engine, test_sessionmaker)
    try:
        yield
    finally:
        # Important: reset module globals after each test
        await db_session.dispose_engine()

# ============================================================
# Schema lifecycle
# ============================================================
@pytest.fixture(autouse=True)
async def create_schema(test_engine, patch_app_db_engine):
    from llm_server.db.session import Base
    from llm_server.db import models  # noqa: F401

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    try:
        yield
    finally:
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

# ============================================================
# Fake LLM controls
# ============================================================
@pytest.fixture
def llm_outputs():
    return ["ok"]

@pytest.fixture
def llm_sleep_s():
    return 0.0

# ============================================================
# App (lifespan will be handled by client fixture)
# ============================================================
@pytest.fixture
def app(monkeypatch, llm_outputs, llm_sleep_s, patch_app_db_engine):
    """
    App wired with FakeLLM. DB is already pointed at test_engine via patch_app_db_engine.
    """
    from llm_server.core.config import get_settings

    get_settings.cache_clear()

    from fakes import FakeLLM
    import llm_server.api.deps as deps
    import llm_server.main as main
    import llm_server.services.llm as llm_svc

    fake = FakeLLM(outputs=list(llm_outputs), sleep_s=float(llm_sleep_s))

    monkeypatch.setattr(deps, "build_llm_from_settings", lambda: fake, raising=True)
    deps._RL.clear()

    monkeypatch.setattr(main, "build_llm_from_settings", lambda: fake, raising=True)
    monkeypatch.setattr(llm_svc, "build_llm_from_settings", lambda: fake, raising=True)

    return main.create_app()

@pytest.fixture
async def client(app):
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            setattr(c, "app", app)
            yield c

# ============================================================
# App client factory (lifespan aware)
# ============================================================
@pytest.fixture
def app_client(monkeypatch, llm_outputs, llm_sleep_s, test_engine, test_sessionmaker):
    """
    Factory for scenarios where you want a fresh app/client pair per call.

    DB wiring is handled by the autouse patch_app_db_engine fixture.
    """
    from fakes import FakeLLM

    async def _make():
        from llm_server.core.config import get_settings

        get_settings.cache_clear()

        import llm_server.api.deps as deps
        import llm_server.main as main
        import llm_server.services.llm as llm_svc

        fake = FakeLLM(outputs=list(llm_outputs), sleep_s=float(llm_sleep_s))

        monkeypatch.setattr(deps, "build_llm_from_settings", lambda: fake, raising=True)
        deps._RL.clear()
        monkeypatch.setattr(main, "build_llm_from_settings", lambda: fake, raising=True)
        monkeypatch.setattr(llm_svc, "build_llm_from_settings", lambda: fake, raising=True)

        app = main.create_app()

        @contextlib.asynccontextmanager
        async def _ctx():
            async with LifespanManager(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                    setattr(c, "app", app)
                    yield c

        return _ctx()

    return _make

# ============================================================
# API key helpers
# ============================================================
@pytest.fixture
async def api_key(test_sessionmaker):
    from llm_server.db.models import ApiKey
    import uuid

    key = f"test_{uuid.uuid4().hex}"
    async with test_sessionmaker() as session:
        session.add(ApiKey(key=key, active=True, quota_monthly=None, quota_used=0))
        await session.commit()
    return key

@pytest.fixture
async def auth_headers(api_key):
    return {"X-API-Key": api_key}