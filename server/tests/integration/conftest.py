# tests/integration/conftest.py
from __future__ import annotations

import contextlib
import os
import sys
import types
from pathlib import Path
from typing import Any

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
# Config routing (root-level config/)
# ============================================================
# ✅ Use profiled config/server.yaml now (APP_PROFILE selects overlay)
APP_YAML = "config/server.yaml"
os.environ["APP_ROOT"] = str(REPO_ROOT)
os.environ["APP_CONFIG_PATH"] = APP_YAML

# ✅ Select profiles for test runs
os.environ["APP_PROFILE"] = "test"
# models.yaml uses MODELS_PROFILE (preferred) or falls back to APP_PROFILE
os.environ["MODELS_PROFILE"] = "test"


def _is_generate_only_module(request: pytest.FixtureRequest) -> bool:
    nid = (getattr(request.node, "nodeid", "") or "").lower()
    return "tests/integration/test_generate_integration.py" in nid


def _ensure_contracts_config_module() -> None:
    """
    Make llm_contracts.config importable in test environments where
    llm_contracts exposes validators via llm_contracts.config.models_config.
    """

    class _ValidationOK:
        ok = True
        error = ""

    def _always_ok(*args, **kwargs):
        return _ValidationOK()

    try:
        import llm_contracts.config as cfg_mod  # type: ignore

        cfg_mod.validate_models_config = _always_ok  # type: ignore[attr-defined]
        cfg_mod.validate_assessment_for_extract = _always_ok  # type: ignore[attr-defined]
        return
    except Exception:
        pass

    shim = types.ModuleType("llm_contracts.config")
    shim.validate_models_config = _always_ok  # type: ignore[attr-defined]
    shim.validate_assessment_for_extract = _always_ok  # type: ignore[attr-defined]
    sys.modules["llm_contracts.config"] = shim


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
    assert cfg.exists(), f"Missing config: {cfg}"


# ============================================================
# Per-test env isolation
# ============================================================
@pytest.fixture(autouse=True)
def _integration_env_defaults(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
):
    # Ensure config + profile selection for every test
    monkeypatch.setenv("APP_ROOT", str(REPO_ROOT))
    monkeypatch.setenv("APP_CONFIG_PATH", APP_YAML)
    monkeypatch.setenv("APP_PROFILE", "test")
    monkeypatch.setenv("MODELS_PROFILE", "test")
    monkeypatch.setenv("SCHEMAS_DIR", str(REPO_ROOT / "schemas" / "model_output"))
    monkeypatch.setenv("SCHEMAS_ROOT", str(REPO_ROOT / "schemas"))
    monkeypatch.delenv("POLICY_DECISION_PATH", raising=False)

    # NOTE:
    # Do NOT hard-set MODEL_LOAD_MODE / REQUIRE_MODEL_READY / TOKEN_COUNTING here.
    # Those must come from config/server.yaml profile:test.
    #
    # You *can* set DEBUG/REDIS_ENABLED defaults, but avoid overriding the YAML
    # unless tests explicitly need it. We'll keep DEBUG low-noise.
    monkeypatch.setenv("DEBUG", "0")
    monkeypatch.setenv("REDIS_ENABLED", "0")

    # Capabilities: extract disabled ONLY for generate-only module
    # (this is a test-specific behavior toggle, and it's ok to override)
    monkeypatch.setenv("ENABLE_GENERATE", "1")
    if _is_generate_only_module(request):
        monkeypatch.setenv("ENABLE_EXTRACT", "0")
    else:
        monkeypatch.setenv("ENABLE_EXTRACT", "1")


# ============================================================
# Assert actual settings object is test config
# ============================================================
@pytest.fixture(autouse=True)
def _assert_test_config_loaded_per_test():
    from llm_server.core.config import get_settings

    get_settings.cache_clear()
    s = get_settings()

    assert s.env == "test", f"Expected env=test, got {s.env}"
    assert (
        "test" in s.service_name.lower()
    ), f"service_name not test-like: {s.service_name}"


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
    return async_sessionmaker(
        bind=test_engine, expire_on_commit=False, class_=AsyncSession
    )


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
# Force-load FakeLLM even if model mode blocks readiness checks
# ============================================================
def _force_llm_loaded(fake: Any) -> None:
    """
    Some readiness gates may block /v1/generate or /v1/extract if the service thinks
    the model isn't loaded. Tests always provide FakeLLM; force it marked as loaded
    for whichever registry pattern your code uses.
    """
    # 1) service-level registry pattern
    try:
        import llm_server.services.llm_runtime.llm_build as llm_svc

        for attr in (
            "LLM",
            "_LLM",
            "llm",
            "_llm",
            "MODEL",
            "_MODEL",
            "model",
            "_model",
        ):
            if hasattr(llm_svc, attr):
                try:
                    setattr(llm_svc, attr, fake)
                    return
                except Exception:
                    pass

        for flag in (
            "LOADED",
            "_LOADED",
            "loaded",
            "_loaded",
            "MODEL_LOADED",
            "_MODEL_LOADED",
        ):
            if hasattr(llm_svc, flag):
                try:
                    setattr(llm_svc, flag, True)
                except Exception:
                    pass
    except Exception:
        pass

    return


# ============================================================
# App (lifespan will be handled by client fixture)
# ============================================================
@pytest.fixture
def app(
    monkeypatch,
    llm_outputs,
    llm_sleep_s,
    patch_app_db_engine,
    request: pytest.FixtureRequest,
):
    """
    App wired with FakeLLM. DB is already pointed at test_engine via patch_app_db_engine.

    The test profile should select:
      - server.yaml: profile=test
      - models.yaml: profile=test (fake model)
    """
    from llm_server.core.config import get_settings

    get_settings.cache_clear()
    _ensure_contracts_config_module()

    from fakes import FakeLLM
    import llm_server.main as main
    import llm_server.services.llm_runtime.llm_build as llm_svc
    from llm_server.services.api_deps.core.auth import clear_rate_limit_state
    from llm_server.services.llm_runtime.model_state import ModelStateStore

    fake = FakeLLM(outputs=list(llm_outputs), sleep_s=float(llm_sleep_s))

    # Ensure every builder returns our fake.
    monkeypatch.setattr(main, "build_llm_from_settings", lambda: fake, raising=True)
    monkeypatch.setattr(llm_svc, "build_llm_from_settings", lambda: fake, raising=True)
    clear_rate_limit_state()

    app = main.create_app()

    _force_llm_loaded(fake)
    ms = ModelStateStore(app.state)
    ms.set_loaded_model_id(getattr(fake, "model_id", None) or "fake")
    ms.set_model_loaded(True)

    return app


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
        _ensure_contracts_config_module()

        import llm_server.main as main
        import llm_server.services.llm_runtime.llm_build as llm_svc
        from llm_server.services.api_deps.core.auth import clear_rate_limit_state
        from llm_server.services.llm_runtime.model_state import ModelStateStore

        fake = FakeLLM(outputs=list(llm_outputs), sleep_s=float(llm_sleep_s))

        monkeypatch.setattr(main, "build_llm_from_settings", lambda: fake, raising=True)
        monkeypatch.setattr(
            llm_svc, "build_llm_from_settings", lambda: fake, raising=True
        )
        clear_rate_limit_state()

        app = main.create_app()
        _force_llm_loaded(fake)
        ms = ModelStateStore(app.state)
        ms.set_loaded_model_id(getattr(fake, "model_id", None) or "fake")
        ms.set_model_loaded(True)

        @contextlib.asynccontextmanager
        async def _ctx():
            async with LifespanManager(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as c:
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


@pytest.fixture(autouse=True)
def _debug_settings_trace():
    from llm_server.core.config import get_settings

    get_settings.cache_clear()
    s = get_settings()

    print(
        "\n[debug] APP_PROFILE=",
        os.getenv("APP_PROFILE"),
        "| MODELS_PROFILE=",
        os.getenv("MODELS_PROFILE"),
        "| settings.env=",
        getattr(s, "env", None),
        "| settings.model_load_mode=",
        getattr(s, "model_load_mode", None),
        "| ENABLE_EXTRACT=",
        os.getenv("ENABLE_EXTRACT"),
        "| ENABLE_GENERATE=",
        os.getenv("ENABLE_GENERATE"),
    )
