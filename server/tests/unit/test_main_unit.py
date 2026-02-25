from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
from starlette.requests import Request
from starlette.responses import Response


@pytest.fixture()
def mmod(monkeypatch):
    try:
        import llm_contracts.config as c  # type: ignore

        if not hasattr(c, "validate_models_config"):
            monkeypatch.setattr(
                c,
                "validate_models_config",
                lambda cfg, allow_generic_deployment_key=False: SimpleNamespace(ok=True, error=None),
                raising=False,
            )
        if not hasattr(c, "validate_assessment_for_extract"):
            monkeypatch.setattr(
                c,
                "validate_assessment_for_extract",
                lambda cfg: SimpleNamespace(ok=True, error=None),
                raising=False,
            )
    except Exception:
        pkg = types.ModuleType("llm_contracts")
        cfg_mod = types.ModuleType("llm_contracts.config")
        cfg_mod.validate_models_config = (
            lambda cfg, allow_generic_deployment_key=False: SimpleNamespace(ok=True, error=None)
        )
        cfg_mod.validate_assessment_for_extract = (
            lambda cfg: SimpleNamespace(ok=True, error=None)
        )
        sys.modules["llm_contracts"] = pkg
        sys.modules["llm_contracts.config"] = cfg_mod

    mod = importlib.import_module("llm_server.main")
    return importlib.reload(mod)


def _request(path="/v1/generate", headers=None):
    hdrs = []
    for k, v in (headers or {}).items():
        hdrs.append((k.lower().encode(), str(v).encode()))
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "headers": hdrs,
        "query_string": b"",
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


def test_mode_warmup_cors_helpers(monkeypatch, mmod):
    s = SimpleNamespace(
        model_load_mode="on",
        env="dev",
        cors_allowed_origins=[" a ", "", None],
    )
    assert mmod._effective_model_load_mode(s) == "eager"
    assert mmod._model_warmup_enabled(SimpleNamespace(env="prod"), "eager") is True
    assert mmod._model_warmup_enabled(SimpleNamespace(env="dev"), "eager") is False

    monkeypatch.setenv("MODEL_WARMUP", "1")
    assert mmod._model_warmup_enabled(SimpleNamespace(env="dev"), "lazy") is True
    monkeypatch.setenv("MODEL_WARMUP_MAX_NEW_TOKENS", "bad")
    assert mmod._warmup_max_new_tokens() == 8
    assert mmod._cors_origins(s) == ["a"]


def test_validate_models_config_or_raise(monkeypatch, mmod):
    monkeypatch.setattr(
        mmod,
        "validate_models_config",
        lambda cfg, allow_generic_deployment_key=False: SimpleNamespace(ok=True, error=None),
        raising=True,
    )
    monkeypatch.setattr(
        mmod,
        "validate_assessment_for_extract",
        lambda cfg: SimpleNamespace(ok=True, error=None),
        raising=True,
    )
    mmod._validate_models_config_or_raise({"x": 1}, mode="lazy")

    monkeypatch.setattr(
        mmod,
        "validate_models_config",
        lambda cfg, allow_generic_deployment_key=False: SimpleNamespace(ok=False, error="bad"),
        raising=True,
    )
    with pytest.raises(RuntimeError):
        mmod._validate_models_config_or_raise({"x": 1}, mode="lazy")


@pytest.mark.anyio
async def test_request_context_middleware_sets_request_id_and_defaults(mmod):
    async def _app(scope, receive, send):
        return None

    mw = mmod.RequestContextMiddleware(_app)
    req = _request(headers={"x-request-id": "rid-abc"})

    async def _call_next(request):
        assert request.state.request_id == "rid-abc"
        assert request.state.route == "/v1/generate"
        assert request.state.model_id == "unknown"
        assert request.state.cached is False
        return Response("ok", status_code=200)

    resp = await mw.dispatch(req, _call_next)
    assert resp.headers["X-Request-ID"] == "rid-abc"


@pytest.mark.anyio
async def test_lifespan_lazy_mode_continues_when_llm_build_fails(monkeypatch, mmod):
    app = SimpleNamespace(state=SimpleNamespace())
    closed = {"v": False}

    async def _close(_redis):
        closed["v"] = True

    class _Loader:
        def __init__(self, state):
            self.state = state

    monkeypatch.setattr(
        mmod,
        "get_settings",
        lambda: SimpleNamespace(
            env="dev",
            debug=False,
            db_instance="db",
            redis_enabled=False,
            model_load_mode="lazy",
            service_name="svc",
            version="1.0",
            cors_allowed_origins=[],
        ),
        raising=True,
    )
    monkeypatch.setattr(
        mmod,
        "load_policy_decision_from_env",
        lambda: SimpleNamespace(
            ok=True,
            source_path=None,
            model_id=None,
            enable_extract=None,
            error=None,
        ),
        raising=True,
    )
    monkeypatch.setattr(mmod, "init_redis", lambda: None, raising=True)
    monkeypatch.setattr(
        mmod, "load_models_config", lambda: SimpleNamespace(models=[], defaults={}), raising=True
    )
    monkeypatch.setattr(
        mmod, "_validate_models_config_or_raise", lambda cfg, mode: None, raising=True
    )
    monkeypatch.setattr(
        mmod,
        "build_llm_from_settings",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=True,
    )
    monkeypatch.setattr(mmod, "RuntimeModelLoader", _Loader, raising=True)
    monkeypatch.setattr(mmod, "close_redis", _close, raising=True)

    async with mmod.lifespan(app):
        assert app.state.models_config is not None
        assert app.state.llm is None
        assert "boom" in (app.state.model_error or "")
        assert isinstance(app.state.runtime_model_loader, _Loader)

    assert closed["v"] is True


@pytest.mark.anyio
async def test_lifespan_eager_mode_raises_on_models_config_failure(monkeypatch, mmod):
    app = SimpleNamespace(state=SimpleNamespace())
    monkeypatch.setattr(
        mmod,
        "get_settings",
        lambda: SimpleNamespace(
            env="prod",
            debug=False,
            db_instance="db",
            redis_enabled=False,
            model_load_mode="eager",
            service_name="svc",
            version="1.0",
            cors_allowed_origins=[],
        ),
        raising=True,
    )
    monkeypatch.setattr(
        mmod,
        "load_policy_decision_from_env",
        lambda: SimpleNamespace(
            ok=True,
            source_path=None,
            model_id=None,
            enable_extract=None,
            error=None,
        ),
        raising=True,
    )
    monkeypatch.setattr(mmod, "init_redis", lambda: None, raising=True)
    monkeypatch.setattr(
        mmod,
        "load_models_config",
        lambda: (_ for _ in ()).throw(RuntimeError("cfg bad")),
        raising=True,
    )
    monkeypatch.setattr(mmod, "close_redis", lambda _redis: None, raising=True)

    with pytest.raises(RuntimeError):
        async with mmod.lifespan(app):
            pass
