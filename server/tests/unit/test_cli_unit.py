from __future__ import annotations

from llm_server import cli


def test_env_flag_truthy_values(monkeypatch):
    monkeypatch.setenv("FLAG_A", "true")
    monkeypatch.setenv("FLAG_B", "ON")
    monkeypatch.setenv("FLAG_C", "0")

    assert cli._env_flag("FLAG_A") is True
    assert cli._env_flag("FLAG_B") is True
    assert cli._env_flag("FLAG_C") is False
    assert cli._env_flag("FLAG_MISSING", "yes") is True


def test_serve_dev_mode_uses_reload_from_env(monkeypatch):
    calls: list[dict] = []

    def _run(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(cli.uvicorn, "run", _run, raising=True)
    monkeypatch.setenv("ENV", "dev")
    monkeypatch.setenv("UVICORN_RELOAD", "1")
    monkeypatch.delenv("APP_PROFILE", raising=False)

    cli.serve(host="127.0.0.1", port=9001, workers=8, reload_=None, proxy_headers=False, profile="host")

    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    assert kwargs["factory"] is True
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 9001
    assert kwargs["reload"] is True
    assert kwargs["workers"] == 1
    assert kwargs["proxy_headers"] is False
    assert cli.os.environ["APP_PROFILE"] == "host"


def test_serve_prod_mode_uses_worker_count(monkeypatch):
    calls: list[dict] = []

    def _run(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(cli.uvicorn, "run", _run, raising=True)
    monkeypatch.setenv("ENV", "prod")

    cli.serve(host="0.0.0.0", port=8000, workers=3, reload_=True, proxy_headers=True, profile=None)

    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    assert kwargs["factory"] is True
    assert kwargs["workers"] == 3
    assert "reload" not in kwargs
    assert kwargs["proxy_headers"] is True
