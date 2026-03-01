from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import cli.commands.dev as dev_mod
from cli.errors import CLIError
from cli.types import GlobalConfig


def _cfg(tmp_path: Path) -> GlobalConfig:
    return GlobalConfig(
        repo_root=tmp_path,
        env_override_file=None,
        compose_yml=tmp_path / "deploy/compose/docker-compose.yml",
        tools_dir=tmp_path / "tools",
        compose_doctor=tmp_path / "tools/compose/compose_doctor.sh",
        server_dir=tmp_path / "server",
        models_yaml=tmp_path / "config/models.yaml",
        project_name="proj",
        api_port="8000",
        ui_port="5173",
        pgadmin_port="5050",
        prom_port="9090",
        grafana_port="3000",
        prom_host_port="9091",
        pg_user="llm",
        pg_db="llm",
    )


def test_migrate_falls_back_to_server_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _compose_exec(_ctx, service, cmd, tty, verbose):
        calls.append(service)
        if service == "server":
            raise RuntimeError("not running")
        return None

    monkeypatch.setattr(dev_mod, "compose_exec", _compose_exec)
    dev_mod._migrate("CTX", verbose=False)
    assert calls == ["server", "server_gpu"]


def test_migrate_raises_when_no_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dev_mod, "compose_exec", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(CLIError, match="No running server/server_gpu container found"):
        dev_mod._migrate("CTX", verbose=False)


def test_run_doctor_checks_file_and_executable(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(CLIError, match="compose doctor not found"):
        dev_mod._run_doctor(cfg, verbose=False)

    cfg2 = GlobalConfig(**{**cfg.__dict__, "compose_doctor": tmp_path / "doctor.sh"})
    cfg2.compose_doctor.write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")
    with pytest.raises(CLIError, match="not executable"):
        dev_mod._run_doctor(cfg2, verbose=False)


def test_handle_dev_cpu_invokes_compose_and_migrate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}
    monkeypatch.setattr(dev_mod, "ensure_bins", lambda *_bins: None)
    monkeypatch.setattr(dev_mod, "build_compose_context", lambda *_args, **_kwargs: "CTX")
    monkeypatch.setattr(dev_mod, "compose_config", lambda ctx, profiles, verbose: called.update({"cfg": (ctx, profiles, verbose)}))
    monkeypatch.setattr(
        dev_mod,
        "compose_up",
        lambda ctx, profiles, detach, build, remove_orphans, verbose: called.update(
            {"up": (ctx, profiles, detach, build, remove_orphans, verbose)}
        ),
    )
    monkeypatch.setattr(dev_mod, "_migrate", lambda ctx, verbose: called.update({"migrate": (ctx, verbose)}))

    args = argparse.Namespace(dev_cmd="dev-cpu", verbose=False, defaults_profile=None, defaults_yaml=None)
    rc = dev_mod._handle(_cfg(tmp_path), args)
    assert rc == 0
    assert called["cfg"] == ("CTX", ["infra", "server"], False)
    assert called["migrate"] == ("CTX", False)


def test_handle_unknown_dev_command_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dev_mod, "ensure_bins", lambda *_bins: None)
    monkeypatch.setattr(dev_mod, "build_compose_context", lambda *_args, **_kwargs: "CTX")
    args = argparse.Namespace(dev_cmd="not-real", verbose=False, defaults_profile=None, defaults_yaml=None)
    with pytest.raises(CLIError, match="Unknown dev command"):
        dev_mod._handle(_cfg(tmp_path), args)
