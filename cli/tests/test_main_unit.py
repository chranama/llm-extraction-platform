from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import cli.main as main_mod
from cli.errors import CLIError


def test_build_global_config_resolves_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(main_mod, "find_repo_root", lambda *_args, **_kwargs: repo_root)

    args = argparse.Namespace(
        env_override_file="envs/local.env",
        project_name="proj",
        compose_yml="deploy/compose/docker-compose.yml",
        tools_dir="tools",
        compose_doctor="tools/compose/compose_doctor.sh",
        server_dir="server",
        models_yaml="config/models.yaml",
        api_port="18000",
        ui_port="15173",
        pgadmin_port="15050",
        prom_port="19090",
        grafana_port="13000",
        prom_host_port="19091",
        pg_user="llm",
        pg_db="llm",
    )

    cfg = main_mod._build_global_config(args)
    assert cfg.repo_root == repo_root
    assert cfg.env_override_file == (repo_root / "envs/local.env").resolve()
    assert cfg.compose_yml == (repo_root / "deploy/compose/docker-compose.yml").resolve()
    assert cfg.server_dir == (repo_root / "server").resolve()
    assert cfg.models_yaml == (repo_root / "config/models.yaml").resolve()
    assert cfg.project_name == "proj"


def test_main_dispatches_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Parser:
        def parse_args(self, _argv):
            ns = argparse.Namespace()
            ns._handler = lambda _cfg, _args: 3
            return ns

    monkeypatch.setattr(main_mod, "build_parser", lambda: _Parser())
    monkeypatch.setattr(main_mod, "_build_global_config", lambda _args: object())
    assert main_mod.main([]) == 3


def test_main_cli_error_exits_with_code(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Parser:
        def parse_args(self, _argv):
            ns = argparse.Namespace()
            ns._handler = lambda _cfg, _args: (_ for _ in ()).throw(CLIError("bad", code=9))
            return ns

    monkeypatch.setattr(main_mod, "build_parser", lambda: _Parser())
    monkeypatch.setattr(main_mod, "_build_global_config", lambda _args: object())

    with pytest.raises(SystemExit) as exc:
        main_mod.main([])
    assert exc.value.code == 9


def test_main_unexpected_error_exits_one(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Parser:
        def parse_args(self, _argv):
            ns = argparse.Namespace()
            ns._handler = lambda _cfg, _args: (_ for _ in ()).throw(RuntimeError("boom"))
            return ns

    monkeypatch.setattr(main_mod, "build_parser", lambda: _Parser())
    monkeypatch.setattr(main_mod, "_build_global_config", lambda _args: object())

    with pytest.raises(SystemExit) as exc:
        main_mod.main([])
    assert exc.value.code == 1
