from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import cli.commands.k8s as k8s_mod
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


def test_kind_up_invokes_run_bash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}
    monkeypatch.setattr(k8s_mod, "ensure_bins", lambda *_bins: None)
    monkeypatch.setattr(
        k8s_mod,
        "run_bash",
        lambda script, verbose: called.update({"script": script, "verbose": verbose}),
    )

    args = argparse.Namespace(k8s_cmd="kind-up", verbose=False)
    rc = k8s_mod._handle(_cfg(tmp_path), args)
    assert rc == 0
    assert "kind create cluster" in str(called["script"])
    assert "kind-config.yaml" in str(called["script"])


def test_unknown_k8s_command_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(k8s_mod, "ensure_bins", lambda *_bins: None)
    with pytest.raises(CLIError, match="Unknown k8s command"):
        k8s_mod._handle(_cfg(tmp_path), argparse.Namespace(k8s_cmd="nope", verbose=False))
