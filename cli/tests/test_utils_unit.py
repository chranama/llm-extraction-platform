from __future__ import annotations

import os
from pathlib import Path

import pytest

from cli.errors import CLIError
from cli.types import GlobalConfig
from cli.utils.compose_config import render_compose_env_dict
from cli.utils.compose_runner import clean_compose_process_env
from cli.utils.env import load_dotenv_file
from cli.utils.paths import resolve_path
import cli.utils.proc as proc_mod


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


def test_run_check_false_returns_nonzero_code() -> None:
    res = proc_mod.run(["bash", "-lc", "exit 3"], check=False, inherit_env=False)
    assert res.code == 3


def test_run_check_true_raises_cli_error() -> None:
    with pytest.raises(CLIError, match="Command failed"):
        proc_mod.run(["bash", "-lc", "exit 4"], check=True, inherit_env=False)


def test_ensure_bins_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proc_mod, "_which", lambda name: None if name == "docker" else "/bin/bash")
    with pytest.raises(CLIError, match="Missing required tools: docker"):
        proc_mod.ensure_bins("bash", "docker")


def test_load_dotenv_file_parses_quotes_and_comments(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "A=1",
                "B='two'",
                'C="three"',
                "INVALID_LINE",
            ]
        ),
        encoding="utf-8",
    )
    out = load_dotenv_file(env_file)
    assert out == {"A": "1", "B": "two", "C": "three"}


def test_resolve_path_handles_none_blank_relative_absolute(tmp_path: Path) -> None:
    assert resolve_path(tmp_path, None) is None
    assert resolve_path(tmp_path, "   ") is None
    assert resolve_path(tmp_path, "a/b") == (tmp_path / "a/b").resolve()
    assert resolve_path(tmp_path, str((tmp_path / "x").resolve())) == (tmp_path / "x").resolve()


def test_render_compose_env_dict_merges_profiles_and_extra(tmp_path: Path) -> None:
    cfg_yaml = tmp_path / "compose-defaults.yaml"
    cfg_yaml.write_text(
        "\n".join(
            [
                "profiles:",
                "  docker:",
                "    APP_PROFILE: docker",
                "    API_PORT: 8000",
                "  jobs:",
                "    APP_PROFILE: jobs",
                "    REDIS_ENABLED: true",
            ]
        ),
        encoding="utf-8",
    )

    env = render_compose_env_dict(
        config_yaml_path=cfg_yaml,
        profile="docker+jobs",
        extra_env={"COMPOSE_PROJECT_NAME": "proj"},
    )
    assert env["APP_PROFILE"] == "jobs"
    assert env["API_PORT"] == "8000"
    assert env["REDIS_ENABLED"] == "1"
    assert env["COMPOSE_PROJECT_NAME"] == "proj"


def test_clean_compose_process_env_enforces_denylist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("API_KEY", "k")
    monkeypatch.setenv("MODEL_ID", "should_not_pass")
    monkeypatch.setenv("PATH", os.getenv("PATH", ""))

    env = clean_compose_process_env(_cfg(tmp_path))
    assert env["COMPOSE_PROJECT_NAME"] == "proj"
    assert env["API_KEY"] == "k"
    assert "MODEL_ID" not in env
