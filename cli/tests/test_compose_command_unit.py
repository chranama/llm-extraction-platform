from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import cli.commands.compose as compose_mod
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


def test_split_profiles_and_args() -> None:
    p, a = compose_mod._split_profiles_and_args(["infra", "server", "--", "up", "-d"])
    assert p == ["infra", "server"]
    assert a == ["up", "-d"]

    p2, a2 = compose_mod._split_profiles_and_args(["infra", "server", "up", "-d"])
    assert p2 == ["infra", "server"]
    assert a2 == ["up", "-d"]


def test_handle_dc_requires_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compose_mod, "build_compose_context", lambda *_args, **_kwargs: object())

    args = argparse.Namespace(
        compose_cmd="dc",
        tokens=["infra", "server"],  # profiles only, no compose verb
        verbose=False,
        defaults_profile=None,
        defaults_yaml=None,
        env_override_file=None,
    )
    with pytest.raises(CLIError, match="compose dc requires compose args"):
        compose_mod._handle(_cfg(tmp_path), args)


def test_handle_config_invokes_compose_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(compose_mod, "build_compose_context", lambda *_args, **_kwargs: "CTX")
    monkeypatch.setattr(
        compose_mod,
        "compose_config",
        lambda ctx, profiles, verbose: calls.update({"ctx": ctx, "profiles": profiles, "verbose": verbose}),
    )

    args = argparse.Namespace(
        compose_cmd="config",
        profiles=["infra"],
        verbose=True,
        defaults_profile=None,
        defaults_yaml=None,
        env_override_file=None,
    )
    rc = compose_mod._handle(_cfg(tmp_path), args)
    assert rc == 0
    assert calls["ctx"] == "CTX"
    assert calls["profiles"] == ["infra"]
    assert calls["verbose"] is True


def test_handle_infra_up_shortcut_invokes_compose_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}
    monkeypatch.setattr(compose_mod, "build_compose_context", lambda *_args, **_kwargs: "CTX")
    monkeypatch.setattr(
        compose_mod,
        "compose_up",
        lambda ctx, profiles, detach, build, remove_orphans, verbose: called.update(
            {
                "ctx": ctx,
                "profiles": profiles,
                "detach": detach,
                "build": build,
                "remove_orphans": remove_orphans,
                "verbose": verbose,
            }
        ),
    )

    args = argparse.Namespace(
        _shortcut="infra-up",
        verbose=False,
        defaults_profile=None,
        defaults_yaml=None,
        env_override_file=None,
    )
    rc = compose_mod._handle(_cfg(tmp_path), args)
    assert rc == 0
    assert called["ctx"] == "CTX"
    assert called["profiles"] == ["infra"]
    assert called["detach"] is True
    assert called["remove_orphans"] is True


def test_handle_logs_forces_follow_true(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}
    monkeypatch.setattr(compose_mod, "build_compose_context", lambda *_args, **_kwargs: "CTX")
    monkeypatch.setattr(
        compose_mod,
        "compose_logs",
        lambda ctx, profiles, follow, tail, verbose=False: called.update(
            {"ctx": ctx, "profiles": profiles, "follow": follow, "tail": tail}
        ),
    )

    args = argparse.Namespace(
        compose_cmd="logs",
        profiles=["infra"],
        follow=False,
        tail=25,
        verbose=False,
        defaults_profile=None,
        defaults_yaml=None,
        env_override_file=None,
    )
    rc = compose_mod._handle(_cfg(tmp_path), args)
    assert rc == 0
    assert called["follow"] is True
    assert called["tail"] == 25
