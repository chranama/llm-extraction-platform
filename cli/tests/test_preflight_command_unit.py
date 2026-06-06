from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import cli.commands.preflight as preflight_mod
from cli.types import GlobalConfig
from cli.utils.compose_runner import ComposeContext


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


def _args(target: str, **overrides):
    data = {
        "target": target,
        "verbose": False,
        "defaults_profile": None,
        "defaults_yaml": None,
        "json": False,
        "strict": False,
        "skip_port_check": True,
        "probe_external_model": False,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def _write_repo_basics(tmp_path: Path, *, models_profiles: tuple[str, ...] = ("test",)) -> None:
    (tmp_path / "deploy/compose").mkdir(parents=True)
    (tmp_path / "deploy/compose/docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config/compose-defaults.yaml").write_text(
        "profiles:\n  docker:\n    APP_PROFILE: docker\n", encoding="utf-8"
    )
    profiles_yaml = "\n".join(f"  {profile}: {{}}" for profile in models_profiles)
    (tmp_path / "config/models.yaml").write_text(
        f"profiles:\n{profiles_yaml}\n", encoding="utf-8"
    )
    (tmp_path / "schemas/model_output").mkdir(parents=True)
    (tmp_path / "schemas/model_output/sroie_receipt_v1.json").write_text(
        '{"type":"object"}\n', encoding="utf-8"
    )


def _ctx(tmp_path: Path, env_text: str, *, defaults_profile: str = "docker") -> ComposeContext:
    env_file = tmp_path / ".tmp" / "llmctl" / "compose-effective.env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(env_text, encoding="utf-8")
    return ComposeContext(
        base_cmd=["docker", "compose", "--env-file", str(env_file)],
        proc_env={},
        env_files=[env_file],
        defaults_profile=defaults_profile,
        rendered_defaults_env=env_file,
        user_env_file=None,
    )


def _mock_external_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight_mod, "_check_docker_daemon", lambda checks, target: None)
    monkeypatch.setattr(preflight_mod, "compose_config_check", lambda *args, **kwargs: None)
    monkeypatch.setattr(preflight_mod.shutil, "which", lambda name: f"/usr/bin/{name}")


def test_payload_reports_overall_status() -> None:
    payload = preflight_mod._payload(
        "smoke",
        [
            preflight_mod.PreflightCheck("smoke", "a", "pass", "ok"),
            preflight_mod.PreflightCheck("smoke", "b", "warn", "warning"),
        ],
    )

    assert payload["status"] == "warn"
    assert payload["summary"] == {"pass": 1, "warn": 1, "fail": 0, "skip": 0}


def test_handle_prints_json_and_returns_failure(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        preflight_mod,
        "_run_target_preflight",
        lambda cfg, args, target: [
            preflight_mod.PreflightCheck(target, "env:API_KEY", "fail", "missing")
        ],
    )

    rc = preflight_mod._handle(_cfg(Path("/tmp/repo")), _args("smoke", json=True))

    out = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert out["target"] == "smoke"
    assert out["status"] == "fail"
    assert out["checks"][0]["name"] == "env:API_KEY"


def test_handle_strict_returns_one_for_warnings(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        preflight_mod,
        "_run_target_preflight",
        lambda cfg, args, target: [
            preflight_mod.PreflightCheck(target, "llama CPU setting", "warn", "not cpu")
        ],
    )

    rc = preflight_mod._handle(_cfg(Path("/tmp/repo")), _args("compose-extract", strict=True))

    assert rc == 1
    assert "WARN" in capsys.readouterr().out


def test_smoke_preflight_fails_when_api_key_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_repo_basics(tmp_path, models_profiles=("test",))
    _mock_external_checks(monkeypatch)
    monkeypatch.setattr(
        preflight_mod,
        "_build_target_context",
        lambda *args, **kwargs: _ctx(tmp_path, "", defaults_profile="docker+reviewer-smoke"),
    )

    checks = preflight_mod._run_target_preflight(_cfg(tmp_path), _args("smoke"), "smoke")

    assert any(
        check.name == "env:API_KEY" and check.status == "fail"
        for check in checks
    )


def test_compose_extract_preflight_fails_when_model_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_repo_basics(tmp_path, models_profiles=("compose-extract",))
    (tmp_path / "policy_out").mkdir()
    (tmp_path / "policy_out/local_extract_allow.json").write_text("{}", encoding="utf-8")
    _mock_external_checks(monkeypatch)
    monkeypatch.setattr(
        preflight_mod,
        "_build_target_context",
        lambda *args, **kwargs: _ctx(
            tmp_path,
            "\n".join(
                [
                    "API_KEY=key",
                    f"LLAMA_MODELS_DIR={tmp_path / 'models'}",
                    "LLAMA_MODEL_FILE=/models/smollm2/Q8_0.gguf",
                    "LLAMA_N_GPU_LAYERS=0",
                    "",
                ]
            ),
            defaults_profile="docker+llama+compose-extract",
        ),
    )

    checks = preflight_mod._run_target_preflight(
        _cfg(tmp_path), _args("compose-extract"), "compose-extract"
    )

    assert any(
        check.name == "llama model file" and check.status == "fail"
        for check in checks
    )


def test_compose_extract_preflight_accepts_model_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_repo_basics(tmp_path, models_profiles=("compose-extract",))
    (tmp_path / "policy_out").mkdir()
    (tmp_path / "policy_out/local_extract_allow.json").write_text("{}", encoding="utf-8")
    model = tmp_path / "models" / "smollm2" / "Q8_0.gguf"
    model.parent.mkdir(parents=True)
    model.write_text("fake gguf", encoding="utf-8")
    _mock_external_checks(monkeypatch)
    monkeypatch.setattr(
        preflight_mod,
        "_build_target_context",
        lambda *args, **kwargs: _ctx(
            tmp_path,
            "\n".join(
                [
                    "API_KEY=key",
                    f"LLAMA_MODELS_DIR={tmp_path / 'models'}",
                    "LLAMA_MODEL_FILE=/models/smollm2/Q8_0.gguf",
                    "LLAMA_N_GPU_LAYERS=0",
                    "",
                ]
            ),
            defaults_profile="docker+llama+compose-extract",
        ),
    )

    checks = preflight_mod._run_target_preflight(
        _cfg(tmp_path), _args("compose-extract"), "compose-extract"
    )

    assert not [check for check in checks if check.status == "fail"]
    assert any(
        check.name == "llama model file" and check.status == "pass"
        for check in checks
    )
