from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pytest

import cli.commands.paths as paths_mod
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


def _args(cmd: str, **overrides):
    data = {
        "path_cmd": cmd,
        "verbose": False,
        "defaults_profile": None,
        "defaults_yaml": None,
        "skip_verify": True,
        "skip_async": True,
        "regenerate": False,
        "volumes": False,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def test_compose_extract_requires_llama_model_file(tmp_path: Path) -> None:
    env = {
        "API_KEY": "key",
        "LLAMA_MODELS_DIR": str(tmp_path / "models"),
        "LLAMA_MODEL_FILE": "/models/smollm2/Q8_0.gguf",
    }
    with pytest.raises(CLIError, match="Could not find"):
        paths_mod._validate_compose_extract_env(env)


def test_compose_extract_accepts_existing_llama_model_file(tmp_path: Path) -> None:
    model = tmp_path / "models" / "smollm2" / "Q8_0.gguf"
    model.parent.mkdir(parents=True)
    model.write_text("fake gguf placeholder", encoding="utf-8")

    paths_mod._validate_compose_extract_env(
        {
            "API_KEY": "key",
            "LLAMA_MODELS_DIR": str(tmp_path / "models"),
            "LLAMA_MODEL_FILE": "/models/smollm2/Q8_0.gguf",
        }
    )


def test_request_json_treats_connection_reset_as_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_connection_reset(_request, timeout):
        raise ConnectionResetError("reset")

    monkeypatch.setattr(urllib.request, "urlopen", raise_connection_reset)

    status, body = paths_mod._request_json(method="GET", url="http://127.0.0.1/healthz")

    assert status == 0
    assert body == {}


def test_verify_schema_surfaces_checks_index_and_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_request_json(method, url, api_key=None, payload=None, timeout=30):
        calls.append(url)
        if url.endswith("/v1/schemas"):
            return 200, [{"schema_id": "sroie_receipt_v1", "title": "Receipt"}]
        if url.endswith("/v1/schemas/sroie_receipt_v1"):
            return 200, {"type": "object", "properties": {}}
        return 404, {}

    monkeypatch.setattr(paths_mod, "_request_json", fake_request_json)

    result = paths_mod._verify_schema_surfaces("http://127.0.0.1:8000", "key")

    assert result["detail"]["type"] == "object"
    assert calls == [
        "http://127.0.0.1:8000/v1/schemas",
        "http://127.0.0.1:8000/v1/schemas/sroie_receipt_v1",
    ]


def test_run_compose_extract_uses_promoted_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}
    model = tmp_path / "models" / "smollm2" / "Q8_0.gguf"
    model.parent.mkdir(parents=True)
    model.write_text("fake gguf placeholder", encoding="utf-8")

    class Ctx:
        proc_env = {}
        env_files = []

    def build_context(_cfg, _args, *, default_profile):
        calls["default_profile"] = default_profile
        return Ctx()

    monkeypatch.setattr(paths_mod, "ensure_bins", lambda *_bins: None)
    monkeypatch.setattr(paths_mod, "_compose_context", build_context)
    monkeypatch.setattr(
        paths_mod,
        "_effective_env",
        lambda _ctx: {
            "API_KEY": "key",
            "LLAMA_MODELS_DIR": str(tmp_path / "models"),
            "LLAMA_MODEL_FILE": "/models/smollm2/Q8_0.gguf",
        },
    )
    monkeypatch.setattr(
        paths_mod,
        "compose_config_check",
        lambda ctx, profiles, verbose: calls.update({"config_profiles": profiles}),
    )

    up_calls: list[tuple[list[str], bool]] = []
    monkeypatch.setattr(
        paths_mod,
        "compose_up",
        lambda ctx, profiles, detach, build, remove_orphans, verbose: up_calls.append(
            (list(profiles), bool(build))
        ),
    )
    monkeypatch.setattr(
        paths_mod, "_migrate", lambda ctx, services, verbose: calls.update({"migrate": services})
    )
    monkeypatch.setattr(paths_mod, "_seed_api_keys", lambda ctx, env, verbose: None)

    rc = paths_mod._run_compose_extract(_cfg(tmp_path), _args("compose-extract"))

    assert rc == 0
    assert calls["default_profile"] == "docker+llama+compose-extract"
    assert calls["config_profiles"] == ["infra", "llama", "server-llama"]
    assert up_calls == [
        (["infra", "llama", "server-llama"], True),
        (["infra", "llama", "worker-llama"], True),
    ]
    assert calls["migrate"] == ["server_llama"]


def test_evidence_validate_runs_validator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}
    monkeypatch.setattr(
        paths_mod,
        "run",
        lambda cmd, verbose=False, **_kwargs: called.update({"cmd": cmd, "verbose": verbose}),
    )

    rc = paths_mod._run_evidence(_cfg(tmp_path), _args("evidence"))

    assert rc == 0
    assert str(called["cmd"][1]).endswith("proof/validate_evidence_manifest.py")


@pytest.mark.parametrize(
    ("runner", "script"),
    [
        (paths_mod._run_policy_eval, "proof/generate_policy_eval_linkage_proof.py"),
        (paths_mod._run_admin_trace, "proof/generate_trace_inspection_proof.py"),
        (paths_mod._run_ops_surface, "proof/generate_ops_surface_proof.py"),
    ],
)
def test_proof_path_commands_run_expected_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner,
    script: str,
) -> None:
    called: dict[str, object] = {}
    monkeypatch.setattr(paths_mod, "ensure_bins", lambda *_bins: None)
    monkeypatch.setattr(
        paths_mod,
        "run",
        lambda cmd, verbose=False, **_kwargs: called.update({"cmd": cmd, "verbose": verbose}),
    )

    rc = runner(_cfg(tmp_path), _args("proof"))

    assert rc == 0
    assert str(called["cmd"][1]).endswith(script)


def test_stop_invokes_compose_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}
    monkeypatch.setattr(paths_mod, "ensure_bins", lambda *_bins: None)
    monkeypatch.setattr(paths_mod, "build_compose_context", lambda cfg: "CTX")
    monkeypatch.setattr(
        paths_mod,
        "compose_down",
        lambda ctx, profiles, volumes, remove_orphans, verbose: called.update(
            {
                "ctx": ctx,
                "profiles": profiles,
                "volumes": volumes,
                "remove_orphans": remove_orphans,
            }
        ),
    )
    monkeypatch.setattr(paths_mod, "compose_ps", lambda *args, **kwargs: None)

    rc = paths_mod._run_stop(_cfg(tmp_path), _args("stop", volumes=True))

    assert rc == 0
    assert called == {
        "ctx": "CTX",
        "profiles": paths_mod.COMPOSE_STOP_PROFILES,
        "volumes": True,
        "remove_orphans": True,
    }
