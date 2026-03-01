from __future__ import annotations

import json
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CONFIG_DIR.parent


def _read_yaml(relative_path: str) -> dict:
    return yaml.safe_load((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def _read_json(relative_path: str) -> dict:
    return json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def test_all_repo_config_files_parse() -> None:
    for rel in (
        "config/server.yaml",
        "config/eval.yaml",
        "config/policy.yaml",
        "config/compose-defaults.yaml",
        "config/models.yaml",
    ):
        parsed = _read_yaml(rel)
        assert isinstance(parsed, dict), rel

    ui = _read_json("config/ui.json")
    assert isinstance(ui, dict)


def test_server_config_references_existing_models_config() -> None:
    cfg = _read_yaml("config/server.yaml")

    assert set(cfg) >= {"base", "profiles"}
    assert isinstance(cfg["base"], dict)
    assert isinstance(cfg["profiles"], dict)
    assert {"host", "docker", "test"} <= set(cfg["profiles"])

    model_cfg_path = cfg["base"]["model"]["models_config_path"]
    assert isinstance(model_cfg_path, str)
    assert (REPO_ROOT / model_cfg_path).exists()


def test_compose_defaults_profiles_have_expected_keys() -> None:
    cfg = _read_yaml("config/compose-defaults.yaml")
    profiles = cfg["profiles"]

    for name in ("docker", "host", "itest", "jobs"):
        assert name in profiles

    assert {"APP_PROFILE", "MODELS_YAML", "MODELS_PROFILE"} <= set(profiles["docker"])
    assert {"APP_PROFILE", "MODELS_YAML", "MODELS_PROFILE"} <= set(profiles["host"])
    assert {"APP_PROFILE", "MODELS_YAML", "MODELS_PROFILE"} <= set(profiles["itest"])
    assert {"API_BASE_URL", "POLICY_OUT_PATH", "POLICY_THRESHOLDS_ROOT"} <= set(profiles["jobs"])


def test_models_yaml_profiles_have_consistent_inventory() -> None:
    cfg = _read_yaml("config/models.yaml")
    assert set(cfg) >= {"base", "profiles"}

    profiles = cfg["profiles"]
    required_profiles = {
        "host-transformers",
        "docker-transformers",
        "host-llama",
        "docker-llama",
        "test",
    }
    assert required_profiles <= set(profiles)

    for name in required_profiles:
        entry = profiles[name]
        assert isinstance(entry.get("models"), list) and entry["models"], name
        model_ids = {m["id"] for m in entry["models"]}
        assert entry["default_model"] in model_ids, name


def test_eval_policy_ui_configs_have_required_shape() -> None:
    eval_cfg = _read_yaml("config/eval.yaml")
    assert isinstance(eval_cfg["service"]["base_url"], str)
    assert isinstance(eval_cfg["defaults"]["model_id"], str)
    assert isinstance(eval_cfg["extraction"]["schema_ids"], list)
    assert eval_cfg["extraction"]["schema_ids"]

    policy_cfg = _read_yaml("config/policy.yaml")
    assert isinstance(policy_cfg["version"], int)
    assert isinstance(policy_cfg["tasks"], dict) and policy_cfg["tasks"]
    assert isinstance(policy_cfg["run"]["outdir_root"], str)

    ui_cfg = _read_json("config/ui.json")
    assert isinstance(ui_cfg["api"]["base_url"], str)
    assert isinstance(ui_cfg["features"], dict)
