from __future__ import annotations

from pathlib import Path

import yaml

from llm_policy.io.models_yaml_patch import patch_models_yaml_extract_capability


def _write(p: Path, obj) -> None:
    p.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def _read(p: Path):
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def test_patch_models_yaml_profiled_sets_extract_and_assessment(tmp_path: Path):
    path = tmp_path / "models.yaml"
    _write(
        path,
        {
            "base": {"models": [{"id": "m1", "capabilities": {"extract": False}}]},
            "profiles": {
                "host-transformers": {"models": [{"id": "m1", "capabilities": {"extract": False}}]}
            },
        },
    )

    res = patch_models_yaml_extract_capability(
        path=path,
        model_id="m1",
        enable=True,
        profile="host-transformers",
        assessed=True,
        assessed_by="policy",
        assessed_pipeline="extract_only",
        eval_run_dir="/tmp/run",
        thresholds_profile="extract/default",
        deployment={"k": "v"},
        deployment_key="dep1",
    )
    assert res.ok is True
    assert res.changed is True

    obj = _read(path)
    model = obj["profiles"]["host-transformers"]["models"][0]
    assert model["capabilities"]["extract"] is True
    assert model["assessment"]["assessed"] is True
    assert model["assessment"]["status"] == "allowed"
    assert model["assessment"]["deployment_key"] == "dep1"


def test_patch_models_yaml_is_idempotent(tmp_path: Path):
    path = tmp_path / "models.yaml"
    _write(
        path,
        {
            "base": {"models": [{"id": "m1", "capabilities": {"extract": False}}]},
            "profiles": {
                "host-transformers": {
                    "models": [
                        {
                            "id": "m1",
                            "capabilities": {"extract": True, "assessment": {"assessed": True}},
                        }
                    ]
                }
            },
        },
    )

    res1 = patch_models_yaml_extract_capability(
        path=path, model_id="m1", enable=True, profile="host-transformers"
    )
    # could still change assessment metadata first time; second call should settle
    res2 = patch_models_yaml_extract_capability(
        path=path, model_id="m1", enable=True, profile="host-transformers"
    )
    assert res2.ok is True
    assert res2.changed is False


def test_patch_models_yaml_file_not_found(tmp_path: Path):
    path = tmp_path / "missing.yaml"
    res = patch_models_yaml_extract_capability(path=path, model_id="m1", enable=True)
    assert res.ok is False
    assert "not found" in res.message.lower()
