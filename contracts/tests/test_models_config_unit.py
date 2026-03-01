from __future__ import annotations

from llm_contracts.config.models_config import (
    validate_assessment_for_extract,
    validate_models_config,
)


def _valid_cfg() -> dict:
    return {
        "primary_id": "m1",
        "model_ids": ["m1", "m2"],
        "defaults": {"selected_profile": "prod", "assessment": {"require_for_extract": False}},
        "models": [
            {
                "id": "m1",
                "backend": "transformers",
                "load_mode": "lazy",
                "deployment_key": "dep-a",
                "capabilities": {"generate": True, "extract": True},
                "assessment": {"assessed": True},
            },
            {
                "id": "m2",
                "backend": "remote",
                "load_mode": "eager",
                "deployment_key": "dep-b",
                "capabilities": {"generate": True, "extract": False},
                "assessment": {"assessed": False},
            },
        ],
    }


def test_validate_models_config_accepts_valid_payload() -> None:
    res = validate_models_config(_valid_cfg())
    assert res.ok is True
    assert res.issues == []
    assert res.snapshot["primary_id"] == "m1"


def test_validate_models_config_rejects_missing_primary_and_model_ids() -> None:
    cfg = _valid_cfg()
    cfg["primary_id"] = ""
    cfg["model_ids"] = []
    res = validate_models_config(cfg)
    assert res.ok is False
    paths = {i.path for i in res.issues}
    assert "primary_id" in paths
    assert "model_ids" in paths


def test_validate_models_config_rejects_primary_not_in_model_ids() -> None:
    cfg = _valid_cfg()
    cfg["primary_id"] = "m3"
    res = validate_models_config(cfg)
    assert res.ok is False
    assert any(i.path == "model_ids" and "primary_id must be present" in i.message for i in res.issues)


def test_validate_models_config_rejects_duplicate_ids_and_invalid_enums() -> None:
    cfg = _valid_cfg()
    cfg["models"] = [
        {
            "id": "m1",
            "backend": "bad-backend",
            "load_mode": "bad-load-mode",
            "deployment_key": "dep-a",
            "capabilities": {"generate": True, "extract": True},
        },
        {
            "id": "m1",
            "backend": "remote",
            "load_mode": "lazy",
            "deployment_key": "dep-b",
            "capabilities": {"generate": True, "extract": True},
        },
    ]
    res = validate_models_config(cfg)
    assert res.ok is False
    assert any(i.path == "models[1].id" and "duplicate model id" in i.message for i in res.issues)
    assert any(i.path == "models[0].backend" for i in res.issues)
    assert any(i.path == "models[0].load_mode" for i in res.issues)


def test_validate_models_config_rejects_generic_deployment_key_by_default() -> None:
    cfg = _valid_cfg()
    cfg["models"][0]["deployment_key"] = "default"
    res = validate_models_config(cfg)
    assert res.ok is False
    assert any(i.path == "models[0].deployment_key" and "must not be generic" in i.message for i in res.issues)


def test_validate_models_config_allows_generic_deployment_key_when_enabled() -> None:
    cfg = _valid_cfg()
    cfg["models"][0]["deployment_key"] = "default"
    res = validate_models_config(cfg, allow_generic_deployment_key=True)
    assert res.ok is True


def test_validate_models_config_rejects_invalid_capabilities_shape() -> None:
    cfg = _valid_cfg()
    cfg["models"][0]["capabilities"] = {"generate": "yes", "invalid_cap": True}
    res = validate_models_config(cfg)
    assert res.ok is False
    assert any(i.path == "models[0].capabilities.invalid_cap" and "invalid capability key" in i.message for i in res.issues)


def test_validate_assessment_for_extract_bypasses_test_profile() -> None:
    cfg = _valid_cfg()
    cfg["defaults"]["selected_profile"] = "test"
    cfg["defaults"]["assessment"] = {"require_for_extract": True}
    cfg["models"][0]["assessment"] = {}
    cfg["models"][1]["assessment"] = {}

    res = validate_assessment_for_extract(cfg, bypass_if_profile_test=True)
    assert res.ok is True
    assert res.issues == []


def test_validate_assessment_for_extract_requires_assessment_blocks_when_enabled() -> None:
    cfg = _valid_cfg()
    cfg["defaults"]["selected_profile"] = "prod"
    cfg["defaults"]["assessment"] = {"require_for_extract": True}
    cfg["models"][0]["assessment"] = {"assessed": True}
    cfg["models"][1]["assessment"] = {}

    res = validate_assessment_for_extract(cfg, bypass_if_profile_test=False)
    assert res.ok is False
    assert any(i.path == "models[1].assessment.assessed" for i in res.issues)


def test_validate_assessment_for_extract_fails_when_models_missing() -> None:
    cfg = _valid_cfg()
    cfg["defaults"]["assessment"] = {"require_for_extract": True}
    cfg["models"] = []

    res = validate_assessment_for_extract(cfg)
    assert res.ok is False
    assert any(i.path == "models" for i in res.issues)
