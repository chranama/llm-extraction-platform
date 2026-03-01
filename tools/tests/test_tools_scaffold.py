from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_compose_helper_scripts_exist_and_use_strict_mode() -> None:
    for rel in ("tools/compose/compose_doctor.sh", "tools/k8s/k8s_smoke.sh"):
        text = _read_text(rel)
        assert text.startswith("#!/usr/bin/env bash"), rel
        assert "set -euo pipefail" in text, rel


def test_compose_doctor_checks_expected_endpoints() -> None:
    text = _read_text("tools/compose/compose_doctor.sh")
    assert "/healthz" in text
    assert "/readyz" in text
    assert "/modelz" in text
    assert "/v1/schemas" in text
    assert "/v1/extract" in text


def test_k8s_smoke_covers_generate_only_contract() -> None:
    text = _read_text("tools/k8s/k8s_smoke.sh")
    assert "/v1/models" in text
    assert "/v1/generate" in text
    assert "/v1/extract" in text
    assert "deployment_capabilities.generate" in text
    assert "deployment_capabilities.extract" in text
