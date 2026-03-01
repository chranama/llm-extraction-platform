from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_yaml(relative_path: str) -> dict:
    return yaml.safe_load((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def test_compose_manifest_has_expected_core_services() -> None:
    compose = _read_yaml("deploy/compose/docker-compose.yml")
    assert set(compose) >= {"services"}
    services = compose["services"]

    required = {
        "postgres",
        "redis",
        "server",
        "server_itest",
        "eval",
        "policy",
        "ui",
        "llama_server",
    }
    assert required <= set(services)


def test_compose_build_dockerfiles_exist() -> None:
    compose = _read_yaml("deploy/compose/docker-compose.yml")
    services = compose["services"]
    for name, svc in services.items():
        build = svc.get("build")
        if not isinstance(build, dict):
            continue
        dockerfile = build.get("dockerfile")
        if isinstance(dockerfile, str):
            assert (REPO_ROOT / dockerfile).exists(), f"{name}: {dockerfile}"


def test_k8s_base_kustomization_resource_paths_exist() -> None:
    base_dir = REPO_ROOT / "deploy/k8s/base"
    kustomization = _read_yaml("deploy/k8s/base/kustomization.yaml")

    assert isinstance(kustomization.get("resources"), list)
    for rel in kustomization["resources"]:
        assert (base_dir / rel).exists(), rel


def test_k8s_overlays_reference_base_and_patches() -> None:
    overlay_paths = [
        "deploy/k8s/overlays/local-generate-only/kustomization.yaml",
        "deploy/k8s/overlays/prod-gpu-full/kustomization.yaml",
    ]
    for overlay in overlay_paths:
        data = _read_yaml(overlay)
        resources = data.get("resources") or []
        assert "../../base" in resources, overlay

        overlay_dir = (REPO_ROOT / overlay).parent
        for patch in data.get("patchesStrategicMerge") or []:
            assert (overlay_dir / patch).exists(), f"{overlay}: {patch}"
