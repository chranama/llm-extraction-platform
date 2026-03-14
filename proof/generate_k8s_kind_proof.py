#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "proof" / "artifacts" / "phase5_k8s_kind"
KIND_CLUSTER = "llm"
NAMESPACE = "llm"
LOCAL_PORT = 18080
REMOTE_PORT = 8000
API_BASE = f"http://127.0.0.1:{LOCAL_PORT}"
SERVICE_NAME = "api"
MIN_DOCKER_FREE_BYTES = 8 * 1024 * 1024 * 1024


def fail(message: str) -> None:
    raise RuntimeError(message)


def ensure_bin(name: str) -> None:
    if shutil.which(name) is None:
        fail(f"missing required binary: {name}")


def log_step(message: str) -> None:
    print(f"[phase5_k8s_kind] {message}", file=sys.stderr, flush=True)


def run(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=check,
        capture_output=capture_output,
        text=text,
        env=env,
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{num_bytes}B"


def ensure_docker_storage_headroom() -> None:
    root_result = run(["docker", "info", "--format", "{{.DockerRootDir}}"], check=False)
    docker_root = root_result.stdout.strip()
    if root_result.returncode != 0 or not docker_root:
        fail("unable to determine DockerRootDir from docker info")

    usage = shutil.disk_usage(docker_root)
    if usage.free < MIN_DOCKER_FREE_BYTES:
        fail(
            "docker storage is too low for kind proof: "
            f"{format_bytes(usage.free)} free under {docker_root}; "
            f"need at least {format_bytes(MIN_DOCKER_FREE_BYTES)}. "
            "Run docker cleanup before retrying."
        )

    log_step(
        "docker storage preflight passed: "
        f"{format_bytes(usage.free)} free under {docker_root}"
    )


def render_overlay(overlay: str, output_path: Path) -> None:
    for cmd in (["kustomize", "build", overlay], ["kubectl", "kustomize", overlay]):
        if shutil.which(cmd[0]) is None:
            continue
        result = run(cmd, check=False)
        if result.returncode == 0:
            if not result.stdout.strip():
                fail(f"rendered overlay is empty: {overlay}")
            write_text(output_path, result.stdout)
            return
    fail(f"unable to render overlay: {overlay}")


def http_request(method: str, path: str, body: bytes | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(f"{API_BASE}{path}", data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError:
        return 0, b""


@contextmanager
def port_forward():
    proc = subprocess.Popen(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "port-forward",
            f"svc/{SERVICE_NAME}",
            f"{LOCAL_PORT}:{REMOTE_PORT}",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        for _ in range(50):
            code, _ = http_request("GET", "/healthz")
            if code == 200:
                break
            if proc.poll() is not None:
                fail("kubectl port-forward exited before /healthz became reachable")
            time.sleep(0.2)
        else:
            fail("timed out waiting for local port-forward healthz")
        yield
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def generate_k8s_kind_proof() -> None:
    for binary in ("kind", "kubectl", "docker", "curl", "python"):
        ensure_bin(binary)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    docker_info = run(["docker", "info"], check=False)
    if docker_info.returncode != 0:
        fail("docker daemon is not reachable; start Docker Desktop (or equivalent) before running the kind proof")
    ensure_docker_storage_headroom()

    clusters = run(["kind", "get", "clusters"], check=False).stdout.splitlines()
    if KIND_CLUSTER not in clusters:
        log_step(f"creating kind cluster {KIND_CLUSTER}")
        run(
            [
                "kind",
                "create",
                "cluster",
                "--config",
                str(ROOT / "deploy" / "k8s" / "kind" / "kind-config.yaml"),
            ]
        )
    else:
        log_step(f"reusing existing kind cluster {KIND_CLUSTER}")

    log_step("building llm-server:dev image")
    run(
        [
            "docker",
            "build",
            "-t",
            "llm-server:dev",
            "-f",
            str(ROOT / "deploy" / "docker" / "Dockerfile.server"),
            str(ROOT),
        ]
    )
    log_step(f"loading llm-server:dev into kind cluster {KIND_CLUSTER}")
    run(["kind", "load", "docker-image", "llm-server:dev", "--name", KIND_CLUSTER])
    log_step("resetting local-generate-only overlay resources")
    run(
        [
            "kubectl",
            "delete",
            "-k",
            str(ROOT / "deploy" / "k8s" / "overlays" / "local-generate-only"),
            "--ignore-not-found",
        ],
        check=False,
    )
    log_step("applying local-generate-only overlay")
    run(
        [
            "kubectl",
            "apply",
            "-k",
            str(ROOT / "deploy" / "k8s" / "overlays" / "local-generate-only"),
        ]
    )

    log_step("waiting for db-migrate job completion")
    migrate_wait = run(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "wait",
            "--for=condition=complete",
            "job/db-migrate",
            "--timeout=240s",
        ]
    )
    write_text(ARTIFACT_DIR / "db_migrate_job_status.txt", migrate_wait.stdout)

    log_step("waiting for api deployment rollout")
    rollout = run(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "rollout",
            "status",
            "deployment/api",
            "--timeout=240s",
        ]
    )
    write_text(ARTIFACT_DIR / "server_rollout_status.txt", rollout.stdout)
    write_text(
        ARTIFACT_DIR / "kubectl_get_pods.txt",
        run(["kubectl", "-n", NAMESPACE, "get", "pods", "-o", "wide"]).stdout,
    )
    write_text(
        ARTIFACT_DIR / "kubectl_get_svc.txt",
        run(["kubectl", "-n", NAMESPACE, "get", "svc"]).stdout,
    )

    log_step("running Kubernetes smoke checks")
    smoke = run(
        [str(ROOT / "tools" / "k8s" / "k8s_smoke.sh")],
        env={
            **dict(os.environ),
            "NAMESPACE": NAMESPACE,
            "API_SVC": SERVICE_NAME,
            "LOCAL_PORT": str(LOCAL_PORT),
            "REMOTE_PORT": str(REMOTE_PORT),
        },
    )
    write_text(ARTIFACT_DIR / "k8s_smoke.log", smoke.stdout)

    log_step("rendering local overlay manifest")
    render_overlay(
        str(ROOT / "deploy" / "k8s" / "overlays" / "local-generate-only"),
        ARTIFACT_DIR / "kustomize_local_generate_only.yaml",
    )
    log_step("rendering prod overlay manifest")
    render_overlay(
        str(ROOT / "deploy" / "k8s" / "overlays" / "prod-gpu-full"),
        ARTIFACT_DIR / "kustomize_prod_gpu_full.yaml",
    )

    log_step("verifying live service via port-forward")
    with port_forward():
        health_code, _ = http_request("GET", "/healthz")
        models_code, models_body = http_request("GET", "/v1/models")
        extract_code, _ = http_request(
            "POST",
            "/v1/extract",
            body=b'{"schema_id":"invoice_v1","text":"probe","cache":false,"repair":false}',
        )

    if health_code != 200:
        fail(f"/healthz returned {health_code}")
    if models_code != 200:
        fail(f"/v1/models returned {models_code}")

    models_payload = json.loads(models_body.decode("utf-8"))
    dep_caps = models_payload.get("deployment_capabilities") or {}
    if dep_caps.get("generate") is not True or dep_caps.get("extract") is not False:
        fail(f"deployment capabilities mismatch: {dep_caps}")
    if extract_code == 200:
        fail("/v1/extract unexpectedly returned 200")

    log_step("writing proof summary")
    summary = {
        "proof_phase": "phase5_k8s_kind",
        "cluster_name": KIND_CLUSTER,
        "namespace": NAMESPACE,
        "overlay": "local-generate-only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "checks": {
            "rollout_status": "pass",
            "healthz": "pass",
            "models_capabilities": "pass",
            "generate_smoke": "pass",
            "extract_disabled": "pass",
            "local_overlay_render": "pass",
            "prod_overlay_render": "pass",
        },
        "deployment_capabilities": {
            "generate": True,
            "extract": False,
        },
        "http_status": {
            "extract_disabled": extract_code,
        },
        "service_name": SERVICE_NAME,
        "api_base": API_BASE,
    }
    write_text(ARTIFACT_DIR / "kind_smoke_summary.json", json.dumps(summary, indent=2) + "\n")


def main() -> None:
    try:
        generate_k8s_kind_proof()
    except Exception as exc:  # pragma: no cover - surfaced in CLI/CI output
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
