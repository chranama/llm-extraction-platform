#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "proof" / "artifacts" / "phase8_compose_llama_extract"
COMPOSE_FILE = ROOT / "deploy" / "compose" / "docker-compose.yml"
COMPOSE_PROJECT = "llmep-phase8"
API_PORT = os.getenv("PHASE8_API_PORT", "18083")
LLAMA_PUBLISH_PORT = os.getenv("PHASE8_LLAMA_PUBLISH_PORT", "18084")
API_BASE = f"http://127.0.0.1:{API_PORT}"
SOURCE_ENV_FILE = ROOT / os.getenv("PHASE8_ENV_FILE", ".env.docker")
EFFECTIVE_ENV_FILE = ROOT / ".tmp" / "phase8-compose-extract.env"
RECEIPT_TEXT = "ACME STORE\n123 MAIN ST\nDATE: 2024-03-10\nTOTAL: $42.18"


def fail(message: str) -> None:
    raise RuntimeError(message)


def run(
    args: list[str],
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=check,
        env=env,
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            out[key] = value
    return out


def write_effective_env() -> dict[str, str]:
    if not SOURCE_ENV_FILE.exists():
        fail(f"missing env file: {SOURCE_ENV_FILE.relative_to(ROOT)}")
    env = read_env_file(SOURCE_ENV_FILE)
    env["API_PORT"] = API_PORT
    env["LLAMA_PUBLISH_PORT"] = LLAMA_PUBLISH_PORT
    env["POLICY_DECISION_PATH"] = "/app/policy_out/local_extract_allow.json"
    env.setdefault("ADMIN_API_KEY", env.get("API_KEY", ""))
    EFFECTIVE_ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    EFFECTIVE_ENV_FILE.write_text(
        "\n".join(f"{key}={value}" for key, value in sorted(env.items())) + "\n",
        encoding="utf-8",
    )
    return env


def http_json(
    method: str,
    path: str,
    *,
    api_key: str | None,
    payload: dict | None = None,
    timeout: float = 60.0,
) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(f"{API_BASE}{path}", data=body, method=method)
    if api_key:
        req.add_header("X-API-Key", api_key)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            payload = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            payload = {"raw": raw}
        return exc.code, payload
    except urllib.error.URLError:
        return 0, {}


def capture_endpoint(path: str, filename: str, *, api_key: str | None = None) -> dict:
    status, body = http_json("GET", path, api_key=api_key, timeout=30)
    payload = {"status_code": status, "body": body}
    write_json(ARTIFACT_DIR / filename, payload)
    return payload


def capture_logs(env_file: Path) -> None:
    base = [
        "docker",
        "compose",
        "--env-file",
        str(env_file),
        "-f",
        str(COMPOSE_FILE),
        "--profile",
        "infra",
        "--profile",
        "llama",
        "--profile",
        "server-llama",
        "--profile",
        "worker-llama",
        "-p",
        COMPOSE_PROJECT,
    ]
    for service, filename in (
        ("server_llama", "server_llama.log"),
        ("llama_server", "llama_server.log"),
        ("worker_llama", "worker_llama.log"),
    ):
        result = run(base + ["logs", "--no-color", "--tail=200", service], check=False)
        write_text(ARTIFACT_DIR / filename, result.stdout + result.stderr)


def wait_for_job(api_key: str, poll_path: str) -> dict:
    deadline = time.time() + 180
    final: dict = {}
    status = 0
    while time.time() < deadline:
        status, final = http_json("GET", poll_path, api_key=api_key, timeout=30)
        if status == 200 and final.get("status") in {"succeeded", "failed"}:
            break
        time.sleep(1)
    return {"status_code": status, "body": final}


def generate_compose_llama_extract_proof() -> None:
    for binary in ("docker", "uv"):
        if shutil.which(binary) is None:
            fail(f"missing required binary: {binary}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    env = write_effective_env()
    api_key = env.get("API_KEY", "").strip()
    if not api_key:
        fail("API_KEY is required in the phase8 env file")

    command = [
        "uv",
        "run",
        "llmctl",
        "--project-name",
        COMPOSE_PROJECT,
        "--env-override-file",
        str(EFFECTIVE_ENV_FILE.relative_to(ROOT)),
        "--api-port",
        API_PORT,
        "compose-extract",
    ]
    result = run(command, check=False)
    write_text(ARTIFACT_DIR / "llmctl_compose_extract.log", result.stdout + result.stderr)
    if result.returncode != 0:
        capture_logs(EFFECTIVE_ENV_FILE)
        fail(f"llmctl compose-extract failed with exit code {result.returncode}")

    capture_endpoint("/healthz", "healthz.json")
    readyz = capture_endpoint("/readyz", "readyz.json")
    models_status = capture_endpoint("/v1/models/status", "models_status.json", api_key=api_key)

    generate_status, generate_body = http_json(
        "POST",
        "/v1/generate",
        api_key=api_key,
        payload={"prompt": "smoke test", "max_new_tokens": 16, "temperature": 0.2},
        timeout=90,
    )
    generate_payload = {"status_code": generate_status, "body": generate_body}
    write_json(ARTIFACT_DIR / "generate_response.json", generate_payload)

    extract_status, extract_body = http_json(
        "POST",
        "/v1/extract",
        api_key=api_key,
        payload={
            "schema_id": "sroie_receipt_v1",
            "text": RECEIPT_TEXT,
            "max_new_tokens": 512,
            "temperature": 0.0,
            "cache": False,
            "repair": True,
        },
        timeout=180,
    )
    extract_payload = {"status_code": extract_status, "body": extract_body}
    write_json(ARTIFACT_DIR / "extract_response.json", extract_payload)

    async_status, async_body = http_json(
        "POST",
        "/v1/extract/jobs",
        api_key=api_key,
        payload={
            "schema_id": "sroie_receipt_v1",
            "text": RECEIPT_TEXT,
            "cache": False,
            "repair": True,
        },
        timeout=60,
    )
    async_submit = {"status_code": async_status, "body": async_body}
    write_json(ARTIFACT_DIR / "async_submit_response.json", async_submit)
    async_final = {"status_code": 0, "body": {}}
    if async_status == 202 and async_body.get("poll_path"):
        async_final = wait_for_job(api_key, str(async_body["poll_path"]))
    write_json(ARTIFACT_DIR / "async_final_response.json", async_final)

    capture_logs(EFFECTIVE_ENV_FILE)

    checks = {
        "readyz": readyz["status_code"] == 200,
        "models_status": models_status["status_code"] == 200,
        "generate": generate_status == 200,
        "extract": extract_status == 200 and bool((extract_body.get("data") or {})),
        "async_extract": async_final["status_code"] == 200
        and (async_final.get("body") or {}).get("status") == "succeeded",
    }
    summary = {
        "proof_phase": "phase8_compose_llama_extract",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "compose_project": COMPOSE_PROJECT,
        "api_base": API_BASE,
        "model_backend": "containerized llama.cpp",
        "model_profile": "compose-extract",
        "model_runtime": "SmolLM2-360M-Instruct GGUF via llama-server",
        "acceleration": "cpu",
        "checks": checks,
        "resolved_model": extract_body.get("model") or generate_body.get("model"),
    }
    write_json(ARTIFACT_DIR / "compose_llama_extract_summary.json", summary)
    if summary["status"] != "pass":
        fail(f"phase8 proof failed checks: {checks}")


def cleanup() -> None:
    if not EFFECTIVE_ENV_FILE.exists():
        return
    run(
        [
            "uv",
            "run",
            "llmctl",
            "--project-name",
            COMPOSE_PROJECT,
            "--env-override-file",
            str(EFFECTIVE_ENV_FILE.relative_to(ROOT)),
            "stop",
            "--volumes",
        ],
        check=False,
    )
    for path in (
        EFFECTIVE_ENV_FILE,
        ROOT / ".tmp" / "llmctl" / "compose-effective-docker_llama_compose-extract.env",
        ROOT / ".tmp" / "llmctl" / "compose-effective-docker.env",
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    try:
        generate_compose_llama_extract_proof()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
