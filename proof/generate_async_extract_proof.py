#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "proof" / "artifacts" / "phase6_extract_async"
COMPOSE_FILE = ROOT / "deploy" / "compose" / "docker-compose.yml"
COMPOSE_PROJECT = "llmep-proof"
POSTGRES_PORT = "5434"
REDIS_PORT = "6381"
API_PORT = "18081"
API_BASE = f"http://127.0.0.1:{API_PORT}"
API_KEY = "proof-async-key"
SCHEMA_ID = "proof_async"
MODELS_YAML = ROOT / "proof" / "fixtures" / "models.async-proof.yaml"
SCHEMAS_DIR = ROOT / "proof" / "fixtures" / "schemas"
WORKER_LOG = ARTIFACT_DIR / "async_worker_log.txt"
SERVER_LOG = ARTIFACT_DIR / "async_server_log.txt"


def fail(message: str) -> None:
    raise RuntimeError(message)


def run(args: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=check,
        env=env,
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def http_request(method: str, path: str, body: bytes | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(f"{API_BASE}{path}", data=body, method=method)
    req.add_header("X-API-Key", API_KEY)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError:
        return 0, b""


def wait_tcp(host: str, port: int, timeout_seconds: float = 60.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    fail(f"timed out waiting for tcp {host}:{port}")


def wait_http(path: str, timeout_seconds: float = 60.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        code, _ = http_request("GET", path)
        if code == 200:
            return
        time.sleep(0.5)
    fail(f"timed out waiting for {path}")


def compose_env() -> dict[str, str]:
    return {
        **dict(os.environ),
        "POSTGRES_HOST_PORT": POSTGRES_PORT,
        "REDIS_HOST_PORT": REDIS_PORT,
    }


def compose_cmd(*parts: str) -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "-p",
        COMPOSE_PROJECT,
        "--profile",
        "infra-host",
        *parts,
    ]


def proof_env() -> dict[str, str]:
    return {
        **dict(os.environ),
        "APP_ROOT": str(ROOT),
        "APP_PROFILE": "test",
        "MODELS_PROFILE": "async-proof",
        "APP_CONFIG_PATH": "config/server.yaml",
        "MODELS_YAML": str(MODELS_YAML),
        "SCHEMAS_DIR": str(SCHEMAS_DIR),
        "DATABASE_URL": f"postgresql+asyncpg://llm:llm@127.0.0.1:{POSTGRES_PORT}/llm",
        "REDIS_URL": f"redis://127.0.0.1:{REDIS_PORT}/0",
        "REDIS_ENABLED": "1",
        "ENABLE_EXTRACT": "1",
        "ENABLE_GENERATE": "1",
        "PYTHONUNBUFFERED": "1",
    }


@contextmanager
def managed_process(args: list[str], *, env: dict[str, str], log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            args,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        try:
            yield proc
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def seed_api_key(env: dict[str, str]) -> None:
    raw_db_url = env["DATABASE_URL"]
    seed_db_url = raw_db_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    code = (
        "import asyncio\n"
        "import asyncpg\n"
        f"DATABASE_URL={seed_db_url!r}\n"
        f"API_KEY={API_KEY!r}\n"
        "async def main():\n"
        "    conn = await asyncpg.connect(DATABASE_URL)\n"
        "    try:\n"
        "        await conn.execute(\"INSERT INTO api_keys (key, active, quota_monthly, quota_used, created_at) VALUES ($1, true, NULL, 0, now()) ON CONFLICT (key) DO UPDATE SET active = EXCLUDED.active\", API_KEY)\n"
        "    finally:\n"
        "        await conn.close()\n"
        "asyncio.run(main())\n"
    )
    run(["uv", "run", "--project", "server", "python", "-c", code], env=env)


def generate_async_extract_proof() -> None:
    for binary in ("docker", "uv"):
        if shutil.which(binary) is None:
            fail(f"missing required binary: {binary}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cenv = compose_env()
    penv = proof_env()

    run(compose_cmd("up", "-d", "postgres_host", "redis_host"), env=cenv)
    wait_tcp("127.0.0.1", int(POSTGRES_PORT))
    wait_tcp("127.0.0.1", int(REDIS_PORT))

    run(
        ["uv", "run", "--project", "server", "python", "-m", "alembic", "-c", "server/alembic.ini", "upgrade", "head"],
        env=penv,
    )
    seed_api_key(penv)

    with managed_process(
        ["uv", "run", "--project", "server", "python", "-m", "uvicorn", "llm_server.main:app", "--host", "127.0.0.1", "--port", API_PORT],
        env=penv,
        log_path=SERVER_LOG,
    ) as server_proc:
        wait_http("/healthz")
        with managed_process(
            ["uv", "run", "--project", "server", "python", "-m", "llm_server.worker.extract_jobs"],
            env=penv,
            log_path=WORKER_LOG,
        ):
            submit_payload = {
                "schema_id": SCHEMA_ID,
                "text": "id 1",
                "cache": False,
                "repair": True,
            }
            code, body = http_request(
                "POST",
                "/v1/extract/jobs",
                body=json.dumps(submit_payload).encode("utf-8"),
            )
            submit_json = json.loads(body.decode("utf-8"))
            write_json(ARTIFACT_DIR / "async_submit_response.json", {"status_code": code, "body": submit_json})
            if code != 202:
                fail(f"async submit returned {code}")

            job_path = submit_json["poll_path"]
            code, body = http_request("GET", job_path)
            initial_json = json.loads(body.decode("utf-8"))
            write_json(ARTIFACT_DIR / "async_job_initial.json", {"status_code": code, "body": initial_json})
            if code != 200:
                fail(f"initial job read returned {code}")
            if initial_json["status"] not in {"queued", "running", "succeeded"}:
                fail(f"unexpected initial job status: {initial_json['status']}")

            deadline = time.time() + 30
            final_json = None
            final_code = None
            while time.time() < deadline:
                final_code, body = http_request("GET", job_path)
                if final_code == 200:
                    data = json.loads(body.decode("utf-8"))
                    if data["status"] in {"succeeded", "failed"}:
                        final_json = data
                        break
                time.sleep(0.5)
            if final_json is None or final_code != 200:
                fail("timed out waiting for final async job state")
            write_json(ARTIFACT_DIR / "async_job_final.json", {"status_code": final_code, "body": final_json})
            if final_json["status"] != "succeeded":
                fail(f"final async job status was {final_json['status']}")
            result = final_json.get("result") or {}
            if result.get("id") != "1":
                fail(f"unexpected async extract result: {result}")

    worker_log = WORKER_LOG.read_text(encoding="utf-8") if WORKER_LOG.exists() else ""
    server_proc_return = server_proc.poll() if server_proc else None
    summary = {
        "proof_phase": "phase6_extract_async",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "job_id": submit_json["job_id"],
        "submission_status": submit_json["status"],
        "final_status": final_json["status"],
        "worker_claimed": submit_json["job_id"] in worker_log,
        "result_valid": True,
        "schema_id": SCHEMA_ID,
        "resolved_model_id": final_json.get("model"),
        "server_exit_code": server_proc_return,
    }
    if not summary["worker_claimed"]:
        fail("worker log did not include job id")
    write_json(ARTIFACT_DIR / "async_job_summary.json", summary)


if __name__ == "__main__":
    generate_async_extract_proof()
