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
ARTIFACT_DIR = ROOT / "proof" / "artifacts" / "phase7_trace_inspection"
COMPOSE_FILE = ROOT / "deploy" / "compose" / "docker-compose.yml"
COMPOSE_PROJECT = "llmep-trace-proof"
POSTGRES_PORT = "5435"
REDIS_PORT = "6382"
API_PORT = "18082"
API_BASE = f"http://127.0.0.1:{API_PORT}"
USER_API_KEY = "proof-trace-key"
ADMIN_API_KEY = "proof-trace-admin"
SCHEMA_ID = "proof_async"
MODELS_YAML = ROOT / "proof" / "fixtures" / "models.async-proof.yaml"
SCHEMAS_DIR = ROOT / "proof" / "fixtures" / "schemas"
WORKER_LOG = ARTIFACT_DIR / "async_worker_log.txt"
SERVER_LOG = ARTIFACT_DIR / "trace_server_log.txt"


def fail(message: str) -> None:
    raise RuntimeError(message)


def run(
    args: list[str], *, env: dict[str, str] | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
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


def http_request(
    method: str,
    path: str,
    *,
    api_key: str,
    body: bytes | None = None,
) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(f"{API_BASE}{path}", data=body, method=method)
    req.add_header("X-API-Key", api_key)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()
    except urllib.error.URLError:
        return 0, {}, b""


def header_value(headers: dict[str, str], name: str) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


def wait_tcp(host: str, port: int, timeout_seconds: float = 60.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    fail(f"timed out waiting for tcp {host}:{port}")


def wait_http(path: str, *, api_key: str = USER_API_KEY, timeout_seconds: float = 60.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        code, _, _ = http_request("GET", path, api_key=api_key)
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
        "EDGE_MODE": "behind_gateway",
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


def seed_api_keys(env: dict[str, str]) -> None:
    raw_db_url = env["DATABASE_URL"]
    seed_db_url = raw_db_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    code = (
        "import asyncio\n"
        "import asyncpg\n"
        f"DATABASE_URL={seed_db_url!r}\n"
        f"USER_API_KEY={USER_API_KEY!r}\n"
        f"ADMIN_API_KEY={ADMIN_API_KEY!r}\n"
        "async def main():\n"
        "    conn = await asyncpg.connect(DATABASE_URL)\n"
        "    try:\n"
        "        role_id = await conn.fetchval(\"INSERT INTO roles (name, created_at) VALUES ('admin', now()) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id\")\n"
        '        await conn.execute("INSERT INTO api_keys (key, active, quota_monthly, quota_used, created_at) VALUES ($1, true, NULL, 0, now()) ON CONFLICT (key) DO UPDATE SET active = EXCLUDED.active", USER_API_KEY)\n'
        '        await conn.execute("INSERT INTO api_keys (key, active, quota_monthly, quota_used, created_at, role_id) VALUES ($1, true, NULL, 0, now(), $2) ON CONFLICT (key) DO UPDATE SET active = EXCLUDED.active, role_id = EXCLUDED.role_id", ADMIN_API_KEY, role_id)\n'
        "    finally:\n"
        "        await conn.close()\n"
        "asyncio.run(main())\n"
    )
    run(["uv", "run", "--project", "server", "python", "-c", code], env=env)


def _write_timeline(path: Path, detail: dict) -> None:
    lines = [
        "# Async Trace Timeline",
        "",
        f"- Trace ID: `{detail['trace_id']}`",
        f"- Status: `{detail['status']}`",
        f"- Request Kind: `{detail['request_kind']}`",
        "",
        "| Time | Event | Stage | Status | Job | Model |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for event in detail.get("events", []):
        lines.append(
            "| {created_at} | {event_name} | {stage} | {status} | {job_id} | {model_id} |".format(
                created_at=event.get("created_at"),
                event_name=event.get("event_name"),
                stage=event.get("stage") or "",
                status=event.get("status"),
                job_id=event.get("job_id") or "",
                model_id=event.get("model_id") or "",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _event_names(detail: dict) -> list[str]:
    return [str(item.get("event_name")) for item in detail.get("events", [])]


def generate_trace_inspection_proof() -> None:
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
        [
            "uv",
            "run",
            "--project",
            "server",
            "python",
            "-m",
            "alembic",
            "-c",
            "server/alembic.ini",
            "upgrade",
            "head",
        ],
        env=penv,
    )
    seed_api_keys(penv)

    with managed_process(
        [
            "uv",
            "run",
            "--project",
            "server",
            "python",
            "-m",
            "uvicorn",
            "llm_server.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            API_PORT,
        ],
        env=penv,
        log_path=SERVER_LOG,
    ) as server_proc:
        wait_http("/healthz")
        with managed_process(
            ["uv", "run", "--project", "server", "python", "-m", "llm_server.worker.extract_jobs"],
            env=penv,
            log_path=WORKER_LOG,
        ):
            sync_payload = {"schema_id": SCHEMA_ID, "text": "id 1", "cache": False, "repair": True}
            sync_code, sync_headers, sync_body = http_request(
                "POST",
                "/v1/extract",
                api_key=USER_API_KEY,
                body=json.dumps(sync_payload).encode("utf-8"),
            )
            sync_json = json.loads(sync_body.decode("utf-8"))
            sync_trace_id = header_value(sync_headers, "X-Request-ID")
            write_json(
                ARTIFACT_DIR / "sync_extract_response.json",
                {
                    "status_code": sync_code,
                    "trace_id": sync_trace_id,
                    "body": sync_json,
                },
            )
            if sync_code != 200 or not sync_trace_id:
                fail(f"sync extract failed with status {sync_code}")

            sync_trace_code, _, sync_trace_body = http_request(
                "GET",
                f"/v1/admin/traces/{sync_trace_id}",
                api_key=ADMIN_API_KEY,
            )
            sync_trace_json = json.loads(sync_trace_body.decode("utf-8"))
            write_json(
                ARTIFACT_DIR / "sync_trace_detail.json",
                {"status_code": sync_trace_code, "body": sync_trace_json},
            )
            if sync_trace_code != 200:
                fail("sync trace detail fetch failed")

            async_payload = {"schema_id": SCHEMA_ID, "text": "id 1", "cache": False, "repair": True}
            async_code, _, async_body = http_request(
                "POST",
                "/v1/extract/jobs",
                api_key=USER_API_KEY,
                body=json.dumps(async_payload).encode("utf-8"),
            )
            async_submit_json = json.loads(async_body.decode("utf-8"))
            write_json(
                ARTIFACT_DIR / "async_submit_response.json",
                {"status_code": async_code, "body": async_submit_json},
            )
            if async_code != 202:
                fail(f"async submit returned {async_code}")

            job_path = async_submit_json["poll_path"]
            final_json = None
            deadline = time.time() + 30
            while time.time() < deadline:
                code, _, body = http_request("GET", job_path, api_key=USER_API_KEY)
                if code == 200:
                    data = json.loads(body.decode("utf-8"))
                    if data["status"] in {"succeeded", "failed"}:
                        final_json = data
                        break
                time.sleep(0.5)
            if final_json is None:
                fail("timed out waiting for async trace proof job completion")
            if final_json["status"] != "succeeded":
                fail(f"async trace proof ended with {final_json['status']}")

            async_trace_id = async_submit_json.get("trace_id")
            async_trace_code, _, async_trace_body = http_request(
                "GET",
                f"/v1/admin/traces/{async_trace_id}",
                api_key=ADMIN_API_KEY,
            )
            async_trace_json = json.loads(async_trace_body.decode("utf-8"))
            write_json(
                ARTIFACT_DIR / "async_trace_detail.json",
                {"status_code": async_trace_code, "body": async_trace_json},
            )
            if async_trace_code != 200:
                fail("async trace detail fetch failed")
            _write_timeline(ARTIFACT_DIR / "async_trace_timeline.md", async_trace_json)

    sync_names = _event_names(sync_trace_json)
    async_names = _event_names(async_trace_json)
    summary = {
        "proof_phase": "phase7_trace_inspection",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "sync_trace_complete": "extract.completed" in sync_names,
        "async_trace_complete": "extract_job.completed" in async_names,
        "async_worker_claimed": "extract_job.worker_claimed" in async_names,
        "async_status_polled": "extract_job.status_polled" in async_names,
        "sync_contains_generate_or_cache_path": (
            "extract.generate_completed" in sync_names or "extract.cache_lookup" in sync_names
        ),
        "trace_ids_present": bool(sync_trace_id) and bool(async_trace_id),
        "sync_trace_id": sync_trace_id,
        "async_trace_id": async_trace_id,
        "async_job_id": async_submit_json.get("job_id"),
        "server_exit_code": server_proc.poll() if server_proc else None,
    }
    if not all(
        bool(summary[key])
        for key in (
            "sync_trace_complete",
            "async_trace_complete",
            "async_worker_claimed",
            "async_status_polled",
            "sync_contains_generate_or_cache_path",
            "trace_ids_present",
        )
    ):
        fail(f"trace proof summary check failed: {summary}")
    write_json(ARTIFACT_DIR / "trace_summary.json", summary)


if __name__ == "__main__":
    generate_trace_inspection_proof()
