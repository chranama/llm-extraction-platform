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
ARTIFACT_DIR = ROOT / "proof" / "artifacts" / "phase9_policy_eval_linkage"
COMPOSE_FILE = ROOT / "deploy" / "compose" / "docker-compose.yml"
COMPOSE_PROJECT = "llmep-policy-eval-proof"
POSTGRES_PORT = "5436"
REDIS_PORT = "6383"
API_PORT = "18084"
API_BASE = f"http://127.0.0.1:{API_PORT}"
USER_API_KEY = "proof-policy-key"
ADMIN_API_KEY = "proof-policy-admin"
SCHEMA_ID = "proof_async"
MODELS_YAML = ROOT / "proof" / "fixtures" / "models.async-proof.yaml"
SCHEMAS_DIR = ROOT / "proof" / "fixtures" / "schemas"
RUNTIME_POLICY = ARTIFACT_DIR / "runtime_policy.json"
SERVER_LOG = ARTIFACT_DIR / "policy_eval_server_log.txt"


def fail(message: str) -> None:
    raise RuntimeError(message)


def run(
    args: list[str], *, env: dict[str, str] | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if check and result.returncode != 0:
        fail(
            "command failed: {cmd}\nreturncode={code}\nstdout:\n{stdout}\nstderr:\n{stderr}".format(
                cmd=" ".join(args),
                code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
    return result


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


def wait_postgres(env: dict[str, str], timeout_seconds: float = 60.0) -> None:
    db_url = env["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://", 1)
    code = (
        "import asyncio\n"
        "import asyncpg\n"
        f"DATABASE_URL={db_url!r}\n"
        "async def main():\n"
        "    conn = await asyncpg.connect(DATABASE_URL)\n"
        "    await conn.close()\n"
        "asyncio.run(main())\n"
    )
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        result = run(
            ["uv", "run", "--project", "server", "python", "-c", code], env=env, check=False
        )
        if result.returncode == 0:
            return
        time.sleep(1)
    fail("timed out waiting for Postgres readiness")


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
        "POLICY_DECISION_PATH": str(RUNTIME_POLICY),
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


def _eval_summary(*, run_id: str, passed: bool) -> dict:
    return {
        "schema_version": "eval_run_summary_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "extract",
        "run_id": run_id,
        "run_dir": str(ARTIFACT_DIR / run_id),
        "passed": passed,
        "metrics": {
            "extract_gate": {"passed": passed},
            "n_total": 2,
            "n_ok": 2 if passed else 1,
            "schema_validity_rate": 99.0 if passed else 60.0,
            "non_200_rate": 0.0 if passed else 10.0,
            "http_5xx_rate": 0.0 if passed else 5.0,
            "timeout_rate": 0.0 if passed else 5.0,
        },
        "model_id": "fake-extract",
        "schema_id": SCHEMA_ID,
        "thresholds_profile": "extract/default",
        "thresholds_version": "proof",
        "deployment_key": "async-proof--fake-extract",
        "deployment": {"provider": "fake", "profile": "async-proof"},
        "counts": {
            "examples_total": 2,
            "examples_ok": 2 if passed else 1,
            "examples_failed": 0 if passed else 1,
        },
        "warnings": [],
        "notes": {"fixture": "phase9_policy_eval_linkage"},
    }


def write_eval_fixture(run_id: str, *, passed: bool) -> Path:
    run_dir = ARTIFACT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "summary.json", _eval_summary(run_id=run_id, passed=passed))
    rows = []
    for idx in range(2):
        ok = passed or idx == 0
        rows.append(
            {
                "id": str(idx + 1),
                "ok": ok,
                "status_code": 200 if ok else 500,
                "schema_valid": ok,
                "deployment_key": "async-proof--fake-extract",
                "deployment": {"provider": "fake", "profile": "async-proof"},
            }
        )
    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return run_dir


def run_policy(run_dir: Path, *, out_json: Path, out_md: Path, expect_code: int) -> dict:
    result = run(
        [
            "uv",
            "run",
            "--project",
            "policy",
            "policy",
            "runtime-decision",
            "--pipeline",
            "extract_only",
            "--run-dir",
            str(run_dir),
            "--artifact-out",
            str(out_json),
            "--report",
            "md",
            "--report-out",
            str(out_md),
        ],
        check=False,
    )
    write_json(
        out_json.with_suffix(".command.json"),
        {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "expected_returncode": expect_code,
        },
    )
    if result.returncode != expect_code:
        fail(f"policy command returned {result.returncode}, expected {expect_code}")
    return json.loads(out_json.read_text(encoding="utf-8"))


def _json_response(path: Path, status_code: int, body: bytes) -> dict:
    try:
        parsed = json.loads(body.decode("utf-8")) if body.strip() else {}
    except json.JSONDecodeError:
        parsed = {"raw": body.decode("utf-8", errors="replace")}
    payload = {"status_code": status_code, "body": parsed}
    write_json(path, payload)
    return payload


def _extract_payload() -> bytes:
    return json.dumps(
        {"schema_id": SCHEMA_ID, "text": "id 1", "cache": False, "repair": True}
    ).encode("utf-8")


def generate_policy_eval_linkage_proof() -> None:
    for binary in ("docker", "uv"):
        if shutil.which(binary) is None:
            fail(f"missing required binary: {binary}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    pass_run = write_eval_fixture("eval_pass", passed=True)
    fail_run = write_eval_fixture("eval_fail", passed=False)
    allow_policy = ARTIFACT_DIR / "policy_allow.json"
    deny_policy = ARTIFACT_DIR / "policy_deny.json"

    allow = run_policy(
        pass_run,
        out_json=allow_policy,
        out_md=ARTIFACT_DIR / "policy_allow.md",
        expect_code=0,
    )
    deny = run_policy(
        fail_run,
        out_json=deny_policy,
        out_md=ARTIFACT_DIR / "policy_deny.md",
        expect_code=2,
    )
    if allow.get("ok") is not True or allow.get("enable_extract") is not True:
        fail("allow policy did not enable extract")
    if deny.get("ok") is not False or deny.get("enable_extract") is not False:
        fail("deny policy did not disable extract")

    cenv = compose_env()
    penv = proof_env()
    shutil.copyfile(allow_policy, RUNTIME_POLICY)

    try:
        run(compose_cmd("up", "-d", "postgres_host", "redis_host"), env=cenv)
        wait_tcp("127.0.0.1", int(POSTGRES_PORT))
        wait_tcp("127.0.0.1", int(REDIS_PORT))
        wait_postgres(penv)

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
        ):
            wait_http("/healthz")

            code, _, body = http_request("GET", "/v1/admin/policy", api_key=ADMIN_API_KEY)
            initial_policy = _json_response(ARTIFACT_DIR / "admin_policy_initial.json", code, body)
            if code != 200 or initial_policy["body"].get("enable_extract") is not True:
                fail("initial admin policy did not expose allow decision")

            code, _, body = http_request(
                "POST", "/v1/extract", api_key=USER_API_KEY, body=_extract_payload()
            )
            allow_extract = _json_response(ARTIFACT_DIR / "extract_allow_response.json", code, body)
            if code != 200:
                fail("extract should pass under allow policy")

            shutil.copyfile(deny_policy, RUNTIME_POLICY)
            code, _, body = http_request("POST", "/v1/admin/policy/reload", api_key=ADMIN_API_KEY)
            reload_policy = _json_response(ARTIFACT_DIR / "admin_policy_reload.json", code, body)
            if code != 200 or reload_policy["body"].get("enable_extract") is not False:
                fail("admin policy reload did not expose deny decision")

            code, _, body = http_request(
                "POST", "/v1/extract", api_key=USER_API_KEY, body=_extract_payload()
            )
            deny_extract = _json_response(ARTIFACT_DIR / "extract_deny_response.json", code, body)
            if code == 200:
                fail("extract should be blocked under deny policy")
    finally:
        run(compose_cmd("down", "--remove-orphans", "-v"), env=cenv, check=False)

    summary = {
        "proof_phase": "phase9_policy_eval_linkage",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "policy_cli_allow_returncode": 0,
        "policy_cli_deny_returncode": 2,
        "allow_eval_run_dir": str(pass_run.relative_to(ROOT)),
        "deny_eval_run_dir": str(fail_run.relative_to(ROOT)),
        "allow_policy_ok": allow.get("ok"),
        "deny_policy_ok": deny.get("ok"),
        "admin_initial_enable_extract": initial_policy["body"].get("enable_extract"),
        "admin_reload_enable_extract": reload_policy["body"].get("enable_extract"),
        "extract_allow_status_code": allow_extract["status_code"],
        "extract_deny_status_code": deny_extract["status_code"],
        "runtime_policy_path": str(RUNTIME_POLICY.relative_to(ROOT)),
    }
    write_json(ARTIFACT_DIR / "policy_eval_linkage_summary.json", summary)


if __name__ == "__main__":
    generate_policy_eval_linkage_proof()
