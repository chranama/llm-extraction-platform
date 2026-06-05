from __future__ import annotations

import argparse
import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from cli.errors import CLIError
from cli.types import GlobalConfig  # type: ignore[attr-defined]
from cli.utils.compose_runner import (
    ComposeContext,
    build_compose_context,
    compose_config_check,
    compose_down,
    compose_exec,
    compose_ps,
    compose_up,
)
from cli.utils.env import load_dotenv_file
from cli.utils.proc import ensure_bins, run

COMPOSE_INFRA_SERVER = ["infra", "server"]
COMPOSE_EXTRACT_PROFILES = ["infra", "llama", "server-llama"]
COMPOSE_EXTRACT_WORKER_PROFILES = ["infra", "llama", "worker-llama"]
COMPOSE_EXTERNAL_MODEL_PROFILES = ["infra", "server-llama-host"]
COMPOSE_STOP_PROFILES = [
    "infra",
    "infra-host",
    "llama",
    "itest",
    "server",
    "server-llama",
    "worker-llama",
    "server-llama-host",
    "server-gpu",
    "ui",
    "admin",
    "obs",
    "obs-host",
    "otel-host",
    "proxy",
    "eval",
    "eval-host",
    "policy",
]
RECEIPT_TEXT = "ACME STORE\n123 MAIN ST\nDATE: 2024-03-10\nTOTAL: $42.18"


def register(sub: argparse._SubParsersAction) -> None:
    smoke = sub.add_parser("smoke", help="Run the reviewer smoke path with a fake extract backend.")
    _add_defaults_args(smoke)
    smoke.add_argument(
        "--skip-verify", action="store_true", help="Start services without API probes."
    )
    smoke.set_defaults(_handler=_handle, path_cmd="smoke")

    extract = sub.add_parser(
        "compose-extract",
        help="Run the Compose extract path with containerized llama.cpp.",
    )
    _add_defaults_args(extract)
    extract.add_argument(
        "--skip-verify", action="store_true", help="Start services without API probes."
    )
    extract.add_argument("--skip-async", action="store_true", help="Skip async extract job probe.")
    extract.set_defaults(_handler=_handle, path_cmd="compose-extract")

    external = sub.add_parser(
        "external-model",
        help="Run a containerized server against an external host model runtime.",
    )
    _add_defaults_args(external)
    external.add_argument(
        "--skip-verify", action="store_true", help="Start services without API probes."
    )
    external.set_defaults(_handler=_handle, path_cmd="external-model")

    kind = sub.add_parser("kind-smoke", help="Run the local kind generate-only proof path.")
    kind.set_defaults(_handler=_handle, path_cmd="kind-smoke")

    policy_eval = sub.add_parser(
        "policy-eval",
        help="Run the policy/eval linkage proof path.",
    )
    policy_eval.set_defaults(_handler=_handle, path_cmd="policy-eval")

    admin_trace = sub.add_parser(
        "admin-trace",
        help="Run the admin trace inspection proof path.",
    )
    admin_trace.set_defaults(_handler=_handle, path_cmd="admin-trace")

    ops_surface = sub.add_parser(
        "ops-surface",
        help="Run the UI, observability, and proxy proof path.",
    )
    ops_surface.set_defaults(_handler=_handle, path_cmd="ops-surface")

    evidence = sub.add_parser("evidence", help="Validate or regenerate the curated proof bundle.")
    evidence.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate canonical artifacts instead of only validating saved artifacts.",
    )
    evidence.set_defaults(_handler=_handle, path_cmd="evidence")

    doctor = sub.add_parser("doctor", help="Inspect the currently running Compose stack.")
    doctor.set_defaults(_handler=_handle, path_cmd="doctor")

    stop = sub.add_parser("stop", help="Stop the current Compose project.")
    stop.add_argument("--volumes", action="store_true", help="Also remove named volumes.")
    stop.set_defaults(_handler=_handle, path_cmd="stop")


def _add_defaults_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--defaults-profile",
        default=None,
        help="Override compose defaults profile(s) for this invocation.",
    )
    parser.add_argument(
        "--defaults-yaml",
        default=None,
        help="Override compose defaults YAML (default: config/compose-defaults.yaml).",
    )


def _handle(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    cmd = args.path_cmd
    if cmd == "smoke":
        return _run_smoke(cfg, args)
    if cmd == "compose-extract":
        return _run_compose_extract(cfg, args)
    if cmd == "external-model":
        return _run_external_model(cfg, args)
    if cmd == "kind-smoke":
        return _run_kind_smoke(cfg, args)
    if cmd == "policy-eval":
        return _run_policy_eval(cfg, args)
    if cmd == "admin-trace":
        return _run_admin_trace(cfg, args)
    if cmd == "ops-surface":
        return _run_ops_surface(cfg, args)
    if cmd == "evidence":
        return _run_evidence(cfg, args)
    if cmd == "doctor":
        return _run_doctor(cfg, args)
    if cmd == "stop":
        return _run_stop(cfg, args)
    raise CLIError(f"Unknown happy-path command: {cmd}", code=2)


def _compose_context(
    cfg: GlobalConfig,
    args: argparse.Namespace,
    *,
    default_profile: str,
) -> ComposeContext:
    return build_compose_context(
        cfg,
        defaults_profile=getattr(args, "defaults_profile", None) or default_profile,
        defaults_yaml=getattr(args, "defaults_yaml", None),
    )


def _effective_env(ctx: ComposeContext) -> dict[str, str]:
    env: dict[str, str] = {}
    if ctx.env_files:
        env.update(load_dotenv_file(ctx.env_files[0]))
    for key in ("API_KEY", "ADMIN_API_KEY", "LLAMA_SERVER_API_KEY"):
        value = ctx.proc_env.get(key) or os.environ.get(key)
        if value:
            env.setdefault(key, value)
    return env


def _require_env(env: Mapping[str, str], keys: list[str]) -> None:
    missing = [key for key in keys if not str(env.get(key) or "").strip()]
    if missing:
        raise CLIError(f"Missing required env value(s): {', '.join(missing)}", code=2)


def _host_llama_model_path(env: Mapping[str, str]) -> Path | None:
    models_dir = str(env.get("LLAMA_MODELS_DIR") or "").strip()
    model_file = str(env.get("LLAMA_MODEL_FILE") or "").strip()
    if not models_dir or not model_file:
        return None

    if model_file.startswith("/models/"):
        return Path(models_dir).expanduser() / model_file.removeprefix("/models/")

    candidate = Path(model_file).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate
    return Path(models_dir).expanduser() / model_file


def _validate_compose_extract_env(env: Mapping[str, str]) -> None:
    _require_env(env, ["API_KEY", "LLAMA_MODELS_DIR", "LLAMA_MODEL_FILE"])
    model_path = _host_llama_model_path(env)
    if model_path is None or not model_path.exists():
        display_path = str(model_path) if model_path is not None else "<unresolved>"
        raise CLIError(
            "Containerized llama.cpp extract requires a local GGUF model file. "
            f"Could not find: {display_path}",
            code=2,
        )


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _seed_api_keys(ctx: ComposeContext, env: Mapping[str, str], *, verbose: bool) -> None:
    api_key = str(env.get("API_KEY") or "").strip()
    admin_key = str(env.get("ADMIN_API_KEY") or api_key).strip()
    _require_env({"API_KEY": api_key}, ["API_KEY"])

    user = str(env.get("POSTGRES_USER") or "llm").strip() or "llm"
    db = str(env.get("POSTGRES_DB") or "llm").strip() or "llm"
    sql = "\n".join(
        [
            "INSERT INTO roles (name, created_at) VALUES ('admin', now()) "
            "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name;",
            "INSERT INTO api_keys (key, active, quota_monthly, quota_used, created_at) "
            f"VALUES ({_sql_literal(api_key)}, true, NULL, 0, now()) "
            "ON CONFLICT (key) DO UPDATE SET active = EXCLUDED.active;",
            "INSERT INTO api_keys (key, active, quota_monthly, quota_used, created_at, role_id) "
            f"SELECT {_sql_literal(admin_key)}, true, NULL, 0, now(), id FROM roles WHERE name = 'admin' "
            "ON CONFLICT (key) DO UPDATE SET active = EXCLUDED.active, role_id = EXCLUDED.role_id;",
        ]
    )
    compose_exec(
        ctx,
        service="postgres",
        cmd=["psql", "-U", user, "-d", db, "-v", "ON_ERROR_STOP=1", "-c", sql],
        tty=False,
        verbose=verbose,
    )


def _migrate(ctx: ComposeContext, *, services: list[str], verbose: bool) -> None:
    last_error: Exception | None = None
    for service in services:
        try:
            compose_exec(
                ctx,
                service=service,
                cmd=[
                    "python",
                    "-m",
                    "alembic",
                    "-c",
                    "/app/server/alembic.ini",
                    "upgrade",
                    "head",
                ],
                tty=False,
                verbose=verbose,
            )
            return
        except Exception as exc:
            last_error = exc
    raise CLIError(
        f"Database migrations failed for services: {', '.join(services)}", code=2
    ) from last_error


def _api_base(cfg: GlobalConfig) -> str:
    return f"http://127.0.0.1:{cfg.api_port}"


def _request_json(
    *,
    method: str,
    url: str,
    api_key: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=body, method=method)
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
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        return exc.code, parsed
    except urllib.error.URLError:
        return 0, {}
    except (ConnectionResetError, TimeoutError, socket.timeout, OSError):
        return 0, {}


def _wait_for_http(
    *,
    url: str,
    api_key: str | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_status = 0
    last_body: dict[str, Any] = {}
    while time.time() < deadline:
        last_status, last_body = _request_json(method="GET", url=url, api_key=api_key, timeout=10)
        if last_status == 200:
            return last_body
        time.sleep(1)
    raise CLIError(
        f"Timed out waiting for {url}; last status={last_status} body={last_body}", code=2
    )


def _verify_generate(api_base: str, api_key: str) -> dict[str, Any]:
    status, body = _request_json(
        method="POST",
        url=f"{api_base}/v1/generate",
        api_key=api_key,
        payload={"prompt": "smoke test", "max_new_tokens": 16, "temperature": 0.2},
        timeout=60,
    )
    if status != 200:
        raise CLIError(f"/v1/generate failed with HTTP {status}: {body}", code=2)
    return body


def _verify_extract(api_base: str, api_key: str) -> dict[str, Any]:
    status, body = _request_json(
        method="POST",
        url=f"{api_base}/v1/extract",
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
    if status != 200:
        raise CLIError(f"/v1/extract failed with HTTP {status}: {body}", code=2)
    data = body.get("data") or {}
    if not isinstance(data, dict) or not data:
        raise CLIError(f"/v1/extract returned no data object: {body}", code=2)
    return body


def _verify_schema_surfaces(api_base: str, api_key: str) -> dict[str, Any]:
    status, body = _request_json(
        method="GET",
        url=f"{api_base}/v1/schemas",
        api_key=api_key,
        timeout=30,
    )
    if status != 200:
        raise CLIError(f"/v1/schemas failed with HTTP {status}: {body}", code=2)
    if not isinstance(body, list) or not body:
        raise CLIError(f"/v1/schemas returned no schema entries: {body}", code=2)

    schema_ids = {str(item.get("schema_id") or "") for item in body if isinstance(item, dict)}
    schema_id = "sroie_receipt_v1"
    if schema_id not in schema_ids:
        raise CLIError(f"/v1/schemas did not include {schema_id}: {body}", code=2)

    detail_status, detail = _request_json(
        method="GET",
        url=f"{api_base}/v1/schemas/{schema_id}",
        api_key=api_key,
        timeout=30,
    )
    if detail_status != 200:
        raise CLIError(
            f"/v1/schemas/{schema_id} failed with HTTP {detail_status}: {detail}", code=2
        )
    if not isinstance(detail, dict) or detail.get("type") != "object":
        raise CLIError(f"/v1/schemas/{schema_id} returned invalid schema detail: {detail}", code=2)
    return {"schemas": body, "detail": detail}


def _verify_async_extract(api_base: str, api_key: str) -> dict[str, Any]:
    status, body = _request_json(
        method="POST",
        url=f"{api_base}/v1/extract/jobs",
        api_key=api_key,
        payload={
            "schema_id": "sroie_receipt_v1",
            "text": RECEIPT_TEXT,
            "cache": False,
            "repair": True,
        },
        timeout=30,
    )
    if status != 202:
        raise CLIError(f"/v1/extract/jobs failed with HTTP {status}: {body}", code=2)
    poll_path = str(body.get("poll_path") or "").strip()
    if not poll_path:
        raise CLIError(f"Async extract response missing poll_path: {body}", code=2)

    deadline = time.time() + 180
    final: dict[str, Any] = {}
    while time.time() < deadline:
        poll_status, final = _request_json(
            method="GET",
            url=f"{api_base}{poll_path}",
            api_key=api_key,
            timeout=30,
        )
        if poll_status == 200 and final.get("status") in {"succeeded", "failed"}:
            break
        time.sleep(1)
    if final.get("status") != "succeeded":
        raise CLIError(f"Async extract did not succeed: {final}", code=2)
    return final


def _run_smoke(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    ensure_bins("docker")
    ctx = _compose_context(cfg, args, default_profile="docker+reviewer-smoke")
    env = _effective_env(ctx)
    _require_env(env, ["API_KEY"])

    compose_config_check(ctx, profiles=COMPOSE_INFRA_SERVER, verbose=args.verbose)
    compose_up(
        ctx,
        profiles=COMPOSE_INFRA_SERVER,
        detach=True,
        build=True,
        remove_orphans=True,
        verbose=args.verbose,
    )
    _migrate(ctx, services=["server"], verbose=args.verbose)
    _seed_api_keys(ctx, env, verbose=args.verbose)

    if not args.skip_verify:
        api_base = _api_base(cfg)
        _wait_for_http(url=f"{api_base}/healthz", timeout_seconds=60)
        _wait_for_http(url=f"{api_base}/readyz", timeout_seconds=60)
        _verify_schema_surfaces(api_base, env["API_KEY"])
        _verify_generate(api_base, env["API_KEY"])
        _verify_extract(api_base, env["API_KEY"])

    print(f"reviewer smoke path running at {_api_base(cfg)}")
    print("inspect: docker compose ps / docker compose logs")
    return 0


def _run_compose_extract(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    ensure_bins("docker")
    ctx = _compose_context(cfg, args, default_profile="docker+llama+compose-extract")
    env = _effective_env(ctx)
    _validate_compose_extract_env(env)

    compose_config_check(ctx, profiles=COMPOSE_EXTRACT_PROFILES, verbose=args.verbose)
    compose_up(
        ctx,
        profiles=COMPOSE_EXTRACT_PROFILES,
        detach=True,
        build=True,
        remove_orphans=True,
        verbose=args.verbose,
    )
    _migrate(ctx, services=["server_llama"], verbose=args.verbose)
    _seed_api_keys(ctx, env, verbose=args.verbose)
    compose_up(
        ctx,
        profiles=COMPOSE_EXTRACT_WORKER_PROFILES,
        detach=True,
        build=True,
        remove_orphans=False,
        verbose=args.verbose,
    )

    if not args.skip_verify:
        api_base = _api_base(cfg)
        _wait_for_http(url=f"{api_base}/healthz", timeout_seconds=120)
        _wait_for_http(url=f"{api_base}/readyz", timeout_seconds=180)
        _wait_for_http(
            url=f"{api_base}/v1/models/status", api_key=env["API_KEY"], timeout_seconds=120
        )
        _verify_generate(api_base, env["API_KEY"])
        _verify_extract(api_base, env["API_KEY"])
        if not args.skip_async:
            _verify_async_extract(api_base, env["API_KEY"])

    print(f"compose extract path running at {_api_base(cfg)}")
    print("model backend: containerized llama.cpp / SmolLM2 / CPU")
    print("inspect: docker compose ps / docker compose logs server_llama llama_server worker_llama")
    return 0


def _run_external_model(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    ensure_bins("docker")
    ctx = _compose_context(cfg, args, default_profile="docker+models-llama")
    env = _effective_env(ctx)
    _require_env(env, ["API_KEY"])

    compose_config_check(ctx, profiles=COMPOSE_EXTERNAL_MODEL_PROFILES, verbose=args.verbose)
    compose_up(
        ctx,
        profiles=COMPOSE_EXTERNAL_MODEL_PROFILES,
        detach=True,
        build=True,
        remove_orphans=True,
        verbose=args.verbose,
    )
    _migrate(ctx, services=["server_llama_host"], verbose=args.verbose)
    _seed_api_keys(ctx, env, verbose=args.verbose)

    if not args.skip_verify:
        api_base = _api_base(cfg)
        _wait_for_http(url=f"{api_base}/healthz", timeout_seconds=120)
        _wait_for_http(url=f"{api_base}/readyz", timeout_seconds=180)
        _verify_generate(api_base, env["API_KEY"])

    print(f"external model path running at {_api_base(cfg)}")
    print("model backend: host or Docker-managed OpenAI-compatible runtime")
    return 0


def _run_kind_smoke(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    ensure_bins("python")
    run(
        ["python", str(cfg.repo_root / "proof" / "generate_k8s_kind_proof.py")],
        verbose=args.verbose,
    )
    return 0


def _run_proof_script(cfg: GlobalConfig, args: argparse.Namespace, script_name: str) -> int:
    ensure_bins("python")
    run(["python", str(cfg.repo_root / "proof" / script_name)], verbose=args.verbose)
    return 0


def _run_policy_eval(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    return _run_proof_script(cfg, args, "generate_policy_eval_linkage_proof.py")


def _run_admin_trace(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    return _run_proof_script(cfg, args, "generate_trace_inspection_proof.py")


def _run_ops_surface(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    return _run_proof_script(cfg, args, "generate_ops_surface_proof.py")


def _run_evidence(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    script = (
        "generate_canonical_manifest.py" if args.regenerate else "validate_evidence_manifest.py"
    )
    run(["python", str(cfg.repo_root / "proof" / script)], verbose=args.verbose)
    return 0


def _run_doctor(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    if not cfg.compose_doctor.exists():
        raise CLIError(f"compose doctor not found: {cfg.compose_doctor}", code=2)
    env = {
        "API_PORT": cfg.api_port,
        "UI_PORT": cfg.ui_port,
        "PGADMIN_PORT": cfg.pgadmin_port,
        "PROM_PORT": cfg.prom_port,
        "GRAFANA_PORT": cfg.grafana_port,
        "PROM_HOST_PORT": cfg.prom_host_port,
        "ENV_FILE": str(cfg.env_override_file or ""),
        "COMPOSE_YML": str(cfg.compose_yml),
    }
    run(["bash", str(cfg.compose_doctor)], env=env, verbose=args.verbose)
    return 0


def _run_stop(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    ensure_bins("docker")
    ctx = build_compose_context(cfg)
    compose_down(
        ctx,
        profiles=COMPOSE_STOP_PROFILES,
        volumes=bool(args.volumes),
        remove_orphans=True,
        verbose=args.verbose,
    )
    compose_ps(ctx, profiles=COMPOSE_STOP_PROFILES, extra_args=None, verbose=args.verbose)
    return 0
