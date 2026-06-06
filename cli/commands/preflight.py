from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import yaml

from cli.commands import paths as path_cmds
from cli.errors import CLIError
from cli.types import GlobalConfig
from cli.utils.compose_runner import ComposeContext, build_compose_context, compose_config_check
from cli.utils.env import load_dotenv_file

Status = Literal["pass", "warn", "fail", "skip"]

TARGETS = (
    "smoke",
    "compose-extract",
    "external-model",
    "kind-smoke",
    "policy-eval",
    "admin-trace",
    "ops-surface",
    "evidence",
)


@dataclass(frozen=True)
class PreflightCheck:
    target: str
    name: str
    status: Status
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "target": self.target,
            "name": self.name,
            "status": self.status,
            "message": self.message,
        }
        if self.detail:
            row["detail"] = self.detail
        return row


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "preflight",
        help="Validate prerequisites for a supported runbook path without starting services.",
    )
    p.add_argument("target", choices=(*TARGETS, "all"))
    p.add_argument(
        "--defaults-profile",
        default=None,
        help="Override compose defaults profile(s) for this invocation.",
    )
    p.add_argument(
        "--defaults-yaml",
        default=None,
        help="Override compose defaults YAML (default: config/compose-defaults.yaml).",
    )
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as a nonzero result.",
    )
    p.add_argument(
        "--skip-port-check",
        action="store_true",
        help="Skip local TCP port availability checks.",
    )
    p.add_argument(
        "--probe-external-model",
        action="store_true",
        help="For external-model, perform an opt-in HTTP probe against LLAMA_SERVER_URL.",
    )
    p.set_defaults(_handler=_handle)


def _handle(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    requested = [args.target] if args.target != "all" else list(TARGETS)
    checks: list[PreflightCheck] = []
    for target in requested:
        checks.extend(_run_target_preflight(cfg, args, target))

    payload = _payload(args.target, checks)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human(args.target, checks)

    if any(check.status == "fail" for check in checks):
        return 2
    if args.strict and any(check.status == "warn" for check in checks):
        return 1
    return 0


def _payload(target: str, checks: Sequence[PreflightCheck]) -> dict[str, Any]:
    return {
        "target": target,
        "status": _overall_status(checks),
        "summary": _summary(checks),
        "checks": [check.to_dict() for check in checks],
    }


def _overall_status(checks: Sequence[PreflightCheck]) -> str:
    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "warn" for check in checks):
        return "warn"
    return "pass"


def _summary(checks: Sequence[PreflightCheck]) -> dict[str, int]:
    return {
        status: sum(1 for check in checks if check.status == status)
        for status in ("pass", "warn", "fail", "skip")
    }


def _print_human(target: str, checks: Sequence[PreflightCheck]) -> None:
    print(f"Preflight target: {target}")
    for check in checks:
        print(f"[{check.status.upper():4}] {check.target}: {check.name} - {check.message}")
    summary = _summary(checks)
    print(
        "Summary: "
        f"{summary['pass']} pass, {summary['warn']} warn, "
        f"{summary['fail']} fail, {summary['skip']} skip"
    )


def _run_target_preflight(
    cfg: GlobalConfig, args: argparse.Namespace, target: str
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    runners: dict[str, Callable[[GlobalConfig, argparse.Namespace, list[PreflightCheck]], None]] = {
        "smoke": _preflight_smoke,
        "compose-extract": _preflight_compose_extract,
        "external-model": _preflight_external_model,
        "kind-smoke": _preflight_kind_smoke,
        "policy-eval": _preflight_policy_eval,
        "admin-trace": _preflight_admin_trace,
        "ops-surface": _preflight_ops_surface,
        "evidence": _preflight_evidence,
    }
    runners[target](cfg, args, checks)
    return checks


def _add(
    checks: list[PreflightCheck],
    target: str,
    name: str,
    status: Status,
    message: str,
    detail: dict[str, Any] | None = None,
) -> None:
    checks.append(PreflightCheck(target, name, status, message, detail or {}))


def _check(
    checks: list[PreflightCheck],
    target: str,
    name: str,
    fn: Callable[[], tuple[str, dict[str, Any] | None] | str | None],
    *,
    warn: bool = False,
) -> None:
    try:
        result = fn()
        detail: dict[str, Any] | None = None
        if isinstance(result, tuple):
            message, detail = result
        elif isinstance(result, str):
            message = result
        else:
            message = "ok"
        _add(checks, target, name, "pass", message, detail)
    except Exception as exc:
        _add(checks, target, name, "warn" if warn else "fail", str(exc))


def _check_binary(checks: list[PreflightCheck], target: str, name: str) -> None:
    def run() -> tuple[str, dict[str, Any]]:
        path = shutil.which(name)
        if path is None:
            raise CLIError(f"missing required binary: {name}", code=2)
        return "found", {"path": path}

    _check(checks, target, f"binary:{name}", run)


def _check_binaries(checks: list[PreflightCheck], target: str, names: Iterable[str]) -> None:
    for name in names:
        _check_binary(checks, target, name)


def _check_docker_daemon(checks: list[PreflightCheck], target: str) -> None:
    def run() -> str:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "").strip()
            if len(message) > 300:
                message = message[-300:]
            raise CLIError(
                "docker daemon is not reachable"
                + (f": {message}" if message else ""),
                code=2,
            )
        return "reachable"

    _check(checks, target, "docker daemon", run)


def _check_file(
    checks: list[PreflightCheck],
    target: str,
    path: Path,
    label: str,
    *,
    executable: bool = False,
) -> None:
    def run() -> tuple[str, dict[str, Any]]:
        if not path.exists():
            raise CLIError(f"missing {label}: {path}", code=2)
        if executable and not path.is_file():
            raise CLIError(f"{label} is not a file: {path}", code=2)
        if executable and not path.stat().st_mode:
            raise CLIError(f"{label} is not readable: {path}", code=2)
        return "present", {"path": str(path)}

    _check(checks, target, label, run)


def _check_json_file(checks: list[PreflightCheck], target: str, path: Path, label: str) -> None:
    def run() -> tuple[str, dict[str, Any]]:
        if not path.exists():
            raise CLIError(f"missing {label}: {path}", code=2)
        json.loads(path.read_text(encoding="utf-8"))
        return "valid JSON", {"path": str(path)}

    _check(checks, target, label, run)


def _port_available(port: str) -> bool:
    value = int(str(port))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", value))
        except OSError:
            return False
    return True


def _check_port(checks: list[PreflightCheck], target: str, port: str, label: str) -> None:
    def run() -> tuple[str, dict[str, Any]]:
        if not _port_available(port):
            raise CLIError(f"port {port} is already in use ({label})", code=2)
        return "available", {"port": str(port), "label": label}

    _check(checks, target, f"port:{label}", run)


def _check_ports(
    checks: list[PreflightCheck],
    target: str,
    args: argparse.Namespace,
    ports: Sequence[tuple[str, str]],
) -> None:
    if getattr(args, "skip_port_check", False):
        for port, label in ports:
            _add(
                checks,
                target,
                f"port:{label}",
                "skip",
                "skipped by --skip-port-check",
                {"port": str(port), "label": label},
            )
        return
    for port, label in ports:
        _check_port(checks, target, port, label)


def _check_repo_basics(cfg: GlobalConfig, checks: list[PreflightCheck], target: str) -> None:
    _check_file(checks, target, cfg.compose_yml, "compose file")
    _check_file(checks, target, cfg.models_yaml, "models config")
    _check_file(
        checks,
        target,
        cfg.repo_root / "config" / "compose-defaults.yaml",
        "compose defaults config",
    )


def _check_schema_basics(cfg: GlobalConfig, checks: list[PreflightCheck], target: str) -> None:
    _check_json_file(
        checks,
        target,
        cfg.repo_root / "schemas" / "model_output" / "sroie_receipt_v1.json",
        "sroie receipt schema",
    )


def _check_models_profile(
    cfg: GlobalConfig, checks: list[PreflightCheck], target: str, profile: str
) -> None:
    def run() -> tuple[str, dict[str, Any]]:
        data = yaml.safe_load(cfg.models_yaml.read_text(encoding="utf-8")) or {}
        profiles = (data.get("profiles") or {}) if isinstance(data, dict) else {}
        if not isinstance(profiles, dict) or profile not in profiles:
            raise CLIError(f"models profile not found: {profile}", code=2)
        return "present", {"profile": profile}

    _check(checks, target, f"models profile:{profile}", run)


def _build_target_context(
    cfg: GlobalConfig,
    args: argparse.Namespace,
    *,
    target: str,
    default_profile: str,
) -> ComposeContext | None:
    try:
        return build_compose_context(
            cfg,
            defaults_profile=getattr(args, "defaults_profile", None) or default_profile,
            defaults_yaml=getattr(args, "defaults_yaml", None),
        )
    except Exception as exc:
        return _ContextFailure(str(exc), target)  # type: ignore[return-value]


class _ContextFailure:
    def __init__(self, message: str, target: str) -> None:
        self.message = message
        self.target = target


def _check_context(
    checks: list[PreflightCheck],
    target: str,
    ctx: ComposeContext | _ContextFailure | None,
) -> bool:
    if isinstance(ctx, _ContextFailure):
        _add(checks, target, "compose context", "fail", ctx.message)
        return False
    if ctx is None:
        _add(checks, target, "compose context", "fail", "compose context was not built")
        return False
    _add(
        checks,
        target,
        "compose context",
        "pass",
        "effective env rendered",
        {
            "defaults_profile": ctx.defaults_profile,
            "env_file": str(ctx.env_files[0]) if ctx.env_files else "",
        },
    )
    return True


def _context_env(ctx: ComposeContext) -> dict[str, str]:
    env: dict[str, str] = {}
    if ctx.env_files:
        env.update(load_dotenv_file(ctx.env_files[0]))
    for key in ("API_KEY", "ADMIN_API_KEY", "LLAMA_SERVER_API_KEY"):
        value = ctx.proc_env.get(key)
        if value and not str(env.get(key) or "").strip():
            env[key] = value
    return env


def _check_required_env(
    checks: list[PreflightCheck],
    target: str,
    env: Mapping[str, str],
    keys: Sequence[str],
) -> None:
    for key in keys:
        def run(key: str = key) -> str:
            if not str(env.get(key) or "").strip():
                raise CLIError(f"missing required env value: {key}", code=2)
            return "set"

        _check(checks, target, f"env:{key}", run)


def _check_compose_config(
    checks: list[PreflightCheck],
    target: str,
    ctx: ComposeContext,
    profiles: Sequence[str],
    verbose: bool,
) -> None:
    def run() -> tuple[str, dict[str, Any]]:
        compose_config_check(ctx, profiles=profiles, verbose=verbose)
        return "renders", {"profiles": list(profiles)}

    _check(checks, target, "compose config", run)


def _preflight_smoke(
    cfg: GlobalConfig, args: argparse.Namespace, checks: list[PreflightCheck]
) -> None:
    target = "smoke"
    _check_binaries(checks, target, ("docker",))
    _check_docker_daemon(checks, target)
    _check_repo_basics(cfg, checks, target)
    _check_schema_basics(cfg, checks, target)
    _check_models_profile(cfg, checks, target, "test")

    ctx = _build_target_context(
        cfg, args, target=target, default_profile="docker+reviewer-smoke"
    )
    if _check_context(checks, target, ctx) and isinstance(ctx, ComposeContext):
        env = _context_env(ctx)
        _check_required_env(checks, target, env, ("API_KEY",))
        _check_compose_config(checks, target, ctx, path_cmds.COMPOSE_INFRA_SERVER, args.verbose)

    _check_ports(checks, target, args, ((cfg.api_port, "api"),))


def _preflight_compose_extract(
    cfg: GlobalConfig, args: argparse.Namespace, checks: list[PreflightCheck]
) -> None:
    target = "compose-extract"
    _check_binaries(checks, target, ("docker",))
    _check_docker_daemon(checks, target)
    _check_repo_basics(cfg, checks, target)
    _check_schema_basics(cfg, checks, target)
    _check_models_profile(cfg, checks, target, "compose-extract")
    _check_json_file(
        checks,
        target,
        cfg.repo_root / "policy_out" / "local_extract_allow.json",
        "local extract allow policy",
    )

    ctx = _build_target_context(
        cfg, args, target=target, default_profile="docker+llama+compose-extract"
    )
    llama_publish_port = "8080"
    if _check_context(checks, target, ctx) and isinstance(ctx, ComposeContext):
        env = _context_env(ctx)
        llama_publish_port = str(env.get("LLAMA_PUBLISH_PORT") or llama_publish_port)
        _check_required_env(
            checks, target, env, ("API_KEY", "LLAMA_MODELS_DIR", "LLAMA_MODEL_FILE")
        )
        _check_llama_model_file(checks, target, env)
        _check_cpu_llama_setting(checks, target, env)
        _check_compose_config(
            checks, target, ctx, path_cmds.COMPOSE_EXTRACT_PROFILES, args.verbose
        )
        _check_compose_config(
            checks, target, ctx, path_cmds.COMPOSE_EXTRACT_WORKER_PROFILES, args.verbose
        )

    _check_ports(
        checks,
        target,
        args,
        (
            (cfg.api_port, "api"),
            (llama_publish_port, "llama-server"),
        ),
    )


def _check_llama_model_file(
    checks: list[PreflightCheck], target: str, env: Mapping[str, str]
) -> None:
    def run() -> tuple[str, dict[str, Any]]:
        model_path = path_cmds._host_llama_model_path(env)
        if model_path is None or not model_path.exists():
            display_path = str(model_path) if model_path is not None else "<unresolved>"
            raise CLIError(f"missing readable GGUF model file: {display_path}", code=2)
        if not model_path.is_file():
            raise CLIError(f"GGUF path is not a file: {model_path}", code=2)
        return "present", {"path": str(model_path)}

    _check(checks, target, "llama model file", run)


def _check_cpu_llama_setting(
    checks: list[PreflightCheck], target: str, env: Mapping[str, str]
) -> None:
    def run() -> str:
        value = str(env.get("LLAMA_N_GPU_LAYERS") or "").strip()
        if value != "0":
            raise CLIError(
                f"LLAMA_N_GPU_LAYERS should be 0 for promoted CPU-only compose-extract; got {value or '<unset>'}",
                code=2,
            )
        return "CPU-only llama.cpp setting is explicit"

    _check(checks, target, "llama CPU setting", run, warn=True)


def _preflight_external_model(
    cfg: GlobalConfig, args: argparse.Namespace, checks: list[PreflightCheck]
) -> None:
    target = "external-model"
    _check_binaries(checks, target, ("docker",))
    _check_docker_daemon(checks, target)
    _check_repo_basics(cfg, checks, target)
    _check_models_profile(cfg, checks, target, "llama-server")

    ctx = _build_target_context(cfg, args, target=target, default_profile="docker+models-llama")
    if _check_context(checks, target, ctx) and isinstance(ctx, ComposeContext):
        env = _context_env(ctx)
        _check_required_env(checks, target, env, ("API_KEY",))
        url = str(env.get("LLAMA_SERVER_URL") or "http://host.docker.internal:8080").strip()
        _add(
            checks,
            target,
            "external model URL",
            "pass",
            "resolved",
            {"url": url},
        )
        if getattr(args, "probe_external_model", False):
            _check_external_model_url(checks, target, url)
        else:
            _add(
                checks,
                target,
                "external model probe",
                "skip",
                "skipped unless --probe-external-model is set",
            )
        _check_compose_config(
            checks, target, ctx, path_cmds.COMPOSE_EXTERNAL_MODEL_PROFILES, args.verbose
        )

    _check_ports(checks, target, args, ((cfg.api_port, "api"),))


def _check_external_model_url(checks: list[PreflightCheck], target: str, url: str) -> None:
    def run() -> tuple[str, dict[str, Any]]:
        probe_url = url.rstrip("/") + "/health"
        request = urllib.request.Request(probe_url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status >= 400:
                    raise CLIError(f"external model returned HTTP {response.status}", code=2)
        except urllib.error.HTTPError as exc:
            raise CLIError(f"external model returned HTTP {exc.code}", code=2) from exc
        except urllib.error.URLError as exc:
            raise CLIError(f"external model is not reachable: {exc}", code=2) from exc
        return "reachable", {"url": probe_url}

    _check(checks, target, "external model probe", run)


def _preflight_kind_smoke(
    cfg: GlobalConfig, args: argparse.Namespace, checks: list[PreflightCheck]
) -> None:
    target = "kind-smoke"
    _check_binaries(checks, target, ("docker", "kind", "kubectl"))
    _check_docker_daemon(checks, target)
    _check_file(
        checks,
        target,
        cfg.repo_root / "deploy" / "k8s" / "kind" / "kind-config.yaml",
        "kind config",
    )
    _check_file(
        checks,
        target,
        cfg.repo_root / "deploy" / "docker" / "Dockerfile.server",
        "server Dockerfile",
    )
    _check_kustomize_overlay(
        checks,
        target,
        cfg.repo_root / "deploy" / "k8s" / "overlays" / "local-generate-only",
    )
    _check_kustomize_overlay(
        checks,
        target,
        cfg.repo_root / "deploy" / "k8s" / "overlays" / "prod-gpu-full",
    )
    _check_ports(checks, target, args, (("18080", "kind-port-forward"),))


def _check_kustomize_overlay(checks: list[PreflightCheck], target: str, overlay: Path) -> None:
    def run() -> tuple[str, dict[str, Any]]:
        if not overlay.exists():
            raise CLIError(f"missing kustomize overlay: {overlay}", code=2)
        commands = []
        if shutil.which("kustomize"):
            commands.append(["kustomize", "build", str(overlay)])
        if shutil.which("kubectl"):
            commands.append(["kubectl", "kustomize", str(overlay)])
        for command in commands:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                return "renders", {"overlay": str(overlay), "command": command[0]}
        raise CLIError(f"unable to render kustomize overlay: {overlay}", code=2)

    _check(checks, target, f"kustomize:{overlay.name}", run)


def _preflight_policy_eval(
    cfg: GlobalConfig, args: argparse.Namespace, checks: list[PreflightCheck]
) -> None:
    target = "policy-eval"
    _check_binaries(checks, target, ("docker", "uv"))
    _check_docker_daemon(checks, target)
    _check_proof_script(cfg, checks, target, "generate_policy_eval_linkage_proof.py")
    _check_async_fixture_files(cfg, checks, target)
    _check_ports(
        checks,
        target,
        args,
        (("5436", "postgres"), ("6383", "redis"), ("18084", "api")),
    )


def _preflight_admin_trace(
    cfg: GlobalConfig, args: argparse.Namespace, checks: list[PreflightCheck]
) -> None:
    target = "admin-trace"
    _check_binaries(checks, target, ("docker", "uv"))
    _check_docker_daemon(checks, target)
    _check_proof_script(cfg, checks, target, "generate_trace_inspection_proof.py")
    _check_async_fixture_files(cfg, checks, target)
    _check_ports(
        checks,
        target,
        args,
        (("5435", "postgres"), ("6382", "redis"), ("18082", "api")),
    )


def _preflight_ops_surface(
    cfg: GlobalConfig, args: argparse.Namespace, checks: list[PreflightCheck]
) -> None:
    target = "ops-surface"
    _check_binaries(checks, target, ("docker", "uv"))
    _check_docker_daemon(checks, target)
    _check_proof_script(cfg, checks, target, "generate_ops_surface_proof.py")
    _check_file(
        checks,
        target,
        cfg.repo_root
        / "deploy"
        / "observability"
        / "grafana"
        / "grafana"
        / "provisioning"
        / "datasources"
        / "datasources.yml",
        "Grafana datasource config",
    )
    _check_file(
        checks,
        target,
        cfg.repo_root / "deploy" / "proxy" / "nginx" / "nginx.compose.conf",
        "compose proxy config",
    )
    ctx = _build_target_context(
        cfg, args, target=target, default_profile="docker+reviewer-smoke"
    )
    if _check_context(checks, target, ctx) and isinstance(ctx, ComposeContext):
        _check_compose_config(
            checks,
            target,
            ctx,
            ["infra", "server", "ui", "obs", "proxy"],
            args.verbose,
        )
    _check_ports(
        checks,
        target,
        args,
        (
            ("18085", "api"),
            ("18086", "ui"),
            ("18087", "prometheus"),
            ("18088", "grafana"),
            ("18089", "proxy"),
        ),
    )


def _check_proof_script(
    cfg: GlobalConfig, checks: list[PreflightCheck], target: str, script_name: str
) -> None:
    _check_file(checks, target, cfg.repo_root / "proof" / script_name, f"proof script:{script_name}")


def _check_async_fixture_files(
    cfg: GlobalConfig, checks: list[PreflightCheck], target: str
) -> None:
    _check_file(
        checks,
        target,
        cfg.repo_root / "proof" / "fixtures" / "models.async-proof.yaml",
        "async proof models fixture",
    )
    _check_json_file(
        checks,
        target,
        cfg.repo_root / "proof" / "fixtures" / "schemas" / "proof_async.json",
        "async proof schema fixture",
    )


def _preflight_evidence(
    cfg: GlobalConfig, args: argparse.Namespace, checks: list[PreflightCheck]
) -> None:
    target = "evidence"
    _check_binaries(checks, target, ("python",))
    manifest = cfg.repo_root / "proof" / "evidence_manifest.latest.json"
    _check_json_file(checks, target, manifest, "evidence manifest")
    _check_json_file(
        checks,
        target,
        cfg.repo_root / "proof" / "evidence_contract.schema.json",
        "evidence contract schema",
    )
    _check_proof_script(cfg, checks, target, "validate_evidence_manifest.py")
    _check_manifest_artifacts(cfg, checks, target, manifest)


def _check_manifest_artifacts(
    cfg: GlobalConfig, checks: list[PreflightCheck], target: str, manifest: Path
) -> None:
    def run() -> tuple[str, dict[str, Any]]:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        missing: list[str] = []
        count = 0
        for claim in data.get("claims", []):
            if not isinstance(claim, dict):
                continue
            for rel in claim.get("artifact_paths", []):
                count += 1
                if not (cfg.repo_root / str(rel)).exists():
                    missing.append(str(rel))
        if missing:
            raise CLIError(f"missing manifest artifact path(s): {', '.join(missing)}", code=2)
        return "all referenced artifacts exist", {"artifact_count": count}

    _check(checks, target, "manifest artifact paths", run)
