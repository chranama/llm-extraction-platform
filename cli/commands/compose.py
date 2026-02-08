# cli/commands/compose.py
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Sequence

from cli.errors import CLIError
from cli.types import GlobalConfig  # type: ignore[attr-defined]
from cli.utils.compose_config import render_compose_env_file
from cli.utils.proc import ensure_bins, run

_COMPOSE_VERBS = {
    "up",
    "down",
    "ps",
    "logs",
    "config",
    "build",
    "pull",
    "push",
    "restart",
    "stop",
    "start",
    "rm",
    "exec",
    "run",
    "kill",
    "pause",
    "unpause",
    "top",
    "events",
    "images",
    "ls",
    "port",
    "cp",
    "create",
}

_PASSTHROUGH_ENV = {
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "HF_TOKEN",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
}

# Critical: prevent shell env from silently overriding your deterministic defaults-profile wiring.
_DENYLIST_ENV = {
    # config selection / profiles (must be controlled by rendered defaults + user .env)
    "APP_PROFILE",
    "MODELS_PROFILE",

    # app behavior & model wiring (compose profile decides; server.yaml decides semantics)
    "MODELS_YAML",
    "MODEL_LOAD_MODE",
    "REQUIRE_MODEL_READY",
    "MODEL_ID",
    "MODEL_DEVICE",

    # infra wiring
    "DATABASE_URL",
    "REDIS_ENABLED",
    "REDIS_URL",

    # misc runtime toggles
    "ENV",
    "WORKERS",
    "UVICORN_RELOAD",

    # artifacts paths
    "POLICY_DECISION_PATH",
    "SLO_OUT_DIR",
}


def _clean_compose_process_env(cfg: GlobalConfig) -> dict[str, str]:
    env: dict[str, str] = {"COMPOSE_PROJECT_NAME": cfg.project_name}
    for k in _PASSTHROUGH_ENV:
        v = os.environ.get(k)
        if v:
            env[k] = v
    for k in list(env.keys()):
        if k in _DENYLIST_ENV:
            env.pop(k, None)
    return env


def _compose_base(cfg: GlobalConfig, *, env_files: list[Path]) -> list[str]:
    cmd: list[str] = ["docker", "compose"]
    for f in env_files:
        cmd += ["--env-file", str(f)]
    cmd += ["-f", str(cfg.compose_yml)]
    return cmd


def _add_profiles(cmd: list[str], profiles: Sequence[str]) -> list[str]:
    for p in profiles:
        cmd += ["--profile", p]
    return cmd


def _split_profiles_and_args(tokens: list[str]) -> tuple[list[str], list[str]]:
    if not tokens:
        return [], []

    if "--" in tokens:
        i = tokens.index("--")
        return tokens[:i], tokens[i + 1 :]

    for i, t in enumerate(tokens):
        if t in _COMPOSE_VERBS:
            return tokens[:i], tokens[i:]

    return tokens, []


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("compose", help="Direct docker compose control (replacement for just dc).")
    p.set_defaults(_handler=_handle)

    p.add_argument(
        "--defaults-profile",
        default=None,
        help="Override compose defaults profile(s) for this invocation (e.g. docker, host, itest, jobs, or docker+jobs).",
    )

    sp = p.add_subparsers(dest="compose_cmd", required=True)

    dc = sp.add_parser("dc", help="Compose with profiles + args.")
    dc.add_argument("tokens", nargs=argparse.REMAINDER)

    cfgp = sp.add_parser("config", help="docker compose config (validates compose).")
    cfgp.add_argument("--profiles", nargs="*", default=[])

    psp = sp.add_parser("ps", help="docker compose ps")
    psp.add_argument("--profiles", nargs="*", default=[])
    psp.add_argument("args", nargs=argparse.REMAINDER)

    lg = sp.add_parser("logs", help="docker compose logs -f --tail=200 (default)")
    lg.add_argument("--profiles", nargs="*", default=[])
    lg.add_argument("--follow", action="store_true")
    lg.add_argument("--tail", type=int, default=200)

    dn = sp.add_parser("down", help="docker compose down --remove-orphans")
    dn.add_argument("--profiles", nargs="*", default=[])
    dn.add_argument("--volumes", action="store_true")
    dn.add_argument("--remove-orphans", action="store_true")
    dn.set_defaults(remove_orphans=True)

    up = sp.add_parser("up", help="docker compose up")
    up.add_argument("--profiles", nargs="*", default=[])
    up.add_argument("-d", "--detach", action="store_true")
    up.add_argument("--build", action="store_true")
    up.add_argument("--remove-orphans", action="store_true")
    up.add_argument("args", nargs=argparse.REMAINDER)

    rm = sp.add_parser("rm-orphans", help="Shortcut: compose up -d --remove-orphans for a profile set")
    rm.add_argument("--profiles", nargs="*", default=[])

    infra = sp.add_parser("infra-up", help="Start postgres+redis (profile infra).")
    infra.set_defaults(_shortcut="infra-up")
    infra2 = sp.add_parser("infra-ps", help="Show infra status.")
    infra2.set_defaults(_shortcut="infra-ps")
    infra3 = sp.add_parser("infra-down", help="Stop infra.")
    infra3.add_argument("--volumes", action="store_true")
    infra3.set_defaults(_shortcut="infra-down")


def _resolve_defaults_yaml(cfg: GlobalConfig, args: argparse.Namespace) -> Path:
    rel = getattr(args, "compose_defaults_yaml", None) or "config/compose-defaults.yaml"
    p = Path(rel)
    return p if p.is_absolute() else (cfg.repo_root / p)


def _resolve_defaults_profile(args: argparse.Namespace) -> str:
    # allow compose subcommand override
    return (getattr(args, "defaults_profile", None) or getattr(args, "compose_defaults_profile", None) or "docker").strip()


def _render_defaults_env_file(cfg: GlobalConfig, args: argparse.Namespace) -> Path:
    defaults_yaml = _resolve_defaults_yaml(cfg, args)
    profile = _resolve_defaults_profile(args)

    tmp_dir = cfg.repo_root / ".tmp" / "llmctl"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    safe_suffix = profile.replace("/", "_").replace("+", "_").replace(",", "_")
    fd, path = tempfile.mkstemp(prefix=f"compose-defaults-{safe_suffix}-", suffix=".env", dir=str(tmp_dir))
    os.close(fd)
    out_path = Path(path)

    render_compose_env_file(
        config_yaml_path=defaults_yaml,
        profile=profile,  # can be "docker+jobs"
        out_env_path=out_path,
        extra_env={"COMPOSE_PROJECT_NAME": cfg.project_name},
    )
    return out_path


def _handle(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    ensure_bins("docker")

    proc_env = _clean_compose_process_env(cfg)
    rendered_defaults = _render_defaults_env_file(cfg, args)

    env_files = [rendered_defaults]
    if cfg.env_file.exists():
        env_files.append(cfg.env_file)

    def base_cmd() -> list[str]:
        return _compose_base(cfg, env_files=env_files)

    if getattr(args, "_shortcut", None) == "infra-up":
        cmd = _add_profiles(base_cmd(), ["infra"]) + ["up", "-d", "--remove-orphans"]
        run(cmd, env=proc_env, verbose=args.verbose, inherit_env=False)
        print("✅ infra up (postgres/redis).")
        return 0

    if getattr(args, "_shortcut", None) == "infra-ps":
        cmd = _add_profiles(base_cmd(), ["infra"]) + ["ps"]
        run(cmd, env=proc_env, verbose=args.verbose, inherit_env=False)
        return 0

    if getattr(args, "_shortcut", None) == "infra-down":
        cmd = _add_profiles(base_cmd(), ["infra"]) + ["down", "--remove-orphans"]
        if getattr(args, "volumes", False):
            cmd.append("-v")
        run(cmd, env=proc_env, verbose=args.verbose, inherit_env=False)
        return 0

    c = args.compose_cmd

    if c == "dc":
        tokens = list(getattr(args, "tokens", []) or [])
        profiles, extra = _split_profiles_and_args(tokens)
        if not extra:
            raise CLIError(
                "compose dc requires compose args. Examples:\n"
                "  llmctl compose dc infra server -- up -d --build\n"
                "  llmctl compose dc infra server up -d --build\n"
                "  llmctl compose dc infra -- ps"
            )
        cmd = _add_profiles(base_cmd(), profiles) + extra
        run(cmd, env=proc_env, verbose=args.verbose, inherit_env=False)
        return 0

    if c == "config":
        cmd = _add_profiles(base_cmd(), list(args.profiles or [])) + ["config"]
        run(cmd, env=proc_env, verbose=args.verbose, inherit_env=False)
        print("✅ compose config OK")
        return 0

    if c == "ps":
        cmd = _add_profiles(base_cmd(), list(args.profiles or [])) + ["ps"] + list(args.args or [])
        run(cmd, env=proc_env, verbose=args.verbose, inherit_env=False)
        return 0

    if c == "logs":
        cmd = _add_profiles(base_cmd(), list(args.profiles or []))
        cmd += ["logs", "-f", f"--tail={args.tail}"]
        run(cmd, env=proc_env, verbose=args.verbose, inherit_env=False)
        return 0

    if c == "down":
        cmd = _add_profiles(base_cmd(), list(args.profiles or [])) + ["down"]
        if args.remove_orphans:
            cmd.append("--remove-orphans")
        if args.volumes:
            cmd.append("-v")
        run(cmd, env=proc_env, verbose=args.verbose, inherit_env=False)
        return 0

    if c == "up":
        cmd = _add_profiles(base_cmd(), list(args.profiles or [])) + ["up"]
        if args.detach:
            cmd.append("-d")
        if args.build:
            cmd.append("--build")
        if args.remove_orphans:
            cmd.append("--remove-orphans")
        cmd += list(args.args or [])
        run(cmd, env=proc_env, verbose=args.verbose, inherit_env=False)
        return 0

    if c == "rm-orphans":
        cmd = _add_profiles(base_cmd(), list(args.profiles or [])) + ["up", "-d", "--remove-orphans"]
        run(cmd, env=proc_env, verbose=args.verbose, inherit_env=False)
        return 0

    raise CLIError(f"Unknown compose command: {c}", code=2)