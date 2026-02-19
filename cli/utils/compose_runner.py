# cli/utils/compose_runner.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from cli.errors import CLIError
from cli.types import GlobalConfig  # type: ignore[attr-defined]
from cli.utils.compose_config import render_compose_env_dict, write_env_file
from cli.utils.env import load_dotenv_file
from cli.utils.proc import ensure_bins, run

# -----------------------------
# Deterministic compose runner
# -----------------------------
#
# Option A semantics (strict):
# - Render compose-defaults.yaml (profile merge) into a dict
# - If user explicitly provides an env override file, load it into a dict
# - Merge them deterministically into ONE effective env file:
#     effective = defaults then overridden by user env (user wins)
# - Pass EXACTLY ONE --env-file to docker compose (the effective file)
# - NEVER auto-include repo .env.* files
# - Do NOT allow shell env to override compose behavior (inherit_env=False)


_PASSTHROUGH_ENV = {
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "API_KEY",
    "HUGGINGFACE_HUB_TOKEN",
    "HF_HOME",
    "HF_TOKEN",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
}

_DENYLIST_ENV = {
    "APP_PROFILE",
    "MODELS_PROFILE",
    "MODELS_YAML",
    "MODEL_LOAD_MODE",
    "REQUIRE_MODEL_READY",
    "MODEL_ID",
    "MODEL_DEVICE",
    "DB_INSTANCE",
    "DATABASE_URL",
    "REDIS_ENABLED",
    "REDIS_URL",
    "ENV",
    "WORKERS",
    "UVICORN_RELOAD",
    "POLICY_DECISION_PATH",
    "SLO_OUT_DIR",
    "COMPOSE_PROJECT_NAME",
}


@dataclass(frozen=True)
class ComposeContext:
    """
    Fully-determined compose invocation context.

    base_cmd: `docker compose ... --env-file <effective> -f ...` prefix (no profiles, no verbs)
    proc_env: process env passed to docker compose (sanitized)
    env_files: env files used (exactly one: effective env file)
    """
    base_cmd: list[str]
    proc_env: dict[str, str]
    env_files: list[Path]
    defaults_profile: str
    rendered_defaults_env: Path  # (kept for drop-in compat; now points to effective env file)
    user_env_file: Path | None


def clean_compose_process_env(cfg: GlobalConfig) -> dict[str, str]:
    env: dict[str, str] = {"COMPOSE_PROJECT_NAME": cfg.project_name}

    for k in _PASSTHROUGH_ENV:
        v = os.environ.get(k)
        if v is not None and str(v).strip() != "":
            env[k] = str(v)

    # Ensure compose-behavior env cannot leak in from caller shell.
    for k in list(env.keys()):
        if k in _DENYLIST_ENV:
            env.pop(k, None)

    env["COMPOSE_PROJECT_NAME"] = cfg.project_name
    return env


def resolve_defaults_yaml(cfg: GlobalConfig, defaults_yaml: str | None) -> Path:
    rel = (defaults_yaml or "config/compose-defaults.yaml").strip() or "config/compose-defaults.yaml"
    p = Path(rel)
    return p if p.is_absolute() else (cfg.repo_root / p)


def normalize_defaults_profile(profile: str | None) -> str:
    s = (profile or "docker").strip()
    return s or "docker"


def _safe_profile_suffix(s: str) -> str:
    return s.replace("/", "_").replace("+", "_").replace(",", "_").replace(" ", "_")


def render_effective_env_file(
    cfg: GlobalConfig,
    *,
    defaults_yaml_path: Path,
    defaults_profile: str,
    user_env_file: Path | None,
    extra_env: Mapping[str, str] | None = None,
) -> Path:
    """
    Write a single deterministic env file for docker compose:

      defaults_env = render_compose_env_dict(...)
      overrides_env = load_dotenv_file(user_env_file) if provided
      effective_env = defaults_env then overridden by overrides_env (user wins)

    Returns the path to the written file.
    """
    tmp_dir = cfg.repo_root / ".tmp" / "llmctl"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_path = tmp_dir / f"compose-effective-{_safe_profile_suffix(defaults_profile)}.env"

    defaults_env = render_compose_env_dict(
        config_yaml_path=defaults_yaml_path,
        profile=defaults_profile,
        extra_env=dict(extra_env) if extra_env is not None else {"COMPOSE_PROJECT_NAME": cfg.project_name},
    )

    overrides: dict[str, str] = {}
    if user_env_file is not None:
        overrides = dict(load_dotenv_file(user_env_file))

    effective = dict(defaults_env)
    effective.update(overrides)  # user wins

    write_env_file(out_env_path=out_path, env=effective)
    return out_path


def build_base_cmd(cfg: GlobalConfig, *, env_files: Iterable[Path]) -> list[str]:
    cmd: list[str] = ["docker", "compose"]
    for f in env_files:
        cmd += ["--env-file", str(f)]
    cmd += ["-f", str(cfg.compose_yml)]
    return cmd


def add_profiles(cmd: list[str], profiles: Sequence[str]) -> list[str]:
    out = list(cmd)
    for p in profiles:
        ps = (p or "").strip()
        if ps:
            out += ["--profile", ps]
    return out


def _resolve_user_env(cfg: GlobalConfig, env_override_file: str | Path | None) -> Path | None:
    """
    Option A: Only use a user env file when explicitly provided.
    """
    if env_override_file is None:
        return None

    s = str(env_override_file).strip()
    if not s:
        return None

    p = Path(s).expanduser()
    if not p.is_absolute():
        p = (cfg.repo_root / p).resolve()

    if not p.exists():
        raise CLIError(f"env override file not found: {p}", code=2)

    return p


def build_compose_context(
    cfg: GlobalConfig,
    *,
    defaults_profile: str | None = None,
    defaults_yaml: str | None = None,
    env_override_file: str | Path | None = None,
) -> ComposeContext:
    ensure_bins("docker")

    dp = normalize_defaults_profile(defaults_profile)
    dy = resolve_defaults_yaml(cfg, defaults_yaml)

    if not dy.exists():
        raise CLIError(f"compose defaults yaml not found: {dy}", code=2)

    # Option A: ONLY include explicit env override file (per invocation or global cfg).
    user_env = _resolve_user_env(cfg, env_override_file if env_override_file is not None else cfg.env_override_file)

    # Create ONE effective env file (defaults merged with user overrides)
    effective_env = render_effective_env_file(
        cfg,
        defaults_yaml_path=dy,
        defaults_profile=dp,
        user_env_file=user_env,
        extra_env={"COMPOSE_PROJECT_NAME": cfg.project_name},
    )

    env_files: list[Path] = [effective_env]

    proc_env = clean_compose_process_env(cfg)
    base_cmd = build_base_cmd(cfg, env_files=env_files)

    return ComposeContext(
        base_cmd=base_cmd,
        proc_env=proc_env,
        env_files=env_files,
        defaults_profile=dp,
        rendered_defaults_env=effective_env,  # kept for compat
        user_env_file=user_env,
    )


def compose_run(
    ctx: ComposeContext,
    *,
    profiles: Sequence[str],
    args: Sequence[str],
    verbose: bool,
) -> None:
    if verbose:
        print(f"+ env_files: {[str(p) for p in ctx.env_files]}")
    cmd = add_profiles(ctx.base_cmd, profiles) + list(args)
    run(cmd, env=ctx.proc_env, verbose=verbose, inherit_env=False)


def compose_config(ctx: ComposeContext, *, profiles: Sequence[str], verbose: bool) -> None:
    compose_run(ctx, profiles=profiles, args=["config"], verbose=verbose)


def compose_up(
    ctx: ComposeContext,
    *,
    profiles: Sequence[str],
    detach: bool = True,
    build: bool = False,
    remove_orphans: bool = True,
    extra_args: Sequence[str] | None = None,
    verbose: bool,
) -> None:
    args: list[str] = ["up"]
    if detach:
        args.append("-d")
    if build:
        args.append("--build")
    if remove_orphans:
        args.append("--remove-orphans")
    if extra_args:
        args += list(extra_args)
    compose_run(ctx, profiles=profiles, args=args, verbose=verbose)


def compose_down(
    ctx: ComposeContext,
    *,
    profiles: Sequence[str],
    volumes: bool = False,
    remove_orphans: bool = True,
    verbose: bool,
) -> None:
    args: list[str] = ["down"]
    if remove_orphans:
        args.append("--remove-orphans")
    if volumes:
        args.append("-v")
    compose_run(ctx, profiles=profiles, args=args, verbose=verbose)


def compose_ps(
    ctx: ComposeContext,
    *,
    profiles: Sequence[str],
    extra_args: Sequence[str] | None = None,
    verbose: bool,
) -> None:
    args: list[str] = ["ps"]
    if extra_args:
        args += list(extra_args)
    compose_run(ctx, profiles=profiles, args=args, verbose=verbose)


def compose_logs(
    ctx: ComposeContext,
    *,
    profiles: Sequence[str],
    follow: bool = True,
    tail: int = 200,
    verbose: bool = False,
) -> None:
    args: list[str] = ["logs"]
    if follow:
        args.append("-f")
    args.append(f"--tail={int(tail)}")
    compose_run(ctx, profiles=profiles, args=args, verbose=verbose)


def compose_exec(
    ctx: ComposeContext,
    *,
    service: str,
    cmd: Sequence[str],
    tty: bool = True,
    verbose: bool,
) -> None:
    if not service or not service.strip():
        raise CLIError("compose_exec requires a service name", code=2)
    if not cmd:
        raise CLIError("compose_exec requires a command", code=2)

    args: list[str] = ["exec"]
    if not tty:
        args.append("-T")
    args.append(service)
    args += list(cmd)

    compose_run(ctx, profiles=[], args=args, verbose=verbose)