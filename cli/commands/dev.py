# cli/commands/dev.py
from __future__ import annotations

import argparse
import os

from cli.errors import CLIError
from cli.types import GlobalConfig  # type: ignore[attr-defined]
from cli.utils.compose_runner import (
    build_compose_context,
    compose_config,
    compose_exec,
    compose_up,
)
from cli.utils.proc import ensure_bins, run


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("dev", help="Golden paths: dev-cpu/dev-gpu, smoke tests, doctor.")
    p.set_defaults(_handler=_handle)

    p.add_argument(
        "--defaults-profile",
        default=None,
        help="Override compose defaults profile(s) for this invocation (e.g. docker, host, itest, jobs, or docker+jobs).",
    )
    p.add_argument(
        "--defaults-yaml",
        default=None,
        help="Override compose defaults YAML (default: config/compose-defaults.yaml).",
    )

    sp = p.add_subparsers(dest="dev_cmd", required=True)

    sp.add_parser("dev-cpu", help="infra+server (cpu) + migrations")
    sp.add_parser("dev-gpu", help="infra+server_gpu + migrations")
    sp.add_parser("dev-cpu-generate-only", help="infra+server (generate-only) + migrations")
    sp.add_parser("dev-gpu-generate-only", help="infra+server_gpu (generate-only) + migrations")

    sp.add_parser("doctor", help="Run tools/compose_doctor.sh")

    sp.add_parser("smoke-cpu", help="dev-cpu + doctor + /v1/generate probe (if API_KEY set)")
    sp.add_parser("smoke-cpu-generate-only", help="dev-cpu-generate-only + doctor")
    sp.add_parser("smoke-gpu", help="dev-gpu + doctor")
    sp.add_parser("smoke-gpu-generate-only", help="dev-gpu-generate-only + doctor")


def _migrate(ctx, verbose: bool) -> None:
    """
    Apply alembic migrations inside the running server container.
    Tries: server, then server_gpu.

    IMPORTANT:
    We always pass an explicit alembic config path so we do not rely
    on container working directory.
    """
    last_err: Exception | None = None

    for svc in ["server", "server_gpu"]:
        try:
            compose_exec(
                ctx,
                service=svc,
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
            print("✅ migrations applied (docker)")
            return
        except Exception as e:
            last_err = e
            continue

    raise CLIError(
        "No running server/server_gpu container found (or migrations failed). Start the server first.",
        code=2,
    ) from last_err


def _run_doctor(cfg: GlobalConfig, verbose: bool) -> None:
    if not cfg.compose_doctor.exists():
        raise CLIError(f"compose doctor not found: {cfg.compose_doctor}", code=2)
    if not os.access(cfg.compose_doctor, os.X_OK):
        raise CLIError(f"compose doctor not executable: chmod +x {cfg.compose_doctor}", code=2)

    env = {
        "API_PORT": cfg.api_port,
        "UI_PORT": cfg.ui_port,
        "PGADMIN_PORT": cfg.pgadmin_port,
        "PROM_PORT": cfg.prom_port,
        "GRAFANA_PORT": cfg.grafana_port,
        "PROM_HOST_PORT": cfg.prom_host_port,
        "ENV_FILE": str(cfg.env_file),
        "COMPOSE_YML": str(cfg.compose_yml),
    }
    run(["bash", str(cfg.compose_doctor)], env=env, verbose=verbose)


def _sub_args(parent: argparse.Namespace, *, dev_cmd: str) -> argparse.Namespace:
    """
    Build an args namespace for internal sub-invocations (smoke paths)
    without relying on argparse internals.
    """
    return argparse.Namespace(
        dev_cmd=dev_cmd,
        verbose=getattr(parent, "verbose", False),
        defaults_profile=getattr(parent, "defaults_profile", None),
        defaults_yaml=getattr(parent, "defaults_yaml", None),
    )


def _handle(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    ensure_bins("docker", "bash")

    ctx = build_compose_context(
        cfg,
        defaults_profile=getattr(args, "defaults_profile", None),
        defaults_yaml=getattr(args, "defaults_yaml", None),
    )

    c = args.dev_cmd

    if c == "dev-cpu":
        compose_config(ctx, profiles=["infra", "server"], verbose=args.verbose)
        compose_up(
            ctx,
            profiles=["infra", "server"],
            detach=True,
            build=True,
            remove_orphans=True,
            verbose=args.verbose,
        )
        print(f"✅ server up (docker) @ http://localhost:{cfg.api_port}")
        _migrate(ctx, args.verbose)
        return 0

    if c == "dev-gpu":
        compose_config(ctx, profiles=["infra", "server-gpu"], verbose=args.verbose)
        compose_up(
            ctx,
            profiles=["infra", "server-gpu"],
            detach=True,
            build=True,
            remove_orphans=True,
            verbose=args.verbose,
        )
        print(f"✅ server_gpu up (docker) @ http://localhost:{cfg.api_port}")
        _migrate(ctx, args.verbose)
        return 0

    if c == "dev-cpu-generate-only":
        compose_config(ctx, profiles=["infra", "server"], verbose=args.verbose)
        compose_up(
            ctx,
            profiles=["infra", "server"],
            detach=True,
            build=True,
            remove_orphans=True,
            verbose=args.verbose,
        )
        print(f"✅ server up (docker, generate-only) @ http://localhost:{cfg.api_port}")
        _migrate(ctx, args.verbose)
        return 0

    if c == "dev-gpu-generate-only":
        compose_config(ctx, profiles=["infra", "server-gpu"], verbose=args.verbose)
        compose_up(
            ctx,
            profiles=["infra", "server-gpu"],
            detach=True,
            build=True,
            remove_orphans=True,
            verbose=args.verbose,
        )
        print(f"✅ server_gpu up (docker, generate-only) @ http://localhost:{cfg.api_port}")
        _migrate(ctx, args.verbose)
        return 0

    if c == "doctor":
        _run_doctor(cfg, args.verbose)
        return 0

    if c == "smoke-cpu":
        _handle(cfg, _sub_args(args, dev_cmd="dev-cpu"))
        _handle(cfg, _sub_args(args, dev_cmd="doctor"))

        api_key = os.getenv("API_KEY", "").strip()
        if not api_key:
            print("ℹ️  API_KEY not set; skipping /v1/generate probe.")
            return 0

        run(
            [
                "bash",
                "-lc",
                f'curl -fsS -X POST "http://localhost:{cfg.api_port}/v1/generate" '
                f'-H "Content-Type: application/json" -H "X-API-Key: {api_key}" '
                f'--data \'{{"prompt":"smoke test","max_new_tokens":16,"temperature":0.2}}\' '
                f'>/dev/null && echo "✅ /v1/generate probe OK"',
            ],
            verbose=args.verbose,
        )
        return 0

    if c == "smoke-cpu-generate-only":
        _handle(cfg, _sub_args(args, dev_cmd="dev-cpu-generate-only"))
        _handle(cfg, _sub_args(args, dev_cmd="doctor"))
        return 0

    if c == "smoke-gpu":
        _handle(cfg, _sub_args(args, dev_cmd="dev-gpu"))
        _handle(cfg, _sub_args(args, dev_cmd="doctor"))
        return 0

    if c == "smoke-gpu-generate-only":
        _handle(cfg, _sub_args(args, dev_cmd="dev-gpu-generate-only"))
        _handle(cfg, _sub_args(args, dev_cmd="doctor"))
        return 0

    raise CLIError(f"Unknown dev command: {c}", code=2)