# cli/commands/compose.py
from __future__ import annotations

import argparse

from cli.errors import CLIError
from cli.types import GlobalConfig  # type: ignore[attr-defined]
from cli.utils.compose_runner import (
    build_compose_context,
    compose_config,
    compose_down,
    compose_logs,
    compose_ps,
    compose_run,
    compose_up,
)

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
    p.add_argument(
        "--defaults-yaml",
        default=None,
        help="Override compose defaults YAML (default: config/compose-defaults.yaml).",
    )

    # Option A: explicit env override only (no implicit .env.*)
    p.add_argument(
        "--env-override-file",
        default=None,
        help="Optional env file to include AFTER rendered compose-defaults for this invocation (e.g. .env.docker).",
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


def _handle(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    ctx = build_compose_context(
        cfg,
        defaults_profile=getattr(args, "defaults_profile", None),
        defaults_yaml=getattr(args, "defaults_yaml", None),
        env_override_file=getattr(args, "env_override_file", None),
    )

    # Shortcuts
    if getattr(args, "_shortcut", None) == "infra-up":
        compose_up(ctx, profiles=["infra"], detach=True, build=False, remove_orphans=True, verbose=args.verbose)
        print("✅ infra up (postgres/redis).")
        return 0

    if getattr(args, "_shortcut", None) == "infra-ps":
        compose_ps(ctx, profiles=["infra"], extra_args=None, verbose=args.verbose)
        return 0

    if getattr(args, "_shortcut", None) == "infra-down":
        compose_down(
            ctx,
            profiles=["infra"],
            volumes=bool(getattr(args, "volumes", False)),
            remove_orphans=True,
            verbose=args.verbose,
        )
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
        compose_run(ctx, profiles=profiles, args=extra, verbose=args.verbose)
        return 0

    if c == "config":
        compose_config(ctx, profiles=list(args.profiles or []), verbose=args.verbose)
        print("✅ compose config OK")
        return 0

    if c == "ps":
        compose_ps(ctx, profiles=list(args.profiles or []), extra_args=list(args.args or []), verbose=args.verbose)
        return 0

    if c == "logs":
        follow = True
        tail = int(getattr(args, "tail", 200))
        compose_logs(ctx, profiles=list(args.profiles or []), follow=follow, tail=tail, verbose=args.verbose)
        return 0

    if c == "down":
        compose_down(
            ctx,
            profiles=list(args.profiles or []),
            volumes=bool(getattr(args, "volumes", False)),
            remove_orphans=bool(getattr(args, "remove_orphans", True)),
            verbose=args.verbose,
        )
        return 0

    if c == "up":
        compose_up(
            ctx,
            profiles=list(args.profiles or []),
            detach=bool(getattr(args, "detach", False)),
            build=bool(getattr(args, "build", False)),
            remove_orphans=bool(getattr(args, "remove_orphans", False)),
            extra_args=list(args.args or []),
            verbose=args.verbose,
        )
        return 0

    if c == "rm-orphans":
        compose_up(ctx, profiles=list(args.profiles or []), detach=True, build=False, remove_orphans=True, verbose=args.verbose)
        return 0

    raise CLIError(f"Unknown compose command: {c}", code=2)