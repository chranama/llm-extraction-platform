# cli/main.py
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import cli.commands.compose as compose_cmd
import cli.commands.dev as dev_cmd
import cli.commands.eval as eval_cmd
import cli.commands.k8s as k8s_cmd
import cli.commands.policy as policy_cmd
from cli.errors import CLIError, die
from cli.types import GlobalConfig
from cli.utils.paths import find_repo_root

# -----------------------
# Defaults
# -----------------------

DEFAULT_ENV_FILE = ".env"

DEFAULT_PROJECT_NAME = "llm-extraction-platform"
DEFAULT_COMPOSE_YML = "deploy/compose/docker-compose.yml"
DEFAULT_TOOLS_DIR = "tools"
DEFAULT_COMPOSE_DOCTOR = "tools/compose_doctor.sh"
DEFAULT_SERVER_DIR = "server"

DEFAULT_MODELS_YAML = "config/models.yaml"

DEFAULT_COMPOSE_DEFAULTS_YAML = "config/compose-defaults.yaml"
DEFAULT_COMPOSE_DEFAULTS_PROFILE = "docker"  # docker | host | itest | jobs

DEFAULT_API_PORT = "8000"
DEFAULT_UI_PORT = "5173"
DEFAULT_PGADMIN_PORT = "5050"
DEFAULT_PROM_PORT = "9090"
DEFAULT_GRAFANA_PORT = "3000"
DEFAULT_PROM_HOST_PORT = "9091"

DEFAULT_PG_USER = "llm"
DEFAULT_PG_DB = "llm"


def _resolve_repo_path(repo_root: Path, raw: str) -> Path:
    """
    Resolve a user-provided path consistently:
      - expand ~
      - if absolute, use as-is
      - else resolve relative to repo_root
      - normalize with resolve()
    """
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def _build_global_config(args: argparse.Namespace) -> GlobalConfig:
    repo_root = find_repo_root(Path.cwd(), compose_rel=DEFAULT_COMPOSE_YML)

    env_file = _resolve_repo_path(repo_root, args.env_file or DEFAULT_ENV_FILE)
    compose_yml = _resolve_repo_path(repo_root, args.compose_yml or DEFAULT_COMPOSE_YML)
    tools_dir = _resolve_repo_path(repo_root, args.tools_dir or DEFAULT_TOOLS_DIR)
    compose_doctor = _resolve_repo_path(repo_root, args.compose_doctor or DEFAULT_COMPOSE_DOCTOR)

    server_dir = _resolve_repo_path(repo_root, args.server_dir or DEFAULT_SERVER_DIR)

    # ✅ argparse flag is --models-yaml => args.models_yaml
    models_yaml = _resolve_repo_path(repo_root, getattr(args, "models_yaml", None) or DEFAULT_MODELS_YAML)

    return GlobalConfig(
        repo_root=repo_root,
        env_file=env_file,
        project_name=args.project_name or DEFAULT_PROJECT_NAME,
        compose_yml=compose_yml,
        tools_dir=tools_dir,
        compose_doctor=compose_doctor,
        server_dir=server_dir,
        models_yaml=models_yaml,
        api_port=args.api_port or DEFAULT_API_PORT,
        ui_port=args.ui_port or DEFAULT_UI_PORT,
        pgadmin_port=args.pgadmin_port or DEFAULT_PGADMIN_PORT,
        prom_port=args.prom_port or DEFAULT_PROM_PORT,
        grafana_port=args.grafana_port or DEFAULT_GRAFANA_PORT,
        prom_host_port=args.prom_host_port or DEFAULT_PROM_HOST_PORT,
        pg_user=args.pg_user or DEFAULT_PG_USER,
        pg_db=args.pg_db or DEFAULT_PG_DB,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llmctl",
        description="Root CLI: compose workflows, dev paths, eval, policy, and kind/k8s helpers.",
    )

    # Global options
    p.add_argument("--env-file", default=os.getenv("LLMCTL_ENV_FILE", DEFAULT_ENV_FILE))
    p.add_argument("--project-name", default=os.getenv("LLMCTL_PROJECT_NAME", DEFAULT_PROJECT_NAME))
    p.add_argument("--compose-yml", default=os.getenv("LLMCTL_COMPOSE_YML", DEFAULT_COMPOSE_YML))
    p.add_argument("--tools-dir", default=os.getenv("LLMCTL_TOOLS_DIR", DEFAULT_TOOLS_DIR))
    p.add_argument("--compose-doctor", default=os.getenv("LLMCTL_COMPOSE_DOCTOR", DEFAULT_COMPOSE_DOCTOR))

    p.add_argument("--server-dir", default=os.getenv("LLMCTL_SERVER_DIR", DEFAULT_SERVER_DIR))

    # Keep this around as a *host-path* convenience for tooling or future commands.
    # (Compose itself should not inherit it due to compose.py denylist.)
    p.add_argument("--models-yaml", dest="models_yaml", default=os.getenv("LLMCTL_MODELS_YAML", DEFAULT_MODELS_YAML))

    # Internal compose defaults (YAML + selected profile)
    p.add_argument(
        "--compose-defaults-yaml",
        default=os.getenv("LLMCTL_COMPOSE_DEFAULTS_YAML", DEFAULT_COMPOSE_DEFAULTS_YAML),
        help="YAML containing internal compose default env values (profiles: docker/host/itest/jobs).",
    )
    p.add_argument(
        "--compose-defaults-profile",
        default=os.getenv("LLMCTL_COMPOSE_DEFAULTS_PROFILE", DEFAULT_COMPOSE_DEFAULTS_PROFILE),
        help="Which defaults profile to render (docker|host|itest|jobs).",
    )

    p.add_argument("--api-port", default=os.getenv("API_PORT", DEFAULT_API_PORT))
    p.add_argument("--ui-port", default=os.getenv("UI_PORT", DEFAULT_UI_PORT))
    p.add_argument("--pgadmin-port", default=os.getenv("PGADMIN_PORT", DEFAULT_PGADMIN_PORT))
    p.add_argument("--prom-port", default=os.getenv("PROM_PORT", DEFAULT_PROM_PORT))
    p.add_argument("--grafana-port", default=os.getenv("GRAFANA_PORT", DEFAULT_GRAFANA_PORT))
    p.add_argument("--prom-host-port", default=os.getenv("PROM_HOST_PORT", DEFAULT_PROM_HOST_PORT))

    p.add_argument("--pg-user", default=os.getenv("POSTGRES_USER", DEFAULT_PG_USER))
    p.add_argument("--pg-db", default=os.getenv("POSTGRES_DB", DEFAULT_PG_DB))

    p.add_argument("--verbose", action="store_true", help="Print the exact commands being executed.")

    sub = p.add_subparsers(dest="cmd", required=True)

    compose_cmd.register(sub)
    dev_cmd.register(sub)
    eval_cmd.register(sub)
    policy_cmd.register(sub)
    k8s_cmd.register(sub)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()

    try:
        args = parser.parse_args(argv)
        cfg = _build_global_config(args)

        handler = getattr(args, "_handler", None)
        if handler is None:
            die("Internal error: no handler registered for command", code=2)

        return int(handler(cfg, args) or 0)

    except CLIError as e:
        die(str(e), code=e.code)
    except KeyboardInterrupt:
        die("Interrupted.", code=130)
    except SystemExit:
        raise
    except Exception as e:
        die(f"Unexpected error: {type(e).__name__}: {e}", code=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())