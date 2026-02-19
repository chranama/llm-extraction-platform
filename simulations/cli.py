# simulations/cli.py
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from simulations.paths import ArtifactPaths, find_repo_root, resolve_under_repo
from simulations.artifacts.main import register_artifacts_subcommands
from simulations.traffic.main import register_traffic_subcommands


# ---------------------------------------------------------------------
# Errors / exit codes
# ---------------------------------------------------------------------


class SimError(Exception):
    def __init__(self, message: str, *, code: int = 2):
        super().__init__(message)
        self.code = int(code)


def die(msg: str, *, code: int = 2) -> int:
    print(msg, file=sys.stderr)
    return code


# ---------------------------------------------------------------------
# Runtime settings
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Runtime:
    repo_root: Path
    base_url: str
    api_key: Optional[str]
    timeout_s: float
    dry_run: bool

    paths: ArtifactPaths
    policy_out_path: Path
    slo_out_path: Path

    # Treated as an override for the *eval pointer* file
    # (normally eval_out/<task>/latest.json). If unset, contracts pick defaults.
    eval_out_path: Path


def _normalize_base_url(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return "http://127.0.0.1:8000"
    return s.rstrip("/")


def _build_runtime(args: argparse.Namespace) -> Runtime:
    repo_root = find_repo_root(Path.cwd())
    ap = ArtifactPaths.from_repo_root(repo_root)

    base_url = _normalize_base_url(getattr(args, "base_url", None) or os.getenv("SIM_BASE_URL", ""))
    api_key = (getattr(args, "api_key", None) or os.getenv("API_KEY") or os.getenv("SIM_API_KEY") or "").strip() or None
    timeout_s = float(getattr(args, "timeout", None) or 20.0)
    dry_run = bool(getattr(args, "dry_run", False))

    policy_out = resolve_under_repo(repo_root, getattr(args, "policy_out", None))
    slo_out = resolve_under_repo(repo_root, getattr(args, "slo_out", None))
    eval_out = resolve_under_repo(repo_root, getattr(args, "eval_out", None))

    return Runtime(
        repo_root=repo_root,
        base_url=base_url,
        api_key=api_key,
        timeout_s=timeout_s,
        dry_run=dry_run,
        paths=ap,
        policy_out_path=policy_out or ap.policy_out_latest,
        slo_out_path=slo_out or ap.slo_generate_latest,
        eval_out_path=eval_out or ap.eval_extract_latest,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _print_paths(rt: Runtime) -> int:
    print(
        json.dumps(
            {
                "repo_root": str(rt.repo_root),
                "base_url": rt.base_url,
                "policy_out": str(rt.policy_out_path),
                "slo_out": str(rt.slo_out_path),
                "eval_out": str(rt.eval_out_path),
                "dry_run": rt.dry_run,
            },
            indent=2,
        )
    )
    return 0


# ---------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sim",
        description="Deterministic simulations: write artifacts and run traffic scenarios (run via uv).",
    )

    # Global flags
    p.add_argument("--base-url", default=os.getenv("SIM_BASE_URL", "http://127.0.0.1:8000"))
    p.add_argument("--api-key", default=os.getenv("SIM_API_KEY", None))
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--dry-run", action="store_true", help="Print actions but do not write files or call HTTP.")
    p.add_argument("--policy-out", default=None, help="Override policy_out/latest.json path (relative to repo root ok).")
    p.add_argument("--slo-out", default=None, help="Override slo_out/generate/latest.json path (relative ok).")

    # Treated as pointer out override (normally eval_out/<task>/latest.json)
    p.add_argument(
        "--eval-out",
        default=None,
        help="Override eval run pointer output path (normally eval_out/<task>/latest.json). Relative to repo root ok.",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # paths
    sp = sub.add_parser("paths", help="Print resolved repo-root + artifact paths.")
    sp.set_defaults(_handler=lambda rt, a: _print_paths(rt))

    # artifacts + traffic
    register_artifacts_subcommands(sub)
    register_traffic_subcommands(sub)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        rt = _build_runtime(args)

        handler = getattr(args, "_handler", None)
        if handler is None:
            raise SimError("Internal error: no handler for command.", code=2)

        return int(handler(rt, args) or 0)

    except SimError as e:
        return die(str(e), code=e.code)
    except KeyboardInterrupt:
        return die("Interrupted.", code=130)


if __name__ == "__main__":
    raise SystemExit(main())