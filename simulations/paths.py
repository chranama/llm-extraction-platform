# simulations/paths.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class PathsError(RuntimeError):
    pass


def find_repo_root(start: Path | None = None) -> Path:
    """
    Resolve repo root by walking upward until we find something canonical.

    Deterministic, no config required. We keep this conservative:
      - prefer repo-level pyproject.toml
      - otherwise accept deploy/compose/docker-compose.yml

    Raises:
      PathsError if root cannot be found within a bounded walk.
    """
    cur = (start or Path.cwd()).resolve()

    for _ in range(25):
        if (cur / "pyproject.toml").exists():
            return cur
        if (cur / "deploy" / "compose" / "docker-compose.yml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent

    raise PathsError(
        "Could not find repo root (expected pyproject.toml or deploy/compose/docker-compose.yml). "
        "Run from within the repo."
    )


@dataclass(frozen=True)
class ArtifactPaths:
    """
    Canonical host-side artifact paths.

    Control-plane invariants used by simulations and preflight:
      - policy_out/latest.json
      - slo_out/generate/latest.json
      - eval_out/<task>/latest.json (task-scoped pointer)

    Notes:
      - Keep eval_out/latest.json as a legacy alias if older tooling expects it.
    """

    repo_root: Path

    # Policy decision artifact (runtime)
    policy_out_latest: Path

    # Generate SLO snapshot artifact (runtime)
    slo_generate_latest: Path

    # Eval pointers (canonical)
    eval_extract_latest: Path

    # Legacy alias pointer (optional compatibility)
    eval_out_latest_legacy: Path

    @staticmethod
    def from_repo_root(repo_root: Path) -> "ArtifactPaths":
        rr = repo_root.resolve()
        return ArtifactPaths(
            repo_root=rr,
            policy_out_latest=rr / "policy_out" / "latest.json",
            slo_generate_latest=rr / "slo_out" / "generate" / "latest.json",
            eval_extract_latest=rr / "eval_out" / "extract" / "latest.json",
            eval_out_latest_legacy=rr / "eval_out" / "latest.json",
        )


def resolve_under_repo(repo_root: Path, raw: str | None) -> Optional[Path]:
    """
    Resolve user-provided path consistently:
      - expand ~
      - if absolute, use as-is
      - else resolve relative to repo_root
      - normalize with resolve()

    Returns None if raw is None/empty/whitespace.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    p = Path(s).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)