from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class EvalCliRun:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_eval_cli(
    *, args: Iterable[str], env: Optional[dict[str, str]] = None, timeout_s: int = 180
) -> EvalCliRun:
    root = repo_root()
    eval_dir = root / "eval"

    argv = [
        "uv",
        "run",
        "--project",
        str(eval_dir),
        "python",
        "-m",
        "llm_eval.cli",
        *list(args),
    ]

    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)

    p = subprocess.run(
        argv,
        cwd=str(root),
        env=merged_env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    return EvalCliRun(argv=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows
