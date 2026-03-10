#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "proof" / "evidence_manifest.latest.json"


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "UNKNOWN"


def main() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    data["repo_commit"] = git_commit()
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {MANIFEST}")


if __name__ == "__main__":
    main()
