from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _schemas_root(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("SCHEMAS_ROOT", str(repo_root / "schemas"))
