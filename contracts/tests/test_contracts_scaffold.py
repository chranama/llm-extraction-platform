from __future__ import annotations

from pathlib import Path

from llm_contracts import schema


def test_schemas_root_defaults_to_cwd_schemas(monkeypatch, tmp_path: Path) -> None:
    # Exercise fallback path behavior when SCHEMAS_ROOT is not provided.
    (tmp_path / "schemas").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SCHEMAS_ROOT", raising=False)

    assert schema.schemas_root() == (tmp_path / "schemas").resolve()


def test_internal_schemas_dir_points_under_configured_root(monkeypatch, tmp_path: Path) -> None:
    custom_root = tmp_path / "custom-schemas"
    monkeypatch.setenv("SCHEMAS_ROOT", str(custom_root))

    assert schema.internal_schemas_dir() == (custom_root / "internal").resolve()
