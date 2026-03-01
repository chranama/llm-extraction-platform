from __future__ import annotations

from pathlib import Path

import pytest

from simulations.paths import ArtifactPaths, PathsError, ensure_parent_dir, find_repo_root, resolve_under_repo


def test_find_repo_root_from_nested_dir(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "a" / "b" / "c"
    nested.mkdir(parents=True, exist_ok=True)
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    assert find_repo_root(nested) == repo.resolve()


def test_find_repo_root_raises_when_not_found(tmp_path: Path) -> None:
    p = tmp_path / "x" / "y"
    p.mkdir(parents=True, exist_ok=True)
    with pytest.raises(PathsError):
        find_repo_root(p)


def test_artifact_paths_from_repo_root(tmp_path: Path) -> None:
    ap = ArtifactPaths.from_repo_root(tmp_path)
    assert ap.policy_out_latest == (tmp_path / "policy_out" / "latest.json")
    assert ap.slo_generate_latest == (tmp_path / "slo_out" / "generate" / "latest.json")
    assert ap.eval_extract_latest == (tmp_path / "eval_out" / "extract" / "latest.json")


def test_resolve_under_repo_none_blank_relative_absolute(tmp_path: Path) -> None:
    assert resolve_under_repo(tmp_path, None) is None
    assert resolve_under_repo(tmp_path, "   ") is None
    assert resolve_under_repo(tmp_path, "a/b.json") == (tmp_path / "a/b.json").resolve()
    abs_p = (tmp_path / "abs.json").resolve()
    assert resolve_under_repo(tmp_path, str(abs_p)) == abs_p


def test_ensure_parent_dir_creates_parent(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "file.json"
    ensure_parent_dir(out)
    assert out.parent.exists()
