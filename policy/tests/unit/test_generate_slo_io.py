from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_policy.io.generate_slo import (
    default_generate_slo_path,
    read_generate_slo_snapshot_result,
    resolve_generate_slo_path,
)


def test_default_generate_slo_path_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLO_OUT_DIR", "/tmp/slo-dir")
    assert default_generate_slo_path() == Path("/tmp/slo-dir/latest.json")


def test_default_generate_slo_path_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLO_OUT_DIR", raising=False)
    assert default_generate_slo_path() == Path("slo_out/generate/latest.json")


def test_resolve_generate_slo_path_rejects_directory_like_path() -> None:
    with pytest.raises(ValueError, match="file path"):
        resolve_generate_slo_path(".")


def test_read_generate_slo_snapshot_result_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "latest.json"
    snap = SimpleNamespace(error=None, schema_version="generate_slo_v1")
    monkeypatch.setattr("llm_policy.io.generate_slo.read_generate_slo", lambda p: snap)

    res = read_generate_slo_snapshot_result(path)

    assert res.ok is True
    assert res.artifact is snap
    assert res.error is None
    assert res.resolved_path == str(path)


def test_read_generate_slo_snapshot_result_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "latest.json"
    snap = SimpleNamespace(error="parse failed", schema_version=None)
    monkeypatch.setattr("llm_policy.io.generate_slo.read_generate_slo", lambda p: snap)

    res = read_generate_slo_snapshot_result(path)

    assert res.ok is False
    assert res.artifact is snap
    assert res.error == "parse failed"
