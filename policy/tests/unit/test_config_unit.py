from __future__ import annotations

from pathlib import Path

import pytest

from llm_policy.config import (
    PolicyConfig,
    load_extract_thresholds,
    load_generate_thresholds,
)


def _write_extract_threshold(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "version: v1",
                "metrics:",
                "  schema_validity_rate:",
                "    min: 95.0",
                "params:",
                "  min_n_total: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_generate_threshold(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "min_requests: 10",
                "error_rate:",
                "  threshold: 0.02",
                "  cap: 128",
                "latency_p95_ms:",
                "  steps:",
                "    1000: 256",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_policy_config_default_uses_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_POLICY_THRESHOLDS_ROOT", str(tmp_path))
    cfg = PolicyConfig.default()
    assert cfg.thresholds_root == str(tmp_path)


def test_load_extract_thresholds_shorthand_profile(tmp_path: Path) -> None:
    _write_extract_threshold(tmp_path / "extract" / "sroie.yaml")
    cfg = PolicyConfig(thresholds_root=str(tmp_path))

    profile, th = load_extract_thresholds(cfg=cfg, profile="sroie")

    assert profile == "extract/sroie"
    assert th.metrics["schema_validity_rate"].min == 95.0


def test_load_extract_thresholds_falls_back_to_default(tmp_path: Path) -> None:
    _write_extract_threshold(tmp_path / "extract" / "default.yaml")
    cfg = PolicyConfig(thresholds_root=str(tmp_path))

    profile, _th = load_extract_thresholds(cfg=cfg, profile="extract/missing")

    assert profile == "extract/default"


def test_load_extract_thresholds_raises_when_no_file_and_no_fallback(tmp_path: Path) -> None:
    cfg = PolicyConfig(thresholds_root=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        load_extract_thresholds(cfg=cfg, profile="extract/missing")


def test_load_extract_thresholds_blocks_path_traversal(tmp_path: Path) -> None:
    cfg = PolicyConfig(thresholds_root=str(tmp_path))
    with pytest.raises(ValueError, match="path traversal"):
        load_extract_thresholds(cfg=cfg, profile="../../etc/passwd")


def test_load_generate_thresholds_shorthand_profile(tmp_path: Path) -> None:
    _write_generate_threshold(tmp_path / "generate" / "portable.yaml")
    cfg = PolicyConfig(thresholds_root=str(tmp_path))

    profile, th = load_generate_thresholds(cfg=cfg, profile="portable")

    assert profile == "generate/portable"
    assert th.error_rate.cap == 128
    assert th.latency_p95_ms.steps[1000] == 256


def test_load_generate_thresholds_falls_back_to_portable(tmp_path: Path) -> None:
    _write_generate_threshold(tmp_path / "generate" / "portable.yaml")
    cfg = PolicyConfig(thresholds_root=str(tmp_path))

    profile, _th = load_generate_thresholds(cfg=cfg, profile="generate/missing")

    assert profile == "generate/portable"
