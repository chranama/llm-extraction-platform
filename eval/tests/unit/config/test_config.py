from __future__ import annotations

import os
from pathlib import Path

import pytest

import llm_eval.config as cfg


def test_expand_env_str_supports_default_and_plain_var(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("API_HOST", "api.local")
    monkeypatch.delenv("MISSING", raising=False)

    out = cfg._expand_env_str("http://${API_HOST}:${PORT:-8080}/x/${MISSING:-fallback}/$API_HOST")
    assert out == "http://api.local:8080/x/fallback/api.local"


def test_expand_env_recurses_dicts_and_lists(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("A", "va")
    payload = {
        "x": "${A}",
        "y": ["$A", {"z": "${B:-vb}"}],
        "n": 7,
    }

    out = cfg._expand_env(payload)
    assert out == {"x": "va", "y": ["va", {"z": "vb"}], "n": 7}


def test_load_eval_yaml_missing_or_invalid_returns_empty(tmp_path: Path):
    missing = tmp_path / "missing.yaml"
    assert cfg.load_eval_yaml(str(missing)) == {}

    p = tmp_path / "bad.yaml"
    p.write_text("- not-a-dict\n", encoding="utf-8")
    assert cfg.load_eval_yaml(str(p)) == {}


def test_load_eval_yaml_expands_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("BASE_URL", "http://svc")
    p = tmp_path / "eval.yaml"
    p.write_text(
        "service:\n"
        "  base_url: ${BASE_URL}\n"
        "  api_key_env: EVAL_KEY\n"
        "run:\n"
        "  outdir_root: ${OUTDIR:-results}\n",
        encoding="utf-8",
    )

    out = cfg.load_eval_yaml(str(p))
    assert out["service"]["base_url"] == "http://svc"
    assert out["run"]["outdir_root"] == "results"


def test_dig_and_get_api_key(monkeypatch: pytest.MonkeyPatch):
    c = {"service": {"api_key_env": "EVAL_KEY"}, "nested": {"a": {"b": 1}}}
    assert cfg.dig(c, "nested", "a", "b") == 1
    assert cfg.dig(c, "nested", "x", default="d") == "d"

    monkeypatch.setenv("EVAL_KEY", "secret")
    assert cfg.get_api_key(c) == "secret"

    monkeypatch.delenv("EVAL_KEY", raising=False)
    assert cfg.get_api_key(c) is None
    assert cfg.get_api_key({}) is None
